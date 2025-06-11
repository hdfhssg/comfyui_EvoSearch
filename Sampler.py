import torch
from comfy.samplers import KSampler
from .evosearch.utils import do_eval 
import comfy
import latent_preview
from torchvision.transforms import ToPILImage

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )

import numpy as np
class EvoSearch_FLUX:
    CATEGORY = "evolution"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT", {"tooltip": "必选初始潜在，格式为 {'samples': Tensor, 'batch_index': [...]}" }),
                "vae": ("VAE", {"tooltip": "用来解码 latent 的 VAE 模型"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "The number of steps used in the denoising process."}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality."}),
                "sampler_name": (KSampler.SAMPLERS,),
                "scheduler": (KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0}),
                "evolution_schedule": ("LIST", {"default": [0,10,20,30,50]}),
                "population_size": ("INT", {"default": 8}),
                "elite_count": ("INT", {"default": 2}),
                "guidance_rewards": ("LIST",  {
                    "default": ["clip_score"],
                    "choices": [
                        "clip_score", "aesthetic_score", "pickscore",
                        "image_reward", "clip_score_only", "human_preference"
                    ]
                }),
                "prompt_text": ("STRING", {"default": "a beautiful landscape"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

   def decode_latents_to_images(self, vae, latent_batch):
        """
        将 latent 通过 VAE 解码到图像，并缩放到 224x224 的 PIL.Image 列表。
        """
        with torch.no_grad():
            # 优先使用 decode_first_stage 输出 [N,3,H,W]
            if hasattr(vae, "decode_first_stage"):
                decoded = vae.decode_first_stage(latent_batch)
            elif hasattr(vae, "decode_latents"):
                decoded = vae.decode_latents(latent_batch)
            else:
                out = vae.decode(latent_batch)
                decoded = out if isinstance(out, torch.Tensor) else out.get("sample", out.get("images"))

        # 将张量值从 [-1,1] 或 [0,1] 归一化到 [0,255]
        decoded = (decoded.clamp(-1,1) + 1) / 2  # now [0,1]
        decoded = (decoded * 255).clamp(0,255).to(torch.uint8)

        images = []
        for img_tensor in decoded:  # img_tensor: [3,H,W]
            # 转到 CPU 并变成 HWC numpy
            np_img = img_tensor.cpu().permute(1, 2, 0).numpy()
            pil = Image.fromarray(np_img)
            # 缩放到 CLIP 常用大小
            pil = pil.resize((224, 224), Image.BICUBIC)
            images.append(pil)
        return images

    def evaluate_images(self, prompt, images, guidance_rewards):
        results = do_eval(
            prompt=[prompt] * len(images),
            images=images,
            metrics_to_compute=guidance_rewards
        )
        scores = []
        for i in range(len(images)):
            vals = [results[metric][i] for metric in guidance_rewards]
            scores.append(float(np.mean(vals)))
        return np.array(scores)

    def generate(self, model, positive, negative, latent, vae,
                 sampler_name, scheduler, cfg, seed, steps, denoise,
                 evolution_schedule, population_size, elite_count,
                 guidance_rewards, prompt_text):

        device = model.load_device
        # 从输入 latent 中读取初始潜在
        base_latent = latent["samples"].to(device)  # Tensor shape: [1, C, H, W]
        batch_index = latent["batch_index"] if "batch_index" in latent else None

        schedule = sorted(set(int(s) for s in evolution_schedule))
        final_step = schedule[-1] if schedule else steps

        # 初始化潜在种群：复制传入 latent
        latents = [
            {"samples": base_latent.clone(), "batch_index": batch_index}
            for _ in range(population_size)
        ]

        prev_step = 0
        for stage_idx, stage in enumerate(schedule):
            # 分段采样
            new_latents = []
            for i, latent_dict in enumerate(latents):
                out = common_ksampler(
                    model=model,
                    seed=seed + i + stage_idx * 1000,
                    steps=final_step,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent=latent_dict,
                    denoise=denoise,
                    disable_noise=True,
                    start_step=prev_step,
                    last_step=stage
                )[0]
                new_latents.append(out)
            latents = new_latents
            prev_step = stage

            # 解码并评估
            lat_batch = lat_batch = torch.cat([d['samples'] for d in latents], dim=0).to(device)
            lat_batch = lat_batch.squeeze(1) if lat_batch.dim() == 5 else lat_batch
            #lat_batch = torch.cat([d['samples'] for d in latents], dim=0)
            images = self.decode_latents_to_images(vae, lat_batch)
            #print(images[0].shape)
            scores = self.evaluate_images(prompt_text, images, guidance_rewards)

            # 选出精英
            top_idx = scores.argsort()[::-1][:elite_count]
            elites = [latents[i]['samples'].clone() for i in top_idx]

            # 重建种群
            latents = []
            for i in range(population_size):
                base = elites[i % elite_count].clone()
                noise = torch.randn_like(base) * 0.01
                latents.append({"samples": base + noise, "batch_index": batch_index})

        # 最终评估并返回最佳 latent
        lat_batch = lat_batch = torch.cat([d['samples'] for d in latents], dim=0).to(device)
        lat_batch = lat_batch.squeeze(1) if lat_batch.dim() == 5 else lat_batch
        #lat_batch = torch.cat([d['samples'] for d in latents], dim=0)
        images = self.decode_latents_to_images(vae, lat_batch)
        scores = self.evaluate_images(prompt_text, images, guidance_rewards)
        best_idx = scores.argmax()
        best_latent = latents[best_idx]
        return (best_latent,)

class EvoSearch_SD21:
    CATEGORY = "sampling"  # or "evolutionary" etc.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Stable Diffusion 2.1 model"}),
                "positive": ("CONDITIONING", {"tooltip": "Positive prompt conditioning"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative prompt conditioning"}),
                "sampler_name": (KSampler.SAMPLERS, {"tooltip": "Sampling algorithm"}),
                "scheduler": (KSampler.SCHEDULERS, {"tooltip": "Noise scheduler"}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 30.0}),
                "seed": ("INT", {"default": 0, "min": 0}),
                "steps": ("INT", {"default": 20, "min": 1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "evolution_schedule": ("LIST", "INT", {"tooltip": "Timesteps for evaluation (e.g. [0,5,20,30])"}),
                "population_size": ("INT", {"default": 10, "min": 1}),
                "elite_count": ("INT", {"default": 1, "min": 1}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    def generate(self, model, positive, negative, sampler_name, scheduler, cfg, seed, steps, denoise, evolution_schedule, population_size, elite_count):
        # Convert schedule to sorted unique list of ints
        schedule = sorted(set(int(s) for s in evolution_schedule))
        final_step = schedule[-1] if schedule else steps
        # Initialize population latents as empty (None -> noise)
        pop_latents = []
        for i in range(population_size):
            pop_latents.append({"samples": None, "batch_index": [0]})

        prev_step = 0
        for stage_idx, stage in enumerate(schedule):
            if stage == 0:
                prev_step = 0
                continue
            # Denoise each latent from prev_step to stage
            new_latents = []
            for idx, latent in enumerate(pop_latents):
                # Use seed offset to diversify initial noise
                out = common_ksampler(model, seed + idx, final_step, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise, start_step=prev_step, last_step=stage)
                # out is (latent_dict,), take the dict
                new_latents.append(out[0])
            pop_latents = new_latents
            prev_step = stage

            # If not final stage, select elites and repopulate
            if stage_idx < len(schedule) - 1:
                # Placeholder scoring: here use mean latent value (replace with real reward)
                scores = [torch.mean(lat["samples"]).item() for lat in pop_latents]
                # Select top indices (higher is better)
                top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:elite_count]
                elites = [pop_latents[i] for i in top_idxs]
                # Repopulate to original population_size by cloning elites with small noise
                new_pop = []
                for i in range(population_size):
                    base = elites[i % elite_count]
                    # Perturb latent slightly for diversity
                    perturbed = base["samples"].clone() + torch.randn_like(base["samples"]) * 0.001
                    new_pop.append({"samples": perturbed, "batch_index": base.get("batch_index", [0])})
                pop_latents = new_pop

        # Final selection from last stage
        scores = [torch.mean(lat["samples"]).item() for lat in pop_latents]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_latent = pop_latents[best_idx]
        return (best_latent,)


class EvoSearch_WAN:
    CATEGORY = "sampling"  # video sampling
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Wan video generation model"}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning"}),
                "sampler_name": (KSampler.SAMPLERS, {"tooltip": "Sampling algorithm"}),
                "scheduler": (KSampler.SCHEDULERS, {"tooltip": "Noise scheduler"}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 30.0}),
                "seed": ("INT", {"default": 0, "min": 0}),
                "steps": ("INT", {"default": 20, "min": 1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "evolution_schedule": ("LIST", "INT", {"tooltip": "Timesteps for evaluation (e.g. [5,20,30,45])"}),
                "population_size": ("INT", {"default": 5, "min": 1}),
                "elite_count": ("INT", {"default": 1, "min": 1}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    def generate(self, model, positive, negative, sampler_name, scheduler, cfg, seed, steps, denoise, evolution_schedule, population_size, elite_count):
        schedule = sorted(set(int(s) for s in evolution_schedule))
        final_step = schedule[-1] if schedule else steps
        pop_latents = []
        for i in range(population_size):
            pop_latents.append({"samples": None, "batch_index": [0]})

        prev_step = 0
        for stage_idx, stage in enumerate(schedule):
            if stage == 0:
                prev_step = 0
                continue
            new_latents = []
            for idx, latent in enumerate(pop_latents):
                out = common_ksampler(model, seed + idx, final_step, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise, start_step=prev_step, last_step=stage)
                new_latents.append(out[0])
            pop_latents = new_latents
            prev_step = stage

            if stage_idx < len(schedule) - 1:
                # Placeholder scoring: average latent value across all frames
                scores = [torch.mean(lat["samples"]).item() for lat in pop_latents]
                top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:elite_count]
                elites = [pop_latents[i] for i in top_idxs]
                new_pop = []
                for i in range(population_size):
                    base = elites[i % elite_count]
                    perturbed = base["samples"].clone() + torch.randn_like(base["samples"]) * 0.001
                    new_pop.append({"samples": perturbed, "batch_index": base.get("batch_index", [0])})
                pop_latents = new_pop

        scores = [torch.mean(lat["samples"]).item() for lat in pop_latents]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_latent = pop_latents[best_idx]
        return (best_latent,)
class EvolutionScheduleGenerator:
    """
    输入一个逗号分隔的字符串，输出一个整数列表（evolution_schedule）。
    """
    CATEGORY = "Params"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_str": ("STRING", {
                    "default": "0,10,20,30,50",
                    "tooltip": "请输入逗号分隔的步数列表，如 0,10,20,30,50"
                })
            }
        }
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("evolution_schedule",)
    FUNCTION = "generate_schedule"

    def generate_schedule(self, schedule_str):
        # 解析逗号分隔的字符串为 int 列表
        parts = [p.strip() for p in schedule_str.split(",") if p.strip() != ""]
        schedule = []
        for p in parts:
            try:
                schedule.append(int(p))
            except ValueError:
                continue
        # 确保有终点为正整数
        schedule = sorted(set(schedule))
        return (schedule,)


class GuidanceRewardsGenerator:
    """
    通过多复选框按钮选择奖励函数，输出 guidance_rewards 列表。
    """
    CATEGORY = "Params"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_score": ("BOOLEAN", {"default": False, "label": "Clip-Score"}),
                "aesthetic_score": ("BOOLEAN", {"default": False, "label": "Aesthetic"}),
                "pickscore": ("BOOLEAN", {"default": False, "label": "Pickscore"}),
                "image_reward": ("BOOLEAN", {"default": False, "label": "ImageReward"}),
                "clip_score_only": ("BOOLEAN", {"default": False, "label": "Clip-Score-only"}),
                "human_preference": ("BOOLEAN", {"default": False, "label": "HumanPreference"}),
            }
        }
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("guidance_rewards",)
    FUNCTION = "generate_rewards"

    def generate_rewards(self, clip_score, aesthetic_score, pickscore,
                         image_reward, clip_score_only, human_preference):
        rewards = []
        if clip_score:
            rewards.append("Clip-Score")
        if aesthetic_score:
            rewards.append("Aesthetic")
        if pickscore:
            rewards.append("Pickscore")
        if image_reward:
            rewards.append("ImageReward")
        if clip_score_only:
            rewards.append("Clip-Score-only")
        if human_preference:
            rewards.append("HumanPreference")
        # 如果用户未选中任何奖励，默认回传 ["clip_score"]
        if not rewards:
            rewards = ["Clip-Score"]
        return (rewards,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "EvoSearch_FLUX": EvoSearch_FLUX,
    "EvoSearch_SD21": EvoSearch_SD21,
    "EvoSearch_WAN": EvoSearch_WAN,
    "EvolutionScheduleGenerator": EvolutionScheduleGenerator,
    "GuidanceRewardsGenerator": GuidanceRewardsGenerator,
}
