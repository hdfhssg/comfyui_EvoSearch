import torch
from comfy.samplers import KSampler
from .evosearch.utils import do_eval 

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
                "sampler_name": (KSampler.SAMPLERS,),
                "scheduler": (KSampler.SCHEDULERS,),
                "cfg": ("FLOAT", {"default": 7.5}),
                "seed": ("INT", {"default": 0}),
                "steps": ("INT", {"default": 50}),
                "denoise": ("FLOAT", {"default": 1.0}),
                "evolution_schedule": ("LIST", "INT", {"default": [0, 10, 20, 30, 50]}),
                "population_size": ("INT", {"default": 8}),
                "elite_count": ("INT", {"default": 2}),
                "guidance_rewards": ("LIST", "STRING", {
                    "default": ["clip_score"],
                    "choices": [
                        "clip_score", "aesthetic_score", "pickscore",
                        "image_reward", "clip_score_only", "human_preference"
                    ]
                }),
                "prompt_text": ("STRING", {"default": "a beautiful landscape"})
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    def decode_latents_to_images(self, vae, latent_batch):
        # Convert latents to decoded RGB images in [0, 255] numpy format
        decoded = vae.decode(latent_batch)['images']
        decoded = (decoded.clamp(0.0, 1.0) * 255).to(torch.uint8)
        images = [img.permute(1, 2, 0).cpu().numpy() for img in decoded]
        return images

    def generate(self, model, positive, negative, sampler_name, scheduler, cfg, seed,
                 steps, denoise, evolution_schedule, population_size, elite_count,
                 guidance_reward, prompt_text):
        device = model.load_device
        latent_shape = (4, 64, 64)  # SD默认 latent shape

        schedule = sorted(set(int(s) for s in evolution_schedule))
        final_step = schedule[-1] if schedule else steps

        # 初始化种群 latent
        latents = [torch.randn(latent_shape, device=device) for _ in range(population_size)]
        latents = [{"samples": l.unsqueeze(0), "batch_index": [0]} for l in latents]

        prev_step = 0
        for stage_idx, stage in enumerate(schedule):
            new_latents = []
            for i, latent in enumerate(latents):
                sampled = common_ksampler(
                    model=model,
                    seed=seed + i + stage_idx * 1000,
                    steps=final_step,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent=latent,
                    denoise=denoise,
                    disable_pbar=True,
                    start_step=prev_step,
                    last_step=stage
                )[0]
                new_latents.append(sampled)
            latents = new_latents
            prev_step = stage

            # decode并打分
            lat_batch = torch.cat([d['samples'] for d in latents], dim=0)
            images = self.decode_latents_to_images(model.model.decode_first_stage, lat_batch)
            results = do_eval(prompt=[prompt_text] * population_size, images=images,
                              metrics_to_compute=[guidance_reward])
            scores = results[guidance_reward]
            scores = np.array(scores)

            # 精英选择 + 复制 + 添加微噪声
            top_idx = scores.argsort()[::-1][:elite_count]
            elite_latents = [latents[i]['samples'].clone() for i in top_idx]

            latents = []
            for i in range(population_size):
                base = elite_latents[i % elite_count].clone()
                noise = torch.randn_like(base) * 0.01
                latents.append({"samples": base + noise, "batch_index": [0]})

        # 最终选择得分最高者
        lat_batch = torch.cat([d['samples'] for d in latents], dim=0)
        images = self.decode_latents_to_images(model.model.decode_first_stage, lat_batch)
        results = do_eval(prompt=[prompt_text] * population_size, images=images,
                          metrics_to_compute=[guidance_reward])
        scores = results[guidance_reward]
        scores = np.array(scores)
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
            rewards.append("clip_score")
        if aesthetic_score:
            rewards.append("aesthetic_score")
        if pickscore:
            rewards.append("pickscore")
        if image_reward:
            rewards.append("image_reward")
        if clip_score_only:
            rewards.append("clip_score_only")
        if human_preference:
            rewards.append("human_preference")
        # 如果用户未选中任何奖励，默认回传 ["clip_score"]
        if not rewards:
            rewards = ["clip_score"]
        return (rewards,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "EvoSearch_FLUX": EvoSearch_FLUX,
    "EvoSearch_SD21": EvoSearch_SD21,
    "EvoSearch_WAN": EvoSearch_WAN,
    "EvolutionScheduleGenerator": EvolutionScheduleGenerator,
    "GuidanceRewardsGenerator": GuidanceRewardsGenerator,
}
