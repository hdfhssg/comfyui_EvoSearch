import torch
from comfy.samplers import KSampler
# 假设 EvoSearch 库中提供了如下函数
from evosearch.reward import aesthetic_score, clip_eval  

class EvoSearch_FLUX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),                 # 已加载的 Flux 模型
                "positive": ("CONDITIONING", {}),    # 正向提示
                "negative": ("CONDITIONING", {}),    # 负向提示
                "sampler_name": (KSampler.SAMPLERS,),# 采样器算法
                "scheduler": (KSampler.SCHEDULERS,), # 调度器
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 15.0}),
                "seed": ("INT", {"default": 42, "min": 0}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "evolution_schedule": ("STRING", {"default": "5,20,30,40"}), # 分段步数
                "steps": ("INT", {"default": 40, "min": 1}),   # 总采样步数
                "population_size": ("INT", {"default": 10, "min": 1}),
                "elite_count": ("INT", {"default": 2, "min": 1}),
            }
        }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent_out",)
    CATEGORY = "EvoSearch"
    FUNCTION = "evo_search"

    def evo_search(self, model, positive, negative, sampler_name, scheduler, cfg, seed, denoise, evolution_schedule, steps, population_size, elite_count):
        # 解析演化调度 (如 "5,20,30,40")
        try:
            schedule = sorted([int(x) for x in evolution_schedule.split(",")])
        except:
            schedule = [0, steps]
        if schedule[-1] < steps:
            schedule.append(steps)

        # 初始化噪声潜在种群
        latent_pop = []
        # 假定 Flux 模型产生的潜在为 4 通道，大小可根据模型调整（此处以 64x64 为例）
        for i in range(population_size):
            noise = torch.randn(1, 4, 64, 64)  # 随机噪声
            latent_pop.append({"samples": noise})

        best_score = -float('inf')
        best_latent = {"samples": torch.zeros(1,4,64,64)}

        # 按照调度阶段循环演化
        for gen in range(len(schedule)):
            # 当前阶段步数
            if gen == 0:
                stage_steps = schedule[0]
            else:
                stage_steps = schedule[gen] - schedule[gen-1]
            # 对每个个体进行 KSampler 采样
            new_pop = []
            scores = []
            for latent in latent_pop:
                out = KSampler.common_ksampler(
                    model=model, seed=seed, steps=stage_steps, cfg=cfg,
                    sampler_name=sampler_name, scheduler=scheduler,
                    positive=positive, negative=negative,
                    latent=latent, denoise=denoise
                )
                new_pop.append(out)
                # 计算评分（可选用美学或CLIP匹配评估）
                try:
                    # 假设out["samples"]可传入评估函数
                    score = aesthetic_score(out["samples"])
                except:
                    # 失败时采用0分
                    score = 0.0
                scores.append(score)
                # 更新最佳 latent
                if score > best_score:
                    best_score = score
                    best_latent = out

            # 选择精英
            if scores:
                top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:elite_count]
                elites = [new_pop[i] for i in top_idx]
            else:
                elites = new_pop[:elite_count]

            # 生成下一代：保留精英并对其做微扰
            latent_pop = elites.copy()
            while len(latent_pop) < population_size:
                for elite in elites:
                    # 对精英进行随机扰动生成新个体
                    mut_noise = torch.randn_like(elite["samples"]) * 0.02
                    mutated = {"samples": elite["samples"] + mut_noise}
                    latent_pop.append(mutated)
                    if len(latent_pop) >= population_size:
                        break

        # 返回得分最高的潜在向量
        return (best_latent,)


from comfy.sample import common_ksampler

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


# 注册节点
NODE_CLASS_MAPPINGS = {
    "EvoSearch_FLUX": EvoSearch_FLUX,
    "EvoSearch_SD21": EvoSearch_SD21,
    "EvoSearch_WAN": EvoSearch_WAN,
}