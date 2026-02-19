# sampling_patch.py — 纯采样侧改进, 零训练改动
# ═══════════════════════════════════════════════════════════════
# 
# 设计原则: 你原始代码训练到 Acc=0.69 是正确的
# 问题只在采样质量, 所以 **只改采样, 不碰训练**
#
# 使用方法:
#   1. 保持你原始的 common_dpm.py 和 mDPM.py 完全不变
#   2. 训练完成后, 用这个文件的函数来生成更好的图像
#   
#   或者: 在 mDPM.py 的训练循环中加入 EMA 更新 (只加一行)
#
# 包含:
#   - EMA 类 (训练时维护影子权重, 不影响梯度)
#   - DDIM 采样器 (50步替代1000步)
#   - 伪 CFG (不需要重新训练, 用 K 个 class 的平均作为 unconditional)
# ═══════════════════════════════════════════════════════════════

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


# ============================================================
# 1. EMA — 完全不影响训练
# ============================================================
class EMA:
    """
    指数移动平均.
    
    用法 (在你原始的训练循环里只需加两处):
    
        # 训练开始前:
        ema = EMA(model, decay=0.9999)
        
        # 每个 batch 的 optimizer.step() 之后:
        ema.update(model)
        
        # 采样时:
        ema_denoiser = ema.get_model().cond_denoiser
        
    EMA 不影响:
        - model 的梯度计算
        - optimizer 的更新
        - E-step / M-step 的任何逻辑
    它只是在旁边维护一份"平滑过的权重副本"
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)
    
    def get_model(self):
        return self.shadow


# ============================================================
# 2. 改进采样: DDIM + 伪 CFG
# ============================================================

@torch.no_grad()
def sample_improved(denoiser, dpm_process, cfg, class_id, n_samples=10,
                    cluster_mapping=None, method='ddim',
                    guidance_scale=1.5, ddim_steps=100, eta=0.0):
    """
    改进采样函数 — 替代原始的 sample_and_save 中的采样逻辑
    
    参数:
        denoiser: 可以是原始 model.cond_denoiser 或 ema.get_model().cond_denoiser
        dpm_process: model.dpm_process
        cfg: Config 对象
        class_id: 要生成的目标数字 (0-9)
        cluster_mapping: 聚类到真实标签的映射
        method: 'ddpm' (原始) / 'ddim' (推荐) / 'ddim_cfg' (DDIM+伪CFG)
        guidance_scale: 伪 CFG 强度 (仅 ddim_cfg 模式)
            1.0 = 无 guidance
            1.5 = 温和 (推荐起点)
            2.0 = 中等
            3.0+ = 强 (可能过饱和)
        ddim_steps: DDIM 步数 (50-200, 推荐 100)
        eta: DDIM 噪声 (0=确定性, 1=等价DDPM)
    """
    T = dpm_process.timesteps
    K = cfg.num_classes
    device = cfg.device
    denoiser.eval()
    
    # 找到对应的 cluster id
    gen_k = class_id
    if cluster_mapping:
        for ck, tk in cluster_mapping.items():
            if tk == class_id:
                gen_k = ck
                break
    
    y_cond = F.one_hot(torch.full((n_samples,), gen_k, dtype=torch.long), K).float().to(device)
    x = torch.randn(n_samples, cfg.image_channels, 28, 28, device=device)
    
    if method == 'ddpm':
        return _sample_ddpm(denoiser, dpm_process, x, y_cond, T)
    elif method == 'ddim':
        return _sample_ddim(denoiser, dpm_process, x, y_cond, T, K, ddim_steps, eta)
    elif method == 'ddim_cfg':
        return _sample_ddim_pseudo_cfg(denoiser, dpm_process, x, y_cond, T, K,
                                        ddim_steps, eta, guidance_scale)
    else:
        raise ValueError(f"Unknown method: {method}")


def _sample_ddpm(denoiser, dpm_process, x, y_cond, T):
    """原始 DDPM 采样 (1000步)"""
    for t_idx in reversed(range(T)):
        n = x.shape[0]
        t_ = torch.full((n,), t_idx, device=x.device, dtype=torch.long)
        pred = denoiser(x, t_, y_cond)
        beta = dpm_process.betas[t_idx]
        alpha = dpm_process.alphas[t_idx]
        alpha_bar = dpm_process.alphas_cumprod[t_idx]
        x = (1.0 / alpha.sqrt()) * (x - beta / (1 - alpha_bar).sqrt() * pred)
        if t_idx > 0:
            x = x + beta.sqrt() * torch.randn_like(x)
    return x


def _sample_ddim(denoiser, dpm_process, x, y_cond, T, K, ddim_steps, eta):
    """DDIM 采样 — 更少步数, 更少累积误差"""
    # 均匀间隔的时间步
    step_indices = torch.linspace(0, T - 1, ddim_steps + 1).long()
    timesteps = step_indices.flip(0)  # T-1 → 0
    
    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i]
        t_prev = timesteps[i + 1]
        n = x.shape[0]
        
        t_ = torch.full((n,), t_cur.item(), device=x.device, dtype=torch.long)
        pred_noise = denoiser(x, t_, y_cond)
        
        alpha_bar_t = dpm_process.alphas_cumprod[t_cur]
        alpha_bar_prev = dpm_process.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        # 预测 x_0
        pred_x0 = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
        pred_x0 = pred_x0.clamp(-1, 1)
        
        # DDIM 方向
        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
        dir_xt = (1 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise
        
        x = alpha_bar_prev.sqrt() * pred_x0 + dir_xt
        if sigma > 0 and t_prev > 0:
            x = x + sigma * torch.randn_like(x)
    
    return x


def _sample_ddim_pseudo_cfg(denoiser, dpm_process, x, y_cond, T, K,
                             ddim_steps, eta, guidance_scale):
    """
    DDIM + 伪 Classifier-Free Guidance
    
    ★ 核心思路: 不需要重新训练!
    
    标准 CFG 需要训练时 drop class → 学 unconditional 分支
    但我们可以用 "所有 class 的平均预测" 近似 unconditional:
    
        pred_uncond ≈ (1/K) * Σ_k pred(x_t, t, class=k)
    
    这在你的模型中是合理的, 因为:
    - 你的 π 固定为 uniform 1/K
    - 所有 class 等概率 → 平均 ≈ 边际分布 ≈ unconditional
    
    然后做 guidance:
        pred_final = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    
    代价: 每步需要 K+1 次 forward (慢 K 倍)
    技巧: 只在前 50% 的步做 CFG (后面噪声小, 不需要)
    """
    step_indices = torch.linspace(0, T - 1, ddim_steps + 1).long()
    timesteps = step_indices.flip(0)
    n = x.shape[0]
    
    # 预计算所有 class 的 one-hot
    all_y_ohs = [F.one_hot(torch.full((n,), k, dtype=torch.long), K).float().to(x.device) 
                  for k in range(K)]
    
    cfg_cutoff = len(timesteps) // 2  # 只在前 50% 的步做 CFG (节省时间)
    
    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i]
        t_prev = timesteps[i + 1]
        
        t_ = torch.full((n,), t_cur.item(), device=x.device, dtype=torch.long)
        
        # 条件预测
        pred_cond = denoiser(x, t_, y_cond)
        
        if i < cfg_cutoff and guidance_scale > 1.0:
            # 伪 unconditional: 所有 class 的平均
            pred_sum = torch.zeros_like(pred_cond)
            for k in range(K):
                pred_sum += denoiser(x, t_, all_y_ohs[k])
            pred_uncond = pred_sum / K
            
            # CFG
            pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        else:
            pred_noise = pred_cond
        
        alpha_bar_t = dpm_process.alphas_cumprod[t_cur]
        alpha_bar_prev = dpm_process.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        pred_x0 = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
        pred_x0 = pred_x0.clamp(-1, 1)
        
        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
        dir_xt = (1 - alpha_bar_prev - sigma ** 2).clamp(min=0).sqrt() * pred_noise
        
        x = alpha_bar_prev.sqrt() * pred_x0 + dir_xt
        if sigma > 0 and t_prev > 0:
            x = x + sigma * torch.randn_like(x)
    
    return x


# ============================================================
# 3. 完整的采样+保存函数
# ============================================================

@torch.no_grad()
def sample_and_save_improved(model, cfg, out_path, n_per_class=10,
                              cluster_mapping=None,
                              use_ema_denoiser=None,
                              method='ddim', guidance_scale=1.5,
                              ddim_steps=100):
    """
    直接替代你原始 mDPM.py 里的 sample_and_save
    
    用法:
        from sampling_patch import EMA, sample_and_save_improved
        
        # 用 EMA 权重 + DDIM 采样:
        sample_and_save_improved(
            model, cfg, "output.png",
            cluster_mapping=mapping,
            use_ema_denoiser=ema.get_model().cond_denoiser,
            method='ddim'
        )
    """
    denoiser = use_ema_denoiser if use_ema_denoiser is not None else model.cond_denoiser
    denoiser.eval()
    
    K = cfg.num_classes
    all_imgs = []
    
    for k in range(K):
        imgs = sample_improved(
            denoiser, model.dpm_process, cfg,
            class_id=k, n_samples=n_per_class,
            cluster_mapping=cluster_mapping,
            method=method,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps
        )
        all_imgs.append(imgs.cpu())
    
    all_imgs = torch.cat(all_imgs, dim=0)
    save_image(all_imgs, out_path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
    print(f"  ✓ [{method}] samples → {out_path}")


# ============================================================
# 4. 集成指南 (读这里!)
# ============================================================
"""
═══════════════════════════════════════════════════════════════
                    集成到你原始 mDPM.py 的方法
═══════════════════════════════════════════════════════════════

你只需要在原始 mDPM.py 中做 3 处改动 (标记为 [PATCH]):

────────────────────────────────────────
改动 1: 导入 (文件顶部)
────────────────────────────────────────

    from sampling_patch import EMA, sample_and_save_improved

────────────────────────────────────────
改动 2: 在 run_training() 中初始化 EMA 并每步更新
────────────────────────────────────────

    def run_training(model, optimizer, unlabeled_loader, val_loader, cfg, ...):
        ...
        # [PATCH] 初始化 EMA
        ema = EMA(model, decay=0.9999)
        
        for epoch in range(1, total_epochs + 1):
            ...
            for x_batch, _ in unlabeled_loader:
                ...
                optimizer.step()
                
                # [PATCH] 更新 EMA (加在 optimizer.step() 后面, 就这一行)
                if epoch >= 5:  # 前几个 epoch 先让模型学
                    ema.update(model)
                
                ...

────────────────────────────────────────
改动 3: 采样时用 EMA + DDIM
────────────────────────────────────────

    # 替换原始的 sample_and_save 调用:
    
    # 原始:
    # sample_and_save(model, cfg, path, cluster_mapping=mapping)
    
    # 改为:
    ema_denoiser = ema.get_model().cond_denoiser
    
    # 方式A: EMA + DDIM (推荐, 最安全)
    sample_and_save_improved(
        model, cfg, "samples_ema_ddim.png",
        cluster_mapping=mapping,
        use_ema_denoiser=ema_denoiser,
        method='ddim', ddim_steps=100
    )
    
    # 方式B: EMA + DDIM + 伪 CFG (更清晰, 但慢 10 倍)
    sample_and_save_improved(
        model, cfg, "samples_ema_ddim_cfg.png",
        cluster_mapping=mapping,
        use_ema_denoiser=ema_denoiser,
        method='ddim_cfg', guidance_scale=1.5, ddim_steps=100
    )
    
    # 方式C: EMA + 原始 DDPM (对比用)
    sample_and_save_improved(
        model, cfg, "samples_ema_ddpm.png",
        cluster_mapping=mapping,
        use_ema_denoiser=ema_denoiser,
        method='ddpm'
    )

就这些。不改 common_dpm.py, 不改 E-step, 不改 M-step, 
不改 ResidualBlock, 不改 Config, 不改任何超参数。
训练过程和你原始到 Acc=0.69 的完全一样。

═══════════════════════════════════════════════════════════════
"""