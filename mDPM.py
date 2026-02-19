# mDPM_improved.py — 改进版: EMA + CFG + 更好的采样
# ═══════════════════════════════════════════════════════════════
# 改动总结 (只改生成质量, 不破坏聚类):
#
# 1. [NEW] EMA — 采样/评估用 EMA 权重 (生成质量 #1 因素)
# 2. [NEW] CFG — 训练 10% drop class, 采样时 guidance_scale=2.0
# 3. [NEW] DDIM 采样 — 50步快速采样, 质量更好
# 4. [MOD] base_channels 64 (容量翻倍)
# 5. [保留] 在线 EM 框架, π 固定, sin 时间权重 — 全部不动
# ═══════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as NMI
from torchvision.utils import save_image

# ★ 导入改进版
from common_dpm import *


# ============================================================
# 0. Utilities
# ============================================================
def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed locked to {seed}")


# ============================================================
# 1. Model — 在线 EM (聚类框架不变, 增加 CFG 训练)
# ============================================================
class mDPM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cond_denoiser = ConditionalUnet(
            in_channels=cfg.image_channels,
            base_channels=cfg.unet_base_channels,
            num_classes=cfg.num_classes,
            time_emb_dim=cfg.unet_time_emb_dim
        )
        self.dpm_process = DPMForwardProcess(
            timesteps=cfg.timesteps,
            schedule='linear',
            image_channels=cfg.image_channels
        )
        self.K = cfg.num_classes
        self.register_buffer('pi', torch.ones(cfg.num_classes) / cfg.num_classes)

        # [NEW] CFG dropout 概率
        self.cfg_dropout_prob = getattr(cfg, 'cfg_dropout_prob', 0.1)

    def estimate_posterior_logits(self, x_0, cfg, scale_factor=1.0):
        """E-step: 不变"""
        B = x_0.size(0)
        K = self.K
        M = getattr(cfg, 'posterior_sample_steps', 5)
        device = x_0.device

        accum_neg_mse = torch.zeros(B, K, device=device)

        with torch.no_grad():
            for _ in range(M):
                t = torch.randint(100, 900, (B,), device=device).long()
                noise = torch.randn_like(x_0)
                x_t = self.dpm_process.q_sample(x_0, t, noise)

                for k in range(K):
                    y_oh = F.one_hot(torch.full((B,), k, device=device,
                                                 dtype=torch.long), K).float()
                    pred = self.cond_denoiser(x_t, t, y_oh)
                    mse = F.mse_loss(pred, noise, reduction='none').view(B, -1).mean(dim=1)
                    accum_neg_mse[:, k] += -mse

            avg_neg_mse = accum_neg_mse / M
            log_pi = torch.log(self.pi.clamp(min=1e-6)).unsqueeze(0)
            logits = log_pi + avg_neg_mse * scale_factor

        return logits

    def forward(self, x_0, cfg, scale_factor=1.0,
                use_hard_label=False, threshold=0.0,
                enable_cfg_dropout=False):
        """
        在线 EM + CFG dropout (仅 REFINE 阶段启用)
        
        关键: EXPLORE 阶段伪标签是噪声, 此时 drop class 会让模型
        学到 "class = 无用信息" → conditioning 死亡 → 聚类坍缩
        只有 REFINE 阶段伪标签有意义后, 才开启 CFG dropout
        """
        B = x_0.size(0)

        # E-step (不变)
        logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=scale_factor)
        resp = F.softmax(logits, dim=1)

        if use_hard_label:
            max_probs, pseudo_labels = resp.max(dim=1)
            mask = (max_probs >= threshold).float()
            y_target = F.one_hot(pseudo_labels, num_classes=self.K).float()
        else:
            pseudo_labels = torch.multinomial(resp, 1).squeeze(1)
            y_target = F.one_hot(pseudo_labels, num_classes=self.K).float()
            mask = torch.ones(B, device=x_0.device)

        # M-step: 训练 denoiser
        t_train = torch.randint(0, cfg.timesteps, (B,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.dpm_process.q_sample(x_0, t_train, noise)

        # [FIX] CFG dropout 仅在 REFINE 阶段 + enable_cfg_dropout=True 时启用
        # EXPLORE 阶段绝不 drop, 保护 conditioning 建立
        if self.training and enable_cfg_dropout and self.cfg_dropout_prob > 0:
            drop_mask = torch.rand(B, device=x_0.device) < self.cfg_dropout_prob
            y_target[drop_mask] = 0.0

        pred_noise = self.cond_denoiser(x_t, t_train, y_target)
        loss_per = F.mse_loss(pred_noise, noise, reduction='none').view(B, -1).mean(dim=1)
        dpm_loss = (loss_per * mask).sum() / (mask.sum() + 1e-8)

        mask_rate = mask.mean().item()

        return dpm_loss, {
            'dpm_loss': dpm_loss.item(),
            'mask_rate': mask_rate,
            'pseudo_labels': pseudo_labels.detach(),
        }


# ============================================================
# 2. Evaluation (不变, 但可选用 EMA 模型)
# ============================================================
@torch.no_grad()
def evaluate_model(model, loader, cfg, use_ema_denoiser=None):
    """
    use_ema_denoiser: 如果传入 EMA 的 denoiser, 用它来评估
    """
    model.eval()
    denoiser = use_ema_denoiser if use_ema_denoiser is not None else model.cond_denoiser
    denoiser.eval()

    preds, ys_true = [], []
    eval_t = cfg.timesteps // 2
    n_repeats = 3

    for x_0, y_true in loader:
        x_0 = x_0.to(cfg.device)
        B = x_0.size(0)
        cumulative_mse = torch.zeros(B, cfg.num_classes, device=cfg.device)

        for _ in range(n_repeats):
            noise = torch.randn_like(x_0)
            t = torch.full((B,), eval_t, device=cfg.device, dtype=torch.long)
            x_t = model.dpm_process.q_sample(x_0, t, noise)
            for k in range(cfg.num_classes):
                y_oh = F.one_hot(torch.full((B,), k, device=x_0.device,
                                             dtype=torch.long), cfg.num_classes).float()
                pred = denoiser(x_t, t, y_oh)
                mse = F.mse_loss(pred, noise, reduction='none').view(B, -1).mean(dim=1)
                cumulative_mse[:, k] += mse

        pred_cluster = torch.argmin(cumulative_mse, dim=1).cpu().numpy()
        preds.append(pred_cluster)
        ys_true.append(y_true.numpy())

    preds = np.concatenate(preds)
    ys_true = np.concatenate(ys_true)

    K = cfg.num_classes
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = -np.sum((ys_true == i) & (preds == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    cluster2label = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned_preds = np.array([cluster2label.get(p, 0) for p in preds])
    acc = np.mean(aligned_preds == ys_true)
    nmi_score = NMI(ys_true, preds)

    freq = np.bincount(preds, minlength=K).astype(float)
    freq = freq / freq.sum()

    return acc, cluster2label, nmi_score, freq


# ============================================================
# 3. Conditioning Diagnostic (不变)
# ============================================================
@torch.no_grad()
def conditioning_diagnostic(model, loader, cfg, n_batches=3):
    model.eval()
    T = cfg.timesteps
    test_timesteps = [T // 10, T * 3 // 10, T * 6 // 10, T * 8 // 10]
    K = cfg.num_classes

    results = {}
    batches_seen = 0

    for x_0, _ in loader:
        if batches_seen >= n_batches:
            break
        x_0 = x_0.to(cfg.device)
        B = x_0.size(0)

        for t_val in test_timesteps:
            t = torch.full((B,), t_val, device=cfg.device, dtype=torch.long)
            noise = torch.randn_like(x_0)
            x_t = model.dpm_process.q_sample(x_0, t, noise)

            mse_per_k = []
            for k in range(K):
                y_oh = F.one_hot(torch.full((B,), k, device=cfg.device,
                                             dtype=torch.long), K).float()
                pred = model.cond_denoiser(x_t, t, y_oh)
                mse = (pred - noise).pow(2).view(B, -1).mean(1).mean().item()
                mse_per_k.append(mse)

            mse_arr = np.array(mse_per_k)
            diff = mse_arr.max() - mse_arr.min()
            avg = mse_arr.mean()
            ratio = diff / (avg + 1e-9)

            if t_val not in results:
                results[t_val] = []
            results[t_val].append(ratio)

        batches_seen += 1

    avg_ratios = {}
    for t_val, ratios in results.items():
        avg_ratios[t_val] = np.mean(ratios)

    return avg_ratios


# ============================================================
# 4. [NEW] CFG 采样 + DDIM
# ============================================================

@torch.no_grad()
def sample_cfg_ddpm(denoiser, dpm_process, cfg, class_id, n_samples=10,
                    guidance_scale=2.0, cluster_mapping=None):
    """
    [NEW] Classifier-Free Guidance DDPM 采样
    
    pred_final = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
    
    guidance_scale=1.0 → 无 guidance (退化为原始)
    guidance_scale=2.0 → 温和 guidance (推荐)
    guidance_scale=3.0+ → 强 guidance (更清晰但多样性低)
    """
    T = dpm_process.timesteps
    K = cfg.num_classes
    device = cfg.device
    denoiser.eval()

    # 确定生成哪个 cluster
    gen_k = class_id
    if cluster_mapping:
        for ck, tk in cluster_mapping.items():
            if tk == class_id:
                gen_k = ck
                break

    # 条件 one-hot
    y_cond = F.one_hot(torch.full((n_samples,), gen_k, dtype=torch.long),
                       K).float().to(device)
    # 无条件 (全零)
    y_uncond = torch.zeros(n_samples, K, device=device)

    x = torch.randn(n_samples, cfg.image_channels, 28, 28, device=device)

    for t_idx in reversed(range(T)):
        t_ = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)

        # 条件预测 + 无条件预测
        pred_cond = denoiser(x, t_, y_cond)
        pred_uncond = denoiser(x, t_, y_uncond)

        # CFG 公式
        pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        beta = dpm_process.betas[t_idx]
        alpha = dpm_process.alphas[t_idx]
        alpha_bar = dpm_process.alphas_cumprod[t_idx]

        x = (1.0 / alpha.sqrt()) * (x - beta / (1 - alpha_bar).sqrt() * pred)
        if t_idx > 0:
            x = x + beta.sqrt() * torch.randn_like(x)

    return x


@torch.no_grad()
def sample_cfg_ddim(denoiser, dpm_process, cfg, class_id, n_samples=10,
                    guidance_scale=2.0, ddim_steps=50, eta=0.0,
                    cluster_mapping=None):
    """
    [NEW] DDIM 采样 + CFG
    
    DDIM 优势:
    - 50步 ≈ 1000步 DDPM 的质量
    - eta=0 → 确定性采样 (更干净)
    - eta=1 → 等价于 DDPM
    """
    T = dpm_process.timesteps
    K = cfg.num_classes
    device = cfg.device
    denoiser.eval()

    gen_k = class_id
    if cluster_mapping:
        for ck, tk in cluster_mapping.items():
            if tk == class_id:
                gen_k = ck
                break

    y_cond = F.one_hot(torch.full((n_samples,), gen_k, dtype=torch.long),
                       K).float().to(device)
    y_uncond = torch.zeros(n_samples, K, device=device)

    # DDIM 时间步子集 (均匀间隔)
    step_indices = torch.linspace(0, T - 1, ddim_steps + 1).long()
    timesteps = step_indices.flip(0)  # 从 T-1 到 0

    x = torch.randn(n_samples, cfg.image_channels, 28, 28, device=device)

    for i in range(len(timesteps) - 1):
        t_cur = timesteps[i]
        t_prev = timesteps[i + 1]

        t_ = torch.full((n_samples,), t_cur.item(), device=device, dtype=torch.long)

        # CFG
        pred_cond = denoiser(x, t_, y_cond)
        pred_uncond = denoiser(x, t_, y_uncond)
        pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        # DDIM 更新
        alpha_bar_t = dpm_process.alphas_cumprod[t_cur]
        alpha_bar_prev = dpm_process.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # 预测 x_0
        pred_x0 = (x - (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_bar_t.sqrt()
        pred_x0 = pred_x0.clamp(-1, 1)  # 稳定性

        # 方向
        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
        dir_xt = (1 - alpha_bar_prev - sigma ** 2).sqrt() * pred_noise

        x = alpha_bar_prev.sqrt() * pred_x0 + dir_xt
        if sigma > 0 and t_prev > 0:
            x = x + sigma * torch.randn_like(x)

    return x


@torch.no_grad()
def sample_and_save(model, cfg, out_path, n_per_class=10, cluster_mapping=None,
                    use_ema_denoiser=None, use_cfg=True, use_ddim=True):
    """
    改进版采样:
    - 优先使用 EMA denoiser
    - 支持 CFG + DDIM
    """
    denoiser = use_ema_denoiser if use_ema_denoiser is not None else model.cond_denoiser
    denoiser.eval()

    guidance_scale = getattr(cfg, 'cfg_guidance_scale', 2.0)
    K = cfg.num_classes
    all_imgs = []

    for k in range(K):
        if use_ddim and use_cfg:
            imgs = sample_cfg_ddim(
                denoiser, model.dpm_process, cfg,
                class_id=k, n_samples=n_per_class,
                guidance_scale=guidance_scale,
                ddim_steps=50, eta=0.0,
                cluster_mapping=cluster_mapping
            )
        elif use_cfg:
            imgs = sample_cfg_ddpm(
                denoiser, model.dpm_process, cfg,
                class_id=k, n_samples=n_per_class,
                guidance_scale=guidance_scale,
                cluster_mapping=cluster_mapping
            )
        else:
            # 原始采样 (fallback)
            imgs = sample_cfg_ddpm(
                denoiser, model.dpm_process, cfg,
                class_id=k, n_samples=n_per_class,
                guidance_scale=1.0,  # 无 guidance
                cluster_mapping=cluster_mapping
            )
        all_imgs.append(imgs.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)
    save_image(all_imgs, out_path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
    print(f"  ✓ samples → {out_path}")


# ============================================================
# 5. Dashboard (不变)
# ============================================================
def plot_dashboard(history, outpath):
    n = len(history.get("loss", []))
    if n == 0:
        return
    epochs = range(1, n + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Online EM + EMA + CFG (Ep {n})', fontsize=18)

    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], 'b-')
    ax.set_title('DPM Loss'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["acc"], 'r-', label='Acc')
    if "nmi" in history:
        ax.plot(epochs, history["nmi"], 'g--', label='NMI')
    ax.set_title('Acc & NMI'); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 2]
    if "pi_entropy" in history and len(history["pi_entropy"]) > 0:
        ax.plot(epochs, history["pi_entropy"], 'purple', marker='.')
        ax.axhline(y=np.log(10), color='gray', linestyle='--', alpha=0.5, label='uniform')
        ax.legend()
    ax.set_title('π Entropy'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["scale"], 'c-')
    ax.set_title('Scale Factor'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "cond_ratio" in history and len(history["cond_ratio"]) > 0:
        ax.plot(range(1, len(history["cond_ratio"])+1),
                history["cond_ratio"], 'orange', marker='o')
        ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='healthy')
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='dead')
        ax.set_title('Conditioning Ratio'); ax.legend()
    else:
        ax.set_title('Conditioning Ratio (pending)')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    if "freq" in history and len(history["freq"]) > 0:
        freq_arr = np.array(history["freq"])
        for kk in range(freq_arr.shape[1]):
            ax.plot(range(1, len(history["freq"])+1), freq_arr[:, kk], alpha=0.6)
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Class Frequencies'); ax.set_ylim(0, 0.5)
    else:
        ax.set_title('Class Frequencies (pending)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()


# ============================================================
# 6. Training — 在线 EM + EMA
# ============================================================
def run_training(model, optimizer, unlabeled_loader, val_loader, cfg,
                 hyperparams=None, is_final=True):
    if hyperparams is None:
        hyperparams = {}

    total_epochs = hyperparams.get('total_epochs', 60)
    target_scale = hyperparams.get('target_scale', 134.0)
    warmup_epochs = hyperparams.get('warmup_epochs', 10)
    threshold_final = hyperparams.get('threshold_final', 0.036)

    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    # [NEW] 初始化 EMA
    ema = EMA(model, decay=getattr(cfg, 'ema_decay', 0.9999))
    ema_start = getattr(cfg, 'ema_start_epoch', 5)

    best_val_acc = 0.0
    best_cluster_mapping = None

    history = {"loss": [], "acc": [], "nmi": [],
               "pass_rate": [], "scale": [],
               "cond_ratio": [], "freq": [], "pi_entropy": []}

    for epoch in range(1, total_epochs + 1):

        # Scale 调度 (不变)
        if epoch <= warmup_epochs:
            use_hard = False
            p1 = epoch / warmup_epochs
            dynamic_scale = 5.0 + (20.0 - 5.0) * p1
            dynamic_threshold = 0.0
            status = "EXPLORE"
        else:
            use_hard = True
            p2 = (epoch - warmup_epochs) / (total_epochs - warmup_epochs + 1e-8)
            dynamic_scale = 20.0 + (target_scale - 20.0) * p2
            dynamic_threshold = threshold_final * p2
            status = "REFINE"

        if is_final:
            print(f"🔥 [Ep {epoch}/{total_epochs}] [{status}] "
                  f"Scale={dynamic_scale:.1f} Thres={dynamic_threshold:.3f}")

        # Training
        model.train()
        ep_loss = 0.0
        ep_mask = 0.0
        n_batches = 0
        epoch_labels = []

        for x_batch, _ in unlabeled_loader:
            x_batch = x_batch.to(cfg.device)
            optimizer.zero_grad()

            loss, info = model(x_batch, cfg,
                               scale_factor=dynamic_scale,
                               use_hard_label=use_hard,
                               threshold=dynamic_threshold,
                               enable_cfg_dropout=use_hard)  # [FIX] 仅 REFINE 阶段开 CFG dropout

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # [NEW] 更新 EMA (在 ema_start_epoch 之后)
            if epoch >= ema_start:
                ema.update(model)

            ep_loss += loss.item()
            ep_mask += info['mask_rate']
            epoch_labels.append(info['pseudo_labels'].cpu())
            n_batches += 1

        # Epoch 统计
        avg_loss = ep_loss / max(n_batches, 1)
        pass_pct = (ep_mask / max(n_batches, 1)) * 100

        all_labels = torch.cat(epoch_labels)
        label_freq = np.bincount(all_labels.numpy(), minlength=cfg.num_classes).astype(float)
        label_freq = label_freq / label_freq.sum()
        pi_entropy = -np.sum(label_freq * np.log(label_freq + 1e-9))

        if is_final:
            freq_str = ", ".join([f"{f:.3f}" for f in label_freq])
            print(f"   freq=[{freq_str}] H={pi_entropy:.3f}")

        # Validation (用 EMA denoiser 评估)
        ema_denoiser = ema.get_model().cond_denoiser if epoch >= ema_start else None
        val_acc, cluster_mapping, val_nmi, val_freq = evaluate_model(
            model, val_loader, cfg, use_ema_denoiser=ema_denoiser)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cluster_mapping = cluster_mapping
            # 保存原始模型 + EMA 模型
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'acc': val_acc, 'nmi': val_nmi,
                'cluster_mapping': cluster_mapping,
            }
            if epoch >= ema_start:
                save_dict['ema_state_dict'] = ema.get_model().state_dict()
            torch.save(save_dict, os.path.join(cfg.output_dir, "best_model.pt"))
            if is_final:
                print(f"   ★ New Best! Acc={best_val_acc:.4f}")

        # Conditioning Diagnostic
        if epoch % 5 == 0 or epoch == 1:
            cond_ratios = conditioning_diagnostic(model, val_loader, cfg)
            avg_ratio = np.mean(list(cond_ratios.values()))
            history["cond_ratio"].append(avg_ratio)
            if is_final:
                ratio_str = " | ".join([f"t={t}: {r:.4f}" for t, r in sorted(cond_ratios.items())])
                print(f"   [Cond] {ratio_str} | avg={avg_ratio:.4f}")
        else:
            if len(history["cond_ratio"]) > 0:
                history["cond_ratio"].append(history["cond_ratio"][-1])
            else:
                history["cond_ratio"].append(0.0)

        # History
        history["loss"].append(avg_loss)
        history["acc"].append(val_acc)
        history["nmi"].append(val_nmi)
        history["pass_rate"].append(pass_pct)
        history["scale"].append(dynamic_scale)
        history["freq"].append(label_freq.tolist())
        history["pi_entropy"].append(pi_entropy)

        if is_final:
            print(f"  → Loss={avg_loss:.4f} | Acc={val_acc:.4f} NMI={val_nmi:.4f} "
                  f"| Pass={pass_pct:.1f}%")

            plot_dashboard(history, os.path.join(cfg.output_dir, "dashboard.png"))

            # [NEW] 采样用 EMA + CFG + DDIM
            if epoch % 5 == 0:
                ema_den = ema.get_model().cond_denoiser if epoch >= ema_start else None
                sample_and_save(
                    model, cfg,
                    os.path.join(sample_dir, f"epoch_{epoch:03d}.png"),
                    cluster_mapping=cluster_mapping,
                    use_ema_denoiser=ema_den,
                    use_cfg=True,
                    use_ddim=True
                )

    return best_val_acc, best_cluster_mapping, ema


# ============================================================
# 7. Main
# ============================================================
def main():
    set_seed(2026)

    cfg = Config()
    cfg.labeled_per_class = 0
    cfg.posterior_sample_steps = 5
    cfg.output_dir = "./mDPM_results_improved"
    cfg.final_epochs = 80  # [MOD] 多训一些

    # [FIX] 保持 base_channels=32 (原始聚类验证值)
    # 容量太大会让模型忽略 class condition → 聚类坍缩
    # 生成质量主要靠 EMA + CFG, 不靠容量
    cfg.unet_base_channels = 32
    cfg.ema_decay = 0.9999
    cfg.ema_start_epoch = 5
    cfg.cfg_dropout_prob = 0.1        # 10% drop class (仅 REFINE 阶段)
    cfg.cfg_guidance_scale = 2.0      # 采样 guidance 强度

    print("=" * 60)
    print("🔓 Improved: Online EM + EMA + CFG + DDIM")
    print(f"   T={cfg.timesteps}, M={cfg.posterior_sample_steps}")
    print(f"   base_channels={cfg.unet_base_channels} (保持原始值)")
    print(f"   EMA decay={cfg.ema_decay}, start_epoch={cfg.ema_start_epoch}")
    print(f"   CFG dropout={cfg.cfg_dropout_prob} (仅REFINE阶段), guidance={cfg.cfg_guidance_scale}")
    print("=" * 60)

    os.makedirs(cfg.output_dir, exist_ok=True)
    _, unlabeled_loader, val_loader = get_semi_loaders(cfg)

    model = mDPM(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   Model params: {n_params:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=4.01e-05)

    hyperparams = {
        'total_epochs': 80,
        'target_scale': 134.37,
        'warmup_epochs': 10,
        'threshold_final': 0.036,
    }

    print(f"\n🚀 Training:")
    print(f"   LR=4.01e-05, Epochs={hyperparams['total_epochs']}")
    print(f"   Scale: 5→20→134, π=uniform (FIXED)")

    best_acc, best_mapping, ema = run_training(
        model, optimizer, unlabeled_loader, val_loader, cfg,
        hyperparams=hyperparams, is_final=True)

    print(f"\n✅ Done. Best Acc: {best_acc:.4f}")

    # 加载 best model + EMA 生成最终 samples
    best_ckpt = os.path.join(cfg.output_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        best_mapping = ckpt.get('cluster_mapping', best_mapping)
        print(f"   Loaded best model (Acc={ckpt.get('acc', '?'):.4f})")

        # 加载 EMA
        if 'ema_state_dict' in ckpt:
            ema.get_model().load_state_dict(ckpt['ema_state_dict'])
            print("   Loaded EMA weights")

    # 生成最终 samples: 对比有无 CFG
    ema_den = ema.get_model().cond_denoiser

    print("\n📸 Generating final samples...")

    # 1. EMA + CFG + DDIM (推荐)
    sample_and_save(model, cfg,
                    os.path.join(cfg.output_dir, "final_ema_cfg_ddim.png"),
                    cluster_mapping=best_mapping,
                    use_ema_denoiser=ema_den,
                    use_cfg=True, use_ddim=True)

    # 2. EMA + CFG + DDPM (对比)
    sample_and_save(model, cfg,
                    os.path.join(cfg.output_dir, "final_ema_cfg_ddpm.png"),
                    cluster_mapping=best_mapping,
                    use_ema_denoiser=ema_den,
                    use_cfg=True, use_ddim=False)

    # 3. EMA only, no CFG (对比)
    sample_and_save(model, cfg,
                    os.path.join(cfg.output_dir, "final_ema_nocfg.png"),
                    cluster_mapping=best_mapping,
                    use_ema_denoiser=ema_den,
                    use_cfg=False, use_ddim=True)

    # 4. 原始采样 (no EMA, no CFG) — 和你原来一样
    sample_and_save(model, cfg,
                    os.path.join(cfg.output_dir, "final_original.png"),
                    cluster_mapping=best_mapping,
                    use_ema_denoiser=None,
                    use_cfg=False, use_ddim=False)

    # 5. [NEW] 不同 guidance scale 对比
    for gs in [1.5, 2.0, 3.0, 5.0]:
        old_gs = cfg.cfg_guidance_scale
        cfg.cfg_guidance_scale = gs
        sample_and_save(model, cfg,
                        os.path.join(cfg.output_dir, f"final_gs{gs:.1f}.png"),
                        cluster_mapping=best_mapping,
                        use_ema_denoiser=ema_den,
                        use_cfg=True, use_ddim=True)
        cfg.cfg_guidance_scale = old_gs

    # Save config
    cfg_dict = {k: v for k, v in vars(cfg).items()
                if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    cfg_dict['hyperparams'] = hyperparams
    cfg_dict['best_acc'] = best_acc
    json.dump(cfg_dict, open(os.path.join(cfg.output_dir, "config.json"), "w"), indent=2)

    print("\n🎉 All done! Check the output directory for comparison images.")


if __name__ == "__main__":
    main()