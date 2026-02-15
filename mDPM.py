# mDPM.py — Strict EM version
# 借鉴 HMM-DPM 的严格 EM 框架:
#   E-step: 全局计算所有样本的 pseudo-label → cached
#   M-step: 用固定标签训练 N 个 epoch
#   π 更新: 闭式解 (count-based), 不用梯度
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
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
# 1. Model
# ============================================================
class mDPM_StrictEM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        K = cfg.num_classes
        # 构建标准 UNet (in_channels=1, 输出也是1)
        self.cond_denoiser = ConditionalUnet(
            in_channels=cfg.image_channels,      # 输出仍为 1 通道
            base_channels=cfg.unet_base_channels,
            num_classes=cfg.num_classes,
            time_emb_dim=cfg.unet_time_emb_dim
        )
        # [关键] 替换 init_conv: 接受 1+K 通道输入 (image + cond broadcast)
        # 借鉴 HMM-DPM: 条件占输入的 K/(1+K) → 结构上不可忽略
        # 输出层不变 (final_conv 仍输出 1 通道)
        self.cond_denoiser.init_conv = nn.Conv2d(
            cfg.image_channels + K,
            cfg.unet_base_channels,
            3, padding=1
        )
        self.dpm_process = DPMForwardProcess(
            timesteps=cfg.timesteps,
            schedule='linear',
            image_channels=cfg.image_channels
        )
        self.K = K
        self.register_buffer('pi', torch.ones(K) / K)

    def _concat_cond(self, x, y_onehot):
        """将条件 broadcast 到空间维度并拼接到输入
        HMM-DPM 的关键设计: 条件占输入的 K/(1+K) → 结构上不可忽略
        """
        B, C, H, W = x.shape
        cond_map = y_onehot[:, :, None, None].expand(-1, -1, H, W)  # [B, K, H, W]
        return torch.cat([x, cond_map], dim=1)  # [B, 1+K, H, W]

    @torch.no_grad()
    def compute_log_likelihood(self, x_0, cfg, scale_factor=5.0, n_mc=5):
        """
        E-step: 计算 log p(x | k) ∝ -MSE_k
        ★ 方案B: 只用高噪声段 t ∈ [600, 1000)
        诊断: t=500 ratio≈0.28 (有信号), t=50 ratio≈0.006 (无信号)
        低t无区分度, 会淹没高t的真实信号, scale放大后导致坍缩
        """
        B = x_0.size(0)
        K = self.K
        device = x_0.device
        T = self.dpm_process.timesteps

        t_lo = T * 3 // 5   # 600
        t_hi = T             # 1000

        total_neg_mse = torch.zeros(B, K, device=device)

        for _ in range(n_mc):
            t = torch.randint(t_lo, t_hi, (B,), device=device).long()
            noise = torch.randn_like(x_0)
            x_t = self.dpm_process.q_sample(x_0, t, noise)

            for k in range(K):
                y_oh = F.one_hot(torch.full((B,), k, device=device,
                                             dtype=torch.long), K).float()
                pred = self.cond_denoiser(self._concat_cond(x_t, y_oh), t, y_oh)
                mse = F.mse_loss(pred, noise, reduction='none').view(B, -1).mean(dim=1)
                total_neg_mse[:, k] += -mse

        avg_neg_mse = total_neg_mse / n_mc
        log_pi = torch.log(self.pi.clamp(min=1e-6)).unsqueeze(0)
        logits = log_pi + avg_neg_mse * scale_factor
        return logits

    def compute_diffusion_loss(self, x_0, y_onehot):
        """
        M-step: DDPM loss, 偏向高 t 采样 (借鉴 HMM-DPM)
        高 t 时 denoiser 被迫更依赖条件输入 → conditioning 更强
        """
        B = x_0.size(0)
        device = x_0.device
        # 偏向高 t: u~U(0,1), t = u^0.5 * T
        u = torch.rand(B, device=device)
        t = (u.sqrt() * self.dpm_process.timesteps).long().clamp(
            0, self.dpm_process.timesteps - 1)
        noise = torch.randn_like(x_0)
        x_t = self.dpm_process.q_sample(x_0, t, noise)
        pred_noise = self.cond_denoiser(self._concat_cond(x_t, y_onehot), t, y_onehot)
        return F.mse_loss(pred_noise, noise)

    def compute_contrastive_loss(self, x_0, y_onehot, margin=0.01):
        """
        对比 loss (借鉴 HMM-DPM): 正确条件 MSE < 错误条件 MSE
        直接增强 conditioning ratio
        """
        B = x_0.size(0)
        device = x_0.device
        t = torch.randint(0, self.dpm_process.timesteps, (B,), device=device).long()
        noise = torch.randn_like(x_0)
        x_t = self.dpm_process.q_sample(x_0, t, noise)

        # 正确条件
        pred_pos = self.cond_denoiser(self._concat_cond(x_t, y_onehot), t, y_onehot)
        mse_pos = (pred_pos - noise).pow(2).view(B, -1).mean(1)

        # 错误条件: shuffle
        perm = torch.randperm(B, device=device)
        y_neg = y_onehot[perm]
        same = (y_neg.argmax(1) == y_onehot.argmax(1))
        if same.any():
            y_neg[same] = torch.roll(y_onehot[same], 1, dims=0)

        pred_neg = self.cond_denoiser(self._concat_cond(x_t, y_neg), t, y_neg)
        mse_neg = (pred_neg - noise).pow(2).view(B, -1).mean(1)

        contrastive = F.relu(mse_pos - mse_neg + margin).mean()
        return contrastive

    def update_pi_closed_form(self, cached_labels):
        """
        闭式解更新 π (借鉴 HMM-DPM)
        π_k = count(label == k) / N
        天然均衡, 不会坍缩
        """
        counts = torch.bincount(cached_labels.view(-1).long(),
                                minlength=self.K).float()
        counts = counts + 1e-6  # Laplace 平滑
        new_pi = counts / counts.sum()
        self.pi.copy_(new_pi.to(self.pi.device))
        return new_pi.cpu().numpy()


# ============================================================
# 2. Evaluation (固定 t=500, argmin MSE)
# ============================================================
@torch.no_grad()
def evaluate_model(model, val_loader, cfg, n_repeats=3, scale_for_eval=50.0):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in val_loader:
        x = x.to(cfg.device)
        logits = model.compute_log_likelihood(x, cfg, scale_factor=scale_for_eval,
                                               n_mc=n_repeats)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(y)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    K = cfg.num_classes
    cost = np.zeros((K, K))
    for pred_k in range(K):
        for true_k in range(K):
            cost[pred_k, true_k] = -np.sum(
                (all_preds == pred_k) & (all_labels == true_k))
    _, col_ind = linear_sum_assignment(cost)

    mapping = {pred_k: true_k for pred_k, true_k in enumerate(col_ind)}
    mapped = np.array([mapping[p] for p in all_preds])
    acc = np.mean(mapped == all_labels)
    nmi = NMI(all_labels, all_preds)
    return acc, mapping, nmi


# ============================================================
# 3. Global E-step (借鉴 HMM-DPM)
# ============================================================
@torch.no_grad()
def global_estep(model, all_x, cfg, scale_factor=5.0, n_mc=5, batch_size=256):
    """
    全局 E-step: 对所有样本计算 pseudo-label, 缓存结果
    这是 HMM-DPM 成功的关键 — 标签在整个 M-step 期间固定
    """
    model.eval()
    device = cfg.device
    N = all_x.size(0)

    all_labels = []
    all_logits = []
    for i in range(0, N, batch_size):
        x = all_x[i:i+batch_size].to(device)
        logits = model.compute_log_likelihood(x, cfg, scale_factor=scale_factor,
                                               n_mc=n_mc)
        labels = logits.argmax(dim=1).cpu()
        all_labels.append(labels)
        all_logits.append(logits.cpu())

    cached_labels = torch.cat(all_labels)    # [N]
    all_logits_t = torch.cat(all_logits)     # [N, K]

    # 统计
    freq = torch.bincount(cached_labels, minlength=cfg.num_classes).float()
    freq = freq / freq.sum()
    n_active = len(cached_labels.unique())

    # Top-2 gap (denoiser confidence)
    vals, _ = torch.topk(all_logits_t, 2, dim=1)
    gap = (vals[:, 0] - vals[:, 1]).mean().item()

    return cached_labels, freq.numpy(), n_active, gap


@torch.no_grad()
def conditioning_diagnostic(model, data_x, cfg, n_samples=200):
    """诊断 denoiser 对不同 class 的区分能力"""
    model.eval()
    device = cfg.device
    K = cfg.num_classes
    x = data_x[:n_samples].to(device)
    B = x.size(0)

    results = {}
    for t_val in [50, 200, 500, 700, 900]:
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_t = model.dpm_process.q_sample(x, t, noise)

        mse_per_k = []
        for k in range(K):
            y_oh = F.one_hot(torch.full((B,), k, device=device,
                                         dtype=torch.long), K).float()
            pred = model.cond_denoiser(model._concat_cond(x_t, y_oh), t, y_oh)
            mse = (pred - noise).pow(2).view(B, -1).mean(1)  # [B]
            mse_per_k.append(mse.mean().item())

        mse_arr = np.array(mse_per_k)
        diff = mse_arr.max() - mse_arr.min()
        avg = mse_arr.mean()
        ratio = diff / (avg + 1e-8)
        results[t_val] = {'diff': diff, 'avg': avg, 'ratio': ratio}

    return results


# ============================================================
# 4. Dashboard & Sampling
# ============================================================
def plot_dashboard(history, outpath):
    n = len(history["acc"])
    if n == 0:
        return
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle(f"Strict EM Dashboard (Round {n})", fontsize=16, fontweight='bold')
    rounds = list(range(1, n + 1))

    # ── Row 1: Core metrics ──
    ax = axes[0, 0]
    if len(history["loss"]) >= n:
        ax.plot(rounds, history["loss"][:n], 'b-', marker='o', linewidth=2)
    ax.set_title('Diffusion Loss', fontweight='bold')
    ax.set_xlabel('EM Round'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(rounds, history["acc"], 'r-', marker='o', linewidth=2, label='Acc')
    ax.plot(rounds, history["nmi"], 'g--', marker='s', linewidth=2, label='NMI')
    ax.set_title('Clustering Performance', fontweight='bold')
    ax.set_ylim(0, 1); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_xlabel('EM Round')

    ax = axes[0, 2]
    ax.plot(rounds, history["n_active"], 'purple', marker='o', linewidth=2)
    ax.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='K=10')
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='collapse threshold')
    ax.set_title('Active Classes', fontweight='bold')
    ax.set_ylim(0, 12); ax.legend(fontsize=9)
    ax.set_xlabel('EM Round'); ax.grid(True, alpha=0.3)

    # ── Row 2: EM internals ──
    ax = axes[1, 0]
    ax.plot(rounds, history["scale"], 'c-', marker='o', linewidth=2)
    ax.set_title('Scale Factor (annealing)', fontweight='bold')
    ax.set_xlabel('EM Round'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(rounds, history["pi_entropy"], 'orange', marker='o', linewidth=2, label='H(π)')
    ax.axhline(y=np.log(10), color='gray', linestyle='--', alpha=0.5, label='uniform=2.303')
    ax.set_title('π Entropy', fontweight='bold')
    ax.legend(fontsize=10); ax.set_xlabel('EM Round'); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(rounds, history["gap"], 'm-', marker='o', linewidth=2)
    ax.set_title('Emission Gap (top1-top2)', fontweight='bold')
    ax.set_xlabel('EM Round'); ax.grid(True, alpha=0.3)

    # ── Row 3: Conditioning & Freq ──
    ax = axes[2, 0]
    if "cond_ratio" in history and len(history["cond_ratio"]) > 0:
        cr = history["cond_ratio"]
        cr_rounds = list(range(1, len(cr) + 1))
        for t_key in ['t50', 't200', 't500']:
            vals = [c.get(t_key, 0) for c in cr]
            ax.plot(cr_rounds, vals, marker='o', linewidth=2, label=t_key)
        ax.set_title('Conditioning Ratio', fontweight='bold')
        ax.legend(fontsize=9); ax.set_xlabel('EM Round'); ax.grid(True, alpha=0.3)
    else:
        ax.set_title('Conditioning Ratio (N/A)'); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    if "freq_history" in history and len(history["freq_history"]) > 0:
        freq_arr = np.array(history["freq_history"])  # [n_rounds, K]
        for k in range(freq_arr.shape[1]):
            ax.plot(list(range(1, freq_arr.shape[0]+1)), freq_arr[:, k],
                    alpha=0.7, linewidth=1.5, label=f'k={k}')
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Class Frequency Distribution', fontweight='bold')
        ax.set_xlabel('EM Round'); ax.grid(True, alpha=0.3)
        if freq_arr.shape[1] <= 10:
            ax.legend(fontsize=7, ncol=2)
    else:
        ax.set_title('Class Frequency (N/A)'); ax.grid(True, alpha=0.3)

    # Info panel
    ax = axes[2, 2]; ax.axis('off')
    if n > 0:
        pi_str = ""
        if "pi_values" in history and len(history["pi_values"]) > 0:
            pi_str = f"\nπ: [{history['pi_values'][-1]}]"
        info = (f"Current Acc:   {history['acc'][-1]:.4f}\n"
                f"Best Acc:      {max(history['acc']):.4f}\n"
                f"Current NMI:   {history['nmi'][-1]:.4f}\n"
                f"Active Classes: {history['n_active'][-1]}\n"
                f"Scale:         {history['scale'][-1]:.1f}\n"
                f"H(π):          {history['pi_entropy'][-1]:.3f}\n"
                f"Gap:           {history['gap'][-1]:.4f}"
                f"{pi_str}")
        ax.text(0.05, 0.95, info, fontsize=12, family='monospace',
                verticalalignment='top', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(); plt.savefig(outpath, dpi=120); plt.close()


@torch.no_grad()
def sample_and_save(model, cfg, out_path, n_per_class=10, cluster_mapping=None):
    T = model.dpm_process.timesteps
    model.cond_denoiser.eval()
    K = cfg.num_classes
    imgs = []
    for k in range(K):
        mapped_k = k
        if cluster_mapping and k in cluster_mapping:
            inv_map = {v: kk for kk, v in cluster_mapping.items()}
            if k in inv_map:
                mapped_k = inv_map[k]

        y_oh = F.one_hot(torch.full((n_per_class,), mapped_k,
                                     dtype=torch.long), K).float().to(cfg.device)
        x = torch.randn(n_per_class, cfg.image_channels, 28, 28,
                         device=cfg.device)
        for t_idx in reversed(range(T)):
            t_ = torch.full((n_per_class,), t_idx, device=cfg.device,
                             dtype=torch.long)
            pred = model.cond_denoiser(model._concat_cond(x, y_oh), t_, y_oh)
            beta = model.dpm_process.betas[t_idx]
            alpha = model.dpm_process.alphas[t_idx]
            alpha_bar = model.dpm_process.alphas_cumprod[t_idx]
            x = (1.0 / alpha.sqrt()) * (
                x - beta / (1 - alpha_bar).sqrt() * pred)
            if t_idx > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
        imgs.append(x.cpu())
    imgs = torch.cat(imgs, dim=0)
    save_image(imgs, out_path, nrow=n_per_class, normalize=True,
               value_range=(-1, 1))
    print(f"  ✓ samples → {out_path}")


# ============================================================
# 5. Strict EM Training (借鉴 HMM-DPM)
# ============================================================
def run_strict_em(model, optimizer, all_x, val_loader, cfg,
                  hyperparams=None, is_final_training=False):
    """
    严格 EM 训练:
      E-step: 全局计算 pseudo-label, 闭式解更新 π
      M-step: 用固定标签训练 denoiser N 个 epoch
    """
    if hyperparams is None:
        hyperparams = {}

    n_em_rounds = hyperparams.get('n_em_rounds', 8)
    m_epochs_first = hyperparams.get('m_epochs_first', 15)
    m_epochs_rest = hyperparams.get('m_epochs_rest', 8)
    scale_start = hyperparams.get('scale_start', 5.0)     # E-step 起始 scale
    scale_end = hyperparams.get('scale_end', 50.0)         # E-step 最终 scale
    n_mc = hyperparams.get('n_mc', 5)
    use_contrastive = hyperparams.get('use_contrastive', False)
    contrastive_weight = hyperparams.get('contrastive_weight', 0.1)
    use_kmeans_init = hyperparams.get('use_kmeans_init', True)

    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    best_val_acc = 0.0
    best_cluster_mapping = None

    history = {"loss": [], "acc": [], "nmi": [], "scale": [],
               "pi_entropy": [], "n_active": [], "gap": [],
               "cond_ratio": [], "freq_history": [], "pi_values": []}

    # ── KMeans 初始化 (借鉴 HMM-DPM) ──
    if use_kmeans_init:
        print("\n🔑 KMeans initialization...")
        images_flat = all_x.view(all_x.size(0), -1).numpy()
        pca = PCA(n_components=50, random_state=42)
        features = pca.fit_transform(images_flat)
        print(f"   PCA: {images_flat.shape} → {features.shape}, "
              f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        kmeans = KMeans(n_clusters=cfg.num_classes, random_state=42, n_init=10)
        cached_labels = torch.tensor(kmeans.fit_predict(features), dtype=torch.long)

        freq = torch.bincount(cached_labels, minlength=cfg.num_classes).float()
        freq = freq / freq.sum()
        print(f"   KMeans freq: [{freq.min():.3f} - {freq.max():.3f}]")
    else:
        cached_labels = torch.randint(0, cfg.num_classes, (all_x.size(0),))

    # 初始 π
    pi_np = model.update_pi_closed_form(cached_labels)
    print(f"   Initial π: [{pi_np.min():.3f} - {pi_np.max():.3f}]")

    # ── EM 主循环 ──
    for em_round in range(n_em_rounds):
        m_epochs = m_epochs_first if em_round == 0 else m_epochs_rest

        # Scale 退火: 从 scale_start 到 scale_end
        progress = em_round / max(n_em_rounds - 1, 1)
        current_scale = scale_start + (scale_end - scale_start) * progress

        print(f"\n{'='*60}")
        print(f"EM Round {em_round+1}/{n_em_rounds} | Scale={current_scale:.1f}")
        print(f"{'='*60}")

        # ── Conditioning 诊断 ──
        diag = conditioning_diagnostic(model, all_x, cfg)
        cond_ratio_entry = {f't{t_val}': d['ratio'] for t_val, d in diag.items()}
        history["cond_ratio"].append(cond_ratio_entry)

        if is_final_training:
            print(f"  [Conditioning Diagnostic]")
            for t_val, d in diag.items():
                print(f"    t={t_val:3d}: diff={d['diff']:.6f} "
                      f"avg={d['avg']:.4f} ratio={d['ratio']:.4f}")

        # ── E-step: 全局重算分配 ──
        if em_round > 0:
            print(f"  [E-step] scale={current_scale:.1f}, n_mc={n_mc}")
            cached_labels, freq_np, n_active, gap = global_estep(
                model, all_x, cfg,
                scale_factor=current_scale, n_mc=n_mc)

            # ── 坍缩检测与恢复 ──
            if n_active < cfg.num_classes * 0.5:  # 活跃类 < 50%
                print(f"  ⚠️ COLLAPSE DETECTED: #active={n_active}/{cfg.num_classes}")
                print(f"  ⚠️ Reverting to previous labels + reducing scale")
                # 恢复上一轮标签
                cached_labels = prev_cached_labels.clone()
                current_scale = max(current_scale * 0.5, scale_start)
                freq = torch.bincount(cached_labels, minlength=cfg.num_classes).float()
                freq_np = (freq / freq.sum()).numpy()
                n_active = len(cached_labels.unique())
                gap = 0.0
                print(f"  ⚠️ Restored: #active={n_active}, scale→{current_scale:.1f}")

            # 闭式解更新 π
            pi_np = model.update_pi_closed_form(cached_labels)
            pi_entropy = -np.sum(pi_np * np.log(pi_np + 1e-9))

            # 详细频率打印
            freq_str = ", ".join([f"{f:.3f}" for f in freq_np])
            print(f"  → #active={n_active} | gap={gap:.4f} | H(π)={pi_entropy:.3f}")
            print(f"  → freq=[{freq_str}]")

            # Evaluate
            val_acc, mapping, val_nmi = evaluate_model(model, val_loader, cfg)
            print(f"  → Acc={val_acc:.4f} NMI={val_nmi:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_cluster_mapping = mapping
                torch.save({
                    'em_round': em_round,
                    'model_state_dict': model.state_dict(),
                    'acc': val_acc, 'nmi': val_nmi,
                    'cluster_mapping': mapping, 'pi': pi_np,
                }, os.path.join(cfg.output_dir, "best_model.pt"))
                print(f"  ★ New Best! Acc={best_val_acc:.4f}")

            history["acc"].append(val_acc)
            history["nmi"].append(val_nmi)
            history["pi_entropy"].append(pi_entropy)
            history["n_active"].append(n_active)
            history["gap"].append(gap)
            history["scale"].append(current_scale)
            history["freq_history"].append(freq_np.tolist())
            history["pi_values"].append(", ".join([f"{p:.3f}" for p in pi_np]))

        # 保存当前标签 (用于坍缩恢复)
        prev_cached_labels = cached_labels.clone()

        # ── M-step: 用固定标签训练 denoiser ──
        print(f"  [M-step] training denoiser for {m_epochs} epochs "
              f"(labels FIXED, {len(cached_labels)} samples)")

        cached_dataset = TensorDataset(all_x, cached_labels)
        cached_loader = DataLoader(cached_dataset, batch_size=cfg.batch_size,
                                   shuffle=True, drop_last=True)

        for m_ep in range(m_epochs):
            model.train()
            ep_diff_loss = 0.0
            ep_cont_loss = 0.0
            n_batches = 0

            for x_batch, labels_batch in cached_loader:
                x_batch = x_batch.to(cfg.device)
                labels_batch = labels_batch.to(cfg.device).long()
                y_onehot = F.one_hot(labels_batch,
                                     num_classes=cfg.num_classes).float()

                optimizer.zero_grad()

                # Diffusion loss (偏向高 t)
                diff_loss = model.compute_diffusion_loss(x_batch, y_onehot)

                # Contrastive loss
                total_loss = diff_loss
                cont_val = 0.0
                if use_contrastive:
                    cont_loss = model.compute_contrastive_loss(
                        x_batch, y_onehot)
                    total_loss = diff_loss + contrastive_weight * cont_loss
                    cont_val = cont_loss.item()

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                ep_diff_loss += diff_loss.item()
                ep_cont_loss += cont_val
                n_batches += 1

            avg_diff = ep_diff_loss / max(n_batches, 1)
            avg_cont = ep_cont_loss / max(n_batches, 1)

            if is_final_training and (m_ep == 0 or m_ep == m_epochs - 1):
                cont_str = f" cont={avg_cont:.4f}" if use_contrastive else ""
                print(f"    M-ep {m_ep+1}/{m_epochs}: "
                      f"diff={avg_diff:.4f}{cont_str}")

        # 记录
        history["loss"].append(avg_diff)

        # Round 0: M-step 后也评估一次
        if em_round == 0:
            val_acc, mapping, val_nmi = evaluate_model(model, val_loader, cfg)
            pi_entropy = -np.sum(pi_np * np.log(pi_np + 1e-9))
            print(f"  → After M-step: Acc={val_acc:.4f} NMI={val_nmi:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_cluster_mapping = mapping
                torch.save({
                    'em_round': em_round,
                    'model_state_dict': model.state_dict(),
                    'acc': val_acc, 'nmi': val_nmi,
                    'cluster_mapping': mapping, 'pi': pi_np,
                }, os.path.join(cfg.output_dir, "best_model.pt"))
                print(f"  ★ New Best! Acc={best_val_acc:.4f}")

            history["acc"].append(val_acc)
            history["nmi"].append(val_nmi)
            history["pi_entropy"].append(pi_entropy)
            history["n_active"].append(len(cached_labels.unique()))
            history["gap"].append(0.0)
            history["scale"].append(current_scale)
            freq_r0 = torch.bincount(cached_labels, minlength=cfg.num_classes).float()
            freq_r0 = (freq_r0 / freq_r0.sum()).numpy()
            history["freq_history"].append(freq_r0.tolist())
            history["pi_values"].append(", ".join([f"{p:.3f}" for p in pi_np]))

        # Dashboard & samples
        if is_final_training and len(history["acc"]) > 0:
            plot_dashboard(history,
                           os.path.join(cfg.output_dir, "dashboard.png"))
            sample_and_save(model, cfg,
                            os.path.join(sample_dir,
                                         f"em_round_{em_round+1:02d}.png"),
                            cluster_mapping=best_cluster_mapping or mapping)

    return best_val_acc, best_cluster_mapping


# ============================================================
# 6. Main
# ============================================================
def main():
    set_seed(2026)

    # ╔══════════════════════════════════════════════════════════╗
    # ║  配置区                                                  ║
    # ╚══════════════════════════════════════════════════════════╝
    USE_KMEANS_INIT = True          # KMeans 初始化 (借鉴 HMM-DPM)
    USE_CONTRASTIVE = False         # 关闭: 无监督下无法判断正确/错误条件

    cfg = Config()
    cfg.alpha_unlabeled = 1.0
    cfg.posterior_sample_steps = 5

    print("=" * 50)
    print("🔓 Mode: UNSUPERVISED (Strict EM)")
    print("=" * 50)

    # 加载数据
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_train = datasets.MNIST('./data', train=True, download=True,
                                transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=True,
                                 transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                            shuffle=False)

    # 预加载所有训练数据到内存 (60k 图, ~50MB)
    print("Loading all training data into memory...")
    all_x = torch.stack([full_train[i][0] for i in range(len(full_train))])
    print(f"   all_x: {all_x.shape}")  # [60000, 1, 28, 28]

    # Model & Optimizer
    model = mDPM_StrictEM(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    cfg.output_dir = "./mDPM_results"
    os.makedirs(cfg.output_dir, exist_ok=True)

    hyperparams = {
        'n_em_rounds': 8,
        'm_epochs_first': 15,         # 恢复原值
        'm_epochs_rest': 8,
        'scale_start': 5.0,
        'scale_end': 50.0,
        'n_mc': 5,
        'use_kmeans_init': USE_KMEANS_INIT,
        'use_contrastive': False,
        'contrastive_weight': 0.1,
    }

    total_epochs = (hyperparams['m_epochs_first'] +
                    hyperparams['m_epochs_rest'] * (hyperparams['n_em_rounds'] - 1))

    print(f"\n🚀 Strict EM Training [实验B: E-step只用高t ∈ [600,1000)]")
    print(f"   EM rounds:    {hyperparams['n_em_rounds']}")
    print(f"   M-step epochs: {hyperparams['m_epochs_first']} (first) / "
          f"{hyperparams['m_epochs_rest']} (rest)")
    print(f"   Scale:         {hyperparams['scale_start']} → "
          f"{hyperparams['scale_end']} (annealing)")
    print(f"   E-step t:      [600, 1000) ★ 只用高噪声段")
    print(f"   KMeans init:   {USE_KMEANS_INIT}")
    print(f"   Total epochs:  ~{total_epochs}")

    best_acc, best_mapping = run_strict_em(
        model, optimizer, all_x, val_loader, cfg,
        hyperparams=hyperparams, is_final_training=True)

    print(f"\n✅ Done. Best Acc: {best_acc:.4f}")

    # 加载 best model 生成 samples
    best_ckpt = os.path.join(cfg.output_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=cfg.device,
                           weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        best_mapping = ckpt.get('cluster_mapping', best_mapping)
        print(f"   Loaded best model (Acc={ckpt.get('acc', '?'):.4f})")

    sample_and_save(model, cfg,
                    os.path.join(cfg.output_dir, "final_samples.png"),
                    cluster_mapping=best_mapping)

    cfg_dict = {k: v for k, v in vars(cfg).items()
                if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    cfg_dict['hyperparams'] = hyperparams
    json.dump(cfg_dict,
              open(os.path.join(cfg.output_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()