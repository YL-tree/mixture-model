# mDPM.py
# ═══════════════════════════════════════════════════════════════
# Mixture DPM — Partially Variational EM (论文 Section 2.2.3)
#   + 完整 log s̃_k 公式 (重构项 + SNR 加权扩散项 + 先验项)
#   + 可学习 π
#   + 全套诊断图像 (仿照 mVAE_aligned.py 风格)
# ═══════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
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
from torchvision.utils import save_image
import itertools
from common_dpm import *


# ============================================================
# Style & Colors (与 mVAE 一致)
# ============================================================
COLORS = ['#2c73d2', '#ff6b6b', '#51cf66', '#ffa94d', '#845ef7',
          '#f06595', '#20c997', '#fab005', '#339af0', '#ff8787']
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'figure.dpi': 150,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight'})


# ============================================================
# 0. Reproducibility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed locked to {seed}")


# ============================================================
# 1. Training Logger (与 mVAE 对齐)
# ============================================================
class TrainingLogger:
    """统一的训练日志记录器，支持保存 / 加载 / 画图"""
    def __init__(self):
        self.records = {}

    def log(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.records:
                self.records[k] = []
            if isinstance(v, np.ndarray):
                v = v.tolist()
            self.records[k].append(v)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.records, f, indent=2)
        print(f"  ✓ log → {path}")


# ============================================================
# 2. Model
# ============================================================
class mDPM_SemiSup(nn.Module):
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
        self.log_pi = nn.Parameter(torch.zeros(cfg.num_classes))

    @property
    def pi(self):
        return F.softmax(self.log_pi, dim=0)

    def estimate_posterior_logits(self, x_0, cfg):
        """
        与 evaluate_model_simple 完全相同的方法:
        - 无 SNR 权重 (SNR 权重会产生极端值, 导致 posterior 坍缩)
        - t ∈ [1, T) 完整范围
        - 简单累加 -mse
        """
        batch_size = x_0.size(0)
        K = cfg.num_classes
        M = getattr(cfg, 'posterior_sample_steps', 10)
        T = cfg.timesteps
        device = x_0.device

        logits = torch.zeros(batch_size, K, device=device)

        with torch.no_grad():
            for _ in range(M):
                t = torch.randint(1, T, (batch_size,), device=device).long()
                noise = torch.randn_like(x_0)
                x_t = self.dpm_process.q_sample(x_0, t, noise)

                for k in range(K):
                    y_oh = F.one_hot(torch.full((batch_size,), k, device=device,
                                                 dtype=torch.long), K).float()
                    pred = self.cond_denoiser(x_t, t, y_oh)
                    mse = F.mse_loss(pred, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                    logits[:, k] += -mse

            log_pi = torch.log(self.pi.clamp(min=1e-6)).unsqueeze(0)
            logits = logits + log_pi

        return logits

    def forward(self, x_0, cfg, y=None, threshold=0.0, use_hard_label=False):
        batch_size = x_0.size(0)

        # Path A: 监督
        if y is not None:
            t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.dpm_process.q_sample(x_0, t, noise)
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean')
            return dpm_loss, {'dpm_loss': dpm_loss.item(), 'label_loss': 0.0,
                              'mask_rate': 1.0, 'resp': None,
                              'resp_entropy': 0.0, 'max_conf': 1.0}

        # Path B: 无监督
        else:
            logits = self.estimate_posterior_logits(x_0, cfg)
            resp = F.softmax(logits, dim=1)

            if use_hard_label:
                max_probs, pseudo_labels = resp.max(dim=1)
                mask = (max_probs >= threshold).float()
                y_target = F.one_hot(pseudo_labels, num_classes=cfg.num_classes).float()
            else:
                pseudo_labels = torch.multinomial(resp, 1).squeeze(1)
                y_target = F.one_hot(pseudo_labels, num_classes=cfg.num_classes).float()
                mask = torch.ones(batch_size, device=x_0.device)

            t_train = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
            pred_noise = self.cond_denoiser(x_t_train, t_train, y_target)

            loss_per_sample = F.mse_loss(pred_noise, noise, reduction='none').view(batch_size, -1).mean(dim=1)
            dpm_loss = (loss_per_sample * mask).sum() / (mask.sum() + 1e-8)

            log_pi = torch.log(self.pi.clamp(min=1e-6)).unsqueeze(0)
            label_loss = -(resp.detach() * log_pi).sum(dim=1).mean()

            total_loss = dpm_loss + label_loss
            mask_rate = mask.mean().item()
            resp_entropy = -(resp * torch.log(resp + 1e-9)).sum(dim=1).mean().item()
            max_conf = resp.max(dim=1)[0].mean().item()

            return total_loss, {
                'dpm_loss': dpm_loss.item(),
                'label_loss': label_loss.item(),
                'mask_rate': mask_rate,
                'resp': resp.detach(),
                'resp_entropy': resp_entropy,
                'max_conf': max_conf,
            }


# ============================================================
# 3. Evaluation
# ============================================================
def evaluate_model(model, loader, cfg):
    model.eval()
    preds, ys_true = [], []

    with torch.no_grad():
        for x_0, y_true in loader:
            x_0 = x_0.to(cfg.device)
            logits = model.estimate_posterior_logits(x_0, cfg)
            pred_cluster = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred_cluster)
            ys_true.append(y_true.numpy())

    preds = np.concatenate(preds)
    ys_true = np.concatenate(ys_true)

    n_classes = cfg.num_classes
    cost_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cost_matrix[i, j] = -np.sum((ys_true == i) & (preds == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    cluster2label = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned_preds = np.array([cluster2label.get(p, 0) for p in preds])
    acc = np.mean(aligned_preds == ys_true)
    nmi_score = NMI(ys_true, preds)

    return acc, cluster2label, nmi_score


@torch.no_grad()
def evaluate_model_simple(model, loader, cfg):
    """
    简化版评估: 直接 conditional MSE, 完整 t 范围。
    这个方法 pretrain 阶段能到 40%。
    """
    model.eval()
    preds, ys_true = [], []
    K = cfg.num_classes
    M = 5

    for x_0, y_true in loader:
        x_0 = x_0.to(cfg.device)
        B = x_0.size(0)
        logits = torch.zeros(B, K, device=x_0.device)

        for _ in range(M):
            t = torch.randint(1, cfg.timesteps, (B,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = model.dpm_process.q_sample(x_0, t, noise)

            for k in range(K):
                y_oh = F.one_hot(torch.full((B,), k, device=x_0.device,
                                             dtype=torch.long), K).float()
                pred = model.cond_denoiser(x_t, t, y_oh)
                mse = F.mse_loss(pred, noise, reduction='none').view(B, -1).mean(dim=1)
                logits[:, k] += -mse

        pred_cluster = logits.argmax(dim=1).cpu().numpy()
        preds.append(pred_cluster)
        ys_true.append(y_true.numpy())

    preds = np.concatenate(preds)
    ys_true = np.concatenate(ys_true)

    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = -np.sum((ys_true == i) & (preds == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    c2l = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned = np.array([c2l.get(p, 0) for p in preds])
    acc = np.mean(aligned == ys_true)
    nmi_score = NMI(ys_true, preds)

    return acc, c2l, nmi_score


# ============================================================
# 4. Diagnostic Figures (仿照 mVAE_aligned.py)
# ============================================================

# --- fig01: Training Curves (2x2) ---
def fig01_training_curves(logger, save_path, pretrain_epochs=0):
    """Loss, Acc/NMI, π evolution, Posterior confidence"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    epochs = logger.records['epoch']

    def _add_boundary(ax):
        """在 pretrain/EM 分界处画竖线"""
        if pretrain_epochs > 0:
            ax.axvline(pretrain_epochs + 0.5, color='gray', linestyle=':',
                       alpha=0.6, linewidth=1.5)

    # (0,0) Loss components
    ax = axes[0, 0]
    ax.plot(epochs, logger.records['loss'], color=COLORS[0], linewidth=1.5, label='Total Loss')
    if 'dpm_loss' in logger.records:
        ax.plot(epochs, logger.records['dpm_loss'], color=COLORS[1], linewidth=1,
                alpha=0.7, label='DPM Loss')
    if 'label_loss' in logger.records:
        ax.plot(epochs, logger.records['label_loss'], color=COLORS[2], linewidth=1,
                alpha=0.7, label='Label Loss')
    ax.legend(fontsize=9); ax.set_title("Training Loss"); ax.grid(alpha=0.3)
    _add_boundary(ax)

    # (0,1) Acc & NMI
    ax = axes[0, 1]
    ax.plot(epochs, logger.records['acc'], label='Accuracy', color=COLORS[1], linewidth=2)
    ax.plot(epochs, logger.records['nmi'], label='NMI', color=COLORS[0],
            linewidth=2, linestyle='--')
    ax.legend(fontsize=10); ax.set_title("Clustering Quality")
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    _add_boundary(ax)

    # (1,0) π evolution
    ax = axes[1, 0]
    if 'pi_values' in logger.records and len(logger.records['pi_values']) > 0:
        pi_array = np.array(logger.records['pi_values'])
        K = pi_array.shape[1]
        for k in range(K):
            ax.plot(epochs, pi_array[:, k], label=f'π_{k}',
                    color=COLORS[k % len(COLORS)], linewidth=1)
        ax.set_ylim(0, max(0.5, pi_array.max() * 1.2))
        ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.set_title("Class Prior π"); ax.grid(alpha=0.3)
    _add_boundary(ax)

    # (1,1) Posterior confidence & entropy
    ax = axes[1, 1]
    if 'resp_entropy' in logger.records:
        ax2 = ax.twinx()
        l1, = ax.plot(epochs, logger.records['max_conf'], label='Max Conf',
                       color=COLORS[4], linewidth=1.5)
        l2, = ax2.plot(epochs, logger.records['resp_entropy'], label='Resp Entropy',
                        color=COLORS[3], linewidth=1.5, linestyle='--')
        ax.legend(handles=[l1, l2], loc='center right', fontsize=9)
        ax2.set_ylabel('Entropy')
    ax.set_title("Posterior Confidence"); ax.grid(alpha=0.3)
    _add_boundary(ax)

    fig.suptitle("mDPM Training (pretrain | EM)" if pretrain_epochs > 0
                 else "mDPM Training Curves",
                 fontsize=16, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig01 → {save_path}")


# --- fig02: Posterior Confidence Histogram + Confusion Matrix ---
@torch.no_grad()
def fig02_posterior_histogram(model, loader, cfg, save_path):
    model.eval()
    all_confs, all_preds, all_true = [], [], []
    for x_0, y_true in loader:
        x_0 = x_0.to(cfg.device)
        logits = model.estimate_posterior_logits(x_0, cfg)
        resp = F.softmax(logits, dim=1)
        max_conf, pred = resp.max(dim=1)
        all_confs.append(max_conf.cpu().numpy())
        all_preds.append(pred.cpu().numpy())
        all_true.append(y_true.numpy())
        if sum(c.shape[0] for c in all_confs) > 5000:
            break

    confs = np.concatenate(all_confs)
    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_true)
    K = cfg.num_classes

    # Hungarian mapping
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = -np.sum((trues == i) & (preds == j))
    _, col_ind = linear_sum_assignment(cost_matrix)
    c2l = {int(c): int(l) for c, l in zip(col_ind, range(K))}
    aligned = np.array([c2l.get(p, 0) for p in preds])
    correct = aligned == trues

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (0,0) 全局 confidence 分布
    ax = axes[0, 0]
    ax.hist(confs, bins=50, color=COLORS[0], alpha=0.7, edgecolor='white')
    ax.axvline(confs.mean(), color='red', linestyle='--', label=f'mean={confs.mean():.3f}')
    ax.legend(); ax.set_title("Overall Posterior Confidence"); ax.set_xlabel("max p(x|z,y)")

    # (0,1) 正确 vs 错误
    ax = axes[0, 1]
    ax.hist(confs[correct], bins=40, alpha=0.6, label='Correct', color=COLORS[2], edgecolor='white')
    ax.hist(confs[~correct], bins=40, alpha=0.6, label='Wrong', color=COLORS[1], edgecolor='white')
    ax.legend(); ax.set_title("Confidence: Correct vs Wrong")

    # (1,0) 每类 confidence 箱线图
    ax = axes[1, 0]
    class_confs = [confs[trues == k] for k in range(K)]
    bp = ax.boxplot(class_confs, labels=[str(k) for k in range(K)], patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(COLORS[i % len(COLORS)])
        patch.set_alpha(0.6)
    ax.set_title("Confidence per True Class"); ax.set_xlabel("True Label")

    # (1,1) 混淆矩阵
    ax = axes[1, 1]
    conf_mat = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            conf_mat[i, j] = np.sum((trues == i) & (preds == j))
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    conf_mat_norm = conf_mat / (row_sums + 1e-8)
    im = ax.imshow(conf_mat_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xlabel("Predicted Cluster"); ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (normalized)")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{conf_mat_norm[i, j]:.2f}',
                    ha='center', va='center', fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Posterior Diagnostics", fontsize=16, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig02 → {save_path}")


# --- fig03: Generated Samples (enhanced) ---
@torch.no_grad()
def fig03_generated_samples(model, cfg, save_path, n_per_class=10, cluster_mapping=None):
    """按映射后的真实数字顺序生成图像网格"""
    T = model.dpm_process.timesteps
    model.cond_denoiser.eval()
    image_c = model.dpm_process.image_channels
    K = cfg.num_classes
    device = cfg.device

    shape = (n_per_class * K, image_c, 28, 28)
    x_t = torch.randn(shape, device=device)

    if cluster_mapping is not None:
        label2cluster = {v: k for k, v in cluster_mapping.items()}
        ordered = [label2cluster.get(d, d) for d in range(K)]
        y_cond = torch.tensor(ordered, device=device).repeat_interleave(n_per_class).long()
    else:
        y_cond = torch.arange(K).to(device).repeat_interleave(n_per_class).long()

    y_cond_vec = F.one_hot(y_cond, K).float()

    for i in reversed(range(0, T)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        alpha_t = model.dpm_process._extract_t(model.dpm_process.alphas, t, shape)
        one_minus_alpha_t_bar = model.dpm_process._extract_t(
            model.dpm_process.sqrt_one_minus_alphas_cumprod, t, shape)

        pred_noise = model.cond_denoiser(x_t, t, y_cond_vec)

        mu_t_1 = (x_t - (1 - alpha_t) / one_minus_alpha_t_bar * pred_noise) / alpha_t.sqrt()
        sigma_t_1 = model.dpm_process._extract_t(
            model.dpm_process.posterior_variance, t, shape).sqrt()

        if i > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        x_t = mu_t_1 + sigma_t_1 * noise

    save_image(x_t.clamp(-1, 1), save_path, nrow=n_per_class,
               normalize=True, value_range=(-1, 1))
    print(f"  ✓ fig03 → {save_path}")


# --- fig04: x-Conditionality (同一噪声输入, 不同类别条件去噪) ---
@torch.no_grad()
def fig04_x_conditionality(model, loader, cfg, save_path, n_samples=6):
    """
    同一个 x_0 加噪到 t=100，用不同 class 条件去噪一步，
    展示 class label 对去噪结果的影响（类似 mVAE 的 fig15）
    """
    model.eval()
    K = cfg.num_classes
    device = cfg.device
    t_vis = 100

    x_0, y_true = next(iter(loader))
    x_0 = x_0[:n_samples].to(device)

    t = torch.full((n_samples,), t_vis, device=device, dtype=torch.long)
    noise = torch.randn_like(x_0)
    x_t = model.dpm_process.q_sample(x_0, t, noise)

    fig, axes = plt.subplots(K + 2, n_samples, figsize=(n_samples * 1.8, (K + 2) * 1.5))

    # Row 0: 原图
    for j in range(n_samples):
        ax = axes[0, j]
        ax.imshow(x_0[j, 0].cpu(), cmap='gray'); ax.axis('off')
        ax.set_title(f"y={y_true[j].item()}", fontsize=8)
    axes[0, 0].set_ylabel("Original", fontsize=9, rotation=0, labelpad=50)

    # Row 1: 加噪图
    for j in range(n_samples):
        ax = axes[1, j]
        ax.imshow(x_t[j, 0].cpu().clamp(-1, 1) * 0.5 + 0.5, cmap='gray'); ax.axis('off')
    axes[1, 0].set_ylabel(f"Noisy\nt={t_vis}", fontsize=9, rotation=0, labelpad=50)

    # Row 2..K+1: 不同 class 条件一步去噪
    alpha_t_val = model.dpm_process.alphas[t_vis]
    sqrt_oma = model.dpm_process.sqrt_one_minus_alphas_cumprod[t_vis]

    for k in range(K):
        y_oh = F.one_hot(torch.full((n_samples,), k, device=device,
                                     dtype=torch.long), K).float()
        eps_pred = model.cond_denoiser(x_t, t, y_oh)
        x_denoised = (x_t - (1 - alpha_t_val) / sqrt_oma * eps_pred) / alpha_t_val.sqrt()

        for j in range(n_samples):
            ax = axes[k + 2, j]
            ax.imshow(x_denoised[j, 0].cpu().clamp(-1, 1) * 0.5 + 0.5, cmap='gray')
            ax.axis('off')
        axes[k + 2, 0].set_ylabel(f"x={k}", fontsize=9, rotation=0, labelpad=30)

    xcond = measure_x_conditionality(model, loader, cfg)
    fig.suptitle(f"x-Conditionality: same noisy input, different class (xcond={xcond:.4f})",
                 fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig04 → {save_path}")


@torch.no_grad()
def measure_x_conditionality(model, loader, cfg, n_samples=100):
    """
    xcond = Var_x[denoised] / (Var_x[denoised] + Var_z[denoised])
    值越高说明 class label 对输出影响越大
    """
    model.eval()
    K = cfg.num_classes
    device = cfg.device
    t_val = 200

    xs = []
    for x_0, _ in loader:
        xs.append(x_0)
        if sum(x.size(0) for x in xs) >= n_samples:
            break
    x_0 = torch.cat(xs)[:n_samples].to(device)

    t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
    noise = torch.randn_like(x_0)
    x_t = model.dpm_process.q_sample(x_0, t, noise)

    alpha_t = model.dpm_process.alphas[t_val]
    sqrt_oma = model.dpm_process.sqrt_one_minus_alphas_cumprod[t_val]

    outputs = []
    for k in range(K):
        y_oh = F.one_hot(torch.full((n_samples,), k, device=device,
                                     dtype=torch.long), K).float()
        eps_pred = model.cond_denoiser(x_t, t, y_oh)
        denoised = (x_t - (1 - alpha_t) / sqrt_oma * eps_pred) / alpha_t.sqrt()
        outputs.append(denoised)

    outputs = torch.stack(outputs, dim=0)  # [K, N, C, H, W]
    D = outputs[0].numel() // n_samples
    flat = outputs.reshape(K, n_samples, D)
    var_x = flat.var(dim=0).mean().item()   # 固定 z, 变 x 的方差
    var_z = flat.var(dim=1).mean().item()   # 固定 x, 变 z 的方差
    return var_x / (var_x + var_z + 1e-9)


# --- fig05: Per-class Denoising (原图 → 加噪 → 去噪) ---
@torch.no_grad()
def fig05_per_class_denoising(model, loader, cfg, save_path,
                               n_per_class=5, cluster_mapping=None):
    model.eval()
    K = cfg.num_classes
    device = cfg.device
    t_vis = 200

    # 收集每类样本
    class_imgs = {k: [] for k in range(K)}
    for x_0, y_true in loader:
        for k in range(K):
            mask = y_true == k
            if mask.sum() > 0 and len(class_imgs[k]) < n_per_class:
                class_imgs[k].extend(x_0[mask][:n_per_class - len(class_imgs[k])])
        if all(len(v) >= n_per_class for v in class_imgs.values()):
            break

    label2cluster = None
    if cluster_mapping is not None:
        label2cluster = {v: k for k, v in cluster_mapping.items()}

    fig, axes = plt.subplots(K, n_per_class * 3, figsize=(n_per_class * 4.5, K * 1.4))

    for k in range(K):
        imgs = torch.stack(class_imgs[k][:n_per_class]).to(device)
        B = imgs.size(0)

        t = torch.full((B,), t_vis, device=device, dtype=torch.long)
        noise = torch.randn_like(imgs)
        x_t = model.dpm_process.q_sample(imgs, t, noise)

        ck = label2cluster.get(k, k) if label2cluster else k
        y_oh = F.one_hot(torch.full((B,), ck, device=device, dtype=torch.long), K).float()
        pred_noise = model.cond_denoiser(x_t, t, y_oh)
        alpha_t_val = model.dpm_process.alphas[t_vis]
        sqrt_oma = model.dpm_process.sqrt_one_minus_alphas_cumprod[t_vis]
        x_denoised = (x_t - (1 - alpha_t_val) / sqrt_oma * pred_noise) / alpha_t_val.sqrt()

        for i in range(n_per_class):
            axes[k, i * 3].imshow(imgs[i, 0].cpu(), cmap='gray')
            axes[k, i * 3].axis('off')
            axes[k, i * 3 + 1].imshow(
                x_t[i, 0].cpu().clamp(-1, 1) * 0.5 + 0.5, cmap='gray')
            axes[k, i * 3 + 1].axis('off')
            axes[k, i * 3 + 2].imshow(
                x_denoised[i, 0].cpu().clamp(-1, 1) * 0.5 + 0.5, cmap='gray')
            axes[k, i * 3 + 2].axis('off')

        axes[k, 0].set_ylabel(f"{k}", fontsize=9, rotation=0, labelpad=15)

    for i in range(n_per_class):
        axes[0, i * 3].set_title("Orig", fontsize=7)
        axes[0, i * 3 + 1].set_title("Noisy", fontsize=7)
        axes[0, i * 3 + 2].set_title("Denoised", fontsize=7)

    fig.suptitle(f"Per-Class Denoising (t={t_vis}): Orig → Noisy → Denoised", fontsize=12)
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig05 → {save_path}")


# --- fig06: π Bar Chart (最终状态) ---
def fig06_pi_barchart(model, cfg, save_path, cluster_mapping=None):
    pi = model.pi.detach().cpu().numpy()
    K = cfg.num_classes

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"c{k}" for k in range(K)]
    if cluster_mapping:
        labels = [f"c{k}→{cluster_mapping.get(k, '?')}" for k in range(K)]

    bars = ax.bar(range(K), pi,
                  color=[COLORS[i % len(COLORS)] for i in range(K)], alpha=0.8)
    ax.set_xticks(range(K)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("π_k"); ax.set_title("Learned Class Prior π")
    ax.axhline(1.0 / K, color='gray', linestyle='--', alpha=0.5,
               label=f'Uniform={1/K:.3f}')
    ax.legend()

    for bar, v in zip(bars, pi):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  ✓ fig06 → {save_path}")


# --- Generate All Figures ---
def generate_all_figures(model, logger, loader, cfg, cluster_mapping=None):
    fig_dir = os.path.join(cfg.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    best_path = os.path.join(cfg.output_dir, "best_model.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        cluster_mapping = ckpt.get('cluster_mapping', cluster_mapping)
    model.eval()

    pretrain_ep = getattr(cfg, 'pretrain_epochs', 0)

    print("\n" + "=" * 50 + "\nGenerating diagnostic figures...\n" + "=" * 50)

    fig01_training_curves(logger,
                          os.path.join(fig_dir, "fig01_training_curves.png"),
                          pretrain_epochs=pretrain_ep)
    fig02_posterior_histogram(model, loader, cfg,
                              os.path.join(fig_dir, "fig02_posterior_histogram.png"))
    fig03_generated_samples(model, cfg,
                            os.path.join(fig_dir, "fig03_generated_samples.png"),
                            cluster_mapping=cluster_mapping)
    fig04_x_conditionality(model, loader, cfg,
                            os.path.join(fig_dir, "fig04_x_conditionality.png"))
    fig05_per_class_denoising(model, loader, cfg,
                               os.path.join(fig_dir, "fig05_per_class_denoising.png"),
                               cluster_mapping=cluster_mapping)
    fig06_pi_barchart(model, cfg,
                      os.path.join(fig_dir, "fig06_pi_distribution.png"),
                      cluster_mapping=cluster_mapping)

    logger.save(os.path.join(fig_dir, "training_log.json"))

    xcond = measure_x_conditionality(model, loader, cfg)
    print(f"\n★ Final x-conditionality: {xcond:.4f}")


# ============================================================
# 5. KMeans Pretrain (打破 EM 冷启动)
# ============================================================
def kmeans_init(loader, cfg):
    """
    对训练数据做 KMeans 聚类，返回:
      - centroids: [K, D] tensor，后续用于在线分配
      - cluster_props: 每个 cluster 的比例，用于初始化 π
    """
    print("\n🔬 Computing KMeans initialization...")
    all_features, all_true = [], []
    for x_0, y_true in loader:
        all_features.append(x_0.view(x_0.size(0), -1).numpy())
        all_true.append(y_true.numpy())

    features = np.concatenate(all_features)
    true_labels = np.concatenate(all_true)

    K = cfg.num_classes
    km = KMeans(n_clusters=K, n_init=10, random_state=42)
    km_labels = km.fit_predict(features)

    # 评估质量
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost_matrix[i, j] = -np.sum((true_labels == i) & (km_labels == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    c2l = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned = np.array([c2l.get(p, 0) for p in km_labels])
    km_acc = np.mean(aligned == true_labels)
    km_nmi = NMI(true_labels, km_labels)

    cluster_counts = np.bincount(km_labels, minlength=K).astype(float)
    cluster_props = cluster_counts / cluster_counts.sum()

    print(f"   KMeans Acc: {km_acc:.4f}  NMI: {km_nmi:.4f}")
    print(f"   Cluster sizes: {cluster_counts.astype(int).tolist()}")

    centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    return centroids, cluster_props


def pretrain_with_kmeans(model, optimizer, loader, val_loader, cfg,
                         centroids, pretrain_epochs=10, logger=None):
    """
    Phase 0: 用 KMeans 伪标签做 supervised DDPM 训练。
    使用独立的高学习率，结束后恢复原 optimizer 状态。
    """
    print(f"\n🏋️ Pretrain: {pretrain_epochs} epochs with KMeans pseudo-labels")
    centroids_dev = centroids.to(cfg.device)

    # Pretrain 用独立的高学习率 optimizer
    pretrain_lr = getattr(cfg, 'pretrain_lr', 2e-4)
    pretrain_opt = torch.optim.Adam(model.parameters(), lr=pretrain_lr)
    print(f"   Pretrain LR: {pretrain_lr}")

    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0

        for x_0, _ in loader:
            x_0 = x_0.to(cfg.device)
            B = x_0.size(0)

            with torch.no_grad():
                flat = x_0.view(B, -1)
                dists = torch.cdist(flat, centroids_dev)
                pseudo_y = dists.argmin(dim=1)

            loss, info = model(x_0, cfg, y=pseudo_y)

            pretrain_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            pretrain_opt.step()

            ep_loss += loss.item()
            n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)
        # Pretrain 用简单 conditional MSE 评估（不用 CFG，因为 unconditional 路径还没训好）
        val_acc, _, val_nmi = evaluate_model_simple(model, val_loader, cfg)

        if logger:
            pi_np = model.pi.detach().cpu().numpy()
            logger.log(
                epoch=epoch, loss=avg_loss, dpm_loss=avg_loss, label_loss=0.0,
                acc=val_acc, nmi=val_nmi, mask_rate=1.0,
                resp_entropy=0.0, max_conf=1.0,
                pi_values=pi_np,
                pi_entropy=float(-(pi_np * np.log(pi_np + 1e-9)).sum()),
                threshold=0.0,
            )

        print(f"  [Pretrain] Ep {epoch}/{pretrain_epochs} "
              f"| Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} NMI: {val_nmi:.4f}")

    print("  ✅ Pretrain complete\n")

    # ---- 诊断: UNet 是否真的在用 class 条件? ----
    _diagnose_conditioning(model, loader, cfg)


@torch.no_grad()
def _diagnose_conditioning(model, loader, cfg):
    """
    测量: 同一个 (x_t, t)，不同 class k 的 UNet 输出差异。
    在多个 t 值测试，找到 class 信号最强的区间。
    """
    model.eval()
    device = cfg.device
    K = cfg.num_classes

    x_0, _ = next(iter(loader))
    x_0 = x_0[:16].to(device)

    print(f"\n🔬 Conditioning Diagnostic (per timestep):")
    print(f"   {'t':>6s}  {'diff':>10s}  {'norm':>10s}  {'ratio':>10s}")
    print(f"   {'-'*42}")

    for t_val in [5, 20, 50, 100, 200, 500]:
        t = torch.full((16,), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = model.dpm_process.q_sample(x_0, t, noise)

        outputs = []
        for k in range(K):
            y_oh = F.one_hot(torch.full((16,), k, device=device, dtype=torch.long), K).float()
            eps_k = model.cond_denoiser(x_t, t, y_oh)
            outputs.append(eps_k)

        diffs = []
        for i in range(K):
            for j in range(i + 1, K):
                d = (outputs[i] - outputs[j]).pow(2).mean().item()
                diffs.append(d)

        avg_diff = np.mean(diffs)
        avg_norm = np.mean([o.pow(2).mean().item() for o in outputs])
        ratio = avg_diff / (avg_norm + 1e-9)
        marker = " ✅" if ratio > 0.001 else " ⚠️"
        print(f"   t={t_val:>4d}  {avg_diff:>10.6f}  {avg_norm:>10.6f}  {ratio:>10.6f}{marker}")

    print()
def run_training_session(model, optimizer, labeled_loader, unlabeled_loader,
                         val_loader, cfg, is_final_training=False, trial=None,
                         hyperparams=None, logger=None, epoch_offset=0):

    total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs

    if hyperparams is None:
        hyperparams = {'warmup_epochs': 15, 'threshold_final': 0.0}

    warmup_epochs = hyperparams.get('warmup_epochs', 15)
    threshold_final = hyperparams.get('threshold_final', 0.0)

    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    best_val_acc = 0.0
    best_cluster_mapping = None

    has_labeled = labeled_loader is not None
    if has_labeled and is_final_training:
        print(f"   📋 Semi-supervised mode: labeled + unlabeled data")

    for epoch in range(1, total_epochs + 1):

        # 调度器
        if epoch <= warmup_epochs:
            use_hard = False
            dynamic_threshold = 0.0
            status = "EXPLORE"
        else:
            use_hard = True
            p2 = (epoch - warmup_epochs) / (total_epochs - warmup_epochs + 1e-8)
            dynamic_threshold = threshold_final * p2
            status = "REFINE"

        if is_final_training:
            pi_str = ", ".join([f"{p:.3f}" for p in model.pi.detach().cpu().numpy()])
            print(f"🔥 [Ep {epoch}/{total_epochs}] [{status}] "
                  f"Thres={dynamic_threshold:.3f}  π=[{pi_str}]")

        model.train()
        ep_loss, ep_dpm, ep_label = 0.0, 0.0, 0.0
        ep_sup_loss = 0.0
        ep_mask, ep_ent, ep_conf = 0.0, 0.0, 0.0
        n_batches = 0

        # ---- 有标签数据: supervised DDPM loss ----
        if has_labeled:
            for x_lab, y_lab in labeled_loader:
                x_lab = x_lab.to(cfg.device)
                y_lab = y_lab.to(cfg.device)

                optimizer.zero_grad()
                sup_loss, _ = model(x_lab, cfg, y=y_lab)
                sup_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_sup_loss += sup_loss.item()

        # ---- 无标签数据: EM loss ----
        if unlabeled_loader is not None:
            for batch_un in unlabeled_loader:
                optimizer.zero_grad()

                x_un, _ = batch_un
                x_un = x_un.to(cfg.device)

                l_unsup, info = model(x_un, cfg, y=None,
                                      threshold=dynamic_threshold,
                                      use_hard_label=use_hard)

                total_loss = cfg.alpha_unlabeled * l_unsup

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                ep_loss += total_loss.item()
                ep_dpm += info['dpm_loss']
                ep_label += info.get('label_loss', 0.0)
                ep_mask += info['mask_rate']
                ep_ent += info.get('resp_entropy', 0.0)
                ep_conf += info.get('max_conf', 0.0)
                n_batches += 1

        # Validation
        val_acc, cluster_mapping, val_nmi = evaluate_model(model, val_loader, cfg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cluster_mapping = cluster_mapping
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': val_acc,
                'nmi': val_nmi,
                'params': hyperparams,
                'cluster_mapping': cluster_mapping,
            }
            torch.save(save_dict, os.path.join(cfg.output_dir, "best_model.pt"))
            if is_final_training:
                print(f"   ★ New Best! Acc={best_val_acc:.4f}")

        # 统计
        N = max(n_batches, 1)
        avg_loss = ep_loss / N
        avg_dpm = ep_dpm / N
        avg_label = ep_label / N
        avg_mask = ep_mask / N
        avg_ent = ep_ent / N
        avg_conf = ep_conf / N
        pi_np = model.pi.detach().cpu().numpy()

        # Logger
        if logger:
            logger.log(
                epoch=epoch + epoch_offset,
                loss=avg_loss,
                dpm_loss=avg_dpm,
                label_loss=avg_label,
                acc=val_acc,
                nmi=val_nmi,
                mask_rate=avg_mask,
                resp_entropy=avg_ent,
                max_conf=avg_conf,
                pi_values=pi_np,
                pi_entropy=float(-(pi_np * np.log(pi_np + 1e-9)).sum()),
                threshold=dynamic_threshold,
            )

        if is_final_training:
            sup_str = f" sup={ep_sup_loss:.4f}" if has_labeled else ""
            print(f"  → Loss={avg_loss:.4f} (dpm={avg_dpm:.4f} label={avg_label:.4f}{sup_str}) "
                  f"| Acc={val_acc:.4f} NMI={val_nmi:.4f} "
                  f"| Conf={avg_conf:.3f} Pass={avg_mask*100:.1f}%")

            if epoch % 5 == 0:
                fig03_generated_samples(
                    model, cfg,
                    os.path.join(sample_dir, f"epoch_{epoch:03d}.png"),
                    cluster_mapping=cluster_mapping
                )

    return best_val_acc, best_cluster_mapping


# ============================================================
# 6. Main
# ============================================================
def objective(trial):
    cfg = Config()
    cfg.alpha_unlabeled = 1.0
    cfg.labeled_per_class = 0
    cfg.posterior_sample_steps = 5
    cfg.optuna_epochs = 35

    lr = trial.suggest_float("lr", 4e-5, 2e-4, log=True)
    warmup_epochs = trial.suggest_int("warmup_epochs", 10, 20)
    threshold_final = trial.suggest_float("threshold_final", 0.0, 0.1)

    hyperparams = {
        'warmup_epochs': warmup_epochs,
        'threshold_final': threshold_final
    }

    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    _, unlabeled_loader, val_loader = get_semi_loaders(cfg)

    acc, _ = run_training_session(
        model, optimizer, None, unlabeled_loader, val_loader, cfg,
        is_final_training=False, trial=trial, hyperparams=hyperparams
    )
    return acc


def main():
    set_seed(42)

    # ╔══════════════════════════════════════════════════════╗
    # ║  在这里选择训练模式                                    ║
    # ╚══════════════════════════════════════════════════════╝
    TRAINING_MODE = "unsupervised"   # "unsupervised" 或 "semi_supervised"
    LABELED_PER_CLASS = 100          # 半监督模式下每类的标注数量
    SKIP_PRETRAIN = False            # True = 从 checkpoint 加载, 跳过 pretrain
    ENABLE_AUTO_SEARCH = False

    cfg = Config()
    cfg.alpha_unlabeled = 1.0
    cfg.posterior_sample_steps = 10
    cfg.pretrain_epochs = 20
    cfg.pretrain_lr = 2e-4

    # ---- 根据模式配置 ----
    if TRAINING_MODE == "unsupervised":
        cfg.labeled_per_class = 0
        print("=" * 50)
        print("🔓 Mode: UNSUPERVISED (全部数据无标签)")
        print("   → KMeans pretrain + EM")
        print("=" * 50)
    elif TRAINING_MODE == "semi_supervised":
        cfg.labeled_per_class = LABELED_PER_CLASS
        print("=" * 50)
        print(f"🏷️  Mode: SEMI-SUPERVISED ({LABELED_PER_CLASS} labels/class)")
        print(f"   → Labeled supervised + Unlabeled EM")
        print("=" * 50)
    else:
        raise ValueError(f"Unknown mode: {TRAINING_MODE}")

    if ENABLE_AUTO_SEARCH:
        print("🔍 [Step 1] Starting Optuna Search...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        best_params = {
            'warmup_epochs': study.best_params['warmup_epochs'],
            'threshold_final': study.best_params['threshold_final']
        }
        best_lr = study.best_params['lr']
    else:
        print("⏩ [Step 1] Skipping Search, using default params")
        best_params = {
            'warmup_epochs': 10,
            'threshold_final': 0.036
        }
        best_lr = 4.01e-05

    print(f"\n🚀 [Step 2] Starting Training...")
    print(f"   Configs: LR={best_lr:.2e}, Params={best_params}")

    cfg.final_epochs = 60
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)

    logger = TrainingLogger()
    epoch_offset = 0

    pretrain_ckpt_path = os.path.join(cfg.output_dir, "pretrain_checkpoint.pt")

    # ── Phase 0: Pretrain 或 加载 Checkpoint ──
    if SKIP_PRETRAIN and os.path.exists(pretrain_ckpt_path):
        # ---- 从 checkpoint 恢复 ----
        print(f"\n⏩ Loading pretrain checkpoint: {pretrain_ckpt_path}")
        ckpt = torch.load(pretrain_ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'logger_data' in ckpt:
            logger.data = ckpt['logger_data']
        epoch_offset = ckpt.get('epoch_offset', cfg.pretrain_epochs)
        print(f"   Restored: Acc={ckpt.get('pretrain_acc', '?'):.4f}, "
              f"epoch_offset={epoch_offset}")
        print(f"   π = {model.pi.detach().cpu().numpy().round(3).tolist()}")
        _diagnose_conditioning(model, val_loader, cfg)

    else:
        # ---- 正常 pretrain ----
        if TRAINING_MODE == "unsupervised":
            data_loader = unlabeled_loader
            centroids, cluster_props = kmeans_init(data_loader, cfg)

            with torch.no_grad():
                init_log_pi = torch.log(torch.tensor(cluster_props, dtype=torch.float32).clamp(min=1e-6))
                model.log_pi.copy_(init_log_pi)
            print(f"   π initialized from KMeans: {model.pi.detach().cpu().numpy().round(3).tolist()}")

            pretrain_with_kmeans(model, optimizer, data_loader, val_loader, cfg,
                                 centroids, pretrain_epochs=cfg.pretrain_epochs, logger=logger)
            epoch_offset = cfg.pretrain_epochs

        elif TRAINING_MODE == "semi_supervised":
            if labeled_loader is not None:
                print(f"\n🏋️ Pretrain: {cfg.pretrain_epochs} epochs with REAL labels")
                pretrain_opt = torch.optim.Adam(model.parameters(), lr=cfg.pretrain_lr)

                for epoch in range(1, cfg.pretrain_epochs + 1):
                    model.train()
                    ep_loss, n = 0.0, 0
                    for x_lab, y_lab in labeled_loader:
                        x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device)
                        loss, _ = model(x_lab, cfg, y=y_lab)
                        pretrain_opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        pretrain_opt.step()
                        ep_loss += loss.item(); n += 1

                    avg_loss = ep_loss / max(n, 1)
                    val_acc, _, val_nmi = evaluate_model_simple(model, val_loader, cfg)

                    if logger:
                        pi_np = model.pi.detach().cpu().numpy()
                        logger.log(epoch=epoch, loss=avg_loss, dpm_loss=avg_loss, label_loss=0.0,
                                   acc=val_acc, nmi=val_nmi, mask_rate=1.0,
                                   resp_entropy=0.0, max_conf=1.0, pi_values=pi_np,
                                   pi_entropy=float(-(pi_np * np.log(pi_np + 1e-9)).sum()),
                                   threshold=0.0)

                    print(f"  [Pretrain] Ep {epoch}/{cfg.pretrain_epochs} "
                          f"| Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} NMI: {val_nmi:.4f}")

                print("  ✅ Pretrain complete\n")
                _diagnose_conditioning(model, val_loader, cfg)
                epoch_offset = cfg.pretrain_epochs

        # ---- 保存 pretrain checkpoint ----
        pretrain_acc, _, pretrain_nmi = evaluate_model_simple(model, val_loader, cfg)
        torch.save({
            'model_state_dict': model.state_dict(),
            'logger_data': logger.data,
            'epoch_offset': epoch_offset,
            'pretrain_acc': pretrain_acc,
            'pretrain_nmi': pretrain_nmi,
            'training_mode': TRAINING_MODE,
        }, pretrain_ckpt_path)
        print(f"💾 Pretrain checkpoint saved → {pretrain_ckpt_path}")
        print(f"   (Acc={pretrain_acc:.4f}, NMI={pretrain_nmi:.4f})")
        print(f"   下次设 SKIP_PRETRAIN=True 即可跳过 pretrain\n")

    # ── Phase 1: EM Training ──
    print("=" * 50)
    print("🔄 Switching to EM training...")
    print("=" * 50)

    best_acc, best_mapping = run_training_session(
        model, optimizer,
        labeled_loader if TRAINING_MODE == "semi_supervised" else None,
        unlabeled_loader, val_loader, cfg,
        is_final_training=True,
        hyperparams=best_params,
        logger=logger,
        epoch_offset=epoch_offset,
    )

    print(f"\n✅ Done. Best Acc: {best_acc:.4f}")

    generate_all_figures(model, logger, val_loader, cfg, cluster_mapping=best_mapping)

    cfg_dict = {k: v for k, v in vars(cfg).items()
                if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    cfg_dict['training_mode'] = TRAINING_MODE
    cfg_dict['labeled_per_class'] = cfg.labeled_per_class
    json.dump(cfg_dict, open(os.path.join(cfg.output_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
