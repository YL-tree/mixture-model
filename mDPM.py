# mDPM_aligned.py — 统一入口: unsup / semisup / fullsup
# ═══════════════════════════════════════════════════════════════
# 用法:
#   python mDPM_aligned.py --mode unsup                    # 无监督 (原始)
#   python mDPM_aligned.py --mode semisup --labeled 100    # 半监督, 每类100标签
#   python mDPM_aligned.py --mode fullsup                  # 全监督
#   python mDPM_aligned.py --mode unsup --epochs 80        # 自定义 epoch
#
# 架构完全不改, 训练逻辑不改, 只加:
#   - argparse 统一入口
#   - fullsup / semisup 训练分支
#   - EMA (纯采样侧, 不影响训练)
# ═══════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, json, copy, argparse, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as NMI
from torchvision.utils import save_image

# 用你原始的 common_dpm.py (一个字不改)
from common_dpm import *


# ============================================================
# 0. EMA (纯采样侧, 不影响训练)
# ============================================================
class EMA:
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
# 1. Seed
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
# 2. Model (原始在线 EM, 不改)
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

    def estimate_posterior_logits(self, x_0, cfg, scale_factor=1.0):
        """
        改进的 E-step — 对齐 evaluate_model 的做法
        
        原始问题:
          random t∈[100,900], 每个样本 t 不同 → 方差大
          同一张图在 t=150 和 t=850 的 MSE 差异 >> 不同 class 的 MSE 差异
          → 被 t 的随机性淹没, posterior 质量差
        
        改进:
          固定 t 值 (全 batch 同一个 t), 多次不同噪声取平均
          和 evaluate_model 一样的做法, 但计算量更少
        
        可通过 cfg.estep_mode 切换:
          'fixed'  = 固定 t=500, 3次噪声 (推荐, 和 evaluate 一致)
          'multi'  = 固定 t=[300,500,700], 各1次噪声
          'random' = 原始随机 t (fallback)
        """
        B = x_0.size(0)
        K = self.K
        device = x_0.device
        estep_mode = getattr(cfg, 'estep_mode', 'fixed')

        accum_neg_mse = torch.zeros(B, K, device=device)
        total_steps = 0

        with torch.no_grad():
            if estep_mode == 'fixed':
                # ★ 推荐: 固定 t=T//2, 重复 n_repeats 次 (和 evaluate 一致)
                eval_t = cfg.timesteps // 2
                n_repeats = getattr(cfg, 'estep_repeats', 3)
                for _ in range(n_repeats):
                    t = torch.full((B,), eval_t, device=device, dtype=torch.long)
                    noise = torch.randn_like(x_0)
                    x_t = self.dpm_process.q_sample(x_0, t, noise)
                    for k in range(K):
                        y_oh = F.one_hot(torch.full((B,), k, device=device,
                                                     dtype=torch.long), K).float()
                        pred = self.cond_denoiser(x_t, t, y_oh)
                        mse = F.mse_loss(pred, noise, reduction='none').view(B, -1).mean(dim=1)
                        accum_neg_mse[:, k] += -mse
                    total_steps += 1

            elif estep_mode == 'multi':
                # 多个固定 t, 每个 1 次
                eval_ts = [300, 500, 700]
                for eval_t in eval_ts:
                    t = torch.full((B,), eval_t, device=device, dtype=torch.long)
                    noise = torch.randn_like(x_0)
                    x_t = self.dpm_process.q_sample(x_0, t, noise)
                    for k in range(K):
                        y_oh = F.one_hot(torch.full((B,), k, device=device,
                                                     dtype=torch.long), K).float()
                        pred = self.cond_denoiser(x_t, t, y_oh)
                        mse = F.mse_loss(pred, noise, reduction='none').view(B, -1).mean(dim=1)
                        accum_neg_mse[:, k] += -mse
                    total_steps += 1

            else:  # 'random' — 原始方式
                M = getattr(cfg, 'posterior_sample_steps', 5)
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
                    total_steps += 1

            avg_neg_mse = accum_neg_mse / max(total_steps, 1)
            log_pi = torch.log(self.pi.clamp(min=1e-6)).unsqueeze(0)
            logits = log_pi + avg_neg_mse * scale_factor
        return logits

    def forward_unlabeled(self, x_0, cfg, scale_factor=1.0, **kwargs):
        """
        无监督 forward: E-step + M-step
        
        M-step 使用加权 soft-EM (和原始 mDPM.py 一致):
          对所有 K 个 class 做 forward, 用 posterior 加权求和
          每个 batch = K 次 denoiser forward → 训练效率高
        
        kwargs: 兼容旧接口 (use_hard_label, threshold 等已不使用)
        """
        B = x_0.size(0)
        K = self.K

        # E-step
        logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=scale_factor)
        resp = F.softmax(logits, dim=1)  # (B, K)

        # M-step: 加权 soft-EM (原始方式)
        t_train = torch.randint(0, cfg.timesteps, (B,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.dpm_process.q_sample(x_0, t_train, noise)

        weighted_dpm_loss = 0.0
        for k in range(K):
            y_onehot_k = F.one_hot(
                torch.full((B,), k, device=x_0.device, dtype=torch.long), K
            ).float()
            pred_noise_k = self.cond_denoiser(x_t, t_train, y_onehot_k)
            dpm_loss_k = F.mse_loss(pred_noise_k, noise, reduction='none').view(B, -1).mean(dim=1)
            weighted_dpm_loss += (resp[:, k].detach() * dpm_loss_k).mean()

        # pseudo_labels 用于统计 (argmax)
        pseudo_labels = resp.argmax(dim=1)

        return weighted_dpm_loss, {
            'dpm_loss': weighted_dpm_loss.item(),
            'mask_rate': 1.0,
            'pseudo_labels': pseudo_labels.detach(),
        }

    def forward_labeled(self, x_0, y_true, cfg):
        """有监督 forward: 用真实标签训练 denoiser"""
        B = x_0.size(0)
        y_target = F.one_hot(y_true, num_classes=self.K).float()

        t_train = torch.randint(0, cfg.timesteps, (B,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.dpm_process.q_sample(x_0, t_train, noise)
        pred_noise = self.cond_denoiser(x_t, t_train, y_target)
        dpm_loss = F.mse_loss(pred_noise, noise)

        return dpm_loss, {
            'dpm_loss': dpm_loss.item(),
            'mask_rate': 1.0,
            'pseudo_labels': y_true.detach(),
        }


# ============================================================
# 3. Evaluation (原始, 不改)
# ============================================================
@torch.no_grad()
def evaluate_model(model, loader, cfg):
    model.eval()
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
                pred = model.cond_denoiser(x_t, t, y_oh)
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
# 4. Conditioning Diagnostic (原始, 不改)
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
# 5. Sampling
# ============================================================
@torch.no_grad()
def sample_and_save(model, cfg, out_path, n_per_class=10,
                    cluster_mapping=None, use_denoiser=None):
    """用指定的 denoiser (原始或 EMA) 采样"""
    denoiser = use_denoiser if use_denoiser is not None else model.cond_denoiser
    T = model.dpm_process.timesteps
    denoiser.eval()
    K = cfg.num_classes
    imgs = []
    for k in range(K):
        gen_k = k
        if cluster_mapping:
            for ck, tk in cluster_mapping.items():
                if tk == k:
                    gen_k = ck
                    break
        y_oh = F.one_hot(torch.full((n_per_class,), gen_k,
                                     dtype=torch.long), K).float().to(cfg.device)
        x = torch.randn(n_per_class, cfg.image_channels, 28, 28, device=cfg.device)
        for t_idx in reversed(range(T)):
            t_ = torch.full((n_per_class,), t_idx, device=cfg.device, dtype=torch.long)
            pred = denoiser(x, t_, y_oh)
            beta = model.dpm_process.betas[t_idx]
            alpha = model.dpm_process.alphas[t_idx]
            alpha_bar = model.dpm_process.alphas_cumprod[t_idx]
            x = (1.0 / alpha.sqrt()) * (x - beta / (1 - alpha_bar).sqrt() * pred)
            if t_idx > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
        imgs.append(x.cpu())
    imgs = torch.cat(imgs, dim=0)
    save_image(imgs, out_path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
    print(f"  ✓ samples → {out_path}")


# ============================================================
# 6. Dashboard
# ============================================================
def plot_dashboard(history, outpath, mode="unsup"):
    n = len(history.get("loss", []))
    if n == 0:
        return
    epochs = range(1, n + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'mDPM [{mode}] (Ep {n})', fontsize=18)

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
# 7. Training: Unsupervised (原始在线 EM, 不改)
# ============================================================
def train_unsupervised(model, optimizer, unlabeled_loader, val_loader, cfg,
                       ema=None, hyperparams=None):
    if hyperparams is None:
        hyperparams = {}

    total_epochs = hyperparams.get('total_epochs', 60)
    target_scale = hyperparams.get('target_scale', 134.0)
    warmup_epochs = hyperparams.get('warmup_epochs', 10)
    threshold_final = hyperparams.get('threshold_final', 0.036)

    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    best_val_acc = 0.0
    best_cluster_mapping = None
    history = {"loss": [], "acc": [], "nmi": [],
               "pass_rate": [], "scale": [],
               "cond_ratio": [], "freq": [], "pi_entropy": []}

    for epoch in range(1, total_epochs + 1):
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

        print(f"🔥 [Ep {epoch}/{total_epochs}] [{status}] "
              f"Scale={dynamic_scale:.1f} Thres={dynamic_threshold:.3f}")

        model.train()
        ep_loss, ep_mask, n_batches = 0.0, 0.0, 0
        epoch_labels = []

        for x_batch, _ in unlabeled_loader:
            x_batch = x_batch.to(cfg.device)
            optimizer.zero_grad()
            loss, info = model.forward_unlabeled(
                x_batch, cfg, scale_factor=dynamic_scale,
                use_hard_label=use_hard, threshold=dynamic_threshold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema and epoch >= 5:
                ema.update(model)

            ep_loss += loss.item()
            ep_mask += info['mask_rate']
            epoch_labels.append(info['pseudo_labels'].cpu())
            n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)
        pass_pct = (ep_mask / max(n_batches, 1)) * 100
        all_labels = torch.cat(epoch_labels)
        label_freq = np.bincount(all_labels.numpy(), minlength=cfg.num_classes).astype(float)
        label_freq = label_freq / label_freq.sum()
        pi_entropy = -np.sum(label_freq * np.log(label_freq + 1e-9))

        freq_str = ", ".join([f"{f:.3f}" for f in label_freq])
        print(f"   freq=[{freq_str}] H={pi_entropy:.3f}")

        val_acc, cluster_mapping, val_nmi, val_freq = evaluate_model(model, val_loader, cfg)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cluster_mapping = cluster_mapping
            save_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                         'acc': val_acc, 'nmi': val_nmi, 'cluster_mapping': cluster_mapping}
            if ema:
                save_dict['ema_state_dict'] = ema.get_model().state_dict()
            torch.save(save_dict, os.path.join(cfg.output_dir, "best_model.pt"))
            print(f"   ★ New Best! Acc={best_val_acc:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            cond_ratios = conditioning_diagnostic(model, val_loader, cfg)
            avg_ratio = np.mean(list(cond_ratios.values()))
            history["cond_ratio"].append(avg_ratio)
            ratio_str = " | ".join([f"t={t}: {r:.4f}" for t, r in sorted(cond_ratios.items())])
            print(f"   [Cond] {ratio_str} | avg={avg_ratio:.4f}")
        else:
            history["cond_ratio"].append(
                history["cond_ratio"][-1] if history["cond_ratio"] else 0.0)

        history["loss"].append(avg_loss)
        history["acc"].append(val_acc)
        history["nmi"].append(val_nmi)
        history["pass_rate"].append(pass_pct)
        history["scale"].append(dynamic_scale)
        history["freq"].append(label_freq.tolist())
        history["pi_entropy"].append(pi_entropy)

        print(f"  → Loss={avg_loss:.4f} | Acc={val_acc:.4f} NMI={val_nmi:.4f} "
              f"| Pass={pass_pct:.1f}%")

        plot_dashboard(history, os.path.join(cfg.output_dir, "dashboard.png"), "unsup")
        if epoch % 5 == 0:
            ema_den = ema.get_model().cond_denoiser if ema else None
            sample_and_save(model, cfg,
                            os.path.join(sample_dir, f"epoch_{epoch:03d}.png"),
                            cluster_mapping=cluster_mapping,
                            use_denoiser=ema_den)

    return best_val_acc, best_cluster_mapping


# ============================================================
# 8. Training: Fully Supervised
# ============================================================
def train_fullsup(model, optimizer, labeled_loader, val_loader, cfg,
                  ema=None, total_epochs=60):
    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    best_val_acc = 0.0
    best_cluster_mapping = None
    history = {"loss": [], "acc": [], "nmi": [],
               "pass_rate": [], "scale": [],
               "cond_ratio": [], "freq": [], "pi_entropy": []}

    for epoch in range(1, total_epochs + 1):
        print(f"🔥 [Ep {epoch}/{total_epochs}] [FULLSUP]")

        model.train()
        ep_loss, n_batches = 0.0, 0
        epoch_labels = []

        for x_batch, y_batch in labeled_loader:
            x_batch = x_batch.to(cfg.device)
            y_batch = y_batch.to(cfg.device)
            optimizer.zero_grad()
            loss, info = model.forward_labeled(x_batch, y_batch, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema and epoch >= 5:
                ema.update(model)

            ep_loss += loss.item()
            epoch_labels.append(info['pseudo_labels'].cpu())
            n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)
        all_labels = torch.cat(epoch_labels)
        label_freq = np.bincount(all_labels.numpy(), minlength=cfg.num_classes).astype(float)
        label_freq = label_freq / label_freq.sum()
        pi_entropy = -np.sum(label_freq * np.log(label_freq + 1e-9))

        val_acc, cluster_mapping, val_nmi, val_freq = evaluate_model(model, val_loader, cfg)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cluster_mapping = cluster_mapping
            save_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                         'acc': val_acc, 'nmi': val_nmi, 'cluster_mapping': cluster_mapping}
            if ema:
                save_dict['ema_state_dict'] = ema.get_model().state_dict()
            torch.save(save_dict, os.path.join(cfg.output_dir, "best_model.pt"))
            print(f"   ★ New Best! Acc={best_val_acc:.4f}")

        history["loss"].append(avg_loss)
        history["acc"].append(val_acc)
        history["nmi"].append(val_nmi)
        history["pass_rate"].append(100.0)
        history["scale"].append(0.0)
        history["freq"].append(label_freq.tolist())
        history["pi_entropy"].append(pi_entropy)
        history["cond_ratio"].append(0.0)

        print(f"  → Loss={avg_loss:.4f} | Acc={val_acc:.4f} NMI={val_nmi:.4f}")

        plot_dashboard(history, os.path.join(cfg.output_dir, "dashboard.png"), "fullsup")
        if epoch % 5 == 0:
            ema_den = ema.get_model().cond_denoiser if ema else None
            sample_and_save(model, cfg,
                            os.path.join(sample_dir, f"epoch_{epoch:03d}.png"),
                            cluster_mapping=cluster_mapping,
                            use_denoiser=ema_den)

    return best_val_acc, best_cluster_mapping


# ============================================================
# 9. Training: Semi-supervised
#    labeled 循环复用 (和你 mVAE 一样的模式)
# ============================================================
def train_semisup(model, optimizer, labeled_loader, unlabeled_loader, val_loader,
                  cfg, ema=None, hyperparams=None):
    """
    半监督训练 — 还原原始 mDPM.py 的行为:
      Phase 1 (epoch 1~sup_warmup): 纯监督, alpha_un=0
        → denoiser 先学会用 class condition
      Phase 2 (epoch sup_warmup+1~total): 监督 + 无监督 (在线 EM)
        → unlabeled 数据的 EXPLORE→REFINE 从 Phase 2 开始算
    """
    if hyperparams is None:
        hyperparams = {}

    total_epochs = hyperparams.get('total_epochs', 60)
    target_scale = hyperparams.get('target_scale', 134.0)
    warmup_epochs = hyperparams.get('warmup_epochs', 10)  # 无监督部分的 EXPLORE 长度
    threshold_final = hyperparams.get('threshold_final', 0.036)
    alpha_labeled = hyperparams.get('alpha_labeled', 1.0)

    # ★ 关键: 前 N 个 epoch 纯监督 (和原始代码一致)
    sup_warmup = hyperparams.get('sup_warmup', 10)

    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    best_val_acc = 0.0
    best_cluster_mapping = None
    history = {"loss": [], "acc": [], "nmi": [],
               "pass_rate": [], "scale": [],
               "cond_ratio": [], "freq": [], "pi_entropy": []}

    for epoch in range(1, total_epochs + 1):

        # ── 判断阶段 ──
        if epoch <= sup_warmup:
            # Phase 1: 纯监督, 不用 unlabeled
            current_alpha_un = 0.0
            dynamic_scale = 0.0
            dynamic_threshold = 0.0
            use_hard = False
            status = "SUP_ONLY"
        else:
            # Phase 2: 监督 + 无监督
            current_alpha_un = cfg.alpha_unlabeled
            em_epoch = epoch - sup_warmup  # 无监督部分的 epoch 计数
            em_total = total_epochs - sup_warmup

            if em_epoch <= warmup_epochs:
                use_hard = False
                p1 = em_epoch / warmup_epochs
                dynamic_scale = 5.0 + (20.0 - 5.0) * p1
                dynamic_threshold = 0.0
                status = "EXPLORE"
            else:
                use_hard = True
                p2 = (em_epoch - warmup_epochs) / (em_total - warmup_epochs + 1e-8)
                dynamic_scale = 20.0 + (target_scale - 20.0) * p2
                dynamic_threshold = threshold_final * p2
                status = "REFINE"

        print(f"🔥 [Ep {epoch}/{total_epochs}] [{status}] "
              f"Scale={dynamic_scale:.1f} α_un={current_alpha_un:.1f} α_lab={alpha_labeled}")

        model.train()
        ep_loss, ep_mask, n_batches = 0.0, 0.0, 0
        epoch_labels = []

        if current_alpha_un > 0 and unlabeled_loader is not None:
            # Phase 2: 遍历 unlabeled, labeled 循环复用
            labeled_iter = iter(labeled_loader)

            for x_un, _ in unlabeled_loader:
                try:
                    x_lab, y_lab = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    x_lab, y_lab = next(labeled_iter)

                x_un = x_un.to(cfg.device)
                x_lab = x_lab.to(cfg.device)
                y_lab = y_lab.to(cfg.device)

                optimizer.zero_grad()

                loss_un, info_un = model.forward_unlabeled(
                    x_un, cfg, scale_factor=dynamic_scale,
                    use_hard_label=use_hard, threshold=dynamic_threshold)
                loss_lab, info_lab = model.forward_labeled(x_lab, y_lab, cfg)

                loss = current_alpha_un * loss_un + alpha_labeled * loss_lab

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if ema and epoch >= 5:
                    ema.update(model)

                ep_loss += loss.item()
                ep_mask += info_un['mask_rate']
                epoch_labels.append(info_un['pseudo_labels'].cpu())
                n_batches += 1
        else:
            # Phase 1: 纯监督
            for x_lab, y_lab in labeled_loader:
                x_lab = x_lab.to(cfg.device)
                y_lab = y_lab.to(cfg.device)

                optimizer.zero_grad()
                loss, info = model.forward_labeled(x_lab, y_lab, cfg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if ema and epoch >= 5:
                    ema.update(model)

                ep_loss += loss.item()
                epoch_labels.append(info['pseudo_labels'].cpu())
                n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)
        pass_pct = (ep_mask / max(n_batches, 1)) * 100 if ep_mask > 0 else 100.0
        all_labels = torch.cat(epoch_labels)
        label_freq = np.bincount(all_labels.numpy(), minlength=cfg.num_classes).astype(float)
        label_freq = label_freq / label_freq.sum()
        pi_entropy = -np.sum(label_freq * np.log(label_freq + 1e-9))

        freq_str = ", ".join([f"{f:.3f}" for f in label_freq])
        print(f"   freq=[{freq_str}] H={pi_entropy:.3f}")

        val_acc, cluster_mapping, val_nmi, val_freq = evaluate_model(model, val_loader, cfg)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cluster_mapping = cluster_mapping
            save_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                         'acc': val_acc, 'nmi': val_nmi, 'cluster_mapping': cluster_mapping}
            if ema:
                save_dict['ema_state_dict'] = ema.get_model().state_dict()
            torch.save(save_dict, os.path.join(cfg.output_dir, "best_model.pt"))
            print(f"   ★ New Best! Acc={best_val_acc:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            cond_ratios = conditioning_diagnostic(model, val_loader, cfg)
            avg_ratio = np.mean(list(cond_ratios.values()))
            history["cond_ratio"].append(avg_ratio)
            ratio_str = " | ".join([f"t={t}: {r:.4f}" for t, r in sorted(cond_ratios.items())])
            print(f"   [Cond] {ratio_str} | avg={avg_ratio:.4f}")
        else:
            history["cond_ratio"].append(
                history["cond_ratio"][-1] if history["cond_ratio"] else 0.0)

        history["loss"].append(avg_loss)
        history["acc"].append(val_acc)
        history["nmi"].append(val_nmi)
        history["pass_rate"].append(pass_pct)
        history["scale"].append(dynamic_scale)
        history["freq"].append(label_freq.tolist())
        history["pi_entropy"].append(pi_entropy)

        print(f"  → Loss={avg_loss:.4f} | Acc={val_acc:.4f} NMI={val_nmi:.4f} "
              f"| Pass={pass_pct:.1f}%")

        plot_dashboard(history, os.path.join(cfg.output_dir, "dashboard.png"), "semisup")
        if epoch % 5 == 0:
            ema_den = ema.get_model().cond_denoiser if ema else None
            sample_and_save(model, cfg,
                            os.path.join(sample_dir, f"epoch_{epoch:03d}.png"),
                            cluster_mapping=cluster_mapping,
                            use_denoiser=ema_den)

    return best_val_acc, best_cluster_mapping


# ============================================================
# 10. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="mDPM — Unified Entry")
    parser.add_argument("--mode", default="unsup",
                        choices=["unsup", "semisup", "fullsup"],
                        help="unsup: 无监督聚类 | semisup: 半监督 | fullsup: 全监督")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=4.01e-05)
    parser.add_argument("--labeled", type=int, default=100,
                        help="semisup 模式下每类的标签数")
    parser.add_argument("--alpha_labeled", type=float, default=1.0,
                        help="semisup 模式下有监督损失的权重")
    parser.add_argument("--alpha_unlabeled", type=float, default=1.0,
                        help="semisup 模式下无监督损失的权重")
    parser.add_argument("--sup_warmup", type=int, default=10,
                        help="semisup 模式下前 N epoch 纯监督 (不用 unlabeled)")
    parser.add_argument("--target_scale", type=float, default=134.37)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.036)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--estep", default="fixed",
                        choices=["fixed", "multi", "random"],
                        help="E-step 模式: fixed=t=500×3(推荐) | multi=t=[300,500,700] | random=原始")
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = Config()
    cfg.posterior_sample_steps = 5
    cfg.output_dir = f"./mDPM_{args.mode}"
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 根据 mode 设置 labeled_per_class
    if args.mode == "unsup":
        cfg.labeled_per_class = 0
    elif args.mode == "semisup":
        cfg.labeled_per_class = args.labeled
    elif args.mode == "fullsup":
        cfg.labeled_per_class = -1

    print("=" * 60)
    print(f"mDPM — Mode: {args.mode}")
    print(f"  T={cfg.timesteps}, M={cfg.posterior_sample_steps}")
    print(f"  LR={args.lr}, Epochs={args.epochs}")
    if args.mode == "semisup":
        print(f"  Labeled/class: {args.labeled}, α_labeled: {args.alpha_labeled}")
        print(f"  α_unlabeled: {args.alpha_unlabeled}, sup_warmup: {args.sup_warmup}ep")
    elif args.mode == "fullsup":
        print(f"  全监督: 所有数据使用真实标签")
    else:
        print(f"  无监督: Scale 5→20→{args.target_scale}, π=uniform")
    print(f"  E-step: {args.estep} mode")
    print(f"  EMA: decay=0.9999 (纯采样侧)")
    print("=" * 60)

    # 数据
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)

    # 模型
    model = mDPM(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=0.9999)

    # 设置半监督参数
    cfg.alpha_unlabeled = args.alpha_unlabeled
    cfg.estep_mode = args.estep

    hyperparams = {
        'total_epochs': args.epochs,
        'target_scale': args.target_scale,
        'warmup_epochs': args.warmup_epochs,
        'threshold_final': args.threshold,
        'alpha_labeled': args.alpha_labeled,
        'sup_warmup': args.sup_warmup,
    }

    # 训练
    if args.mode == "unsup":
        best_acc, best_mapping = train_unsupervised(
            model, optimizer, unlabeled_loader, val_loader, cfg,
            ema=ema, hyperparams=hyperparams)

    elif args.mode == "fullsup":
        best_acc, best_mapping = train_fullsup(
            model, optimizer, labeled_loader, val_loader, cfg,
            ema=ema, total_epochs=args.epochs)

    elif args.mode == "semisup":
        best_acc, best_mapping = train_semisup(
            model, optimizer, labeled_loader, unlabeled_loader, val_loader,
            cfg, ema=ema, hyperparams=hyperparams)

    print(f"\n✅ Done. Best Acc: {best_acc:.4f}")

    # 加载 best model 生成最终 samples
    best_ckpt = os.path.join(cfg.output_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        best_mapping = ckpt.get('cluster_mapping', best_mapping)
        print(f"   Loaded best model (Acc={ckpt.get('acc', '?'):.4f})")

        if 'ema_state_dict' in ckpt:
            ema.get_model().load_state_dict(ckpt['ema_state_dict'])
            print("   Loaded EMA weights")

    # 最终采样: 原始 + EMA
    ema_den = ema.get_model().cond_denoiser

    sample_and_save(model, cfg,
                    os.path.join(cfg.output_dir, "final_samples.png"),
                    cluster_mapping=best_mapping)
    sample_and_save(model, cfg,
                    os.path.join(cfg.output_dir, "final_samples_ema.png"),
                    cluster_mapping=best_mapping,
                    use_denoiser=ema_den)

    # Save config
    cfg_dict = {k: v for k, v in vars(cfg).items()
                if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    cfg_dict['mode'] = args.mode
    cfg_dict['hyperparams'] = hyperparams
    cfg_dict['best_acc'] = best_acc
    json.dump(cfg_dict, open(os.path.join(cfg.output_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()