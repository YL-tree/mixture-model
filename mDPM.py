# mDPM.py
# ═══════════════════════════════════════════════════════════════
# Mixture DPM — 综合版
#   基于验证过的 Acc=0.5 机制 (Embedding + AdaGN + get_time_weight)
#   + 论文 M-step π 更新 (label_loss, λ_π 平衡)
#   + KMeans pretrain (可选) + 半监督 + checkpoint
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
        # π: register_buffer 作为默认值, log_pi 作为可学习参数
        # 当 enable_pi_update=True 时通过 label_loss 更新
        self.register_buffer('default_pi', torch.ones(cfg.num_classes) / cfg.num_classes)
        self.log_pi = nn.Parameter(torch.zeros(cfg.num_classes), requires_grad=False)
        # requires_grad 默认 False, 只在 enable_pi_update 时开启

    @property
    def pi(self):
        if self.log_pi.requires_grad:
            return F.softmax(self.log_pi, dim=0)
        else:
            return self.default_pi

    def enable_pi_update(self, enable=True):
        """开启/关闭 π 的梯度更新"""
        self.log_pi.requires_grad_(enable)
        if enable:
            # 用当前 default_pi 初始化 log_pi
            with torch.no_grad():
                self.log_pi.copy_(torch.log(self.default_pi.clamp(min=1e-6)))

    def estimate_posterior_logits(self, x_0, cfg, scale_factor=1.0):
        """
        E-step: -MSE/M × scale_factor
        t ∈ [100, 900]: 验证过有效, 过滤极端 t 减少方差
        scale_factor = tempered posterior 温度参数
        """
        B = x_0.size(0)
        K = cfg.num_classes
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

    def forward(self, x_0, cfg, y=None, scale_factor=1.0, lambda_pi=0.0):
        B = x_0.size(0)

        # Path A: 监督
        if y is not None:
            t = torch.randint(0, cfg.timesteps, (B,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.dpm_process.q_sample(x_0, t, noise)
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean')
            return dpm_loss, {'dpm_loss': dpm_loss.item(), 'label_loss': 0.0,
                              'mask_rate': 1.0}

        # Path B: 无监督 EM (论文: 始终 soft sampling)
        logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=scale_factor)
        resp = F.softmax(logits, dim=1)

        # Soft sampling: scale 增大时自动趋近 hard
        pseudo_labels = torch.multinomial(resp, 1).squeeze(1)
        y_target = F.one_hot(pseudo_labels, num_classes=cfg.num_classes).float()

        # M-step: denoiser loss
        t_train = torch.randint(0, cfg.timesteps, (B,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
        pred_noise = self.cond_denoiser(x_t_train, t_train, y_target)
        dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean')

        # M-step: π 更新 (论文公式, 延迟开启)
        label_loss_val = 0.0
        total_loss = dpm_loss
        if lambda_pi > 0 and self.log_pi.requires_grad:
            log_pi = torch.log(self.pi.clamp(min=1e-6)).unsqueeze(0)
            label_loss = -(resp.detach() * log_pi).sum(dim=1).mean()
            total_loss = dpm_loss + lambda_pi * label_loss
            label_loss_val = label_loss.item()

        return total_loss, {
            'dpm_loss': dpm_loss.item(),
            'label_loss': label_loss_val,
            'mask_rate': 1.0,
        }


# ============================================================
# 2. Evaluation (固定 t=500, argmin MSE)
# ============================================================
def evaluate_model(model, loader, cfg):
    model.eval()
    preds, ys_true = [], []
    eval_t = 500
    n_repeats = 3

    with torch.no_grad():
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
    return acc, cluster2label, nmi_score


# ============================================================
# 3. Dashboard & Diagnostics
# ============================================================
def plot_dashboard(history, outpath):
    n = len(history.get("loss", []))
    if n == 0:
        return
    epochs = range(1, n + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Dashboard (Ep {n})', fontsize=16)

    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], 'b-')
    ax.set_title('Loss'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["acc"], 'r-', label='Acc')
    if "nmi" in history:
        ax.plot(epochs, history["nmi"], 'g--', label='NMI')
    ax.set_title('Acc & NMI'); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 2]
    ax.plot(epochs, history["pass_rate"], 'm-')
    ax.set_title('Pass Rate (%)'); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["scale"], 'c-')
    ax.set_title('Dynamic Scale'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "pi_entropy" in history and len(history["pi_entropy"]) > 0:
        ax.plot(epochs, history["pi_entropy"], 'orange', label='π entropy')
        ax.axhline(y=np.log(10), color='gray', linestyle='--', alpha=0.5, label='uniform')
        ax.set_title('π Entropy'); ax.grid(True, alpha=0.3); ax.legend()
    else:
        ax.set_title('π Entropy (N/A)'); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]; ax.axis('off')
    pi_str = ""
    if "pi_values" in history and len(history["pi_values"]) > 0:
        pi_str = f"\nπ: {history['pi_values'][-1]}"
    info = (f"Current Acc: {history['acc'][-1]:.4f}\n"
            f"Best Acc:    {max(history['acc']):.4f}\n"
            f"Scale:       {history['scale'][-1]:.1f}\n"
            f"Pass Rate:   {history['pass_rate'][-1]:.1f}%"
            f"{pi_str}")
    ax.text(0.05, 0.5, info, fontsize=13, family='monospace')

    plt.tight_layout(); plt.savefig(outpath); plt.close()


@torch.no_grad()
def sample_and_save(model, cfg, out_path, n_per_class=10, cluster_mapping=None):
    T = model.dpm_process.timesteps
    model.cond_denoiser.eval()
    K = cfg.num_classes
    device = cfg.device

    shape = (n_per_class * K, cfg.image_channels, 28, 28)
    x_t = torch.randn(shape, device=device)

    if cluster_mapping is not None:
        label2cluster = {v: k for k, v in cluster_mapping.items()}
        ordered = [label2cluster.get(d, d) for d in range(K)]
        y_cond = torch.tensor(ordered, device=device).repeat_interleave(n_per_class).long()
    else:
        y_cond = torch.arange(K, device=device).repeat_interleave(n_per_class).long()

    y_vec = F.one_hot(y_cond, K).float()

    for i in reversed(range(0, T)):
        t_batch = torch.full((shape[0],), i, device=device, dtype=torch.long)
        alpha_t = model.dpm_process._extract_t(model.dpm_process.alphas, t_batch, shape)
        sqrt_oma = model.dpm_process._extract_t(
            model.dpm_process.sqrt_one_minus_alphas_cumprod, t_batch, shape)
        pred_noise = model.cond_denoiser(x_t, t_batch, y_vec)
        mu = (x_t - (1 - alpha_t) / sqrt_oma * pred_noise) / alpha_t.sqrt()
        sigma = model.dpm_process._extract_t(
            model.dpm_process.posterior_variance, t_batch, shape).sqrt()
        noise = torch.randn_like(x_t) if i > 0 else torch.zeros_like(x_t)
        x_t = mu + sigma * noise

    save_image(x_t.clamp(-1, 1), out_path, nrow=n_per_class,
               normalize=True, value_range=(-1, 1))
    print(f"  ✓ samples → {out_path}")


@torch.no_grad()
def diagnose_conditioning(model, loader, cfg):
    model.eval()
    device = cfg.device
    K = cfg.num_classes
    x_0, _ = next(iter(loader))
    x_0 = x_0[:16].to(device)

    print(f"\n🔬 Conditioning Diagnostic:")
    print(f"   {'t':>6s}  {'diff':>10s}  {'norm':>10s}  {'ratio':>10s}")
    print(f"   {'-'*42}")

    for t_val in [5, 50, 200, 500]:
        t = torch.full((16,), t_val, device=device, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = model.dpm_process.q_sample(x_0, t, noise)
        outputs = []
        for k in range(K):
            y_oh = F.one_hot(torch.full((16,), k, device=device, dtype=torch.long), K).float()
            outputs.append(model.cond_denoiser(x_t, t, y_oh))
        diffs = [(outputs[i] - outputs[j]).pow(2).mean().item()
                 for i in range(K) for j in range(i + 1, K)]
        avg_diff = np.mean(diffs)
        avg_norm = np.mean([o.pow(2).mean().item() for o in outputs])
        ratio = avg_diff / (avg_norm + 1e-9)
        marker = " ✅" if ratio > 0.001 else " ⚠️"
        print(f"   t={t_val:>4d}  {avg_diff:>10.6f}  {avg_norm:>10.6f}  {ratio:>10.6f}{marker}")
    print()


# ============================================================
# 4. KMeans Initialization (可选)
# ============================================================
def kmeans_init(loader, cfg):
    print("🔄 Computing KMeans initialization...")
    features, true_labels = [], []
    for x, y in loader:
        features.append(x.view(x.size(0), -1))
        true_labels.append(y)
    features = torch.cat(features).numpy()
    ys = torch.cat(true_labels).numpy()

    kmeans = KMeans(n_clusters=cfg.num_classes, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    K = cfg.num_classes
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = -np.sum((ys == i) & (labels == j))
    ri, ci = linear_sum_assignment(cost)
    mapping = {int(c): int(l) for c, l in zip(ci, ri)}
    aligned = np.array([mapping.get(p, 0) for p in labels])
    print(f"   KMeans Acc: {np.mean(aligned == ys):.4f}  NMI: {NMI(ys, labels):.4f}")

    cluster_sizes = np.bincount(labels, minlength=K)
    cluster_props = cluster_sizes / cluster_sizes.sum()
    print(f"   Cluster sizes: {cluster_sizes.tolist()}")
    return centroids, cluster_props


def pretrain_with_kmeans(model, unlabeled_loader, val_loader, cfg,
                         centroids, pretrain_epochs=20):
    print(f"\n🏋️ Pretrain: {pretrain_epochs} epochs with KMeans pseudo-labels")
    pretrain_lr = getattr(cfg, 'pretrain_lr', 2e-4)
    print(f"   Pretrain LR: {pretrain_lr}")

    pretrain_opt = torch.optim.Adam(model.parameters(), lr=pretrain_lr)
    centroids_dev = centroids.to(cfg.device)

    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        ep_loss, n_batches = 0.0, 0
        for x_0, _ in unlabeled_loader:
            x_0 = x_0.to(cfg.device)
            with torch.no_grad():
                flat = x_0.view(x_0.size(0), -1)
                pseudo_y = torch.cdist(flat, centroids_dev).argmin(dim=1)
            loss, _ = model(x_0, cfg, y=pseudo_y)
            pretrain_opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            pretrain_opt.step()
            ep_loss += loss.item(); n_batches += 1

        avg_loss = ep_loss / max(n_batches, 1)
        val_acc, _, val_nmi = evaluate_model(model, val_loader, cfg)
        print(f"  [Pretrain] Ep {epoch}/{pretrain_epochs} "
              f"| Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} NMI: {val_nmi:.4f}")

    print("  ✅ Pretrain complete\n")


# ============================================================
# 5. Training Engine
# ============================================================
def run_training_session(model, optimizer, labeled_loader, unlabeled_loader,
                         val_loader, cfg, is_final_training=False,
                         trial=None, hyperparams=None):

    total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs

    if hyperparams is None:
        hyperparams = {'target_scale': 134.0, 'warmup_epochs': 10, 'threshold_final': 0.036}

    target_scale = hyperparams.get('target_scale', 134.0)
    warmup_epochs = hyperparams.get('warmup_epochs', 10)
    threshold_final = hyperparams.get('threshold_final', 0.036)
    lambda_pi = hyperparams.get('lambda_pi', 0.0)
    pi_start_epoch = hyperparams.get('pi_start_epoch', 30)  # π 更新延迟到 Ep30+

    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    best_val_acc = 0.0
    best_cluster_mapping = None
    has_labeled = labeled_loader is not None

    history = {"loss": [], "acc": [], "nmi": [],
               "pass_rate": [], "scale": [],
               "pi_entropy": [], "pi_values": []}

    for epoch in range(1, total_epochs + 1):

        # ── Scale 调度: 5 → 20 → target_scale ──
        # 始终用 soft sampling (论文 M-step), scale 退火自动实现 soft→近似hard 的平滑过渡
        if epoch <= warmup_epochs:
            p1 = epoch / warmup_epochs
            dynamic_scale = 5.0 + (20.0 - 5.0) * p1
            status = "EXPLORE"
        else:
            p2 = (epoch - warmup_epochs) / (total_epochs - warmup_epochs + 1e-8)
            dynamic_scale = 20.0 + (target_scale - 20.0) * p2
            status = "REFINE"

        # π 信息
        pi_np = model.pi.detach().cpu().numpy()
        pi_entropy = -np.sum(pi_np * np.log(pi_np + 1e-9))

        if is_final_training:
            pi_str = ", ".join([f"{p:.3f}" for p in pi_np])
            pi_status = f"λ_π={lambda_pi}" if epoch >= pi_start_epoch else "π=frozen"
            print(f"🔥 [Ep {epoch}/{total_epochs}] [{status}] "
                  f"Scale={dynamic_scale:.1f} [{pi_status}] "
                  f"π=[{pi_str}] H(π)={pi_entropy:.3f}")

        model.train()
        ep_loss, ep_dpm, ep_label, ep_mask = 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        # Labeled (半监督)
        if has_labeled:
            for x_lab, y_lab in labeled_loader:
                x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device)
                optimizer.zero_grad()
                sup_loss, _ = model(x_lab, cfg, y=y_lab)
                sup_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # Unlabeled (EM)
        if unlabeled_loader is not None:
            for x_un, _ in unlabeled_loader:
                x_un = x_un.to(cfg.device)
                optimizer.zero_grad()

                loss, info = model(x_un, cfg, y=None,
                                   scale_factor=dynamic_scale,
                                   lambda_pi=lambda_pi if epoch >= pi_start_epoch else 0.0)

                total_loss = cfg.alpha_unlabeled * loss
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                ep_loss += total_loss.item()
                ep_dpm += info['dpm_loss']
                ep_label += info['label_loss']
                ep_mask += info['mask_rate']
                n_batches += 1

        # Validation
        val_acc, cluster_mapping, val_nmi = evaluate_model(model, val_loader, cfg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cluster_mapping = cluster_mapping
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'acc': val_acc, 'nmi': val_nmi, 'params': hyperparams,
                'cluster_mapping': cluster_mapping,
            }, os.path.join(cfg.output_dir, "best_model.pt"))
            if is_final_training:
                print(f"   ★ New Best! Acc={best_val_acc:.4f}")

        N = max(n_batches, 1)
        avg_loss = ep_loss / N
        avg_dpm = ep_dpm / N
        avg_label = ep_label / N
        pass_pct = (ep_mask / N) * 100

        history["loss"].append(avg_loss)
        history["acc"].append(val_acc)
        history["nmi"].append(val_nmi)
        history["pass_rate"].append(pass_pct)
        history["scale"].append(dynamic_scale)
        history["pi_entropy"].append(pi_entropy)
        history["pi_values"].append(", ".join([f"{p:.3f}" for p in pi_np]))

        if is_final_training:
            label_str = f" π_loss={avg_label:.4f}×{lambda_pi}" if lambda_pi > 0 else ""
            print(f"  → Loss={avg_loss:.4f} (dpm={avg_dpm:.4f}{label_str}) "
                  f"| Acc={val_acc:.4f} NMI={val_nmi:.4f} | Pass={pass_pct:.1f}%")
            plot_dashboard(history, os.path.join(cfg.output_dir, "dashboard.png"))
            if epoch % 5 == 0:
                sample_and_save(model, cfg,
                                os.path.join(sample_dir, f"epoch_{epoch:03d}.png"),
                                cluster_mapping=cluster_mapping)

        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

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
    target_scale = trial.suggest_float("target_scale", 120.0, 180.0)
    warmup_epochs = trial.suggest_int("warmup_epochs", 10, 20)
    threshold_final = trial.suggest_float("threshold_final", 0.0, 0.1)

    hyperparams = {'target_scale': target_scale,
                   'warmup_epochs': warmup_epochs,
                   'threshold_final': threshold_final,
                   'lambda_pi': 0.0}

    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    _, unlabeled_loader, val_loader = get_semi_loaders(cfg)

    acc, _ = run_training_session(
        model, optimizer, None, unlabeled_loader, val_loader, cfg,
        is_final_training=False, trial=trial, hyperparams=hyperparams
    )
    return acc


def main():
    set_seed(2026)

    # ╔══════════════════════════════════════════════════════════╗
    # ║  配置区                                                  ║
    # ╚══════════════════════════════════════════════════════════╝
    TRAINING_MODE = "unsupervised"    # "unsupervised" | "semi_supervised"
    LABELED_PER_CLASS = 100           # 半监督: 每类标注数量
    ENABLE_PRETRAIN = False           # True = KMeans pretrain, False = 直接 EM
    SKIP_PRETRAIN = False             # True = 从 checkpoint 恢复
    ENABLE_AUTO_SEARCH = False

    # π 更新配置 (论文 M-step)
    ENABLE_PI_UPDATE = True           # True = 开启 π 更新 (论文要求)
    LAMBDA_PI = 0.01                  # label_loss 系数 (平衡 dpm_loss≈0.02 和 label_loss≈2.3)

    cfg = Config()
    cfg.alpha_unlabeled = 1.0
    cfg.posterior_sample_steps = 5
    cfg.pretrain_epochs = 20
    cfg.pretrain_lr = 2e-4

    if TRAINING_MODE == "unsupervised":
        cfg.labeled_per_class = 0
        print("=" * 50)
        print("🔓 Mode: UNSUPERVISED")
        print("=" * 50)
    elif TRAINING_MODE == "semi_supervised":
        cfg.labeled_per_class = LABELED_PER_CLASS
        print("=" * 50)
        print(f"🏷️  Mode: SEMI-SUPERVISED ({LABELED_PER_CLASS}/class)")
        print("=" * 50)

    if ENABLE_AUTO_SEARCH:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        best_params = {
            'target_scale': study.best_params['target_scale'],
            'warmup_epochs': study.best_params['warmup_epochs'],
            'threshold_final': study.best_params['threshold_final'],
        }
        best_lr = study.best_params['lr']
    else:
        best_params = {'target_scale': 134.37, 'warmup_epochs': 10, 'threshold_final': 0.036}
        best_lr = 4.01e-05

    # 添加 π 更新参数
    best_params['lambda_pi'] = LAMBDA_PI if ENABLE_PI_UPDATE else 0.0
    best_params['pi_start_epoch'] = 30  # 延迟到 Ep30: Scale≈60, Acc 应已稳定

    pi_info = f"λ_π={LAMBDA_PI} (starts Ep30)" if ENABLE_PI_UPDATE else "π=fixed"
    print(f"\n🚀 Training: LR={best_lr:.2e}, {pi_info}, Params={best_params}")

    cfg.final_epochs = 60
    cfg.output_dir = "./mDPM_results"
    os.makedirs(cfg.output_dir, exist_ok=True)

    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)

    pretrain_ckpt = os.path.join(cfg.output_dir, "pretrain_checkpoint.pt")

    # ── Phase 0: Pretrain (可选) ──
    if ENABLE_PRETRAIN:
        if SKIP_PRETRAIN and os.path.exists(pretrain_ckpt):
            print(f"\n⏩ Loading checkpoint: {pretrain_ckpt}")
            ckpt = torch.load(pretrain_ckpt, map_location=cfg.device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"   Acc={ckpt.get('acc', '?')}")
        else:
            if TRAINING_MODE == "unsupervised":
                centroids, cluster_props = kmeans_init(unlabeled_loader, cfg)
                # 初始化 π 从 KMeans 比例
                with torch.no_grad():
                    model.default_pi.copy_(torch.tensor(cluster_props, dtype=torch.float32))
                print(f"   π from KMeans: {model.pi.detach().cpu().numpy().round(3).tolist()}")
                pretrain_with_kmeans(model, unlabeled_loader, val_loader, cfg,
                                     centroids, pretrain_epochs=cfg.pretrain_epochs)

            elif TRAINING_MODE == "semi_supervised" and labeled_loader is not None:
                print(f"\n🏋️ Pretrain with REAL labels ({cfg.pretrain_epochs} ep)")
                pt_opt = torch.optim.Adam(model.parameters(), lr=cfg.pretrain_lr)
                for ep in range(1, cfg.pretrain_epochs + 1):
                    model.train()
                    ep_loss, n = 0.0, 0
                    for x, y in labeled_loader:
                        x, y = x.to(cfg.device), y.to(cfg.device)
                        loss, _ = model(x, cfg, y=y)
                        pt_opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        pt_opt.step(); ep_loss += loss.item(); n += 1
                    acc, _, nmi = evaluate_model(model, val_loader, cfg)
                    print(f"  [Pretrain] Ep {ep}/{cfg.pretrain_epochs} "
                          f"| Loss: {ep_loss/max(n,1):.4f} | Acc: {acc:.4f} NMI: {nmi:.4f}")
                print("  ✅ Pretrain complete\n")

            # Save checkpoint
            acc_now, _, nmi_now = evaluate_model(model, val_loader, cfg)
            torch.save({
                'model_state_dict': model.state_dict(),
                'acc': acc_now, 'nmi': nmi_now,
            }, pretrain_ckpt)
            print(f"💾 Checkpoint saved → {pretrain_ckpt} (Acc={acc_now:.4f})")
            print(f"   下次设 SKIP_PRETRAIN=True 即可跳过\n")
    else:
        print("\n⏩ No pretrain, starting EM directly")

    # 开启 π 更新
    if ENABLE_PI_UPDATE:
        model.enable_pi_update(True)
        print(f"📊 π update ENABLED (λ_π={LAMBDA_PI})")
    else:
        print(f"📊 π update DISABLED (fixed uniform)")

    diagnose_conditioning(model, val_loader, cfg)

    # ── Phase 1: EM Training ──
    print("=" * 50)
    print("🔄 EM Training...")
    print("=" * 50)

    best_acc, best_mapping = run_training_session(
        model, optimizer,
        labeled_loader if TRAINING_MODE == "semi_supervised" else None,
        unlabeled_loader, val_loader, cfg,
        is_final_training=True,
        hyperparams=best_params,
    )

    print(f"\n✅ Done. Best Acc: {best_acc:.4f}")
    sample_and_save(model, cfg, os.path.join(cfg.output_dir, "final_samples.png"),
                    cluster_mapping=best_mapping)

    # Save config
    cfg_dict = {k: v for k, v in vars(cfg).items()
                if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    cfg_dict['training_mode'] = TRAINING_MODE
    cfg_dict['enable_pi_update'] = ENABLE_PI_UPDATE
    cfg_dict['lambda_pi'] = LAMBDA_PI
    json.dump(cfg_dict, open(os.path.join(cfg.output_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()