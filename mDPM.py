# mDPM.py — 在线 EM + 闭式解 π 更新版
# ═══════════════════════════════════════════════════════════════
# 回到验证过的在线 EM 框架 (Acc=0.5-0.6):
#   每个 batch 内: E-step(算posterior) + M-step(训denoiser)
#   标签每 batch 刷新 → conditioning 不会死
#
# π 更新: 闭式解 + clamp 防坍缩
#   原版 π 通过 label_loss 梯度更新 → rich-get-richer 坍缩
#   现版: π_k = count(label==k)/N, clamp(min=0.05) → 反映数据分布但不饿死
#
# 同时加入 conditioning diagnostic 监控 denoiser 健康度
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
# 1. Model — 在线 EM (原始框架)
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
        # π: 闭式解更新 (不用梯度), clamp 防饿死
        self.register_buffer('pi', torch.ones(cfg.num_classes) / cfg.num_classes)

    def update_pi_closed_form(self, all_pseudo_labels, min_ratio=0.5):
        """
        闭式解更新 π (借鉴 HMM-DPM):
          π_k = count(label==k) / N
          clamp: π_k >= 1/(2K) 防止任何 class 被饿死
        
        Args:
            all_pseudo_labels: 一个 epoch 所有 batch 的 pseudo_label 拼起来 [N]
            min_ratio: clamp 下限 = 1/(K * 1/min_ratio), 0.5 → min=1/(2K)=0.05
        """
        counts = torch.bincount(all_pseudo_labels, minlength=self.K).float()
        freq = counts / counts.sum()
        # clamp: 不低于 uniform * min_ratio = 0.1 * 0.5 = 0.05
        min_val = (1.0 / self.K) * min_ratio
        clamped = freq.clamp(min=min_val)
        new_pi = clamped / clamped.sum()
        self.pi.copy_(new_pi.to(self.pi.device))
        return new_pi.cpu().numpy()

    def estimate_posterior_logits(self, x_0, cfg, scale_factor=1.0):
        """
        E-step: 每个 batch 当场算 posterior
        logits = log π_k + (-avg_mse_k) * scale_factor
        π 固定 → log π 是常数, 不影响 argmax, 但保留以备将来开启
        """
        B = x_0.size(0)
        K = self.K
        M = getattr(cfg, 'posterior_sample_steps', 5)
        device = x_0.device

        accum_neg_mse = torch.zeros(B, K, device=device)

        with torch.no_grad():
            for _ in range(M):
                t = torch.randint(1, cfg.timesteps, (B,), device=device).long()
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
                use_hard_label=False, threshold=0.0):
        """
        在线 EM: 一个 forward 里同时做 E-step + M-step
        这是验证过能到 Acc=0.5-0.6 的原始框架
        """
        B = x_0.size(0)

        # E-step: 当场算 posterior
        logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=scale_factor)
        resp = F.softmax(logits, dim=1)

        if use_hard_label:
            # REFINE 阶段: hard argmax + confidence threshold
            max_probs, pseudo_labels = resp.max(dim=1)
            mask = (max_probs >= threshold).float()
            y_target = F.one_hot(pseudo_labels, num_classes=self.K).float()
        else:
            # EXPLORE 阶段: soft sampling (多样性)
            pseudo_labels = torch.multinomial(resp, 1).squeeze(1)
            y_target = F.one_hot(pseudo_labels, num_classes=self.K).float()
            mask = torch.ones(B, device=x_0.device)

        # M-step: 用 pseudo_label 训练 denoiser
        t_train = torch.randint(0, cfg.timesteps, (B,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.dpm_process.q_sample(x_0, t_train, noise)
        pred_noise = self.cond_denoiser(x_t, t_train, y_target)
        loss_per = F.mse_loss(pred_noise, noise, reduction='none').view(B, -1).mean(dim=1)
        dpm_loss = (loss_per * mask).sum() / (mask.sum() + 1e-8)

        # π 固定, 不需要 label_loss
        mask_rate = mask.mean().item()

        return dpm_loss, {
            'dpm_loss': dpm_loss.item(),
            'mask_rate': mask_rate,
            'pseudo_labels': pseudo_labels.detach(),
        }


# ============================================================
# 2. Evaluation
# ============================================================
@torch.no_grad()
def evaluate_model(model, loader, cfg):
    """
    稳定评估: 固定 t=T//2, 重复 3 次, argmin MSE
    """
    model.eval()
    preds, ys_true = [], []
    eval_t = cfg.timesteps // 2  # T=100 → t=50
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

    # 统计 class 频率
    freq = np.bincount(preds, minlength=K).astype(float)
    freq = freq / freq.sum()

    return acc, cluster2label, nmi_score, freq


# ============================================================
# 3. Conditioning Diagnostic
# ============================================================
@torch.no_grad()
def conditioning_diagnostic(model, loader, cfg, n_batches=3):
    """
    测量 denoiser 是否在使用条件输入
    ratio = (max_MSE - min_MSE) / avg_MSE
    ratio > 0.2 = 健康, < 0.05 = conditioning 死了
    """
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

    # 平均
    avg_ratios = {}
    for t_val, ratios in results.items():
        avg_ratios[t_val] = np.mean(ratios)

    return avg_ratios


# ============================================================
# 4. Dashboard & Sampling
# ============================================================
def plot_dashboard(history, outpath):
    n = len(history.get("loss", []))
    if n == 0:
        return
    epochs = range(1, n + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Online EM + Closed-form π (Ep {n})', fontsize=18)

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
        for k in range(freq_arr.shape[1]):
            ax.plot(range(1, len(history["freq"])+1), freq_arr[:, k], alpha=0.6)
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        ax.set_title('Class Frequencies'); ax.set_ylim(0, 0.5)
    else:
        ax.set_title('Class Frequencies (pending)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=100)
    plt.close()


@torch.no_grad()
def sample_and_save(model, cfg, out_path, n_per_class=10, cluster_mapping=None):
    T = model.dpm_process.timesteps
    model.cond_denoiser.eval()
    K = cfg.num_classes
    imgs = []
    for k in range(K):
        # 如果有映射, 用映射后的 class 生成
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
            pred = model.cond_denoiser(x, t_, y_oh)
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
# 5. Training — 在线 EM (原始框架, π 固定)
# ============================================================
def run_training(model, optimizer, unlabeled_loader, val_loader, cfg,
                 hyperparams=None, is_final=True):
    """
    在线 EM 训练:
      每个 batch: E-step(算posterior) + M-step(训denoiser)
      标签每 batch 都不同 → conditioning 不死
      π 闭式解 + clamp → 不坍缩
    """
    if hyperparams is None:
        hyperparams = {}

    total_epochs = hyperparams.get('total_epochs', 60)
    target_scale = hyperparams.get('target_scale', 134.0)
    warmup_epochs = hyperparams.get('warmup_epochs', 10)
    threshold_final = hyperparams.get('threshold_final', 0.036)
    diag_every = hyperparams.get('diag_every', 5)

    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    best_val_acc = 0.0
    best_cluster_mapping = None

    history = {"loss": [], "acc": [], "nmi": [],
               "pass_rate": [], "scale": [],
               "cond_ratio": [], "freq": [], "pi_entropy": []}

    for epoch in range(1, total_epochs + 1):

        # ── Scale 调度 (验证过的: 5→20→134) ──
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

        # ── Training (在线 EM) ──
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
                               threshold=dynamic_threshold)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_mask += info['mask_rate']
            epoch_labels.append(info['pseudo_labels'].cpu())
            n_batches += 1

        # ── Epoch 统计 ──
        avg_loss = ep_loss / max(n_batches, 1)
        pass_pct = (ep_mask / max(n_batches, 1)) * 100

        # 统计本 epoch 的 label 分布
        all_labels = torch.cat(epoch_labels)
        label_freq = np.bincount(all_labels.numpy(), minlength=cfg.num_classes).astype(float)
        label_freq = label_freq / label_freq.sum()

        # ── 闭式解更新 π (每 epoch 结束后) ──
        pi_np = model.update_pi_closed_form(all_labels)
        pi_entropy = -np.sum(pi_np * np.log(pi_np + 1e-9))
        if is_final:
            pi_str = ", ".join([f"{p:.3f}" for p in pi_np])
            print(f"   π=[{pi_str}] H(π)={pi_entropy:.3f}")

        # ── Validation ──
        val_acc, cluster_mapping, val_nmi, val_freq = evaluate_model(
            model, val_loader, cfg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cluster_mapping = cluster_mapping
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'acc': val_acc, 'nmi': val_nmi,
                'cluster_mapping': cluster_mapping,
            }, os.path.join(cfg.output_dir, "best_model.pt"))
            if is_final:
                print(f"   ★ New Best! Acc={best_val_acc:.4f}")

        # ── Conditioning Diagnostic ──
        if epoch % diag_every == 0 or epoch == 1:
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

        # ── History ──
        history["loss"].append(avg_loss)
        history["acc"].append(val_acc)
        history["nmi"].append(val_nmi)
        history["pass_rate"].append(pass_pct)
        history["scale"].append(dynamic_scale)
        history["freq"].append(label_freq.tolist())
        history["pi_entropy"].append(pi_entropy)

        if is_final:
            freq_str = ", ".join([f"{f:.3f}" for f in label_freq])
            print(f"  → Loss={avg_loss:.4f} | Acc={val_acc:.4f} NMI={val_nmi:.4f} "
                  f"| Pass={pass_pct:.1f}% | freq=[{freq_str}]")

            plot_dashboard(history, os.path.join(cfg.output_dir, "dashboard.png"))
            if epoch % 5 == 0:
                sample_and_save(model, cfg,
                                os.path.join(sample_dir, f"epoch_{epoch:03d}.png"),
                                cluster_mapping=cluster_mapping)

    return best_val_acc, best_cluster_mapping


# ============================================================
# 6. Main
# ============================================================
def main():
    set_seed(2026)

    cfg = Config()
    cfg.labeled_per_class = 0
    cfg.posterior_sample_steps = 5
    cfg.output_dir = "./mDPM_results"
    cfg.final_epochs = 60

    print("=" * 60)
    print("🔓 Online EM + Closed-form π (clamp防坍缩)")
    print(f"   T={cfg.timesteps}, M={cfg.posterior_sample_steps}")
    print("=" * 60)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # 数据
    _, unlabeled_loader, val_loader = get_semi_loaders(cfg)

    # 模型
    model = mDPM(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4.01e-05)

    hyperparams = {
        'total_epochs': 60,
        'target_scale': 134.37,
        'warmup_epochs': 10,
        'threshold_final': 0.036,
        'diag_every': 5,
    }

    print(f"\n🚀 Training:")
    print(f"   LR=4.01e-05")
    print(f"   Scale: 5→20 (warmup {hyperparams['warmup_epochs']}ep) → 134 (refine)")
    print(f"   π = closed-form + clamp(min=0.05)")
    print(f"   Epochs: {hyperparams['total_epochs']}")

    best_acc, best_mapping = run_training(
        model, optimizer, unlabeled_loader, val_loader, cfg,
        hyperparams=hyperparams, is_final=True)

    print(f"\n✅ Done. Best Acc: {best_acc:.4f}")

    # 加载 best model 生成 samples
    best_ckpt = os.path.join(cfg.output_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        best_mapping = ckpt.get('cluster_mapping', best_mapping)
        print(f"   Loaded best model (Acc={ckpt.get('acc', '?'):.4f})")

    sample_and_save(model, cfg, os.path.join(cfg.output_dir, "final_samples.png"),
                    cluster_mapping=best_mapping)

    # Save config
    cfg_dict = {k: v for k, v in vars(cfg).items()
                if not k.startswith('_') and isinstance(v, (int, float, str, bool))}
    cfg_dict['hyperparams'] = hyperparams
    cfg_dict['best_acc'] = best_acc
    json.dump(cfg_dict, open(os.path.join(cfg.output_dir, "config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()