# mDPM.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as NMI
from torchvision.utils import save_image
import itertools
from common_dpm import *

# -----------------------
# 0. 随机种子设置 (Reproducibility)
# -----------------------
def set_seed(seed=42):
    """锁定所有随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed locked to {seed}")

# -----------------------
# 1. 绘图辅助函数 (简化版仪表盘)
# -----------------------
def plot_advanced_curves(history, outpath):
    """
    绘制 2x3 仪表盘：
    Row 1: Loss, Acc & NMI, Pass Rate
    Row 2: Scale, Threshold, Text Info
    """
    n = len(history["loss"])
    epochs = range(1, n + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Dashboard (Ep {n})', fontsize=18)
    
    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], 'b-', label='MSE Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Acc & NMI
    ax = axes[0, 1]
    ax.plot(epochs, history["acc"], 'r-', label='Accuracy')
    if "nmi" in history and len(history["nmi"]) > 0:
        ax.plot(epochs, history["nmi"], 'g--', label='NMI')
    ax.set_title('Clustering (Acc/NMI)')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Pass Rate
    ax = axes[0, 2]
    ax.plot(epochs, history["pass_rate"], 'm-', label='Pass Rate')
    ax.set_title('Pass Rate')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # 4. Scale
    ax = axes[1, 0]
    ax.plot(epochs, history["scale"], 'c-', label='Scale')
    ax.set_title('Dynamic Scale')
    ax.grid(True, alpha=0.3)
    
    # 5. Threshold
    ax = axes[1, 1]
    ax.plot(epochs, history["threshold"], 'orange', label='Threshold')
    ax.set_title('Dynamic Threshold')
    ax.grid(True, alpha=0.3)
    
    # 6. Text Info
    ax = axes[1, 2]
    ax.axis('off')
    
    info_text = (f"Current Acc: {history['acc'][-1]:.4f}\n"
                 f"Best Acc:    {max(history['acc']):.4f}\n"
                 f"Scale:       {history['scale'][-1]:.1f}\n"
                 f"Pass Rate:   {history['pass_rate'][-1]:.1f}%")
    ax.text(0.1, 0.5, info_text, fontsize=16, family='monospace')
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -----------------------
# 2. 模型定义 (保持不变)
# -----------------------
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
        self.register_buffer('registered_pi', torch.ones(cfg.num_classes) / cfg.num_classes)
        
    def estimate_posterior_logits(self, x_0, cfg, scale_factor=1.0):
        batch_size = x_0.size(0)
        num_classes = cfg.num_classes
        M = cfg.posterior_sample_steps
        
        accum_neg_mse = torch.zeros(batch_size, num_classes, device=x_0.device)
        
        with torch.no_grad():
            for _ in range(M):
                t = torch.randint(100, 900, (batch_size,), device=x_0.device).long()
                noise = torch.randn_like(x_0)
                x_t = self.dpm_process.q_sample(x_0, t, noise)
                
                for k in range(num_classes):
                    y_cond = torch.full((batch_size,), k, device=x_0.device, dtype=torch.long)
                    y_onehot = F.one_hot(y_cond, num_classes=num_classes).float()
                    pred_noise = self.cond_denoiser(x_t, t, y_onehot)
                    mse = F.mse_loss(pred_noise, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                    accum_neg_mse[:, k] += -mse

        avg_neg_mse = accum_neg_mse / M
        log_pi = torch.log(torch.clamp(self.registered_pi, min=1e-6)).unsqueeze(0)
        final_logits = log_pi + (avg_neg_mse * scale_factor)
        return final_logits

    def forward(self, x_0, cfg, y=None, current_scale=1.0, current_lambda=0.0, threshold=0.0, use_hard_label=False):
        batch_size = x_0.size(0)

        # Path A: 监督
        if y is not None:
            t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.dpm_process.q_sample(x_0, t, noise)
            
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean') 
            return dpm_loss, -dpm_loss.item(), dpm_loss.item(), 1.0, None, None
            
        # Path B: 无监督
        else:
            logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=current_scale)
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
            
            total_loss = dpm_loss
            mask_rate = mask.mean().item()
            return total_loss, -total_loss.item(), dpm_loss.item(), mask_rate, resp.detach(), None

# -----------------------
# 3. 评估与可视化
# -----------------------
def evaluate_model(model, loader, cfg):
    try:
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics import normalized_mutual_info_score as NMI
    except ImportError:
        return 0.0, {}, 0.0

    model.eval()
    preds, ys_true = [], []
    eval_timesteps = [500] 
    n_repeats = 3 
    
    with torch.no_grad():
        for x_0, y_true in loader:
            x_0 = x_0.to(cfg.device)
            batch_size = x_0.size(0)
            cumulative_mse = torch.zeros(batch_size, cfg.num_classes, device=cfg.device)
            
            for t_val in eval_timesteps:
                mse_t_sum = torch.zeros(batch_size, cfg.num_classes, device=cfg.device)
                for _ in range(n_repeats):
                    noise = torch.randn_like(x_0)
                    current_t = torch.full((batch_size,), t_val, device=cfg.device, dtype=torch.long)
                    x_t = model.dpm_process.q_sample(x_0, current_t, noise)
                    for k in range(cfg.num_classes):
                        y_vec = F.one_hot(torch.full((batch_size,), k, device=x_0.device), cfg.num_classes).float()
                        pred = model.cond_denoiser(x_t, current_t, y_vec)
                        loss = F.mse_loss(pred, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                        mse_t_sum[:, k] += loss
                cumulative_mse += (mse_t_sum / n_repeats)

            pred_cluster = torch.argmin(cumulative_mse, dim=1).cpu().numpy()
            preds.append(pred_cluster)
            ys_true.append(y_true.numpy())

    preds = np.concatenate(preds)
    ys_true = np.concatenate(ys_true)
    
    # 1. Hungarian Matching
    n_classes = cfg.num_classes
    cost_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cost_matrix[i, j] = -np.sum((ys_true == i) & (preds == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    cluster2label = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned_preds = np.array([cluster2label.get(p, 0) for p in preds])
    acc = np.mean(aligned_preds == ys_true)
    
    # 2. NMI
    nmi_score = NMI(ys_true, preds)
    
    return acc, cluster2label, nmi_score

def sample_and_save_dpm(denoiser, dpm_process, num_classes, out_path, device, n_per_class=10, cluster_mapping=None):
    """
    生成图像网格 (按真实数字顺序)
    """
    T = dpm_process.timesteps
    denoiser.eval()
    image_c = dpm_process.image_channels

    with torch.no_grad():
        shape = (n_per_class * num_classes, image_c, 28, 28)
        x_t = torch.randn(shape, device=device)
        
        # 顺序映射
        if cluster_mapping is not None:
            label2cluster = {v: k for k, v in cluster_mapping.items()}
            ordered_internal_labels = []
            for true_digit in range(num_classes):
                internal_c = label2cluster.get(true_digit, true_digit)
                ordered_internal_labels.append(internal_c)
            y_cond = torch.tensor(ordered_internal_labels, device=device).repeat_interleave(n_per_class).long()
        else:
            y_cond = torch.arange(num_classes).to(device).repeat_interleave(n_per_class).long()
            
        y_cond_vec = F.one_hot(y_cond, num_classes).float()
        
        for i in reversed(range(0, T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            alpha_t = dpm_process._extract_t(dpm_process.alphas, t, shape)
            one_minus_alpha_t_bar = dpm_process._extract_t(dpm_process.sqrt_one_minus_alphas_cumprod, t, shape)
            pred_noise = denoiser(x_t, t, y_cond_vec)
            mu_t_1 = (x_t - (1 - alpha_t) / one_minus_alpha_t_bar * pred_noise) / alpha_t.sqrt()
            sigma_t_1 = dpm_process._extract_t(dpm_process.posterior_variance, t, shape).sqrt()
            
            if i > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
            x_t = mu_t_1 + sigma_t_1 * noise

        save_image(x_t.clamp(-1, 1), out_path, nrow=n_per_class, normalize=True, value_range=(-1, 1))

# -----------------------
# 4. 训练引擎
# -----------------------
def run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                         is_final_training=False, trial=None, hyperparams=None):
    
    total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs
    
    if hyperparams is None:
        hyperparams = {'target_scale': 150.0, 'warmup_epochs': 15, 'threshold_final': 0.0}

    target_scale = hyperparams.get('target_scale', 150.0)
    warmup_epochs = hyperparams.get('warmup_epochs', 15)
    threshold_final = hyperparams.get('threshold_final', 0.0)

    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    start_epoch = 1
    best_val_acc = 0.0
    
    # 历史记录 (移除了 FID/IS)
    history = {
        "loss": [], "acc": [], "nmi": [], 
        "pass_rate": [], "scale": [], "threshold": []
    }
    
    mode = "UNSUPERVISED"

    for epoch in range(start_epoch, total_epochs + 1):
        
        # 调度器
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
            dynamic_threshold = 0.0 + (threshold_final - 0.0) * p2
            status = "REFINE"

        if is_final_training and epoch % 1 == 0:
            print(f"🔥 [Scheduler] Ep {epoch} [{status}]: Scale={dynamic_scale:.1f}, Thres={dynamic_threshold:.3f}")

        model.train()
        loss_accum = 0.0
        mask_rate_accum = 0.0
        n_batches = 0
        
        iterator = ((None, batch) for batch in unlabeled_loader)

        for _, batch_un in iterator:
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=cfg.device)

            if batch_un is not None:
                x_un, _ = batch_un
                x_un = x_un.to(cfg.device)
                
                l_unsup, _, _, mask_rate, _, _ = model(x_un, cfg, y=None, 
                                                       current_scale=dynamic_scale,
                                                       current_lambda=0.01,
                                                       threshold=dynamic_threshold,
                                                       use_hard_label=use_hard)
                
                total_loss += cfg.alpha_unlabeled * l_unsup
                mask_rate_accum += mask_rate

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_accum += total_loss.item()
            n_batches += 1

        # Validation (Acc & NMI)
        val_acc, cluster_mapping, val_nmi = evaluate_model(model, val_loader, cfg)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': val_acc,
                'nmi': val_nmi,
                'params': hyperparams
            }
            torch.save(save_dict, os.path.join(cfg.output_dir, "best_model.pt"))
            if is_final_training:
                print(f"   ★ New Best Model Saved! (Acc: {best_val_acc:.4f})")

        # 统计 & 记录
        avg_loss = loss_accum / n_batches if n_batches > 0 else 0.0
        avg_mask = mask_rate_accum / n_batches if n_batches > 0 else 0
        pass_rate_percent = avg_mask * 100
        
        history["loss"].append(avg_loss)
        history["acc"].append(val_acc)
        history["nmi"].append(val_nmi)
        history["pass_rate"].append(pass_rate_percent)
        history["scale"].append(dynamic_scale)
        history["threshold"].append(dynamic_threshold)
        
        # 实时画图
        if trial is not None:
            curve_name = f"optuna_trial_{trial.number}_dashboard.png"
        else:
            curve_name = "final_training_dashboard.png"
        plot_advanced_curves(history, os.path.join(cfg.output_dir, curve_name))

        if is_final_training:
            print(f"Ep {epoch} | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} | NMI: {val_nmi:.4f} | Pass: {pass_rate_percent:.1f}%")
            
            # 画生成图 (每5轮)
            if epoch % 5 == 0:
                sample_and_save_dpm(
                    model.cond_denoiser, model.dpm_process, cfg.num_classes,
                    os.path.join(sample_dir, f"epoch_{epoch:03d}.png"), cfg.device,
                    cluster_mapping=cluster_mapping
                )
    
    return best_val_acc, {}

# -----------------------
# 5. 主入口
# -----------------------
def objective(trial):
    # Optuna logic (如果你想继续搜)
    cfg = Config()
    cfg.alpha_unlabeled = 1.0
    cfg.labeled_per_class = 0
    cfg.posterior_sample_steps = 5 
    cfg.optuna_epochs = 35 
    
    lr = trial.suggest_float("lr", 4e-5, 2e-4, log=True)
    target_scale = trial.suggest_float("target_scale", 120.0, 180.0)
    warmup_epochs = trial.suggest_int("warmup_epochs", 10, 20)
    threshold_final = trial.suggest_float("threshold_final", 0.0, 0.1)
    
    hyperparams = {
        'target_scale': target_scale,
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
    # [设置] 随机种子 (例如 42)
    set_seed(42)
    
    # 自动搜索开关
    ENABLE_AUTO_SEARCH = False 
    
    cfg = Config()
    cfg.alpha_unlabeled = 1.0
    cfg.labeled_per_class = 0
    cfg.posterior_sample_steps = 5 
    
    if ENABLE_AUTO_SEARCH:
        print("🔍 [Step 1] Starting Optuna Search...")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        
        print("\n" + "="*40 + "\n🎉 Search Finished!\n" + "="*40)
        best_params = {
            'target_scale': study.best_params['target_scale'],
            'warmup_epochs': study.best_params['warmup_epochs'],
            'threshold_final': study.best_params['threshold_final']
        }
        best_lr = study.best_params['lr']
        
    else:
        print("⏩ [Step 1] Skipping Search, using Trial 3 BEST params")
        # 您上次获得的最佳参数 (Acc ~0.605)
        best_params = {
            'target_scale': 134.37,
            'warmup_epochs': 10,
            'threshold_final': 0.036
        }
        best_lr = 4.01e-05

    print("\n🚀 [Step 2] Starting Final Training (Clean Version)...")
    print(f"   Configs: LR={best_lr:.2e}, Params={best_params}")
    
    cfg.final_epochs = 60 
    
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    _, unlabeled_loader, val_loader = get_semi_loaders(cfg)
    
    run_training_session(
        model, optimizer, None, unlabeled_loader, val_loader, cfg, 
        is_final_training=True,
        hyperparams=best_params
    )

if __name__ == "__main__":
    main()