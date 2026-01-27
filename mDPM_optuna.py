# mDPM_optuna.py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.trial import TrialState
import os
import gc
import random
import numpy as np
from scipy.optimize import linear_sum_assignment
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
# Model Definition (保持不变)
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

        # Path A: 监督模式
        if y is not None:
            t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.dpm_process.q_sample(x_0, t, noise)
            
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean') 
            return dpm_loss, -dpm_loss.item(), dpm_loss.item(), 1.0, None, None
            
        # Path B: 无监督模式
        else:
            logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=current_scale)
            resp = F.softmax(logits, dim=1) 
            
            if use_hard_label:
                # FixMatch Mode
                max_probs, pseudo_labels = resp.max(dim=1)
                mask = (max_probs >= threshold).float()
                y_target = F.one_hot(pseudo_labels, num_classes=cfg.num_classes).float()
            else:
                # Exploration Mode
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
# Evaluation Utils (保持不变)
# -----------------------
def evaluate_model(model, loader, cfg):
    try:
        from scipy.optimize import linear_sum_assignment
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
    
    n_classes = cfg.num_classes
    cost_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cost_matrix[i, j] = -np.sum((ys_true == i) & (preds == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster2label = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned_preds = np.array([cluster2label.get(p, 0) for p in preds])
    acc = np.mean(aligned_preds == ys_true)
    
    return acc, cluster2label, 0.0

def sample_and_save_dpm(denoiser, dpm_process, num_classes, out_path, device, n_per_class=10):
    T = dpm_process.timesteps
    denoiser.eval()
    image_c = dpm_process.image_channels

    with torch.no_grad():
        shape = (n_per_class * num_classes, image_c, 28, 28)
        x_t = torch.randn(shape, device=device)
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
# Training Engine (Modified for Optuna)
# -----------------------
def run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                         is_final_training=False, trial=None, hyperparams=None):
    
    total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs
    
    if hyperparams is None:
        hyperparams = {}

    target_scale = hyperparams.get('target_scale', 150.0)
    warmup_epochs = hyperparams.get('warmup_epochs', 10)
    threshold_start = hyperparams.get('threshold_start', 0.8)
    threshold_final = hyperparams.get('threshold_final', 0.95)
    
    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    start_epoch = 1
    best_val_acc = 0.0

    # 模式检测
    mode = "UNKNOWN"
    if cfg.labeled_per_class > 0 and labeled_loader is not None: 
        mode = "SEMI_SUPERVISED"
    else:
        mode = "UNSUPERVISED"
        cfg.alpha_unlabeled = 1.0

    print(f"🚀 Training Mode: {mode} (Total Epochs: {total_epochs})")
    metrics = {"DPM_Loss": [], "PosteriorAcc": []}

    for epoch in range(start_epoch, total_epochs + 1):
        progress = (epoch - 1) / total_epochs
        
        # ==========================================
        # [调度器] 分支处理：互不干扰
        # ==========================================
        
        if mode == "UNSUPERVISED":
            # 【保持原样】您原来的无监督逻辑 (Soft Explore -> Hard Refine)
            current_alpha = 1.0 # 无监督始终需要无标签数据
            
            if epoch <= warmup_epochs:
                use_hard = False
                p1 = epoch / warmup_epochs
                dynamic_scale = 5.0 + (20.0 - 5.0) * p1
                dynamic_threshold = 0.0 
                status = "EXPLORE (Unsup)"
            else:
                use_hard = True
                p2 = (epoch - warmup_epochs) / (total_epochs - warmup_epochs + 1e-8)
                dynamic_scale = 20.0 + (target_scale - 20.0) * p2
                dynamic_threshold = 0.0 + (threshold_final - 0.0) * p2
                status = "REFINE (Unsup)"

        elif mode == "SEMI_SUPERVISED":
            # 【本次修改】半监督专用逻辑 (Sup Warmup -> FixMatch)
            
            if epoch <= warmup_epochs:
                # Phase 1: 纯监督预热 (前10轮)
                # 关键：强制 alpha=0，完全屏蔽无标签噪音
                current_alpha = 0.0
                
                # 只是为了日志显示正常，参数设为监督模式
                dynamic_scale = 1.0 
                dynamic_threshold = 0.0
                use_hard = True
                status = "WARMUP (Sup Only)"
            
            else:
                # Phase 2: 正式半监督 (第11轮起)
                current_alpha = 1.0
                
                # 开启高 Scale 和 FixMatch
                dynamic_scale = target_scale
                use_hard = True
                
                # 门槛爬坡
                p_refine = (epoch - warmup_epochs) / (total_epochs - warmup_epochs + 1e-8)
                dynamic_threshold = threshold_start + (threshold_final - threshold_start) * p_refine
                status = "SEMI-SUP (FixMatch)"
                
        # ------------------------------------------

        if is_final_training and epoch % 1 == 0:
            print(f"🔥 [Scheduler] Ep {epoch} [{status}]: Alpha={current_alpha}, Scale={dynamic_scale:.1f}, Thres={dynamic_threshold:.2f}")

        model.train()
        loss_accum = 0.0
        mask_rate_accum = 0.0
        n_batches = 0
        
        if mode == "SEMI_SUPERVISED": 
            iterator = zip(itertools.cycle(labeled_loader), unlabeled_loader)
        else:
            iterator = ((None, batch) for batch in unlabeled_loader)

        for batch_lab, batch_un in iterator:
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=cfg.device)

            # 1. 监督 Loss
            if batch_lab is not None:
                x_lab, y_lab = batch_lab
                x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device).long()
                l_sup, _, _, _, _, _ = model(x_lab, cfg, y=y_lab)
                total_loss += l_sup

            # 2. 无监督 Loss (仅当 Alpha > 0 时计算)
            if batch_un is not None:
                if current_alpha > 0: # 关键判断
                    x_un, _ = batch_un
                    x_un = x_un.to(cfg.device)
                    
                    l_unsup, _, _, mask_rate, _, _ = model(x_un, cfg, y=None, 
                                                           current_scale=dynamic_scale,
                                                           current_lambda=0.01,
                                                           threshold=dynamic_threshold,
                                                           use_hard_label=use_hard)
                    
                    total_loss += current_alpha * l_unsup
                    mask_rate_accum += mask_rate

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_accum += total_loss.item()
            n_batches += 1

        # Validation ... (保持不变)
        val_acc, _, _ = evaluate_model(model, val_loader, cfg)
        if val_acc > best_val_acc: best_val_acc = val_acc
        
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()

        avg_loss = loss_accum / n_batches if n_batches > 0 else 0.0
        avg_mask = mask_rate_accum / n_batches if n_batches > 0 else 0
        metrics["DPM_Loss"].append(avg_loss)
        metrics["PosteriorAcc"].append(val_acc)
        
        if trial is not None:
            curve_name = f"optuna_trial_{trial.number}_curve.png"
        else:
            curve_name = "training_curves_final.png"
        plot_training_curves(metrics, os.path.join(cfg.output_dir, curve_name))

        if is_final_training:
            print(f"Ep {epoch} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | Pass: {avg_mask*100:.1f}%")
            if epoch % 5 == 0:
                sample_and_save_dpm(model.cond_denoiser, model.dpm_process, cfg.num_classes,
                                    os.path.join(sample_dir, f"epoch_{epoch:03d}.png"), cfg.device)
    
    return best_val_acc, {}


# -----------------------
# Optuna Objective
# -----------------------
def objective(trial):
    cfg = Config()
    
    # ==========================================
    # 1. 设置搜索模式
    # ==========================================
    SEARCH_MODE = "SEMI_SUPERVISED" 
    
    # ==========================================
    # 2. 定义搜索空间
    # ==========================================
    if SEARCH_MODE == "UNSUPERVISED":
        cfg.labeled_per_class = 0
        cfg.alpha_unlabeled = 1.0
        
        lr = trial.suggest_float("lr", 4e-5, 2e-4, log=True)
        target_scale = trial.suggest_float("target_scale", 120.0, 180.0)
        warmup_epochs = trial.suggest_int("warmup_epochs", 10, 20)
        threshold_start = 0.0 
        threshold_final = trial.suggest_float("threshold_final", 0.0, 0.1)

    else: # SEMI_SUPERVISED
        cfg.labeled_per_class = 10  
        cfg.alpha_unlabeled = 1.0
        # 搜参时降 M 以加速
        cfg.posterior_sample_steps = 1
        
        lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
        target_scale = trial.suggest_float("target_scale", 100.0, 160.0)
        warmup_epochs = trial.suggest_int("warmup_epochs", 5, 10)
        # warmup_epochs = 0 # 半监督不用 warmup
        threshold_start = trial.suggest_float("threshold_start", 0.3, 0.7)
        threshold_final = trial.suggest_float("threshold_final", 0.8, 0.99)
    
    cfg.optuna_epochs = 15
    
    hyperparams = {
        'target_scale': target_scale,
        'warmup_epochs': warmup_epochs,
        'threshold_start': threshold_start,
        'threshold_final': threshold_final
    }
    
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)
    
    accuracy, _ = run_training_session(
        model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
        is_final_training=False, 
        trial=trial,
        hyperparams=hyperparams
    )
    
    return accuracy

def main():
    set_seed(42)
    
    # 模式开关
    ENABLE_AUTO_SEARCH = False 
    
    if ENABLE_AUTO_SEARCH:
        pass # (搜索代码略)
    else:
        print("⏩ [Step 1] Skipping Search, using SEMI-SUPERVISED params...")
        
        # [半监督专用参数]
        best_params = {
            'target_scale': 150.0,
            
            # 这里的 warmup_epochs 对应 SEMI_SUPERVISED 分支里的“纯监督预热”时长
            'warmup_epochs': 10,        
            
            # 正式开始半监督时的门槛
            'threshold_start': 0.80,    
            'threshold_final': 0.95
        }
        best_lr = 4.48e-4

    print("\n🚀 [Step 2] Starting Final Training...")
    
    cfg = Config()
    cfg.final_epochs = 60
    cfg.posterior_sample_steps = 5 
    
    # 确保是半监督模式
    cfg.labeled_per_class = 10 
    
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)
    
    run_training_session(
        model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg, 
        is_final_training=True,
        hyperparams=best_params
    )

if __name__ == "__main__":
    main()