# mDPM.py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import os
import gc  # 显式内存管理
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision.utils import save_image
import itertools  # <--- 必须加这个
# 导入 common 组件
from common_dpm import *

# -----------------------
# Model Definition (mDPM Adaptation)
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
        # 类别分布先验 (Uniform initialization)
        self.register_buffer('registered_pi', torch.ones(cfg.num_classes) / cfg.num_classes)
        
    # [修改] 增加 scale_factor 参数，接收动态调度值
    def estimate_posterior_logits(self, x_0, cfg, scale_factor=5.0):
        batch_size = x_0.size(0)
        num_classes = cfg.num_classes
        M = cfg.posterior_sample_steps
        
        accum_log_lik = torch.zeros(batch_size, num_classes, device=x_0.device)
        
        with torch.no_grad():
            for _ in range(M):
                # 语义区间 [300, 700]
                t_start = int(0.3 * cfg.timesteps)
                t_end = int(0.7 * cfg.timesteps)
                
                # 采样 t
                t = torch.randint(t_start, t_end, (batch_size,), device=x_0.device).long()
                
                noise = torch.randn_like(x_0)
                x_t = self.dpm_process.q_sample(x_0, t, noise)
                
                for k in range(num_classes):
                    y_cond = torch.full((batch_size,), k, device=x_0.device, dtype=torch.long)
                    y_onehot = F.one_hot(y_cond, num_classes=num_classes).float()
                    
                    pred_noise = self.cond_denoiser(x_t, t, y_onehot)
                    
                    # Log Likelihood Proxy
                    mse = F.mse_loss(pred_noise, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                    accum_log_lik[:, k] += -mse

        # [修改] 使用传入的动态 Scale
        accum_log_lik = accum_log_lik * scale_factor
        
        log_pi = torch.log(self.registered_pi + 1e-8).unsqueeze(0)
        final_logits = log_pi + (accum_log_lik / M)
        
        return final_logits

    # [修改] 增加 current_scale 和 current_lambda 参数
    def forward(self, x_0, cfg, y=None, current_scale=5.0, current_lambda=0.05):
        """
        前向传播包含 E-Step 和 M-Step 的损失计算
        """
        batch_size = x_0.size(0)
        t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.dpm_process.q_sample(x_0, t, noise)

        # -------------------
        # 监督模式 (Labeled Data)
        # -------------------
        if y is not None:
            t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.dpm_process.q_sample(x_0, t, noise)
            
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean') 
            
            return dpm_loss, -dpm_loss.item(), dpm_loss.item(), 0.0, None, None
            
        # -------------------
        # 无监督模式 (Unlabeled Data) - Soft-EM with Dynamic Annealing
        # -------------------
        else:
            # === E-Step: 推断潜变量 x 的分布 ===
            # [修改] 传入动态 Scale
            logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=current_scale)
            
            # 使用 Softmax 获取概率 (Soft-EM)
            # 在推理阶段，直接用 Softmax 比 Gumbel 更稳定，因为我们已经在 Logits 层面加了 scale
            resp = F.softmax(logits, dim=1)
            
            # === M-Step: 训练去噪网络 ===
            t_train = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
            
            weighted_dpm_loss = 0.0
            
            # 计算加权 Loss
            for k in range(cfg.num_classes):
                y_onehot_k = F.one_hot(torch.full((batch_size,), k, device=x_0.device),
                                       num_classes=cfg.num_classes).float()
                
                pred_noise_k = self.cond_denoiser(x_t_train, t_train, y_onehot_k)
                dpm_loss_k = F.mse_loss(pred_noise_k, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                
                weighted_dpm_loss += (resp[:, k].detach() * dpm_loss_k).mean()
            
            # === 辅助损失 ===
            # [修改] 使用动态 Lambda
            entropy = -(resp * torch.log(resp + 1e-8)).sum(dim=1).mean()
            total_loss = weighted_dpm_loss + current_lambda * entropy

            # --- 更新 Prior ---
            if self.training:
                with torch.no_grad():
                    current_counts = resp.mean(0)
                    self.registered_pi.copy_(0.99 * self.registered_pi + 0.01 * current_counts)
            
            return total_loss, -total_loss.item(), weighted_dpm_loss.item(), entropy.item(), resp.detach(), None
            
            
# -----------------------
# Evaluation Utils
# -----------------------
def evaluate_model(model, loader, cfg):
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("Scipy not found, skipping detailed evaluation.")
        return 0.0, {}, 0.0

    model.eval()
    preds, ys_true = [], []
    
    # 黄金区间评估
    eval_timesteps = [300, 500, 700] 
    n_repeats = 5
    
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
                        pred_noise = model.cond_denoiser(x_t, current_t, y_vec)
                        
                        loss = F.mse_loss(pred_noise, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                        mse_t_sum[:, k] += loss
                
                cumulative_mse += (mse_t_sum / n_repeats)

            pred_cluster = torch.argmin(cumulative_mse, dim=1).cpu().numpy()
            preds.append(pred_cluster)
            ys_true.append(y_true.numpy())

    preds = np.concatenate(preds)
    ys_true = np.concatenate(ys_true)
    
    # --- 计算指标 ---
    nmi = NMI(ys_true, preds)
    
    n_classes = cfg.num_classes
    cost_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cost_matrix[i, j] = -np.sum((ys_true == i) & (preds == j))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster2label = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned_preds = np.array([cluster2label.get(p, 0) for p in preds])
    posterior_acc = np.mean(aligned_preds == ys_true)
    
    return posterior_acc, cluster2label, nmi

def sample_and_save_dpm(denoiser, dpm_process, num_classes, out_path, device, n_per_class=10):
    """修正后的 DPM 逆向采样过程"""
    T = dpm_process.timesteps
    denoiser.eval()
    image_c = dpm_process.image_channels

    with torch.no_grad():
        shape = (n_per_class * num_classes, image_c, 28, 28)
        x_t = torch.randn(shape, device=device)
        y_cond = torch.arange(num_classes).to(device).repeat_interleave(n_per_class).long()
        
        for i in reversed(range(0, T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            alpha_t = dpm_process._extract_t(dpm_process.alphas, t, shape)
            one_minus_alpha_t_bar = dpm_process._extract_t(dpm_process.sqrt_one_minus_alphas_cumprod, t, shape)
            pred_noise = denoiser(x_t, t, y_cond)
            mu_t_1 = (x_t - (1 - alpha_t) / one_minus_alpha_t_bar * pred_noise) / alpha_t.sqrt()
            sigma_t_1 = dpm_process._extract_t(dpm_process.posterior_variance, t, shape).sqrt()
            
            if i > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
            x_t = mu_t_1 + sigma_t_1 * noise

        save_image(x_t.clamp(-1, 1), out_path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
    print(f"💾 Saved DPM samples to {out_path}")

# -----------------------
# Training Engine
# -----------------------
def run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                         is_final_training=False, trial_id=None):
    
    total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs
    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    metrics = {"Loss": [], "NMI": [], "Acc": []}
    best_val_nmi = -np.inf
    
    # 自动判断模式
    mode = "UNKNOWN"
    if labeled_loader is not None and unlabeled_loader is not None:
        mode = "SEMI_SUPERVISED"
        print("🚀 模式检测: 半监督训练 (Semi-Supervised)")
    elif labeled_loader is not None and unlabeled_loader is None:
        mode = "SUPERVISED"
        cfg.alpha_unlabeled = 0.0 
        print("🚀 模式检测: 全监督训练 (Fully Supervised)")
    elif labeled_loader is None and unlabeled_loader is not None:
        mode = "UNSUPERVISED"
        print("🚀 模式检测: 无监督训练 (Unsupervised)")
    else:
        raise ValueError("❌ 错误: Labeled 和 Unlabeled loader 不能同时为空！")

    # 训练循环
    for epoch in range(1, total_epochs + 1):
        
        # === 动态退火调度器 ===
        progress = epoch / total_epochs
        dynamic_scale = 3.0 + (10.0 - 3.0) * progress
        
        if epoch < 20:
            dynamic_lambda = 0.0
        else:
            denom = max(1, total_epochs - 20)
            lambda_progress = (epoch - 20) / denom
            dynamic_lambda = 0.0 + (0.2 - 0.0) * lambda_progress
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"🔥 [Scheduler] Epoch {epoch}: Scale={dynamic_scale:.2f}, Lambda={dynamic_lambda:.4f}")
            
        model.train()
        loss_accum = 0.0
        n_batches = 0
        
        # Warm-up: 前 10 Epoch alpha=0
        current_alpha_un = cfg.alpha_unlabeled
        if mode == "SEMI_SUPERVISED" and epoch <= 10:
            current_alpha_un = 0.0
        
        # === [关键修复] 构造迭代器 ===
        if mode == "SEMI_SUPERVISED":
            # 使用 itertools.cycle 确保 labeled_loader 循环使用，
            # 这样 Epoch 长度由 unlabeled_loader (长) 决定，而不是 labeled_loader (短)
            import itertools 
            iterator = zip(itertools.cycle(labeled_loader), unlabeled_loader)
        elif mode == "SUPERVISED":
            iterator = ((batch, None) for batch in labeled_loader)
        elif mode == "UNSUPERVISED":
            iterator = ((None, batch) for batch in unlabeled_loader)

        # Batch 循环
        for batch_lab, batch_un in iterator:
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=cfg.device)
            resp = None 

            # 有监督部分
            if batch_lab is not None:
                x_lab, y_lab = batch_lab
                x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device).long()
                loss_lab, _, _, _, _, _ = model(x_lab, cfg, y_lab)
                total_loss += loss_lab

            # 无监督部分
            if batch_un is not None and current_alpha_un > 0:
                x_un, _ = batch_un 
                x_un = x_un.to(cfg.device)
                loss_un, _, _, _, resp, _ = model(x_un, cfg, None, 
                                                  current_scale=dynamic_scale, 
                                                  current_lambda=dynamic_lambda)
                total_loss += current_alpha_un * loss_un
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss_accum += total_loss.item()
            n_batches += 1

            if resp is not None:
                with torch.no_grad():
                    momentum = 0.9 if mode == "UNSUPERVISED" else 0.99
                    model.registered_pi.copy_(momentum * model.registered_pi + (1-momentum) * resp.mean(0).detach())

        # 评估
        raw_acc, _, val_nmi = evaluate_model(model, val_loader, cfg)
        
        target_metric = raw_acc if mode == "SUPERVISED" else val_nmi
        if target_metric > best_val_nmi:
            best_val_nmi = target_metric
            if is_final_training:
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))

        log_tag = "FINAL" if is_final_training else f"TRIAL-{trial_id}"
        print(f"[{log_tag}] Mode: {mode} | Epoch {epoch} | Loss: {loss_accum/n_batches:.4f} | "
              f"Acc: {raw_acc:.4f} | NMI: {val_nmi:.4f}")

        if is_final_training and (epoch % 10 == 0 or epoch == total_epochs):
            sample_and_save_dpm(model.cond_denoiser, model.dpm_process, cfg.num_classes,
                                os.path.join(sample_dir, f"epoch_{epoch:03d}.png"), cfg.device)
    
    return best_val_nmi, metrics

def objective(trial):
    cfg = Config()
    cfg.output_dir = "./mDPM_optuna_temp"
    cfg.unet_base_channels = 32
    
    # Optuna 这里也可以搜 LR，但为了验证动态退火，我们建议手动设小 LR
    cfg.lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    
    model = None
    optimizer = None
    
    try:
        model = mDPM_SemiSup(cfg).to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)

        best_nmi, _ = run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                                           is_final_training=False, trial_id=trial.number)
        return -best_nmi
        
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.TrialPruned()
    finally:
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

def main():
    RUN_OPTUNA = False 
    cfg = Config()

    if RUN_OPTUNA:
        print("--- Starting Optuna Hyperparameter Search ---")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5) 
        print("Best params found:", study.best_params)
        for k, v in study.best_params.items():
            setattr(cfg, k, v)
        with open(os.path.join(cfg.output_dir, "optuna_best_params.json"), "w") as f:
            json.dump(study.best_params, f, indent=4)
    else:
        print("--- Skipping Optuna: Using Manual/Default Config ---")
        cfg.unet_base_channels = 32
        
        # [关键] 手动设置较小的学习率，配合 Hardening 阶段
        # cfg.lr = 2e-5 
        print(f"🎯 Manual LR set to: {cfg.lr}")
        
    print("\n" + "="*30)
    print("--- Starting Final Training (Dynamic Annealing Enabled) ---")
    print(f"Config: Channels={cfg.unet_base_channels}, LR={cfg.lr}")
    print("="*30 + "\n")

    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)
    
    run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg, is_final_training=True)
    
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "final_model.pt"))
    print(f"✅ Done. Model saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()