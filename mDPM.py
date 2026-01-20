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
        
    def estimate_posterior_logits(self, x_0, cfg, scale_factor=1.0):
        batch_size = x_0.size(0)
        num_classes = cfg.num_classes
        M = cfg.posterior_sample_steps
        
        # 使用负 MSE 累加器 (Log Likelihood ∝ -MSE)
        accum_neg_mse = torch.zeros(batch_size, num_classes, device=x_0.device)
        
        with torch.no_grad():
            for _ in range(M):
                # 采样时间步，建议覆盖中间大部分区域
                t = torch.randint(100, 900, (batch_size,), device=x_0.device).long()
                noise = torch.randn_like(x_0)
                x_t = self.dpm_process.q_sample(x_0, t, noise)
                
                # 计算所有类别的条件去噪误差
                for k in range(num_classes):
                    y_cond = torch.full((batch_size,), k, device=x_0.device, dtype=torch.long)
                    y_onehot = F.one_hot(y_cond, num_classes=num_classes).float()
                    pred_noise = self.cond_denoiser(x_t, t, y_onehot)
                    
                    # MSE (Batch,)
                    mse = F.mse_loss(pred_noise, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                    accum_neg_mse[:, k] += -mse

        # 平均化 Monte Carlo 步数
        avg_neg_mse = accum_neg_mse / M
        
        # 加入 Prior: log P(y) + Scale * log P(x|y)
        # 限制 log_pi 防止无穷小
        log_pi = torch.log(torch.clamp(self.registered_pi, min=1e-6)).unsqueeze(0)
        
        # [逻辑修复] 去除 Z-Score，使用直接缩放
        # Scale 很大 (e.g. 100) 因为 MSE 差值很小 (e.g. 0.01)
        final_logits = log_pi + (avg_neg_mse * scale_factor)

        return final_logits

    def forward(self, x_0, cfg, y=None, current_scale=100.0, current_lambda=0.05):
        """
        前向传播包含 E-Step 和 M-Step 的损失计算
        """
        batch_size = x_0.size(0)

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
        # 无监督模式 (Unlabeled Data) - Hard-EM
        # -------------------
        else:
            # === E-Step: 推断潜变量 y 的分布 ===
            logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=current_scale)
            resp = F.softmax(logits, dim=1) # Shape: (B, K)
            
            # === M-Step: Hard Sampling (伪标签) ===
            # 论文做法: 从后验分布中采样类别
            pseudo_y = torch.multinomial(resp, 1).squeeze(1) # (B,)
            
            # 构造训练数据
            t_train = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
            
            # 使用伪标签作为条件
            y_onehot_pseudo = F.one_hot(pseudo_y, num_classes=cfg.num_classes).float()
            
            # 计算 DPM Loss
            pred_noise = self.cond_denoiser(x_t_train, t_train, y_onehot_pseudo)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean')
            
            # === 辅助损失: 熵正则化 (Minimization) ===
            # 鼓励模型做出确信的预测
            entropy = -(resp * torch.log(resp + 1e-8)).sum(dim=1).mean()
            
            total_loss = dpm_loss + current_lambda * entropy

            # === Update Prior (Momentum) ===
            if self.training:
                with torch.no_grad():
                    # 动量更新，防止震荡
                    momentum = 0.99
                    current_counts = resp.mean(0).detach()
                    self.registered_pi.copy_(momentum * self.registered_pi + (1 - momentum) * current_counts)
            
            return total_loss, -total_loss.item(), dpm_loss.item(), entropy.item(), resp.detach(), None

        # -------------------
        # 无监督模式 (Unlabeled Data) - Soft-EM with Dynamic Annealing
        # -------------------
        # else:
        #     # === E-Step: 推断潜变量 x 的分布 ===
        #     # [修改] 传入动态 Scale
        #     logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=current_scale)
            
        #     # 使用 Softmax 获取概率 (Soft-EM)
        #     # 在推理阶段，直接用 Softmax 比 Gumbel 更稳定，因为我们已经在 Logits 层面加了 scale
        #     resp = F.softmax(logits, dim=1)
            
        #     # === M-Step: 训练去噪网络 ===
        #     t_train = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
        #     noise = torch.randn_like(x_0)
        #     x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
            
        #     weighted_dpm_loss = 0.0
            
        #     # 计算加权 Loss
        #     for k in range(cfg.num_classes):
        #         y_onehot_k = F.one_hot(torch.full((batch_size,), k, device=x_0.device),
        #                                num_classes=cfg.num_classes).float()
                
        #         pred_noise_k = self.cond_denoiser(x_t_train, t_train, y_onehot_k)
        #         dpm_loss_k = F.mse_loss(pred_noise_k, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                
        #         weighted_dpm_loss += (resp[:, k].detach() * dpm_loss_k).mean()
            
        #     # === 辅助损失 ===
        #     # [修改] 使用动态 Lambda
        #     entropy = -(resp * torch.log(resp + 1e-8)).sum(dim=1).mean()
        #     total_loss = weighted_dpm_loss + current_lambda * entropy

        #     # --- 更新 Prior ---
        #     if self.training:
        #         with torch.no_grad():
        #             current_counts = resp.mean(0)
        #             self.registered_pi.copy_(0.99 * self.registered_pi + 0.01 * current_counts)
            
        #     return total_loss, -total_loss.item(), weighted_dpm_loss.item(), entropy.item(), resp.detach(), None
        # 无监督部分
        if y is None:
            # === E-Step ===
            logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=current_scale)
            resp = F.softmax(logits, dim=1) # shape: (B, K)
            
            # === M-Step (使用 Hard Sampling 以符合论文并加速) ===
            # 论文: "By drawing samples... obtain class labels... proceed with noise prediction"
            
            # 1. 采样伪标签
            pseudo_y = torch.multinomial(resp, 1).squeeze(1) # (B,)
            
            # 2. 构造训练数据
            t_train = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
            
            y_onehot_pseudo = F.one_hot(pseudo_y, num_classes=cfg.num_classes).float()
            
            # 3. 计算 DPM Loss (只算一次 forward，不用算 K 次)
            pred_noise = self.cond_denoiser(x_t_train, t_train, y_onehot_pseudo)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean')
            
            # 4. 辅助 Loss (Entropy Regularization)
            entropy = -(resp * torch.log(resp + 1e-8)).sum(dim=1).mean()
            
            total_loss = dpm_loss + current_lambda * entropy

            # 更新 Prior (Momentum Update)
            if self.training:
                with torch.no_grad():
                    # 这里可以用 resp 的均值，也可以用 pseudo_y 的 one-hot 均值，resp 更平滑
                    self.registered_pi.copy_(0.99 * self.registered_pi + 0.01 * resp.mean(0).detach())
            
            return total_loss, -total_loss.item(), dpm_loss.item(), entropy.item(), resp.detach(), None
            
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
                         is_final_training=False, trial_id=None, resume_path=None):
    
    total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs
    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    start_epoch = 1
    best_val_nmi = -np.inf
    metrics = {"Loss": [], "NMI": [], "Acc": []}

    # === Resume Logic ===
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 Resuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_nmi = checkpoint.get('best_nmi', -np.inf)
        print(f"✅ Resumed at Epoch {start_epoch}, Best NMI: {best_val_nmi:.4f}")

    # Detect Mode
    mode = "UNKNOWN"
    if labeled_loader is not None and unlabeled_loader is not None: mode = "SEMI_SUPERVISED"
    elif labeled_loader is not None: mode = "SUPERVISED"; cfg.alpha_unlabeled = 0.0
    elif unlabeled_loader is not None: mode = "UNSUPERVISED"
    print(f"🚀 Training Mode: {mode}")

    # Training Loop
    for epoch in range(start_epoch, total_epochs + 1):
        # === Dynamic Scheduler ===
        # Scale: 50 -> 300 (适应 Raw MSE)
        progress = epoch / total_epochs
        dynamic_scale = 50.0 + (300.0 - 50.0) * progress
        
        # Lambda: 0.0 -> 0.2 (后期增强熵惩罚)
        if epoch < 10: dynamic_lambda = 0.0
        else: dynamic_lambda = 0.0 + (0.2) * ((epoch - 10) / (total_epochs - 10))
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"🔥 [Scheduler] Epoch {epoch}: Scale={dynamic_scale:.1f}, Lambda={dynamic_lambda:.4f}")

        model.train()
        loss_accum = 0.0
        n_batches = 0
        
        # Warm-up alpha
        current_alpha_un = cfg.alpha_unlabeled
        if mode == "SEMI_SUPERVISED" and epoch <= 5: current_alpha_un = 0.0
        
        # Iterator Setup
        if mode == "SEMI_SUPERVISED": iterator = zip(itertools.cycle(labeled_loader), unlabeled_loader)
        elif mode == "SUPERVISED": iterator = ((batch, None) for batch in labeled_loader)
        elif mode == "UNSUPERVISED": iterator = ((None, batch) for batch in unlabeled_loader)

        for batch_lab, batch_un in iterator:
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=cfg.device)
            resp = None 

            # Labeled
            if batch_lab is not None:
                x_lab, y_lab = batch_lab
                x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device).long()
                loss_lab, _, _, _, _, _ = model(x_lab, cfg, y_lab)
                total_loss += loss_lab

            # Unlabeled
            if batch_un is not None and current_alpha_un > 0:
                # [修复] 正确解包，获取 y_un_true 用于 Debug
                x_un, y_un_true = batch_un 
                x_un = x_un.to(cfg.device)
                y_un_true = y_un_true.to(cfg.device)
                
                loss_un, _, _, _, resp, _ = model(x_un, cfg, None, 
                                                  current_scale=dynamic_scale, 
                                                  current_lambda=dynamic_lambda)
                total_loss += current_alpha_un * loss_un
                
                # === 深度监控 ===
                if n_batches % 50 == 0:
                    with torch.no_grad():
                        pseudo_labels = resp.argmax(dim=1)
                        acc_unsup = (pseudo_labels == y_un_true).float().mean().item()
                        conf = resp.max(dim=1)[0].mean().item()
                        class_counts = torch.bincount(pseudo_labels, minlength=cfg.num_classes).cpu().numpy()
                        # 仅打印简要信息
                        print(f"   [Debug] Unsup Acc: {acc_unsup:.2f} | Conf: {conf:.2f} | Dist: {class_counts}")

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_accum += total_loss.item()
            n_batches += 1

        # Validation
        raw_acc, _, val_nmi = evaluate_model(model, val_loader, cfg)
        target_metric = raw_acc if mode == "SUPERVISED" else val_nmi
        
        # === Checkpointing ===
        # Save Last
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_nmi': best_val_nmi
        }
        torch.save(ckpt, os.path.join(cfg.output_dir, "checkpoint_last.pt"))
        
        # Save Best
        if target_metric > best_val_nmi:
            best_val_nmi = target_metric
            if is_final_training:
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
                print(f"   ★ New Best! NMI: {best_val_nmi:.4f}")

        log_tag = "FINAL" if is_final_training else f"TRIAL-{trial_id}"
        print(f"[{log_tag}] Ep {epoch} | Loss: {loss_accum/n_batches:.4f} | Acc: {raw_acc:.4f} | NMI: {val_nmi:.4f}")

        if is_final_training and (epoch % 10 == 0 or epoch == total_epochs):
            sample_and_save_dpm(model.cond_denoiser, model.dpm_process, cfg.num_classes,
                                os.path.join(sample_dir, f"epoch_{epoch:03d}.png"), cfg.device)
    
    return best_val_nmi, metrics

def main():
    # [开关] 是否断点续训
    RESUME_TRAINING = True  
    
    cfg = Config()
    print("="*30)
    print(f"--- Starting Training (M={cfg.posterior_sample_steps}) ---")
    print(f"Config: LR={cfg.lr}, Scale Range=50->300")
    print("="*30)

    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)
    
    resume_path = os.path.join(cfg.output_dir, "checkpoint_last.pt") if RESUME_TRAINING else None
    
    run_training_session(
        model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg, 
        is_final_training=True,
        resume_path=resume_path
    )
    
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "final_model.pt"))
    print(f"✅ Done. Results saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()