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
        # 无监督模式 (Unlabeled Data) - 改为 Gumbel-Softmax
        # -------------------
        else:
            # === E-Step: 推断潜变量 y 的分布 ===
            # 获取 Logits (注意：Scale 依然很重要，Logits 差异过小会导致 Gumbel 输出趋向均匀)
            logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=current_scale)
            
            # 仅用于指标监控 (Softmax)
            resp = F.softmax(logits, dim=1) 

            
            # === M-Step: Gumbel-Softmax Sampling ===
            # [关键修改] 使用可导的软采样
            # hard=False 表示我们需要软概率向量 (e.g., [0.1, 0.8, 0.1]) 而不是 One-hot
            # 这样梯度可以流向所有类别，如果不确定是类2还是类7，两个类都会得到更新
            # [新增] 计算置信度权重
            # 获取每个样本最大的概率值 (B,)
            max_probs, _ = resp.max(dim=1)
            
            # [关键策略] 只有置信度 > 0.4 的样本才贡献 Loss
            # 或者是软权重: weight = max_probs^2 (让确信的样本权重更大)
            mask = (max_probs > 0.4).float() 

            y_soft = F.gumbel_softmax(logits, tau=gumbel_temp, hard=False)
            
            t_train = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
            
            pred_noise = self.cond_denoiser(x_t_train, t_train, y_soft)
            
            # 计算 element-wise MSE: (B, C, H, W) -> (B,)
            loss_per_sample = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=[1,2,3])
            
            # [关键] 应用 Mask，只训练高质量样本
            # 加上一个极小值防止除以0
            dpm_loss = (loss_per_sample * mask).sum() / (mask.sum() + 1e-8)
        
            
            # === 辅助损失: 熵正则化 ===
            entropy = -(resp * torch.log(resp + 1e-8)).sum(dim=1).mean()
            
            total_loss = dpm_loss + current_lambda * entropy

            # === Update Prior (Momentum) ===
            if self.training:
                with torch.no_grad():
                    # 稍微减慢更新速度，防止初期波动
                    momentum = 0.995 
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

    # ==========================================
    # 🔄 Resume Logic (断点续训)
    # ==========================================
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 Resuming from: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=cfg.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        best_val_nmi = checkpoint.get('best_nmi', -np.inf)
        print(f"✅ Resumed at Epoch {start_epoch}, Best NMI: {best_val_nmi:.4f}")
    else:
        if resume_path:
            print(f"⚠️ Checkpoint not found at {resume_path}, starting from scratch.")

    # ==========================================
    # Detect Mode (模式检测)
    # ==========================================
    mode = "UNKNOWN"
    if labeled_loader is not None and unlabeled_loader is not None: 
        mode = "SEMI_SUPERVISED"
    elif labeled_loader is not None: 
        mode = "SUPERVISED"
        cfg.alpha_unlabeled = 0.0
    elif unlabeled_loader is not None: 
        mode = "UNSUPERVISED"
    
    print(f"🚀 Training Mode: {mode}")

    # ==========================================
    # Training Loop
    # ==========================================
    for epoch in range(start_epoch, total_epochs + 1):
        # --- Schedulers (关键参数调度) ---
        progress = (epoch - 1) / total_epochs # 0.0 -> 1.0
        
        # 1. Dynamic Scale: 放大 Logits 差异
        # [修改] 提高起始值到 150，防止初期 Logits 太平滑导致 Gumbel 也是均匀分布
        dynamic_scale = 300.0 + (600.0 - 300.0) * progress
        
        # 2. Dynamic Lambda: 熵惩罚
        if epoch < 10: 
            dynamic_lambda = 0.0
        else: 
            dynamic_lambda = 0.0 + (0.2) * ((epoch - 10) / (max(1, total_epochs - 10)))
        
        # 3. [新增] Gumbel Temperature: 探索 -> 确定
        # 初期 1.0 (平滑，探索所有类)，后期 0.5 (尖锐，接近 One-hot)
        gumbel_temp = 1.0 - (0.5 * progress)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"🔥 [Scheduler] Ep {epoch}: Scale={dynamic_scale:.1f}, Lambda={dynamic_lambda:.4f}, Temp={gumbel_temp:.2f}")

        model.train()
        loss_accum = 0.0
        n_batches = 0
        
        # Warm-up alpha (前5轮不进行无监督训练，先学好 backbone)
        current_alpha_un = cfg.alpha_unlabeled
        if mode == "SEMI_SUPERVISED" and epoch <= 5: 
            current_alpha_un = 0.0
        
        # Iterator Setup
        if mode == "SEMI_SUPERVISED": 
            iterator = zip(itertools.cycle(labeled_loader), unlabeled_loader)
        elif mode == "SUPERVISED": 
            iterator = ((batch, None) for batch in labeled_loader)
        elif mode == "UNSUPERVISED": 
            iterator = ((None, batch) for batch in unlabeled_loader)

        for batch_lab, batch_un in iterator:
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=cfg.device)
            resp = None 

            # --- A. Supervised Part ---
            if batch_lab is not None:
                x_lab, y_lab = batch_lab
                x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device).long()
                loss_lab, _, _, _, _, _ = model(x_lab, cfg, y=y_lab) # y is provided
                total_loss += loss_lab

            # --- B. Unsupervised Part (Gumbel-Softmax) ---
            if batch_un is not None and current_alpha_un > 0:
                # [修复] 正确解包，获取 y_un_true 用于 Debug
                x_un, y_un_true = batch_un 
                x_un = x_un.to(cfg.device)
                y_un_true = y_un_true.to(cfg.device)
                
                # 调用 forward，传入 gumbel_temp
                loss_un, _, _, _, resp, _ = model(x_un, cfg, y=None, 
                                                  current_scale=dynamic_scale, 
                                                  current_lambda=dynamic_lambda,
                                                  gumbel_temp=gumbel_temp) # <--- 传入温度
                
                total_loss += current_alpha_un * loss_un
                
                # === 深度监控 (Deep Monitoring) ===
                # 每 50 个 batch 打印一次，观察是否发生 Mode Collapse
                if n_batches % 50 == 0:
                    with torch.no_grad():
                        # resp 是 Softmax 后的概率，用于观察模型"想"选哪个
                        pseudo_labels = resp.argmax(dim=1)
                        
                        # 计算伪标签准确率 (仅供参考，不参与梯度)
                        acc_unsup = (pseudo_labels == y_un_true).float().mean().item()
                        
                        # 计算平均置信度
                        conf = resp.max(dim=1)[0].mean().item()
                        
                        # 统计类别分布 (最重要！检查是否全部分到了某一类)
                        class_counts = torch.bincount(pseudo_labels, minlength=cfg.num_classes).cpu().numpy()
                        
                        print(f"   [Debug] Unsup Acc: {acc_unsup:.2f} | Conf: {conf:.2f} | Dist: {class_counts}")

            # Optimization
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_accum += total_loss.item()
            n_batches += 1

        # ==========================================
        # Validation & Checkpointing
        # ==========================================
        # 这里的验证比较耗时，但很有必要
        raw_acc, _, val_nmi = evaluate_model(model, val_loader, cfg)
        target_metric = raw_acc if mode == "SUPERVISED" else val_nmi
        
        # 准备 Checkpoint 数据
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_nmi': best_val_nmi,
            'config': cfg.__dict__
        }
        
        # 1. 保存 "Latest" (用于 Resume)
        last_ckpt_path = os.path.join(cfg.output_dir, "checkpoint_last.pt")
        torch.save(checkpoint_dict, last_ckpt_path)

        # 2. 保存 "Periodic" (每10轮备份，防止跑崩了没后悔药)
        if epoch % 10 == 0:
            periodic_path = os.path.join(cfg.output_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            torch.save(checkpoint_dict, periodic_path)
            print(f"   💾 [Backup] Periodic checkpoint saved: {periodic_path}")

        # 3. 保存 "Best" (最佳指标)
        if target_metric > best_val_nmi:
            best_val_nmi = target_metric
            if is_final_training:
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
                torch.save(checkpoint_dict, os.path.join(cfg.output_dir, "checkpoint_best.pt"))
                print(f"   ★ New Best Model Saved! (NMI: {best_val_nmi:.4f})")

        # Log
        log_tag = "FINAL" if is_final_training else f"TRIAL-{trial_id}"
        print(f"[{log_tag}] Ep {epoch} | Loss: {loss_accum/n_batches:.4f} | Acc: {raw_acc:.4f} | NMI: {val_nmi:.4f}")

        # 采样看图
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