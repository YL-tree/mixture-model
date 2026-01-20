# mDPM.py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import os
import gc
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision.utils import save_image
import itertools
from common_dpm import *

# -----------------------
# Model Definition
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
        # 类别先验：初始化为均匀分布
        self.register_buffer('registered_pi', torch.ones(cfg.num_classes) / cfg.num_classes)
        
    def estimate_posterior_logits(self, x_0, cfg, scale_factor=1.0):
        batch_size = x_0.size(0)
        num_classes = cfg.num_classes
        M = cfg.posterior_sample_steps
        
        # Log Likelihood 累加器
        accum_neg_mse = torch.zeros(batch_size, num_classes, device=x_0.device)
        
        with torch.no_grad():
            for _ in range(M):
                # 采样时间步 (覆盖中间区域)
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
        
        # Logits = Prior + Scale * Likelihood
        final_logits = log_pi + (avg_neg_mse * scale_factor)
        return final_logits

    def forward(self, x_0, cfg, y=None, current_scale=100.0, current_lambda=0.0, threshold=0.95):
        """
        前向传播：根据 y 是否存在，自动选择 Supervised 或 FixMatch 路径
        """
        batch_size = x_0.size(0)

        # -------------------
        # Path A: 监督模式 (有标签) - 老师教
        # -------------------
        if y is not None:
            t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.dpm_process.q_sample(x_0, t, noise)
            
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean') 
            
            # 返回格式: (total_loss, neg_elbo, dpm_loss, mask_rate, resp, None)
            return dpm_loss, -dpm_loss.item(), dpm_loss.item(), 1.0, None, None
            
        # Path B: 无监督
        else:
            # 1. E-Step
            logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=current_scale)
            resp = F.softmax(logits, dim=1) 
            
            # --- 分支 1: 硬标签模式 (FixMatch) ---
            if use_hard_label:
                max_probs, pseudo_labels = resp.max(dim=1)
                mask = (max_probs >= threshold).float()
                y_target = F.one_hot(pseudo_labels, num_classes=cfg.num_classes).float()
                
                # M-Step
                t_train = torch.randint(0, cfg.timesteps, (x_0.size(0),), device=x_0.device).long()
                noise = torch.randn_like(x_0)
                x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
                pred_noise = self.cond_denoiser(x_t_train, t_train, y_target)
                
                loss_per_sample = F.mse_loss(pred_noise, noise, reduction='none').view(x_0.size(0), -1).mean(dim=1)
                dpm_loss = (loss_per_sample * mask).sum() / (mask.sum() + 1e-8)
                mask_rate = mask.mean().item()

            # --- 分支 2: 软标签模式 (Soft-EM) [新增] ---
            # 适用于无监督初期，防止坍塌
            else:
                y_target = resp # 直接用概率分布作为 Target
                mask_rate = 1.0 # 所有样本都参与，不搞阈值
                
                # M-Step (Gumbel-Softmax 或 直接 Soft)
                # 为了稳定，这里建议直接把 Softmax 概率喂进去，
                # 但前提是你的 ConditionalUnet 支持 float 类型的 label (你的代码支持)
                t_train = torch.randint(0, cfg.timesteps, (x_0.size(0),), device=x_0.device).long()
                noise = torch.randn_like(x_0)
                x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
                pred_noise = self.cond_denoiser(x_t_train, t_train, y_target)
                
                # 损失：MSE * 权重(resp) ? 不，直接算 MSE 即可
                # 因为 y_target 已经是混合 embedding 了，网络会自动学去噪
                dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean')

            # 辅助损失
            entropy = -(resp * torch.log(resp + 1e-8)).sum(dim=1).mean()
            total_loss = dpm_loss + current_lambda * entropy
            
            return total_loss, -total_loss.item(), dpm_loss.item(), mask_rate, resp.detach(), None
# -----------------------
# Evaluation Utils
# -----------------------
def evaluate_model(model, loader, cfg):
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return 0.0, {}, 0.0

    model.eval()
    preds, ys_true = [], []
    
    # 简化评估：只用中间时刻
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
    
    nmi = 0.0 # NMI(ys_true, preds) # 暂时省略 NMI 以加速
    
    # 计算 Acc
    n_classes = cfg.num_classes
    cost_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cost_matrix[i, j] = -np.sum((ys_true == i) & (preds == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster2label = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned_preds = np.array([cluster2label.get(p, 0) for p in preds])
    acc = np.mean(aligned_preds == ys_true)
    
    return acc, cluster2label, nmi

def sample_and_save_dpm(denoiser, dpm_process, num_classes, out_path, device, n_per_class=10):
    """生成图像网格：每行一个类别"""
    T = dpm_process.timesteps
    denoiser.eval()
    image_c = dpm_process.image_channels

    with torch.no_grad():
        # 形状: (100, 1, 28, 28)
        shape = (n_per_class * num_classes, image_c, 28, 28)
        x_t = torch.randn(shape, device=device)
        # 生成条件: [0,0,..,0, 1,1,..,1, ...]
        y_cond = torch.arange(num_classes).to(device).repeat_interleave(n_per_class).long()
        y_cond_vec = F.one_hot(y_cond, num_classes).float() # Hard Label Cond
        
        for i in reversed(range(0, T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # 标准 DDPM 采样步
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
    print(f"   [Visual] Samples saved to {out_path}")

# -----------------------
# Training Engine
# -----------------------
def run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                         is_final_training=False, trial_id=None, resume_path=None):
    
    total_epochs = cfg.final_epochs
    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    start_epoch = 1
    best_val_acc = 0.0

    # Resume Logic
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_nmi', 0.0)
        print(f"🔄 Resumed at Ep {start_epoch}")

    # ==========================================
    # [修复] 完善的模式检测逻辑
    # ==========================================
    mode = "UNKNOWN"
    if labeled_loader is not None and unlabeled_loader is not None: 
        mode = "SEMI_SUPERVISED"
    elif labeled_loader is not None: 
        mode = "SUPERVISED"
        cfg.alpha_unlabeled = 0.0 # 强制关闭无监督权重
    elif unlabeled_loader is not None: 
        # 这种情况就是你现在需要的
        mode = "UNSUPERVISED"
        cfg.alpha_unlabeled = 1.0 # 强制开启无监督权重
    
    print(f"🚀 Training Mode: {mode}")

    for epoch in range(start_epoch, total_epochs + 1):
        progress = (epoch - 1) / total_epochs
        
    
        if mode == "UNSUPERVISED":
            # 1. 阶段划分
            # 前 10 Epoch: 探索期 (Soft, Low Scale)
            # 后期: 硬化期 (Hard, High Scale)
            if epoch <= 10:
                use_hard = False
                # Scale 从 1.0 慢慢涨到 10.0 (非常低，让 MSE 自然竞争)
                dynamic_scale = 1.0 + (10.0 - 1.0) * (epoch / 10)
                dynamic_threshold = 0.0 # 不生效
            else:
                use_hard = True
                # Scale 开始飙升: 10.0 -> 100.0
                p2 = (epoch - 10) / (total_epochs - 10)
                dynamic_scale = 10.0 + (100.0 - 10.0) * p2
                # 阈值慢慢加上来
                dynamic_threshold = 0.5 + (0.9 - 0.5) * p2
        
        else: 
            # 半监督/监督模式维持原判
            use_hard = True
            dynamic_scale = 300.0 + (600.0 - 300.0) * progress
            dynamic_threshold = 0.70 + (0.95 - 0.70) * progress

        if epoch % 5 == 0 or epoch == 1:
            status = "HARD" if use_hard else "SOFT"
            print(f"🔥 [Scheduler] Ep {epoch} ({status}): Scale={dynamic_scale:.1f}, Thres={dynamic_threshold:.2f}")

        
        model.train()
        loss_accum = 0.0
        mask_rate_accum = 0.0
        n_batches = 0
        
        # Warm-up (仅在半监督时生效，无监督不跳过)
        current_alpha = cfg.alpha_unlabeled
        if mode == "SEMI_SUPERVISED" and epoch <= 5: 
            current_alpha = 0.0
        
        
        # ==========================================
        # [修复] Iterator 分发逻辑
        # ==========================================
        if mode == "SEMI_SUPERVISED": 
            iterator = zip(itertools.cycle(labeled_loader), unlabeled_loader)
        elif mode == "SUPERVISED":
            iterator = ((batch, None) for batch in labeled_loader)
        elif mode == "UNSUPERVISED":
            # 纯无监督：第一个位置(labeled)传 None，第二个位置(unlabeled)传数据
            iterator = ((None, batch) for batch in unlabeled_loader)

        for batch_lab, batch_un in iterator:
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=cfg.device)

            # A. 有标签部分 (UNSUPERVISED 模式下 batch_lab 为 None，会自动跳过)
            if batch_lab is not None:
                x, y = batch_lab
                x, y = x.to(cfg.device), y.to(cfg.device).long()
                l_sup, _, _, _, _, _ = model(x, cfg, y=y)
                total_loss += l_sup

            # B. 无标签部分 (SUPERVISED 模式下 batch_un 为 None，会自动跳过)
            if batch_un is not None and current_alpha > 0:
                x_un, _ = batch_un
                x_un = x_un.to(cfg.device)
                
                # [修改] 传入 use_hard_label
                l_unsup, _, _, mask_rate, _, _ = model(x_un, cfg, y=None, 
                                                       current_scale=dynamic_scale,
                                                       current_lambda=0.01,
                                                       threshold=dynamic_threshold,
                                                       use_hard_label=use_hard) # <--- 传入
                # ...
                total_loss += current_alpha * l_unsup
                mask_rate_accum += mask_rate

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_accum += total_loss.item()
            n_batches += 1

        # Validation
        val_acc, _, _ = evaluate_model(model, val_loader, cfg)
        
        # Checkpointing
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_nmi': max(val_acc, best_val_acc)
        }
        torch.save(ckpt, os.path.join(cfg.output_dir, "checkpoint_last.pt"))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, os.path.join(cfg.output_dir, "best_model.pt"))
            print(f"   ★ New Best! Acc: {best_val_acc:.4f}")

        avg_mask = mask_rate_accum / n_batches if n_batches > 0 else 0
        print(f"Ep {epoch} | Loss: {loss_accum/n_batches:.4f} | Val Acc: {val_acc:.4f} | Pass: {avg_mask*100:.1f}%")

        if epoch % 1 == 0:
            sample_and_save_dpm(model.cond_denoiser, model.dpm_process, cfg.num_classes,
                                os.path.join(sample_dir, f"epoch_{epoch:03d}.png"), cfg.device)
    
    return best_val_acc, {}

def main():
    cfg = Config()
    # 强制覆盖配置以确保 FixMatch 生效
    cfg.alpha_unlabeled = 0.1  # 权重必须小
    cfg.posterior_sample_steps = 5 # 采样必须准
    
    print("="*30)
    print(f"--- FixMatch Training (Threshold=0.95) ---")
    print(f"Config: LR={cfg.lr}, Alpha={cfg.alpha_unlabeled}")
    print("="*30)

    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)
    
    # 默认尝试 Resume
    resume_path = os.path.join(cfg.output_dir, "checkpoint_last.pt")
    
    run_training_session(
        model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg, 
        is_final_training=True,
        resume_path=resume_path
    )

if __name__ == "__main__":
    main()