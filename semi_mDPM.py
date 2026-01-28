# mDPM_SemiSup_SoftHard.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from torchvision.utils import save_image
import itertools
from common_dpm import *

# -----------------------
# 0. 基础设置
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"🔒 Seed locked to {seed}")

def plot_curves(history, outpath):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history["loss"]) + 1)
    
    # Loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, history["loss"], color='tab:blue', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Acc
    ax2 = ax1.twinx()
    ax2.set_ylabel('Acc', color='tab:red')
    ax2.plot(epochs, history["acc"], color='tab:red', linestyle='--', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('Training Dynamics (Soft-to-Hard)')
    fig.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -----------------------
# 1. 模型定义 (必须支持 use_hard_label 开关)
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

    def forward(self, x_0, cfg, y=None, current_scale=1.0, threshold=0.0, use_hard_label=False):
        batch_size = x_0.size(0)

        # Path A: 监督 (始终开启)
        if y is not None:
            t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.dpm_process.q_sample(x_0, t, noise)
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean') 
            return dpm_loss, 1.0 # mask_rate=1 for labeled

        # Path B: 无监督 (软/硬切换)
        else:
            logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=current_scale)
            resp = F.softmax(logits, dim=1) 
            
            if use_hard_label:
                # [阶段二：REFINE] 硬标签 + 阈值过滤
                max_probs, pseudo_labels = resp.max(dim=1)
                mask = (max_probs >= threshold).float()
                y_target = F.one_hot(pseudo_labels, num_classes=cfg.num_classes).float()
            else:
                # [阶段一：EXPLORE] 软标签 (采样) + 全通过
                # Multinomial Sampling: 依概率采样，保持多样性
                pseudo_labels = torch.multinomial(resp, 1).squeeze(1)
                y_target = F.one_hot(pseudo_labels, num_classes=cfg.num_classes).float()
                mask = torch.ones(batch_size, device=x_0.device)

            t_train = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
            pred_noise = self.cond_denoiser(x_t_train, t_train, y_target)
            
            loss_per_sample = F.mse_loss(pred_noise, noise, reduction='none').view(batch_size, -1).mean(dim=1)
            # 应用 Mask
            dpm_loss = (loss_per_sample * mask).sum() / (mask.sum() + 1e-8)
            
            return dpm_loss, mask.mean().item()

# -----------------------
# 2. 评估函数
# -----------------------
def evaluate_model(model, loader, cfg):
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return 0.0

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
                        mse_t_sum[:, k] += F.mse_loss(pred, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                cumulative_mse += (mse_t_sum / n_repeats)
            preds.append(torch.argmin(cumulative_mse, dim=1).cpu().numpy())
            ys_true.append(y_true.numpy())

    preds = np.concatenate(preds)
    ys_true = np.concatenate(ys_true)
    
    cost_matrix = np.zeros((cfg.num_classes, cfg.num_classes))
    for i in range(cfg.num_classes):
        for j in range(cfg.num_classes):
            cost_matrix[i, j] = -np.sum((ys_true == i) & (preds == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster2label = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned_preds = np.array([cluster2label.get(p, 0) for p in preds])
    
    return np.mean(aligned_preds == ys_true)

def sample_visual(model, cfg, epoch, save_dir):
    model.eval()
    with torch.no_grad():
        shape = (100, cfg.image_channels, 28, 28)
        x_t = torch.randn(shape, device=cfg.device)
        y_cond = torch.arange(10).to(cfg.device).repeat_interleave(10).long()
        y_vec = F.one_hot(y_cond, 10).float()
        
        for i in reversed(range(0, cfg.timesteps)):
            t = torch.full((100,), i, device=cfg.device).long()
            pred = model.cond_denoiser(x_t, t, y_vec)
            # ... DPM Sampling Logic (简化) ...
            # 为了代码简洁，这里假设 dpm_process 在 model 里
            # 实际运行请确保 dpm_process 的采样逻辑正确引用
            alpha_t = model.dpm_process._extract_t(model.dpm_process.alphas, t, shape)
            one_minus_bar = model.dpm_process._extract_t(model.dpm_process.sqrt_one_minus_alphas_cumprod, t, shape)
            mu = (x_t - (1-alpha_t)/one_minus_bar * pred) / alpha_t.sqrt()
            sigma = model.dpm_process._extract_t(model.dpm_process.posterior_variance, t, shape).sqrt()
            noise = torch.randn_like(x_t) if i > 0 else 0
            x_t = mu + sigma * noise
            
        save_image(x_t.clamp(-1, 1), os.path.join(save_dir, f"ep_{epoch:03d}.png"), nrow=10, normalize=True)

# -----------------------
# 3. 训练循环 (Soft-to-Hard Scheduler)
# -----------------------
def run_training(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg):
    history = {"loss": [], "acc": []}
    sample_dir = os.path.join(cfg.output_dir, "soft_hard_samples")
    os.makedirs(sample_dir, exist_ok=True)
    
    # 关键参数
    warmup_epochs = 15     # 探索期时长
    target_scale = 150.0   # 最终自信度
    threshold_final = 0.95 # 最终门槛
    
    print(f"🚀 Soft-to-Hard Semi-Sup: Warmup={warmup_epochs}, TargetScale={target_scale}")

    for epoch in range(1, cfg.final_epochs + 1):
        
        # === 核心调度器 ===
        if epoch <= warmup_epochs:
            # Phase 1: EXPLORE (Soft Mode)
            # 允许无标签数据以“软”的方式参与，Scale很低，目的是对齐特征，而不是强行分类
            status = "EXPLORE (Soft)"
            use_hard = False
            
            # Scale: 5.0 -> 20.0
            p = epoch / warmup_epochs
            curr_scale = 5.0 + (20.0 - 5.0) * p
            
            # Threshold: 0 (软模式不需要门槛，全盘接收作为探索)
            curr_thres = 0.0
            
        else:
            # Phase 2: REFINE (Hard/FixMatch Mode)
            # 开启高压政策，强行拉开类别
            status = "REFINE (Hard)"
            use_hard = True
            
            # Scale: 20.0 -> 150.0
            p = (epoch - warmup_epochs) / (cfg.final_epochs - warmup_epochs)
            curr_scale = 20.0 + (target_scale - 20.0) * p
            
            # Threshold: 0.0 -> 0.95 (逐渐收紧)
            curr_thres = 0.0 + (threshold_final - 0.0) * p

        print(f"Ep {epoch} [{status}] Scale={curr_scale:.1f}, Thres={curr_thres:.2f}")

        model.train()
        loss_acc = 0
        mask_acc = 0
        batches = 0
        
        # 同时遍历有标签和无标签
        iterator = zip(itertools.cycle(labeled_loader), unlabeled_loader)
        
        for batch_lab, batch_un in iterator:
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=cfg.device)
            
            # 1. 有标签 Loss (永远存在!)
            x_lab, y_lab = batch_lab
            x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device).long()
            l_sup, _ = model(x_lab, cfg, y=y_lab)
            total_loss += l_sup
            
            # 2. 无标签 Loss (Soft -> Hard)
            x_un, _ = batch_un
            x_un = x_un.to(cfg.device)
            l_unsup, mask_rate = model(x_un, cfg, y=None, 
                                       current_scale=curr_scale, 
                                       threshold=curr_thres,
                                       use_hard_label=use_hard)
            
            # 这里的 alpha 可以设为 1.0，因为 Soft Mode 下 Scale 很小，梯度很温和
            # 不需要像之前那样设为 0.1
            total_loss += 1.0 * l_unsup
            
            total_loss.backward()
            optimizer.step()
            
            loss_acc += total_loss.item()
            mask_acc += mask_rate
            batches += 1
            
        # Eval
        val_acc = evaluate_model(model, val_loader, cfg)
        history["loss"].append(loss_acc/batches)
        history["acc"].append(val_acc)
        
        print(f"   Loss: {loss_acc/batches:.4f} | Val Acc: {val_acc:.4f} | Pass: {mask_acc/batches*100:.1f}%")
        plot_curves(history, os.path.join(cfg.output_dir, "training_curve.png"))
        
        if epoch % 5 == 0:
            sample_visual(model, cfg, epoch, sample_dir)

def main():
    set_seed(42)
    cfg = Config()
    cfg.labeled_per_class = 10  # 确保有标签数据开启
    cfg.final_epochs = 60
    
    print("="*40)
    print("Starting Semi-Supervised Training (Unsupervised Style Schedule)")
    print("Strategy: Ep1-15 Soft Explore -> Ep16+ Hard Refine")
    print("="*40)
    
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5) # 推荐用小一点的 LR
    
    lab_loader, unlab_loader, val_loader = get_semi_loaders(cfg)
    
    run_training(model, optimizer, lab_loader, unlab_loader, val_loader, cfg)

if __name__ == "__main__":
    main()