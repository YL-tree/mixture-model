# semi_mDPM_final_fix.py
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
from common_dpm import * # 确保您的目录下有 common_dpm.py

# -----------------------
# 0. 基础设置 (Reproducibility)
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"🔒 Seed locked to {seed}")

def plot_dashboard(history, outpath):
    """绘制训练监控面板：Loss, Acc, Scale, PassRate"""
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Training Dashboard (Ep {len(history["loss"])})', fontsize=16)
    
    # 1. Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], 'b-', label='Total Loss')
    ax.set_title('Loss')
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["acc"], 'r-', label='Val Acc')
    ax.set_title('Validation Accuracy')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    
    # 3. Scale
    ax = axes[1, 0]
    ax.plot(epochs, history["scale"], 'g-', label='Scale')
    ax.set_title('Dynamic Scale')
    ax.grid(True, alpha=0.3)
    
    # 4. Pass Rate
    ax = axes[1, 1]
    ax.plot(epochs, history["pass_rate"], 'm-', label='Pass Rate (%)')
    ax.set_title('Unlabeled Pass Rate')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -----------------------
# 1. 模型定义
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

        # Path A: 监督路径 (老师教)
        if y is not None:
            t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.dpm_process.q_sample(x_0, t, noise)
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean') 
            return dpm_loss, 1.0 # mask_rate=1 for labeled

        # Path B: 无监督路径 (自学)
        else:
            logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=current_scale)
            resp = F.softmax(logits, dim=1) 
            
            if use_hard_label:
                # [Phase 2: Hard Mode] 
                # 严厉模式：只有置信度 > threshold (0.95) 才能通过
                max_probs, pseudo_labels = resp.max(dim=1)
                mask = (max_probs >= threshold).float()
                y_target = F.one_hot(pseudo_labels, num_classes=cfg.num_classes).float()
            else:
                # [Phase 1: Soft Mode]
                # 探索模式：Multinomial 采样 + 全通过 (Mask=1)
                # 即使猜错了也没关系，Scale很小，目的是学特征
                pseudo_labels = torch.multinomial(resp, 1).squeeze(1)
                y_target = F.one_hot(pseudo_labels, num_classes=cfg.num_classes).float()
                mask = torch.ones(batch_size, device=x_0.device)

            t_train = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
            pred_noise = self.cond_denoiser(x_t_train, t_train, y_target)
            
            loss_per_sample = F.mse_loss(pred_noise, noise, reduction='none').view(batch_size, -1).mean(dim=1)
            
            # 只有 mask=1 的样本贡献 Loss
            dpm_loss = (loss_per_sample * mask).sum() / (mask.sum() + 1e-8)
            
            return dpm_loss, mask.mean().item()

# -----------------------
# 2. 评估与可视化
# -----------------------
def evaluate_model(model, loader, cfg):
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return 0.0

    model.eval()
    preds, ys_true = [], []
    eval_timesteps = [500] 
    
    with torch.no_grad():
        for x_0, y_true in loader:
            x_0 = x_0.to(cfg.device)
            batch_size = x_0.size(0)
            cumulative_mse = torch.zeros(batch_size, cfg.num_classes, device=cfg.device)
            for t_val in eval_timesteps:
                mse_t_sum = torch.zeros(batch_size, cfg.num_classes, device=cfg.device)
                for _ in range(3): # n_repeats
                    noise = torch.randn_like(x_0)
                    current_t = torch.full((batch_size,), t_val, device=cfg.device, dtype=torch.long)
                    x_t = model.dpm_process.q_sample(x_0, current_t, noise)
                    for k in range(cfg.num_classes):
                        y_vec = F.one_hot(torch.full((batch_size,), k, device=x_0.device), cfg.num_classes).float()
                        pred = model.cond_denoiser(x_t, current_t, y_vec)
                        mse_t_sum[:, k] += F.mse_loss(pred, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                cumulative_mse += (mse_t_sum / 3)
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
            
            alpha_t = model.dpm_process._extract_t(model.dpm_process.alphas, t, shape)
            one_minus_bar = model.dpm_process._extract_t(model.dpm_process.sqrt_one_minus_alphas_cumprod, t, shape)
            mu = (x_t - (1-alpha_t)/one_minus_bar * pred) / alpha_t.sqrt()
            sigma = model.dpm_process._extract_t(model.dpm_process.posterior_variance, t, shape).sqrt()
            
            noise = torch.randn_like(x_t) if i > 0 else 0
            x_t = mu + sigma * noise
            
        save_image(x_t.clamp(-1, 1), os.path.join(save_dir, f"ep_{epoch:03d}.png"), nrow=10, normalize=True)

# -----------------------
# 3. 训练引擎 (修正版)
# -----------------------
def run_training(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg):
    history = {"loss": [], "acc": [], "scale": [], "pass_rate": []}
    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)
    
    # === [核心参数配置] ===
    warmup_epochs = 20      # Phase 1 探索期
    final_scale = 150.0     # 最终 Scale
    
    # [关键] 门槛锁死 0.95
    fixed_threshold = 0.95  
    
    # [关键] Alpha 恒定 0.1
    alpha_unlabeled = 0.1   

    print(f"🚀 Strategy: Soft-to-Hard | Warmup={warmup_epochs} | Alpha={alpha_unlabeled} | Thres={fixed_threshold}")

    for epoch in range(1, cfg.final_epochs + 1):
        
        # === Scheduler ===
        if epoch <= warmup_epochs:
            # Phase 1: EXPLORE (Soft)
            status = "EXPLORE"
            use_hard = False
            # Scale: 5 -> 20 (温和)
            p = epoch / warmup_epochs
            curr_scale = 5.0 + (20.0 - 5.0) * p
            # 软模式不需要门槛
            curr_thres = 0.0
            
        else:
            # Phase 2: REFINE (Hard)
            status = "REFINE"
            use_hard = True
            # Scale: 50 -> 150 (跳变起步，为了过 0.95 门槛)
            p = (epoch - warmup_epochs) / (cfg.final_epochs - warmup_epochs)
            curr_scale = 50.0 + (final_scale - 50.0) * p
            # 硬模式门槛直接锁死 0.95
            curr_thres = fixed_threshold

        model.train()
        loss_acc = 0
        mask_acc = 0
        batches = 0
        
        # 混合数据流
        iterator = zip(itertools.cycle(labeled_loader), unlabeled_loader)
        
        for batch_lab, batch_un in iterator:
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=cfg.device)
            
            # 1. Labeled Data (Teacher)
            x_lab, y_lab = batch_lab
            x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device).long()
            l_sup, _ = model(x_lab, cfg, y=y_lab)
            total_loss += l_sup
            
            # 2. Unlabeled Data (Student)
            x_un, _ = batch_un
            x_un = x_un.to(cfg.device)
            
            l_unsup, mask_rate = model(x_un, cfg, y=None, 
                                       current_scale=curr_scale, 
                                       threshold=curr_thres,
                                       use_hard_label=use_hard)
            
            # [关键] 乘以 0.1 的保护系数
            total_loss += alpha_unlabeled * l_unsup
            
            total_loss.backward()
            optimizer.step()
            
            loss_acc += total_loss.item()
            mask_acc += mask_rate
            batches += 1
            
        # Logging
        val_acc = evaluate_model(model, val_loader, cfg)
        avg_loss = loss_acc/batches if batches > 0 else 0
        avg_pass = (mask_acc/batches)*100 if batches > 0 else 0
        
        history["loss"].append(avg_loss)
        history["acc"].append(val_acc)
        history["scale"].append(curr_scale)
        history["pass_rate"].append(avg_pass)
        
        print(f"Ep {epoch} [{status}] Scale={curr_scale:.1f} Thres={curr_thres:.2f} | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f} | Pass: {avg_pass:.1f}%")
        
        plot_dashboard(history, os.path.join(cfg.output_dir, "training_dashboard.png"))
        
        if epoch % 5 == 0:
            sample_visual(model, cfg, epoch, sample_dir)
            # 保存 Checkpoint
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "last_model.pt"))

def main():
    set_seed(42)
    cfg = Config()
    
    # [关键] 必须设置每类标签数，否则 labeled_loader 为空
    cfg.labeled_per_class = 10 
    
    cfg.final_epochs = 60
    cfg.posterior_sample_steps = 5
    
    print("="*40)
    print(">>> FINAL FIXED SEMI-SUPERVISED TRAINING <<<")
    print(f"Config: Labeled={cfg.labeled_per_class}, Ep={cfg.final_epochs}")
    print("="*40)
    
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
    
    lab_loader, unlab_loader, val_loader = get_semi_loaders(cfg)
    
    run_training(model, optimizer, lab_loader, unlab_loader, val_loader, cfg)

if __name__ == "__main__":
    main()