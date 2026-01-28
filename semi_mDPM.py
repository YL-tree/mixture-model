# semi_mDPM_resume.py
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
from common_dpm import * # Ensure common_dpm.py exists in your directory

# -----------------------
# 0. Basic Settings (Reproducibility)
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"🔒 Seed locked to {seed}")

def plot_dashboard(history, outpath):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Loss
    if len(history["loss"]) > 0:
        ax = axes[0, 0]
        ax.plot(epochs, history["loss"], 'b-', label='Total Loss')
        ax.set_title('Loss')
        ax.grid(True, alpha=0.3)
    
    # 2. Acc
    if len(history["acc"]) > 0:
        ax = axes[0, 1]
        ax.plot(epochs, history["acc"], 'r-', label='Val Acc')
        ax.set_title(f'Validation Acc (Best: {max(history["acc"]):.4f})')
        ax.grid(True, alpha=0.3)
    
    # 3. Scale
    if len(history["scale"]) > 0:
        ax = axes[1, 0]
        ax.plot(epochs, history["scale"], 'g-', label='Scale')
        ax.set_title('Dynamic Scale')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -----------------------
# 1. Model Definition (Soft-Only Logic)
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

    def forward(self, x_0, cfg, y=None, current_scale=1.0):
        batch_size = x_0.size(0)

        # Path A: Supervised Path
        if y is not None:
            t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.dpm_process.q_sample(x_0, t, noise)
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean') 
            return dpm_loss

        # Path B: Unsupervised Path (Full Soft Mode)
        else:
            # 1. Calc Distribution
            logits = self.estimate_posterior_logits(x_0, cfg, scale_factor=current_scale)
            resp = F.softmax(logits, dim=1) 
            
            # 2. Soft Sampling (Multinomial)
            pseudo_labels = torch.multinomial(resp, 1).squeeze(1)
            y_target = F.one_hot(pseudo_labels, num_classes=cfg.num_classes).float()

            # 3. Training
            t_train = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
            pred_noise = self.cond_denoiser(x_t_train, t_train, y_target)
            
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean')
            
            return dpm_loss

# -----------------------
# 2. Evaluation Utils
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
                for _ in range(3):
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

# -----------------------
# 3. Training Engine (With Resume)
# -----------------------
def run_training(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg, resume=False):
    history = {"loss": [], "acc": [], "scale": []}
    start_epoch = 1
    
    # === Resume Logic ===
    ckpt_path = os.path.join(cfg.output_dir, "last_model.pt")
    
    if resume and os.path.exists(ckpt_path):
        print(f"🔄 Found checkpoint at {ckpt_path}, loading...")
        checkpoint = torch.load(ckpt_path, map_location=cfg.device)
        
        # Load Model & Optimizer
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore Epoch & History
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)
        
        print(f"✅ Successfully resumed from Epoch {start_epoch - 1}")
        print(f"   Last Acc: {history['acc'][-1] if history['acc'] else 'N/A'}")
    else:
        if resume:
            print("⚠️ Resume requested but no checkpoint found. Starting from scratch.")
        else:
            print("🆕 Starting fresh training.")

    # === [Optimized Strategy: Full Soft Mode] ===
    # Start high to avoid "ambiguity trap"
    start_scale = 30.0
    end_scale = 100.0
    alpha_unlabeled = 0.1 

    print(f"🚀 Strategy: Full Soft Mode | Scale: {start_scale}->{end_scale} | Alpha={alpha_unlabeled}")
    print(f"   Labeled Weight: x5.0 (Important!)")

    for epoch in range(start_epoch, cfg.final_epochs + 1):
        
        # Linear Scale Increase
        p = (epoch - 1) / cfg.final_epochs
        curr_scale = start_scale + (end_scale - start_scale) * p
        
        status = "SOFT_TRAIN"

        model.train()
        loss_acc = 0
        batches = 0
        
        iterator = zip(itertools.cycle(labeled_loader), unlabeled_loader)
        
        for batch_lab, batch_un in iterator:
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=cfg.device)
            
            # 1. Teacher (Labeled) - Weight x5.0
            x_lab, y_lab = batch_lab
            x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device).long()
            l_sup = model(x_lab, cfg, y=y_lab)
            total_loss += 5.0 * l_sup  # Anchoring
            
            # 2. Student (Unlabeled Soft)
            x_un, _ = batch_un
            x_un = x_un.to(cfg.device)
            l_unsup = model(x_un, cfg, y=None, current_scale=curr_scale)
            
            total_loss += alpha_unlabeled * l_unsup
            
            total_loss.backward()
            optimizer.step()
            loss_acc += total_loss.item()
            batches += 1
            
        # Logging
        val_acc = evaluate_model(model, val_loader, cfg)
        avg_loss = loss_acc/batches if batches > 0 else 0
        
        history["loss"].append(avg_loss)
        history["acc"].append(val_acc)
        history["scale"].append(curr_scale)
        
        print(f"Ep {epoch} [{status}] Scale={curr_scale:.1f} | Loss: {avg_loss:.4f} | Acc: {val_acc:.4f}")
        
        plot_dashboard(history, os.path.join(cfg.output_dir, "training_dashboard.png"))
        
        # Save Checkpoint (Every Epoch)
        # Includes epoch, state_dict, and history for resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, os.path.join(cfg.output_dir, "last_model.pt"))

def main():
    set_seed(42)
    cfg = Config()
    cfg.labeled_per_class = 10 
    cfg.final_epochs = 60
    cfg.posterior_sample_steps = 5
    
    # ================================
    # [CONTROL SWITCH]
    # Set to True to load 'last_model.pt' and continue
    # Set to False to start a new training
    # ================================
    RESUME_TRAINING = False  
    
    print("="*40)
    print(">>> FULL SOFT-MODE SEMI-SUPERVISED TRAINING <<<")
    print(f"RESUME MODE: {RESUME_TRAINING}")
    print("="*40)
    
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-5)
    
    lab_loader, unlab_loader, val_loader = get_semi_loaders(cfg)
    
    run_training(model, optimizer, lab_loader, unlab_loader, val_loader, cfg, resume=RESUME_TRAINING)

if __name__ == "__main__":
    main()