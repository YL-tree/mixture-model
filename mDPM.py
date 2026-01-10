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
        
    def forward(self, x_0, cfg, y=None):
        batch_size = x_0.size(0)
        
        # 1. 采样连续潜在变量 (Sample x_t)
        t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.dpm_process.q_sample(x_0, t, noise)
        
        # -------------------
        # 监督模式 (Labeled Data)
        # -------------------
        if y is not None:
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            
            # 简单的 MSE 损失
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean') 
            
            return dpm_loss, -dpm_loss.item(), dpm_loss.item(), 0.0, None, None
            
        # -------------------
        # 无监督模式 (Unlabeled Data) - PVEM
        # -------------------
        else:
            log_pi = torch.log(self.registered_pi + 1e-8).unsqueeze(0).to(x_0.device)
            log_lik_proxy = []
            
            # E-Step: 估计 p(x|z)
            for k in range(cfg.num_classes):
                y_onehot_k = F.one_hot(torch.full((batch_size,), k, device=x_0.device),
                                       num_classes=cfg.num_classes).float()
                
                # 使用负 MSE 作为 Log-Likelihood 的代理
                pred_noise_k = self.cond_denoiser(x_t, t, y_onehot_k)
                dpm_loss_k = F.mse_loss(pred_noise_k, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                
                log_lik_proxy.append((-dpm_loss_k).unsqueeze(1))
                
            log_lik_proxy = torch.cat(log_lik_proxy, dim=1)
            logits = log_pi + log_lik_proxy
            
            # Gumbel Softmax (松弛后验)
            resp = gumbel_softmax_sample(logits, cfg.current_gumbel_temp)
            
            # M-Step: 加权 DPM Loss
            weighted_dpm_loss = 0.0
            for k in range(cfg.num_classes):
                y_onehot_k = F.one_hot(torch.full((batch_size,), k, device=x_0.device),
                                       num_classes=cfg.num_classes).float()
                
                pred_noise_k = self.cond_denoiser(x_t, t, y_onehot_k)
                dpm_loss_k = F.mse_loss(pred_noise_k, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                
                weighted_dpm_loss += (resp[:, k] * dpm_loss_k).mean()
            
            # 熵最小化 (Entropy Minimization) - 鼓励高置信度
            # H(p) = - sum p log p (这是一个正数)
            entropy = -(resp * torch.log(resp + 1e-8)).sum(dim=1).mean()
            
            # 修正: 使用 + 号来惩罚高熵 (Entropy Minimization)
            total_loss = weighted_dpm_loss + cfg.lambda_entropy * entropy
            
            return total_loss, -total_loss.item(), weighted_dpm_loss.item(), 0.0, resp.detach(), None

# -----------------------
# Evaluation Utils
# -----------------------
def evaluate_model(model, loader, cfg):
    """计算后验 Accuracy 和 NMI"""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("Scipy not found, skipping detailed evaluation.")
        return 0.0, {}, 0.0

    model.eval()
    preds, ys_true = [], []
    
    # 评估时使用固定的中间时间步
    t_eval_val = cfg.timesteps // 2 
    
    with torch.no_grad():
        for x_0, y_true in loader:
            x_0 = x_0.to(cfg.device)
            batch_size = x_0.size(0)
            
            current_noise = torch.randn_like(x_0)
            current_t = torch.full((batch_size,), t_eval_val, device=cfg.device, dtype=torch.long)
            x_t = model.dpm_process.q_sample(x_0, current_t, current_noise)
            
            log_pi = torch.log(model.registered_pi + 1e-8).unsqueeze(0).to(x_0.device)
            dpm_loss_proxies = []

            for k in range(cfg.num_classes):
                y_onehot_k = F.one_hot(torch.full((batch_size,), k, device=x_0.device),
                                       num_classes=cfg.num_classes).float()
                
                pred_noise_k = model.cond_denoiser(x_t, current_t, y_onehot_k)
                dpm_loss_k = F.mse_loss(pred_noise_k, current_noise, reduction='none').view(batch_size, -1).mean(dim=1)
                dpm_loss_proxies.append((-dpm_loss_k).unsqueeze(1))
            
            logits = torch.cat(dpm_loss_proxies, dim=1) + log_pi 
            pred_cluster = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred_cluster)
            ys_true.append(y_true.numpy())

    preds = np.concatenate(preds)
    ys_true = np.concatenate(ys_true)
    
    # 计算 NMI
    nmi = NMI(ys_true, preds)
    
    # 计算 Accuracy (匈牙利算法对齐)
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
        x_t = torch.randn(shape, device=device) # x_T
        
        # 构造条件
        y_cond = torch.arange(num_classes).to(device).repeat_interleave(n_per_class).long()
        
        # 修正: 循环范围从 T-1 到 0
        for i in reversed(range(0, T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            alpha_t = dpm_process._extract_t(dpm_process.alphas, t, shape)
            one_minus_alpha_t_bar = dpm_process._extract_t(dpm_process.sqrt_one_minus_alphas_cumprod, t, shape)
            
            # 预测噪声
            pred_noise = denoiser(x_t, t, y_cond)
            
            # 计算均值 mu_{t-1}
            mu_t_1 = (x_t - (1 - alpha_t) / one_minus_alpha_t_bar * pred_noise) / alpha_t.sqrt()
            
            # 计算标准差
            sigma_t_1 = dpm_process._extract_t(dpm_process.posterior_variance, t, shape).sqrt()
            
            # 修正: 只有当 i > 0 时才添加噪声
            if i > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)
                
            x_t = mu_t_1 + sigma_t_1 * noise

        save_image(x_t.clamp(0, 1), out_path, nrow=n_per_class, normalize=True)
    print(f"💾 Saved DPM samples to {out_path}")

# -----------------------
# Training Engine
# -----------------------
def run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                         is_final_training=False, trial_id=None):
    total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs
    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    metrics = {"Neg_ELBO": [], "DPM_Loss": [], "NMI": [], "PosteriorAcc": []}
    best_val_nmi = -np.inf

    for epoch in range(1, total_epochs + 1):
        model.train()
        
        # Tau 退火
        if epoch > total_epochs * 0.5:
            cfg.current_gumbel_temp = max(cfg.min_gumbel_temp, cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)

        loss_accum = 0.0
        
        for (x_lab, y_lab), (x_un, _) in zip(labeled_loader, unlabeled_loader):
            x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device).long()
            x_un = x_un.to(cfg.device)
            
            # Loss Calculation
            loss_lab, _, _, _, _, _ = model(x_lab, cfg, y_lab)
            loss_un, _, _, _, resp, _ = model(x_un, cfg, None)
            
            loss = loss_lab + cfg.alpha_unlabeled * loss_un

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss_accum += loss.item()

            # EMA Update for Prior
            with torch.no_grad():
                if resp is not None:
                    model.registered_pi.copy_(0.99 * model.registered_pi + 0.01 * resp.mean(0).detach())

        # Validation
        posterior_acc, cluster2label, val_nmi = evaluate_model(model, val_loader, cfg)
        
        metrics["NMI"].append(val_nmi)
        metrics["PosteriorAcc"].append(posterior_acc)

        if val_nmi > best_val_nmi:
            best_val_nmi = val_nmi

        mode = "FINAL" if is_final_training else f"TRIAL-{trial_id}"
        print(f"[{mode}] Epoch {epoch} | Loss: {loss_accum/len(labeled_loader):.4f} | "
              f"NMI: {val_nmi:.4f} | Acc: {posterior_acc:.4f} | τ: {cfg.current_gumbel_temp:.3f}")

        if is_final_training and (epoch % 10 == 0 or epoch == total_epochs):
            sample_and_save_dpm(model.cond_denoiser, model.dpm_process, cfg.num_classes,
                                os.path.join(sample_dir, f"epoch_{epoch:03d}.png"), cfg.device)
    
    if is_final_training:
        with open(os.path.join(cfg.output_dir, "posterior_mapping.json"), "w") as f:
            json.dump(cluster2label, f, indent=2)

    return best_val_nmi, metrics

def objective(trial):
    cfg = Config()
    cfg.output_dir = "./mDPM_optuna_temp"
    
    # Hyperparameters to tune
    cfg.unet_base_channels = trial.suggest_categorical("base_channels", [32, 64])
    cfg.lambda_entropy = trial.suggest_float("lambda_entropy", 0.1, 5.0)
    cfg.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    
    model = None
    optimizer = None
    
    try:
        model = mDPM_SemiSup(cfg).to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)

        best_nmi, _ = run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                                           is_final_training=False, trial_id=trial.number)
        return -best_nmi # Optuna minimizes
        
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.TrialPruned()
    finally:
        # 显式内存清理
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

def main():
    # 1. Optuna Search
    print("--- Starting Optuna ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5) 
    
    print("Best params:", study.best_params)
    
    # 2. Final Training
    print("--- Starting Final Training ---")
    cfg = Config()
    for k, v in study.best_params.items():
        setattr(cfg, k, v)
    
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)
    
    run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg, is_final_training=True)
    
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "final_model.pt"))
    print("Done.")

if __name__ == "__main__":
    main()