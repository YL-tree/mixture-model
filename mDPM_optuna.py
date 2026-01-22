# mDPM.py
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
from optuna.trial import TrialState
import os
import gc
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision.utils import save_image
import itertools
from common_dpm import *

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
    
    # 如果是 Optuna 模式，epoch 数由 cfg.optuna_epochs 决定
    # 如果是最终训练，由 cfg.final_epochs 决定
    total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs
    
    # 默认超参数（手动调出的最佳实践）
    if hyperparams is None:
        hyperparams = {
            'target_scale': 150.0,
            'warmup_epochs': 15,
            'threshold_final': 0.0
        }

    target_scale = hyperparams.get('target_scale', 150.0)
    warmup_epochs = hyperparams.get('warmup_epochs', 15)
    threshold_final = hyperparams.get('threshold_final', 0.0)

    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    start_epoch = 1
    best_val_acc = 0.0

    metrics = {
        "DPM_Loss": [],      # 记录 Loss
        "PosteriorAcc": []   # 记录 Accuracy
    }
    
    # 模式检测
    mode = "UNSUPERVISED" # 强制无监督

    for epoch in range(start_epoch, total_epochs + 1):
        
        # ==========================================
        # [动态调度器] 由 Optuna 参数控制
        # ==========================================
        
        # Phase 1: 探索 (Scale 较小，Multinomial)
        if epoch <= warmup_epochs:
            use_hard = False
            # Scale: 5.0 -> 20.0
            p1 = epoch / warmup_epochs
            dynamic_scale = 5.0 + (20.0 - 5.0) * p1
            dynamic_threshold = 0.0 
            status = "EXPLORE"
        
        # Phase 2: 精炼 (Scale 变大，FixMatch)
        else:
            use_hard = True
            # 进度条：从 warmup 结束开始算
            p2 = (epoch - warmup_epochs) / (total_epochs - warmup_epochs + 1e-8)
            
            # Scale: 20.0 -> target_scale (由 Optuna 决定)
            dynamic_scale = 20.0 + (target_scale - 20.0) * p2
            
            # Threshold: 始终保持低位 (根据之前的经验，设为 0 是最稳的)
            # 但也可以让 Optuna 尝试微小的阈值
            dynamic_threshold = 0.0 + (threshold_final - 0.0) * p2
            
            status = "REFINE"

        if is_final_training and epoch % 1 == 0:
            print(f"🔥 [Scheduler] Ep {epoch} [{status}]: Scale={dynamic_scale:.1f}, Thres={dynamic_threshold:.2f}")

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

        # Validation
        val_acc, _, _ = evaluate_model(model, val_loader, cfg)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        # [Optuna Pruning] 保持不变
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # 计算平均 Loss
        avg_loss = loss_accum / n_batches if n_batches > 0 else 0.0
        avg_mask = mask_rate_accum / n_batches if n_batches > 0 else 0
        
        # [新增 2] 记录数据并绘图
        metrics["DPM_Loss"].append(avg_loss)
        metrics["PosteriorAcc"].append(val_acc)
        
        # 每一轮都更新图片
        # 如果是 Optuna 搜索，文件名可以加上 trial id 防止覆盖，或者就叫 training_curves.png 实时看
        if trial is not None:
            curve_name = f"optuna_trial_{trial.number}_curve.png"
        else:
            curve_name = "training_curves_final.png"
            
        plot_path = os.path.join(cfg.output_dir, curve_name)
        plot_training_curves(metrics, plot_path)

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
    
    # 1. 强制无监督设置
    cfg.alpha_unlabeled = 1.0
    cfg.labeled_per_class = 0
    cfg.posterior_sample_steps = 5 
    # [重要] 搜索时不需要跑 100 轮，跑 30-40 轮足够看趋势了
    cfg.optuna_epochs = 35 
    
    # 2. 定义搜索空间 (Search Space)
    # 基于之前的经验，我们在敏感区间附近搜索
    
    # (A) 学习率: 之前 5e-5 太稳，1e-4 可能太冲，搜一下中间值
    lr = trial.suggest_float("lr", 4e-5, 2e-4, log=True)
    
    # (B) 最终 Scale: 150 是个甜点，但可能 140 或 170 更好
    target_scale = trial.suggest_float("target_scale", 120.0, 180.0)
    
    # (C) 预热轮数: 之前 15 轮，也许 10 轮就够了，或者 20 轮更稳
    warmup_epochs = trial.suggest_int("warmup_epochs", 10, 20)
    
    # (D) 阈值: 虽然 0.0 最好，但也试探一下极小值 (0.0 ~ 0.1)
    # 如果还是 0.0 胜出，说明结论非常硬
    threshold_final = trial.suggest_float("threshold_final", 0.0, 0.1)
    
    hyperparams = {
        'target_scale': target_scale,
        'warmup_epochs': warmup_epochs,
        'threshold_final': threshold_final
    }
    
    # 3. 初始化模型
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    _, unlabeled_loader, val_loader = get_semi_loaders(cfg)
    
    # 4. 运行训练
    accuracy, _ = run_training_session(
        model, optimizer, None, unlabeled_loader, val_loader, cfg,
        is_final_training=False, # 标记为搜索模式
        trial=trial,
        hyperparams=hyperparams
    )
    
    return accuracy

# mDPM.py 中的 main 函数替换为以下内容：

def main():
    # ==========================
    # 全自动开关
    # ==========================
    # True: 先搜参数，搜完自动跑最终训练
    # False: 跳过搜索，直接用下方手动指定的参数跑最终训练
    ENABLE_AUTO_SEARCH = True 
    
    cfg = Config()
    
    # 强制配置
    cfg.alpha_unlabeled = 1.0
    cfg.labeled_per_class = 0
    cfg.posterior_sample_steps = 5 
    
    # -------------------------------------------
    # 步骤 1: 参数搜索 (Optuna)
    # -------------------------------------------
    if ENABLE_AUTO_SEARCH:
        print("🔍 [Step 1] Starting Optuna Hyperparameter Search...")
        
        # 定义搜索轮数 (比如搜 20 次)
        n_trials = 20 
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        print("\n" + "="*40)
        print("🎉 Search Finished!")
        print(f"  Best Acc: {study.best_value:.4f}")
        print("  Best Params found:")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")
        print("="*40)
        
        # 提取最佳参数
        best_params = {
            'target_scale': study.best_params['target_scale'],
            'warmup_epochs': study.best_params['warmup_epochs'],
            'threshold_final': study.best_params['threshold_final']
        }
        best_lr = study.best_params['lr']
        
    else:
        # 如果不搜参数，就用这里手动指定的默认值
        print("⏩ [Step 1] Skipping Search, using manual params...")
        best_params = {
            'target_scale': 150.0,
            'warmup_epochs': 15,
            'threshold_final': 0.0
        }
        best_lr = 1e-4

    # -------------------------------------------
    # 步骤 2: 最终训练 (Final Training)
    # -------------------------------------------
    print("\n🚀 [Step 2] Starting Final Training with BEST parameters...")
    print(f"   Configs: LR={best_lr:.2e}, Params={best_params}")
    
    # 设置最终训练的时长
    cfg.final_epochs = 100 
    
    # [关键] 必须重新实例化模型和优化器，确保是从头开始训练
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    _, unlabeled_loader, val_loader = get_semi_loaders(cfg)
    
    # 运行最终训练
    # 注意：这里我们传入 best_params，并没有传入 resume_path，
    # 意味着它是用最佳参数“从零开始”跑一个完美的 100 轮。
    run_training_session(
        model, optimizer, None, unlabeled_loader, val_loader, cfg, 
        is_final_training=True,
        hyperparams=best_params
    )

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()