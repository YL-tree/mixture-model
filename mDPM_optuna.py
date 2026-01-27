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
    
    if hyperparams is None:
        hyperparams = {
            'target_scale': 150.0,
            'warmup_epochs': 15,
            'threshold_start': 0.0,  # 默认值
            'threshold_final': 0.0
        }

    target_scale = hyperparams.get('target_scale', 150.0)
    warmup_epochs = hyperparams.get('warmup_epochs', 15)
    threshold_start = hyperparams.get('threshold_start', 0.0) # 新增接收
    threshold_final = hyperparams.get('threshold_final', 0.0)
    
    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    start_epoch = 1
    best_val_acc = 0.0

    # 模式检测
    mode = "UNKNOWN"
    if labeled_loader is not None and unlabeled_loader is not None: 
        mode = "SEMI_SUPERVISED"
    elif labeled_loader is not None: 
        mode = "SUPERVISED"
        cfg.alpha_unlabeled = 0.0
    elif unlabeled_loader is not None: 
        mode = "UNSUPERVISED"
        cfg.alpha_unlabeled = 1.0
    

    print(f"🚀 Training Mode: {mode} (Total Epochs: {total_epochs})")

    metrics = {
        "DPM_Loss": [],      # 记录 Loss
        "PosteriorAcc": []   # 记录 Accuracy
    }
    
    

    for epoch in range(start_epoch, total_epochs + 1):
        progress = (epoch - 1) / total_epochs
        # ==========================================
        # [动态调度器] 由 Optuna 参数控制
        # ==========================================
        # ==========================================
        # [调度器] 根据模式选择策略
        # ==========================================
        if mode == "UNSUPERVISED":
            # --- 策略 A: 无监督 (Optuna 优化的探索策略) ---
            # 特点：先软后硬，低门槛，适合从零发现结构
            if epoch <= warmup_epochs:
                use_hard = False
                p1 = epoch / warmup_epochs
                dynamic_scale = 5.0 + (20.0 - 5.0) * p1
                dynamic_threshold = 0.0 
                status = "EXPLORE"
            else:
                use_hard = True
                p2 = (epoch - warmup_epochs) / (total_epochs - warmup_epochs + 1e-8)
                dynamic_scale = 20.0 + (target_scale - 20.0) * p2
                dynamic_threshold = 0.0 + (threshold_final - 0.0) * p2
                status = "REFINE"

         # SEMI_SUPERVISED
            # --- 策略 B: 半监督 (FixMatch 策略) ---
            # 特点：全程硬标签，高自信，中等门槛
            # 既然有标签指路，直接自信地 Scale=150
            # 半监督/监督模式保持原样
        elif mode == "SEMI_SUPERVISED":
            use_hard = True 
            # 搜索出来的 Scale
            dynamic_scale = target_scale 
            
            # 搜索出来的爬坡曲线
            # 比如从 0.4 爬到 0.95
            progress = (epoch - 1) / total_epochs
            dynamic_threshold = threshold_start + (threshold_final - threshold_start) * progress
            
            status = "SEMI-SUP"
        
            

        if is_final_training and epoch % 1 == 0:
            print(f"🔥 [Scheduler] Ep {epoch} [{status}]: Scale={dynamic_scale:.1f}, Thres={dynamic_threshold:.2f}")

        model.train()
        loss_accum = 0.0
        mask_rate_accum = 0.0
        n_batches = 0
        
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

            # --- 1. 计算监督 Loss (Anchor) ---
            if batch_lab is not None:
                x_lab, y_lab = batch_lab
                x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device).long()
                # 监督 forward (Scale=1.0, 纯 MSE)
                l_sup, _, _, _, _, _ = model(x_lab, cfg, y=y_lab)
                total_loss += l_sup

            # --- 2. 计算无监督 Loss (Consistency) ---
            if batch_un is not None:
                x_un, _ = batch_un
                x_un = x_un.to(cfg.device)
                
                # 传入调度器参数
                l_unsup, _, _, mask_rate, _, _ = model(x_un, cfg, y=None, 
                                                       current_scale=dynamic_scale,
                                                       current_lambda=0.01,
                                                       threshold=dynamic_threshold,
                                                       use_hard_label=use_hard)
                
                # 半监督时通常 alpha=1.0 即可
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

        # ==========================================
        # [Optuna] 保留剪枝功能
        # ==========================================
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # 记录数据
        avg_loss = loss_accum / n_batches if n_batches > 0 else 0.0
        avg_mask = mask_rate_accum / n_batches if n_batches > 0 else 0
        
        metrics["DPM_Loss"].append(avg_loss)
        metrics["PosteriorAcc"].append(val_acc)
        
        # 绘图
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
    
    # ==========================================
    # 1. 设置搜索模式 (从 main 函数外部控制，或在这里修改)
    # ==========================================
    # 选项: "UNSUPERVISED" 或 "SEMI_SUPERVISED"
    # 注意：Optuna 运行时这里的变量需要能被访问，建议直接在这里硬编码当前想搜的模式
    SEARCH_MODE = "SEMI_SUPERVISED" 
    
    # ==========================================
    # 2. 定义不同模式下的搜索空间
    # ==========================================
    
    if SEARCH_MODE == "UNSUPERVISED":
        # --- 无监督配置 ---
        cfg.labeled_per_class = 0
        cfg.alpha_unlabeled = 1.0
        
        # [搜索空间 - 无监督]
        # 策略：低自信，低门槛，小心翼翼
        lr = trial.suggest_float("lr", 4e-5, 2e-4, log=True)
        target_scale = trial.suggest_float("target_scale", 120.0, 180.0)
        warmup_epochs = trial.suggest_int("warmup_epochs", 10, 20)
        
        # 无监督的门槛通常很低
        threshold_start = 0.0 # 固定
        threshold_final = trial.suggest_float("threshold_final", 0.0, 0.1)

    else: # SEMI_SUPERVISED
        # --- 半监督配置 ---
        cfg.labeled_per_class = 10  # 每类 10 张
        cfg.alpha_unlabeled = 1.0
        
        # [搜索空间 - 半监督]
        # 策略：高自信，高门槛，强力纠偏
        lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True) # 稍微大一点的 LR
        
        # Scale 甚至可以更大一点，因为有标签拉着
        target_scale = trial.suggest_float("target_scale", 500.0, 600.0)
        
        # 半监督不需要太长的 Warmup，因为一开始就有标签
        warmup_epochs = trial.suggest_int("warmup_epochs", 5, 10)
        
        # [关键] 搜索门槛的“爬坡”曲线
        # 起点：不能是 0 (太吵)，也不能是 0.9 (太硬)
        threshold_start = trial.suggest_float("threshold_start", 0.3, 0.7)
        # 终点：必须很高 (FixMatch 标准)
        threshold_final = trial.suggest_float("threshold_final", 0.8, 0.99)
    
    # ------------------------------------------
    # 3. 运行训练
    # ------------------------------------------
    
    # 同样只跑 35 轮看趋势
    cfg.optuna_epochs = 35 
    
    # 打包参数传给 run_training_session
    hyperparams = {
        'target_scale': target_scale,
        'warmup_epochs': warmup_epochs,
        'threshold_start': threshold_start, # 新增
        'threshold_final': threshold_final
    }
    
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 获取数据 loader
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)
    
    # 运行
    accuracy, _ = run_training_session(
        model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
        is_final_training=False, 
        trial=trial,
        hyperparams=hyperparams
    )
    
    return accuracy

# mDPM.py 中的 main 函数替换为以下内容：
def main():
    # ==========================
    # 全自动开关
    # ==========================
    ENABLE_AUTO_SEARCH = True 
    
    # 当前任务模式：用于 main 函数里非搜索部分的配置
    # 注意：objective 函数里的 SEARCH_MODE 需要单独改，或者你可以把 objective 定义在 main 里面闭包调用
    CURRENT_MODE = "SEMI_SUPERVISED" 
    
    if ENABLE_AUTO_SEARCH:
        print(f"🔍 [Step 1] Starting Optuna Search for {CURRENT_MODE}...")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20) # 跑 20 次
        
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
            'threshold_start': study.best_params.get('threshold_start', 0.0), # 兼容性写法
            'threshold_final': study.best_params['threshold_final']
        }
        best_lr = study.best_params['lr']
        
    else:
        print("⏩ [Step 1] Skipping Search, using manual params...")
        # 手动指定（示例）
        best_params = {
            'target_scale': 150.0,
            'warmup_epochs': 10,
            'threshold_start': 0.5,
            'threshold_final': 0.95
        }
        best_lr = 1e-4

    # -------------------------------------------
    # 步骤 2: 最终训练
    # -------------------------------------------
    print("\n🚀 [Step 2] Starting Final Training...")
    
    cfg = Config()
    cfg.final_epochs = 60
    
    if CURRENT_MODE == "SEMI_SUPERVISED":
        cfg.labeled_per_class = 10
    else:
        cfg.labeled_per_class = 0
        
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