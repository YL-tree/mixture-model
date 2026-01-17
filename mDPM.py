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
        
    def estimate_posterior_logits(self, x_0, cfg):
        batch_size = x_0.size(0)
        num_classes = cfg.num_classes
        M = cfg.posterior_sample_steps
        
        accum_log_lik = torch.zeros(batch_size, num_classes, device=x_0.device)
        
        with torch.no_grad():
            for _ in range(M):
                # [关键修改]
                # 不再从 [0, 1000] 均匀采样
                # 而是专注于 "语义区间" [300, 700]
                # 这不是 hack，这是 "降低估计方差" 的数学手段
                t_start = int(0.3 * cfg.timesteps)
                t_end = int(0.7 * cfg.timesteps)
                
                # 采样 t
                t = torch.randint(t_start, t_end, (batch_size,), device=x_0.device).long()
                
                noise = torch.randn_like(x_0)
                x_t = self.dpm_process.q_sample(x_0, t, noise)
                
                for k in range(num_classes):
                    y_cond = torch.full((batch_size,), k, device=x_0.device, dtype=torch.long)
                    y_onehot = F.one_hot(y_cond, num_classes=num_classes).float()
                    
                    # 此时 U-Net 内部会自动给 y_emb 乘上比较大的权重 (因为 t 在中间)
                    # 所以如果 k 是错的，pred_noise 就会错得很离谱 -> MSE 很大
                    # 如果 k 是对的，pred_noise 就会很准 -> MSE 很小
                    pred_noise = self.cond_denoiser(x_t, t, y_onehot)
                    
                    # Log Likelihood Proxy
                    mse = F.mse_loss(pred_noise, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                    accum_log_lik[:, k] += -mse

        # 在循环结束后，accum_log_lik 里面存的是 -MSE
        # MSE 的数值通常很小 (0.02 左右)，导致差异只有 0.001 级别
        
        # [关键修改] 手动放大差异 (Scale Factor)
        # 这相当于人为降低了 E-Step 的 "温度"
        # 让猜对的类别的 Logits 显著高于猜错的
        scale_factor = 1.0  
        accum_log_lik = accum_log_lik * scale_factor
        
        log_pi = torch.log(self.registered_pi + 1e-8).unsqueeze(0)
        final_logits = log_pi + (accum_log_lik / M)
        
        return final_logits

    def forward(self, x_0, cfg, y=None):
        """
        前向传播包含 E-Step 和 M-Step 的损失计算
        """
        batch_size = x_0.size(0)
        
        # -------------------
        # 监督模式 (Labeled Data)
        # -------------------
        if y is not None:
            # 标准 DDPM 训练：采样 1 个 t
            t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t = self.dpm_process.q_sample(x_0, t, noise)
            
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean') 
            
            return dpm_loss, -dpm_loss.item(), dpm_loss.item(), 0.0, None, None
            
        # -------------------
        # 无监督模式 (Unlabeled Data) - PVEM
        # -------------------
        else:
            # === E-Step: 推断潜变量 x 的分布 ===
            # 这里使用了 Multi-step 近似，比原来的单步更准
            logits = self.estimate_posterior_logits(x_0, cfg)
            
            # 使用 Gumbel Softmax 进行重参数化或松弛采样
            # 这里的 resp 对应论文中的 \tilde{p}(x|z,y)
            resp = gumbel_softmax_sample(logits, cfg.current_gumbel_temp)
            
            # === M-Step: 训练去噪网络 ===
            # 论文: Sample x ~ p(x|z,y) then train DDPM
            # 实际操作: 使用 resp 加权的 Loss (Soft-EM)，这在深度学习中比 Hard Sampling 更稳定
            
            # 重新采样一个 t 用于训练 (标准 DDPM 做法)
            t_train = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
            noise = torch.randn_like(x_0)
            x_t_train = self.dpm_process.q_sample(x_0, t_train, noise)
            
            weighted_dpm_loss = 0.0
            
            # 计算加权 Loss
            # L = Sum_k q(x=k) * ||eps - eps_theta(x_t, t, k)||^2
            for k in range(cfg.num_classes):
                y_onehot_k = F.one_hot(torch.full((batch_size,), k, device=x_0.device),
                                       num_classes=cfg.num_classes).float()
                
                # 这里需要梯度，用于更新 cond_denoiser
                pred_noise_k = self.cond_denoiser(x_t_train, t_train, y_onehot_k)
                
                # Per-sample loss
                dpm_loss_k = F.mse_loss(pred_noise_k, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                
                # 使用 E-Step 算出来的 resp 进行加权
                # resp.detach() 很关键！确保梯度不回传到 E-Step 逻辑
                weighted_dpm_loss += (resp[:, k].detach() * dpm_loss_k).mean()
            
            # === 辅助损失 ===
            # 熵最小化: 鼓励模型做出确定的预测 (Paper context: Self-consistent)
            # entropy = -(resp * torch.log(resp + 1e-8)).sum(dim=1).mean()
            
            # total_loss = weighted_dpm_loss - cfg.lambda_entropy * entropy
            total_loss = weighted_dpm_loss
            
            return total_loss, -total_loss.item(), weighted_dpm_loss.item(), 0.0, resp.detach(), None

# -----------------------
# Evaluation Utils
# -----------------------
def evaluate_model(model, loader, cfg):
    """
    改进版评估：使用多个时间步累积 Loss 来降低方差，提高分类准确率。
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("Scipy not found, skipping detailed evaluation.")
        return 0.0, {}, 0.0

    model.eval()
    preds, ys_true = [], []
    
    # [修正 1] 剔除 700, 900，只保留信号最强的区间
    # eval_timesteps = [300, 400, 500] 
    eval_timesteps = [60, 100, 140] 
    
    # [修正 2] 增加重复次数 (训练时为了速度可以用 3-5 次，不用 10 次)
    n_repeats = 5
    
    with torch.no_grad():
        for x_0, y_true in loader:
            x_0 = x_0.to(cfg.device)
            batch_size = x_0.size(0)
            
            # (Batch, Num_Classes)
            cumulative_mse = torch.zeros(batch_size, cfg.num_classes, device=cfg.device)
            
            for t_val in eval_timesteps:
                mse_t_sum = torch.zeros(batch_size, cfg.num_classes, device=cfg.device)
                
                # 重复采样以消除方差
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
            # [新增调试打印]
            unique_preds, counts = np.unique(pred_cluster, return_counts=True)
            print(f"DEBUG: Predicted Clusters Distribution: {dict(zip(unique_preds, counts))}")
        
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

        
        save_image(x_t.clamp(-1, 1), out_path, nrow=n_per_class, normalize=True, value_range=(-1, 1))
    print(f"💾 Saved DPM samples to {out_path}")

# -----------------------
# Training Engine
# -----------------------
def run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                         is_final_training=False, trial_id=None):
    """
    通用训练函数：兼容全监督、半监督、无监督。
    通过检测 loader 是否为 None 以及 cfg.alpha_unlabeled 来自动切换策略。
    """
    total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs
    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    metrics = {"Loss": [], "NMI": [], "Acc": []}
    best_val_nmi = -np.inf
    
    # --- 1. 自动判断训练模式 ---
    mode = "UNKNOWN"
    if labeled_loader is not None and unlabeled_loader is not None:
        mode = "SEMI_SUPERVISED"
        print("🚀 模式检测: 半监督训练 (Semi-Supervised)")
    elif labeled_loader is not None and unlabeled_loader is None:
        mode = "SUPERVISED"
        # 强制修正：如果没无标签数据，alpha 必须为 0
        cfg.alpha_unlabeled = 0.0 
        print("🚀 模式检测: 全监督训练 (Fully Supervised)")
    elif labeled_loader is None and unlabeled_loader is not None:
        mode = "UNSUPERVISED"
        print("🚀 模式检测: 无监督训练 (Unsupervised)")
    else:
        raise ValueError("❌ 错误: Labeled 和 Unlabeled loader 不能同时为空！")

    # --- 2. 训练循环 ---
    for epoch in range(1, total_epochs + 1):
        model.train()
        loss_accum = 0.0
        n_batches = 0
        
        # === 策略 A: 无监督模式下的温度退火 ===
        # 无监督需要激进的退火 (High -> Low)
        if mode == "UNSUPERVISED":
             if epoch > 5:
                cfg.current_gumbel_temp = max(cfg.min_gumbel_temp, cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)
        # 半监督/全监督通常保持较低温度或缓慢退火
        elif epoch > total_epochs * 0.5:
             cfg.current_gumbel_temp = max(cfg.min_gumbel_temp, cfg.current_gumbel_temp * 0.995)

        # === 策略 B: 半监督模式下的 Warm-up ===
        # 前 10 个 Epoch 强制只看有标签数据
        current_alpha_un = cfg.alpha_unlabeled
        if mode == "SEMI_SUPERVISED" and epoch <= 10:
            current_alpha_un = 0.0
        
        # === 3. 构造通用迭代器 ===
        # 技巧：将不同的 Loader 包装成统一的 (batch_lab, batch_un) 格式
        if mode == "SEMI_SUPERVISED":
            # 取 min length，或者用 itertools.cycle 循环较短的那个
            iterator = zip(labeled_loader, unlabeled_loader)
            loader_len = len(labeled_loader) # 以有标签的为准
        elif mode == "SUPERVISED":
            # 伪造一个空的 unlabeled batch
            iterator = ((batch, None) for batch in labeled_loader)
            loader_len = len(labeled_loader)
        elif mode == "UNSUPERVISED":
            # 伪造一个空的 labeled batch
            iterator = ((None, batch) for batch in unlabeled_loader)
            loader_len = len(unlabeled_loader)

        # === 4. Batch 循环 ===
        for batch_lab, batch_un in iterator:
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=cfg.device)
            resp = None # 用于更新 Prior

            # --- 计算有监督部分 ---
            if batch_lab is not None:
                x_lab, y_lab = batch_lab
                x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device).long()
                
                # 有标签 Loss (始终权重为 1.0 或自定义 alpha_labeled)
                loss_lab, _, _, _, _, _ = model(x_lab, cfg, y_lab)
                total_loss += loss_lab

            # --- 计算无监督部分 ---
            if batch_un is not None and current_alpha_un > 0:
                x_un, _ = batch_un # 忽略标签
                x_un = x_un.to(cfg.device)
                
                # 无标签 Loss
                loss_un, _, _, _, resp, _ = model(x_un, cfg, None)
                total_loss += current_alpha_un * loss_un
            
            # --- 反向传播 ---
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            loss_accum += total_loss.item()
            n_batches += 1

            # EMA 更新 Prior (仅当有无监督推断发生时)
            if resp is not None:
                with torch.no_grad():
                    # 无监督模式下，动量设为 0.9 更快响应，强迫模型注意那些“没人选”的类别
                    momentum = 0.9 if mode == "UNSUPERVISED" else 0.99
                    model.registered_pi.copy_(momentum * model.registered_pi + (1-momentum) * resp.mean(0).detach())

        # === 5. 评估与日志 ===
        # 使用修正后的 evaluate_model (包含黄金区间和多次采样)
        raw_acc, _, val_nmi = evaluate_model(model, val_loader, cfg)
        
        # 记录最佳模型
        target_metric = raw_acc if mode == "SUPERVISED" else val_nmi
        if target_metric > best_val_nmi:
            best_val_nmi = target_metric
            if is_final_training:
                torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))

        log_tag = "FINAL" if is_final_training else f"TRIAL-{trial_id}"
        print(f"[{log_tag}] Mode: {mode} | Epoch {epoch} | Loss: {loss_accum/n_batches:.4f} | "
              f"Acc: {raw_acc:.4f} | NMI: {val_nmi:.4f} | τ: {cfg.current_gumbel_temp:.3f}")

        # 定期保存图片
        if is_final_training and (epoch % 10 == 0 or epoch == total_epochs):
            sample_and_save_dpm(model.cond_denoiser, model.dpm_process, cfg.num_classes,
                                os.path.join(sample_dir, f"epoch_{epoch:03d}.png"), cfg.device)
    
    return best_val_nmi, metrics

def objective(trial):
    cfg = Config()
    cfg.output_dir = "./mDPM_optuna_temp"
    
    # Hyperparameters to tune
    # cfg.unet_base_channels = trial.suggest_categorical("base_channels", [32, 64])
    # 强制让维度为32
    cfg.unet_base_channels = 32

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
    # ==========================================
    # 🎛️ 控制开关：是否进行 Optuna 超参数搜索
    # True  = 运行搜索，找到最优参后训练 (慢)
    # False = 跳过搜索，直接用 Config 默认参数训练 (快)
    # ==========================================
    RUN_OPTUNA = False 

    # 初始化基础配置
    cfg = Config()

    if RUN_OPTUNA:
        # --- 1. 运行 Optuna 搜索 ---
        print("--- Starting Optuna Hyperparameter Search ---")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=5) 
        
        print("Best params found:", study.best_params)
        
        # 将搜索到的最优参数覆盖到 cfg 中
        for k, v in study.best_params.items():
            setattr(cfg, k, v)
            
        # 保存最优参数备份
        with open(os.path.join(cfg.output_dir, "optuna_best_params.json"), "w") as f:
            json.dump(study.best_params, f, indent=4)
            
    else:
        # --- 2. 跳过搜索，使用默认/手动配置 ---
        print("--- Skipping Optuna: Using Manual/Default Config ---")
        
        # [关键安全设置] 
        # 之前我们在 objective 里强制改成了 32 以防爆显存
        # 如果跳过 Optuna，必须在这里手动设为 32，否则会读 common_dpm 里的默认值 64
        cfg.unet_base_channels = 32
        
        # 你也可以在这里手动微调其他参数，例如：
        # cfg.lr = 1e-3
        # cfg.lambda_entropy = 2.0
        
    # --- 3. 开始最终训练 ---
    print("\n" + "="*30)
    print("--- Starting Final Training ---")
    print(f"Config: Channels={cfg.unet_base_channels}, LR={cfg.lr}")
    print("="*30 + "\n")

    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)
    
    # 运行训练
    run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg, is_final_training=True)
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "final_model.pt"))
    print(f"✅ Done. Model saved to {cfg.output_dir}")

if __name__ == "__main__":
    main()