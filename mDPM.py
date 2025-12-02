import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设您的项目中有 DPM 相关的组件
from common_dpm import * # DPMEncoder, ConditionalDPM, DPMForwardProcess, DPMBackwardProcess
import optuna

# -----------------------
# Model Definition (mDPM Adaptation)
# -----------------------
# -----------------------
# Model Definition (mDPM Adaptation - Z simplified to Noise)
# -----------------------
class mDPM_SemiSup(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ConditionalDPM 只需要条件 y (离散标签) 和时间步 t。
        self.cond_denoiser = ConditionalUnet(
            in_channels=cfg.image_channels,
            base_channels=cfg.unet_base_channels,
            num_classes=cfg.num_classes,
            time_emb_dim=cfg.unet_time_emb_dim
        )
        self.dpm_process = DPMForwardProcess(cfg.timesteps) 
        self.register_buffer('registered_pi', torch.ones(cfg.num_classes) / cfg.num_classes)
        
    def forward(self, x_0, cfg, y=None):
        batch_size = x_0.size(0)
        
        # Monte Carlo 估计：采样时间步 t 和噪声 epsilon (不再有 VAE 的 KL-z)
        t = torch.randint(0, cfg.timesteps, (batch_size,), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.dpm_process.q_sample(x_0, t, noise) # x_t 即为 DPM 的连续潜在变量
        
        # -------------------
        # 监督模式 (Labeled Data) - 简化 C-DPM 损失
        # -------------------
        if y is not None:
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            
            # DPM Loss: L_simple (预测噪声与真实噪声的 L2 损失)
            pred_noise = self.cond_denoiser(x_t, t, y_onehot)
            dpm_loss = F.mse_loss(pred_noise, noise, reduction='mean') 
            
            # 由于 z 被简化，KL_z = 0。 total_loss = dpm_loss
            return dpm_loss, -dpm_loss.item(), dpm_loss.item(), 0.0, None, None
            
        # -------------------
        # 无监督模式 (Unlabeled Data) - PVEM E/M-Step
        # -------------------
        else:
            log_pi = torch.log(self.registered_pi + 1e-8).unsqueeze(0).to(x_0.device)
            log_lik_proxy = []
            
            # E-Step (近似计算 log P(x_0|x=k))
            for k in range(cfg.num_classes):
                y_onehot_k = F.one_hot(torch.full((batch_size,), k, device=x_0.device),
                                       num_classes=cfg.num_classes).float()
                
                # 对数空间多步近似 & Monte Carlo 估计：用负的 L_t 作为 log lik proxy
                pred_noise_k = self.cond_denoiser(x_t, t, y_onehot_k)
                dpm_loss_k = F.mse_loss(pred_noise_k, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                
                log_p_proxy = -dpm_loss_k # log P(x_0|x=k) 代理
                log_lik_proxy.append(log_p_proxy.unsqueeze(1))
                
            log_lik_proxy = torch.cat(log_lik_proxy, dim=1)
            logits = log_pi + log_lik_proxy # 近似 Log P(x=k|x_0)
            
            # Gumbel Softmax (松弛 E-Step)
            resp = gumbel_softmax_sample(logits, cfg.current_gumbel_temp)
            
            # M-Step: 计算期望 DPM 损失 (Recon)
            weighted_dpm_loss = 0.0
            for k in range(cfg.num_classes):
                y_onehot_k = F.one_hot(torch.full((batch_size,), k, device=x_0.device),
                                       num_classes=cfg.num_classes).float()
                
                pred_noise_k = self.cond_denoiser(x_t, t, y_onehot_k)
                dpm_loss_k = F.mse_loss(pred_noise_k, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                
                weighted_dpm_loss += (resp[:, k] * dpm_loss_k).mean()
            
            # 熵惩罚
            entropy = -(resp * torch.log(resp + 1e-8)).sum(dim=1).mean()
            
            # 总损失：加权的 DPM 损失 - 熵惩罚
            total_loss = weighted_dpm_loss - cfg.lambda_entropy * entropy
            
            return total_loss, -total_loss.item(), weighted_dpm_loss.item(), 0.0, resp.detach(), None
# -----------------------
# Posterior Accuracy Evaluation (与 mVAE 略有不同)
# -----------------------
# -----------------------
# Posterior Accuracy Evaluation 
# -----------------------
def evaluate_model(model, loader, cfg):
    """
    计算后验聚类标签与真实标签的对齐准确率和 NMI，使用 DPM 损失 L_t 作为负对数似然的代理。
    """
    model.eval()
    preds, ys_true = [], []
    
    # 使用固定的时间步 T/2 进行评估
    t_eval_val = cfg.timesteps // 2 
    
    with torch.no_grad():
        for x_0, y_true in loader:
            x_0 = x_0.to(cfg.device)
            batch_size = x_0.size(0)
            
            # 1. 采样 x_t (连续潜在变量)
            current_noise = torch.randn_like(x_0)
            current_t = torch.full((batch_size,), t_eval_val, device=cfg.device, dtype=torch.long)
            x_t = model.dpm_process.q_sample(x_0, current_t, current_noise)
            
            # 2. 计算近似 Log P(x|x0)
            log_pi = torch.log(model.registered_pi + 1e-8).unsqueeze(0).to(x_0.device)
            dpm_loss_proxies = [] # -L_t 代理

            for k in range(cfg.num_classes):
                # 构造硬标签 one-hot 向量 (用于 ConditinalUnet 的输入)
                y_onehot_k = F.one_hot(torch.full((batch_size,), k, device=x_0.device),
                                       num_classes=cfg.num_classes).float()
                
                # 计算条件 DPM 损失 L_t(k)
                pred_noise_k = model.cond_denoiser(x_t, current_t, y_onehot_k)
                dpm_loss_k = F.mse_loss(pred_noise_k, current_noise, reduction='none').view(batch_size, -1).mean(dim=1)
                
                # 使用 -L_t 作为 log lik proxy
                dpm_loss_proxies.append((-dpm_loss_k).unsqueeze(1))
            
            logits = torch.cat(dpm_loss_proxies, dim=1) + log_pi # 近似 Log P(x|x0)
            
            pred_cluster = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred_cluster)
            ys_true.append(y_true.numpy())

    # 3. 聚类对齐 (使用匈牙利算法)
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
    
    # 4. 计算指标
    posterior_acc = np.mean(aligned_preds == ys_true)
    nmi = NMI(ys_true, preds)
    
    # 返回 Acc, Mapping, NMI
    return posterior_acc, cluster2label, nmi

# =====================================================
# 4. 采样过程 (DPM Backward Process)
# =====================================================
def sample_and_save_dpm(denoiser, dpm_process, num_classes, out_path, device, n_per_class=10):
    """
    使用 DPM 逆过程从噪声 X_T 和类别条件 x 生成样本。
    """
    T = dpm_process.timesteps
    denoiser.eval()

    with torch.no_grad():
        # 1. 初始化噪声 x_T
        shape = (n_per_class * num_classes, dpm_process.image_channels, 28, 28)
        x_t = torch.randn(shape, device=device)
        
        # 2. 构造类别条件 (硬标签索引)
        y_cond = torch.arange(num_classes).to(device).repeat_interleave(n_per_class).long()
        
        # 3. 逆向采样循环 (从 T-1 迭代到 0)
        for i in reversed(range(1, T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # 提取参数
            alpha_t = dpm_process._extract_t(dpm_process.alphas, t, shape)
            alpha_bar_t = dpm_process._extract_t(dpm_process.alphas_cumprod, t, shape)
            one_minus_alpha_t_bar = dpm_process._extract_t(dpm_process.sqrt_one_minus_alphas_cumprod, t, shape)
            
            # 预测噪声
            pred_noise = denoiser(x_t, t, y_cond)
            
            # 估计 x_0 (可选，用于截断或加速采样)
            # pred_x0 = (x_t - pred_noise * one_minus_alpha_t_bar) / alpha_bar_t.sqrt()
            
            # 计算均值 mu_t-1
            mu_t = (x_t - (1 - alpha_t) / one_minus_alpha_t_bar * pred_noise) / alpha_t.sqrt()
            
            # 计算方差 sigma_t-1 (通常为 beta_t)
            sigma_t = dpm_process._extract_t(dpm_process.betas, t, shape).sqrt()
            
            if i > 1:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t) # 在最后一步不加噪声
                
            x_t = mu_t + sigma_t * noise # 更新 x_{t-1}

        final_samples = x_t
        save_image(final_samples, out_path, nrow=n_per_class, normalize=True)
    print(f"💾 Saved DPM samples to {out_path}")

# =====================================================
# E. 训练循环和主函数 (Training Loop and Main)
# =====================================================

def run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                         is_final_training=False, trial_id=None):
    total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs
    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    metrics = {"Neg_ELBO": [], "DPM_Loss": [], "KLz": [], "NMI": [], "PosteriorAcc": [], "tau": []}
    best_val_nmi = -np.inf

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_neg_elbo, epoch_dpm_loss = 0.0, 0.0
        
        # Tau Annealing
        if epoch > total_epochs * 0.5:
            cfg.current_gumbel_temp = max(cfg.min_gumbel_temp, cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)

        # 确保 zip 循环长度一致 (以较短的为准，这是半监督的常见做法)
        for (x_lab, y_lab), (x_un, _) in zip(labeled_loader, unlabeled_loader):
            x_lab, y_lab = x_lab.to(cfg.device), y_lab.to(cfg.device).long()
            x_un = x_un.to(cfg.device)
            
            # 监督损失
            loss_lab, elbo_lab, recon_lab, kl_lab, _, _ = model(x_lab, cfg, y_lab)
            
            # 无监督损失 (返回 resp 用于 EMA)
            loss_un, elbo_un, recon_un, kl_un, resp, _ = model(x_un, cfg, None)
            
            loss = loss_lab + cfg.alpha_unlabeled * loss_un

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # 记录指标
            epoch_neg_elbo += (loss_lab.item() + loss_un.item()) / 2
            epoch_dpm_loss += (recon_lab + recon_un) / 2 # recon_lab/recon_un 对应 DPM_Loss

        # EMA update
        with torch.no_grad():
            if resp is not None:
                model.registered_pi.copy_(0.95 * model.registered_pi + 0.05 * resp.mean(0).detach())

        # ---- Evaluate ----
        posterior_acc, cluster2label, val_nmi = evaluate_model(model, val_loader, cfg)
        
        metrics["Neg_ELBO"].append(epoch_neg_elbo / len(labeled_loader))
        metrics["DPM_Loss"].append(epoch_dpm_loss / len(labeled_loader))
        metrics["KLz"].append(0.0)
        metrics["NMI"].append(val_nmi)
        metrics["PosteriorAcc"].append(posterior_acc)
        metrics["tau"].append(cfg.current_gumbel_temp)

        if val_nmi > best_val_nmi:
            best_val_nmi = val_nmi

        mode = "FINAL" if is_final_training else "OPTUNA"
        print(f"[{mode}] Epoch {epoch}/{total_epochs} | NMI={val_nmi:.4f} | Acc={posterior_acc:.4f} "
              f"| τ={cfg.current_gumbel_temp:.3f}")

        # ---- Save Samples ----
        if is_final_training and (epoch % 10 == 0 or epoch == total_epochs):
            sample_and_save_dpm(model.cond_denoiser, model.dpm_process, cfg.num_classes,
                                os.path.join(sample_dir, f"final_epoch{epoch:03d}.png"), cfg.device)
    
    # 保存最终后验映射
    with open(os.path.join(cfg.output_dir, "posterior_mapping.json"), "w") as f:
        json.dump(cluster2label, f, indent=2)
    print(f"✅ Final posterior accuracy: {posterior_acc:.4f}")

    return best_val_nmi, metrics

def objective(trial):
    """Optuna 目标函数：最小化 -NMI"""
    from common_dpm import Config
    cfg = Config()
    cfg.output_dir = "./mDPM_semi_optuna"
    
    # 建议的超参数 (根据 mVAE 和 DPM 特点调整)
    cfg.unet_base_channels = trial.suggest_categorical("base_channels", [32, 64])
    cfg.lambda_entropy = trial.suggest_float("lambda_entropy", 1.0, 10.0)
    cfg.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    cfg.alpha_unlabeled = trial.suggest_float("alpha_unlabeled", 0.5, 2.0)
    
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)

    best_nmi, _ = run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                                       is_final_training=False, trial_id=trial.number)
    return -best_nmi


def main():
    # 1. Optuna 超参搜索
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    print("--- Starting Optuna Hyperparameter Search ---")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    print("\n--- Optuna Complete ---")
    print("Best Parameters:", best_params)

    # 2. 最终训练配置
    cfg = Config()
    for k, v in best_params.items():
        setattr(cfg, k, v)
    json.dump(best_params, open(os.path.join(cfg.output_dir, "mDPM_best_params.json"), "w"), indent=4)

    # 3. 最终训练
    model = mDPM_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)

    print("\n--- Starting Final Training ---")
    final_nmi, metrics = run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                                              is_final_training=True)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "mDPM_best_model.pt"))

    # 4. 可视化
    generate_visualizations(model, val_loader, metrics, cfg)
    print(f"✅ Training and Visualization Complete. Final NMI: {final_nmi:.4f}")

if __name__ == "__main__":
    main()