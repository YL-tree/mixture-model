# mVAE_semisup_aligned.py (Refactored for clarity and modularity)
import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as NMI
from common import * # 保持你项目中 common.py 的工具与模型组件

# -----------------------
# Model Definition (Unchanged)
# -----------------------
class mVAE_SemiSup(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = Encoder(cfg.latent_dim)
        self.dec = ConditionalDecoder(cfg.latent_dim, cfg.num_classes)
        self.register_buffer('registered_pi', torch.ones(cfg.num_classes) / cfg.num_classes)

    def forward(self, x, cfg, y=None):
        mu, logvar = self.enc(x)
        z = reparameterize(mu, logvar)
        batch_size = x.size(0)
        if y is not None:
            y_onehot = F.one_hot(y, num_classes=cfg.num_classes).float()
            x_recon = self.dec(z, y_onehot)
            recon_loss = F.mse_loss(x_recon, x, reduction='sum') / batch_size
            kl_z = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            elbo = -(recon_loss + cfg.beta * kl_z)
            return -elbo, elbo.item(), recon_loss.item(), kl_z.item(), None, mu.detach()
        else:
            log_pi = torch.log(self.registered_pi + 1e-8).unsqueeze(0).to(x.device)
            recon_loglik = []
            for k in range(cfg.num_classes):
                y_onehot = F.one_hot(torch.full((batch_size,), k, device=x.device), num_classes=cfg.num_classes).float()
                x_recon = self.dec(z, y_onehot)
                log_p = -F.mse_loss(x_recon, x, reduction='none').view(batch_size, -1).sum(dim=1)
                recon_loglik.append(log_p.unsqueeze(1))
            recon_loglik = torch.cat(recon_loglik, dim=1)
            logits = log_pi + recon_loglik
            resp = gumbel_softmax_sample(logits, cfg.current_gumbel_temp)
            x_recons = []
            for k in range(cfg.num_classes):
                y_onehot = F.one_hot(torch.full((batch_size,), k, device=x.device), num_classes=cfg.num_classes).float()
                x_recon = self.dec(z, y_onehot)
                x_recons.append(x_recon.unsqueeze(1))
            x_recons = torch.cat(x_recons, dim=1)
            x_mean = (resp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * x_recons).sum(dim=1)
            kl_z = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            recon_loss = F.mse_loss(x_mean, x, reduction='sum') / batch_size
            elbo = -(recon_loss + cfg.beta * kl_z)
            entropy = -(resp * torch.log(resp + 1e-8)).sum(dim=1).mean()
            total_loss = -elbo + cfg.lambda_entropy * entropy
            return total_loss, elbo.item(), recon_loss.item(), kl_z.item(), resp.detach(), mu.detach()

# -----------------------
# Data Loader (Unchanged)
# -----------------------
def get_semi_loaders(cfg, labeled_per_class=100):
    # ... (这部分代码与您原来的一样，无需改动)
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    labels = np.array(dataset.targets)
    labeled_idx, unlabeled_idx = [], []
    for c in range(cfg.num_classes):
        idx_c = np.where(labels == c)[0]
        labeled_idx.extend(idx_c[:labeled_per_class])
        unlabeled_idx.extend(idx_c[labeled_per_class:])
    labeled_set = Subset(dataset, labeled_idx)
    unlabeled_set = Subset(dataset, unlabeled_idx)
    val_set = Subset(dataset, list(range(int(0.1 * len(dataset)))))
    labeled_loader = DataLoader(labeled_set, batch_size=cfg.batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    return labeled_loader, unlabeled_loader, val_loader

### 放入common.py了
# # -----------------------
# # Helper: Performance Evaluation
# # -----------------------
# def evaluate_performance(model, loader, cfg):
#     """在给定的数据集上评估模型，返回 NMI 分数。"""
#     model.eval()
#     zs, ys_true = [], []
#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(cfg.device)
#             mu, _ = model.enc(x)
#             zs.append(mu.cpu().numpy())
#             ys_true.append(y.numpy())
    
#     zs = np.concatenate(zs)
#     ys_true = np.concatenate(ys_true)
#     if len(zs) < cfg.num_classes: return 0.0

#     km = KMeans(n_clusters=cfg.num_classes, n_init='auto', random_state=42).fit(zs)
#     nmi = NMI(ys_true, km.labels_)
#     return nmi

    
# # -----------------------
# # NEW: Unified Training Function
# # -----------------------
# def run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg, is_final_training=False):
#     """
#     一个统一的训练函数，既可用于Optuna搜索，也可用于最终训练。
    
#     Args:
#         model: The PyTorch model.
#         optimizer: The optimizer.
#         ...: Data loaders and config.
#         is_final_training (bool): 如果为True，会启用更长的训练周期、退火机制和保存样本等操作。
    
#     Returns:
#         best_val_nmi (float): 在所有周期中达到的最佳NMI分数。
#         metrics (dict): 包含所有指标历史记录的字典。
#     """
#     total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs
#     kl_anneal_epochs = 5 # KL预热的周期数
#     sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    
#     beta_target = cfg.beta
#     if is_final_training:
#         cfg.beta = beta_target * 0.01 # 最终训练时从极小值开始退火
#         print(f"Starting Final Training with Beta Annealing. Target beta: {beta_target:.2f}")
#         os.makedirs(sample_dir, exist_ok=True)
        
#     metrics = {"ELBO": [], "Recon": [], "KLz": [], "NMI": [], "tau": []}
#     best_val_nmi = -np.inf

#     for epoch in range(1, total_epochs + 1):
#         model.train()
#         epoch_loss, epoch_elbo, epoch_recon, epoch_kl = 0.0, 0.0, 0.0, 0.0
#         num_batches = 0
        
#         # --- 启用退火机制 (对最终训练至关重要) ---
#         if is_final_training:
            
#             # Beta Annealing (KL Warmup)
#             current_beta = beta_target * min(1.0, epoch / kl_anneal_epochs)
#             cfg.beta = current_beta 
            
#             # Tau Schedule: 保持一半周期，然后退火
#             if epoch > total_epochs * 0.5:
#                 cfg.current_gumbel_temp = max(cfg.min_gumbel_temp, cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)
#             if (epoch+1) % 5 == 0:
#                 sample_and_save(model.dec, cfg.latent_dim, cfg.num_classes,
#                             out_path=os.path.join(cfg.output_dir, f"samples_epoch{epoch+1}.png"),device=cfg.device)
#         else:
#             # Optuna试运行时，使用简单退火
#             cfg.current_gumbel_temp = max(cfg.min_gumbel_temp, cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)
        
#         for (x_lab, y_lab), (x_un, _) in zip(labeled_loader, unlabeled_loader):
#             x_lab, y_lab, x_un = x_lab.to(cfg.device), y_lab.to(cfg.device), x_un.to(cfg.device)
            
#             loss_lab, elbo_lab, recon_lab, kl_lab, _, _ = model(x_lab, cfg, y_lab)
#             loss_un, elbo_un, recon_un, kl_un, resp, _ = model(x_un, cfg, None)

#             loss = loss_lab + cfg.alpha_unlabeled * loss_un
#             optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()

#             epoch_elbo += (elbo_lab + elbo_un) / 2
#             epoch_recon += (recon_lab + recon_un) / 2
#             epoch_kl += (kl_lab + kl_un) / 2
#             num_batches += 1

#         # EMA update registered_pi to avoid collapse
#         with torch.no_grad():
#             if resp is not None:
#                 mean_resp = resp.mean(0).detach()
#                 model.registered_pi.copy_(0.95 * model.registered_pi + 0.05 * mean_resp)

#         # --- 评估与记录 ---
#         val_nmi = evaluate_performance(model, val_loader, cfg)
#         metrics["ELBO"].append(epoch_elbo / num_batches)
#         metrics["Recon"].append(epoch_recon / num_batches)
#         metrics["KLz"].append(epoch_kl / num_batches)
#         metrics["NMI"].append(val_nmi)
#         metrics["tau"].append(cfg.current_gumbel_temp)
        
#         if val_nmi > best_val_nmi:
#             best_val_nmi = val_nmi
        
#         log_prefix = "[Final Train]" if is_final_training else "[Optuna Trial]"
#         print(f"{log_prefix} Epoch {epoch}/{total_epochs} | NMI: {val_nmi:.4f} | ELBO: {metrics['ELBO'][-1]:.2f} | Beta: {cfg.beta:.2f} | Tau: {cfg.current_gumbel_temp:.3f}")

#         # --- 保存样本 (仅在最终训练时) ---
#         if is_final_training and (epoch % 5 == 0 or epoch == total_epochs):
#             sample_and_save(model.dec, cfg.latent_dim, cfg.num_classes,
#                             out_path=os.path.join(cfg.output_dir, f"samples_epoch{epoch:03d}.png"), device=cfg.device)
   

#     return best_val_nmi, metrics

# -----------------------
# NEW: Standalone Visualization Function
# -----------------------
def generate_visualizations(model, val_loader, metrics, cfg):
    """
    生成并保存所有可视化结果。
    """
    print("\n--- Generating Final Visualizations ---")
    output_dir = cfg.output_dir
    final_nmi = metrics["NMI"][-1]

    # 1. 绘制并保存训练指标曲线
    plt.figure(figsize=(12, 8))
    for key, value in metrics.items():
        plt.plot(value, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Semi-supervised Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "semi_training_metrics.png"))
    plt.close()
    print(f"Saved training metrics to {os.path.join(output_dir, 'semi_training_metrics.png')}")

    # 2. 绘制并保存潜空间散点图
    model.eval()
    with torch.no_grad():
        zs_vis, ys_vis = [], []
        for x, y in val_loader:
            x = x.to(cfg.device)
            mu, _ = model.enc(x)
            zs_vis.append(mu.cpu().numpy())
            ys_vis.append(y.numpy())
        zs_vis = np.concatenate(zs_vis)
        ys_vis = np.concatenate(ys_vis)

    plt.figure(figsize=(8, 7))
    if zs_vis.shape[1] == 2:
        scatter = plt.scatter(zs_vis[:, 0], zs_vis[:, 1], c=ys_vis, cmap='tab10', s=10, alpha=0.8)
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
    else: # 使用PCA降维
        pca = PCA(n_components=2)
        proj = pca.fit_transform(zs_vis)
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=ys_vis, cmap='tab10', s=10, alpha=0.8)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
    
    plt.title(f"Latent Space Visualization (Validation Set) | Final NMI: {final_nmi:.4f}")
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(cfg.num_classes)))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "semi_latent_scatter.png"))
    plt.close()
    print(f"Saved latent space scatter plot to {os.path.join(output_dir, 'semi_latent_scatter.png')}")

    # 3. 生成并保存最终的条件样本
    with torch.no_grad():
        n_per_class = 10
        z = torch.randn(n_per_class, cfg.latent_dim).to(cfg.device).repeat(cfg.num_classes, 1)
        y_cond = torch.arange(cfg.num_classes).to(cfg.device).repeat_interleave(n_per_class)
        y_onehot = F.one_hot(y_cond, num_classes=cfg.num_classes).float()
        final_samples = model.dec(z, y_onehot)
        save_image(final_samples, os.path.join(output_dir, "semi_final_samples.png"), nrow=n_per_class, normalize=True)
    print(f"Saved final conditional samples to {os.path.join(output_dir, 'semi_final_samples.png')}")

# # -----------------------
# # Main Execution Logic (Refactored)
# # -----------------------
# def objective(trial):
#     """Optuna的目标函数。"""
#     cfg = Config()
#     # 定义超参数搜索空间
#     cfg.latent_dim = trial.suggest_categorical("latent_dim", [2, 4, 6, 8])
#     cfg.beta = trial.suggest_float("beta", 0.1, 5.0)
#     cfg.lambda_entropy = trial.suggest_float("lambda_entropy", 1.0, 10.0)
#     cfg.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
#     cfg.alpha_unlabeled = trial.suggest_float("alpha_unlabeled", 0.5, 2.0)
    
#     # 实例化模型和数据加载器
#     model = mVAE_SemiSup(cfg).to(cfg.device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
#     labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)

#     # 调用统一的训练函数 (试运行模式)
#     best_val_nmi, _ = run_training_session(
#         model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg, is_final_training=False
#     )
    
#     # Optuna的目标是最大化NMI，所以返回其负值
#     return -best_val_nmi

# def main():
#     """主执行函数。"""
#     # 1. 超参数搜索
#     study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
#     study.optimize(objective, n_trials=20) # 建议增加试运行次数以获得更好结果
    
#     best_params = study.best_params
#     print("\n--- Optuna Search Complete ---")
#     print(f"Best parameters found: {best_params}")
    
#     # 2. 使用最佳参数进行最终训练
#     print("\n--- Starting Final Training with Best Parameters ---")
#     cfg = Config()
#     cfg.output_dir = "./semi_final"
#     for key, value in best_params.items():
#         setattr(cfg, key, value)
    
#     os.makedirs(cfg.output_dir, exist_ok=True)
#     json.dump(best_params, open(os.path.join(cfg.output_dir, "semi_best_params.json"), "w"), indent=4)

#     # 实例化模型、优化器和数据加载器
#     model = mVAE_SemiSup(cfg).to(cfg.device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
#     labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)

#     # 调用统一的训练函数 (最终训练模式)
#     final_nmi, metrics_history = run_training_session(
#         model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg, is_final_training=True
#     )
    
#     # 保存最终模型
#     torch.save(model.state_dict(), os.path.join(cfg.output_dir, "semi_best_model.pt"))
#     print(f"\nSaved final model to {os.path.join(cfg.output_dir, 'semi_best_model.pt')}")

#     # 3. 生成所有可视化结果
#     generate_visualizations(model, val_loader, metrics_history, cfg)
    
#     # 绘制指标（子图）
#     plot_metrics_subplots(metrics_history, cfg.output_dir)

# if __name__ == "__main__":
#     main()

# mVAE_semisup_aligned.py (Enhanced with Posterior Accuracy + Optuna Sampling)
import os, json, torch, optuna
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from common import *  # 使用你现有的 Encoder / Decoder / reparameterize / gumbel_softmax_sample 等定义


# =====================================================
# Posterior Accuracy Evaluation
# =====================================================
def evaluate_posterior_accuracy(model, loader, cfg):
    """
    计算后验聚类标签与真实标签的对齐准确率。
    返回:
        posterior_acc, cluster2label(dict), label2cluster(dict)
    """
    model.eval()
    preds, ys_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(cfg.device)
            mu, _ = model.enc(x)
            z = mu
            batch_size = x.size(0)
            log_pi = torch.log(model.registered_pi + 1e-8).unsqueeze(0).to(x.device)
            recon_loglik = []
            for k in range(cfg.num_classes):
                y_onehot = F.one_hot(torch.full((batch_size,), k, device=x.device),
                                     num_classes=cfg.num_classes).float()
                x_recon = model.dec(z, y_onehot)
                log_p = -F.mse_loss(x_recon, x, reduction='none').view(batch_size, -1).sum(dim=1)
                recon_loglik.append(log_p.unsqueeze(1))
            logits = torch.cat(recon_loglik, dim=1) + log_pi
            pred_cluster = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred_cluster)
            ys_true.append(y.numpy())

    preds = np.concatenate(preds)
    ys_true = np.concatenate(ys_true)
    n_classes = cfg.num_classes
    cost_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cost_matrix[i, j] = -np.sum((ys_true == i) & (preds == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster2label = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    label2cluster = {int(v): int(k) for k, v in cluster2label.items()}
    aligned_preds = np.array([cluster2label.get(p, 0) for p in preds])
    posterior_acc = np.mean(aligned_preds == ys_true)
    return posterior_acc, cluster2label, label2cluster


# =====================================================
# Sampling Utility (used by both Optuna + Final Training)
# =====================================================
def sample_and_save(decoder, latent_dim, num_classes, out_path, device):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    decoder.eval()
    with torch.no_grad():
        n_per_class = 10
        z = torch.randn(n_per_class, latent_dim).to(device)
        grids = []
        for k in range(num_classes):
            y = torch.full((n_per_class,), k, dtype=torch.long, device=device)
            y_onehot = F.one_hot(y, num_classes=num_classes).float()
            x_gen = decoder(z, y_onehot)
            grids.append(x_gen)
        grid = torch.cat(grids, dim=0)
        save_image(grid, out_path, nrow=n_per_class, normalize=True)
    print(f"💾 Saved samples to {out_path}")


# =====================================================
# Training Loop (Optuna + Final Unified)
# =====================================================
def run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                         is_final_training=False, trial_id=None):
    total_epochs = cfg.final_epochs if is_final_training else cfg.optuna_epochs
    kl_anneal_epochs = 5
    sample_dir = os.path.join(cfg.output_dir, "sample_progress")
    os.makedirs(sample_dir, exist_ok=True)

    beta_target = cfg.beta
    if is_final_training:
        cfg.beta = beta_target * 0.01
        print(f"🔥 Starting Final Training with Beta Annealing (Target β={beta_target:.2f})")

    metrics = {"ELBO": [], "Recon": [], "KLz": [], "NMI": [], "PosteriorAcc": [], "tau": []}
    best_val_nmi = -np.inf

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_elbo, epoch_recon, epoch_kl = 0.0, 0.0, 0.0

        # Beta Annealing
        if is_final_training:
            cfg.beta = beta_target * min(1.0, epoch / kl_anneal_epochs)
        # Tau Annealing
        if epoch > total_epochs * 0.5:
            cfg.current_gumbel_temp = max(cfg.min_gumbel_temp, cfg.current_gumbel_temp * cfg.gumbel_anneal_rate)

        for (x_lab, y_lab), (x_un, _) in zip(labeled_loader, unlabeled_loader):
            x_lab, y_lab, x_un = x_lab.to(cfg.device), y_lab.to(cfg.device), x_un.to(cfg.device)
            loss_lab, elbo_lab, recon_lab, kl_lab, _, _ = model(x_lab, cfg, y_lab)
            loss_un, elbo_un, recon_un, kl_un, resp, _ = model(x_un, cfg, None)
            loss = loss_lab + cfg.alpha_unlabeled * loss_un

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_elbo += (elbo_lab + elbo_un) / 2
            epoch_recon += (recon_lab + recon_un) / 2
            epoch_kl += (kl_lab + kl_un) / 2

        # EMA update
        with torch.no_grad():
            if resp is not None:
                model.registered_pi.copy_(0.95 * model.registered_pi + 0.05 * resp.mean(0).detach())

        # ---- Evaluate ----
        val_nmi = evaluate_performance(model, val_loader, cfg)
        posterior_acc, cluster2label, _ = evaluate_posterior_accuracy(model, val_loader, cfg)
        metrics["ELBO"].append(epoch_elbo / len(labeled_loader))
        metrics["Recon"].append(epoch_recon / len(labeled_loader))
        metrics["KLz"].append(epoch_kl / len(labeled_loader))
        metrics["NMI"].append(val_nmi)
        metrics["PosteriorAcc"].append(posterior_acc)
        metrics["tau"].append(cfg.current_gumbel_temp)

        if val_nmi > best_val_nmi:
            best_val_nmi = val_nmi

        mode = "FINAL" if is_final_training else "OPTUNA"
        print(f"[{mode}] Epoch {epoch}/{total_epochs} | NMI={val_nmi:.4f} | Acc={posterior_acc:.4f} "
              f"| β={cfg.beta:.3f} | τ={cfg.current_gumbel_temp:.3f}")

        # ---- Save Samples ----
        if is_final_training:
            if epoch % 5 == 0 or epoch == total_epochs:
                sample_and_save(model.dec, cfg.latent_dim, cfg.num_classes,
                                os.path.join(sample_dir, f"final_epoch{epoch:03d}.png"), cfg.device)
        else:
            # Optuna trial 保存样本（低频）
            if epoch % 3 == 0 or epoch == total_epochs:
                sample_and_save(model.dec, cfg.latent_dim, cfg.num_classes,
                                os.path.join(sample_dir, f"trial{trial_id}_epoch{epoch:03d}.png"), cfg.device)

    # ---- Save final posterior mapping ----
    posterior_acc, cluster2label, _ = evaluate_posterior_accuracy(model, val_loader, cfg)
    with open(os.path.join(cfg.output_dir, "posterior_mapping.json"), "w") as f:
        json.dump(cluster2label, f, indent=2)
    print(f"✅ Posterior accuracy: {posterior_acc:.4f}")

    return best_val_nmi, metrics


# =====================================================
# Optuna Objective Function
# =====================================================
def objective(trial):
    cfg = Config()
    cfg.output_dir = "./semi_final"
    cfg.latent_dim = trial.suggest_categorical("latent_dim", [2, 4, 8, 16])
    cfg.beta = trial.suggest_float("beta", 0.1, 5.0)
    cfg.lambda_entropy = trial.suggest_float("lambda_entropy", 1.0, 10.0)
    cfg.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    cfg.alpha_unlabeled = trial.suggest_float("alpha_unlabeled", 0.5, 2.0)

    model = mVAE_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)

    best_nmi, _ = run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                                       is_final_training=False, trial_id=trial.number)
    return -best_nmi


# =====================================================
# Final Training + Visualization
# =====================================================
def main():
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    print("\n--- Optuna Complete ---")
    print(best_params)

    cfg = Config()
    cfg.output_dir = "./semi_final"
    os.makedirs(cfg.output_dir, exist_ok=True)
    for k, v in best_params.items():
        setattr(cfg, k, v)
    json.dump(best_params, open(os.path.join(cfg.output_dir, "semi_best_params.json"), "w"), indent=4)

    model = mVAE_SemiSup(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    labeled_loader, unlabeled_loader, val_loader = get_semi_loaders(cfg)

    final_nmi, metrics = run_training_session(model, optimizer, labeled_loader, unlabeled_loader, val_loader, cfg,
                                              is_final_training=True)
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "semi_best_model.pt"))

    plot_metrics_subplots(metrics, cfg.output_dir)
    generate_visualizations(model, val_loader, metrics, cfg)
    print("✅ Training and Visualization Complete.")


if __name__ == "__main__":
    main()
