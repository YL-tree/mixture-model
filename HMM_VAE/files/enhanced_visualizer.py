"""
增强的可视化模块
包含详细的VAE质量监控和训练过程可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import norm


class EnhancedVisualizer:
    """增强的可视化工具,包含VAE质量检查"""
    
    @staticmethod
    def check_vae_quality(y_real, y_recon, z_data, stage_name="VAE", filename="vae_quality_check.png", states=None):
        """
        详细检查VAE重建质量

        y_real: (n_samples, seq_len, n_features) 或 (n_samples, n_features) - 真实数据
        y_recon: (n_samples, seq_len, n_features) 或 (n_samples, n_features) - 重建数据
        z_data: (n_samples, latent_dim) - 潜变量
        stage_name: 阶段名称
        states: (n_samples,) 可选, 状态标签用于PCA着色
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 处理维度
        if len(y_real.shape) == 3:
            y_real_avg = y_real.mean(axis=1)  # (n_samples, n_features)
            y_recon_avg = y_recon.mean(axis=1)
            # 用于时间序列可视化
            y_real_ts = y_real.mean(axis=2)  # (n_samples, seq_len)
            y_recon_ts = y_recon.mean(axis=2)
        else:
            y_real_avg = y_real
            y_recon_avg = y_recon
            y_real_ts = None
            y_recon_ts = None
        
        # === 1. 重建样本对比 (时间序列) ===
        ax1 = fig.add_subplot(gs[0, :])
        n_samples_to_show = min(5, len(y_real))
        indices = np.random.choice(len(y_real), n_samples_to_show, replace=False)
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_samples_to_show))
        
        for i, idx in enumerate(indices):
            if y_real_ts is not None:
                real_seq = y_real_ts[idx]
                recon_seq = y_recon_ts[idx]
            else:
                # 如果是2D,取部分特征展示
                real_seq = y_real[idx, :30]
                recon_seq = y_recon[idx, :30]
            
            offset = i * 2.0
            ax1.plot(real_seq + offset, color=colors[i], alpha=0.7, linewidth=1.5, label=f'Real {i}')
            ax1.plot(recon_seq + offset, color=colors[i], linestyle='--', linewidth=2, label=f'Recon {i}')
        
        ax1.set_title(f'{stage_name}: Reconstruction Quality', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value (with offset)')
        ax1.legend(ncol=5, fontsize=8, loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # === 2. 潜空间分布 (PCA) ===
        ax2 = fig.add_subplot(gs[1, 0])
        if z_data.shape[1] > 2:
            pca = PCA(n_components=2)
            z_vis = pca.fit_transform(z_data)
            explained_var = pca.explained_variance_ratio_
            title = f'Latent Space (PCA)\nVar: {explained_var[0]:.1%} + {explained_var[1]:.1%}'
        else:
            z_vis = z_data
            title = 'Latent Space'
        
        if states is not None:
            unique_states = np.unique(states)
            colors_map = plt.cm.Set1(np.linspace(0, 0.8, len(unique_states)))
            for idx, s in enumerate(unique_states):
                mask = states == s
                ax2.scatter(z_vis[mask, 0], z_vis[mask, 1], alpha=0.4, s=15,
                           color=colors_map[idx], label=f'State {s}')
            ax2.legend(fontsize=8, markerscale=2)
        else:
            scatter = ax2.scatter(z_vis[:, 0], z_vis[:, 1], alpha=0.4, s=15,
                                 c=np.arange(len(z_vis)), cmap='viridis')
            plt.colorbar(scatter, ax=ax2, label='Sample Index')
        ax2.set_title(title, fontsize=10, fontweight='bold')
        ax2.set_xlabel('PC1' if z_data.shape[1] > 2 else 'Dim 1')
        ax2.set_ylabel('PC2' if z_data.shape[1] > 2 else 'Dim 2')
        ax2.grid(True, alpha=0.3)
        
        # === 3. 重建误差分布 ===
        ax3 = fig.add_subplot(gs[1, 1])
        errors = np.abs(y_real_avg - y_recon_avg).mean(axis=1)
        ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        ax3.axvline(errors.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {errors.mean():.4f}')
        ax3.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(errors):.4f}')
        ax3.set_title('Reconstruction Error Distribution', fontsize=10, fontweight='bold')
        ax3.set_xlabel('MAE per Sample')
        ax3.set_ylabel('Frequency')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # === 4. 潜变量分布 vs 标准正态 ===
        ax4 = fig.add_subplot(gs[1, 2])
        z_flat = z_data.flatten()
        counts, bins, patches = ax4.hist(z_flat, bins=60, alpha=0.6, edgecolor='black', 
                                         density=True, color='lightgreen', label='Latent z')
        
        # 叠加标准正态分布
        x_range = np.linspace(z_flat.min(), z_flat.max(), 100)
        ax4.plot(x_range, norm.pdf(x_range, 0, 1), 'r-', linewidth=2.5, label='N(0,1)')
        
        # 叠加实际分布
        actual_mean, actual_std = z_flat.mean(), z_flat.std()
        ax4.plot(x_range, norm.pdf(x_range, actual_mean, actual_std), 'b--', 
                linewidth=2, label=f'N({actual_mean:.2f},{actual_std:.2f})')
        
        ax4.set_title('Latent Distribution vs Prior', fontsize=10, fontweight='bold')
        ax4.set_xlabel('Value')
        ax4.set_ylabel('Density')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # === 5. 特征级重建对比 (散点图) ===
        ax5 = fig.add_subplot(gs[2, 0])
        sample_size = min(10000, y_real_avg.size)
        sample_indices = np.random.choice(y_real_avg.size, sample_size, replace=False)
        y_real_sample = y_real_avg.flatten()[sample_indices]
        y_recon_sample = y_recon_avg.flatten()[sample_indices]
        
        ax5.hexbin(y_real_sample, y_recon_sample, gridsize=40, cmap='Blues', alpha=0.8)
        # 添加y=x参考线
        min_val = min(y_real_sample.min(), y_recon_sample.min())
        max_val = max(y_real_sample.max(), y_recon_sample.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        ax5.set_title('Real vs Reconstructed (Hexbin)', fontsize=10, fontweight='bold')
        ax5.set_xlabel('Real Value')
        ax5.set_ylabel('Reconstructed Value')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # === 6. 重建质量指标表 ===
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        # 计算详细指标
        mse = np.mean((y_real_avg - y_recon_avg) ** 2)
        mae = np.mean(np.abs(y_real_avg - y_recon_avg))
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((y_real_avg - y_recon_avg) ** 2)
        ss_tot = np.sum((y_real_avg - y_real_avg.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        
        # 相关系数
        corr = np.corrcoef(y_real_avg.flatten(), y_recon_avg.flatten())[0, 1]
        
        # 潜变量统计
        z_mean = z_data.mean()
        z_std = z_data.std()
        z_min, z_max = z_data.min(), z_data.max()
        
        # KL散度 (近似)
        kl_div = 0.5 * np.mean(z_data**2 + 1 - 2*np.log(np.abs(z_data) + 1e-10))
        
        metrics_text = f"""
╔═══════════════════════════════╗
║  RECONSTRUCTION METRICS       ║
╠═══════════════════════════════╣
║  MSE:        {mse:10.6f}     ║
║  MAE:        {mae:10.6f}     ║
║  RMSE:       {rmse:10.6f}     ║
║  R²:         {r2:10.4f}     ║
║  Corr:       {corr:10.4f}     ║
╠═══════════════════════════════╣
║  LATENT STATISTICS            ║
╠═══════════════════════════════╣
║  Mean:       {z_mean:10.4f}     ║
║  Std:        {z_std:10.4f}     ║
║  Min:        {z_min:10.4f}     ║
║  Max:        {z_max:10.4f}     ║
║  KL(approx): {kl_div:10.4f}     ║
╠═══════════════════════════════╣
║  DATA SHAPE                   ║
╠═══════════════════════════════╣
║  Samples:    {len(y_real):10d}     ║
║  Features:   {y_real_avg.shape[1]:10d}     ║
║  Latent Dim: {z_data.shape[1]:10d}     ║
╚═══════════════════════════════╝
        """
        
        ax6.text(0.05, 0.5, metrics_text, fontsize=9, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        
        # === 7. 每个特征的重建误差 ===
        ax7 = fig.add_subplot(gs[2, 2])
        dim_errors = np.abs(y_real_avg - y_recon_avg).mean(axis=0)
        ax7.fill_between(range(len(dim_errors)), 0, dim_errors, alpha=0.5, color='coral')
        ax7.plot(dim_errors, linewidth=1.5, color='darkred')
        ax7.axhline(dim_errors.mean(), color='blue', linestyle='--', 
                   linewidth=2, label=f'Mean: {dim_errors.mean():.4f}')
        ax7.set_title('Per-Feature Reconstruction Error', fontsize=10, fontweight='bold')
        ax7.set_xlabel('Feature Index')
        ax7.set_ylabel('MAE')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3)
        
        # 总标题
        plt.suptitle(f'🔍 {stage_name} Quality Assessment', 
                    fontsize=15, fontweight='bold', y=0.998)
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved VAE quality check: {filename}")
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'corr': corr,
            'z_mean': z_mean,
            'z_std': z_std
        }
    
    @staticmethod
    def plot_training_dashboard(history, filename="training_dashboard.png"):
        """
        绘制综合训练监控面板
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        epochs = range(1, len(history['vae_recon']) + 1)
        
        # === 1. VAE重建损失 ===
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, history['vae_recon'], linewidth=2, color='blue')
        ax1.fill_between(epochs, 0, history['vae_recon'], alpha=0.3, color='blue')
        ax1.set_title('VAE Reconstruction Loss', fontsize=11, fontweight='bold')
        ax1.set_ylabel('MSE Loss')
        ax1.set_xlabel('Epoch')
        ax1.grid(True, alpha=0.3)
        
        # === 2. KL散度 ===
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, history['vae_kld'], linewidth=2, color='green')
        ax2.fill_between(epochs, 0, history['vae_kld'], alpha=0.3, color='green')
        ax2.set_title('VAE KL Divergence', fontsize=11, fontweight='bold')
        ax2.set_ylabel('KL Divergence')
        ax2.set_xlabel('Epoch')
        ax2.grid(True, alpha=0.3)
        
        # === 3. HMM负对数似然 ===
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, history['hmm_nll'], linewidth=2, color='red')
        ax3.fill_between(epochs, 0, history['hmm_nll'], alpha=0.3, color='red')
        ax3.set_title('HMM Negative Log-Likelihood', fontsize=11, fontweight='bold')
        ax3.set_ylabel('NLL')
        ax3.set_xlabel('Epoch')
        ax3.grid(True, alpha=0.3)
        
        # === 4. 总损失 ===
        ax4 = fig.add_subplot(gs[1, 0])
        if 'total' in history:
            ax4.plot(epochs, history['total'], linewidth=2, color='purple')
            ax4.fill_between(epochs, 0, history['total'], alpha=0.3, color='purple')
        ax4.set_title('Total Loss', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Loss')
        ax4.set_xlabel('Epoch')
        ax4.grid(True, alpha=0.3)
        
        # === 5. 状态分离度 ===
        ax5 = fig.add_subplot(gs[1, 1])
        if 'state_separation' in history and len(history['state_separation']) > 0:
            ax5.plot(epochs, history['state_separation'], linewidth=2, color='orange')
            ax5.set_title('State Separation', fontsize=11, fontweight='bold')
            ax5.set_ylabel('Distance')
            ax5.set_xlabel('Epoch')
            ax5.grid(True, alpha=0.3)
        
        # === 6. 学习率曲线(如果有) ===
        ax6 = fig.add_subplot(gs[1, 2])
        if 'temperature' in history and len(history['temperature']) > 0:
            ax6.plot(epochs, history['temperature'], linewidth=2, color='brown')
            ax6.set_title('Gumbel Temperature', fontsize=11, fontweight='bold')
            ax6.set_ylabel('Temperature')
            ax6.set_xlabel('Epoch')
            ax6.grid(True, alpha=0.3)
        
        # === 7. VAE各组件损失对比 ===
        ax7 = fig.add_subplot(gs[2, :])
        ax7_twin = ax7.twinx()
        
        line1 = ax7.plot(epochs, history['vae_recon'], linewidth=2, color='blue', 
                        label='Recon Loss')
        ax7.set_ylabel('Reconstruction Loss', color='blue')
        ax7.tick_params(axis='y', labelcolor='blue')
        
        line2 = ax7_twin.plot(epochs, history['vae_kld'], linewidth=2, color='green',
                             linestyle='--', label='KL Divergence')
        ax7_twin.set_ylabel('KL Divergence', color='green')
        ax7_twin.tick_params(axis='y', labelcolor='green')
        
        ax7.set_title('VAE Components Over Time', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Epoch')
        ax7.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax7.legend(lines, labels, loc='upper right')
        
        plt.suptitle('📊 Training Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training dashboard: {filename}")


# 导出
__all__ = ['EnhancedVisualizer']
