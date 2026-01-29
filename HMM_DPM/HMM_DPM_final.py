"""
HMM-DPM 完整实现 (修复版)

主要改进:
1. 正确实现发射概率 p(y, z | x)
2. 添加解码器 p(y | z_1, x)
3. 完整的EM训练流程
4. 与VAE相同的监控和可视化
5. 完整的交易策略回测
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from data import download_csi500_data

# ==========================================
# 0. 全局配置
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
sns.set_theme(style="whitegrid")

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f">>> Random Seed set to: {seed}")


# ==========================================
# 1. 扩散模型组件
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalDiffusionModel(nn.Module):
    """
    条件扩散模型 (修复版)
    
    关键修复:
    1. 添加解码器 p(y | z_1, x_state)
    2. 完整的前向/反向扩散
    """
    def __init__(self, input_dim, hidden_dim=128, n_states=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_states = n_states
        
        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 状态嵌入
        self.state_emb = nn.Embedding(n_states, hidden_dim)
        
        # 噪声预测网络 (去噪)
        self.noise_pred = nn.Sequential(
            nn.Linear(input_dim + hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # ===== 关键修复: 添加解码器 =====
        self.decoder = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def predict_noise(self, x_t, t, state):
        """
        噪声预测: ε_θ(x_t, t, state)
        """
        # 时间嵌入
        t_emb = self.time_mlp(t)
        
        # 状态嵌入
        s_emb = self.state_emb(state)
        
        # 拼接
        h = torch.cat([x_t, t_emb, s_emb], dim=-1)
        
        return self.noise_pred(h)
    
    def decode(self, z_1, state):
        """
        解码器: p(y | z_1, x_state)
        
        z_1: (batch, input_dim) - 扩散的第一步
        state: (batch,) - 状态索引
        """
        s_emb = self.state_emb(state)
        h = torch.cat([z_1, s_emb], dim=-1)
        return self.decoder(h)


# ==========================================
# 2. HMM-DPM 完整模型
# ==========================================
class HMM_DPM(nn.Module):
    """
    HMM-DPM 模型 (修复版)
    
    关键修复:
    1. 正确的发射概率计算
    2. 完整的扩散采样
    3. EM训练流程
    """
    def __init__(self, input_dim, n_states=3, n_steps=50, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.n_states = n_states
        self.n_steps = n_steps
        
        # 扩散模型
        self.diffusion = ConditionalDiffusionModel(input_dim, hidden_dim, n_states)
        
        # HMM参数
        self.trans_logits = nn.Parameter(torch.zeros(n_states, n_states))
        self.start_logits = nn.Parameter(torch.zeros(n_states))
        
        # 扩散schedule
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
    
    def get_diffusion_loss(self, y, state):
        """
        标准的扩散损失: 噪声预测
        
        y: (batch, seq_len, input_dim) 或 (batch, input_dim)
        state: (batch, seq_len) 或 (batch,)
        """
        # Flatten
        if y.dim() == 3:
            B, L, D = y.shape
            y_flat = y.reshape(B * L, D)
            state_flat = state.reshape(B * L)
        else:
            y_flat = y
            state_flat = state
        
        batch_size = y_flat.shape[0]
        
        # 随机采样时间步
        t = torch.randint(0, self.n_steps, (batch_size,), device=y.device).long()
        
        # 添加噪声
        noise = torch.randn_like(y_flat)
        alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)
        y_t = torch.sqrt(alpha_bar_t) * y_flat + torch.sqrt(1 - alpha_bar_t) * noise
        
        # 预测噪声
        pred_noise = self.diffusion.predict_noise(y_t, t, state_flat)
        
        return F.mse_loss(pred_noise, noise)
    
    def get_reconstruction_loss(self, y, z_1, state):
        """
        重建损失: p(y | z_1, x_state)
        
        y: (batch, seq_len, input_dim)
        z_1: (batch, seq_len, input_dim) - 扩散第一步
        state: (batch, seq_len)
        """
        B, L, D = y.shape
        
        y_flat = y.reshape(B * L, D)
        z_1_flat = z_1.reshape(B * L, D)
        state_flat = state.reshape(B * L)
        
        # 解码
        y_recon = self.diffusion.decode(z_1_flat, state_flat)
        
        # MSE损失
        return F.mse_loss(y_recon, y_flat)
    
    def forward_diffusion(self, y, return_trajectory=False):
        """
        前向扩散: y -> z_T
        
        q(z_t | z_{t-1}) = N(√α_t z_{t-1}, (1-α_t)I)
        """
        batch_size = y.shape[0]
        
        if return_trajectory:
            z_trajectory = [y]
            z = y
            for t in range(1, self.n_steps):
                noise = torch.randn_like(z)
                alpha_t = self.alphas[t]
                z = torch.sqrt(alpha_t) * z + torch.sqrt(1 - alpha_t) * noise
                z_trajectory.append(z)
            return torch.stack(z_trajectory, dim=1)  # (batch, n_steps, dim)
        else:
            # 直接跳到z_T (一步加噪)
            noise = torch.randn_like(y)
            alpha_bar = self.alphas_cumprod[-1]
            z_T = torch.sqrt(alpha_bar) * y + torch.sqrt(1 - alpha_bar) * noise
            return z_T
    
    def compute_emission_probs_corrected(self, y, n_samples=3):
        """
        === 关键修复: 正确的发射概率计算 ===
        
        根据论文公式:
        log p(y, z | x=k) = log p(y | z_1, x=k) + log p(z轨迹 | x=k) + log p(z_T)
        
        简化版(不计算完整轨迹):
        log p(y, z | x=k) ≈ log p(y | z_1, x=k) - reconstruction_error
        
        y: (batch, seq_len, input_dim)
        返回: (batch, seq_len, n_states)
        """
        B, L, D = y.shape
        log_probs = torch.zeros(B, L, self.n_states).to(y.device)
        
        y_flat = y.reshape(B * L, D)
        
        for k in range(self.n_states):
            state_tensor = torch.full((B * L,), k, device=y.device).long()
            
            # 蒙特卡洛采样
            sample_log_probs = []
            
            for _ in range(n_samples):
                # 1. 前向扩散: y -> z (一步加噪,简化)
                noise = torch.randn_like(y_flat)
                t = torch.randint(self.n_steps // 2, self.n_steps, (B * L,), device=y.device).long()
                alpha_bar_t = self.alphas_cumprod[t].view(-1, 1)
                z_t = torch.sqrt(alpha_bar_t) * y_flat + torch.sqrt(1 - alpha_bar_t) * noise
                
                # 2. 反向去噪: z_t -> z_1 (一步)
                with torch.no_grad():
                    pred_noise = self.diffusion.predict_noise(z_t, t, state_tensor)
                
                # 近似z_1 (DDIM-style)
                alpha_bar_1 = self.alphas_cumprod[0]
                z_1_approx = (z_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
                z_1_approx = torch.sqrt(alpha_bar_1) * z_1_approx
                
                # 3. 解码: log p(y | z_1, x=k)
                y_recon = self.diffusion.decode(z_1_approx, state_tensor)
                
                # 负MSE作为log概率 (高斯似然)
                log_prob = -0.5 * ((y_flat - y_recon) ** 2).sum(dim=-1)
                
                # 4. 加上先验: log p(z)
                log_prior = -0.5 * (z_1_approx ** 2).sum(dim=-1)
                
                sample_log_probs.append(log_prob + 0.1 * log_prior)
            
            # 平均
            avg_log_prob = torch.stack(sample_log_probs, dim=0).mean(dim=0)
            log_probs[:, :, k] = avg_log_prob.reshape(B, L)
        
        return log_probs
    
    def forward_backward_sampling(self, y):
        """
        Forward-Backward采样 (正确实现)
        """
        log_emission = self.compute_emission_probs_corrected(y, n_samples=3)
        B, L, K = log_emission.shape
        
        log_trans = F.log_softmax(self.trans_logits, dim=1)
        log_start = F.log_softmax(self.start_logits, dim=0)
        
        # Forward
        alpha = torch.zeros(B, L, K).to(y.device)
        alpha[:, 0, :] = log_start.unsqueeze(0) + log_emission[:, 0, :]
        
        for t in range(1, L):
            prev = alpha[:, t-1, :].unsqueeze(2) + log_trans.unsqueeze(0)
            alpha[:, t, :] = log_emission[:, t, :] + torch.logsumexp(prev, dim=1)
        
        # Backward sampling
        sampled_states = torch.zeros(B, L, dtype=torch.long).to(y.device)
        
        probs_T = F.softmax(alpha[:, -1, :], dim=1)
        sampled_states[:, -1] = torch.multinomial(probs_T, 1).squeeze()
        
        for t in range(L - 2, -1, -1):
            next_state = sampled_states[:, t+1]
            trans_cols = log_trans[:, next_state].T
            log_prob = alpha[:, t, :] + trans_cols
            probs = F.softmax(log_prob, dim=1)
            sampled_states[:, t] = torch.multinomial(probs, 1).squeeze()
        
        return sampled_states
    
    def viterbi_decode(self, y):
        """Viterbi解码(用于预测)"""
        log_emission = self.compute_emission_probs_corrected(y, n_samples=5)
        B, L, K = log_emission.shape
        log_trans = F.log_softmax(self.trans_logits, dim=1)
        log_start = F.log_softmax(self.start_logits, dim=0)
        
        dp = torch.zeros(B, L, K).to(y.device)
        pointers = torch.zeros(B, L, K, dtype=torch.long).to(y.device)
        dp[:, 0, :] = log_start.unsqueeze(0) + log_emission[:, 0, :]
        
        for t in range(1, L):
            scores = dp[:, t-1, :].unsqueeze(2) + log_trans.unsqueeze(0)
            max_scores, prev_states = torch.max(scores, dim=1)
            dp[:, t, :] = max_scores + log_emission[:, t, :]
            pointers[:, t, :] = prev_states
        
        best_paths = []
        for b in range(B):
            path = []
            last_state = torch.argmax(dp[b, -1, :]).item()
            path.append(last_state)
            for t in range(L-1, 0, -1):
                last_state = pointers[b, t, last_state].item()
                path.append(last_state)
            best_paths.append(path[::-1])
        
        return torch.tensor(best_paths, dtype=torch.long).to(y.device)


# ==========================================
# 3. 数据准备 (修复数据泄漏)
# ==========================================
def prepare_data_no_leakage(returns_data, seq_len=30, train_ratio=0.8):
    """数据准备 (修复数据泄漏)"""
    N, D = returns_data.shape
    
    # 时间划分
    split_idx = int(N * train_ratio)
    train_returns = returns_data[:split_idx]
    test_returns = returns_data[split_idx:]
    
    print(f"时间划分: 训练 [0:{split_idx}], 测试 [{split_idx}:{N}]")
    
    # 标准化 (只在训练集上fit)
    scaler = StandardScaler()
    train_flat = np.clip(train_returns.flatten().reshape(-1, 1), -10, 10)
    scaler.fit(train_flat)
    
    train_scaled = scaler.transform(train_returns.flatten().reshape(-1, 1)).reshape(train_returns.shape)
    test_scaled = scaler.transform(test_returns.flatten().reshape(-1, 1)).reshape(test_returns.shape)
    
    # 构建序列
    X_train = []
    for i in range(len(train_scaled) - seq_len):
        X_train.append(train_scaled[i : i + seq_len])
    X_train = np.array(X_train)
    
    X_test = []
    for i in range(len(test_scaled) - seq_len):
        X_test.append(test_scaled[i : i + seq_len])
    X_test = np.array(X_test)
    
    print(f"✓ 数据准备: 训练集{X_train.shape}, 测试集{X_test.shape}")
    
    return X_train, X_test, scaler


# ==========================================
# 4. 增强的可视化 (与VAE相同)
# ==========================================
class DPMVisualizer:
    @staticmethod
    def plot_training_dashboard(history, filename="dpm_training_dashboard.png"):
        """训练监控面板"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. Warmup损失
        axes[0].plot(history['warmup_loss'], linewidth=2, color='orange')
        axes[0].set_title('Stage 1: Supervised Warmup (Diffusion Loss)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # 2. EM训练损失
        axes[1].plot(history['em_diffusion'], label='Diffusion Loss', linewidth=2, color='blue')
        axes[1].plot(history['em_reconstruction'], label='Reconstruction Loss', linewidth=2, color='green')
        axes[1].set_title('Stage 2: EM Training', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 状态分离度
        axes[2].plot(history['state_separation'], linewidth=2, color='purple')
        axes[2].set_title('State Persistence (Diagonal of Transition Matrix)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Avg Diagonal')
        axes[2].set_xlabel('Epoch')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✓ Saved: {filename}")
    
    @staticmethod
    def plot_state_analysis(trans_matrix, states, returns, filename="dpm_state_analysis.png"):
        """状态分析"""
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        
        # 1. 转移矩阵
        ax1 = fig.add_subplot(gs[0])
        sns.heatmap(trans_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=[f'S{i}' for i in range(len(trans_matrix))],
                    yticklabels=[f'S{i}' for i in range(len(trans_matrix))],
                    vmin=0, vmax=1, ax=ax1, cbar_kws={'label': 'Probability'})
        ax1.set_title('Transition Matrix', fontsize=12, fontweight='bold')
        
        # 2. 状态收益率分布
        ax2 = fig.add_subplot(gs[1])
        for s in range(trans_matrix.shape[0]):
            mask = states == s
            if np.any(mask):
                ax2.boxplot(returns[mask], positions=[s], widths=0.6,
                           patch_artist=True,
                           boxprops=dict(facecolor=f'C{s}', alpha=0.7))
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('State')
        ax2.set_ylabel('Return')
        ax2.set_title('Return Distribution by State', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✓ Saved: {filename}")


# ==========================================
# 5. 交易策略
# ==========================================
class TradingStrategy:
    """交易策略 (与VAE相同)"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def predict_states(self, X_test):
        """预测状态序列"""
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).float().to(self.device)
            states = self.model.viterbi_decode(X_test_tensor)
        return states.cpu().numpy()
    
    def backtest_state_timing(self, states, returns):
        """
        策略: 状态择时
        只在牛市状态持仓
        """
        # 计算每个状态的平均收益
        state_returns = {}
        for s in range(self.model.n_states):
            mask = states[:, -1] == s
            if np.any(mask):
                state_returns[s] = np.mean(returns[mask])
            else:
                state_returns[s] = -999.0
        
        bull_state = max(state_returns, key=state_returns.get)
        
        print(f"\n状态分析:")
        for s, ret in state_returns.items():
            print(f"  State {s}: 平均收益 = {ret:.4%}")
        print(f"  识别的牛市状态: {bull_state}")
        
        # 执行策略
        final_states = states[:, -1]
        strategy_returns = np.where(final_states == bull_state, returns, 0)
        
        # 计算净值
        nav = np.cumprod(1 + strategy_returns)
        nav_benchmark = np.cumprod(1 + returns)
        
        # 计算指标
        sharpe = (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9)) * np.sqrt(252)
        win_rate = np.mean(strategy_returns > 0)
        
        # 最大回撤
        running_max = np.maximum.accumulate(nav)
        drawdown = (nav - running_max) / running_max
        max_dd = drawdown.min()
        
        return {
            'nav': nav,
            'nav_benchmark': nav_benchmark,
            'returns': strategy_returns,
            'states': final_states,
            'bull_state': bull_state,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'max_drawdown': max_dd
        }


# ==========================================
# 6. 完整训练pipeline
# ==========================================
def train_hmm_dpm(returns_data, config, device):
    """完整训练"""
    print("\n" + "="*60)
    print("HMM-DPM 训练")
    print("="*60)
    
    # 准备数据
    X_train, X_test, scaler = prepare_data_no_leakage(
        returns_data,
        seq_len=config['seq_len'],
        train_ratio=0.8
    )
    
    X_train = torch.from_numpy(X_train).float().to(device)
    X_test_np = X_test  # 保存numpy版本用于回测
    X_test = torch.from_numpy(X_test).float().to(device)
    
    train_loader = DataLoader(
        TensorDataset(X_train),
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # 初始化模型
    model = HMM_DPM(
        input_dim=500,
        n_states=config['n_states'],
        n_steps=50,
        hidden_dim=config['hidden_dim']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    
    # 初始化HMM参数 (KMeans)
    print("\n>>> KMeans初始化...")
    volatility = np.std(X_train.cpu().numpy(), axis=2).mean(axis=1).reshape(-1, 1)
    kmeans = KMeans(n_clusters=config['n_states'], random_state=42, n_init=10).fit(volatility)
    labels = torch.tensor(kmeans.labels_, device=device).long()
    init_labels = labels.unsqueeze(1).repeat(1, config['seq_len'])
    
    # 转移矩阵初始化
    trans_init = np.eye(config['n_states']) * 0.85 + 0.15 / config['n_states']
    model.trans_logits.data = torch.tensor(np.log(trans_init + 1e-10), dtype=torch.float32).to(device)
    
    # 训练历史
    history = {
        'warmup_loss': [],
        'em_diffusion': [],
        'em_reconstruction': [],
        'state_separation': []
    }
    
    # ===== Stage 1: Warmup (有监督预训练) =====
    print("\n>>> Stage 1: Warmup (有监督预训练)...")
    warmup_dataset = TensorDataset(X_train, init_labels)
    warmup_loader = DataLoader(warmup_dataset, batch_size=config['batch_size'], shuffle=True)
    
    for epoch in range(config['warmup_epochs']):
        model.train()
        epoch_loss = 0
        
        for bx, by in warmup_loader:
            optimizer.zero_grad()
            loss = model.get_diffusion_loss(bx, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        history['warmup_loss'].append(epoch_loss / len(warmup_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['warmup_epochs']}: Loss = {history['warmup_loss'][-1]:.4f}")
    
    # ===== Stage 2: EM Training =====
    print("\n>>> Stage 2: EM联合训练...")
    
    for epoch in range(config['em_epochs']):
        model.train()
        epoch_diff_loss = 0
        epoch_recon_loss = 0
        
        # 打乱数据
        perm = torch.randperm(len(X_train))
        X_shuffled = X_train[perm]
        n_batches = len(X_train) // config['batch_size']
        
        for i in range(n_batches):
            bx = X_shuffled[i * config['batch_size'] : (i+1) * config['batch_size']]
            
            # E-step: 采样状态
            with torch.no_grad():
                sampled_states = model.forward_backward_sampling(bx)
            
            # M-step: 优化模型
            optimizer.zero_grad()
            
            # 1. 扩散损失
            diff_loss = model.get_diffusion_loss(bx, sampled_states)
            
            # 2. 重建损失
            with torch.no_grad():
                # 简化: 用原始y作为z_1的近似
                z_1 = bx + torch.randn_like(bx) * 0.1
            recon_loss = model.get_reconstruction_loss(bx, z_1, sampled_states)
            
            # 总损失
            total_loss = diff_loss + 5.0 * recon_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_diff_loss += diff_loss.item()
            epoch_recon_loss += recon_loss.item()
        
        history['em_diffusion'].append(epoch_diff_loss / n_batches)
        history['em_reconstruction'].append(epoch_recon_loss / n_batches)
        
        # 计算状态分离度
        with torch.no_grad():
            trans_matrix = F.softmax(model.trans_logits, dim=1).cpu().numpy()
            state_persistence = np.diag(trans_matrix).mean()
            history['state_separation'].append(state_persistence)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{config['em_epochs']}: "
                  f"Diff = {history['em_diffusion'][-1]:.4f}, "
                  f"Recon = {history['em_reconstruction'][-1]:.4f}, "
                  f"StatePer = {state_persistence:.3f}")
    
    print("\n✓ 训练完成!")
    
    return model, history, (X_train, X_test, X_test_np, scaler)


def main():
    """主函数"""
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 配置
    config = {
        'seq_len': 30,
        'n_states': 3,
        'hidden_dim': 128,
        'batch_size': 128,
        'lr': 1e-3,
        'warmup_epochs': 50,
        'em_epochs': 100
    }
    
    print("\n配置参数:")
    for k, v in config.items():
        print(f"  {k:15s}: {v}")
    
    # 加载数据
    print("\n>>> 加载数据...")
    df = download_csi500_data(cache_path="csi500_dataset.csv")
    returns_data = df.values.astype(np.float32)
    returns_data = np.clip(returns_data, -0.1, 0.1)
    
    print(f"数据形状: {returns_data.shape}")
    
    # 训练
    model, history, data = train_hmm_dpm(returns_data, config, device)
    X_train, X_test, X_test_np, scaler = data
    
    # 可视化训练过程
    print("\n>>> 生成可视化...")
    visualizer = DPMVisualizer()
    visualizer.plot_training_dashboard(history)
    
    # 转移矩阵
    trans_matrix = F.softmax(model.trans_logits, dim=1).detach().cpu().numpy()
    
    # 回测
    print("\n>>> 回测策略...")
    strategy = TradingStrategy(model, device)
    states = strategy.predict_states(X_test_np)
    
    # 对应的真实收益
    test_start = int(len(returns_data) * 0.8) + config['seq_len']
    test_returns = returns_data[test_start : test_start + len(states)].mean(axis=1)
    
    results = strategy.backtest_state_timing(states, test_returns)
    
    # 状态分析图
    visualizer.plot_state_analysis(trans_matrix, results['states'], test_returns)
    
    # 绩效报告
    print("\n" + "="*60)
    print("回测结果")
    print("="*60)
    print(f"Sharpe Ratio:    {results['sharpe']:.4f}")
    print(f"Win Rate:        {results['win_rate']:.2%}")
    print(f"Max Drawdown:    {results['max_drawdown']:.2%}")
    print(f"Final NAV:       {results['nav'][-1]:.4f}")
    print(f"Benchmark NAV:   {results['nav_benchmark'][-1]:.4f}")
    
    # 净值曲线
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(results['nav'], label='Strategy', linewidth=2)
    axes[0].plot(results['nav_benchmark'], label='Benchmark', linewidth=2, alpha=0.7, linestyle='--')
    axes[0].set_title('Net Asset Value', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(results['states'], drawstyle='steps-post', linewidth=1.5, color='purple')
    axes[1].set_title('Predicted States', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('State')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dpm_backtest_results.png', dpi=150)
    plt.close()
    print("\n✓ Saved: dpm_backtest_results.png")
    
    print("\n✅ 所有流程完成!")


if __name__ == "__main__":
    main()