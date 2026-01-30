"""
完整的HMM-VAE实现 - 严格按照论文框架
Partially Variational EM for Hidden Markov Models with VAE Emissions

主要特点:
1. Forward-Backward采样算法
2. 条件VAE解码器 (以状态为条件)
3. EM框架训练
4. 正确的发射概率计算 p(y,z|x)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random

# ==========================================
# 0. 全局设置
# ==========================================
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f">>> Random Seed set to: {seed}")

os.environ["OMP_NUM_THREADS"] = "1"
sns.set_theme(style="whitegrid")


# ==========================================
# 1. 条件VAE模型 (状态条件解码器)
# ==========================================
class ConditionalVAE(nn.Module):
    """
    条件VAE: p(y | z, x_state)
    - 编码器: q_φ(z | y)
    - 解码器: p_θ(y | z, x_state) [关键: 以状态为条件]
    """
    def __init__(self, input_dim, latent_dim, n_states, hidden_dim=512):
        super(ConditionalVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_states = n_states

        # 编码器: y -> (mu, logvar) [4层]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器: FiLM条件解码 (z, x_state) -> y
        # 独立线性层 (输入不含 n_states，状态通过 FiLM 调制注入)
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

        # FiLM生成器: state one-hot -> (gamma, beta) 对每层
        self.film1 = nn.Linear(n_states, hidden_dim * 2)
        self.film2 = nn.Linear(n_states, hidden_dim * 2)
        self.film3 = nn.Linear(n_states, hidden_dim * 2)

        # State-Conditional Prior: p(z|x=k) = N(μ_k, σ²_k)
        # 每个状态有独立的先验中心,KL项会把不同状态的z推向不同区域
        # 初始化要小 (0.5): 太大会让KL压倒重建,导致z坍缩
        # 模型训练中会自动学习合适的分离度
        init_prior_mu = torch.zeros(n_states, latent_dim)
        for k in range(n_states):
            init_prior_mu[k, k % latent_dim] = 0.5
        self.prior_mu = nn.Parameter(init_prior_mu)
        self.prior_logvar = nn.Parameter(torch.zeros(n_states, latent_dim))
    
    def encode(self, y):
        """
        编码器: q_φ(z | y)
        y: (batch, seq_len, input_dim)
        返回: mu, logvar (batch, seq_len, latent_dim)
        """
        h = self.encoder(y)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_divergence(self, mu, logvar, state_onehot):
        """
        State-Conditional KL: KL(q(z|y) || p(z|x))

        不同状态有不同的先验中心,KL项会把z推向对应状态的先验区域,
        从而在潜空间中形成按状态分离的聚类。

        mu, logvar: (..., latent_dim)
        state_onehot: (..., n_states) — 可以是hard one-hot或soft概率
        返回: scalar (mean KL)
        """
        # 根据状态加权得到当前样本的先验参数
        # state_onehot @ prior_mu: (..., n_states) x (n_states, latent_dim) → (..., latent_dim)
        p_mu = torch.matmul(state_onehot, self.prior_mu)
        p_logvar = torch.matmul(state_onehot, self.prior_logvar)

        # KL(N(mu, sigma²) || N(p_mu, p_sigma²))
        kl = 0.5 * (p_logvar - logvar
                     + (logvar.exp() + (mu - p_mu) ** 2) / (p_logvar.exp() + 1e-8)
                     - 1)
        return kl.sum(dim=-1).mean()

    def _film_layer(self, h, film_gen, state_onehot):
        """FiLM调制: h' = gamma * h + beta"""
        film_params = film_gen(state_onehot)  # (..., hidden_dim * 2)
        gamma, beta = film_params.chunk(2, dim=-1)  # 各 (..., hidden_dim)
        gamma = gamma + 1.0  # 中心化在1附近，避免初始化时缩放过大
        return gamma * h + beta

    def decode(self, z, state_onehot):
        """
        FiLM条件解码器: p_θ(y | z, x_state)
        z: (batch, seq_len, latent_dim) 或 (batch, latent_dim)
        state_onehot: (batch, seq_len, n_states) 或 (batch, n_states)
        """
        h = self.dec_fc1(z)
        h = self._film_layer(h, self.film1, state_onehot)
        h = F.relu(h)

        h = self.dec_fc2(h)
        h = self._film_layer(h, self.film2, state_onehot)
        h = F.relu(h)

        h = self.dec_fc3(h)
        h = self._film_layer(h, self.film3, state_onehot)
        h = F.relu(h)

        return self.dec_out(h)
    
    def forward(self, y, state_onehot=None):
        """
        完整前向传播
        如果state_onehot为None,使用均匀软状态(预训练模式),
        FiLM解码器下均匀分布不引入噪声,保护重建质量
        """
        mu, logvar = self.encode(y)
        z = self.reparameterize(mu, logvar)

        if state_onehot is None:
            # 使用均匀软状态,避免FiLM随机调制破坏重建质量
            # 均匀分布让所有样本获得相同的gamma/beta,无噪声
            uniform_val = 1.0 / self.n_states
            if y.dim() == 3:
                batch, seq_len = y.shape[0], y.shape[1]
                state_onehot = torch.full((batch, seq_len, self.n_states),
                                          uniform_val, device=y.device)
            else:
                batch = y.shape[0]
                state_onehot = torch.full((batch, self.n_states),
                                          uniform_val, device=y.device)

        recon = self.decode(z, state_onehot)
        return recon, mu, logvar, z


# ==========================================
# 2. HMM模型 (Forward-Backward采样)
# ==========================================
class HMM_ForwardBackward(nn.Module):
    """
    HMM with Forward-Backward Sampling
    
    关键方法:
    1. forward_algorithm(): 计算α(forward variables)
    2. backward_sampling(): 从p(X|Z,Y)采样状态序列
    3. compute_emission_logprob(): 计算log p(y,z|x) = log p(y|z,x) + log p(z)
    """
    def __init__(self, n_states, latent_dim, vae):
        super(HMM_ForwardBackward, self).__init__()
        self.n_states = n_states
        self.latent_dim = latent_dim
        self.vae = vae
        
        # HMM参数
        self.start_logits = nn.Parameter(torch.zeros(n_states))
        init_logits = torch.randn(n_states, n_states) * 0.1
        init_logits += torch.eye(n_states) * 2.0  # softmax后对角≈0.7
        self.trans_logits = nn.Parameter(init_logits)
    
    def get_transition_matrix(self):
        """获取转移矩阵 A"""
        return F.softmax(self.trans_logits, dim=1)
    
    def get_start_prob(self):
        """获取初始分布 Π"""
        return F.softmax(self.start_logits, dim=0)
    
    def compute_emission_logprob(self, y, z):
        """
        计算发射概率: log p(y, z | x_k)

        根据论文:
        p(y_i, z_i | x_i = s_k) = p(y_i | z_i, x_i = s_k) · p(z_i | x_i = s_k)

        使用 State-Conditional Prior: p(z|x=k) = N(μ_k, σ²_k)
        不同状态有不同的先验,z靠近哪个状态中心就更可能属于该状态。

        y: (batch, input_dim)
        z: (batch, latent_dim)
        返回: (batch, n_states)
        """
        batch = y.shape[0]
        log_probs = torch.zeros(batch, self.n_states).to(y.device)

        for k in range(self.n_states):
            # 创建状态one-hot编码
            state_onehot = torch.zeros(batch, self.n_states).to(y.device)
            state_onehot[:, k] = 1.0

            # 1. 计算 log p(y | z, x=k) 通过VAE解码器
            with torch.no_grad():
                y_recon = self.vae.decode(z, state_onehot)

            log_p_y_given_z = -0.5 * ((y - y_recon) ** 2).sum(dim=-1)

            # 2. 计算 log p(z | x=k) = log N(z; μ_k, σ²_k) — State-Conditional Prior
            prior_mu_k = self.vae.prior_mu[k]           # (latent_dim,)
            prior_logvar_k = self.vae.prior_logvar[k]    # (latent_dim,)
            log_p_z = -0.5 * (
                prior_logvar_k.sum()
                + ((z - prior_mu_k.unsqueeze(0)) ** 2 / (prior_logvar_k.exp().unsqueeze(0) + 1e-8)).sum(dim=-1)
                + self.latent_dim * np.log(2 * np.pi)
            )

            # 3. 组合
            log_probs[:, k] = log_p_y_given_z + log_p_z

        return log_probs
    
    def forward_algorithm(self, y_seq, z_seq):
        """
        Forward算法: 计算α变量
        
        α_k(i) = p(y_{1:i}, z_{1:i}, x_i = s_k | θ, Π, A)
        
        初始化: α_k(1) = π_k · b_k(y_1, z_1)
        递推: α_k(i) = b_k(y_i, z_i) · Σ_j α_j(i-1) · a_{jk}
        
        y_seq: (batch, seq_len, input_dim)
        z_seq: (batch, seq_len, latent_dim)
        返回: alpha (batch, seq_len, n_states) - log空间
        """
        batch, seq_len, _ = y_seq.shape
        alpha = torch.zeros(batch, seq_len, self.n_states).to(y_seq.device)
        
        # 初始化
        log_start = F.log_softmax(self.start_logits, dim=0)  # (n_states,)
        emission_0 = self.compute_emission_logprob(y_seq[:, 0], z_seq[:, 0])  # (batch, n_states)
        alpha[:, 0, :] = log_start.unsqueeze(0) + emission_0
        
        # 递推
        log_trans = F.log_softmax(self.trans_logits, dim=1)  # (n_states, n_states)
        
        for t in range(1, seq_len):
            emission_t = self.compute_emission_logprob(y_seq[:, t], z_seq[:, t])
            
            for k in range(self.n_states):
                # α_k(t) = b_k(y_t, z_t) · Σ_j α_j(t-1) · a_{jk}
                # log空间: log α_k(t) = log b_k + logsumexp(log α_j(t-1) + log a_{jk})
                trans_score = alpha[:, t-1, :] + log_trans[:, k].unsqueeze(0)  # (batch, n_states)
                alpha[:, t, k] = torch.logsumexp(trans_score, dim=1) + emission_t[:, k]
        
        return alpha
    
    def backward_sampling(self, alpha, temperature=1.0):
        """
        Backward采样: 从p(X | Z, Y, θ)采样状态序列
        
        1. 采样 x_n ~ p(x_n | Z, Y) ∝ α_k(n)
        2. 向前采样 x_i ~ p(x_i | x_{i+1}, Z, Y) ∝ α_k(i) · a_{k, x_{i+1}}
        
        alpha: (batch, seq_len, n_states)
        temperature: Gumbel-Softmax温度
        返回: (batch, seq_len, n_states) - one-hot或soft samples
        """
        batch, seq_len, n_states = alpha.shape
        sampled_states = torch.zeros(batch, seq_len, n_states).to(alpha.device)
        
        # 1. 采样最后一个状态: p(x_n) ∝ α(n)
        logits_n = alpha[:, -1, :] / temperature
        sampled_states[:, -1, :] = F.gumbel_softmax(logits_n, tau=temperature, hard=True)
        
        # 2. 向前采样
        log_trans = F.log_softmax(self.trans_logits, dim=1)
        
        for t in range(seq_len - 2, -1, -1):
            # 对于每个batch,计算 p(x_t | x_{t+1})
            # p(x_t = k | x_{t+1} = j) ∝ α_k(t) · a_{kj}
            
            # 获取下一时刻的状态 (one-hot)
            next_state_onehot = sampled_states[:, t+1, :]  # (batch, n_states)
            
            # 计算每个可能的当前状态的log概率
            # alpha[:, t, :]: (batch, n_states)
            # log_trans: (n_states, n_states) - a_{kj}
            
            # 对于每个样本,计算 α_k(t) + Σ_j log(a_{kj}) * next_state_j
            logits_t = alpha[:, t, :].clone()  # (batch, n_states)
            
            for b in range(batch):
                # 找到下一个状态 (argmax of one-hot)
                next_state_idx = torch.argmax(next_state_onehot[b]).item()
                # 加上转移概率
                logits_t[b, :] += log_trans[:, next_state_idx]
            
            # Gumbel-Softmax采样
            sampled_states[:, t, :] = F.gumbel_softmax(logits_t / temperature, tau=temperature, hard=True)
        
        return sampled_states
    
    def viterbi(self, y_seq, z_seq):
        """
        Viterbi算法: 用于测试时找最优路径
        返回: (batch, seq_len) - 状态索引
        """
        with torch.no_grad():
            batch, seq_len, _ = y_seq.shape
            
            # DP table
            dp = torch.zeros(batch, seq_len, self.n_states).to(y_seq.device)
            pointers = torch.zeros(batch, seq_len, self.n_states, dtype=torch.long).to(y_seq.device)
            
            # 初始化
            log_start = F.log_softmax(self.start_logits, dim=0)
            emission_0 = self.compute_emission_logprob(y_seq[:, 0], z_seq[:, 0])
            dp[:, 0, :] = log_start.unsqueeze(0) + emission_0
            
            # 递推
            log_trans = F.log_softmax(self.trans_logits, dim=1)
            
            for t in range(1, seq_len):
                emission_t = self.compute_emission_logprob(y_seq[:, t], z_seq[:, t])
                
                for k in range(self.n_states):
                    scores = dp[:, t-1, :] + log_trans[:, k].unsqueeze(0)
                    max_scores, best_prev = torch.max(scores, dim=1)
                    dp[:, t, k] = max_scores + emission_t[:, k]
                    pointers[:, t, k] = best_prev
            
            # 回溯
            best_paths = []
            for b in range(batch):
                path = []
                curr = torch.argmax(dp[b, -1, :]).item()
                path.append(curr)
                for t in range(seq_len - 1, 0, -1):
                    curr = pointers[b, t, curr].item()
                    path.append(curr)
                best_paths.append(path[::-1])
            
            return torch.tensor(best_paths, dtype=torch.long).to(y_seq.device)


# ==========================================
# 3. EM训练器
# ==========================================
class EM_Trainer:
    """
    Partially Variational EM训练器
    
    E-step: 采样隐状态 X ~ p(X | Z, Y, θ, Π, A)
    M-step: 更新参数 θ, φ, Π, A
    """
    def __init__(self, vae, hmm, device):
        self.vae = vae
        self.hmm = hmm
        self.device = device
        
        # 优化器
        self.optimizer_vae = optim.Adam(vae.parameters(), lr=1e-3)
        self.optimizer_hmm = optim.Adam(hmm.parameters(), lr=1e-2)
    
    def em_step(self, y_batch, temperature=1.0, warmup=False, kl_weight=0.1,
                freeze_hmm=False):
        """
        一次EM迭代

        y_batch: (batch, seq_len, input_dim)
        warmup: 若为True,使用更大recon权重、更小KL权重
        kl_weight: KL散度权重(可由外部KL annealing控制)
        freeze_hmm: 若为True,只更新VAE(冻结HMM)
        返回: loss_dict
        """
        batch, seq_len, input_dim = y_batch.shape

        # warm-up阶段调整权重
        if warmup:
            effective_kl_weight = kl_weight * 0.1  # 更小KL
        else:
            effective_kl_weight = kl_weight

        # ================== E-step ==================
        # 1. 编码: 获取 z ~ q_φ(z | y)
        with torch.no_grad():
            mu, logvar = self.vae.encode(y_batch)
            z = self.vae.reparameterize(mu, logvar)

        # 2. Forward算法: 计算α
        with torch.no_grad():
            alpha = self.hmm.forward_algorithm(y_batch, z)

        # 3. Backward采样: 采样 X ~ p(X | Z, Y)
        with torch.no_grad():
            sampled_states = self.hmm.backward_sampling(alpha, temperature)  # (batch, seq_len, n_states)

        # ================== M-step ==================
        # 1. 更新VAE参数
        self.optimizer_vae.zero_grad()

        # 重新编码(带梯度)
        mu, logvar = self.vae.encode(y_batch)
        z = self.vae.reparameterize(mu, logvar)

        # 条件解码
        y_recon = self.vae.decode(z, sampled_states)

        # VAE损失 (State-Conditional KL)
        recon_loss = F.mse_loss(y_recon, y_batch)
        kld_loss = self.vae.kl_divergence(mu, logvar, sampled_states)
        vae_loss = recon_loss + effective_kl_weight * kld_loss

        vae_loss.backward()
        self.optimizer_vae.step()

        # 2. 更新HMM参数 (freeze_hmm时跳过)
        hmm_loss_val = 0.0
        if not freeze_hmm:
            self.optimizer_hmm.zero_grad()

            # 重新计算z(带梯度)
            with torch.no_grad():
                mu, logvar = self.vae.encode(y_batch)
            z = self.vae.reparameterize(mu, logvar)

            # 计算HMM负对数似然
            alpha = self.hmm.forward_algorithm(y_batch, z)
            log_likelihood = torch.logsumexp(alpha[:, -1, :], dim=1)
            hmm_loss = -log_likelihood.mean()

            hmm_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.hmm.parameters(), 1.0)
            self.optimizer_hmm.step()
            hmm_loss_val = hmm_loss.item()

        return {
            'vae_recon': recon_loss.item(),
            'vae_kld': kld_loss.item(),
            'hmm_nll': hmm_loss_val,
            'total': vae_loss.item() + hmm_loss_val
        }


# ==========================================
# 4. 数据加载和预处理
# ==========================================
def prepare_stock_data(returns_data, seq_len=30, train_ratio=0.8):
    """
    准备股票收益率数据
    
    returns_data: (n_days, n_stocks) - 日收益率
    seq_len: 序列长度
    返回: train_loader, test_data, scaler
    """
    N, D = returns_data.shape
    
    # 标准化
    scaler = StandardScaler()
    returns_flat = returns_data.flatten().reshape(-1, 1)
    returns_flat = np.clip(returns_flat, -10, 10)
    scaled_returns = scaler.fit_transform(returns_flat).reshape(N, D)
    
    # 构建序列
    X = []
    for i in range(len(scaled_returns) - seq_len):
        X.append(scaled_returns[i : i + seq_len])
    X = np.array(X)
    
    # 划分训练/测试
    split_idx = int(len(X) * train_ratio)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    
    return X_train, X_test, scaler


# ==========================================
# 5. 可视化工具
# ==========================================
class Visualizer:
    @staticmethod
    def plot_training_curves(history, filename="training_curves.png"):
        """绘制训练曲线"""
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # VAE重建损失
        axes[0].plot(history['vae_recon'], label='Reconstruction Loss')
        axes[0].set_title('VAE Reconstruction Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # KL散度
        axes[1].plot(history['vae_kld'], label='KL Divergence', color='green')
        axes[1].set_title('VAE KL Divergence')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # HMM负对数似然
        axes[2].plot(history['hmm_nll'], label='HMM NLL', color='red')
        axes[2].set_title('HMM Negative Log-Likelihood')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved training curves to {filename}")
    
    @staticmethod
    def plot_state_transitions(trans_matrix, filename="transition_matrix.png"):
        """可视化状态转移矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(trans_matrix, annot=True, fmt='.3f', cmap='YlOrRd', 
                    xticklabels=[f'State {i}' for i in range(len(trans_matrix))],
                    yticklabels=[f'State {i}' for i in range(len(trans_matrix))])
        plt.title('HMM State Transition Matrix')
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved transition matrix to {filename}")
    
    @staticmethod
    def plot_latent_space(z_data, states, filename="latent_space.png"):
        """可视化潜空间"""
        if z_data.shape[1] > 2:
            pca = PCA(n_components=2)
            z_vis = pca.fit_transform(z_data)
        else:
            z_vis = z_data
        
        plt.figure(figsize=(10, 6))
        for state in np.unique(states):
            mask = states == state
            plt.scatter(z_vis[mask, 0], z_vis[mask, 1], 
                       label=f'State {state}', alpha=0.6, s=20)
        plt.title('Latent Space Visualization (PCA)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved latent space to {filename}")
    
    @staticmethod
    def plot_portfolio_performance(nav, benchmark, states, filename="portfolio_performance.png"):
        """绘制投资组合表现"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 净值曲线
        axes[0].plot(nav, label='Strategy NAV', linewidth=2)
        axes[0].plot(benchmark, label='Benchmark NAV', linewidth=2, alpha=0.7)
        axes[0].set_title('Portfolio Net Asset Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 状态时序
        axes[1].plot(states, drawstyle='steps-post', color='purple', linewidth=1.5)
        axes[1].set_title('Predicted Market States')
        axes[1].set_ylabel('State')
        axes[1].set_xlabel('Time')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved portfolio performance to {filename}")


# ==========================================
# 6. 主训练函数
# ==========================================
def train_hmm_vae(returns_data, config, device):
    """
    主训练函数
    
    config: {
        'seq_len': 序列长度,
        'latent_dim': 潜变量维度,
        'n_states': HMM状态数,
        'batch_size': 批大小,
        'n_epochs': 训练轮数,
        'temperature_start': Gumbel-Softmax初始温度,
        'temperature_end': Gumbel-Softmax最终温度
    }
    """
    # 准备数据
    X_train, X_test, scaler = prepare_stock_data(
        returns_data, 
        seq_len=config['seq_len']
    )
    
    n_stocks = X_train.shape[2]
    
    # 转换为Tensor
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    
    train_loader = DataLoader(
        TensorDataset(X_train_tensor), 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    # 初始化模型
    print(">>> Initializing models...")
    vae = ConditionalVAE(
        input_dim=n_stocks,
        latent_dim=config['latent_dim'],
        n_states=config['n_states']
    ).to(device)
    
    hmm = HMM_ForwardBackward(
        n_states=config['n_states'],
        latent_dim=config['latent_dim'],
        vae=vae
    ).to(device)
    
    # EM训练器
    trainer = EM_Trainer(vae, hmm, device)
    
    # 训练历史
    history = {
        'vae_recon': [],
        'vae_kld': [],
        'hmm_nll': [],
        'total': []
    }
    
    # 温度退火
    temperature_schedule = np.linspace(
        config['temperature_start'],
        config['temperature_end'],
        config['n_epochs']
    )
    
    print(">>> Starting EM training...")
    for epoch in range(config['n_epochs']):
        epoch_losses = {'vae_recon': 0, 'vae_kld': 0, 'hmm_nll': 0, 'total': 0}
        
        for batch_idx, (batch_y,) in enumerate(train_loader):
            losses = trainer.em_step(batch_y, temperature=temperature_schedule[epoch])
            
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        # 记录平均损失
        for key in epoch_losses:
            avg_loss = epoch_losses[key] / len(train_loader)
            history[key].append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['n_epochs']} | "
                  f"Recon: {history['vae_recon'][-1]:.4f} | "
                  f"KLD: {history['vae_kld'][-1]:.4f} | "
                  f"HMM NLL: {history['hmm_nll'][-1]:.4f} | "
                  f"Temp: {temperature_schedule[epoch]:.3f}")
    
    print(">>> Training completed!")
    
    return vae, hmm, history, (X_train_tensor, X_test_tensor, scaler)


# ==========================================
# 7. 评估和预测
# ==========================================
def evaluate_and_predict(vae, hmm, X_test, original_returns, device, config):
    """
    评估模型并进行交易回测
    
    X_test: (n_samples, seq_len, n_stocks)
    original_returns: (n_days, n_stocks) - 原始收益率
    """
    vae.eval()
    hmm.eval()
    
    with torch.no_grad():
        # 编码测试数据
        mu, logvar = vae.encode(X_test)
        z = vae.reparameterize(mu, logvar)
        
        # Viterbi解码获取状态序列
        state_sequence = hmm.viterbi(X_test, z)  # (n_samples, seq_len)
    
    # 获取每个序列最后一个状态
    final_states = state_sequence[:, -1].cpu().numpy()
    
    # 对应的真实收益率
    test_start_idx = len(original_returns) - len(X_test) - config['seq_len']
    test_returns = original_returns[test_start_idx + config['seq_len']:].mean(axis=1)
    
    # 截断到相同长度
    min_len = min(len(final_states), len(test_returns))
    final_states = final_states[:min_len]
    test_returns = test_returns[:min_len]
    
    # 识别牛熊状态
    state_returns = {}
    for state in range(config['n_states']):
        mask = final_states == state
        if np.any(mask):
            state_returns[state] = np.mean(test_returns[mask])
        else:
            state_returns[state] = 0.0
    
    bull_state = max(state_returns, key=state_returns.get)
    
    print(f"\n>>> State Analysis:")
    for state, ret in state_returns.items():
        print(f"State {state}: Avg Return = {ret:.4f}")
    print(f"Identified Bull State: {bull_state}")
    
    # 交易策略: 只在牛市状态持有
    strategy_returns = np.where(final_states == bull_state, test_returns, 0)
    
    # 计算指标
    sharpe = (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9)) * np.sqrt(252)
    win_rate = np.mean(strategy_returns > 0)
    
    # 净值曲线
    nav_strategy = np.cumprod(1 + strategy_returns)
    nav_benchmark = np.cumprod(1 + test_returns)
    
    print(f"\n>>> Performance Metrics:")
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Final NAV (Strategy): {nav_strategy[-1]:.4f}")
    print(f"Final NAV (Benchmark): {nav_benchmark[-1]:.4f}")
    
    return {
        'states': final_states,
        'returns': test_returns,
        'nav_strategy': nav_strategy,
        'nav_benchmark': nav_benchmark,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'z_latent': z[:, -1, :].cpu().numpy()[:min_len]
    }


# ==========================================
# 8. 完整pipeline
# ==========================================
def main():
    # 设置
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 配置
    config = {
        'seq_len': 30,
        'latent_dim': 8,
        'n_states': 3,  # 尝试3个状态: 牛市/熊市/震荡
        'batch_size': 64,
        'n_epochs': 100,
        'temperature_start': 5.0,
        'temperature_end': 0.5
    }
    
    # 加载数据 (示例: 需要替换为真实的CSI500数据)
    print(">>> Loading data...")
    # 这里需要您的download_csi500_data函数
    # df = download_csi500_data(start_date="20160101", end_date="20231231")
    # returns_data = df.values.astype(np.float32)
    
    # 示例数据(随机生成,用于演示)
    np.random.seed(42)
    n_days = 1000
    n_stocks = 500
    returns_data = np.random.randn(n_days, n_stocks) * 0.02
    
    # 训练
    vae, hmm, history, (X_train, X_test, scaler) = train_hmm_vae(
        returns_data, config, device
    )
    
    # 可视化训练过程
    print("\n>>> Generating visualizations...")
    vis = Visualizer()
    vis.plot_training_curves(history)
    
    # 转移矩阵
    trans_matrix = hmm.get_transition_matrix().detach().cpu().numpy()
    vis.plot_state_transitions(trans_matrix)
    
    # 评估
    results = evaluate_and_predict(vae, hmm, X_test, returns_data, device, config)
    
    # 可视化结果
    vis.plot_latent_space(results['z_latent'], results['states'])
    vis.plot_portfolio_performance(
        results['nav_strategy'],
        results['nav_benchmark'],
        results['states']
    )
    
    print("\n>>> All done!")
    return vae, hmm, results


if __name__ == "__main__":
    vae, hmm, results = main()
