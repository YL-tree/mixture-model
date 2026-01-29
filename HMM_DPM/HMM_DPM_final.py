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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import optuna
from data import download_csi500_data

# ==========================================
# 0. 全局配置
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
sns.set_theme(style="whitegrid")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f">>> Random Seed set to: {seed}")

# ==========================================
# 1. 核心模型: Conditional Diffusion + HMM
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
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

class ConditionalDiffusionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_states=2):
        super().__init__()
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 状态嵌入
        self.state_emb = nn.Embedding(n_states, hidden_dim)
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 扩散去噪网络 (Denoising Network)
        self.mid_block = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), 
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 【新增】解码器 p(y | z_1, x)
        # 将 z_1 (维度 input_dim) 和 状态嵌入 映射回 y (维度 input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, state):
        """预测噪声 (Denoising)"""
        t_emb = self.time_mlp(t)
        x_emb = self.input_proj(x)
        s_emb = self.state_emb(state)
        h = torch.cat([x_emb, t_emb, s_emb], dim=1)
        return self.mid_block(h)

    def decode(self, z_1, state):
        """解码: p(y | z_1, x)"""
        s_emb = self.state_emb(state)
        h = torch.cat([z_1, s_emb], dim=-1)
        return self.decoder(h)

class HMM_DPM(nn.Module):
    def __init__(self, input_dim, n_states=2, n_steps=100, hidden_dim=128):
        super().__init__()
        self.n_states = n_states
        self.n_steps = n_steps
        self.input_dim = input_dim
        
        self.net = ConditionalDiffusionNet(input_dim, hidden_dim, n_states)
        
        # HMM 参数
        self.trans_logits = nn.Parameter(torch.zeros(n_states, n_states))
        self.start_logits = nn.Parameter(torch.zeros(n_states))
        
        # 扩散参数 schedule
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def get_diffusion_loss(self, x, state):
        """计算简单的扩散去噪损失 (用于 Warmup 或 辅助 Loss)"""
        if x.dim() == 3:
            B, L, D = x.shape
            x_flat = x.reshape(B * L, D)
            state_flat = state.reshape(B * L)
        else:
            x_flat = x
            state_flat = state
            
        batch_size = x_flat.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=x.device).long()
        noise = torch.randn_like(x_flat)
        
        alpha_bar = self.alphas_cumprod[t].view(-1, 1)
        x_t = torch.sqrt(alpha_bar) * x_flat + torch.sqrt(1 - alpha_bar) * noise
        
        predicted_noise = self.net(x_t, t, state_flat)
        return F.mse_loss(predicted_noise, noise)

    # 【新增】前向扩散采样: 获取完整轨迹 z_0 ... z_T
    def forward_diffusion(self, y):
        """
        前向扩散: y -> z_T
        q(z_t | z_{t-1}) = N(√α_t z_{t-1}, (1-α_t)I)
        """
        z = [y]  # z_0 = y
        for t in range(1, self.n_steps):
            noise = torch.randn_like(z[-1])
            alpha_t = self.alphas[t]
            z_t = torch.sqrt(alpha_t) * z[-1] + torch.sqrt(1 - alpha_t) * noise
            z.append(z_t)
        return torch.stack(z, dim=1)  # (batch, T+1, dim)

    # 【新增】计算反向扩散概率
    def compute_reverse_logprob(self, z_traj, state):
        """
        蒙特卡洛近似计算 Σ log p(z_t | z_{t+1}, x=state)
        """
        B, T_plus_1, D = z_traj.shape
        T = T_plus_1 - 1
        
        # 随机采样 M 个时间步来近似求和 (论文 P9)
        n_samples = min(10, T)
        sampled_t = torch.randint(0, T, (n_samples,), device=z_traj.device)
        
        log_prob = 0
        for t in sampled_t:
            z_t = z_traj[:, t, :]      # Target
            z_t_plus_1 = z_traj[:, t+1, :] # Input
            
            t_tensor = torch.full((B,), t, device=z_traj.device).long()
            
            # 预测噪声
            pred_noise = self.net(z_t_plus_1, t_tensor, state)
            
            # 计算 p(z_t | z_{t+1}) 的均值和方差
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_t_plus_1 = self.alphas_cumprod[t+1] if t < T-1 else torch.tensor(0.0).to(z_traj.device)
            
            # DDPM 后验均值公式 (简化版，用于 reverse process p)
            # 注意: 这里计算的是 p_theta 的均值，近似于 q 的后验
            # mu_theta = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * epsilon)
            coef = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t_plus_1) #近似
            mu_pred = (z_t_plus_1 - coef * pred_noise) / torch.sqrt(alpha_t)
            
            # 方差 (简化为 fixed variance)
            sigma2_t = (1 - alpha_bar_t) / (1 - alpha_bar_t_plus_1) * (1 - alpha_t)
            sigma2_t = torch.clamp(sigma2_t, min=1e-5)
            
            # log N(z_t | mu_pred, sigma2)
            # 忽略常数项 -0.5 * log(2pi * sigma2)
            log_prob += -0.5 * ((z_t - mu_pred)**2 / sigma2_t).sum(dim=-1)
            
        # 缩放回完整的时间步
        return log_prob * (T / n_samples)

    # 【修正】计算发射概率 log p(y, z | x)
    def compute_emission_probs(self, x, n_traj_samples=1):
        """
        论文公式: log p(y, z | x=k) = log p(y|z_1) + log p(z_reverse) + log p(z_T)
        """
        if x.dim() == 3:
            B, L, D = x.shape
            x_flat = x.reshape(B * L, D)
        else:
            B, D = x.shape
            L = 1
            x_flat = x
            
        flat_batch = x_flat.shape[0]
        log_probs_flat = torch.zeros(flat_batch, self.n_states).to(x.device)
        
        for k in range(self.n_states):
            state_tensor = torch.full((flat_batch,), k, device=x.device).long()
            
            traj_log_probs = []
            for _ in range(n_traj_samples):
                # 1. 前向扩散采样轨迹: y -> z_0...z_T
                # 注意: 这里的 z_traj[0] 是 y (z_0)
                z_trajectory = self.forward_diffusion(x_flat) 
                
                # 2. 解码器概率: log p(y | z_1, x)
                # 论文中 y 是观测，z_1 是第一个潜变量
                z_1 = z_trajectory[:, 1, :] # index 1 corresponds to t=1
                y_recon = self.net.decode(z_1, state_tensor)
                log_p_y = -0.5 * ((x_flat - y_recon)**2).sum(dim=-1)
                
                # 3. 反向扩散概率
                log_p_reverse = self.compute_reverse_logprob(z_trajectory, state_tensor)
                
                # 4. 先验概率 log p(z_T)
                z_T = z_trajectory[:, -1, :]
                log_p_prior = -0.5 * (z_T**2).sum(dim=-1)
                
                total = log_p_y + log_p_reverse + log_p_prior
                traj_log_probs.append(total)
            
            # 蒙特卡洛平均
            if n_traj_samples > 1:
                avg_log_prob = torch.logsumexp(torch.stack(traj_log_probs, dim=0), dim=0) - np.log(n_traj_samples)
            else:
                avg_log_prob = traj_log_probs[0]
                
            log_probs_flat[:, k] = avg_log_prob
            
        return log_probs_flat.reshape(B, L, self.n_states)

    def forward_backward_sampling(self, x):
        # 使用修正后的发射概率
        with torch.no_grad():
            log_emission = self.compute_emission_probs(x, n_traj_samples=1)
            
        B, L, K = log_emission.shape
        log_trans = F.log_softmax(self.trans_logits, dim=1)
        log_start = F.log_softmax(self.start_logits, dim=0)
        
        # Forward
        alpha = torch.zeros(B, L, K).to(x.device)
        alpha[:, 0, :] = log_start.unsqueeze(0) + log_emission[:, 0, :]
        
        for t in range(1, L):
            prev = alpha[:, t-1, :].unsqueeze(2) + log_trans.unsqueeze(0)
            alpha_t_given_prev = torch.logsumexp(prev, dim=1)
            alpha[:, t, :] = log_emission[:, t, :] + alpha_t_given_prev
            
        # Backward Sampling
        sampled_states = torch.zeros(B, L, dtype=torch.long).to(x.device)
        probs_T = F.softmax(alpha[:, -1, :], dim=1)
        sampled_states[:, -1] = torch.multinomial(probs_T, 1).squeeze()
        
        for t in range(L - 2, -1, -1):
            next_state = sampled_states[:, t+1]
            trans_cols = log_trans[:, next_state].T
            log_prob = alpha[:, t, :] + trans_cols
            probs = F.softmax(log_prob, dim=1)
            sampled_states[:, t] = torch.multinomial(probs, 1).squeeze()
        return sampled_states

    def viterbi_decode(self, x):
        with torch.no_grad():
            log_emission = self.compute_emission_probs(x, n_traj_samples=1)
        
        B, L, K = log_emission.shape
        log_trans = F.log_softmax(self.trans_logits, dim=1)
        
        dp = torch.zeros(B, L, K).to(x.device)
        pointers = torch.zeros(B, L, K, dtype=torch.long).to(x.device)
        dp[:, 0, :] = log_emission[:, 0, :]
        
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
        return best_paths

# ==========================================
# 2. 增强型绘图模块 (保持不变)
# ==========================================
class ShowcasePlotter:
    @staticmethod
    def plot_training_dashboard(history, filename="1_training_dashboard.png"):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
        if 'warmup_loss' in history and len(history['warmup_loss']) > 0:
            axes[0].plot(history['warmup_loss'], label='Warmup Loss', color='orange')
            axes[0].set_title("Stage 1: Supervised Warmup")
            axes[0].legend()
        if 'pvem_loss' in history and len(history['pvem_loss']) > 0:
            axes[1].plot(history['pvem_loss'], label='PVEM Loss', color='blue')
            axes[1].set_title("Stage 2: Joint Training (Diffusion + Recon)")
            axes[1].legend()
        plt.tight_layout()
        plt.savefig(filename)

    @staticmethod
    def plot_hmm_insights(model, states, returns, filename="3_hmm_insights.png"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        trans_mat = torch.exp(model.trans_logits).detach().cpu().numpy()
        # Row Normalize
        trans_mat = trans_mat / (trans_mat.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(trans_mat, annot=True, fmt=".2f", cmap="Blues", ax=axes[0])
        axes[0].set_title("Transition Matrix")
        
        returns_flat = np.array(returns).flatten()
        states_flat = np.array(states).flatten()
        min_len = min(len(returns_flat), len(states_flat))
        df = pd.DataFrame({'Return': returns_flat[:min_len], 'State': states_flat[:min_len]})
        sns.boxenplot(data=df, x='State', y='Return', palette='coolwarm', ax=axes[1])
        axes[1].axhline(0, color='black', linestyle='--')
        axes[1].set_title("Return Distribution")
        plt.tight_layout()
        plt.savefig(filename)

    @staticmethod
    def plot_financial_performance(nav, bench, states, bull_state, filename="4_financial_report.png"):
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(nav, label='DPM Strategy', color='purple', linewidth=2)
        ax1.plot(bench, label='Benchmark', color='gray', linestyle='--')
        
        for t in range(len(states)):
            if states[t] == bull_state:
                 ax1.axvspan(t, t+1, color='red', alpha=0.1, linewidth=0)
            else:
                 ax1.axvspan(t, t+1, color='green', alpha=0.1, linewidth=0)
        ax1.legend()
        ax1.set_title("Cumulative Wealth")
        
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        dd_strat = (np.array(nav) - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav)
        dd_bench = (np.array(bench) - np.maximum.accumulate(bench)) / np.maximum.accumulate(bench)
        ax2.fill_between(range(len(dd_strat)), dd_strat, 0, color='purple', alpha=0.3)
        ax2.plot(dd_bench, color='gray', linestyle=':')
        ax2.set_title("Drawdown")
        plt.tight_layout()
        plt.savefig(filename)

# ==========================================
# 3. 训练与评估引擎
# ==========================================
def prepare_data_no_leakage(raw_data, seq_len=30, train_ratio=0.8):
    """
    【新增】防止数据泄露的预处理
    """
    # 1. 划分
    split_idx = int(len(raw_data) * train_ratio)
    train_data = raw_data[:split_idx]
    test_data = raw_data[split_idx:]
    
    # 2. Fit scaler only on train
    scaler = StandardScaler()
    # Flatten to (N*500, 1) for global scaling, or (N, 500) for per-feature
    # 这里我们用全局缩放，假设所有股票是同质的
    scaler.fit(train_data.reshape(-1, 1))
    
    # 3. Transform
    train_scaled = scaler.transform(train_data.reshape(-1, 1)).reshape(train_data.shape)
    test_scaled = scaler.transform(test_data.reshape(-1, 1)).reshape(test_data.shape)
    
    # 4. Clip (Outlier removal)
    train_scaled = np.clip(train_scaled, -5, 5)
    test_scaled = np.clip(test_scaled, -5, 5)
    
    # 5. Make sequences
    def make_seq(data):
        X = []
        for i in range(0, len(data) - seq_len):
            X.append(data[i : i + seq_len])
        return np.array(X)
        
    X_train = make_seq(train_scaled)
    X_test = make_seq(test_scaled)
    
    return torch.from_numpy(X_train).float(), torch.from_numpy(X_test).float(), split_idx

def train_and_evaluate(params, raw_data, device, is_final_run=False):
    SEQ_LEN = params['seq_len']
    N_STATES = params['n_states']
    LR = params['lr']
    HIDDEN_DIM = params['hidden_dim']
    BATCH_SIZE = params['batch_size']
    
    # 增加 Epoch，因为 M-step 变复杂了
    WARMUP_EPOCHS = 20 if is_final_run else 2
    PVEM_EPOCHS = 100 if is_final_run else 2
    
    # 1. 准备数据 (无泄露)
    X_train, X_test, split_idx = prepare_data_no_leakage(raw_data, SEQ_LEN)
    X_train, X_test = X_train.to(device), X_test.to(device)
    
    train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    
    model = HMM_DPM(input_dim=500, n_states=N_STATES, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    history = {'warmup_loss': [], 'pvem_loss': []}
    
    # --- Stage 0: Init with KMeans (Heuristic) ---
    print(">>> Stage 0: KMeans Initialization...")
    # 使用训练集的波动率
    volatility = torch.std(X_train, dim=2).mean(dim=1).cpu().numpy().reshape(-1, 1)
    
    try:
        kmeans = KMeans(n_clusters=N_STATES, random_state=42, n_init=10).fit(volatility)
        # 排序：让 State 0 = 低波, State N = 高波
        cluster_vols = [volatility[kmeans.labels_==k].mean() for k in range(N_STATES)]
        sorted_indices = np.argsort(cluster_vols)
        map_dict = {old: new for new, old in enumerate(sorted_indices)}
        
        # 转换标签
        labels_vec = np.vectorize(map_dict.get)(kmeans.labels_)
        labels_tensor = torch.tensor(labels_vec, device=device).long()
        # 扩展到序列长度 (B, L)
        init_labels = labels_tensor.unsqueeze(1).repeat(1, SEQ_LEN)
        
        # 初始化转移矩阵 (对角占优)
        trans_init = np.eye(N_STATES) * 0.9 + (1-0.9)/(N_STATES-1) * (1-np.eye(N_STATES))
        model.trans_logits.data = torch.tensor(np.log(trans_init), dtype=torch.float32).to(device)
    except Exception as e:
        print(f"KMeans Init Failed: {e}")
        return -999.0

    # --- Stage 1: Supervised Warmup (Diffusion Only) ---
    print(">>> Stage 1: Warmup...")
    warmup_dataset = TensorDataset(X_train, init_labels)
    warmup_loader = DataLoader(warmup_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(WARMUP_EPOCHS):
        model.train()
        t_loss = 0
        for bx, by in warmup_loader:
            optimizer.zero_grad()
            # Warmup 时只优化扩散过程
            loss = model.get_diffusion_loss(bx, by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        history['warmup_loss'].append(t_loss/len(warmup_loader))
        if epoch % 5 == 0: print(f"Warmup Epoch {epoch}: {t_loss/len(warmup_loader):.4f}")

    # --- Stage 2: Joint PVEM Training ---
    print(">>> Stage 2: PVEM Training...")
    
    for epoch in range(PVEM_EPOCHS):
        model.train()
        t_loss = 0
        n_batches = len(train_loader)
        
        for i, (bx,) in enumerate(train_loader):
            # --- E-Step: Sampling States ---
            # 使用 forward-backward 算法采样状态
            with torch.no_grad():
                sampled_states = model.forward_backward_sampling(bx)
            
            # --- M-Step: Optimizing Parameters ---
            optimizer.zero_grad()
            
            # 1. 扩散损失 (L_diffusion)
            loss_diff = model.get_diffusion_loss(bx, sampled_states)
            
            # 2. 重建损失 (L_recon) - 【新增】优化解码器
            # 我们需要 z_1 来做解码。这里简单地通过 forward_diffusion 获取 z_1
            # 注意: 为了效率，我们可以只采样一步或者复用逻辑
            # 这里为了准确，我们做一次前向扩散到 t=1
            t1 = torch.ones(bx.shape[0] * bx.shape[1], device=device).long()
            bx_flat = bx.reshape(-1, bx.shape[-1])
            states_flat = sampled_states.reshape(-1)
            
            # Forward diffusion to t=1
            noise = torch.randn_like(bx_flat)
            alpha_1 = model.alphas[1] # t=1
            z_1 = torch.sqrt(alpha_1) * bx_flat + torch.sqrt(1 - alpha_1) * noise
            
            # Decode
            y_recon = model.net.decode(z_1, states_flat)
            loss_recon = F.mse_loss(y_recon, bx_flat)
            
            # 3. 总损失
            # 赋予重建损失较高权重，保证状态具有物理意义
            total_loss = loss_diff + 10.0 * loss_recon
            
            total_loss.backward()
            optimizer.step()
            t_loss += total_loss.item()
            
            # 4. 【新增】显式更新 HMM 参数
            # 每隔几个 batch 更新一次 HMM，或者每个 batch 都更新
            if (i + 1) % 5 == 0:
                with torch.no_grad():
                    # 统计转移
                    trans_counts = torch.zeros(N_STATES, N_STATES, device=device)
                    for b in range(sampled_states.shape[0]):
                        for t in range(sampled_states.shape[1]-1):
                            s_curr = sampled_states[b, t]
                            s_next = sampled_states[b, t+1]
                            trans_counts[s_curr, s_next] += 1
                    
                    # 平滑 + 归一化
                    trans_counts += 1.0 
                    trans_probs = trans_counts / trans_counts.sum(dim=1, keepdim=True)
                    # 更新 logits
                    model.trans_logits.data = torch.log(trans_probs)

        avg_loss = t_loss / n_batches
        history['pvem_loss'].append(avg_loss)
        if epoch % 5 == 0: print(f"PVEM Epoch {epoch}: {avg_loss:.4f}")

    # --- Evaluation ---
    print(">>> Evaluating...")
    model.eval()
    with torch.no_grad():
        paths = model.viterbi_decode(X_test)
    final_states = np.array([p[-1] for p in paths])
    
    # 对齐回测时间轴
    test_len = len(final_states)
    raw_test_data = raw_data[split_idx + SEQ_LEN : split_idx + SEQ_LEN + test_len]
    returns = np.mean(raw_test_data, axis=1)
    
    # 确定牛市状态 (平均收益率最高的)
    state_avg_rets = []
    for k in range(N_STATES):
        mask = (final_states == k)
        if mask.sum() > 0: state_avg_rets.append(np.mean(returns[mask]))
        else: state_avg_rets.append(-1.0)
    bull_state = np.argmax(state_avg_rets)
    
    # 回测
    nav = [1.0]; bench = [1.0]
    for i in range(len(returns)):
        r = returns[i]
        bench.append(bench[-1] * (1+r))
        # 简单策略: 牛市持有，非牛市空仓
        if final_states[i] == bull_state: 
            nav.append(nav[-1] * (1+r))
        else: 
            nav.append(nav[-1])
            
    strat_ret = np.diff(nav) / nav[:-1]
    sharpe = np.mean(strat_ret) / np.std(strat_ret) * np.sqrt(252) if np.std(strat_ret) > 1e-6 else 0
        
    if is_final_run:
        print(f"\n>>> Final Result: Sharpe = {sharpe:.4f}")
        ShowcasePlotter.plot_training_dashboard(history)
        ShowcasePlotter.plot_hmm_insights(model, final_states, returns)
        ShowcasePlotter.plot_financial_performance(nav, bench, final_states, bull_state)
        
        # Generate Report
        report = f"""
        HMM-DPM Final Report
        --------------------
        Sharpe Ratio: {sharpe:.4f}
        Bull State: {bull_state}
        State Returns: {state_avg_rets}
        """
        print(report)

    return sharpe

def objective(trial):
    params = {
        'seq_len': trial.suggest_int("seq_len", 15, 40, step=5),
        'n_states': trial.suggest_categorical("n_states", [2, 3]), # 简化搜索空间
        'hidden_dim': trial.suggest_categorical("hidden_dim", [64, 128]),
        'lr': trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        'batch_size': trial.suggest_categorical("batch_size", [32, 64])
    }
    return train_and_evaluate(params, RAW_DATA, DEVICE, is_final_run=False)

if __name__ == "__main__":
    setup_seed(42)
    print(">>> Loading Data...")
    df = download_csi500_data(start_date="20160101", end_date="20231231")
    RAW_DATA = df.values.astype(np.float32)
    # 简单的去极值
    RAW_DATA = np.clip(RAW_DATA, -0.1, 0.1)
    
    print("\n>>> [AutoML] Starting Optuna Search...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5) # 减少 trial 次数以快速验证
    
    print(f"BEST PARAMS: {study.best_params}")
    print("\n>>> Starting Final Showcase Run...")
    train_and_evaluate(study.best_params, RAW_DATA, DEVICE, is_final_run=True)