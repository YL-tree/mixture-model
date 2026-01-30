"""
HMM-DPM 完整实现 (AdaLN + T+1回测修复版)

核心修复:
1. 回测逻辑对齐 T+1 (消除未来函数)
2. 新增训练集校准 (Calibration) 步骤
3. AdaLN 强制状态控制
4. Epoch 级 EMA 更新
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
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
# 1. AdaLN + 扩散模型组件
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


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization
    用 conditioning vector 生成 scale 和 shift
    """
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)
        # 用小随机值初始化
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class ConditionalDiffusionNet(nn.Module):
    """带 AdaLN 的条件扩散网络"""
    def __init__(self, input_dim, hidden_dim=128, n_states=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 时间编码
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # 状态嵌入
        self.state_emb = nn.Embedding(n_states, hidden_dim)

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim * 2)

        # 网络层 + AdaLN
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.adaln1 = AdaLN(hidden_dim * 2, hidden_dim)

        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.adaln2 = AdaLN(hidden_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_t, t, state):
        t_emb = self.time_mlp(t)
        s_emb = self.state_emb(state)
        cond = t_emb + s_emb 

        h = self.input_proj(x_t)
        h = self.linear1(F.silu(h))
        h = self.adaln1(h, cond)
        h = self.linear2(F.silu(h))
        h = self.adaln2(h, cond)
        return self.output(F.silu(h))


# ==========================================
# 2. HMM-DPM 模型
# ==========================================
class HMM_DPM(nn.Module):
    def __init__(self, input_dim, n_states=3, n_steps=50, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.n_states = n_states
        self.n_steps = n_steps

        self.net = ConditionalDiffusionNet(input_dim, hidden_dim, n_states)

        # HMM 参数
        self.trans_logits = nn.Parameter(torch.zeros(n_states, n_states))
        self.start_logits = nn.Parameter(torch.zeros(n_states))

        # 扩散 schedule
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # 默认 emission scale
        self.emission_scale = 0.1

    def get_diffusion_loss(self, y, state):
        if y.dim() == 3:
            B, L, D = y.shape
            y_flat = y.reshape(B * L, D)
            state_flat = state.reshape(B * L)
        else:
            y_flat = y
            state_flat = state

        batch_size = y_flat.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=y.device).long()
        noise = torch.randn_like(y_flat)

        alpha_bar = self.alphas_cumprod[t].view(-1, 1)
        y_t = torch.sqrt(alpha_bar) * y_flat + torch.sqrt(1 - alpha_bar) * noise

        pred_noise = self.net(y_t, t, state_flat)
        return F.mse_loss(pred_noise, noise)

    def compute_emission_logprob(self, x, n_time_samples=8):
        """用 diffusion MSE 作为 emission probability"""
        if x.dim() == 3:
            B, L, D = x.shape
            x_flat = x.reshape(B * L, D)
        else:
            B, D = x.shape
            L = 1
            x_flat = x

        flat_batch = x_flat.shape[0]
        log_probs_flat = torch.zeros(flat_batch, self.n_states, device=x.device)

        # 共享采样
        t_samples = torch.randint(0, self.n_steps, (n_time_samples, flat_batch), device=x.device).long()
        noise_samples = torch.randn(n_time_samples, flat_batch, D, device=x.device)

        for k in range(self.n_states):
            state_tensor = torch.full((flat_batch,), k, device=x.device).long()
            sse_accum = torch.zeros(flat_batch, device=x.device)

            for s in range(n_time_samples):
                t = t_samples[s]
                noise = noise_samples[s]
                alpha_bar = self.alphas_cumprod[t].view(-1, 1)
                x_t = torch.sqrt(alpha_bar) * x_flat + torch.sqrt(1 - alpha_bar) * noise
                pred_noise = self.net(x_t, t, state_tensor)
                sse = ((pred_noise - noise) ** 2).sum(dim=-1)
                sse_accum += sse

            avg_sse = sse_accum / n_time_samples
            log_probs_flat[:, k] = -0.5 * avg_sse

        return log_probs_flat.reshape(B, L, self.n_states)

    def forward_backward_sampling(self, y, emission_scale=1.0):
        with torch.no_grad():
            log_emission = self.compute_emission_logprob(y)
            log_emission = log_emission - log_emission.mean(dim=2, keepdim=True)
            log_emission = log_emission * emission_scale

        B, L, K = log_emission.shape
        log_trans = F.log_softmax(self.trans_logits, dim=1)
        log_start = F.log_softmax(self.start_logits, dim=0)

        alpha = torch.zeros(B, L, K, device=y.device)
        alpha[:, 0, :] = log_start.unsqueeze(0) + log_emission[:, 0, :]

        for t in range(1, L):
            prev = alpha[:, t-1, :].unsqueeze(2) + log_trans.unsqueeze(0)
            alpha[:, t, :] = log_emission[:, t, :] + torch.logsumexp(prev, dim=1)

        sampled_states = torch.zeros(B, L, dtype=torch.long, device=y.device)
        probs_T = F.softmax(alpha[:, -1, :], dim=1)
        sampled_states[:, -1] = torch.multinomial(probs_T, 1).squeeze(-1)

        for t in range(L - 2, -1, -1):
            next_state = sampled_states[:, t+1]
            trans_cols = log_trans[:, next_state].T
            log_prob = alpha[:, t, :] + trans_cols
            probs = F.softmax(log_prob, dim=1)
            sampled_states[:, t] = torch.multinomial(probs, 1).squeeze(-1)

        return sampled_states

    def viterbi_decode(self, y, emission_scale=1.0):
        with torch.no_grad():
            log_emission = self.compute_emission_logprob(y, n_time_samples=16)
            log_emission = log_emission - log_emission.mean(dim=2, keepdim=True)
            log_emission = log_emission * emission_scale

        B, L, K = log_emission.shape
        log_trans = F.log_softmax(self.trans_logits, dim=1)
        log_start = F.log_softmax(self.start_logits, dim=0)

        dp = torch.zeros(B, L, K, device=y.device)
        pointers = torch.zeros(B, L, K, dtype=torch.long, device=y.device)
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

        return torch.tensor(best_paths, dtype=torch.long, device=y.device)


# ==========================================
# 3. 数据准备 (含防泄露逻辑)
# ==========================================
def prepare_data_no_leakage(returns_data, seq_len=30, train_ratio=0.8):
    N = returns_data.shape[0]
    split_idx = int(N * train_ratio)
    
    # 划分训练/测试集
    train_returns = returns_data[:split_idx]
    test_returns = returns_data[split_idx:]

    print(f"时间划分: 训练 [0:{split_idx}], 测试 [{split_idx}:{N}]")

    # 仅在训练集上Fit
    scaler = StandardScaler()
    train_flat = np.clip(train_returns.flatten().reshape(-1, 1), -10, 10)
    scaler.fit(train_flat)

    # 变换
    train_scaled = scaler.transform(train_returns.flatten().reshape(-1, 1)).reshape(train_returns.shape)
    test_scaled = scaler.transform(test_returns.flatten().reshape(-1, 1)).reshape(test_returns.shape)

    # 制作滑窗序列
    def make_seq(data):
        if len(data) <= seq_len:
            return np.empty((0, seq_len, data.shape[1]))
        return np.array([data[i:i+seq_len] for i in range(len(data) - seq_len)])

    X_train = make_seq(train_scaled)
    X_test = make_seq(test_scaled)

    print(f"  数据准备: 训练集{X_train.shape}, 测试集{X_test.shape}")
    return X_train, X_test, scaler


# ==========================================
# 4. 可视化
# ==========================================
class DPMVisualizer:
    @staticmethod
    def plot_training_dashboard(history, filename="dpm_training_dashboard.png"):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        axes[0].plot(history['warmup_loss'], linewidth=2, color='orange')
        axes[0].set_title('Stage 1: Supervised Warmup (Diffusion Loss)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['em_diffusion'], label='Diffusion Loss', linewidth=2, color='blue')
        axes[1].set_title('Stage 2: EM Training (Diffusion Only)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(history['state_separation'], linewidth=2, color='purple')
        axes[2].set_title('State Persistence (Diagonal of Transition Matrix)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Avg Diagonal')
        axes[2].set_xlabel('Epoch')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"  Saved: {filename}")

    @staticmethod
    def plot_state_analysis(trans_matrix, states, returns, filename="dpm_state_analysis.png"):
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 2, wspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        sns.heatmap(trans_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=[f'S{i}' for i in range(len(trans_matrix))],
                    yticklabels=[f'S{i}' for i in range(len(trans_matrix))],
                    vmin=0, vmax=1, ax=ax1, cbar_kws={'label': 'Probability'})
        ax1.set_title('Transition Matrix', fontsize=12, fontweight='bold')

        ax2 = fig.add_subplot(gs[1])
        # 简单过滤
        if len(returns) != len(states):
            min_len = min(len(returns), len(states))
            returns = returns[:min_len]
            states = states[:min_len]
            
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
        print(f"  Saved: {filename}")


# ==========================================
# 5. 绩效指标 & 交易策略
# ==========================================
def calculate_metrics(strategy_returns, benchmark_returns):
    n_days = len(strategy_returns)
    total_return = (1 + strategy_returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    annual_vol = strategy_returns.std() * np.sqrt(252)
    sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-9)) * np.sqrt(252)

    cum_returns = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    win_rate = np.mean(strategy_returns > 0)
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }

def print_metrics(metrics, name):
    print(f"\n{name} 绩效指标:")
    print(f"  总收益率:   {metrics['total_return']:.2%}")
    print(f"  年化收益:   {metrics['annual_return']:.2%}")
    print(f"  夏普比率:   {metrics['sharpe']:.4f}")
    print(f"  最大回撤:   {metrics['max_drawdown']:.2%}")


class TradingStrategy:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.state_weights = None  # 校准后的仓位权重
        self.bull_state = None

    def predict_states(self, X_test):
        self.model.eval()
        with torch.no_grad():
            if isinstance(X_test, np.ndarray):
                X_test_tensor = torch.from_numpy(X_test).float().to(self.device)
            else:
                X_test_tensor = X_test
            
            es = getattr(self.model, 'emission_scale', 1.0)
            states = self.model.viterbi_decode(X_test_tensor, emission_scale=es)
        return states.cpu().numpy()

    def calibrate(self, train_data, train_returns):
        """
        在训练集上校准状态-收益映射 (消除未来函数)
        
        train_data: (N, seq_len, 500) Tensor
        train_returns: (N, 500) Numpy Array (T+1日收益)
        """
        print("\n>>> 正在校准策略 (使用训练集)...")
        states = self.predict_states(train_data)
        final_states = states[:, -1]

        state_avg_ret = {}
        for s in range(self.model.n_states):
            mask = final_states == s
            if np.any(mask):
                # 收益率取所有股票平均
                state_avg_ret[s] = np.mean(train_returns[mask].mean(axis=1))
            else:
                state_avg_ret[s] = 0.0
        
        # 计算权重 (Softmax/Sigmoid 变体)
        rets = np.array([state_avg_ret[s] for s in range(self.model.n_states)])
        
        # Z-score 标准化后通过 Sigmoid 映射到 [0, 1]
        if rets.std() > 1e-9:
            z = (rets - rets.mean()) / rets.std()
        else:
            z = np.zeros_like(rets)
        
        self.state_weights = 1.0 / (1.0 + np.exp(-z * 2.0))
        self.bull_state = np.argmax(rets)

        print("  校准结果:")
        for s in range(self.model.n_states):
            print(f"    State {s}: AvgRet={state_avg_ret[s]:.4%}, W={self.state_weights[s]:.2f}")

    def strategy_state_timing(self, test_data_np, real_returns):
        """
        使用校准后的权重进行择时
        """
        # 1. 预测状态
        states_all = self.predict_states(test_data_np)
        final_states = states_all[:, -1]

        # 2. 应用权重
        portfolio_returns = []
        for i in range(len(final_states)):
            s = final_states[i]
            # 如果未校准(不应该发生)，默认全仓
            w = self.state_weights[s] if self.state_weights is not None else 1.0
            
            # 策略收益 = 权重 * T+1日市场均价收益
            ret = w * real_returns[i].mean()
            portfolio_returns.append(ret)

        portfolio_returns = np.array(portfolio_returns)
        nav = np.cumprod(1 + portfolio_returns)
        return nav, final_states, portfolio_returns


# ==========================================
# 6. 训练 Pipeline
# ==========================================
def train_hmm_dpm(returns_data, config, device):
    print("\n" + "="*60)
    print("HMM-DPM 训练 (AdaLN + T+1 Fix)")
    print("="*60)

    X_train, X_test, scaler = prepare_data_no_leakage(
        returns_data, seq_len=config['seq_len'], train_ratio=config['train_ratio']
    )

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)

    model = HMM_DPM(
        input_dim=500,
        n_states=config['n_states'],
        n_steps=50,
        hidden_dim=config['hidden_dim']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    # KMeans 初始化
    print("\n>>> KMeans 初始化...")
    volatility = np.std(X_train, axis=2).mean(axis=1).reshape(-1, 1)
    kmeans = KMeans(n_clusters=config['n_states'], random_state=42, n_init=10).fit(volatility)
    
    # 排序
    cluster_vols = [volatility[kmeans.labels_ == k].mean() for k in range(config['n_states'])]
    sorted_indices = np.argsort(cluster_vols)
    label_map = {old: new for new, old in enumerate(sorted_indices)}
    mapped_labels = np.vectorize(label_map.get)(kmeans.labels_)
    
    labels = torch.tensor(mapped_labels, device=device).long()
    init_labels = labels.unsqueeze(1).repeat(1, config['seq_len'])

    # 转移矩阵初始化
    trans_init = np.eye(config['n_states']) * 0.85 + 0.15 / config['n_states']
    model.trans_logits.data = torch.tensor(np.log(trans_init + 1e-10), dtype=torch.float32).to(device)

    history = {'warmup_loss': [], 'em_diffusion': [], 'state_separation': []}

    # Stage 1: Warmup
    print(f"\n>>> Stage 1: Warmup ({config['warmup_epochs']} epochs)...")
    warmup_loader = DataLoader(TensorDataset(X_train_tensor, init_labels), 
                               batch_size=config['batch_size'], shuffle=True)
    
    for epoch in range(config['warmup_epochs']):
        model.train()
        ep_loss = 0
        for bx, by in warmup_loader:
            optimizer.zero_grad()
            loss = model.get_diffusion_loss(bx, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        history['warmup_loss'].append(ep_loss / len(warmup_loader))

    # Stage 2: EM
    print(f"\n>>> Stage 2: EM 联合训练 ({config['em_epochs']} epochs)...")
    EMA_ALPHA = 0.3
    EMISSION_SCALE_START = 0.3
    EMISSION_SCALE_END = 1.0

    for epoch in range(config['em_epochs']):
        model.train()
        ep_diff_loss = 0
        progress = epoch / max(config['em_epochs'] - 1, 1)
        emission_scale = EMISSION_SCALE_START + (EMISSION_SCALE_END - EMISSION_SCALE_START) * progress
        
        epoch_trans_counts = torch.zeros(config['n_states'], config['n_states'], device=device)
        
        # Shuffle batching
        perm = torch.randperm(len(X_train_tensor))
        X_shuffled = X_train_tensor[perm]
        n_batches = len(X_train_tensor) // config['batch_size']

        for i in range(n_batches):
            bx = X_shuffled[i * config['batch_size'] : (i+1) * config['batch_size']]
            
            # E-Step
            with torch.no_grad():
                sampled_states = model.forward_backward_sampling(bx, emission_scale=emission_scale)
                
            # Count transitions
            with torch.no_grad():
                for b in range(sampled_states.shape[0]):
                    for t in range(sampled_states.shape[1]-1):
                        epoch_trans_counts[sampled_states[b, t], sampled_states[b, t+1]] += 1
            
            # M-Step (Diffusion)
            optimizer.zero_grad()
            diff_loss = model.get_diffusion_loss(bx, sampled_states)
            diff_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_diff_loss += diff_loss.item()

        # Epoch End: M-Step (HMM) — 纯 Laplace 平滑 + EMA，不加 sticky bias
        with torch.no_grad():
            epoch_trans_counts += 1.0  # Laplace 平滑
            new_probs = epoch_trans_counts / epoch_trans_counts.sum(dim=1, keepdim=True)

            old_probs = F.softmax(model.trans_logits, dim=1)
            smoothed = EMA_ALPHA * new_probs + (1 - EMA_ALPHA) * old_probs
            model.trans_logits.data = torch.log(smoothed + 1e-8)

        history['em_diffusion'].append(ep_diff_loss / n_batches)
        
        with torch.no_grad():
            tm = F.softmax(model.trans_logits, dim=1).cpu().numpy()
            history['state_separation'].append(np.diag(tm).mean())

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Diff={ep_diff_loss/n_batches:.4f}, "
                  f"StatePer={history['state_separation'][-1]:.3f}")

    model.emission_scale = EMISSION_SCALE_END
    # 返回 X_train_tensor 用于校准，X_test_tensor 用于测试
    return model, history, (X_train_tensor, X_test_tensor)


# ==========================================
# 7. 主函数
# ==========================================
def plot_backtest(nav_strategy, nav_benchmark, ret_strategy, ret_benchmark, states, filename="dpm_backtest_results.png"):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(nav_strategy, label='Strategy', linewidth=2)
    ax1.plot(nav_benchmark, label='Benchmark', linewidth=2, alpha=0.7, linestyle='--')
    ax1.set_title('NAV Comparison (T+1 Fix)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(states, color='purple', linewidth=1.5, drawstyle='steps-post')
    ax2.set_title('Predicted States', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    dd = (nav_strategy - np.maximum.accumulate(nav_strategy)) / np.maximum.accumulate(nav_strategy)
    ax3.fill_between(range(len(dd)), dd, 0, alpha=0.5, color='red')
    ax3.set_title('Drawdown', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def main():
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        'seq_len': 30,
        'n_states': 3,
        'hidden_dim': 128,
        'batch_size': 128,
        'lr': 1e-3,
        'warmup_epochs': 50,
        'em_epochs': 100,
        'train_ratio': 0.8  # 明确指定切分比例
    }

    print("\n>>> 加载数据...")
    df = download_csi500_data(cache_path="csi500_dataset.csv")
    returns_data = df.values.astype(np.float32)
    returns_data = np.clip(returns_data, -0.1, 0.1)
    
    # 训练模型
    model, history, (X_train_tensor, X_test_tensor) = train_hmm_dpm(returns_data, config, device)
    
    # 可视化训练
    vis = DPMVisualizer()
    vis.plot_training_dashboard(history)
    trans_matrix = F.softmax(model.trans_logits, dim=1).detach().cpu().numpy()
    
    # ==========================================
    # 回测准备: 严格 T+1 对齐
    # ==========================================
    print("\n" + "="*60)
    print("回测分析 (T+1)")
    print("="*60)
    
    split_idx = int(len(returns_data) * config['train_ratio'])
    seq_len = config['seq_len']
    
    # --- 1. 准备校准数据 (训练集 T+1) ---
    # X_train_tensor 对应的 T+1 收益
    # X_train[i] 覆盖 [i, i+seq_len)，最后一天是 i+seq_len-1
    # 目标收益是 returns_data[i+seq_len]
    real_returns_train = []
    n_train = X_train_tensor.shape[0]
    
    for i in range(n_train):
        target_idx = i + seq_len
        # 确保 target_idx 在 split_idx 之前(属于训练集范围)
        if target_idx < split_idx:
            real_returns_train.append(returns_data[target_idx])
        else:
            break # 超过训练集边界
            
    real_returns_train = np.array(real_returns_train)
    # 截断 X_train 以匹配可用收益长度
    X_train_calib = X_train_tensor[:len(real_returns_train)]
    
    print(f"校准样本数: {len(X_train_calib)}")
    
    # --- 2. 准备测试数据 (测试集 T+1) ---
    # X_test[i] 覆盖 [split_idx+i, split_idx+i+seq_len)
    # 目标收益是 returns_data[split_idx+i+seq_len]
    real_returns_test = []
    n_test = X_test_tensor.shape[0]
    
    for i in range(n_test):
        target_idx = split_idx + i + seq_len
        if target_idx < len(returns_data):
            real_returns_test.append(returns_data[target_idx])
        else:
            break
            
    real_returns_test = np.array(real_returns_test)
    X_test_backtest = X_test_tensor[:len(real_returns_test)]
    
    print(f"回测样本数: {len(X_test_backtest)}")
    
    # ==========================================
    # 执行策略
    # ==========================================
    strategy = TradingStrategy(model, device)
    
    # 1. 在训练集上校准
    strategy.calibrate(X_train_calib, real_returns_train)
    
    # 2. 在测试集上回测
    nav, states, ret_strategy = strategy.strategy_state_timing(
        X_test_backtest, real_returns_test
    )
    
    # 基准
    benchmark_ret = real_returns_test.mean(axis=1)
    nav_benchmark = np.cumprod(1 + benchmark_ret)
    
    # 结果
    metrics = calculate_metrics(ret_strategy, benchmark_ret)
    print_metrics(metrics, "状态择时(T+1)")
    
    vis.plot_state_analysis(trans_matrix, states, benchmark_ret)
    plot_backtest(nav, nav_benchmark, ret_strategy, benchmark_ret, states)
    
    print("\n完成! 生成文件: dpm_backtest_results.png")

if __name__ == "__main__":
    main()