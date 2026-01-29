"""
HMM-DPM 完整实现 (AdaLN 修复版)

核心修复:
1. AdaLN 强制状态控制 — 不同状态产生不同的 scale/shift，网络无法忽略状态
2. Diffusion-loss-based emission — 直接用去噪误差衡量状态匹配度，低方差
3. Epoch 级 EMA 更新转移矩阵
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
    用 conditioning vector 生成 scale 和 shift，
    使得不同状态产生完全不同的激活分布。
    """
    def __init__(self, hidden_dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(cond_dim, hidden_dim * 2)
        # 初始化为 identity transform (scale=1, shift=0)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class ConditionalDiffusionNet(nn.Module):
    """
    带 AdaLN 的条件扩散网络

    关键: 状态不再通过拼接输入，而是通过 AdaLN 控制每一层的归一化参数。
    这迫使网络为不同状态学到不同的去噪行为。
    """
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

        # 输入投影 (只有 x_t，不再拼接 state)
        self.input_proj = nn.Linear(input_dim, hidden_dim * 2)

        # 第一层 + AdaLN
        self.linear1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.adaln1 = AdaLN(hidden_dim * 2, hidden_dim)

        # 第二层 + AdaLN
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.adaln2 = AdaLN(hidden_dim, hidden_dim)

        # 输出
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_t, t, state):
        """预测噪声 eps_theta(x_t, t, state)"""
        t_emb = self.time_mlp(t)
        s_emb = self.state_emb(state)
        cond = t_emb + s_emb  # 融合条件

        h = self.input_proj(x_t)
        h = self.linear1(F.silu(h))
        h = self.adaln1(h, cond)   # 状态在此控制激活分布
        h = self.linear2(F.silu(h))
        h = self.adaln2(h, cond)   # 再次控制
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

    def get_diffusion_loss(self, y, state):
        """标准扩散去噪损失"""
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
        """
        用 diffusion MSE 作为 emission probability:
          log p(y | state=k) ∝ -0.5 * E_t[ ||eps - eps_theta(x_t, t, k)||^2 ]

        所有状态共享相同的 noise 和 timestep，保证公平比较。
        """
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
                # SSE: sum over features (高斯 log-likelihood)
                sse = ((pred_noise - noise) ** 2).sum(dim=-1)
                sse_accum += sse

            avg_sse = sse_accum / n_time_samples
            log_probs_flat[:, k] = -0.5 * avg_sse

        return log_probs_flat.reshape(B, L, self.n_states)

    def forward_backward_sampling(self, y):
        """Forward-Backward 采样"""
        with torch.no_grad():
            log_emission = self.compute_emission_logprob(y)

        B, L, K = log_emission.shape
        log_trans = F.log_softmax(self.trans_logits, dim=1)
        log_start = F.log_softmax(self.start_logits, dim=0)

        # Forward
        alpha = torch.zeros(B, L, K, device=y.device)
        alpha[:, 0, :] = log_start.unsqueeze(0) + log_emission[:, 0, :]

        for t in range(1, L):
            prev = alpha[:, t-1, :].unsqueeze(2) + log_trans.unsqueeze(0)
            alpha[:, t, :] = log_emission[:, t, :] + torch.logsumexp(prev, dim=1)

        # Backward Sampling
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

    def viterbi_decode(self, y):
        """Viterbi 解码"""
        with torch.no_grad():
            log_emission = self.compute_emission_logprob(y, n_time_samples=16)

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
# 3. 数据准备
# ==========================================
def prepare_data_no_leakage(returns_data, seq_len=30, train_ratio=0.8):
    N = returns_data.shape[0]
    split_idx = int(N * train_ratio)
    train_returns = returns_data[:split_idx]
    test_returns = returns_data[split_idx:]

    print(f"时间划分: 训练 [0:{split_idx}], 测试 [{split_idx}:{N}]")

    scaler = StandardScaler()
    train_flat = np.clip(train_returns.flatten().reshape(-1, 1), -10, 10)
    scaler.fit(train_flat)

    train_scaled = scaler.transform(train_returns.flatten().reshape(-1, 1)).reshape(train_returns.shape)
    test_scaled = scaler.transform(test_returns.flatten().reshape(-1, 1)).reshape(test_returns.shape)

    def make_seq(data):
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
# 5. 交易策略
# ==========================================
class TradingStrategy:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_states(self, X_test):
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).float().to(self.device)
            states = self.model.viterbi_decode(X_test_tensor)
        return states.cpu().numpy()

    def backtest_state_timing(self, states, returns):
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

        final_states = states[:, -1]
        strategy_returns = np.where(final_states == bull_state, returns, 0)

        nav = np.cumprod(1 + strategy_returns)
        nav_benchmark = np.cumprod(1 + returns)

        sharpe = (np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9)) * np.sqrt(252)
        win_rate = np.mean(strategy_returns > 0)

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
# 6. 训练 Pipeline
# ==========================================
def train_hmm_dpm(returns_data, config, device):
    print("\n" + "="*60)
    print("HMM-DPM 训练 (AdaLN)")
    print("="*60)

    # 准备数据
    X_train, X_test, scaler = prepare_data_no_leakage(
        returns_data, seq_len=config['seq_len'], train_ratio=0.8
    )

    X_train = torch.from_numpy(X_train).float().to(device)
    X_test_np = X_test
    X_test = torch.from_numpy(X_test).float().to(device)

    # 初始化模型
    model = HMM_DPM(
        input_dim=500,
        n_states=config['n_states'],
        n_steps=50,
        hidden_dim=config['hidden_dim']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    # KMeans 初始化
    print("\n>>> KMeans 初始化...")
    volatility = np.std(X_train.cpu().numpy(), axis=2).mean(axis=1).reshape(-1, 1)
    kmeans = KMeans(n_clusters=config['n_states'], random_state=42, n_init=10).fit(volatility)

    # 排序：State 0 = 低波, State N = 高波
    cluster_vols = [volatility[kmeans.labels_ == k].mean() for k in range(config['n_states'])]
    sorted_indices = np.argsort(cluster_vols)
    label_map = {old: new for new, old in enumerate(sorted_indices)}
    mapped_labels = np.vectorize(label_map.get)(kmeans.labels_)

    labels = torch.tensor(mapped_labels, device=device).long()
    init_labels = labels.unsqueeze(1).repeat(1, config['seq_len'])

    # 转移矩阵初始化 (对角占优)
    trans_init = np.eye(config['n_states']) * 0.85 + 0.15 / config['n_states']
    model.trans_logits.data = torch.tensor(np.log(trans_init + 1e-10), dtype=torch.float32).to(device)

    history = {
        'warmup_loss': [],
        'em_diffusion': [],
        'state_separation': []
    }

    # ===== Stage 1: Supervised Warmup =====
    print(f"\n>>> Stage 1: Warmup ({config['warmup_epochs']} epochs)...")
    warmup_loader = DataLoader(
        TensorDataset(X_train, init_labels),
        batch_size=config['batch_size'], shuffle=True
    )

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

        avg = epoch_loss / len(warmup_loader)
        history['warmup_loss'].append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{config['warmup_epochs']}: Loss = {avg:.4f}")

    # ===== Stage 2: EM Training =====
    print(f"\n>>> Stage 2: EM 联合训练 ({config['em_epochs']} epochs)...")
    EMA_ALPHA = 0.3

    for epoch in range(config['em_epochs']):
        model.train()
        epoch_diff_loss = 0

        # 每 epoch 收集全局转移统计量
        epoch_trans_counts = torch.zeros(config['n_states'], config['n_states'], device=device)

        perm = torch.randperm(len(X_train))
        X_shuffled = X_train[perm]
        n_batches = len(X_train) // config['batch_size']

        for i in range(n_batches):
            bx = X_shuffled[i * config['batch_size'] : (i+1) * config['batch_size']]

            # E-step
            with torch.no_grad():
                sampled_states = model.forward_backward_sampling(bx)

            # 累积转移统计
            with torch.no_grad():
                for b in range(sampled_states.shape[0]):
                    for t_idx in range(sampled_states.shape[1] - 1):
                        s_curr = sampled_states[b, t_idx]
                        s_next = sampled_states[b, t_idx + 1]
                        epoch_trans_counts[s_curr, s_next] += 1

            # M-step: 只有 diffusion loss
            optimizer.zero_grad()
            diff_loss = model.get_diffusion_loss(bx, sampled_states)
            diff_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_diff_loss += diff_loss.item()

        # Epoch 结束: EMA 更新转移矩阵
        with torch.no_grad():
            epoch_trans_counts += 1.0
            new_trans_probs = epoch_trans_counts / epoch_trans_counts.sum(dim=1, keepdim=True)
            old_trans_probs = F.softmax(model.trans_logits, dim=1)
            smoothed_probs = EMA_ALPHA * new_trans_probs + (1 - EMA_ALPHA) * old_trans_probs
            model.trans_logits.data = torch.log(smoothed_probs + 1e-8)

        avg_diff = epoch_diff_loss / max(n_batches, 1)
        history['em_diffusion'].append(avg_diff)

        with torch.no_grad():
            trans_matrix = F.softmax(model.trans_logits, dim=1).cpu().numpy()
            state_persistence = np.diag(trans_matrix).mean()
            history['state_separation'].append(state_persistence)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{config['em_epochs']}: "
                  f"Diff = {avg_diff:.4f}, "
                  f"StatePer = {state_persistence:.3f}")

    print("\n  训练完成!")
    return model, history, (X_train, X_test, X_test_np, scaler)


# ==========================================
# 7. 主函数
# ==========================================
def main():
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

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
    _, _, X_test_np, _ = data

    # 可视化
    print("\n>>> 生成可视化...")
    visualizer = DPMVisualizer()
    visualizer.plot_training_dashboard(history)

    trans_matrix = F.softmax(model.trans_logits, dim=1).detach().cpu().numpy()

    # 回测
    print("\n>>> 回测策略...")
    strategy = TradingStrategy(model, device)
    states = strategy.predict_states(X_test_np)

    test_start = int(len(returns_data) * 0.8) + config['seq_len']
    test_returns = returns_data[test_start : test_start + len(states)].mean(axis=1)

    results = strategy.backtest_state_timing(states, test_returns)

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
    print("\n  Saved: dpm_backtest_results.png")
    print("\n  所有流程完成!")


if __name__ == "__main__":
    main()
