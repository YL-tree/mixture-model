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
        # 用小随机值初始化，让不同状态从一开始就产生不同的 scale/shift
        nn.init.normal_(self.proj.weight, std=0.02)
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

    def forward_backward_sampling(self, y, emission_scale=1.0):
        """Forward-Backward 采样"""
        with torch.no_grad():
            log_emission = self.compute_emission_logprob(y)
            log_emission = log_emission * emission_scale

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

    def viterbi_decode(self, y, emission_scale=1.0):
        """Viterbi 解码"""
        with torch.no_grad():
            log_emission = self.compute_emission_logprob(y, n_time_samples=16)
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
# 5. 绩效指标 & 交易策略 (与 HMM_VAE 对齐)
# ==========================================
def calculate_metrics(strategy_returns, benchmark_returns):
    """计算绩效指标"""
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

    excess_returns = strategy_returns - benchmark_returns
    ir = (excess_returns.mean() / (excess_returns.std() + 1e-9)) * np.sqrt(252)
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'information_ratio': ir,
        'calmar': calmar,
        'total_return': total_return
    }


def print_metrics(metrics, strategy_name):
    """打印指标"""
    print(f"\n{strategy_name} 绩效指标:")
    print(f"  总收益率:        {metrics['total_return']:.2%}")
    print(f"  年化收益率:      {metrics['annual_return']:.2%}")
    print(f"  年化波动率:      {metrics['annual_vol']:.2%}")
    print(f"  夏普比率:        {metrics['sharpe']:.4f}")
    print(f"  信息比率:        {metrics['information_ratio']:.4f}")
    print(f"  Calmar比率:      {metrics['calmar']:.4f}")
    print(f"  最大回撤:        {metrics['max_drawdown']:.2%}")
    print(f"  胜率:            {metrics['win_rate']:.2%}")


class TradingStrategy:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_states(self, X_test):
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).float().to(self.device)
            es = getattr(self.model, 'emission_scale', 0.1)
            states = self.model.viterbi_decode(X_test_tensor, emission_scale=es)
        return states.cpu().numpy()

    def strategy_state_timing(self, test_data_np, real_returns):
        """
        策略: 状态择时 — 牛市持有，非牛市空仓

        test_data_np: (n_samples, seq_len, n_stocks) numpy
        real_returns:  (n_samples, n_stocks) numpy — 每个序列最后一天的原始收益
        返回: nav, states, portfolio_returns
        """
        # Viterbi 解码
        states_all = self.predict_states(test_data_np)   # (n_samples, seq_len)
        final_states = states_all[:, -1]                  # 每个序列最后一步的状态

        # 每个状态的平均日收益（所有股票等权）
        state_returns = {}
        for s in range(self.model.n_states):
            mask = final_states == s
            if np.any(mask):
                state_returns[s] = np.mean(real_returns[mask].mean(axis=1))
            else:
                state_returns[s] = 0.0

        bull_state = max(state_returns, key=state_returns.get)

        print(f"\n  识别的牛市状态: {bull_state}")
        for s, ret in state_returns.items():
            print(f"    State {s}: 平均收益 = {ret:.4%}")

        # 执行策略
        portfolio_returns = []
        for i in range(len(final_states)):
            if final_states[i] == bull_state:
                daily_return = real_returns[i].mean()   # 等权持有所有股票
            else:
                daily_return = 0.0
            portfolio_returns.append(daily_return)
        portfolio_returns = np.array(portfolio_returns)

        nav = np.cumprod(1 + portfolio_returns)
        return nav, final_states, portfolio_returns


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
    # Emission scale 退火: 从 0.005 (transition 主导) 到 0.1 (emission 有意义但不碾压)
    # raw emission diff ≈ 1.0/step, transition diff ≈ 2.84/step
    # scale=0.005 → effective emission diff ≈ 0.005 (transition 主导 500x)
    # scale=0.1   → effective emission diff ≈ 0.1   (transition 仍主导 ~30x, 但 emission 有影响)
    EMISSION_SCALE_START = 0.005
    EMISSION_SCALE_END = 0.1

    for epoch in range(config['em_epochs']):
        model.train()
        epoch_diff_loss = 0

        # 退火 emission scale
        progress = epoch / max(config['em_epochs'] - 1, 1)
        emission_scale = EMISSION_SCALE_START + (EMISSION_SCALE_END - EMISSION_SCALE_START) * progress

        # 每 epoch 收集全局转移统计量
        epoch_trans_counts = torch.zeros(config['n_states'], config['n_states'], device=device)

        perm = torch.randperm(len(X_train))
        X_shuffled = X_train[perm]
        n_batches = len(X_train) // config['batch_size']

        for i in range(n_batches):
            bx = X_shuffled[i * config['batch_size'] : (i+1) * config['batch_size']]

            # E-step (with emission scaling)
            with torch.no_grad():
                sampled_states = model.forward_backward_sampling(bx, emission_scale=emission_scale)

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
                  f"StatePer = {state_persistence:.3f}, "
                  f"EmScale = {emission_scale:.4f}")

    # 保存最终 emission_scale 供测试时使用
    model.emission_scale = EMISSION_SCALE_END
    print("\n  训练完成!")
    return model, history, (X_train, X_test, X_test_np, scaler)


# ==========================================
# 7. 主函数
# ==========================================
def plot_backtest(nav_strategy, nav_benchmark, ret_strategy, ret_benchmark, states,
                  filename="dpm_backtest_results.png"):
    """回测可视化 (与 HMM_VAE 对齐)"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. 净值曲线
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(nav_strategy, label='Strategy: State Timing', linewidth=2)
    ax1.plot(nav_benchmark, label='Benchmark: Buy & Hold', linewidth=2, alpha=0.7, linestyle='--')
    ax1.set_title('NAV Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('NAV')

    # 2. 日收益率分布
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(ret_strategy, bins=50, alpha=0.6, label='Strategy', density=True)
    ax2.hist(ret_benchmark, bins=50, alpha=0.6, label='Benchmark', density=True)
    ax2.set_title('Return Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Daily Return')

    # 3. 回撤曲线
    ax3 = fig.add_subplot(gs[1, 1])
    dd_s = (nav_strategy - np.maximum.accumulate(nav_strategy)) / np.maximum.accumulate(nav_strategy)
    dd_b = (nav_benchmark - np.maximum.accumulate(nav_benchmark)) / np.maximum.accumulate(nav_benchmark)
    ax3.fill_between(range(len(dd_s)), dd_s, 0, alpha=0.5, label='Strategy')
    ax3.plot(dd_b, label='Benchmark', linewidth=2, alpha=0.7, linestyle='--')
    ax3.set_title('Drawdown', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('Drawdown')

    # 4. 状态时序
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(states, drawstyle='steps-post', linewidth=2, color='purple')
    ax4.set_title('Predicted Market States', fontsize=12)
    ax4.set_ylabel('State')
    ax4.set_xlabel('Time')
    ax4.grid(True, alpha=0.3)

    # 5. 累计超额收益
    ax5 = fig.add_subplot(gs[2, 1])
    excess = np.cumprod(1 + ret_strategy) / np.cumprod(1 + ret_benchmark)
    ax5.plot(excess, linewidth=2, color='green')
    ax5.axhline(1.0, color='black', linestyle='--', linewidth=1)
    ax5.set_title('Cumulative Excess Return', fontsize=12)
    ax5.set_ylabel('Strategy / Benchmark')
    ax5.set_xlabel('Time')
    ax5.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


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

    # 可视化训练过程
    print("\n>>> 生成可视化...")
    visualizer = DPMVisualizer()
    visualizer.plot_training_dashboard(history)

    trans_matrix = F.softmax(model.trans_logits, dim=1).detach().cpu().numpy()

    # ==========================================
    # 回测: 对齐真实收益率 (与 HMM_VAE 完全一致)
    # ==========================================
    print("\n" + "="*60)
    print("回测分析")
    print("="*60)

    split_idx = int(len(returns_data) * 0.8)
    seq_len = config['seq_len']
    n_test_samples = X_test_np.shape[0]

    # 每个测试序列最后一天的原始收益率 (n_samples, n_stocks)
    real_returns_test = []
    for i in range(n_test_samples):
        day_idx = split_idx + i + seq_len - 1
        if day_idx < len(returns_data):
            real_returns_test.append(returns_data[day_idx])
        else:
            break
    real_returns_test = np.array(real_returns_test)

    # 截断到相同长度
    min_len = min(n_test_samples, len(real_returns_test))
    X_test_backtest = X_test_np[:min_len]
    real_returns_test = real_returns_test[:min_len]

    print(f"回测样本数: {min_len}")
    print(f"测试集收益率均值: {real_returns_test.mean():.6f}")

    # 执行策略
    print("\n### 策略: 状态择时 ###")
    strategy = TradingStrategy(model, device)
    nav_strategy, states, ret_strategy = strategy.strategy_state_timing(
        X_test_backtest, real_returns_test
    )

    # 基准: 等权买入持有
    benchmark_ret = real_returns_test.mean(axis=1)
    nav_benchmark = np.cumprod(1 + benchmark_ret)

    # 绩效指标
    metrics = calculate_metrics(ret_strategy, benchmark_ret)
    print_metrics(metrics, "状态择时")

    # 状态分析图
    visualizer.plot_state_analysis(trans_matrix, states, benchmark_ret)

    # 回测可视化
    plot_backtest(nav_strategy, nav_benchmark, ret_strategy, benchmark_ret, states)

    # 总结
    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)
    print(f"\n状态择时: Sharpe={metrics['sharpe']:.4f}, "
          f"MaxDD={metrics['max_drawdown']:.2%}, "
          f"Annual={metrics['annual_return']:.2%}")
    print(f"Benchmark: Total={benchmark_ret.sum():.2%}")

    print("\n生成的文件:")
    print("  dpm_training_dashboard.png - 训练监控面板")
    print("  dpm_state_analysis.png     - 状态分析")
    print("  dpm_backtest_results.png   - 回测对比图")


if __name__ == "__main__":
    main()
