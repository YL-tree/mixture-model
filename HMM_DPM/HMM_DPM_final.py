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
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_emb = nn.Embedding(n_states, hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.mid_block = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), 
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, state):
        t_emb = self.time_mlp(t)
        x_emb = self.input_proj(x)
        s_emb = self.state_emb(state)
        h = torch.cat([x_emb, t_emb, s_emb], dim=1)
        return self.mid_block(h)

class HMM_DPM(nn.Module):
    def __init__(self, input_dim, n_states=2, n_steps=100, hidden_dim=128):
        super().__init__()
        self.n_states = n_states
        self.n_steps = n_steps
        self.net = ConditionalDiffusionNet(input_dim, hidden_dim, n_states)
        
        self.trans_logits = nn.Parameter(torch.zeros(n_states, n_states))
        self.start_logits = nn.Parameter(torch.zeros(n_states))
        
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, n_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def get_diffusion_loss(self, x, state):
        # Flatten sequence for diffusion: (B, L, D) -> (B*L, D)
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

    def compute_emission_probs(self, x, n_samples=5):
        if x.dim() == 3:
            B, L, D = x.shape
            x_flat = x.reshape(B * L, D)
        else:
            B, D = x.shape
            L = 1
            x_flat = x
        
        flat_batch = x_flat.shape[0]
        log_scores = []
        for k in range(self.n_states):
            total_mse = 0
            state_tensor = torch.full((flat_batch,), k, device=x.device).long()
            for _ in range(n_samples):
                t = torch.randint(0, self.n_steps, (flat_batch,), device=x.device).long()
                noise = torch.randn_like(x_flat)
                alpha_bar = self.alphas_cumprod[t].view(-1, 1)
                x_t = torch.sqrt(alpha_bar) * x_flat + torch.sqrt(1 - alpha_bar) * noise
                with torch.no_grad():
                    pred_noise = self.net(x_t, t, state_tensor)
                mse = torch.mean((noise - pred_noise)**2, dim=1)
                total_mse += mse
            avg_mse = total_mse / n_samples
            log_scores.append(-avg_mse * 100) 
        
        scores_flat = torch.stack(log_scores, dim=1) 
        return scores_flat.reshape(B, L, self.n_states)

    def forward_backward_sampling(self, x):
        log_emission = self.compute_emission_probs(x, n_samples=3)
        B, L, K = log_emission.shape
        log_trans = F.log_softmax(self.trans_logits, dim=1)
        log_start = F.log_softmax(self.start_logits, dim=0)
        
        alpha = torch.zeros(B, L, K).to(x.device)
        alpha[:, 0, :] = log_start.unsqueeze(0) + log_emission[:, 0, :]
        
        for t in range(1, L):
            prev = alpha[:, t-1, :].unsqueeze(2) + log_trans.unsqueeze(0)
            alpha_t_given_prev = torch.logsumexp(prev, dim=1)
            alpha[:, t, :] = log_emission[:, t, :] + alpha_t_given_prev
            
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
        log_emission = self.compute_emission_probs(x, n_samples=10)
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
# 2. 增强型绘图模块
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
            axes[1].set_title("Stage 2: Joint Training")
            axes[1].legend()
        plt.tight_layout()
        plt.savefig(filename)

    @staticmethod
    def plot_emission_scores(model, X_sample, states, filename="2_emission_scores.png"):
        B, L, D = X_sample.shape
        X_flat = X_sample.reshape(B*L, D)
        limit = 1000
        if len(X_flat) > limit:
            idx = np.random.choice(len(X_flat), limit, replace=False)
            X_flat = X_flat[idx]
        with torch.no_grad():
            log_scores = model.compute_emission_probs(X_flat, n_samples=5).squeeze(1).cpu().numpy()
        n_states = log_scores.shape[1]
        fig, axes = plt.subplots(1, n_states, figsize=(6*n_states, 5), sharey=True)
        if n_states == 1: axes = [axes]
        for k in range(n_states):
            ax = axes[k]
            for s in range(n_states):
                sns.kdeplot(log_scores[:, s], ax=ax, label=f'Score S{s}', fill=True, alpha=0.3)
            ax.set_title(f"Scores Dist")
            ax.legend()
        plt.tight_layout()
        plt.savefig(filename)

    @staticmethod
    def plot_hmm_insights(model, states, returns, filename="3_hmm_insights.png"):
        n_states = model.n_states
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        trans_mat = torch.exp(model.trans_logits).detach().cpu().numpy()
        trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)
        sns.heatmap(trans_mat, annot=True, fmt=".2f", cmap="Blues", ax=axes[0])
        axes[0].set_title("Transition Matrix")
        
        returns_flat = np.array(returns).flatten()
        states_flat = np.array(states).flatten()
        min_len = min(len(returns_flat), len(states_flat))
        df = pd.DataFrame({'Return': returns_flat[:min_len], 'State': states_flat[:min_len]})
        sns.boxenplot(data=df, x='State', y='Return', palette='coolwarm', ax=axes[1])
        axes[1].axhline(0, color='black', linestyle='--')
        axes[1].set_ylim(-0.04, 0.04)
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
        
        # 染色
        for t in range(len(states)):
            c = 'red' if states[t] == bull_state else 'green'
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
def train_and_evaluate(params, raw_data, device, is_final_run=False):
    SEQ_LEN = params['seq_len']
    N_STATES = params['n_states']
    LR = params['lr']
    HIDDEN_DIM = params['hidden_dim']
    BATCH_SIZE = params['batch_size']
    
    WARMUP_EPOCHS = 60 if is_final_run else 5
    PVEM_EPOCHS = 500 if is_final_run else 5
    
    X = []
    stride = 1 if is_final_run else 5
    for i in range(0, len(raw_data) - SEQ_LEN, stride):
        X.append(raw_data[i : i + SEQ_LEN])
    X = np.array(X)
    
    split_idx = int(len(X) * 0.8)
    X_train = torch.from_numpy(X[:split_idx]).to(device)
    X_test = torch.from_numpy(X[split_idx:]).to(device)
    train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    
    model = HMM_DPM(input_dim=500, n_states=N_STATES, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    history = {'warmup_loss': [], 'pvem_loss': []}
    
    # --- Stage 0: Init ---
    volatility = np.std(X[:split_idx], axis=2).mean(axis=1).reshape(-1, 1)
    try:
        kmeans = KMeans(n_clusters=N_STATES, random_state=42, n_init=10).fit(volatility)
        labels_vec = torch.tensor(kmeans.labels_, device=device).long()
        init_labels = labels_vec.unsqueeze(1).repeat(1, SEQ_LEN)
        
        cluster_vols = [volatility[kmeans.labels_==k].mean() for k in range(N_STATES)]
        sorted_indices = np.argsort(cluster_vols)
        map_dict = {old: new for new, old in enumerate(sorted_indices)}
        new_labels = init_labels.clone()
        for old, new in map_dict.items():
            new_labels[init_labels == old] = new
        init_labels = new_labels
        
        trans_init = np.eye(N_STATES) * 0.9 + (1-0.9)/(N_STATES-1) * (1-np.eye(N_STATES))
        model.trans_logits.data = torch.tensor(np.log(trans_init), dtype=torch.float32).to(device)
    except:
        return -999.0

    # --- Stage 1 & 2 ---
    # ... (训练代码同上，省略以节省空间) ...
    warmup_dataset = TensorDataset(X_train, init_labels)
    warmup_loader = DataLoader(warmup_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(WARMUP_EPOCHS):
        model.train(); t_loss = 0
        for bx, by in warmup_loader:
            optimizer.zero_grad()
            loss = model.get_diffusion_loss(bx, by); loss.backward(); optimizer.step(); t_loss += loss.item()
        history['warmup_loss'].append(t_loss/len(warmup_loader))
        
    for epoch in range(PVEM_EPOCHS):
        model.train(); t_loss = 0; perm = torch.randperm(len(X_train)); X_shuffled = X_train[perm]; n_batches = len(X_train)//BATCH_SIZE
        for i in range(n_batches):
            bx = X_shuffled[i*BATCH_SIZE : (i+1)*BATCH_SIZE]; optimizer.zero_grad()
            with torch.no_grad(): sampled_states = model.forward_backward_sampling(bx)
            loss = model.get_diffusion_loss(bx, sampled_states); loss.backward(); optimizer.step(); t_loss += loss.item()
        history['pvem_loss'].append(t_loss/n_batches)

    # --- Evaluation ---
    model.eval()
    with torch.no_grad(): paths = model.viterbi_decode(X_test)
    final_states = np.array([p[-1] for p in paths])
    
    target_indices = np.arange(split_idx + SEQ_LEN, split_idx + SEQ_LEN + len(final_states))
    valid_mask = target_indices < len(raw_data)
    target_indices = target_indices[valid_mask]
    final_states = final_states[valid_mask]
    returns = np.mean(raw_data[target_indices], axis=1)
    
    state_avg_rets = []
    for k in range(N_STATES):
        mask = (final_states == k)
        if mask.sum() > 0: state_avg_rets.append(np.mean(returns[mask]))
        else: state_avg_rets.append(-1.0)
    bull_state = np.argmax(state_avg_rets)
    
    nav = [1.0]; bench = [1.0]
    for i in range(len(returns)):
        r = returns[i]
        bench.append(bench[-1] * (1+r))
        if final_states[i] == bull_state: nav.append(nav[-1] * (1+r))
        else: nav.append(nav[-1])
            
    strat_ret = np.diff(nav) / nav[:-1]
    sharpe = np.mean(strat_ret) / np.std(strat_ret) * np.sqrt(252) if np.std(strat_ret) > 1e-6 else 0
        
    if is_final_run:
        print(f"\n>>> Final Result: Sharpe = {sharpe:.4f}")
        ShowcasePlotter.plot_training_dashboard(history)
        ShowcasePlotter.plot_emission_scores(model, X_test[:50], final_states[:50]) 
        ShowcasePlotter.plot_hmm_insights(model, final_states, returns)
        ShowcasePlotter.plot_financial_performance(nav, bench, final_states, bull_state)
        
        # === 新增：详细指标报告生成逻辑 ===
        def calc_mdd(ts): return np.min((ts - np.maximum.accumulate(ts)) / np.maximum.accumulate(ts))
        def calc_ann_ret(ts): return (ts[-1])**(252/len(ts)) - 1
        
        strat_arr = np.array(nav)
        bench_arr = np.array(bench)
        
        metrics = {
            'Total Return': (strat_arr[-1]-1, bench_arr[-1]-1),
            'Ann Return': (calc_ann_ret(strat_arr), calc_ann_ret(bench_arr)),
            'Max Drawdown': (calc_mdd(strat_arr), calc_mdd(bench_arr)),
            'Sharpe': (sharpe, (np.mean(returns)/np.std(returns)*np.sqrt(252) if np.std(returns)>0 else 0))
        }
        
        # 动态生成状态统计字符串
        state_stats_str = ""
        for k in range(N_STATES):
            state_stats_str += f"State {k} Avg Ret     : {state_avg_rets[k]*100:.4f}%\n"
            
        report = f"""
==================================================
          HMM-DPM STRATEGY REPORT
==================================================
[Parameters]
Seq Len: {SEQ_LEN} | N States: {N_STATES}
Hidden Dim: {HIDDEN_DIM} | LR: {LR:.2e}
STAGE_NUM: {N_STATES}

[Performance Metrics]
Indicator           | Strategy      | Benchmark
--------------------|---------------|---------------
Total Return        | {metrics['Total Return'][0]*100:6.2f}%       | {metrics['Total Return'][1]*100:6.2f}%
Annualized Return   | {metrics['Ann Return'][0]*100:6.2f}%       | {metrics['Ann Return'][1]*100:6.2f}%
Max Drawdown        | {metrics['Max Drawdown'][0]*100:6.2f}%       | {metrics['Max Drawdown'][1]*100:6.2f}%
Sharpe Ratio        | {metrics['Sharpe'][0]:6.2f}        | {metrics['Sharpe'][1]:6.2f}

[Regime Statistics]
Bull State ID       : {bull_state}
{state_stats_str}
==================================================
"""
        print(report)
        with open("final_metrics_report.txt", "w") as f: f.write(report)
        print("Saved: final_metrics_report.txt")

    return sharpe

def objective(trial):
    params = {
        'seq_len': trial.suggest_int("seq_len", 15, 40, step=5),
        'n_states': trial.suggest_categorical("n_states", [2, 3, 4]),
        'hidden_dim': trial.suggest_categorical("hidden_dim", [64, 128]),
        'lr': trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        'batch_size': trial.suggest_categorical("batch_size", [32, 64])
    }
    return train_and_evaluate(params, RAW_DATA, DEVICE, is_final_run=False)

if __name__ == "__main__":
    setup_seed(42)
    print(">>> Loading Data...")
    df = download_csi500_data(start_date="20160101", end_date="20231231")
    RAW_DATA = df.values.astype(np.float32)
    RAW_DATA = np.clip(RAW_DATA, -0.1, 0.1)
    
    print("\n>>> [AutoML] Starting Optuna Search...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print(f"BEST PARAMS: {study.best_params}")
    print("\n>>> Starting Final Showcase Run...")
    train_and_evaluate(study.best_params, RAW_DATA, DEVICE, is_final_run=True)