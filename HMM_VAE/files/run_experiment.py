"""
使用您的 data.py 进行完整训练

这个脚本:
1. 使用您的 data.py 加载数据
2. 包含所有改进(数据泄漏修复、VAE监控等)
3. 如果有 csi500_dataset.csv 缓存,直接使用
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import sys

# 导入模型和工具
from hmm_vae_complete import (
    ConditionalVAE, HMM_ForwardBackward, EM_Trainer,
    setup_seed
)
from enhanced_visualizer import EnhancedVisualizer

# 导入您的数据加载器
from data import download_csi500_data


def prepare_stock_data_no_leakage(returns_data, seq_len=30, train_ratio=0.8):
    """
    准备数据 (修复数据泄漏)
    """
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
    
    print(f"✓ 数据准备完成: 训练集{X_train.shape}, 测试集{X_test.shape}")
    
    return X_train, X_test, scaler


def compute_pseudo_labels(X_train, n_states):
    """
    基于波动率KMeans聚类生成伪标签

    对每个序列窗口 (seq_len, n_stocks):
      1. 计算每个时间步的截面波动率 std(stocks)
      2. 取序列内均值和最大值作为特征
      3. KMeans聚类 → 每个序列一个伪标签,广播到所有时间步

    X_train: (n_samples, seq_len, n_stocks)
    返回: pseudo_labels (n_samples, seq_len) int64
    """
    n_samples, seq_len, n_stocks = X_train.shape

    # 每个序列的波动率特征
    # (n_samples, seq_len) — 每个时间步的截面标准差
    cross_vol = np.std(X_train, axis=2)  # (n_samples, seq_len)

    # 聚类特征: 均值波动率 + 最大波动率 + 波动率变化幅度
    vol_mean = cross_vol.mean(axis=1, keepdims=True)   # (n_samples, 1)
    vol_max = cross_vol.max(axis=1, keepdims=True)     # (n_samples, 1)
    vol_std = cross_vol.std(axis=1, keepdims=True)     # (n_samples, 1)
    features = np.concatenate([vol_mean, vol_max, vol_std], axis=1)  # (n_samples, 3)

    kmeans = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)  # (n_samples,)

    # 按波动率均值对簇重新编号: 0=低波动, n_states-1=高波动
    cluster_vol = [vol_mean[labels == k].mean() for k in range(n_states)]
    rank = np.argsort(cluster_vol)
    label_map = {old: new for new, old in enumerate(rank)}
    labels = np.array([label_map[l] for l in labels])

    # 打印聚类统计
    print(f"\n>>> 波动率伪标签聚类 (n_states={n_states}):")
    for k in range(n_states):
        mask = labels == k
        print(f"  State {k}: {mask.sum():4d} samples, "
              f"avg_vol={vol_mean[mask].mean():.4f}, "
              f"max_vol={vol_max[mask].mean():.4f}")

    # 广播到 (n_samples, seq_len): 同一序列所有时间步共享标签
    pseudo_labels = np.repeat(labels[:, np.newaxis], seq_len, axis=1)  # (n_samples, seq_len)
    return pseudo_labels.astype(np.int64)


def pretrain_vae(vae, train_loader, device, config, visualizer):
    """VAE预训练 - 使用波动率伪标签 + KL annealing"""
    print("\n" + "="*60)
    print("阶段1: VAE预训练 (波动率伪标签 + KL annealing)")
    print("="*60)

    optimizer = optim.Adam(vae.parameters(), lr=config.get('lr_vae', 1e-3))

    history = {
        'recon_loss': [],
        'kl_loss': [],
        'total_loss': []
    }

    n_epochs = config.get('vae_pretrain_epochs', 50)
    kl_weight_target = config.get('kl_weight', 0.1)

    for epoch in range(n_epochs):
        vae.train()
        epoch_recon = 0
        epoch_kl = 0
        epoch_total = 0

        # KL annealing: 从0线性增长到目标值
        kl_weight = kl_weight_target * min(1.0, epoch / max(n_epochs * 0.5, 1))

        for batch_data in train_loader:
            batch_x = batch_data[0]
            optimizer.zero_grad()

            # 使用波动率伪标签作为FiLM条件
            if len(batch_data) > 1:
                batch_labels = batch_data[1]  # (batch, seq_len) int64
                state_onehot = nn.functional.one_hot(
                    batch_labels, vae.n_states
                ).float()
            else:
                state_onehot = None

            recon, mu, logvar, z = vae(batch_x, state_onehot=state_onehot)

            recon_loss = nn.functional.mse_loss(recon, batch_x, reduction='mean')
            # State-Conditional KL: 把z推向对应状态的先验中心
            if state_onehot is not None:
                kl_loss = vae.kl_divergence(mu, logvar, state_onehot)
            else:
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            total_loss = recon_loss + kl_weight * kl_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()
            epoch_total += total_loss.item()

        n_batches = len(train_loader)
        history['recon_loss'].append(epoch_recon / n_batches)
        history['kl_loss'].append(epoch_kl / n_batches)
        history['total_loss'].append(epoch_total / n_batches)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Recon: {history['recon_loss'][-1]:.4f} | "
                  f"KL: {history['kl_loss'][-1]:.4f} | "
                  f"KLw: {kl_weight:.4f} | "
                  f"Total: {history['total_loss'][-1]:.4f}")
    
    # 评估VAE质量
    print("\n>>> 评估预训练VAE质量...")
    vae.eval()
    with torch.no_grad():
        sample_x, sample_labels = [], []
        for i, batch_data in enumerate(train_loader):
            if i < 5:
                sample_x.append(batch_data[0])
                if len(batch_data) > 1:
                    sample_labels.append(batch_data[1])
            else:
                break

        all_x = torch.cat(sample_x, dim=0)
        # 评估时也用伪标签,反映真实预训练质量
        if sample_labels:
            all_labels = torch.cat(sample_labels, dim=0)
            state_onehot = nn.functional.one_hot(
                all_labels, vae.n_states
            ).float()
            recon, mu, logvar, z = vae(all_x, state_onehot=state_onehot)
        else:
            recon, mu, logvar, z = vae(all_x)

        y_real = all_x.cpu().numpy()
        y_recon = recon.cpu().numpy()
        z_data = mu.cpu().numpy()[:, -1, :]

        # 伪标签用于PCA着色
        stage1_states = None
        if sample_labels:
            stage1_states = all_labels[:, -1].cpu().numpy()  # 每个序列最后一步的状态

        metrics = visualizer.check_vae_quality(
            y_real, y_recon, z_data,
            stage_name="Stage1_VAE_Pretrain",
            filename="stage1_vae_quality.png",
            states=stage1_states
        )

        print(f"VAE预训练质量: MSE={metrics['mse']:.6f}, R²={metrics['r2']:.4f}")

    return history, metrics


def joint_training_em(vae, hmm, train_loader, device, config, visualizer):
    """EM联合训练 - 含VAE warm-up阶段 + KL annealing"""
    print("\n" + "="*60)
    print("阶段2: EM联合训练 (含VAE warm-up)")
    print("="*60)

    trainer = EM_Trainer(vae, hmm, device)

    history = {
        'vae_recon': [],
        'vae_kld': [],
        'hmm_nll': [],
        'total': [],
        'temperature': [],
        'state_separation': []
    }

    n_epochs = config.get('n_epochs', 100)
    em_warmup_epochs = config.get('em_warmup_epochs', 10)
    kl_weight_target = config.get('kl_weight', 0.1)
    temp_start = config.get('temperature_start', 5.0)
    temp_end = config.get('temperature_end', 0.5)
    temperature_schedule = np.linspace(temp_start, temp_end, n_epochs)

    for epoch in range(n_epochs):
        epoch_losses = {
            'vae_recon': 0,
            'vae_kld': 0,
            'hmm_nll': 0,
            'total': 0
        }

        # Change G: 前em_warmup_epochs轮冻结HMM,只更新VAE
        is_warmup = (epoch < em_warmup_epochs)
        # Change E: EM阶段KL annealing
        kl_weight = kl_weight_target * min(1.0, epoch / max(n_epochs * 0.3, 1))

        for batch_data in train_loader:
            batch_x = batch_data[0]  # EM阶段忽略伪标签,由HMM采样状态
            losses = trainer.em_step(
                batch_x,
                temperature=temperature_schedule[epoch],
                warmup=is_warmup,
                kl_weight=kl_weight,
                freeze_hmm=is_warmup
            )

            for key in epoch_losses:
                epoch_losses[key] += losses[key]

        n_batches = len(train_loader)
        for key in epoch_losses:
            history[key].append(epoch_losses[key] / n_batches)

        history['temperature'].append(temperature_schedule[epoch])

        with torch.no_grad():
            trans_matrix = hmm.get_transition_matrix().cpu().numpy()
            state_persistence = np.diag(trans_matrix).mean()
            history['state_separation'].append(state_persistence)

        if (epoch + 1) % 10 == 0:
            phase = "WARMUP" if is_warmup else "EM"
            print(f"[{phase:6s}] Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Recon: {history['vae_recon'][-1]:.4f} | "
                  f"KL: {history['vae_kld'][-1]:.4f} | "
                  f"KLw: {kl_weight:.4f} | "
                  f"HMM: {history['hmm_nll'][-1]:.1f} | "
                  f"Temp: {temperature_schedule[epoch]:.2f} | "
                  f"StatePer: {state_persistence:.3f}")
    
    # 评估最终VAE质量 — 使用Viterbi状态解码,而非均匀状态
    print("\n>>> 评估最终VAE质量 (Viterbi状态解码)...")
    vae.eval()
    hmm.eval()
    with torch.no_grad():
        sample_batches = []
        for i, batch_data in enumerate(train_loader):
            if i < 5:
                sample_batches.append(batch_data[0])
            else:
                break

        all_x = torch.cat(sample_batches, dim=0)

        # 编码
        mu, logvar = vae.encode(all_x)
        z = vae.reparameterize(mu, logvar)

        # Viterbi解码获取状态序列
        state_seq = hmm.viterbi(all_x, z)  # (n_samples, seq_len)
        state_onehot = nn.functional.one_hot(
            state_seq, vae.n_states
        ).float().to(all_x.device)

        # 用Viterbi状态条件解码
        recon = vae.decode(z, state_onehot)

        y_real = all_x.cpu().numpy()
        y_recon = recon.cpu().numpy()
        z_data = mu.cpu().numpy()[:, -1, :]
        viterbi_states = state_seq[:, -1].cpu().numpy()  # 每个序列最后一步的状态

        final_metrics = visualizer.check_vae_quality(
            y_real, y_recon, z_data,
            stage_name="Stage2_Final_VAE",
            filename="stage2_vae_quality.png",
            states=viterbi_states
        )

        # 打印每个状态的样本数和各自的R²
        print(f"最终VAE质量: MSE={final_metrics['mse']:.6f}, R²={final_metrics['r2']:.4f}")
        for s in range(vae.n_states):
            mask = viterbi_states == s
            n_s = mask.sum()
            if n_s > 0:
                y_r = y_real[mask].mean(axis=1) if len(y_real.shape) == 3 else y_real[mask]
                y_p = y_recon[mask].mean(axis=1) if len(y_recon.shape) == 3 else y_recon[mask]
                ss_res = np.sum((y_r - y_p) ** 2)
                ss_tot = np.sum((y_r - y_r.mean()) ** 2)
                r2_s = 1 - ss_res / (ss_tot + 1e-10)
                print(f"  State {s}: {n_s:4d} samples, R²={r2_s:.4f}")

    return history, final_metrics


# ==========================================
# 交易策略
# ==========================================
class TradingStrategy:
    """
    基于HMM-VAE的交易策略

    策略1: 状态择时 - 基于训练集校准的状态仓位权重
    策略2: 多空对冲 - 做多预测收益高的,做空预测收益低的
    """
    def __init__(self, vae, hmm, device, n_long=10, n_short=10):
        self.vae = vae
        self.hmm = hmm
        self.device = device
        self.n_long = n_long
        self.n_short = n_short
        self.state_weights = None  # 校准后的状态仓位权重

    def calibrate(self, train_data, train_returns):
        """
        在训练集上校准状态-收益映射,避免测试集数据窥探

        train_data: (n_samples, seq_len, n_stocks) tensor - 标准化训练数据
        train_returns: (n_samples, n_stocks) numpy - 对应的原始收益率
        """
        print("\n>>> 校准状态-收益映射 (训练集)...")
        n_samples = len(train_data)

        # 批量预测状态 (避免逐样本推理太慢)
        states = self._batch_predict_states(train_data)

        # 计算每个状态的平均收益
        state_avg_returns = {}
        for s in range(self.hmm.n_states):
            mask = states == s
            if np.any(mask):
                state_avg_returns[s] = np.mean(train_returns[mask].mean(axis=1))
            else:
                state_avg_returns[s] = 0.0

        # 用 softmax 把收益映射为仓位权重 [0, 1]
        # 正收益的状态获得更高权重,负收益的状态权重更低
        returns_arr = np.array([state_avg_returns[s] for s in range(self.hmm.n_states)])
        # 放大差异: 用z-score后sigmoid
        if returns_arr.std() > 1e-9:
            z_scores = (returns_arr - returns_arr.mean()) / returns_arr.std()
        else:
            z_scores = np.zeros_like(returns_arr)
        self.state_weights = 1.0 / (1.0 + np.exp(-z_scores * 2))  # sigmoid with scale=2

        print(f"  状态收益映射 (训练集):")
        for s in range(self.hmm.n_states):
            print(f"    State {s}: avg_return={state_avg_returns[s]:.4%}, "
                  f"position_weight={self.state_weights[s]:.3f}, "
                  f"samples={np.sum(states == s)}")

    def _batch_predict_states(self, data, batch_size=256):
        """批量Viterbi预测, 取每个序列的众数状态 (比单取末尾更稳定)"""
        all_states = []
        n = len(data)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch = data[start:end]
                mu, logvar = self.vae.encode(batch)
                z = self.vae.reparameterize(mu, logvar)
                state_seq = self.hmm.viterbi(batch, z)  # (batch, seq_len)
                # 取每个序列的众数状态,比只看最后一步更稳定
                for i in range(len(batch)):
                    seq_states = state_seq[i].cpu().numpy()
                    mode_state = np.bincount(seq_states, minlength=self.hmm.n_states).argmax()
                    all_states.append(mode_state)
        return np.array(all_states)

    def predict_state(self, y_seq):
        """
        预测当前市场状态 (取整个序列的众数)

        y_seq: (seq_len, n_stocks) 或 (1, seq_len, n_stocks)
        返回: state (int)
        """
        if len(y_seq.shape) == 2:
            y_seq = y_seq.unsqueeze(0)

        with torch.no_grad():
            mu, logvar = self.vae.encode(y_seq)
            z = self.vae.reparameterize(mu, logvar)
            state_seq = self.hmm.viterbi(y_seq, z)  # (1, seq_len)
            # 众数状态
            seq_states = state_seq[0].cpu().numpy()
            return np.bincount(seq_states, minlength=self.hmm.n_states).argmax()

    def predict_returns(self, y_seq, state):
        """
        在给定状态下预测下一日收益率

        y_seq: (seq_len, n_stocks)
        state: int
        返回: (n_stocks,) 预测收益率
        """
        if len(y_seq.shape) == 2:
            y_seq = y_seq.unsqueeze(0)

        with torch.no_grad():
            mu, _ = self.vae.encode(y_seq)
            z_last = mu[:, -1, :]

            state_onehot = torch.zeros(1, self.vae.n_states).to(self.device)
            state_onehot[0, state] = 1.0

            pred_returns = self.vae.decode(z_last, state_onehot)
            return pred_returns.squeeze(0).cpu().numpy()

    def strategy_state_timing(self, test_data, real_returns):
        """
        策略1: 状态择时 - 根据训练集校准的仓位权重调整持仓

        test_data: (n_samples, seq_len, n_stocks) tensor
        real_returns: (n_samples, n_stocks) numpy
        返回: nav, states, portfolio_returns
        """
        # 批量预测状态
        states = self._batch_predict_states(test_data)

        # 打印测试集状态分布
        print(f"\n>>> 测试集状态分布:")
        for s in range(self.hmm.n_states):
            n_s = np.sum(states == s)
            avg_ret = real_returns[states == s].mean() if n_s > 0 else 0
            weight = self.state_weights[s] if self.state_weights is not None else "未校准"
            print(f"  State {s}: {n_s:4d} samples, test_avg_return={avg_ret:.4%}, "
                  f"position_weight={weight}")

        # 根据校准权重确定仓位
        portfolio_returns = []
        for i in range(len(states)):
            if self.state_weights is not None:
                weight = self.state_weights[states[i]]
            else:
                weight = 1.0  # 未校准时全仓
            daily_return = weight * real_returns[i].mean()
            portfolio_returns.append(daily_return)

        portfolio_returns = np.array(portfolio_returns)
        nav = np.cumprod(1 + portfolio_returns)

        return nav, states, portfolio_returns

    def strategy_long_short(self, test_data, real_returns):
        """
        策略2: 多空对冲 - 做多Top N / 做空Bottom N

        test_data: (n_samples, seq_len, n_stocks) tensor
        real_returns: (n_samples, n_stocks) numpy
        返回: nav, positions_history, portfolio_returns
        """
        n_samples = len(test_data)
        portfolio_returns = []
        positions_history = []

        for i in range(n_samples):
            state = self.predict_state(test_data[i])
            pred_ret = self.predict_returns(test_data[i], state)

            top_long = np.argsort(pred_ret)[-self.n_long:]
            top_short = np.argsort(pred_ret)[:self.n_short]

            long_return = real_returns[i, top_long].mean()
            short_return = -real_returns[i, top_short].mean()

            total_return = (long_return + short_return) / 2
            portfolio_returns.append(total_return)

            positions_history.append({
                'long': top_long.tolist(),
                'short': top_short.tolist(),
                'state': state
            })

        portfolio_returns = np.array(portfolio_returns)
        nav = np.cumprod(1 + portfolio_returns)

        return nav, positions_history, portfolio_returns


# ==========================================
# 回测评估
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


def plot_comparison(nav1, nav2, nav_bench, ret1, ret2, ret_bench, states):
    """回测对比可视化"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. 净值曲线对比
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(nav1, label='Strategy 1: State Timing', linewidth=2)
    ax1.plot(nav2, label='Strategy 2: Long-Short', linewidth=2)
    ax1.plot(nav_bench, label='Benchmark: Buy & Hold', linewidth=2, alpha=0.7, linestyle='--')
    ax1.set_title('NAV Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('NAV')

    # 2. 日收益率分布
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(ret1, bins=50, alpha=0.6, label='Strategy 1', density=True)
    ax2.hist(ret_bench, bins=50, alpha=0.6, label='Benchmark', density=True)
    ax2.set_title('Return Distribution (S1 vs Benchmark)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Daily Return')

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(ret2, bins=50, alpha=0.6, label='Strategy 2', color='green', density=True)
    ax3.hist(ret_bench, bins=50, alpha=0.6, label='Benchmark', density=True)
    ax3.set_title('Return Distribution (S2 vs Benchmark)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Daily Return')

    # 3. 回撤曲线
    ax4 = fig.add_subplot(gs[2, 0])
    dd1 = (nav1 - np.maximum.accumulate(nav1)) / np.maximum.accumulate(nav1)
    dd2 = (nav2 - np.maximum.accumulate(nav2)) / np.maximum.accumulate(nav2)
    dd_bench = (nav_bench - np.maximum.accumulate(nav_bench)) / np.maximum.accumulate(nav_bench)

    ax4.fill_between(range(len(dd1)), dd1, 0, alpha=0.5, label='Strategy 1')
    ax4.fill_between(range(len(dd2)), dd2, 0, alpha=0.5, label='Strategy 2')
    ax4.plot(dd_bench, label='Benchmark', linewidth=2, alpha=0.7, linestyle='--')
    ax4.set_title('Drawdown', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylabel('Drawdown')

    # 4. 状态时序
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(states, drawstyle='steps-post', linewidth=2, color='purple')
    ax5.set_title('Predicted Market States', fontsize=12)
    ax5.set_ylabel('State')
    ax5.set_xlabel('Time')
    ax5.grid(True, alpha=0.3)

    plt.savefig('backtest_comparison.png', dpi=150, bbox_inches='tight')
    print(">>> Saved backtest_comparison.png")
    plt.close()


def comprehensive_backtest(strategy, test_data, real_returns):
    """全面回测分析"""
    print("\n" + "="*60)
    print("回测分析")
    print("="*60)

    # 策略1: 状态择时
    print("\n### 策略1: 状态择时 ###")
    nav1, states, ret1 = strategy.strategy_state_timing(test_data, real_returns)

    # 基准: 买入持有
    benchmark_ret = real_returns.mean(axis=1)
    nav_benchmark = np.cumprod(1 + benchmark_ret)

    metrics1 = calculate_metrics(ret1, benchmark_ret)
    print_metrics(metrics1, "状态择时")

    # 策略2: 多空对冲
    print("\n### 策略2: 多空对冲 (Top10 Long + Top10 Short) ###")
    nav2, positions, ret2 = strategy.strategy_long_short(test_data, real_returns)

    metrics2 = calculate_metrics(ret2, benchmark_ret)
    print_metrics(metrics2, "多空对冲")

    # 可视化
    plot_comparison(nav1, nav2, nav_benchmark, ret1, ret2, benchmark_ret, states)

    return {
        'strategy1': {'nav': nav1, 'returns': ret1, 'states': states, 'metrics': metrics1},
        'strategy2': {'nav': nav2, 'returns': ret2, 'positions': positions, 'metrics': metrics2},
        'benchmark': {'nav': nav_benchmark, 'returns': benchmark_ret}
    }


def main():
    """主函数 - 使用您的数据"""
    # 设置
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}\n")
    
    # 配置
    config = {
        'seq_len': 30,
        'latent_dim': 32,
        'n_states': 3,
        'batch_size': 128,
        'train_ratio': 0.8,
        # VAE预训练
        'vae_pretrain_epochs': 50,
        'lr_vae': 1e-3,
        'kl_weight': 0.1,
        # EM训练
        'n_epochs': 100,
        'em_warmup_epochs': 10,
        'temperature_start': 5.0,
        'temperature_end': 0.5
    }
    
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    
    # ==========================================
    # 加载您的数据
    # ==========================================
    print("\n" + "="*60)
    print("加载数据 (使用您的 data.py)")
    print("="*60)
    
    # 检查缓存文件
    cache_path = "csi500_dataset.csv"
    
    if os.path.exists(cache_path):
        print(f"✓ 发现缓存文件: {cache_path}")
    else:
        print(f"❌ 未找到缓存文件: {cache_path}")
        print(f"请先运行 data.py 下载数据,或将您的数据文件重命名为 {cache_path}")
        return None
    
    # 使用您的函数加载数据
    df = download_csi500_data(
        start_date="20160101",
        end_date="20231231",
        cache_path=cache_path
    )
    
    if df is None:
        print("❌ 数据加载失败!")
        return None
    
    # 转换为numpy
    returns_data = df.values.astype(np.float32)
    
    print(f"\n✓ 数据加载成功")
    print(f"  形状: {returns_data.shape}")
    print(f"  日期范围: {df.index[0]} 到 {df.index[-1]}")
    
    # 数据统计
    returns_flat = returns_data.flatten()
    print(f"\n数据统计:")
    print(f"  平均收益率: {returns_flat.mean():.6f}")
    print(f"  标准差: {returns_flat.std():.6f}")
    print(f"  最小值: {returns_flat.min():.6f}")
    print(f"  最大值: {returns_flat.max():.6f}")
    
    # ==========================================
    # 准备训练数据
    # ==========================================
    print("\n" + "="*60)
    print("数据预处理")
    print("="*60)
    
    X_train, X_test, scaler = prepare_stock_data_no_leakage(
        returns_data,
        seq_len=config['seq_len'],
        train_ratio=config['train_ratio']
    )
    
    n_stocks = X_train.shape[2]

    # 计算波动率伪标签
    pseudo_labels = compute_pseudo_labels(X_train, n_states=config['n_states'])

    # 转换为Tensor
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    pseudo_labels_tensor = torch.from_numpy(pseudo_labels).long().to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, pseudo_labels_tensor),
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # ==========================================
    # 初始化模型
    # ==========================================
    print("\n>>> 初始化模型...")
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
    
    visualizer = EnhancedVisualizer()
    
    # ==========================================
    # 训练
    # ==========================================
    # 阶段1: VAE预训练
    vae_history, vae_metrics = pretrain_vae(
        vae, train_loader, device, config, visualizer
    )
    
    # 阶段2: EM联合训练
    em_history, final_metrics = joint_training_em(
        vae, hmm, train_loader, device, config, visualizer
    )
    
    # ==========================================
    # 生成可视化
    # ==========================================
    print("\n>>> 生成训练监控面板...")
    visualizer.plot_training_dashboard(em_history, filename="training_dashboard.png")
    
    # 状态转移矩阵
    import seaborn as sns
    trans_matrix = hmm.get_transition_matrix().detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(trans_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'S{i}' for i in range(config['n_states'])],
                yticklabels=[f'S{i}' for i in range(config['n_states'])],
                vmin=0, vmax=1, ax=ax)
    ax.set_title('HMM State Transition Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('transition_matrix.png', dpi=150)
    plt.close()
    print("✓ Saved transition matrix")
    
    # ==========================================
    # 阶段3: 策略回测
    # ==========================================
    print("\n" + "="*60)
    print("阶段3: 策略回测")
    print("="*60)

    # 对齐测试集真实收益率
    # 序列 i 覆盖 [split_idx+i, split_idx+i+seq_len), 最后一天是 T = split_idx+i+seq_len-1
    # T 日收盘后用该序列预测状态 → 决定 T+1 日仓位 → 赚/亏 T+1 日收益
    # 所以 real_returns 应取 T+1 = split_idx + i + seq_len
    split_idx = int(len(returns_data) * config['train_ratio'])
    seq_len = config['seq_len']

    # T+1 日的原始收益率
    real_returns_test = []
    n_test_samples = X_test_tensor.shape[0]
    for i in range(n_test_samples):
        day_idx = split_idx + i + seq_len  # T+1
        if day_idx < len(returns_data):
            real_returns_test.append(returns_data[day_idx])
        else:
            break
    real_returns_test = np.array(real_returns_test)

    # 截断到相同长度
    min_len = min(len(X_test_tensor), len(real_returns_test))
    X_test_backtest = X_test_tensor[:min_len]
    real_returns_test = real_returns_test[:min_len]

    print(f"回测样本数: {min_len}")
    print(f"测试集收益率均值: {real_returns_test.mean():.6f}")

    # 初始化策略
    strategy = TradingStrategy(
        vae, hmm, device,
        n_long=10,
        n_short=10
    )

    # 校准: 在训练集上建立状态-收益映射
    # 训练序列 i 覆盖 [i, i+seq_len), T+1 = i+seq_len
    real_returns_train = []
    n_train_samples = X_train_tensor.shape[0]
    for i in range(n_train_samples):
        day_idx = i + seq_len  # T+1: 用 T 日收盘预测, 赚 T+1 日收益
        if day_idx < split_idx:
            real_returns_train.append(returns_data[day_idx])
        else:
            break
    real_returns_train = np.array(real_returns_train)
    X_train_calib = X_train_tensor[:len(real_returns_train)]

    strategy.calibrate(X_train_calib, real_returns_train)

    # 执行回测
    backtest_results = comprehensive_backtest(
        strategy,
        X_test_backtest,
        real_returns_test
    )

    # ==========================================
    # 总结
    # ==========================================
    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)
    print(f"\n预训练VAE: MSE={vae_metrics['mse']:.6f}, R²={vae_metrics['r2']:.4f}")
    print(f"最终VAE:   MSE={final_metrics['mse']:.6f}, R²={final_metrics['r2']:.4f}")
    print(f"改进: R² {vae_metrics['r2']:.4f} -> {final_metrics['r2']:.4f} "
          f"({(final_metrics['r2'] - vae_metrics['r2']) * 100:+.2f}%)")

    s1_m = backtest_results['strategy1']['metrics']
    s2_m = backtest_results['strategy2']['metrics']
    print(f"\n状态择时: Sharpe={s1_m['sharpe']:.4f}, "
          f"MaxDD={s1_m['max_drawdown']:.2%}, "
          f"Annual={s1_m['annual_return']:.2%}")
    print(f"多空对冲: Sharpe={s2_m['sharpe']:.4f}, "
          f"MaxDD={s2_m['max_drawdown']:.2%}, "
          f"Annual={s2_m['annual_return']:.2%}")

    print("\n生成的文件:")
    print("  stage1_vae_quality.png  - VAE预训练质量")
    print("  stage2_vae_quality.png  - 最终VAE质量")
    print("  training_dashboard.png  - 训练监控面板")
    print("  transition_matrix.png   - 状态转移矩阵")
    print("  backtest_comparison.png - 回测对比图")

    # 保存模型
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'hmm_state_dict': hmm.state_dict(),
        'config': config,
        'vae_metrics': vae_metrics,
        'final_metrics': final_metrics,
        'backtest_metrics': {
            'state_timing': s1_m,
            'long_short': s2_m
        }
    }, 'hmm_vae_model.pth')
    print("  hmm_vae_model.pth      - 训练好的模型")

    return vae, hmm, (vae_history, em_history), (X_train_tensor, X_test_tensor, scaler), backtest_results


if __name__ == "__main__":
    result = main()

    if result is not None:
        vae, hmm, history, data, backtest = result
        print("\n训练成功! 模型和回测结果已保存。")
    else:
        print("\n训练失败,请检查数据文件。")