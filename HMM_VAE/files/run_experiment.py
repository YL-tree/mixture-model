"""
完整运行示例: HMM-VAE股票交易策略

此脚本展示如何:
1. 下载CSI500数据
2. 训练HMM-VAE模型
3. 实施交易策略
4. 评估回测表现
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# 导入自定义模块
from hmm_vae_complete import (
    ConditionalVAE, HMM_ForwardBackward, EM_Trainer,
    prepare_stock_data, Visualizer, setup_seed
)
from data_downloader import download_csi500_data, load_saved_data


# ==========================================
# 1. 高级交易策略实现
# ==========================================
class TradingStrategy:
    """
    基于HMM-VAE的交易策略
    
    策略1: 状态择时 - 只在牛市持仓
    策略2: 多空对冲 - 做多预测收益高的,做空预测收益低的
    """
    def __init__(self, vae, hmm, device, n_long=10, n_short=10, capital_per_stock=1000):
        self.vae = vae
        self.hmm = hmm
        self.device = device
        self.n_long = n_long
        self.n_short = n_short
        self.capital_per_stock = capital_per_stock
        
    def predict_state(self, y_seq):
        """
        预测当前市场状态
        
        y_seq: (seq_len, n_stocks) 或 (1, seq_len, n_stocks)
        返回: state (int)
        """
        if len(y_seq.shape) == 2:
            y_seq = y_seq.unsqueeze(0)
        
        with torch.no_grad():
            mu, logvar = self.vae.encode(y_seq)
            z = self.vae.reparameterize(mu, logvar)
            
            # 使用Viterbi获取最可能的状态序列
            state_seq = self.hmm.viterbi(y_seq, z)
            
            # 返回最后一个状态
            return state_seq[0, -1].item()
    
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
            # 编码
            mu, _ = self.vae.encode(y_seq)
            z_last = mu[:, -1, :]  # 最后一个时间步的z
            
            # 构造状态one-hot
            state_onehot = torch.zeros(1, self.vae.n_states).to(self.device)
            state_onehot[0, state] = 1.0
            
            # 条件解码
            pred_returns = self.vae.decode(z_last, state_onehot)
            
            return pred_returns.squeeze(0).cpu().numpy()
    
    def strategy_state_timing(self, test_data, real_returns):
        """
        策略1: 状态择时
        
        test_data: (n_samples, seq_len, n_stocks)
        real_returns: (n_samples, n_stocks) 真实收益率
        
        返回: nav, positions, states
        """
        n_samples = len(test_data)
        
        # 识别牛熊状态
        states = []
        for i in range(n_samples):
            state = self.predict_state(test_data[i])
            states.append(state)
        
        states = np.array(states)
        
        # 计算每个状态的平均收益
        state_returns = {}
        for s in range(self.hmm.n_states):
            mask = states == s
            if np.any(mask):
                state_returns[s] = np.mean(real_returns[mask].mean(axis=1))
            else:
                state_returns[s] = 0.0
        
        bull_state = max(state_returns, key=state_returns.get)
        print(f"\n>>> 识别的牛市状态: {bull_state}")
        for s, ret in state_returns.items():
            print(f"State {s}: 平均收益 = {ret:.4%}")
        
        # 执行策略
        portfolio_returns = []
        for i in range(n_samples):
            if states[i] == bull_state:
                # 牛市: 持有市场组合
                daily_return = real_returns[i].mean()
            else:
                # 非牛市: 空仓
                daily_return = 0.0
            
            portfolio_returns.append(daily_return)
        
        portfolio_returns = np.array(portfolio_returns)
        nav = np.cumprod(1 + portfolio_returns)
        
        return nav, states, portfolio_returns
    
    def strategy_long_short(self, test_data, real_returns):
        """
        策略2: 多空对冲
        
        每天:
        - 预测下一日收益率
        - 做多预测收益最高的n_long只股票
        - 做空预测收益最低的n_short只股票
        """
        n_samples = len(test_data)
        portfolio_returns = []
        positions_history = []
        
        for i in range(n_samples):
            # 预测状态
            state = self.predict_state(test_data[i])
            
            # 预测收益率
            pred_ret = self.predict_returns(test_data[i], state)
            
            # 选股
            top_long = np.argsort(pred_ret)[-self.n_long:]  # 预测收益最高
            top_short = np.argsort(pred_ret)[:self.n_short]  # 预测收益最低
            
            # 计算当日收益
            long_return = real_returns[i, top_long].mean()
            short_return = -real_returns[i, top_short].mean()  # 做空收益为负的真实收益
            
            # 总收益 (平均分配资金)
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
# 2. 完整回测分析
# ==========================================
def comprehensive_backtest(strategy, test_data, real_returns):
    """
    全面的回测分析
    """
    print("\n" + "="*60)
    print("回测分析")
    print("="*60)
    
    # 策略1: 状态择时
    print("\n### 策略1: 状态择时 ###")
    nav1, states, ret1 = strategy.strategy_state_timing(test_data, real_returns)
    
    # 基准: 买入持有
    benchmark_ret = real_returns.mean(axis=1)
    nav_benchmark = np.cumprod(1 + benchmark_ret)
    
    # 计算指标
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


def calculate_metrics(strategy_returns, benchmark_returns):
    """计算绩效指标"""
    # 年化收益率
    n_days = len(strategy_returns)
    total_return = (1 + strategy_returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / n_days) - 1
    
    # 年化波动率
    annual_vol = strategy_returns.std() * np.sqrt(252)
    
    # 夏普比率
    sharpe = (strategy_returns.mean() / (strategy_returns.std() + 1e-9)) * np.sqrt(252)
    
    # 最大回撤
    cum_returns = np.cumprod(1 + strategy_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 胜率
    win_rate = np.mean(strategy_returns > 0)
    
    # 信息比率 (相对基准)
    excess_returns = strategy_returns - benchmark_returns
    ir = (excess_returns.mean() / (excess_returns.std() + 1e-9)) * np.sqrt(252)
    
    # Calmar比率
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
    """对比可视化"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 净值曲线对比
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(nav1, label='策略1: 状态择时', linewidth=2)
    ax1.plot(nav2, label='策略2: 多空对冲', linewidth=2)
    ax1.plot(nav_bench, label='基准: 买入持有', linewidth=2, alpha=0.7, linestyle='--')
    ax1.set_title('净值曲线对比', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('净值')
    
    # 2. 日收益率分布
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(ret1, bins=50, alpha=0.6, label='策略1', density=True)
    ax2.hist(ret_bench, bins=50, alpha=0.6, label='基准', density=True)
    ax2.set_title('收益率分布 (策略1 vs 基准)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('日收益率')
    
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(ret2, bins=50, alpha=0.6, label='策略2', color='green', density=True)
    ax3.hist(ret_bench, bins=50, alpha=0.6, label='基准', density=True)
    ax3.set_title('收益率分布 (策略2 vs 基准)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('日收益率')
    
    # 3. 回撤曲线
    ax4 = fig.add_subplot(gs[2, 0])
    drawdown1 = (nav1 - np.maximum.accumulate(nav1)) / np.maximum.accumulate(nav1)
    drawdown2 = (nav2 - np.maximum.accumulate(nav2)) / np.maximum.accumulate(nav2)
    drawdown_bench = (nav_bench - np.maximum.accumulate(nav_bench)) / np.maximum.accumulate(nav_bench)
    
    ax4.fill_between(range(len(drawdown1)), drawdown1, 0, alpha=0.5, label='策略1')
    ax4.fill_between(range(len(drawdown2)), drawdown2, 0, alpha=0.5, label='策略2')
    ax4.plot(drawdown_bench, label='基准', linewidth=2, alpha=0.7, linestyle='--')
    ax4.set_title('回撤曲线', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylabel('回撤 (%)')
    
    # 4. 状态时序
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(states, drawstyle='steps-post', linewidth=2, color='purple')
    ax5.set_title('预测市场状态', fontsize=12)
    ax5.set_ylabel('状态')
    ax5.set_xlabel('时间')
    ax5.grid(True, alpha=0.3)
    
    plt.savefig('backtest_comparison.png', dpi=150, bbox_inches='tight')
    print("\n>>> 回测对比图已保存: backtest_comparison.png")
    plt.close()


# ==========================================
# 3. 主函数
# ==========================================
def main():
    """完整pipeline"""
    # 设置
    setup_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    
    # ==========================================
    # 阶段1: 数据准备
    # ==========================================
    print("=" * 60)
    print("阶段1: 数据准备")
    print("=" * 60)
    
    # 尝试加载已保存的数据
    returns_df = load_saved_data("csi500_returns.csv")
    
    if returns_df is None:
        print("未找到保存的数据,开始下载...")
        returns_df = download_csi500_data(
            start_date="20180101",
            end_date="20231231",
            min_data_ratio=0.95
        )
    
    returns_data = returns_df.values
    print(f"数据形状: {returns_data.shape}")
    
    # ==========================================
    # 阶段2: 模型训练
    # ==========================================
    print("\n" + "=" * 60)
    print("阶段2: 模型训练")
    print("=" * 60)
    
    config = {
        'seq_len': 30,
        'latent_dim': 12,
        'n_states': 3,
        'batch_size': 128,
        'n_epochs': 100,
        'temperature_start': 5.0,
        'temperature_end': 0.5
    }
    
    print(f"配置: {config}")
    
    # 准备数据
    X_train, X_test, scaler = prepare_stock_data(
        returns_data,
        seq_len=config['seq_len'],
        train_ratio=0.8
    )
    
    n_stocks = X_train.shape[2]
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    
    # 转换为Tensor
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    
    from torch.utils.data import DataLoader, TensorDataset
    train_loader = DataLoader(
        TensorDataset(X_train_tensor),
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    # 初始化模型
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
    
    # EM训练
    trainer = EM_Trainer(vae, hmm, device)
    
    history = {
        'vae_recon': [],
        'vae_kld': [],
        'hmm_nll': [],
        'total': []
    }
    
    temperature_schedule = np.linspace(
        config['temperature_start'],
        config['temperature_end'],
        config['n_epochs']
    )
    
    print("\n开始EM训练...")
    for epoch in range(config['n_epochs']):
        epoch_losses = {'vae_recon': 0, 'vae_kld': 0, 'hmm_nll': 0, 'total': 0}
        
        for batch_y, in train_loader:
            losses = trainer.em_step(batch_y, temperature=temperature_schedule[epoch])
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
        
        for key in epoch_losses:
            history[key].append(epoch_losses[key] / len(train_loader))
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | "
                  f"Recon: {history['vae_recon'][-1]:.4f} | "
                  f"KLD: {history['vae_kld'][-1]:.4f} | "
                  f"HMM: {history['hmm_nll'][-1]:.1f} | "
                  f"Temp: {temperature_schedule[epoch]:.2f}")
    
    # 可视化训练过程
    vis = Visualizer()
    vis.plot_training_curves(history)
    
    trans_matrix = hmm.get_transition_matrix().detach().cpu().numpy()
    vis.plot_state_transitions(trans_matrix)
    
    # ==========================================
    # 阶段3: 策略回测
    # ==========================================
    print("\n" + "=" * 60)
    print("阶段3: 策略回测")
    print("=" * 60)
    
    # 初始化策略
    strategy = TradingStrategy(
        vae, hmm, device,
        n_long=10,
        n_short=10,
        capital_per_stock=1000
    )
    
    # 准备测试数据
    test_start_idx = len(returns_data) - len(X_test) - config['seq_len']
    real_returns_test = returns_data[test_start_idx + config['seq_len']:]
    min_len = min(len(X_test), len(real_returns_test))
    
    X_test_tensor = X_test_tensor[:min_len]
    real_returns_test = real_returns_test[:min_len]
    
    # 执行回测
    results = comprehensive_backtest(
        strategy,
        X_test_tensor,
        real_returns_test
    )
    
    print("\n" + "=" * 60)
    print("回测完成!")
    print("=" * 60)
    
    return vae, hmm, results, history


if __name__ == "__main__":
    vae, hmm, results, history = main()
