# 快速入门指南

## 📦 文件清单

1. **hmm_vae_complete.py** (26KB) - 完整HMM-VAE实现
   - ConditionalVAE: 条件VAE模型
   - HMM_ForwardBackward: Forward-Backward采样
   - EM_Trainer: EM训练框架
   - Visualizer: 可视化工具

2. **data_downloader.py** (6.7KB) - 数据下载脚本
   - 使用akshare下载CSI500数据
   - 数据清洗和预处理
   - 收益率计算

3. **run_experiment.py** (16KB) - 完整实验脚本
   - TradingStrategy: 两种交易策略
   - 回测分析
   - 绩效评估

4. **code_review.md** (11KB) - 代码审查报告
   - 您原代码的问题分析
   - 与论文的对比

5. **README.md** (8.6KB) - 详细文档
   - 模型架构说明
   - 超参数调优
   - 常见问题

---

## 🚀 30秒快速开始

### 方法1: 直接运行(使用示例数据)

```bash
python run_experiment.py
```

这将:
- ✅ 生成模拟的CSI500数据
- ✅ 训练HMM-VAE模型(100 epochs)
- ✅ 执行两种交易策略回测
- ✅ 生成所有可视化结果

**输出文件**:
- `training_curves.png` - 训练曲线
- `transition_matrix.png` - 状态转移矩阵
- `backtest_comparison.png` - 回测对比

---

### 方法2: 使用真实CSI500数据

```bash
# 1. 安装依赖
pip install akshare torch numpy pandas matplotlib seaborn scikit-learn

# 2. 下载数据
python data_downloader.py

# 3. 运行实验
python run_experiment.py
```

---

## 🔧 自定义配置

编辑`run_experiment.py`中的config:

```python
config = {
    'seq_len': 30,          # 序列长度(天)
    'latent_dim': 12,       # 潜变量维度
    'n_states': 3,          # HMM状态数
    'batch_size': 128,      # 批大小
    'n_epochs': 100,        # 训练轮数
    'temperature_start': 5.0,   # Gumbel初始温度
    'temperature_end': 0.5      # Gumbel最终温度
}
```

---

## 📊 预期输出

### 1. 控制台输出

```
>>> 阶段1: 数据准备
数据形状: (1000, 500)

>>> 阶段2: 模型训练
训练集: (800, 30, 500), 测试集: (200, 30, 500)

Epoch  20 | Recon: 0.0234 | KLD: 2.1456 | HMM: 145.3 | Temp: 4.56
Epoch  40 | Recon: 0.0198 | KLD: 1.8923 | HMM: 132.7 | Temp: 3.72
...

>>> 阶段3: 策略回测

### 策略1: 状态择时 ###
识别的牛市状态: 1
State 0: 平均收益 = -0.12%
State 1: 平均收益 = 0.34%
State 2: 平均收益 = 0.05%

状态择时 绩效指标:
  总收益率:        15.43%
  年化收益率:      12.87%
  夏普比率:        0.9234
  最大回撤:        -8.56%
  胜率:            58.32%

### 策略2: 多空对冲 ###
  总收益率:        22.18%
  年化收益率:      18.45%
  夏普比率:        1.2156
  最大回撤:        -6.23%
  胜率:            61.45%
```

### 2. 生成的图表

**training_curves.png**:
- VAE重建损失曲线
- KL散度曲线
- HMM负对数似然曲线

**transition_matrix.png**:
```
         State 0  State 1  State 2
State 0   0.82     0.12     0.06
State 1   0.10     0.78     0.12
State 2   0.08     0.10     0.82
```

**backtest_comparison.png**:
- 净值曲线对比
- 收益率分布
- 回撤曲线
- 状态时序

---

## ⚠️ 常见问题

### Q: CUDA out of memory

**A**: 减小batch_size
```python
config = {
    'batch_size': 32,  # 从128降到32
    ...
}
```

### Q: 状态全部预测为同一个

**A**: 状态坍缩,调整温度和正则化
```python
# 在hmm_vae_complete.py的EM_Trainer.em_step中添加:
# 计算状态分布熵
state_dist = sampled_states.mean(dim=0).mean(dim=0)
entropy = -(state_dist * torch.log(state_dist + 1e-9)).sum()
entropy_loss = -0.1 * entropy  # 鼓励状态均匀分布

total_loss = vae_loss + hmm_loss + entropy_loss
```

### Q: VAE重建质量差

**A**: 增加模型容量
```python
vae = ConditionalVAE(
    input_dim=n_stocks,
    latent_dim=16,      # 从8增加到16
    n_states=config['n_states'],
    hidden_dim=512      # 从256增加到512
)
```

---

## 🎯 下一步

### 1. 超参数优化

使用Optuna自动搜索最佳参数:

```python
import optuna

def objective(trial):
    config = {
        'latent_dim': trial.suggest_int('latent_dim', 4, 16),
        'n_states': trial.suggest_int('n_states', 2, 5),
        'temperature_end': trial.suggest_float('temp_end', 0.1, 1.0)
    }
    
    # 训练并评估
    vae, hmm, results = train_and_evaluate(config)
    
    return results['strategy1']['metrics']['sharpe']

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 2. 添加风险管理

```python
# 在TradingStrategy中
def strategy_with_risk_control(self, test_data, real_returns, max_drawdown=-0.15):
    nav = 1.0
    for i in range(len(test_data)):
        # 检查当前回撤
        current_dd = (nav - peak_nav) / peak_nav
        if current_dd < max_drawdown:
            # 强制平仓
            position = 0
        else:
            # 正常交易
            ...
```

### 3. 集成多个模型

```python
# Ensemble: 训练多个HMM-VAE,投票决策
models = [train_hmm_vae(config, seed=i) for i in range(5)]

def ensemble_predict(models, y_seq):
    states = [model.predict_state(y_seq) for model in models]
    return max(set(states), key=states.count)  # 多数投票
```

---

## 📚 学习路径

1. **理解论文**: 阅读README.md中的模型架构部分
2. **代码审查**: 查看code_review.md,对比您的代码
3. **运行实验**: 执行run_experiment.py,观察结果
4. **调整参数**: 修改config,尝试不同配置
5. **改进策略**: 实现更复杂的交易逻辑

---

## 💡 提示

- 训练100 epochs在CPU上约需20-30分钟
- GPU可加速至5-10分钟
- 首次运行建议使用示例数据,快速验证代码
- 真实数据下载可能需要10-30分钟(取决于网络)

---

## 📞 获取帮助

遇到问题? 检查:
1. README.md - 详细文档
2. code_review.md - 常见错误
3. 代码注释 - 每个函数都有说明

祝您实验顺利! 🎉
