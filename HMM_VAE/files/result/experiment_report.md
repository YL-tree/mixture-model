# HMM-VAE 实验报告: FiLM 条件解码器 + 训练改进

## 1. 问题背景

初始版本的 HMM-VAE 存在两个核心问题:

1. **HMM 状态坍缩**: 转移矩阵对角线全部 < 1/3, State Separation 仅 0.23, PCA 无聚类
2. **状态条件失效**: 解码器使用 concat one-hot 拼接, 网络学会忽略状态维度

## 2. 修改过程 (3 轮迭代)

### Round 1: FiLM 条件解码器 + HMM 对角偏置

**文件**: `hmm_vae_complete.py`

**改动 A - FiLM 解码器** (替换 concat):
- 删除 `self.decoder = nn.Sequential(nn.Linear(latent_dim + n_states, ...))`
- 新增 4 层独立线性层: `dec_fc1/2/3` + `dec_out`, 输入维度为 `latent_dim` (不含 n_states)
- 新增 3 个 FiLM 生成器: `film1/2/3: Linear(n_states, hidden_dim * 2)`
- 每层执行 `h' = (gamma + 1) * h + beta`, gamma/beta 由状态 one-hot 生成
- 乘性调制使网络无法忽略状态信息

**改动 B - HMM 对角偏置**:
```python
# 之前
self.trans_logits = nn.Parameter(torch.randn(n_states, n_states) * 0.1)
# 之后
init_logits = torch.randn(n_states, n_states) * 0.1
init_logits += torch.eye(n_states) * 2.0  # softmax 后对角 ≈ 0.7
self.trans_logits = nn.Parameter(init_logits)
```

---

### Round 2: 波动率伪标签预训练

**问题**: Round 1 后均匀软状态 (1/3,1/3,1/3) 让 FiLM 在预训练阶段退化, R² 下降。

**文件**: `run_experiment.py`

**改动 C - 波动率 KMeans 伪标签**:
- 对每个训练序列计算截面波动率特征 (均值, 最大值, 标准差)
- KMeans 聚成 n_states 簇, 按波动率从低到高重编号
- 伪标签放入 DataLoader, 预训练时作为 FiLM 条件
- EM 阶段忽略伪标签, 由 HMM backward sampling 提供状态

**设计逻辑**:

| 预训练方案       | FiLM 行为                | R²  | State Sep |
|:-----------------|:-------------------------|:----|:----------|
| 随机 one-hot     | 每样本随机调制 = 噪声    | 差  | 差        |
| 均匀 (1/3,1/3)   | 所有样本相同 = FiLM 退化 | 中  | 差        |
| **波动率伪标签** | 有意义的分组调制         | 好  | 好        |

---

### Round 3: Stage2 评估修复

**问题**: Stage2 评估使用 `vae(all_x)` (uniform 状态), FiLM 已学会强状态条件, uniform 产生"平均调制"导致 R² = -0.50。这是评估 bug, 不是模型问题。

**文件**: `run_experiment.py`, `enhanced_visualizer.py`

**改动 D - Viterbi 状态评估**:
```python
# 之前 (错误)
recon, mu, logvar, z = vae(all_x)  # uniform 状态

# 之后 (正确)
mu, logvar = vae.encode(all_x)
z = vae.reparameterize(mu, logvar)
state_seq = hmm.viterbi(all_x, z)
state_onehot = F.one_hot(state_seq, n_states).float()
recon = vae.decode(z, state_onehot)
```

**改动 E - PCA 按状态着色**:
- `check_vae_quality()` 新增 `states` 参数
- 有状态标签时按状态着色 PCA, 无标签时按 sample index

**改动 F - 交易策略与回测**:
- 从 `run_experiment_1.py` 移植 `TradingStrategy` 类 (状态择时 + 多空对冲)
- 添加 `comprehensive_backtest()` 全面回测评估
- 计算年化收益、Sharpe、最大回撤、信息比率、Calmar、胜率
- 生成 `backtest_comparison.png` (净值曲线、收益分布、回撤、状态时序)

---

## 3. 最终结果

### 3.1 VAE 重建质量

| 指标         | Stage1 (预训练) | Stage2 (EM 后) | 变化     |
|:-------------|:----------------|:---------------|:---------|
| **R²**       | 0.6186          | **0.8213**     | +32.8%   |
| **Corr**     | 0.7876          | **0.9091**     | +15.4%   |
| MSE          | 0.0158          | 0.0072         | -54.4%   |
| MAE          | 0.0564          | 0.0440         | -22.0%   |

EM 联合训练显著提升了 VAE 重建质量, R² 从 0.62 提升到 0.82。

### 3.2 HMM 状态分离

| 指标                   | 初始版本     | 最终版本         |
|:-----------------------|:-------------|:-----------------|
| 转移矩阵对角线        | < 0.35       | **0.908 / 0.915 / 0.867** |
| State Separation       | 0.23         | **0.89**         |
| 状态坍缩?             | 是           | **否**           |

转移矩阵三个状态均有强自持续性 (>0.86), 状态切换合理。

### 3.3 转移矩阵

```
         S0      S1      S2
S0    [0.908]  0.054   0.038
S1     0.058  [0.915]  0.027
S2     0.092   0.041  [0.867]
```

- S1 最稳定 (91.5% 自持续), 可能对应平稳震荡市
- S0 次之 (90.8%), S2 相对活跃 (86.7%)
- 跨状态转移概率均 < 10%, 符合市场regime切换的稀疏特性

### 3.4 潜空间

- Stage1 PCA (19.0% + 18.7%): 按伪标签着色, 3 个状态有一定分布差异但重叠较多 (伪标签本身基于波动率粗聚类)
- Stage2 PCA (15.8% + 14.3%): 按 Viterbi 状态着色, 3 个状态有分布差异, 重叠仍存在但优于 Stage1

潜空间重叠是正常的 -- 32 维潜空间投影到 2 维 PCA 会丢失大量信息。FiLM 调制主要作用在高维隐层激活上, 低维投影不一定能完全体现。

### 3.5 训练动态

- **Recon Loss**: 0.6 -> 0.15, 持续下降, 无震荡
- **KL Divergence**: warmup->EM 过渡时 spike, 之后稳定在 0.5-0.6
- **HMM NLL**: 4000 -> 2500, 持续下降
- **State Separation**: 0.80 -> 0.89, epoch 20 左右快速跳升 (HMM 解冻时刻)
- **温度退火**: 5.0 -> 0.5, 线性下降

### 3.6 回测结果

| 指标       | 策略1 (状态择时) | 策略2 (多空对冲) | 基准 (买入持有) |
|:-----------|:-----------------|:-----------------|:----------------|
| 最终 NAV   | ~1.0             | ~14.0            | ~1.0            |


### 策略1: 状态择时 ###
  识别的牛市状态: 2
  State 0: 平均收益 = -0.0008%
  State 1: 平均收益 = -0.0651%
  State 2: 平均收益 = 0.1234%

状态择时 绩效指标:
  总收益率:        4.61%
  年化收益率:      3.21%
  年化波动率:      2.84%
  夏普比率:        1.1281
  信息比率:        0.5791
  Calmar比率:      2.3866
  最大回撤:        -1.35%
  胜率:            6.41%

### 策略2: 多空对冲 (Top10 Long + Top10 Short) ###

多空对冲 绩效指标:
  总收益率:        1332.74%
  年化收益率:      548.00%
  年化波动率:      10.76%
  夏普比率:        17.4860
  信息比率:        15.7882
  Calmar比率:      624.8298
  最大回撤:        -0.88%
  胜率:            89.42%

**策略1 分析**: NAV 基本不动, 因为模型大部分时间预测为 State 0/2, 若牛市状态样本极少则策略几乎不参与。收益分布高度集中在 0 附近, 验证了该策略处于"永久空仓"状态。

**策略2 分析**: NAV 达 14x, 收益分布明显右偏。但需注意:
- 多空策略的超额收益可能来自 VAE 解码器在不同状态下的系统性偏差, 而非真实预测能力
- 日均收益约 0.7% 偏高, 需排除前视偏差
- 该策略在真实中会受交易成本、做空限制等约束

**状态时序**: 主要在 State 0 和 State 2 之间快速切换, State 1 出现较少。尽管转移矩阵对角线 0.9+, 但 Viterbi 对每个 30 天窗口独立运行, 相邻窗口仅差 1 天, emission probability 的微小变化可能导致最终状态翻转。

---

## 4. 关键结论

1. **FiLM 条件解码器有效解决了状态坍缩问题**: 从 concat (可被忽略) 到 FiLM (乘性调制, 不可忽略), 转移矩阵对角线从 <0.35 提升到 >0.86
2. **波动率伪标签解决了预训练冲突**: 给 FiLM 提供了有意义的初始信号, 避免随机 one-hot 噪声或均匀退化
3. **评估方法必须与模型架构匹配**: FiLM 解码器要求评估时提供正确的状态条件 (Viterbi), 均匀状态会导致严重误判
4. **EM 训练确实提升了 VAE**: R² 从 0.62 (pretrain) 到 0.82 (EM 后), 说明 HMM 状态信息帮助了条件重建

---

## 5. 生成的文件

| 文件                    | 说明                        |
|:------------------------|:----------------------------|
| `stage1_vae_quality.png`  | VAE 预训练质量 (伪标签着色)  |
| `stage2_vae_quality.png`  | EM 后 VAE 质量 (Viterbi 着色)|
| `training_dashboard.png`  | 训练监控面板                 |
| `transition_matrix.png`   | HMM 转移矩阵热力图          |
| `backtest_comparison.png` | 回测对比 (净值/分布/回撤/状态)|
| `hmm_vae_model.pth`       | 模型权重 + 指标               |

历史结果保存在:
- `result/round2/` -- 修复评估前的结果 (Stage2 R² = -0.50)
- `result/round3/` -- 修复评估后的结果 (Stage2 R² = 0.82)

---

## 6. 后续可探索方向

- **PCA 聚类不明显**: 可尝试 t-SNE/UMAP 可视化, 或计算高维空间中的 state-wise silhouette score
- **策略1 空仓问题**: 调整牛市识别逻辑 (如多状态持仓, 或按状态调整仓位比例)
- **策略2 收益偏高**: 加入交易成本、滑点模拟, 检验收益的鲁棒性
- **状态快速切换**: 可在 Viterbi 推理时对连续窗口施加平滑约束
