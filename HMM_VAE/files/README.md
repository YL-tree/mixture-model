# HMM-VAE完整实现 - 使用指南

## 项目简介

本项目实现了论文《Partially Variational Expectation-Maximization for Deep Generative Models》中的HMM-VAE框架,用于股票收益率预测和交易策略。

### 核心特性

✅ **Forward-Backward采样算法** - 严格按照论文实现  
✅ **条件VAE解码器** - p(y | z, x_state)  
✅ **EM训练框架** - E步采样,M步优化  
✅ **Gumbel-Softmax采样** - 可微分的离散采样  
✅ **完整的发射概率** - p(y, z | x) = p(y | z, x) · p(z)  

---

## 文件结构

```
.
├── hmm_vae_complete.py      # 完整模型实现
├── data_downloader.py        # 数据下载脚本
├── run_experiment.py         # 运行实验(本文件)
└── README.md                 # 文档
```

---

## 安装依赖

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn akshare
```

---

## 快速开始

### 方法1: 使用真实CSI500数据

```python
from data_downloader import download_csi500_data
from hmm_vae_complete import train_hmm_vae, evaluate_and_predict, Visualizer
import torch

# 1. 下载数据
returns_df = download_csi500_data(
    start_date="20180101",
    end_date="20231231"
)
returns_data = returns_df.values

# 2. 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = {
    'seq_len': 30,          # 用30天历史预测
    'latent_dim': 8,        # 潜变量维度
    'n_states': 3,          # 3个市场状态
    'batch_size': 64,
    'n_epochs': 100,
    'temperature_start': 5.0,
    'temperature_end': 0.5
}

# 3. 训练模型
vae, hmm, history, data = train_hmm_vae(returns_data, config, device)

# 4. 评估
X_train, X_test, scaler = data
results = evaluate_and_predict(vae, hmm, X_test, returns_data, device, config)

# 5. 可视化
vis = Visualizer()
vis.plot_training_curves(history)
vis.plot_portfolio_performance(
    results['nav_strategy'],
    results['nav_benchmark'],
    results['states']
)
```

### 方法2: 使用示例数据(快速测试)

```python
from hmm_vae_complete import main

# 一键运行完整pipeline
vae, hmm, results = main()
```

---

## 模型架构详解

### 1. 条件VAE (ConditionalVAE)

```
编码器: y → (μ, σ²)
        [Linear(500, 256) → ReLU → Linear(256, 256) → ReLU]
        ├─ μ: Linear(256, 8)
        └─ σ²: Linear(256, 8)

解码器: (z, x_state) → y_recon
        [Concat(z, state_onehot) → Linear(8+3, 256) → ReLU → Linear(256, 500)]
```

**关键点**: 解码器必须以状态x_state为条件,这样不同状态下可以生成不同的收益率分布。

### 2. HMM Forward-Backward采样

**Forward算法** (计算α):
```
α_k(1) = π_k · b_k(y_1, z_1)
α_k(i) = b_k(y_i, z_i) · Σ_j [α_j(i-1) · a_{jk}]
```

**Backward采样** (从后验采样):
```
1. x_n ~ Categorical(softmax(α(:, n)))
2. For i = n-1, ..., 1:
   x_i ~ Categorical(softmax(α(:, i) + log_trans(:, x_{i+1})))
```

**发射概率**:
```python
log p(y_i, z_i | x_i = k) = log p(y_i | z_i, x_i = k) + log p(z_i)
                           = -||y_i - decode(z_i, k)||² - 0.5||z_i||²
```

### 3. EM训练流程

```
For each epoch:
    For each batch:
        # E-step
        z ~ q_φ(z | y)
        α = forward_algorithm(y, z)
        X ~ backward_sampling(α)  # 关键!
        
        # M-step
        # 更新VAE
        y_recon = decode(z, X)
        L_vae = ||y - y_recon||² + KL(q(z|y) || p(z))
        
        # 更新HMM
        L_hmm = -log p(X | y, z)
        
        # 反向传播
        (L_vae + L_hmm).backward()
```

---

## 交易策略

### 预测流程

```python
# 测试阶段
for day_t in test_period:
    # 1. 编码当天数据
    y_t = returns[t-30:t]  # 过去30天
    z_t = vae.encode(y_t)
    
    # 2. 预测当天状态
    alpha_t = hmm.forward_algorithm(y_t, z_t)
    state_t = viterbi_decode(alpha_t)
    
    # 3. 识别牛熊
    if state_t == bull_state:
        持仓(做多)
    else:
        空仓
```

### 策略细节

**简化版** (当前实现):
- 识别收益率最高的状态为"牛市"
- 牛市: 全仓持有市场组合
- 其他: 空仓

**增强版** (可选扩展):
```python
# 在预测状态下生成具体股票收益率
pred_returns = vae.decode(z_t, state_onehot[state_t])

# 做多前10只预测收益最高的
top10_long = pred_returns.argsort()[-10:]

# 做空后10只预测收益最低的  
top10_short = pred_returns.argsort()[:10]

# 每只1000元
positions = {
    'long': {stock: 1000 for stock in top10_long},
    'short': {stock: 1000 for stock in top10_short}
}
```

---

## 超参数调优建议

### 关键超参数

| 参数 | 建议范围 | 说明 |
|------|---------|------|
| `seq_len` | 20-60 | 序列长度,太短信息不足,太长计算慢 |
| `latent_dim` | 4-16 | 潜变量维度,500只股票用8-12维足够 |
| `n_states` | 2-5 | 状态数,3个(牛/熊/震荡)是常见选择 |
| `temperature_start` | 3-10 | Gumbel初始温度,太高采样随机 |
| `temperature_end` | 0.1-1.0 | 最终温度,太低梯度消失 |
| `n_epochs` | 50-200 | 训练轮数,观察收敛曲线 |

### 优化建议

1. **状态坍缩问题**: 如果所有样本都预测为同一状态
   - 降低HMM损失权重
   - 增加状态互斥正则化
   - 调高Gumbel温度

2. **VAE重建质量差**: 
   - 增加`latent_dim`
   - 降低KL权重
   - 增加hidden层维度

3. **过拟合**:
   - 添加Dropout
   - 减少模型容量
   - 增加训练数据

---

## 预期结果

### 训练曲线

**正常情况**:
- VAE重建损失: 单调下降,最终稳定在0.01-0.05
- KL散度: 稳定在1-5之间
- HMM负对数似然: 逐步下降

**异常情况**:
- VAE重建损失不下降 → 学习率过高/模型容量不足
- KL散度爆炸 → KL权重过低,缺少正则化
- HMM损失震荡 → 状态采样不稳定,调整温度

### 状态转移矩阵示例

```
         State 0  State 1  State 2
State 0   0.850    0.100    0.050   # 牛市倾向保持
State 1   0.150    0.700    0.150   # 震荡市
State 2   0.050    0.100    0.850   # 熊市倾向保持
```

**健康指标**:
- 对角线元素 > 0.7 (状态持续性)
- 转移概率 > 0.05 (状态可达性)

### 回测表现

**示例指标**:
- Sharpe Ratio: 0.8 - 1.5 (年化)
- 胜率: 55% - 65%
- 最大回撤: < 20%

---

## 常见问题

### Q1: 为什么要用Forward-Backward采样而不是Viterbi?

**A**: 
- Viterbi是确定性算法,找最优路径
- EM算法需要从后验分布**采样**,不是找最优
- 采样才能正确计算期望,保证EM收敛性

### Q2: 为什么VAE解码器要以状态为条件?

**A**: 论文公式要求 p(y | z, **x**)
- 不同市场状态下,股票收益率分布不同
- 条件解码器可以学习"状态依赖"的生成模型
- 这是HMM-VAE的核心创新

### Q3: Gumbel-Softmax的作用?

**A**:
- 离散采样不可微,无法反向传播
- Gumbel-Softmax提供连续松弛版本
- 温度退火: 训练初期软采样(可微),后期硬采样(离散)

### Q4: 如何处理状态坍缩?

**A**: 添加状态互斥正则化
```python
# 鼓励不同状态的发射分布分离
state_centers = hmm.emission_mu  # (n_states, latent_dim)
pairwise_dist = torch.pdist(state_centers)
repulsion_loss = -torch.mean(pairwise_dist)  # 最大化距离
```

---

## 代码对比: 您的实现 vs 论文要求

| 组件 | 您的代码 | 论文要求 | 本实现 |
|------|---------|---------|--------|
| 状态推断 | Viterbi | Forward-Backward采样 | ✅ |
| 发射概率 | p(z\|x) | p(y,z\|x) | ✅ |
| VAE解码器 | decode(z) | decode(z, x) | ✅ |
| 训练框架 | 联合优化 | EM迭代 | ✅ |
| 采样方法 | - | Gumbel-Softmax | ✅ |

---

## 扩展方向

### 1. 使用HMM-DPM (扩散模型)

论文第7-9页描述了用扩散模型替代VAE:

```python
# 前向扩散
q(z_t | z_{t-1}) = N(√α_t z_{t-1}, (1-α_t)I)

# 后向去噪(条件于状态)
p(z_{t-1} | z_t, x_state) = N(μ_θ(z_t, t, x_state), σ_t²I)
```

### 2. 多资产配置

```python
# 为每只股票单独建模
for stock_i in range(n_stocks):
    z_i = vae.encode(y[:, stock_i])
    state_i = hmm.predict(z_i)
    weight_i = f(state_i, z_i)  # 动态权重
```

### 3. 风险控制

```python
# 在预测中加入不确定性估计
z_samples = [vae.encode(y) for _ in range(100)]  # MC采样
pred_var = np.var([vae.decode(z) for z in z_samples])

# 高不确定性 → 降低仓位
position_size = base_size * (1 - pred_var / threshold)
```

---

## 引用

如果使用本代码,请引用原论文:

```
@article{hmm_vae_2025,
  title={Partially Variational Expectation-Maximization for Deep Generative Models},
  author={Author 1 and Author 2},
  journal={},
  year={2025}
}
```

---

## 许可

MIT License

---

## 联系方式

Issues and Pull Requests are welcome!


---                                                                                                                                        
  核心问题：预训练时 VAE 的状态条件输入是常量，导致解码器学会了忽略状态信息                                                                  
                                                                                                                                             
  这是最根本的问题。看 hmm_vae_complete.py:103-114 的 forward 方法：                                                                         
                                                                                                                                             
  def forward(self, y, state_onehot=None):                                                                                                   
      if state_onehot is None:                                                                                                               
          # 训练初期,使用均匀状态分布                                                                                                        
          state_onehot = torch.ones(batch, seq_len, self.n_states).to(y.device) / self.n_states                                              
      recon = self.decode(z, state_onehot)                                                                                                   
                                                                                                                                             
  预训练阶段 (run_experiment.py:95) 调用 vae(batch_x) 时不传 state_onehot，所以每次输入到解码器的状态都是 [1/3, 1/3,                         
  1/3]。50个epoch的预训练下来，解码器完全学会了忽略状态输入——因为它是常量，梯度为零，与状态相关的权重几乎没有更新。                          
                                                                                                                                             
  这直接导致了一个鸡生蛋的死循环：                                                                                                           
  1. 解码器不区分状态 → 不同状态的发射概率几乎相同                                                                                           
  2. 发射概率相同 → HMM 无法区分状态 → 采样出的状态接近随机                                                                                  
  3. 随机状态 → 解码器仍然学不到有意义的状态条件模式                                                                                         
                                                                                                                                             
  从你的结果可以直接验证这一点：                                                                                                             
                                                                                                                                             
  - 转移矩阵接近均匀分布：对角线值 (0.269, 0.495, 0.340) 没有明显的自持性，说明 HMM 没学到有意义的状态结构                                   
  - Stage1 和 Stage2 的 VAE 质量差异不大：R² 从 0.5086 → 0.6920，说明 EM 联合训练对 VAE 的提升有限，状态条件没真正起作用                     
                                                                                                                                             
  ---                                                                                                                                        
  其他问题                                                                                                                                   
                                                                                                                                             
  1. 压缩比过高：500维 → 12维                                                                                                                
                                                                                                                                             
  500个股票的特征压缩到12维潜空间，R² 只有 0.5，说明信息损失严重。Latent Distribution vs Prior 图中，latent z 的实际分布 N(0.00, 0.81)       
  方差偏小，说明 VAE 的表达能力受限。                                                                                                        
                                                                                                                                             
  2. Training Dashboard 中 Recon Loss 在 EM 阶段先降后升                                                                                     
                                                                                                                                             
  底部的 "VAE Components Over Time" 图显示，Reconstruction Loss 在 EM 训练后半段（约 epoch 60 之后）开始反弹上升，同时 KL Divergence         
  持续增长。这是典型的 KL-Recon 冲突——HMM 采样出的状态引入了噪声，但由于状态条件不起作用，VAE 无法利用这些信息来改善重建，反而被干扰。       
                                                                                                                                             
  3. compute_emission_logprob 中解码器用了 torch.no_grad()                                                                                   
                                                                                                                                             
  hmm_vae_complete.py:170-171：                                                                                                              
  with torch.no_grad():                                                                                                                      
      y_recon = self.vae.decode(z, state_onehot)                                                                                             
                                                                                                                                             
  这意味着 HMM 优化发射概率时，VAE 解码器是完全冻结的。虽然这在标准 EM                                                                       
  中是合理的，但在你的情况下加剧了问题——因为解码器本身就不区分状态，冻住它让 HMM 更难学到好的状态分配。                                      
                                                                                                                                             
  ---                                                                                                                                        
  建议修复方向                                                                                                                               
                                                                                                                                             
  1. 预训练时使用随机状态采样，而不是固定均匀分布，让解码器从一开始就学会利用状态信息                                                        
  2. 增大 latent_dim（比如 32 或 64），或增加 encoder/decoder 层数，提升 VAE 表达能力                                                        
  3. EM 训练初期，先冻结 HMM 只更新 VAE 几轮，让 VAE 先适应状态条件解码                                                                      
  4. 考虑用 KL annealing（从 0 逐渐增加到目标权重），避免 KL-Recon 冲突    