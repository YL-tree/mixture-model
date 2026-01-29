# HMM-VAE 代码审查报告

## 总体评价
您的代码实现了一个HMM-VAE混合模型,但与论文中描述的"Partially Variational EM"框架存在**重大偏差**。

---

## 主要问题分析

### 1. **缺失核心算法:Forward-Backward采样** ⚠️⚠️⚠️

**论文要求(第10-12页):**
```
采用forward-backward采样策略来采样隐状态序列X:

1. Forward Pass (前向算法):
   α_k(i) = p(y_{1:i}, z_{1:i}, x_i=s_k | θ, Π, A)
   
   初始化: α_k(1) = π_k · b_k(y_1, z_1)
   递推: α_k(i) = b_k(y_i, z_i) · Σ_j α_j(i-1) · a_{jk}

2. Backward Sampling (后向采样):
   - 从 x_n 开始,根据 α_k(n) 采样
   - 然后向前采样 x_{n-1}, ..., x_1
   p(x_i = s_k | x_{i+1} = s_j) ∝ α_k(i) · a_{kj}
```

**您的代码(第73-96行):**
```python
def viterbi(self, x):
    # 使用Viterbi算法寻找最优路径
    # 这是确定性的解码,不是采样!
```

**问题:**
- ❌ 您使用的是**Viterbi算法**(找最优路径),论文要求的是**Forward-Backward采样**(从后验分布采样)
- ❌ Viterbi是确定性的,无法实现EM算法中的E步骤
- ❌ 缺少α(forward variables)的计算用于采样

**影响:** 这导致您的模型无法正确执行论文中的Partially Variational EM算法。

---

### 2. **发射概率计算不符合论文定义** ⚠️⚠️

**论文定义(第10页,公式):**
```
b_k(y_i, z_i) = p(y_i, z_i | x_i = s_k, θ)
              = p(y_i | z_i, x_i = s_k, θ) · p(z_i)
```
其中:
- `p(y_i | z_i, x_i, θ)`: VAE的解码器
- `p(z_i)`: 标准高斯先验 N(0, I)

**您的代码(第47-58行):**
```python
def emission_log_prob(self, x):
    # x 这里是 z (latent code),不是 y (observation)
    # 计算的是 p(z | x_state),而不是 p(y, z | x_state)
    log_prob = -0.5 * (...使用emission_mu和emission_logvar...)
```

**问题:**
- ❌ 您的发射概率只计算了 `p(z | x_state)`,没有包含 `p(y | z, x_state)`
- ❌ 完全忽略了原始观测数据y的条件概率
- ❌ VAE解码器没有参与HMM的推断过程

**正确应该是:**
```python
def compute_emission_log_prob(self, y_i, z_i, state_k):
    # 1. 通过VAE解码器计算 p(y | z, x=k)
    y_recon = self.vae.decode(z_i, condition=state_k)  # 条件解码
    log_p_y_given_z = -F.mse_loss(y_recon, y_i, reduction='none').sum(-1)
    
    # 2. 计算先验 p(z) = N(0, I)
    log_p_z = -0.5 * (z_i**2).sum(-1) - 0.5 * z_i.shape[-1] * np.log(2*np.pi)
    
    # 3. 组合
    return log_p_y_given_z + log_p_z
```

---

### 3. **训练流程与EM算法不符** ⚠️⚠️

**论文要求(第10-11页):**
```
ELBO = E_{Z~q_φ(Z|Y)} E_{X~p(X|Z,Y,θ,Π,A)} [...]
     = L_emission(θ,φ) + L_transition(Π,A)

训练步骤:
1. 用编码器得到 z_i ~ q_φ(z_i | y_i) 对所有i
2. 用Forward-Backward采样得到 X ~ p(X | Z, Y, θ, Π, A)
3. 更新VAE参数θ,φ和HMM参数Π,A
```

**您的代码(第404-446行):**
```python
# Stage 3: Joint Training
for epoch in range(HMM_EPOCHS):
    for batch_x, in train_loader:
        recon, mu, logvar, z = vae(batch_x)
        recon_loss = F.mse_loss(recon, batch_x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        hmm_nll = -hmm(z)  # 这里调用的是forward算法,不是采样
        
        # 没有采样X!直接优化
        total_loss = hmm_nll + 50.0 * recon_loss + 0.05 * kld + repulsion
        total_loss.backward()
```

**问题:**
- ❌ 没有采样隐状态序列X,直接用梯度下降优化
- ❌ `hmm(z)`计算的是forward算法的对数似然,不是ELBO
- ❌ 缺少EM的E步骤(采样X)
- ❌ 您的训练更像是一个联合优化,而不是EM算法

---

### 4. **VAE解码器缺少条件输入** ⚠️

**论文要求(第10页):**
```
p(y_i | z_i, x_i = s_k, θ)
```
解码器必须以隐状态x_i作为条件

**您的代码(第104-110行):**
```python
self.dec = nn.Sequential(
    nn.Linear(latent_dim, hidden_dim),  # 只接受z作为输入
    ...
)

def decode(self, z):
    return self.dec(z)  # 没有x_state作为条件!
```

**问题:**
- ❌ 解码器没有接受状态x作为条件输入
- ❌ 应该是 `decode(z, x_state)` 形式

**正确示例:**
```python
def decode(self, z, x_state_onehot):
    # 拼接z和状态信息
    combined = torch.cat([z, x_state_onehot], dim=-1)
    return self.dec(combined)
```

---

### 5. **初始化策略有问题** ⚠️

**您的代码(第376-402行):**
```python
# 用波动率的中位数分成两类
vol_median = np.median(vol_train)
labels = (vol_train > vol_median).astype(int)
```

**问题:**
- ⚠️ 这种初始化忽略了时序依赖性
- ⚠️ HMM的状态应该有平滑的转移,但波动率阈值会产生频繁跳跃
- ⚠️ 论文建议使用聚类初始化(如GMM或KMeans),而不是简单阈值

---

### 6. **Gumbel Softmax在哪里?** ⚠️

**论文提到(第12页末尾):**
```
"Then we can train the VAE after sampling X using Gumbel softmax mentioned above."
```

**您的代码:**
- ❌ 完全没有使用Gumbel Softmax
- ❌ Viterbi算法是硬分配,不是可微的软采样

**应该实现:**
```python
def gumbel_softmax_sample(logits, temperature=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)
```

---

## 核心缺失功能清单

### 必须实现:
1. ✅ **Forward算法计算α** - 您有(在forward函数中)
2. ❌ **Backward采样算法** - 完全缺失
3. ❌ **条件VAE解码器** - 缺失状态条件
4. ❌ **正确的发射概率** - 当前只考虑z,忽略y
5. ❌ **Gumbel Softmax采样** - 缺失
6. ❌ **EM迭代框架** - 当前是端到端优化

---

## 对股票收益率预测的影响

### 当前架构的问题:
1. **状态无意义**: 由于发射概率不考虑y,HMM状态与收益率的关系被切断
2. **预测不可靠**: Viterbi路径是基于z的聚类,而不是y和z的联合后验
3. **无法捕捉市场制度**: 状态转移没有正确建模,因为缺少forward-backward采样

### 预测流程应该是:
```python
# 测试时
for day_t in test_period:
    # 1. 编码当天数据
    z_t = encoder(y_t)
    
    # 2. 根据昨天状态预测今天状态
    state_probs = transition_matrix[prev_state]  # 使用学到的转移矩阵
    pred_state = argmax(state_probs)
    
    # 3. 在预测状态下生成收益率
    pred_returns = conditional_decoder(z_t, pred_state)  # 条件解码
```

您的代码缺少步骤3的条件生成。

---

## 建议的修改优先级

### 🔴 Critical (必须修改):
1. **实现Forward-Backward采样** 替换Viterbi
2. **修改发射概率** 包含p(y|z,x)
3. **添加条件解码器** x作为输入

### 🟡 Important (强烈建议):
4. 实现Gumbel Softmax采样
5. 重构训练循环为EM框架
6. 改进初始化策略

### 🟢 Nice to have:
7. 添加正则化防止状态坍缩
8. 实现温度退火
9. 可视化状态转移概率

---

## 代码修正示例(核心部分)

### 1. Forward算法(计算α,用于采样):
```python
def forward_algorithm(self, y_seq, z_seq):
    """
    y_seq: (batch, seq_len, n_features) - 原始观测
    z_seq: (batch, seq_len, latent_dim) - VAE编码
    返回: alpha (batch, seq_len, n_states)
    """
    batch, seq_len, _ = y_seq.shape
    alpha = torch.zeros(batch, seq_len, self.n_states).to(y_seq.device)
    
    # 初始化: α_k(1) = π_k · b_k(y_1, z_1)
    log_start = F.log_softmax(self.start_logits, dim=0)
    emission_1 = self.compute_emission_logprob(y_seq[:, 0], z_seq[:, 0])
    alpha[:, 0, :] = log_start + emission_1
    
    # 递推
    log_trans = F.log_softmax(self.trans_logits, dim=1)
    for t in range(1, seq_len):
        emission_t = self.compute_emission_logprob(y_seq[:, t], z_seq[:, t])
        for k in range(self.n_states):
            # α_k(t) = b_k(y_t, z_t) · Σ_j α_j(t-1) · a_{jk}
            trans_score = alpha[:, t-1, :] + log_trans[:, k]
            alpha[:, t, k] = torch.logsumexp(trans_score, dim=1) + emission_t[:, k]
    
    return alpha

def compute_emission_logprob(self, y, z):
    """
    计算 log p(y, z | x_k) = log p(y | z, x_k) + log p(z)
    """
    batch = y.shape[0]
    log_probs = torch.zeros(batch, self.n_states).to(y.device)
    
    for k in range(self.n_states):
        # 条件解码
        state_onehot = F.one_hot(torch.tensor([k]), self.n_states).float()
        state_onehot = state_onehot.expand(batch, -1).to(y.device)
        y_recon = self.vae.decode(z, state_onehot)
        
        # log p(y | z, x=k)
        log_p_y_given_z = -F.mse_loss(y_recon, y, reduction='none').sum(-1)
        
        # log p(z) = log N(0, I)
        log_p_z = -0.5 * (z**2).sum(-1) - 0.5 * z.shape[-1] * np.log(2*np.pi)
        
        log_probs[:, k] = log_p_y_given_z + log_p_z
    
    return log_probs
```

### 2. Backward采样:
```python
def backward_sampling(self, alpha):
    """
    从后验分布采样状态序列
    alpha: (batch, seq_len, n_states)
    返回: sampled_states (batch, seq_len)
    """
    batch, seq_len, n_states = alpha.shape
    sampled = torch.zeros(batch, seq_len, dtype=torch.long)
    
    # 采样最后一个状态
    probs_n = F.softmax(alpha[:, -1, :], dim=1)
    sampled[:, -1] = torch.multinomial(probs_n, 1).squeeze(-1)
    
    # 向前采样
    log_trans = F.log_softmax(self.trans_logits, dim=1)
    for t in range(seq_len-2, -1, -1):
        for b in range(batch):
            next_state = sampled[b, t+1].item()
            # p(x_t = k | x_{t+1}) ∝ α_k(t) · a_{k,next}
            logits = alpha[b, t, :] + log_trans[:, next_state]
            probs = F.softmax(logits, dim=0)
            sampled[b, t] = torch.multinomial(probs, 1).item()
    
    return sampled
```

### 3. 修改训练循环:
```python
# EM框架
for epoch in range(num_epochs):
    for batch_y in dataloader:
        # E-step: 采样隐状态
        with torch.no_grad():
            mu, logvar = vae.encode(batch_y)
            z = vae.reparameterize(mu, logvar)
            alpha = hmm.forward_algorithm(batch_y, z)
            sampled_states = hmm.backward_sampling(alpha)  # 关键!
        
        # M-step: 更新参数
        optimizer.zero_grad()
        
        # VAE部分
        z_new = vae.reparameterize(mu, logvar)
        state_onehot = F.one_hot(sampled_states, n_states).float()
        y_recon = vae.decode(z_new, state_onehot)
        vae_loss = F.mse_loss(y_recon, batch_y)
        
        # HMM部分(使用采样的状态更新转移矩阵)
        # ... 计算状态转移的极大似然估计
        
        total_loss = vae_loss + hmm_loss
        total_loss.backward()
        optimizer.step()
```

---

## 总结

您的实现在以下方面偏离了论文:
1. ❌ 使用Viterbi而不是Forward-Backward采样
2. ❌ 发射概率定义错误
3. ❌ VAE解码器缺少状态条件
4. ❌ 没有实现EM框架
5. ❌ 缺少Gumbel Softmax

这些问题导致模型无法正确学习"状态依赖的收益率分布",从而影响预测性能。

建议从头实现Forward-Backward采样和条件VAE,才能符合论文的Partially Variational EM框架。
