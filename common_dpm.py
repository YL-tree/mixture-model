import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
# 注意: 你的 evaluate_model 依赖 linear_sum_assignment，这通常需要 scipy。
# 如果你没有安装 scipy，evaluate_model 可能会失败。
# 为了保持代码完整性，我在这里不导入它，但请确保在运行时环境中有它。
# from scipy.optimize import linear_sum_assignment 

# -----------------------------------------------------
# A. 配置类
# -----------------------------------------------------

class Config:
    """
    mDPM_SemiSup 模型的配置参数
    """
    def __init__(self):
        # ---------------------
        # 训练和硬件设置
        # ---------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "./mDPM_sup"
        self.batch_size = 128
        self.final_epochs = 50 
        self.optuna_epochs = 10 
        self.lr = 2e-4                    # 学习率
        self.labeled_per_class = 100      # 每类用于监督学习的样本数 (半监督)
        
        # ---------------------
        # PVEM 框架权重 (用于无监督损失)
        # ---------------------
        # self.beta = 1.0                   # KL 散度权重 (VAE的z已移除，此参数可视为 0 或用于其他正则化)
        self.alpha_unlabeled = 1        # 无标签数据损失的权重
        self.lambda_entropy = 5.0         # 熵惩罚项的权重 (鼓励 q(x|x0) 软分配)
        
        # ---------------------
        # Gumbel Softmax 退火参数 (用于离散潜在变量 x 的推理)
        # ---------------------
        self.initial_gumbel_temp = 1.0    # Gumbel Softmax 初始温度 (tau)
        self.min_gumbel_temp = 0.1        # Gumbel Softmax 最小温度
        self.gumbel_anneal_rate = 0.995   # 每 epoch 的退火率
        self.current_gumbel_temp = self.initial_gumbel_temp # 当前温度
        
        # ---------------------
        # 模型结构和 DPM 参数
        # ---------------------
        self.latent_dim = 0               # 连续潜在变量 z 已移除/简化，设为 0
        self.num_classes = 10             # 离散潜在变量 x 的类别数 (例如 MNIST)
        
        # DPM 特有参数
        self.timesteps = 1000             # 扩散总时间步 T
        self.image_channels = 1           # 输入图像通道数 (例如 MNIST 是 1)
        
        # ConditionalUnet/DPM 参数
        self.unet_base_channels = 64      # U-Net 初始通道数
        self.unet_time_emb_dim = 256      # 时间和类别嵌入维度
        
        # 在实际训练中，需要确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

# -----------------------------------------------------
# B. DPM 前向过程
# -----------------------------------------------------

class DPMForwardProcess(nn.Module):
    """
    DDPM 扩散前向过程：定义 βt, αt 等参数，并实现 q(x_t | x_0) 的采样。
    """
    def __init__(self, timesteps: int = 1000, schedule: str = 'linear'):
        super().__init__()
        self.timesteps = timesteps

        # 定义 β 调度
        if schedule == 'linear':
            # 从 1e-4 到 0.02 的线性调度
            self.register_buffer('betas', torch.linspace(1e-4, 0.02, timesteps))
        else:
            raise NotImplementedError(f"Schedule {schedule} not implemented.")

        # 计算 α 参数
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        """
        根据 q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I) 采样 x_t。
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # 提取对应时间步 t 的参数
        # 确保形状匹配：(B,) -> (B, 1, 1, 1)
        sqrt_alphas_cumprod_t = self._extract_t(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_t(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t

    def _extract_t(self, a, t, x_shape):
        """
        从参数张量 a 中提取对应时间步 t 的值，并重塑以匹配 x_shape。
        """
        batch_size = t.shape[0]
        # 使用 t.to(a.device) 确保索引和张量在同一设备上
        out = a.gather(-1, t.to(a.device)) 
        # 重塑： (B,) -> (B, 1, 1, 1)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# -----------------------------------------------------
# C. U-Net 组件
# -----------------------------------------------------

class SinusoidalPositionalEmbedding(nn.Module):
    """
    时间步 t 的正弦位置嵌入 (Time Step t)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        # 计算 log(10000) / (dim/2 - 1)
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float().unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """
    带时间步条件注入的残差块。
    确保所有 3x3 卷积使用 padding=1，以保持空间尺寸不变。
    """
    def __init__(self, in_channels, out_channels, time_embed_dim, kernel_size=3):
        super().__init__()
        
        # 确保 padding 被正确设置 (对于 kernel_size=3, padding=1)
        padding = kernel_size // 2
        
        # 1. 主路径 (Conv -> GroupNorm -> SiLU)
        # 使用 padding=padding 确保输入和输出的 H/W 尺寸一致
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        
        # 2. 第二层 (Conv -> GroupNorm -> SiLU)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        
        # 3. 时间嵌入投影层
        self.time_mlp = nn.Linear(time_embed_dim, out_channels)
        
        # 4. 残差跳跃连接 (如果通道数不匹配则进行 1x1 卷积)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        """
        Args:
            x (Tensor): 输入特征图 (B, C_in, H, W)
            t_emb (Tensor): 时间和类别联合嵌入 (B, time_embed_dim)
            
        Returns:
            Tensor: 输出特征图 (B, C_out, H, W)
        """
        # 1. 第一层卷积和激活
        h = self.conv1(x)
        h = self.act1(self.norm1(h))
        
        # 2. 注入时间条件
        # 将 (B, time_embed_dim) 形状的 t_emb 投影并重塑为 (B, C_out, 1, 1)，以便通过广播进行加法
        time_emb_projected = self.time_mlp(t_emb)[:, :, None, None] 
        h = h + time_emb_projected
        
        # 3. 第二层卷积和归一化
        h = self.conv2(h)
        h = self.norm2(h)
        
        # 4. 残差连接: h + x
        # 注意: residual_conv(x) 确保 x 的通道数匹配 h 的通道数
        return self.act2(h + self.residual_conv(x))

# 紧接在 ResidualBlock(nn.Module) 定义之后
# --- ResidualBlock Sanity Check ---
try:
    temp_block = ResidualBlock(in_channels=32, out_channels=32, time_embed_dim=256)
    temp_x = torch.randn(2, 32, 28, 28)
    temp_t = torch.randn(2, 256)
    temp_out = temp_block(temp_x, temp_t)
    assert temp_out.shape[2] == 28 and temp_out.shape[3] == 28, \
        f"ResidualBlock is shrinking the image! Input 28x28, Output {temp_out.shape[2]}x{temp_out.shape[3]}"
    print("ResidualBlock check: PASS (Size kept).")
except AssertionError as e:
    print(f"ResidualBlock check: FAILED! {e}")
    # 强制退出，因为这是最可能的原因
    import sys; sys.exit(1)
except Exception as e:
    print(f"ResidualBlock check: ERROR! {e}")
# -----------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    """
    自注意力块。
    使用 1x1 卷积（kernel=1, padding=0）确保空间尺寸保持不变。
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        
        # 强制使用 padding=0 且 kernel_size=1
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, padding=0, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, cond_emb=None):
        # 添加cond_emb参数但不使用它，以兼容U-Net中的统一调用方式
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1) # (B, C, H, W)
        
        # 将空间维度合并到 batch 维度进行矩阵乘法
        q = q.flatten(2).transpose(1, 2) # (B, H*W, C)
        k = k.flatten(2)                # (B, C, H*W)
        v = v.flatten(2).transpose(1, 2) # (B, H*W, C)
        
        # Scaled Dot-Product Attention
        attn = (q @ k) * (q.shape[-1] ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = attn @ v                  # (B, H*W, C)
        out = out.transpose(1, 2).reshape(x.shape) # 还原回 (B, C, H, W)
        
        return x + self.proj_out(out) # 残差连接

class ConditionalUnet(nn.Module):
    """
    Conditional U-Net (修复了 nn.Sequential 导致的条件传递错误).
    - 使用 nn.ModuleList 嵌套结构，在 forward 中手动控制 cond_emb 传递。
    """
    def __init__(self, in_channels=1, base_channels=64, num_classes=10, time_emb_dim=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes

        # time and label embeddings
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        # channels configuration: ch0=64, ch1=128, ch2=256, ch3=512
        # ⚠️ 修正：只使用 3 个通道等级，ch0=64, ch1=128, ch2=256 (Bottleneck size)
        ch = [base_channels, base_channels * 2, base_channels * 4] 
        
        # initial conv
        self.init_conv = nn.Conv2d(in_channels, ch[0], 3, padding=1)

        # Encoder blocks (Down-sampling steps) - 只有 2 个 Down Stages
        self.downs = nn.ModuleList([
            nn.ModuleList([ # Stage 1: 28x28 -> 14x14
                ResidualBlock(ch[0], ch[0], time_emb_dim),
                ResidualBlock(ch[0], ch[1], time_emb_dim), 
                nn.MaxPool2d(2)
            ]),
            nn.ModuleList([ # Stage 2: 14x14 -> 7x7 (Bottleneck Input)
                ResidualBlock(ch[1], ch[2], time_emb_dim), 
                AttentionBlock(ch[2]),
                nn.MaxPool2d(2) # ⚠️ 修正：这是最后一个 MaxPool
            ]),
            # 移除第三个 Down Stage
        ])

        # Bottleneck (现在在 7x7 上运行)
        self.bottleneck = nn.ModuleList([
            ResidualBlock(ch[2], ch[2], time_emb_dim),
            AttentionBlock(ch[2]),
            ResidualBlock(ch[2], ch[2], time_emb_dim),
        ])

        # Decoder blocks (Up-sampling steps) - 只有 2 个 Up Stages
        self.ups = nn.ModuleList([
            nn.ModuleList([ # Stage 1 Up: 7x7 -> 14x14
                ResidualBlock(ch[2] + ch[1], ch[1], time_emb_dim), # Concat ch[2]+ch[1] -> ch[1]
                AttentionBlock(ch[1])
            ]),
            nn.ModuleList([ # Stage 2 Up: 14x14 -> 28x28
                ResidualBlock(ch[1] + ch[0], ch[0], time_emb_dim), # Concat ch[1]+ch[0] -> ch[0]
            ]),
            # 移除第三个 Up Stage
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # final conv
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], in_channels, 3, padding=1)
        )

    def forward(self, x, t, y_cond):
        """
        x: (B, C, H, W)
        t: (B,) long
        y_cond: LongTensor (B,) or FloatTensor (B, num_classes)
        """
        # embed time and label
        t_emb = self.time_mlp(t)
        if y_cond.dim() == 2 and y_cond.size(1) == self.num_classes:
            # 软标签/概率输入，进行矩阵乘法得到嵌入
            y_emb = y_cond @ self.label_emb.weight
        elif y_cond.dim() == 1 and y_cond.dtype == torch.long:
            # 硬标签索引输入，进行 Embedding 查找
            y_emb = self.label_emb(y_cond)
        else:
            raise ValueError("y_cond 必须是 LongTensor 索引 (B) 或 FloatTensor 概率 (B x C)。")
        cond_emb = t_emb + y_emb

        # initial conv
        x = self.init_conv(x)
        
        # Encoder (保存跳跃连接)
        skips = [x] 
        
        # ⚠️ 注意: 移除的 debug 代码，避免干扰
        for stage_idx, down_block_set in enumerate(self.downs):
            for module in down_block_set:
                if isinstance(module, (ResidualBlock, AttentionBlock)):
                    x = module(x, cond_emb)
                else:
                    x = module(x)
            
            # 检查 MaxPool 后的尺寸是否正确（14x14 或 7x7）
            skips.append(x)
            
        skips.pop() # 移除最后一个 down-sample 的输出 (它将进入 Bottleneck)
        
        # Bottleneck
        for block in self.bottleneck:
            x = block(x, cond_emb)

        # Decoder (反向使用跳跃连接)
        for up_block_set, skip in zip(self.ups, reversed(skips)):
            
            # 1. Upsample: 7->14, 14->28。所有尺寸均为 2 的幂次方，因此 Upsample 完美对齐。
            x = self.upsample(x)
            
            # 2. Safety Align/Concat Skip: 理论上 x.shape == skip.shape
            if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
                 # ⚠️ 理论上此裁剪不再发生，但保留作为安全措施
                 skip = skip[:, :, :x.shape[2], :x.shape[3]] 
            x = torch.cat([x, skip], dim=1) 
            
            # 3. Apply Up Blocks 
            for module in up_block_set:
                if isinstance(module, (ResidualBlock, AttentionBlock)):
                    x = module(x, cond_emb)
                else:
                    x = module(x) 

        # final conv
        out = self.final_conv(x)

        return out

# -----------------------------------------------------
# D. 辅助函数 (保持不变)
# -----------------------------------------------------

def gumbel_softmax_sample(logits, temperature):
    """计算 Gumbel Softmax 软分配."""
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    return F.softmax((logits + gumbel) / (temperature + 1e-9), dim=-1)

def get_mnist_loader(batch_size=128, train=True, shuffle=True, download=True):
    """标准的 MNIST DataLoader."""
    ds = datasets.MNIST('./data', train=train, download=download, transform=transforms.ToTensor())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

def plot_training_curves(metrics, outpath):
    """绘制训练指标曲线 (已适配 mDPM 指标)。"""
    plt.figure(figsize=(12, 5))
    
    # 损失和 ELBO (左侧 Y 轴)
    ax1 = plt.gca()
    if "Neg_ELBO" in metrics: ax1.plot(metrics["Neg_ELBO"], label="-ELBO", color='tab:blue')
    if "DPM_Loss" in metrics: ax1.plot(metrics["DPM_Loss"], label="DPM Loss", color='tab:orange')
    ax1.set_ylabel("Loss / -ELBO")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # 准确率和 NMI (右侧 Y 轴)
    ax2 = ax1.twinx()
    if "NMI" in metrics: ax2.plot(metrics["NMI"], label="NMI", color='tab:green', linestyle='--')
    if "PosteriorAcc" in metrics: ax2.plot(metrics["PosteriorAcc"], label="Acc", color='tab:red', linestyle='--')
    ax2.set_ylabel("Accuracy / NMI")
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper right')
    
    plt.xlabel("Epoch"); plt.title("mDPM Training Metrics")
    plt.grid(True); plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# =====================================================
# E. 数据加载辅助函数 (Semi-supervised Loader) (保持不变)
# =====================================================

def get_semi_loaders(cfg, labeled_per_class=100):
    """创建半监督学习所需的 labeled, unlabeled, 和 val loaders."""
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    labels = np.array(dataset.targets)
    labeled_idx, unlabeled_idx = [], []
    for c in range(cfg.num_classes):
        idx_c = np.where(labels == c)[0]
        count = min(labeled_per_class, len(idx_c))
        labeled_idx.extend(idx_c[:count])
        unlabeled_idx.extend(idx_c[count:])
        
    labeled_set = Subset(dataset, labeled_idx)
    unlabeled_set = Subset(dataset, unlabeled_idx)
    
    # 使用训练集的前 10% 作为验证集
    full_train_indices = list(range(len(dataset)))
    val_indices = full_train_indices[:int(0.1 * len(dataset))]
    val_set = Subset(dataset, val_indices)
    
    labeled_loader = DataLoader(labeled_set, batch_size=cfg.batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    return labeled_loader, unlabeled_loader, val_loader

# =====================================================
# F. 性能评估 (DPM 后验准确率) (保持不变，但需要 scipy.optimize.linear_sum_assignment)
# =====================================================

def evaluate_model(model, loader, cfg):
    """
    计算后验聚类标签与真实标签的对齐准确率和 NMI，使用 DPM 损失作为负对数似然的代理。
    返回: posterior_acc, cluster2label, nmi
    """
    # ⚠️ 注意: 此函数需要从 scipy.optimize 导入 linear_sum_assignment
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("WARNING: linear_sum_assignment from scipy not found. Returning 0 for Acc.")
        return 0.0, {}, 0.0 # 无法计算准确率

    model.eval()
    preds, ys_true = [], []
    
    # 使用固定的 t 作为评估时间步 (例如总时间步的一半)
    t_eval_val = cfg.timesteps // 2
    
    with torch.no_grad():
        for x_0, y_true in loader:
            x_0 = x_0.to(cfg.device)
            batch_size = x_0.size(0)
            
            # Monte Carlo 估计：采样噪声
            current_noise = torch.randn_like(x_0)
            current_t = torch.full((batch_size,), t_eval_val, device=cfg.device, dtype=torch.long)
            x_t = DPMForwardProcess(cfg.timesteps).q_sample(x_0, current_t, current_noise) # 需要实例化 DPMForwardProcess 或将其作为模型的一部分
            
            # 假设 model.registered_pi 存在并已初始化
            log_pi = torch.log(torch.ones(cfg.num_classes) / cfg.num_classes + 1e-8).unsqueeze(0).to(x_0.device)
            dpm_loss_proxies = []

            for k in range(cfg.num_classes):
                y_label_k = torch.full((batch_size,), k, device=x_0.device).long()
                
                # 计算条件 DPM 损失 L_t(k)
                pred_noise_k = model.cond_denoiser(x_t, current_t, y_label_k)
                dpm_loss_k = F.mse_loss(pred_noise_k, current_noise, reduction='none').view(batch_size, -1).mean(dim=1)
                
                # log P(x_0|x=k) proxy
                dpm_loss_proxies.append(-dpm_loss_k.unsqueeze(1))
            
            logits = torch.cat(dpm_loss_proxies, dim=1) + log_pi
            pred_cluster = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred_cluster)
            ys_true.append(y_true.numpy())

    preds = np.concatenate(preds)
    ys_true = np.concatenate(ys_true)
    n_classes = cfg.num_classes
    
    # 1. 计算 NMI
    nmi = NMI(ys_true, preds) 

    # 2. 计算准确率 (Alignment Calculation using Hungarian Algorithm)
    cost_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes): 
        for j in range(n_classes): 
            cost_matrix[i, j] = -np.sum((ys_true == i) & (preds == j))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster2label = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned_preds = np.array([cluster2label.get(p, 0) for p in preds])
    posterior_acc = np.mean(aligned_preds == ys_true)
    
    return posterior_acc, cluster2label, nmi

# =====================================================
# G. 采样和可视化辅助函数 (保持不变)
# =====================================================

def sample_and_save_dpm(denoiser, dpm_process, num_classes, out_path, device, n_per_class=10):
    """
    使用 DPM 逆过程从噪声 X_T 和类别条件 x 生成样本。
    """
    T = dpm_process.timesteps
    denoiser.eval()

    with torch.no_grad():
        # 1. 初始化噪声 x_T
        shape = (n_per_class * num_classes, dpm_process.image_channels, 28, 28)
        x_t = torch.randn(shape, device=device)
        
        # 2. 构造类别条件 (硬标签索引)
        y_cond_idx = torch.arange(num_classes).to(device).repeat_interleave(n_per_class).long()
        
        # 使用 LongTensor 索引作为条件 (ConditionalUnet 支持 LongTensor)
        y_cond = y_cond_idx 

        # 3. 逆向采样循环
        for i in reversed(range(1, T)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # 提取参数
            alpha_t = dpm_process._extract_t(dpm_process.alphas, t, shape)
            one_minus_alpha_t_bar = dpm_process._extract_t(dpm_process.sqrt_one_minus_alphas_cumprod, t, shape)
            
            # 预测噪声
            pred_noise = denoiser(x_t, t, y_cond)
            
            # 计算均值 mu_t-1 (使用 DPM 理论的去噪公式)
            mu_t = (x_t - (1 - alpha_t) / one_minus_alpha_t_bar * pred_noise) / alpha_t.sqrt()
            
            # 计算方差 sigma_t-1 (通常为 beta_t)
            sigma_t = dpm_process._extract_t(dpm_process.betas, t, shape).sqrt()
            
            if i > 1:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t) # 最后一个时间步不加噪声
                
            x_t = mu_t + sigma_t * noise # 更新 x_{t-1}

        final_samples = x_t.clamp(0, 1) # 确保输出在 [0, 1] 范围内
        save_image(final_samples, out_path, nrow=n_per_class, normalize=True)
    print(f"💾 Saved DPM samples to {out_path}")

def generate_visualizations(model, val_loader, metrics, cfg):
    """生成并保存最终可视化结果 (指标曲线和最终样本)。"""
    print("\n--- Generating Final Visualizations ---")
    output_dir = cfg.output_dir
    
    # 1. 绘制并保存训练指标曲线
    plot_training_curves(metrics, os.path.join(output_dir, "mDPM_training_metrics.png"))
    print(f"Saved training metrics to {os.path.join(output_dir, 'mDPM_training_metrics.png')}")

    # 2. 生成并保存最终的条件样本
    dpm_process = DPMForwardProcess(cfg.timesteps).to(cfg.device) # 临时实例化
    sample_and_save_dpm(model.cond_denoiser, dpm_process, cfg.num_classes,
                        os.path.join(output_dir, "mDPM_final_samples.png"), cfg.device)

# -----------------------------------------------------
# H. 运行时的形状检查 (使用修正后的 U-Net)
# -----------------------------------------------------

if __name__ == "__main__":
    print("==== Running ConditionalUnet shape check (Fixed) ====")

    # 确保运行环境中的 data 目录存在
    os.makedirs('./data', exist_ok=True)
    
    device = "cpu"
    
    # 实例化配置，用于获取 DPM 参数
    cfg = Config()
    
    model = ConditionalUnet(
        in_channels=cfg.image_channels,
        base_channels=32,   # 小一点速度更快
        num_classes=cfg.num_classes,
        time_emb_dim=cfg.unet_time_emb_dim
    ).to(device)

    # 随机输入，符合你 MNIST 的 (B=4, C=1, H=28, W=28)
    x = torch.randn(4, 1, 28, 28).to(device)
    t = torch.randint(0, cfg.timesteps, (4,), device=device)
    
    # y_cond 可以是 one-hot 或 long label —— 两个都测
    y_long = torch.randint(0, cfg.num_classes, (4,), device=device)
    y_onehot = F.one_hot(y_long, num_classes=cfg.num_classes).float()

    print("\nTest 1: Using LongTensor labels (y_cond = Long)")
    try:
        out1 = model(x, t, y_long)
        # 预期的输出形状：(B, C, H, W)
        expected_shape = torch.Size([4, 1, 28, 28])
        assert out1.shape == expected_shape, f"Expected {expected_shape}, but got {out1.shape}"
        print(" ✓ Passed. Output shape:", out1.shape)
    except Exception as e:
        print(" ✗ FAILED with LongTensor labels!")
        raise e

    print("\nTest 2: Using one-hot labels (y_cond = Float)")
    try:
        out2 = model(x, t, y_onehot)
        expected_shape = torch.Size([4, 1, 28, 28])
        assert out2.shape == expected_shape, f"Expected {expected_shape}, but got {out2.shape}"
        print(" ✓ Passed. Output shape:", out2.shape)
    except Exception as e:
        print(" ✗ FAILED with one-hot labels!")
        raise e

    print("\n==== Shape check finished successfully! The ConditionalUnet structure is now correct. ====")