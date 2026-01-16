# common_dpm.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI

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
        # 自动检测可用设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "./mDPM_results_unsupervised"
        self.batch_size = 16
        self.final_epochs = 50 
        self.optuna_epochs = 10 
        self.lr = 2e-4                    # 学习率
        self.labeled_per_class = 100      # 每类用于监督学习的样本数 (半监督)
        
        # ---------------------
        # PVEM 框架权重
        # ---------------------
        self.alpha_unlabeled = 1        # 无标签数据损失的权重
        self.lambda_entropy = 5.0         # 熵惩罚项的权重 (Minimization)
        
        # ---------------------
        # Gumbel Softmax 退火参数
        # ---------------------
        self.initial_gumbel_temp = 2.0    
        self.min_gumbel_temp = 0.1   

        self.gumbel_anneal_rate = 0.9   
        self.current_gumbel_temp = self.initial_gumbel_temp 
        
        # ---------------------
        # 模型结构和 DPM 参数
        # ---------------------
        self.num_classes = 10             # MNIST
        self.timesteps = 1000             # 扩散总时间步 T
        self.image_channels = 1           # MNIST
        
        # U-Net 参数
        self.unet_base_channels = 32      
        self.unet_time_emb_dim = 256      
        
        os.makedirs(self.output_dir, exist_ok=True)

# -----------------------------------------------------
# B. DPM 前向过程
# -----------------------------------------------------

class DPMForwardProcess(nn.Module):
    def __init__(self, timesteps: int = 1000, schedule: str = 'linear', image_channels: int = 1):
        super().__init__()
        self.timesteps = timesteps
        self.image_channels = image_channels

        if schedule == 'linear':
            self.register_buffer('betas', torch.linspace(1e-4, 0.02, timesteps))
        else:
            raise NotImplementedError(f"Schedule {schedule} not implemented.")

        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # 辅助变量
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('posterior_variance', self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self._extract_t(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_t(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def _extract_t(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t.to(a.device)) 
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# -----------------------------------------------------
# C. U-Net 组件
# -----------------------------------------------------

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.float().unsqueeze(1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

# common_dpm.py 中的 ResidualBlock 类

class ResidualBlock(nn.Module):
    """
    改进版 ResidualBlock：使用 AdaGN (Adaptive Group Norm) 替代简单的加法。
    增强了条件 y 对生成过程的控制力，大幅提升分类时的判别度。
    """
    def __init__(self, in_channels, out_channels, time_embed_dim, kernel_size=3):
        super().__init__()
        
        padding = kernel_size // 2
        
        # 1. 正常的卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        
        # [核心修改 1] 输出维度翻倍 (out_channels * 2)
        # 因为我们要同时预测 Scale (乘法系数) 和 Shift (加法偏置)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels * 2)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        """
        x: (B, C, H, W)
        t_emb: (B, time_embed_dim) -> 包含了 time + label 的信息
        """
        # --- 主干路径 ---
        
        # 1. 第一层卷积
        h = self.conv1(x)
        
        # 2. [核心修改 2] AdaGN 注入机制
        # 先归一化
        h = self.norm1(h)
        
        # 计算 Scale 和 Shift
        # style shape: (B, 2*C) -> (B, 2*C, 1, 1)
        style = self.time_mlp(t_emb)[:, :, None, None]
        scale, shift = style.chunk(2, dim=1) # 分割成两半
        
        # 执行仿射变换: h = h * (1 + scale) + shift
        # 这种乘法交互让条件 y 能强力控制特征图
        h = h * (1 + scale) + shift
        
        # 激活函数
        h = self.act1(h)
        
        # 3. 第二层卷积
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        
        # --- 残差连接 ---
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, padding=0, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, cond_emb=None):
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        v = v.flatten(2).transpose(1, 2)
        attn = (q @ k) * (q.shape[-1] ** -0.5)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(x.shape)
        return x + self.proj_out(out)

class ConditionalUnet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_classes=10, time_emb_dim=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        ch = [base_channels, base_channels * 2, base_channels * 4]
        self.init_conv = nn.Conv2d(in_channels, ch[0], 3, padding=1)

        self.downs = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock(ch[0], ch[0], time_emb_dim),
                ResidualBlock(ch[0], ch[1], time_emb_dim), 
                nn.MaxPool2d(2)
            ]),
            nn.ModuleList([
                ResidualBlock(ch[1], ch[2], time_emb_dim), 
                AttentionBlock(ch[2]),
                nn.MaxPool2d(2)
            ]),
        ])

        self.bottleneck = nn.ModuleList([
            ResidualBlock(ch[2], ch[2], time_emb_dim),
            AttentionBlock(ch[2]),
            ResidualBlock(ch[2], ch[2], time_emb_dim),
        ])

        self.ups = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock(ch[2] + ch[1], ch[1], time_emb_dim),
                AttentionBlock(ch[1])
            ]),
            nn.ModuleList([
                ResidualBlock(ch[1] + ch[0], ch[0], time_emb_dim),
            ]),
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], in_channels, 3, padding=1)
        )

    def forward(self, x, t, y_cond):
        t_emb = self.time_mlp(t)
        if y_cond.dim() == 2 and y_cond.size(1) == self.num_classes:
            y_emb = y_cond @ self.label_emb.weight
        elif y_cond.dim() == 1:
            y_emb = self.label_emb(y_cond)
        else:
            raise ValueError("y_cond format error")
        cond_emb = t_emb + y_emb

        x = self.init_conv(x)
        skips = [x]
        
        for down_block_set in self.downs:
            for module in down_block_set:
                if isinstance(module, (ResidualBlock, AttentionBlock)):
                    x = module(x, cond_emb)
                else:
                    x = module(x)
            skips.append(x)
            
        skips.pop()
        
        for block in self.bottleneck:
            x = block(x, cond_emb)

        for up_block_set, skip in zip(self.ups, reversed(skips)):
            x = self.upsample(x)
            if x.shape[2] != skip.shape[2]:
                 skip = skip[:, :, :x.shape[2], :x.shape[3]] 
            x = torch.cat([x, skip], dim=1) 
            for module in up_block_set:
                if isinstance(module, (ResidualBlock, AttentionBlock)):
                    x = module(x, cond_emb)
                else:
                    x = module(x) 

        return self.final_conv(x)

# -----------------------------------------------------
# D. 辅助函数
# -----------------------------------------------------

def gumbel_softmax_sample(logits, temperature):
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    return F.softmax((logits + gumbel) / (temperature + 1e-9), dim=-1)



# common_dpm.py

def get_semi_loaders(cfg, labeled_per_class=None):
    if labeled_per_class is None: 
        labeled_per_class = cfg.labeled_per_class

    # 1. 预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    labels = np.array(dataset.targets)
    
    # 2. 划分数据索引
    labeled_idx, unlabeled_idx = [], []
    
    # 特殊情况：如果 labeled_per_class 为 -1，表示全监督（所有数据都有标签）
    if labeled_per_class == -1:
        print("Dataset Mode: Fully Supervised (All Data Labeled)")
        labeled_idx = list(range(len(dataset)))
        unlabeled_idx = [] # 无标签为空
    else:
        # 正常半监督或无监督逻辑
        for c in range(cfg.num_classes):
            idx_c = np.where(labels == c)[0]
            
            # 如果 labeled_per_class 是 0，这里 count 就是 0
            count = min(labeled_per_class, len(idx_c))
            
            labeled_idx.extend(idx_c[:count])
            unlabeled_idx.extend(idx_c[count:])

    # 3. 构造 DataLoader
    
    # --- 处理有标签数据 ---
    if len(labeled_idx) > 0:
        labeled_set = Subset(dataset, labeled_idx)
        labeled_loader = DataLoader(labeled_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    else:
        # 关键修改：如果没有标签数据，直接返回 None
        # 这会触发 run_training_session 进入 "UNSUPERVISED" 模式
        labeled_loader = None
        
    # --- 处理无标签数据 ---
    if len(unlabeled_idx) > 0:
        unlabeled_set = Subset(dataset, unlabeled_idx)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    else:
        unlabeled_loader = None
        
    # --- 处理验证集 ---
    val_indices = list(range(len(dataset)))[:int(0.1 * len(dataset))]
    val_set = Subset(dataset, val_indices)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
    
    # 打印数据集统计信息，防止配错
    print(f"Dataset Split -> Labeled: {len(labeled_idx)} | Unlabeled: {len(unlabeled_idx)}")
    
    return labeled_loader, unlabeled_loader, val_loader

def plot_training_curves(metrics, outpath):
    plt.figure(figsize=(12, 5))
    ax1 = plt.gca()
    if "Neg_ELBO" in metrics: ax1.plot(metrics["Neg_ELBO"], label="-ELBO", color='tab:blue')
    if "DPM_Loss" in metrics: ax1.plot(metrics["DPM_Loss"], label="DPM Loss", color='tab:orange')
    ax1.set_ylabel("Loss")
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    if "NMI" in metrics: ax2.plot(metrics["NMI"], label="NMI", color='tab:green', linestyle='--')
    if "PosteriorAcc" in metrics: ax2.plot(metrics["PosteriorAcc"], label="Acc", color='tab:red', linestyle='--')
    ax2.set_ylabel("Metric")
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

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