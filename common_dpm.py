# common_dpm_improved.py — 改进版: 增加 EMA + CFG + 更大容量
# 
# 改动总结:
#   1. [NEW] EMA 类 — 生成质量的关键
#   2. [NEW] CFG 支持 — 训练时随机 drop class (10%), 采样时做 guidance
#   3. [MOD] base_channels 32→64, 增加模型容量
#   4. [MOD] ResidualBlock 增加 Dropout 防过拟合
#   5. [保留] get_time_weight sin 曲线 (聚类核心, 不动)

import os
import copy
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
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "./mDPM_results"
        self.batch_size = 64
        self.final_epochs = 100
        self.optuna_epochs = 10
        self.lr = 1e-5
        self.labeled_per_class = 0

        self.alpha_unlabeled = 1
        self.posterior_sample_steps = 5

        # Gumbel Softmax
        self.initial_gumbel_temp = 1.0
        self.min_gumbel_temp = 0.5
        self.gumbel_anneal_rate = 0.99
        self.current_gumbel_temp = self.initial_gumbel_temp

        self.num_classes = 10
        self.timesteps = 1000
        self.image_channels = 1

        # [FIX] 保持 32 — 容量太大会让模型忽略 class condition
        # 生成质量主要靠 EMA + CFG, 不靠容量
        self.unet_base_channels = 32
        self.unet_time_emb_dim = 256

        # [NEW] EMA 参数
        self.ema_decay = 0.9999          # EMA 衰减率
        self.ema_start_epoch = 5         # 前几个 epoch 不用 EMA (让模型先学)

        # [NEW] CFG 参数
        self.cfg_dropout_prob = 0.1      # 训练时 10% 概率 drop class condition
        self.cfg_guidance_scale = 2.0    # 采样时 guidance 强度 (1.0=无, 2.0=温和, 3.0+=强)

        os.makedirs(self.output_dir, exist_ok=True)


# -----------------------------------------------------
# [NEW] EMA — 指数移动平均
# -----------------------------------------------------
class EMA:
    """
    指数移动平均: 维护模型参数的平滑版本
    采样时用 EMA 权重, 训练时用原始权重
    这是扩散模型生成质量的 #1 关键因素
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        # 冻结 shadow 的梯度
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """每个 training step 后调用"""
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1 - self.decay)

    def get_model(self):
        """返回 EMA 模型 (用于采样和评估)"""
        return self.shadow


# -----------------------------------------------------
# B. DPM 前向过程 (无改动)
# -----------------------------------------------------
class DPMForwardProcess(nn.Module):
    def __init__(self, timesteps=1000, schedule='linear', image_channels=1):
        super().__init__()
        self.timesteps = timesteps
        self.image_channels = image_channels

        if schedule == 'linear':
            self.register_buffer('betas', torch.linspace(1e-4, 0.02, timesteps))
        else:
            raise NotImplementedError(f"Schedule {schedule} not implemented.")

        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        self.register_buffer('posterior_variance', self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ac = self._extract_t(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_omc = self._extract_t(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_ac * x_0 + sqrt_omc * noise

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
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# 保留原始 sin 曲线时间权重 (聚类核心)
def get_time_weight(t, max_steps=1000):
    t_norm = t.float() / max_steps
    base = 1.0
    peak = 3.0
    weights = base + peak * torch.sin(t_norm * torch.pi)
    return weights.view(-1, 1)


class ResidualBlock(nn.Module):
    """
    Combined AdaGN ResidualBlock
    [MOD] 增加 Dropout (防止大模型过拟合)
    """
    def __init__(self, in_channels, out_channels, time_embed_dim, class_embed_dim=None,
                 kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

        # [MOD] Dropout
        self.dropout = nn.Dropout(dropout)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels * 2)
        )

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond_emb, c_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)

        style = self.time_mlp(cond_emb)[:, :, None, None]
        scale, shift = style.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.act1(h)
        h = self.dropout(h)  # [MOD]

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

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
    """
    改进版 UNet:
    [MOD] base_channels 默认 64 (从32翻倍)
    [NEW] CFG: 支持 uncond_flag, 当 y_cond 全零时 = unconditional
    """
    def __init__(self, in_channels=1, base_channels=64, num_classes=10, time_emb_dim=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.num_classes = num_classes
        self.in_channels = in_channels

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
                ResidualBlock(ch[0], ch[0], time_emb_dim, time_emb_dim),
                ResidualBlock(ch[0], ch[1], time_emb_dim, time_emb_dim),
                nn.MaxPool2d(2)
            ]),
            nn.ModuleList([
                ResidualBlock(ch[1], ch[2], time_emb_dim, time_emb_dim),
                AttentionBlock(ch[2]),
                nn.MaxPool2d(2)
            ]),
        ])

        self.bottleneck = nn.ModuleList([
            ResidualBlock(ch[2], ch[2], time_emb_dim, time_emb_dim),
            AttentionBlock(ch[2]),
            ResidualBlock(ch[2], ch[2], time_emb_dim, time_emb_dim),
        ])

        self.ups = nn.ModuleList([
            nn.ModuleList([
                ResidualBlock(ch[2] + ch[1], ch[1], time_emb_dim, time_emb_dim),
                AttentionBlock(ch[1])
            ]),
            nn.ModuleList([
                ResidualBlock(ch[1] + ch[0], ch[0], time_emb_dim, time_emb_dim),
            ]),
        ])
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, ch[0]),
            nn.SiLU(),
            nn.Conv2d(ch[0], in_channels, 3, padding=1)
        )

    def forward(self, x, t, y_cond):
        """
        y_cond: LongTensor, one-hot FloatTensor, 或全零 (unconditional)
        全零 one-hot → y_emb 全零 → 纯 time conditioning = unconditional
        """
        t_emb = self.time_mlp(t)

        if y_cond.dim() == 1:
            y_onehot = F.one_hot(y_cond, self.num_classes).float()
        elif y_cond.dim() == 2 and y_cond.size(1) == self.num_classes:
            y_onehot = y_cond
        else:
            raise ValueError("y_cond format error")

        y_emb = y_onehot @ self.label_emb.weight

        # 时间步动态权重 (保留)
        w_t = get_time_weight(t, max_steps=1000)
        y_emb = y_emb * w_t

        cond_emb = t_emb + y_emb

        # UNet forward
        x = self.init_conv(x)
        skips = [x]

        for down_block_set in self.downs:
            for module in down_block_set:
                if isinstance(module, ResidualBlock):
                    x = module(x, cond_emb)
                elif isinstance(module, AttentionBlock):
                    x = module(x)
                else:
                    x = module(x)
            skips.append(x)

        skips.pop()

        for block in self.bottleneck:
            if isinstance(block, ResidualBlock):
                x = block(x, cond_emb)
            else:
                x = block(x)

        for up_block_set, skip in zip(self.ups, reversed(skips)):
            x = self.upsample(x)
            if x.shape[2] != skip.shape[2]:
                skip = skip[:, :, :x.shape[2], :x.shape[3]]
            x = torch.cat([x, skip], dim=1)
            for module in up_block_set:
                if isinstance(module, ResidualBlock):
                    x = module(x, cond_emb)
                elif isinstance(module, AttentionBlock):
                    x = module(x)
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


def get_semi_loaders(cfg, labeled_per_class=None):
    if labeled_per_class is None:
        labeled_per_class = cfg.labeled_per_class

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    labels = np.array(dataset.targets)

    labeled_idx, unlabeled_idx = [], []

    if labeled_per_class == -1:
        print("Dataset Mode: Fully Supervised")
        labeled_idx = list(range(len(dataset)))
        unlabeled_idx = []
    else:
        for c in range(cfg.num_classes):
            idx_c = np.where(labels == c)[0]
            count = min(labeled_per_class, len(idx_c))
            labeled_idx.extend(idx_c[:count])
            unlabeled_idx.extend(idx_c[count:])

    if len(labeled_idx) > 0:
        labeled_set = Subset(dataset, labeled_idx)
        labeled_loader = DataLoader(labeled_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    else:
        labeled_loader = None

    if len(unlabeled_idx) > 0:
        unlabeled_set = Subset(dataset, unlabeled_idx)
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    else:
        unlabeled_loader = None

    val_indices = list(range(len(dataset)))[:int(0.1 * len(dataset))]
    val_set = Subset(dataset, val_indices)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)

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
    print("==== Shape check (Improved) ====")
    os.makedirs('./data', exist_ok=True)
    device = "cpu"
    cfg = Config()

    model = ConditionalUnet(
        in_channels=cfg.image_channels,
        base_channels=cfg.unet_base_channels,
        num_classes=cfg.num_classes,
        time_emb_dim=cfg.unet_time_emb_dim
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M")

    x = torch.randn(4, 1, 28, 28).to(device)
    t = torch.randint(0, cfg.timesteps, (4,), device=device)

    # Test conditional
    y_long = torch.randint(0, cfg.num_classes, (4,), device=device)
    out1 = model(x, t, y_long)
    assert out1.shape == (4, 1, 28, 28)
    print("✓ Conditional forward OK")

    # Test unconditional (全零 one-hot = no class info)
    y_uncond = torch.zeros(4, cfg.num_classes, device=device)
    out2 = model(x, t, y_uncond)
    assert out2.shape == (4, 1, 28, 28)
    print("✓ Unconditional forward OK")

    # Test EMA
    ema = EMA(model, decay=0.9999)
    ema.update(model)
    print("✓ EMA OK")

    print("==== All checks passed! ====")