
# common.py
# 公共网络结构与工具函数（供 mVAE_unsup.py 与 mVAE_semisup.py 使用）

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI

# ----------------------
# Config（默认值，训练脚本可以覆盖）
# ----------------------
class Config:
    latent_dim = 16
    hidden_dim = 256
    num_classes = 10
    beta = 1.0
    lambda_entropy = 1.0
    # 半监督特有
    alpha_unlabeled = 1.0
    # gumbel
    init_gumbel_temp = 0.7
    min_gumbel_temp = 0.1
    gumbel_anneal_rate = 0.95
    current_gumbel_temp = init_gumbel_temp
    # opt / training
    lr = 1e-3
    batch_size = 128
    epochs = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = "./results_common"

# 确保输出目录存在
os.makedirs(Config.output_dir, exist_ok=True)

# ----------------------
# Encoder (与原文件一致)
# ----------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

# ----------------------
# Conditional Decoder (与原文件一致)
# ----------------------
class ConditionalDecoder(nn.Module):
    def __init__(self, latent_dim=16, num_classes=10, hidden_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim + num_classes, hidden_dim)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, 64 * 7 * 7), nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()
        )

    def forward(self, z, y_onehot):
        h = torch.cat([z, y_onehot], dim=1)
        return self.decoder(self.fc(h))

# ----------------------
# reparameterize & gumbel-softmax
# ----------------------
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def gumbel_softmax_sample(logits, temperature):
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    return F.softmax((logits + gumbel) / (temperature + 1e-9), dim=-1)

# ----------------------
# MNIST loader helper
# ----------------------
def get_mnist_loader(batch_size=128, train=True, shuffle=True, download=True):
    ds = datasets.MNIST('./data', train=train, download=download, transform=transforms.ToTensor())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

# ----------------------
# sample & save helper (class-conditional sampling)
# ----------------------
def sample_and_save(dec, latent_dim, num_classes, n_per_class=16, out_path="samples.png", device=None):
    device = device or torch.device("cpu")
    z = torch.randn(n_per_class, latent_dim).to(device)
    grids = []
    for k in range(num_classes):
        y = torch.full((n_per_class,), k, dtype=torch.long, device=device)
        y_onehot = F.one_hot(y, num_classes=num_classes).float()
        samples = dec(z, y_onehot)
        grids.append(samples)
    grid = torch.cat(grids, dim=0)
    save_image(grid, out_path, nrow=n_per_class)

# ----------------------
# plotting helper
# ----------------------
def plot_training_curves(metrics, outpath):
    plt.figure(figsize=(8,5))
    if "elbo" in metrics: plt.plot(metrics["elbo"], label="ELBO")
    if "recon" in metrics: plt.plot(metrics["recon"], label="Reconstruction")
    if "kl" in metrics: plt.plot(metrics["kl"], label="KL")
    if "acc" in metrics: plt.plot(metrics["acc"], label="Accuracy")
    plt.legend(); plt.xlabel("Epoch"); plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# ----------------------
# compute NMI via KMeans (增强鲁棒性)
# ----------------------
def compute_NMI_via_KMeans(Z, Y, n_clusters=10, n_init=20, min_cluster_ratio=0.5):
    """
    Z: numpy array (N, D) latent features
    Y: numpy array (N,) true labels
    返回 NMI，如果出现 collapse（形成的簇数小于阈值）则返回 0.0
    """
    try:
        if len(Z) == 0:
            return 0.0
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        y_pred = kmeans.fit_predict(Z)
        unique_clusters = len(np.unique(y_pred))
        if unique_clusters < max(1, int(n_clusters * min_cluster_ratio)):
            print(f"[Warning] KMeans collapse: only {unique_clusters}/{n_clusters} clusters formed.")
            return 0.0
        return float(NMI(Y, y_pred))
    except Exception as e:
        print(f"[Error in compute_NMI_via_KMeans]: {e}")
        return 0.0
