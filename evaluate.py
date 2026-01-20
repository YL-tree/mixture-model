import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from common_dpm import Config
from mDPM import mDPM_SemiSup, evaluate_model # 确保引用正确

def verify_model_performance(model_path="mDPM_results_semisupervised/best_model.pt"):
    # 1. 准备配置和环境
    cfg = Config()
    device = cfg.device
    print(f"🔍 Loading model from {model_path}...")

    # 2. 加载模型
    model = mDPM_SemiSup(cfg).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # 兼容只保存了 state_dict 或保存了完整 checkpoint 的情况
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    model.eval()

    # 3. 加载真正的 Test Set (10k images)
    print("📦 Loading MNIST Test Set...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 4. 运行评估 (复用你现有的 evaluate_model)
    # 注意：evaluate_model 内部会跑 t=500 的采样，比较慢，请耐心等待
    print("🚀 Running inference on Test Set (this may take a while)...")
    acc, cluster2label, nmi = evaluate_model(model, test_loader, cfg)

    print("\n" + "="*40)
    print(f"✅ Final Test Results:")
    print(f"   Accuracy (ACC): {acc*100:.2f}%")
    print(f"   NMI Score:      {nmi:.4f}")
    print("="*40 + "\n")

    # 5. (可选) 绘制混淆矩阵
    # 为了画混淆矩阵，我们需要拿到原始的 preds 和 targets
    # 这里简单重新手动跑一遍获取数据（为了代码独立性）
    print("🎨 Generating Confusion Matrix...")
    all_preds = []
    all_targets = []
    
    # 简化版快速推理 (只用 t=500, repeat=1)
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            batch_size = x.size(0)
            
            # 快速预测：只采样一次，t=500
            t_val = 500
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            noise = torch.randn_like(x)
            x_t = model.dpm_process.q_sample(x, t, noise)
            
            mse_scores = []
            for k in range(cfg.num_classes):
                y_vec = torch.nn.functional.one_hot(torch.full((batch_size,), k, device=device), cfg.num_classes).float()
                pred = model.cond_denoiser(x_t, t, y_vec)
                loss = torch.nn.functional.mse_loss(pred, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                mse_scores.append(loss.unsqueeze(1))
            
            mse_scores = torch.cat(mse_scores, dim=1) # (B, 10)
            raw_preds = torch.argmin(mse_scores, dim=1).cpu().numpy()
            
            # 使用 evaluate_model 算出来的映射关系对齐标签
            aligned_preds = [cluster2label.get(p, p) for p in raw_preds]
            
            all_preds.extend(aligned_preds)
            all_targets.extend(y.numpy())

    # 绘制
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Acc: {acc:.2%})')
    plt.savefig('confusion_matrix_test.png')
    print("💾 Confusion matrix saved to 'confusion_matrix_test.png'")

if __name__ == "__main__":
    verify_model_performance()