import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# 引入项目中的定义
from common_dpm import Config, get_semi_loaders
from mDPM import mDPM_SemiSup, set_seed, evaluate_model, sample_and_save_dpm

# 引入 FID/IS
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    HAS_METRICS = True
except ImportError:
    print("❌ Error: torchmetrics not installed.")
    print("   Run: pip install torchmetrics torch-fidelity")
    exit()

def run_full_evaluation(checkpoint_path, num_fid_samples=5000):
    print(f"\n" + "="*50)
    print(f"🚀 STARTING FULL EVALUATION")
    print(f"   Model: {checkpoint_path}")
    print(f"="*50 + "\n")
    
    # 1. 初始化
    set_seed(42)
    cfg = Config()
    cfg.alpha_unlabeled = 1.0
    cfg.labeled_per_class = 0
    device = cfg.device
    
    # 2. 加载模型
    if not os.path.exists(checkpoint_path):
        print(f"❌ File not found: {checkpoint_path}")
        return

    model = mDPM_SemiSup(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    train_epoch = checkpoint.get('epoch', '?')
    print(f"✅ Model loaded (Trained for {train_epoch} epochs)")

    # 3. 准备数据
    _, _, val_loader = get_semi_loaders(cfg)

    # ==========================================
    # Phase 1: 聚类性能评估 & 获取映射
    # ==========================================
    print(f"\n[Phase 1] Evaluating Clustering Performance...")
    acc, cluster_mapping, nmi = evaluate_model(model, val_loader, cfg)
    
    print(f"   🏆 Accuracy: {acc:.4f}")
    print(f"   🔗 NMI:      {nmi:.4f}")
    print(f"   🗺️  Mapping found: {cluster_mapping}")

    # ==========================================
    # Phase 2: 生成可视化图像 (按顺序)
    # ==========================================
    print(f"\n[Phase 2] Generating Ordered Visualization (0-9)...")
    vis_save_path = "eval_vis_ordered.png"
    
    sample_and_save_dpm(
        model.cond_denoiser, 
        model.dpm_process, 
        cfg.num_classes, 
        vis_save_path, 
        device, 
        n_per_class=10, 
        cluster_mapping=cluster_mapping # <--- 传入刚才算出的映射
    )
    print(f"   🖼️  Saved to: {vis_save_path} (Check this file!)")

    # ==========================================
    # Phase 3: 计算 FID / IS
    # ==========================================
    print(f"\n[Phase 3] Calculating Generative Metrics ({num_fid_samples} samples)...")
    
    # 初始化指标
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    is_metric = InceptionScore(feature=2048, normalize=False).to(device)
    
    # 3.1 收集真图 (Real Images)
    print(f"   📸 Collecting Real Images...")
    val_dataset = val_loader.dataset
    if len(val_dataset) < num_fid_samples:
        num_fid_samples = len(val_dataset)
    
    eval_loader_large = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)
    real_cnt = 0
    for x, _ in tqdm(eval_loader_large, desc="   Real"):
        if real_cnt >= num_fid_samples: break
        x = x.to(device)
        x_rgb = x.repeat(1, 3, 1, 1)
        x_uint8 = ((x_rgb.clamp(-1, 1) * 0.5 + 0.5) * 255).to(torch.uint8)
        fid_metric.update(x_uint8, real=True)
        real_cnt += x.size(0)

    # 3.2 生成假图 (Fake Images)
    print(f"   🎨 Generating Fake Images...")
    fake_cnt = 0
    batch_size = 100
    
    with torch.no_grad():
        pbar = tqdm(total=num_fid_samples, desc="   Fake")
        while fake_cnt < num_fid_samples:
            curr_bs = min(batch_size, num_fid_samples - fake_cnt)
            
            # 生成
            x_fake = torch.randn(curr_bs, 1, 28, 28, device=device)
            y_gen = torch.randint(0, cfg.num_classes, (curr_bs,), device=device).long()
            y_vec = F.one_hot(y_gen, cfg.num_classes).float()
            
            for i in reversed(range(0, cfg.timesteps)):
                t = torch.full((curr_bs,), i, device=device, dtype=torch.long)
                
                # Extract helpers
                dpm = model.dpm_process
                alpha_t = dpm._extract_t(dpm.alphas, t, x_fake.shape)
                one_minus = dpm._extract_t(dpm.sqrt_one_minus_alphas_cumprod, t, x_fake.shape)
                post_var = dpm._extract_t(dpm.posterior_variance, t, x_fake.shape)
                
                pred_noise = model.cond_denoiser(x_fake, t, y_vec)
                mu = (x_fake - (1 - alpha_t) / one_minus * pred_noise) / alpha_t.sqrt()
                
                if i > 0:
                    x_fake = mu + post_var.sqrt() * torch.randn_like(x_fake)
                else:
                    x_fake = mu
            
            # 处理 & 更新
            fake_rgb = x_fake.repeat(1, 3, 1, 1)
            fake_uint8 = ((fake_rgb.clamp(-1, 1) * 0.5 + 0.5) * 255).to(torch.uint8)
            
            fid_metric.update(fake_uint8, real=False)
            is_metric.update(fake_uint8)
            
            fake_cnt += curr_bs
            pbar.update(curr_bs)
        pbar.close()

    # 3.3 计算结果
    print(f"   🧮 Computing final scores...")
    fid_score = fid_metric.compute().item()
    is_score, is_std = is_metric.compute()
    
    print("\n" + "="*40)
    print("FINAL REPORT")
    print("-" * 40)
    print(f"Model Epoch: {train_epoch}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"NMI Score:   {nmi:.4f}")
    print("-" * 40)
    print(f"FID Score:   {fid_score:.4f}  (Lower is better)")
    print(f"IS Score:    {is_score:.4f} ± {is_std:.4f}")
    print("="*40)

if __name__ == "__main__":
    # 指向您的最佳模型路径
    CKPT_PATH = "./final_training_FID/best_model.pt"
    
    # 运行
    run_full_evaluation(CKPT_PATH, num_fid_samples=5000)