import torch
import os
import json
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as NMI

# 导入你的模型定义
from common_dpm import Config, get_semi_loaders
from mDPM import mDPM_SemiSup  # 确保 mDPM.py 中包含 mDPM_SemiSup 类

# ==========================================
# 核心评估函数 (包含 Low-T 和 Monte Carlo 策略)
# ==========================================
def robust_evaluate(model, loader, cfg):
    """
    V3 强力评估版：
    1. 专注于低时间步 (Low t strategy) - 信号最强区域
    2. 蒙特卡洛多次采样 (Monte Carlo Averaging) - 消除噪声方差
    """
    model.eval()
    preds, ys_true = [], []
    
    # [关键策略 1] 只评估中低时间步
    # 避开 t > 500 的高噪声区域
    eval_timesteps = [50, 100, 150, 200] 
    
    # [关键策略 2] 对每个 t 重复采样次数
    n_repeats = 5 
    
    print(f"🔍 开始评估: TimeSteps={eval_timesteps}, Repeats={n_repeats}")

    with torch.no_grad():
        for i, (x_0, y_true) in enumerate(loader):
            x_0 = x_0.to(cfg.device)
            batch_size = x_0.size(0)
            
            # (Batch, Num_Classes)
            cumulative_mse = torch.zeros(batch_size, cfg.num_classes, device=cfg.device)
            
            for t_val in eval_timesteps:
                mse_t_sum = torch.zeros(batch_size, cfg.num_classes, device=cfg.device)
                
                for _ in range(n_repeats):
                    # 1. 采样噪声
                    noise = torch.randn_like(x_0)
                    current_t = torch.full((batch_size,), t_val, device=cfg.device, dtype=torch.long)
                    x_t = model.dpm_process.q_sample(x_0, current_t, noise)
                    
                    # 2. 对每个类别计算 MSE
                    for k in range(cfg.num_classes):
                        y_onehot_k = F.one_hot(torch.full((batch_size,), k, device=x_0.device),
                                               num_classes=cfg.num_classes).float()
                        
                        pred_noise = model.cond_denoiser(x_t, current_t, y_onehot_k)
                        
                        # 计算单个样本的 MSE
                        loss_k = F.mse_loss(pred_noise, noise, reduction='none').view(batch_size, -1).mean(dim=1)
                        mse_t_sum[:, k] += loss_k
                
                # 平均该时间步的 MSE 并累加
                cumulative_mse += (mse_t_sum / n_repeats)

            # MSE 越小，似然概率越大。Logits = -MSE
            log_pi = torch.log(model.registered_pi + 1e-8).unsqueeze(0).to(x_0.device)
            final_logits = -cumulative_mse + log_pi
            
            pred_cluster = torch.argmax(final_logits, dim=1).cpu().numpy()
            preds.append(pred_cluster)
            ys_true.append(y_true.numpy())
            
            if i % 10 == 0:
                print(f"   处理 Batch {i}...")

    preds = np.concatenate(preds)
    ys_true = np.concatenate(ys_true)
    
    # --- 指标计算 ---
    nmi = NMI(ys_true, preds)
    
    # 匈牙利算法对齐 (哪怕是全监督也可以跑一下，确认是否对齐)
    n_classes = cfg.num_classes
    cost_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cost_matrix[i, j] = -np.sum((ys_true == i) & (preds == j))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster2label = {int(c): int(l) for c, l in zip(col_ind, row_ind)}
    aligned_preds = np.array([cluster2label.get(p, 0) for p in preds])
    
    # 原始准确率 (假设类别 ID 一一对应)
    raw_acc = np.mean(preds == ys_true)
    # 对齐后准确率
    posterior_acc = np.mean(aligned_preds == ys_true)
    
    return raw_acc, posterior_acc, nmi, cluster2label

# ==========================================
# 主加载逻辑
# ==========================================
def load_and_run():
    # 1. 初始化配置
    cfg = Config()
    
    # [重要] 必须与训练时的配置一致，否则模型权重加载会报错
    # 如果你在训练时修改了 batch_size 或 channels，这里也要改
    cfg.unet_base_channels = 32  # 请确认你训练时是用 32 还是 64
    cfg.batch_size = 32          # 评估时 Batch 可以小一点以防显存溢出
    
    # 模型路径
    model_path = os.path.join(cfg.output_dir, "mDPM_best_model.pt") 
    # 或者如果你是在全监督文件夹下：
    # model_path = "./mDPM_results_supervised/mDPM_best_model.pt" 
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        return

    print(f"📂 正在加载模型: {model_path}")
    print(f"⚙️  设备: {cfg.device}")

    # 2. 初始化模型架构
    model = mDPM_SemiSup(cfg).to(cfg.device)
    
    # 3. 加载权重
    try:
        checkpoint = torch.load(model_path, map_location=cfg.device)
        model.load_state_dict(checkpoint)
        print("✅ 模型权重加载成功！")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("提示：请检查 cfg.unet_base_channels 是否与训练时一致。")
        return

    # 4. 获取验证集数据
    # get_semi_loaders 返回 (labeled, unlabeled, val_loader)
    _, _, val_loader = get_semi_loaders(cfg)
    
    # 5. 运行评估
    print("\n🚀 开始运行 Robust Evaluate...")
    raw_acc, post_acc, nmi, mapping = robust_evaluate(model, val_loader, cfg)
    
    print("\n" + "="*30)
    print(f"📊 最终评估结果")
    print("="*30)
    print(f"Raw Accuracy (无对齐):  {raw_acc:.4f}")
    print(f"Aligned Accuracy (对齐后): {post_acc:.4f}")
    print(f"NMI Score:              {nmi:.4f}")
    print("-" * 30)
    print(f"类别映射 (Cluster -> Label): {mapping}")
    print("="*30)

if __name__ == "__main__":
    load_and_run()