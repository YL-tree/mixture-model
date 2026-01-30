"""
数据适配器 - 使用您的 data.py 加载数据

这个脚本将您的 data.py 适配到训练代码中
"""

import numpy as np
import pandas as pd
import sys
import os

# 导入您的数据加载函数
from data import download_csi500_data


def load_data_for_training(cache_path="csi500_dataset.csv", 
                           start_date="20160101", 
                           end_date="20231231"):
    """
    使用您的 data.py 加载数据,并转换为训练所需格式
    
    返回:
        returns_data: (n_days, n_stocks=500) numpy数组,日收益率
        dates: 日期索引
    """
    print("="*60)
    print("使用您的 data.py 加载数据")
    print("="*60)
    
    # 1. 加载数据 (如果有缓存会直接读取)
    df = download_csi500_data(
        start_date=start_date,
        end_date=end_date,
        cache_path=cache_path
    )
    
    if df is None:
        raise RuntimeError("数据加载失败!")
    
    # 2. 数据检查
    print(f"\n✓ 数据加载成功")
    print(f"  形状: {df.shape}")
    print(f"  日期范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"  股票数量: {df.shape[1]}")
    
    # 检查缺失值
    missing_ratio = df.isnull().sum().sum() / df.size
    print(f"  缺失值比例: {missing_ratio:.2%}")
    
    if missing_ratio > 0.1:
        print("  ⚠️  警告: 缺失值超过10%,建议检查数据质量")
    
    # 3. 统计信息
    returns_flat = df.values.flatten()
    returns_flat = returns_flat[~np.isnan(returns_flat)]  # 去除NaN
    
    print(f"\n数据统计:")
    print(f"  平均收益率: {returns_flat.mean():.6f}")
    print(f"  标准差: {returns_flat.std():.6f}")
    print(f"  最小值: {returns_flat.min():.6f}")
    print(f"  最大值: {returns_flat.max():.6f}")
    print(f"  中位数: {np.median(returns_flat):.6f}")
    
    # 检查是否有极端值
    extreme_pos = np.sum(returns_flat > 0.15)
    extreme_neg = np.sum(returns_flat < -0.15)
    if extreme_pos > 0 or extreme_neg > 0:
        print(f"  ⚠️  极端值: {extreme_pos}个>15%, {extreme_neg}个<-15%")
    
    # 4. 转换为numpy数组
    returns_data = df.values.astype(np.float32)
    dates = df.index
    
    # 5. 最终检查
    assert returns_data.shape[1] == 500, f"维度错误: 期望500列,实际{returns_data.shape[1]}列"
    
    print(f"\n✓ 数据准备完成,可以开始训练!")
    print("="*60 + "\n")
    
    return returns_data, dates


def quick_data_check(cache_path="csi500_dataset.csv"):
    """
    快速检查已缓存的数据
    """
    if not os.path.exists(cache_path):
        print(f"❌ 未找到缓存文件: {cache_path}")
        print(f"请先运行 data.py 下载数据,或提供正确的缓存路径")
        return False
    
    print(f"✓ 发现缓存文件: {cache_path}")
    
    # 读取并检查
    df = pd.read_csv(cache_path, index_col=0, parse_dates=True, nrows=5)
    print(f"  形状(前5行): {df.shape}")
    print(f"  列数: {df.shape[1]}")
    
    if df.shape[1] != 500:
        print(f"  ⚠️  警告: 列数不是500,是{df.shape[1]}")
    else:
        print(f"  ✓ 列数正确(500)")
    
    return True


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 方法1: 快速检查缓存
    print("方法1: 检查缓存文件")
    quick_data_check("csi500_dataset.csv")
    
    print("\n" + "="*60)
    
    # 方法2: 加载完整数据
    print("方法2: 加载完整数据用于训练")
    try:
        returns_data, dates = load_data_for_training(
            cache_path="csi500_dataset.csv"
        )
        
        print(f"返回数据类型: {type(returns_data)}")
        print(f"返回数据形状: {returns_data.shape}")
        print(f"返回数据dtype: {returns_data.dtype}")
        
        # 显示前5天,前5只股票
        print(f"\n前5天,前5只股票的收益率:")
        print(returns_data[:5, :5])
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
