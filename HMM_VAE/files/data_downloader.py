"""
CSI500成分股数据下载和预处理
使用akshare库下载A股数据
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def get_csi500_constituents():
    """
    获取中证500成分股列表
    返回: DataFrame with columns ['code', 'name']
    """
    try:
        # 获取中证500成分股
        df = ak.index_stock_cons_csindex(symbol="000905")
        print(f"获取到 {len(df)} 只中证500成分股")
        return df
    except Exception as e:
        print(f"获取成分股失败: {e}")
        return None


def download_stock_data(stock_code, start_date, end_date, adjust="qfq"):
    """
    下载单只股票的历史数据
    
    stock_code: 股票代码 (6位数字)
    start_date: 开始日期 "YYYYMMDD"
    end_date: 结束日期 "YYYYMMDD"
    adjust: 复权方式 "qfq"前复权/"hfq"后复权/""不复权
    
    返回: DataFrame with ['date', 'close', ...]
    """
    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        
        if df is not None and len(df) > 0:
            df['stock_code'] = stock_code
            return df[['日期', '收盘', 'stock_code']].rename(columns={'日期': 'date', '收盘': 'close'})
        else:
            return None
    except Exception as e:
        print(f"下载 {stock_code} 失败: {e}")
        return None


def download_csi500_data(start_date="20160101", end_date="20231231", min_data_ratio=0.95):
    """
    下载中证500成分股数据并清理
    
    start_date: 开始日期
    end_date: 结束日期
    min_data_ratio: 最小数据完整度(删除缺失数据过多的股票)
    
    返回: DataFrame - (n_days, n_stocks) 收益率矩阵
    """
    print(f"开始下载CSI500数据: {start_date} 到 {end_date}")
    
    # 1. 获取成分股列表
    constituents = get_csi500_constituents()
    if constituents is None:
        print("无法获取成分股列表,使用示例数据")
        return generate_sample_data(start_date, end_date)
    
    # 2. 下载每只股票数据
    all_data = []
    success_count = 0
    
    for idx, row in constituents.iterrows():
        stock_code = row['成分券代码']
        stock_name = row['成分券名称']
        
        print(f"下载 [{idx+1}/{len(constituents)}] {stock_code} - {stock_name}", end='\r')
        
        df = download_stock_data(stock_code, start_date, end_date, adjust="qfq")
        
        if df is not None and len(df) > 0:
            all_data.append(df)
            success_count += 1
    
    print(f"\n成功下载 {success_count}/{len(constituents)} 只股票")
    
    if len(all_data) == 0:
        print("没有成功下载任何数据,使用示例数据")
        return generate_sample_data(start_date, end_date)
    
    # 3. 合并数据
    print("合并数据...")
    df_combined = pd.concat(all_data, ignore_index=True)
    
    # 4. 转换为宽表格式 (日期 x 股票)
    df_pivot = df_combined.pivot(index='date', columns='stock_code', values='close')
    df_pivot = df_pivot.sort_index()
    
    print(f"原始数据形状: {df_pivot.shape}")
    
    # 5. 数据清理
    # 5.1 删除缺失数据过多的股票
    missing_ratio = df_pivot.isnull().sum() / len(df_pivot)
    valid_stocks = missing_ratio[missing_ratio < (1 - min_data_ratio)].index
    df_clean = df_pivot[valid_stocks].copy()
    
    print(f"删除缺失数据过多的股票后: {df_clean.shape}")
    
    # 5.2 前向填充缺失值 (停牌日用前一日收盘价)
    df_clean = df_clean.fillna(method='ffill')
    
    # 5.3 删除仍有缺失的行
    df_clean = df_clean.dropna()
    
    print(f"清理后数据形状: {df_clean.shape}")
    
    # 6. 计算收益率
    returns = df_clean.pct_change().dropna()
    
    # 7. 异常值处理 (clip到合理范围)
    returns = returns.clip(-0.15, 0.15)  # 限制在±15%
    
    # 8. 删除全为0的行 (可能是异常交易日)
    returns = returns[(returns != 0).any(axis=1)]
    
    print(f"最终收益率矩阵: {returns.shape}")
    print(f"日期范围: {returns.index[0]} 到 {returns.index[-1]}")
    print(f"股票数量: {returns.shape[1]}")
    
    # 保存数据
    returns.to_csv("csi500_returns.csv")
    print("数据已保存到 csi500_returns.csv")
    
    return returns


def generate_sample_data(start_date, end_date, n_stocks=500):
    """
    生成示例数据(用于测试)
    """
    print("生成示例数据用于测试...")
    
    # 转换日期
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    n_days = (end - start).days
    
    # 生成日期序列(只包含交易日,约252天/年)
    dates = pd.date_range(start, end, freq='B')  # 'B' = business day
    
    # 生成收益率数据
    # 模拟3个市场状态: 牛市/熊市/震荡
    np.random.seed(42)
    returns = np.zeros((len(dates), n_stocks))
    
    # 状态切换
    state_durations = [60, 100, 80, 120, 90]  # 每个状态持续天数
    state_means = [0.001, -0.001, 0.0005, 0.002, -0.0015]  # 每个状态的平均收益
    
    current_idx = 0
    for duration, mean in zip(state_durations, state_means):
        end_idx = min(current_idx + duration, len(dates))
        returns[current_idx:end_idx] = np.random.randn(end_idx - current_idx, n_stocks) * 0.02 + mean
        current_idx = end_idx
        if current_idx >= len(dates):
            break
    
    # 剩余部分
    if current_idx < len(dates):
        returns[current_idx:] = np.random.randn(len(dates) - current_idx, n_stocks) * 0.02
    
    df = pd.DataFrame(returns, index=dates, columns=[f'stock_{i:03d}' for i in range(n_stocks)])
    
    print(f"示例数据形状: {df.shape}")
    return df


def load_saved_data(filepath="csi500_returns.csv"):
    """
    加载已保存的数据
    """
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"加载数据: {df.shape}")
        return df
    except Exception as e:
        print(f"加载失败: {e}")
        return None


if __name__ == "__main__":
    # 下载数据
    returns_df = download_csi500_data(
        start_date="20180101",
        end_date="20231231",
        min_data_ratio=0.95
    )
    
    # 查看统计信息
    print("\n=== 数据统计 ===")
    print(f"平均收益率: {returns_df.mean().mean():.6f}")
    print(f"收益率标准差: {returns_df.std().mean():.6f}")
    print(f"最大收益率: {returns_df.max().max():.4f}")
    print(f"最小收益率: {returns_df.min().min():.4f}")
