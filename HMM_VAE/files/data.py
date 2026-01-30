import akshare as ak
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime
import time      # <--- 1. 必须导入 time 模块
import random    # <--- 2. 导入 random 模块（可选，为了让间隔更自然）



def download_csi500_data(start_date="20160101", end_date="20231231", cache_path="csi500_dataset.csv"):
    """
    下载并处理中证500成分股数据（对齐填充版）
    :param start_date: 建议从 20160101 开始
    :param end_date: 结束日期
    :param cache_path: 保存路径
    :return: 清洗后的 DataFrame (Index=Date, Columns=Stock_Codes)
    """
    
    if os.path.exists(cache_path):
        print(f"检测到本地缓存 {cache_path}，正在读取...")
        df_merged = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"数据读取完毕，形状: {df_merged.shape}")
        return df_merged

    print("未检测到缓存，开始从 Akshare 下载...")
    
    # --- 1. 获取基准交易日历 (Benchmark) ---
    # 为了对齐，我们需要一个“标准的时间轴”。通常用大盘指数（如沪深300或上证指数）的交易日作为标准。
    try:
        print("正在获取基准交易日历（以上证指数为例）...")
        # 000001 是上证指数
        index_df = ak.stock_zh_index_daily(symbol="sh000001")
        index_df['date'] = pd.to_datetime(index_df['date'])
        
        # 截取我们需要的时间段
        mask = (index_df['date'] >= pd.to_datetime(start_date)) & (index_df['date'] <= pd.to_datetime(end_date))
        benchmark_dates = index_df.loc[mask, 'date'].sort_values().unique()
        print(f"基准交易日共 {len(benchmark_dates)} 天。")
    except Exception as e:
        print(f"获取基准日历失败: {e}")
        # 如果获取失败，fallback 到后面 concat 自动生成的索引，但可能不准
        benchmark_dates = None

    # --- 2. 获取成分股列表 ---
    try:
        print("正在获取中证500成分股列表...")
        index_stock_cons = ak.index_stock_cons(symbol="000905")
        
        # 兼容不同列名逻辑
        if 'variety' in index_stock_cons.columns:
            stock_codes = index_stock_cons['variety'].tolist()
        elif 'symbol' in index_stock_cons.columns:
            stock_codes = index_stock_cons['symbol'].tolist()
        elif 'stock_code' in index_stock_cons.columns:
            stock_codes = index_stock_cons['stock_code'].tolist()
        else:
            stock_codes = index_stock_cons.iloc[:, 0].tolist()
            
        print(f"共获取到 {len(stock_codes)} 只成分股。")
        
        # 确保只取前500只（虽然中证500理论上是500，但有时候接口会返回调整名单导致数量不对）
        # 我们只取前500个以固定维度
        stock_codes = stock_codes[:500] 

    except Exception as e:
        print(f"获取成分股列表失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- 3. 循环下载 ---
    data_dict = {}
    error_codes = []
    
    print(f"开始下载成分股数据...")
    pbar = tqdm(stock_codes)
    for code in pbar:
        time.sleep(random.uniform(0.1, 0.3)) # 稍微快一点，AKshare最近限制还好
        str_code = str(code).zfill(6)
        pbar.set_description(f"下载 {str_code}")
        
        try:
            df = ak.stock_zh_a_hist(
                symbol=str_code, 
                period="daily", 
                start_date=start_date, 
                end_date=end_date, 
                adjust="hfq"
            )
            
            if df.empty:
                continue
                
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            
            # 计算收益率
            # 注意：pct_change 会导致第一天变成 NaN
            pct_change = df['收盘'].pct_change()
            
            # 去除极值 (Winsorize): 防止新股上市首日涨幅 200% 这种数据破坏模型
            # 简单的 Clip 策略：限制在 -10.5% 到 +10.5% 之间 (考虑到科创板是20%，可以设宽一点，比如 0.2)
            pct_change = pct_change.clip(lower=-0.2, upper=0.2)
            
            data_dict[str_code] = pct_change
            
        except Exception as e:
            error_codes.append(str_code)
            continue
            
    if len(data_dict) == 0:
        raise RuntimeError("所有股票下载均失败！")

    # --- 4. 数据对齐与填充 (核心修改) ---
    print("正在进行数据对齐与填充...")
    
    # 使用基准日历创建一个空的 DataFrame
    if benchmark_dates is not None:
        df_base = pd.DataFrame(index=benchmark_dates)
        # 将数据合并到基准日历上 (Left Join)
        # 这样可以保证即使某天所有股票都停牌，日期行也会保留（全为NaN）
        # 从而保证时间序列的连续性，这对 HMM 很重要
        df_merged = pd.concat([df_base, pd.DataFrame(data_dict)], axis=1, join='outer')
        df_merged = df_merged.loc[benchmark_dates] # 确保只保留交易日
    else:
        df_merged = pd.concat(data_dict, axis=1)

    # 填充策略:
    # 1. 此时 df_merged 中的 NaN 可能代表：
    #    a. 股票未上市
    #    b. 股票已退市
    #    c. 股票停牌
    #    d. 节假日（虽然用了基准日历应该避免了，但防止万一）
    
    # 简单粗暴但有效的策略：填充 0
    # 含义：你持有现金或停牌股票，当日收益为 0
    df_merged = df_merged.fillna(0.0)

    # --- 5. 最终检查 ---
    print(f"清洗完毕。最终形状: {df_merged.shape}")
    
    # 如果列数不足 500 (比如只有490只下载成功)，补齐剩余的列为全0
    # 这对于神经网络输入维度固定很重要 (比如 VAE 输入必须是 500)
    current_cols = df_merged.shape[1]
    if current_cols < 500:
        print(f"警告: 下载成功的股票只有 {current_cols} 只，正在补齐至 500 只...")
        for i in range(500 - current_cols):
            df_merged[f'dummy_{i}'] = 0.0
            
    # 如果列数超过 500，截取前 500
    if df_merged.shape[1] > 500:
        df_merged = df_merged.iloc[:, :500]

    # 保存
    df_merged.to_csv(cache_path)
    print(f"数据已保存至 {cache_path}")
    
    return df_merged

# --- 测试运行 ---
if __name__ == "__main__":
    # 建议选取最近 2-3 年的数据
    # 注意：Akshare 下载 500 只股票可能需要 10-20 分钟，请耐心等待
    df = download_csi500_data(start_date="20160101", end_date="20231231")
    
    # 打印前几行看看
    if df is not None:
        print("\n数据预览 (前5行, 前5列):")
        print(df.iloc[:5, :5])