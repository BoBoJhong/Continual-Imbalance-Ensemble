"""
自動下載 Stock 市場資料
使用 Yahoo Finance API 下載股票資料並轉換為適合訓練的格式
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# 添加專案路徑
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_logger

try:
    import yfinance as yf
    print("✓ yfinance 已安裝")
except ImportError:
    print("⚠ 需要安裝 yfinance")
    print("執行: pip install yfinance")
    sys.exit(1)


def download_stock_data(
    symbol: str = "^GSPC",  # S&P 500
    start_date: str = "2010-01-01",
    end_date: str = "2020-12-31",
    output_dir: str = "data/raw/stock"
):
    """
    下載股票市場資料
    
    Args:
        symbol: 股票代碼（^GSPC=S&P 500, ^DJI=Dow Jones）
        start_date: 開始日期
        end_date: 結束日期
        output_dir: 輸出目錄
    """
    logger = get_logger("StockDownloader", console=True, file=False)
    
    logger.info(f"下載 {symbol} 資料: {start_date} 到 {end_date}")
    
    # 下載資料
    data = yf.download(symbol, start=start_date, end=end_date, progress=True)
    
    if data.empty:
        logger.error(f"無法下載 {symbol} 的資料")
        return None
    
    logger.info(f"✓ 下載了 {len(data)} 筆資料")
    
    # 計算技術指標
    logger.info("計算技術指標...")
    
    # 價格變化
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # 移動平均
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_60'] = data['Close'].rolling(window=60).mean()
    
    # 波動率
    data['Volatility_20'] = data['Returns'].rolling(window=20).std()
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 創建崩盤事件標籤（目標變數）
    # 定義：未來 20 天內跌幅超過 5% 為崩盤事件
    data['Future_Returns_20'] = data['Close'].pct_change(20).shift(-20)
    data['Crash_Event'] = (data['Future_Returns_20'] < -0.05).astype(int)
    
    # 移除 NaN
    data = data.dropna()
    
    logger.info(f"✓ 處理後剩餘 {len(data)} 筆資料")
    logger.info(f"✓ 崩盤事件: {data['Crash_Event'].sum()} 次 ({data['Crash_Event'].mean()*100:.2f}%)")
    
    # 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / "stock_data.csv"
    data.to_csv(output_file)
    
    logger.info(f"✓ 資料已保存到: {output_file}")
    
    # 保存摘要
    summary = {
        'Symbol': symbol,
        'Date Range': f"{start_date} to {end_date}",
        'Total Records': len(data),
        'Crash Events': int(data['Crash_Event'].sum()),
        'Crash Rate': f"{data['Crash_Event'].mean()*100:.2f}%",
        'Features': data.columns.tolist()
    }
    
    summary_file = output_path / "data_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"✓ 摘要已保存到: {summary_file}")
    
    return data


def download_multiple_indices():
    """下載多個指數的資料進行比較"""
    logger = get_logger("StockDownloader", console=True, file=False)
    
    indices = {
        'SP500': '^GSPC',      # S&P 500
        'NASDAQ': '^IXIC',     # NASDAQ
        'DJI': '^DJI',         # Dow Jones
        'FTSE': '^FTSE',       # FTSE 100 (UK)
        'N225': '^N225'        # Nikkei 225 (Japan)
    }
    
    logger.info("下載多個指數資料...")
    
    for name, symbol in indices.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"下載 {name} ({symbol})")
        logger.info(f"{'='*60}")
        
        download_stock_data(
            symbol=symbol,
            start_date="2010-01-01",
            end_date="2020-12-31",
            output_dir=f"data/raw/stock/{name}"
        )


def main():
    """主函數"""
    print("="*60)
    print("Stock 市場資料下載工具")
    print("="*60)
    
    # 選項 1: 下載 S&P 500（推薦用於研究）
    print("\n選項 1: 下載 S&P 500 指數（推薦）")
    download_stock_data(
        symbol="^GSPC",
        start_date="2000-01-01",  # 20 年資料
        end_date="2020-12-31",
        output_dir="data/raw/stock"
    )
    
    # 選項 2: 下載多個指數（如果需要比較）
    # print("\n選項 2: 下載多個指數")
    # download_multiple_indices()
    
    print("\n" + "="*60)
    print("✅ 下載完成！")
    print("="*60)
    print("\n資料位置:")
    print("  - data/raw/stock/stock_data.csv")
    print("  - data/raw/stock/data_summary.txt")
    print("\n下一步:")
    print("  1. 檢查資料: python -c \"import pandas as pd; print(pd.read_csv('data/raw/stock/stock_data.csv').head())\"")
    print("  2. 開始實驗: 使用 DataLoader 載入資料")
    print("="*60)


if __name__ == "__main__":
    main()
