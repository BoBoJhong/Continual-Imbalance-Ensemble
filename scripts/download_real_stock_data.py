"""
下載真實 S&P 500 歷史資料（2000-2020），繞過 yfinance SSL 問題。
資料來源: Stooq（免費、無需帳號、直接 CSV 下載）
計算技術指標並輸出符合 common_dataset.py 格式的 stock_data.csv

Run: python scripts/download_real_stock_data.py
"""
import sys
import ssl
import urllib.request
import io
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent

# Stooq 下載 S&P 500（^SPX），2000-01-01 到 2020-12-31
STOOQ_URL = "https://stooq.com/q/d/l/?s=^spx&d1=20000101&d2=20201231&i=d"

def download_csv_no_ssl(url: str) -> pd.DataFrame:
    """使用 urllib 繞過 SSL 驗證（Python 3.14 相容）"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    return pd.read_csv(io.StringIO(raw))


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """計算技術指標，符合 STOCK_COLUMNS 格式。"""
    df = df.sort_values("Date").reset_index(drop=True)
    df["Returns"] = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_60"] = df["Close"].rolling(60).mean()
    df["Volatility_20"] = df["Returns"].rolling(20).std()
    df["RSI"] = calc_rsi(df["Close"], 14)
    df["Future_Returns_20"] = df["Close"].pct_change(20).shift(-20)
    df["Crash_Event"] = (df["Future_Returns_20"] < -0.05).astype(int)

    # 去掉 NaN（前期指標計算期間）
    df = df.dropna(subset=["SMA_60", "Future_Returns_20"])
    return df


def main():
    out_dir = project_root / "data" / "raw" / "stock"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "stock_data.csv"

    print("=" * 60)
    print("下載真實 S&P 500 資料（Stooq，2000-2020）")
    print("=" * 60)

    try:
        print(f"URL: {STOOQ_URL}")
        raw_df = download_csv_no_ssl(STOOQ_URL)
        print(f"Raw rows: {len(raw_df)}, columns: {raw_df.columns.tolist()}")
    except Exception as e:
        print(f"[ERROR] Stooq 下載失敗: {e}")
        print("嘗試備用來源 (^GSPC)...")
        backup_url = "https://stooq.com/q/d/l/?s=%5Egspc&d1=20000101&d2=20201231&i=d"
        raw_df = download_csv_no_ssl(backup_url)
        print(f"Raw rows: {len(raw_df)}, columns: {raw_df.columns.tolist()}")

    # Stooq 的欄位是 Date, Open, High, Low, Close, Volume
    raw_df.columns = [c.strip() for c in raw_df.columns]
    df = raw_df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # 計算指標
    df = build_stock_features(df)

    print(f"\n處理後筆數: {len(df)}")
    print(f"期間: {df.Date.iloc[0]} 至 {df.Date.iloc[-1]}")
    print(f"Crash_Event 比例: {df.Crash_Event.mean()*100:.1f}%")

    # 以符合 download_stock_data.py 原始標題格式儲存
    # （common_dataset.py 用 skiprows=2，所以前兩列是 header）
    header1 = ",".join(["Ticker"] + ["^GSPC"] * (len(df.columns) - 1))
    header2 = ",".join(df.columns.tolist())

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(header1 + "\n")
        f.write(header2 + "\n")
        df.to_csv(f, index=False, header=False)

    print(f"\nSaved to: {out_file}")
    print("=" * 60)
    print("Done! 現在可以重跑實驗:")
    print("  python experiments/05_stock_baseline_ensemble.py")
    print("  python experiments/07_stock_des.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
