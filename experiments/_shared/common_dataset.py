"""
共用：Stock / Medical 資料載入、切割（5-fold block CV 或 chronological）、前處理。

切割模式：
  - block_cv：5-fold block CV（1+2 折=歷史、3+4 折=新營運、第 5 折=測試）
  - chronological：依年份切割
      Stock  SPX/DJI/NDX：2000-2013 歷史、2014-2016 新營運、2017-2020 測試
      Medical（UCI Diabetes 1999-2008）：1999-2004 歷史、2005-2006 新營運、2007-2008 測試
"""
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent

STOCK_COLUMNS = [
    "Date", "Close", "High", "Low", "Open", "Volume",
    "Returns", "Log_Returns", "SMA_5", "SMA_20", "SMA_60",
    "Volatility_20", "RSI", "Future_Returns_20", "Crash_Event",
]

# 各資料集 chronological 切點（年份整數）
CHRONO_CUTS = {
    "stock": {"historical_end": 2013, "new_operating_end": 2016},
    "medical": {"historical_end": 2004, "new_operating_end": 2006},
}


def _load_stock(logger, ticker="spx", keep_year=False):
    """載入 Stock 資料（Kaggle/學長論文格式，skip 前兩列標題）。
    keep_year=True 時在 X 中保留 '_year' 欄位，供 chronological_split 使用。
    """
    path = project_root / f"data/raw/stock/stock_{ticker}.csv"
    if not path.exists():
        fallback_path = project_root / "data/raw/stock/stock_data.csv"
        if fallback_path.exists():
            path = fallback_path
        else:
            raise FileNotFoundError(f"Stock 資料不存在: {path}")
    df = pd.read_csv(path, skiprows=2, names=STOCK_COLUMNS, header=None)
    df = df.dropna(subset=["Crash_Event"])
    df["Crash_Event"] = df["Crash_Event"].astype(int)
    y = df["Crash_Event"]
    # 移除 Future_Returns_20（Crash_Event 由此衍生，若保留則造成資料洩漏 AUC≈1.0）
    drop_cols = ["Date", "Crash_Event", "Future_Returns_20"]
    X = df.drop(columns=drop_cols)
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    if keep_year:
        X = X.copy()
        X["_year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year.values
    logger.info(f"Stock 資料大小: {X.shape}, Crash 率: {y.mean()*100:.2f}%")
    return X, y


def _load_medical(logger, keep_year=False):
    """載入 Medical 資料（UCI Diabetes 130-US Hospitals 1999-2008，目標: 30天再入院）。
    keep_year=True 時在 X 中保留 '_year' 欄位，供 chronological_split 使用。

    資料來源: UCI ML Repository #296
    正類（<30d readmission）比例約 11%（少數類）。
    """
    uci_path = project_root / "data/raw/medical/diabetes130/diabetes130_medical.csv"
    synthetic_path = project_root / "data/raw/medical/synthetic/synthetic_medical_data.csv"

    if uci_path.exists():
        path = uci_path
        logger.info("使用真實 UCI Diabetes 130-US Hospitals 資料")
    elif synthetic_path.exists():
        path = synthetic_path
        logger.info("[WARNING] 使用合成 Medical 資料（建議換成真實 UCI 資料）")
    else:
        raise FileNotFoundError(
            "Medical 資料不存在，請執行:\n"
            "  python scripts/download_real_medical_data.py"
        )

    df = pd.read_csv(path)
    y = df["mortality"]
    X = df.drop(columns=["date", "mortality"])
    X = X.astype(np.float64)
    if keep_year:
        X = X.copy()
        X["_year"] = pd.to_datetime(df["date"], errors="coerce").dt.year.values
    logger.info(f"Medical 資料大小: {X.shape}, 正類率: {y.mean()*100:.2f}%")
    return X, y


def get_splits(dataset_name, logger, split_mode="block_cv"):
    """
    載入 Stock 或 Medical、切割、前處理。
    回傳 (X_hist_scaled, y_hist, X_new_scaled, y_new, X_test_scaled, y_test)。

    dataset_name: "stock" | "stock_spx" | "stock_dji" | "stock_ndx" | "medical"
    split_mode  : "block_cv" | "chronological" | "random"

    chronological 切點：
      stock   → 2000-2013 歷史 / 2014-2016 新營運 / 2017-2020 測試
      medical → 1999-2004 歷史 / 2005-2006 新營運 / 2007-2008 測試
    """
    import sys
    sys.path.insert(0, str(project_root))
    from src.data import DataSplitter, DataPreprocessor

    logger.info(f"步驟 1: 載入 {dataset_name} 資料  (split_mode={split_mode})")
    keep_year = (split_mode == "chronological")

    if dataset_name.startswith("stock"):
        parts = dataset_name.split("_")
        ticker = parts[1] if len(parts) > 1 else "spx"
        X, y = _load_stock(logger, ticker=ticker, keep_year=keep_year)
        ds_key = "stock"
    elif dataset_name == "medical":
        X, y = _load_medical(logger, keep_year=keep_year)
        ds_key = "medical"
    else:
        raise ValueError(f"不支援的 dataset_name: {dataset_name}")

    logger.info("步驟 2: 資料切割")
    splitter = DataSplitter()

    if split_mode == "chronological":
        cuts = CHRONO_CUTS[ds_key]
        logger.info(
            f"  chronological 切點: historical≤{cuts['historical_end']}, "
            f"new_operating≤{cuts['new_operating_end']}, testing>{cuts['new_operating_end']}"
        )
        splits = splitter.chronological_split(
            X, y,
            time_column="_year",
            historical_end=cuts["historical_end"],
            new_operating_end=cuts["new_operating_end"],
        )
        # 切完後移除輔助 _year 欄位
        splits = {
            k: (v[0].drop(columns=["_year"], errors="ignore"), v[1])
            for k, v in splits.items()
        }
    elif split_mode == "block_cv":
        splits = splitter.block_cv_split(
            X, y,
            n_folds=5,
            historical_folds=[1, 2],
            new_operating_folds=[3, 4],
            testing_fold=5,
        )
    else:
        from sklearn.model_selection import train_test_split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_hist, X_new, y_hist, y_new = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        splits = {
            "historical": (X_hist, y_hist),
            "new_operating": (X_new, y_new),
            "testing": (X_test, y_test),
        }
        logger.info(f"Historical: {len(X_hist)}, New: {len(X_new)}, Test: {len(X_test)}")

    X_hist, y_hist = splits["historical"]
    X_new, y_new = splits["new_operating"]
    X_test, y_test = splits["testing"]
    logger.info(
        f"  切割結果 → historical={len(X_hist)}, new_operating={len(X_new)}, testing={len(X_test)}"
    )

    logger.info("步驟 3: 資料前處理")
    preprocessor = DataPreprocessor()
    X_hist_clean = preprocessor.handle_missing_values(X_hist)
    X_new_clean = preprocessor.handle_missing_values(X_new)
    X_test_clean = preprocessor.handle_missing_values(X_test)
    X_hist_scaled, X_test_scaled = preprocessor.scale_features(
        X_hist_clean, X_test_clean, fit=True
    )
    X_new_scaled, _ = preprocessor.scale_features(X_new_clean, fit=False)

    return X_hist_scaled, y_hist, X_new_scaled, y_new, X_test_scaled, y_test
