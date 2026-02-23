"""
共用：Stock / Medical 資料載入、切割（5-fold block CV）、前處理。
"""
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent

STOCK_COLUMNS = [
    "Date", "Close", "High", "Low", "Open", "Volume",
    "Returns", "Log_Returns", "SMA_5", "SMA_20", "SMA_60",
    "Volatility_20", "RSI", "Future_Returns_20", "Crash_Event",
]


def _load_stock(logger):
    """載入 Stock 資料（Kaggle/學長論文格式，skip 前兩列標題）。"""
    path = project_root / "data/raw/stock/stock_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"Stock 資料不存在: {path}")
    df = pd.read_csv(path, skiprows=2, names=STOCK_COLUMNS, header=None)
    df = df.dropna(subset=["Crash_Event"])
    df["Crash_Event"] = df["Crash_Event"].astype(int)
    y = df["Crash_Event"]
    # 移除 Future_Returns_20（Crash_Event 由此衍生，若保留則造成資料洩漏 AUC≈1.0）
    X = df.drop(columns=["Date", "Crash_Event", "Future_Returns_20"])
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(X.median())
    logger.info(f"Stock 資料大小: {X.shape}, Crash 率: {y.mean()*100:.2f}%")
    return X, y


def _load_medical(logger):
    """載入 Medical 資料（UCI Diabetes 130-US Hospitals 1999-2008，目標: 30天再入院）。
    
    資料來源: UCI ML Repository #296
    https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008
    正類（<30d readmission）比例約 11%（少數類）。
    """
    # Priority: real UCI data > synthetic fallback
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
            f"Medical 資料不存在，請執行:\n"
            f"  python scripts/download_real_medical_data.py"
        )
    
    df = pd.read_csv(path)
    y = df["mortality"]
    X = df.drop(columns=["date", "mortality"])
    X = X.astype(np.float64)
    logger.info(f"Medical 資料大小: {X.shape}, 正類率: {y.mean()*100:.2f}%")
    return X, y


def get_splits(dataset_name, logger, split_mode="block_cv"):
    """
    載入 Stock 或 Medical、切割、前處理，回傳 (X_hist_scaled, y_hist, X_new_scaled, y_new, X_test_scaled, y_test)。

    dataset_name: "stock" | "medical"
    split_mode: "block_cv" | "random"
    """
    import sys
    sys.path.insert(0, str(project_root))
    from src.data import DataSplitter, DataPreprocessor

    logger.info(f"步驟 1: 載入 {dataset_name} 資料")
    if dataset_name == "stock":
        X, y = _load_stock(logger)
    elif dataset_name == "medical":
        X, y = _load_medical(logger)
    else:
        raise ValueError(f"不支援的 dataset_name: {dataset_name}")

    logger.info("步驟 2: 資料切割")
    splitter = DataSplitter()
    if split_mode == "block_cv":
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
