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
            "  python scripts/data/download_real_medical_data.py"
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


# ---------------------------------------------------------------------------
# 醫療年份切割：固定 test=2007-2008，彈性 old_end_year
# ---------------------------------------------------------------------------

MEDICAL_YEAR_SPLITS = [
    ("split_1+7", 1999),   # 1999      Old (1yr) / 2000-2006 New (7yr)
    ("split_2+6", 2000),   # 1999-2000 Old (2yr) / 2001-2006 New (6yr)
    ("split_3+5", 2001),   # 1999-2001 Old (3yr) / 2002-2006 New (5yr)
    ("split_4+4", 2002),   # 1999-2002 Old (4yr) / 2003-2006 New (4yr)
    ("split_5+3", 2003),   # 1999-2003 Old (5yr) / 2004-2006 New (3yr)
    ("split_6+2", 2004),   # 1999-2004 Old (6yr) / 2005-2006 New (2yr)
    ("split_7+1", 2005),   # 1999-2005 Old (7yr) / 2006      New (1yr)
]


def get_medical_year_split(logger, old_end_year: int, return_years: bool = False):
    """
    固定 test = 2007-2008，依 old_end_year 切割：
      Old  = year <= old_end_year
      New  = old_end_year+1 <= year <= 2006
      Test = 2007 <= year <= 2008

    回傳 (X_old, y_old, X_new, y_new, X_test, y_test)（未縮放，由實驗腳本決定縮放方式）
    若 return_years=True，額外回傳 (year_old, year_new, year_test)。
    """
    X, y = _load_medical(logger, keep_year=True)
    year = X["_year"].astype(int)

    mask_old  = year <= old_end_year
    mask_new  = (year > old_end_year) & (year <= 2006)
    mask_test = (year >= 2007) & (year <= 2008)

    X_old,  y_old  = X[mask_old].drop(columns=["_year"]),  y[mask_old]
    X_new,  y_new  = X[mask_new].drop(columns=["_year"]),  y[mask_new]
    X_test, y_test = X[mask_test].drop(columns=["_year"]), y[mask_test]
    year_old = year[mask_old].reset_index(drop=True)
    year_new = year[mask_new].reset_index(drop=True)
    year_test = year[mask_test].reset_index(drop=True)

    logger.info(
        f"  Old={len(X_old)}({y_old.mean()*100:.1f}%+)  "
        f"New={len(X_new)}({y_new.mean()*100:.1f}%+)  "
        f"Test={len(X_test)}({y_test.mean()*100:.1f}%+)"
    )
    base_ret = (
        X_old.reset_index(drop=True),  y_old.reset_index(drop=True),
        X_new.reset_index(drop=True),  y_new.reset_index(drop=True),
        X_test.reset_index(drop=True), y_test.reset_index(drop=True),
    )
    if return_years:
        return base_ret + (
            year_old,
            year_new,
            year_test,
        )
    return base_ret


# ---------------------------------------------------------------------------
# 股市年份切割：固定 test=2017-2020，彈性 old_end_year（SPX，S&P 500）
# ---------------------------------------------------------------------------

STOCK_BASE_YEAR = 2001   # 訓練窗口 2001-2016（16 年）
STOCK_TRAIN_END_YEAR = 2016


def _build_stock_year_splits(base_year: int, train_end: int):
    total_years = train_end - base_year + 1
    return [
        (
            f"split_{old_years}+{total_years - old_years}",
            base_year + old_years - 1,
        )
        for old_years in range(1, total_years - 1)
    ]


STOCK_YEAR_SPLITS = _build_stock_year_splits(STOCK_BASE_YEAR, STOCK_TRAIN_END_YEAR)


def get_stock_year_split(logger, old_end_year: int, ticker: str = "spx", return_years: bool = False):
    """
    固定 test = 2017-2020，訓練窗口 2001-2016（16 年），依 old_end_year 切割
    （預設使用 S&P 500 / SPX）：
      Old  = 2001 <= year <= old_end_year
      New  = old_end_year+1 <= year <= 2016
      Test = 2017 <= year <= 2020

    回傳 (X_old, y_old, X_new, y_new, X_test, y_test)（未縮放）
    若 return_years=True，額外回傳 (year_old, year_new, year_test)。
    """
    X, y = _load_stock(logger, ticker=ticker, keep_year=True)
    year = X["_year"].astype(int)

    mask_old  = (year >= 2001) & (year <= old_end_year)
    mask_new  = (year > old_end_year) & (year <= 2016)
    mask_test = (year >= 2017) & (year <= 2020)

    X_old,  y_old  = X[mask_old].drop(columns=["_year"]),  y[mask_old]
    X_new,  y_new  = X[mask_new].drop(columns=["_year"]),  y[mask_new]
    X_test, y_test = X[mask_test].drop(columns=["_year"]), y[mask_test]
    year_old = year[mask_old].reset_index(drop=True)
    year_new = year[mask_new].reset_index(drop=True)
    year_test = year[mask_test].reset_index(drop=True)

    logger.info(
        f"  Old={len(X_old)}({y_old.mean()*100:.2f}%Crash)  "
        f"New={len(X_new)}({y_new.mean()*100:.2f}%Crash)  "
        f"Test={len(X_test)}({y_test.mean()*100:.2f}%Crash)"
    )
    base_ret = (
        X_old.reset_index(drop=True),  y_old.reset_index(drop=True),
        X_new.reset_index(drop=True),  y_new.reset_index(drop=True),
        X_test.reset_index(drop=True), y_test.reset_index(drop=True),
    )
    if return_years:
        return base_ret + (
            year_old,
            year_new,
            year_test,
        )
    return base_ret
