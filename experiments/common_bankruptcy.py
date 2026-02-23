"""
共用：Bankruptcy 資料載入、切割（含 5-fold block CV、chronological）、前處理。
- 支援 Taiwan 1999-2009（data.csv）或 US 1999-2018（american_bankruptcy_dataset.csv）
- 切割 b：5-fold CV（1+2 折=歷史、3+4 折=新營運、第 5 折=測試）
- 切割 a（僅 US 有年份時）：1999-2011 歷史、2012-2014 新營運、2015-2018 測試
"""
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
BANKRUPTCY_DIR = project_root / "data" / "raw" / "bankruptcy"
US_CSV = BANKRUPTCY_DIR / "american_bankruptcy_dataset.csv"
TAIWAN_CSV = BANKRUPTCY_DIR / "data.csv"


def _load_taiwan(logger):
    """Taiwan Economic Journal 破產資料（約 1999-2009），無年份欄。"""
    df = pd.read_csv(TAIWAN_CSV)
    if "Bankrupt?" in df.columns:
        y = df["Bankrupt?"]
        X = df.drop(columns=["Bankrupt?"])
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    return X, y


def _load_us_1999_2018(logger):
    """US 破產資料 1999-2018（sowide/Kaggle），含 fyear。"""
    df = pd.read_csv(US_CSV)
    if "status_label" not in df.columns:
        raise ValueError("US 資料需有 status_label 欄（alive/failed）")
    y = (df["status_label"] == "failed").astype(int)
    drop_cols = ["company_name", "status_label"]
    if "Division" in df.columns:
        drop_cols.append("Division")
    X = df.drop(columns=drop_cols)
    if "fyear" not in X.columns:
        raise ValueError("US 資料需有 fyear 欄（年份）")
    return X, y


def get_bankruptcy_splits(logger, split_mode="block_cv", dataset="auto"):
    """
    載入 Bankruptcy 資料、切割、前處理，回傳 (X_hist_scaled, y_hist, X_new_scaled, y_new, X_test_scaled, y_test)。

    dataset:
      - "auto": 若存在 american_bankruptcy_dataset.csv 則用 US 1999-2018，否則用 Taiwan data.csv
      - "us_1999_2018": 使用 US 1999-2018（須有 american_bankruptcy_dataset.csv）
      - "taiwan": 使用 Taiwan data.csv

    split_mode:
      - "chronological": 僅 US 有 fyear 時有效；1999-2011 歷史、2012-2014 新營運、2015-2018 測試
      - "block_cv": 5-fold block CV（1+2 折=歷史、3+4 折=新營運、第 5 折=測試）
      - "random": 60-20-20 隨機切（stratify）
    """
    import sys
    sys.path.insert(0, str(project_root))
    from src.data import DataSplitter, DataPreprocessor

    # 決定資料集
    if dataset == "auto":
        use_us = US_CSV.exists()
        dataset = "us_1999_2018" if use_us else "taiwan"
    if dataset == "us_1999_2018":
        if not US_CSV.exists():
            raise FileNotFoundError(
                f"US 1999-2018 資料檔不存在: {US_CSV}\n"
                "請從 GitHub https://github.com/sowide/bankruptcy_dataset 或 "
                "Kaggle american-companies-bankruptcy-prediction-dataset 下載 "
                "american_bankruptcy_dataset.csv 放到 data/raw/bankruptcy/"
            )
        logger.info("步驟 1: 載入 US 1999-2018 破產資料")
        X, y = _load_us_1999_2018(logger)
        time_col = "fyear"
    else:
        logger.info("步驟 1: 載入 Taiwan 破產資料 (data.csv)")
        X, y = _load_taiwan(logger)
        time_col = None

    logger.info(f"資料大小: {X.shape}, 破產率: {y.mean()*100:.2f}%")

    # US 1999-2018 有年份時，預設用 chronological（1999-2011 / 2012-2014 / 2015-2018）
    if dataset == "us_1999_2018" and split_mode == "block_cv" and time_col:
        split_mode = "chronological"
        logger.info("使用 US 1999-2018，改為 chronological 切割（1999-2011 / 2012-2014 / 2015-2018）")

    logger.info("步驟 2: 資料切割")
    splitter = DataSplitter()

    if split_mode == "chronological" and time_col and time_col in X.columns:
        # 切割 a：1999-2011 歷史、2012-2014 新營運、2015-2018 測試
        splits = splitter.chronological_split(
            X, y,
            time_column=time_col,
            historical_end=2011,
            new_operating_end=2014,
        )
        # 訓練用特徵不包含年份，移除 fyear
        for key in splits:
            X_part, y_part = splits[key]
            if time_col in X_part.columns:
                X_part = X_part.drop(columns=[time_col])
            splits[key] = (X_part, y_part)
    elif split_mode == "block_cv":
        splits = splitter.block_cv_split(
            X, y,
            n_folds=5,
            historical_folds=[1, 2],
            new_operating_folds=[3, 4],
            testing_fold=5,
        )
        if time_col and time_col in X.columns:
            for key in splits:
                X_part, y_part = splits[key]
                if time_col in X_part.columns:
                    X_part = X_part.drop(columns=[time_col])
                splits[key] = (X_part, y_part)
    else:
        from sklearn.model_selection import train_test_split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_hist, X_new, y_hist, y_new = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        if time_col and time_col in X.columns:
            X_hist = X_hist.drop(columns=[time_col])
            X_new = X_new.drop(columns=[time_col])
            X_test = X_test.drop(columns=[time_col])
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
