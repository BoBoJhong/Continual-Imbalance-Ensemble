"""
共用：Bankruptcy 資料載入、切割（含 5-fold block CV）、前處理。
對應老師要求：切割 b. 5-fold CV（1+2 折=歷史、3+4 折=新營運、第 5 折=測試）
"""
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent


def get_bankruptcy_splits(logger, split_mode="block_cv"):
    """
    載入 Bankruptcy 資料、切割、前處理，回傳 (X_hist_scaled, y_hist, X_new_scaled, y_new, X_test_scaled, y_test)。

    split_mode:
      - "block_cv": 5-fold block CV（1+2 折=歷史、3+4 折=新營運、第 5 折=測試）
      - "random": 60-20-20 隨機切（stratify）
    """
    import sys
    sys.path.insert(0, str(project_root))
    from src.data import DataSplitter, DataPreprocessor

    logger.info("步驟 1: 載入 Bankruptcy 資料")
    df = pd.read_csv(project_root / "data/raw/bankruptcy/data.csv")
    if "Bankrupt?" in df.columns:
        y = df["Bankrupt?"]
        X = df.drop(columns=["Bankrupt?"])
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    logger.info(f"資料大小: {X.shape}, 破產率: {y.mean()*100:.2f}%")

    logger.info("步驟 2: 資料切割")
    splitter = DataSplitter()
    if split_mode == "block_cv":
        # 老師要求 b：1+2 折=歷史、3+4 折=新營運、第 5 折=測試
        splits = splitter.block_cv_split(
            X, y,
            n_folds=5,
            historical_folds=[1, 2],
            new_operating_folds=[3, 4],
            testing_fold=5,
        )
    elif split_mode == "chronological" and "Year" in X.columns:
        splits = splitter.chronological_split(
            X, y,
            time_column="Year",
            historical_end=2011,
            new_operating_end=2014,
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
