"""
Phase 3 — Study II: 全集成方法 (Static + DES) 與特徵選擇的結合
=============================================================================
測試所有 FS 配置（no_fs / mutual_info_r50 / mutual_info_r80 / shap_r50 / shap_r80）
在所有集成策略（含動態選擇 DES）下的表現。

FS 配置與靜態實驗 (xgb_bankruptcy_fs_advanced_raw.csv) 完全對齊，確保比較公平性。
年份切割：15 組 (split_1+15 … split_15+1)
"""
from __future__ import annotations

import sys
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import DataPreprocessor, ImbalanceSampler
from src.models import XGBoostWrapper
from src.features import FeatureSelector
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split
from experiments.phase2_ensemble.xgb_oldnew_ensemble_common import (
    ensemble_metrics_with_threshold,
    dynamic_ensemble_metrics_with_threshold,
    DYNAMIC_DES_METHODS
)

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

SAMPLING_STRATEGIES = ["undersampling", "oversampling", "hybrid"]
# FS 配置與靜態集成實驗（xgb_bankruptcy_fs_advanced_raw.csv）完全對齊
# 確保同一階段所有實驗使用相同的 FS 方法與比例，結果才具有可比較性
FS_CONFIGS = [
    ("no_fs",           None,          None),
    ("mutual_info_r50", "mutual_info", 0.5),
    ("mutual_info_r80", "mutual_info", 0.8),
    ("shap_r50",        "shap",        0.5),
    ("shap_r80",        "shap",        0.8),
]
METRICS = ["AUC", "F1", "Recall"]
OUTPUT_DIR = project_root / "results" / "phase3_feature"

# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def _apply_fs(method, ratio, X_old, y_old, X_new, X_test, logger):
    if method is None:
        return X_old, X_new, X_test
    k = max(1, int(X_old.shape[1] * ratio))
    fs = FeatureSelector(method=method, k=k)
    X_old_fs = fs.fit_transform(X_old, y_old)
    X_new_fs = fs.transform(X_new)
    X_test_fs = fs.transform(X_test)
    logger.info(f"    FS [{method}]: {X_old.shape[1]} -> {X_old_fs.shape[1]} features")
    return X_old_fs, X_new_fs, X_test_fs

# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main():
    logger = get_logger("Phase3_FS_Full_Ensemble", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from experiments.phase2_ensemble.xgb_year_split_shared import (
        _split_fit_val_by_year, train_one_sampling_xgb
    )

    static_all = []
    des_all = []

    for split_idx, (label, old_end_year) in enumerate(YEAR_SPLITS, 1):
        logger.info(f"\n[{split_idx}/{len(YEAR_SPLITS)}] Processing {label}...")
        
        # 載入原始資料
        X_old_raw, y_old, X_new_raw, y_new, X_test_raw, y_test, year_old, year_new, _ = \
            get_bankruptcy_year_split(logger, old_end_year=old_end_year, return_years=True)
        
        y_old = np.asarray(y_old)
        y_new = np.asarray(y_new)
        y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)

        for fs_tag, fs_method, fs_ratio in FS_CONFIGS:
            logger.info(f"  Running FS: {fs_tag}")
            
            # 1. Feature Selection
            X_old_fs, X_new_fs, X_test_fs = _apply_fs(fs_method, fs_ratio, X_old_raw, y_old, X_new_raw, X_test_raw, logger)
            
            # 2. Split Fit/Val
            X_old_fit, y_old_fit, X_old_val, y_old_val = _split_fit_val_by_year(X_old_fs, y_old, year_old)
            X_new_fit, y_new_fit, X_new_val, y_new_val = _split_fit_val_by_year(X_new_fs, y_new, year_new)

            # 3. Scaling
            pre_old = DataPreprocessor()
            X_old_fit_s = pre_old.scale_features(X_old_fit, fit=True)[0]
            X_old_val_s_old = pre_old.scale_features(X_old_val, fit=False)[0]
            X_new_val_s_old = pre_old.scale_features(X_new_val, fit=False)[0]
            X_test_s_old = pre_old.scale_features(X_test_fs, fit=False)[0]

            pre_new = DataPreprocessor()
            X_new_fit_s = pre_new.scale_features(X_new_fit, fit=True)[0]
            X_new_val_s_new = pre_new.scale_features(X_new_val, fit=False)[0]
            X_test_s_new = pre_new.scale_features(X_test_fs, fit=False)[0]

            # Validation data for thresholding and DES
            X_val_s_old_all = np.concatenate([X_old_val_s_old, X_new_val_s_old], axis=0)
            X_val_s_new_all = np.concatenate([pre_new.scale_features(X_old_val, fit=False)[0], X_new_val_s_new], axis=0)
            y_val_all = np.concatenate([y_old_val, y_new_val], axis=0)

            # 4. Train 6 Models
            old_val_probas = []
            old_test_probas = []
            new_val_probas = []
            new_test_probas = []
            sampler = ImbalanceSampler()

            for s in SAMPLING_STRATEGIES:
                # Old models
                X_r, y_r = sampler.apply_sampling(X_old_fit_s, y_old_fit, strategy=s)
                m = XGBoostWrapper(name=f"old_{label}_{fs_tag}_{s}")
                m.fit(X_r, y_r)
                old_val_probas.append(m.predict_proba(X_val_s_old_all))
                old_test_probas.append(m.predict_proba(X_test_s_old))

                # New models
                X_r, y_r = sampler.apply_sampling(X_new_fit_s, y_new_fit, strategy=s)
                m = XGBoostWrapper(name=f"new_{label}_{fs_tag}_{s}")
                m.fit(X_r, y_r)
                new_val_probas.append(m.predict_proba(X_val_s_new_all))
                new_test_probas.append(m.predict_proba(X_test_s_new))

            # 5. Static Ensembles
            year_combo = label.split("split_", 1)[-1]
            combos = {
                "Old_3": (list(range(3)), []),
                "New_3": ([], list(range(3))),
                "All_6": (list(range(3)), list(range(3)))
            }

            for name, (oi, ni) in combos.items():
                v_ps = [old_val_probas[i] for i in oi] + [new_val_probas[i] for i in ni]
                t_ps = [old_test_probas[i] for i in oi] + [new_test_probas[i] for i in ni]
                res = ensemble_metrics_with_threshold(y_val_all, v_ps, y_test_arr, t_ps)
                static_all.append({
                    "split": label, "fs": fs_tag, "ensemble": name,
                    **{k: res[k] for k in METRICS}
                })

            # 6. Dynamic Ensembles (DES)
            # Use New scaler's validation features for DES
            X_val_np = np.asarray(X_val_s_new_all, dtype=np.float64)
            X_test_np = np.asarray(X_test_s_new, dtype=np.float64)
            
            val_6 = old_val_probas + new_val_probas
            test_6 = old_test_probas + new_test_probas

            for method_key, ens_name in DYNAMIC_DES_METHODS:
                d_res = dynamic_ensemble_metrics_with_threshold(
                    y_val_all, X_val_np, val_6, y_test_arr, X_test_np, test_6, 
                    method=method_key, k_neighbors=7
                )
                des_all.append({
                    "split": label, "fs": fs_tag, "ensemble": ens_name,
                    **{k: d_res[k] for k in METRICS}
                })

    # Save results（覆蓋原有檔案，包含所有 5 種 FS 配置）
    static_path = OUTPUT_DIR / "xgb_bankruptcy_fs_full_static.csv"
    des_path    = OUTPUT_DIR / "xgb_bankruptcy_fs_full_des.csv"
    pd.DataFrame(static_all).to_csv(static_path, index=False)
    pd.DataFrame(des_all).to_csv(des_path, index=False)
    logger.info(f"Static results saved: {static_path}  ({len(static_all)} rows)")
    logger.info(f"DES results saved:    {des_path}  ({len(des_all)} rows)")
    logger.info(f"Done! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
