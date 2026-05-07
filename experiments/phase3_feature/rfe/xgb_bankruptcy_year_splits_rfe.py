"""
Phase 3 — Study II: Wrapper RFE（破產）— 與 MI / SHAP 同一套 FeatureSelector
=============================================================================
與 `xgb_bankruptcy_year_splits_fs_full.py`、`xgb_bankruptcy_year_splits_fs_dcs.py`
使用相同的 `FeatureSelector(method="rfe")`（sklearn RFE + DecisionTree 基底），
僅輸出靜態集成 All_6 一欄，方便快速重跑或對照。

完整靜態／DES／DCS 請跑 `xgb_bankruptcy_year_splits_fs_full.py` 與 `xgb_bankruptcy_year_splits_fs_dcs.py`。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import DataPreprocessor, ImbalanceSampler
from src.models import XGBoostWrapper
from src.features import FeatureSelector
from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split
from experiments.phase2_ensemble.xgb_oldnew_ensemble_common import ensemble_metrics_with_threshold
from experiments.phase3_feature._core.paths import results_method_dir

# ---------------------------------------------------------------------------
# 常數
# ---------------------------------------------------------------------------

FS_RATIO = 0.8  # 保留 80%
OUTPUT_DIR = results_method_dir(project_root, "rfe")

# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main():
    logger = get_logger("Phase3_RFE_Bankruptcy", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from experiments.phase2_ensemble.xgb_year_split_shared import (
        _split_fit_val_by_year
    )

    results = []

    for split_idx, (label, old_end_year) in enumerate(YEAR_SPLITS, 1):
        logger.info(f"\n[{split_idx}/{len(YEAR_SPLITS)}] RFE Processing {label}...")
        
        X_old_raw, y_old, X_new_raw, y_new, X_test_raw, y_test, year_old, year_new, _ = \
            get_bankruptcy_year_split(logger, old_end_year=old_end_year, return_years=True)
        
        y_old = np.asarray(y_old)
        y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)

        # 1. RFE（與 fs_full / fs_dcs 相同之 FeatureSelector，在 Old 上 fit）
        n_features = X_old_raw.shape[1]
        n_to_select = max(1, int(n_features * FS_RATIO))

        logger.info(
            f"    Running RFE (FeatureSelector, DecisionTree base): "
            f"{n_to_select} / {n_features} features..."
        )
        fs = FeatureSelector(method="rfe", k=n_to_select)
        X_old_fs = fs.fit_transform(X_old_raw, y_old)
        X_new_fs = fs.transform(X_new_raw)
        X_test_fs = fs.transform(X_test_raw)

        # 2. 訓練與評估 (使用靜態集成 All_6 作為標竿)
        # 這裡簡化流程，直接在 FS 後的資料上跑一次標準集成
        X_old_fit, y_old_fit, X_old_val, y_old_val = _split_fit_val_by_year(X_old_fs, y_old, year_old)
        X_new_fit, y_new_fit, X_new_val, y_new_val = _split_fit_val_by_year(X_new_fs, y_new, year_new)

        pre_old = DataPreprocessor()
        X_old_fit_s = pre_old.scale_features(X_old_fit, fit=True)[0]
        X_old_val_s_old = pre_old.scale_features(X_old_val, fit=False)[0]
        X_new_val_s_old = pre_old.scale_features(X_new_val, fit=False)[0]
        X_test_s_old = pre_old.scale_features(X_test_fs, fit=False)[0]

        pre_new = DataPreprocessor()
        X_new_fit_s = pre_new.scale_features(X_new_fit, fit=True)[0]
        X_new_val_s_new = pre_new.scale_features(X_new_val, fit=False)[0]
        X_test_s_new = pre_new.scale_features(X_test_fs, fit=False)[0]

        y_val_all = np.concatenate([y_old_val, y_new_val], axis=0)
        sampler = ImbalanceSampler()
        
        old_val_p, old_test_p = [], []
        new_val_p, new_test_p = [], []

        for s in ["undersampling", "oversampling", "hybrid"]:
            # Old
            xr, yr = sampler.apply_sampling(X_old_fit_s, y_old_fit, strategy=s)
            m = XGBoostWrapper()
            m.fit(xr, yr)
            old_val_p.append(m.predict_proba(np.concatenate([X_old_val_s_old, X_new_val_s_old], axis=0)))
            old_test_p.append(m.predict_proba(X_test_s_old))
            # New
            xr, yr = sampler.apply_sampling(X_new_fit_s, y_new_fit, strategy=s)
            m = XGBoostWrapper()
            m.fit(xr, yr)
            new_val_p.append(m.predict_proba(np.concatenate([pre_new.scale_features(X_old_val, fit=False)[0], X_new_val_s_new], axis=0)))
            new_test_p.append(m.predict_proba(X_test_s_new))

        # All_6 Ensemble
        res = ensemble_metrics_with_threshold(y_val_all, old_val_p + new_val_p, y_test_arr, old_test_p + new_test_p)
        results.append({
            "split": label, "fs": "rfe_r80", "AUC": res["AUC"], "F1": res["F1"], "Recall": res["Recall"]
        })
        logger.info(f"    RFE All_6 Result: AUC={res['AUC']:.4f}")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "xgb_bankruptcy_rfe_results.csv", index=False)
    logger.info(f"RFE Experiment Finished. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
