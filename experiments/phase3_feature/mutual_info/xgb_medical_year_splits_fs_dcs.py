"""
Phase 3 — Study II: 醫療資料集 + FS + DCS（對齊破產流程）
=============================================================================
與破產 mutual_info 版本一致，輸出拆分到：
results/phase3_feature/{mutual_info,shap,rfe}/xgb_medical_fs_full_dcs.csv
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
from experiments._shared.common_dataset import MEDICAL_YEAR_SPLITS, get_medical_year_split
from experiments._shared.common_dcs import _Wrapper, run_dcs_all_variants_from_pool
from experiments.phase3_feature._core.triad_csv import save_medical_fs_dcs_split

SAMPLING_STRATEGIES = ["undersampling", "oversampling", "hybrid"]
FS_CONFIGS = [
    ("no_fs", None, None),
    ("mi_r80", "mutual_info", 0.8),
    ("shap_r80", "shap", 0.8),
    ("rfe_r80", "rfe", 0.8),
]
METRICS = ["AUC", "F1", "Recall"]


def _apply_fs(fs_tag, fs_method, fs_ratio, X_old, y_old, X_new, X_test, logger):
    if fs_method is None:
        return X_old, X_new, X_test
    k = max(1, int(X_old.shape[1] * fs_ratio))
    fs = FeatureSelector(method=fs_method, k=k)
    X_old_fs = fs.fit_transform(X_old, y_old)
    X_new_fs = fs.transform(X_new)
    X_test_fs = fs.transform(X_test)
    logger.info(f"    FS [{fs_tag}]: {X_old.shape[1]} -> {X_old_fs.shape[1]} features")
    return X_old_fs, X_new_fs, X_test_fs


def main():
    logger = get_logger("Phase3_FS_DCS_Medical", console=True, file=True)
    set_seed(42)

    all_rows = []

    for split_idx, (label, old_end_year) in enumerate(MEDICAL_YEAR_SPLITS, 1):
        logger.info(f"\n[{split_idx}/{len(MEDICAL_YEAR_SPLITS)}] {label} (old_end={old_end_year})")

        X_old_raw, y_old, X_new_raw, y_new, X_test_raw, y_test = get_medical_year_split(
            logger, old_end_year=old_end_year
        )

        y_old = np.asarray(y_old)
        y_new = np.asarray(y_new)
        y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)

        for fs_tag, fs_method, fs_ratio in FS_CONFIGS:
            logger.info(f"  FS: {fs_tag}")
            X_old_fs, X_new_fs, X_test_fs = _apply_fs(
                fs_tag, fs_method, fs_ratio, X_old_raw, y_old, X_new_raw, X_test_raw, logger
            )

            pre = DataPreprocessor()
            X_all = pd.concat([X_old_fs, X_new_fs], axis=0).reset_index(drop=True)
            X_all_s = pre.scale_features(X_all, fit=True)[0]
            n_old = len(X_old_fs)
            X_h = X_all_s.iloc[:n_old]
            X_n = X_all_s.iloc[n_old:]
            X_t = pre.scale_features(X_test_fs, fit=False)[0]

            sampler = ImbalanceSampler()
            pool_models = []
            for s in SAMPLING_STRATEGIES:
                X_r, y_r = sampler.apply_sampling(X_h, y_old, strategy=s)
                m_old = XGBoostWrapper(name=f"old_med_{label}_{fs_tag}_{s}")
                m_old.fit(X_r, y_r)
                pool_models.append(_Wrapper(m_old))
            for s in SAMPLING_STRATEGIES:
                X_r, y_r = sampler.apply_sampling(X_n, y_new, strategy=s)
                m_new = XGBoostWrapper(name=f"new_med_{label}_{fs_tag}_{s}")
                m_new.fit(X_r, y_r)
                pool_models.append(_Wrapper(m_new))

            results = run_dcs_all_variants_from_pool(
                pool_models, X_h, y_old, X_n, y_new, X_t, y_test_arr, logger, k=7
            )
            for variant_name, metrics in results.items():
                all_rows.append(
                    {
                        "split": label,
                        "fs": fs_tag,
                        "ensemble": variant_name,
                        **{k: metrics[k] for k in METRICS if k in metrics},
                    }
                )

    save_medical_fs_dcs_split(project_root, all_rows)
    logger.info(f"Done! Medical DCS rows saved: {len(all_rows)}")


if __name__ == "__main__":
    main()
