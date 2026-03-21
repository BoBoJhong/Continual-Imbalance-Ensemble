"""Phase 2 — XGB Medical：動態 DCS，年份切割。輸出：dynamic/dcs/"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import DataPreprocessor
from src.models import ModelPool, XGBoostWrapper
from src.utils import set_seed, get_logger

from experiments._shared.common_dataset import MEDICAL_YEAR_SPLITS, get_medical_year_split
from experiments._shared.common_dcs import _Wrapper, run_dcs_all_variants_from_pool


def main():
    logger = get_logger("XGB_Medical_DCS_YearSplits", console=True, file=True)
    set_seed(42)

    out = project_root / "results" / "phase2_ensemble" / "dynamic" / "dcs"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for label, old_end_year in MEDICAL_YEAR_SPLITS:
        logger.info(f"\n[Medical DCS] Split={label}")

        X_old, y_old, X_new, y_new, X_test, y_test = get_medical_year_split(
            logger, old_end_year=old_end_year
        )

        y_old = np.asarray(y_old)
        y_new = np.asarray(y_new)

        pre = DataPreprocessor()
        X_all = pd.concat([X_old, X_new], axis=0)
        X_all_s = pre.scale_features(X_all, fit=True)[0]
        no = len(X_old)
        X_h = X_all_s.iloc[:no]
        X_n = X_all_s.iloc[no:]
        X_t = pre.scale_features(X_test, fit=False)[0]

        old_pool = ModelPool(pool_name="old")
        old_pool.create_pool(X_h, y_old, prefix="old", model_class=XGBoostWrapper)
        new_pool = ModelPool(pool_name="new")
        new_pool.create_pool(X_n, y_new, prefix="new", model_class=XGBoostWrapper)

        pool_models = [
            _Wrapper(m)
            for m in old_pool.get_all_models() + new_pool.get_all_models()
        ]

        results = run_dcs_all_variants_from_pool(
            pool_models,
            X_h,
            y_old,
            X_n,
            y_new,
            X_t,
            y_test,
            logger,
            k=7,
        )

        year_combo = label.split("split_", 1)[-1] if str(label).startswith("split_") else str(label)
        for variant_name, m in results.items():
            rows.append(
                {
                    "dataset": year_combo,
                    "split": label,
                    "ensemble": variant_name,
                    "sampling_col": "pool6_xgb",
                    "subset": "",
                    "subset_indices": "",
                    "AUC": m["AUC"],
                    "F1": m["F1"],
                    "Recall": m["Recall"],
                }
            )

    df = pd.DataFrame(rows)
    path = out / "xgb_oldnew_ensemble_dcs_by_sampling_raw_medical.csv"
    df.to_csv(path, index=False, float_format="%.4f")
    logger.info(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
