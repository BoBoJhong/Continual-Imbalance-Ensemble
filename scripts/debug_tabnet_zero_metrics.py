"""
Debug TabNet zero-metric cases on bankruptcy year splits.

This script re-runs the TabNet bankruptcy year split experiment and records
cases where F1/Recall/Precision/G_Mean are zero. It logs class counts and
probability distribution stats to help explain why metrics are 0.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler, DataPreprocessor
from src.models import TabNetWrapper
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split

SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
OUTPUT_DIR = project_root / "results" / "phase1_baseline" / "tabnet"


def _scale(X_train, X_test_raw):
    pre = DataPreprocessor()
    X_tr_s, X_te_s = pre.scale_features(X_train, X_test_raw, fit=True)
    return X_tr_s, X_te_s


def _run_one(X_train, y_train, X_test, y_test, sampler, strategy, tag, logger):
    X_r, y_r = sampler.apply_sampling(X_train, np.asarray(y_train), strategy=strategy)
    model = TabNetWrapper(name=f"{tag}_{strategy}")
    model.fit(X_r, y_r)

    y_true = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    y_proba = model.predict_proba(X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = compute_metrics(y_true, y_proba, y_pred=y_pred)
    zero_flags = {
        "F1": metrics["F1"] == 0.0,
        "Recall": metrics["Recall"] == 0.0,
        "Precision": metrics["Precision"] == 0.0,
        "G_Mean": metrics["G_Mean"] == 0.0,
    }

    if any(zero_flags.values()):
        pos_true = int((y_true == 1).sum())
        neg_true = int((y_true == 0).sum())
        pos_pred = int((y_pred == 1).sum())
        neg_pred = int((y_pred == 0).sum())
        logger.info(
            "ZERO_METRIC | %s %s | pos_true=%d neg_true=%d pos_pred=%d neg_pred=%d "
            "proba[min/mean/max]=%.4f/%.4f/%.4f",
            tag,
            strategy,
            pos_true,
            neg_true,
            pos_pred,
            neg_pred,
            float(np.min(y_proba)),
            float(np.mean(y_proba)),
            float(np.max(y_proba)),
        )

        return {
            "method": tag,
            "sampling": strategy,
            **metrics,
            "pos_true": pos_true,
            "neg_true": neg_true,
            "pos_pred": pos_pred,
            "neg_pred": neg_pred,
            "proba_min": float(np.min(y_proba)),
            "proba_mean": float(np.mean(y_proba)),
            "proba_max": float(np.max(y_proba)),
            **{f"zero_{k}": v for k, v in zero_flags.items()},
        }

    return None


def main():
    logger = get_logger("BK_YearSplits_TabNet_ZeroDebug", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, old_end_year in YEAR_SPLITS:
        logger.info(
            f"\n{'=' * 60}\nSplit: {label}  (Old<=2{old_end_year}, New=2{old_end_year + 1}-2014, Test=2015-2018)\n{'=' * 60}"
        )
        try:
            X_old, y_old, X_new, y_new, X_test, y_test = get_bankruptcy_year_split(
                logger, old_end_year=old_end_year
            )
        except Exception as exc:
            logger.error("[ERROR] %s: %s", label, exc)
            import traceback

            logger.error(traceback.format_exc())
            continue

        sampler = ImbalanceSampler()

        X_old_s, X_test_s_old = _scale(X_old, X_test)
        for strat in SAMPLING_STRATEGIES:
            row = _run_one(X_old_s, y_old, X_test_s_old, y_test, sampler, strat, "Old", logger)
            if row:
                row["split"] = label
                rows.append(row)

        X_combined = pd.concat([X_old, X_new], ignore_index=True)
        y_combined = pd.concat([y_old.reset_index(drop=True), y_new.reset_index(drop=True)], ignore_index=True)
        X_combined_s, X_test_s_comb = _scale(X_combined, X_test)
        for strat in SAMPLING_STRATEGIES:
            row = _run_one(
                X_combined_s,
                y_combined,
                X_test_s_comb,
                y_test,
                sampler,
                strat,
                "Old+New",
                logger,
            )
            if row:
                row["split"] = label
                rows.append(row)

        X_new_s, X_test_s_new = _scale(X_new, X_test)
        for strat in SAMPLING_STRATEGIES:
            row = _run_one(X_new_s, y_new, X_test_s_new, y_test, sampler, strat, "New", logger)
            if row:
                row["split"] = label
                rows.append(row)

    if not rows:
        logger.info("No zero-metric cases found.")
        return

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "bankruptcy_year_splits_tabnet_zero_debug.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved -> %s (%d rows)", out_path.name, len(df))


if __name__ == "__main__":
    main()