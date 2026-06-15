"""
Weighted Old/New ensemble validation across bankruptcy year splits.

This addresses the limitation that the Phase 5 weighted ensemble was originally
reported on one final boundary setting. Here we evaluate all 15 chronological
Old/New boundaries, select the Old/New weight by validation F1 within each split,
and compare against equal weighting with paired Wilcoxon tests.

Outputs:
    results/phase5_weighted/bk_year_split_weight_sweep.csv
    results/phase5_weighted/bk_year_split_weight_val_selected_summary.csv
    results/phase5_weighted/bk_year_split_weight_wilcoxon.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments._shared.common_bankruptcy import YEAR_SPLITS
from experiments.phase4_drift.bankruptcy_drift_stream import (
    TRAIN_END_YEAR,
    _select_threshold,
    load_bankruptcy_with_year,
)
from experiments.phase4_drift.bankruptcy_ross_fs_static_ensemble import _apply_old_fit_fs
from experiments.phase4_drift.bankruptcy_ross_static_ensemble import _old_new_split, _test_split
from experiments.phase5_weighted.bankruptcy_ross_weight_sweep import (
    WEIGHT_GRID,
    _train_old_new_mean_probas,
)
from src.evaluation import compute_metrics
from src.utils import get_logger, set_seed


OUTPUT_DIR = project_root / "results" / "phase5_weighted"
METRICS = ("AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error")


def _run_split(
    *,
    split_label: str,
    drift_start_year: int,
    fs_variant: str,
    X_train_all: pd.DataFrame,
    y_train_all: pd.Series,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    logger,
) -> list[dict]:
    X_old, y_old, X_new, y_new = _old_new_split(X_train_all, y_train_all, drift_start_year)
    n_features_before = X_old.shape[1]
    n_features_after = n_features_before
    selected_preview = ""
    X_test_used = X_test

    if fs_variant == "fs":
        X_old, X_new, X_test_used, n_features_after, selected_preview = _apply_old_fit_fs(
            X_old,
            y_old,
            X_new,
            X_test,
            logger,
        )
    elif fs_variant != "no_fs":
        raise ValueError(f"Unknown fs_variant: {fs_variant}")

    y_val, old_val, new_val, old_test, new_test = _train_old_new_mean_probas(
        X_old,
        y_old,
        X_new,
        y_new,
        X_test_used,
        tag=f"{split_label}_{fs_variant}",
        logger=logger,
    )

    rows: list[dict] = []
    for w_new in WEIGHT_GRID:
        w_old = float(1.0 - w_new)
        val_proba = w_old * old_val + float(w_new) * new_val
        test_proba = w_old * old_test + float(w_new) * new_test
        threshold = _select_threshold(y_val, val_proba)
        val_metrics = compute_metrics(y_val, val_proba, threshold=threshold)
        test_metrics = compute_metrics(y_test, test_proba, threshold=threshold)
        row = {
            "split": split_label,
            "drift_start_year": drift_start_year,
            "old_window": f"1999-{drift_start_year - 1}",
            "new_window": f"{drift_start_year}-{TRAIN_END_YEAR}",
            "fs_variant": fs_variant,
            "n_features_before": n_features_before,
            "n_features_after": n_features_after,
            "selected_preview": selected_preview,
            "w_old": w_old,
            "w_new": float(w_new),
            "threshold": threshold,
        }
        for metric in METRICS:
            row[f"val_{metric}"] = val_metrics[metric]
            row[f"test_{metric}"] = test_metrics[metric]
        rows.append(row)
    return rows


def _build_val_selected_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    selected = (
        sweep.sort_values(
            ["split", "fs_variant", "val_F1", "val_AUC", "val_Precision"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["split", "fs_variant"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    equal = sweep[np.isclose(sweep["w_new"], 0.5)].copy()
    equal = equal.rename(columns={f"test_{m}": f"equal_test_{m}" for m in METRICS})
    equal = equal[["split", "fs_variant", "w_new", *[f"equal_test_{m}" for m in METRICS]]]
    equal = equal.rename(columns={"w_new": "equal_w_new"})
    summary = selected.merge(equal, on=["split", "fs_variant"], how="left")
    for metric in METRICS:
        summary[f"delta_vs_equal_{metric}"] = summary[f"test_{metric}"] - summary[f"equal_test_{metric}"]
    return summary


def _build_wilcoxon(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for fs_variant, group in summary.groupby("fs_variant"):
        for metric in ("AUC", "F1", "Recall", "Precision"):
            selected = group[f"test_{metric}"].to_numpy(dtype=float)
            equal = group[f"equal_test_{metric}"].to_numpy(dtype=float)
            diff = selected - equal
            if np.allclose(diff, 0.0):
                p_greater = 1.0
                p_two_sided = 1.0
            else:
                p_greater = float(wilcoxon(selected, equal, alternative="greater").pvalue)
                p_two_sided = float(wilcoxon(selected, equal, alternative="two-sided").pvalue)
            rows.append(
                {
                    "fs_variant": fs_variant,
                    "metric": metric,
                    "n_pairs": len(group),
                    "selected_mean": float(np.mean(selected)),
                    "equal_mean": float(np.mean(equal)),
                    "mean_diff_selected_minus_equal": float(np.mean(diff)),
                    "n_selected_better": int(np.sum(selected > equal)),
                    "n_equal_better": int(np.sum(equal > selected)),
                    "p_greater": p_greater,
                    "p_two_sided": p_two_sided,
                    "significant_greater": bool(p_greater < 0.05),
                    "significant_two_sided": bool(p_two_sided < 0.05),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    logger = get_logger("WeightedSplitValidation", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_all, y_all = load_bankruptcy_with_year(logger)
    train_mask = (X_all["fyear"] >= 1999) & (X_all["fyear"] <= TRAIN_END_YEAR)
    X_train_all = X_all.loc[train_mask].reset_index(drop=True)
    y_train_all = y_all.loc[train_mask].reset_index(drop=True)
    X_test, y_test = _test_split(X_all, y_all)

    rows: list[dict] = []
    for idx, (split_label, old_end_year) in enumerate(YEAR_SPLITS, 1):
        drift_start_year = old_end_year + 1
        logger.info(
            f"\n[{idx}/{len(YEAR_SPLITS)}] {split_label}: "
            f"Old=1999-{old_end_year}, New={drift_start_year}-{TRAIN_END_YEAR}"
        )
        for fs_variant in ("no_fs", "fs"):
            rows.extend(
                _run_split(
                    split_label=split_label,
                    drift_start_year=drift_start_year,
                    fs_variant=fs_variant,
                    X_train_all=X_train_all,
                    y_train_all=y_train_all,
                    X_test=X_test,
                    y_test=y_test,
                    logger=logger,
                )
            )

    sweep = pd.DataFrame(rows)
    summary = _build_val_selected_summary(sweep)
    wilcoxon_df = _build_wilcoxon(summary)

    sweep_path = OUTPUT_DIR / "bk_year_split_weight_sweep.csv"
    summary_path = OUTPUT_DIR / "bk_year_split_weight_val_selected_summary.csv"
    wilcoxon_path = OUTPUT_DIR / "bk_year_split_weight_wilcoxon.csv"
    sweep.to_csv(sweep_path, index=False, float_format="%.8f")
    summary.to_csv(summary_path, index=False, float_format="%.8f")
    wilcoxon_df.to_csv(wilcoxon_path, index=False, float_format="%.8f")

    logger.info(f"\nSaved sweep -> {sweep_path}")
    logger.info(f"Saved val-selected summary -> {summary_path}")
    logger.info(f"Saved Wilcoxon -> {wilcoxon_path}")
    logger.info("\nWeighted split Wilcoxon:")
    logger.info("\n" + wilcoxon_df.to_string(index=False))


if __name__ == "__main__":
    main()
