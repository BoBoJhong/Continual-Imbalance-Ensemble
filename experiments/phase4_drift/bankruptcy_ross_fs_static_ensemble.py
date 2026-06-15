"""
Phase 4 - Bankruptcy ROSS + Feature Selection Static Ensemble
=============================================================

Compare Fixed vs ROSS boundaries with and without Phase 3-style feature
selection.

Boundaries
----------
Fixed:
    Old = 1999-2011, New = 2012-2014

ROSS:
    Old = 1999-2007, New = 2008-2014

Feature selection
-----------------
The FS variant follows Phase 3:

    FeatureSelector(method="kbest_f", k=50% of features)

The selector is fitted only on the Old window, then applied to New and
Test. This keeps the feature selection step aligned with the historical
knowledge available before observing post-drift data.

Outputs
-------
    results/phase4_drift/bk_ross_fs_static_ensemble_comparison.csv
    results/phase4_drift/bk_ross_fs_static_ensemble_summary.csv

Usage
-----
    python experiments/phase4_drift/bankruptcy_ross_fs_static_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features import FeatureSelector
from src.utils import get_logger, set_seed

from experiments.phase4_drift.bankruptcy_drift_stream import (
    TRAIN_END_YEAR,
    TEST_END_YEAR,
    TEST_START_YEAR,
    load_bankruptcy_with_year,
    train_eval_ensemble,
)
from experiments.phase4_drift.bankruptcy_ross_static_ensemble import (
    FIXED_DRIFT_START_YEAR,
    OUTPUT_DIR,
    _old_new_split,
    _read_ross_boundary,
    _test_split,
)


FS_METHOD = "kbest_f"
FS_RATIO = 0.5


def _apply_old_fit_fs(
    X_old: pd.DataFrame,
    y_old: np.ndarray,
    X_new: pd.DataFrame,
    X_test: pd.DataFrame,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int, str]:
    """
    Fit FS on Old only, transform Old/New/Test.

    Mean imputation is fitted from Old only to avoid peeking at New/Test
    distribution during feature selection.
    """
    fill_values = X_old.mean(numeric_only=True)
    X_old_f = X_old.fillna(fill_values)
    X_new_f = X_new.fillna(fill_values)
    X_test_f = X_test.fillna(fill_values)

    k = max(1, int(X_old_f.shape[1] * FS_RATIO))
    selector = FeatureSelector(method=FS_METHOD, k=k)
    X_old_fs = selector.fit_transform(X_old_f, y_old)
    X_new_fs = selector.transform(X_new_f)
    X_test_fs = selector.transform(X_test_f)

    selected_preview = ",".join(selector.selected_cols_[:8])
    logger.info(
        f"  FS({FS_METHOD}): {X_old.shape[1]} -> {selector.n_selected} features; "
        f"preview={selected_preview}"
    )
    return X_old_fs, X_new_fs, X_test_fs, selector.n_selected, selected_preview


def _run_boundary_variant(
    *,
    label: str,
    boundary_source: str,
    drift_start_year: int,
    fs_variant: str,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    logger,
) -> list[dict]:
    X_old, y_old, X_new, y_new = _old_new_split(X_all, y_all, drift_start_year)
    n_features_before = X_old.shape[1]
    n_features_after = n_features_before
    selected_preview = ""

    logger.info(
        f"\n[{label}|{fs_variant}] Old=1999-{drift_start_year - 1} "
        f"({len(X_old):,} rows, {y_old.mean()*100:.2f}%+) | "
        f"New={drift_start_year}-{TRAIN_END_YEAR} "
        f"({len(X_new):,} rows, {y_new.mean()*100:.2f}%+)"
    )

    if fs_variant == "fs":
        X_old, X_new, X_test_used, n_features_after, selected_preview = _apply_old_fit_fs(
            X_old,
            y_old,
            X_new,
            X_test,
            logger,
        )
    elif fs_variant == "no_fs":
        X_test_used = X_test
    else:
        raise ValueError(f"Unknown fs_variant: {fs_variant}")

    metrics_by_combo = train_eval_ensemble(
        X_old,
        y_old,
        X_new,
        y_new,
        X_test_used,
        y_test,
        tag=f"{label}_{fs_variant}",
        logger=logger,
    )

    rows = []
    for combo, metrics in metrics_by_combo.items():
        rows.append(
            {
                "method": label,
                "boundary_source": boundary_source,
                "drift_start_year": drift_start_year,
                "old_window": f"1999-{drift_start_year - 1}",
                "new_window": f"{drift_start_year}-{TRAIN_END_YEAR}",
                "fs_variant": fs_variant,
                "fs_method": FS_METHOD if fs_variant == "fs" else "",
                "n_features_before": n_features_before,
                "n_features_after": n_features_after,
                "selected_preview": selected_preview,
                "combo": combo,
                **metrics,
            }
        )
    return rows


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    best = (
        df.sort_values(["method", "fs_variant", "F1", "AUC"], ascending=[True, True, False, False])
        .groupby(["method", "fs_variant"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    fixed_no_fs = best[(best["method"] == "Fixed_2012") & (best["fs_variant"] == "no_fs")].iloc[0]
    summary = best.copy()
    summary["delta_vs_fixed_no_fs_AUC"] = summary["AUC"] - fixed_no_fs["AUC"]
    summary["delta_vs_fixed_no_fs_F1"] = summary["F1"] - fixed_no_fs["F1"]
    summary["delta_vs_fixed_no_fs_Recall"] = summary["Recall"] - fixed_no_fs["Recall"]
    summary["delta_vs_fixed_no_fs_Precision"] = summary["Precision"] - fixed_no_fs["Precision"]

    no_fs_by_method = {
        row["method"]: row
        for _, row in best[best["fs_variant"] == "no_fs"].iterrows()
    }
    summary["delta_vs_same_boundary_no_fs_AUC"] = summary.apply(
        lambda r: r["AUC"] - no_fs_by_method[r["method"]]["AUC"], axis=1
    )
    summary["delta_vs_same_boundary_no_fs_F1"] = summary.apply(
        lambda r: r["F1"] - no_fs_by_method[r["method"]]["F1"], axis=1
    )
    return summary


def main() -> None:
    logger = get_logger("Phase4_BK_ROSS_FS_StaticEns", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ross_start = _read_ross_boundary()
    X_all, y_all = load_bankruptcy_with_year(logger)
    train_mask = (X_all["fyear"] >= 1999) & (X_all["fyear"] <= TRAIN_END_YEAR)
    X_train_all = X_all.loc[train_mask].reset_index(drop=True)
    y_train_all = y_all.loc[train_mask].reset_index(drop=True)
    X_test, y_test = _test_split(X_all, y_all)

    logger.info(
        f"Test={TEST_START_YEAR}-{TEST_END_YEAR}: {len(X_test):,} rows, "
        f"{y_test.mean()*100:.2f}%+"
    )

    configs = [
        ("Fixed_2012", "manual fixed boundary", FIXED_DRIFT_START_YEAR),
        ("ROSS_2008", "ROSS selected boundary", ross_start),
    ]

    rows: list[dict] = []
    for label, source, drift_start in configs:
        for fs_variant in ["no_fs", "fs"]:
            rows.extend(
                _run_boundary_variant(
                    label=label,
                    boundary_source=source,
                    drift_start_year=drift_start,
                    fs_variant=fs_variant,
                    X_all=X_train_all,
                    y_all=y_train_all,
                    X_test=X_test,
                    y_test=y_test,
                    logger=logger,
                )
            )

    df = pd.DataFrame(rows)
    comparison_path = OUTPUT_DIR / "bk_ross_fs_static_ensemble_comparison.csv"
    df.to_csv(comparison_path, index=False, float_format="%.6f")

    summary = _build_summary(df)
    summary_path = OUTPUT_DIR / "bk_ross_fs_static_ensemble_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    logger.info(f"\nSaved comparison -> {comparison_path}")
    logger.info(f"Saved summary -> {summary_path}")
    logger.info("\nBest static ensemble by boundary and FS variant:")
    logger.info(
        "\n"
        + summary[
            [
                "method",
                "fs_variant",
                "drift_start_year",
                "combo",
                "n_features_after",
                "AUC",
                "F1",
                "Recall",
                "Precision",
                "delta_vs_fixed_no_fs_AUC",
                "delta_vs_fixed_no_fs_F1",
                "delta_vs_same_boundary_no_fs_F1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
