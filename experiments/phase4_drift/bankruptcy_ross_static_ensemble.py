"""
Phase 4 - Bankruptcy ROSS Static Ensemble
=========================================

Compare Phase 2-style static Old/New ensembles under two boundaries:

1. Fixed boundary:
       Old = 1999-2011, New = 2012-2014

2. ROSS boundary:
       Old = 1999-2007, New = 2008-2014

The ROSS boundary is read from:

    results/phase4_drift/bk_ross_selected_boundary.csv

Outputs:

    results/phase4_drift/bk_ross_static_ensemble_comparison.csv
    results/phase4_drift/bk_ross_static_ensemble_summary.csv

Usage:

    python experiments/phase4_drift/bankruptcy_ross_static_ensemble.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_logger, set_seed

from experiments.phase4_drift.bankruptcy_drift_stream import (
    TRAIN_END_YEAR,
    TEST_END_YEAR,
    TEST_START_YEAR,
    load_bankruptcy_with_year,
    train_eval_ensemble,
)


OUTPUT_DIR = project_root / "results" / "phase4_drift"
ROSS_SELECTED_PATH = OUTPUT_DIR / "bk_ross_selected_boundary.csv"
FIXED_DRIFT_START_YEAR = 2012


def _test_split(X_all: pd.DataFrame, y_all: pd.Series):
    mask_test = (X_all["fyear"] >= TEST_START_YEAR) & (X_all["fyear"] <= TEST_END_YEAR)
    X_test = X_all.loc[mask_test].drop(columns=["fyear"]).reset_index(drop=True)
    y_test = np.asarray(y_all.loc[mask_test])
    return X_test, y_test


def _old_new_split(X_all: pd.DataFrame, y_all: pd.Series, drift_start_year: int):
    mask_old = (X_all["fyear"] >= 1999) & (X_all["fyear"] < drift_start_year)
    mask_new = (X_all["fyear"] >= drift_start_year) & (X_all["fyear"] <= TRAIN_END_YEAR)
    X_old = X_all.loc[mask_old].drop(columns=["fyear"]).reset_index(drop=True)
    y_old = np.asarray(y_all.loc[mask_old])
    X_new = X_all.loc[mask_new].drop(columns=["fyear"]).reset_index(drop=True)
    y_new = np.asarray(y_all.loc[mask_new])
    return X_old, y_old, X_new, y_new


def _read_ross_boundary() -> int:
    if not ROSS_SELECTED_PATH.exists():
        raise FileNotFoundError(
            f"ROSS selected boundary not found: {ROSS_SELECTED_PATH}\n"
            "Run experiments/phase4_drift/bankruptcy_ross.py first."
        )
    selected = pd.read_csv(ROSS_SELECTED_PATH)
    ross = selected[selected["method"].astype(str).str.lower() == "ross"]
    if ross.empty:
        raise ValueError(f"No ROSS row found in {ROSS_SELECTED_PATH}")
    return int(ross.iloc[0]["drift_start_year"])


def _run_boundary(
    *,
    label: str,
    boundary_source: str,
    drift_start_year: int,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    logger,
) -> list[dict]:
    X_old, y_old, X_new, y_new = _old_new_split(X_all, y_all, drift_start_year)
    logger.info(
        f"\n[{label}] Old=1999-{drift_start_year - 1} "
        f"({len(X_old):,} rows, {y_old.mean()*100:.2f}%+) | "
        f"New={drift_start_year}-{TRAIN_END_YEAR} "
        f"({len(X_new):,} rows, {y_new.mean()*100:.2f}%+)"
    )

    metrics_by_combo = train_eval_ensemble(
        X_old,
        y_old,
        X_new,
        y_new,
        X_test,
        y_test,
        tag=label,
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
                "combo": combo,
                **metrics,
            }
        )
    return rows


def main() -> None:
    logger = get_logger("Phase4_BK_ROSS_StaticEns", console=True, file=True)
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

    rows: list[dict] = []
    rows.extend(
        _run_boundary(
            label="Fixed_2012",
            boundary_source="manual fixed boundary",
            drift_start_year=FIXED_DRIFT_START_YEAR,
            X_all=X_train_all,
            y_all=y_train_all,
            X_test=X_test,
            y_test=y_test,
            logger=logger,
        )
    )
    rows.extend(
        _run_boundary(
            label="ROSS_2008",
            boundary_source="ROSS selected boundary",
            drift_start_year=ross_start,
            X_all=X_train_all,
            y_all=y_train_all,
            X_test=X_test,
            y_test=y_test,
            logger=logger,
        )
    )

    df = pd.DataFrame(rows)
    comparison_path = OUTPUT_DIR / "bk_ross_static_ensemble_comparison.csv"
    df.to_csv(comparison_path, index=False, float_format="%.6f")

    best = (
        df.sort_values(["method", "F1", "AUC"], ascending=[True, False, False])
        .groupby("method", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    fixed_best = best[best["method"] == "Fixed_2012"].iloc[0]
    summary = best.copy()
    summary["delta_vs_fixed_AUC"] = summary["AUC"] - fixed_best["AUC"]
    summary["delta_vs_fixed_F1"] = summary["F1"] - fixed_best["F1"]
    summary["delta_vs_fixed_Recall"] = summary["Recall"] - fixed_best["Recall"]
    summary["delta_vs_fixed_Precision"] = summary["Precision"] - fixed_best["Precision"]

    summary_path = OUTPUT_DIR / "bk_ross_static_ensemble_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    logger.info(f"\nSaved comparison -> {comparison_path}")
    logger.info(f"Saved summary -> {summary_path}")
    logger.info("\nBest static ensemble by boundary:")
    logger.info(
        "\n"
        + summary[
            [
                "method",
                "drift_start_year",
                "old_window",
                "new_window",
                "combo",
                "AUC",
                "F1",
                "Recall",
                "Precision",
                "delta_vs_fixed_AUC",
                "delta_vs_fixed_F1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
