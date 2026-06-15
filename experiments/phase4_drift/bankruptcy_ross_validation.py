"""
Phase 4/5 - Validation-based ROSS-WE for Bankruptcy
===================================================

This is the leakage-safe version of ROSS:

1. Candidate boundary search uses only 1999-2011 for training and
   2012-2014 for validation.
2. The final evaluation uses 2015-2018 only once as the held-out test set.

Compared with the earlier Oracle ROSS analysis, this script does not use
the 2015-2018 test period to select the drift boundary.

Outputs
-------
Phase 4:
    results/phase4_drift/bk_validation_ross_candidates.csv
    results/phase4_drift/bk_validation_ross_selected_boundary.csv

Phase 5:
    results/phase5_weighted/bk_validation_ross_weight_sweep.csv
    results/phase5_weighted/bk_validation_ross_weight_summary.csv

Usage
-----
    python experiments/phase4_drift/bankruptcy_ross_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import ImbalanceSampler
from src.evaluation import compute_metrics
from src.models import XGBoostWrapper
from src.utils import get_logger, set_seed

from experiments.phase4_drift.bankruptcy_drift_stream import (
    SAMPLING_STRATEGIES,
    TRAIN_END_YEAR,
    _preprocess,
    _select_threshold,
    load_bankruptcy_with_year,
)
from experiments.phase4_drift.bankruptcy_ross_static_ensemble import (
    FIXED_DRIFT_START_YEAR,
    _test_split,
)
from experiments.phase5_weighted.bankruptcy_ross_weight_sweep import (
    _run_config,
)


PHASE4_OUT = project_root / "results" / "phase4_drift"
PHASE5_OUT = project_root / "results" / "phase5_weighted"

SEARCH_START_YEAR = 1999
SEARCH_END_YEAR = 2011
VALIDATION_START_YEAR = 2012
VALIDATION_END_YEAR = 2014


def _eval_single_on_validation(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_val_raw: pd.DataFrame,
    y_val: np.ndarray,
    sampling: str,
    tag: str,
) -> dict:
    """Train one XGB model and evaluate on the boundary-selection validation period."""
    X_train_s, X_val_s, _ = _preprocess(X_train_raw, X_val_raw)
    sampler = ImbalanceSampler()
    X_r, y_r = sampler.apply_sampling(X_train_s, y_train, strategy=sampling)
    model = XGBoostWrapper(name=f"val_ross_{tag}_{sampling}", use_imbalance=False)
    model.fit(X_r, y_r)
    proba = model.predict_proba(X_val_s)
    threshold = _select_threshold(y_val, proba)
    metrics = compute_metrics(y_val, proba, threshold=threshold)
    return {"sampling": sampling, "threshold": threshold, **metrics}


def _best_single_on_validation(
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    X_val_raw: pd.DataFrame,
    y_val: np.ndarray,
    tag: str,
) -> dict:
    rows = [
        _eval_single_on_validation(X_train_raw, y_train, X_val_raw, y_val, sampling, tag)
        for sampling in SAMPLING_STRATEGIES
    ]
    return sorted(rows, key=lambda r: (r["AUC"], r["F1"], r["Precision"]), reverse=True)[0]


def build_validation_candidates(
    X_all: pd.DataFrame,
    y_all: pd.Series,
    logger,
) -> pd.DataFrame:
    """Enumerate drift boundaries inside 1999-2011 and select by 2012-2014 validation."""
    val_mask = (
        (X_all["fyear"] >= VALIDATION_START_YEAR)
        & (X_all["fyear"] <= VALIDATION_END_YEAR)
    )
    X_val = X_all.loc[val_mask].drop(columns=["fyear"]).reset_index(drop=True)
    y_val = np.asarray(y_all.loc[val_mask])

    rows = []
    for old_end_year in range(SEARCH_START_YEAR, SEARCH_END_YEAR):
        drift_start_year = old_end_year + 1
        old_mask = (X_all["fyear"] >= SEARCH_START_YEAR) & (X_all["fyear"] <= old_end_year)
        new_mask = (X_all["fyear"] >= drift_start_year) & (X_all["fyear"] <= SEARCH_END_YEAR)

        X_old = X_all.loc[old_mask].drop(columns=["fyear"]).reset_index(drop=True)
        y_old = np.asarray(y_all.loc[old_mask])
        X_new = X_all.loc[new_mask].drop(columns=["fyear"]).reset_index(drop=True)
        y_new = np.asarray(y_all.loc[new_mask])

        logger.info(
            f"Candidate drift_start={drift_start_year}: "
            f"Old=1999-{old_end_year} ({len(X_old):,}), "
            f"New={drift_start_year}-{SEARCH_END_YEAR} ({len(X_new):,})"
        )

        old_best = _best_single_on_validation(X_old, y_old, X_val, y_val, f"old_{old_end_year}")
        new_best = _best_single_on_validation(X_new, y_new, X_val, y_val, f"new_{drift_start_year}")

        rows.append(
            {
                "old_end_year": old_end_year,
                "drift_start_year": drift_start_year,
                "old_window": f"{SEARCH_START_YEAR}-{old_end_year}",
                "new_window": f"{drift_start_year}-{SEARCH_END_YEAR}",
                "validation_window": f"{VALIDATION_START_YEAR}-{VALIDATION_END_YEAR}",
                "old_n": len(X_old),
                "new_n": len(X_new),
                "val_n": len(X_val),
                "best_old_sampling": old_best["sampling"],
                "best_new_sampling": new_best["sampling"],
                "old_val_AUC": old_best["AUC"],
                "new_val_AUC": new_best["AUC"],
                "gap_new_minus_old_val_AUC": new_best["AUC"] - old_best["AUC"],
                "old_val_F1": old_best["F1"],
                "new_val_F1": new_best["F1"],
                "gap_new_minus_old_val_F1": new_best["F1"] - old_best["F1"],
                "old_val_Recall": old_best["Recall"],
                "new_val_Recall": new_best["Recall"],
                "old_val_Precision": old_best["Precision"],
                "new_val_Precision": new_best["Precision"],
            }
        )

    candidates = pd.DataFrame(rows)
    candidates["rank_by_new_val_auc"] = candidates["new_val_AUC"].rank(
        ascending=False, method="min"
    ).astype(int)
    candidates["rank_by_new_val_f1"] = candidates["new_val_F1"].rank(
        ascending=False, method="min"
    ).astype(int)
    return candidates.sort_values(["rank_by_new_val_auc", "rank_by_new_val_f1"])


def _build_weight_summary(df: pd.DataFrame) -> pd.DataFrame:
    best = (
        df.sort_values(["method", "fs_variant", "F1", "AUC"], ascending=[True, True, False, False])
        .groupby(["method", "fs_variant"], as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    fixed_equal = df[
        (df["method"] == "Fixed_2012")
        & (df["fs_variant"] == "no_fs")
        & (np.isclose(df["w_new"], 0.5))
    ].iloc[0]
    summary = best.copy()
    summary["delta_vs_fixed_equal_AUC"] = summary["AUC"] - fixed_equal["AUC"]
    summary["delta_vs_fixed_equal_F1"] = summary["F1"] - fixed_equal["F1"]
    summary["delta_vs_fixed_equal_Precision"] = summary["Precision"] - fixed_equal["Precision"]
    return summary


def run_final_weighted_eval(
    selected_drift_start: int,
    X_all: pd.DataFrame,
    y_all: pd.Series,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """After validation selection, retrain on 1999-2014 and test on 2015-2018."""
    train_mask = (X_all["fyear"] >= 1999) & (X_all["fyear"] <= TRAIN_END_YEAR)
    X_train_all = X_all.loc[train_mask].reset_index(drop=True)
    y_train_all = y_all.loc[train_mask].reset_index(drop=True)
    X_test, y_test = _test_split(X_all, y_all)

    configs = [
        ("Fixed_2012", "manual fixed boundary", FIXED_DRIFT_START_YEAR),
        (f"ValROSS_{selected_drift_start}", "validation-selected ROSS boundary", selected_drift_start),
    ]

    rows: list[dict] = []
    for method, source, drift_start in configs:
        for fs_variant in ["no_fs", "fs"]:
            rows.extend(
                _run_config(
                    method=method,
                    boundary_source=source,
                    drift_start_year=drift_start,
                    fs_variant=fs_variant,
                    X_train_all=X_train_all,
                    y_train_all=y_train_all,
                    X_test=X_test,
                    y_test=y_test,
                    logger=logger,
                )
            )

    sweep = pd.DataFrame(rows)
    summary = _build_weight_summary(sweep)
    return sweep, summary


def main() -> None:
    logger = get_logger("Phase4_Validation_ROSS", console=True, file=True)
    set_seed(42)
    PHASE4_OUT.mkdir(parents=True, exist_ok=True)
    PHASE5_OUT.mkdir(parents=True, exist_ok=True)

    X_all, y_all = load_bankruptcy_with_year(logger)
    candidates = build_validation_candidates(X_all, y_all, logger)
    selected = candidates.iloc[0].copy()
    selected_start = int(selected["drift_start_year"])

    candidates_path = PHASE4_OUT / "bk_validation_ross_candidates.csv"
    selected_path = PHASE4_OUT / "bk_validation_ross_selected_boundary.csv"
    candidates.to_csv(candidates_path, index=False, float_format="%.6f")
    pd.DataFrame([selected]).to_csv(selected_path, index=False, float_format="%.6f")

    logger.info(f"\nSelected validation ROSS boundary: {selected_start}")
    logger.info(f"Saved validation candidates -> {candidates_path}")
    logger.info(f"Saved selected boundary -> {selected_path}")

    sweep, summary = run_final_weighted_eval(selected_start, X_all, y_all, logger)
    sweep_path = PHASE5_OUT / "bk_validation_ross_weight_sweep.csv"
    summary_path = PHASE5_OUT / "bk_validation_ross_weight_summary.csv"
    sweep.to_csv(sweep_path, index=False, float_format="%.6f")
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    logger.info(f"\nSaved validation ROSS weight sweep -> {sweep_path}")
    logger.info(f"Saved validation ROSS weight summary -> {summary_path}")
    logger.info("\nValidation ROSS weighted summary:")
    logger.info(
        "\n"
        + summary[
            [
                "method",
                "fs_variant",
                "drift_start_year",
                "w_new",
                "AUC",
                "F1",
                "Recall",
                "Precision",
                "delta_vs_fixed_equal_F1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
