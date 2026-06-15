"""
Phase 5 - Bankruptcy ROSS Weighted Ensemble
===========================================

Motivation
----------
Phase 4 showed that ROSS selects 2008 as a meaningful drift boundary, but
equal Old/New averaging can dilute the stronger post-drift New pool. Phase 5
therefore tests whether a New-dominant weighted ensemble improves
imbalance-sensitive metrics.

Experiment
----------
Boundaries:
    Fixed: Old=1999-2011, New=2012-2014
    ROSS : Old=1999-2007, New=2008-2014

Feature variants:
    no_fs
    fs     (kbest_f, 50%, fitted on Old only)

Weighted prediction:
    p = (1 - w_new) * mean(Old3) + w_new * mean(New3)

where:
    Old3 = old_under / old_over / old_hybrid
    New3 = new_under / new_over / new_hybrid

Outputs
-------
    results/phase5_weighted/bk_ross_weight_sweep.csv
    results/phase5_weighted/bk_ross_weight_summary.csv

Usage
-----
    python experiments/phase5_weighted/bankruptcy_ross_weight_sweep.py
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
    POOL_SAMPLING,
    TRAIN_END_YEAR,
    _preprocess,
    _select_threshold,
    load_bankruptcy_with_year,
)
from experiments.phase4_drift.bankruptcy_ross_static_ensemble import (
    FIXED_DRIFT_START_YEAR,
    _old_new_split,
    _read_ross_boundary,
    _test_split,
)
from experiments.phase4_drift.bankruptcy_ross_fs_static_ensemble import (
    _apply_old_fit_fs,
)


OUTPUT_DIR = project_root / "results" / "phase5_weighted"
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def _train_old_new_mean_probas(
    X_old_raw: pd.DataFrame,
    y_old: np.ndarray,
    X_new_raw: pd.DataFrame,
    y_new: np.ndarray,
    X_test_raw: pd.DataFrame,
    tag: str,
    logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train Old3 and New3 XGBoost pools and return mean validation/test probas.

    Returns:
        y_val, old_val_mean, new_val_mean, old_test_mean, new_test_mean
    """
    X_old_s, X_new_s, X_te_s, _ = _preprocess(X_old_raw, X_new_raw, X_test_raw)

    n_old_val = max(1, int(len(X_old_s) * 0.2))
    n_new_val = max(1, int(len(X_new_s) * 0.2))
    X_old_fit = X_old_s.iloc[:-n_old_val]
    y_old_fit = y_old[:-n_old_val]
    X_new_fit = X_new_s.iloc[:-n_new_val]
    y_new_fit = y_new[:-n_new_val]

    X_val = pd.concat([X_old_s.iloc[-n_old_val:], X_new_s.iloc[-n_new_val:]], ignore_index=True)
    y_val = np.concatenate([y_old[-n_old_val:], y_new[-n_new_val:]])

    sampler = ImbalanceSampler()
    old_val_probas: list[np.ndarray] = []
    old_test_probas: list[np.ndarray] = []
    new_val_probas: list[np.ndarray] = []
    new_test_probas: list[np.ndarray] = []

    for sampling in POOL_SAMPLING:
        X_r, y_r = sampler.apply_sampling(X_old_fit, y_old_fit, strategy=sampling)
        model = XGBoostWrapper(name=f"{tag}_old_{sampling}", use_imbalance=False)
        model.fit(X_r, y_r)
        old_val_probas.append(model.predict_proba(X_val))
        old_test_probas.append(model.predict_proba(X_te_s))

        X_r, y_r = sampler.apply_sampling(X_new_fit, y_new_fit, strategy=sampling)
        model = XGBoostWrapper(name=f"{tag}_new_{sampling}", use_imbalance=False)
        model.fit(X_r, y_r)
        new_val_probas.append(model.predict_proba(X_val))
        new_test_probas.append(model.predict_proba(X_te_s))

    return (
        y_val,
        np.mean(old_val_probas, axis=0),
        np.mean(new_val_probas, axis=0),
        np.mean(old_test_probas, axis=0),
        np.mean(new_test_probas, axis=0),
    )


def _run_config(
    *,
    method: str,
    boundary_source: str,
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

    logger.info(
        f"\n[{method}|{fs_variant}] Old=1999-{drift_start_year - 1} "
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
    elif fs_variant != "no_fs":
        raise ValueError(f"Unknown fs_variant: {fs_variant}")

    y_val, old_val, new_val, old_test, new_test = _train_old_new_mean_probas(
        X_old,
        y_old,
        X_new,
        y_new,
        X_test_used,
        tag=f"{method}_{fs_variant}",
        logger=logger,
    )

    rows: list[dict] = []
    for w_new in WEIGHT_GRID:
        w_old = float(1.0 - w_new)
        val_proba = w_old * old_val + float(w_new) * new_val
        test_proba = w_old * old_test + float(w_new) * new_test
        threshold = _select_threshold(y_val, val_proba)
        metrics = compute_metrics(y_test, test_proba, threshold=threshold)

        rows.append(
            {
                "method": method,
                "boundary_source": boundary_source,
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
                **metrics,
            }
        )

    best = max(rows, key=lambda r: (r["F1"], r["AUC"]))
    logger.info(
        f"  best w_new={best['w_new']:.2f} | "
        f"AUC={best['AUC']:.4f} F1={best['F1']:.4f} "
        f"Recall={best['Recall']:.4f} Precision={best['Precision']:.4f}"
    )
    return rows


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
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
    ross_equal = df[
        (df["method"] == "ROSS_2008")
        & (df["fs_variant"] == "no_fs")
        & (np.isclose(df["w_new"], 0.5))
    ].iloc[0]

    summary = best.copy()
    summary["delta_vs_fixed_equal_AUC"] = summary["AUC"] - fixed_equal["AUC"]
    summary["delta_vs_fixed_equal_F1"] = summary["F1"] - fixed_equal["F1"]
    summary["delta_vs_fixed_equal_Precision"] = summary["Precision"] - fixed_equal["Precision"]
    summary["delta_vs_ross_equal_AUC"] = summary["AUC"] - ross_equal["AUC"]
    summary["delta_vs_ross_equal_F1"] = summary["F1"] - ross_equal["F1"]
    summary["delta_vs_ross_equal_Precision"] = summary["Precision"] - ross_equal["Precision"]
    return summary


def main() -> None:
    logger = get_logger("Phase5_BK_ROSS_Weighted", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ross_start = _read_ross_boundary()
    X_all, y_all = load_bankruptcy_with_year(logger)
    train_mask = (X_all["fyear"] >= 1999) & (X_all["fyear"] <= TRAIN_END_YEAR)
    X_train_all = X_all.loc[train_mask].reset_index(drop=True)
    y_train_all = y_all.loc[train_mask].reset_index(drop=True)
    X_test, y_test = _test_split(X_all, y_all)

    configs = [
        ("Fixed_2012", "manual fixed boundary", FIXED_DRIFT_START_YEAR),
        ("ROSS_2008", "ROSS selected boundary", ross_start),
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

    df = pd.DataFrame(rows)
    sweep_path = OUTPUT_DIR / "bk_ross_weight_sweep.csv"
    df.to_csv(sweep_path, index=False, float_format="%.6f")

    summary = _build_summary(df)
    summary_path = OUTPUT_DIR / "bk_ross_weight_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    logger.info(f"\nSaved sweep -> {sweep_path}")
    logger.info(f"Saved summary -> {summary_path}")
    logger.info("\nBest weighted ensemble by boundary and FS variant:")
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
                "delta_vs_ross_equal_F1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
