"""
Multi-boundary ROSS (k=2) for Bankruptcy
=========================================
Generalisation of DAWCE to k=2 drift boundaries.

Standard ROSS (k=1):
    |--- Old ---|--- New ---|--- Val ---|--- Test ---|
    Searches ONE boundary b* to maximise Validation F1.

Multi-boundary ROSS (k=2):
    |--- P1 ---|--- P2 ---|--- P3 ---|--- Val ---|--- Test ---|
    Searches TWO boundaries (b1*, b2*) and THREE group weights
    (w1, w2, w3) jointly to maximise Validation F1.

Prediction:
    p_hat = w1 * mean_proba(P1) + w2 * mean_proba(P2) + w3 * mean_proba(P3)

k selection:
    Compare validation F1 of k=1 vs k=2 (adjusted for complexity).
    If k=2 improves val-F1 by >= IMPROVEMENT_THRESHOLD, use k=2.

Outputs
-------
    results/phase4_drift/bk_multi_boundary_k2_candidates.csv
    results/phase4_drift/bk_multi_boundary_k2_selected.csv
    results/phase4_drift/bk_multi_boundary_k_comparison.csv

Usage
-----
    python experiments/phase4_drift/bankruptcy_multi_boundary_ross.py
"""
from __future__ import annotations

import itertools
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

OUTPUT_DIR = project_root / "results" / "phase4_drift"

# ── experiment settings ────────────────────────────────────────────────────
TRAIN_START        = 1999
TRAIN_SEARCH_END   = 2011   # candidate search uses 1999-2011 ONLY (leakage-safe)
VAL_START          = 2012   # validation period for boundary selection
VAL_END            = 2014
TEST_START         = 2015
TEST_END           = 2018

# Minimum years per period (avoid tiny pools)
MIN_PERIOD_YEARS = 3
# Boundary search range: each period must have >= MIN_PERIOD_YEARS
# Last period ends at TRAIN_SEARCH_END=2011, so the last boundary must be <= 2011-3=2008
SEARCH_MIN = TRAIN_START + MIN_PERIOD_YEARS           # 2002
SEARCH_MAX = TRAIN_SEARCH_END - MIN_PERIOD_YEARS + 1  # 2009

# k=2 improvement threshold to justify extra complexity
K2_IMPROVEMENT_THRESHOLD = 0.005   # val-F1 must improve by at least 0.5%

# Weight grid for k=1 (same as original DAWCE)
W_GRID_1 = np.round(np.arange(0.0, 1.0001, 0.05), 2)

# Weight grid for k=2: w1, w2; w3 = 1 - w1 - w2
# Use coarser grid (0.1 step) to keep computation manageable
W_STEP_2 = 0.1
W_VALS   = np.round(np.arange(0.0, 1.0001, W_STEP_2), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _yr(X_all, y_all, yr_from, yr_to):
    m = (X_all["fyear"] >= yr_from) & (X_all["fyear"] <= yr_to)
    X = X_all.loc[m].drop(columns=["fyear"]).reset_index(drop=True)
    y = np.asarray(y_all.loc[m])
    return X, y


def get_val_test(X_all, y_all):
    X_val,  y_val  = _yr(X_all, y_all, VAL_START,  VAL_END)
    X_test, y_test = _yr(X_all, y_all, TEST_START, TEST_END)
    return X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Model pool helpers
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_pool(X_fit, y_fit, X_val_s, X_test_s, tag, sampler):
    """Train 3 models (one per sampling strategy) and return mean probas."""
    val_probas, test_probas = [], []
    for sampling in POOL_SAMPLING:
        X_r, y_r = sampler.apply_sampling(X_fit, y_fit, strategy=sampling)
        m = XGBoostWrapper(name=f"{tag}_{sampling}", use_imbalance=False)
        m.fit(X_r, y_r)
        val_probas.append(m.predict_proba(X_val_s))
        test_probas.append(m.predict_proba(X_test_s))
    return np.mean(val_probas, axis=0), np.mean(test_probas, axis=0)


def train_pools(periods_train, X_val, X_test, tag_prefix, logger):
    """
    Train one 3-model pool per period.

    Parameters
    ----------
    periods_train : list[(X_raw, y_array, label_str)]

    Returns
    -------
    val_means  : list[np.ndarray]   (one per period)
    test_means : list[np.ndarray]
    """
    # Fit scaler on the FIRST (oldest) period only to avoid leakage
    # _preprocess(X_train, *others) returns (X_train_s, *others_s, scaler)
    n_periods = len(periods_train)
    X0_raw     = periods_train[0][0]
    other_Xs   = [p[0] for p in periods_train[1:]] + [X_val, X_test]

    scaled = _preprocess(X0_raw, *other_Xs)
    # scaled indices: 0..n_periods-1 = period scaled dfs, -3=X_val_s, -2=X_test_s, -1=scaler
    X_val_s  = scaled[n_periods]        # n_periods-th other = X_val
    X_test_s = scaled[n_periods + 1]    # (n_periods+1)-th other = X_test

    # Re-assemble period scaled dataframes
    period_scaled = [scaled[0]] + [scaled[i] for i in range(1, n_periods)]

    sampler = ImbalanceSampler()
    val_means, test_means = [], []

    for i, (_, y_fit, label) in enumerate(periods_train):
        X_fit_s = period_scaled[i]
        logger.info(
            f"  [{tag_prefix}] Training pool for {label}: "
            f"{len(X_fit_s):,} rows, {y_fit.mean()*100:.1f}% positive"
        )
        vm, tm = _train_one_pool(X_fit_s, y_fit, X_val_s, X_test_s,
                                  f"{tag_prefix}_{label}", sampler)
        val_means.append(vm)
        test_means.append(tm)

    return val_means, test_means, X_val_s, X_test_s


# ─────────────────────────────────────────────────────────────────────────────
# k=1 ROSS (single boundary)
# ─────────────────────────────────────────────────────────────────────────────

def search_k1(X_all, y_all, X_val, y_val, X_test, y_test, logger):
    """
    Original ROSS: enumerate single boundary b in [SEARCH_MIN, SEARCH_MAX].
    Returns DataFrame of all candidates + best row.
    """
    logger.info("\n" + "="*60)
    logger.info("k=1 ROSS boundary search")
    logger.info("="*60)
    rows = []

    for b in range(SEARCH_MIN, SEARCH_MAX + 1):
        X_p1, y_p1 = _yr(X_all, y_all, TRAIN_START, b - 1)
        X_p2, y_p2 = _yr(X_all, y_all, b,           TRAIN_SEARCH_END)

        if len(X_p1) < 100 or len(X_p2) < 100:
            continue

        tag = f"k1_b{b}"
        periods = [(X_p1, y_p1, f"P1_{TRAIN_START}-{b-1}"),
                   (X_p2, y_p2, f"P2_{b}-{TRAIN_END_YEAR}")]
        vm, tm, _, _ = train_pools(periods, X_val, X_test, tag, logger)

        best_f1, best_w2, best_thr = -1.0, 0.5, 0.5
        for w2 in W_GRID_1:
            w1 = float(1.0 - w2)
            vp = w1 * vm[0] + float(w2) * vm[1]
            thr = _select_threshold(y_val, vp)
            f1  = compute_metrics(y_val, vp, threshold=thr)["F1"]
            if f1 > best_f1:
                best_f1, best_w2, best_thr = f1, float(w2), thr

        best_w1 = round(1.0 - best_w2, 2)
        vp_best = best_w1 * vm[0] + best_w2 * vm[1]
        tp_best = best_w1 * tm[0] + best_w2 * tm[1]
        val_m   = compute_metrics(y_val,  vp_best, threshold=best_thr)
        test_m  = compute_metrics(y_test, tp_best, threshold=best_thr)

        rows.append({
            "k":            1,
            "b1":           b,
            "b2":           None,
            "periods":      f"{TRAIN_START}-{b-1} | {b}-{TRAIN_SEARCH_END}",
            "n_p1":         len(y_p1),
            "n_p2":         len(y_p2),
            "w1":           best_w1,
            "w2":           best_w2,
            "w3":           None,
            "val_F1":       val_m["F1"],
            "val_AUC":      val_m["AUC"],
            "test_F1":      test_m["F1"],
            "test_AUC":     test_m["AUC"],
            "test_Recall":  test_m["Recall"],
            "test_Precision": test_m["Precision"],
        })
        logger.info(f"  k=1 b={b}: val_F1={val_m['F1']:.4f}  test_F1={test_m['F1']:.4f}")

    df = pd.DataFrame(rows).sort_values("val_F1", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# k=2 ROSS (two boundaries)
# ─────────────────────────────────────────────────────────────────────────────

def search_k2(X_all, y_all, X_val, y_val, X_test, y_test, logger):
    """
    Multi-boundary ROSS with k=2: enumerate all (b1, b2) pairs.
    """
    logger.info("\n" + "="*60)
    logger.info("k=2 ROSS boundary search")
    logger.info("="*60)

    # Generate all valid (b1, b2) with b1 < b2, min gap enforced
    boundary_candidates = []
    year_range = list(range(SEARCH_MIN, SEARCH_MAX + 1))
    for b1, b2 in itertools.combinations(year_range, 2):
        if (b2 - b1) < MIN_PERIOD_YEARS:
            continue
        # P3 must have at least MIN_PERIOD_YEARS (P3 ends at TRAIN_SEARCH_END)
        if (TRAIN_SEARCH_END - b2 + 1) < MIN_PERIOD_YEARS:
            continue
        boundary_candidates.append((b1, b2))

    logger.info(f"  Total (b1, b2) candidate pairs: {len(boundary_candidates)}")
    rows = []

    for b1, b2 in boundary_candidates:
        X_p1, y_p1 = _yr(X_all, y_all, TRAIN_START, b1 - 1)
        X_p2, y_p2 = _yr(X_all, y_all, b1,          b2 - 1)
        X_p3, y_p3 = _yr(X_all, y_all, b2,           TRAIN_SEARCH_END)

        if any(len(y) < 50 for y in [y_p1, y_p2, y_p3]):
            continue

        tag = f"k2_b{b1}_{b2}"
        periods = [
            (X_p1, y_p1, f"P1_{TRAIN_START}-{b1-1}"),
            (X_p2, y_p2, f"P2_{b1}-{b2-1}"),
            (X_p3, y_p3, f"P3_{b2}-{TRAIN_END_YEAR}"),
        ]
        vm, tm, _, _ = train_pools(periods, X_val, X_test, tag, logger)

        best_f1 = -1.0
        best_w = (1/3, 1/3, 1/3)

        # Grid search over (w1, w2); w3 = 1 - w1 - w2
        for w1 in W_VALS:
            for w2 in W_VALS:
                w3 = round(1.0 - float(w1) - float(w2), 6)
                if w3 < -1e-9:
                    continue
                w3 = max(0.0, w3)
                vp = float(w1) * vm[0] + float(w2) * vm[1] + w3 * vm[2]
                thr = _select_threshold(y_val, vp)
                f1  = compute_metrics(y_val, vp, threshold=thr)["F1"]
                if f1 > best_f1:
                    best_f1 = f1
                    best_w  = (float(w1), float(w2), w3)

        w1, w2, w3 = best_w
        vp_best = w1 * vm[0] + w2 * vm[1] + w3 * vm[2]
        tp_best = w1 * tm[0] + w2 * tm[1] + w3 * tm[2]
        best_thr = _select_threshold(y_val, vp_best)

        val_m  = compute_metrics(y_val,  vp_best, threshold=best_thr)
        test_m = compute_metrics(y_test, tp_best, threshold=best_thr)

        rows.append({
            "k":             2,
            "b1":            b1,
            "b2":            b2,
            "periods":       f"{TRAIN_START}-{b1-1} | {b1}-{b2-1} | {b2}-{TRAIN_SEARCH_END}",
            "n_p1":          len(y_p1),
            "n_p2":          len(y_p2),
            "n_p3":          len(y_p3),
            "w1":            round(w1, 2),
            "w2":            round(w2, 2),
            "w3":            round(w3, 2),
            "val_F1":        val_m["F1"],
            "val_AUC":       val_m["AUC"],
            "test_F1":       test_m["F1"],
            "test_AUC":      test_m["AUC"],
            "test_Recall":   test_m["Recall"],
            "test_Precision": test_m["Precision"],
        })
        logger.info(
            f"  k=2 ({b1},{b2}): val_F1={val_m['F1']:.4f}  test_F1={test_m['F1']:.4f}  "
            f"w=({w1:.1f},{w2:.1f},{w3:.1f})"
        )

    df = pd.DataFrame(rows).sort_values("val_F1", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# k selection comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_k(df_k1: pd.DataFrame, df_k2: pd.DataFrame) -> pd.DataFrame:
    """
    Compare best k=1 vs best k=2 on validation F1.
    Returns summary DataFrame.
    """
    best_k1 = df_k1.iloc[0]
    best_k2 = df_k2.iloc[0]

    k2_better = best_k2["val_F1"] - best_k1["val_F1"]
    selected_k = 2 if k2_better >= K2_IMPROVEMENT_THRESHOLD else 1
    selected   = best_k2 if selected_k == 2 else best_k1

    rows = [
        {
            "k":              1,
            "best_boundaries": f"b*={int(best_k1['b1'])}",
            "val_F1":         round(best_k1["val_F1"],  4),
            "val_AUC":        round(best_k1["val_AUC"], 4),
            "test_F1":        round(best_k1["test_F1"], 4),
            "test_AUC":       round(best_k1["test_AUC"],4),
            "delta_val_F1":   0.0,
            "selected":       selected_k == 1,
        },
        {
            "k":              2,
            "best_boundaries": f"b1*={int(best_k2['b1'])}, b2*={int(best_k2['b2'])}",
            "val_F1":         round(best_k2["val_F1"],  4),
            "val_AUC":        round(best_k2["val_AUC"], 4),
            "test_F1":        round(best_k2["test_F1"], 4),
            "test_AUC":       round(best_k2["test_AUC"],4),
            "delta_val_F1":   round(k2_better, 4),
            "selected":       selected_k == 2,
        },
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger = get_logger("MultiBoundaryROSS", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_all, y_all = load_bankruptcy_with_year(logger)
    X_val, y_val, X_test, y_test = get_val_test(X_all, y_all)

    logger.info(
        f"Val: {VAL_START}-{VAL_END} ({len(y_val):,} rows)  "
        f"Test: {TEST_START}-{TEST_END} ({len(y_test):,} rows)"
    )

    # ── k=1 search ────────────────────────────────────────────────────────
    df_k1 = search_k1(X_all, y_all, X_val, y_val, X_test, y_test, logger)
    out_k1 = OUTPUT_DIR / "bk_multi_boundary_k1_candidates.csv"
    df_k1.to_csv(out_k1, index=False, float_format="%.6f")
    logger.info(f"\nk=1 candidates saved -> {out_k1}")
    logger.info(f"Best k=1: boundary={int(df_k1.iloc[0]['b1'])}  "
                f"val_F1={df_k1.iloc[0]['val_F1']:.4f}  "
                f"test_F1={df_k1.iloc[0]['test_F1']:.4f}")

    # ── k=2 search ────────────────────────────────────────────────────────
    df_k2 = search_k2(X_all, y_all, X_val, y_val, X_test, y_test, logger)
    out_k2 = OUTPUT_DIR / "bk_multi_boundary_k2_candidates.csv"
    df_k2.to_csv(out_k2, index=False, float_format="%.6f")
    logger.info(f"\nk=2 candidates saved -> {out_k2}")
    logger.info(f"Best k=2: b1={int(df_k2.iloc[0]['b1'])}, b2={int(df_k2.iloc[0]['b2'])}  "
                f"val_F1={df_k2.iloc[0]['val_F1']:.4f}  "
                f"test_F1={df_k2.iloc[0]['test_F1']:.4f}")

    # ── k comparison ──────────────────────────────────────────────────────
    cmp = compare_k(df_k1, df_k2)
    out_cmp = OUTPUT_DIR / "bk_multi_boundary_k_comparison.csv"
    cmp.to_csv(out_cmp, index=False, float_format="%.6f")

    print("\n" + "="*70)
    print("Multi-boundary ROSS: k=1 vs k=2 comparison")
    print("="*70)
    print(cmp.to_string(index=False))
    print(f"\nSaved: {out_k1}")
    print(f"Saved: {out_k2}")
    print(f"Saved: {out_cmp}")

    selected_row = cmp[cmp["selected"] == True].iloc[0]
    print(f"\n>>> Selected k = {int(selected_row['k'])} "
          f"(threshold for k=2 adoption = +{K2_IMPROVEMENT_THRESHOLD:.3f} val-F1)")


if __name__ == "__main__":
    main()
