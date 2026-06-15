"""
AWE vs DAWCE Comparison (Bankruptcy)
=====================================
Implements Accuracy Weighted Ensemble (AWE; Street & Kim, 2001) in the
batch-yearly setting and compares it against DAWCE (group weighting) and
Equal-weight baseline on the same train/val/test split.

AWE in this setting
-------------------
All 6 base classifiers (Old_{under,over,hybrid}, New_{under,over,hybrid})
are trained on their respective periods.  Each model's individual AUC on
the *Validation* period (2012-2014) is used as its accuracy weight.
Models with AUC < 0.5 (below random) are pruned.  Final prediction:

    p_AWE = sum_i(w_i * p_i)   where w_i = AUC_i / sum_j(AUC_j)

Comparison methods
------------------
1. Equal_weight   – all 6 models, w = 1/6
2. AWE            – per-model validation-AUC weight
3. DAWCE          – group val-F1 grid search (val-selected w_old / w_new)

Boundaries tested: ROSS_2009 and Fixed_2012  x  FS / no-FS

Outputs
-------
    results/phase5_weighted/bk_awe_comparison.csv
    results/phase5_weighted/bk_awe_model_weights.csv

Usage
-----
    python experiments/phase5_weighted/awe_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data import ImbalanceSampler
from src.evaluation import compute_metrics
from src.features import FeatureSelector
from src.models import XGBoostWrapper
from src.utils import get_logger, set_seed

from experiments.phase4_drift.bankruptcy_drift_stream import (
    POOL_SAMPLING,
    TRAIN_END_YEAR,
    _preprocess,
    _select_threshold,
    load_bankruptcy_with_year,
)
from experiments.phase4_drift.bankruptcy_ross_static_ensemble import FIXED_DRIFT_START_YEAR
from experiments.phase4_drift.bankruptcy_ross_fs_static_ensemble import FS_METHOD, FS_RATIO

OUTPUT_DIR   = project_root / "results" / "phase5_weighted"
ROSS_BOUNDARY  = 2009
FIXED_BOUNDARY = FIXED_DRIFT_START_YEAR   # 2012
TRAIN_START    = 1999
VAL_START, VAL_END   = 2012, 2014
TEST_START, TEST_END = 2015, 2018
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Data splitting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _yr_split(X_all, y_all, yr_from, yr_to):
    m = (X_all["fyear"] >= yr_from) & (X_all["fyear"] <= yr_to)
    return (
        X_all.loc[m].drop(columns=["fyear"]).reset_index(drop=True),
        np.asarray(y_all.loc[m]),
    )


def get_periods(X_all, y_all, boundary):
    X_old, y_old   = _yr_split(X_all, y_all, TRAIN_START, boundary - 1)
    X_new, y_new   = _yr_split(X_all, y_all, boundary,    TRAIN_END_YEAR)
    X_val, y_val   = _yr_split(X_all, y_all, VAL_START,   VAL_END)
    X_test, y_test = _yr_split(X_all, y_all, TEST_START,  TEST_END)
    return X_old, y_old, X_new, y_new, X_val, y_val, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection (fit on Old, transform New / Val / Test)
# ─────────────────────────────────────────────────────────────────────────────

def apply_fs(X_old, y_old, X_new, X_val, X_test, logger):
    fill_vals = X_old.mean(numeric_only=True)
    k = max(1, int(X_old.shape[1] * FS_RATIO))
    sel = FeatureSelector(method=FS_METHOD, k=k)
    X_old_fs  = sel.fit_transform(X_old.fillna(fill_vals), y_old)
    X_new_fs  = sel.transform(X_new.fillna(fill_vals))
    X_val_fs  = sel.transform(X_val.fillna(fill_vals))
    X_test_fs = sel.transform(X_test.fillna(fill_vals))
    logger.info(f"  FS: {X_old.shape[1]} -> {sel.n_selected} features")
    return X_old_fs, X_new_fs, X_val_fs, X_test_fs


# ─────────────────────────────────────────────────────────────────────────────
# Train 6-model pool, return per-model info
# ─────────────────────────────────────────────────────────────────────────────

def train_pool(X_old, y_old, X_new, y_new, X_val, y_val, X_test, tag, logger):
    """
    Returns
    -------
    models : list[dict]  with keys: name, side, sampling, val_auc,
                          val_proba, test_proba
    """
    # Fit scaler on Old only; transform New / Val / Test
    X_old_s, X_new_s, X_val_s, X_test_s, _ = _preprocess(X_old, X_new, X_val, X_test)

    sampler = ImbalanceSampler()
    models = []

    for side, X_fit, y_fit in [("old", X_old_s, y_old), ("new", X_new_s, y_new)]:
        for sampling in POOL_SAMPLING:
            X_r, y_r = sampler.apply_sampling(X_fit, y_fit, strategy=sampling)
            m = XGBoostWrapper(name=f"{tag}_{side}_{sampling}", use_imbalance=False)
            m.fit(X_r, y_r)

            vp = m.predict_proba(X_val_s)
            tp = m.predict_proba(X_test_s)
            va = float(roc_auc_score(y_val, vp))

            models.append({
                "name":       f"{side}_{sampling}",
                "side":       side,
                "sampling":   sampling,
                "val_auc":    va,
                "val_proba":  vp,
                "test_proba": tp,
            })
            logger.info(f"  {side}_{sampling}: val_AUC={va:.4f}")

    return models


# ─────────────────────────────────────────────────────────────────────────────
# Ensembling methods
# ─────────────────────────────────────────────────────────────────────────────

def equal_weight(models, y_val, y_test):
    vp = np.mean([m["val_proba"]  for m in models], axis=0)
    tp = np.mean([m["test_proba"] for m in models], axis=0)
    thr = _select_threshold(y_val, vp)
    return compute_metrics(y_test, tp, threshold=thr), {"w_old": 1/6, "w_new": 1/6}, []


def awe_weight(models, y_val, y_test):
    """Per-model AUC-normalised weighting; prune < 0.5 AUC."""
    active = [m for m in models if m["val_auc"] >= 0.5] or models
    raw_w  = np.array([m["val_auc"] for m in active], dtype=float)
    norm_w = raw_w / raw_w.sum()

    vp = sum(w * m["val_proba"]  for w, m in zip(norm_w, active))
    tp = sum(w * m["test_proba"] for w, m in zip(norm_w, active))
    thr = _select_threshold(y_val, vp)

    weight_info = [
        {"name": m["name"], "side": m["side"], "val_auc": m["val_auc"], "awe_weight": float(w)}
        for w, m in zip(norm_w, active)
    ]
    old_w = float(np.mean([w for w, m in zip(norm_w, active) if m["side"] == "old"]))
    new_w = float(np.mean([w for w, m in zip(norm_w, active) if m["side"] == "new"]))
    return compute_metrics(y_test, tp, threshold=thr), {"w_old": old_w, "w_new": new_w}, weight_info


def dawce_group_weight(models, y_val, y_test):
    """Val-F1 grid search over w_new in group-level weighting."""
    old_vp = np.mean([m["val_proba"]  for m in models if m["side"] == "old"], axis=0)
    new_vp = np.mean([m["val_proba"]  for m in models if m["side"] == "new"], axis=0)
    old_tp = np.mean([m["test_proba"] for m in models if m["side"] == "old"], axis=0)
    new_tp = np.mean([m["test_proba"] for m in models if m["side"] == "new"], axis=0)

    best_f1, best_w_new = -1.0, 0.5
    for w_new in WEIGHT_GRID:
        w_old = 1.0 - float(w_new)
        vp = w_old * old_vp + float(w_new) * new_vp
        thr = _select_threshold(y_val, vp)
        f1  = compute_metrics(y_val, vp, threshold=thr)["F1"]
        if f1 > best_f1:
            best_f1, best_w_new = f1, float(w_new)

    best_w_old = 1.0 - best_w_new
    tp  = best_w_old * old_tp + best_w_new * new_tp
    vp2 = best_w_old * old_vp + best_w_new * new_vp
    thr = _select_threshold(y_val, vp2)
    return compute_metrics(y_test, tp, threshold=thr), {"w_old": round(best_w_old, 2), "w_new": round(best_w_new, 2)}, []


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(X_all, y_all, logger):
    results, weight_records = [], []

    configs = [
        ("ROSS_2009_noFS",  ROSS_BOUNDARY,  False),
        ("ROSS_2009_FS",    ROSS_BOUNDARY,  True),
        ("Fixed_2012_noFS", FIXED_BOUNDARY, False),
        ("Fixed_2012_FS",   FIXED_BOUNDARY, True),
    ]

    for label, boundary, use_fs in configs:
        logger.info(f"\n{'='*60}\n{label}\n{'='*60}")

        X_old, y_old, X_new, y_new, X_val, y_val, X_test, y_test = \
            get_periods(X_all, y_all, boundary)

        if use_fs:
            X_old, X_new, X_val, X_test = apply_fs(X_old, y_old, X_new, X_val, X_test, logger)

        logger.info(
            f"  Old 1999-{boundary-1}: {len(X_old):,}  "
            f"New {boundary}-{TRAIN_END_YEAR}: {len(X_new):,}  "
            f"Val {VAL_START}-{VAL_END}: {len(X_val):,}  "
            f"Test {TEST_START}-{TEST_END}: {len(X_test):,}"
        )

        models = train_pool(X_old, y_old, X_new, y_new, X_val, y_val, X_test, label, logger)

        for method_name, fn in [
            ("Equal_weight", equal_weight),
            ("AWE",          awe_weight),
            ("DAWCE",        dawce_group_weight),
        ]:
            metrics, weights, winfo = fn(models, y_val, y_test)
            results.append({
                "config":    label,
                "boundary":  boundary,
                "fs":        use_fs,
                "method":    method_name,
                "w_old":     weights["w_old"],
                "w_new":     weights["w_new"],
                **metrics,
            })
            logger.info(
                f"  [{method_name:<14}] AUC={metrics['AUC']:.4f}  "
                f"F1={metrics['F1']:.4f}  Recall={metrics['Recall']:.4f}  "
                f"w_old={weights['w_old']:.3f}  w_new={weights['w_new']:.3f}"
            )
            for wi in winfo:
                weight_records.append({"config": label, **wi})

    return pd.DataFrame(results), pd.DataFrame(weight_records)


def main():
    logger = get_logger("AWE_Comparison", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_all, y_all = load_bankruptcy_with_year(logger)
    df, wdf = run_comparison(X_all, y_all, logger)

    out_cmp = OUTPUT_DIR / "bk_awe_comparison.csv"
    out_wgt = OUTPUT_DIR / "bk_awe_model_weights.csv"
    df.to_csv(out_cmp,  index=False, float_format="%.6f")
    wdf.to_csv(out_wgt, index=False, float_format="%.6f")

    print("\n" + "="*70)
    print("AWE vs DAWCE vs Equal-weight — Bankruptcy Test 2015-2018")
    print("="*70)
    cols = ["config", "method", "AUC", "F1", "Recall", "Precision", "w_old", "w_new"]
    print(df[cols].to_string(index=False))
    print(f"\nSaved: {out_cmp}")
    print(f"Saved: {out_wgt}")


if __name__ == "__main__":
    main()
