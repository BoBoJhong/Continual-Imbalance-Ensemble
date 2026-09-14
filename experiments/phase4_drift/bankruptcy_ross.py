"""
Phase 4 - Bankruptcy ROSS (Retrospective Optimal Split Selection)
=================================================================

Purpose
-------
Use the existing Phase 1 year-split results to identify a data-driven
concept drift boundary for the US bankruptcy dataset.

This script does not retrain models. It reads:

    results/phase1_baseline/xgb/bankruptcy_year_splits_xgb_raw.csv

and ranks all sliding Old/New boundaries by the performance of the New
model on the fixed future test period (2015-2018).

Interpretation
--------------
For each split:

    Old = 1999 ... old_end_year
    New = drift_start_year ... 2014
    Test = 2015 ... 2018

If the New-only model achieves the best future-test performance when
New starts at a specific year, that year is treated as the retrospective
drift boundary. This is not an online detector; it is a retrospective
selection method for studying where the most useful post-drift operating
window begins.

Outputs
-------
    results/phase4_drift/bk_ross_boundary_candidates.csv
    results/phase4_drift/bk_ross_selected_boundary.csv

Usage
-----
    python experiments/phase4_drift/bankruptcy_ross.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_logger


TRAIN_START_YEAR = 1999
TRAIN_END_YEAR = 2014
FIXED_DRIFT_START_YEAR = 2012

RAW_PATH = (
    project_root
    / "results"
    / "phase1_baseline"
    / "xgb"
    / "bankruptcy_year_splits_xgb_raw.csv"
)
OUTPUT_DIR = project_root / "results" / "phase4_drift"

METRICS = ["AUC", "F1", "G_Mean", "Recall", "Precision", "Type1_Error", "Type2_Error"]


def parse_split(split: str) -> tuple[int, int, int]:
    """Return (old_years, new_years, old_end_year) from split_9+7."""
    m = re.match(r"split_(\d+)\+(\d+)$", str(split))
    if not m:
        raise ValueError(f"Unexpected split label: {split}")
    old_years = int(m.group(1))
    new_years = int(m.group(2))
    old_end_year = TRAIN_START_YEAR + old_years - 1
    return old_years, new_years, old_end_year


def best_row(df: pd.DataFrame, method: str, split: str, criterion: str) -> pd.Series:
    sub = df[(df["method"] == method) & (df["split"] == split)].copy()
    if sub.empty:
        raise ValueError(f"No rows for method={method}, split={split}")
    return sub.sort_values([criterion, "AUC", "F1"], ascending=False).iloc[0]


def build_candidates(df: pd.DataFrame, criterion: str = "AUC") -> pd.DataFrame:
    """Build one candidate row per split using best New performance."""
    rows = []
    split_order = sorted(
        df["split"].unique(),
        key=lambda s: parse_split(s)[0],
    )

    for split in split_order:
        old_years, new_years, old_end = parse_split(split)
        drift_start = old_end + 1

        new_best = best_row(df, "New", split, criterion)
        old_best = best_row(df, "Old", split, criterion)

        row = {
            "split": split,
            "old_years": old_years,
            "new_years": new_years,
            "old_end_year": old_end,
            "drift_start_year": drift_start,
            "old_window": f"{TRAIN_START_YEAR}-{old_end}",
            "new_window": f"{drift_start}-{TRAIN_END_YEAR}",
            "best_new_sampling": new_best["sampling"],
            "best_old_sampling": old_best["sampling"],
        }

        for metric in METRICS:
            row[f"new_{metric}"] = float(new_best[metric])
            row[f"old_{metric}"] = float(old_best[metric])
            row[f"gap_new_minus_old_{metric}"] = float(new_best[metric] - old_best[metric])

        rows.append(row)

    candidates = pd.DataFrame(rows)
    candidates["rank_by_new_auc"] = candidates["new_AUC"].rank(ascending=False, method="min").astype(int)
    candidates["rank_by_new_f1"] = candidates["new_F1"].rank(ascending=False, method="min").astype(int)
    candidates["rank_by_gap_auc"] = (
        candidates["gap_new_minus_old_AUC"].rank(ascending=False, method="min").astype(int)
    )
    candidates["ross_rank"] = candidates[[ "rank_by_new_auc", "rank_by_new_f1" ]].mean(axis=1)
    return candidates.sort_values(["rank_by_new_auc", "rank_by_new_f1", "drift_start_year"])


def build_selected_summary(candidates: pd.DataFrame) -> pd.DataFrame:
    """Build a compact comparison: fixed boundary vs ROSS-selected boundary."""
    ross = candidates.sort_values(["rank_by_new_auc", "rank_by_new_f1"]).iloc[0]
    fixed = candidates[candidates["drift_start_year"] == FIXED_DRIFT_START_YEAR]
    fixed_row = fixed.iloc[0] if not fixed.empty else None

    rows = []
    if fixed_row is not None:
        rows.append({
            "method": "Fixed boundary",
            "boundary_source": "manual",
            "drift_start_year": int(fixed_row["drift_start_year"]),
            "old_window": fixed_row["old_window"],
            "new_window": fixed_row["new_window"],
            "best_new_sampling": fixed_row["best_new_sampling"],
            "new_AUC": fixed_row["new_AUC"],
            "new_F1": fixed_row["new_F1"],
            "new_Recall": fixed_row["new_Recall"],
            "new_Precision": fixed_row["new_Precision"],
            "gap_new_minus_old_AUC": fixed_row["gap_new_minus_old_AUC"],
        })

    rows.append({
        "method": "ROSS",
        "boundary_source": "best New AUC across sliding splits",
        "drift_start_year": int(ross["drift_start_year"]),
        "old_window": ross["old_window"],
        "new_window": ross["new_window"],
        "best_new_sampling": ross["best_new_sampling"],
        "new_AUC": ross["new_AUC"],
        "new_F1": ross["new_F1"],
        "new_Recall": ross["new_Recall"],
        "new_Precision": ross["new_Precision"],
        "gap_new_minus_old_AUC": ross["gap_new_minus_old_AUC"],
    })

    selected = pd.DataFrame(rows)
    if len(selected) == 2:
        selected["delta_vs_fixed_AUC"] = selected["new_AUC"] - selected.loc[0, "new_AUC"]
        selected["delta_vs_fixed_F1"] = selected["new_F1"] - selected.loc[0, "new_F1"]
    return selected


def main() -> None:
    if "--allow-test-oracle" not in sys.argv:
        raise SystemExit(
            "Test-derived ROSS is an oracle, not a deployable model selection rule. "
            "Use bankruptcy_ross_validation.py; legacy reproduction requires --allow-test-oracle."
        )
    logger = get_logger("Phase4_BK_ROSS", console=True, file=True)
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Phase 1 raw result not found: {RAW_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    candidates = build_candidates(df, criterion="AUC")
    selected = build_selected_summary(candidates)
    for frame in (candidates, selected):
        frame["selection_source"] = "test_oracle"
        frame["evidence_status"] = "exploratory_not_confirmatory"

    candidates_path = OUTPUT_DIR / "bk_ross_boundary_candidates.csv"
    selected_path = OUTPUT_DIR / "bk_ross_selected_boundary.csv"

    candidates.to_csv(candidates_path, index=False, float_format="%.6f")
    selected.to_csv(selected_path, index=False, float_format="%.6f")

    logger.info(f"Saved candidates -> {candidates_path}")
    logger.info(f"Saved selected boundary -> {selected_path}")

    logger.info("\nTop 5 ROSS candidates by New AUC:")
    top_cols = [
        "split",
        "drift_start_year",
        "old_window",
        "new_window",
        "best_new_sampling",
        "new_AUC",
        "new_F1",
        "gap_new_minus_old_AUC",
    ]
    logger.info("\n" + candidates[top_cols].head(5).to_string(index=False))

    logger.info("\nFixed vs ROSS:")
    logger.info("\n" + selected.to_string(index=False))


if __name__ == "__main__":
    main()
