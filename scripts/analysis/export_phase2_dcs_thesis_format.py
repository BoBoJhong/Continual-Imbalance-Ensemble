"""
Export Phase 2 bankruptcy DCS results to thesis-format CSV.

Usage:
  python scripts/analysis/export_phase2_dcs_thesis_format.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")
TRAIN_START_YEAR = 1999
TRAIN_END_YEAR = 2014
TEST_WINDOW = "2015~2018"


def parse_split_windows(split_id: str) -> tuple[str, str]:
    m = SPLIT_RE.match(str(split_id).strip())
    if not m:
        return "", ""

    old_years = int(m.group(1))
    old_start = TRAIN_START_YEAR
    old_end = TRAIN_START_YEAR + old_years - 1
    new_start = old_end + 1
    new_end = TRAIN_END_YEAR

    old_window = f"{old_start}~{old_end}"
    new_window = f"{new_start}~{new_end}"
    return old_window, new_window


def pick_best_row(g: pd.DataFrame) -> pd.Series:
    sort_cols = ["AUC"]
    asc = [False]
    for extra in ("F1", "Recall"):
        if extra in g.columns:
            sort_cols.append(extra)
            asc.append(False)
    return g.sort_values(sort_cols, ascending=asc).iloc[0]


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_path = (
        project_root
        / "results"
        / "phase2_ensemble"
        / "dynamic"
        / "dcs"
        / "xgb_oldnew_ensemble_dcs_by_sampling_raw_bankruptcy.csv"
    )
    out_path = (
        project_root
        / "results"
        / "phase2_ensemble"
        / "dynamic"
        / "dcs"
        / "xgb_oldnew_ensemble_dcs_thesis_format_bankruptcy.csv"
    )

    df = pd.read_csv(raw_path)
    rows: list[dict] = []
    for split, g in df.groupby("split", sort=False):
        best = pick_best_row(g)
        old_window, new_window = parse_split_windows(str(split))
        rows.append(
            {
                "dataset_name": "Bankruptcy_US",
                "split_id": str(split),
                "old_window": old_window,
                "new_window": new_window,
                "test_window": TEST_WINDOW,
                "method": "DynamicDCS",
                "train_scope": "Old/New pools + dynamic classifier selection",
                "best_ensemble_size": str(best.get("sampling_col", "")),
                "best_subset": str(best.get("ensemble", "")),
                "AUC": float(best["AUC"]),
                "F1": float(best["F1"]),
                "Recall": float(best["Recall"]),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"[SAVED] {out_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
