"""
Export Phase 3 (FS + ensemble) results to thesis-format CSV.

Currently Phase 3 thesis-format CSVs exist for Bankruptcy. This script adds the same
export for Medical (and can be reused for other datasets if needed).

Outputs:
  - results/phase3_feature/thesis/xgb_{suffix}_fs_static_thesis_format.csv
  - results/phase3_feature/thesis/xgb_{suffix}_fs_des_thesis_format.csv

Usage:
  python scripts/analysis/export_phase3_fs_thesis_format.py --dataset medical
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")


def parse_split_windows(split_id: str, *, train_start: int, train_end: int) -> tuple[str, str]:
    m = SPLIT_RE.match(str(split_id).strip())
    if not m:
        return "", ""

    old_years = int(m.group(1))
    old_start = int(train_start)
    old_end = int(train_start) + old_years - 1
    new_start = old_end + 1
    new_end = int(train_end)
    return f"{old_start}~{old_end}", f"{new_start}~{new_end}"


def pick_best_row(g: pd.DataFrame) -> pd.Series:
    sort_cols = ["AUC"]
    asc = [False]
    for extra in ("F1", "Recall"):
        if extra in g.columns:
            sort_cols.append(extra)
            asc.append(False)
    return g.sort_values(sort_cols, ascending=asc).iloc[0]


def _dataset_meta(dataset: str) -> dict:
    ds = str(dataset).strip().lower()
    if ds == "bankruptcy":
        return {
            "suffix": "bankruptcy",
            "dataset_name": "Bankruptcy_US",
            "train_start": 1999,
            "train_end": 2014,
            "test_window": "2015~2018",
        }
    if ds == "medical":
        return {
            "suffix": "medical",
            "dataset_name": "Medical_UCI",
            "train_start": 1999,
            "train_end": 2006,
            "test_window": "2007~2008",
        }
    raise SystemExit("--dataset must be one of: bankruptcy, medical")


def export_one(
    *,
    project_root: Path,
    meta: dict,
    src_csv: Path,
    out_csv: Path,
    method_label: str,
    train_scope: str,
) -> Path:
    df = pd.read_csv(src_csv)
    rows: list[dict] = []
    for split, g in df.groupby("split", sort=False):
        best = pick_best_row(g)
        old_window, new_window = parse_split_windows(
            str(split), train_start=meta["train_start"], train_end=meta["train_end"]
        )
        rows.append(
            {
                "dataset_name": meta["dataset_name"],
                "split_id": str(split),
                "old_window": old_window,
                "new_window": new_window,
                "test_window": meta["test_window"],
                "method": method_label,
                "train_scope": train_scope,
                "best_fs": str(best.get("fs", "")),
                "best_ensemble": str(best.get("ensemble", "")),
                "AUC": float(best["AUC"]),
                "F1": float(best["F1"]),
                "Recall": float(best["Recall"]),
            }
        )

    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, float_format="%.4f")
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Export Phase3 FS+ensemble results to thesis-format CSV")
    p.add_argument("--dataset", required=True, choices=["bankruptcy", "medical"])
    args = p.parse_args()

    meta = _dataset_meta(args.dataset)
    project_root = Path(__file__).resolve().parent.parent.parent

    suffix = meta["suffix"]
    src_static = project_root / "results" / "phase3_feature" / "mutual_info" / f"xgb_{suffix}_fs_full_static.csv"
    src_des = project_root / "results" / "phase3_feature" / "mutual_info" / f"xgb_{suffix}_fs_full_des.csv"

    out_dir = project_root / "results" / "phase3_feature" / "thesis"
    out_static = out_dir / f"xgb_{suffix}_fs_static_thesis_format.csv"
    out_des = out_dir / f"xgb_{suffix}_fs_des_thesis_format.csv"

    saved_static = export_one(
        project_root=project_root,
        meta=meta,
        src_csv=src_static,
        out_csv=out_static,
        method_label="FS_Static",
        train_scope="Old/New pools + FS + static soft voting",
    )
    saved_des = export_one(
        project_root=project_root,
        meta=meta,
        src_csv=src_des,
        out_csv=out_des,
        method_label="FS_DES",
        train_scope="Old/New pools + FS + dynamic selection (DES)",
    )

    rel = lambda p: str(p.relative_to(project_root)).replace("\\", "/")
    print("[SAVED]")
    print(" - " + rel(saved_static))
    print(" - " + rel(saved_des))


if __name__ == "__main__":
    main()

