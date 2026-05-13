"""
Export Phase 2 XGB ensemble results to thesis-format CSV (Bankruptcy / Medical).

This script converts the existing Phase 2 raw CSVs into a compact per-split table
used in thesis writing (pick the best configuration per split).

Outputs (created if missing):
  - results/phase2_ensemble/static/xgb_oldnew_static_thesis_format_{suffix}.csv
  - results/phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_thesis_format_{suffix}.csv
  - results/phase2_ensemble/dynamic/dcs/xgb_oldnew_ensemble_dcs_thesis_format_{suffix}.csv

Usage:
  python scripts/analysis/export_phase2_thesis_format.py --dataset medical
  python scripts/analysis/export_phase2_thesis_format.py --dataset bankruptcy
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


def export_static(project_root: Path, meta: dict) -> Path:
    suffix = meta["suffix"]
    raw_path = (
        project_root
        / "results"
        / "phase2_ensemble"
        / "static"
        / f"xgb_oldnew_ensemble_static_by_sampling_raw_{suffix}.csv"
    )
    out_path = (
        project_root
        / "results"
        / "phase2_ensemble"
        / "static"
        / f"xgb_oldnew_static_thesis_format_{suffix}.csv"
    )

    df = pd.read_csv(raw_path)
    rows: list[dict] = []

    for split, g in df.groupby("split", sort=False):
        # Bankruptcy raw contains k-subset candidates (2/3/4-model subsets).
        # Medical raw only contains the three pools (Old/New/Retrain) per sampling.
        if suffix == "bankruptcy":
            cand = g[(g["type"] == "k_subset") & (g["ensemble"].astype(str).str.endswith("models"))].copy()
            if cand.empty:
                continue
            best = pick_best_row(cand)
            best_size = str(best.get("ensemble", ""))
            best_subset = str(best.get("subset", ""))
        else:
            cand = g[g["ensemble"].isin(["Old", "New", "Retrain"])].copy()
            if cand.empty:
                continue
            best = pick_best_row(cand)
            sampling_col = str(best.get("sampling_col", best.get("type", "")))
            ens = str(best.get("ensemble", ""))
            best_size = "1model"
            best_subset = f"{ens}_{sampling_col}".strip("_")

        old_window, new_window = parse_split_windows(str(split), train_start=meta["train_start"], train_end=meta["train_end"])
        rows.append(
            {
                "dataset_name": meta["dataset_name"],
                "split_id": str(split),
                "old_window": old_window,
                "new_window": new_window,
                "test_window": meta["test_window"],
                "method": "StaticEnsemble",
                "train_scope": "Old/New pools + static soft voting",
                "best_ensemble_size": best_size,
                "best_subset": best_subset,
                "AUC": float(best["AUC"]),
                "F1": float(best["F1"]),
                "Recall": float(best["Recall"]),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, float_format="%.4f")
    return out_path


def export_des(project_root: Path, meta: dict) -> Path:
    suffix = meta["suffix"]
    raw_path = (
        project_root
        / "results"
        / "phase2_ensemble"
        / "dynamic"
        / "des"
        / f"xgb_oldnew_ensemble_des_by_sampling_raw_{suffix}.csv"
    )
    out_path = (
        project_root
        / "results"
        / "phase2_ensemble"
        / "dynamic"
        / "des"
        / f"xgb_oldnew_ensemble_des_thesis_format_{suffix}.csv"
    )

    df = pd.read_csv(raw_path)
    rows: list[dict] = []
    for split, g in df.groupby("split", sort=False):
        best = pick_best_row(g)
        old_window, new_window = parse_split_windows(str(split), train_start=meta["train_start"], train_end=meta["train_end"])
        rows.append(
            {
                "dataset_name": meta["dataset_name"],
                "split_id": str(split),
                "old_window": old_window,
                "new_window": new_window,
                "test_window": meta["test_window"],
                "method": "DynamicDES",
                "train_scope": "Old/New pools + dynamic selection (DES)",
                "best_ensemble_size": str(best.get("type", "")),
                "best_subset": str(best.get("ensemble", "")),
                "AUC": float(best["AUC"]),
                "F1": float(best["F1"]),
                "Recall": float(best["Recall"]),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False, float_format="%.4f")
    return out_path


def export_dcs(project_root: Path, meta: dict) -> Path:
    suffix = meta["suffix"]
    raw_path = (
        project_root
        / "results"
        / "phase2_ensemble"
        / "dynamic"
        / "dcs"
        / f"xgb_oldnew_ensemble_dcs_by_sampling_raw_{suffix}.csv"
    )
    out_path = (
        project_root
        / "results"
        / "phase2_ensemble"
        / "dynamic"
        / "dcs"
        / f"xgb_oldnew_ensemble_dcs_thesis_format_{suffix}.csv"
    )

    df = pd.read_csv(raw_path)
    rows: list[dict] = []
    for split, g in df.groupby("split", sort=False):
        best = pick_best_row(g)
        old_window, new_window = parse_split_windows(str(split), train_start=meta["train_start"], train_end=meta["train_end"])
        rows.append(
            {
                "dataset_name": meta["dataset_name"],
                "split_id": str(split),
                "old_window": old_window,
                "new_window": new_window,
                "test_window": meta["test_window"],
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
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Export Phase2 ensemble results to thesis-format CSV")
    p.add_argument("--dataset", required=True, choices=["bankruptcy", "medical"])
    args = p.parse_args()

    meta = _dataset_meta(args.dataset)
    project_root = Path(__file__).resolve().parent.parent.parent

    out_static = export_static(project_root, meta)
    out_des = export_des(project_root, meta)
    out_dcs = export_dcs(project_root, meta)

    rel = lambda p: str(p.relative_to(project_root)).replace("\\", "/")
    print("[SAVED]")
    print(" - " + rel(out_static))
    print(" - " + rel(out_des))
    print(" - " + rel(out_dcs))


if __name__ == "__main__":
    main()

