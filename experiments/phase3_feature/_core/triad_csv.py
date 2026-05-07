"""Split bankruptcy fs_full (static / DES / DCS) rows into mutual_info / shap / rfe result dirs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from experiments.phase3_feature._core.paths import results_method_dir, results_phase3

# Triad tags aligned with FS_CONFIGS in fs_full / fs_dcs
TAG_TO_METHOD = {
    "mi_r80": "mutual_info",
    "shap_r80": "shap",
    "rfe_r80": "rfe",
}


def save_bankruptcy_fs_full_split(
    project_root: Path,
    static_rows: Iterable[dict],
    des_rows: Iterable[dict],
    *,
    static_name: str = "xgb_bankruptcy_fs_full_static.csv",
    des_name: str = "xgb_bankruptcy_fs_full_des.csv",
) -> None:
    static_df = pd.DataFrame(static_rows)
    des_df = pd.DataFrame(des_rows)
    for tag, method in TAG_TO_METHOD.items():
        out = results_method_dir(project_root, method)
        fs_keep = {"no_fs", tag}
        static_df[static_df["fs"].isin(fs_keep)].to_csv(out / static_name, index=False)
        des_df[des_df["fs"].isin(fs_keep)].to_csv(out / des_name, index=False)


def save_bankruptcy_fs_dcs_split(
    project_root: Path,
    rows: Iterable[dict],
    *,
    csv_name: str = "xgb_bankruptcy_fs_full_dcs.csv",
) -> None:
    df = pd.DataFrame(rows)
    for tag, method in TAG_TO_METHOD.items():
        out = results_method_dir(project_root, method)
        df[df["fs"].isin({"no_fs", tag})].to_csv(out / csv_name, index=False, float_format="%.6f")


def save_medical_fs_full_split(
    project_root: Path,
    static_rows: Iterable[dict],
    des_rows: Iterable[dict],
    *,
    static_name: str = "xgb_medical_fs_full_static.csv",
    des_name: str = "xgb_medical_fs_full_des.csv",
) -> None:
    static_df = pd.DataFrame(static_rows)
    des_df = pd.DataFrame(des_rows)
    for tag, method in TAG_TO_METHOD.items():
        out = results_method_dir(project_root, method)
        fs_keep = {"no_fs", tag}
        static_df[static_df["fs"].isin(fs_keep)].to_csv(out / static_name, index=False)
        des_df[des_df["fs"].isin(fs_keep)].to_csv(out / des_name, index=False)


def save_medical_fs_dcs_split(
    project_root: Path,
    rows: Iterable[dict],
    *,
    csv_name: str = "xgb_medical_fs_full_dcs.csv",
) -> None:
    df = pd.DataFrame(rows)
    for tag, method in TAG_TO_METHOD.items():
        out = results_method_dir(project_root, method)
        df[df["fs"].isin({"no_fs", tag})].to_csv(out / csv_name, index=False, float_format="%.6f")


def write_bankruptcy_fs_full_combined_fallback(project_root: Path, static_df: pd.DataFrame, des_df: pd.DataFrame) -> None:
    """Optional legacy single-file copy at results/phase3_feature/ (for quick grep)."""
    root = results_phase3(project_root)
    root.mkdir(parents=True, exist_ok=True)
    static_df.to_csv(root / "xgb_bankruptcy_fs_full_static.csv", index=False)
    des_df.to_csv(root / "xgb_bankruptcy_fs_full_des.csv", index=False)
