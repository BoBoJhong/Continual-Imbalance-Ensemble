"""
將 results/phase3_feature 內既有 CSV 依三法目錄分類。
- mutual_info / shap / rfe：依 triad 與欄位 fs 分流
- thesis/：論文彙整用寬表
- combined/：保留「全方法合併」大表（方便舊腳本與全文 grep）
執行：專案根目錄下
  python scripts/reports/migrate_phase3_results_to_method_dirs.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "results" / "phase3_feature"
MI = P3 / "mutual_info"
SH = P3 / "shap"
RF = P3 / "rfe"
TH = P3 / "thesis"
CB = P3 / "combined"


def _ensure() -> None:
    for d in (MI, SH, RF, TH, CB):
        d.mkdir(parents=True, exist_ok=True)


def _split_full_static_des() -> None:
    for name in ("xgb_bankruptcy_fs_full_static.csv", "xgb_bankruptcy_fs_full_des.csv"):
        p = P3 / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "fs" not in df.columns:
            continue
        mapping = {
            "mi_r80": MI,
            "shap_r80": SH,
            "rfe_r80": RF,
        }
        for tag, out in mapping.items():
            sub = df[df["fs"].isin({"no_fs", tag})]
            sub.to_csv(out / name, index=False)


def _split_dcs() -> None:
    p = P3 / "xgb_bankruptcy_fs_full_dcs.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    if "fs" not in df.columns:
        return

    def pick_mi(fs: str) -> bool:
        return fs in {"no_fs", "mi_r80", "mutual_info_r50", "mutual_info_r80"}

    def pick_sh(fs: str) -> bool:
        return fs in {"no_fs", "shap_r50", "shap_r80"}

    def pick_rf(fs: str) -> bool:
        return fs in {"no_fs", "rfe_r80"}

    df[df["fs"].apply(pick_mi)].to_csv(MI / "xgb_bankruptcy_fs_full_dcs.csv", index=False, float_format="%.6f")
    df[df["fs"].apply(pick_sh)].to_csv(SH / "xgb_bankruptcy_fs_full_dcs.csv", index=False, float_format="%.6f")
    df[df["fs"].apply(pick_rf)].to_csv(RF / "xgb_bankruptcy_fs_full_dcs.csv", index=False, float_format="%.6f")


def _split_advanced() -> None:
    raw_p = P3 / "xgb_bankruptcy_fs_advanced_raw.csv"
    if raw_p.exists():
        df = pd.read_csv(raw_p)
        if "fs" in df.columns:

            def mi_fs(fs: str) -> bool:
                s = str(fs)
                return s == "no_fs" or s.startswith("mutual_info") or s == "mi_r80"

            def sh_fs(fs: str) -> bool:
                s = str(fs)
                return s == "no_fs" or s.startswith("shap")

            def rf_fs(fs: str) -> bool:
                s = str(fs)
                return s == "no_fs" or s.startswith("rfe") or s == "rfe_r80"

            df[df["fs"].apply(mi_fs)].to_csv(MI / "xgb_bankruptcy_fs_advanced_raw.csv", index=False)
            df[df["fs"].apply(sh_fs)].to_csv(SH / "xgb_bankruptcy_fs_advanced_raw.csv", index=False)
            df[df["fs"].apply(rf_fs)].to_csv(RF / "xgb_bankruptcy_fs_advanced_raw.csv", index=False)

    for fname in ("xgb_bankruptcy_fs_advanced_summary.csv", "xgb_bankruptcy_fs_advanced_auc_diff.csv"):
        p = P3 / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "fs" not in df.columns:
            continue
        df[df["fs"].astype(str).apply(lambda x: x == "no_fs" or x.startswith("mutual_info") or x == "mi_r80")].to_csv(
            MI / fname, index=False
        )
        df[df["fs"].astype(str).apply(lambda x: x == "no_fs" or x.startswith("shap"))].to_csv(SH / fname, index=False)
        df[df["fs"].astype(str).apply(lambda x: x == "no_fs" or x.startswith("rfe"))].to_csv(RF / fname, index=False)


def _move_whole(src_rel: str, dest_dir: Path) -> None:
    src = P3 / src_rel
    if not src.exists():
        return
    dest = dest_dir / Path(src_rel).name
    shutil.move(str(src), str(dest))


def _move_to_combined(src_rel: str) -> None:
    src = P3 / src_rel
    if not src.exists():
        return
    dest = CB / Path(src_rel).name
    shutil.move(str(src), str(dest))


def main() -> None:
    _ensure()

    _split_full_static_des()
    _split_dcs()
    _split_advanced()

    # 合併版大表移至 combined/（子目錄已各有分流切片）
    for rel in (
        "xgb_bankruptcy_fs_full_static.csv",
        "xgb_bankruptcy_fs_full_des.csv",
        "xgb_bankruptcy_fs_full_dcs.csv",
        "xgb_bankruptcy_fs_advanced_raw.csv",
        "xgb_bankruptcy_fs_advanced_summary.csv",
        "xgb_bankruptcy_fs_advanced_auc_diff.csv",
    ):
        _move_to_combined(rel)

    # 線性 FS 掃描（kbest 等）→ mutual_info
    for rel in (
        "xgb_bankruptcy_fs_raw.csv",
        "xgb_bankruptcy_fs_summary.csv",
        "xgb_bankruptcy_fs_auc_diff.csv",
    ):
        _move_whole(rel, MI)

    # 醫療／股票 triad 類 full → mutual_info（含 mi/cart/ga）
    for rel in (
        "xgb_medical_fs_full_static.csv",
        "xgb_medical_fs_full_des.csv",
        "xgb_medical_fs_summary_diff.csv",
        "xgb_stock_fs_full_static.csv",
        "xgb_stock_fs_full_des.csv",
    ):
        _move_whole(rel, MI)

    # RFE 專表 → rfe
    for rel in ("xgb_bankruptcy_rfe_results.csv", "xgb_medical_rfe_results.csv"):
        _move_whole(rel, RF)

    # 論文用寬表 → thesis
    for rel in (
        "xgb_bankruptcy_p3_thesis_table.csv",
        "xgb_bankruptcy_fs_static_thesis_format.csv",
        "xgb_bankruptcy_fs_des_thesis_format.csv",
    ):
        _move_whole(rel, TH)

    print("Done.")
    print(f"  {MI}")
    print(f"  {SH}")
    print(f"  {RF}")
    print(f"  {TH}")
    print(f"  {CB}")


if __name__ == "__main__":
    main()
