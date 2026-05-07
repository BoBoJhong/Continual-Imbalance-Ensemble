"""
Archive Phase3 FS results into method-specific folders.

Folders:
  results/phase3_feature/archive/mi
  results/phase3_feature/archive/cart
  results/phase3_feature/archive/ga
"""
from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULT_DIR = PROJECT_ROOT / "results" / "phase3_feature"
ARCHIVE_ROOT = RESULT_DIR / "archive"

METHOD_TAGS = {
    "mi": "mi_r80",
    "cart": "cart_r80",
    "ga": "ga_r80",
}

RESULT_FILES = [
    "xgb_bankruptcy_fs_full_static.csv",
    "xgb_bankruptcy_fs_full_des.csv",
    "xgb_bankruptcy_fs_full_dcs.csv",
    "xgb_stock_fs_full_static.csv",
    "xgb_stock_fs_full_des.csv",
    "xgb_medical_fs_full_static.csv",
    "xgb_medical_fs_full_des.csv",
]


def _best_rows(df: pd.DataFrame, fs_tag: str) -> pd.DataFrame:
    if df.empty or "fs" not in df.columns:
        return pd.DataFrame()
    sub = df[df["fs"] == fs_tag].copy()
    if sub.empty:
        return pd.DataFrame()
    if {"split", "AUC"}.issubset(sub.columns):
        idx = sub.groupby("split")["AUC"].idxmax()
        return sub.loc[idx].sort_values("split").reset_index(drop=True)
    if "AUC" in sub.columns:
        return sub.sort_values("AUC", ascending=False).head(1).reset_index(drop=True)
    return sub.reset_index(drop=True)


def main() -> None:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

    for method_name, fs_tag in METHOD_TAGS.items():
        method_dir = ARCHIVE_ROOT / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        for filename in RESULT_FILES:
            src = RESULT_DIR / filename
            if not src.exists():
                continue

            dst = method_dir / filename
            shutil.copy2(src, dst)

            df = pd.read_csv(src)
            best = _best_rows(df, fs_tag)
            if not best.empty:
                best.insert(0, "source_file", filename)
                summary_rows.append(best)

        if summary_rows:
            pd.concat(summary_rows, ignore_index=True).to_csv(
                method_dir / f"{method_name}_best_summary.csv",
                index=False,
            )

    print(f"Archived FS results to: {ARCHIVE_ROOT}")


if __name__ == "__main__":
    main()
