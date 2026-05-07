"""
Archive bankruptcy FS triad results:
- Filter   : MI
- Embedded : SHAP
- Wrapper  : RFE
"""
from __future__ import annotations

from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULT_DIR = PROJECT_ROOT / "results" / "phase3_feature"
ARCHIVE_DIR = RESULT_DIR / "archive" / "bankruptcy_fs_triad"

FILES = [
    "xgb_bankruptcy_fs_full_static.csv",
    "xgb_bankruptcy_fs_full_des.csv",
    "xgb_bankruptcy_fs_full_dcs.csv",
]

# 新版腳本產物為 mi_r80；舊版 CSV 可能為 mutual_info_r80，歸檔時兩者擇一
METHOD_TAGS = {
    "mi": ("mi_r80", "mutual_info_r80"),
    "shap": ("shap_r80",),
    "rfe": ("rfe_r80",),
}


def _pick_best(df: pd.DataFrame, fs_tags: tuple[str, ...]) -> pd.DataFrame:
    if "fs" not in df.columns or "AUC" not in df.columns:
        return pd.DataFrame()
    sub = pd.DataFrame()
    for tag in fs_tags:
        sub = df[df["fs"] == tag].copy()
        if not sub.empty:
            break
    if sub.empty:
        return pd.DataFrame()
    if "split" in sub.columns:
        idx = sub.groupby("split")["AUC"].idxmax()
        return sub.loc[idx].sort_values("split").reset_index(drop=True)
    return sub.sort_values("AUC", ascending=False).head(1).reset_index(drop=True)


def main() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for method_name, fs_tags in METHOD_TAGS.items():
        method_dir = ARCHIVE_DIR / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        best_parts = []
        for name in FILES:
            src = RESULT_DIR / name
            if not src.exists():
                continue
            dst = method_dir / name
            shutil.copy2(src, dst)
            df = pd.read_csv(src)
            best = _pick_best(df, fs_tags)
            if not best.empty:
                best.insert(0, "source_file", name)
                best_parts.append(best)
        if best_parts:
            pd.concat(best_parts, ignore_index=True).to_csv(
                method_dir / f"{method_name}_best_summary.csv",
                index=False,
            )
    print(f"Archived to {ARCHIVE_DIR}")


if __name__ == "__main__":
    main()
