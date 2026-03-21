"""
Phase 1 XGB Baseline 視覺化
==========================
讀取 results/phase1_baseline/xgb/ 內 *year_splits_xgb_raw.csv，
產出折線圖與（若存在）bk_xgb_compact_summary 熱力圖。

使用：
    python scripts/plots/visualize_phase1_xgb_baseline.py
    python scripts/plots/visualize_phase1_xgb_baseline.py --metrics AUC F1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
project_root = _scripts_dir.parent.parent
sys.path.insert(0, str(_scripts_dir))

from phase1_baseline_plotting import (
    METHOD_COLORS_XGB,
    METHOD_ORDER_XGB,
    dataset_label_from_raw_stem,
    plot_compact_heatmaps,
    plot_year_split_lines,
)
import pandas as pd

XGB_DIR = project_root / "results" / "phase1_baseline" / "xgb"
PLOTS_DIR = XGB_DIR / "plots"
RAW_SUFFIX = "_year_splits_xgb_raw"
RAW_GLOB = f"*{RAW_SUFFIX}.csv"


def discover_raw_files() -> list[Path]:
    if not XGB_DIR.is_dir():
        return []
    return sorted(XGB_DIR.glob(RAW_GLOB))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["AUC", "F1"],
        help="要畫折線圖的指標（須為 raw CSV 欄位）",
    )
    args = ap.parse_args()

    raw_files = discover_raw_files()
    if not raw_files:
        print(f"找不到 raw CSV：{XGB_DIR}")
        sys.exit(1)

    print("Phase 1 XGB baseline 圖表輸出…")
    for path in raw_files:
        df = pd.read_csv(path)
        need = {"split", "method", "sampling", *args.metrics}
        missing = need - set(df.columns)
        if missing:
            print(f"  [SKIP] {path.name} 缺少欄位: {missing}")
            continue
        label = dataset_label_from_raw_stem(path.stem, RAW_SUFFIX)
        out = PLOTS_DIR / f"{path.stem.replace(RAW_SUFFIX, '')}_year_splits_{'_'.join(args.metrics)}.png"
        plot_year_split_lines(
            df,
            label,
            list(args.metrics),
            out,
            model_title="XGB Baseline",
            method_order=METHOD_ORDER_XGB,
            method_colors=METHOD_COLORS_XGB,
        )
        print(f"  [SAVED] {out.relative_to(project_root)}")

    summary = XGB_DIR / "bk_xgb_compact_summary.csv"
    if summary.exists():
        print("\nCompact summary 熱力圖…")
        plot_compact_heatmaps(
            summary,
            PLOTS_DIR / "bk_xgb",
            model_title="XGB Baseline",
            method_order=METHOD_ORDER_XGB,
            project_root=project_root,
        )

    print(f"\n完成。圖表目錄: {PLOTS_DIR.relative_to(project_root)}")


if __name__ == "__main__":
    main()
