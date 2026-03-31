"""
Phase 1 Random Forest Baseline 視覺化
====================================
讀取 results/phase1_baseline/random_forest/*_year_splits_rf_raw.csv。
預設不繪製 Finetune；需畫入時加 --include-finetune。

使用：
    python scripts/plots/visualize_phase1_random_forest_baseline.py
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
    METHOD_ORDER_NO_FINETUNE,
    METHOD_ORDER_XGB,
    run_phase1_baseline_visualization,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["AUC", "F1"],
        help="要畫折線圖的指標（須為 raw CSV 欄位）",
    )
    ap.add_argument(
        "--include-finetune",
        action="store_true",
        help="一併包含 Finetune（預設排除）",
    )
    args = ap.parse_args()

    method_order = METHOD_ORDER_XGB if args.include_finetune else METHOD_ORDER_NO_FINETUNE

    ok = run_phase1_baseline_visualization(
        project_root,
        list(args.metrics),
        result_subdir="random_forest",
        raw_glob="*_year_splits_rf_raw.csv",
        raw_suffix="_year_splits_rf_raw",
        model_title="Random Forest Baseline",
        method_order=method_order,
        method_colors=METHOD_COLORS_XGB,
        compact_summary_name="bk_rf_compact_summary.csv",
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
