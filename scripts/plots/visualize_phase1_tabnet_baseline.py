"""
Phase 1 TabNet Baseline 視覺化
============================
讀取 results/phase1_baseline/tabnet/*_year_splits_tabnet_raw.csv（三策略：Old / Old+New / New）。

使用：
    python scripts/plots/visualize_phase1_tabnet_baseline.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
project_root = _scripts_dir.parent.parent
sys.path.insert(0, str(_scripts_dir))

from phase1_baseline_plotting import (
    METHOD_COLORS_LEGACY_THREE,
    METHOD_ORDER_LEGACY_THREE,
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
    args = ap.parse_args()

    ok = run_phase1_baseline_visualization(
        project_root,
        list(args.metrics),
        result_subdir="tabnet",
        raw_glob="*_year_splits_tabnet_raw.csv",
        raw_suffix="_year_splits_tabnet_raw",
        model_title="TabNet Baseline",
        method_order=METHOD_ORDER_LEGACY_THREE,
        method_colors=METHOD_COLORS_LEGACY_THREE,
        compact_summary_name=None,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
