"""
Phase 1 SVM Baseline 視覺化
=========================
讀取 results/phase1_baseline/svm/*_year_splits_svm_raw.csv，
產出折線圖與（若存在）bk_svm_compact_summary 熱力圖。

使用：
    python scripts/plots/visualize_phase1_svm_baseline.py
    python scripts/plots/visualize_phase1_svm_baseline.py --metrics AUC F1 Recall
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
        result_subdir="svm",
        raw_glob="*_year_splits_svm_raw.csv",
        raw_suffix="_year_splits_svm_raw",
        model_title="SVM Baseline",
        method_order=METHOD_ORDER_XGB,
        method_colors=METHOD_COLORS_XGB,
        compact_summary_name="bk_svm_compact_summary.csv",
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
