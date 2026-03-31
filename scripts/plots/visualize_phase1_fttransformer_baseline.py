"""
Phase 1 FT-Transformer Baseline 視覺化
=====================================
讀取 results/phase1_baseline/fttransformer/*_year_splits_fttransformer_raw.csv，
產出折線圖與（若存在）bk_fttransformer_compact_summary 熱力圖。
輸出目錄：results/phase1_baseline/fttransformer/plots/（實驗腳本不會自動產圖，須手動執行本程式）。
須已安裝 matplotlib、seaborn（requirements.txt 已列；僅 requirements-core 時請另裝）。
預設不繪製 Finetune；需畫入時加 --include-finetune。

使用：
    python scripts/plots/visualize_phase1_fttransformer_baseline.py
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
        result_subdir="fttransformer",
        raw_glob="*_year_splits_fttransformer_raw.csv",
        raw_suffix="_year_splits_fttransformer_raw",
        model_title="FT-Transformer Baseline",
        method_order=method_order,
        method_colors=METHOD_COLORS_XGB,
        compact_summary_name="bk_fttransformer_compact_summary.csv",
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
