"""
Phase 1 Torch MLP Baseline 視覺化
================================
讀取 results/phase1_baseline/torch_mlp/ 內 *year_splits_torch_mlp_raw.csv，
產出折線圖與（若存在）bk_torch_mlp_compact_summary 熱力圖。

與 XGB 相同三策略：Old / New / Retrain（見 bankruptcy_year_splits_torch_mlp.py）。
若 raw 仍出現 Old+New，代表為舊版三策略輸出，請重新跑實驗腳本（15 折、Retrain 僅一次時 raw 為 124 rows）。

使用：
    python scripts/plots/visualize_phase1_torch_mlp_baseline.py
    python scripts/plots/visualize_phase1_torch_mlp_baseline.py --metrics AUC F1 Recall
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_scripts_dir = Path(__file__).resolve().parent
project_root = _scripts_dir.parent.parent
sys.path.insert(0, str(_scripts_dir))

from phase1_baseline_plotting import (
    METHOD_COLORS_MLP,
    METHOD_ORDER_MLP,
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
        result_subdir="torch_mlp",
        raw_glob="*_year_splits_torch_mlp_raw.csv",
        raw_suffix="_year_splits_torch_mlp_raw",
        model_title="Torch MLP Baseline",
        method_order=METHOD_ORDER_MLP,
        method_colors=METHOD_COLORS_MLP,
        compact_summary_name="bk_torch_mlp_compact_summary.csv",
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
