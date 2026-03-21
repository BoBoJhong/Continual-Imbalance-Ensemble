"""
從既有的 by_sampling raw CSV（欄位 type 或 sampling_col）重產精簡寬表，無需重訓練。

範例（raw 與輸出目錄需對應 static／des）：
  python experiments/phase2_ensemble/static/regenerate_phase2_ensemble_summary_from_raw.py \\
    results/phase2_ensemble/static/xgb_oldnew_ensemble_static_by_sampling_raw_bankruptcy.csv bankruptcy
  python experiments/phase2_ensemble/static/regenerate_phase2_ensemble_summary_from_raw.py \\
    results/phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_by_sampling_raw_bankruptcy.csv bankruptcy \\
    --out-dir results/phase2_ensemble/dynamic/des
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.phase2_ensemble.xgb_oldnew_ensemble_common import build_summary_wide


def main() -> None:
    p = argparse.ArgumentParser(description="由 raw CSV 重產 xgb_oldnew_ensemble_*_summary_wide_*.csv")
    p.add_argument(
        "raw_csv",
        type=Path,
        help="例如 static 的 xgb_oldnew_ensemble_static_by_sampling_raw_*.csv 或 des 的 …_des_by_sampling_raw_*.csv",
    )
    p.add_argument(
        "suffix",
        type=str,
        help="檔名後綴，與實驗腳本一致：bankruptcy / medical / stock_spx",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="預設與 raw_csv 同目錄",
    )
    args = p.parse_args()
    out_dir = args.out_dir or args.raw_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.raw_csv)
    for metric in ("AUC", "F1", "Recall"):
        if metric not in df.columns:
            continue
        wide = build_summary_wide(df, metric)
        path = out_dir / f"xgb_oldnew_ensemble_{metric}_summary_wide_{args.suffix}.csv"
        wide.to_csv(path, index=False, float_format="%.4f")
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
