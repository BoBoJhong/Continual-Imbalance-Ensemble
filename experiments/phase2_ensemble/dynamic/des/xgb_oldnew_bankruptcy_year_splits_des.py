"""
Phase 2 — XGB Bankruptcy：動態 DES（KNORA / DES-KNN），年份切割。
輸出：results/phase2_ensemble/dynamic/des/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger

from experiments.phase2_ensemble.xgb_year_split_shared import iter_bankruptcy_year_splits
from experiments.phase2_ensemble.xgb_oldnew_ensemble_common import (
    export_ensemble_long_tables_and_raw,
    expected_summary_wide_columns_des_only,
)


def main():
    parser = argparse.ArgumentParser(description="Phase2 XGB bankruptcy dynamic DES year splits")
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="額外輸出子資料夾名稱（例如 tuned_rerun），結果會寫到 dynamic/des/<tag>/",
    )
    args = parser.parse_args()

    logger = get_logger("XGB_Bankruptcy_DES_YearSplits", console=True, file=True)
    set_seed(42)

    out = project_root / "results" / "phase2_ensemble" / "dynamic" / "des"
    tag = args.output_tag.strip()
    if tag:
        out = out / tag
    out.mkdir(parents=True, exist_ok=True)

    _, des_rows = iter_bankruptcy_year_splits(logger)
    df = pd.DataFrame(des_rows)

    export_ensemble_long_tables_and_raw(
        df,
        out,
        metric_cols=["AUC", "F1", "Recall"],
        table_filename_fmt="xgb_oldnew_ensemble_des_{metric}_table_bankruptcy.csv",
        raw_csv_name="xgb_oldnew_ensemble_des_by_sampling_raw_bankruptcy.csv",
        logger=logger,
        summary_wide_suffix="bankruptcy",
        summary_wide_filename_fmt="xgb_oldnew_ensemble_des_{metric}_summary_wide_{suffix}.csv",
        summary_wide_columns=expected_summary_wide_columns_des_only(),
    )
    logger.info("\n完成。results/phase2_ensemble/dynamic/des/（動態 DES xgb_oldnew_ensemble_des_*）")


if __name__ == "__main__":
    main()
