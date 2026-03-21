"""Phase 2 — XGB Stock(spx)：動態 DES。輸出：results/phase2_ensemble/dynamic/des/"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger

from experiments.phase2_ensemble.xgb_year_split_shared import iter_stock_year_splits
from experiments.phase2_ensemble.xgb_oldnew_ensemble_common import (
    export_ensemble_long_tables_and_raw,
    expected_summary_wide_columns_des_only,
)


def main():
    ticker = "spx"
    logger = get_logger("XGB_Stock_DES_YearSplits", console=True, file=True)
    set_seed(42)

    out = project_root / "results" / "phase2_ensemble" / "dynamic" / "des"
    out.mkdir(parents=True, exist_ok=True)

    _, des_rows = iter_stock_year_splits(logger, ticker=ticker)
    df = pd.DataFrame(des_rows)

    suf = f"stock_{ticker}"
    export_ensemble_long_tables_and_raw(
        df,
        out,
        metric_cols=["AUC", "F1", "Recall"],
        table_filename_fmt=f"xgb_oldnew_ensemble_des_{{metric}}_table_{suf}.csv",
        raw_csv_name=f"xgb_oldnew_ensemble_des_by_sampling_raw_{suf}.csv",
        logger=logger,
        summary_wide_suffix=suf,
        summary_wide_filename_fmt=f"xgb_oldnew_ensemble_des_{{metric}}_summary_wide_{{suffix}}.csv",
        summary_wide_columns=expected_summary_wide_columns_des_only(),
    )
    logger.info(f"\n完成。dynamic/des/ xgb_oldnew_ensemble_des_*_{suf}.csv")


if __name__ == "__main__":
    main()
