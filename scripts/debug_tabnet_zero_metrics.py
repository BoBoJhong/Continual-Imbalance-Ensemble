"""
Debug TabNet zero-metric cases on bankruptcy year splits.

主實驗見 experiments/phase1_baseline/bankruptcy_year_splits_tabnet.py（與 XGB 同協議：
逐年 validation、F1 閾值、Old/New/Retrain）。本腳本僅 **抽樣** 呼叫同一套
`_train_eval`，在少數 split 上檢查 F1/Recall/Precision/G_Mean 是否為 0，並寫出摘要 CSV。
"""
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from experiments._shared.common_bankruptcy import YEAR_SPLITS, get_bankruptcy_year_split
from experiments.phase1_baseline.bankruptcy_year_splits_tabnet import (
    SAMPLING_STRATEGIES,
    _train_eval,
)

OUTPUT_DIR = project_root / "results" / "phase1_baseline" / "tabnet"


def main():
    logger = get_logger("BK_YearSplits_TabNet_ZeroDebug", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    # 抽前 3 個 split 降低執行時間；要全跑可改為 YEAR_SPLITS
    for label, old_end_year in YEAR_SPLITS[:3]:
        logger.info(
            "\n%s\nSplit: %s  (Old<=%s, New=%s-2014, Test=2015-2018)\n%s",
            "=" * 60,
            label,
            old_end_year,
            old_end_year + 1,
            "=" * 60,
        )
        try:
            X_old, y_old, X_new, y_new, X_test, y_test, year_old, year_new, _yt = (
                get_bankruptcy_year_split(logger, old_end_year=old_end_year, return_years=True)
            )
        except Exception as exc:
            logger.error("[ERROR] %s: %s", label, exc)
            import traceback

            logger.error(traceback.format_exc())
            continue

        sampler = ImbalanceSampler()
        for tag, X_tr, y_tr, year_tr in (
            ("Old", X_old, y_old, year_old),
            ("New", X_new, y_new, year_new),
        ):
            for strat in SAMPLING_STRATEGIES:
                m = _train_eval(
                    X_tr,
                    y_tr,
                    X_test,
                    y_test,
                    sampler,
                    strat,
                    tag,
                    logger,
                    year_train=year_tr,
                )
                zero_flags = {
                    "F1": m["F1"] == 0.0,
                    "Recall": m["Recall"] == 0.0,
                    "Precision": m["Precision"] == 0.0,
                    "G_Mean": m["G_Mean"] == 0.0,
                }
                if any(zero_flags.values()):
                    rows.append(
                        {
                            "split": label,
                            "method": tag,
                            "sampling": strat,
                            **m,
                            **{f"zero_{k}": v for k, v in zero_flags.items()},
                        }
                    )

    if not rows:
        logger.info("No zero-metric cases in sampled splits (Old/New, first 3 splits).")
        return

    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "bankruptcy_year_splits_tabnet_zero_debug.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved -> %s (%d rows)", out_path.name, len(df))


if __name__ == "__main__":
    main()
