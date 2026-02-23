"""
實驗 7: Stock 資料集 - DES (KNORA-E)
"""
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.evaluation import compute_metrics, print_results_table, results_to_dataframe
from experiments.common_dataset import get_splits
from experiments.common_des import run_des

SPLIT_MODE = "block_cv"


def run_experiment(dataset_name, logger):
    logger.info(f"\n[{dataset_name.upper()}] 開始執行")
    X_hist, y_hist, X_new, y_new, X_test, y_test = get_splits(dataset_name, logger, split_mode=SPLIT_MODE)

    logger.info("  - 執行 DES (KNORA-E)")
    res = run_des(X_hist, y_hist, X_new, y_new, X_test, y_test, logger, k=7)
    
    results = {"DES_KNORAE": res}
    return results_to_dataframe(results)


def main():
    logger = get_logger("Stock_DES", console=True, file=True)
    set_seed(42)
    logger.info("=" * 80)
    logger.info("實驗 7: Stock DES (KNORA-E) - SPX, DJI, NDX")
    logger.info("=" * 80)

    indices = ["stock_spx", "stock_dji", "stock_ndx"]
    all_results = {}

    for ds in indices:
        try:
            res_df = run_experiment(ds, logger)
            out_dir = project_root / "results/des"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"{ds}_des_results.csv"
            res_df.to_csv(out_csv)
            logger.info(f"  -> {ds.upper()} 結果已保存: {out_csv}")
            all_results[ds] = res_df
        except Exception as e:
            logger.error(f"  [ERROR] {ds} 執行失敗: {e}")

    if all_results:
        avg_df = sum(all_results.values()) / len(all_results)
        out_csv = project_root / "results/des/stock_des_results.csv"
        avg_df.to_csv(out_csv)
        logger.info(f"\n三大指數平均 DES 結果已保存: {out_csv}")
        print_results_table(avg_df.T.to_dict(), title="Stock DES Average Results")

    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    main()
    print("\n實驗 7 完成！")
