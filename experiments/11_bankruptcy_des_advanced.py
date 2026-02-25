"""
實驗 9: 進階 DES 比較（後續論文用）
比較：baseline DES、時間加權 DES、少數類加權 DES、combined（時間+少數類）
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from experiments.common_bankruptcy import get_bankruptcy_splits
from experiments.common_des import run_des
from experiments.common_des_advanced import run_des_advanced

SPLIT_MODE = "block_cv"


def main():
    logger = get_logger("Bankruptcy_DES_Advanced", console=True, file=True)
    set_seed(42)

    logger.info("=" * 80)
    logger.info("實驗 9: 進階 DES 比較（時間加權 + 少數類加權）")
    logger.info("=" * 80)

    X_hist, y_hist, X_new, y_new, X_test, y_test = get_bankruptcy_splits(
        logger, split_mode=SPLIT_MODE
    )

    results = {}

    # 1. Baseline DES (KNORA-E)
    logger.info("\n--- DES baseline (KNORA-E) ---")
    results["DES_baseline"] = run_des(
        X_hist, y_hist, X_new, y_new, X_test, y_test, logger, k=7
    )
    logger.info(f"AUC={results['DES_baseline']['AUC']:.4f}, F1={results['DES_baseline']['F1']:.4f}")

    # 2. 時間加權 DES（新營運期樣本權重 2.0）
    logger.info("\n--- DES 時間加權 (time_weight_new=2.0) ---")
    results["DES_time_weighted"] = run_des_advanced(
        X_hist, y_hist, X_new, y_new, X_test, y_test, logger,
        k=7, time_weight_new=2.0, minority_weight=1.0
    )
    logger.info(f"AUC={results['DES_time_weighted']['AUC']:.4f}, F1={results['DES_time_weighted']['F1']:.4f}")

    # 3. 少數類加權 DES（少數類樣本權重 2.0）
    logger.info("\n--- DES 少數類加權 (minority_weight=2.0) ---")
    results["DES_minority_weighted"] = run_des_advanced(
        X_hist, y_hist, X_new, y_new, X_test, y_test, logger,
        k=7, time_weight_new=1.0, minority_weight=2.0
    )
    logger.info(f"AUC={results['DES_minority_weighted']['AUC']:.4f}, F1={results['DES_minority_weighted']['F1']:.4f}")

    # 4. Combined（時間 + 少數類 皆 2.0）
    logger.info("\n--- DES combined (time_weight_new=2.0, minority_weight=2.0) ---")
    results["DES_combined"] = run_des_advanced(
        X_hist, y_hist, X_new, y_new, X_test, y_test, logger,
        k=7, time_weight_new=2.0, minority_weight=2.0
    )
    logger.info(f"AUC={results['DES_combined']['AUC']:.4f}, F1={results['DES_combined']['F1']:.4f}")

    # 儲存
    results_df = pd.DataFrame(results).T
    output_dir = project_root / "results" / "des_advanced"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "bankruptcy_des_advanced_comparison.csv"
    results_df.to_csv(output_file)
    logger.info(f"\n結果已保存: {output_file}")

    best_auc = results_df["AUC"].idxmax()
    logger.info(f"最佳 AUC: {best_auc} = {results_df.loc[best_auc, 'AUC']:.4f}")
    logger.info("=" * 80)
    return results_df


if __name__ == "__main__":
    main()
    print("\n實驗 9 完成！結果在 results/des_advanced/bankruptcy_des_advanced_comparison.csv")
