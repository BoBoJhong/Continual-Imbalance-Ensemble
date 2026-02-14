"""
實驗 8: Medical 資料集 - DES (KNORA-E)
"""
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from experiments.common_dataset import get_splits
from experiments.common_des import run_des

SPLIT_MODE = "block_cv"


def main():
    logger = get_logger("Medical_DES", console=True, file=True)
    set_seed(42)
    logger.info("=" * 80)
    logger.info("實驗 8: Medical DES (KNORA-E)")
    logger.info("=" * 80)
    X_hist, y_hist, X_new, y_new, X_test, y_test = get_splits("medical", logger, split_mode=SPLIT_MODE)
    logger.info("\n步驟 4: 執行 DES")
    res = run_des(X_hist, y_hist, X_new, y_new, X_test, y_test, logger, k=7)
    logger.info(f"DES (KNORA-E) Results: {res}")
    results_df = pd.DataFrame([res], index=["DES_KNORAE"])
    out_dir = project_root / "results/des"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "medical_des_results.csv"
    results_df.to_csv(out_csv)
    logger.info(f"結果已保存: {out_csv}")
    return results_df


if __name__ == "__main__":
    main()
    print("\n實驗 8 完成！結果在 results/des/medical_des_results.csv")
