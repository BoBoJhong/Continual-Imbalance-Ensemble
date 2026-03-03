"""
Phase 3 - DES Standard: KNORA-E 動態集成選擇
涵蓋全部資料集 (Bankruptcy / Stock / Medical)。
"""
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.evaluation import results_to_dataframe
from experiments._shared.common_bankruptcy import get_bankruptcy_splits
from experiments._shared.common_dataset import get_splits
from experiments._shared.common_des import run_des

SPLIT_MODE = "block_cv"
OUTPUT_DIR = project_root / "results" / "phase3_dynamic"


def run_experiment(X_h, y_h, X_n, y_n, X_t, y_t, logger):
    metrics = run_des(X_h, y_h, X_n, y_n, X_t, y_t, logger, k=7)
    return results_to_dataframe({"DES_KNORAE": metrics})


def _run_and_save(name, X_h, y_h, X_n, y_n, X_t, y_t, logger):
    df = run_experiment(X_h, y_h, X_n, y_n, X_t, y_t, logger)
    df.to_csv(OUTPUT_DIR / f"{name}_des_standard.csv")
    logger.info(f"  {name}: AUC={df['AUC'].iloc[0]:.4f}")


def main():
    logger = get_logger("Phase3_DES_Standard", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70 + "\nBankruptcy  (DES Standard)\n" + "=" * 70)
    _run_and_save("bankruptcy", *get_bankruptcy_splits(logger, split_mode=SPLIT_MODE), logger)

    for ds in ["stock_spx", "stock_dji", "stock_ndx"]:
        logger.info(f"{'=' * 70}\n{ds.upper()}  (DES Standard)\n{'=' * 70}")
        try:
            _run_and_save(ds, *get_splits(ds, logger, split_mode=SPLIT_MODE), logger)
        except Exception as e:
            logger.error(f"[ERROR] {ds}: {e}")

    logger.info("=" * 70 + "\nMedical  (DES Standard)\n" + "=" * 70)
    try:
        _run_and_save("medical", *get_splits("medical", logger, split_mode=SPLIT_MODE), logger)
    except Exception as e:
        logger.error(f"[ERROR] medical: {e}")

    logger.info("\nPhase 3 DES Standard 完成。results/phase3_dynamic/")


if __name__ == "__main__":
    main()
