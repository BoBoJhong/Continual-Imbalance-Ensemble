"""
Phase 4 — Proportion Study: 新舊資料比例對集成效能的影響
涵蓋全部資料集 (Bankruptcy / Stock / Medical)。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from experiments._shared.common_bankruptcy import get_bankruptcy_splits
from experiments._shared.common_dataset import get_splits
from experiments._shared.common_des import run_des
from experiments._shared.common_des_advanced import run_des_advanced

SPLIT_MODE  = "block_cv"
NEW_RATIOS  = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
OUTPUT_DIR  = project_root / "results" / "phase4_analysis"


def run_proportion_study(X_h, y_h, X_n, y_n, X_t, y_t, logger):
    results = {}
    n_new   = len(X_n)
    for ratio in NEW_RATIOS:
        n_take = max(1, int(n_new * ratio))
        X_n_sub = X_n.iloc[:n_take]
        y_n_sub = y_n.iloc[:n_take] if hasattr(y_n, "iloc") else y_n[:n_take]
        tag = f"new_{int(ratio*100):03d}"
        try:
            results[f"DES_baseline_{tag}"] = run_des(
                X_h, y_h, X_n_sub, y_n_sub, X_t, y_t, logger, k=7)
            results[f"DES_combined_{tag}"] = run_des_advanced(
                X_h, y_h, X_n_sub, y_n_sub, X_t, y_t, logger, k=7,
                time_weight_new=2.0, minority_weight=2.0)
            logger.info(f"  ratio={ratio:.1f}: DES_baseline AUC={results[f'DES_baseline_{tag}']['AUC']:.4f}")
        except Exception as e:
            logger.warning(f"  ratio={ratio:.1f} SKIP: {e}")
    return pd.DataFrame(results).T


def _run_and_save(name, X_h, y_h, X_n, y_n, X_t, y_t, logger):
    df = run_proportion_study(X_h, y_h, X_n, y_n, X_t, y_t, logger)
    df.to_csv(OUTPUT_DIR / f"{name}_proportion_study.csv")


def main():
    logger = get_logger("Phase4_ProportionStudy", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70 + "\nBankruptcy  (Proportion Study)\n" + "=" * 70)
    _run_and_save("bankruptcy", *get_bankruptcy_splits(logger, split_mode=SPLIT_MODE), logger)

    for ds in ["stock_spx", "stock_dji", "stock_ndx"]:
        logger.info(f"{ds.upper()}  (Proportion Study)")
        try:
            _run_and_save(ds, *get_splits(ds, logger, split_mode=SPLIT_MODE), logger)
        except Exception as e:
            logger.error(f"[ERROR] {ds}: {e}")

    logger.info("=" * 70 + "\nMedical  (Proportion Study)\n" + "=" * 70)
    try:
        _run_and_save("medical", *get_splits("medical", logger, split_mode=SPLIT_MODE), logger)
    except Exception as e:
        logger.error(f"[ERROR] medical: {e}")

    logger.info("\nPhase 4 Proportion Study 完成。results/phase4_analysis/")


if __name__ == "__main__":
    main()
