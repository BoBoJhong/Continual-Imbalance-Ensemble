"""
Phase 1 - Baseline 2: Fine-tuning
先以歷史資料訓練，再以新營運資料微調，搭配 4 種採樣策略，涵蓋全部資料集。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import get_bankruptcy_splits
from experiments._shared.common_dataset import get_splits

SPLIT_MODE         = "block_cv"
SAMPLING_STRATEGIES = ["none", "undersampling", "oversampling", "hybrid"]
OUTPUT_DIR         = project_root / "results" / "phase1_baseline"


def run_finetune(X_hist, y_hist, X_new, y_new, X_test, y_test, logger):
    """Baseline 2: 歷史訓練 → 新營運微調，回傳 DataFrame。"""
    sampler = ImbalanceSampler()
    y_t     = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    results = {}
    for strat in SAMPLING_STRATEGIES:
        # 第一階段：歷史資料
        X_hr, y_hr = sampler.apply_sampling(X_hist, y_hist.values, strategy=strat)
        m = LightGBMWrapper(name=f"finetune_{strat}")
        m.fit(X_hr, y_hr)
        # 第二階段：新營運微調
        X_nr, y_nr = sampler.apply_sampling(X_new, y_new.values, strategy=strat)
        m.fit(X_nr, y_nr)
        results[f"finetune_{strat}"] = compute_metrics(y_t, m.predict_proba(X_test))
        logger.info(f"  {strat:15s}: AUC={results[f'finetune_{strat}']['AUC']:.4f}")
    return pd.DataFrame(results).T


def _run_and_save(name, X_h, y_h, X_n, y_n, X_t, y_t, logger):
    df = run_finetune(X_h, y_h, X_n, y_n, X_t, y_t, logger)
    df.to_csv(OUTPUT_DIR / f"{name}_finetune.csv")
    logger.info(f"  Saved -> {name}_finetune.csv")
    return df


def main():
    logger = get_logger("Phase1_Finetune", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70 + "\nBankruptcy  (Fine-tuning)\n" + "=" * 70)
    _run_and_save("bankruptcy", *get_bankruptcy_splits(logger, split_mode=SPLIT_MODE), logger)

    for ds in ["stock_spx", "stock_dji", "stock_ndx"]:
        logger.info("=" * 70 + f"\n{ds.upper()}  (Fine-tuning)\n" + "=" * 70)
        try:
            _run_and_save(ds, *get_splits(ds, logger, split_mode=SPLIT_MODE), logger)
        except Exception as e:
            logger.error(f"[ERROR] {ds}: {e}")

    logger.info("=" * 70 + "\nMedical  (Fine-tuning)\n" + "=" * 70)
    try:
        _run_and_save("medical", *get_splits("medical", logger, split_mode=SPLIT_MODE), logger)
    except Exception as e:
        logger.error(f"[ERROR] medical: {e}")

    logger.info("\nPhase 1 Fine-tuning 完成。results/phase1_baseline/")


if __name__ == "__main__":
    main()
