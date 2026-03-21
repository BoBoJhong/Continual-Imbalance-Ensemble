"""
Phase 2 - Ensemble (欠採樣)
使用 undersampling 採樣訓練 Old / New 模型池，比較歷史池、新資料池與組合效果。
涵蓋全部資料集 (Bankruptcy / Stock / Medical)。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.models import ModelPool
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import get_bankruptcy_splits
from experiments._shared.common_dataset import get_splits

SPLIT_MODE    = "block_cv"
SAMPLING_TYPE = "undersampling"          # undersampling | oversampling | hybrid
OUTPUT_DIR    = project_root / "results" / "phase2_ensemble" / "static"

COMBINATIONS = {
    "old_only":     [f"old_{SAMPLING_TYPE}"],
    "new_only":     [f"new_{SAMPLING_TYPE}"],
    "old_new_pair": [f"old_{SAMPLING_TYPE}", f"new_{SAMPLING_TYPE}"],
}


def run_ensemble(X_hist, y_hist, X_new, y_new, X_test, y_test, logger):
    y_t = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)

    old_pool = ModelPool(pool_name="old")
    old_pool.create_model_with_sampling(X_hist, y_hist.values if hasattr(y_hist,"values") else y_hist,
                                        SAMPLING_TYPE, f"old_{SAMPLING_TYPE}")

    new_pool = ModelPool(pool_name="new")
    new_pool.create_model_with_sampling(X_new, y_new.values if hasattr(y_new,"values") else y_new,
                                        SAMPLING_TYPE, f"new_{SAMPLING_TYPE}")

    all_proba = {}
    all_proba.update(old_pool.predict_proba(X_test))
    all_proba.update(new_pool.predict_proba(X_test))

    results = {}
    for combo_name, keys in COMBINATIONS.items():
        probs         = [all_proba[k] for k in keys if k in all_proba]
        y_proba_avg   = np.mean(probs, axis=0)
        results[combo_name] = compute_metrics(y_t, y_proba_avg)
        logger.info(f"  {combo_name:25s}: AUC={results[combo_name]['AUC']:.4f}")
    return pd.DataFrame(results).T


def _run_and_save(name, X_h, y_h, X_n, y_n, X_t, y_t, logger):
    df = run_ensemble(X_h, y_h, X_n, y_n, X_t, y_t, logger)
    df.to_csv(OUTPUT_DIR / f"{name}_ensemble_{SAMPLING_TYPE}.csv")


def main():
    logger = get_logger(f"Phase2_Ensemble_{SAMPLING_TYPE}", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70 + f"\nBankruptcy  (Ensemble - {SAMPLING_TYPE})\n" + "=" * 70)
    _run_and_save("bankruptcy", *get_bankruptcy_splits(logger, split_mode=SPLIT_MODE), logger)

    for ds in ["stock_spx", "stock_dji", "stock_ndx"]:
        logger.info(f"{ds.upper()}  (Ensemble - {SAMPLING_TYPE})")
        try:
            _run_and_save(ds, *get_splits(ds, logger, split_mode=SPLIT_MODE), logger)
        except Exception as e:
            logger.error(f"[ERROR] {ds}: {e}")

    logger.info("=" * 70 + f"\nMedical  (Ensemble - {SAMPLING_TYPE})\n" + "=" * 70)
    try:
        _run_and_save("medical", *get_splits("medical", logger, split_mode=SPLIT_MODE), logger)
    except Exception as e:
        logger.error(f"[ERROR] medical: {e}")

    logger.info(f"Phase 2 Ensemble ({SAMPLING_TYPE}) 完成。results/phase2_ensemble/static/")


if __name__ == "__main__":
    main()
