"""
實驗 7: Medical 資料集 - Baseline
Time series medical (UCI/synthetic)，5-fold block CV，與 Bankruptcy 相同流程。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import compute_metrics
from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper
from experiments.common_dataset import get_splits

SPLIT_MODE = "block_cv"

def main():
    logger = get_logger("Medical_Baseline", console=True, file=True)
    set_seed(42)

    logger.info("=" * 80)
    logger.info("實驗 7: Medical Baseline")
    logger.info("=" * 80)

    X_hist, y_hist, X_new, y_new, X_test, y_test = get_splits(
        "medical", logger, split_mode=SPLIT_MODE
    )
    y_test = np.asarray(y_test.values) if hasattr(y_test, "values") else np.asarray(y_test)
    sampler = ImbalanceSampler()
    results = {}

    # ========== Baselines ==========
    logger.info("\n--- Baseline 1: Re-training ---")
    X_combined = pd.concat([X_hist, X_new], axis=0)
    y_combined = pd.concat([y_hist, y_new], axis=0)
    
    sampling_strategies = ["none", "undersampling", "oversampling", "hybrid"]
    for strategy in sampling_strategies:
        X_r, y_r = sampler.apply_sampling(X_combined, y_combined.values, strategy=strategy)
        m = LightGBMWrapper(name=f"retrain_{strategy}")
        m.fit(X_r, y_r)
        proba = m.predict_proba(X_test)
        results[f"retrain_{strategy}"] = compute_metrics(y_test, proba)

    logger.info("\n--- Baseline 2: Fine-tuning ---")
    for strategy in sampling_strategies:
        X_hr, y_hr = sampler.apply_sampling(X_hist, y_hist.values, strategy=strategy)
        m = LightGBMWrapper(name=f"finetune_{strategy}")
        m.fit(X_hr, y_hr)
        X_nr, y_nr = sampler.apply_sampling(X_new, y_new.values, strategy=strategy)
        m.fit(X_nr, y_nr)
        proba = m.predict_proba(X_test)
        results[f"finetune_{strategy}"] = compute_metrics(y_test, proba)

    # ========== 儲存 ==========
    results_df = pd.DataFrame(results).T
    out_dir = project_root / "results/medical"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "medical_baseline_results.csv"
    results_df.to_csv(out_csv)
    logger.info(f"\n結果已保存: {out_csv}")
    logger.info(results_df.to_string())
    logger.info("\n" + "=" * 80)
    return results_df

if __name__ == "__main__":
    main()
    print("\n實驗 7 完成！結果在 results/medical/medical_baseline_results.csv")
