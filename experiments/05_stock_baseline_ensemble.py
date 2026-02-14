"""
實驗 5: Stock 資料集 - Baseline + Ensemble
學長論文 / Kaggle 格式，5-fold block CV，與 Bankruptcy 相同流程。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool
from experiments.common_dataset import get_splits

SPLIT_MODE = "block_cv"


def _evaluate(y_true, y_proba, y_pred=None):
    if y_pred is None:
        y_pred = (y_proba > 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }


def main():
    logger = get_logger("Stock_Baseline_Ensemble", console=True, file=True)
    set_seed(42)

    logger.info("=" * 80)
    logger.info("實驗 5: Stock Baseline + Ensemble")
    logger.info("=" * 80)

    X_hist, y_hist, X_new, y_new, X_test, y_test = get_splits(
        "stock", logger, split_mode=SPLIT_MODE
    )
    y_test = np.asarray(y_test.values) if hasattr(y_test, "values") else np.asarray(y_test)
    sampler = ImbalanceSampler()
    results = {}

    # ========== Baselines ==========
    logger.info("\n--- Baseline 1: Re-training ---")
    X_combined = pd.concat([X_hist, X_new], axis=0)
    y_combined = pd.concat([y_hist, y_new], axis=0)
    X_r, y_r = sampler.apply_sampling(X_combined, y_combined.values, strategy="hybrid")
    m = LightGBMWrapper(name="retrain")
    m.fit(X_r, y_r)
    proba = m.predict_proba(X_test)
    results["retrain"] = _evaluate(y_test, proba)

    logger.info("\n--- Baseline 2: Fine-tuning ---")
    X_hr, y_hr = sampler.apply_sampling(X_hist, y_hist.values, strategy="hybrid")
    m = LightGBMWrapper(name="finetune")
    m.fit(X_hr, y_hr)
    X_nr, y_nr = sampler.apply_sampling(X_new, y_new.values, strategy="hybrid")
    m.fit(X_nr, y_nr)
    proba = m.predict_proba(X_test)
    results["finetune"] = _evaluate(y_test, proba)

    logger.info("\n--- Baseline 3: Ensemble (Old 3) ---")
    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_hist, y_hist.values, prefix="old")
    probas = list(old_pool.predict_proba(X_test).values())
    proba_avg = np.mean(probas, axis=0)
    results["ensemble_old_3"] = _evaluate(y_test, proba_avg)

    # ========== Ensemble 組合 ==========
    logger.info("\n--- Ensemble: New 3 + 組合 ---")
    new_pool = ModelPool(pool_name="new")
    new_pool.create_pool(X_new, y_new.values, prefix="new")
    all_proba = {}
    all_proba.update(old_pool.predict_proba(X_test))
    all_proba.update(new_pool.predict_proba(X_test))
    combinations = {
        "ensemble_new_3": ["new_under", "new_over", "new_hybrid"],
        "ensemble_all_6": list(all_proba.keys()),
        "ensemble_2_old_hybrid_new_hybrid": ["old_hybrid", "new_hybrid"],
        "ensemble_3_type_a": ["old_under", "old_over", "new_hybrid"],
        "ensemble_3_type_b": ["old_hybrid", "new_over", "new_hybrid"],
    }
    for name, keys in combinations.items():
        probs = [all_proba[k] for k in keys]
        proba_avg = np.mean(probs, axis=0)
        results[name] = _evaluate(y_test, proba_avg)

    # ========== 儲存 ==========
    results_df = pd.DataFrame(results).T
    out_dir = project_root / "results/stock"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "stock_baseline_ensemble_results.csv"
    results_df.to_csv(out_csv)
    logger.info(f"\n結果已保存: {out_csv}")
    logger.info(results_df.to_string())
    logger.info("\n" + "=" * 80)
    return results_df


if __name__ == "__main__":
    main()
    print("\n實驗 5 完成！結果在 results/stock/stock_baseline_ensemble_results.csv")
