"""
實驗 6: Stock 資料集 - Ensemble
為美國三大指數 (S&P 500, Dow Jones, NASDAQ) 各跑一次 Static Ensemble 實驗。
學長論文 / Kaggle 格式，5-fold block CV，與 Bankruptcy 相同流程。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import compute_metrics
from src.utils import set_seed, get_logger
from src.models import ModelPool
from experiments.common_dataset import get_splits

SPLIT_MODE = "block_cv"

def run_experiment(dataset_name, logger):
    logger.info(f"\n[{dataset_name.upper()}] 開始執行 Ensemble")
    X_hist, y_hist, X_new, y_new, X_test, y_test = get_splits(
        dataset_name, logger, split_mode=SPLIT_MODE
    )
    y_test = np.asarray(y_test.values) if hasattr(y_test, "values") else np.asarray(y_test)
    results = {}

    # ========== Ensemble ==========
    logger.info("  - Ensemble: Old 3")
    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_hist, y_hist.values, prefix="old")
    probas = list(old_pool.predict_proba(X_test).values())
    proba_avg = np.mean(probas, axis=0)
    results["ensemble_old_3"] = compute_metrics(y_test, proba_avg)

    logger.info("  - Ensemble: New 3 + 組合")
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
        results[name] = compute_metrics(y_test, proba_avg)

    return pd.DataFrame(results).T

def main():
    logger = get_logger("Stock_Ensemble", console=True, file=True)
    set_seed(42)

    logger.info("=" * 80)
    logger.info("實驗 6: Stock Ensemble (SPX, DJI, NDX)")
    logger.info("=" * 80)

    indices = ["stock_spx", "stock_dji", "stock_ndx"]
    all_results = {}

    for ds in indices:
        try:
            res_df = run_experiment(ds, logger)
            out_dir = project_root / "results/stock"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"{ds}_ensemble_results.csv"
            res_df.to_csv(out_csv)
            logger.info(f"  -> 結果已保存: {out_csv}")
            all_results[ds] = res_df
        except Exception as e:
            logger.error(f"  [ERROR] {ds} 執行失敗: {e}")

    if all_results:
        avg_df = sum(all_results.values()) / len(all_results)
        out_csv = project_root / "results/stock/stock_ensemble_results.csv"
        avg_df.to_csv(out_csv)
        logger.info(f"\n三大指數平均結果已保存: {out_csv}")
        logger.info(avg_df.to_string())

    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    main()
    print("\n實驗 6 完成！")
