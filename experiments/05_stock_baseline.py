"""
實驗 5: Stock 資料集 - Baseline
為美國三大指數 (S&P 500, Dow Jones, NASDAQ) 各跑一次 Baseline 實驗 (Re-training & Fine-tuning)。
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
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper
from experiments.common_dataset import get_splits

SPLIT_MODE = "block_cv"

def run_experiment(dataset_name, logger):
    logger.info(f"\n[{dataset_name.upper()}] 開始執行 Baseline")
    X_hist, y_hist, X_new, y_new, X_test, y_test = get_splits(
        dataset_name, logger, split_mode=SPLIT_MODE
    )
    y_test = np.asarray(y_test.values) if hasattr(y_test, "values") else np.asarray(y_test)
    sampler = ImbalanceSampler()
    results = {}

    # ========== Baselines ==========
    logger.info("  - Baseline 1: Re-training")
    X_combined = pd.concat([X_hist, X_new], axis=0)
    y_combined = pd.concat([y_hist, y_new], axis=0)
    
    sampling_strategies = ["none", "undersampling", "oversampling", "hybrid"]
    for strategy in sampling_strategies:
        X_r, y_r = sampler.apply_sampling(X_combined, y_combined.values, strategy=strategy)
        m = LightGBMWrapper(name=f"retrain_{strategy}")
        m.fit(X_r, y_r)
        proba = m.predict_proba(X_test)
        results[f"retrain_{strategy}"] = compute_metrics(y_test, proba)

    logger.info("  - Baseline 2: Fine-tuning")
    for strategy in sampling_strategies:
        X_hr, y_hr = sampler.apply_sampling(X_hist, y_hist.values, strategy=strategy)
        m = LightGBMWrapper(name=f"finetune_{strategy}")
        m.fit(X_hr, y_hr)
        X_nr, y_nr = sampler.apply_sampling(X_new, y_new.values, strategy=strategy)
        m.fit(X_nr, y_nr)
        proba = m.predict_proba(X_test)
        results[f"finetune_{strategy}"] = compute_metrics(y_test, proba)

    return pd.DataFrame(results).T

def main():
    logger = get_logger("Stock_Baseline", console=True, file=True)
    set_seed(42)

    logger.info("=" * 80)
    logger.info("實驗 5: Stock Baseline (SPX, DJI, NDX)")
    logger.info("=" * 80)

    indices = ["stock_spx", "stock_dji", "stock_ndx"]
    all_results = {}

    for ds in indices:
        try:
            res_df = run_experiment(ds, logger)
            out_dir = project_root / "results/stock"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / f"{ds}_baseline_results.csv"
            res_df.to_csv(out_csv)
            logger.info(f"  -> 結果已保存: {out_csv}")
            all_results[ds] = res_df
        except Exception as e:
            logger.error(f"  [ERROR] {ds} 執行失敗: {e}")

    if all_results:
        avg_df = sum(all_results.values()) / len(all_results)
        out_csv = project_root / "results/stock/stock_baseline_results.csv"
        avg_df.to_csv(out_csv)
        logger.info(f"\n三大指數平均結果已保存: {out_csv}")
        logger.info(avg_df.to_string())

    logger.info("\n" + "=" * 80)

if __name__ == "__main__":
    main()
    print("\n實驗 5 完成！")
