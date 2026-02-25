"""
實驗 2: Bankruptcy 資料集 - Ensemble 實驗
建立 Old + New 模型池，測試多種 ensemble 組合（軟投票平均）
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation import compute_metrics

from src.utils import set_seed, get_logger, get_config_loader
from src.data import DataPreprocessor, DataSplitter, ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool
from experiments.common_bankruptcy import get_bankruptcy_splits

# 切割模式：block_cv = 5-fold（1+2 歷史、3+4 新營運、5 測試）；random = 60-20-20 隨機
SPLIT_MODE = "block_cv"





def main():
    logger = get_logger("Bankruptcy_Ensemble", console=True, file=True)
    set_seed(42)

    logger.info("=" * 80)
    logger.info("實驗 2: Bankruptcy Ensemble (Old + New 模型池)")
    logger.info("=" * 80)

    X_hist, y_hist, X_new, y_new, X_test, y_test = get_bankruptcy_splits(logger, split_mode=SPLIT_MODE)

    # ========== 建立 Old 與 New 模型池 ==========
    logger.info("\n步驟 4: 建立 Old 模型池 (Historical)")
    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_hist, y_hist.values, prefix="old")

    logger.info("步驟 5: 建立 New 模型池 (New Operating)")
    new_pool = ModelPool(pool_name="new")
    new_pool.create_pool(X_new, y_new.values, prefix="new")

    # 合併為 6 個模型的預測字典 (old_under, old_over, old_hybrid, new_under, new_over, new_hybrid)
    all_proba = {}
    all_proba.update(old_pool.predict_proba(X_test))
    all_proba.update(new_pool.predict_proba(X_test))

    # ========== Ensemble 組合（對應 des_config 概念）==========
    # 名稱對應: old_1_under->old_under, old_2_over->old_over, old_3_hybrid->old_hybrid, 同理 new
    combinations = {
        "ensemble_old_3": ["old_under", "old_over", "old_hybrid"],
        "ensemble_new_3": ["new_under", "new_over", "new_hybrid"],
        "ensemble_all_6": list(all_proba.keys()),
        "ensemble_2_old_hybrid_new_hybrid": ["old_hybrid", "new_hybrid"],
        "ensemble_3_type_a": ["old_under", "old_over", "new_hybrid"],
        "ensemble_3_type_b": ["old_hybrid", "new_over", "new_hybrid"],
        "ensemble_4": ["old_under", "old_over", "new_over", "new_hybrid"],
        "ensemble_5": ["old_under", "old_over", "old_hybrid", "new_over", "new_hybrid"],
    }

    results = {}
    for combo_name, model_names in combinations.items():
        probs = [all_proba[n] for n in model_names]
        y_proba_avg = np.mean(probs, axis=0)
        results[combo_name] = compute_metrics(y_test.values, y_proba_avg)
        logger.info(f"{combo_name}: AUC={results[combo_name]['AUC']:.4f}, G-Mean={results[combo_name]['G_Mean']:.4f}")

    # ========== 結果總結與儲存 ==========
    logger.info("\n" + "=" * 80)
    logger.info("Ensemble 實驗結果總結")
    logger.info("=" * 80)

    results_df = pd.DataFrame(results).T
    logger.info(f"\n{results_df}")

    output_dir = project_root / "results/ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "bankruptcy_ensemble_results.csv"
    results_df.to_csv(output_file)
    logger.info(f"\n結果已保存到: {output_file}")

    best_method = results_df["AUC"].idxmax()
    best_auc = results_df["AUC"].max()
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ 最佳組合: {best_method}")
    logger.info(f"✅ 最佳 AUC: {best_auc:.4f}")
    logger.info("=" * 80)

    return results_df


if __name__ == "__main__":
    main()
    print("\n實驗 2 完成！結果在 results/ensemble/bankruptcy_ensemble_results.csv")
