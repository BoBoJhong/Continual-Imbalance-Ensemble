"""
實驗 1: Bankruptcy 資料集 - Baseline 實驗
測試所有基準方法並記錄結果
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger, get_config_loader
from src.data import DataPreprocessor, DataSplitter, ImbalanceSampler
from src.models import LightGBMWrapper, XGBoostWrapper, ModelPool
from experiments.common_bankruptcy import get_bankruptcy_splits

# 切割模式：block_cv = 5-fold（1+2 歷史、3+4 新營運、5 測試）；random = 60-20-20 隨機
SPLIT_MODE = "block_cv"


def main():
    """主實驗流程"""
    
    # 初始化
    logger = get_logger("Bankruptcy_Baseline", console=True, file=True)
    config = get_config_loader()
    set_seed(42)
    
    logger.info("="*80)
    logger.info("實驗 1: Bankruptcy Baseline")
    logger.info("="*80)
    
    # ========== 1~3. 載入、切割（5-fold block CV 或隨機）、前處理 ==========
    X_hist_scaled, y_hist, X_new_scaled, y_new, X_test_scaled, y_test = get_bankruptcy_splits(
        logger, split_mode=SPLIT_MODE
    )
    logger.info("前處理完成")
    
    # ========== 4. Baseline 實驗 ==========
    results = {}
    
    # --- Baseline 1: Re-training (Historical + New) ---
    logger.info("\n" + "="*80)
    logger.info("Baseline 1: Re-training (Historical + New Operating)")
    logger.info("="*80)
    
    X_combined = pd.concat([X_hist_scaled, X_new_scaled])
    y_combined = pd.concat([y_hist, y_new])
    
    sampler = ImbalanceSampler()
    X_resampled, y_resampled = sampler.apply_sampling(
        X_combined, y_combined.values,
        strategy="hybrid"
    )
    
    model_retrain = LightGBMWrapper(name="retrain")
    model_retrain.fit(X_resampled, y_resampled)
    
    y_pred = model_retrain.predict(X_test_scaled)
    y_proba = model_retrain.predict_proba(X_test_scaled)
    
    results['retrain'] = {
        'AUC': roc_auc_score(y_test, y_proba),
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }
    
    logger.info(f"Re-training Results: {results['retrain']}")
    
    # --- Baseline 2: Fine-tuning (僅用新資料做第二階段訓練，符合老師「solely using new data」) ---
    logger.info("\n" + "="*80)
    logger.info("Baseline 2: Fine-tuning")
    logger.info("="*80)

    # 先用 Historical 訓練
    X_hist_resampled, y_hist_resampled = sampler.apply_sampling(
        X_hist_scaled, y_hist.values, strategy="hybrid"
    )
    model_finetune = LightGBMWrapper(name="finetune")
    model_finetune.fit(X_hist_resampled, y_hist_resampled)

    # 僅用 New Operating 做第二階段訓練（微調階段不使用 historical，評估僅在 test）
    X_new_resampled, y_new_resampled = sampler.apply_sampling(
        X_new_scaled, y_new.values, strategy="hybrid"
    )
    model_finetune.fit(X_new_resampled, y_new_resampled)
    
    y_pred = model_finetune.predict(X_test_scaled)
    y_proba = model_finetune.predict_proba(X_test_scaled)
    
    results['finetune'] = {
        'AUC': roc_auc_score(y_test, y_proba),
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }
    
    logger.info(f"Fine-tuning Results: {results['finetune']}")
    
    # --- Baseline 3: Ensemble (Old Models Pool) ---
    logger.info("\n" + "="*80)
    logger.info("Baseline 3: Ensemble (Old Models)")
    logger.info("="*80)
    
    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_hist_scaled, y_hist.values, prefix="old")
    
    old_predictions = old_pool.predict_proba(X_test_scaled)
    
    # 簡單平均集成
    y_proba_avg = np.mean(list(old_predictions.values()), axis=0)
    y_pred_avg = (y_proba_avg > 0.5).astype(int)
    
    results['ensemble_old'] = {
        'AUC': roc_auc_score(y_test, y_proba_avg),
        'F1': f1_score(y_test, y_pred_avg),
        'Precision': precision_score(y_test, y_pred_avg),
        'Recall': recall_score(y_test, y_pred_avg)
    }
    
    logger.info(f"Ensemble (Old) Results: {results['ensemble_old']}")
    
    # ========== 5. 結果總結 ==========
    logger.info("\n" + "="*80)
    logger.info("實驗結果總結")
    logger.info("="*80)
    
    results_df = pd.DataFrame(results).T
    logger.info(f"\n{results_df}")
    
    # 保存結果
    output_dir = project_root / 'results/baseline'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'bankruptcy_baseline_results.csv'
    results_df.to_csv(output_file)
    logger.info(f"\n結果已保存到: {output_file}")
    
    # 找出最佳方法
    best_method = results_df['AUC'].idxmax()
    best_auc = results_df['AUC'].max()
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ 最佳方法: {best_method}")
    logger.info(f"✅ 最佳 AUC: {best_auc:.4f}")
    logger.info("="*80)
    
    return results_df


if __name__ == "__main__":
    results = main()
    print("\n實驗完成！查看 logs/ 目錄查看詳細日誌")
