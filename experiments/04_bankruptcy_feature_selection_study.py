"""
Study II: 特徵選擇對 ensemble 的影響
比較「無特徵選擇」vs「SelectKBest 特徵選擇」下的 ensemble 表現。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import DataPreprocessor, ImbalanceSampler
from src.models import ModelPool
from experiments.common_bankruptcy import get_bankruptcy_splits

SPLIT_MODE = "block_cv"
N_FEATURES_SELECT = 50  # 特徵選擇保留數（對應 feature_config.yaml）


def _evaluate(y_true, y_proba, y_pred=None):
    if y_pred is None:
        y_pred = (y_proba > 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y_true, y_proba),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
    }


def _run_ensemble(X_hist, y_hist, X_new, y_new, X_test, y_test, logger):
    """與實驗 2 相同的 ensemble 流程，回傳各組合的指標 dict。"""
    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_hist, y_hist.values, prefix="old")
    new_pool = ModelPool(pool_name="new")
    new_pool.create_pool(X_new, y_new.values, prefix="new")
    all_proba = {}
    all_proba.update(old_pool.predict_proba(X_test))
    all_proba.update(new_pool.predict_proba(X_test))
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
        y_pred_avg = (y_proba_avg > 0.5).astype(int)
        results[combo_name] = _evaluate(y_test.values, y_proba_avg, y_pred_avg)
    return results


def main():
    logger = get_logger("Bankruptcy_FS_Study", console=True, file=True)
    set_seed(42)

    logger.info("=" * 80)
    logger.info("Study II: 特徵選擇對 Ensemble 的影響")
    logger.info("=" * 80)

    X_hist, y_hist, X_new, y_new, X_test, y_test = get_bankruptcy_splits(
        logger, split_mode=SPLIT_MODE
    )

    # ========== 無特徵選擇 ==========
    logger.info("\n--- 無特徵選擇 ---")
    results_no_fs = _run_ensemble(X_hist, y_hist, X_new, y_new, X_test, y_test, logger)
    for k, v in results_no_fs.items():
        logger.info(f"  {k}: AUC={v['AUC']:.4f}")

    # ========== 特徵選擇：SelectKBest (f_classif)，fit 於 historical ==========
    logger.info(f"\n--- 特徵選擇 (SelectKBest k={N_FEATURES_SELECT}) ---")
    k = min(N_FEATURES_SELECT, X_hist.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_hist, y_hist.values)
    X_hist_fs = pd.DataFrame(
        selector.transform(X_hist),
        index=X_hist.index,
        columns=[f"f{i}" for i in range(k)],
    )
    X_new_fs = pd.DataFrame(
        selector.transform(X_new),
        index=X_new.index,
        columns=[f"f{i}" for i in range(k)],
    )
    X_test_fs = pd.DataFrame(
        selector.transform(X_test),
        index=X_test.index,
        columns=[f"f{i}" for i in range(k)],
    )
    results_with_fs = _run_ensemble(
        X_hist_fs, y_hist, X_new_fs, y_new, X_test_fs, y_test, logger
    )
    for k, v in results_with_fs.items():
        logger.info(f"  {k}: AUC={v['AUC']:.4f}")

    # ========== 比較表 ==========
    rows = []
    for combo in results_no_fs:
        r_no = results_no_fs[combo]
        r_fs = results_with_fs[combo]
        rows.append({
            "combination": combo,
            "AUC_no_fs": r_no["AUC"],
            "AUC_with_fs": r_fs["AUC"],
            "AUC_diff": r_fs["AUC"] - r_no["AUC"],
            "F1_no_fs": r_no["F1"],
            "F1_with_fs": r_fs["F1"],
            "F1_diff": r_fs["F1"] - r_no["F1"],
        })
    comparison_df = pd.DataFrame(rows)

    output_dir = project_root / "results/feature_study"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "bankruptcy_fs_comparison.csv"
    comparison_df.to_csv(out_csv, index=False)
    logger.info(f"\n比較結果已保存: {out_csv}")
    logger.info("\n" + comparison_df.to_string())

    logger.info("\n" + "=" * 80)
    logger.info("Study II 完成")
    logger.info("=" * 80)
    return comparison_df


if __name__ == "__main__":
    main()
    print("\nStudy II 完成！結果在 results/feature_study/bankruptcy_fs_comparison.csv")
