"""
Study II: 特徵選擇對 ensemble 的影響 (修正版)
比較「無特徵選擇」vs「FeatureSelector」下的 ensemble 表現。

Bug fix: 原版 selector.transform() 後欄位變成 f0/f1...，
         但 ModelPool 用相同資料造成 0 差異。
         現改用 src.features.FeatureSelector 並斷言特徵數確實減少。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.models import ModelPool
from src.features import FeatureSelector
from src.evaluation import compute_metrics, print_results_table, results_to_dataframe
from experiments.common_bankruptcy import get_bankruptcy_splits

SPLIT_MODE      = "block_cv"
FS_RATIO        = 0.5      # 保留前 FS_RATIO 比例的特徵（自動適應特徵數）
FS_METHOD       = "kbest_f"   # 可改：kbest_chi2 / lasso


def _run_ensemble(X_hist, y_hist, X_new, y_new, X_test, y_test, logger, tag=""):
    """與實驗 2 相同的 ensemble 流程，回傳各組合的指標 dict。"""
    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_hist, y_hist.values, prefix="old")
    new_pool = ModelPool(pool_name="new")
    new_pool.create_pool(X_new, y_new.values, prefix="new")

    all_proba = {}
    all_proba.update(old_pool.predict_proba(X_test))
    all_proba.update(new_pool.predict_proba(X_test))

    combinations = {
        "ensemble_old_3":                   ["old_under", "old_over", "old_hybrid"],
        "ensemble_new_3":                   ["new_under", "new_over", "new_hybrid"],
        "ensemble_all_6":                   list(all_proba.keys()),
        "ensemble_2_old_hybrid_new_hybrid": ["old_hybrid", "new_hybrid"],
        "ensemble_3_type_a":                ["old_under", "old_over", "new_hybrid"],
        "ensemble_3_type_b":                ["old_hybrid", "new_over", "new_hybrid"],
        "ensemble_4":                       ["old_under", "old_over", "new_over", "new_hybrid"],
        "ensemble_5":                       ["old_under", "old_over", "old_hybrid", "new_over", "new_hybrid"],
    }

    results = {}
    y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    for combo_name, model_names in combinations.items():
        probs = [all_proba[n] for n in model_names]
        y_proba_avg = np.mean(probs, axis=0)
        metrics = compute_metrics(y_test_arr, y_proba_avg)
        results[f"{combo_name}{tag}"] = metrics
        logger.info(f"  {combo_name}{tag}: AUC={metrics['AUC']:.4f}, G-Mean={metrics['G_Mean']:.4f}")
    return results


def main():
    logger = get_logger("Bankruptcy_FS_Study", console=True, file=True)
    set_seed(42)

    logger.info("=" * 80)
    logger.info("Study II: 特徵選擇對 Ensemble 的影響")
    logger.info(f"  FS method={FS_METHOD}, ratio={FS_RATIO}, split={SPLIT_MODE}")
    logger.info("=" * 80)

    X_hist, y_hist, X_new, y_new, X_test, y_test = get_bankruptcy_splits(
        logger, split_mode=SPLIT_MODE
    )
    n_orig = X_hist.shape[1]
    logger.info(f"原始特徵數: {n_orig}")

    # ===== 無特徵選擇 =====
    logger.info("\n--- 無特徵選擇 ---")
    results_no_fs = _run_ensemble(X_hist, y_hist, X_new, y_new, X_test, y_test, logger)

    # ===== 特徵選擇 (fit 於 Historical，transform 所有段) =====
    # k = 至少 1，最多 n_orig-1（確保真正有篩選）
    k_features = max(1, min(int(n_orig * FS_RATIO), n_orig - 1))
    logger.info(f"\n--- 特徵選擇 ({FS_METHOD}, k={k_features}/{n_orig}) ---")
    selector = FeatureSelector(method=FS_METHOD, k=k_features)
    X_hist_fs = selector.fit_transform(X_hist, y_hist.values)
    X_new_fs  = selector.transform(X_new)
    X_test_fs = selector.transform(X_test)

    n_selected = selector.n_selected
    assert n_selected < n_orig, (
        f"特徵選擇無效！selected={n_selected} >= original={n_orig}。"
        f"請調整 FS_RATIO（目前={FS_RATIO}）或 FS_METHOD。"
    )
    logger.info(f"選出特徵數: {n_selected} / {n_orig}（{n_selected/n_orig*100:.1f}%）")
    logger.info(f"選出特徵（前 10）: {selector.selected_cols_[:10]}")

    results_with_fs = _run_ensemble(X_hist_fs, y_hist, X_new_fs, y_new, X_test_fs, y_test, logger, tag="_fs")

    # ===== 比較表 =====
    rows = []
    for combo in [k for k in results_no_fs]:
        r_no = results_no_fs[combo]
        r_fs = results_with_fs.get(f"{combo}_fs", {})
        rows.append({
            "combination":  combo,
            "AUC_no_fs":    r_no["AUC"],
            "AUC_with_fs":  r_fs.get("AUC", float("nan")),
            "AUC_diff":     r_fs.get("AUC", float("nan")) - r_no["AUC"],
            "F1_no_fs":     r_no["F1"],
            "F1_with_fs":   r_fs.get("F1", float("nan")),
            "F1_diff":      r_fs.get("F1", float("nan")) - r_no["F1"],
            "GMean_no_fs":  r_no["G_Mean"],
            "GMean_with_fs":r_fs.get("G_Mean", float("nan")),
            "GMean_diff":   r_fs.get("G_Mean", float("nan")) - r_no["G_Mean"],
        })

    comparison_df = pd.DataFrame(rows)
    logger.info("\n比較表（AUC_diff = with_fs - no_fs）：")
    logger.info("\n" + comparison_df[["combination","AUC_no_fs","AUC_with_fs","AUC_diff"]].to_string(index=False))

    output_dir = project_root / "results/feature_study"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "bankruptcy_fs_comparison.csv"
    comparison_df.to_csv(out_csv, index=False)
    logger.info(f"\n比較結果已保存: {out_csv}")

    logger.info("=" * 80)
    logger.info("Study II 完成")
    logger.info("=" * 80)
    return comparison_df


if __name__ == "__main__":
    main()
    print("\nStudy II 完成！結果在 results/feature_study/bankruptcy_fs_comparison.csv")
