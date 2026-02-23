"""
實驗 10: 比例實驗（後續論文用）
改變 historical vs new 比例（20% new / 50% new / 80% new），比較 retrain、ensemble_new_3、DES、DES_combined。
目的：分析「何時適應策略（ensemble/DES）領先 retrain 最多」。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool
from experiments.common_bankruptcy import get_bankruptcy_splits
from experiments.common_des import run_des
from experiments.common_des_advanced import run_des_advanced

SPLIT_MODE = "block_cv"
RATIOS_NEW = [0.2, 0.5, 0.8]  # 新營運資料佔訓練集比例


def _subsample_stratified(X, y, n, random_state=42):
    """分層抽樣取 n 筆（若 n >= len(X) 則全取）。"""
    n = min(n, len(X))
    if n >= len(X):
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=random_state
    )
    return X_sub, y_sub


def _run_retrain(X_hist, y_hist, X_new, y_new, X_test, y_test):
    """合併 hist+new → 取樣 → 訓練單一模型 → 在 test 評估。"""
    X_combined = pd.concat([X_hist, X_new])
    y_combined = pd.concat([y_hist, y_new])
    sampler = ImbalanceSampler()
    X_res, y_res = sampler.apply_sampling(X_combined, y_combined.values, strategy="hybrid")
    model = LightGBMWrapper(name="retrain")
    model.fit(X_res, y_res)
    y_proba = model.predict_proba(X_test)
    y_pred = (y_proba > 0.5).astype(int)
    return roc_auc_score(y_test, y_proba), f1_score(y_test, y_pred)


def _run_ensemble_new_3(X_hist, y_hist, X_new, y_new, X_test, y_test):
    """Old 池用 hist、New 池用 new，取 ensemble_new_3（3 個 New 模型）在 test 評估。"""
    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_hist, y_hist.values if hasattr(y_hist, "values") else y_hist, prefix="old")
    new_pool = ModelPool(pool_name="new")
    new_pool.create_pool(X_new, y_new.values if hasattr(y_new, "values") else y_new, prefix="new")
    proba = new_pool.predict_proba(X_test)
    y_proba = np.mean(list(proba.values()), axis=0)
    y_pred = (y_proba > 0.5).astype(int)
    y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    return roc_auc_score(y_test_arr, y_proba), f1_score(y_test_arr, y_pred)


def main():
    logger = get_logger("Bankruptcy_Proportion_Study", console=True, file=True)
    set_seed(42)

    logger.info("=" * 80)
    logger.info("實驗 10: 比例實驗（historical vs new 比例）")
    logger.info("=" * 80)

    X_hist, y_hist, X_new, y_new, X_test, y_test = get_bankruptcy_splits(
        logger, split_mode=SPLIT_MODE
    )
    n_hist_total, n_new_total = len(X_hist), len(X_new)
    logger.info(f"hist={n_hist_total}, new={n_new_total}, test={len(X_test)}")

    # 比例 = 新資料佔訓練集比例。固定用「全部 new」，對 hist 分層抽樣使 n_new/(n_hist_sub+n_new)=ratio_new
    # 故 n_hist_use = n_new * (1-r)/r，cap 不超過 len(X_hist)
    rows = []
    for ratio_new in RATIOS_NEW:
        n_new_use = n_new_total  # 全部 new
        n_hist_use = min(int(n_new_use * (1 - ratio_new) / ratio_new), n_hist_total)
        if n_hist_use < 1:
            logger.warning(f"ratio_new={ratio_new} 跳過（n_hist_use={n_hist_use}）")
            continue
        X_hist_sub, y_hist_sub = _subsample_stratified(X_hist, y_hist, n_hist_use)
        X_new_sub, y_new_sub = X_new, y_new  # 全部 new，不抽樣
        label = f"new_{int(ratio_new*100)}"
        logger.info(f"\n--- ratio_new={ratio_new} (hist={n_hist_use}, new={n_new_use}) ---")

        # Retrain
        auc, f1 = _run_retrain(X_hist_sub, y_hist_sub, X_new_sub, y_new_sub, X_test, y_test)
        rows.append({"ratio_new": label, "method": "retrain", "AUC": auc, "F1": f1})
        logger.info(f"  retrain: AUC={auc:.4f}, F1={f1:.4f}")

        # Ensemble_new_3
        auc, f1 = _run_ensemble_new_3(X_hist_sub, y_hist_sub, X_new_sub, y_new_sub, X_test, y_test)
        rows.append({"ratio_new": label, "method": "ensemble_new_3", "AUC": auc, "F1": f1})
        logger.info(f"  ensemble_new_3: AUC={auc:.4f}, F1={f1:.4f}")

        # DES baseline
        res = run_des(X_hist_sub, y_hist_sub, X_new_sub, y_new_sub, X_test, y_test, logger, k=7)
        rows.append({"ratio_new": label, "method": "DES_baseline", "AUC": res["AUC"], "F1": res["F1"]})
        logger.info(f"  DES_baseline: AUC={res['AUC']:.4f}, F1={res['F1']:.4f}")

        # DES combined
        res = run_des_advanced(
            X_hist_sub, y_hist_sub, X_new_sub, y_new_sub, X_test, y_test, logger,
            k=7, time_weight_new=2.0, minority_weight=2.0
        )
        rows.append({"ratio_new": label, "method": "DES_combined", "AUC": res["AUC"], "F1": res["F1"]})
        logger.info(f"  DES_combined: AUC={res['AUC']:.4f}, F1={res['F1']:.4f}")

    df = pd.DataFrame(rows)
    output_dir = project_root / "results" / "proportion_study"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "bankruptcy_ratio_comparison.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"\n結果已保存: {out_csv}")
    logger.info("=" * 80)
    return df


if __name__ == "__main__":
    main()
    print("\n實驗 10 完成！結果在 results/proportion_study/bankruptcy_ratio_comparison.csv")
