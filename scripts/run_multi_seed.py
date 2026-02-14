"""
多 seed 重跑實驗 01（Baseline），產出 mean±std 供論文/報告使用。
對應 config/experiment_config.yaml 的 reproducibility.random_seeds。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 使用前 3 個 seed 以節省時間；可改為 [42, 123, 456, 789, 2024] 跑滿 5 次
SEEDS = [42, 123, 456]


def run_baseline_once(seed):
    from src.utils import set_seed, get_logger
    from src.data import ImbalanceSampler
    from src.models import LightGBMWrapper, ModelPool
    from experiments.common_bankruptcy import get_bankruptcy_splits
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    set_seed(seed)
    logger = get_logger("MultiSeed", console=False, file=False)
    X_hist, y_hist, X_new, y_new, X_test, y_test = get_bankruptcy_splits(logger, split_mode="block_cv")
    y_test = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    sampler = ImbalanceSampler()
    results = {}

    X_combined = pd.concat([X_hist, X_new], axis=0)
    y_combined = pd.concat([y_hist, y_new], axis=0)
    X_r, y_r = sampler.apply_sampling(X_combined, y_combined.values, strategy="hybrid")
    m = LightGBMWrapper(name="retrain")
    m.fit(X_r, y_r)
    p = m.predict_proba(X_test)
    results["retrain"] = {"AUC": roc_auc_score(y_test, p), "F1": f1_score(y_test, (p >= 0.5).astype(int))}

    X_hr, y_hr = sampler.apply_sampling(X_hist, y_hist.values, strategy="hybrid")
    m = LightGBMWrapper(name="finetune")
    m.fit(X_hr, y_hr)
    X_nr, y_nr = sampler.apply_sampling(X_new, y_new.values, strategy="hybrid")
    m.fit(X_nr, y_nr)
    p = m.predict_proba(X_test)
    results["finetune"] = {"AUC": roc_auc_score(y_test, p), "F1": f1_score(y_test, (p >= 0.5).astype(int))}

    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_hist, y_hist.values, prefix="old")
    probs = list(old_pool.predict_proba(X_test).values())
    p_avg = np.mean(probs, axis=0)
    results["ensemble_old"] = {"AUC": roc_auc_score(y_test, p_avg), "F1": f1_score(y_test, (p_avg >= 0.5).astype(int))}

    return pd.DataFrame(results).T


def main():
    print("多 seed 重跑 Bankruptcy Baseline，seeds =", SEEDS)
    all_runs = []
    for seed in SEEDS:
        df = run_baseline_once(seed)
        df["seed"] = seed
        all_runs.append(df)
    combined = pd.concat(all_runs)
    mean_std = combined.groupby(combined.index).agg({"AUC": ["mean", "std"], "F1": ["mean", "std"]})
    mean_std.columns = ["AUC_mean", "AUC_std", "F1_mean", "F1_std"]
    out_dir = project_root / "results/baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "bankruptcy_baseline_mean_std.csv"
    mean_std.to_csv(out_csv)
    print(f"已保存: {out_csv}")
    print(mean_std.to_string())
    return mean_std


if __name__ == "__main__":
    main()
