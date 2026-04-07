"""
多 Seed 重現性實驗（擴展版）
涵蓋 Bankruptcy / Stock / Medical 的 Baseline + Ensemble + DES。
輸出每個資料集的 mean±std CSV 到 results/multi_seed/。

使用方式：
    python scripts/run/run_multi_seed.py
    python scripts/run/run_multi_seed.py --seeds 42 123 456 789 2024
    python scripts/run/run_multi_seed.py --dataset bankruptcy --seeds 42 123
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

DEFAULT_SEEDS = [42, 123, 456]          # 可改為 [42,123,456,789,2024] 跑滿 5 次
OUT_DIR = project_root / "results" / "multi_seed"


# ─────────────────────────── per-seed runners ────────────────────────────

def _run_bankruptcy_once(seed: int) -> pd.DataFrame:
    """跑一次 Bankruptcy：Baseline(retrain) + Ensemble(old3/new3/all6) + DES。"""
    from src.utils import set_seed, get_logger
    from src.models import ModelPool, LightGBMWrapper
    from src.data import ImbalanceSampler
    from src.evaluation import compute_metrics
    from experiments.common_bankruptcy import get_bankruptcy_splits
    from experiments.common_des import run_des

    set_seed(seed)
    logger = get_logger("MultiSeed_BK", console=False, file=False)
    X_hist, y_hist, X_new, y_new, X_test, y_test = get_bankruptcy_splits(logger, split_mode="block_cv")
    y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    sampler = ImbalanceSampler(random_state=seed)
    lgbm_params = {"seed": seed, "verbose": -1}
    results = {}

    # Baseline: Re-training
    X_c = pd.concat([X_hist, X_new]); y_c = pd.concat([y_hist, y_new])
    Xr, yr = sampler.apply_sampling(X_c, y_c.values, strategy="hybrid")
    m = LightGBMWrapper(name="retrain", **lgbm_params); m.fit(Xr, yr)
    p = m.predict_proba(X_test)
    results["retrain"] = compute_metrics(y_test_arr, p)

    # Ensemble: Old 3 / New 3 / All 6
    old_pool = ModelPool(pool_name="old", random_state=seed); old_pool.create_pool(X_hist, y_hist.values, prefix="old")
    new_pool = ModelPool(pool_name="new", random_state=seed); new_pool.create_pool(X_new,  y_new.values,  prefix="new")
    all_proba = {**old_pool.predict_proba(X_test), **new_pool.predict_proba(X_test)}
    for name, keys in [
        ("ensemble_old_3", ["old_under","old_over","old_hybrid"]),
        ("ensemble_new_3", ["new_under","new_over","new_hybrid"]),
        ("ensemble_all_6", list(all_proba.keys())),
    ]:
        p_avg = np.mean([all_proba[k] for k in keys], axis=0)
        results[name] = compute_metrics(y_test_arr, p_avg)

    # DES KNORA-E
    results["DES_KNORAE"] = run_des(X_hist, y_hist, X_new, y_new, X_test, y_test, logger)

    return pd.DataFrame(results).T


def _run_stock_once(seed: int) -> pd.DataFrame:
    """跑一次 Stock：Baseline + Ensemble + DES。"""
    from src.utils import set_seed, get_logger
    from src.models import ModelPool, LightGBMWrapper
    from src.data import ImbalanceSampler
    from src.evaluation import compute_metrics
    from experiments.common_dataset import get_splits
    from experiments.common_des import run_des

    set_seed(seed)
    logger = get_logger("MultiSeed_Stock", console=False, file=False)
    X_hist, y_hist, X_new, y_new, X_test, y_test = get_splits("stock", logger)
    y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    sampler = ImbalanceSampler(random_state=seed)
    results = {}

    X_c = pd.concat([X_hist, X_new]); y_c = pd.concat([y_hist, y_new])
    Xr, yr = sampler.apply_sampling(X_c, y_c.values, strategy="hybrid")
    m = LightGBMWrapper(name="retrain", seed=seed, verbose=-1); m.fit(Xr, yr)
    results["retrain"] = compute_metrics(y_test_arr, m.predict_proba(X_test))

    old_pool = ModelPool(pool_name="old", random_state=seed); old_pool.create_pool(X_hist, y_hist.values, prefix="old")
    new_pool = ModelPool(pool_name="new", random_state=seed); new_pool.create_pool(X_new,  y_new.values,  prefix="new")
    all_proba = {**old_pool.predict_proba(X_test), **new_pool.predict_proba(X_test)}
    for name, keys in [
        ("ensemble_old_3", ["old_under", "old_over", "old_hybrid"]),
        ("ensemble_new_3", ["new_under", "new_over", "new_hybrid"]),
        ("ensemble_all_6", list(all_proba.keys())),
    ]:
        p_avg = np.mean([all_proba[k] for k in keys], axis=0)
        results[name] = compute_metrics(y_test_arr, p_avg)
    results["DES_KNORAE"] = run_des(X_hist, y_hist, X_new, y_new, X_test, y_test, logger)

    return pd.DataFrame(results).T


def _run_medical_once(seed: int) -> pd.DataFrame:
    """跑一次 Medical：Baseline + Ensemble + DES。"""
    from src.utils import set_seed, get_logger
    from src.models import ModelPool, LightGBMWrapper
    from src.data import ImbalanceSampler
    from src.evaluation import compute_metrics
    from experiments.common_dataset import get_splits
    from experiments.common_des import run_des

    set_seed(seed)
    logger = get_logger("MultiSeed_Medical", console=False, file=False)
    X_hist, y_hist, X_new, y_new, X_test, y_test = get_splits("medical", logger)
    y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    sampler = ImbalanceSampler(random_state=seed)
    results = {}

    X_c = pd.concat([X_hist, X_new]); y_c = pd.concat([y_hist, y_new])
    Xr, yr = sampler.apply_sampling(X_c, y_c.values, strategy="hybrid")
    m = LightGBMWrapper(name="retrain", seed=seed, verbose=-1); m.fit(Xr, yr)
    results["retrain"] = compute_metrics(y_test_arr, m.predict_proba(X_test))

    old_pool = ModelPool(pool_name="old", random_state=seed); old_pool.create_pool(X_hist, y_hist.values, prefix="old")
    new_pool = ModelPool(pool_name="new", random_state=seed); new_pool.create_pool(X_new,  y_new.values,  prefix="new")
    all_proba = {**old_pool.predict_proba(X_test), **new_pool.predict_proba(X_test)}
    for name, keys in [
        ("ensemble_old_3", ["old_under", "old_over", "old_hybrid"]),
        ("ensemble_new_3", ["new_under", "new_over", "new_hybrid"]),
        ("ensemble_all_6", list(all_proba.keys())),
    ]:
        p_avg = np.mean([all_proba[k] for k in keys], axis=0)
        results[name] = compute_metrics(y_test_arr, p_avg)
    results["DES_KNORAE"] = run_des(X_hist, y_hist, X_new, y_new, X_test, y_test, logger)

    return pd.DataFrame(results).T


RUNNERS = {
    "bankruptcy": _run_bankruptcy_once,
    "stock":      _run_stock_once,
    "medical":    _run_medical_once,
}


# ──────────────────────────── aggregation ────────────────────────────────

def aggregate_seeds(all_runs: list[pd.DataFrame]) -> pd.DataFrame:
    """將多次 seed 結果合併為 mean±std DataFrame。"""
    combined = pd.concat(all_runs)
    metrics = [c for c in combined.columns if c != "seed"]
    agg = {}
    for m in metrics:
        group = combined.groupby(combined.index)[m]
        agg[f"{m}_mean"] = group.mean()
        agg[f"{m}_std"]  = group.std().fillna(0)
    return pd.DataFrame(agg)


def run_dataset(dataset_name: str, seeds: list[int]) -> pd.DataFrame:
    runner = RUNNERS[dataset_name]
    print(f"\n{'='*60}")
    print(f"  {dataset_name.upper()} — seeds={seeds}")
    print(f"{'='*60}")
    all_runs = []
    for seed in seeds:
        print(f"  Running seed={seed}...", end=" ", flush=True)
        try:
            df = runner(seed)
            df["seed"] = seed
            all_runs.append(df)
            print("OK")
        except Exception as e:
            print(f"SKIP ({e})")

    if not all_runs:
        print(f"  No valid runs for {dataset_name}, skipping.")
        return pd.DataFrame()

    result = aggregate_seeds(all_runs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{dataset_name}_multi_seed.csv"
    result.to_csv(out_csv)
    print(f"\n  已保存: {out_csv}")
    # 只顯示 AUC
    auc_cols = [c for c in result.columns if "AUC" in c]
    print(result[auc_cols].to_string(float_format="{:.4f}".format))
    return result


# ──────────────────────────── main ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="多 Seed 重現性實驗")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--dataset", choices=list(RUNNERS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    datasets = list(RUNNERS.keys()) if args.dataset == "all" else [args.dataset]
    print(f"Seeds: {args.seeds}")
    print(f"Datasets: {datasets}")

    for ds in datasets:
        run_dataset(ds, args.seeds)

    print("\n\n多 Seed 實驗完成！結果在 results/multi_seed/")


if __name__ == "__main__":
    main()
