"""Run the legacy LightGBM ensemble comparison across configured seeds.

This protocol uses fixed block/time splits and is a reproducibility diagnostic,
not the confirmatory rolling evaluation. Per-seed rows are retained in Git.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
OUT_DIR = PROJECT_ROOT / "results" / "multi_seed"


def get_default_seeds() -> list[int]:
    """Load the canonical seed list from ``config/base_config.yaml``."""
    from src.utils import get_config_loader

    values = get_config_loader().get("base_config", "base_config.random_seeds", default=[42])
    return [int(value) for value in values]


def _get_dataset_splits(dataset_name: str, logger):
    if dataset_name == "bankruptcy":
        from experiments._shared.common_bankruptcy import get_bankruptcy_splits

        return get_bankruptcy_splits(logger, split_mode="block_cv")

    from experiments._shared.common_dataset import get_splits

    return get_splits(dataset_name, logger)


def _run_once(dataset_name: str, seed: int) -> pd.DataFrame:
    from experiments._shared.common_des import run_des
    from src.data import ImbalanceSampler
    from src.evaluation import compute_metrics
    from src.models import LightGBMWrapper, ModelPool
    from src.utils import get_logger, set_seed

    set_seed(seed)
    logger = get_logger(f"MultiSeed_{dataset_name}", console=False, file=False)
    X_hist, y_hist, X_new, y_new, X_test, y_test = _get_dataset_splits(dataset_name, logger)
    y_hist_array = np.asarray(y_hist)
    y_new_array = np.asarray(y_new)
    y_test_array = np.asarray(y_test)

    combined_X = pd.concat([X_hist, X_new])
    combined_y = np.concatenate([y_hist_array, y_new_array])
    sampler = ImbalanceSampler(random_state=seed)
    resampled_X, resampled_y = sampler.apply_sampling(combined_X, combined_y, strategy="hybrid")
    retrained = LightGBMWrapper(name="retrain", seed=seed, verbose=-1)
    retrained.fit(resampled_X, resampled_y)
    results = {"retrain": compute_metrics(y_test_array, retrained.predict_proba(X_test))}

    old_pool = ModelPool(pool_name="old", random_state=seed)
    old_pool.create_pool(X_hist, y_hist_array, prefix="old")
    new_pool = ModelPool(pool_name="new", random_state=seed)
    new_pool.create_pool(X_new, y_new_array, prefix="new")
    probabilities = {
        **old_pool.predict_proba(X_test),
        **new_pool.predict_proba(X_test),
    }
    combinations = {
        "ensemble_old_3": ["old_under", "old_over", "old_hybrid"],
        "ensemble_new_3": ["new_under", "new_over", "new_hybrid"],
        "ensemble_all_6": list(probabilities),
    }
    for method_name, model_names in combinations.items():
        average = np.mean([probabilities[name] for name in model_names], axis=0)
        results[method_name] = compute_metrics(y_test_array, average)

    results["DES_KNORAE"] = run_des(
        X_hist,
        y_hist,
        X_new,
        y_new,
        X_test,
        y_test,
        logger,
        random_state=seed,
    )
    return pd.DataFrame(results).T


def aggregate_seeds(all_runs: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate metric means and sample standard deviations by method."""
    combined = pd.concat(all_runs)
    metrics = [column for column in combined.columns if column != "seed"]
    aggregated = {}
    for metric in metrics:
        grouped = combined.groupby(combined.index)[metric]
        aggregated[f"{metric}_mean"] = grouped.mean()
        aggregated[f"{metric}_std"] = grouped.std().fillna(0)
    return pd.DataFrame(aggregated)


def run_dataset(dataset_name: str, seeds: list[int]) -> pd.DataFrame:
    print(f"\n{'=' * 60}\n{dataset_name.upper()} seeds={seeds}\n{'=' * 60}")
    all_runs: list[pd.DataFrame] = []
    failures: list[tuple[int, str]] = []
    for seed in seeds:
        print(f"Running seed={seed}...", end=" ", flush=True)
        try:
            frame = _run_once(dataset_name, seed)
        except Exception as exc:  # Keep other seeds auditable if one run fails.
            failures.append((seed, f"{type(exc).__name__}: {exc}"))
            print(f"FAILED ({failures[-1][1]})")
            continue
        frame["seed"] = seed
        all_runs.append(frame)
        print("OK")

    if not all_runs:
        raise RuntimeError(f"No successful runs for {dataset_name}: {failures}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(all_runs)
    raw.index.name = "method"
    raw.to_csv(OUT_DIR / f"{dataset_name}_multi_seed_raw.csv")
    result = aggregate_seeds(all_runs)
    result.to_csv(OUT_DIR / f"{dataset_name}_multi_seed.csv")

    if failures:
        failure_frame = pd.DataFrame(failures, columns=["seed", "error"])
        failure_frame.to_csv(OUT_DIR / f"{dataset_name}_multi_seed_errors.csv", index=False)
    print(result.filter(like="AUC").to_string(float_format="{:.4f}".format))
    return result


def main() -> int:
    datasets = ("bankruptcy", "stock", "medical")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=get_default_seeds())
    parser.add_argument("--dataset", choices=(*datasets, "all"), default="all")
    args = parser.parse_args()

    selected = datasets if args.dataset == "all" else (args.dataset,)
    for dataset_name in selected:
        run_dataset(dataset_name, args.seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
