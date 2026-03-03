"""Experiment 29: Systematic All-Combination Ensemble Study.

Tests all C(6,k) k=2..6 subsets with >=1 Old + >=1 New model.
Covers: 3 datasets x 2 split modes = 6 configs, 49 combos each.
"""
import sys
import itertools
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.models import ModelPool
from src.evaluation import compute_metrics
from experiments._shared.common_dataset import get_splits
from experiments._shared.common_bankruptcy import get_bankruptcy_splits

OLD_KEYS = ["old_under", "old_over", "old_hybrid"]
NEW_KEYS = ["new_under", "new_over", "new_hybrid"]
ALL_KEYS = OLD_KEYS + NEW_KEYS


def build_combinations():
    """Generate all subsets of size 2-6 with >=1 Old and >=1 New."""
    combos = []
    for size in range(2, 7):
        for subset in itertools.combinations(ALL_KEYS, size):
            has_old = any(k in OLD_KEYS for k in subset)
            has_new = any(k in NEW_KEYS for k in subset)
            if has_old and has_new:
                o = sum(1 for k in subset if k in OLD_KEYS)
                n = sum(1 for k in subset if k in NEW_KEYS)
                lbl = (f"combo_size{size}_o{o}n{n}_"
                       + "_".join(k.replace("old_","O").replace("new_","N") for k in subset))
                combos.append((lbl, list(subset)))
    return combos


COMBINATIONS = build_combinations()


def run_dataset(dataset_name, split_mode, logger):
    set_seed(42)
    if dataset_name == "bankruptcy":
        sm = "chronological" if split_mode == "chronological" else "block_cv"
        X_h,y_h,X_n,y_n,X_t,y_t = get_bankruptcy_splits(logger, split_mode=sm, dataset="us_1999_2018")
    else:
        X_h,y_h,X_n,y_n,X_t,y_t = get_splits(dataset_name, logger, split_mode=split_mode)
    yt = np.asarray(y_t)
    logger.info(f"  [{dataset_name}|{split_mode}] hist={len(X_h)} new={len(X_n)} test={len(X_t)} pos={yt.mean()*100:.1f}%")
    op = ModelPool(pool_name="old")
    op.create_pool(X_h, y_h.values if hasattr(y_h,"values") else y_h, prefix="old")
    np_ = ModelPool(pool_name="new")
    np_.create_pool(X_n, y_n.values if hasattr(y_n,"values") else y_n, prefix="new")
    ap = {**op.predict_proba(X_t), **np_.predict_proba(X_t)}
    rows = []
    for label, keys in COMBINATIONS:
        probs = np.mean([ap[k] for k in keys], axis=0)
        m = compute_metrics(yt, probs)
        o = sum(1 for k in keys if k in OLD_KEYS)
        n = len(keys) - o
        rows.append({"combo": label, "size": len(keys), "n_old": o, "n_new": n,
                     "models": ",".join(keys), "dataset": dataset_name,
                     "split_mode": split_mode, **m})
    return pd.DataFrame(rows)


DATASETS_SPLITS = [
    ("bankruptcy",  "chronological"),
    ("bankruptcy",  "block_cv"),
    ("stock_spx",   "chronological"),
    ("stock_spx",   "block_cv"),
    ("stock_dji",   "chronological"),
    ("stock_dji",   "block_cv"),
    ("stock_ndx",   "chronological"),
    ("stock_ndx",   "block_cv"),
    ("medical",     "chronological"),
    ("medical",     "block_cv"),
]


def main():
    logger = get_logger("AllCombos", console=True, file=True)
    logger.info("=" * 70)
    logger.info("Experiment 29: Systematic All-Combination Ensemble Study")
    logger.info(f"Total combinations per config: {len(COMBINATIONS)}")
    logger.info("=" * 70)
    all_parts = []
    for ds, mode in DATASETS_SPLITS:
        logger.info(f">>> {ds} | {mode}")
        try:
            df = run_dataset(ds, mode, logger)
            top5 = df.nlargest(5, "AUC")
            for _, r in top5.iterrows():
                logger.info(f"  size={int(r['size'])} o{int(r['n_old'])}n{int(r['n_new'])} AUC={r['AUC']:.4f} F1={r['F1']:.4f} Recall={r['Recall']:.4f} | {r['combo']}")
            all_parts.append(df)
        except Exception as exc:
            logger.error(f"  FAILED {ds}|{mode}: {exc}")
    if all_parts:
        combined = pd.concat(all_parts, ignore_index=True)
        out = project_root / "results" / "phase2_ensemble"
        out.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out / "all_combinations_systematic.csv", index=False)
        logger.info("Saved: results/phase2_ensemble/all_combinations_systematic.csv")
        summary = (
            combined.sort_values("AUC", ascending=False)
            .groupby(["dataset","split_mode","size"]).first().reset_index()
            [["dataset","split_mode","size","n_old","n_new","AUC","F1","Recall","G_Mean","Type1_Error","combo"]]
        )
        summary.to_csv(out / "all_combinations_best_per_size.csv", index=False)
        logger.info("Saved: results/phase2_ensemble/all_combinations_best_per_size.csv")
    else:
        logger.error("No results produced.")


if __name__ == "__main__":
    main()
