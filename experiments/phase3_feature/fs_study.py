"""
Phase 3 — Study II: 特徵選擇對集成的影響
比較「無特徵選擇」vs「FeatureSelector (kbest_f, FS_RATIO=0.5)」，涵蓋全部資料集。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.models import ModelPool
from src.features import FeatureSelector
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import get_bankruptcy_splits
from experiments._shared.common_dataset import get_splits

SPLIT_MODE = "block_cv"
FS_RATIO   = 0.5
FS_METHOD  = "kbest_f"
OUTPUT_DIR = project_root / "results" / "phase3_feature"

COMBOS = {
    "ensemble_old_3":   ["old_under", "old_over", "old_hybrid"],
    "ensemble_new_3":   ["new_under", "new_over", "new_hybrid"],
    "ensemble_all_6":   ["old_under", "old_over", "old_hybrid", "new_under", "new_over", "new_hybrid"],
    "ensemble_pair_hy": ["old_hybrid", "new_hybrid"],
}


def _build_ensemble(X_h, y_h, X_n, y_n, X_t, y_t, logger):
    y_t = np.asarray(y_t.values if hasattr(y_t, "values") else y_t)
    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_h, y_h.values if hasattr(y_h,"values") else y_h, prefix="old")
    new_pool = ModelPool(pool_name="new")
    new_pool.create_pool(X_n, y_n.values if hasattr(y_n,"values") else y_n, prefix="new")
    all_proba = {**old_pool.predict_proba(X_t), **new_pool.predict_proba(X_t)}
    results = {}
    for name, keys in COMBOS.items():
        probs = [all_proba[k] for k in keys]
        results[name] = compute_metrics(y_t, np.mean(probs, axis=0))
    return results


def run_fs_study(X_h, y_h, X_n, y_n, X_t, y_t, logger):
    # 無 FS
    res_no_fs = _build_ensemble(X_h, y_h, X_n, y_n, X_t, y_t, logger)

    # 套用 FS
    k = max(1, int(X_h.shape[1] * FS_RATIO))
    fs = FeatureSelector(method=FS_METHOD, k=k)
    X_h2 = fs.fit_transform(X_h, y_h.values if hasattr(y_h,"values") else y_h)
    X_n2 = fs.transform(X_n)
    X_t2 = fs.transform(X_t)
    logger.info(f"  FS: {X_h.shape[1]} -> {X_h2.shape[1]} features")
    res_fs = _build_ensemble(X_h2, y_h, X_n2, y_n, X_t2, y_t, logger)

    combined = {}
    for k2, v in res_no_fs.items():
        combined[f"{k2}_no_fs"] = v
    for k2, v in res_fs.items():
        combined[f"{k2}_fs"]    = v
    return pd.DataFrame(combined).T


def _run_and_save(name, X_h, y_h, X_n, y_n, X_t, y_t, logger):
    df = run_fs_study(X_h, y_h, X_n, y_n, X_t, y_t, logger)
    df.to_csv(OUTPUT_DIR / f"{name}_fs_study.csv")
    logger.info(f"  Saved -> {name}_fs_study.csv")


def main():
    logger = get_logger("Phase3_FS_Study", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70 + "\nBankruptcy  (FS Study)\n" + "=" * 70)
    _run_and_save("bankruptcy", *get_bankruptcy_splits(logger, split_mode=SPLIT_MODE), logger)

    for ds in ["stock_spx", "stock_dji", "stock_ndx"]:
        logger.info(f"{'=' * 70}\n{ds.upper()}  (FS Study)\n{'=' * 70}")
        try:
            _run_and_save(ds, *get_splits(ds, logger, split_mode=SPLIT_MODE), logger)
        except Exception as e:
            logger.error(f"[ERROR] {ds}: {e}")

    logger.info("=" * 70 + "\nMedical  (FS Study)\n" + "=" * 70)
    try:
        _run_and_save("medical", *get_splits("medical", logger, split_mode=SPLIT_MODE), logger)
    except Exception as e:
        logger.error(f"[ERROR] medical: {e}")

    logger.info("\nPhase 3 FS Study 完成。results/phase3_feature/")


if __name__ == "__main__":
    main()
