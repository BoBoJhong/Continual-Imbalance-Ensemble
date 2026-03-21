"""
Phase 3 — FS Sweep: 特徵選擇方法 / 保留比例 全面掃描
FS_METHODS x FS_RATIOS，涵蓋全部資料集 (Bankruptcy / Stock / Medical)。
"""
import sys, warnings
warnings.filterwarnings("ignore")
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
FS_RATIOS  = [0.2, 0.5, 0.8]
FS_METHODS = ["kbest_f", "kbest_chi2", "lasso"]
OUTPUT_DIR = project_root / "results" / "phase3_feature"


def _ensemble_new3(X_h, y_h, X_n, y_n, X_t, y_t):
    old = ModelPool(pool_name="old"); old.create_pool(X_h, y_h.values if hasattr(y_h,"values") else y_h, prefix="old")
    new = ModelPool(pool_name="new"); new.create_pool(X_n, y_n.values if hasattr(y_n,"values") else y_n, prefix="new")
    ap = {**old.predict_proba(X_t), **new.predict_proba(X_t)}
    yt = np.asarray(y_t.values if hasattr(y_t,"values") else y_t)
    return compute_metrics(yt, np.mean([ap["new_under"], ap["new_over"], ap["new_hybrid"]], axis=0))


def run_fs_sweep(X_h, y_h, X_n, y_n, X_t, y_t, logger):
    results = {}
    # baseline (no FS)
    results["no_fs"] = _ensemble_new3(X_h, y_h, X_n, y_n, X_t, y_t)
    # sweep
    for method in FS_METHODS:
        for ratio in FS_RATIOS:
            k = max(1, int(X_h.shape[1] * ratio))
            try:
                fs = FeatureSelector(method=method, k=k)
                Xh2 = fs.fit_transform(X_h, y_h.values if hasattr(y_h,"values") else y_h)
                Xn2 = fs.transform(X_n)
                Xt2 = fs.transform(X_t)
                tag = f"{method}_r{int(ratio*100)}"
                results[tag] = _ensemble_new3(Xh2, y_h, Xn2, y_n, Xt2, y_t)
                logger.info(f"  {tag:25s}: AUC={results[tag]['AUC']:.4f}")
            except Exception as e:
                logger.warning(f"  [{method} ratio={ratio}] SKIP: {e}")
    return pd.DataFrame(results).T


def _run_and_save(name, X_h, y_h, X_n, y_n, X_t, y_t, logger):
    df = run_fs_sweep(X_h, y_h, X_n, y_n, X_t, y_t, logger)
    df.to_csv(OUTPUT_DIR / f"{name}_fs_sweep.csv")
    logger.info(f"  Saved -> {name}_fs_sweep.csv")


def main():
    logger = get_logger("Phase3_FS_Sweep", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70 + "\nBankruptcy  (FS Sweep)\n" + "=" * 70)
    _run_and_save("bankruptcy", *get_bankruptcy_splits(logger, split_mode=SPLIT_MODE), logger)

    for ds in ["stock_spx", "stock_dji", "stock_ndx"]:
        logger.info(f"{'=' * 70}\n{ds.upper()}  (FS Sweep)\n{'=' * 70}")
        try:
            _run_and_save(ds, *get_splits(ds, logger, split_mode=SPLIT_MODE), logger)
        except Exception as e:
            logger.error(f"[ERROR] {ds}: {e}")

    logger.info("=" * 70 + "\nMedical  (FS Sweep)\n" + "=" * 70)
    try:
        _run_and_save("medical", *get_splits("medical", logger, split_mode=SPLIT_MODE), logger)
    except Exception as e:
        logger.error(f"[ERROR] medical: {e}")

    logger.info("\nPhase 3 FS Sweep 完成。results/phase3_feature/")


if __name__ == "__main__":
    main()
