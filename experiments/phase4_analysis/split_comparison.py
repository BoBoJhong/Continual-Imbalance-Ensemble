"""
Phase 4 — Split Comparison: Chronological vs Block-CV 切割方式比較
涵蓋全部資料集 (Bankruptcy / Stock / Medical)。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import get_bankruptcy_splits
from experiments._shared.common_dataset import get_splits
from experiments._shared.common_des import run_des

OUTPUT_DIR = project_root / "results" / "phase4_analysis"


def run_once(get_data_fn, split_mode, logger):
    sm     = ImbalanceSampler()
    X_h, y_h, X_n, y_n, X_t, y_t = get_data_fn(split_mode)
    yt     = np.asarray(y_t.values if hasattr(y_t,"values") else y_t)
    res    = {}
    X_c    = pd.concat([X_h, X_n])
    y_c    = pd.concat([y_h, y_n])
    for strat in ["none", "undersampling", "hybrid"]:
        X_r, y_r = sm.apply_sampling(X_c, y_c.values, strategy=strat)
        m = LightGBMWrapper(name="retrain"); m.fit(X_r, y_r)
        res[f"retrain_{strat}"] = compute_metrics(yt, m.predict_proba(X_t))

        X_hr, y_hr = sm.apply_sampling(X_h, y_h.values, strategy=strat)
        mf = LightGBMWrapper(name="finetune"); mf.fit(X_hr, y_hr)
        X_nr, y_nr = sm.apply_sampling(X_n, y_n.values, strategy=strat)
        mf.fit(X_nr, y_nr)
        res[f"finetune_{strat}"] = compute_metrics(yt, mf.predict_proba(X_t))

    res["DES_KNORAE"] = run_des(X_h, y_h, X_n, y_n, X_t, y_t, logger, k=7)
    return pd.DataFrame(res).T


def compare_splits(name, get_bk_fn, get_cv_fn, logger):
    df_bk = run_once(get_bk_fn, "chronological", logger)
    df_cv = run_once(get_cv_fn, "block_cv",       logger)
    df_bk.index = [f"{i}_chrono" for i in df_bk.index]
    df_cv.index = [f"{i}_blockcv" for i in df_cv.index]
    df = pd.concat([df_bk, df_cv])
    df.to_csv(OUTPUT_DIR / f"{name}_split_comparison.csv")
    logger.info(f"  Saved -> {name}_split_comparison.csv")
    best = df["AUC"].idxmax()
    logger.info(f"  Best AUC: {best} = {df['AUC'].max():.4f}")


def main():
    logger = get_logger("Phase4_SplitComparison", console=True, file=True)
    set_seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    compare_splits("bankruptcy",
        lambda m: get_bankruptcy_splits(logger, split_mode=m),
        lambda m: get_bankruptcy_splits(logger, split_mode=m), logger)

    for ds in ["stock_spx", "medical"]:
        compare_splits(ds,
            lambda m, d=ds: get_splits(d, logger, split_mode=m),
            lambda m, d=ds: get_splits(d, logger, split_mode=m), logger)

    logger.info("\nPhase 4 Split Comparison 完成。results/phase4_analysis/")


if __name__ == "__main__":
    main()
