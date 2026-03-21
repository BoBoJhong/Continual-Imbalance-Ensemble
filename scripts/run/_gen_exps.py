"""Helper: generate experiment files 23, 24, 25"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXP = ROOT / "experiments"

# ─────────────────────────────── EXP 23 ────────────────────────────────
EXP23 = '''\
"""Experiment 23: Bankruptcy – Chronological vs Block-CV split comparison."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool
from src.evaluation import compute_metrics
from experiments.common_bankruptcy import get_bankruptcy_splits
from experiments.common_des import run_des


def run_once(split_mode, logger):
    set_seed(42)
    sm = ImbalanceSampler()
    X_h, y_h, X_n, y_n, X_t, y_t = get_bankruptcy_splits(
        logger, split_mode=split_mode, dataset="us_1999_2018"
    )
    yt = np.asarray(y_t)
    logger.info(
        "  [%s] hist=%d new=%d test=%d pos_rate=%.1f%%",
        split_mode, len(X_h), len(X_n), len(X_t), yt.mean() * 100,
    )
    res = {}
    X_c = pd.concat([X_h, X_n])
    y_c = pd.concat([y_h, y_n])

    for strat in ["none", "undersampling", "hybrid"]:
        Xr, yr = sm.apply_sampling(X_c, y_c.values, strategy=strat)
        m = LightGBMWrapper(name="retrain_" + strat)
        m.fit(Xr, yr)
        res["retrain_" + strat] = compute_metrics(yt, m.predict_proba(X_t))

        Xhr, yhr = sm.apply_sampling(X_h, y_h.values, strategy=strat)
        mf = LightGBMWrapper(name="finetune_" + strat)
        mf.fit(Xhr, yhr)
        Xnr, ynr = sm.apply_sampling(X_n, y_n.values, strategy=strat)
        mf.fit(Xnr, ynr)
        res["finetune_" + strat] = compute_metrics(yt, mf.predict_proba(X_t))

    op = ModelPool(pool_name="old")
    op.create_pool(X_h, y_h.values, prefix="old")
    np_ = ModelPool(pool_name="new")
    np_.create_pool(X_n, y_n.values, prefix="new")
    ap = {**op.predict_proba(X_t), **np_.predict_proba(X_t)}

    for nm, keys in [
        ("ensemble_old_3", ["old_under", "old_over", "old_hybrid"]),
        ("ensemble_new_3", ["new_under", "new_over", "new_hybrid"]),
        ("ensemble_all_6", list(ap.keys())),
    ]:
        res[nm] = compute_metrics(yt, np.mean([ap[k] for k in keys], axis=0))

    try:
        res["DES_KNORAE"] = run_des(X_h, y_h, X_n, y_n, X_t, y_t, logger)
    except Exception as exc:
        logger.warning("DES failed: %s", exc)

    df = pd.DataFrame(res).T
    df["split_mode"] = split_mode
    return df


def main():
    logger = get_logger("BK_SplitComp", console=True, file=True)
    logger.info("=" * 70)
    logger.info("Experiment 23: Bankruptcy Split Mode Comparison")
    logger.info("=" * 70)

    parts = []
    for mode in ["chronological", "block_cv"]:
        logger.info(">>> split_mode=%s", mode)
        df = run_once(mode, logger)
        for idx, r in df.iterrows():
            logger.info(
                "  %-35s AUC=%.4f  F1=%.4f  Recall=%.4f",
                idx, r["AUC"], r["F1"], r["Recall"],
            )
        parts.append(df)

    combined = pd.concat(parts)
    out = project_root / "results" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "bankruptcy_split_comparison.csv")
    logger.info("Saved to results/baseline/bankruptcy_split_comparison.csv")
    return combined


if __name__ == "__main__":
    main()
'''

# ─────────────────────────────── EXP 24 ────────────────────────────────
EXP24 = '''\
"""Experiment 24: Advanced feature selection sweep (kbest_f / mutual_info / shap)."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool
from src.evaluation import compute_metrics
from src.features import FeatureSelector
from experiments.common_bankruptcy import get_bankruptcy_splits
from experiments.common_dataset import get_stock_splits, get_medical_splits


METHODS = ["none", "kbest_f", "mutual_info", "shap"]
K = 30


def apply_fs(method, X_h, y_h, X_n, y_n, X_t):
    if method == "none":
        return X_h, X_n, X_t
    fs = FeatureSelector(method=method, k=K)
    fs.fit(X_h, y_h.values)
    return fs.transform(X_h), fs.transform(X_n), fs.transform(X_t)


def evaluate_fs(method, X_h, y_h, X_n, y_n, X_t, y_t, logger):
    Xh2, Xn2, Xt2 = apply_fs(method, X_h, y_h, X_n, y_n, X_t)
    yt = np.asarray(y_t)
    sm = ImbalanceSampler()
    op = ModelPool(pool_name="old")
    op.create_pool(Xh2, y_h.values, prefix="old")
    np_ = ModelPool(pool_name="new")
    np_.create_pool(Xn2, y_n.values, prefix="new")
    ap = {**op.predict_proba(Xt2), **np_.predict_proba(Xt2)}
    results = {}
    for nm, keys in [
        ("ensemble_old_3", ["old_under", "old_over", "old_hybrid"]),
        ("ensemble_new_3", ["new_under", "new_over", "new_hybrid"]),
        ("ensemble_all_6", list(ap.keys())),
    ]:
        results[nm] = compute_metrics(yt, np.mean([ap[k] for k in keys], axis=0))
    Xm = pd.concat([Xh2, Xn2])
    ym = pd.concat([y_h, y_n])
    Xr, yr = sm.apply_sampling(Xm, ym.values, strategy="hybrid")
    mf = LightGBMWrapper(name="finetune_hybrid")
    mf.fit(*sm.apply_sampling(Xh2, y_h.values, strategy="hybrid"))
    mf.fit(*sm.apply_sampling(Xn2, y_n.values, strategy="hybrid"))
    results["finetune_hybrid"] = compute_metrics(yt, mf.predict_proba(Xt2))
    df = pd.DataFrame(results).T
    df["fs_method"] = method
    n_feat = Xh2.shape[1]
    df["n_features"] = n_feat
    logger.info("  [%s] features=%d", method, n_feat)
    for idx, r in df.iterrows():
        logger.info("    %-20s AUC=%.4f  F1=%.4f", idx, r["AUC"], r["F1"])
    return df


def main():
    logger = get_logger("FS_Advanced", console=True, file=True)
    logger.info("=" * 70)
    logger.info("Experiment 24: Advanced Feature Selection Sweep")
    logger.info("=" * 70)

    set_seed(42)
    splits = {
        "bankruptcy": get_bankruptcy_splits(logger, split_mode="chronological", dataset="us_1999_2018"),
        "stock": get_stock_splits(logger),
        "medical": get_medical_splits(logger),
    }

    parts = []
    for ds_name, (X_h, y_h, X_n, y_n, X_t, y_t) in splits.items():
        logger.info("\\n>>> Dataset: %s", ds_name)
        for method in METHODS:
            logger.info("  FS method: %s", method)
            try:
                df = evaluate_fs(method, X_h, y_h, X_n, y_n, X_t, y_t, logger)
                df["dataset"] = ds_name
                parts.append(df)
            except Exception as exc:
                logger.warning("  FAILED (%s): %s", method, exc)

    combined = pd.concat(parts)
    out = project_root / "results" / "feature_study"
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "fs_advanced_comparison.csv")
    logger.info("Saved to results/feature_study/fs_advanced_comparison.csv")
    return combined


if __name__ == "__main__":
    main()
'''

# ─────────────────────────────── EXP 25 ────────────────────────────────
EXP25 = '''\
"""Experiment 25: Base learner comparison – LightGBM vs XGBoost vs RandomForest."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper, XGBoostWrapper, RandomForestWrapper, ModelPool
from src.evaluation import compute_metrics
from experiments.common_bankruptcy import get_bankruptcy_splits
from experiments.common_dataset import get_stock_splits, get_medical_splits

SAMPLING_STRATEGIES = ["undersampling", "oversampling", "hybrid"]
LEARNERS = {
    "LightGBM": LightGBMWrapper,
    "XGBoost": XGBoostWrapper,
    "RandomForest": RandomForestWrapper,
}


def build_manual_pool(model_class, X_train, y_train, prefix, sm):
    models = {}
    for strat in SAMPLING_STRATEGIES:
        Xr, yr = sm.apply_sampling(X_train, y_train, strategy=strat)
        short = strat[:2] if strat == "oversampling" else strat.split("s")[0]
        names = {"undersampling": "under", "oversampling": "over", "hybrid": "hybrid"}
        key = prefix + "_" + names[strat]
        m = model_class(name=key)
        m.fit(Xr, yr)
        models[key] = m
    return models


def evaluate_learner(learner_name, model_class, X_h, y_h, X_n, y_n, X_t, y_t, logger):
    sm = ImbalanceSampler()
    yt = np.asarray(y_t)
    old_models = build_manual_pool(model_class, X_h, y_h.values, "old", sm)
    new_models = build_manual_pool(model_class, X_n, y_n.values, "new", sm)
    ap = {}
    for key, m in {**old_models, **new_models}.items():
        ap[key] = m.predict_proba(X_t)

    results = {}
    for nm, keys in [
        ("ensemble_old_3", ["old_under", "old_over", "old_hybrid"]),
        ("ensemble_new_3", ["new_under", "new_over", "new_hybrid"]),
        ("ensemble_all_6", list(ap.keys())),
    ]:
        results[nm] = compute_metrics(yt, np.mean([ap[k] for k in keys], axis=0))

    Xhr, yhr = sm.apply_sampling(X_h, y_h.values, strategy="hybrid")
    mf = model_class(name="finetune_hybrid")
    mf.fit(Xhr, yhr)
    Xnr, ynr = sm.apply_sampling(X_n, y_n.values, strategy="hybrid")
    mf.fit(Xnr, ynr)
    results["finetune_hybrid"] = compute_metrics(yt, mf.predict_proba(X_t))

    df = pd.DataFrame(results).T
    df["learner"] = learner_name
    for idx, r in df.iterrows():
        logger.info("    %-20s AUC=%.4f  F1=%.4f  Recall=%.4f", idx, r["AUC"], r["F1"], r["Recall"])
    return df


def main():
    logger = get_logger("BaseLearner", console=True, file=True)
    logger.info("=" * 70)
    logger.info("Experiment 25: Base Learner Comparison")
    logger.info("=" * 70)

    set_seed(42)
    splits = {
        "bankruptcy": get_bankruptcy_splits(logger, split_mode="chronological", dataset="us_1999_2018"),
        "stock": get_stock_splits(logger),
        "medical": get_medical_splits(logger),
    }

    parts = []
    for ds_name, (X_h, y_h, X_n, y_n, X_t, y_t) in splits.items():
        logger.info("\\n>>> Dataset: %s", ds_name)
        for learner_name, model_class in LEARNERS.items():
            logger.info("  Learner: %s", learner_name)
            try:
                df = evaluate_learner(learner_name, model_class, X_h, y_h, X_n, y_n, X_t, y_t, logger)
                df["dataset"] = ds_name
                parts.append(df)
            except Exception as exc:
                logger.warning("  FAILED (%s): %s", learner_name, exc)

    combined = pd.concat(parts)
    out = project_root / "results" / "base_learner"
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "base_learner_comparison.csv")
    logger.info("Saved to results/base_learner/base_learner_comparison.csv")
    return combined


if __name__ == "__main__":
    main()
'''

(EXP / "23_bankruptcy_split_comparison.py").write_text(EXP23, encoding="utf-8")
(EXP / "24_fs_advanced_sweep.py").write_text(EXP24, encoding="utf-8")
(EXP / "25_base_learner_comparison.py").write_text(EXP25, encoding="utf-8")
print("All 3 files written.")
print("  23:", (EXP / "23_bankruptcy_split_comparison.py").stat().st_size, "bytes")
print("  24:", (EXP / "24_fs_advanced_sweep.py").stat().st_size, "bytes")
print("  25:", (EXP / "25_base_learner_comparison.py").stat().st_size, "bytes")
