"""Experiment 25: Base learner comparison – LightGBM vs XGBoost vs RandomForest."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import ImbalanceSampler
from src.models import LightGBMWrapper, XGBoostWrapper, RandomForestWrapper, ModelPool
from src.evaluation import compute_metrics
from experiments._shared.common_bankruptcy import get_bankruptcy_splits
from experiments._shared.common_dataset import get_stock_splits, get_medical_splits

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
        logger.info("\n>>> Dataset: %s", ds_name)
        for learner_name, model_class in LEARNERS.items():
            logger.info("  Learner: %s", learner_name)
            try:
                df = evaluate_learner(learner_name, model_class, X_h, y_h, X_n, y_n, X_t, y_t, logger)
                df["dataset"] = ds_name
                parts.append(df)
            except Exception as exc:
                logger.warning("  FAILED (%s): %s", learner_name, exc)

    combined = pd.concat(parts)
    out = project_root / "results" / "phase4_analysis"
    out.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out / "base_learner_comparison.csv")
    logger.info("Saved to results/phase4_analysis/base_learner_comparison.csv")
    return combined


if __name__ == "__main__":
    main()
