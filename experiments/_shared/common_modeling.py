"""Shared sampling labels, model factories and training-only feature selection.

Callers own the allowed fitting partition. This module preserves the fixed
settings used by the fixed-year and annual FIFO experiments.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier

from src.models import (
    LightGBMWrapper,
    LogisticRegressionWrapper,
    MLPWrapper,
    RandomForestWrapper,
    SVMWrapper,
    XGBoostWrapper,
)

SAMPLER_NAMES = {"none": "None", "under": "TomekLinks", "over": "ADASYN", "hybrid": "SMOTEENN"}
STRATEGIES = {"none": "none", "under": "undersampling", "over": "oversampling", "hybrid": "hybrid"}
SEEDS = (42, 123, 456, 789, 2024, 7, 21, 84, 168, 999)


def make_model(classifier, seed, threads):
    # Sampling is the only class-imbalance intervention in this factorial study.
    factories = {
        "lr": lambda: LogisticRegressionWrapper(random_state=seed, class_weight=None),
        "rf": lambda: RandomForestWrapper(random_state=seed, class_weight=None, n_jobs=threads),
        "xgb": lambda: XGBoostWrapper(use_imbalance=False, random_state=seed, seed=seed, n_jobs=threads),
        "lgb": lambda: LightGBMWrapper(use_imbalance=False, seed=seed, num_threads=threads),
        # Fixed epochs avoid a random early-stopping holdout created after oversampling.
        "mlp": lambda: MLPWrapper(random_state=seed, early_stopping=False),
        "svm": lambda: SVMWrapper(random_state=seed, class_weight=None),
    }
    return factories[classifier]()


def feature_columns(X, y, method, ratio, seed, threads):
    if method == "none":
        return list(X.columns)
    k = max(1, int(X.shape[1] * ratio))
    tree = DecisionTreeClassifier(random_state=seed, class_weight="balanced")
    if method == "rfe":
        selector = RFE(tree, n_features_to_select=k, step=1).fit(X, y)
        return list(X.columns[selector.get_support()])
    if method == "mi":
        scores = mutual_info_classif(X, y, random_state=seed)
    elif method == "cart":
        scores = tree.fit(X, y).feature_importances_
    elif method == "shap":
        from lightgbm import LGBMClassifier
        selector = LGBMClassifier(n_estimators=100, random_state=seed, verbosity=-1,
                                 class_weight="balanced", n_jobs=threads)
        selector.fit(X, y)
        # Exact tree contributions; exclude the bias column. No fallback ranking.
        scores = np.abs(selector.booster_.predict(X, pred_contrib=True)[:, :-1]).mean(axis=0)
    else:
        raise ValueError(f"Unknown FS method: {method}")
    indices = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return list(X.columns[indices])

