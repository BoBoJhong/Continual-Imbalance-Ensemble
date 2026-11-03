"""Official DESlib algorithms with a sklearn data-validation compatibility bridge.

The bridge only replaces the removed BaseEstimator._validate_data method;
competence, selection, voting and the unanimous-prediction shortcut stay in DESlib.
"""

import numpy as np
import pandas as pd
from deslib.dcs import LCA, OLA
from deslib.des import KNORAE, KNORAU
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data

METHODS = ("KNORA_E", "KNORA_U", "DCS_OLA", "DCS_LCA")


class ProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """Expose an already-fitted project's binary model through sklearn's API."""

    def __init__(self, model, columns):
        self.model = model
        self.columns = columns
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = len(columns)

    def fit(self, X, y=None):
        raise RuntimeError("This adapter wraps a frozen fitted model; refitting is forbidden")

    def predict_proba(self, X):
        frame = pd.DataFrame(np.asarray(X), columns=self.columns)
        positive = np.asarray(self.model.predict_proba(frame), dtype=float)
        if positive.shape != (len(frame),) or not np.isfinite(positive).all():
            raise ValueError("Base model returned invalid binary probabilities")
        if np.any((positive < 0) | (positive > 1)):
            raise ValueError("Base probabilities must be in [0, 1]")
        return np.column_stack([1 - positive, positive])

    def predict(self, X):
        # Argmax resolves probability=0.5 to class 0, as sklearn does.
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class _ValidationBridge:
    def _validate_data(self, *args, **kwargs):
        return validate_data(self, *args, **kwargs)


class StandardKNORAE(_ValidationBridge, KNORAE):
    pass


class StandardKNORAU(_ValidationBridge, KNORAU):
    pass


class StandardOLA(_ValidationBridge, OLA):
    pass


class StandardLCA(_ValidationBridge, LCA):
    pass


def fit_selector(method, pool, X_dsel, y_dsel, *, k=7, seed=42, threads=4):
    """Fit competence on an explicit held-out DSEL, never on pool training rows.

    DES uses soft voting; DCS uses the official deterministic 'best' tie rule.
    No frienemy pruning, instance-hardness routing or time weighting is enabled.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown standard dynamic method: {method}")
    if not isinstance(k, int) or k < 1 or len(X_dsel) < k:
        raise ValueError("k must be positive and no larger than DSEL")
    if set(np.unique(y_dsel)) != {0, 1}:
        raise ValueError("DSEL must contain both binary classes")
    common = dict(pool_classifiers=pool, k=k, DFP=False, with_IH=False,
                  random_state=seed, n_jobs=threads)
    factories = {"KNORA_E": StandardKNORAE, "KNORA_U": StandardKNORAU,
                 "DCS_OLA": StandardOLA, "DCS_LCA": StandardLCA}
    options = {"voting": "soft"} if method.startswith("KNORA") else {
        "selection_method": "best"}
    return factories[method](**common, **options).fit(np.asarray(X_dsel), np.asarray(y_dsel))
