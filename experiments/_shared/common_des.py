"""Shared KNORA-E-like dynamic ensemble selection helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class _SklearnCompatWrapper:
    """Expose a consistent two-column probability interface."""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
        return self.model.predict(X)

    def predict_proba(self, X):
        X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
        p1 = np.asarray(self.model.predict_proba(X)).ravel()
        return np.column_stack([1.0 - p1, p1])


def run_des(
    X_hist,
    y_hist,
    X_new,
    y_new,
    X_test,
    y_test,
    logger,
    k: int = 7,
    random_state: int = 42,
):
    """Train old/new pools and evaluate local-oracle ensemble selection.

    The DSEL set is the concatenation of the two training periods. This helper
    remains for legacy multi-seed experiments; rolling experiments provide the
    stricter out-of-time evaluation protocol.
    """
    from src.evaluation import compute_metrics
    from src.models import ModelPool

    del logger  # Retained in the signature for existing callers.
    y_test = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    old_pool = ModelPool(pool_name="old", random_state=random_state)
    old_pool.create_pool(
        X_hist,
        y_hist.values if hasattr(y_hist, "values") else y_hist,
        prefix="old",
    )
    new_pool = ModelPool(pool_name="new", random_state=random_state)
    new_pool.create_pool(
        X_new,
        y_new.values if hasattr(y_new, "values") else y_new,
        prefix="new",
    )

    pool_models = [
        _SklearnCompatWrapper(info["model"])
        for pool in (old_pool, new_pool)
        for info in pool.models.values()
    ]
    X_dsel = pd.concat([X_hist, X_new], axis=0).reset_index(drop=True)
    y_hist_arr = y_hist.values if hasattr(y_hist, "values") else np.asarray(y_hist)
    y_new_arr = y_new.values if hasattr(y_new, "values") else np.asarray(y_new)
    y_dsel = np.concatenate([y_hist_arr, y_new_arr])
    X_dsel_arr = np.asarray(X_dsel, dtype=np.float64)
    X_test_arr = np.asarray(X_test, dtype=np.float64)

    dsel_preds = np.column_stack([model.predict(X_dsel_arr) for model in pool_models])
    test_proba = np.column_stack([model.predict_proba(X_test_arr)[:, 1] for model in pool_models])
    n_neighbors = min(k, len(X_dsel_arr))
    neighbors = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="minkowski",
        p=2,
    ).fit(X_dsel_arr)
    _, indices = neighbors.kneighbors(X_test_arr)

    y_proba_pos = np.zeros(len(X_test_arr))
    for row_index, neighbor_indices in enumerate(indices):
        neighbor_labels = y_dsel[neighbor_indices]
        neighbor_predictions = dsel_preds[neighbor_indices]
        locally_perfect = (neighbor_predictions == neighbor_labels.reshape(-1, 1)).all(axis=0)
        candidates = test_proba[row_index, locally_perfect]
        y_proba_pos[row_index] = (
            candidates.mean() if candidates.size else test_proba[row_index].mean()
        )

    return compute_metrics(y_test, y_proba_pos)
