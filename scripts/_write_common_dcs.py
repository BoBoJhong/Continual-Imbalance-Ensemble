from pathlib import Path

code = r"""from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.evaluation import compute_metrics


class _Wrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
        return self.model.predict(X)
    def predict_proba(self, X):
        X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
        p1 = np.asarray(self.model.predict_proba(X)).ravel()
        return np.column_stack([1.0 - p1, p1])


def _build_pool(X_hist, y_hist, X_new, y_new):
    from src.models import ModelPool
    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_hist, y_hist.values if hasattr(y_hist, "values") else y_hist, prefix="old")
    new_pool = ModelPool(pool_name="new")
    new_pool.create_pool(X_new, y_new.values if hasattr(y_new, "values") else y_new, prefix="new")
    models = []
    for _, info in old_pool.models.items():
        models.append(_Wrapper(info["model"]))
    for _, info in new_pool.models.items():
        models.append(_Wrapper(info["model"]))
    return models


def run_dcs(X_hist, y_hist, X_new, y_new, X_test, y_test, logger,
            k=7, method="OLA", time_weight_new=1.0):
    """
    Dynamic Classifier Selection.

    method='OLA': Overall Local Accuracy
        For each test sample, pick the single classifier with the highest
        (weighted) accuracy on its K nearest neighbours in DSEL.

    method='LCA': Local Class Accuracy
        For each classifier, only consider the neighbours where it predicts
        the same class as it does for the test sample, then measure accuracy
        on that subset. Pick the classifier with the highest LCA score.

    time_weight_new > 1:  neighbours from the 'new' period get higher weight,
        making the selection lean towards models that are accurate on recent data.
    """
    y_test_arr = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    pool_models = _build_pool(X_hist, y_hist, X_new, y_new)

    X_dsel = pd.concat([X_hist, X_new], axis=0).reset_index(drop=True)
    y_dsel = np.concatenate([
        y_hist.values if hasattr(y_hist, "values") else np.asarray(y_hist),
        y_new.values  if hasattr(y_new,  "values") else np.asarray(y_new),
    ])
    n_hist = len(X_hist)
    X_dsel_arr = np.asarray(X_dsel, dtype=np.float64)
    X_test_arr = np.asarray(X_test, dtype=np.float64)
    n_pool = len(pool_models)
    n_dsel = len(X_dsel_arr)
    n_test = X_test_arr.shape[0]

    # Predictions on DSEL
    dsel_preds = np.zeros((n_dsel, n_pool), dtype=np.int64)
    for i, m in enumerate(pool_models):
        dsel_preds[:, i] = m.predict(X_dsel_arr)

    # Probabilities and predictions on test set
    test_proba = np.zeros((n_test, n_pool))
    test_preds = np.zeros((n_test, n_pool), dtype=np.int64)
    for i, m in enumerate(pool_models):
        p = m.predict_proba(X_test_arr)
        test_proba[:, i] = p[:, 1] if p.shape[1] == 2 else p.ravel()
        test_preds[:, i] = m.predict(X_test_arr)

    # Sample weights (time-aware)
    sample_weight = np.ones(n_dsel)
    if time_weight_new != 1.0:
        sample_weight[n_hist:] *= time_weight_new

    nn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2).fit(X_dsel_arr)
    _, idx = nn.kneighbors(X_test_arr)

    y_proba_pos = np.zeros(n_test)
    for j in range(n_test):
        nbr_y    = y_dsel[idx[j]]
        nbr_pred = dsel_preds[idx[j]]
        nbr_w    = sample_weight[idx[j]]
        nbr_w    = nbr_w / nbr_w.sum()

        if method == "OLA":
            correct = (nbr_pred == nbr_y.reshape(-1, 1)).astype(float)
            scores  = np.dot(nbr_w, correct)
        else:  # LCA
            scores = np.zeros(n_pool)
            c_vec  = test_preds[j]
            for i in range(n_pool):
                c_i  = c_vec[i]
                mask = nbr_pred[:, i] == c_i
                if mask.sum() == 0:
                    scores[i] = np.dot(nbr_w, (nbr_pred[:, i] == nbr_y).astype(float))
                else:
                    w_sub = nbr_w[mask] / nbr_w[mask].sum()
                    scores[i] = np.dot(w_sub, (nbr_y[mask] == c_i).astype(float))

        best_mask = scores == scores.max()
        y_proba_pos[j] = test_proba[j, best_mask].mean()

    return compute_metrics(y_test_arr, y_proba_pos)


def run_dcs_all_variants(X_hist, y_hist, X_new, y_new, X_test, y_test, logger, k=7):
    """Run OLA / LCA / OLA_TW / LCA_TW and return {name: metrics}."""
    variants = {
        "DCS_OLA":    dict(method="OLA", time_weight_new=1.0),
        "DCS_LCA":    dict(method="LCA", time_weight_new=1.0),
        "DCS_OLA_TW": dict(method="OLA", time_weight_new=3.0),
        "DCS_LCA_TW": dict(method="LCA", time_weight_new=3.0),
    }
    results = {}
    for name, kwargs in variants.items():
        logger.info(f"Running {name}...")
        m = run_dcs(X_hist, y_hist, X_new, y_new, X_test, y_test, logger, k=k, **kwargs)
        results[name] = m
        logger.info(f"  {name}: AUC={m['AUC']:.4f}  F1={m['F1']:.4f}  Recall={m['Recall']:.4f}  Type1_Err={m['Type1_Error']:.4f}")
    return results
"""

Path("experiments/common_dcs.py").write_text(code, encoding="utf-8")
print("Done. Lines:", code.count("\n"))
