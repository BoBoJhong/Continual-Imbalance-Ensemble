from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.evaluation import compute_metrics


def _labels_1d(y):
    """與 pandas Series／(n,1) ndarray 相容，統一為 1D 整數標籤。"""
    a = np.asarray(y.values if hasattr(y, "values") else y)
    return np.ravel(a)


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


def run_dcs_from_pool_models(
    pool_models,
    X_hist,
    y_hist,
    X_new,
    y_new,
    X_test,
    y_test,
    k=7,
    method="OLA",
    time_weight_new=1.0,
):
    """
    與 run_dcs 相同演算法，但使用已訓練好的 pool_models（須有 predict / predict_proba）。
    """
    y_test_arr = _labels_1d(y_test)
    X_dsel = pd.concat([X_hist, X_new], axis=0).reset_index(drop=True)
    y_dsel = np.concatenate([_labels_1d(y_hist), _labels_1d(y_new)])
    n_hist = len(X_hist)
    X_dsel_arr = np.asarray(X_dsel, dtype=np.float64)
    X_test_arr = np.asarray(X_test, dtype=np.float64)
    n_pool = len(pool_models)
    n_dsel = len(X_dsel_arr)
    n_test = X_test_arr.shape[0]
    dsel_preds = np.zeros((n_dsel, n_pool), dtype=np.int64)
    for i, m in enumerate(pool_models):
        dsel_preds[:, i] = m.predict(X_dsel_arr)
    test_proba = np.zeros((n_test, n_pool))
    test_preds = np.zeros((n_test, n_pool), dtype=np.int64)
    for i, m in enumerate(pool_models):
        p = m.predict_proba(X_test_arr)
        test_proba[:, i] = p[:, 1] if p.shape[1] == 2 else p.ravel()
        test_preds[:, i] = m.predict(X_test_arr)
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
        else:
            scores = np.zeros(n_pool)
            c_vec  = test_preds[j]
            for ci in range(n_pool):
                c_i  = c_vec[ci]
                mask = nbr_pred[:, ci] == c_i
                if mask.sum() == 0:
                    scores[ci] = np.dot(nbr_w, (nbr_pred[:, ci] == nbr_y).astype(float))
                else:
                    w_sub = nbr_w[mask] / nbr_w[mask].sum()
                    scores[ci] = np.dot(w_sub, (nbr_y[mask] == c_i).astype(float))
        best_mask = scores == scores.max()
        y_proba_pos[j] = test_proba[j, best_mask].mean()
    return compute_metrics(y_test_arr, y_proba_pos)


def run_dcs(X_hist, y_hist, X_new, y_new, X_test, y_test, logger,
            k=7, method="OLA", time_weight_new=1.0):
    pool_models = _build_pool(X_hist, y_hist, X_new, y_new)
    return run_dcs_from_pool_models(
        pool_models,
        X_hist,
        y_hist,
        X_new,
        y_new,
        X_test,
        y_test,
        k=k,
        method=method,
        time_weight_new=time_weight_new,
    )


def run_dcs_all_variants(X_hist, y_hist, X_new, y_new, X_test, y_test, logger, k=7):
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
        logger.info(f"  {name}: AUC={m['AUC']:.4f}  F1={m['F1']:.4f}  Recall={m['Recall']:.4f}  Type1={m['Type1_Error']:.4f}")
    return results


def run_dcs_all_variants_from_pool(
    pool_models,
    X_hist,
    y_hist,
    X_new,
    y_new,
    X_test,
    y_test,
    logger,
    k=7,
):
    """run_dcs_all_variants 的 pool 版本（例如 XGB ModelPool）。"""
    variants = {
        "DCS_OLA": dict(method="OLA", time_weight_new=1.0),
        "DCS_LCA": dict(method="LCA", time_weight_new=1.0),
        "DCS_OLA_TW": dict(method="OLA", time_weight_new=3.0),
        "DCS_LCA_TW": dict(method="LCA", time_weight_new=3.0),
    }
    results = {}
    for name, kwargs in variants.items():
        logger.info(f"Running {name}...")
        m = run_dcs_from_pool_models(
            pool_models,
            X_hist,
            y_hist,
            X_new,
            y_new,
            X_test,
            y_test,
            k=k,
            **kwargs,
        )
        results[name] = m
        logger.info(
            f"  {name}: AUC={m['AUC']:.4f}  F1={m['F1']:.4f}  "
            f"Recall={m['Recall']:.4f}  Type1={m['Type1_Error']:.4f}"
        )
    return results