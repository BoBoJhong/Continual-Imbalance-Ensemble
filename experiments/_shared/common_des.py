"""
共用：KNORA-E 風格 DES 邏輯，供 Bankruptcy / Stock / Medical 使用。
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class _SklearnCompatWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
        return self.model.predict(X)

    def predict_proba(self, X):
        X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
        p1 = np.asarray(self.model.predict_proba(X)).ravel()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def run_des(X_hist, y_hist, X_new, y_new, X_test, y_test, logger, k=7):
    """
    建立 Old+New 模型池，執行 KNORA-E 風格 DES。
    回傳指標 dict：AUC, F1, G_Mean, Recall, Precision。
    """
    from src.models import ModelPool
    from src.evaluation import compute_metrics

    y_test = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_hist, y_hist.values if hasattr(y_hist, "values") else y_hist, prefix="old")
    new_pool = ModelPool(pool_name="new")
    new_pool.create_pool(X_new, y_new.values if hasattr(y_new, "values") else y_new, prefix="new")

    pool_models = []
    for _, info in old_pool.models.items():
        pool_models.append(_SklearnCompatWrapper(info["model"]))
    for _, info in new_pool.models.items():
        pool_models.append(_SklearnCompatWrapper(info["model"]))

    X_dsel = pd.concat([X_hist, X_new], axis=0).reset_index(drop=True)
    y_hist_arr = y_hist.values if hasattr(y_hist, "values") else np.asarray(y_hist)
    y_new_arr  = y_new.values  if hasattr(y_new,  "values") else np.asarray(y_new)
    y_dsel = np.concatenate([y_hist_arr, y_new_arr])
    X_dsel_arr = np.asarray(X_dsel, dtype=np.float64)
    X_test_arr = np.asarray(X_test, dtype=np.float64)

    n_test = X_test_arr.shape[0]
    n_pool = len(pool_models)
    dsel_preds = np.zeros((len(X_dsel_arr), n_pool), dtype=np.int64)
    for i, w in enumerate(pool_models):
        dsel_preds[:, i] = w.predict(X_dsel_arr)
    test_proba = np.zeros((n_test, n_pool))
    for i, w in enumerate(pool_models):
        p = w.predict_proba(X_test_arr)
        test_proba[:, i] = p[:, 1] if p.shape[1] == 2 else p.ravel()

    nn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2).fit(X_dsel_arr)
    _, idx = nn.kneighbors(X_test_arr)
    y_proba_pos = np.zeros(n_test)
    for j in range(n_test):
        neighbors_y    = y_dsel[idx[j]]
        neighbors_pred = dsel_preds[idx[j]]
        correct = (neighbors_pred == neighbors_y.reshape(-1, 1)).all(axis=0)
        if correct.any():
            y_proba_pos[j] = test_proba[j, correct].mean()
        else:
            y_proba_pos[j] = test_proba[j].mean()

    # 使用統一的 compute_metrics（含 G-Mean）
    return compute_metrics(y_test, y_proba_pos)
