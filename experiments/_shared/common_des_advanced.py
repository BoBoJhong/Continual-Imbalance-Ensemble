"""
進階 DES：時間／漂移感知 + 少數類導向（供後續論文用）。
- 時間加權：DSEL 中「新營運期」樣本權重提高，讓在新資料上正確的模型更容易被選中。
- 少數類加權：DSEL 中少數類（y=1）樣本權重提高，讓在少數類上正確的模型更容易被選中。
- 兩者可同時啟用（combined）。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors

from experiments._shared.common_des import _SklearnCompatWrapper, run_des


def _build_pool_and_dsel(X_hist, y_hist, X_new, y_new, X_test):
    """建立 Old+New 池與 DSEL，回傳 pool_models, X_dsel_arr, y_dsel, dsel_sample_weight, test_proba, n_hist."""
    from src.models import ModelPool

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
    y_hist_arr = np.asarray(y_hist.values if hasattr(y_hist, "values") else y_hist)
    y_new_arr = np.asarray(y_new.values if hasattr(y_new, "values") else y_new)
    y_dsel = np.concatenate([y_hist_arr, y_new_arr])
    n_hist = len(X_hist)
    X_dsel_arr = np.asarray(X_dsel, dtype=np.float64)
    X_test_arr = np.asarray(X_test, dtype=np.float64)

    n_pool = len(pool_models)
    dsel_preds = np.zeros((len(X_dsel_arr), n_pool), dtype=np.int64)
    for i, w in enumerate(pool_models):
        dsel_preds[:, i] = w.predict(X_dsel_arr)
    test_proba = np.zeros((X_test_arr.shape[0], n_pool))
    for i, w in enumerate(pool_models):
        p = w.predict_proba(X_test_arr)
        test_proba[:, i] = p[:, 1] if p.shape[1] == 2 else p.ravel()

    return pool_models, X_dsel_arr, y_dsel, dsel_preds, test_proba, n_hist, X_test_arr


def run_des_advanced(
    X_hist,
    y_hist,
    X_new,
    y_new,
    X_test,
    y_test,
    logger,
    k=7,
    time_weight_new=1.0,
    minority_weight=1.0,
):
    """
    進階 DES：可選時間加權、少數類加權或兩者同時啟用。

    - time_weight_new: 新營運期 DSEL 樣本權重（>1 表示更重視在新資料上正確的模型）
    - minority_weight: 少數類（y=1）DSEL 樣本權重（>1 表示更重視在少數類上正確的模型）
    - 若兩者皆為 1.0，行為等同 run_des（KNORA-E baseline）
    """
    y_test = np.asarray(y_test.values if hasattr(y_test, "values") else y_test)
    (
        pool_models,
        X_dsel_arr,
        y_dsel,
        dsel_preds,
        test_proba,
        n_hist,
        X_test_arr,
    ) = _build_pool_and_dsel(X_hist, y_hist, X_new, y_new, X_test)

    n_test = X_test_arr.shape[0]
    n_pool = len(pool_models)
    # DSEL 樣本權重：來自 new 的乘上 time_weight_new，少數類乘上 minority_weight
    dsel_from_new = np.zeros(len(y_dsel), dtype=bool)
    dsel_from_new[n_hist:] = True
    dsel_minority = y_dsel == 1
    sample_weight = np.ones(len(y_dsel))
    sample_weight[dsel_from_new] *= time_weight_new
    sample_weight[dsel_minority] *= minority_weight

    nn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2).fit(X_dsel_arr)
    _, idx = nn.kneighbors(X_test_arr)

    y_pred = np.zeros(n_test, dtype=np.int64)
    y_proba_pos = np.zeros(n_test)
    for j in range(n_test):
        neighbors_y = y_dsel[idx[j]]
        neighbors_pred = dsel_preds[idx[j]]  # (k, n_pool)
        neighbor_weights = sample_weight[idx[j]]  # (k,)
        # 每個模型在鄰居上的「加權正確數」
        correct_mask = neighbors_pred == neighbors_y.reshape(-1, 1)  # (k, n_pool)
        weighted_correct = np.dot(neighbor_weights, correct_mask.astype(float))  # (n_pool,)
        if np.any(weighted_correct > 0):
            selected = weighted_correct > 0
            # 以加權正確數做軟投票權重
            w_sum = weighted_correct[selected].sum()
            y_proba_pos[j] = (test_proba[j, selected] * weighted_correct[selected]).sum() / w_sum
        else:
            y_proba_pos[j] = test_proba[j].mean()
        y_pred[j] = 1 if y_proba_pos[j] >= 0.5 else 0

    return {
        "AUC": roc_auc_score(y_test, y_proba_pos),
        "F1": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
    }
