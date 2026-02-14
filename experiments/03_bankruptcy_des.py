"""
實驗 3: Bankruptcy 資料集 - DES (Dynamic Ensemble Selection)
使用 deslib 的 KNORA-E，與靜態 ensemble 比較。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import set_seed, get_logger
from src.data import DataPreprocessor, DataSplitter, ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool
from experiments.common_bankruptcy import get_bankruptcy_splits

# 切割模式：block_cv = 5-fold（1+2 歷史、3+4 新營運、5 測試）；random = 60-20-20 隨機
SPLIT_MODE = "block_cv"


class _SklearnCompatWrapper:
    """讓我們的 model 符合 sklearn/deslib 介面：X 用 array，predict_proba 回 (n, 2)。"""
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
        return self.model.predict(X)

    def predict_proba(self, X):
        X = np.asarray(X) if not isinstance(X, pd.DataFrame) else X
        if hasattr(X, "shape") and len(X.shape) == 2 and X.shape[1] == 0:
            X = np.atleast_2d(X)
        p1 = np.asarray(self.model.predict_proba(X)).ravel()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def main():
    logger = get_logger("Bankruptcy_DES", console=True, file=True)
    set_seed(42)

    logger.info("=" * 80)
    logger.info("實驗 3: Bankruptcy DES (KNORA-E)")
    logger.info("=" * 80)

    X_hist, y_hist, X_new, y_new, X_test, y_test = get_bankruptcy_splits(logger, split_mode=SPLIT_MODE)
    y_test = np.asarray(y_test)

    # ========== 建立 Old + New 模型池（與實驗 2 相同）==========
    logger.info("\n步驟 4: 建立 Old 與 New 模型池")
    old_pool = ModelPool(pool_name="old")
    old_pool.create_pool(X_hist, y_hist.values, prefix="old")
    new_pool = ModelPool(pool_name="new")
    new_pool.create_pool(X_new, y_new.values, prefix="new")

    # 合併為 6 個 model，並包成 deslib 可用的 list（sklearn 介面）
    pool_models = []
    for name, info in old_pool.models.items():
        pool_models.append(_SklearnCompatWrapper(info["model"]))
    for name, info in new_pool.models.items():
        pool_models.append(_SklearnCompatWrapper(info["model"]))

    # ========== DSEL：Historical + New 合併，用於估計 competence ==========
    X_dsel = pd.concat([X_hist, X_new], axis=0).reset_index(drop=True)
    y_dsel = np.concatenate([y_hist.values, y_new.values])
    X_dsel_arr = np.asarray(X_dsel, dtype=np.float64)
    X_test_arr = np.asarray(X_test, dtype=np.float64)

    # ========== 精簡版 KNORA-E：依樣本動態選「區域內全對」的模型再軟投票 ==========
    logger.info("步驟 5: KNORA-E 風格 DES (k=7, soft voting)")
    from sklearn.neighbors import NearestNeighbors
    k = 7
    n_test = X_test_arr.shape[0]
    n_pool = len(pool_models)

    # 各模型在 DSEL 上的預測
    dsel_preds = np.zeros((len(X_dsel_arr), n_pool), dtype=np.int64)
    for i, w in enumerate(pool_models):
        dsel_preds[:, i] = w.predict(X_dsel_arr)

    # 各模型在 test 上的機率 (取正類)
    test_proba = np.zeros((n_test, n_pool))
    for i, w in enumerate(pool_models):
        p = w.predict_proba(X_test_arr)
        test_proba[:, i] = p[:, 1] if p.shape[1] == 2 else p.ravel()

    nn = NearestNeighbors(n_neighbors=k, metric="minkowski", p=2).fit(X_dsel_arr)
    dist, idx = nn.kneighbors(X_test_arr)

    y_pred = np.zeros(n_test, dtype=np.int64)
    y_proba_pos = np.zeros(n_test)
    for j in range(n_test):
        neighbors_y = y_dsel[idx[j]]
        neighbors_pred = dsel_preds[idx[j]]  # (k, n_pool)
        # 哪些模型在 k 個鄰居上全對（local oracle）
        correct = (neighbors_pred == neighbors_y.reshape(-1, 1)).all(axis=0)
        if correct.any():
            sel_proba = test_proba[j, correct]
            y_proba_pos[j] = sel_proba.mean()
        else:
            y_proba_pos[j] = test_proba[j].mean()
        y_pred[j] = 1 if y_proba_pos[j] >= 0.5 else 0

    logger.info("步驟 6: 評估 DES 結果")
    results = {
        "DES_KNORAE": {
            "AUC": roc_auc_score(y_test, y_proba_pos),
            "F1": f1_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
        }
    }
    logger.info(f"DES (KNORA-E) Results: {results['DES_KNORAE']}")

    # ========== 儲存 ==========
    results_df = pd.DataFrame(results).T
    output_dir = project_root / "results/des"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "bankruptcy_des_results.csv"
    results_df.to_csv(output_file)
    logger.info(f"\n結果已保存到: {output_file}")
    logger.info("=" * 80)
    return results_df


if __name__ == "__main__":
    main()
    print("\n實驗 3 完成！結果在 results/des/bankruptcy_des_results.csv")
