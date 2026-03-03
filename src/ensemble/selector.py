"""
src/ensemble/selector.py

Dynamic Ensemble Selection (DES) 與靜態集成模組。
抽象化 experiments/03_bankruptcy_des.py 的 KNORA-E 邏輯，使其可重用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from itertools import combinations

from sklearn.neighbors import NearestNeighbors


class DynamicEnsembleSelector:
    """
    KNORA-E 風格的 Dynamic Ensemble Selection。

    每個測試樣本在 DSEL 集中找 k 個最近鄰居，
    選出在這些鄰居上「全對」的模型做 soft voting。
    若無模型全對，退回所有模型的平均。

    Args:
        k: 鄰居數，預設 7
        metric: 距離指標，預設 'minkowski'

    Example:
        des = DynamicEnsembleSelector(k=7)
        des.fit(pool_models, X_dsel, y_dsel)
        y_proba, y_pred = des.predict(X_test)
    """

    def __init__(self, k: int = 7, metric: str = "minkowski"):
        self.k = k
        self.metric = metric
        self._nn = None
        self._pool_models = None
        self._dsel_preds = None
        self._y_dsel = None

    def fit(self, pool_models: list, X_dsel, y_dsel) -> "DynamicEnsembleSelector":
        """
        在 DSEL 資料集上 fit。

        Args:
            pool_models: 模型列表，各模型需有 predict(X) 與 predict_proba(X) 方法，
                         predict_proba 需回傳 shape=(n, 2) 的矩陣。
            X_dsel:      DSEL 特徵（Historical + New Operating 合併）
            y_dsel:      DSEL 標籤（0/1）

        Returns:
            self
        """
        self._pool_models = pool_models
        X_arr = np.asarray(X_dsel, dtype=np.float64)
        self._y_dsel = np.asarray(y_dsel)

        # 各模型在 DSEL 上的預測
        n_pool = len(pool_models)
        self._dsel_preds = np.zeros((len(X_arr), n_pool), dtype=np.int64)
        for i, model in enumerate(pool_models):
            self._dsel_preds[:, i] = model.predict(X_arr)

        # 建立 kNN 索引
        self._nn = NearestNeighbors(
            n_neighbors=self.k, metric=self.metric, p=2
        ).fit(X_arr)
        return self

    def predict(self, X_test) -> Tuple[np.ndarray, np.ndarray]:
        """
        KNORA-E 動態選擇並預測。

        Args:
            X_test: 測試集特徵

        Returns:
            (y_proba_pos, y_pred):
                y_proba_pos: 正類機率 (n_test,)
                y_pred:      預測標籤 (n_test,)
        """
        if self._nn is None:
            raise RuntimeError("請先呼叫 fit() 再使用 predict()。")

        X_arr = np.asarray(X_test, dtype=np.float64)
        n_test = X_arr.shape[0]
        n_pool = len(self._pool_models)

        # 各模型在 test 上的正類機率
        test_proba = np.zeros((n_test, n_pool))
        for i, model in enumerate(self._pool_models):
            p = model.predict_proba(X_arr)
            test_proba[:, i] = p[:, 1] if p.shape[1] == 2 else p.ravel()

        # kNN 查詢
        _, idx = self._nn.kneighbors(X_arr)

        y_proba_pos = np.zeros(n_test)
        for j in range(n_test):
            neighbors_y = self._y_dsel[idx[j]]
            neighbors_pred = self._dsel_preds[idx[j]]  # (k, n_pool)
            # 哪些模型在 k 個鄰居上全對
            correct = (neighbors_pred == neighbors_y.reshape(-1, 1)).all(axis=0)
            if correct.any():
                y_proba_pos[j] = test_proba[j, correct].mean()
            else:
                y_proba_pos[j] = test_proba[j].mean()

        y_pred = (y_proba_pos >= 0.5).astype(int)
        return y_proba_pos, y_pred


class EnsembleCombiner:
    """
    靜態集成組合器，管理 2~6 個基分類器的平均 soft voting。

    研究方向定義的組合（見 .agent/rules.md）：
      - 2 models: 一 Old + 一 New
      - 3 models: 2 Old + 1 New (type_a) 或 1 Old + 2 New (type_b)
      - 4/5/6 models: 任意組合

    Args:
        old_proba_dict: {'old_under': ndarray, 'old_over': ..., 'old_hybrid': ...}
        new_proba_dict: {'new_under': ndarray, 'new_over': ..., 'new_hybrid': ...}

    Example:
        combiner = EnsembleCombiner(old_proba, new_proba)
        results = combiner.run_all_combinations(y_test, evaluate_fn)
    """

    OLD_KEYS = ["old_under", "old_over", "old_hybrid"]
    NEW_KEYS = ["new_under", "new_over", "new_hybrid"]

    def __init__(
        self,
        old_proba_dict: Dict[str, np.ndarray],
        new_proba_dict: Dict[str, np.ndarray],
    ):
        self.old = old_proba_dict
        self.new = new_proba_dict
        self._all = {**old_proba_dict, **new_proba_dict}

    def _avg(self, keys: List[str]) -> np.ndarray:
        return np.mean([self._all[k] for k in keys], axis=0)

    def get_predefined_combinations(self) -> Dict[str, np.ndarray]:
        """
        回傳研究設計中所有預定義的集成組合（平均 soft voting）。

        Returns:
            {combo_name: y_proba_avg}
        """
        combos: Dict[str, np.ndarray] = {}

        # 2 models: 所有 (1 old, 1 new) 對
        for o in self.OLD_KEYS:
            for n in self.NEW_KEYS:
                name = f"ensemble_2_{o.split('_')[1]}_{n.split('_')[1]}"
                combos[name] = self._avg([o, n])

        # 3 models type_a: 2 Old + 1 New
        for n in self.NEW_KEYS:
            name = f"ensemble_3a_2old_{n.split('_')[1]}"
            combos[name] = self._avg(self.OLD_KEYS + [n])

        # 3 models type_b: 1 Old + 2 New
        for o in self.OLD_KEYS:
            name = f"ensemble_3b_{o.split('_')[1]}_2new"
            combos[name] = self._avg([o] + self.NEW_KEYS)

        # 4 models: C(6,4) = 15 組
        all_keys = self.OLD_KEYS + self.NEW_KEYS
        for keys in combinations(all_keys, 4):
            name = "ensemble_4_" + "_".join(k.split("_")[1] for k in keys)
            combos[name] = self._avg(list(keys))

        # 5 models: C(6,5) = 6 組
        for keys in combinations(all_keys, 5):
            name = "ensemble_5_" + "_".join(k.split("_")[1] for k in keys)
            combos[name] = self._avg(list(keys))

        # 6 models (all)
        combos["ensemble_6_all"] = self._avg(all_keys)

        return combos

    def run_all_combinations(self, y_test, evaluate_fn) -> Dict[str, dict]:
        """
        執行所有預定義集成組合並評估。

        Args:
            y_test:      真實標籤
            evaluate_fn: callable(y_true, y_proba) -> dict of metrics

        Returns:
            {combo_name: {metric_name: value}}
        """
        combos = self.get_predefined_combinations()
        results = {}
        for name, y_proba in combos.items():
            results[name] = evaluate_fn(y_test, y_proba)
        return results


class DynamicClassifierSelector:
    """
    Dynamic Classifier Selection (DCS)。

    與 DES（選出「一群」模型做集成投票）不同，DCS 對每個測試樣本
    只選出「單一最佳分類器」進行預測。

    本類別實作兩種主流 DCS 方法：
      - OLA (Overall Local Accuracy)：
          選在 K 個最近鄰上整體正確率最高的分類器。
          直覺：找在該局部區域最準的單一模型。
      - LCA (Local Class Accuracy)：
          只計算分類器在鄰域中「與其預測類別相同的鄰居」上的正確率。
          直覺：對每個分類器，只看它敢出手的樣本，有沒有真的出手正確。

    若多個分類器同分，則對同分者做軟投票平均（退化為集成）。

    Args:
        k      : 鄰居數，預設 7
        method : 'OLA'（預設）或 'LCA'
        metric : 距離指標，預設 'minkowski'

    Example:
        dcs = DynamicClassifierSelector(k=7, method='OLA')
        dcs.fit(pool_models, X_dsel, y_dsel)
        y_proba, y_pred = dcs.predict(X_test)
    """

    SUPPORTED_METHODS = ("OLA", "LCA")

    def __init__(
        self,
        k: int = 7,
        method: str = "OLA",
        metric: str = "minkowski",
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"method 必須為 {self.SUPPORTED_METHODS}，收到 '{method}'")
        self.k = k
        self.method = method
        self.metric = metric

        self._nn = None
        self._pool_models = None
        self._dsel_preds: Optional[np.ndarray] = None
        self._y_dsel: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(
        self, pool_models: list, X_dsel, y_dsel
    ) -> "DynamicClassifierSelector":
        """
        在 DSEL 資料集上建立 kNN 索引並記錄各模型的預測。

        Args:
            pool_models : 模型列表，需有 predict(X) 與 predict_proba(X) 方法
            X_dsel      : DSEL 特徵（Historical + New Operating 合併）
            y_dsel      : DSEL 標籤（0/1）
        Returns:
            self
        """
        self._pool_models = pool_models
        X_arr = np.asarray(X_dsel, dtype=np.float64)
        self._y_dsel = np.asarray(y_dsel)

        n_pool = len(pool_models)
        self._dsel_preds = np.zeros((len(X_arr), n_pool), dtype=np.int64)
        for i, model in enumerate(pool_models):
            self._dsel_preds[:, i] = model.predict(X_arr)

        self._nn = NearestNeighbors(
            n_neighbors=self.k, metric=self.metric, p=2
        ).fit(X_arr)
        return self

    # ------------------------------------------------------------------
    def predict(self, X_test) -> Tuple[np.ndarray, np.ndarray]:
        """
        DCS 預測。

        Returns:
            (y_proba_pos, y_pred):
                y_proba_pos: 正類機率 (n_test,)
                y_pred     : 預測標籤 (n_test,)
        """
        if self._nn is None:
            raise RuntimeError("請先呼叫 fit() 再使用 predict()。")

        X_arr = np.asarray(X_test, dtype=np.float64)
        n_test = X_arr.shape[0]
        n_pool = len(self._pool_models)

        # 各模型在 test 上的正類機率
        test_proba = np.zeros((n_test, n_pool))
        for i, model in enumerate(self._pool_models):
            p = model.predict_proba(X_arr)
            test_proba[:, i] = p[:, 1] if p.shape[1] == 2 else p.ravel()

        # 各模型在 test 上的預測類別（供 LCA 使用）
        test_preds = np.zeros((n_test, n_pool), dtype=np.int64)
        for i, model in enumerate(self._pool_models):
            test_preds[:, i] = model.predict(X_arr)

        # kNN 查詢
        _, idx = self._nn.kneighbors(X_arr)

        y_proba_pos = np.zeros(n_test)
        for j in range(n_test):
            neighbors_y    = self._y_dsel[idx[j]]           # (k,)
            neighbors_pred = self._dsel_preds[idx[j]]        # (k, n_pool)

            if self.method == "OLA":
                scores = self._ola_scores(neighbors_y, neighbors_pred, n_pool)
            else:  # LCA
                scores = self._lca_scores(
                    neighbors_y, neighbors_pred, n_pool, test_preds[j]
                )

            # 選分數最高的模型（若同分則平均）
            best_mask = scores == scores.max()
            y_proba_pos[j] = test_proba[j, best_mask].mean()

        y_pred = (y_proba_pos >= 0.5).astype(int)
        return y_proba_pos, y_pred

    # ------------------------------------------------------------------
    @staticmethod
    def _ola_scores(
        neighbors_y: np.ndarray,
        neighbors_pred: np.ndarray,
        n_pool: int,
    ) -> np.ndarray:
        """
        OLA：每個模型在 k 個鄰居上的整體正確率。

        scores[i] = Σ_k 1[pred_i(x_k) == y_k] / k
        """
        correct = (neighbors_pred == neighbors_y.reshape(-1, 1))  # (k, n_pool)
        return correct.mean(axis=0)  # (n_pool,)

    @staticmethod
    def _lca_scores(
        neighbors_y: np.ndarray,
        neighbors_pred: np.ndarray,
        n_pool: int,
        test_pred_i: np.ndarray,
    ) -> np.ndarray:
        """
        LCA：每個模型只在它「預測為某類別」的鄰居子集上計算正確率。

        對模型 i，預測類別 c_i = test_pred_i[i]：
          - 找出模型 i 在 k 個鄰居中預測為 c_i 的樣本集合 S_i
          - scores[i] = 在 S_i 中真實標籤也是 c_i 的比例
          - 若 S_i 為空（模型在鄰域都不預測此類），退回 OLA 分數
        """
        scores = np.zeros(n_pool)
        for i in range(n_pool):
            c_i = test_pred_i[i]
            # S_i：模型 i 在鄰居中預測為 c_i 的樣本
            mask_pred_c = neighbors_pred[:, i] == c_i  # (k,)
            if mask_pred_c.sum() == 0:
                # 退回 OLA
                scores[i] = (neighbors_pred[:, i] == neighbors_y).mean()
            else:
                # 在 S_i 中，真實標籤也是 c_i 的比例
                scores[i] = (neighbors_y[mask_pred_c] == c_i).mean()
        return scores
