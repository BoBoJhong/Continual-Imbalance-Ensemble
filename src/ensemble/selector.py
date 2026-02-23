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
