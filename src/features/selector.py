"""
src/features/selector.py

特徵選擇模組。
支援：SelectKBest (f_classif / chi2)、LASSO (SelectFromModel)、
     Mutual Information (mutual_info_classif)、SHAP (TreeExplainer)。
fit 於 Historical Data，transform 用於所有切割段（確保無資料洩漏）。
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, List

from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    chi2,
    SelectFromModel,
    mutual_info_classif,
    RFE,
)
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler


class FeatureSelector:
    """
    特徵選擇器，符合 sklearn 的 fit/transform 介面。

    使用方式（fit 於 Historical Data，transform 全部切割段）：
        selector = FeatureSelector(method='kbest_f', k=50)
        X_hist_fs = selector.fit_transform(X_hist, y_hist)
        X_new_fs  = selector.transform(X_new)
        X_test_fs = selector.transform(X_test)

    Attributes:
        method:         'kbest_f' | 'kbest_chi2' | 'lasso' | 'mutual_info' | 'shap' | 'rfe' | 'cart' | 'ga'
        k:              保留的特徵數（lasso 以重要性自動決定，shap/mi 手動給 k）
        selected_cols_: fit 後選出的欄位名稱清單
    """

    SUPPORTED_METHODS = ("kbest_f", "kbest_chi2", "lasso", "mutual_info", "shap", "rfe", "cart", "ga")

    def __init__(self, method: str = "kbest_f", k: int = 50):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"method 必須為 {self.SUPPORTED_METHODS}，得到 '{method}'"
            )
        self.method = method
        self.k = k
        self._selector = None
        self.selected_cols_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y) -> "FeatureSelector":
        """
        在 Historical Data 上 fit 特徵選擇器。

        Args:
            X: 特徵 DataFrame（已前處理、已 scale）
            y: 標籤（0/1）

        Returns:
            self
        """
        y = np.asarray(y)
        n_features = X.shape[1]
        k = min(self.k, n_features)

        if self.method == "kbest_f":
            self._selector = SelectKBest(score_func=f_classif, k=k)
            self._selector.fit(X, y)

        elif self.method == "kbest_chi2":
            # chi2 需要非負數，先做 MinMaxScaler
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            self._selector = SelectKBest(score_func=chi2, k=k)
            self._selector.fit(X_scaled, y)
            self._chi2_scaler = scaler

        elif self.method == "lasso":
            lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
            lasso.fit(X, y)
            self._selector = SelectFromModel(lasso, prefit=True)

        elif self.method == "mutual_info":
            scores = mutual_info_classif(X, y, random_state=42)
            k = min(self.k, n_features)
            top_idx = np.argsort(scores)[::-1][:k]
            self._mi_scores = scores
            self._mi_top_idx = top_idx
            self._selector = None  # 不用 sklearn selector，自行記錄 idx

        elif self.method == "cart":
            cart = DecisionTreeClassifier(random_state=42, class_weight="balanced")
            cart.fit(X, y)
            importances = np.asarray(cart.feature_importances_, dtype=float)
            k = min(self.k, n_features)
            top_idx = np.argsort(importances)[::-1][:k]
            self._cart_scores = importances
            self._cart_top_idx = top_idx
            self._selector = None

        elif self.method == "ga":
            # GA wrapper：以交叉驗證 AUC 搜尋最佳特徵子集，再取前 k 個基因
            top_idx = self._run_ga_feature_search(X, y, k=min(self.k, n_features))
            self._ga_top_idx = top_idx
            self._selector = None

        elif self.method == "shap":
            try:
                import shap
                from lightgbm import LGBMClassifier
            except ImportError:
                raise ImportError("SHAP 方法需要安裝 shap 與 lightgbm：pip install shap lightgbm")
            _clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1,
                                  is_unbalance=True)
            _clf.fit(X, y)
            explainer = shap.TreeExplainer(_clf)
            shap_values = explainer.shap_values(X)
            # 對二元分類取正類的 SHAP（list[1] 或直接 array）
            if isinstance(shap_values, list):
                sv = np.abs(shap_values[1])
            else:
                sv = np.abs(shap_values)
            mean_abs_shap = sv.mean(axis=0)
            k = min(self.k, n_features)
            top_idx = np.argsort(mean_abs_shap)[::-1][:k]
            self._shap_scores = mean_abs_shap
            self._shap_top_idx = top_idx
            self._selector = None  # 自行記錄 idx

        elif self.method == "rfe":
            est = DecisionTreeClassifier(random_state=42, class_weight="balanced")
            self._selector = RFE(estimator=est, n_features_to_select=k, step=1)
            self._selector.fit(X, y)

        # 記錄選出的欄位名稱
        if self.method in ("mutual_info",):
            self.selected_cols_ = list(np.array(X.columns)[self._mi_top_idx])
        elif self.method == "cart":
            self.selected_cols_ = list(np.array(X.columns)[self._cart_top_idx])
        elif self.method == "ga":
            self.selected_cols_ = list(np.array(X.columns)[self._ga_top_idx])
        elif self.method == "shap":
            self.selected_cols_ = list(np.array(X.columns)[self._shap_top_idx])
        else:
            mask = self._selector.get_support()
            self.selected_cols_ = list(np.array(X.columns)[mask])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        套用已 fit 的特徵選擇，回傳精簡後的 DataFrame。

        Args:
            X: 與 fit 時相同欄位的特徵 DataFrame

        Returns:
            pd.DataFrame，欄位為 selected_cols_
        """
        if self.selected_cols_ is None:
            raise RuntimeError("請先呼叫 fit() 再使用 transform()。")

        if self.method == "kbest_chi2":
            X_scaled = self._chi2_scaler.transform(X)
            arr = self._selector.transform(X_scaled)
        elif self.method in ("mutual_info", "cart", "ga", "shap"):
            arr = X[self.selected_cols_].values
        else:
            arr = self._selector.transform(X)

        return pd.DataFrame(arr, index=X.index, columns=self.selected_cols_)

    def fit_transform(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """Convenience: fit + transform on the same data."""
        return self.fit(X, y).transform(X)

    @property
    def n_selected(self) -> int:
        """選出的特徵數量。"""
        return len(self.selected_cols_) if self.selected_cols_ else 0

    def summary(self) -> str:
        """印出摘要。"""
        if self.selected_cols_ is None:
            return "FeatureSelector: not fitted yet"
        return (
            f"FeatureSelector(method={self.method}, "
            f"selected={self.n_selected} features)"
        )

    def _run_ga_feature_search(self, X: pd.DataFrame, y: np.ndarray, k: int) -> np.ndarray:
        """
        小型 GA wrapper（不依賴額外套件）：
        - 個體: 二元向量（1 表示選特徵）
        - 適應度: 3-fold AUC 平均值（DecisionTreeClassifier）
        """
        rng = np.random.default_rng(42)
        n_features = X.shape[1]
        if n_features == 1:
            return np.array([0], dtype=int)

        pop_size = 18
        n_gen = 10
        elite_size = 4
        mutation_rate = 0.03
        min_keep = max(1, min(k, n_features))
        max_keep = max(min_keep, min(n_features, int(max(k, 2) * 1.5)))

        base_est = DecisionTreeClassifier(random_state=42, class_weight="balanced")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        X_np = np.asarray(X)

        def _repair(mask: np.ndarray) -> np.ndarray:
            m = mask.copy()
            n_on = int(m.sum())
            if n_on < min_keep:
                off_idx = np.where(~m)[0]
                if len(off_idx) > 0:
                    add = rng.choice(off_idx, size=min_keep - n_on, replace=False)
                    m[add] = True
            elif n_on > max_keep:
                on_idx = np.where(m)[0]
                drop = rng.choice(on_idx, size=n_on - max_keep, replace=False)
                m[drop] = False
            return m

        def _fitness(mask: np.ndarray) -> float:
            m = _repair(mask)
            cols = np.where(m)[0]
            if len(cols) == 0:
                return -1.0
            est = clone(base_est)
            scores = cross_val_score(est, X_np[:, cols], y, cv=cv, scoring="roc_auc")
            return float(np.mean(scores))

        population = []
        for _ in range(pop_size):
            mask = rng.random(n_features) < (min_keep / max(n_features, 1))
            population.append(_repair(mask))

        best_mask = population[0]
        best_score = -1.0

        for _ in range(n_gen):
            scores = np.array([_fitness(ind) for ind in population], dtype=float)
            order = np.argsort(scores)[::-1]
            population = [population[i] for i in order]
            scores = scores[order]
            if scores[0] > best_score:
                best_score = float(scores[0])
                best_mask = population[0].copy()

            next_pop = population[:elite_size]
            while len(next_pop) < pop_size:
                p1 = population[int(rng.integers(0, max(2, pop_size // 2)))]
                p2 = population[int(rng.integers(0, max(2, pop_size // 2)))]
                cut = int(rng.integers(1, n_features))
                child = np.concatenate([p1[:cut], p2[cut:]])
                mut = rng.random(n_features) < mutation_rate
                child = np.logical_xor(child, mut)
                next_pop.append(_repair(child))
            population = next_pop

        final_cols = np.where(best_mask)[0]
        if len(final_cols) > k:
            # 若 GA 選超過 k，按 CART 重要度再截斷，保持輸出維度穩定
            cart = DecisionTreeClassifier(random_state=42, class_weight="balanced")
            cart.fit(X_np[:, final_cols], y)
            imp = np.asarray(cart.feature_importances_)
            keep_local_idx = np.argsort(imp)[::-1][:k]
            final_cols = final_cols[keep_local_idx]
        elif len(final_cols) < k:
            # 若不足 k，以 CART 全域重要度補齊
            cart = DecisionTreeClassifier(random_state=42, class_weight="balanced")
            cart.fit(X_np, y)
            imp = np.asarray(cart.feature_importances_)
            for idx in np.argsort(imp)[::-1]:
                if idx not in final_cols:
                    final_cols = np.append(final_cols, idx)
                if len(final_cols) >= k:
                    break

        return np.asarray(final_cols[:k], dtype=int)
