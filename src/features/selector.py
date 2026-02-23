"""
src/features/selector.py

特徵選擇模組。
支援：SelectKBest (f_classif / chi2)、LASSO (SelectFromModel)。
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
)
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler


class FeatureSelector:
    """
    特徵選擇器，符合 sklearn 的 fit/transform 介面。

    使用方式（fit 於 Historical Data，transform 全部切割段）：
        selector = FeatureSelector(method='kbest', k=50)
        X_hist_fs = selector.fit_transform(X_hist, y_hist)
        X_new_fs  = selector.transform(X_new)
        X_test_fs = selector.transform(X_test)

    Attributes:
        method:         'kbest_f' | 'kbest_chi2' | 'lasso'
        k:              SelectKBest 保留的特徵數（method='kbest_*' 時有效）
        selected_cols_: fit 後選出的欄位名稱清單
    """

    SUPPORTED_METHODS = ("kbest_f", "kbest_chi2", "lasso")

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

        # 記錄選出的欄位名稱
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
        if self._selector is None:
            raise RuntimeError("請先呼叫 fit() 再使用 transform()。")

        if self.method == "kbest_chi2":
            X_scaled = self._chi2_scaler.transform(X)
            arr = self._selector.transform(X_scaled)
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
