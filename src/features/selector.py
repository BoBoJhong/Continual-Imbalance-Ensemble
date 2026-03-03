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
)
from sklearn.linear_model import LassoCV
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
        method:         'kbest_f' | 'kbest_chi2' | 'lasso' | 'mutual_info' | 'shap'
        k:              保留的特徵數（lasso 以重要性自動決定，shap/mi 手動給 k）
        selected_cols_: fit 後選出的欄位名稱清單
    """

    SUPPORTED_METHODS = ("kbest_f", "kbest_chi2", "lasso", "mutual_info", "shap")

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

        # 記錄選出的欄位名稱
        if self.method in ("mutual_info",):
            self.selected_cols_ = list(np.array(X.columns)[self._mi_top_idx])
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
        elif self.method in ("mutual_info", "shap"):
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
