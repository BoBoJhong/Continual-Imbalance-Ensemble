"""MLP model wrapper (sklearn MLPClassifier)."""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from typing import Optional


class MLPWrapper:
    """
    Wrapper for sklearn MLPClassifier，介面與 XGBoostWrapper 完全一致。

    架構：
      Input → Linear(128) + ReLU → Linear(64) + ReLU → Linear(32) + ReLU → Linear(1) + Sigmoid
    BatchNorm 等效：sklearn MLP 內建無 BN，以 early_stopping 防止過擬合。

    不平衡處理：
      - 訓練前由 ImbalanceSampler 負責（採樣後資料餵入此模型）
      - model 本身不額外設定 class_weight（避免雙重處理）
    """

    def __init__(
        self,
        name: str = "mlp",
        hidden_layer_sizes: tuple = (128, 64, 32),
        activation: str = "relu",
        max_iter: int = 300,
        learning_rate_init: float = 1e-3,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        random_state: int = 42,
        **kwargs,
    ):
        self.name = name
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="adam",
            max_iter=max_iter,
            learning_rate_init=learning_rate_init,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=random_state,
            verbose=False,
            **kwargs,
        )

    def fit(self, X_train, y_train, **kwargs):
        """訓練模型。X_train 可為 DataFrame 或 ndarray。"""
        X = X_train.values if hasattr(X_train, "values") else np.asarray(X_train)
        y = y_train.values if hasattr(y_train, "values") else np.asarray(y_train)
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        X = X.values if hasattr(X, "values") else np.asarray(X)
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """回傳正類機率（shape: (n,)），與 XGBoostWrapper 介面一致。"""
        X = X.values if hasattr(X, "values") else np.asarray(X)
        return self.model.predict_proba(X)[:, 1]
