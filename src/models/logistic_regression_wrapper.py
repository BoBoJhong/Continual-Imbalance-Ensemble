"""Logistic Regression wrapper for imbalance-aware baseline comparisons."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from ..utils import get_logger


class LogisticRegressionWrapper:
    """Wrapper for LogisticRegression with class_weight support."""

    def __init__(
        self,
        name: str = "logistic_regression",
        random_state: int = 42,
        class_weight: str = "balanced",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        **kwargs,
    ):
        self.name = name
        self.random_state = random_state
        self.logger = get_logger(f"LR-{name}", console=True, file=False)

        params = dict(
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            warm_start=False,
        )
        params.update(kwargs)
        self.params = params
        self.model = LogisticRegression(**params)
        self.logger.info(f"Initialized LogisticRegression with params: {params}")

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, **fit_params):
        continue_training = bool(fit_params.pop("continue_training", False))
        self.model.set_params(warm_start=continue_training)

        self.logger.info(f"Training LogisticRegression on {len(X_train)} samples")
        self.model.fit(X_train, y_train)
        self.logger.info("Training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba.ravel()
