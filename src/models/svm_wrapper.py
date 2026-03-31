"""SVM (RBF) wrapper for imbalance-aware baseline comparisons."""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from ..utils import get_logger


class SVMWrapper:
    """
    Wrapper for sklearn SVC (RBF) with probability estimates.
    `predict_proba` requires probability=True (Platt scaling after fit).
    """

    def __init__(
        self,
        name: str = "svm",
        random_state: int = 42,
        class_weight: str = "balanced",
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        probability: bool = True,
        max_iter: int = -1,
        cache_size: int = 500,
        **kwargs,
    ):
        self.name = name
        self.random_state = random_state
        self.logger = get_logger(f"SVM-{name}", console=True, file=False)
        params = dict(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            class_weight=class_weight,
            random_state=random_state,
            max_iter=max_iter,
            cache_size=cache_size,
        )
        params.update(kwargs)
        self.params = params
        self.model = SVC(**params)
        self.logger.info(f"Initialized SVC with params: {params}")

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, **fit_params):
        fit_params.pop("continue_training", None)
        self.logger.info(f"Training SVC on {len(X_train)} samples")
        self.model.fit(X_train, y_train)
        self.logger.info("Training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba.ravel()
