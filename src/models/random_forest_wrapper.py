"""src/models/random_forest_wrapper.py - Random Forest wrapper for ensemble experiments."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Optional
from ..utils import get_logger


class RandomForestWrapper:
    """Wrapper for RandomForestClassifier with imbalance handling."""

    def __init__(self, name: str = "random_forest", random_state: int = 42,
                 n_estimators: int = 200, class_weight: str = "balanced", **kwargs):
        self.name = name
        self.random_state = random_state
        self.logger = get_logger(f"RF-{name}", console=True, file=False)
        params = dict(
            n_estimators=n_estimators,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
            max_depth=None,
        )
        params.update(kwargs)
        self.params = params
        self.base_n_estimators = int(n_estimators)
        self.model = RandomForestClassifier(**params)
        self.logger.info(f"Initialized RandomForest with params: {params}")

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, **fit_params):
        continue_training = bool(fit_params.pop("continue_training", False))
        additional_estimators = int(fit_params.pop("additional_estimators", self.base_n_estimators))

        if continue_training:
            new_estimators = int(self.model.n_estimators) + max(1, additional_estimators)
            self.model.set_params(warm_start=True, n_estimators=new_estimators)
        else:
            # 重訓時重置模型，避免前一次 fit 狀態殘留
            reset_params = dict(self.params)
            reset_params["warm_start"] = False
            self.model = RandomForestClassifier(**reset_params)

        self.logger.info(f"Training RandomForest on {len(X_train)} samples")
        self.model.fit(X_train, y_train)
        self.logger.info("Training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability of positive class (shape: (n,))."""
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba.ravel()

    def get_feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_
