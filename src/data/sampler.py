"""Imbalanced data sampling utilities."""

from copy import deepcopy
from typing import Any, Tuple

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler, TomekLinks

from ..utils import get_config_loader, get_logger


class ImbalanceSampler:
    """Handle imbalanced data using various sampling strategies."""

    def __init__(self, random_state: int = 42):
        """Initialize ImbalanceSampler."""
        self.logger = get_logger("ImbalanceSampler", console=True, file=False)
        self.config = get_config_loader()
        self.sampler = None
        self.random_state = random_state

    def _params(self, key_path: str) -> dict[str, Any]:
        """Return a mutable parameter mapping with the run seed applied."""
        params = deepcopy(self.config.get("sampling_config", key_path, default={}) or {})
        if "random_state" in params:
            params["random_state"] = self.random_state
        return params

    def apply_undersampling(
        self, X: pd.DataFrame, y: np.ndarray, method: str = "tomek"
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply under-sampling.

        Args:
            X: Features DataFrame
            y: Target array
            method: Undersampling method ('tomek', 'random')

        Returns:
            Tuple of (resampled X, resampled y)
        """
        self.logger.info(f"Applying under-sampling: {method}")

        original_dist = pd.Series(y).value_counts().to_dict()
        self.logger.info(f"Original distribution: {original_dist}")

        if method == "tomek":
            params = self._params("sampling_strategies.undersampling.params")
            self.sampler = TomekLinks(**params)
        elif method == "random":
            params = self._params("alternative_methods.random_undersampling.params")
            self.sampler = RandomUnderSampler(**params)
        else:
            raise ValueError(f"Unknown undersampling method: {method}")

        X_resampled, y_resampled = self.sampler.fit_resample(X, y)

        new_dist = pd.Series(y_resampled).value_counts().to_dict()
        self.logger.info(f"Resampled distribution: {new_dist}")

        return X_resampled, y_resampled

    def apply_oversampling(
        self, X: pd.DataFrame, y: np.ndarray, method: str = "adasyn"
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply over-sampling.

        Args:
            X: Features DataFrame
            y: Target array
            method: Oversampling method ('smote', 'adasyn', 'random')

        Returns:
            Tuple of (resampled X, resampled y)
        """
        self.logger.info(f"Applying over-sampling: {method}")

        original_dist = pd.Series(y).value_counts().to_dict()
        self.logger.info(f"Original distribution: {original_dist}")

        if method == "smote":
            params = self._params("alternative_methods.smote.params")
            self.sampler = SMOTE(**params)
        elif method == "adasyn":
            params = self._params("sampling_strategies.oversampling.params")
            self.sampler = ADASYN(**params)
        elif method == "random":
            params = self._params("alternative_methods.random_oversampling.params")
            self.sampler = RandomOverSampler(**params)
        else:
            raise ValueError(f"Unknown oversampling method: {method}")

        X_resampled, y_resampled = self.sampler.fit_resample(X, y)

        new_dist = pd.Series(y_resampled).value_counts().to_dict()
        self.logger.info(f"Resampled distribution: {new_dist}")

        # Convert back to DataFrame with original column names
        if isinstance(X_resampled, np.ndarray):
            X_resampled = pd.DataFrame(
                X_resampled, columns=X.columns if hasattr(X, "columns") else None
            )

        return X_resampled, y_resampled

    def apply_hybrid_sampling(
        self, X: pd.DataFrame, y: np.ndarray, method: str = "smoteenn"
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply hybrid sampling (combination of over and under).

        Args:
            X: Features DataFrame
            y: Target array
            method: Hybrid method ('smoteenn', 'smotetomek')

        Returns:
            Tuple of (resampled X, resampled y)
        """
        self.logger.info(f"Applying hybrid sampling: {method}")

        original_dist = pd.Series(y).value_counts().to_dict()
        self.logger.info(f"Original distribution: {original_dist}")

        if method == "smoteenn":
            params = self._params("sampling_strategies.hybrid.params")
            smote_params = params.pop("smote", {})
            enn_params = params.pop("enn", {})
            if "random_state" in smote_params:
                smote_params["random_state"] = self.random_state
            params["random_state"] = self.random_state
            self.sampler = SMOTEENN(
                smote=SMOTE(**smote_params),
                enn=EditedNearestNeighbours(**enn_params),
                **params,
            )
        elif method == "smotetomek":
            params = self._params("alternative_methods.smotetomek.params")
            params.setdefault("random_state", self.random_state)
            self.sampler = SMOTETomek(**params)
        else:
            raise ValueError(f"Unknown hybrid method: {method}")

        X_resampled, y_resampled = self.sampler.fit_resample(X, y)

        new_dist = pd.Series(y_resampled).value_counts().to_dict()
        self.logger.info(f"Resampled distribution: {new_dist}")

        # Convert back to DataFrame
        if isinstance(X_resampled, np.ndarray):
            X_resampled = pd.DataFrame(
                X_resampled, columns=X.columns if hasattr(X, "columns") else None
            )

        return X_resampled, y_resampled

    def apply_sampling(
        self, X: pd.DataFrame, y: np.ndarray, strategy: str = "hybrid"
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply sampling strategy.

        Args:
            X: Features DataFrame
            y: Target array
            strategy: Sampling strategy ('undersampling', 'oversampling', 'hybrid')

        Returns:
            Tuple of (resampled X, resampled y)
        """
        if strategy == "undersampling":
            return self.apply_undersampling(X, y, method="tomek")
        elif strategy == "oversampling":
            return self.apply_oversampling(X, y, method="adasyn")
        elif strategy == "hybrid":
            return self.apply_hybrid_sampling(X, y, method="smoteenn")
        elif strategy == "none":
            self.logger.info("No sampling applied")
            return X, y
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def get_imbalance_ratio(self, y: np.ndarray) -> float:
        """
        Calculate imbalance ratio.

        Args:
            y: Target array

        Returns:
            Imbalance ratio (minority / majority)
        """
        counts = pd.Series(y).value_counts()
        ratio = counts.min() / counts.max()

        self.logger.info(f"Imbalance ratio: {ratio:.4f}")

        return ratio
