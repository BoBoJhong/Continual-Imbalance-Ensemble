"""XGBoost model wrapper."""
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Optional
from ..utils import get_logger, get_config_loader


class XGBoostWrapper:
    """Wrapper for XGBoost classifier with imbalance handling."""
    
    def __init__(
        self,
        name: str = "xgboost",
        use_imbalance: bool = True,
        **custom_params
    ):
        """
        Initialize XGBoost wrapper.
        
        Args:
            name: Model name
            use_imbalance: Whether to use imbalance parameters
            **custom_params: Custom parameters
        """
        self.name = name
        self.logger = get_logger(f"XGBoost-{name}", console=True, file=False)
        self.config = get_config_loader()
        
        base_params = self.config.get("model_config", "xgboost.base_params", {})
        
        if use_imbalance:
            imbalance_params = self.config.get("model_config", "xgboost.imbalance_params", {})
            base_params.update(imbalance_params)
        
        base_params.update(custom_params)
        
        self.params = base_params
        self.model = None
        self.best_params = None
        
        self.logger.info(f"Initialized XGBoost with params: {self.params}")
    
    def _calculate_scale_pos_weight(self, y: np.ndarray) -> float:
        """Calculate scale_pos_weight for imbalanced data."""
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) != 2:
            return 1.0
        
        neg_count = counts[0] if unique[0] == 0 else counts[1]
        pos_count = counts[1] if unique[1] == 1 else counts[0]
        
        return neg_count / pos_count
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        **fit_params
    ):
        """Train the model."""
        self.logger.info(f"Training XGBoost on {len(X_train)} samples")
        
        # Auto-calculate scale_pos_weight if set to 'auto'
        if self.params.get('scale_pos_weight') == 'auto':
            self.params['scale_pos_weight'] = self._calculate_scale_pos_weight(y_train)
            self.logger.info(f"Auto scale_pos_weight: {self.params['scale_pos_weight']:.2f}")
        
        # Create model
        self.model = xgb.XGBClassifier(**self.params)
        
        # Prepare eval set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
            **fit_params
        )
        
        self.logger.info("Training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        feature_names = self.model.get_booster().feature_names
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
