"""LightGBM model wrapper."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, Optional, Tuple
from ..utils import get_logger, get_config_loader


class LightGBMWrapper:
    """Wrapper for LightGBM classifier with imbalance handling."""
    
    def __init__(
        self,
        name: str = "lightgbm",
        use_imbalance: bool = True,
        **custom_params
    ):
        """
        Initialize LightGBM wrapper.
        
        Args:
            name: Model name
            use_imbalance: Whether to use imbalance parameters
            **custom_params: Custom parameters to override config
        """
        self.name = name
        self.logger = get_logger(f"LightGBM-{name}", console=True, file=False)
        self.config = get_config_loader()
        
        # Load config parameters
        base_params = self.config.get("model_config", "lightgbm.base_params", {})
        
        if use_imbalance:
            imbalance_params = self.config.get("model_config", "lightgbm.imbalance_params", {})
            base_params.update(imbalance_params)
        
        # Override with custom params
        base_params.update(custom_params)
        
        self.params = base_params
        self.model = None
        self.best_params = None
        self._num_boost_round = None

        # sklearn API 常用 n_estimators；對 lgb.train 對應 num_boost_round
        if "n_estimators" in self.params and "num_boost_round" not in self.params:
            try:
                self._num_boost_round = int(self.params.pop("n_estimators"))
            except Exception:
                self._num_boost_round = None
        
        self.logger.info(f"Initialized LightGBM with params: {self.params}")
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        **fit_params
    ):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            **fit_params: Additional fit parameters
        """
        self.logger.info(f"Training LightGBM on {len(X_train)} samples")
        
        # Create dataset
        train_data = lgb.Dataset(X_train, label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train model
        num_boost_round = fit_params.pop("num_boost_round", None)
        if num_boost_round is None:
            num_boost_round = self._num_boost_round

        train_kwargs = dict(
            params=self.params,
            train_set=train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            **fit_params,
        )
        if num_boost_round is not None:
            train_kwargs["num_boost_round"] = int(num_boost_round)

        self.model = lgb.train(**train_kwargs)
        
        self.logger.info("Training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        param_grid: Optional[Dict[str, Any]] = None,
        n_iter: int = 50,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid (if None, use config)
            n_iter: Number of iterations
            cv: Number of CV folds
            
        Returns:
            Best parameters
        """
        self.logger.info("Starting hyperparameter tuning")
        
        if param_grid is None:
            param_grid = self.config.get("model_config", "lightgbm.param_grid", {})
        
        # Create LGBMClassifier
        lgbm_clf = lgb.LGBMClassifier(**self.params)
        
        # Random search
        random_search = RandomizedSearchCV(
            lgbm_clf,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best score: {random_search.best_score_:.4f}")
        
        # Update params
        self.params.update(self.best_params)
        
        return self.best_params
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: Type of importance ('gain', 'split')
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        feature_names = self.model.feature_name()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, path: str):
        """Save model to file."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.save_model(path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)
        self.logger.info(f"Model loaded from {path}")
