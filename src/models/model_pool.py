"""Model pool manager for ensemble."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .lightgbm_wrapper import LightGBMWrapper
from ..data import ImbalanceSampler
from ..utils import get_logger


class ModelPool:
    """Manage a pool of models for ensemble."""
    
    def __init__(self, pool_name: str = "default", random_state: int = 42):
        """
        Initialize ModelPool.
        
        Args:
            pool_name: Name of the pool ('old' or 'new')
            random_state: Random seed for reproducibility
        """
        self.pool_name = pool_name
        self.logger = get_logger(f"ModelPool-{pool_name}", console=True, file=False)
        self.models: Dict[str, Any] = {}
        self.random_state = random_state
        self.sampler = ImbalanceSampler(random_state=random_state)
        
    def create_model_with_sampling(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sampling_strategy: str,
        model_name: str,
        model_class = LightGBMWrapper
    ):
        """
        Create and train a model with specific sampling strategy.
        
        Args:
            X_train: Training features
            y_train: Training target
            sampling_strategy: 'undersampling', 'oversampling', or 'hybrid'
            model_name: Name for the model
            model_class: Model class to use
        """
        self.logger.info(
            f"Creating model '{model_name}' with {sampling_strategy}"
        )
        
        # Apply sampling
        X_resampled, y_resampled = self.sampler.apply_sampling(
            X_train, y_train, strategy=sampling_strategy
        )
        
        # Create and train model
        model = model_class(name=model_name)
        model.fit(X_resampled, y_resampled)
        
        # Store model
        self.models[model_name] = {
            'model': model,
            'sampling': sampling_strategy,
            'n_samples': len(X_resampled)
        }
        
        self.logger.info(f"Model '{model_name}' created and trained")
    
    def create_pool(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        prefix: str = "model"
    ):
        """
        Create a complete model pool (3 models with different sampling).
        
        Args:
            X_train: Training features
            y_train: Training target
            prefix: Prefix for model names
        """
        self.logger.info(f"Creating model pool with prefix '{prefix}'")
        
        strategies = {
            f"{prefix}_under": "undersampling",
            f"{prefix}_over": "oversampling",
            f"{prefix}_hybrid": "hybrid"
        }
        
        for model_name, strategy in strategies.items():
            self.create_model_with_sampling(
                X_train, y_train, strategy, model_name
            )
        
        self.logger.info(
            f"Model pool created with {len(self.models)} models"
        )
    
    def get_model(self, model_name: str):
        """Get a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in pool")
        return self.models[model_name]['model']
    
    def get_all_models(self) -> List[Any]:
        """Get all models in the pool."""
        return [info['model'] for info in self.models.values()]
    
    def predict(
        self,
        X: pd.DataFrame,
        model_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get predictions from model(s).
        
        Args:
            X: Features
            model_name: Specific model name (if None, use all)
            
        Returns:
            Dictionary of predictions
        """
        predictions = {}
        
        if model_name:
            model = self.get_model(model_name)
            predictions[model_name] = model.predict(X)
        else:
            for name, info in self.models.items():
                predictions[name] = info['model'].predict(X)
        
        return predictions
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        model_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Get probability predictions."""
        probabilities = {}
        
        if model_name:
            model = self.get_model(model_name)
            probabilities[model_name] = model.predict_proba(X)
        else:
            for name, info in self.models.items():
                probabilities[name] = info['model'].predict_proba(X)
        
        return probabilities
