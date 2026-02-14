"""Data splitting utilities for time series."""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List, Dict, Optional
from ..utils import get_logger, get_config_loader


class DataSplitter:
    """Split data for continual learning experiments."""
    
    def __init__(self):
        """Initialize DataSplitter."""
        self.logger = get_logger("DataSplitter", console=True, file=False)
        self.config = get_config_loader()
        
    def chronological_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_column: str,
        historical_end: str,
        new_operating_end: str
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data chronologically into historical, new operating, and testing.
        
        Args:
            X: Features DataFrame
            y: Target Series
            time_column: Name of time column
            historical_end: End time for historical period
            new_operating_end: End time for new operating period
            
        Returns:
            Dictionary with 'historical', 'new_operating', 'testing' splits
        """
        self.logger.info("Performing chronological split")
        
        if time_column not in X.columns:
            raise ValueError(f"Time column '{time_column}' not found")
        
        # Sort by time
        sort_idx = X[time_column].argsort()
        X_sorted = X.iloc[sort_idx]
        y_sorted = y.iloc[sort_idx]
        
        # Create masks for each period
        historical_mask = X_sorted[time_column] <= historical_end
        new_operating_mask = (
            (X_sorted[time_column] > historical_end) & 
            (X_sorted[time_column] <= new_operating_end)
        )
        testing_mask = X_sorted[time_column] > new_operating_end
        
        # Split data
        X_historical = X_sorted[historical_mask]
        y_historical = y_sorted[historical_mask]
        
        X_new_operating = X_sorted[new_operating_mask]
        y_new_operating = y_sorted[new_operating_mask]
        
        X_testing = X_sorted[testing_mask]
        y_testing = y_sorted[testing_mask]
        
        self.logger.info(f"Historical: {len(X_historical)} samples")
        self.logger.info(f"New Operating: {len(X_new_operating)} samples")
        self.logger.info(f"Testing: {len(X_testing)} samples")
        
        return {
            'historical': (X_historical, y_historical),
            'new_operating': (X_new_operating, y_new_operating),
            'testing': (X_testing, y_testing)
        }
    
    def block_cv_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        historical_folds: List[int] = [1, 2],
        new_operating_folds: List[int] = [3, 4],
        testing_fold: int = 5
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """
        Split data using block-based cross-validation.
        
        Args:
            X: Features DataFrame
            y: Target Series
            n_folds: Number of folds
            historical_folds: Fold indices for historical data
            new_operating_folds: Fold indices for new operating data
            testing_fold: Fold index for testing data
            
        Returns:
            Dictionary with 'historical', 'new_operating', 'testing' splits
        """
        self.logger.info(f"Performing {n_folds}-fold block CV split")
        
        # Calculate fold sizes
        n_samples = len(X)
        fold_size = n_samples // n_folds
        
        # Create fold assignments
        fold_assignments = np.array([i // fold_size for i in range(n_samples)])
        fold_assignments = np.minimum(fold_assignments, n_folds - 1)  # Handle remainder
        
        # Combine historical folds
        historical_mask = np.isin(fold_assignments, [f - 1 for f in historical_folds])
        X_historical = X[historical_mask]
        y_historical = y[historical_mask]
        
        # Combine new operating folds
        new_operating_mask = np.isin(fold_assignments, [f - 1 for f in new_operating_folds])
        X_new_operating = X[new_operating_mask]
        y_new_operating = y[new_operating_mask]
        
        # Testing fold
        testing_mask = fold_assignments == (testing_fold - 1)
        X_testing = X[testing_mask]
        y_testing = y[testing_mask]
        
        self.logger.info(f"Historical (folds {historical_folds}): {len(X_historical)} samples")
        self.logger.info(f"New Operating (folds {new_operating_folds}): {len(X_new_operating)} samples")
        self.logger.info(f"Testing (fold {testing_fold}): {len(X_testing)} samples")
        
        return {
            'historical': (X_historical, y_historical),
            'new_operating': (X_new_operating, y_new_operating),
            'testing': (X_testing, y_testing)
        }
    
    def create_validation_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_ratio: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Create training and validation split (time-aware).
        
        Args:
            X: Features DataFrame
            y: Target Series
            validation_ratio: Ratio for validation set
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        split_idx = int(len(X) * (1 - validation_ratio))
        
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        
        self.logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}")
        
        return X_train, y_train, X_val, y_val
    
    def get_time_series_cv(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None
    ) -> TimeSeriesSplit:
        """
        Get TimeSeriesSplit cross-validator.
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set
            
        Returns:
            TimeSeriesSplit instance
        """
        return TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
