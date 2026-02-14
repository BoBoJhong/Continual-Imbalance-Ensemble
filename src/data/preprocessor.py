"""Data preprocessing utilities."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, List
from ..utils import get_logger


class DataPreprocessor:
    """Preprocess datasets for modeling."""
    
    def __init__(self):
        """Initialize DataPreprocessor."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.logger = get_logger("DataPreprocessor", console=True, file=False)
        self.feature_names = None
        
    def handle_missing_values(
        self, 
        X: pd.DataFrame,
        strategy: str = "mean"
    ) -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            X: Features DataFrame
            strategy: Strategy for imputation ('mean', 'median', 'forward_fill')
            
        Returns:
            DataFrame with imputed values
        """
        self.logger.info(f"Handling missing values with strategy: {strategy}")
        
        missing_count = X.isnull().sum().sum()
        if missing_count == 0:
            self.logger.info("No missing values found")
            return X
        
        self.logger.info(f"Found {missing_count} missing values")
        
        X_imputed = X.copy()
        
        if strategy == "mean":
            X_imputed = X_imputed.fillna(X_imputed.mean())
        elif strategy == "median":
            X_imputed = X_imputed.fillna(X_imputed.median())
        elif strategy == "forward_fill":
            X_imputed = X_imputed.fillna(method='ffill').fillna(method='bfill')
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.logger.info("Missing values handled")
        
        return X_imputed
    
    def remove_outliers(
        self, 
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 3.0
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove outliers using Z-score method.
        
        Args:
            X: Features DataFrame
            y: Target Series
            threshold: Z-score threshold
            
        Returns:
            Tuple of (X without outliers, y without outliers)
        """
        self.logger.info(f"Removing outliers with threshold: {threshold}")
        
        # Calculate Z-scores
        z_scores = np.abs((X - X.mean()) / X.std())
        
        # Find rows with any Z-score above threshold
        outlier_mask = (z_scores > threshold).any(axis=1)
        n_outliers = outlier_mask.sum()
        
        self.logger.info(f"Found {n_outliers} outliers ({n_outliers/len(X)*100:.2f}%)")
        
        # Remove outliers
        X_clean = X[~outlier_mask]
        y_clean = y[~outlier_mask]
        
        return X_clean, y_clean
    
    def scale_features(
        self, 
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            fit: Whether to fit scaler on training data
            
        Returns:
            Tuple of (scaled X_train, scaled X_test)
        """
        self.logger.info("Scaling features")
        
        self.feature_names = X_train.columns.tolist()
        
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(
                X_test_scaled,
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def encode_labels(
        self, 
        y: pd.Series,
        fit: bool = True
    ) -> np.ndarray:
        """
        Encode labels to integers.
        
        Args:
            y: Target Series
            fit: Whether to fit encoder
            
        Returns:
            Encoded labels
        """
        if fit:
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)
        
        self.logger.info(f"Encoded {len(np.unique(y_encoded))} classes")
        
        return y_encoded
    
    def select_time_period(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        time_column: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select data from specific time period.
        
        Args:
            X: Features DataFrame
            y: Target Series
            time_column: Name of time column
            start_time: Start time (inclusive)
            end_time: End time (inclusive)
            
        Returns:
            Tuple of (filtered X, filtered y)
        """
        if time_column not in X.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")
        
        mask = pd.Series([True] * len(X), index=X.index)
        
        if start_time is not None:
            mask &= X[time_column] >= start_time
        
        if end_time is not None:
            mask &= X[time_column] <= end_time
        
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        self.logger.info(
            f"Selected time period {start_time} to {end_time}: "
            f"{len(X_filtered)} samples"
        )
        
        return X_filtered, y_filtered
    
    def preprocess_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        handle_missing: bool = True,
        remove_outliers: bool = False,
        scale: bool = True,
        encode: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional[pd.DataFrame], Optional[np.ndarray]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional)
            y_test: Test target (optional)
            handle_missing: Whether to handle missing values
            remove_outliers: Whether to remove outliers
            scale: Whether to scale features
            encode: Whether to encode labels
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test) all preprocessed
        """
        self.logger.info("Starting preprocessing pipeline")
        
        # Handle missing values
        if handle_missing:
            X_train = self.handle_missing_values(X_train)
            if X_test is not None:
                X_test = self.handle_missing_values(X_test)
        
        # Remove outliers (only on training data)
        if remove_outliers:
            X_train, y_train = self.remove_outliers(X_train, y_train)
        
        # Scale features
        if scale:
            X_train, X_test = self.scale_features(X_train, X_test, fit=True)
        
        # Encode labels
        if encode:
            y_train = self.encode_labels(y_train, fit=True)
            if y_test is not None:
                y_test = self.encode_labels(y_test, fit=False)
        else:
            y_train = y_train.values
            if y_test is not None:
                y_test = y_test.values
        
        self.logger.info("Preprocessing pipeline completed")
        
        return X_train, y_train, X_test, y_test
