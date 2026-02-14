"""Data loading utilities."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from ..utils import get_logger


class DataLoader:
    """Load datasets from various sources."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing raw data
        """
        self.data_dir = Path(data_dir)
        self.logger = get_logger("DataLoader", console=True, file=False)
        
    def load_bankruptcy(
        self, 
        file_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load bankruptcy prediction dataset.
        
        Args:
            file_path: Path to bankruptcy data file
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if file_path is None:
            file_path = self.data_dir / "bankruptcy" / "data.csv"
        
        self.logger.info(f"Loading bankruptcy data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Assume target column is 'Bankrupt?' and time column is 'Year'
        target_col = 'Bankrupt?'
        time_col = 'Year'
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.logger.info(f"Loaded {len(df)} samples with {X.shape[1]} features")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def load_medical(
        self, 
        file_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load medical time series dataset (MIMIC-III).
        
        Args:
            file_path: Path to medical data file
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if file_path is None:
            file_path = self.data_dir / "medical" / "data.csv"
        
        self.logger.info(f"Loading medical data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Assume target column is 'mortality'
        target_col = 'mortality'
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.logger.info(f"Loaded {len(df)} samples with {X.shape[1]} features")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def load_stock(
        self, 
        file_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load stock market crash prediction dataset.
        
        Args:
            file_path: Path to stock data file
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if file_path is None:
            file_path = self.data_dir / "stock" / "data.csv"
        
        self.logger.info(f"Loading stock data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Assume target column is 'crash_event'
        target_col = 'crash_event'
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.logger.info(f"Loaded {len(df)} samples with {X.shape[1]} features")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def load_dataset(
        self, 
        dataset_name: str,
        file_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load dataset by name.
        
        Args:
            dataset_name: Name of dataset ('bankruptcy', 'medical', 'stock')
            file_path: Optional custom file path
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        loaders = {
            'bankruptcy': self.load_bankruptcy,
            'medical': self.load_medical,
            'stock': self.load_stock
        }
        
        if dataset_name not in loaders:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Choose from {list(loaders.keys())}"
            )
        
        return loaders[dataset_name](file_path)
    
    def get_dataset_info(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Get dataset information.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'feature_names': X.columns.tolist(),
            'class_distribution': y.value_counts().to_dict(),
            'imbalance_ratio': y.value_counts().min() / y.value_counts().max(),
            'missing_values': X.isnull().sum().sum(),
            'dtypes': X.dtypes.value_counts().to_dict()
        }
        
        self.logger.info(f"Dataset info: {info}")
        
        return info
