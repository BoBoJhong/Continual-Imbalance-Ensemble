"""Data loading utilities."""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from ..utils import get_logger


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
STOCK_COLUMNS = [
    "Date", "Close", "High", "Low", "Open", "Volume", "Returns",
    "Log_Returns", "SMA_5", "SMA_20", "SMA_60", "Volatility_20",
    "RSI", "Future_Returns_20", "Crash_Event",
]


class DataLoader:
    """Load datasets from various sources."""
    
    def __init__(self, data_dir: str | Path | None = None):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing raw data
        """
        self.data_dir = Path(data_dir).resolve() if data_dir else DEFAULT_DATA_DIR
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
            us_path = self.data_dir / "bankruptcy" / "american_bankruptcy_dataset.csv"
            taiwan_path = self.data_dir / "bankruptcy" / "data.csv"
            file_path = us_path if us_path.exists() else taiwan_path
        
        self.logger.info(f"Loading bankruptcy data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        if "status_label" in df.columns:
            y = (df["status_label"].astype(str).str.lower() == "failed").astype(int)
            drop_columns = ["status_label", "company_name", "Division"]
            X = df.drop(columns=[c for c in drop_columns if c in df.columns])
        elif "Bankrupt?" in df.columns:
            y = df["Bankrupt?"].astype(int)
            X = df.drop(columns=["Bankrupt?"])
        else:
            raise ValueError(
                "Bankruptcy data must contain 'status_label' or 'Bankrupt?'"
            )
        
        self.logger.info(f"Loaded {len(df)} samples with {X.shape[1]} features")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def load_medical(
        self, 
        file_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load the processed UCI Diabetes 130 time-series dataset.
        
        Args:
            file_path: Path to medical data file
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if file_path is None:
            file_path = (
                self.data_dir / "medical" / "diabetes130" / "diabetes130_medical.csv"
            )
        
        self.logger.info(f"Loading medical data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        target_col = "mortality"
        if target_col not in df.columns:
            raise ValueError(f"Medical data must contain '{target_col}'")
        y = df[target_col].astype(int)
        X = df.drop(columns=[target_col])
        if "date" in X.columns:
            X = X.copy()
            X["Year"] = pd.to_datetime(X.pop("date"), errors="coerce").dt.year
        
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
            file_path = self.data_dir / "stock" / "stock_spx.csv"
        
        self.logger.info(f"Loading stock data from {file_path}")
        
        df = pd.read_csv(file_path, skiprows=2, names=STOCK_COLUMNS, header=None)
        df = df.dropna(subset=["Crash_Event"])
        y = pd.to_numeric(df["Crash_Event"], errors="raise").astype(int)
        X = df.drop(columns=["Crash_Event", "Future_Returns_20"]).copy()
        X["Year"] = pd.to_datetime(X.pop("Date"), errors="coerce").dt.year
        for column in X.columns:
            X[column] = pd.to_numeric(X[column], errors="coerce")
        
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
