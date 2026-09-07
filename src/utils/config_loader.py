"""Configuration loader utility."""
import yaml
from pathlib import Path
from typing import Dict, Any


DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


class ConfigLoader:
    """Load and manage YAML configuration files."""
    
    def __init__(self, config_dir: str | Path | None = None):
        """
        Initialize ConfigLoader.
        
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir).resolve() if config_dir else DEFAULT_CONFIG_DIR
        self._configs = {}
        
    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of config file (without .yaml extension)
            
        Returns:
            Configuration dictionary
        """
        if config_name in self._configs:
            return self._configs[config_name]
            
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Configuration root must be a mapping: {config_path}")
            
        self._configs[config_name] = config
        return config
    
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files.
        
        Returns:
            Dictionary of all configurations
        """
        config_files = [
            "base_config",
            "model_config",
            "sampling_config",
            "des_config",
            "feature_config",
            "experiment_config"
        ]
        
        all_configs = {}
        for config_name in config_files:
            all_configs[config_name] = self.load(config_name)
            
        return all_configs
    
    def get(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get a specific value from config using dot notation.
        
        Args:
            config_name: Name of config file
            key_path: Path to key using dots (e.g., "lightgbm.base_params.seed")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        config = self.load(config_name)
        
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value


# Cache one loader per resolved configuration directory.
_config_loaders: Dict[Path, ConfigLoader] = {}

def get_config_loader(config_dir: str | Path | None = None) -> ConfigLoader:
    """Get a cached ConfigLoader for the requested configuration directory."""
    resolved = Path(config_dir).resolve() if config_dir else DEFAULT_CONFIG_DIR
    if resolved not in _config_loaders:
        _config_loaders[resolved] = ConfigLoader(resolved)
    return _config_loaders[resolved]
