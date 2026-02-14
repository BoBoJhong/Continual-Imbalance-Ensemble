"""Random seed management for reproducibility."""
import random
import numpy as np
import os


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # TensorFlow (if used in future)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    # PyTorch (if used in future)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f"[OK] Random seed set to: {seed}")


def get_seeds_from_config(config_loader) -> list:
    """
    Get random seeds from configuration.
    
    Args:
        config_loader: ConfigLoader instance
        
    Returns:
        List of random seeds
    """
    return config_loader.get("base_config", "random_seeds", [42])
