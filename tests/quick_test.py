"""Quick test to verify basic functionality."""
import sys
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    import yaml
    print("✓ yaml")
except ImportError as e:
    print(f"✗ yaml: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✓ pandas")
except ImportError as e:
    print(f"✗ pandas: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")
    sys.exit(1)

try:
    import lightgbm as lgb
    print("✓ lightgbm")
except ImportError as e:
    print(f"✗ lightgbm: {e}")
    sys.exit(1)

try:
    from imblearn.over_sampling import SMOTE
    print("✓ imbalanced-learn")
except ImportError as e:
    print(f"✗ imbalanced-learn: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("✅ All critical packages installed successfully!")
print("="*50)

# Test config loading
print("\nTesting configuration loader...")
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils import get_config_loader, set_seed
    print("✓ Utils imported")
    
    loader = get_config_loader()
    base_config = loader.load("base_config")
    print(f"✓ Config loaded: {len(base_config)} sections")
    
    set_seed(42)
    print("✓ Seed set")
    
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("🎉 SETUP SUCCESSFUL! Ready to start experiments.")
print("="*50)
