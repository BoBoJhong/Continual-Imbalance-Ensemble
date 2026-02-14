"""
Test script to verify project setup and configuration loading.
Run this after installation to ensure everything is working correctly.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import get_config_loader, set_seed, get_logger


def test_config_loading():
    """Test configuration loading."""
    print("\n" + "="*50)
    print("Testing Configuration Loading")
    print("="*50)
    
    try:
        loader = get_config_loader()
        
        # Load all configs
        print("\n✓ Loading all configurations...")
        all_configs = loader.load_all()
        
        for config_name, config in all_configs.items():
            print(f"  ✓ {config_name}: {len(config)} sections")
        
        # Test specific value retrieval
        print("\n✓ Testing specific value retrieval...")
        random_seed = loader.get("base_config", "random_seed")
        print(f"  Random seed: {random_seed}")
        
        lgb_objective = loader.get("model_config", "lightgbm.base_params.objective")
        print(f"  LightGBM objective: {lgb_objective}")
        
        des_method = loader.get("des_config", "des_algorithm.method")
        print(f"  DES method: {des_method}")
        
        print("\n✅ Configuration loading: SUCCESS")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration loading: FAILED")
        print(f"  Error: {e}")
        return False


def test_seed_setting():
    """Test seed setting."""
    print("\n" + "="*50)
    print("Testing Seed Setting")
    print("="*50)
    
    try:
        set_seed(42)
        print("✅ Seed setting: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Seed setting: FAILED - {e}")
        return False


def test_logging():
    """Test logging."""
    print("\n" + "="*50)
    print("Testing Logging System")
    print("="*50)
    
    try:
        logger = get_logger("test", console=True, file=False)
        logger.info("This is a test log message")
        logger.warning("This is a test warning")
        print("✅ Logging system: SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Logging system: FAILED - {e}")
        return False


def test_directory_structure():
    """Test directory structure."""
    print("\n" + "="*50)
    print("Testing Directory Structure")
    print("="*50)
    
    required_dirs = [
        "config",
        "src",
        "src/data",
        "src/models",
        "src/ensemble",
        "src/features",
        "src/evaluation",
        "src/utils",
        "data/raw",
        "data/processed",
        "data/splits",
        "results",
        "experiments",
        "notebooks",
        "tests",
        "docs",
        "logs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_path}")
        if not exists:
            all_exist = False
    
    if all_exist:
        print("\n✅ Directory structure: SUCCESS")
    else:
        print("\n⚠ Directory structure: INCOMPLETE")
    
    return all_exist


def main():
    """Run all tests."""
    print("\n" + "="*50)
    print("CONTINUAL-IMBALANCE-ENSEMBLE SETUP TEST")
    print("="*50)
    
    results = {
        "Directory Structure": test_directory_structure(),
        "Configuration Loading": test_config_loading(),
        "Seed Setting": test_seed_setting(),
        "Logging System": test_logging()
    }
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Setup is complete.")
        print("="*50)
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Download datasets to data/raw/")
        print("3. Run experiments: python experiments/run_baseline.py")
    else:
        print("⚠ SOME TESTS FAILED. Please review the errors above.")
    print("="*50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
