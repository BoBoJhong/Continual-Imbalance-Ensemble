"""
簡單測試：驗證所有核心模組都能正常運作
"""
import sys
from pathlib import Path

# 確保能找到 src 模組
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("測試 1: 導入所有模組")
print("="*60)

try:
    from src.utils import set_seed, get_config_loader, get_logger
    print("✓ utils 模組")
except Exception as e:
    print(f"✗ utils 模組: {e}")
    sys.exit(1)

try:
    from src.data import DataLoader, DataPreprocessor, DataSplitter, ImbalanceSampler
    print("✓ data 模組")
except Exception as e:
    print(f"✗ data 模組: {e}")
    sys.exit(1)

try:
    from src.models import LightGBMWrapper, XGBoostWrapper, ModelPool
    print("✓ models 模組")
except Exception as e:
    print(f"✗ models 模組: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("測試 2: 配置系統")
print("="*60)

set_seed(42)
loader = get_config_loader()
configs = loader.load_all()
print(f"✓ 載入了 {len(configs)} 個配置文件")

for name in configs.keys():
    print(f"  - {name}")

print("\n" + "="*60)
print("測試 3: 基本功能")
print("="*60)

# 測試 logger
logger = get_logger("test", console=True, file=False)
logger.info("Logger 測試成功")
print("✓ Logger 功能正常")

# 測試配置讀取
lgb_params = loader.get("model_config", "lightgbm.base_params")
print(f"✓ 讀取 LightGBM 參數: {len(lgb_params)} 個")

print("\n" + "="*60)
print("🎉 所有測試通過！")
print("="*60)
print("\n準備就緒，可以開始使用：")
print("1. 所有模組都能正常導入")
print("2. 配置系統運作正常")
print("3. 工具函數都可使用")
print("\n下一步: 下載真實資料集並開始實驗！")
print("="*60)
