# 快速開始指南

## 1. 安裝依賴

```powershell
# 創建虛擬環境
python -m venv venv

# 啟動環境
.\venv\Scripts\activate

# 安裝依賴 (需要 5-10 分鐘)
pip install -r requirements.txt

# 快速測試
python tests\quick_test.py
```

## 2. 使用範例

### 載入資料

```python
from src.data import DataLoader

# 初始化載入器
loader = DataLoader(data_dir="data/raw")

# 載入資料集
X, y = loader.load_dataset("bankruptcy")
```

### 前處理資料

```python
from src.data import DataPreprocessor, DataSplitter

# 前處理
preprocessor = DataPreprocessor()
X_clean = preprocessor.handle_missing_values(X)
X_scaled, _ = preprocessor.scale_features(X_clean)

# 切割資料
splitter = DataSplitter()
splits = splitter.chronological_split(
    X_scaled, y,
    time_column="Year",
    historical_end=2011,
    new_operating_end=2014
)
```

### 處理不平衡

```python
from src.data import ImbalanceSampler

sampler = ImbalanceSampler()

# 使用 SMOTEENN
X_resampled, y_resampled = sampler.apply_sampling(
    X_train, y_train,
    strategy="hybrid"  # SMOTEENN
)
```

### 訓練模型

```python
from src.models import LightGBMWrapper, ModelPool

# 單一模型
model = LightGBMWrapper(name="my_model")
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 模型池 (6 個模型)
pool = ModelPool(pool_name="old_models")
pool.create_pool(X_historical, y_historical, prefix="old")

# 獲取所有預測
all_predictions = pool.predict_proba(X_test)
```

## 3. 完整實驗流程

```python
from src.data import DataLoader, DataPreprocessor, DataSplitter, ImbalanceSampler
from src.models import ModelPool
from src.utils import set_seed, get_logger

# 設定隨機種子
set_seed(42)

# 載入資料
loader = DataLoader()
X, y = loader.load_dataset("bankruptcy")

# 切割資料
splitter = DataSplitter()
splits = splitter.chronological_split(
    X, y, time_column="Year",
    historical_end=2011,
    new_operating_end=2014
)

X_historical, y_historical = splits['historical']
X_new, y_new = splits['new_operating']
X_test, y_test = splits['testing']

# 創建 Old Models (Historical Data)
old_pool = ModelPool(pool_name="old")
old_pool.create_pool(X_historical, y_historical, prefix="old")

# 創建 New Models (New Operating Data)
new_pool = ModelPool(pool_name="new")
new_pool.create_pool(X_new, y_new, prefix="new")

# 測試
old_predictions = old_pool.predict_proba(X_test)
new_predictions = new_pool.predict_proba(X_test)

print("Old models predictions:", old_predictions.keys())
print("New models predictions:", new_predictions.keys())
```

## 4. 配置文件

所有參數都在 `config/` 目錄：

- `base_config.yaml` - 基礎設定
- `model_config.yaml` - 模型超參數
- `sampling_config.yaml` - 採樣策略
- `des_config.yaml` - 集成配置
- `feature_config.yaml` - 特徵選擇
- `experiment_config.yaml` - 實驗設定

## 5. 疑難排解

### 安裝失敗

如果遇到安裝問題：

```powershell
# 更新 pip
python -m pip install --upgrade pip

# 單獨安裝關鍵套件
pip install numpy pandas scikit-learn
pip install lightgbm xgboost
pip install imbalanced-learn deslib
pip install pyyaml
```

### 測試失敗

```powershell
# 快速測試
python tests\quick_test.py

# 完整測試
python tests\test_setup.py
```

## 6. 下一步

1. **下載資料集**
   - Bankruptcy: [Kaggle Taiwan Economic Journal](https://www.kaggle.com/)
   - Stock: [Stock Market Crash Prediction](https://www.kaggle.com/)
   - Medical: [MIMIC-III](https://physionet.org/content/mimiciii/)

2. **資料探索**
   - 使用 Jupyter Notebook: `jupyter lab`
   - 創建 `notebooks/01_data_exploration.ipynb`

3. **開始實驗**
   - 根據上面的範例建立實驗腳本
   - 調整 `config/` 中的參數

需要更多幫助？查看 `docs/` 目錄！
