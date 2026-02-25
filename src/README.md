# 核心模組目錄 (Source Code Modules)

本目錄 (`src/`) 存放了本專案所有**可共用的核心邏輯函式庫**。

> ⛔ **開發規範限制**
> 根據 **Rule 1**：此目錄**禁止包含任何實驗執行邏輯**、**禁止硬編碼 (Hard-code) 路徑**。所有的實驗腳本都在 `experiments/` 內，這裡的程式碼僅負責提供 API 給它們呼叫。

## 📦 模組架構說明

為了避免義大利麵條式的程式碼積累，專案將不同的功能拆分為以下子模組。開發者在新增功能時，請確保放到正確的分類下。

### 1. `data/` —— 資料處理與載入

- `DataLoader`：負責從 `data/raw/` 讀取資料集並實作對應年份切割。
- `DataSplitter`：實作 "chronological" (依時間序列) 與 "block_cv" (區塊交叉驗證) 的切割邏輯。
- `DataPreprocessor`：負責資料清理、標準化以及特徵工程。
- `ImbalanceSampler`：封裝了 Imbalanced-Learn 的方法，如 Undersampling, SMOTE (Oversampling), SMOTEENN (Hybrid)。

### 2. `models/` —— 基礎模型封裝

- `LightGBMWrapper`：將 LightGBM 的訓練、預測介面統一化。
- `XGBoostWrapper` (若有實作)：統一 XGBoost 介面。
- `ModelPool`：負責管理大量「歷史模型」與「新模型」的集合運算與批次訓練，供 Ensemble 與 DES 使用。

### 3. `ensemble/` —— 集成學習核心

- `EnsembleCombiner`：負責靜態集成的平均預測 (`np.mean`) 與投票演算法。
- `DynamicEnsembleSelector`：實作了 **DES (KNORA-E)** 等動態選擇演算法的核心邏輯。包含計算 Competence 領域與選擇鄰居。

### 4. `features/` —— 特徵挑選

- `FeatureSelector`：利用 SelectKBest 或 LASSO 等方法，依據比例 (`FS_RATIO`) 自動替資料集篩選最具影響力的變數組合。

### 5. `evaluation/` —— 評估與輸出

- `metrics.py`：統一定義本專案的所有評估指標。**禁止單獨使用 Accuracy**，必須計算 `AUC-ROC`、`F1-Score`、`G-Mean` 與 `Recall`。
- 其他腳本負責產生比較報表與 P-Value 矩陣。

### 6. `utils/` —— 工具與設定

- 提供專案全域共用的 `get_logger` (日誌) 與 `set_seed` (隨機種子固定，保證可重現性)。
- 負責載入與解析 `config/` 中的 YAML 設定檔 (`get_config_loader`)。

## 🔗 Import 快速參考

為了維持整潔，各子模組都有 `__init__.py`，因此你可以像這樣直接引用：

```python
from src.evaluation import compute_metrics
from src.ensemble import DynamicEnsembleSelector
from src.features import FeatureSelector
from src.data import DataLoader, ImbalanceSampler
from src.utils import set_seed, get_logger
```
