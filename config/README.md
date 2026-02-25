# 設定檔目錄 (Configurations)

本目錄 (`config/`) 存放了本專案的所有 **YAML 設定檔**。

> ⛔ **開發規範限制**
> 根據 **Rule 6 (設定檔規範)**：所有的實驗參數與模型超參數**嚴禁硬編碼 (Hard-code)** 在 `experiments/` 或 `src/` 中的 Python 程式碼裡。所有變動必須回歸到本目錄下的 YAML 檔中，以確保實驗的可重現性與快速迭代調整。

## ⚙️ YAML 設定檔說明

目前預期的設定檔分類如下（隨專案擴充可能會增加）：

- `lgb_config.yaml`：定義 LightGBM 模型的底層超參數 (例如 `learning_rate`, `n_estimators`, `max_depth` 等)。
- `des_config.yaml`：定義動態集成選擇 (DES) 相關的演算法參數 (例如 KNN 的 `k` 值、Competence 選擇門檻等)。
- `data_config.yaml`：(如有) 存放不同資料集預設的切割時間點或 Fold 數量設定。
- `env_config.yaml`：(如有) 專案全域環境變數，例如日誌層級 (Log Level)、執行緒數量等。

## 💡 如何使用

設定檔由 `src/utils/` 下的載入模組 (通常包含 `get_config_loader`) 進行解析，並作為 `Dict` 或設定物件回傳給實驗腳本。

例如：

```python
from src.utils import load_config
# 載入 LightGBM 模型設定
lgb_params = load_config("lgb_config.yaml")

# 將參數傳入 Model
model = LightGBMWrapper(**lgb_params)
```

若有新增實驗或新的模型演算法，請在新增 Python 腳本前，優先考量將其參數獨立撰寫為一份 YAML 設定檔並置於此處。
