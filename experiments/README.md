# 實驗腳本目錄 (Experiments)

本目錄存放所有可直接執行的實驗 Python 腳本。

根據專案規範 **Rule 8 (實驗腳本細粒度控制)**，為了確保實驗數據的可讀性與未來擴展彈性，所有的基準實驗 (Baseline) 與靜態集成 (Static Ensemble) 皆獨立為不同的腳本，並輸出獨立的結果檔案。

## 🧪 實驗腳本清單

| 編號 | 檔名 | 資料集 | 實驗說明 |
| :--- | :--- | :--- | :--- |
| **01** | `01_bankruptcy_baseline.py` | Bankruptcy | **基準實驗**：執行 Re-training 與 Fine-tuning。 |
| **02** | `02_bankruptcy_ensemble.py` | Bankruptcy | **靜態集成**：執行 Old 3, New 3, Type A/B 等組合。 |
| **03** | `03_bankruptcy_des.py` | Bankruptcy | **動態集成 (DES)**：執行 KNORA-E 等動態選擇策略。 |
| **04** | `04_bankruptcy_feature_selection_study.py` | Bankruptcy | **特徵選擇研究**：探討特徵篩選比例對模型效能的影響。 |
| **05** | `05_stock_baseline.py` | Stock (US) | **基準實驗**：針對 SPX, DJI, NDX 執行 Re-training 與 Fine-tuning。 |
| **06** | `06_stock_ensemble.py` | Stock (US) | **靜態集成**：針對 SPX, DJI, NDX 執行靜態集成組合。 |
| **07** | `07_medical_baseline.py` | Medical | **基準實驗**：醫療資料集的 Re-training 與 Fine-tuning。 |
| **08** | `08_medical_ensemble.py` | Medical | **靜態集成**：醫療資料集的靜態集成組合。 |
| **09** | `09_stock_des.py` | Stock (US) | **動態集成 (DES)**：針對 SPX, DJI, NDX 執行 DES。 |
| **10** | `10_medical_des.py` | Medical | **動態集成 (DES)**：執行醫療資料集的 DES。 |
| **11** | `11_bankruptcy_des_advanced.py` | Bankruptcy | **進階 DES**：加入 Time-weighted 等進階策略。 |
| **12** | `12_bankruptcy_proportion_study.py` | Bankruptcy | **資料比例研究**：分析不同比例新資料對集成的影響。 |

## 🛠️ 共用腳本 (Common Scripts)

開頭為 `common_` 的腳本，是提供各資料集執行的「**基礎流程邏輯**」或「**資料加載邏輯**」，**不能直接執行**，而是由 `01`~`12` 的主腳本引用。

- `common_dataset.py`：負責載入 Stock (預設為 SPX) 與 Medical 等資料集，並進行基本的資料分割 (Split)。
- `common_bankruptcy.py`：專門負責 Bankruptcy 資料集的下載、清洗、特徵處理與分割。
- `common_des.py`：動態集成 (DES) 模型的核心訓練與預測流程封裝，供 `03`, `09`, `10` 呼叫。
- `common_des_advanced.py`：進階 DES 的流程邏輯封裝，供 `11`, `12` 呼叫。

> ⚠️ **開發規範提醒**
> 任何新增的腳本都必須依序編號 (例如 `13_new_experiment.py`)。若新增了全新的資料集，請先查閱 `.agents/rules/rules.md` 中的 **Rule 7：新增資料集擴展 SOP**。
