# 破產資料：改用 US 1999-2018

老師要求使用 **1999-2018** 的破產資料，並以**切割 a**（1999-2011 歷史、2012-2014 新營運、2015-2018 測試）進行實驗。

## 一、取得 US 1999-2018 資料

### 方式 1：GitHub（sowide/bankruptcy_dataset）

1. 開啟：<https://github.com/sowide/bankruptcy_dataset>
2. 下載 **american_bankruptcy_dataset.csv**（或 clone 後複製該檔）
3. 放到專案目錄：**`data/raw/bankruptcy/american_bankruptcy_dataset.csv`**

### 方式 2：Kaggle

1. 搜尋 **American Companies Bankruptcy Prediction Dataset**（例如 utkarshx27）
2. 下載 CSV（檔名可能為 `american_bankruptcy_dataset.csv` 或類似）
3. 放到 **`data/raw/bankruptcy/american_bankruptcy_dataset.csv`**

## 二、資料格式要求

CSV 需包含：

- **fyear**：年份（1999–2018）
- **status_label**：`"alive"` 或 `"failed"`（破產=1、未破產=0 會由程式轉換）
- **company_name**：可保留，程式會自動排除於特徵外
- **X1, X2, …**：數值特徵
- **Division**（若有）：程式會排除；**MajorGroup** 會保留為數值特徵

## 三、程式行為

- **dataset="auto"**（預設）：若存在 `american_bankruptcy_dataset.csv`，則使用 **US 1999-2018**；否則使用 Taiwan `data.csv`。
- 使用 US 資料時，**預設改為 chronological 切割**：1999-2011 歷史、2012-2014 新營運、2015-2018 測試（無需改各實驗腳本的 `SPLIT_MODE`）。
- 各實驗（01–04）無需改程式，只要把 US CSV 放到上述路徑後照常執行即可。

## 四、目錄結構

```
data/raw/bankruptcy/
├── data.csv                          # Taiwan 1999-2009（可保留作對照）
└── american_bankruptcy_dataset.csv   # US 1999-2018（放這裡即啟用）
```

放好 US 檔後，直接執行：

```powershell
python experiments\01_bankruptcy_baseline.py
# 或
python scripts\run_all_experiments.py
```

即會自動載入 US 1999-2018 並使用切割 a。
