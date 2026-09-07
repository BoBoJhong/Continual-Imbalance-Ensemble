# 資料集與治理說明

本文件記錄目前程式實際使用的資料格式。資料檔是否能隨原始碼公開散布，仍需依各來源授權條款逐一確認；在確認前，不應將「可下載」視為「可重新散布」。

| 資料集 | 專案路徑 | 目標欄位 | 時間欄位 | 重要處理 |
| --- | --- | --- | --- | --- |
| US Corporate Bankruptcy 1999–2018 | `data/raw/bankruptcy/american_bankruptcy_dataset.csv` | `status_label` (`failed` = 1) | `fyear` | 移除 `company_name`、`Division` |
| UCI Diabetes 130 | `data/raw/medical/diabetes130/diabetes130_medical.csv` | `mortality` | `date` | 同一病人可能有多次就診；病人層級分析需使用 group-safe split |
| Stock SPX | `data/raw/stock/stock_spx.csv` | `Crash_Event` | `Date` | 必須移除產生標籤用的 `Future_Returns_20`，避免 target leakage |

## 必要紀錄

每次建立或替換資料檔時，建議一併保存：

- 來源頁面、下載日期及原始檔名。
- 授權或使用條款版本。
- SHA-256 checksum。
- 原始列數、欄數、時間範圍及正類比例。
- 從原始資料轉換為專案格式的腳本與參數。
- 是否含個人資料、受限欄位或不可公開再散布內容。

## 洩漏防護

- 補值、縮放、特徵選擇與採樣只能在訓練資料上 fit。
- Validation/Test 僅能使用訓練階段已學得的轉換參數。
- 股票資料不得將 `Future_Returns_20` 放入特徵。
- 醫療資料除時間隔離外，應檢查同一病人是否跨越 Train、Validation 與 Test。
- 最終 Test 不得參與 threshold、drift boundary、feature ratio 或 ensemble weight 的選擇。
