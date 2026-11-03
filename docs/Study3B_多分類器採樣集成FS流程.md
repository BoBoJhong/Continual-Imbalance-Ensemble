# Study 3B：多分類器、採樣、集成與 FS（2026-09-29）

依使用者指示重新走單模型 → 集成 → Validation 選候選 → FS 的研究流程。
本流程不恢復 B-data，所有基模型仍以完整的年度窗口訓練。

## 本次執行範圍

- 分類器：Logistic Regression、Random Forest、XGBoost、LightGBM、MLP、RBF SVM。
- 採樣：none、under=TomekLinks、over=ADASYN、hybrid=SMOTEENN，沿用專案採樣參數。
- 固定三年訓練窗口、每年推進一年、FIFO 容量三個；共同 Test 特徵年 2009–2018。
- 種子：42、123、456、789、2024、7、21、84、168、999。
- 使用 X1–X18；沿用目前重建的次年事件 target。精確財報與標籤可用日期仍未驗證。
- 本輪固定各 wrapper 的基礎參數，不使用旧 Test 最佳超參數；所有預測分類器關閉额外類別權重。
- MLP 採固定最多 300 次迭代及既有收斂條件，停用採樣後的隨機 early-stopping holdout。
- 深度表格分類器與 DES/DCS 尚未納入本次預設執行。

## 年度契約

測試年為 t：每個模型最多使用 t−2 的訓練資料，t−1 僅供 Validation，t 為 Test。
第一年先建立完整三模型池；此後每年只新增一個窗口模型／分類器／採樣配置。
單模型 baseline 重用當年的最新模型，沒有額外訓練。

所有 fit 都依序執行：訓練資料均值補值 → 標準化 → 可選 FS → 採樣 → 分類器。
Validation 和 Test 只套用各模型保存的轉換與特徵集合，不做採樣。
特徵選擇器只讀原始訓練窗口，不讀合成樣本、Validation 或 Test 標籤。

## 階段 1：全配置單模型

六分類器 × 四採樣配置 = 每年 24 個最新窗口單模型。
十年完整三模型池共需 12 個不同窗口，因此每個 seed 的 no-FS 基模型實際擬合數為 288，
十個 seeds 共 2,880 次；FS 額外成本取決於當年選中的集成成員，會逐筆稽核。

## 階段 2：固定集成集合

| 集成類型 | 定義 | 每年配置数 |
|---|---|---:|
| 跨時間 | 同分類器、同採樣的三個年度窗口模型等權 | 24 |
| 跨採樣 | 同分類器的 under/over/hybrid，分最新窗口及 FIFO | 12 |
| 跨分類器 | 六分類器同採樣，分最新窗口及 FIFO | 8 |
| 全部合併 | 六分類器 × 三類採樣，分最新窗口及 FIFO | 2 |

合計 46 個集成配置，加上 24 個單模型，共 70 個預先定義的方法。
none 作独立對照，不併入跨採樣的三票中。全部使用機率等權平均，沒有事後 Test 子集搜尋。
本輪不包含所有可能子集、動態選擇或最佳權重搜尋。

每個年度使用該年 t−1 Validation AP 選一個集成配置；同分依配置名稱排序。
這是隨年度選擇的策略，不能用全十年 Validation 平均去反選早期年度，避免早期 Test
在後續成為 Validation 時造成回看。選擇紀錄在 Test 指標計算前寫入。

## 階段 3：選定集成的 FS 對照

每年度只針對階段 2 選中的集成成員重訓：

- no-FS 對照直接重用原來的模型預測。
- MI、CART、RFE、SHAP，各保留 50% 與 80%：18 維對應 9／14 個特徵（向下取整）。
- MI 使用 mutual information；CART 為 balanced decision tree 的重要性；RFE 使用相同樹種。
- SHAP 使用訓練窗口上 100 棵樹的 balanced LightGBM，依訓練樣本平均絕對 TreeSHAP
  contribution 排序，排除 bias 欄；不以一般 importance 冒充 SHAP。
- 選擇器的 seed 跟隨該次實驗 seed，欄位與時間窗口逐個模型保存。

保留全部九種 FS 對照及 Validation 再選擇的策略結果，依相同年度 Test 評估。
FS 配置可能每年度不同；機率平均允許每個成員擁有不同特徵集合。
依序重用同一 Validation 進行集成／FS／門檻選擇，因此 Validation 分數不能當無偏效能。
此分階段搜尋只回答 FS 是否改善所選配置，不宣稱全配置空間的最佳解。

## 評估與輸出

主排序指標為 AP（輸出欄名 PR_AUC）；另報 AUC、F1、G-Mean、Recall、Precision、
Balanced Accuracy、Specificity、Brier，以及由 Validation FPR≤5% 選門檻後的實際 Test 操作指標。
F1 門檻由 Validation 的 0.01–0.99 網格選取。Test FPR 不保證仍小於 5%。
年度平均與 pooled 指標分開；seed 標準差不是公司或時間不確定性的信賴區間。
此次先產生可追溯逐筆結果，未以 seed 當獨立資料做顯著性結論。

每個 run 保存來源／資料／套件版本、設定、進度、每次 fit 的採樣前後樣本數、實際欄位、
模型參數、訓練警告、耗時、年度選擇、逐筆 Test 預測及彙總。
同一凍結模型的未來機率只用未來特徵計算，快取不含未來標籤；模型本體不永久保存，
此為節省磁碟的 frozen-model replay，不是部署 checkpoint。
若任一分類器／採樣訓練失敗，记录錯誤並繼續其餘基模型；階段 1 結束後拒絕以不完整候選進入集成。

## 執行與續跑

```powershell
python experiments/phase_flexible/study3b_full_pipeline.py
python experiments/phase_flexible/study3b_full_pipeline.py --resume "results/phase_flexible/study3b_full_pipeline/runs/<timestamp>"
python scripts/analysis/study3b_pipeline_status.py "results/phase_flexible/study3b_full_pipeline/runs/<timestamp>"
```

續跑使用原 manifest 的設定；來源、資料或套件有變則拒絕混用快取，必須另開 run。
`RUNNING.lock` 防止同一 run 有兩個寫入者；非正常關機若殘留鎖，須先確認原程序已結束才移除。
每個訓練任務有原子寫入的機率快取與 SHA-256，已完成的任務不重新擬合。
預設全部十個 seeds 會依序自動執行三階段。

本次正式 run：[`20260928T185026381432Z/manifest.json`](../results/phase_flexible/study3b_full_pipeline/runs/20260928T185026381432Z/manifest.json)。
該時間戳為 UTC；啟動後即持續在背景執行。以 `progress.json`、任務快取及作業系統程序狀態共同確認進度，
不能只憑 manifest 的 running 判定程序仍存活。另在 `smoke/` 以真實資料、單分類器及單一 Test 年
驗證三階段串接；此為工程檢查，不併入正式多分類器結果。

## 主張紀錄

分表輸出可在任意完成的 seed 邊界更新；只納入整個三階段完成的 seeds：

```powershell
python scripts/analysis/export_study3b_separated_tables.py "results/phase_flexible/study3b_full_pipeline/runs/<timestamp>" --update-thesis
```

每次產生新的快照資料夾，分開單模型、各集成機制及 FS，並分開 pooled／年度平均、各 seed／逐年資料。
`--update-thesis` 只更新論文中本流程的方法與結果區塊，不改動其餘章節。
2026-09-29 首次分表快照使用 seeds 42、123、456，已同步論文 §3.10.7／§4.5.8。

供人工彙整的 Study 1–2 同格式 CSV 另由下列入口匯出，分類器、集成方式、FS 方法各自存檔，
附 `compact_summary`、`AUC_only`／其他指標、逐年與各 seed 原始值，並打包 ZIP：

```powershell
python scripts/analysis/export_study3b_study12_format.py "results/phase_flexible/study3b_full_pipeline/runs/<timestamp>"
```

輸出在該 run 的 `manual_spreadsheets/<snapshot>/`；與論文 Markdown 分表用途不同。

| 主張 | 證據 | 狀態 |
|---|---|---|
| 已具備完整三階段入口及年度選擇隔離 | `study3b_full_pipeline.py`、`test_study3b_full_pipeline.py` | 已實作並測試 |
| 已完成全部正式實驗 | 本次 run 的 `manifest.json` | 依實際 status 判定；不能以啟動代替完成 |
| 採樣、集成或 FS 改善預測 | 本次 run 完成後的逐筆配對結果 | 尚待評估，不預先宣稱 |
