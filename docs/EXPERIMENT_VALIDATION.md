# 實驗正當性檢查（碩論需求對照）

本文件確認目前實驗的**跑法是否正當**、是否符合寫碩論的常見要求，並標註論文中應寫清楚的幾點。

---

## 一、已符合、可放心寫進論文的項目

### 1. 資料切割（無洩漏）

| 項目 | 實作 | 說明 |
|------|------|------|
| **切割方式** | 5-fold block CV：1+2 折＝歷史、3+4 折＝新營運、第 5 折＝測試 | 與老師要求「切割 b」一致。 |
| **切分邏輯** | `DataSplitter.block_cv_split()` 依**樣本順序**切為 5 塊，再合併 1+2、3+4、5 | 測試集（第 5 折）從未參與訓練或前處理擬合。 |
| **前處理擬合範圍** | 標準化：`StandardScaler` 僅在 **historical** 上 `fit`，再 `transform` historical / new / test | 無測試集資訊洩漏。 |
| **Study II 特徵選擇** | `SelectKBest` 僅在 **historical** 上 `fit`，再 `transform` hist / new / test | 無測試集洩漏。 |

**論文建議寫法**：  
「資料依 5-fold block 切為 historical（fold 1+2）、new operating（fold 3+4）、testing（fold 5）。標準化與特徵選擇僅以 historical 擬合，再套用至 new 與 test，避免測試集洩漏。」

---

### 2. Baseline 定義（與老師要求一致）

| 老師要求 | 實作 | 正當性 |
|----------|------|--------|
| **Re-training**：歷史＋新營運資料一起訓練 | 合併 historical + new → 取樣 → 訓練一個模型 → 在 **test** 上評估 | ✓ 僅用 train（hist+new），評估在 test。 |
| **Fine-tuning**：僅用新資料微調 | 先在 historical 上訓練，再**僅用 new** 做第二次 `fit`，在 **test** 上評估 | ✓ 微調階段只用 new；評估在 test。 |

**論文建議寫法**：  
「Baseline 1：Re-training，合併 historical 與 new operating 資料訓練單一模型。Baseline 2：Fine-tuning，先在 historical 上預訓練，再僅以 new operating 資料進行第二階段訓練（sequential training on new data only）。」

若口委問「fine-tuning 是否用 reduced learning rate」：可說明目前實作為「僅用新資料做第二階段訓練」；若要做嚴格意義的 LR 降低式 fine-tuning，可再補實驗並在論文中對比。

---

### 3. Ensemble 與 DES（無洩漏）

| 項目 | 實作 | 正當性 |
|------|------|--------|
| **Old 模型** | 僅用 historical 訓練（under/over/hybrid 三種） | ✓ |
| **New 模型** | 僅用 new operating 訓練（三種） | ✓ |
| **預測與評估** | 所有 ensemble / DES 的預測與指標皆在 **test** 上計算 | ✓ |
| **DES 的 DSEL** | DSEL = historical + new（訓練用資料），用於估計 competence；預測在 test | ✓ 測試集未參與 DSEL。 |

---

### 4. 評估指標與評估集

| 項目 | 實作 | 說明 |
|------|------|------|
| **評估集** | 一律使用 **testing（第 5 折）** | 未在訓練集上報績效。 |
| **指標** | AUC-ROC、F1、Precision、Recall | 符合 imbalanced 與 experiment_config 常用指標。 |

**可選補充**：若老師或口委要求 G-mean、balanced accuracy，可在同一評估流程上加算並在論文中報告。

---

### 5. 可重複性

| 項目 | 實作 | 說明 |
|------|------|------|
| **隨機種子** | 各實驗開頭 `set_seed(42)` | 同一環境下結果可重現。 |
| **多 seed** | `scripts/run_multi_seed.py` 可跑多組 seed，產出 mean±std | 可支援論文中「mean ± std」或後續統計檢定。 |

---

## 二、論文中建議「寫清楚」的幾點（跑法仍屬正當）

### 1. Block CV 與「時間」的關係

- **目前實作**：5-fold 是依**資料列順序**切塊（第 1 塊、第 2 塊…），**未**依時間欄位排序。
- **影響**：Bankruptcy 資料若無年份欄，則「historical / new」是**順序上的前 40% / 中 40% / 後 20%**，而非嚴格時間上的 1999–2011 / 2012–2014 / 2015–2018。
- **論文建議寫法**：  
  「本實驗採用 5-fold block 切割（1+2 折＝historical、3+4 折＝new operating、第 5 折＝testing）。在無時間欄位之資料集上，block 依樣本索引順序劃分，以模擬前、中、後段資料之分布差異。」

若未來改用具年份欄的資料，可改為 `chronological_split`（切割 a），並在論文中對比兩種切割。

---

### 2. Fine-tuning 的實作定義

- **目前實作**：先 `fit(historical)`，再 `fit(new)`；LightGBM 的第二次 `fit` 會重新訓練，等同「僅用 new 資料從頭訓練一個新模型」或「以 new 為主的第二階段訓練」。
- **論文建議寫法**：  
  「Fine-tuning baseline：先以 historical 訓練，再僅以 new operating 資料進行第二階段訓練（sequential training on new data only），評估時僅使用 testing set。」  
  若口委要求「classical fine-tuning（降低學習率、少數輪數）」，可於論文中列為限制或未來工作，並可再補一組實驗對比。

---

### 3. 統計檢定（Wilcoxon 等）

- **目前**：有 mean±std（多 seed）、單次 run 的 AUC/F1 等，**尚未**內建 Wilcoxon 或多重比較校正。
- **論文建議**：  
  若老師要求「方法間要做統計檢定」，可於**論文中**撰寫：「在 5 個（或 N 個）隨機種子下重複實驗，得到各方法之 AUC/F1，再以 Wilcoxon signed-rank test 比較，並依需要進行 Bonferroni 校正。」實際檢定可用 R/Python 事後對現有結果做一次即可，不一定要寫進程式碼。

---

## 三、總結：是否正當、是否符合碩論需求

| 檢查項目 | 結論 |
|----------|------|
| 測試集是否參與訓練或前處理擬合？ | **否**，正當。 |
| 切割是否符合老師要求（5-fold block）？ | **是**。 |
| Baseline（Re-training / Fine-tuning）定義是否一致？ | **是**。 |
| Ensemble / DES 是否僅在 test 上評估？ | **是**。 |
| 指標（AUC、F1 等）是否合理？ | **是**。 |
| 可重複性（seed、多 seed）是否具備？ | **是**。 |

**結論**：  
目前實驗**跑法正當**、**符合寫碩論的常見要求**。只要在論文中把上述「建議寫清楚」的幾點（block 與時間、fine-tuning 定義、若要做統計檢定則如何做）說明清楚即可。  
若老師或口委特別要求「嚴格 fine-tuning（降 LR）」「一定要 Wilcoxon」「一定要 G-mean」等，再針對性補實驗或補一節說明即可。
