# Continual-Imbalance-Ensemble

非平穩環境下類別不平衡之持續學習集成框架

A continual learning framework for class imbalance in non-stationary datasets, combining **Dynamic Ensemble Selection (DES)**、**Hybrid Sampling** 與多時期（Old / New）模型池。

---

## 目錄

| 區塊 | 說明 |
|------|------|
| [研究目標](#研究目標) | 問題設定、與 `docs/`／UML 對照 |
| [研究現況](#研究現況-current-progress) | 各資料集**進度表**、破產數值快照、待辦（不含方法細節） |
| [專案結構](#專案結構) | `src/`、`experiments/`、Phase 2 腳本、`scripts/`、`results/` |
| [資料集與時間切割](#資料集與時間切割) | Old／New／Test、各資料集年份規則 |
| [實驗方法說明](#實驗方法說明) | **Phase 1 Baseline** 與 **Phase 2 集成**分開撰寫 |
| [評估指標](#評估指標) | AUC、F1、G-Mean、Recall |
| [Quick Start](#quick-start) · [Import 範例](#import-範例) | 環境、指令、常用 import |
| [技術棧](#技術棧) · [文件索引](#文件索引) | 依賴與延伸閱讀 |

---

## 研究目標

在**非平穩**與**類別不平衡**並存時，比較再訓練、微調、**靜態集成**、**動態 DES**（鄰域選擇）與 **DCS**（鄰域競爭、OLA／LCA 等），並可延伸探討特徵選擇之影響。

詳見 [docs/RESEARCH_SPEC.md](docs/RESEARCH_SPEC.md)、[docs/研究方向.md](docs/研究方向.md)。

### 研究方向與實驗階段及 UML 對照

以下將 [docs/研究方向.md](docs/研究方向.md) 的條目，對到本儲存庫的**階段／腳本**與 **PlantUML**（細節見 [UML/README_圖表建議.md](UML/README_圖表建議.md)、完整矩陣見 [docs/研究方向對照表.md](docs/研究方向對照表.md)）。

| 研究方向.md 要點 | 實驗階段與主要路徑 | 建議對照之 UML |
|------------------|-------------------|----------------|
| 三資料集（破產／股票／醫療）、非平穩情境 | Phase 1～4；`_shared/common_*.py` | `current_experiment_architecture.puml`；總覽可選 `research_pipeline_flow_like_reference.puml` |
| 切割 (a) 年份：歷史／新營運／測試 | Phase 1/2：`*_year_splits_*.py` | `study1_baseline_and_ensemble_flow.puml`；切割概念 `include/_split_data.puml` |
| 切割 (b) 5-fold block CV | 程式支援 `block_cv`（主結果仍以年份切割為主） | 同上總覽圖；對照表註明「備援切割」 |
| Baseline：Re-training、Fine-tuning | **Phase 1** — 與研究方向**完全一致**之 **Retrain／Finetune** 目前僅 **破產 XGB**（+ MLP 微調）；三資料集差異見 [Phase 1 Baseline](#method-phase1-baseline) | `baseline_flow.puml`（四訓練設定 + 閾值） |
| 集成：Old×3、New×3、採樣、2～6 組合 | **Phase 2** 見 [Phase 2 集成](#method-phase2-ensemble)；`xgb_oldnew_ensemble_common.py` | `ensemble_flow.puml`；組合枚舉 `ensemble_combinations_full_list.puml` |
| 動態 **DES** | **Phase 2** `dynamic/des/` | 總覽含於 `study1_baseline_and_ensemble_flow.puml`；實作目錄見 UML README「與程式目錄對照」 |
| 動態 **DCS** | **Phase 2** `dynamic/dcs/` | 同上；靜態與 DCS scaler 差異見腳本註解 |
| Study II：特徵選擇對集成之影響 | **Phase 3** `phase3_feature/` | `study2_feature_selection_flow.puml`（定稿前可 `study2_feature_selection_template.puml`） |

上表為「方法論 ↔ 圖檔」對照；**各資料集是否跑完**以 [研究現況](#研究現況-current-progress) 之進度表為準。

---

## 研究現況 (Current Progress)

> 最後更新：2026-03-21

### 依資料集之進度

**破產**為論文主線完成度最高；**股票、醫療**自 Phase 2 起至 Phase 4 多為**未完成**（程式可跑，結果尚未跑齊／彙整）。「未完成」＝未達該階段預定產出；數值以 `results/` 內最新 CSV 為準。

**Phase 2** 欄位細分：**靜態集成**／**動態 DES**／**動態 DCS**。下表欄位：**階段** → **項目** → **方法／分類器** → **狀態** → **備註**。

**與方法說明的分工**：本節只列**完成度、路徑與快照數值**；Phase 1 四策略、Retrain／Finetune、Phase 2 集成（六槽位、DES、DCS）之**定義與實作對照**見下方 [實驗方法說明](#實驗方法說明)。

#### 破產 (Bankruptcy)

| 階段 | 項目 | 方法／分類器 | 狀態 | 備註／代表數值 |
|------|------|--------------|------|----------------|
| Phase 1 | Baseline（`Old`／`New`／**`Retrain`**／**`Finetune`**；另有 MLP） | **XGBoost**（`XGBoostWrapper`）；MLP 見 `bankruptcy_year_splits_torch_mlp.py` | 完成 | XGB Fine-tune AUC 例：0.8552（`results/phase1_baseline/xgb/bk_xgb_table_*_finetune.csv`） |
| Phase 2｜靜態 | 年份切割、Old／New／k 子集、等權平均與閾值 | **XGB** 六槽位集成（靜態平均） | 完成 | `results/phase2_ensemble/static/`、`xgb_oldnew_ensemble_static_*_bankruptcy*` |
| Phase 2｜DES | 鄰域動態選模 | **XGB** 池 + **DES**（KNORA-E／U、DES-KNN 等） | 完成 | `…/dynamic/des/`、`xgb_oldnew_ensemble_des_*_bankruptcy*` |
| Phase 2｜DCS | 鄰域競爭選模 | **XGB** 池 + **DCS**（OLA／LCA／TW） | 完成 | `…/dynamic/dcs/`、`xgb_oldnew_ensemble_dcs_*`（列數精簡，見 [Phase 2 集成](#method-phase2-ensemble)） |
| Phase 3 | Study II 特徵選擇 | **LightGBM** `ModelPool` + `FeatureSelector`（見 `phase3_feature/fs_study.py`） | 完成 | 篩選比例已修正（`FS_RATIO` 等） |
| Phase 4 | 比例、DES Advanced、split 比較等 | 視子實驗（多為 **XGB**／既有集成元件） | 完成 | 見 `results/phase4_analysis/` |

#### 股票 (Stock，預設 SPX)

| 階段 | 項目 | 方法／分類器 | 狀態 | 備註 |
|------|------|--------------|------|------|
| Phase 1 | Baseline（`Old`／`Old+New`／`New`） | **XGBoost**（`stock_year_splits_xgb.py`） | 完成 | **無** Retrain／Finetune（見 [Phase 1 Baseline](#method-phase1-baseline)）；輸出 `stk_xgb_table_*_{old,oldnew,new}.csv` |
| Phase 2｜靜態 | 年份切割靜態集成 | **XGB** 六槽位 + 靜態平均 | **未完成** | `experiments/phase2_ensemble/static/xgb_oldnew_stock_year_splits_static.py` |
| Phase 2｜DES | 鄰域動態選模 | **XGB** 池 + **DES** | **未完成** | `experiments/phase2_ensemble/dynamic/des/xgb_oldnew_stock_year_splits_des.py` |
| Phase 2｜DCS | 鄰域競爭選模 | **XGB** 池 + **DCS** | **未完成** | `experiments/phase2_ensemble/dynamic/dcs/xgb_oldnew_stock_year_splits_dcs.py` |
| Phase 3 | Study II 特徵選擇 | 預設與破產相同管線（**LightGBM** `ModelPool` + FS） | **未完成** | — |
| Phase 4 | 補充分析（如閾值成本） | 視腳本（多與 **XGB** 或成本曲線相關） | **未完成** | — |

#### 醫療 (Medical)

| 階段 | 項目 | 方法／分類器 | 狀態 | 備註 |
|------|------|--------------|------|------|
| Phase 1 | Baseline（`Old`／`Old+New`／`New`） | **XGBoost**（`medical_year_splits_xgb.py`） | 完成 | **無** Retrain／Finetune（見 [Phase 1 Baseline](#method-phase1-baseline)）；輸出 `med_xgb_table_*_{old,oldnew,new}.csv` |
| Phase 2｜靜態 | 年份切割靜態集成 | **XGB** 六槽位 + 靜態平均 | **未完成** | `experiments/phase2_ensemble/static/xgb_oldnew_medical_year_splits_static.py` |
| Phase 2｜DES | 鄰域動態選模 | **XGB** 池 + **DES** | **未完成** | `experiments/phase2_ensemble/dynamic/des/xgb_oldnew_medical_year_splits_des.py` |
| Phase 2｜DCS | 鄰域競爭選模 | **XGB** 池 + **DCS** | **未完成** | `experiments/phase2_ensemble/dynamic/dcs/xgb_oldnew_medical_year_splits_dcs.py` |
| Phase 3 | Study II 特徵選擇 | 預設與破產相同管線（**LightGBM** `ModelPool` + FS） | **未完成** | — |
| Phase 4 | 補充分析 | 視腳本 | **未完成** | — |

### 破產｜主要指標快覽（代表性快照）

| 方法 | AUC | F1 |
|------|-----|-----|
| Re-training | 0.8047 | 0.134 |
| Fine-tuning | 0.8552 | 0.194 |
| Ensemble (Old 3) | 0.8086 | 0.133 |
| **Ensemble (New 3)** | **0.8693** | **0.239** |
| Ensemble (All 6) | 0.8575 | 0.216 |
| DES KNORA-E | 0.8560 | 0.222 |

### 待辦 (TODO)

- [x] Study II 特徵選擇比例修正（`FS_RATIO`、`src.features.FeatureSelector`；**破產**）
- [x] `src.evaluation.compute_metrics` 指標與各資料集對齊（程式面）
- [x] 多 seed：`scripts/run/run_multi_seed.py`
- [x] Wilcoxon：`scripts/analysis/statistical_test.py`
- [x] 結果視覺化：`scripts/plots/visualize_results.py`
- [x] Phase 2 XGB 年份切割：靜態／DES／DCS 分腳本與分目錄（`experiments/phase2_ensemble/static/`、`dynamic/des/`、`dynamic/dcs/`）
- [ ] **股票**：Phase 2（靜態／DES／DCS）與 Phase 3～4 跑齊與論文彙整（見「依資料集之進度」）
- [ ] **醫療**：Phase 2（靜態／DES／DCS）與 Phase 3～4 跑齊與論文彙整（見「依資料集之進度」）
- [ ] **（選）股票／醫療 Phase 1**：若需與破產一致，於 `stock_year_splits_xgb.py`／`medical_year_splits_xgb.py` 補 **Retrain**／**Finetune**（對照 `bankruptcy_year_splits_xgb.py`）

---

## 專案結構

**程式碼**

```text
src/             ← 核心模組（資料處理、模型封裝、集成、評估）
experiments/     ← 可執行實驗（phase1_baseline … phase4_analysis）
scripts/         ← 工具腳本（見下表）
config/          ← YAML 實驗與模型超參數
```

共用邏輯：`experiments/_shared/common_*.py`（非舊版 `experiments/common_*.py`）。

**研究階段與 `experiments/`（論文主線）**

| 階段 | 內容 | 程式路徑 |
|------|------|----------|
| Phase 1 | Baseline（多分類器、多資料集） | `experiments/phase1_baseline/` |
| Phase 2 | 集成（XGB）：靜態／DES／DCS 分腳本；產物分子目錄 | 見下表「Phase 2」 |
| Phase 3（FS） | 特徵選取（Study II） | `experiments/phase3_feature/` |
| Phase 4 | 補充分析（比例、split、成本等） | `experiments/phase4_analysis/` |

**Phase 2 集成腳本與結果（以破產為例；`*` = 資料集後綴）**

| 類型 | 腳本 | 結果目錄 | 檔名前綴 |
|------|------|----------|----------|
| 靜態 | `experiments/phase2_ensemble/static/xgb_oldnew_bankruptcy_year_splits_static.py` | `results/phase2_ensemble/static/` | `xgb_oldnew_ensemble_static_*` |
| 動態 DES | `experiments/phase2_ensemble/dynamic/des/xgb_oldnew_bankruptcy_year_splits_des.py` | `results/phase2_ensemble/dynamic/des/` | `xgb_oldnew_ensemble_des_*` |
| 動態 DCS | `experiments/phase2_ensemble/dynamic/dcs/xgb_oldnew_bankruptcy_year_splits_dcs.py` | `results/phase2_ensemble/dynamic/dcs/` | `xgb_oldnew_ensemble_dcs_*`（列數精簡，見腳本） |

共用邏輯：`experiments/phase2_ensemble/xgb_oldnew_ensemble_common.py`、`xgb_year_split_shared.py`。`python scripts/run/run_all_experiments.py` 會一併觸發股票／醫療 Phase 2；**論文主線完成度**仍以 [研究現況](#研究現況-current-progress) 為準。

**UML／方法圖**（PlantUML）：[UML/README_圖表建議.md](UML/README_圖表建議.md)。與 [研究目標](#研究目標) 內之 UML 對照表呼應。

**`scripts/`**（專案根執行；完整指令：[scripts/README.md](scripts/README.md)）

| 路徑 | 用途 |
|------|------|
| `scripts/data/` | 資料下載、合成資料 |
| `scripts/run/` | 一鍵跑全 phase、多 seed、批次切割 |
| `scripts/analysis/` | 結果彙總、統計檢定 |
| `scripts/plots/` | 通用視覺化、Phase1 baseline 圖表 |
| `scripts/reports/` | 報表匯出（如 Excel） |

**資料**

```text
data/
├── raw/         ← 原始資料（Bankruptcy / Stock / Medical）
├── processed/   ← 預處理後
└── splits/      ← 時序或 CV 切分
```

**實驗結果（節選）**

```text
results/
├── summary_all_datasets.csv
├── phase1_baseline/
├── phase2_ensemble/
│   ├── static/
│   └── dynamic/
│       ├── des/
│       └── dcs/          ← DCS 建池 scaler 與靜態／DES 可能不同，見腳本註解
├── phase3_feature/
├── phase4_analysis/
├── multi_seed/
└── visualizations/
```

詳見 [results/README.md](results/README.md)。

---

## 資料集與時間切割

以下只說明**資料**與 **Old／New／Test** 時間軸。訓練策略（Phase 1）與集成（Phase 2）請讀 [實驗方法說明](#實驗方法說明)；進度與路徑請讀 [研究現況](#研究現況-current-progress)。

| 資料集 | 來源 | 時間範圍 | 切割方式 |
|--------|------|----------|----------|
| Bankruptcy | 公開資料集（如 american_bankruptcy） | 1999–2018 | Chronological 年份窗 |
| Stock | 專案內建／下載 | 依腳本 | Block / 年份 |
| Medical (UCI) | UCI 等 | 依腳本 | Block / 年份 |

實際檔名與路徑以各 `experiments/_shared` 載入邏輯為準；缺檔時可執行 `scripts/data/` 內對應下載腳本。

### Old／New（O+N）怎麼切、為何要分兩段

**總則（三塊時間軸）**  
實驗腳本把有**年份**的樣本，依**時間先後**切成三塊（互不重叠）：

1. **Old（歷史／舊窗）**：較早的一段時間，代表「過去環境下」累積的資料分佈。  
2. **New（新營運／新窗）**：接在 Old 之後、且在**固定測試窗之前**的一段時間，代表「較新環境」下的分佈（可能與 Old 有 **covariate shift** 或非平穩）。  
3. **Test（測試）**：時間上最後、且**只保留給最終評估**；**不參與訓練、不參與 validation 調參**，避免洩漏。

**為何要分 Old 與 New（而不是只併成一池）**  
- 對應研究問題：**非平穩**與**持續學習**情境——「舊資料上學的規則」未必在「新資料分佈」上仍最適用；因此需要 **Old-only**、**New-only**、**兩段合併（Retrain）**、**先舊後新（Finetune）** 等對照，定義見 [Phase 1 Baseline](#method-phase1-baseline)。  
- **集成**階段再進一步：在 Old／New **各自**做欠採樣／過採樣／混合得到 **6 個槽位**，再依 [Phase 2 集成](#method-phase2-ensemble) 做靜態／DES／DCS。

**怎麼「分」出界線（依資料集）**  
界線由一個參數 **`old_end_year`**（或醫療的 `old_end_year` 對應標籤）決定：**該年（含）以前** → Old；**次年起到訓練窗結束年** → New。實作見 `experiments/_shared/common_bankruptcy.py`、`common_dataset.py`。

| 資料集 | 訓練窗（Train）與測試窗（Test） | Old／New 規則（摘要） |
|--------|----------------------------------|------------------------|
| **破產（US）** | Train：`fyear` ∈ 1999–2014；**Test：2015–2018 固定** | `Old`：`fyear ≤ old_end`；`New`：`old_end < fyear ≤ 2014`。`split_a+b` 表示約 **a 年 Old + b 年 New**（在 16 年訓練窗內滑動 `old_end`，共 **14 組**，見 `YEAR_SPLITS`）。 |
| **股票（SPX）** | Train：2001–2016；**Test：2017–2020 固定** | `Old`：2001–`old_end`；`New`：`old_end`+1–2016。同樣 7 組對稱切割（見 `STOCK_YEAR_SPLITS`）。 |
| **醫療** | Train：1999–2006；**Test：2007–2008 固定** | `Old`：`year ≤ old_end`；`New`：`old_end`+1–2006。7 組（見 `MEDICAL_YEAR_SPLITS`）。 |

**多組 O+N 切割（例如破產 `split_2+14` … `split_14+2`）的目的**  
在**同一個 Test** 下，改變 **Old 與 New 的相對長度**（歷史長、新營運短 → 反之），看 **baseline／集成／動態方法** 是否依賴「哪一段訓練資料較多」而表現不穩，屬於**實證設計上的敏感度分析**，而非單一隨機切分。

**與 [docs/研究方向.md](docs/研究方向.md) 裡例句的關係**  
文件中的「1999–2011／2012–2014／2015–2018」是**概念範例**（a 類切割）；本 repo 主實驗為 **US 破產**上 **連續滑動 `old_end`** 的 **14 組**年份窗（`split_1+15` … `split_14+2`），規則與上表一致。

**Phase 1 四種 baseline**（Old／New／Retrain／Finetune）的訓練定義見 [Phase 1 Baseline](#method-phase1-baseline)。

---

## 實驗方法說明

本節集中說明**怎麼做實驗**，與「[資料集與時間切割](#資料集與時間切割)」（資料從哪來、怎麼依年份切）及「[研究現況](#研究現況-current-progress)」（跑完沒、結果路徑）分開，避免 baseline 與集成敘事混在一起。

<a id="method-phase1-baseline"></a>
### Phase 1：Baseline（單一基學習器）

> **三資料集並非同一套 Phase 1 定義**：僅 **破產** `bankruptcy_year_splits_xgb.py` 實作 **Retrain**（Old+New 全量合併重訓）與 **Finetune**（Old 訓練後以 New 接續同一 booster）。**股票、醫療** 的 `stock_year_splits_xgb.py`、`medical_year_splits_xgb.py` 為 **Old／Old+New（平衡混合）／New** 三種，**沒有**與破產同名的 Retrain／Finetune；若論文要三資料集嚴格可比，需另補程式或改寫腳本。

與 [docs/研究方向.md](docs/研究方向.md) 對齊時，**單一基學習器**在「每個年份切割」下會比較下列 **四種訓練策略**（再各自 × **四種採樣**：none／undersampling／oversampling／hybrid）。**完整實作**見 **破產** `bankruptcy_year_splits_xgb.py`、`bankruptcy_year_splits_torch_mlp.py`；流程圖見 `UML/baseline_flow.puml`。

| 策略 | 怎麼 train | 備註 |
|------|------------|------|
| **Old** | 僅使用 **歷史窗（Old）** 的樣本訓練；特徵於訓練折 **fit scaler**，於固定 **Test** 上評估。 | 只在舊分佈上學。 |
| **New** | 僅使用 **新營運窗（New）** 的樣本訓練；train/val 切在 New 內，**Test** 固定。 | 只在新分佈上學。 |
| **Retrain** | 將 **Old 與 New 全量串接**（concat），在合併後訓練集上 **從頭訓練** 一個新模型。 | 研究方向之「歷史＋新營運合併重訓」；**非**股票／醫療的「平衡混合 Old+New」。 |
| **Finetune** | **第一段**只在 **Old** 上訓練；**第二段** **同一個模型**在 **New** 上 **接續訓練**（XGB：`xgb_model=`；Torch MLP：`continue_fit`）。閾值在 **New 的 validation** 上依 F1 搜尋，再套到 Test。 | 「先舊再新」微調；**不是**另訓一個獨立 New-only。 |

#### 案例：破產 US 資料、`split_8+8`（程式裡 `old_end_year=2006`）

與 `experiments/_shared/common_bankruptcy.py` 的 `YEAR_SPLITS` 一致：**測試窗固定**為 **2015–2018**（`fyear`，完全不參與訓練）；**訓練窗**為 **1999–2014**。在 `split_8+8` 這一組切割下：

| 區塊 | 年份（`fyear`） | 白話 |
|------|-----------------|------|
| **Old** | 1999–2006（共 8 個會計年度） | 「較早的歷史窗」 |
| **New** | 2007–2014（共 8 個會計年度） | 「較近的新營運窗」 |
| **Test** | 2015–2018 | 最終只拿來算指標 |

**Retrain 在這一組切割裡實際做什麼**  
把 **Old ∪ New** 的列 **全部** 串成**一張表**（`pd.concat([X_old, X_new])`，時間上就是 **1999–2014 所有樣本**），再在這張「已合併的訓練池」上切 80/20、fit scaler、採樣、然後 **從零** `fit` 一株新的 XGB（**沒有** `xgb_model=` 接續）。可以想成：**假設從一開始就拿到 1999–2014 整包資料，用同一套流程重練一個模型**；樹的結構完全由這次合併後的資料決定，**不保留**「先在 Old 上學到」的 booster。

**Finetune 在這一組切割裡實際做什麼**  
- **第一段**：只在 **Old（1999–2006）** 的 train fold（經採樣後）上訓練，得到 **第一版 booster**。  
- **第二段**：**同一顆 booster**，再用 **New（2007–2014）** 的 train fold（經採樣後）以 `xgb_model=` **接續訓練**，等於在舊樹結構上繼續長／調整。  
- **閾值**：只在 **New 的 validation** 上依 F1 搜尋，再套到 **2015–2018**。  
可以想成：**先適應 1999–2006 的規則，再用 2007–2014 把同一個模型往較新環境推進**；與 Retrain「整段 1999–2014 一次重練、不繼承第一段」不同。

**和「只選一邊」的對照**  
- **Old-only**：只看 1999–2006，**沒有** 2007–2014。  
- **New-only**：只看 2007–2014，**沒有** 1999–2006。  
- **Retrain**：**兩段資料都有**，但是 **concat 後一次重練**，不是兩階段接續。  
- **Finetune**：**兩段資料分兩階段進同一顆模型**，順序是 Old → New。

其他六組切割（例如 `split_14+2`：`Old`=1999–2012、`New`=2013–2014）只是 **Old／New 長度比例**不同，**Retrain／Finetune 的程式邏輯不變**，差別在樣本數與分佈。

**推論與閾值（Old／New／Retrain 共用流程）**：在**該方法可用的訓練資料**上先做 **80%／20%** `train_test_split`（`stratify` 若可行）；**標準化（scaler）只在 train fold 上 fit**，再 transform validation 與 **固定 Test**；採樣（none／under／over／hybrid）只套在 **train fold**；以 **validation 上之 F1** 搜尋最佳分類 **閾值**，最後在 **Test** 上算指標。

#### Retrain、Fine-tuning 在程式裡的實際作法（破產 XGB，與研究方向對齊）

以下對應 `experiments/phase1_baseline/bankruptcy_year_splits_xgb.py`；Torch MLP 邏輯見同目錄 `bankruptcy_year_splits_torch_mlp.py`（`continue_fit` 與閾值選在 New val）。

**Retrain（Re-training）**

1. **合併訓練集**：`pd.concat([X_old, X_new])`、`concat` 標籤，**不抽樣、不截斷**，樣本數 = `len(Old)+len(New)`。  
2. **與 Old／New 單段相同的 `_train_eval`**：在合併後的矩陣上切 train/val → **scaler 僅在 train fold fit** → 採樣僅在 train fold → **全新** `XGBoostWrapper(...).fit`（**沒有** `xgb_model` 接續）。  
3. 每一種採樣策略各訓練 **一個獨立模型**；閾值由該次 train/val 決定，Test 僅評估。

**Fine-tuning（Finetune）— `_finetune_eval`**

1. **分別切 Old、New**：Old、New **各自**做 80/20 train/val（stratify 若可行）。  
2. **單一特徵空間（關鍵）**：**標準化只在 Old 的 train fold 上 fit**；**Old val、New train、New val、Test** 皆以 **同一組 scaler 參數** transform（`fit=False`）。這樣第二段仍在「第一段學到的特徵尺度」上更新，而不是在 New 上重 fit 一個新 scaler。  
3. **第一段**：在 **採樣後的 Old train** 上 `model.fit`（全新 booster）。  
4. **第二段**：在 **採樣後的 New train** 上呼叫 **同一個** `xgboost` 的 `fit(..., xgb_model=model.model.get_booster())`，接續訓練；**`scale_pos_weight` 依第二段採樣後的標籤**重新計算（與類別比例對齊）。  
5. **閾值**：**只在 New 的 validation** 上算 `predict_proba` 並搜尋最佳 F1 閾值（呼應「微調階段以新資料驗證為主」）；**Test** 僅用此閾值做最終指標。  

**與「只用 New」的差異**：Finetune **不是** `_train_eval(X_new, ...)`；它先吃 Old 再 **同一顆 booster** 吃 New，且閾值選在 **New val**，不是 Old val。

**股票／醫療** Phase 1 為 **Old／Old+New（平衡混合）／New** 三種，**無**上表之 **Retrain**／**Finetune** 同名流程，見本節開頭 blockquote。

<a id="method-phase2-ensemble"></a>
### Phase 2：集成（靜態／DES／DCS）

| 類型 | 摘要 |
|------|------|
| 池 | Old／New 各 3 個 XGB（under／over／hybrid），共 6 槽位 |
| 靜態 | 子集內正類機率 **平均** + validation 上 **F1 最佳閾值** |
| DES／DCS | 在鄰域內 **選模或加權**，仍以 **機率** 輸出（見下「集成怎麼做」） |

**Old / New 各 3 個基學習器**（Under / Over / Hybrid 採樣）：

- **Old**：歷史期訓練；**New**：新營運期訓練。
- **靜態集成**：機率平均 + 驗證集 F1 最佳閾值；長表會枚舉多種 k 子集與組合（列數與 DES 相近）。
- **動態 DES**（Phase 2 XGB）：在 New scaler 特徵空間之 kNN 鄰域上，採 KNORA-E / KNORA-U / DES-KNN 等；實作見 `experiments/phase2_ensemble/xgb_oldnew_ensemble_common.py`（與 `dynamic/des/` 腳本共用）。
- **動態 DCS**（Phase 2 XGB）：固定 6 個 XGB 池模型，在測試樣本鄰域內以 OLA／LCA（可選時間加權）競爭選模；輸出為「每個年份窗 × 少數變體」的精簡長表，而非與靜態相同的 k 子集枚舉。

集成規模：2～6 模型子集（論文設定含至少一 Old、一 New）。DCS 實驗固定使用完整 6 槽位池。

#### 集成怎麼做

**1. 基學習器從哪來（六槽位池）**  
在每一組 **Old／New 年份切割**下，於 **Old** 與 **New** 訓練窗上**各自**訓練三種採樣策略的 **XGBoost**（欠採樣／過採樣／混合），共 **3（Old）+ 3（New）= 6** 個獨立模型。每個模型對樣本輸出 **正類機率** `P(y=1|x)`，供後續融合使用。

**2. 靜態集成（等權 soft voting）**  
對選定的子集（例如某 2～6 個模型），在 **validation** 與 **test** 上皆對各模型之正類機率做 **算術平均**（等權 **soft voting**：先對機率平均，再決定類別）。接著**不是**固定以 0.5 切分：在 **validation** 上對「平均後的機率」做 **F1 最大化的閾值格搜**，再將該閾值套到 **test**。核心實作見 `ensemble_metrics_with_threshold()`（`experiments/phase2_ensemble/xgb_oldnew_ensemble_common.py`）。  
抽象元件 `EnsembleCombiner`（`src/ensemble/selector.py`）亦描述為「**平均 soft voting**」。

**3. 動態 DES（鄰域上再選模／加權，仍輸出機率）**  
以 **驗證集** 當 **DSEL**，在 **New 段 scaler 後的特徵空間** 做 **kNN**（驗證點上為 **leave-one-out** 鄰居，避免洩漏）。依 **KNORA-E**／**KNORA-U**／**DES-KNN** 等規則，用鄰居上「硬預測是否正確」篩選或加權模型，再對 **test** 上被選模型的 **正類機率** 做平均或加權平均（見 `_dynamic_proba_rows()` 同上檔案）。最後同樣在 validation 上選最佳 **threshold** 再評 test。

**4. 動態 DCS（鄰域競爭選模）**  
在 **DSEL（通常為 Old+New 合併）** 特徵空間對每個 **test** 樣本找 k 鄰居，以 **OLA** 或 **LCA** 等為池中每個模型算**競爭分數**；取 **分數最高** 的模型（若同分則多個一併納入），對這些模型在該樣本上的 **正類機率取平均** 作為最終分數（`experiments/_shared/common_dcs.py` 之 `run_dcs_from_pool_models`）。可選 **New 側時間加權**（`time_weight_new`）。

**5. 與「硬投票（hard voting）」的差異**  
本專案 **沒有** 採用「多數決類別標籤」的 **VotingClassifier(voting='hard')** 式集成。融合層一律基於 **機率**（soft voting 或動態加權後的機率），以利與 **AUC**、**F1 閾值** 等指標一致；若需對照文獻中的 hard voting，需另擴充實驗。

---

## 評估指標

> 類別不平衡：**勿單獨報告 Accuracy**

- **AUC-ROC** — 排序能力  
- **F1** — 精確度與召回平衡  
- **G-Mean** — 敏感度與特異度幾何平均  
- **Recall** — 少數類召回  

---

## Quick Start

```powershell
# 1. 虛擬環境
.\venv\Scripts\activate

# 2. 依賴
pip install -r requirements.txt

# 3. 資料（必要時）
#    python scripts\data\download_real_medical_data.py
#    其餘見 scripts\README.md

# 4. 一鍵依序跑各 phase（耗時長；會含股票／醫療 Phase 2，與 README「進度」是否完成無關）
python scripts\run\run_all_experiments.py

# 或只跑單一實驗
python experiments\phase1_baseline\bankruptcy_year_splits_xgb.py

# Phase 2 破產集成（靜態 → DES → DCS，可只跑其中一支）
python experiments\phase2_ensemble\static\xgb_oldnew_bankruptcy_year_splits_static.py
python experiments\phase2_ensemble\dynamic\des\xgb_oldnew_bankruptcy_year_splits_des.py
python experiments\phase2_ensemble\dynamic\dcs\xgb_oldnew_bankruptcy_year_splits_dcs.py

# 5. 彙總三資料集結果
python scripts\analysis\compare_all_results.py
```

Windows 批次檔：[run_all_experiments.bat](run_all_experiments.bat)。

更多說明：[experiments/README.md](experiments/README.md)、[docs/STRUCTURE.md](docs/STRUCTURE.md)。

---

## Import 範例

```python
from src.data import DataLoader, DataPreprocessor, DataSplitter, ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool, XGBoostWrapper
from src.ensemble import DynamicEnsembleSelector, EnsembleCombiner
from src.features import FeatureSelector
from src.evaluation import compute_metrics
from src.utils import get_logger, set_seed
```

---

## 技術棧

- **Python** 3.10+（建議與 `requirements.txt` 一致）
- **梯度提升**：LightGBM、XGBoost  
- **神經網路**：PyTorch（Phase1 Torch MLP baseline）  
- **傳統 ML / 指標**：scikit-learn  
- **不平衡**：imbalanced-learn（SMOTE、SMOTEENN 等）  
- **DES**：`src/ensemble/selector.py`（KNORA-E 等）；文獻級方法可參考 `deslib`（已列於 requirements）  
- **設定**：PyYAML、python-dotenv  

---

## 文件索引

| 文件 | 說明 |
|------|------|
| [scripts/README.md](scripts/README.md) | 工具腳本分類與指令 |
| [experiments/README.md](experiments/README.md) | 實驗目錄與執行慣例 |
| [docs/STRUCTURE.md](docs/STRUCTURE.md) | 完整目錄結構 |
| [docs/RESEARCH_SPEC.md](docs/RESEARCH_SPEC.md) | 研究方向規格 |
| [docs/研究方向.md](docs/研究方向.md) | 研究課題說明（中文） |
| [docs/研究方向對照表.md](docs/研究方向對照表.md) | 實作與論文對照 |
| [UML/README_圖表建議.md](UML/README_圖表建議.md) | PlantUML 流程圖與論文用圖建議 |
| [results/README.md](results/README.md) | 結果目錄說明 |
| [.agents/rules/rules.md](.agents/rules/rules.md) | 專案規範（路徑、import） |
