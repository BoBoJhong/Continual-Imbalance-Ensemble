---
trigger: always_on
---

# 專案規範 (Project Rules)

> Antigravity 在每次對話開始時自動讀取此文件。

---

## 研究背景

本專案為碩士論文研究，主題：**非平穩環境下類別不平衡之持續學習集成框架**。
詳細研究方向見 `docs/RESEARCH_SPEC.md`，指導教授需求見 `docs/TEACHER_REQUIREMENTS_CHECKLIST.md`。

---

## Rule 1 — 目錄職責 (Directory Responsibilities)

| 目錄 | 用途 | 限制 |
| :--- | :--- | :--- |
| `src/` | **可 import 的函式庫**。資料處理、模型、集成、特徵選擇、評估指標 | 禁止放實驗邏輯、禁止硬編碼路徑 |
| `experiments/` | **可直接執行的實驗腳本**，依研究階段分子目錄 | 腳本只 import `src/` 及 `experiments._shared`，不含可重用邏輯 |
| `scripts/` | **工具腳本**：一鍵執行、資料下載、結果比較 | 非實驗邏輯 |
| `config/` | **所有 YAML 設定檔** | 禁止在 Python 程式碼中硬編碼超參數 |
| `data/raw/` | **原始資料**（.gitignore 排除） | 禁止修改，只讀 |
| `data/processed/` | **前處理後資料** | 由 `src/data/preprocessor.py` 輸出 |
| `results/` | **實驗結果 CSV** | 由實驗腳本自動輸出，禁止手動編輯 |
| `docs/` | **文件**（Markdown） | 命名規則：UPPER_CASE.md |
| `.agent/` | **Antigravity 設定**（workflows、rules） | 不含業務邏輯 |

根目錄：只保留 `README.md`、`requirements*.txt`、`*.bat`、`.gitignore`。

`experiments/` 子目錄慣例：
- `_shared/`：資料載入 & 流程封裝，**不可直接執行**
- `phase1_baseline/`、`phase2_ensemble/static/`、`phase2_ensemble/dynamic/des/`、`phase2_ensemble/dynamic/dcs/`、`phase3_feature/`、`phase4_analysis/`：各階段執行腳本
- 每支腳本跨全部資料集（Bankruptcy / Stock / Medical）一次執行完畢

---

## Rule 2 — `src/` 模組結構

每個子模組必須有 `__init__.py` 並 export 公開 API：

```text
src/
├── data/         → DataLoader, DataPreprocessor, DataSplitter, ImbalanceSampler
├── models/       → LightGBMWrapper, XGBoostWrapper, ModelPool
├── ensemble/     → DynamicEnsembleSelector, EnsembleCombiner
├── features/     → FeatureSelector
├── evaluation/   → compute_metrics, print_results_table
└── utils/        → get_logger, set_seed, get_config_loader
```

**Import 範例**（正確用法）：

```python
from src.ensemble import DynamicEnsembleSelector
from src.evaluation import compute_metrics
from src.features import FeatureSelector
```

---

## Rule 3 — 命名規範 (Naming Conventions)

| 對象 | 規範 | 範例 |
| :--- | :--- | :--- |
| 實驗腳本 | `{description}.py`，放在對應 phase 子目錄 | `retrain.py`, `standard.py` |
| 共用函式檔 | `common_{scope}.py`，放在 `_shared/` | `common_bankruptcy.py` |
| 結果目錄 | `results/phase?_???/`，小寫底線分隔 | `results/phase1_baseline/` |
| 文件檔案 | `UPPER_CASE.md` | `EXECUTION_GUIDE.md` |
| 設定檔 | `snake_case_config.yaml` | `des_config.yaml` |
| Python 類別 | PascalCase | `DynamicEnsembleSelector` |
| Python 函式 | snake_case | `compute_metrics()` |

---

## Rule 4 — 評估指標規範

由於資料具**類別不平衡**特性，**禁止單獨使用 Accuracy**。
所有實驗必須輸出：**AUC-ROC、F1-Score、G-Mean、Recall**。

統一使用 `src/evaluation/metrics.py` 的 `compute_metrics()` 函式。

---

## Rule 5 — 資料切割規範

支援兩種模式（在 `src/data/splitter.py` 實作）：

- **Mode A（chronological）**：依時間序列切分 (例如 Historical / New / Test)。
- **Mode B（block_cv）**：5-fold 區塊交叉驗證。

預設使用 **Mode B（block_cv）**，除非資料集有明確的時間屬性，且該實驗設計強烈要求依時間推進（例如原有的 US Bankruptcy 1999-2018 資料）。

---

## Rule 6 — 設定檔規範 (Configuration)

所有實驗與模型超參數必須定義在 `config/` 下的 YAML 檔案中。

- `config/` 目錄下應有清晰的參數分類，例如 `config/des_config.yaml`, `config/lgb_config.yaml`。
- 如果新增實驗或模型，**嚴禁**在 `experiments/*.py` 或 `src/models/*.py` 內硬編碼超參數。

---

## Rule 7 — 新增資料集擴展 SOP

若未來需新增其他資料集（例如 `new_dataset`），請嚴格遵守以下擴展流程：

1. **資料放置**：原始資料存放於 `data/raw/new_dataset/`。
2. **共用讀取腳本**：在 `experiments/_shared/` 新增 `common_new_dataset.py`，封裝載入 & 前處理邏輯。
3. **整合現有腳本**：在各 phase 的現有腳本（`retrain.py`、`standard.py` 等）中加入新資料集的迴圈項目，**不新增獨立腳本**。
4. **結果輸出**：結果統一輸出至對應的 `results/phase?_???/`，命名格式為 `{new_dataset}_{description}.csv`。
5. **更新 README**：同步更新根目錄 `README.md` 的資料集表格。

---

## Rule 8 — 實驗腳本細粒度控制 (Fine-Grained Control)

切割維度為**方法 (method)**，而非資料集：

- Phase 2 依取樣策略分檔：`undersampling.py` / `oversampling.py` / `hybrid.py`
- Phase 3 依演算法分子目錄：`des/standard.py` / `des/advanced.py` / `dcs/comparison.py`
- 每支腳本跨全部資料集執行，輸出結果命名格式：`{dataset}_{description}.csv`
- **禁止**為單一資料集新增獨立腳本（如 `bankruptcy_baseline.py`），應整合進對應 phase 腳本的迴圈

---

## Rule 9 — 說明文件全面同步規範 (Documentation Synchronization)

為了維持專案的極高可讀性，本專案高度依賴總目錄的 `README.md` 以及四個核心子目錄的局部 `README.md`：

1. **根目錄 `README.md`**：專案入口，記載總體實驗進度表、目錄導航與最佳結果。
2. **四個子模組導航**：`experiments/`, `results/`, `src/`, `config/`。

**強制規定**：

- 凡是有「新增實驗腳本」、「產出新階段結果」或「加入新資料集」時，**必須優先更新根目錄的 `README.md`**（尤其是實驗完成狀況表）。
- 若在四個核心目錄內進行「新增檔案」、「刪除腳本」或「大幅修改架構」時，**必須同步更新其對應目錄下的局部 `README.md`**，確保雙重文件防護永遠不過期。

---

## Rule 10 — 嚴格的重現性保障 (Strict Reproducibility)

為了確保任何時候執行程式都能得到完全相同的學術實驗結果：

1. **強制固定亂數種子**：所有實驗腳本在 `main()` 函數的頭兩行內，**必須強制呼叫 `set_seed(42)`**。不允許任何具有隨機性的模型（如 LightGBM、SMOTE）在沒有固定 Seed 的情況下運作。
2. **環境依賴鎖定**：若開發過程中使用了 `pip install` 安裝了新的套件，必須 **當下同步更新 `requirements.txt`**。

---

## Rule 11 — 實驗追蹤與日誌紀律 (Logging Discipline)

為了方便日後查證每個參數設定下的模型表現與錯誤追蹤：

1. **全面禁用 `print()`**：在 `src/` 與 `experiments/` 內，嚴禁使用 `print()` 來印出重要資訊（進度條除外）。
2. **統一使用 Logger**：所有執行進度、模型訓練設定、警告訊息，都必須統一呼叫 `src.utils.get_logger()` 來輸出。不僅要顯示在終端機，更要**自動寫入檔案日誌**。

---

## Rule 12 — 實驗腳本的「純淨度」限制 (Experiment Script Purity)

這是為了防止實驗邏輯與資料處理混雜在一起，導致程式碼難以閱讀與維護：

1. `experiments/phase?_*/` 下的主實驗腳本，**嚴禁撰寫任何「特徵工程」、「遺失值填補」或「異常值剃除」的底層邏輯**。
2. 實驗腳本的作用應極度單純，僅限於：**呼叫 `_shared/common_*.py` 取得資料 ➔ 呼叫模型池 ➔ 計算指標 ➔ 報表匯出**。
3. 任何資料處理邏輯必須下沉到 `src/data/` 或 `experiments/_shared/common_*.py` 中維護。

---

## Rule 13 — 實驗成果與洞察彙整 (Research Summary Synchronization - **STRICT ENFORCEMENT**)

為了確保論文撰寫時有最即時、最詳盡的參考依據，所有最新的實驗數據、比較圖表與結果分析，都必須統整至 `docs/reserch_summary.md` 中。**絕對不允許有任何實驗已經執行，但數據或方法卻未同步紀錄於該文件中的情況發生。**

1. **數據與流程同步**：當實驗腳本產出新的數據結果，或是重跑實驗後數據發生變動，必須同步將**最完整、最詳細的數據指標（包含各個 Sampling 策略、各種細度比較數值表）與實驗步驟流程**摘要至該文件中。
   - **涵蓋範圍要求**：三大資料集 (Bankruptcy, Stock, Medical) 均必須詳列完整的 (1) Baseline 與靜態集成 (2) 動態集成選擇 (DES) (3) 進階研究如特徵選擇、加權 DES 與比例衰退研究。
2. **洞察紀錄強制填寫 (No Placeholders)**：除了冰冷的數據表，文件中的「實驗說明與洞察」區塊**絕對不允許留下 `[描述...]` 這種待填寫的佔位符**。每次更新數據時，AI 務必根據最新實驗的 AUC, F1 等交叉數據，直接幫忙填入具體的分析結論（例如：指出何種策略勝出、特徵選擇對該資料集的實際負面/正面影響等），以確保總結文件能立刻被應用於論文撰寫。

---

## 資料集位置

```text
data/raw/
├── bankruptcy/   → american_bankruptcy_dataset.csv
├── stock/        → stock_spx.csv, stock_dji.csv, stock_ndx.csv
└── medical/
    ├── diabetes130/  → diabetes130_medical.csv   (真實 UCI)
    └── synthetic/    → synthetic_medical_data.csv (備用)
```

下載說明見 `docs/DATASET_DOWNLOAD_GUIDE.md`。

---

## 常用指令

```powershell
# 啟動虛擬環境
.venv\Scripts\activate

# 執行所有實驗（依 phase 順序）
python scripts\run\run_all_experiments.py

# 執行單一腳本
python experiments/phase1_baseline/retrain.py

# 查看彙總結果
python scripts\analysis\compare_all_results.py
```

## project_root 深度規範

| 腳本位置 | `project_root` 寫法 |
|----------|--------------------|
| `_shared/`, `phase1_baseline/`, `phase3_feature/` 等（深度 3） | `Path(__file__).parent.parent.parent` |
| `phase2_ensemble/static/` | `Path(__file__).resolve().parent.parent.parent.parent` |
| `phase2_ensemble/dynamic/des/`, `phase2_ensemble/dynamic/dcs/` | `Path(__file__).resolve().parent.parent.parent.parent.parent` |
| `scripts/data/`, `scripts/run/`, `scripts/analysis/`, `scripts/plots/`, `scripts/reports/` | `Path(__file__).resolve().parent.parent.parent` |
