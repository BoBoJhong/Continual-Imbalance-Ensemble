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
|------|------|------|
| `src/` | **可 import 的函式庫**。資料處理、模型、集成、特徵選擇、評估指標 | 禁止放實驗邏輯、禁止硬編碼路徑 |
| `experiments/` | **可直接執行的實驗腳本**（`python experiments/XX_*.py`） | 只 import `src/`，不含可重用邏輯 |
| `scripts/` | **工具腳本**：一鍵執行、資料下載、結果比較 | 非實驗邏輯 |
| `config/` | **所有 YAML 設定檔** | 禁止在 Python 程式碼中硬編碼超參數 |
| `data/raw/` | **原始資料**（.gitignore 排除） | 禁止修改，只讀 |
| `data/processed/` | **前處理後資料** | 由 `src/data/preprocessor.py` 輸出 |
| `results/` | **實驗結果 CSV** | 由實驗腳本自動輸出，禁止手動編輯 |
| `docs/` | **文件**（Markdown） | 命名規則：UPPER_CASE.md |
| `.agent/` | **Antigravity 設定**（workflows、rules） | 不含業務邏輯 |

根目錄：只保留 `README.md`、`requirements*.txt`、`*.bat`、`.gitignore`。

---

## Rule 2 — `src/` 模組結構

每個子模組必須有 `__init__.py` 並 export 公開 API：

```
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
|------|------|------|
| 實驗腳本 | `NN_dataset_description.py`（兩位數編號） | `01_bankruptcy_baseline.py` |
| 共用函式檔 | `common_dataset.py` | `common_bankruptcy.py` |
| 結果目錄 | 小寫，底線分隔 | `results/feature_study/` |
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

1. **資料放置**：將原始資料存放於 `data/raw/new_dataset/data.csv`。
2. **共用讀取腳本**：於 `experiments/` 專案目錄下新增 `common_new_dataset.py`，封裝該資料集獨立的前處理與載入邏輯。
3. **實驗命名與腳本**：新增該資料集的執行腳本並依序編號，例如 `11_new_dataset_baseline.py`。
4. **結果輸出**：確保該資料集的結果統一存放於 `results/new_dataset/`，並在 README 更新對應紀錄。

---

## Rule 8 — 實驗腳本細粒度控制 (Fine-Grained Control)

為了確保實驗數據的可讀性與未來擴展的彈性，所有資料集的實驗腳本必須保持**細粒度切割**：

- **基準實驗 (Baseline)** 與 **靜態集成 (Static Ensemble)** 必須拆分為兩支獨立的腳本。
- 各實驗必須能對應與輸出獨立的結果檔案 (例如 `baseline.csv` 與 `ensemble.csv`)，避免混合導致判讀困難與後續抽換模組的困擾。

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

1. 所有位於 `experiments/` 並且以編號開頭的主實驗腳本（如 `01_*.py`, `13_*.py` 等），**嚴禁撰寫任何「特徵工程 (Feature Engineering)」、「遺失值填補 (Imputation)」或「異常值剃除」的底層邏輯**。
2. 實驗腳本的作用應極度單純，僅限於：**載入處理好的資料 ➔ 呼叫模型池 ➔ 計算指標 ➔ 報表匯出**。
3. 任何針對資料欄位的變動計算，必須下沉 (Push-down) 到 `src/data/preprocessor.py` 或對應的 `common_*.py` 中統一維護。

---

## 資料集位置

```
data/raw/
├── bankruptcy/   → american_bankruptcy_dataset.csv (或 data.csv)
├── stock/        → data.csv
└── medical/      → data.csv
```

下載說明見 `docs/DATASET_DOWNLOAD_GUIDE.md`。

---

## 常用指令

```powershell
# 啟動環境
.\\venv\\Scripts\\activate

# 執行所有實驗
python scripts\\run_all_experiments.py

# 查看結果
python scripts\\compare_all_results.py

# 多 seed 重現性實驗
python scripts\\run_multi_seed.py
```
