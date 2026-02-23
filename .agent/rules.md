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

- **Mode A（chronological）**：1999-2011 Historical / 2012-2014 New / 2015-2018 Test
- **Mode B（block_cv）**：5-fold，Fold 1+2 Historical / 3+4 New / 5 Test

預設使用 **Mode B（block_cv）**，除非資料集有明確年份欄且使用 US 1999-2018 資料。

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
