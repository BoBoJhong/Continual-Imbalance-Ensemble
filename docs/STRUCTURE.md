# 專案結構說明

本文件說明目前程式碼與檔案的組織方式，方便維護與擴充。

---

## 一、目錄總覽

```
Continual-Imbalance-Ensemble/
├── config/                    # 設定檔（YAML）
├── data/                      # 資料目錄（raw/processed 依 .gitignore）
├── docs/                      # 文件
│   ├── STRUCTURE.md          # 本檔：結構說明
│   ├── EXPERIMENT_CHECKLIST.md  # 實驗對照表（老師要求 vs 狀態）
│   └── ...
├── experiments/               # 實驗腳本（依編號執行）
│   ├── common_bankruptcy.py  # 共用：Bankruptcy 載入與切割
│   ├── common_dataset.py     # 共用：Stock / Medical 載入與切割
│   ├── common_des.py         # 共用：KNORA-E DES 邏輯
│   ├── 01_bankruptcy_baseline.py
│   ├── 02_bankruptcy_ensemble.py
│   ├── 03_bankruptcy_des.py
│   ├── 04_bankruptcy_feature_selection_study.py
│   ├── 05_stock_baseline_ensemble.py
│   ├── 06_medical_baseline_ensemble.py
│   ├── 07_stock_des.py
│   └── 08_medical_des.py
├── scripts/                  # 工具腳本（非單一實驗）
│   ├── run_all_experiments.py    # 一鍵執行 01~08
│   ├── compare_baseline_ensemble.py  # Bankruptcy 合併表
│   ├── compare_all_results.py      # 三資料集彙總
│   ├── run_multi_seed.py           # 多 seed mean±std
│   └── download_*.py               # 資料下載（可選）
├── src/                      # 核心程式庫
│   ├── data/                 # 資料：loader, preprocessor, splitter, sampler
│   ├── models/               # 模型：LightGBM, XGBoost, ModelPool
│   └── utils/                # 工具：logger, config, seed
├── results/                  # 實驗結果（CSV，依 .gitignore 可不提交）
├── logs/                     # 日誌（依 .gitignore）
├── examples/                 # 示範與測試用
├── tests/                    # 單元測試
├── EXECUTION_GUIDE.md        # 執行步驟與流程
├── run_experiment.bat        # Windows：單跑 01
└── run_all_experiments.bat   # Windows：一鍵跑 01~08
```

---

## 二、experiments/ 說明

| 檔案 | 用途 | 依賴共用模組 |
|------|------|----------------|
| **common_bankruptcy.py** | Bankruptcy 資料載入、5-fold block CV 切割、前處理 | DataSplitter, DataPreprocessor |
| **common_dataset.py** | Stock / Medical 載入、切割、前處理 | 同上 |
| **common_des.py** | KNORA-E 風格 DES（建池 → DSEL → 動態選擇） | ModelPool |
| **01_bankruptcy_baseline.py** | Baseline：retrain / finetune / ensemble_old | common_bankruptcy |
| **02_bankruptcy_ensemble.py** | 靜態 ensemble（2/3/4/5/6 模型組合） | common_bankruptcy |
| **03_bankruptcy_des.py** | DES (KNORA-E) | common_bankruptcy, common_des |
| **04_bankruptcy_feature_selection_study.py** | Study II：有/無特徵選擇 | common_bankruptcy |
| **05_stock_baseline_ensemble.py** | Stock baseline + ensemble | common_dataset |
| **06_medical_baseline_ensemble.py** | Medical baseline + ensemble | common_dataset |
| **07_stock_des.py** | Stock DES | common_dataset, common_des |
| **08_medical_des.py** | Medical DES | common_dataset, common_des |

**切割模式**：各實驗頂端有 `SPLIT_MODE = "block_cv"`（5-fold：1+2 歷史、3+4 新營運、5 測試）；可改為 `"random"` 使用 60-20-20 隨機切。

---

## 三、scripts/ 說明

| 腳本 | 用途 | 產出 |
|------|------|------|
| **run_all_experiments.py** | 依序執行 01→02→…→08 | 各實驗的 results/、logs/ |
| **compare_baseline_ensemble.py** | 合併 Bankruptcy 的 baseline + ensemble + DES | results/bankruptcy_all_results.csv |
| **compare_all_results.py** | 彙總三資料集所有方法 | results/summary_all_datasets.csv |
| **run_multi_seed.py** | Bankruptcy baseline 多 seed 重跑 | results/baseline/bankruptcy_baseline_mean_std.csv |

---

## 四、src/ 說明

| 模組 | 內容 |
|------|------|
| **src.data** | DataLoader, DataPreprocessor, DataSplitter, ImbalanceSampler |
| **src.models** | LightGBMWrapper, XGBoostWrapper, ModelPool |
| **src.utils** | get_logger, get_config_loader, set_seed |

實驗與腳本皆從**專案根目錄**執行，並依賴 `sys.path` 或 `PYTHONPATH` 包含專案根目錄。

---

## 五、config/ 說明

- **base_config.yaml**：基礎設定  
- **model_config.yaml**：模型超參數  
- **sampling_config.yaml**：取樣策略  
- **des_config.yaml**：DES 與 ensemble 組合  
- **feature_config.yaml**：特徵選擇  
- **experiment_config.yaml**：實驗設計、可重複性、評估  

---

## 六、建議工作流程

1. **單次跑完**：`python scripts/run_all_experiments.py`（或 `run_all_experiments.bat`）  
2. **看結果**：`python scripts/compare_baseline_ensemble.py`、`python scripts/compare_all_results.py`  
3. **可選**：`python scripts/run_multi_seed.py` 產出 mean±std  

詳細步驟見 **EXECUTION_GUIDE.md**，實驗對照見 **docs/EXPERIMENT_CHECKLIST.md**。
