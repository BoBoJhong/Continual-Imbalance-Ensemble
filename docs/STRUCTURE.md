# Continual-Imbalance-Ensemble — 目錄結構說明

> 最後更新：2026-02-23

## 整體架構

```
Continual-Imbalance-Ensemble/
│
├── .agent/
│   ├── rules.md                    ← 專案規範（Antigravity 自動讀取）
│   └── workflows/                  ← Antigravity 工作流程
│
├── README.md                       ← 專案入口（Quick Start）
├── requirements.txt                ← 完整依賴
├── requirements-core.txt           ← 核心依賴（精簡版）
├── run_all_experiments.bat         ← Windows 一鍵執行
├── run_experiment.bat              ← 單一實驗執行
├── company-bankruptcy-prediction.zip ← Bankruptcy 原始資料
│
├── config/                         ← 所有 YAML 設定（禁止在 Python 中硬編碼超參數）
│   ├── base_config.yaml            ← 全局基本設定（路徑、seed、log）
│   ├── model_config.yaml           ← 模型超參數（LightGBM、XGBoost）
│   ├── sampling_config.yaml        ← 採樣策略設定
│   ├── des_config.yaml             ← DES/DCS 設定（k、metric）
│   ├── feature_config.yaml         ← 特徵選擇設定（method、k_features）
│   └── experiment_config.yaml      ← 實驗流程設定（split_mode、seeds）
│
├── src/                            ← 可 import 的函式庫（禁止含實驗邏輯）
│   ├── __init__.py                 ← 版本 v0.2.0，列出所有子模組
│   ├── data/
│   │   ├── loader.py               ← 各資料集載入（Bankruptcy/Stock/Medical）
│   │   ├── preprocessor.py         ← 缺值處理、特徵縮放（StandardScaler）
│   │   ├── sampler.py              ← Under/Over/Hybrid Sampling（SMOTE/ENN）
│   │   └── splitter.py             ← 兩種切割：chronological、block_cv (5-fold)
│   ├── models/
│   │   ├── lightgbm_wrapper.py     ← LightGBM 封裝（fit/predict/predict_proba）
│   │   ├── xgboost_wrapper.py      ← XGBoost 封裝
│   │   └── model_pool.py           ← ModelPool（6 個基分類器管理）
│   ├── ensemble/                   ← [v0.2.0 補完]
│   │   ├── __init__.py
│   │   └── selector.py             ← DynamicEnsembleSelector (KNORA-E)
│   │                                  EnsembleCombiner (2~6 模型組合)
│   ├── features/                   ← [v0.2.0 補完]
│   │   ├── __init__.py
│   │   └── selector.py             ← FeatureSelector (kbest_f/chi2/lasso)
│   ├── evaluation/                 ← [v0.2.0 補完]
│   │   ├── __init__.py
│   │   └── metrics.py              ← compute_metrics (AUC/F1/G-Mean/Recall)
│   │                                  print_results_table, results_to_dataframe
│   └── utils/
│       ├── config_loader.py        ← YAML 設定載入器
│       ├── logger.py               ← 統一 Logger（console + file）
│       └── seed.py                 ← set_seed（numpy/random/torch）
│
├── experiments/                    ← 可直接執行的實驗腳本（只 import src/）
│   ├── common_bankruptcy.py        ← Bankruptcy 資料載入共用函式
│   ├── common_dataset.py           ← Stock/Medical 資料載入共用函式
│   ├── common_des.py               ← DES 實驗共用邏輯
│   ├── common_des_advanced.py      ← 進階 DES 共用邏輯
│   ├── 01_bankruptcy_baseline.py   ← Baseline: Re-train / Fine-tune
│   ├── 02_bankruptcy_ensemble.py   ← 靜態集成（2~6 模型組合）
│   ├── 03_bankruptcy_des.py        ← Dynamic Ensemble Selection (KNORA-E)
│   ├── 04_bankruptcy_feature_selection_study.py  ← Study II: 特徵選擇效果
│   ├── 05_stock_baseline_ensemble.py
│   ├── 06_medical_baseline_ensemble.py
│   ├── 07_stock_des.py
│   ├── 08_medical_des.py
│   ├── 09_bankruptcy_des_advanced.py   ← 進階 DES（時間/少數類加權）
│   └── 10_bankruptcy_proportion_study.py ← New data 比例研究
│
├── scripts/                        ← 工具腳本（非實驗）
│   ├── README.md                   ← 腳本說明
│   ├── run_all_experiments.py      ← 依序執行 01~10
│   ├── run_multi_seed.py           ← 多 Seed 重現性（mean±std）
│   ├── compare_baseline_ensemble.py ← Bankruptcy 結果彙總
│   ├── compare_all_results.py      ← 三資料集彙總（summary CSV）
│   ├── download_medical_data.py    ← UCI 醫療資料下載
│   ├── download_stock_data.py      ← Stock 資料下載
│   └── download_us_bankruptcy.py   ← US Bankruptcy 資料下載
│
├── data/                           ← 資料（raw/ 已 .gitignore）
│   ├── raw/
│   │   ├── bankruptcy/             ← american_bankruptcy_dataset.csv 或 data.csv
│   │   ├── stock/                  ← data.csv（私有）
│   │   └── medical/                ← data.csv（UCI）
│   ├── processed/                  ← 前處理後資料
│   └── splits/                     ← 切割後快取
│
├── results/                        ← 實驗結果（由腳本自動輸出）
│   ├── baseline/                   ← 01 的輸出
│   ├── ensemble/                   ← 02 的輸出
│   ├── des/                        ← 03/07/08 的輸出
│   ├── des_advanced/               ← 09 的輸出
│   ├── feature_study/              ← 04 的輸出（Study II）
│   ├── proportion_study/           ← 10 的輸出
│   ├── stock/                      ← 05/07 的股票結果
│   ├── medical/                    ← 06/08 的醫療結果
│   └── visualizations/             ← 產出圖表
│
├── docs/                           ← 核心文件（~10 個，全 UPPER_CASE.md）
│   ├── STRUCTURE.md                ← 本文件（目錄說明）
│   ├── EXECUTION_GUIDE.md          ← 執行步驟與結果解讀
│   ├── RESEARCH_SPEC.md            ← 指導教授研究方向規格
│   ├── EXPERIMENT_CHECKLIST.md     ← 實驗完成進度追蹤
│   ├── TEACHER_REQUIREMENTS_CHECKLIST.md ← 指導教授需求對照
│   ├── DATASET_DOWNLOAD_GUIDE.md   ← 資料集下載說明
│   ├── DATA_BANKRUPTCY_US.md       ← US Bankruptcy 資料格式說明
│   ├── MIMIC_III_APPLICATION_GUIDE.md ← MIMIC-III 申請指南
│   ├── MEDICAL_ALTERNATIVES.md     ← 醫療資料集替代方案
│   ├── RESEARCH_EXTENSIONS.md      ← 研究延伸方向
│   └── RESULTS_09_10_INTERPRETATION.md ← 實驗 9/10 結果解讀
│
├── examples/                       ← 示範腳本
├── notebooks/                      ← Jupyter Notebooks（EDA）
└── tests/                          ← 單元測試
```

## Import 範例

```python
# 資料處理
from src.data import DataLoader, DataPreprocessor, DataSplitter, ImbalanceSampler

# 模型
from src.models import LightGBMWrapper, XGBoostWrapper, ModelPool

# 集成選擇
from src.ensemble import DynamicEnsembleSelector, EnsembleCombiner

# 特徵選擇
from src.features import FeatureSelector

# 評估指標（禁止單用 Accuracy）
from src.evaluation import compute_metrics, print_results_table

# 工具
from src.utils import get_logger, set_seed, get_config_loader
```
