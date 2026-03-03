# Continual-Imbalance-Ensemble — 目錄結構說明

> 最後更新：2026-03-03

## 整體架構

```
Continual-Imbalance-Ensemble/
│
├── README.md                       ← 專案入口（Quick Start）
├── requirements.txt                ← 完整依賴
├── requirements-core.txt           ← 核心依賴（精簡版）
├── run_all_experiments.bat         ← Windows 一鍵執行
├── advisor_report_20260302.md      ← 指導教授報告（2026-03-02）
├── _fix_bom.py                     ← BOM 修正工具腳本
│
├── config/                         ← 所有 YAML 設定（禁止在 Python 中硬編碼超參數）
│   ├── README.md
│   ├── base_config.yaml            ← 全局基本設定（路徑、seed、log）
│   ├── model_config.yaml           ← 模型超參數（LightGBM、XGBoost）
│   ├── sampling_config.yaml        ← 採樣策略設定
│   ├── des_config.yaml             ← DES/DCS 設定（k、metric）
│   ├── feature_config.yaml         ← 特徵選擇設定（method、k_features）
│   └── experiment_config.yaml      ← 實驗流程設定（split_mode、seeds）
│
├── src/                            ← 可 import 的函式庫（禁止含實驗邏輯）
│   ├── __init__.py                 ← 版本 v0.2.0，列出所有子模組
│   ├── README.md
│   ├── data/
│   │   ├── loader.py               ← 各資料集載入（Bankruptcy/Stock/Medical）
│   │   ├── preprocessor.py         ← 缺值處理、特徵縮放（StandardScaler）
│   │   ├── sampler.py              ← Under/Over/Hybrid Sampling（SMOTE/ENN）
│   │   └── splitter.py             ← 兩種切割：chronological、block_cv (5-fold)
│   ├── models/
│   │   ├── lightgbm_wrapper.py     ← LightGBM 封裝（fit/predict/predict_proba）
│   │   ├── xgboost_wrapper.py      ← XGBoost 封裝
│   │   ├── random_forest_wrapper.py← RandomForest 封裝
│   │   └── model_pool.py           ← ModelPool（6 個基分類器管理）
│   ├── ensemble/
│   │   ├── __init__.py
│   │   └── selector.py             ← DynamicEnsembleSelector (KNORA-E)
│   │                                  EnsembleCombiner (2~6 模型組合)
│   ├── features/
│   │   ├── __init__.py
│   │   └── selector.py             ← FeatureSelector (kbest_f/chi2/lasso)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              ← compute_metrics (AUC/F1/G-Mean/Recall)
│   │                                  print_results_table, results_to_dataframe
│   └── utils/
│       ├── config_loader.py        ← YAML 設定載入器
│       ├── logger.py               ← 統一 Logger（console + file）
│       └── seed.py                 ← set_seed（numpy/random/torch）
│
├── experiments/                    ← 可直接執行的實驗腳本（只 import src/）
│   ├── __init__.py
│   ├── README.md
│   ├── _shared/                    ← 跨 phase 共用函式
│   │   ├── common_bankruptcy.py    ← Bankruptcy 資料載入共用函式
│   │   ├── common_dataset.py       ← Stock/Medical 資料載入共用函式
│   │   ├── common_des.py           ← DES 實驗共用邏輯
│   │   ├── common_des_advanced.py  ← 進階 DES 共用邏輯
│   │   └── common_dcs.py           ← DCS 實驗共用邏輯
│   ├── phase1_baseline/            ← Baseline 實驗
│   │   ├── retrain.py              ← Re-train（歷史 + 新資料合訓）
│   │   └── finetune.py             ← Fine-tune（歷史預訓 + 新資料微調）
│   ├── phase2_ensemble/            ← 靜態集成實驗
│   │   ├── undersampling.py        ← 欠採樣集成
│   │   ├── oversampling.py         ← 過採樣集成
│   │   ├── hybrid.py               ← 混合採樣集成
│   │   └── all_combinations.py     ← 2~6 模型全組合枚舉
│   ├── phase3_dynamic/             ← 動態選擇實驗
│   │   ├── des/
│   │   │   ├── standard.py         ← DES（KNORA-E）標準實驗
│   │   │   └── advanced.py         ← 進階 DES（時間/少數類加權）
│   │   └── dcs/
│   │       └── comparison.py       ← DCS 跨資料集比較
│   ├── phase4_feature/             ← 特徵選擇研究
│   │   ├── fs_study.py             ← 特徵選擇方法比較
│   │   └── fs_sweep.py             ← 特徵數量 sweep
│   └── phase5_analysis/            ← 深度分析
│       ├── base_learner_comparison.py  ← 基學習器比較（LGB/XGB/RF）
│       ├── proportion_study.py         ← New data 比例研究
│       ├── split_comparison.py         ← chronological vs. block_cv 比較
│       └── stock_threshold_cost.py     ← 股票門檻/成本分析
│
├── scripts/                        ← 工具腳本（非實驗）
│   ├── README.md
│   ├── run_all_experiments.py      ← 依序執行所有 phase
│   ├── run_multi_seed.py           ← 多 Seed 重現性（mean±std）
│   ├── compare_baseline_ensemble.py← Bankruptcy 結果彙總
│   ├── compare_all_results.py      ← 三資料集彙總（summary CSV）
│   ├── statistical_test.py         ← 統計顯著性檢定
│   ├── visualize_results.py        ← 結果視覺化（PNG 輸出）
│   ├── generate_advisor_excel.py   ← 產生指導教授報告 Excel
│   ├── generate_synthetic_data.py  ← 合成資料產生
│   ├── download_medical_data.py    ← UCI 醫療資料下載（合成版）
│   ├── download_real_medical_data.py   ← 真實醫療資料下載
│   ├── download_stock_data.py      ← Stock 資料下載（合成版）
│   ├── download_real_stock_data.py ← 真實 Stock 資料下載
│   ├── download_us_bankruptcy.py   ← US Bankruptcy 資料下載
│   ├── _gen_exps.py                ← 實驗腳本自動產生工具
│   └── _write_common_dcs.py        ← common_dcs.py 產生工具
│
├── data/                           ← 資料（raw/ 已 .gitignore）
│   ├── raw/
│   │   ├── bankruptcy/
│   │   │   └── american_bankruptcy_dataset.csv
│   │   ├── stock/
│   │   │   ├── stock_data.csv      ← 合成/舊版資料
│   │   │   ├── stock_spx.csv       ← S&P 500
│   │   │   ├── stock_dji.csv       ← 道瓊工業
│   │   │   └── stock_ndx.csv       ← NASDAQ-100
│   │   └── medical/
│   │       ├── diabetes130/
│   │       │   └── diabetes130_medical.csv  ← Diabetes 130-US Hospitals
│   │       └── synthetic/          ← 合成醫療資料
│   ├── processed/                  ← 前處理後資料（執行時產生）
│   └── splits/                     ← 切割後快取（執行時產生）
│
├── results/                        ← 實驗結果（由腳本自動輸出）
│   ├── README.md
│   ├── summary_all_datasets.csv    ← 三資料集總覽
│   ├── summary_all_datasets_detailed.csv
│   ├── phase1_baseline/            ← retrain / finetune 輸出
│   ├── phase2_ensemble/            ← 靜態集成輸出
│   ├── phase3_dynamic/             ← DES / DCS 輸出
│   ├── phase4_feature/             ← 特徵選擇研究輸出
│   ├── phase5_analysis/            ← 深度分析輸出
│   ├── multi_seed/                 ← 多 Seed 重現性結果
│   │   ├── bankruptcy_multi_seed.csv
│   │   ├── medical_multi_seed.csv
│   │   └── stock_multi_seed.csv
│   └── visualizations/             ← 產出圖表（PNG）
│
├── docs/                           ← 核心文件
│   ├── STRUCTURE.md                ← 本文件（目錄說明）
│   ├── RESEARCH_SPEC.md            ← 指導教授研究方向規格
│   ├── reserch_summary.md          ← 研究摘要
│   └── 研究方向.md                 ← 研究方向規劃
│
├── examples/                       ← 示範腳本
│   ├── demo_complete_workflow.py   ← 完整流程示範
│   └── test_modules.py             ← 模組測試示範
│
└── tests/                          ← 單元測試
    ├── quick_test.py
    └── test_setup.py
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
