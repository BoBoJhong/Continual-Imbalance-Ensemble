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
│   ├── phase2_ensemble/            ← Phase2 集成（主線 XGB：`static/` + `dynamic/des/`，檔名 xgb_*）
│   │   ├── xgb_oldnew_ensemble_common.py  ← XGB Old/New 共用
│   │   ├── static/                 ← 靜態集成
│   │   │   ├── undersampling.py / oversampling.py / hybrid.py
│   │   │   ├── all_combinations.py
│   │   │   └── …
│   │   ├── dynamic/
│   │   │   ├── des/                ← DES（standard / advanced / xgb year-split）
│   │   │   └── dcs/                ← DCS（comparison）
│   │   └── …
│   ├── phase3_feature/             ← 特徵選擇研究（Phase 3 FS）
│   │   ├── fs_study.py             ← 特徵選擇方法比較
│   │   └── fs_sweep.py             ← 特徵數量 sweep
│   └── phase4_analysis/            ← 深度／補充分析
│       ├── base_learner_comparison.py  ← 基學習器比較（LGB/XGB/RF）
│       ├── proportion_study.py         ← New data 比例研究
│       ├── split_comparison.py         ← chronological vs. block_cv 比較
│       └── stock_threshold_cost.py     ← 股票門檻/成本分析
│
├── scripts/                        ← 工具腳本（非實驗），詳見 scripts/README.md
│   ├── README.md
│   ├── data/                       ← 資料下載、合成資料
│   │   ├── download_us_bankruptcy.py
│   │   ├── download_medical_data.py / download_real_medical_data.py
│   │   ├── download_stock_data.py / download_real_stock_data.py
│   │   └── generate_synthetic_data.py
│   ├── run/                        ← 一鍵執行、多 seed、批次切割
│   │   ├── run_all_experiments.py
│   │   ├── run_multi_seed.py
│   │   ├── _run_both_splits.py
│   │   ├── _gen_exps.py
│   │   └── _write_common_dcs.py
│   ├── analysis/                   ← 彙總、統計檢定
│   │   ├── compare_all_results.py
│   │   ├── compare_baseline_ensemble.py
│   │   └── statistical_test.py
│   ├── plots/                      ← 視覺化與 Phase1 繪圖
│   │   ├── visualize_results.py
│   │   ├── phase1_baseline_plotting.py
│   │   ├── visualize_phase1_xgb_baseline.py
│   │   └── visualize_phase1_torch_mlp_baseline.py
│   └── reports/                    ← 報表匯出
│       └── generate_advisor_excel.py
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
│   ├── phase2_ensemble/            ← static/、dynamic/des/（XGB 主線）；dynamic/dcs/ 僅舊版腳本可寫入
│   ├── phase3_feature/             ← 特徵選擇研究輸出
│   ├── phase4_analysis/            ← 深度／補充分析輸出
│   ├── multi_seed/                 ← 多 Seed 重現性結果
│   │   ├── bankruptcy_multi_seed.csv
│   │   ├── medical_multi_seed.csv
│   │   └── stock_multi_seed.csv
│   └── visualizations/             ← 產出圖表（PNG）
│
├── UML/                            ← PlantUML 方法／流程圖（見 UML/README_圖表建議.md）
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
