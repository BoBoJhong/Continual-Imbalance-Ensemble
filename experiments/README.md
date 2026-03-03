# 實驗腳本目錄 (Experiments)

本目錄存放所有可直接執行的實驗腳本，依**研究階段與方法類型**分資料夾。  
每個腳本均跨全部資料集（Bankruptcy / Stock / Medical），一次執行出結果。

## 目錄結構

```
experiments/
├── _shared/                      # 共用資料載入 / 模型流程（不可直接執行）
│   ├── common_bankruptcy.py      # Bankruptcy 資料集切割
│   ├── common_dataset.py         # Stock / Medical 資料集切割
│   ├── common_des.py             # DES KNORA-E 核心流程
│   ├── common_des_advanced.py    # 進階 DES（時間加權 / 少數類加權）
│   └── common_dcs.py             # DCS OLA/LCA/TW 流程
│
├── phase1_baseline/              # Phase 1：基準方法
│   ├── retrain.py                # Re-training（全資料集 × 4種取樣策略）
│   └── finetune.py               # Fine-tuning（全資料集 × 4種取樣策略）
│
├── phase2_ensemble/              # Phase 2：靜態集成
│   ├── undersampling.py          # Undersampling 模型池（old/new/pair）
│   ├── oversampling.py           # Oversampling 模型池（old/new/pair）
│   ├── hybrid.py                 # Hybrid 模型池（old/new/pair）
│   └── all_combinations.py       # 全組合系統性測試（2~6 模型）
│
├── phase3_dynamic/               # Phase 3：動態集成選擇（核心貢獻）
│   ├── des/
│   │   ├── standard.py           # KNORA-E DES（全資料集）
│   │   └── advanced.py           # 時間加權 / 少數類加權 DES（全資料集）
│   └── dcs/
│       └── comparison.py         # OLA / LCA / OLA_TW / LCA_TW 比較（全資料集）
│
├── phase4_feature/               # Phase 4：Study II—特徵選擇的影響
│   ├── fs_study.py               # 基本特徵選擇研究（KBest_F, ratio=0.5）
│   └── fs_sweep.py               # 方法 × 比例全掃描（KBest_F/Chi2/LASSO × 0.2/0.5/0.8）
│
└── phase5_analysis/              # Phase 5：補充分析
    ├── split_comparison.py       # Chronological vs Block-CV 比較（全資料集）
    ├── proportion_study.py       # 新資料比例研究 [0.1–1.0]（全資料集）
    ├── base_learner_comparison.py# Base Learner 比較（LightGBM / XGBoost / RF）
    └── stock_threshold_cost.py   # Stock：決策閾值 & 成本分析
```

## 共用輔助函式速查

| 函式 | 所在模組 | 功能 |
|------|----------|------|
| `get_bankruptcy_splits(logger, split_mode)` | `common_bankruptcy` | 取得 bankruptcy 三段切割資料 |
| `get_splits(dataset, logger, split_mode)` | `common_dataset` | 取得 stock/medical 三段切割資料 |
| `run_des(X_h,y_h,X_n,y_n,X_t,y_t,logger,k=7)` | `common_des` | KNORA-E DES 流程，回傳 metrics dict |
| `run_des_advanced(...,time_weight,minority_weight)` | `common_des_advanced` | 進階 DES，回傳 metrics dict |
| `run_dcs_all_variants(X_h,y_h,X_n,y_n,X_t,y_t,logger,k=7)` | `common_dcs` | DCS 全變體，回傳 variants dict |

## 執行方式

```powershell
# 執行單一腳本
python experiments/phase1_baseline/retrain.py

# 一鍵執行全部（依階段順序）
python scripts/run_all_experiments.py
```

## 重要慣例

- `project_root`：位於 `_shared/` 及 phase1~5 的腳本 → `Path(__file__).parent.parent.parent`  
  位於 `des/` 或 `dcs/` 子目錄的腳本 → `Path(__file__).parent.parent.parent.parent`
- 所有結果輸出至 `results/phase?_??*/`，命名格式：`{dataset}_{description}.csv`
- 所有 import 使用 `from experiments._shared.common_*`
