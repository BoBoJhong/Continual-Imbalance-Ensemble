# Scripts 目錄說明

所有腳本皆假設在**專案根目錄**下執行（`Continual-Imbalance-Ensemble/`）。

## 目錄一覽

| 子資料夾 | 用途 |
|----------|------|
| **`data/`** | 資料下載、合成資料產生 |
| **`run/`** | 一鍵跑實驗、多 seed、批次切割輔助 |
| **`analysis/`** | 結果彙總、baseline 比較、統計檢定 |
| **`plots/`** | 圖表與 Phase1 baseline 視覺化 |
| **`reports/`** | 報表匯出（如指導教授 Excel） |

## 常用指令（由根目錄執行）

```text
# 依序執行各 phase 實驗
python scripts/run/run_all_experiments.py

# 三資料集結果彙總 → results/summary_all_datasets.csv
python scripts/analysis/compare_all_results.py

# 多 seed（Bankruptcy baseline）
python scripts/run/run_multi_seed.py

# Phase1 XGB / Torch MLP 圖表
python scripts/plots/visualize_phase1_xgb_baseline.py
python scripts/plots/visualize_phase1_torch_mlp_baseline.py

# 真實醫療資料下載
python scripts/data/download_real_medical_data.py
```

## 檔案索引

### `data/`

| 檔案 | 說明 |
|------|------|
| `download_us_bankruptcy.py` | US 破產資料 |
| `download_medical_data.py` | UCI 醫療（可選／合成） |
| `download_real_medical_data.py` | 真實醫療資料 |
| `download_stock_data.py` | 股價（可選／合成） |
| `download_real_stock_data.py` | 真實股價 |
| `generate_synthetic_data.py` | 合成資料 |

### `run/`

| 檔案 | 說明 |
|------|------|
| `run_all_experiments.py` | 依序執行 experiments/ 各 phase |
| `run_multi_seed.py` | 多 seed baseline |
| `_run_both_splits.py` | 多資料集 × chronological / block_cv |
| `_gen_exps.py` | 產生實驗檔輔助（開發用） |
| `_write_common_dcs.py` | 寫入 `common_dcs` 模板（開發用） |

### `analysis/`

| 檔案 | 說明 |
|------|------|
| `compare_all_results.py` | 全資料集彙總 CSV |
| `compare_baseline_ensemble.py` | Bankruptcy baseline + ensemble |
| `statistical_test.py` | Wilcoxon 等顯著性檢定 |

### `plots/`

| 檔案 | 說明 |
|------|------|
| `visualize_results.py` | 通用結果視覺化 |
| `phase1_baseline_plotting.py` | Phase1 共用繪圖函式（被其他腳本 import） |
| `visualize_phase1_xgb_baseline.py` | Phase1 XGB 圖 |
| `visualize_phase1_torch_mlp_baseline.py` | Phase1 Torch MLP 圖 |

### `reports/`

| 檔案 | 說明 |
|------|------|
| `generate_advisor_excel.py` | 指導教授報告 Excel |

單一實驗請直接執行 `experiments/` 下對應腳本，見 **docs/STRUCTURE.md** 與 **README.md**（專案根目錄）。
