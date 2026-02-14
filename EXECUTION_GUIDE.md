# 執行指南

**專案結構與檔案說明**：見 **docs/STRUCTURE.md**、**scripts/README.md**、**README.md**。

## ✅ 當前狀態

所有模組已安裝並測試通過！

## 🚀 立即執行（在虛擬環境中）

### 1. 確認環境

```powershell
# 確保您在虛擬環境中（看到 (venv) 提示符）
.\venv\Scripts\activate
```

### 2. 運行完整示範

```powershell
python examples\demo_complete_workflow.py
```

**預期輸出**：

- 創建 1000 個樣本的示範資料
- 切割為 Historical/New/Testing
- 訓練 6 個模型（3 Old + 3 New）
- 顯示每個模型的 AUC 分數

### 3. 測試特定功能

#### 測試資料載入

```powershell
python -c "from src.data import DataLoader; print('DataLoader OK')"
```

#### 測試模型訓練

```powershell
python -c "from src.models import LightGBMWrapper; print('LightGBM OK')"
```

#### 測試配置系統

```powershell
python -c "from src.utils import get_config_loader; loader = get_config_loader(); print(f'載入了 {len(loader.load_all())} 個配置')"
```

## 📋 實驗流程（跑完之後的順序）

### 1. 一鍵跑完所有實驗（推薦）

```powershell
python scripts\run_all_experiments.py
```

會依序執行 01~08（Bankruptcy baseline/ensemble/DES/Study II、Stock baseline+ensemble+DES、Medical baseline+ensemble+DES）。

### 2. 或手動逐個跑（預設 5-fold block CV）

```powershell
python experiments\01_bankruptcy_baseline.py
python experiments\02_bankruptcy_ensemble.py
python experiments\03_bankruptcy_des.py
python experiments\04_bankruptcy_feature_selection_study.py
python experiments\05_stock_baseline_ensemble.py
python experiments\06_medical_baseline_ensemble.py
python experiments\07_stock_des.py
python experiments\08_medical_des.py
```

### 3. 看合併比較（建議先跑這兩個，一次看全部）

```powershell
python scripts\compare_baseline_ensemble.py   # Bankruptcy 合併表
python scripts\compare_all_results.py        # 三資料集彙總 + 終端顯示各資料集最佳 AUC
```

### 4. 如何看整個實驗結果

**第一步：產出彙總表**

- 跑完 01～08 後，執行上面兩個比較腳本，會產出：
  - **`results/summary_all_datasets.csv`**：三資料集（bankruptcy / stock / medical）所有方法的 **AUC、F1**，一表看全部。
  - **`results/bankruptcy_all_results.csv`**：Bankruptcy 的 baseline + ensemble + DES，含 **AUC、F1、Precision、Recall**。

**第二步：用 Excel / 記事本 / 試算表開 CSV**

| 檔案 | 內容 | 怎麼看 |
|------|------|--------|
| **results/summary_all_datasets.csv** | 各資料集、各方法、AUC、F1 | 依 `dataset` 篩選，比較同一資料集下哪個 `method` 的 AUC 最高。 |
| **results/bankruptcy_all_results.csv** | Bankruptcy 所有方法、AUC/F1/Precision/Recall | 看「實驗」欄區分 baseline / ensemble / des；可依 AUC 排序。 |
| **results/baseline/bankruptcy_baseline_results.csv** | 僅 Bankruptcy baseline（retrain / finetune / ensemble_old） | 三個 baseline 的詳細指標。 |
| **results/ensemble/bankruptcy_ensemble_results.csv** | Bankruptcy 靜態 ensemble（2～6 模型組合） | 各組合的 AUC、F1 等。 |
| **results/des/bankruptcy_des_results.csv** | Bankruptcy DES (KNORA-E) | DES 單一結果。 |
| **results/feature_study/bankruptcy_fs_comparison.csv** | Study II：有/無特徵選擇 | 比較無 FS vs 有 FS 的 AUC。 |
| **results/stock/stock_baseline_ensemble_results.csv** | Stock baseline + ensemble | 同結構。 |
| **results/medical/medical_baseline_ensemble_results.csv** | Medical baseline + ensemble | 同結構。 |
| **results/des/stock_des_results.csv**、**des/medical_des_results.csv** | Stock、Medical 的 DES | 同結構。 |

**第三步：終端摘要**

- 執行 **`python scripts\compare_all_results.py`** 時，終端會印出「各資料集最佳 AUC」，例如：
  - `bankruptcy: ensemble_2_old_hybrid_new_hybrid AUC=0.9735`
  - `stock: finetune AUC=1.0000`
  - `medical: ensemble_3_type_a AUC=0.9773`

**寫論文時**：可直接用 `summary_all_datasets.csv` 做「跨資料集比較表」、用 `bankruptcy_all_results.csv` 做「Bankruptcy 方法對照表」；Study II 用 `feature_study/bankruptcy_fs_comparison.csv`。

### 5. 多 seed 重跑（可重複性，mean±std）

```powershell
python scripts\run_multi_seed.py
```

產出 **`results/baseline/bankruptcy_baseline_mean_std.csv`**（retrain / finetune / ensemble_old 的 AUC、F1 的 mean±std）。

### 6. 接下來可選

- **畫圖**：`pip install matplotlib` 後再跑一次比較腳本（會嘗試產出 AUC 比較圖）。
- **其他資料集**：把 Stock / Medical 資料放到 `data/raw/` 後，可仿照 01/02/03 寫對應實驗。
- **多 seed / 統計**：用不同 `random_state` 重跑數次，算 mean±std、做 Wilcoxon 等（見 `config/experiment_config.yaml`）。
- **寫論文**：用 `results/bankruptcy_all_results.csv` 做表格與討論（baseline vs 靜態 ensemble vs DES）。

---

## 📋 下一步（資料與探索）

### A. 下載資料集

1. **Bankruptcy**: <https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction>
2. **Stock**: 在 Kaggle 搜尋 "stock market crash"
3. **Medical**: <https://physionet.org/content/mimiciii/>

### B. 資料放置

```
data/raw/
├── bankruptcy/
│   └── data.csv
├── stock/
│   └── data.csv
└── medical/
    └── data.csv
```

### C. 開始資料探索

```powershell
# 安裝 Jupyter（如果需要）
pip install jupyter matplotlib seaborn

# 啟動
jupyter lab
```

## 🎯 您已經擁有

✅ 完整的專案結構
✅ 所有核心模組（13 個）
✅ 配置系統（6 個 YAML）
✅ 測試腳本
✅ 示範程式

**準備好開始您的研究了！** 🎉
