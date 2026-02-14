# 實驗方向對照表（老師要求 vs 目前狀態）

**專案結構與腳本說明**：見 **docs/STRUCTURE.md**、**EXECUTION_GUIDE.md**、**scripts/README.md**。

## 一、資料集與切割

| 老師要求 | 目前狀態 | 說明 |
|----------|----------|------|
| **Continual learning + class imbalance** | ✅ 有 | 所有實驗皆用 imbalance 取樣（under/over/hybrid） |
| **Bankruptcy (1999~2018 Kaggle)** | ✅ 有 | 實驗 01~04，5-fold block CV（資料無 Year 欄故未用時間切） |
| **切割 a：1999~2011 歷史、2012~2014 新營運、2015~2018 測試** | ⚠️ 程式有 | `DataSplitter.chronological_split()` 已實作；Bankruptcy CSV 無 Year 欄，無法使用。若有帶年份資料可改 `split_mode="chronological"` |
| **切割 b：5-fold CV（1+2 折=歷史、3+4 折=新營運、第 5 折=測試）** | ✅ **已用** | 所有實驗預設使用 `get_bankruptcy_splits` / `get_splits` 的 `split_mode="block_cv"` |
| **Stock 預測（學長論文）** | ✅ **已跑** | 實驗 05 baseline+ensemble、實驗 07 DES，結果在 `results/stock/`、`results/des/stock_des_results.csv` |
| **Time series medical (UCI)** | ✅ **已跑** | 實驗 06 baseline+ensemble、實驗 08 DES，結果在 `results/medical/`、`results/des/medical_des_results.csv` |

---

## 二、Baselines

| 老師要求 | 目前狀態 |
|----------|----------|
| 1. Re-training（歷史 + 新營運資料一起訓練） | ✅ **有**（實驗 01、05、06 的 `retrain`） |
| 2. Fine-tuning（僅用新資料微調） | ✅ **有**（實驗 01、05、06 的 `finetune`） |

---

## 三、Ensemble（含 DCS/DES）

| 老師要求 | 目前狀態 |
|----------|----------|
| **Old 模型 1/2/3**（歷史資料 + under/over/hybrid） | ✅ **有**（實驗 02、05、06） |
| **New 模型 4/5/6**（新營運資料 + under/over/hybrid） | ✅ **有**（實驗 02、05、06） |
| **組合 a：兩兩組合（Old + New 各一）** | ✅ **有** |
| **組合 b：三模型（2 Old + 1 New 或 1 Old + 2 New）** | ✅ **有** |
| **組合 c：四模型** | ✅ **有**（Bankruptcy 實驗 02） |
| **組合 d：五模型** | ✅ **有**（Bankruptcy 實驗 02） |
| **組合 e：六模型** | ✅ **有**（ensemble_all_6） |
| **Dynamic classifier / ensemble selection** | ✅ **有**（實驗 03 Bankruptcy、07 Stock、08 Medical，KNORA-E 風格 DES） |

---

## 四、Study II：特徵選擇對 ensemble 的影響

| 老師要求 | 目前狀態 |
|----------|----------|
| 特徵選擇對 ensemble 的影響 | ✅ **已實作**（實驗 04，`results/feature_study/bankruptcy_fs_comparison.csv`） |

---

## 五、可重複性與彙總

| 項目 | 目前狀態 |
|------|----------|
| **多 seed 重跑（mean±std）** | ✅ **已實作**（`scripts/run_multi_seed.py`，產出 `results/baseline/bankruptcy_baseline_mean_std.csv`） |
| **一鍵跑完所有實驗** | ✅ **已實作**（`scripts/run_all_experiments.py`，依序執行 01~08） |
| **跨資料集結果彙總** | ✅ **已實作**（`scripts/compare_all_results.py`，產出 `results/summary_all_datasets.csv`） |

---

## 六、實驗與腳本一覽（全部完成）

| 編號 | 腳本 | 資料集 | 內容 | 結果位置 |
|------|------|--------|------|----------|
| 01 | `experiments/01_bankruptcy_baseline.py` | Bankruptcy | Baseline（retrain / finetune / ensemble_old） | results/baseline/ |
| 02 | `experiments/02_bankruptcy_ensemble.py` | Bankruptcy | 靜態 ensemble（2/3/4/5/6 模型組合） | results/ensemble/ |
| 03 | `experiments/03_bankruptcy_des.py` | Bankruptcy | DES (KNORA-E) | results/des/ |
| 04 | `experiments/04_bankruptcy_feature_selection_study.py` | Bankruptcy | Study II：有/無特徵選擇 | results/feature_study/ |
| 05 | `experiments/05_stock_baseline_ensemble.py` | Stock | Baseline + 靜態 ensemble | results/stock/ |
| 06 | `experiments/06_medical_baseline_ensemble.py` | Medical | Baseline + 靜態 ensemble | results/medical/ |
| 07 | `experiments/07_stock_des.py` | Stock | DES (KNORA-E) | results/des/ |
| 08 | `experiments/08_medical_des.py` | Medical | DES (KNORA-E) | results/des/ |
| — | `scripts/run_all_experiments.py` | — | 一鍵執行 01~08 | — |
| — | `scripts/compare_baseline_ensemble.py` | Bankruptcy | 合併 baseline + ensemble + DES | results/bankruptcy_all_results.csv |
| — | `scripts/compare_all_results.py` | 全部 | 彙總各資料集 AUC/F1、最佳方法 | results/summary_all_datasets.csv |
| — | `scripts/run_multi_seed.py` | Bankruptcy | 多 seed baseline → mean±std | results/baseline/bankruptcy_baseline_mean_std.csv |

---

## 七、結論

**老師指定的實驗方向已全部實作並可執行。**

- 切割：使用 5-fold block CV（1+2 / 3+4 / 5）；時間切割 a 需資料具 Year 欄方可啟用。
- 三資料集：Bankruptcy、Stock、Medical 皆具 baseline、靜態 ensemble、DES。
- Study II、多 seed、一鍵跑完、跨資料集彙總腳本均已就緒。

執行方式見 **EXECUTION_GUIDE.md**。
