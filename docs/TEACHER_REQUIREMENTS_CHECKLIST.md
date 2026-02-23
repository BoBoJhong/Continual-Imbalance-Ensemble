# 老師方向逐條對照（Class Imbalance 3）

以下依老師給的條文逐項對應到本專案的實作與檔案位置，方便口試或論文對照。

---

## 一、Continual learning with class imbalance (non-stationary datasets)

| 老師要求 | 本專案對應 | 程式/結果位置 |
|----------|------------|----------------|
| **bankruptcy prediction (1999~2018) (Kaggle)** | ✅ 使用 US 1999–2018 破產資料；有 `american_bankruptcy_dataset.csv` 時自動載入並以**切割 a** 實驗 | `data/raw/bankruptcy/american_bankruptcy_dataset.csv`、`experiments/common_bankruptcy.py` |
| **stock prediction（學長的論文）** | ✅ 實驗 05 baseline+ensemble、實驗 07 DES | `experiments/05_stock_baseline_ensemble.py`、`07_stock_des.py`，`results/stock/`、`results/des/stock_des_results.csv` |
| **time series medical datasets (UCI)** | ✅ 實驗 06 baseline+ensemble、實驗 08 DES | `experiments/06_medical_baseline_ensemble.py`、`08_medical_des.py`，`results/medical/`、`results/des/medical_des_results.csv` |

所有實驗皆在 **class imbalance** 下進行（under-/over-/hybrid sampling），見 `src/data/imbalance_sampler.py`、各實驗的 `ImbalanceSampler` 與 `ModelPool.create_pool`。

---

## 二、資料集切割

| 老師要求 | 本專案對應 | 程式位置 |
|----------|------------|----------|
| **a. 1999~2011 as historical, 2012~2014 as new operating, 2015~2018 for testing** | ✅ 使用 US 1999–2018 時**自動**改為 chronological 切割 | `experiments/common_bankruptcy.py`（`split_mode="chronological"`、`historical_end=2011`、`new_operating_end=2014`）、`src/data/splitter.py` 的 `chronological_split()` |
| **b. 5-fold CV: 1st+2nd folds = historical, 3rd+4th = new operating, 5th fold = testing** | ✅ 所有資料集皆支援；無 US 檔或 Stock/Medical 使用 block_cv | `src/data/splitter.py` 的 `block_cv_split()`，各實驗 `get_bankruptcy_splits` / `get_splits` 的 `split_mode="block_cv"` |

---

## 三、Baselines

| 老師要求 | 本專案對應 | 程式位置 |
|----------|------------|----------|
| **1. Re-training the model using historical and new operating data** | ✅ Baseline 1：合併 historical + new → 取樣 → 訓練單一模型 → 在 test 評估 | `experiments/01_bankruptcy_baseline.py`（`retrain`）、05、06 同理 |
| **2. Fine-tuning the model solely using new data** | ✅ Baseline 2：先在 historical 訓練，再**僅用 new operating 資料**做第二階段訓練，在 test 評估 | `experiments/01_bankruptcy_baseline.py`（`finetune`）、05、06 同理 |

---

## 四、Ensemble with/without dynamic classifier selection / dynamic ensemble selection

| 老師要求 | 本專案對應 | 程式位置 |
|----------|------------|----------|
| **1. 'Old' models 1/2/3 (trained by historical data with under-/over-/hybrid sampling)** | ✅ Old 模型池：3 個模型（under / over / hybrid），僅用 historical 訓練 | `src/models/model_pool.py`（`create_pool` → `old_under`, `old_over`, `old_hybrid`），實驗 02、05、06 |
| **2. 'New' models 4/5/6 (trained by new operating data with under-/over-/hybrid sampling)** | ✅ New 模型池：3 個模型（under / over / hybrid），僅用 new operating 訓練 | 同上，`new_under`, `new_over`, `new_hybrid` |
| **3a. two combined models: pairs of 'Old' and 'New'** | ✅ 例如 `ensemble_2_old_hybrid_new_hybrid`（Old+New 各一） | `experiments/02_bankruptcy_ensemble.py` 的 `combinations` |
| **3b. three combined: 2 Old + 1 New or 1 Old + 2 New** | ✅ `ensemble_3_type_a`（2 Old + 1 New）、`ensemble_3_type_b`（1 Old + 2 New） | 同上 |
| **3c. four combined models** | ✅ `ensemble_4`（4 模型組合） | 同上 |
| **3d. five combined models** | ✅ `ensemble_5`（5 模型組合） | 同上 |
| **3e. six combined models** | ✅ `ensemble_all_6`（6 模型全上） | 同上 |
| **Dynamic classifier / ensemble selection** | ✅ KNORA-E 風格 DES：DSEL = historical + new，依鄰近樣本選擇模型並軟投票 | `experiments/common_des.py`，實驗 03（Bankruptcy）、07（Stock）、08（Medical） |

---

## 五、Study II: the effect of feature selection on the ensemble classifiers

| 老師要求 | 本專案對應 | 程式/結果位置 |
|----------|------------|----------------|
| **特徵選擇對 ensemble 的影響** | ✅ 有/無 SelectKBest 特徵選擇下跑相同 ensemble 流程，比較 AUC/F1 | `experiments/04_bankruptcy_feature_selection_study.py`，`results/feature_study/bankruptcy_fs_comparison.csv` |

---

## 六、結論

**以上老師所列方向，本專案皆有對應實作並可執行。**

- 三資料集：Bankruptcy (1999–2018)、Stock、Medical。
- 切割 a（1999–2011 / 2012–2014 / 2015–2018）與切割 b（5-fold block CV）皆支援。
- Baselines：Re-training、Fine-tuning。
- Ensemble：Old 1/2/3、New 4/5/6、組合 2/3/4/5/6 模型、DES。
- Study II：特徵選擇對 ensemble 的影響。

執行方式見 **EXECUTION_GUIDE.md**；實驗與結果對照見 **EXPERIMENT_CHECKLIST.md**。
