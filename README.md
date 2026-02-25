# Continual-Imbalance-Ensemble

非平穩環境下類別不平衡之持續學習集成框架

A continual learning framework for class imbalance in non-stationary datasets, combining **Dynamic Ensemble Selection (DES)** and **Hybrid Sampling**.

---

## 📊 研究現況 (Current Progress)

> 最後更新：2026-02-23

### 實驗完成狀況

| 實驗 | 資料集 | 狀態 | 最佳 AUC |
|------|--------|------|----------|
| 01 Baseline (Re-train / Fine-tune) | Bankruptcy | ✅ 完成 | 0.8552 (finetune) |
| 02 Static Ensemble (2~6 models) | Bankruptcy | ✅ 完成 | 0.8693 (new_3) |
| 03 DES (KNORA-E) | Bankruptcy | ✅ 完成 | 0.8560 |
| 04 Study II: Feature Selection | Bankruptcy | ✅ 完成 (已修正特徵篩選比例) | 0.8720 (new_3) |
| 05 Baseline (Re-train / Fine-tune) | Stock | ✅ 完成 | 0.5911 (finetune) |
| 06 Static Ensemble | Stock | ✅ 完成 | |
| 07 Baseline (Re-train / Fine-tune) | Medical | ✅ 完成 | |
| 08 Static Ensemble | Medical | ✅ 完成 | 0.6665 (new_3) |
| 09 DES | Stock | ✅ 完成 | 0.5564 |
| 10 DES | Medical | ✅ 完成 | 0.6374 |
| 11 DES Advanced | Bankruptcy | ✅ 完成 | 0.8626 (time_weighted) |
| 12 Proportion Study | Bankruptcy | ✅ 完成 | 0.8593 (DES_combined, new_20) |

### Bankruptcy 實驗結果快覽

| 方法 | AUC | F1 |
|------|-----|----|
| Re-training | 0.8047 | 0.134 |
| Fine-tuning | 0.8552 | 0.194 |
| Ensemble (Old 3) | 0.8086 | 0.133 |
| **Ensemble (New 3)** | **0.8693** | **0.239** |
| Ensemble (All 6) | 0.8575 | 0.216 |
| DES KNORA-E | 0.8560 | 0.222 |

### 專案現況目錄 (Project Inventory)

包含專案當前的程式碼、資料目錄與結果紀錄清單。

**1. 程式碼 (Source Code)**

```text
src/             ← 核心模組 (資料處理, 模型, 集成, 評估, 視覺化等)
experiments/     ← 各項實驗腳本 (01~10 系列, baselines, DES 等)
scripts/         ← 輔助及工具腳本 (自動化執行, 分析, 繪圖, 下載器等)
config/          ← 存放實驗參數與模型超參數的 YAML 設定檔
```

> 💡 **備註**：`experiments/common_*.py` 是供各個執行腳本共用的基礎流程邏輯，以減少重複程式碼並提升未來的實驗擴展性。

**2. 原始資料與處理 (Data)**

```text
data/
├── raw/         ← 原始下載的公開資料集 (Bankruptcy, Stock, Medical)
├── processed/   ← 經過預處理、正規化後的資料
└── splits/      ← 依據時序或 CV 切分好的訓練集與測試集
```

**3. 實驗結果 (Results)**

```text
results/
├── *.csv                ← 各種綜合比較報表 (如 summary_all_datasets.csv)
├── baseline/            ← Baseline 實驗分析數據
├── ensemble/            ← 靜態集成實驗分析數據
├── des/                 ← 動態集成 (DES) 實驗分析數據
├── des_advanced/        ← 進階 DES 分析數據
├── feature_study/       ← 特徵選取研究結果
├── proportion_study/    ← 資料比例分析結果
├── multi_seed/          ← 多隨機數種子實驗結果
├── stock/               ← Stock 資料集專屬實驗結果 (含 Baseline, Ensemble, DES)
├── medical/             ← Medical 資料集專屬實驗結果 (含 Baseline, Ensemble, DES)
└── visualizations/      ← 統計圖表與視覺化圖檔 (.png)
```

### 待辦 (TODO)

- [x] ~~Study II 特徵選擇結果為 0 差異~~ → 修正（改用 `FS_RATIO=0.5` 比例篩選，`src.features.FeatureSelector`）
- [x] ~~Stock / Medical DES 腳本較陽春~~ → 強化（加入 G-Mean，使用 `src.evaluation.compute_metrics`）
- [x] ~~多 Seed 重現性實驗~~ → 擴展 `run_multi_seed.py`（支援 3 個資料集，`--seeds`/`--dataset` 參數）
- [x] ~~統計顯著性檢定（Wilcoxon）~~ → 新增 `scripts/statistical_test.py`（pairwise Wilcoxon, p-value 矩陣 CSV）
- [x] ~~結果視覺化圖表~~ → 新增 `scripts/visualize_results.py`（4 張 PNG，存於 `results/visualizations/`）

---

## 🎯 研究目標

解決**非平穩資料環境**中**類別不平衡**對預測模型的效能衰退問題，提出結合 Dynamic Ensemble Selection 與 Hybrid Sampling 的持續學習框架，並探討特徵選擇的影響。

詳見 [`docs/RESEARCH_SPEC.md`](docs/RESEARCH_SPEC.md)。

---

## 🗂️ 資料集

| 資料集 | 來源 | 時間範圍 | 切割方式 |
|--------|------|----------|----------|
| Bankruptcy Prediction | Kaggle | 1999–2018 | Chronological (2011/2014) |
| Stock Prediction | 私有 | — | Block 5-fold CV |
| Medical (UCI) | UCI | — | Block 5-fold CV |

資料放置路徑：`data/raw/{bankruptcy,stock,medical}/data.csv`

下載說明：[`docs/DATASET_DOWNLOAD_GUIDE.md`](docs/DATASET_DOWNLOAD_GUIDE.md)

---

## 🏗️ 模型架構

**Old Models**（Historical Data 訓練）：

- Model 1: Undersampling
- Model 2: Oversampling (ADASYN)
- Model 3: Hybrid (SMOTEENN)

**New Models**（New Operating Data 訓練）：

- Model 4: Undersampling
- Model 5: Oversampling (ADASYN)
- Model 6: Hybrid (SMOTEENN)

**集成組合**：2 / 3 (type A: 2 Old+1 New, type B: 1 Old+2 New) / 4 / 5 / 6 models

---

## 📈 評估指標

> 類別不平衡資料：**禁止單獨使用 Accuracy**

- **AUC-ROC** — 排序能力
- **F1-Score** — Precision × Recall 平衡
- **G-Mean** — 多數類與少數類平衡準確率
- **Recall (Sensitivity)** — 少數類抓取能力

---

## 🚀 Quick Start

```powershell
# 1. 啟動虛擬環境
.\venv\Scripts\activate

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 放入資料（見 docs/DATASET_DOWNLOAD_GUIDE.md）

# 4. 一鍵執行所有實驗
python scripts\run_all_experiments.py

# 5. 查看彙總結果
python scripts\compare_all_results.py
```

完整執行步驟：[`docs/EXECUTION_GUIDE.md`](docs/EXECUTION_GUIDE.md)

---

## 📁 Import 快速參考

```python
from src.data import DataLoader, DataSplitter, ImbalanceSampler
from src.models import LightGBMWrapper, ModelPool
from src.ensemble import DynamicEnsembleSelector, EnsembleCombiner
from src.features import FeatureSelector
from src.evaluation import compute_metrics   # AUC / F1 / G-Mean / Recall
from src.utils import get_logger, set_seed
```

---

## 🛠️ 技術棧

- **Python** 3.8+
- **ML**: LightGBM, XGBoost, scikit-learn
- **Imbalanced**: imbalanced-learn (SMOTEENN, ADASYN)
- **DES**: 自行實作 KNORA-E（`src/ensemble`）
- **Feature Selection**: scikit-learn SelectKBest / LASSO

---

## � 文件索引

| 文件 | 說明 |
|------|------|
| [`docs/STRUCTURE.md`](docs/STRUCTURE.md) | 完整目錄結構說明 |
| [`docs/EXECUTION_GUIDE.md`](docs/EXECUTION_GUIDE.md) | 執行步驟與結果解讀 |
| [`docs/RESEARCH_SPEC.md`](docs/RESEARCH_SPEC.md) | 指導教授研究方向規格 |
| [`docs/EXPERIMENT_CHECKLIST.md`](docs/EXPERIMENT_CHECKLIST.md) | 實驗完成進度追蹤 |
| [`docs/TEACHER_REQUIREMENTS_CHECKLIST.md`](docs/TEACHER_REQUIREMENTS_CHECKLIST.md) | 指導教授需求對照 |
| [`docs/DATASET_DOWNLOAD_GUIDE.md`](docs/DATASET_DOWNLOAD_GUIDE.md) | 資料集下載說明 |
| [`.agent/rules.md`](.agent/rules.md) | 專案規範（目錄、命名、import 規則） |
