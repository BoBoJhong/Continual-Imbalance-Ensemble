# 實驗腳本目錄 (Experiments)

本目錄為可直接執行的實驗腳本，依**研究階段**與**方法類型**分資料夾。

## Phase 1 — Baseline

`phase1_baseline/`：再訓練、微調、依資料集／年份切割之 XGB、Torch MLP、TabNet 等（見該目錄內檔案）。

## Phase 2 — 集成（主線：XGB，僅分靜態／動態）

**論文主線**為 **XGBoost Old／New 六槽位 + 年份切割**：不再另開方法資料夾，以**檔名**區分（例如 `xgb_oldnew_ensemble_*`）。

| 子路徑 | 說明 |
|--------|------|
| **`phase2_ensemble/static/`** | **靜態**：`xgb_oldnew_<資料集>_year_splits_static.py` → `xgb_oldnew_ensemble_static_*`（raw／長表／寬表） |
| **`phase2_ensemble/dynamic/des/`** | **動態 DES（XGB）**：`xgb_oldnew_<資料集>_year_splits_des.py` → `xgb_oldnew_ensemble_des_*` |
| **`phase2_ensemble/dynamic/dcs/`** | **動態 DCS（XGB 池）**：`xgb_oldnew_<資料集>_year_splits_dcs.py` → `xgb_oldnew_ensemble_dcs_by_sampling_raw_*.csv` |

**共用模組**：`phase2_ensemble/xgb_oldnew_ensemble_common.py`。

**輸出（預設）**：`results/phase2_ensemble/static/`、`…/dynamic/des/`、`…/dynamic/dcs/` 三處分開；檔名前綴分別為 `xgb_oldnew_ensemble_static_`、`xgb_oldnew_ensemble_des_`、`*_dcs_*`。

**舊版（LightGBM `ModelPool`）**：`undersampling` / `oversampling` / `hybrid` / `all_combinations`、`dynamic/des/standard|advanced`、`dynamic/dcs/comparison` 仍留在庫內供對照，**一鍵腳本預設不執行**。

## Phase 3 — 特徵選取（Study II）

`phase3_feature/`：`fs_study.py`、`fs_sweep.py`。

## Phase 4 — 補充分析

`phase4_analysis/`：split 比較、比例、基學習器、股票閾值成本等。

---

## 共用函式（`_shared/`）

| 模組 | 用途 |
|------|------|
| `common_bankruptcy` | Bankruptcy 切割 |
| `common_dataset` | Stock / Medical 切割 |
| `common_des` / `common_des_advanced` | DES 流程 |
| `common_dcs` | DCS 流程 |

Import 慣例：`from experiments._shared.common_* import ...`

## `project_root` 深度

| 腳本位置 | `project_root` |
|----------|----------------|
| `phase1_baseline/*.py`（多數） | `Path(__file__).parent.parent.parent` |
| `phase2_ensemble/static/*.py` | `Path(__file__).resolve().parent.parent.parent.parent` |
| `phase2_ensemble/dynamic/des/*.py`、`dynamic/dcs/*.py` | `Path(__file__).resolve().parent.parent.parent.parent.parent` |
| `phase2_ensemble/xgb_oldnew_ensemble_common.py`（根） | `Path(__file__).resolve().parent.parent.parent` |
| `phase3_feature/*.py` | `Path(__file__).parent.parent.parent` |

## 執行範例

```powershell
python experiments/phase2_ensemble/static/xgb_oldnew_bankruptcy_year_splits_static.py
python experiments/phase2_ensemble/dynamic/des/xgb_oldnew_bankruptcy_year_splits_des.py
python experiments/phase2_ensemble/dynamic/dcs/xgb_oldnew_bankruptcy_year_splits_dcs.py
python scripts/run/run_all_experiments.py
```
