# 破產預測實驗結果圖

本目錄收錄 Phase 1–3 的十張結果圖，各有 PNG、SVG 與作圖 CSV；同名檔直接覆寫，不另存版本。研究方法流程圖仍位於 `assets/research/diagrams`。

AUC 為主要指標，F1-score／Recall 以 failed 為正類。圖中延續概括標示；具體算法、設定代碼與來源如下。這些是結果圖，不新增訓練或成效實驗。

## 圖檔

| 階段 | 圖 | 用途 |
|---|---|---|
| phase1 | [PNG](phase1/phase1-hybrid-boundary-auc.png)／[SVG](phase1/phase1-hybrid-boundary-auc.svg)／[CSV](phase1/phase1-hybrid-boundary-auc.csv) | How does AUC vary over 15 Old/New boundaries under fixed hybrid sampling? |
| phase1 | [PNG](phase1/phase1-hybrid-boundary-f1.png)／[SVG](phase1/phase1-hybrid-boundary-f1.svg)／[CSV](phase1/phase1-hybrid-boundary-f1.csv) | How does F1-score vary over 15 Old/New boundaries under fixed hybrid sampling? |
| phase1 | [PNG](phase1/phase1-hybrid-boundary-recall.png)／[SVG](phase1/phase1-hybrid-boundary-recall.svg)／[CSV](phase1/phase1-hybrid-boundary-recall.csv) | How does Recall vary over 15 Old/New boundaries under fixed hybrid sampling? |
| phase2 | [PNG](phase2/phase2-pooled-ensemble-comparison.png)／[SVG](phase2/phase2-pooled-ensemble-comparison.svg)／[CSV](phase2/phase2-pooled-ensemble-comparison.csv) | How do fixed ensemble strategies compare on the same pooled Test rows? |
| phase2 | [PNG](phase2/phase2-static-subset-size.png)／[SVG](phase2/phase2-static-subset-size.svg)／[CSV](phase2/phase2-static-subset-size.csv) | How do average candidate metrics vary with mixed-subset size? |
| phase2 | [PNG](phase2/phase2-annual-test-comparison.png)／[SVG](phase2/phase2-annual-test-comparison.svg)／[CSV](phase2/phase2-annual-test-comparison.csv) | How do prespecified strategies differ within each of four Test years? |
| phase3 | [PNG](phase3/phase3-fs-paired-auc.png)／[SVG](phase3/phase3-fs-paired-auc.svg)／[CSV](phase3/phase3-fs-paired-auc.csv) | What is the same-seed change in AUC after feature selection? |
| phase3 | [PNG](phase3/phase3-fs-paired-f1.png)／[SVG](phase3/phase3-fs-paired-f1.svg)／[CSV](phase3/phase3-fs-paired-f1.csv) | What is the same-seed change in F1-score after feature selection? |
| phase3 | [PNG](phase3/phase3-fs-paired-recall.png)／[SVG](phase3/phase3-fs-paired-recall.svg)／[CSV](phase3/phase3-fs-paired-recall.csv) | What is the same-seed change in Recall after feature selection? |
| phase3 | [PNG](phase3/phase3-fs-selected-static-policy.png)／[SVG](phase3/phase3-fs-selected-static-policy.svg)／[CSV](phase3/phase3-fs-selected-static-policy.csv) | How does the validation-selected static policy perform across all FS configurations? |

## Phase 1：歷史切割敏感度

固定使用 hybrid 採樣，不按 Test 選最佳採樣。Old／New 各有十五組邊界；Old-window end year 是切割參數，不能把折線當成部署年度的成效趨勢。Retrain 每個候選與採樣僅有一筆全資料結果，以固定水平參考線呈現，不複製成十五次獨立實驗。

候選來源依已盤點且完整涵蓋十五組邊界的既有檔案固定指定，未依 Test 排名挑選版本。各檔調參與執行來源不同，圖面只作描述性比較，不能據此宣稱某分類器已被驗證為最佳。舊 CSV 的獨立 Finetune 分支未混入使用者所定義的 New／Fine-tune。

| 圖中代碼 | 類別 | 實作標籤 | 固定來源 |
|---|---|---|---|
| C1 | Traditional ML | XGBoost | `results/phase1_baseline/xgb/tuned/bankruptcy_year_splits_xgb_raw.csv` |
| C2 | Traditional ML | LightGBM | `results/phase1_baseline/lgbm/bk_lgb_tuned48_20260421/tuned/bankruptcy_year_splits_lgb_raw.csv` |
| C3 | Traditional ML | Random Forest | `results/phase1_baseline/random_forest/bankruptcy_year_splits_rf_raw.csv` |
| C4 | Traditional ML | Logistic Regression | `results/phase1_baseline/logistic_regression/bk_lr_tuned4_20260421/tuned/bankruptcy_year_splits_lr_raw.csv` |
| C5 | Deep learning | TabM | `results/phase1_baseline/tabm/tuned_light_full_20260410/tuned/bankruptcy_year_splits_tabm_raw.csv` |
| C6 | Deep learning | TabICL-labelled entry; inspect recorded backend | `results/phase1_baseline/tabicl/tabicl_gpu_20260506_setA_temp075_ctx2048/bankruptcy_year_splits_tabicl_raw.csv` |

完整三策略×四採樣來源另存 [baseline-source-grid.csv](phase1/baseline-source-grid.csv)，含所有原始列定位及調參欄位。三張折線主圖僅使用 hybrid 固定切片。

## Phase 2：同協定集成比較

來源固定為 `results/phase2_ensemble/standard_dynamic/runs/20261004T202331745995Z`，completed 且有既存獨立 audit。作圖時重新核對 90 份配置 metrics 的 checksum，並重建總 metrics；本次不重訓模型或重跑全套機率 audit。

Old-fit 1999–2011、New-fit 2012–2013、DSEL／Calibration 為 2014 年互斥子集、Test 2015–2018。Test 共有 12,282 列，其中 287 列 failed。

- pooled 點圖：八個預先指定方法，同一 Test、十 seeds 的平均值±標準差。
- 模型數折線：先在每個 seed 內平均該模型數的所有靜態混合子集指標，再跨十 seeds 計算平均值與標準差。每個 seed 的 2／3／4／5／6 模型子集數為 9／18／15／6／1；不是挑 Test 最佳組合，也不是把各子集再次組成一個新模型。
- 年度點圖：只有四個測試年度，已核對來源沒有更細時間粒度，故用離散點比較，不補造月資料或外插年度趨勢。2015／2016／2017／2018 的 failed 列數為 111／83／57／36，年樣本數顯示於圖中。

| 圖中名稱 | 原始 method | 說明 |
|---|---|---|
| Old-only／New-only／Equal-six | Old3／New3／Equal6 | 群組或完整六模型等權集成 |
| V-static | ValidationSelectedStatic | Calibration AP 選靜態候選；Calibration F1 選門檻 |
| DCS-1／DCS-2 | DCS_OLA／DCS_LCA | 官方 OLA／LCA |
| DES-1／DES-2 | KNORA_E／KNORA_U | 官方 KNORA-E／KNORA-U |

## Phase 3：固定方法差值與選模政策分開

三張熱圖顯示 FS 減 no-FS 的同 seed、同方法、同 Test 差值，取十 seeds 平均；以 Equal-six、DCS-1／2、DES-1／2 五個固定方法比較。重新計算的 400 筆 paired rows 與正式差值 CSV 一致。三圖使用相同、以零為中心的色階；正值代表指標較高，負值代表較低，不能當成統計顯著性。FS 每組重新訓練及依驗證資料重新固定門檻，所以差值反映整個 FS 訓練流程，不是獨立的特徵刪減因果效應。

另有一張 V-static 點圖，候選組合可隨 FS 變動，故是整體選模政策比較，不混入固定方法的純配對圖。No FS 僅為結果分析對照，不加入第三階段方法流程圖。

| 圖中代碼 | 原始 fs | 實作 | 特徵數 |
|---|---|---|---|
| FS1 | mi_r50 | MI | 9 |
| FS2 | mi_r80 | MI | 14 |
| FS3 | cart_r50 | CART | 9 |
| FS4 | cart_r80 | CART | 14 |
| FS5 | shap_r50 | SHAP | 9 |
| FS6 | shap_r80 | SHAP | 14 |
| FS7 | rfe_r50 | RFE | 9 |
| FS8 | rfe_r80 | RFE | 14 |

SHAP 是模型解釋後的特徵排序，不能僅因方法主圖概括為 Filter／Wrapper／Embedded，就改稱純 Embedded。具體算法保留於本對照表。

## 使用界線與重建

- Phase 1 與最新 Phase 2／3 分開呈現；不能把三階段分數差直接當成集成或 FS 的改善量。
- 誤差線是 seeds 的樣本標準差，不是資料獨立重抽樣的信賴區間。相同 Test 列與相關的模型不能視為獨立觀測。
- 沒有補作同協定 Retrain、選最佳分類器或統計顯著性檢定。
- 原始公司最終 status 標籤不等於已驗證的提前一年破產事件預測。
- SVG 保留文字與向量；PNG 為 300 dpi。圖寬 7.2 英吋，縮至 180 mm 時約為原尺寸 98%，仍須依論文版心檢查。
- 絕對指標圖統一使用 AUC 0.45–1、F1-score 0–0.35、Recall 0–1 的明示刻度，各指標在所有圖保持相同範圍。熱圖使用差值刻度，不能與絕對分數混讀。

重建入口：`scripts/plots/build_bankruptcy_experiment_figures.py`；從專案根目錄執行：

```powershell
.venv/Scripts/python.exe scripts/plots/build_bankruptcy_experiment_figures.py
```

[圖表契約、來源雜湊與限制](chart-contracts.json)；[數值與版面檢查](quality-checks.json)；[視覺檢視紀錄](visual-review.json)。

## 證據帳

| 主張 | 證據 | 狀態 |
|---|---|---|
| 六個歷史候選含完整三策略×四採樣網格 | 原 CSV、盤點 SHA-256、source-grid | 檔案身份與覆蓋已核對；歷史執行未重建 |
| 固定年份圖來源涵蓋 90 配置與 28,350 列指標 | manifest、配置 checksum、重建 metrics | 本次來源與作圖核對通過 |
| 固定方法 FS 差值可由同 seed 原指標重算 | 400 paired rows、正式差值 CSV | 三項指標重算一致 |
| 圖面不足以證明最佳分類器、跨階段效益或顯著性 | 不同協定／seed 相依性／未補執行 | 明確限定，仍需補同協定證據 |
