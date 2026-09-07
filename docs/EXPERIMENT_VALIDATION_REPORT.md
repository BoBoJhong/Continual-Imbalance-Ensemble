# 實驗與論文資料科學審查報告

審查日期：2026-09-08  
審查範圍：Bankruptcy 主實驗之原始資料、時序切割、前處理、取樣、模型池、Study 1–4 結果、統計檢定、結果產物與 `thesis/THESIS_FULL.md`。

## 結論

**整體判定：核心結果可作為方向性與探索性證據，但在完成下列修正與補充實驗前，仍屬 Needs revision，不建議把所有目前的顯著性敘述當成最終投稿結論。**

目前最可信的結果是十個年度 walk-forward 測試：固定 `New_under` 的 pooled AUC/F1 為 0.8403/0.3287；AdaptiveChoice 為 0.8221/0.3283。Holm 校正後，AdaptiveChoice 僅在 AUC 上顯著優於 Equal6，未顯著優於 `New_under`。因此專案支持「近期資料訓練的強單模型通常占優」與「validation-guided weighting 可避免等權納入弱模型」，但尚未證明 DAWCE、ROSS 或動態選擇普遍優於強單模型。

## 已完成的驗證

- 測試：`pytest -q`，19 tests passed。
- 結果產物：985 個 CSV、199,972 列可讀；嚴格稽核無不可讀檔案，且未發現完全重複列。
- 原始資料：78,682 筆、23 欄、1999–2018、8,971 家公司；破產 5,220 筆（6.63%）；無缺失值、無無限值、無完全重複列、無重複 company-year。
- 語法檢查：`python -m compileall -q src experiments scripts` 已通過；一鍵 runner 的 21 個維護中入口亦通過路徑驗證。
- 可追溯性：`results/RESULT_MANIFEST.json` 已記錄結果與 raw data 的 SHA-256、CSV profile、套件版本及 Git 狀態。
- 主要 rolling 數值、15-split 比較與公平消融皆由儲存的 raw 結果重新抽查，與論文主要表格方向一致。

## 資料與實驗單位的重要發現

此資料是公司年度 panel data，而不是彼此獨立的公司樣本。1999–2014 訓練期有 8,503 家公司，2015–2018 測試期有 3,700 家，其中 3,232 家曾出現在訓練期，占測試公司 87.35%。此外，每家公司在資料內的 `status_label` 不隨年度改變。

模型已移除 `company_name`，因此沒有直接把公司識別碼當特徵；然而目前研究問題實際是「已知與新公司混合的未來年度風險辨識」，不是「對完全未見公司的 entity-holdout 泛化」。同一公司跨年重複也使年度批次之間可能存在公司層級相關性。論文已據此降低獨立性主張；投稿前建議增加按公司分群的 sensitivity analysis 或 entity-holdout 評估。

## 主要問題與風險

| 嚴重度 | 問題 | 影響 | 建議 |
|---|---|---|---|
| 高 | 特徵穩定性把 15 個 split 的 105 個兩兩 Jaccard 當 paired Wilcoxon 樣本 | 每個 split 重複出現在 14 個 pair，造成偽重複；`p < 1e-6` 不能作確認性推論 | 保留 Jaccard 均值作描述；改用 split-level permutation/bootstrap 或重複 temporal resampling |
| 高 | 10-seed 實驗不是主 XGBoost 流程的完整重複 | 該流程使用 LightGBM/block-CV；舊結果的 DES 未傳入 seed；且 DES vs retrain AUC 並非顯著 | seed 傳遞已修正，但舊 CSV 尚未重跑；仍須以主流程與 10+ seeds 執行 |
| 高 | 成本式 `FPR + r × FNR` 被稱為 Expected Cost | 未依類別盛行率加權，不能直接解讀成每家公司或金額期望成本 | 改稱「盛行率中性的條件錯誤成本分數」；部署分析使用 `(1-π)c_FP FPR + πc_FN FNR` 與實際成本 |
| 高 | 公司跨年度重複，年度測試並非 entity-level 獨立 | p-value 與泛化範圍可能被高估 | 加 company-cluster bootstrap、entity-holdout 或對已見/未見公司分層報告 |
| 中 | 15-split 多數共享 2015–2018 Test，且訓練窗巢狀重疊 | Wilcoxon p-value 只能視為診斷性，不能當 15 次獨立複驗 | 以 rolling 結果作主證據；15-split 報效果方向與敏感度 |
| 中 | 舊版 DES/DCS 以訓練過模型的 Old+New 樣本作 DSEL | competence 估計可能偏樂觀，且比較不能代表一般 DES/DCS 上限 | 以獨立 validation DSEL 重跑，並明確區分 legacy 與新版流程 |
| 中 | PR-AUC 已在程式實作，但主結果與論文未一致報告 | ROC-AUC 在嚴重不平衡時可能過度樂觀 | 重跑或彙整主要方法的 PR-AUC，連同信賴區間報告 |
| 部分修正 | 既有結果產生時 `SMOTEENN` 未明確套用 YAML 內部參數 | 新舊實驗設定可能不同 | 現行 sampler 已接線 YAML，論文已區分歷史與現行設定；採用新設定的主結果仍須重跑 |
| 已修正 | `scripts/run/_write_common_dcs.py` 曾有 `IndentationError` | 全專案 compileall 曾無法通過 | 已改為無副作用的相容性檢查，compileall 通過 |
| 已修正 | `bankruptcy_feature_stability_errors.csv` 曾為空白檔 | 驗證器無法讀取 | 已由原產生器輸出固定欄位 schema，strict audit 通過 |
| 部分修正 | 多數舊結果 CSV 沒有逐列 `seed` 欄 | 舊產物的 provenance 有限 | 已新增整庫 manifest；新主流程仍應逐列保存 seed 與 experiment ID |

## 結果抽查

### Rolling walk-forward（10 個年度批次）

| 方法 | pooled AUC | pooled F1 |
|---|---:|---:|
| New_under | 0.8403 | 0.3287 |
| DAWCE-AUC | 0.8237 | 0.3275 |
| AdaptiveChoice-AUC | 0.8221 | 0.3283 |
| Retrain-under | 0.8116 | 0.2943 |
| Equal6 | 0.7971 | 0.2830 |

在 16 項 rolling 比較作 Holm 校正後，AdaptiveChoice 對 Equal6 的 AUC 為 raw `p=0.001953`、Holm `p=0.031250`；AdaptiveChoice 對 `New_under` 的 AUC/F1 均不顯著。

### 公平消融（15 splits，診斷性）

- 無 FS `New_under`：AUC 0.8498、F1 0.2143。
- 無 FS DAWCE-F1：AUC 0.8336、F1 0.1886。
- 無 FS selected weighting 對 Equal6：AUC +0.0250、F1 +0.0275；但 splits 共用測試期，p-value 不作獨立確認性證據。

## 前處理與資料洩漏審查

Study 3 與 rolling 新流程使用 training-only imputation/scaling，相關測試已通過。`experiments/_shared/common_bankruptcy.py` 的 `get_bankruptcy_year_split` 仍會分別對 Old、New、Test 呼叫補值程序，屬轉導式前處理限制；但本次原始 bankruptcy 檔案沒有缺失值，因此此路徑目前不會改變數值結果。論文不應再概括宣稱所有歷史實驗完全 leakage-free。

## 投稿前最低必要工作

1. 重新設計特徵穩定性的推論單位，移除 105 個相依 pair 所產生的確認性 p-value。
2. 執行真正對齊主流程的 multi-seed replication，確保 XGBoost、sampling 與 DES/DCS 的所有隨機元件均由 seed 控制。
3. 對主要模型補報 PR-AUC，以及 bootstrap 信賴區間。
4. 增加 company-cluster 或 entity-holdout 敏感度分析，明確界定對新公司的泛化能力。
5. 若成本敏感分析要作部署建議，加入測試期盛行率與可辯護的 FP/FN 金額成本。
6. 修復全專案語法檢查錯誤，並讓空錯誤報表有固定 schema。

## 論文更新狀態

本次已同步修正 `thesis/THESIS_FULL.md` 中的資料單位、統計相依性、特徵穩定性、multi-seed、成本分數、DES/DCS 比較、取樣參數與軟體版本等敘述。這些是文字與主張層級的修正；上列需重跑的實驗仍不得視為已完成。
