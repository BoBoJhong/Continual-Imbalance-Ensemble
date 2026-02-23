# 接下來可以做什麼

實驗 01～10 與進階 DES、比例實驗都已完成，以下是依目標整理的**下一步**。

---

## 一、以碩論畢業為目標

| 步驟 | 做什麼 | 參考 |
|------|--------|------|
| **1. 寫論文** | 依「方法」「實驗」「結果與討論」章節撰寫；對照老師方向用 **docs/TEACHER_REQUIREMENTS_CHECKLIST.md**，實驗正當性用 **docs/EXPERIMENT_VALIDATION.md**，碩論適用性用 **docs/THESIS_READINESS.md** | 結果表：`results/summary_all_datasets.csv`、`results/bankruptcy_all_results.csv`、`results/feature_study/`、`results/des_advanced/`、`results/proportion_study/` |
| **2. 補 mean±std（若老師要求）** | 跑多 seed：`python scripts\run_multi_seed.py`，產出 baseline 的 mean±std；若要 09/10 的 mean±std，可改腳本用多個 `random_state` 重跑並彙總 | `results/baseline/bankruptcy_baseline_mean_std.csv` |
| **3. 統計檢定（若老師要求）** | 在多 seed 結果上做 Wilcoxon signed-rank test（R 或 Python），在論文中加一小節「方法間顯著性」 | 可用現有 CSV 事後分析 |
| **4. 口試準備** | 用 **docs/TEACHER_REQUIREMENTS_CHECKLIST.md** 對照「每條老師要求對應到哪裡」；用 **docs/EXPERIMENT_VALIDATION.md** 說明無洩漏、切割、baseline 定義 | — |

---

## 二、以後續論文投稿為目標

| 步驟 | 做什麼 | 參考 |
|------|--------|------|
| **1. 跑齊並檢查結果** | 確認 09、10 都跑過：`results/des_advanced/bankruptcy_des_advanced_comparison.csv`、`results/proportion_study/bankruptcy_ratio_comparison.csv`；必要時重跑 `python experiments\09_bankruptcy_des_advanced.py`、`python experiments\10_bankruptcy_proportion_study.py` | **docs/RESEARCH_EXTENSIONS.md** 第五節 |
| **2. 分析與寫作** | 進階 DES：比較 baseline vs 時間加權 vs 少數類加權 vs combined，寫「方法→實驗→討論」；比例實驗：畫「比例 vs AUC」圖、寫「何時適應策略領先 retrain」 | 09 結果：時間/combined 略優於 baseline；10 結果：可畫表與圖 |
| **3. 可選加強** | (1) 多 seed 跑 09/10 報 mean±std；(2) 調 `time_weight_new`、`minority_weight`（如 1.5, 2.0, 2.5）做小實驗；(3) 把進階 DES 套到 Stock/Medical（改 07、08 或加 11、12） | `common_des_advanced.run_des_advanced(..., time_weight_new=, minority_weight=)` |
| **4. 選會議/期刊** | 依主題選：continual learning、imbalanced learning、ensemble、applications（bankruptcy/finance） | — |

---

## 三、可選的技術加強（有時間再做）

| 項目 | 說明 |
|------|------|
| **進階 DES 用於 Stock/Medical** | 在 07、08 或新腳本中改為呼叫 `run_des_advanced`，比較 baseline DES vs time/minority/combined，產出三資料集對照表 |
| **權重搜尋** | 對 `time_weight_new`、`minority_weight` 做小網格（如 1.0, 1.5, 2.0, 2.5），記錄 AUC/F1，寫入 CSV 或圖表 |
| **比例實驗多 seed** | 實驗 10 用多個 `random_state` 重跑，產出各比例、各方法的 mean±std，方便做誤差棒或檢定 |
| **G-mean / balanced accuracy** | 在評估流程加算 G-mean 或 balanced accuracy，寫入結果表與論文 |

---

## 四、一鍵檢查清單

- [ ] 論文初稿：方法、實驗設計、結果表與討論
- [ ] 結果檔齊全：`summary_all_datasets.csv`、`bankruptcy_all_results.csv`、`feature_study/`、`des_advanced/`、`proportion_study/`
- [ ] （可選）多 seed：`run_multi_seed.py` 跑過、mean±std 已寫進論文
- [ ] （可選）統計檢定：若老師要求，已做並寫入論文
- [ ] 口試對照：TEACHER_REQUIREMENTS_CHECKLIST、EXPERIMENT_VALIDATION 已看過

---

**結論**：**接下來**優先「寫碩論」＋「必要時補 mean±std／統計」；若要衝後續論文，再以 09/10 為主做分析與寫作，並可依時間加權上述可選加強。
