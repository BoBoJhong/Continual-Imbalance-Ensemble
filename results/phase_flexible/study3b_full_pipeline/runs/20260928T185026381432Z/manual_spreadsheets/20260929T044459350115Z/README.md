# Study 1–2 形式的獨立結果試算表

納入 seeds：42, 123, 456（3/10 完成）。
全部檔案為 UTF-8 BOM CSV，可直接用 Excel 開啟；不是把所有結果塞入一張表。

- `01_baseline/<分類器>/`：每個分類器獨立檔案，四採樣並排，沿用 `bk_*_compact_summary.csv` 與 `compact_AUC_only` 等命名。
- `02_ensemble/fifo/<分類器>/`：各分類器 FIFO 獨立檔案，四採樣並排。
- `02_ensemble/<集成方式>/`：跨採樣、跨分類器及合併集成分別存檔。
- `03_fs/<FS方法>/`：MI、CART、RFE、SHAP 分開存檔，50%／80% 保留率列為不同配置；none 為獨立對照。

`compact_summary`／`pooled_summary` 是先對每個 seed 合併 Test 計算，再對已完成 seeds 取平均。
`annual_mean_summary` 是年度等權平均，另檔保存。`pooled_by_seed` 保存各 seed 原值。
`by_year_raw` 保存完整逐年原值；每個採樣另有獨立 `*_by_year.csv`。
集成與 FS 的 `thesis_format` 為各 Test 年度的 seed 平均，`AUC_only` 等為年度×配置寬表。
所有結果保留 AUC、AP（欄名 PR_AUC）、F1、G-Mean、Recall、Precision，不挑 Test 最佳列。

資料源：`C:\0_git workspace\Continual-Imbalance-Ensemble\results\phase_flexible\study3b_full_pipeline\runs\20260928T185026381432Z`。原始預測與既有論文快照不變；本次僅另行匯出試算表。
FS 基礎集成依年度／seed 的 Validation 選擇不同，查 `by_year_raw` 的 `selected_config` 與 `members`。
檔案總覽見 `file_index.csv`。本匯出為固定快照，不把未完成 seeds 混入。
