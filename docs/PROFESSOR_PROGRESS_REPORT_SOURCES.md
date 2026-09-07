# 教授進度報告：主張與來源台帳

更新日期：2026-09-08

本表是 `PROFESSOR_PROGRESS_REPORT.md` 的 claim ledger。數值主張必須能回到
版本化資料；推論與限制不得寫成已被實驗確認的事實。

| ID | 主張 | 類型 | 主要來源 | 狀態／限定 |
|---|---|---|---|---|
| C1 | Rolling pooled `New_under` AUC/F1 = 0.8403/0.3287 | 結果 | `results/phase_flexible/rolling_bankruptcy/rolling_pooled_summary.csv` | verified |
| C2 | AdaptiveChoice_AUC 對 Equal6 的年度 AUC 經 Holm 校正顯著 | 統計 | `results/phase_flexible/rolling_bankruptcy/rolling_wilcoxon_holm.csv` | verified；Holm p=0.03125 |
| C3 | AdaptiveChoice_AUC 未顯著優於 `New_under` | 統計 | 同上 | verified；四項指標皆未通過 Holm |
| C4 | Rolling 共 10 個測試年度、每方法 33,636 筆 pooled 預測 | 結果 | `rolling_by_year.csv`、`rolling_predictions.csv` | verified |
| C5 | Fair ablation 的 no-FS `New_under` 高於 DAWCE 與 Equal6 | 結果 | `results/phase5_weighted/bk_fair_ablation_summary.csv` | verified；15 splits 共用測試期，統計只作診斷 |
| C6 | 多數 Old/New scope 的 r80 Jaccard 高於 r50 | 結果 | `results/phase3_feature/stability/bankruptcy_feature_stability_summary.csv` | verified；pair 非獨立；old_new 資料集合不變 |
| C7 | Bankruptcy raw data 為 78,682 列、23 欄 | 原始資料 | `data/raw/bankruptcy/american_bankruptcy_dataset.csv` | verified |
| C8 | 結果共有 985 個 CSV、199,972 列，strict audit 無讀取問題 | QA | `docs/result_artifact_audit.json` | verified at 2026-09-08 |
| C9 | 19 項自動測試通過，compileall 通過 | QA | 本地驗證命令 | verified at 2026-09-08 |
| C10 | 結果與 raw data 已建立 SHA-256 manifest | provenance | `results/RESULT_MANIFEST.json` | verified；manifest 記錄產生當下 Git 狀態 |

## 圖表來源

| 圖 | 輸入 | 輸出 |
|---|---|---|
| Rolling pooled performance | `rolling_pooled_summary.csv` | `docs/figures/professor_progress/01_rolling_pooled_performance.png` |
| Annual rolling AUC | `rolling_by_year.csv` | `docs/figures/professor_progress/02_rolling_annual_auc.png` |
| Fair ablation | `bk_fair_ablation_summary.csv` | `docs/figures/professor_progress/03_fair_ablation_no_fs.png` |
| Feature stability | `bankruptcy_feature_stability_summary.csv` | `docs/figures/professor_progress/04_feature_stability.png` |
| Annual class imbalance | Bankruptcy raw CSV | `docs/figures/professor_progress/05_class_imbalance_by_year.png` |

重新產圖：

```powershell
python scripts/plots/generate_professor_report_figures.py
```
