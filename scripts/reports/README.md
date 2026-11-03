# 報告工具

[回工具導航](../README.md)。

- [build_study3b_advisor_report.py](build_study3b_advisor_report.py)：指定 run 的完成 seeds 快照；預設新建 docs/reports/YYYY-MM 時間戳 Markdown，同目錄保存查核 JSON，連結依實際輸出位置計算。
- [build_experiment_analysis_report.py](build_experiment_analysis_report.py)：已有完整流程來源的分析資料匯出，須指定 --output；不自動合併所有最新核心結果。

8 支 FS／CART／Excel 匯出及 archive／migrate／sync 工具已歸入 [legacy](../legacy/README.md)。明確執行可能訓練、寫表或搬移結果；本輪只封存程式。

閱讀已產生文件見 [docs/reports](../../docs/reports/README.md)。
