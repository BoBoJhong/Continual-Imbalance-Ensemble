# 圖表工具

[回工具導航](../README.md)。

| 類別 | 入口／範圍 | 定位 |
|---|---|---|
| 最新主圖 | [build_simplified_research_diagrams.py](build_simplified_research_diagrams.py) | 概括三階段設計；圖面不表示所有 baseline 已執行 |
| 破產三階段結果圖 | [build_bankruptcy_experiment_figures.py](build_bankruptcy_experiment_figures.py) | 來源與配置 checksum 檢查；輸出十張 PNG／SVG、作圖 CSV 至 assets/research/experiment-results/bankruptcy；協定分開 |
| 詳細方法圖 | [build_research_method_diagrams.py](build_research_method_diagrams.py) | 詳細視圖與沿革 |
| 報告圖 | [generate_professor_report_figures.py](generate_professor_report_figures.py) | runner 保留；逐張核對來源，部分為舊擴充結果 |
| Phase 1 比較 | visualize_phase1_* | 保留 baseline／深度學習比較；驗證協定仍須對齊 |
| Phase 1 共用 | [phase1_baseline_plotting.py](phase1_baseline_plotting.py) | 多支圖表 import，保持原位 |
| 舊 Phase 2／3 | [legacy 圖表](../legacy/README.md) | 9 支已封存；部分選 Test 最佳列，只能按歷史／探索協定引用 |

上述歷史工具整理當時沒有重畫圖或覆寫 assets／results。最新破產結果圖依使用者要求另行生成至 assets；原始 results 不變。舊圖的 mkdir 只在實際執行觸發；import 不輸出成果。
