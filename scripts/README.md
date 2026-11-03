# 工具導航

由專案根目錄執行。先看[實驗索引](../docs/實驗索引.md)確認 target、協定與結果版本；維護清單不等於同協定論文比較集。

| 工作 | 工具目錄 | 入口與行為 |
|---|---|---|
| 查看維護實驗 | [run](run/README.md) | run_all_experiments.py --list；省略 --list 會執行並寫入結果 |
| 結果清冊與查核 | [analysis](analysis/README.md) | 逐檔分類、完成狀態、數值重播、切割、消融、統計與匯出；各自來源不同 |
| 研究圖／Phase 1 圖 | [plots](plots/README.md) | 現行流程圖與保留的 baseline 圖；9 支舊 Phase 2／3 圖已封存 |
| 進度報告 | [reports](reports/README.md) | Study 3B 快照報告與分析；新建時間戳版本 |
| 資料準備 | [data](data/README.md) | 原始來源、衍生資料與 MOVER 工具；各自讀／下載／寫入資料 |
| 歷史追溯 | [legacy](legacy/README.md) | 23 支舊圖表、匯出、遷移、批次與統計工具；不在維護 runner |

## 常用查看指令

```powershell
python scripts/run/run_all_experiments.py --list
python scripts/analysis/study3b_pipeline_status.py results/phase_flexible/study3b_full_pipeline/runs/20260928T185026381432Z
```

清冊重建會另建 metadata 快照：

```powershell
python scripts/analysis/build_result_catalog.py
```

## 保存與歷史工具

23 支工具的原／新路徑、來源 hash 與原文 ZIP 見 [legacy](legacy/README.md)。原計算函式保留；六支舊圖表的 mkdir 與舊 Excel 輸出改為明確執行才啟動，import 不再讀寫研究表或搬移成果。明確執行舊工具仍可能訓練或寫入／搬移歷史輸出；本輪未執行這些動作。

Phase 1 圖表 helper、資料準備、MOVER 修正與 TabNet 診斷保留；不僅因未列入 runner 就判定過時。完整範圍見[第四輪紀錄](../docs/第四輪整理紀錄_20261012.md)。
