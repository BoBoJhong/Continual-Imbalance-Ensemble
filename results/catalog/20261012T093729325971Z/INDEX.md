# 結果分類入口

產生時間（UTC）：2026-10-12T09:38:21.897974+00:00

共 14440 份原有檔案；輸出清冊目錄不納入自身盤點。分類依證據與範圍，不按修改時間判定過時。

[所有逐檔紀錄](catalog.json)；[內容相同檔案](duplicates.md)

## 已選定的完成 run

| 研究 | run | target | manifest 狀態 |
|---|---|---|---|
| 固定年份 Study I／II | [results/phase2_ensemble/standard_dynamic/runs/20261004T202331745995Z](../../phase2_ensemble/standard_dynamic/runs/20261004T202331745995Z) | 原始公司最終 status | completed |
| 滑動視窗完整 Study 3B | [results/phase_flexible/study3b_full_pipeline/runs/20260928T185026381432Z](../../phase_flexible/study3b_full_pipeline/runs/20260928T185026381432Z) | 回溯重建事件 target | completed |

## 檔案分類

| 類別 | 檔案數 | MiB | 使用限制 |
|---|---|---|---|
| [現行核心完成 run](current_core.md) | 459 | 205.96 | 已選定的固定年份 run 且 manifest completed；不代表全部 baseline 已完成。 |
| [現行擴充完成 run](current_extension.md) | 12345 | 1800.83 | 已選定的完整滑動視窗 run，與核心 target 分開。 |
| [部分 seeds／階段快照](interim_snapshot.md) | 409 | 5.34 | 沿用快照身份，不自動混入完成後的 seeds。 |
| [失敗 run](failed_run.md) | 0 | 0.00 | manifest 記錄 failed；保留診斷，不用作正式成效。 |
| [試跑／smoke](smoke.md) | 0 | 0.00 | 有限設定或樣本；命名分類仍需核對其 metadata。 |
| [中途／恢復檔](checkpoint_partial.md) | 9 | 0.01 | 只標記該檔，不能據此推論整個實驗失敗。 |
| [已退出研究範圍](retired_scope.md) | 0 | 0.00 | 保留沿革；退出不代表方法無效。 |
| [Test-selected／oracle 探索](oracle_exploration.md) | 9 | 0.03 | Test 參與選擇或事後排名；不是無偏選擇證據。 |
| [歷史協定／匯出](historical_protocol.md) | 837 | 10.60 | 不同切割、方法或檢定來源；按版本引用。 |
| [其他擴充／探索](exploratory_extension.md) | 303 | 451.33 | 逐 run 核對來源；部分含已退出方法，不能整個目錄判廢。 |
| [輔助資料／分析](auxiliary.md) | 60 | 0.49 | 跨領域與穩定性等獨立分析；適用範圍需交代。 |
| [審查／provenance 快照](audit_snapshot.md) | 3 | 0.73 | 查核時點的證據，不代表當前所有檔案已涵蓋。 |
| [人工導航](documentation.md) | 1 | 0.00 | 目錄使用說明。 |
| [待確認來源](needs_review.md) | 5 | 0.82 | 缺少足夠規則或完成證據，未宣稱過時或有效。 |

## 解讀與重建

分類不是全部數值重新驗證；不同 target、協定、部分 seeds 與 pooled／年度口徑仍須分開。

政策：[config/result_catalog_config.json](../../../config/result_catalog_config.json)；產生工具：[scripts/analysis/build_result_catalog.py](../../../scripts/analysis/build_result_catalog.py)

由專案根目錄執行 python scripts/analysis/build_result_catalog.py；每次建立新時間戳快照，更新 catalog/README 導航。
