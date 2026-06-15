# 實驗流程圖

本目錄集中放置研究與實驗流程的 PlantUML 原始檔。

| 圖檔 | 說明 |
| --- | --- |
| `phase1-baseline-flow.puml` | Phase 1：單模型 Baseline 實驗流程 |
| `phase2-ensemble-flow.puml` | Phase 2：Static / DES / DCS 集成流程 |
| `phase3-fs-ensemble-flow.puml` | Phase 3：特徵選取與集成流程 |
| `phase4-batch-adaptive-flow.puml` | 目標方法：任意時間批次下的 ROSS、DAWCE 與 AdaptiveChoice rolling 流程 |
| `phase4-batch-adaptive-flow.png` | Phase 4 研究架構圖的論文／簡報用輸出 |

`phase4-batch-adaptive-flow.puml` 採用資料科學方法架構圖的語意配色：

- 藍色：時序資料與資料切分。
- 紫色：候選模型與方法生成。
- 黃色：Validation 選擇。
- 綠色：最終選定與部署策略。
- 灰色：未知資料、評估與決策紀錄。

圖中的回饋虛線代表下一批標籤取得後的 rolling update；此迴圈為研究目標流程，目前核心元件已實作，完整 rolling runner 尚待整合。
