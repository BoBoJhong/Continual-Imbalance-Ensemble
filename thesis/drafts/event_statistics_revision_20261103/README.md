# 事件配對統計整合與全文一致性修訂

本輪依使用者核准的四項工作修改本地 LaTeX。主要原稿為 thesis/latex/，未修改 THESIS_FULL.md，未操作線上 Overleaf。before/ 保存本輪前的工作稿、相關證據及原 PDF；該工作稿已有未提交修訂，本輪未回退。

| 核准項目 | 處理與位置 | 證據／界線 |
|---|---|---|
| 第四章補事件統計 | 新增事件公司配對分析小節、AUC 差值表與附錄完整 165 項檢定 | event-statistics-20261103/；原分析 9,999 次，未重跑或改模型 |
| 第五章修結論 | 回應三個研究問題；區分平均、校正後顯著與未達顯著；移除成員／規則相對重要性的斷言 | New3、V-static 相對 Equal6 的 AUC 有校正支持；V-static F1 增加未通過；未檢定兩者直接差異 |
| 第二章修矛盾／定位 | TabM、TabICL 只列背景；主 Baseline 每組 24 候選；用三項受控比較界定實證補充價值 | 不新增首次性、普遍優越性宣稱；未逐篇重審全部引文 |
| 來源及編譯 | 歷史分析程式快照綁定、事件分析來源與生成表綁定；同步檢查與嚴格字體原生編譯 | source-audit.json；verification.json；build-report.json |

## 連帶一致性修訂

第一章、第三章、中英文摘要與附錄同步事件分析的完成範圍、兩標籤各自的重抽次數與校正家族。主要目標明確寫為回溯重建的代理標籤，年度可用性是協定假設，不冒稱實際申請日或財報公開日已驗證。英文摘要精簡至單頁。第四章靜態比較小節避免標題與表格分頁。

## 來源處理

- 原最終狀態分析：results/statistical_tests/fixed_two_experiments/20261027T051346477136Z，49,999 次重抽。
- 事件分析：results/statistical_tests/fixed_two_experiments_event/20261031T120706072983Z，9,999 次重抽。
- 兩批各 454 項來源、5 項輸出已核對，原分析 manifest 與 results/ 數值均不改寫。
- 原分析程式快照來自 Git HEAD，其完整位元組 hash 與舊 manifest 相同；舊／現行 sha、holm、sort_inputs、weighted_scores、probability_for 的 AST 相同。main 的差異為支援 run／protocol、事件標籤的群聚檢查與正類公司計數等，未改統計核心。
- config/thesis_sources.json 的 source_snapshots 只允許論文 drafts 下的 .py 快照；預測資料不能由這個機制替換。四項新來源綁定測試涵蓋歷史程式、竄改、資料替換與未知來源。
- 既有表格由原擁有腳本重建以更新相依證據；既有 .tex／.csv 內容與修訂前逐一相同。新增事件統計表由 export_event_statistics.py 匯出，不手改 generated/。
- 共用表格匯出器變動連帶重建 method_inventory、method_clarifications、chapter_results、chapter_figures、event_results 證據。這些刷新不代表重新訓練或重新驗證全部外部文獻。

## 主張與數字

主張查核表 ES-01–05 同步在 docs/研究方向.md；來源為事件分析 paired_tests.csv 的 experiment、fs_a、method_a、metric 篩選。所有指標為先算每個 seed 的 pooled 指標再平均；不是平均機率的新模型。十個 seed 為 42、123、456、789、2024、7、21、84、168、999。

## 仍保留的研究限制

代理事件日期尚未逐筆驗證；驗證正例僅 33；集成與 Baseline 的調參／驗證條件不同；事件尾端 p 值有 Monte Carlo 精度限制；單一資料與固定年份、不證明持續適應。下一步可增加重抽次數或規劃調參模型池控制，但未在本輪執行。
