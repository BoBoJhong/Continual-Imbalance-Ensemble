# 本專案論文檔案

先讀 README.md 與 ../docs/thesis_environment/README.md。
研究規範沿用 ../.agents/skills/research-paper-agent/SKILL.md；
../docs/研究方向.md 為研究定義來源。

本地論文修改使用 ../.agents/skills/thesis-local-edit/SKILL.md；
要求 Overleaf 比對或同步時使用 ../.agents/skills/thesis-overleaf-sync/SKILL.md。
交接規格與指令見 ../docs/thesis_environment/HANDOFF.md。

- THESIS_FULL.md 是既有 Markdown 草稿；latex/ 是新增可編譯模板。依使用者指定的範圍逐章遷入。
- 原 NCU IM class、字體與 main.tex 已保留；原檔雜湊在 ../docs/thesis_environment/。
- 真實數字由 ../results/ 的明確 run 匯出，不手改 generated/ 或原結果 CSV。
- 更換來源時更新 ../config/thesis_sources.json，執行 sync_results.py，再檢查 --check。
- 分開固定年份公司最終狀態 target、滑動視窗回溯事件 target、pooled 與年度平均。
- 保留每段數字的 run、seed、篩選條件及來源；不捏造結果與文獻，不按 Test 選最佳方法。
- 只有資料表更新不代表摘要、結論或正文的數字已同步，修改後逐段核對。
- 使用者要求線上同步時，先讀 ../config/thesis_overleaf.json 確認指定目標；
  首次核對存取、主文件、編譯器及線上差異。記錄目標網址本身不代表要求立即同步。
- 未提供的實驗或格式要求列為待辦，不把模板占位或編譯成功當作已完成研究。
