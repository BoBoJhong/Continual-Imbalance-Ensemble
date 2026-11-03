# 兩個論文技能的獨立前向演練

此報告僅記錄離線演練。沒有操作瀏覽器、線上帳號或實際 Overleaf；所有寫入都在 `scratch/thesis_skill_forward_test/`。技能與必要文件、論文及結果來源唯讀。不是新版論文交付，也不是線上同步成功證明。

## 情境一：「用本地論文技能修訂第三章研究方法，交付新版論文 PDF」

### 實際選稿與發現

- 依技能讀取 `thesis/AGENTS.md`、`thesis/README.md`、環境 README、HANDOFF、共同研究技能的 project-context 與 writing-workflows。
- 讀取 `thesis/THESIS_FULL.md` 第 382–693 行的第三章（312 行），隔離摘錄於 `scenario1/source-method.md`。
- 實際 `03_method.tex` 僅示例實驗設計、Recall 方程與資料洩漏提醒；其 SHA256 `c00aff0e966fd95931a40a1fcdea9d1f8575b8f2f87e0f96373b2d4dc6fe7206` 等於 `original-files.sha256.json` 的原模板雜湊，不能據此宣稱第三章已遷入。
- `main.tex` 確實載入 `chapters/03_method.tex`，但也載入其他原示例章節。現有 passed 編譯報告對應模板 PDF；它不是修訂 Markdown 後的論文成果。
- Markdown §3.1 和 §3.5仍含歷史 New 2012–2014 訓練及重疊 DSEL 協定；`docs/研究方向.md` 的 C39–C44 與修正協定指出正式固定年份新 run 使用 New-fit 2012–2013、2014 分為 DSEL／Calibration。直接讀取正式 run 的 `partition_summary.csv`，確認五個分區及年份。因此此任務需要內容修訂，不能只把 Markdown 原樣轉進模板。
- 實際唯讀執行 `python scripts/thesis/sync_results.py --check`，回傳 `PASS: pinned sources and generated thesis tables are current`。此通過只表示指定來源／生成表格一致，不表示第三章全文已與最新協定一致。

### 按技能會如何完成授權任務

1. 以 Markdown 第三章為原稿，保留歷史協定身分，對齊現行正式固定年份方法；不將最終狀態 target 與 Study 3B 回溯事件 target 混寫。Retrain 或最佳分類器尚缺證據時保留缺口，不新增實驗成果。
2. 為 PDF 交付將使用者要求的第三章遷入 LaTeX，記錄此章後續主要維護 `chapters/03_method.tex`；必要引文以實際查證後的 `references.bib` 相依變更處理。此演練沒有進行新的文獻查證或正文修改。
3. 正式任務使用本專案的 `thesis.py build --document main`，讀最終 log，核對 Biber、引用、缺字、溢出，渲染修改頁並視覺檢查。不能以 results-preview 或原模板 PDF 取代修訂成果。
4. 若只遷第三章，其餘仍為模板，交付應明確稱「第三章已修訂的 LaTeX 預覽／論文整合稿」，不能稱整份論文已遷入或完成。若用戶實際需要完整正式論文 PDF，單章範圍與其餘未遷入章節造成的缺口需要明列；不能自行轉整份。
5. 可編譯後才以指定檔案建立 `handoff.py prepare`、執行 `check`。不把本地修訂授權擴張成線上寫入。

本情境沒有真的修訂／編譯主論文，故不能報稱交付新版論文 PDF。演練成功辨認原稿、必要遷入與交付驗證界線。

## 情境二：「同步已編譯的指定變更」

### 真正執行的離線鏈

`independent_forward_test.py` 以 import 直接呼叫現有 `scripts/thesis/handoff.py` 的 prepare、check、import_backup、baseline、plan；ROOT 參數全部指向隔離 fixture-project，不改動真正 config。本次重跑共 28 個案例，並對預期拒絕及三組計畫動作加入斷言。

最小 LaTeX 稿使用 main.tex 及三個指定內容檔。真正執行本機 XeLaTeX（停用 shell escape），PDF 與 log 位於 fixture-project/thesis/latex/build；依實際輸入雜湊、PDF 雜湊及 compiler return code 製作相容編譯報告。這是離線最小稿的真實編譯，非主論文編譯，也沒有 Biber／NCU 格式驗證。報告的 `offline_test` 和 font_mode 都明確標示測試用途。

兩份完整 fixture 來源 ZIP 是自行建立的離線素材，都附 `OFFLINE_TEST_PROVENANCE.json`；不是從線上下載。另建衝突版、全部相同版各一份 ZIP，逐份匯入新的目錄，不靠改寫已保存的備份模擬新取回。使用全零 fictional project ID，完全沒有存取該 URL。

實際 import_backup 保留 `source.zip`、產出 `source-receipt.json`，remote-root 指向 `output/source`。baseline／plan 都真正經過 verified_backup 的 ZIP 與完整檔案清單雜湊核對。本輪位置見 `scenario2/zip-run-location.json`。相同文字案例使用原始 bytes，避免 Windows 換行正規化改變雜湊；第一次重跑曾因測試素材 LF／CRLF 不同而被判 update，調整為實際 bytes 後預期相同案例通過。這是測試素材問題，不是交接工具失效。

### 實際觀察

| 案例 | 行為 |
|---|---|
| 成功編譯並 prepare/check | 固定 files/ 快照、編譯報告、PDF、來源與目標通過；manifest 明示 prepared_not_uploaded |
| 四份完整 fixture ZIP 匯入 | 都保留 ZIP 與收據，完整檔案清單可驗證；不是線上下載證據 |
| baseline 後新備份仍為舊版 | 第三章 update；原先不存在的 appendix create；相同內容 current already-current |
| 第二份備份第三章不同於 baseline | 第三章 conflict；保留基準與本地／目前遠端雜湊 |
| 第二份備份全部等於固定快照 | 全部 already-current，即使不同於旧基準也可略過 |
| baseline 重複寫入 | 拒絕覆蓋既有基準 |
| 工作稿建立包後有變 | check 拒絕 stale package |
| files/ 快照被改写 | check 拒絕與 compiled source 不一致 |
| 包內 PDF 被改写 | check 拒絕 build proof/PDF 改變 |
| config 更換目標 | check 拒絕 target changed |
| ../ 越界或重複檔案 | prepare 拒絕 |
| failed 編譯報告 | prepare 拒絕未成功編譯 |
| 空目錄 baseline／plan | 兩者都拒絕，不能再產生空目錄全檔 create 計畫 |
| 匯入後缺檔／改檔／多出檔案 | plan 拒絕完整來源清單與收據不一致 |
| 保存 ZIP 被改寫 | plan 拒絕 ZIP 雜湊不一致 |
| 收據目標 ID 更換 | plan 拒絕目標不一致 |
| 匯入時錯誤目標 ID | import_backup 拒絕，不建立錯誤目標備份 |
| 重用已有備份目录 | import_backup 拒絕覆蓋，原備份保留 |
| 恢復未改寫的備份 | plan 再次通過，動作仍為 update／create／already-current |

完整原始回傳、錯誤與每個版本雜湊在 `scenario2/observed-events.json`。`scenario2/package-location.json` 指向本次主要測試交接包。`protected-files.json` 記錄七個主專案重要輸入在測試前後的雜湊，全部相同。

### 發現的限制與影響

1. **原空目錄與匯入後內容不完整的問題已修復。** 目前 baseline/plan 要求收據、原 ZIP、目標 ID 與完整來源清單一致；此次空目錄、缺檔、改檔、增加檔案、改 ZIP、錯目標均實際拒絕。不再沿用初版報告的「空目錄也能產生全檔 create」結論。
2. **ZIP 的線上出處、真正完整性與即時性仍須操作核對。** 此工具證明從提供的 ZIP 解出的所有檔案完整保留且未變更，不能證明呼叫者提供的 ZIP 原本就是線上完整下載，也不能獨立認證用戶帳號或該次觀察的專案 ID。receipt 明示 caller-provided browser observation；拿舊備份再次呼叫 plan 仍可通過。因此寫前重新讀取及核對真實專案不能省略。
3. **授權與模板避免覆蓋屬操作層，不是腳本強制門檻。** manifest 的 authorization 只是提示文字；matches_original_template 只記錄旗標。測試的 current.tex 匹配原模板仍可 prepare。工具不會讀取用戶的具體批准，也不會阻擋模板覆蓋正文。技能現有文字明確限制此事，所以執行者不可把 plan 的 update 當成授权。
4. **套件固定內容與版本保護可實際接通。** 程式的 stale／target／snapshot／PDF 檢查有可重現證據；但不會自動辨識檔案相依性。加入引用、圖片或 input 相依時，操作者仍須將必要相依檔案列入已授權交接，不能把單檔計畫當成線上一定可編譯。
5. **在線成功條件未驗證。** 此次沒有測試存取、下載、寫前重新讀取、逐檔保存回讀、線上 XeLaTeX／Biber／字體與 PDF 頁面驗證。不能因離線鏈成功就宣稱整個線上技能實際同步成功。
6. **操作說明的匯入步驟已補上。** 重跑開始時的 thesis-overleaf-sync/SKILL.md 與 HANDOFF.md 仍只有 baseline／plan 舊範例；回報主代理後再次唯讀檢查，技能第 35 行已加入 import-backup／收據要求，HANDOFF 第 30–33 行已加入 ZIP 匯入及 `/source` 的 remote-root 範例。新版文件可以接上此次實際演練的函式流程；此測試未自行修改技能。

判定：新版 ZIP 匯入與備份完整性檢查通過 28 個隔離案例，原空目錄缺口已修復，新版技能及交接文件也已接上匯入步驟。原稿辨認與固定版本交接仍成立。完整論文 PDF 與真正線上同步的操作驗證尚未完成；線上出處／即時性與授權範圍仍須技能流程維持。
