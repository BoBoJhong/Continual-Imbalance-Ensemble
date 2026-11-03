# 本地修改與 Overleaf 同步的版本交接

本地技能生成可審閱的修改與編譯成果；線上技能處理使用者要求的指定變更。
`handoff.py` 僅在本地讀寫，不會登入、連網、上傳或改動 Overleaf。

## 建立與檢查

```powershell
python scripts/thesis/handoff.py prepare --document main --file chapters/03_method.tex
python scripts/thesis/handoff.py check --package thesis/handoffs/指定時間戳
```

交接包含 manifest.json、files/ 的指定檔案快照、編譯報告與 compiled-preview.pdf。
清單記錄本地來源、線上目標 ID／網址、編譯字體模式、PDF SHA256、來源 SHA256，
以及哪些指定檔案仍與原模板相同。建立包之前必須先成功編譯對應 document。
Markdown 草稿不能直接當作 Overleaf 的 LaTeX 檔案；尚未遷入時由本地技能回報待辦。

檢查會拒絕過期編譯、目標已改變、交接快照／PDF 被改寫、工作稿或數據來源改變。
因此同步前必須使用這次確認過的交接包，而不是隨意上傳目前整個資料夾。
`.cls`、字體、簽署文件只有在使用者指定且授權的同步範圍內才處理。

## 線上基準與衝突（只比對本地備份）

先透過瀏覽器取得指定專案的完整 source ZIP，安全解壓縮至新的私有備份目錄；
保留 ZIP、本次取回網址與時間。先使用 import-backup，將完整 ZIP 安全匯入新目錄，
保存 ZIP 與完整清單的 SHA256。remote-root 必須是產生的 source 根目錄，不能用空目錄或
只包含可見幾行內容的截圖來推論線上檔案不存在。

```powershell
python scripts/thesis/handoff.py import-backup --zip "下載的完整來源.zip" --output .local/overleaf-backups/首次完整來源 --project-id 6acef24f1aecda6740296760
python scripts/thesis/handoff.py baseline --package thesis/handoffs/指定時間戳 --remote-root .local/overleaf-backups/首次完整來源/source
# 再次下載時用新的 output 目錄匯入，保留舊基準。
python scripts/thesis/handoff.py plan --package thesis/handoffs/指定時間戳 --remote-root .local/overleaf-backups/新取回完整來源/source
```

baseline 只保存完整取得的目標檔案雜湊，不代表批准覆蓋這些檔案。
plan 只列出 update／create／already-current／conflict，遇衝突回傳失敗狀態。
未知的首次線上差異要先審閱；實際寫入之前還須用瀏覽器重新讀取當下的內容。
更舊的 source ZIP、其他專案或另一個帳號的測試不能當成目前目標的成功證據。
匯入收據拒絕空目錄、解壓縮後缺檔／改檔或 ZIP 變動；但專案 ID 是呼叫者依瀏覽器提供，
腳本不能自行證明登入身分、下載來源或目前網站狀態，仍須核對實際頁面。

目前資料夾 `thesis/handoffs/` 放本地交接紀錄；`.local/` 放線上備份與私人瀏覽器狀態。
圖表／字體／簽署文件的二進位快照不自動加入 Git。記錄目標網址不會觸發同步。
