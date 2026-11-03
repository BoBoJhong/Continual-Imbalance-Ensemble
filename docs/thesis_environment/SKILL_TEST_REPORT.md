# 論文技能測試報告

日期：2026-10-15（台灣時間）。技能及工具建立於 Continual-Imbalance-Ensemble 專案。

追加的專案 Playwright MCP 實際瀏覽器測試見 [PLAYWRIGHT_MCP_TEST.md](PLAYWRIGHT_MCP_TEST.md)：
已成功開啟網站、讀取結構與操作登入入口；登入後的專案與編譯檢查仍待完成。

## 已完成

- 新增 `.agents/skills/thesis-local-edit/` 與 `.agents/skills/thesis-overleaf-sync/`，含 SKILL.md、介面名稱與預設指令；兩者結構驗證通過。
- 既有 research-paper-agent 保留，新增本地／線上流程入口；研究及引用規範共用。
- 版本交接固定指定檔案、目標、編譯報告、PDF、原始結果與工作稿 SHA256；來源或快照變動時拒絕使用過期版本。
- 完整來源 ZIP 安全匯入新目錄，保留 ZIP、收據與完整檔案清單；空目錄、缺檔、改檔與 ZIP 變動不能作為有效線上備份。
- `tests/test_thesis_sync.py` 與 `tests/test_thesis_handoff.py` 共 **17 項測試通過**。
- 獨立隔離前向演練完成 **28 個案例**，包括真實 XeLaTeX 最小稿編譯，以及 update／create／already-current／conflict、版本過期與備份完整性拒絕。
- 本地正式字體重新編譯：模板主稿 **17 頁**，4 個未插入正式簽署文件的提示；數據核對預覽 **5 頁**，0 警告。原模板 21 檔 SHA256 檢查通過，選定 run 與生成表格新鮮度檢查通過。
- 專案 MCP 啟動及協定檢查通過，發現 25 個工具。實際線上檢查使用 Codex 內建瀏覽器，兩種驗證分開記錄。

## 真實 Overleaf 檢查

目標：https://www.overleaf.com/project/6acef24f1aecda6740296760

使用者自行登入後，已確認專案名稱 NCU_IM_Thesis、可存取、main.tex、XeLaTeX、TeX Live 2026。
已下載完整 source ZIP（21 檔），匯入 `.local/overleaf-backups/20261014T162215Z/source`，
逐檔與本地保留模板比較，全部內容雜湊相符。

本地測試交接包：`thesis/handoffs/20261014T162413846931Z`。
指定第三章模板快照通過 prepare／check；以本次備份建立基準並比對，結果 already-current。
這項結果只證明本次下載與本地指定快照相同，不代表後續網站沒有變動。

實際按下 Recompile 後，網站再次顯示 **Your compile timed out**，說明超過免費方案編譯時間限制。
原始 log 顯示 XeTeX／TeX Live 2026 已載入 Times New Roman 與 fonts/kaiu.ttf 標楷體。
尚無成功的本次線上 PDF，亦未完成 Biber、頁面與保存後內容的線上成功驗證。
截圖：`evidence/overleaf-compile-timeout.jpg`。

## 驗證界線

- 沒有線上論文檔案寫入、全專案覆蓋或新建測試檔案；線上寫入與保存回讀尚未測試。
- THESIS_FULL.md 是既有 Markdown 草稿，LaTeX 章節目前仍是模板；17 頁 PDF 不代表已完成正式論文遷移或修訂。
- 收據的專案 ID 由操作者依實際頁面提供；工具不能自行驗證登入身分、下載出處或網站即時狀態，寫入前仍須重新核對。
- 交接計畫不等於授權。使用者要求同步已審閱的指定變更時，再執行備份、衝突核對、寫入、回讀與線上編譯。

## 使用方式

1. `$thesis-local-edit 請依已完成的實驗修訂第三章，核對來源並交付本地 PDF 與交接包。`
2. 審閱後：`$thesis-overleaf-sync 請同步這次確認的交接包到已指定的 Overleaf。`

只檢查線上差異時可直接要求 `$thesis-overleaf-sync 先比對差異，不寫入。`。
詳細工具指令見 HANDOFF.md；專案環境與首次安裝見 README.md。
