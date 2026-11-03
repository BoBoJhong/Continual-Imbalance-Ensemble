# 專案 Playwright MCP 實際瀏覽器測試

日期：2026-10-15（台灣時間）。本次使用專案既有 launcher，沒有以內建瀏覽器代替。

## 實際連線

- 伺服器：`node scripts/thesis/start-playwright.mjs`。
- 套件：固定的 `@playwright/mcp@0.0.83`；伺服器回報 Playwright `1.64.0-alpha-1790635538000`。
- Chrome：本機 `154.0.8037.98`，使用專案 `.local/overleaf-profile`。
- 透過 MCP initialize、tools/list 與 tools/call 進行實測，工具共 25 個。
- 第一個受限執行的診斷用戶端未在等待期限內收到工具回覆；改用經核准的執行環境後，實際網站工具呼叫成功。

## 已成功的網站操作

1. `browser_navigate` 開啟指定的 Overleaf 專案。
2. `browser_snapshot` 取得可讀取的網頁結構。
3. `browser_click` 選擇必要 Cookie。
4. `browser_click` 操作頁面的 Log in 入口，抵達 Overleaf 登入頁。

以上已證明此專案的 Playwright MCP 可以啟動 Chrome、連線網站、讀取內容與點擊控制項。
這與之前僅測 initialize／tools/list 的結果不同。

## 目前登入界線

未登入的專用 Chrome 存取指定專案回傳 HTTP 403，頁面為 Restricted，並提供 Log in 入口。
這是本次專用 Chrome 的登入／專案存取尚未完成，不能將其判為 Playwright MCP 不可用。
使用者之前在內建瀏覽器登入，不會自動登入此專用 Chrome。

已開啟 `https://www.overleaf.com/login`，等待使用者直接在該 Chrome 完成登入；
不讀取、複製或保存帳密與其他瀏覽器的登入秘密。

此 MCP 尚未完成登入後的專案內容讀取、設定核對與線上編譯，也沒有線上檔案寫入。
之前的免費方案編譯逾時是由內建瀏覽器觀察到，不能當成本次 MCP 的編譯實測結果。

本次診斷原始回傳在 `scratch/thesis_playwright_live/`，瀏覽器快照在 `.local/mcp-output/`；
兩者均忽略於版本控制。後續登入完成後，應更新本報告與設定檔中的驗證狀態。
