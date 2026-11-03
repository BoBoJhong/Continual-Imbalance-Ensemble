# 論文工具

從專案根目錄執行；完整說明在 [docs/thesis_environment](../../docs/thesis_environment/README.md)。

| 工具 | 用途 |
|---|---|
| sync_results.py | 讀取 config/thesis_sources.json，新建有來源雜湊的表格快照；--check 驗證新鮮度 |
| thesis.py doctor | 查核編譯器、套件與正式字體 |
| build.ps1／build.sh | XeLaTeX + Biber；-Document results 另編譯真實數據核對預覽 |
| thesis.py export-overleaf | 論文與目前 LaTeX 快照 ZIP；略過原始結果 CSV |
| handoff.py prepare／check | 固定指定檔案與編譯證據，檢查版本及數據來源 |
| handoff.py import-backup／baseline／plan | 保存完整 ZIP 收據，在本地比較線上備份並偵測衝突 |
| prepare-fonts.ps1 | 從合法已安裝的 Windows 字體準備字體檔，不覆蓋原附標楷體 |
| bootstrap-windows.ps1 | 固定可攜式 TinyTeX；只寫入本專案 .tools/thesis/ |
| start-playwright.mjs | 固定版本 MCP 與本專案獨立 Chrome profile |

這些工具不重訓、不手改原結果，也不自動改寫 THESIS_FULL.md 或線上 Overleaf。
# 方法核對補充

`resolve_method_clarifications.py` 讀取已固定的原作者公開版本副本、既有 Baseline 驗證欄位及本機設備查核紀錄，產生附錄驗證彙整與設備表。只作描述性驗證彙整，不按 Test 選基礎分類器；`--check` 檢查來源與生成表雜湊。`method_reference_inventory.py` 另負責原始資料計數、主要函式 AST 及 FS 入選明細。兩類核對清單均由論文來源設定固定，並納入 `sync_results.py --check`。

