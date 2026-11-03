# 論文圖表候選成果

正式圖表的原始來源仍在 results/ 或 assets/research/。
在此只放準備進入論文的 figures/、tables/，附 manifest.json，逐項記錄 path、sha256、
run_id、data_version、code_commit、config。path 必須在 figures/ 或 tables/。

從專案根目錄執行 `python scripts/thesis/thesis.py sync-artifacts` 先預覽，
確認後加 --apply 複製到 thesis/latex/；不同內容的同名檔會被拒絕覆蓋。
這是本機複製，並非 Overleaf 即時同步；線上上傳另由 Agent 執行使用者授權的操作。
