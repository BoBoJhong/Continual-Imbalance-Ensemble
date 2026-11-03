# 由結果來源產生的論文資產

本目錄由 scripts/thesis/sync_results.py 管理。current.json／current.tex 指向本次選定快照，
每次更新建立新的 UTC 時間戳資料夾，舊快照保留。
source.csv 保存原摘要精度，selected.csv 保存篩選後列，tex 為顯示用四位小數表格，
manifest.json 記錄來源、欄位、條件、seed 與 SHA256。

不要手改生成表格或數據。修改來源選擇時，更新 config/thesis_sources.json 後重建；
--check 會偵測來源變動或生成檔被改寫。Overleaf 匯出僅包含目前快照的 LaTeX 與 metadata，
略過 CSV 與歷史快照。引用固定快照可讓既有章節的數字保有明確版本身份。
