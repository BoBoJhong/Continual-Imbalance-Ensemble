# 固定的補充 TeX 套件

本資料夾保存此次實際編譯所需、但 TinyTeX 2026.10 snapshot 未提供的套件封存檔。
manifest.json 記錄來源、TeX Live revision、SHA256 與原套件中繼資料。
封存檔含原套件的作者與授權告示；chngcntr、tocloft、xeCJK、zhnumber、ctex
依各自 LPPL 授權提供，詳見封存檔及 https://www.latex-project.org/lppl/。

install-runtime-packages.py 僅補入這份論文實際使用的檔案，然後重建檔案資料庫。
ctex 在本論文中只供 xeCJK 的 ctexhook/ctexpatch 使用；沒有安裝 ctex 全系列其他
引擎與字體的相依套件。若將專案改成 ctex class、LuaLaTeX 或其他內容，請改用
完整 TeX Live 並建立新的鎖定環境；不要把本 snapshot 當成通用完整 TeX Live。

執行 bootstrap 不會連網更新這些套件，也不會將它們登記到 tlmgr 相依管理資料庫。
請勿對此固定可攜環境執行 tlmgr update；需要升版時另建環境並完整測試。
