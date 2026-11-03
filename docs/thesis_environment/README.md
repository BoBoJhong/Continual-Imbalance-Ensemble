# 本專案的論文環境與數據更新

## 2026-10-27 依使用者提供的 114.09.25 格式 PDF 修訂

目前版面以使用者提供的 32 頁撰寫體例為核對來源；規範頁次與差異見 `thesis/NCU_IM_FORMAT.md`。本輪新增 `ncu-reference-style.tex` 並更新 class、文件順序開關及符號表；交接時須包含該樣式檔。附錄採中文「附錄一」，正文段次採 1.，目錄置中。數據匯出與原有雙文獻後端編譯方式維持既定流程。早期模板說明中的 APA、字型回退與附錄 A 等描述，以最新來源與格式對照為準。

## 2026-10-27 內建編輯器預覽修正

後續匯出同步修正：`main.tex` 預設 BibTeX 供內建預覽；`environment-main.tex` 宣告原生環境標記後，以固定 Biber 匯出同一論文。兩個路徑均保持 numeric 引用格式並已驗證。內建預覽成功不代表 `build/main.pdf` 已更新，交付匯出 PDF 時必須重新編譯並直接核對 PDF 內文與書籤。

目前主文件的 BibLaTeX 後端為 `bibtex`，數字型引用與既有文獻資料維持原設定。
這項修正使內建編輯器不再依賴會觸發本機 MiKTeX 登錄權限錯誤的外部 Biber。
使用原第三章路徑執行內建編譯已回報成功，無需關閉目前文件、重開分頁或另行匯出 PDF。
既有 TinyTeX 原生工具與字體保留；下文早期環境準備中對 Biber 的安裝說明為歷史準備紀錄，
不代表目前主文件仍使用該文獻後端。

論文環境現在位於 **Continual-Imbalance-Ensemble 專案內**。
實驗繼續使用既有 experiments/、src/、config/、data/、results/；
既有 `thesis/THESIS_FULL.md` 是 Markdown 草稿，新增 `thesis/latex/` 是可編譯的 NCU IM 模板。
本次沒有將示例模板覆蓋既有草稿，也沒有宣稱已將整份 Markdown 論文轉為 LaTeX。

```text
Continual-Imbalance-Ensemble/
├─ experiments/、src/       實驗程式與共用模組（沿用）
├─ results/                 原始 run、metrics、summary、來源與稽核（沿用）
├─ config/
│  ├─ thesis_sources.json   論文採用的明確 run、seed、欄位與篩選條件
│  └─ thesis_environment.lock.json
├─ thesis/
│  ├─ THESIS_FULL.md        既有論文草稿（保留）
│  ├─ latex/               NCU IM LaTeX 專案
│  │  ├─ main.tex、chapters/、references.bib
│  │  ├─ generated/        結果表格、來源快照、current 指標
│  │  └─ build/            PDF、log、編譯報告（不進 Git）
│  └─ overleaf_exports/    核對後的圖表候選成果
├─ scripts/thesis/         匯出、更新檢查、字體、編譯與 MCP 啟動
├─ tools/thesis-agent/     固定版本 Playwright MCP 相依套件
├─ tools/thesis-tex/       固定 TeX 補充套件
└─ docs/thesis_environment/ 操作說明與整合紀錄
```

## Windows 第一次準備

由本專案根目錄執行（Python 3.11+、Node.js 20+、Chrome）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/thesis/bootstrap-windows.ps1
powershell -ExecutionPolicy Bypass -File scripts/thesis/prepare-fonts.ps1
npm --prefix tools/thesis-agent ci --ignore-scripts --no-audit --no-fund
python scripts/thesis/thesis.py doctor
```

bootstrap 把固定 TinyTeX 2026.10 安裝至本專案 `.tools/thesis/`，核對 SHA256 並補入固定套件。
不改動實驗的 Python 依賴或全域 PATH。首次需下載約 247 MB；可先將相同官方 ZIP
放在 `.tools/thesis/TinyTeX-v2026.10.zip`。prepare-fonts 從已安裝的 Windows 字體
複製 Times New Roman 四個字重檔，與原附 kaiu.ttf 組成 strict 字體模式。
字體、簽署文件、工具安裝與瀏覽器 profile 不進 Git；仍須依各自授權處理散布。

若使用完整 TeX Live，無須 bootstrap；讓 xelatex、latexmk、biber 在 PATH 上即可。
亦可用環境變數 NCU_TEX_BIN 指定既有工具目錄。Docker 固定映像設定也保留於 lock，
先 pull 該 digest，再執行 `thesis.py build --backend docker`；本輪不啟動 Docker 或操作線上 Overleaf。

## 實驗數據如何更新到論文

```powershell
# 1. 在 config/thesis_sources.json 明確選定新的完成 run、seed 集合與欄位。
# 2. 產生新的論文來源快照，不重訓、不修改原結果。
python scripts/thesis/sync_results.py
# 3. 檢查來源或生成檔是否改變。
python scripts/thesis/sync_results.py --check
# 4. 編譯數據核對預覽；來源變動時會停止，要求先重新匯出。
powershell -ExecutionPolicy Bypass -File scripts/thesis/build.ps1 -Document results
```

每次匯出新建 `thesis/latex/generated/<UTC timestamp>/`，保存：

- 原 summary CSV 的完整精度快照與本表篩選後 CSV。
- LaTeX 表格、run、protocol、seed 集合、篩選條件、來源 SHA256。
- `current.json`／`current.tex` 指向此次選定快照；舊快照保留。

目前綁定專案已有的固定年份與完整 Study 3B run，表格分開呈現；
固定年份選 `no_fs`＋`pooled`，Study 3B 選 pooled 的 `*_seed_mean/std` 欄位。
不使用三 seed 歷史分表，不與 annual_mean 欄位混用，不依 Test 排名挑選候選。
這個工具核對來源、完成狀態、seed 與格式，不代替完整方法／統計稽核，也不補齊尚缺的 Retrain。

後續將核對過的 LaTeX 內容寫入章節時，可使用：

```latex
\input{generated/current.tex}
% 或固定引用某一快照：
\input{generated/你的快照時間戳/fixed_no_fs_pooled.tex}
```

current 是一次明確匯出的選定版本，並非自動尋找最新檔案。
新實驗完成後，需更新來源設定並重新執行匯出；實驗 CSV 不手改數字。
要更新正文中的數字與解讀，仍須逐段核對來源，不能只更新表格就視為全文同步。

## 論文與 Overleaf

### 分開使用兩個技能

技能保存在本專案 `.agents/skills/`，既有 research-paper-agent 繼續負責研究與引用規範。
本地修改及線上同步分成兩個步驟：

1. `$thesis-local-edit 請依已完成的實驗結果修訂第三章，完成本地核對與編譯。`
2. 審閱修改與 PDF 後：`$thesis-overleaf-sync 請將這次確認的交接包同步到已指定的 Overleaf。`

也可以只要求第二個技能「比對差異」，這時只檢查及備份。
使用者明確授權同一交接內容同步後，技能不重複要求批准。
每個交接包固定檔案、目標與已編譯版本；工作稿或數據再變動時要重新核對與編譯。
操作詳見 [版本交接說明](HANDOFF.md)，驗證界線見 [技能測試報告](SKILL_TEST_REPORT.md)。

### 已指定的線上同步目標

使用者已指定 [Overleaf 專案](https://www.overleaf.com/project/6acef24f1aecda6740296760)，
設定保存在 [thesis_overleaf.json](../../config/thesis_overleaf.json)，本地來源為 thesis/latex/。
2026-10-15（台灣時間）已登入核對專案 NCU_IM_Thesis、存取權限、main.tex、XeLaTeX 與 TeX Live 2026。
已下載完整來源備份；實際重新編譯仍超過免費方案時間限制，未進行線上內容寫入。
正式／測試用途尚未由使用者指定。
設定網址不會啟動線上同步；之後使用者要求同步時，先讀此設定，再核對線上專案與檔案差異。
登入由使用者完成，設定檔不保存密碼或 Cookie。

```powershell
# 原模板編譯（仍為示例；既有 THESIS_FULL.md 尚未轉檔）
powershell -ExecutionPolicy Bypass -File scripts/thesis/build.ps1
# 僅匯出論文及目前選定的 LaTeX 表格，不攜帶實驗資料與原結果 CSV
python scripts/thesis/thesis.py export-overleaf
```

PDF：`thesis/latex/build/main.pdf`；數據核對 PDF：`results-preview.pdf`。
Overleaf ZIP：`dist/NCU_IM_Thesis_Overleaf.zip`；主文件 main.tex，編譯器 XeLaTeX，文獻 Biber。
匯出略過歷史生成快照、CSV、實驗程式、工具安裝、瀏覽器 profile 與新增 Times 字體檔。
Overleaf 使用原 class 選字體，需查 log 確認實際字體。

VS Code `MCP: List Servers` 啟動 `.vscode/mcp.json` 的 playwright，
由使用者登入並選定 Overleaf 測試副本，再讀取、提出修訂及執行已授權變更。
既有 `.vscode/settings.json` 的 Python 設定保持不變；新增任務可用 `Tasks: Run Task` 執行。
LaTeX Workshop 是選用的編輯／PDF 檢視工具，編譯以本包任務或腳本為準。
Codex 範例位於 config/thesis_codex_mcp.example.toml，尚未寫入使用者全域設定。

## 既有草稿與下一階段

Markdown 草稿與 LaTeX 目前是兩份不同用途的檔案，沒有自動雙向同步。
下一階段應逐章將已核對的 THESIS_FULL.md 內容遷入 chapters/，
同步參考文獻與圖表引用，並將每個數字綁定至結果快照；不要直接以示例章節當成正式正文。
