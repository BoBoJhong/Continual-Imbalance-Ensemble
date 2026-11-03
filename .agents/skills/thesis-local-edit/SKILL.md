---
name: thesis-local-edit
description: 修改本專案的本地論文、核對實驗數字與文獻、更新相關章節、編譯 LaTeX 並準備指定檔案的同步交接包。用於修改論文、遷入 LaTeX 或更新論文數據；不操作線上 Overleaf。
---

# 本地論文修改

以繁體中文執行 Continual-Imbalance-Ensemble 的本地論文工作。
先讀專案 `thesis/AGENTS.md`、`thesis/README.md` 與
`docs/thesis_environment/README.md`；研究判斷沿用
[research-paper-agent](../research-paper-agent/SKILL.md)。下列路徑均相對專案根目錄。

## 選定原稿與範圍

- 讀取使用者指定的章節及實際檔案，列出本次修改範圍。檢查／討論要求只提供分析；
  明確要求修改即可完成已授權的本地修改，不重複要求批准同一範圍。
- 現在 `thesis/THESIS_FULL.md` 含既有 Markdown 草稿，`thesis/latex/` 是新增模板。
  先確認該章是否已遷入 LaTeX；模板的 PDF 不能當成 Markdown 修訂後的編譯成果。
  只遷入使用者要求的章節，並記錄目前主要維護檔案；不自行轉換整份論文。
- 保留已有未提交變更。修改題目、class、字體或研究協定時，確認它們在本次範圍內。

## 修改與證據

- 文獻、研究設計與結論依共用研究技能核對。每個新增／改動數字記錄 run、target、
  seed、指標口徑及 CSV 欄位／篩選條件；不手改 results/ 或 generated/ 的數值。
- 數據變動時檢查 `config/thesis_sources.json`，使用 `scripts/thesis/sync_results.py`
  產生新快照，再執行 `--check`。新實驗只有計畫或未完成時，記錄缺口，不填入成果。
- 同步受影響的文字、表格、圖說、摘要、結論及研究規格中的主張，限於本次授權。
  超出範圍的連帶修訂在交付中明列。表格更新不代表全文的數字與解讀已同步。

## 驗證與交接

從專案根目錄執行適合本次內容的檢查：

```powershell
python scripts/thesis/sync_results.py --check
python scripts/thesis/thesis.py build --document main
# 僅核對生成數據時：
python scripts/thesis/thesis.py build --document results
```

讀取最終 log，檢查引文、交叉引用、缺字與溢出；對修改頁渲染並檢視 PDF。
純 Markdown 修改可交付草稿與來源核對，明確標示尚無對應的 LaTeX 編譯結果。

可編譯的 LaTeX 修改完成後，只為本次指定檔案建立交接包：

```powershell
python scripts/thesis/handoff.py prepare --document main --file chapters/03_method.tex
python scripts/thesis/handoff.py check --package thesis/handoffs/指定時間戳
```

`--file` 可重複指定。依 [交接規格](../../../docs/thesis_environment/HANDOFF.md)
確認檔案、來源、PDF 與目標；交接包含固定內容快照與雜湊，建立交接包不會上傳。
原模板內容會標示於清單，不能因此宣稱它已成為完整論文。

交付修改差異、證據／待辦、實際編譯結果與交接包位置。
本技能到本地交付為止，線上同步由使用者另行要求。
