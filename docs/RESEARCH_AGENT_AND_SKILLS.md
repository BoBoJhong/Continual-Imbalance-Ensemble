# 研究與論文 Agent：使用方式與外部 Skill 評估

> 查核日期：2026-09-07

## 專案內建 Agent

本專案新增兩個互補元件：

- `/.codex/agents/researcher.toml`：可委派獨立研究工作的 Codex custom agent。
- `/.agents/skills/research-paper-agent/`：主 agent 與 researcher 共用的研究工作規範。

它涵蓋文獻搜尋、來源分級、引註驗證、研究設計、資料科學方法檢查、結果詮釋、論文章節撰寫、同儕審查與修訂追蹤。此版本特別綁定本專案的時間切割、類別不平衡指標、結果檔與 BibTeX 管理規則。

在 Codex 中可明確呼叫：

```text
$research-paper-agent 幫我針對概念漂移與不平衡串流分類做系統性文獻回顧，並更新 docs/RELATED_LITERATURE.md 與 docs/references.bib。
```

也可用自然語言要求寫方法、結果、討論、citation audit 或 reviewer response；任務符合 description 時，Codex 可自動選用。若新 skill 未出現在選單，重新啟動 Codex。

需要獨立 agent thread 時，可要求：

```text
請使用 researcher agent 搜尋近五年的相關論文，先回傳證據表，不要改 thesis。
```

Custom agent 適合把範圍明確、偏讀取的研究工作與主對話分開；skill 則適合一般研究任務，成本較低。Subagent 會使用額外 token，因此不必在每個小修改中啟用。

## 目前環境已具備的資料分析 Skills

目前已有下列互補能力，不必重複安裝：

- `data-analytics:analyze-data-quality`：資料品質與證據可信度。
- `data-analytics:jupyter-notebooks`：可重現的 Python／SQL Notebook。
- `data-analytics:build-report`：具來源脈絡的分析報告。
- `data-analytics:visualize-data`：量化圖表設計與 QA。
- `data-analytics:validate-data`：方法、計算、比較與結論驗證。
- `data-analytics:product-business-analysis`：需要數據證據的決策分析；研究用途須再套用本專案的學術規範。
- `deep-research-work:deep-research`：完整深度研究流程；僅在使用者明確要求「Deep research」時啟用。

`research-paper-agent` 負責學術脈絡與證據完整性；上述 skills 負責特定的資料分析產物。

## 網路搜尋結果與建議

### 1. Academic Research Suite for Codex

- 套件：`imbad0202/academic-research-skills-codex@academic-research-suite`
- skills.sh 安裝數：約 1.4K（查核當日）
- GitHub：約 9.5K stars（查核當日）
- 優點：Codex-native；涵蓋研究問題、文獻回顧、系統性回顧、寫作、審稿與 revision pipeline；支援繁體中文。
- 注意：規模很大，會與本專案 skill 的部分功能重疊；採 CC BY-NC 4.0，並非 OSI 定義的開源授權。安裝前應檢查依賴、權限與資料外傳設定。
- 安裝：`npx skills add imbad0202/academic-research-skills-codex@academic-research-suite`
- 連結：https://skills.sh/imbad0202/academic-research-skills-codex/academic-research-suite
- 原始碼：https://github.com/Imbad0202/academic-research-skills-codex

評估：若需要完整的跨專案研究流水線，這是最相符的第三方候選；本專案目前先使用較小且可審核的 repo-scoped skill 即可。

### 2. K-Dense Scientific Writing

- 套件：`k-dense-ai/scientific-agent-skills@scientific-writing`
- skills.sh 安裝數：約 1.7K（查核當日）
- GitHub repository：約 43K stars（skills.sh 查核當日顯示）
- 優點：強調證據 provenance、作者責任、保密、報告規範、claim-level verification，核心指引為平台中立。
- 注意：整個 repository 含大量生醫與外部資料庫 skills；只應選裝所需 skill。skills.sh 顯示部分安全掃描為 Warn，安裝前需逐項閱讀掃描報告與腳本。
- 安裝：`npx skills add k-dense-ai/scientific-agent-skills@scientific-writing`
- 連結：https://skills.sh/k-dense-ai/scientific-agent-skills/scientific-writing
- 原始碼：https://github.com/K-Dense-AI/scientific-agent-skills

評估：適合補強投稿前的科學寫作與證據稽核，但不如本專案 skill 熟悉 DAWCE／ROSS、時間洩漏與 results 目錄。

### 3. Academic Research Skills（Claude 版）

- 套件：`imbad0202/academic-research-skills@academic-paper`
- skills.sh 安裝數：約 8.9K（查核當日）
- 優點：功能完整，且另有 reviewer、deep-research、pipeline 等 skills。
- 注意：主要為 Claude Code 封裝；Codex 應優先選上面的 Codex-native sibling。授權同為 CC BY-NC 4.0。
- 安裝：`npx skills add imbad0202/academic-research-skills@academic-paper`
- 連結：https://skills.sh/imbad0202/academic-research-skills/academic-paper
- 原始碼：https://github.com/Imbad0202/academic-research-skills

評估：不建議在本專案直接安裝，除非需要比對原始 Claude 工作流。

### 4. Google Cloud Agentic Data Science Workflow

- 套件：`google/skills@google-cloud-solution-agentic-ai-data-science-workflow`
- skills.sh 安裝數：約 2.6K（查核當日）
- 優點：Google 官方來源，適合規劃 Google Cloud 上的多產品 agentic data-science architecture。
- 注意：它不是通用資料分析或學術寫作 skill，重點是 Google Cloud 架構、部署與 IaC。
- 安裝：`npx skills add google/skills@google-cloud-solution-agentic-ai-data-science-workflow`
- 連結：https://skills.sh/google/skills/google-cloud-solution-agentic-ai-data-science-workflow
- 原始碼：https://github.com/google/skills

評估：目前專案不需要；只有在準備把資料科學 agent 部署到 Google Cloud 時才值得安裝。

## 建議採用方式

目前先採用專案內建的 `$research-paper-agent`，搭配已安裝的 data-analytics skills。不要一次安裝整套大型第三方研究套件，以免出現重複觸發、上下文過長、外部工具依賴或授權不符。

若後續要進行正式的 PRISMA 系統性回顧、多人式審稿流程或跨專案研究，可再選擇安裝 `academic-research-suite`。安裝屬於額外外部變更，應先取得使用者確認。

## 查核來源

- OpenAI Build skills：https://developers.openai.com/codex/skills
- Agent Skills Directory：https://skills.sh/
- ARS-Codex repository：https://github.com/Imbad0202/academic-research-skills-codex
- K-Dense Scientific Agent Skills：https://github.com/K-Dense-AI/scientific-agent-skills
- Google Agent Skills：https://github.com/google/skills
