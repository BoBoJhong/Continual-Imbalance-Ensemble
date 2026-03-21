# 論文／簡報用圖表建議（UML）

研究問題條目見 [docs/研究方向.md](../docs/研究方向.md)；**條目 ↔ 實驗階段 ↔ 本目錄圖檔** 總表見專案根目錄 [README.md](../README.md)「研究方向與實驗階段及 UML 對照」與 [docs/研究方向對照表.md](../docs/研究方向對照表.md)。

## 與程式目錄對照（Phase 2）

| 圖／主題 | 建議對應程式與結果 |
|----------|-------------------|
| `baseline_flow.puml` | `experiments/phase1_baseline/`、`results/phase1_baseline/` |
| `ensemble_flow.puml`（靜態集成） | `experiments/phase2_ensemble/static/`、`results/phase2_ensemble/static/` |
| 動態 **DES**（KNORA、進階、XGB 年份切割） | `experiments/phase2_ensemble/dynamic/des/`、`results/phase2_ensemble/dynamic/des/`（XGB 主線檔名 `xgb_oldnew_*`） |
| 動態 **DCS** | `experiments/phase2_ensemble/dynamic/dcs/`、`results/phase2_ensemble/dynamic/dcs/` |
| Study II 特徵選取 | `experiments/phase3_feature/`（Phase 3 FS） |

## 已建立的檔案

| 檔案 | 用途 |
|------|------|
| `study1_baseline_and_ensemble_flow.puml` | **Study 1 總覽**：**由左而右**—Dataset / Split / Old / New → **Step 1 Baseline** → **Step 2 Ensemble**（`P1 --> P2` 表實作順序）→ **Test** → **Metrics**；內部各步仍為橫向管線；無錯誤資料流；五種子集規模並行進 Fusion。 |
| `study2_feature_selection_flow.puml` | **Study 2 一張圖**：特徵選取 → 子集 → 下游重跑 Study 1 管線，與 Test / 對照全特徵之指標；與 Study 1 **分開**。 |
| `baseline_flow.puml` | **Phase 1 Baseline**（單獨細節圖）：四種採樣（含 none）、**四種訓練設定**（Old-only、New-only、**Re-training** 合併 Old+New、**Fine-tuning** Old→New）、Classifier(XGB)、Val 閾值、Test。 |
| `ensemble_flow.puml` | **Phase 2 靜態集成**（單獨細節圖）：舊新池、組合設計、Fusion、動態分支佔位。 |
| `ensemble_combinations_full_list.puml` | **附錄圖**：英文、單色、五欄面板列舉 **49** 個合法子集（依 \|S\|=2…6）；風格接近論文 appendix / table figure，非 UML 彩色流程。 |
| `research_pipeline_flow_like_reference.puml` | **總覽**（含 Baseline 虛線、DCS/DES／Study II 併列）：適合「一章開頭一張總圖」。 |
| `current_experiment_architecture.puml` | 較早的架構總覽，可與上列擇一使用。 |
| `phase3_dcs_des_planned.puml` | **Phase 3 佔位**：尚未實作時僅表示「之後會接在模型池後面」。 |
| `study2_feature_selection_template.puml` | **Study II 模板**：等特徵選取實驗設計定稿後再替換方塊內文字與箭頭。 |

## 是否要「DCS/DES 一張、特徵選取一張」？

**建議（較好呈現、又不會圖太多）：**

1. **正文方法章**  
   - **Study 1 總覽**：`study1_baseline_and_ensemble_flow.puml`（Baseline + Ensemble 一張）  
   - **Study 2**：`study2_feature_selection_flow.puml`（與 Study 1 分開）  
   - 若需細節拆解：再加 **圖 A** `baseline_flow.puml`、**圖 B** `ensemble_flow.puml`。

2. **總覽（可選）**  
   - 用 `research_pipeline_flow_like_reference.puml` 當 **一頁式 roadmap**，指向「詳見圖 A、圖 B」。

3. **DCS / DES**（實作於 `experiments/phase2_ensemble/dynamic/`）  
   - 若圖檔仍為佔位：可用 **`phase3_dcs_des_planned.puml`**（檔名沿用歷史）當極簡示意。  
   - 正式論文可再獨立一張「動態選擇」細節圖（competence、鄰域、權重），與 `ensemble_flow.puml`（靜態）分開。

4. **特徵選取（Study II）**  
   - **實驗與方法未定前**：不必放正式圖，或只用 **`study2_feature_selection_template.puml`** 在簡報草稿。  
   - **定案後**：**獨立一張**「特徵選取 → 子集 → 下游重跑 Baseline/Ensemble」，避免把 Study II 擠進 Baseline 或 Ensemble 主圖。

## PlantUML 預覽失敗（Windows / `dot.exe` / `cgraph.dll`）

若錯誤類似 `FileNotFoundException: ...\Temp\_graphviz\dot.exe` 或「檔案正由另一個程式使用」，**不是 `.puml` 寫錯**，而是 PlantUML 內嵌 Graphviz 解壓到暫存目錄失敗。

**做法（擇一）：**

1. **已在本專案主要圖檔加上** `!pragma layout smetana`（純 Java 排版，通常**不需**安裝 Graphviz）。若某張圖仍失敗，在該檔 `@startuml` 下一行同樣加上即可。  
2. **安裝 [Graphviz](https://graphviz.org/download/)**，並在 VS Code / Cursor 的 PlantUML 設定中指定 `dot` 路徑（例如 `C:\Program Files\Graphviz\bin\dot.exe`）。  
3. 關閉會鎖定暫存檔的程式、暫時排除防毒對 `%TEMP%\_graphviz` 的掃描，或清理後重開編輯器再試。

## 小結

- **Baseline 一張 + Ensemble 一張**：**是**，這樣最好讀。  
- **DCS/DES**：**現階段**用一張極簡佔位或不用圖；**做完後**再一張專用圖。  
- **特徵選取**：**等實驗跑完、方法寫死**後再一張專用圖；現在用模板即可。
