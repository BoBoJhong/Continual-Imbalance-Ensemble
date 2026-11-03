# 實驗流程圖

Phase 1–3 已於 2026-10-12 更新為分類器與 FS 概括版。每個階段只保留一套可編輯 draw.io、PNG 與 SVG，三頁整合檔同步更新；同名檔直接覆寫，不另存舊版。

| 階段 | 可編輯原始檔 | 圖檔 |
|---|---|---|
| Phase 1：Baseline | [phase1-baseline-flow.drawio](phase1-baseline-flow.drawio) | [PNG](phase1-baseline-flow.png)、[SVG](phase1-baseline-flow.svg) |
| Phase 2：Ensemble | [phase2-ensemble-flow.drawio](phase2-ensemble-flow.drawio) | [PNG](phase2-ensemble-flow.png)、[SVG](phase2-ensemble-flow.svg) |
| Phase 3：Feature Selection | [phase3-fs-ensemble-flow.drawio](phase3-fs-ensemble-flow.drawio) | [PNG](phase3-fs-ensemble-flow.png)、[SVG](phase3-fs-ensemble-flow.svg) |

[三階段整合 draw.io](phase1-3-research-methods.drawio) 含三個可獨立編輯的頁面，各頁為原生形狀、文字與連線。

Phase 1 的合併資料節點標為 Old + New（Retrain），表示歷史與新營運資料合併後用於重新訓練。

Phase 1 的 Classifiers 概括分成 Traditional machine learning／Deep learning，Validation 路徑明示 Classifier selection + fixed threshold。Training strategies 列 Old、New／Fine-tune、Retrain，搭配分類器與採樣形成平行設定。Phase 2／3 使用 Selected classifier，表示以選定分類器分別訓練 Old×3／New×3，維持六模型池。三張圖不列具體分類器名稱。

這表達使用者確認的「先比較分類器，再以選定分類器進行集成與 FS」研究順序。最新 runner 固定使用的算法仍需第一階段驗證比較紀錄支持其最佳身份；圖面不等同已完成選擇證據。Retrain 是獨立 baseline，不加入六模型池；目前最新固定年份 run 尚無同協定 Retrain 對照，詳見圖說。

三階段均以獨立的 Preprocess 節點表示前處理。Phase 1／2 為 Preprocess → Sampling，Phase 3 為 Preprocess → Feature selection → Sampling。

三階段最終評估均標示 AUC、F1-score 與 Recall，AUC 為主指標，F1-score 與 Recall 均以破產少數類別為正類；New 表示本研究的 New／Fine-tune 分支。Phase 3 特徵選取面板只列 Filter、Wrapper、Embedded 三類，不列具體方法名；實作對應與模型解釋排序的界線留在圖說。主圖不放 Phase 2 或跨階段比較節點。Validation 為主圖的簡化表達，實際 DSEL 與 Calibration 分配及用途仍分開。

- [三階段圖說](../../../docs/三階段流程圖圖說.md)
- [圖面品質評估](../../../docs/三階段流程圖品質評估.md)
- [結構與字級檢查](phase1-3-quality-checks.json)

PNG 尺寸為 2720 × 1640；論文排版可使用 SVG。以 180 mm 寬排版，最小字級約 9 pt；窄欄排版需重新調整字級。

從專案根目錄重新產生三階段檔案，會直接更新本目錄的同名檔案：

```powershell
python scripts/plots/build_research_method_diagrams.py
```

Phase 4 是另一路延伸研究方法，本次保留其現行 `phase4-batch-adaptive-flow.puml`、PNG 與 SVG。
