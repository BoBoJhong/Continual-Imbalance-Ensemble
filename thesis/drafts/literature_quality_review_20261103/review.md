# 現行 LaTeX 選文品質初步評估

查核日期：2026-11-03（臺北）。範圍：`thesis/latex/references.bib` 與第二章現行引用；不包含尚未遷入 LaTeX 的健康資料候選文獻。本次為代表性來源查核，非系統性文獻回顧，也非逐篇方法、勘誤或撤稿查核。未修改正文、引用庫或結果。

## 檢索與篩選紀錄

- 先讀現行引用庫、第二章與研究規格，按概念漂移、批次增量集成、動態選擇、表格分類及特徵選取分類。
- 以引用庫的正式 URL 直接開啟出版者、會議論文集及作者機構原文。
- 搜尋 `site.iclr.cc virtual 2025 TabM Advancing Tabular Deep Learning Parameter Efficient Ensembling`，以 ICLR 正式論文集交叉核對；OpenReview 當次遇驗證頁。
- 搜尋 `site.dl.acm.org "A Survey on Concept Drift Adaptation" 2523813`，核對 ACM 出版紀錄。
- 排除非作者或非出版者的轉載與摘要聚合頁作為結論證據。MDPI 原頁本次回應 429，未由此推論文章品質。

## 證據矩陣

| 來源 | 本次核對位置／層級 | 可支持的角色 | 使用界線 |
|---|---|---|---|
| Gama 等，2014，ACM Computing Surveys，DOI 10.1145/2523813 | ACM 出版頁、摘要；verified-metadata | 漂移適應與評估的背景綜述 | 不能證明本資料已出現特定漂移或本方法有效 |
| Elwell、Polikar，2011，IEEE Transactions on Neural Networks，DOI 10.1109/TNN.2011.2160459 | 作者機構全文，第 1517 頁摘要；verified-full（僅限所列主張） | 連續批次各訓練分類器、動態加權整合，支持模型層級增量集成定位 | 本研究只有一次 Old／New 更新，且未重現該權重法；原文效能不移用 |
| TabM，2025，ICLR | 正式論文集題名、作者、年份及摘要；verified-metadata | 表格深度學習與參數有效率集成的候選基準 | 模型內部集成不同於跨時間模型池；不證明本研究的持續學習效益 |
| TabICL，2025，ICML，PMLR 267:50817–50847 | 正式論文集及摘要；verified-metadata | 表格基礎模型、情境學習候選基準 | 情境學習不自動等於持續學習；原文不能驗證本地歷史執行後端 |
| DESlib，2020，JMLR 21(8):1–5 | JMLR 正式頁及摘要；verified-metadata | DCS／DES／靜態集成軟體實作來源 | 軟體論文不等於各演算法優越性證據 |
| Hancock 等，2025，Journal of Big Data 12:12，DOI 10.1186/s40537-024-01041-1 | 出版者全文 Introduction、Case study data；verified-full（僅限所列主張） | SHAP 排序用於特徵選取的應用例子 | 以單一信用卡詐欺資料模擬三種標籤可得情境；不能推論破產或跨時間資料普遍有效 |
| Goldwasser、Hooker，2025，UAI，PMLR 286:1476–1496 | 正式論文集及摘要；verified-metadata | 特徵重要性排名具有抽樣不確定性的補充文獻 | 本研究入選頻率尚非該文的排名檢定 |
| Lombardo 等，2022，Future Internet，DOI 10.3390/fi14080244 | 本次出版頁限流；沿用現行引用庫紀錄，不提高驗證層級 | 資料來源與原始基準背景 | 不以出版者品牌評斷單篇品質；資料版本與標籤須獨立核對 |

## 判斷與待補強處

現有選文具可信的學術來源，且含原始方法、背景綜述、近期正式會議論文與實作文件。足以支持多數方法說明；這不等同逐篇方法品質均已通過完整審查。

第二章的直接缺口在於：對同時處理批次更新、非穩態與類別不平衡的研究，尚缺集中且近期的橫向比較。Learn++.NSE 很貼近模型池增量更新機制，但單篇舊研究不足以完整定位當前缺口。建議優先補少量直接相關研究，比較新舊模型保留方式、更新頻率、少數類處理、時間驗證與知識保留評估；先完成檢索再決定篇數，不預設需要新增模型實驗。

官方套件文件與資料庫說明適合支持版本、介面與資料出處，不應當作額外的同儕審查方法研究。現有第二章已交代多處使用界線；應保留這些限定，並加強跨研究綜合比較。

## 來源

- https://doi.org/10.1145/2523813
- https://users.rowan.edu/~polikar/RESEARCH/PUBLICATIONS/tnn11.pdf
- https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html
- https://proceedings.mlr.press/v267/qu25d.html
- https://jmlr.org/papers/v21/18-144.html
- https://link.springer.com/article/10.1186/s40537-024-01041-1
- https://proceedings.mlr.press/v286/goldwasser25a.html
