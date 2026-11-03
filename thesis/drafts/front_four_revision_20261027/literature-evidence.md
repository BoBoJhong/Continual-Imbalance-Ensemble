# 前四章文獻與內容核對

日期：2026-10-27。此為配合兩項實驗的主題式文獻整理，非系統性回顧，也未主張已搜尋全部最新方法。外部論文只提供方法背景；本研究結果來自既有保存資料。

## 參考論文的使用範圍

使用者提供林鈺惠（2026）《多策略資料前處理之研究：結合特徵選取、樣本選取與重採樣方法》。完整抽取見 `tmp/pdfs/front-four-reference-20261027/reference.txt`，相關頁次摘錄已存同目錄。參照的是章節功能與圖表呈現，未複製原句，也未移入其二十份資料、樣本選取、五折驗證或 Wilcoxon 結果。

| 參考論文安排（PDF 頁次） | 本研究處理 |
|---|---|
| 第一章背景、動機、目的與架構（9–12） | 寫入時間、不平衡、公開企業案例、三研究問題與限制。 |
| 第二章分主題回顧（13–24） | 依時間、採樣、分類器、集成、FS/XAI、評估組織；直接核對原始方法來源。 |
| 第三章架構、方法、參數及驗證（25–44） | 保留既定兩項實驗、現有參數、通用開頭與固定年份配置。 |
| 第四章分實驗結果與分析（45 起） | 保留真實表格，新增三圖與研究問題討論，附錄保留完整比較。 |
| 章次-序號的圖表 | 依使用者明確指定套用，並記錄與較早學校格式參考的差異。 |

## 搜尋紀錄與篩選

搜尋日：2026-10-27；語言：英文原始文獻。工具：網頁搜尋、出版社／論文集、作者原稿與套件官方文件。查詢包含「DESlib JMLR 2020」「Combination of Multiple Classifiers Using Local Accuracy Estimates」「From dynamic classifier selection to dynamic ensemble selection」「Guyon Elisseeff feature selection 2003」「ADASYN He 2008」「Batista balancing 2004」「Gama concept drift 2523813」「Saito Precision Recall 2015」「Random Forests 1010933404324」「TabM ICLR 2025」「TabICL ICML 2025」「LightGBM NeurIPS 2017」「TomekLinks imbalanced-learn」。

納入：與實際方法直接相關、可核對書目及支持段落的原文／官方摘要。排除：僅由二手引用推得的敘述、TabICLv2 等未用於現行實驗的版本、與本研究無關的最新方法排名。OpenReview、IEEE 及 ACM 部分頁面受驗證或 403 限制，改用作者版本、正式論文集或研究機構頁，未推定未取得的全文內容。

## 證據表

Verified-full 指已核對原文中的支持段落；Verified-abstract 指僅核對原作者／出版來源的摘要與書目，不宣稱查完全文。所有敘述均自行撰寫、無逐字引文。

| 引用鍵 | 來源與定位 | 支持主張 | 核對層級／限制 |
|---|---|---|---|
| gama2014drift | DOI 10.1145/2523813；原稿 p.1（作者、文章44、37頁及摘要） | 概念漂移關心輸入與目標關係隨時間改變 | Verified-full；未以本研究年度分數診斷漂移 |
| saito2015pr | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432 ，摘要與 Introduction | 不平衡下需檢視正類預測；ROC 與 PR 解讀不同 | Verified-full；本文仍照既定規則報 AUC/F1/Recall |
| breiman2001rf | https://link.springer.com/article/10.1023/A:1010933404324 ，書目與 Abstract | 隨機樹組合與樹間相關性 | Verified-abstract |
| ke2017lightgbm | https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html | 梯度提升樹的效率設計 | Verified-abstract；未宣稱本地實作使用 GOSS |
| gorishniy2025tabm | https://arxiv.org/abs/2410.24210 ，v3 摘要、ICLR2025；作者 https://github.com/yandex-research/tabm/blob/main/paper/README.md 書目 | 共享參數的 MLP 多預測 | Verified-abstract；非本地結果保證 |
| qu2025tabicl | https://proceedings.mlr.press/v267/qu25d.html ，正式書目與摘要 | 原 TabICL 的情境學習 | Verified-abstract；本地後端未完整證實，避免誤認新版 |
| guyon2003selection | https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf ，p.1157–1158 | 選取目的及 Filter/Wrapper/Embedded | Verified-full |
| he2008adasyn | DOI 10.1109/IJCNN.2008.4633969；原稿 https://sci2s.ugr.es/keel/pdf/algorithm/congreso/2008-He-ieee.pdf p.1 | 困難少數類區域獲得較多合成樣本 | Verified-full；Edwardo A. Garcia 依原稿拼寫 |
| batista2004balancing | DOI 10.1145/1007730.1007735；原稿 https://storm.cis.fordham.edu/~gweiss/selected-papers/batista-study-balancing-training-data.pdf ，摘要與方法 | 少數類不足／類別重疊與 SMOTE+ENN | Verified-full；未搬用其效果到本資料 |
| woods1997local | DOI 10.1109/34.588027；https://cse.msu.edu/~cse802/Papers/WoodsBowyer_ClassifierCombinationLocalAccuracyEstimates.pdf | 局部正確程度作選擇依據 | 原稿第一次取得支持段落，後續重取逾時；細節另由既有 OLA/LCA 官方文件核對 |
| ko2008dynamic | DOI 10.1016/j.patcog.2007.10.015；https://pure.etsmtl.ca/en/publications/from-dynamic-classifier-selection-to-dynamic-ensemble-selection/ 書目與摘要 | 靜態與動態成員選擇及 oracle-based 方法 | Verified-abstract；KNORA細節另用官方文件 |
| cruz2020deslib | https://jmlr.org/papers/v21/18-144.html ，書目與摘要 | DCS、DES、static 三模組 | Verified-abstract |
| imblearn-tomek-docs | https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.TomekLinks.html ，類別移除參數及數量注意事項 | TomekLinks 為清理式低採樣、不保證完全平衡 | 官方文件已讀；實驗版本仍依既存環境表 |

既有 SMOTE、SHAP、SHAP-FS、特徵排名、bootstrap、群聚 ROC、Holm、資料集與套件文件的核對沿用前輪證據。新文獻 DOI 已與研究規格及 BibTeX 去重；不以參考論文的二手書目代替原文查核。

## 主張帳本

| ID | 主張 | 實證／文字來源 | 狀態 |
|---|---|---|---|
| F1 | 論文採兩項實驗，三研究問題對應 Baseline、集成、FS | 第一、三、四章及既有固定年份協定 | 對齊，無新增實驗 |
| F2 | 前兩章為自行組織的研究敘述與已核對原文背景 | 本表；references.bib | 非逐字移植；不提供未實測的抄襲率或 AI 百分比 |
| F3 | 三張新結果圖與既有表格同來源 | figures-proof.json，既有 results-proof.json，統計 manifest | 21列Baseline、24格集成指標、48格FS AUC；固定順序 |
| F4 | 圖表依章重置，正文4-1與附錄A-1不同前綴 | ncuimthesis.cls；最終編譯交叉引用 | 已通過原生／編輯器編譯及目錄、圖表編號檢查；見 verification.json |
| F5 | New描述性結果、V-static相對Equal6差異、FS無顯著AUC改善 | baseline-summary.csv、descriptive.csv、paired_tests.csv | 與既有數值相同；非全域優越或等效主張 |
| F6 | 第一至第四章內容齊備不代表整本可送審 | 第五章、摘要、config及正式表件 | 範圍限制保留，另需作者審閱與正式資料 |
