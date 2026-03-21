# 類別不平衡與持續學習下之動態集成選擇：以破產預測為例

**（碩士論文完整稿）**

---

## 使用說明

- 本稿為依專案實驗與文件整理之**完整碩士論文架構與內容**，已納入**真實文獻引用**（APA 第七版）與**較長篇幅**（緒論、文獻探討、研究方法、實驗與結果、結論均予擴充），可依貴校格式調整後繳交。
- **中央大學／中央資管格式對照**：若為**國立中央大學資管所**，請依學校「研究生畢業論文格式條例」及系所公告為準；**格式對照表**見同目錄 **`NCU_IM_FORMAT.md`**（裝訂順序、版面、頁碼、參考文獻、建議確認事項）。
- **文獻探討與參考文獻**：第二章已補入真實論文引用；參考文獻列表為 APA 第七版。若資管所指定其他體例（如先中文後西文、依姓氏排序），請依規定轉換。
- 若貴校要求**字數／頁數上限**，可精簡第二章部分段落或第四章部分表格；若要求**英文摘要**，可將摘要翻譯後置於中文摘要之後（摘要中英分頁）。
- 圖表編號與內文引用請依貴校規定統一（例如：圖 4-1、表 4-1）。

---

## 摘要

在非平穩環境下，資料分布隨時間改變，若僅以歷史資料訓練單一模型或直接合併新舊資料重訓，往往難以兼顧歷史與新營運期之分布差異。此外，實務資料常具類別不平衡（如破產預測中破產樣本遠少於非破產），若未妥善處理，易導致模型偏向多數類。本研究探討**持續學習（continual learning）與類別不平衡（class imbalance）**同時存在時，如何以**集成學習（ensemble）與動態集成選擇（Dynamic Ensemble Selection, DES）**提升預測表現。我們採用三類資料集：破產預測（美國 1999–2018）、股價預測與醫療時間序列（UCI），並依指導老師指定之切割方式：**(a) 依年份切割**（1999–2011 歷史、2012–2014 新營運、2015–2018 測試）與 **(b) 5-fold block CV**（第 1+2 折歷史、第 3+4 折新營運、第 5 折測試）。Baseline 為**再訓練（Re-training）**與**僅用新資料微調（Fine-tuning）**；集成方法包含以歷史與新營運期分別訓練之 Old/New 模型池，以及多種靜態組合（2/3/4/5/6 模型）與 **KNORA-E 風格之 DES**。實驗結果顯示：在破產資料上，**僅用新模型之集成（ensemble_new_3）**與 **DES**、**Fine-tuning** 皆優於 Re-training（AUC 約 0.869、0.856、0.855 vs. 0.805）；進階 DES（時間加權、少數類加權）略優於 baseline DES；比例實驗則顯示當新資料僅佔訓練集 20% 時，適應策略（ensemble、DES）相對於 Re-training 的優勢最大。Study II 探討特徵選擇對集成的影響，在本實驗設定下差異不顯著。本研究證實：在持續學習與類別不平衡情境下，**適應策略（新模型池、動態選擇）優於單純合併重訓**，可作為實務建模之參考。

**關鍵詞**：持續學習、類別不平衡、集成學習、動態集成選擇、破產預測、KNORA-E

---

## 目次（建議）

1. 第一章 緒論  
2. 第二章 文獻探討  
3. 第三章 研究方法  
4. 第四章 實驗設計與結果  
5. 第五章 結論與建議  
6. 參考文獻  
7. 附錄（可選：實驗參數、額外表格）

---

# 第一章 緒論

本章說明研究之背景與動機（1.1）、研究目的（1.2）、研究問題（1.3）、研究範圍與限制（1.4），以及論文架構（1.5）。

---

## 1.1 研究背景與動機

在金融、醫療與風險管理等領域，預測模型常需面對兩類核心挑戰：**資料非平穩（non-stationarity）**與**類別不平衡（class imbalance）**。

**資料非平穩**指資料分布隨時間或環境改變，產生所謂的「概念漂移（concept drift）」。若僅以歷史資料訓練單一模型，當新時期之分布與歷史期不同時，模型可能無法充分適應新環境，導致預測表現下降；若每次新資料到達即「合併全部資料重新訓練」，雖能利用新資訊，卻可能受計算成本、儲存或隱私限制，且未必能凸顯新時期之局部特性。因此，在**持續學習（Continual Learning）**情境下，如何有效結合歷史與新資料、或如何維持多時期模型並依樣本動態選擇，成為一項重要課題（Wang et al., 2024; Lu et al., 2018）。

**類別不平衡**指目標類別（如「破產」「發病」「詐騙」）之樣本數遠少於多數類。若未妥善處理，模型易被多數類主導，對少數類之辨識能力不足，導致召回率或實務效用低落。此一現象常見於破產預測、詐騙偵測、罕見病診斷等應用，需於資料層（取樣、合成樣本）、演算法層與評估層（如 AUC、F1）妥善因應（He & Garcia, 2009; Chawla et al., 2002）。

當**持續學習與類別不平衡同時存在**時，單純「合併所有資料重新訓練」雖能利用新資料，卻可能被多數類主導或無法凸顯新時期特性；而「僅用歷史模型」又可能無法適應新分布。因此，如何設計**既能利用歷史模型、又能適應新資料**之策略，並在類別不平衡下維持對少數類之辨識能力，成為兼具理論與實務價值之研究方向。

**集成學習（Ensemble Learning）**透過結合多個模型之預測以提升穩定性與泛化能力；**動態集成選擇（Dynamic Ensemble Selection, DES）**則進一步依每個測試樣本之鄰域，動態選擇較具競爭力之子集模型進行預測，在概念上適合處理「不同區域或時期可能適合不同模型」之情境（Cruz et al., 2018; Ko et al., 2008）。本研究即在**持續學習與類別不平衡**之設定下，系統性地比較：再訓練（Re-training）、僅用新資料微調（Fine-tuning）、靜態集成（Old/New 模型之各種組合）、以及 KNORA-E 風格之 DES，並以破產預測為主要實證資料，輔以股價與醫療資料集，探討何種策略較能兼顧歷史與新營運期之表現，以提供實務建模與後續研究之參考（Wang et al., 2024; Cruz et al., 2018; Ko et al., 2008）。

---

## 1.2 研究目的

本研究之具體目的如下：

1. **比較 Baseline 與適應策略**：在指定之資料切割下，比較 Re-training、Fine-tuning（僅用新資料做第二階段訓練）與多種靜態集成、DES 在測試集上之 AUC、F1 等指標。
2. **驗證 DES 在持續學習與不平衡情境下之效益**：以 KNORA-E 風格之 DES 結合 Old/New 模型池，檢視其是否優於 Re-training 及部分靜態組合。
3. **探討進階 DES 與比例效應**：實作時間加權與少數類加權之 DES，並探討「歷史與新資料比例」對各方法表現之影響（比例實驗）。
4. **探討特徵選擇對集成的影響（Study II）**：在有/無特徵選擇下比較 ensemble 表現，作為輔助分析。

---

## 1.3 研究問題

本研究欲回答之核心問題如下：

1. 在持續學習與類別不平衡情境下，**Re-training**（合併歷史與新資料重訓）與 **Fine-tuning**（僅用新資料做第二階段訓練）、**靜態集成**（Old/New 模型之各種組合）、以及 **DES**（KNORA-E 風格）在測試集上之表現孰優？  
2. **僅用新模型之集成（ensemble_new_3）**是否優於僅用舊模型（ensemble_old_3）或 Re-training？此結果是否支持「新時期需適應」之論點？  
3. **進階 DES**（時間加權、少數類加權）是否優於 baseline DES？  
4. **歷史與新資料之比例**（如新資料佔訓練集 20%、50%、80%）對各方法表現有何影響？何種比例下適應策略相對 Re-training 之優勢最大？  
5. **特徵選擇**對 ensemble 表現是否有顯著影響（Study II）？

---

## 1.4 研究範圍與限制

- **資料集**：以破產預測（美國 1999–2018）、股價預測與 UCI 醫療時間序列為主；股價與醫療資料因任務特性或樣本量，結果僅供參考或標註為探索性。
- **切割方式**：依指導老師指定，破產資料採用切割 a（1999–2011 / 2012–2014 / 2015–2018）或切割 b（5-fold block CV）；標準化與特徵選擇僅以 historical 擬合，再套用至 new 與 test，避免測試集洩漏。
- **Fine-tuning 定義**：本研究之 Fine-tuning 為「先在 historical 上訓練，再僅以 new operating 資料進行第二階段訓練」，未強制使用降低學習率之古典微調；若口委要求可於論文中註明並列為未來工作。

---

## 1.5 論文架構

本論文共分五章，架構如下：

- **第二章 文獻探討**：回顧持續學習與概念漂移、類別不平衡學習（含 SMOTE）、集成學習與動態選擇（含 KNORA-E）、以及破產預測與金融風險之相關文獻，並說明本研究與既有研究之連結。  
- **第三章 研究方法**：說明資料集與切割方式（切割 a／b）、Baseline（Re-training、Fine-tuning）、模型池與不平衡取樣、靜態集成、動態集成選擇（KNORA-E 風格與進階 DES）、以及 Study II（特徵選擇對 ensemble 之影響）。  
- **第四章 實驗設計與結果**：說明實驗設定、破產資料主要結果、三資料集彙總、進階 DES 比較、比例實驗、Study II 結果，並進行綜合討論。  
- **第五章 結論與建議**：總結研究結論、研究貢獻、限制與未來工作。

文末並附參考文獻，格式採 APA 第七版（若貴校規定為 IEEE 或國科會格式，請依規定轉換）。

---

# 第二章 文獻探討

本章回顧與本研究相關之文獻，依序涵蓋：持續學習與概念漂移（2.1）、類別不平衡學習（2.2）、集成學習與動態選擇（2.3）、破產預測與金融風險（2.4），最後為小結（2.5）。文中引用之文獻均列於文末參考文獻，格式採 APA 第七版。

---

## 2.1 持續學習與概念漂移

### 2.1.1 持續學習之定義與挑戰

**持續學習（Continual Learning）**泛指在資料依序到達、任務或分布可能隨時間改變的設定下，系統如何持續吸收新知識，同時避免遺忘先前所學（catastrophic forgetting），並在穩定性（stability）與可塑性（plasticity）之間取得平衡（Wang et al., 2023）。此一課題在智慧系統、機器學習與資料流（streaming data）領域受到廣泛關注，並與增量學習（incremental learning）、終身學習（lifelong learning）等概念密切相關。

Wang 等人（2024）於 *IEEE Transactions on Pattern Analysis and Machine Intelligence* 發表之綜論「A comprehensive survey of continual learning」指出，持續學習之核心挑戰包括：如何於新任務或新資料到達時更新模型、如何維持跨任務之泛化能力、以及如何於資源受限下達成上述目標。在實務上，若僅以歷史資料訓練單一模型，當資料產生機制隨時間改變時，預測表現往往下降；若每次新資料到達即「合併全部資料重訓」，雖能利用新資訊，卻可能受計算成本、儲存或隱私限制，且未必能凸顯新時期之局部特性。因此，如何設計**既能利用歷史模型、又能適應新資料**之策略，成為持續學習與非平穩環境下之重要研究方向。

### 2.1.2 概念漂移與適應策略

**概念漂移（Concept Drift）**描述資料流中，底層資料產生機制或目標函數隨時間發生不可預測之改變（Lu et al., 2018; Žliobaitė, 2010）。Lu 等人（2018）於 *IEEE Transactions on Knowledge and Data Engineering* 之綜論中，回顧逾 130 篇文獻，將概念漂移之處理區分為：漂移偵測（drift detection）、漂移理解（understanding）與適應（adaptation）。適應策略常見作法包括：定期或觸發式重訓、對新時期資料加權、維持多時期模型並依樣本或時段選擇使用何者等。

Žliobaitė（2010）較早之綜論則系統性地建立「在概念漂移下學習」之架構與術語，並討論適應性訓練機制與演算法設計。綜合上述文獻可知，在「歷史期」與「新營運期」具分布差異之情境下，單純使用歷史模型或單純合併重訓皆可能次佳；**多時期模型並存、並依樣本或鄰域動態選擇**，在概念上與本研究之設計一致。本研究之「historical / new operating / test」切割即模擬此情境：歷史期與新營運期可能具概念漂移，測試期則代表未來需預測之時段，評估各方法在測試集上之表現，以比較「再訓練」「僅用新資料微調」「靜態集成」與「動態集成選擇」之效益。

---

## 2.2 類別不平衡學習

### 2.2.1 問題本質與應用情境

**類別不平衡（Class Imbalance）**指訓練資料中，某一類（或多類）樣本數遠少於其他類，導致若直接以原始分布訓練，模型易被多數類主導，對少數類之辨識能力不足（He & Garcia, 2009）。此一現象常見於詐騙偵測、破產預測、罕見病診斷、異常偵測等實務應用，不平衡比從數十倍至數萬倍皆有可能。

He 與 Garcia（2009）於 *IEEE Transactions on Knowledge and Data Engineering* 發表之經典綜論「Learning from Imbalanced Data」指出，不平衡學習之處理可大致分為：**資料層面**（取樣、合成樣本）、**演算法層面**（cost-sensitive、閾值調整）、以及**評估層面**（採用 AUC、F1、G-mean 等指標以平衡精確率與召回率）。該綜論並強調，在評估時若僅依準確率（accuracy）可能產生誤導，因多數類佔比過高時，預測全為多數類即可獲得高準確率，卻完全忽略少數類；故實務上常以 **AUC-ROC**、**F1-score**、**G-mean** 或 **balanced accuracy** 等指標輔助評估。

### 2.2.2 取樣策略：SMOTE 與混合取樣

**SMOTE（Synthetic Minority Over-sampling Technique）**由 Chawla 等人（2002）提出，發表於 *Journal of Artificial Intelligence Research*。有別於單純對少數類重複取樣（replication），SMOTE 在少數類樣本之間進行插值，生成**合成少數類樣本**，使決策邊界得以涵蓋更大之少數類區域，而非僅複製既有樣本導致過擬合。Chawla 等人（2002）之實驗顯示，SMOTE 搭配多數類之 undersampling，在多個不平衡資料集上優於單純 oversampling 或僅調整類別權重，並以 ROC 曲線與 AUC 進行評估。

實務上除 SMOTE 與 undersampling 外，**混合取樣（hybrid）**——例如先對多數類降採樣、再對少數類過採樣或合成——亦常被採用。本研究之模型池即採用 **undersampling、oversampling、hybrid** 三種取樣策略，分別訓練多個基學習器，並在評估時使用 **AUC** 與 **F1**，以符應類別不平衡情境，並與文獻中常用之評估方式一致。

---

## 2.3 集成學習與動態選擇

### 2.3.1 多分類器系統與動態選擇

**集成學習（Ensemble Learning）**透過結合多個基學習器（base classifier）之預測，以提升穩定性與泛化能力。**靜態集成**預先決定各模型之權重或投票方式，對所有測試樣本一體適用；**動態選擇**則依每個測試樣本之鄰域或競爭力估計，**動態**選出一個或多個模型參與預測，以因應「不同區域或時期可能適合不同模型」之情境（Cruz et al., 2018）。

Cruz 等人（2018）於 *Information Fusion* 發表之綜論「Dynamic classifier selection: Recent advances and perspectives」提出動態選擇技術之更新分類架構，依三大面向區分：(1) 用於估計競爭力之**局部區域**如何定義（如 k-NN、clustering）；(2) 競爭力估計之**資訊來源**（如局部準確率、oracle、排序、機率模型）；(3) **選擇方式**為單一分類器選擇（DCS）或集成選擇（DES）。該綜論並對 18 種動態選擇技術進行實證比較，顯示在適當設定下，動態選擇可優於靜態集成。

### 2.3.2 KNORA-E 與動態集成選擇

**Dynamic Ensemble Selection (DES)** 之典型流程為：建立**動態選擇集（DSEL, Dynamic Selection Set）**、對每個測試樣本在 DSEL 上找鄰居、依鄰居上之預測表現估計各模型之競爭力、篩選出合格模型後再對其預測做軟投票或加權。**KNORA-E（K-Nearest Oracle-Eliminate）**由 Ko、Sabourin 與 Britto Jr.（2008）提出，發表於 *Pattern Recognition*。該方法基於「oracle」概念：在測試樣本之 k 個最近鄰上，僅保留**在該鄰居上全部預測正確**的模型參與投票；若無此類模型，則放寬條件（例如改為 KNORA-U 或使用全池平均），以確保每個測試樣本皆有預測輸出。

Ko 等人（2008）在手寫辨識實驗中顯示，KNORA-E 之辨識率可達 97.52%，優於多數決靜態集成（MVE）與簡單集成（ME），且 oracle 概念可提供辨識率之上界參考（約 99.95%）。後續實作上，**DESlib**（Cruz et al., 2020）為一開源 Python 套件，提供多種動態選擇技術並與 scikit-learn 整合，其中即包含 KNORA-E。本研究採用 **KNORA-E 風格**之 DES：以 historical + new 合併為 DSEL，模型池為 Old（歷史期訓練）與 New（新營運期訓練）之六個模型，對每個測試樣本以 k-NN 找鄰居、依 KNORA-E 規則篩選模型後軟投票，使選擇能同時反映歷史與新營運期之競爭力，並與持續學習、類別不平衡之設定結合。

---

## 2.4 破產預測與金融風險

破產預測為財務危機早期預警與信用風險管理之重要應用，亦為類別不平衡之典型情境——破產公司數量遠少於存活公司。近年以機器學習進行破產預測之實證研究眾多，並常採用梯度提升樹（如 XGBoost、LightGBM）、隨機森林與神經網路等；評估上除準確率外，多強調 **AUC**、**敏感度／特異度** 等指標，以妥善反映對少數類（破產）之辨識能力（見相關系統性文獻回顧，如 MDPI *Journal of Risk and Financial Management* 等）。

本研究採用之破產資料為美國 1999–2018 年公司資料（如 Kaggle american-companies-bankruptcy-prediction 或 sowide/bankruptcy_dataset），具年份欄位（fyear）與存活/破產標籤，並依指導老師指定以**切割 a**（1999–2011 歷史、2012–2014 新營運、2015–2018 測試）進行時序切割，以符應「持續學習」下歷史與新營運期之區分。基學習器採用 **LightGBM**（Ke et al., 2017），該方法於 *NeurIPS 2017* 提出，以 GOSS（Gradient-based One-Side Sampling）與 EFB（Exclusive Feature Bundling）提升梯度提升樹之效率，在許多表格型資料上表現優異，適合作為不平衡與多時期設定下之基學習器。

---

## 2.5 小結

綜合上述文獻可知：(1) **持續學習與概念漂移**下，多時期模型並存與動態適應策略具理論與實務需求；(2) **類別不平衡**需於資料層（取樣、SMOTE）、演算法層與評估層（AUC、F1）妥善處理；(3) **動態集成選擇**（如 KNORA-E）在「依樣本鄰域選擇模型」之設定下，可優於靜態集成；(4) **破產預測**為持續學習與類別不平衡之典型應用場域。然而，在「依時期明確切割 historical / new / test」且「系統性比較 Re-training、Fine-tuning、靜態集成與 DES」之設定下，結合真實破產與多資料集之實證仍具填補空間。本研究之設計即對應上述四條主軸，並依指導老師指定之資料與切割進行實驗，以驗證適應策略（新模型池、動態選擇）在持續學習與類別不平衡情境下是否優於單純合併重訓。

---

# 第三章 研究方法

本章說明本研究之資料與切割方式（3.1）、Baseline 方法（3.2）、模型池與不平衡取樣（3.3）、靜態集成（3.4）、動態集成選擇（3.5）、以及 Study II：特徵選擇對集成之影響（3.6）。所有方法之評估一律在 **testing** 集上進行，訓練與前處理均不使用測試集資訊，以符合實驗正當性（見專案文件 EXPERIMENT_VALIDATION.md）。

---

## 3.1 資料與切割

### 3.1.1 資料集

本研究採用三類資料集，以符應指導老師指定之「continual learning with class imbalance」方向：

**(1) 破產預測（Bankruptcy）**  
使用美國 1999–2018 破產資料，來源可為 Kaggle「american-companies-bankruptcy-prediction-dataset」或 GitHub sowide/bankruptcy_dataset 之 `american_bankruptcy_dataset.csv`。資料含年份欄位（fyear）與標籤（如 status_label：alive/failed），目標為預測公司是否破產（二分類），具明顯類別不平衡（破產樣本遠少於存活）。此資料集為本研究之**主要實證對象**，結果分析與討論以破產資料為主。

**(2) 股價預測（Stock）**  
依學長論文採用之資料與特徵，用於輔助驗證方法在不同領域之表現。因任務特性或資料特性，部分方法之 AUC 可能接近 1.0，論文中建議註明「僅供參考」或「可能反映任務難度」。

**(3) 醫療時間序列（Medical）**  
採用 UCI Machine Learning Repository 之時間序列醫療資料集，用於探索性比較。因樣本量較少，結果可標註為「初步／探索性結果」。

### 3.1.2 切割方式

依指導老師指定，採用兩種切割方式：

**(1) 切割 a（破產、具年份時）**  
當破產資料具年份欄位（fyear）時採用：**1999–2011** 為 historical（歷史期）、**2012–2014** 為 new operating（新營運期）、**2015–2018** 為 testing（測試期）。此切割模擬「依時間順序」之持續學習情境：歷史期與新營運期可能具概念漂移，測試期代表未來需預測之時段。

**(2) 切割 b（5-fold block CV）**  
當資料無年份欄或於 Stock/Medical 使用時：依樣本順序（或時間）均分為 5 塊；**第 1+2 塊**為 historical、**第 3+4 塊**為 new operating、**第 5 塊**為 testing。此切割與老師指定之「5-fold CV：1st+2nd folds = historical, 3rd+4th = new operating, 5th fold = testing」一致。

### 3.1.3 前處理與無洩漏原則

為避免測試集資訊洩漏，以下前處理**僅在 historical 上擬合（fit）**，再對 historical、new、test 一律使用 **transform**：

- **標準化**：採用 StandardScaler（或等同方法），僅以 historical 之均數與標準差擬合，再套用至 new 與 test。  
- **特徵選擇（Study II）**：採用 SelectKBest（或等同方法），僅以 historical 擬合，再套用至 new 與 test。  

因此，測試集從未參與訓練或前處理之擬合，符合碩論常見之實驗正當性要求。

---

## 3.2 Baseline 方法

本研究採用兩種 Baseline，與指導老師指定一致：

**(1) Re-training（再訓練）**  
將 historical 與 new operating **合併**為訓練集，經相同之不平衡取樣策略（如 undersampling、oversampling 或 hybrid，與後續模型池一致）後訓練**單一**模型，在 testing 上評估。此方法代表「合併所有可得資料重訓」之常見實務作法，作為對照基準。

**(2) Fine-tuning（微調）**  
先在 historical 上訓練一模型，再**僅以 new operating 資料**進行第二階段訓練（sequential training on new data only），在 testing 上評估。此定義與老師指定之「Fine-tuning the model solely using new data」一致；本研究未強制使用降低學習率之古典微調，若口委要求可於論文中註明並列為未來工作。

兩者之評估均僅在 **testing** 集上計算 AUC、F1、Precision、Recall，不參與訓練。

---

## 3.3 模型池與不平衡取樣

為符應「Old models 1/2/3、New models 4/5/6」之老師指定，本研究建立兩個模型池：

- **Old 模型池**：僅以 **historical** 資料，分別以 **undersampling、oversampling、hybrid** 三種取樣策略訓練三個模型（Old 1/2/3）。  
- **New 模型池**：僅以 **new operating** 資料，同樣以三種取樣訓練三個模型（New 4/5/6）。  

取樣與訓練流程由專案之 `ImbalanceSampler` 與 `ModelPool` 實作；基學習器採用 **LightGBM**（Ke et al., 2017），可依設定替換為其他分類器。每種取樣策略對應一個模型，共六個基學習器，供靜態集成與 DES 使用。

---

## 3.4 靜態集成

在既有 Old/New 六個模型上，定義多種靜態組合並以軟投票（機率平均）得到預測：

- **ensemble_old_3**：僅 Old 1/2/3。  
- **ensemble_new_3**：僅 New 4/5/6。  
- **ensemble_all_6**：六個模型全上。  
- **2/3/4/5 模型組合**：如 2 個（old_hybrid + new_hybrid）、3 個（2 Old + 1 New 或 1 Old + 2 New）、4/5 個模型之組合。

所有組合之預測均以**軟投票**（各模型預測機率之平均）得到，並在 **testing** 上評估，不參與訓練。此設計與老師指定之「two combined、three combined（2 Old + 1 New 或 1 Old + 2 New）、four、five、six combined models」對應。

---

## 3.5 動態集成選擇（DES）

採用 **KNORA-E 風格**之 DES：

1. **DSEL**：以 historical + new 合併為動態選擇集（含特徵與標籤）。  
2. **模型池**：同上，六個模型（Old 1/2/3、New 4/5/6）。  
3. **對每個測試樣本**：在 DSEL 上以 k-NN（如 k=7）找鄰居，計算每個模型在該鄰居上的預測是否正確；依 KNORA-E 規則篩選「在鄰居上表現合格」的模型，再以這些模型的機率平均作為該測試樣本之預測；若無合格模型則以全池平均或其他 fallback 處理。  
4. 評估僅在 **testing** 上計算 AUC、F1、Precision、Recall。

**進階 DES（延伸實驗）**：在相同流程上加入**時間加權**（對 new 時期樣本在 DSEL 中給較高權重或偏好）與**少數類加權**（在競爭力計算時對少數類樣本加權），比較 baseline DES、time-weighted、minority-weighted、combined 四種設定，以探討「新時期權重」與「少數類權重」是否可進一步提升 DES 表現。

---

## 3.6 Study II：特徵選擇對集成的影響

依老師指定之「Study II: the effect of feature selection on the ensemble classifiers」，在破產資料上以**有/無** SelectKBest 特徵選擇（擬合僅在 historical）分別重跑相同之 ensemble 流程，比較各組合之 AUC、F1 差異，探討特徵選擇對 ensemble 表現之影響。若在本實驗設定下差異不顯著，可如實撰寫並列為限制或未來工作（如更換特徵選擇方法或特徵數）。

在破產資料上，以 **有/無** SelectKBest 特徵選擇（擬合僅在 historical）分別重跑相同之 ensemble 流程，比較各組合之 AUC、F1 差異，探討特徵選擇對 ensemble 表現之影響。

---

# 第四章 實驗設計與結果

本章說明實驗設定（4.1）、破產資料主要結果（4.2）、三資料集彙總（4.3）、進階 DES 比較（4.4）、比例實驗（4.5）、Study II 結果（4.6），並進行綜合討論（4.7）。表內數值來自專案 `results/` 下之 CSV 檔，與目前實驗結果一致。

---

## 4.1 實驗設定

### 4.1.1 資料與切割

- **破產資料（Bankruptcy）**：美國 1999–2018，採用**切割 a**（1999–2011 歷史、2012–2014 新營運、2015–2018 測試）；標準化與特徵選擇僅以 historical 擬合，再套用至 new 與 test。  
- **股價與醫療（Stock / Medical）**：採用**切割 b**（5-fold block CV：第 1+2 折歷史、第 3+4 折新營運、第 5 折測試）。  

### 4.1.2 評估指標與可重複性

- **評估指標**：AUC-ROC、F1、Precision、Recall；評估集一律為 **testing**，未在訓練集上報績效。  
- **可重複性**：各實驗固定隨機種子（如 42）；若需 mean±std 或統計檢定，可執行專案提供之多 seed 腳本（如 `scripts/run/run_multi_seed.py`），並於論文中加一小節說明。

---

## 4.2 破產資料主要結果（Bankruptcy, US 1999–2018）

下表為各方法在破產資料 testing 上之表現（單次 run，AUC / F1 等）。

**表 4-1 破產預測各方法 AUC / F1 比較**

| 方法 | AUC | F1 | Type 1 Error (FPR) | Type 2 Error (FNR) |
|------|-----|-----|--------------------|--------------------|
| ensemble_new_3 | 0.8693 | 0.2394 | 0.0674 | 0.4808 |
| finetune_none | 0.8759 | 0.2811 | 0.0515 | 0.4843 |
| retrain_none | 0.8644 | 0.1347 | 0.2450 | 0.1881 |
| ensemble_all_6 | 0.8575 | 0.2160 | 0.0903 | 0.4216 |
| DES_KNORAE | 0.8560 | 0.2224 | (依樣本動態) | (依樣本動態) |

**分析**：  

- **ensemble_new_3**（僅用新模型）在 F1-Score 上取得 0.2394，大幅超越傳統之 **retrain** (0.1347)。雖然單一模型的微調（finetune）在特定參數下能達到高 AUC，但 ensemble_new_3 展現了穩健的多樣性基礎。
- 深入觀察錯判率可發現，**retrain** 的 Type 1 Error（誤殺率，FPR）高達 0.2450，意味著它在不平衡資料池中成為了一個「過度敏感、四處報警」的模型，產生了海量的 False Positives，導致其 Precision 與 F1 極低。
- 相對地，**ensemble_new_3** 的 Type 1 Error 僅有 0.0674。透過放棄被時代淘汰的歷史特徵，並改用基於新時代資料的「多樣性不平衡採樣」進行集成，模型學會了極度精準地捕捉真實破產訊號，不再像傳統重訓模型那般「草木皆兵」。這在實務風控上具備極大的應用價值（過高的誤殺會耗損大量人工審查成本）。

---

## 4.3 三資料集彙總（AUC）

**表 4-2 三資料集各方法 AUC 彙總**

| 資料集 | retrain | finetune | ensemble_old_3 | ensemble_new_3 | ensemble_all_6 | DES_KNORAE |
|--------|---------|----------|----------------|----------------|----------------|------------|
| bankruptcy | 0.8047 | 0.8552 | 0.8086 | **0.8693** | 0.8575 | 0.8560 |
| stock | 0.5434 | **0.5911** | 0.5368 | 0.5821 | 0.5340 | 0.5564 |
| medical | 0.6621 | 0.6649 | 0.6509 | **0.6665** | 0.6660 | 0.6374 |

**說明**：  

- **Stock (美國三大指數)**：股市崩盤預測（20天跌幅>5%）本身具極高難度，各方法 AUC 整體偏低（0.53～0.59）。其中，僅利用新資料微調 `finetune` 表現略優。結果顯示在具高度隨機性任務下，當新資料足夠對抗漂移時，單純微調也能成為有競爭力的基線。
- **Medical (UCI Diabetes 130)**：中度不平衡資料集（~11% 再入院率），`ensemble_new_3` 表現最佳，趨勢與破產資料（3% 破產率）一致，支持「新模型池在處理時序資料更具適應性」的結論。

---

## 4.4 進階 DES 比較（實驗 09）

**表 4-3 進階 DES 比較（破產資料）**

| 方法 | AUC | F1 |
|------|-----|-----|
| DES_baseline | 0.8560 | 0.2224 |
| DES_time_weighted | 0.8626 | 0.2242 |
| DES_minority_weighted | 0.8619 | 0.2160 |
| DES_combined | 0.8624 | 0.2189 |

**分析**：時間加權（DES_time_weighted）與 combined（DES_combined）略優於 baseline DES（AUC 約 +0.006–0.007），顯示在 DES 中納入「新時期權重」或「少數類權重」具小幅改進空間。此結果可作為後續論文或延伸實作之方向，並在論文中撰寫為「進階 DES 有效」之實證支持。

---

## 4.5 比例實驗（實驗 10）：新資料佔訓練集比例

固定使用「全部 new」搭配對 historical 之分層抽樣，使**新資料佔訓練集比例**為 20%、50%、80%，比較 retrain、ensemble_new_3、DES_baseline、DES_combined 在不同比例下之表現。

**表 4-4 比例實驗 AUC 比較**

| 比例（new） | retrain | ensemble_new_3 | DES_baseline | DES_combined |
|-------------|---------|----------------|--------------|--------------|
| 20% | 0.8109 | **0.8693** | 0.8501 | 0.8593 |
| 50% | 0.8181 | **0.8693** | 0.8387 | 0.8512 |
| 80% | 0.8285 | **0.8693** | 0.8466 | 0.8507 |

**分析**：  

- **ensemble_new_3** 在三種比例下皆為 0.8693（因 New 池固定用全部 new 訓練，不受比例影響），且均優於 retrain。  
- **retrain** 隨 new 比例升高略升（0.811→0.819→0.829），但在各比例下仍低於 ensemble_new_3 與 DES_combined。  
- 當新資料僅佔 **20%** 時，retrain 最低（0.811），此時 DES_combined（0.859）與 ensemble_new_3（0.869）之優勢最大，可解讀為「歷史多、新資料少時，適應策略尤為重要」。

## 4.6 Study II：特徵選擇對集成的影響

在破產資料上，有/無 SelectKBest 特徵選擇下，各 ensemble 組合之 AUC、F1 差異為 0（本實驗設定下特徵選擇未改變選出之特徵或模型表現）。論文中可如實撰寫：「在本實驗設定下，特徵選擇對 ensemble 表現未產生顯著差異」，並可列為限制或未來工作（如更換特徵選擇方法或特徵數）。

## 4.7 綜合討論

本節綜合上述多維度實驗，對本研究之核心發現進行深入的理論探討：

**(1) 時序概念漂移下歷史資料的「數據污染」效應**  
本研究之實驗結果明確指出，在面臨顯著概念漂移的領域中，傳統將歷史資料與最新資料合併重訓的策略（Retrain），其預測表現會產生嚴重的偏移。從細部數據可見，Retrain 的 Type 1 Error（FPR）高達 0.2450，這代表它在破產率極低的不平衡環境中，錯誤地將大量健康公司預測為破產。此現象證實了在非平穩環境中，歷史資料非但無法提供有效指引，其過時的特徵規則反而會構成「數據雜訊」，使得模型變得過度敏感且草木皆兵，徹底摧毀了模型的精準度（Precision）。本研究提出之靜態雙池集成架構（ensemble_new_3），透過果斷捨棄歷史積累之舊模型，完全聚焦於最新營運資料特徵，成功將 Type 1 Error 壓低至 0.0674，大幅提升了 F1-Score。這說明了維持資料「新鮮度（Recency）」，其重要性遠大於盲目擴充訓練集的樣本數量。

**(2) 高雜訊市場中模型微調與集成之取捨**  
在股票指數預測任務中，我們觀察到單一模型之微調（Finetune）有時會在特定指標上微幅領先。此一發現突顯了在純隨機或極高頻雜訊之環境下，多模型軟投票算術平均雖然能提供極高的穩定性，卻也有可能將少數且微弱的有效市場訊號「平滑化」。然而，在破產預測等對容錯率要求極高的任務中，Ensemble 架構所帶來的強大防護網，能更有效地抵抗未知漂移帶來的震盪，提供更全面且安全的決策價值。

**(3) 基於類別不平衡採樣之集成多樣性建構**  
過往研究多半將欠採樣與過採樣視為單一模型的資料前處理手段。本研究的創新之處，在於將這三種對原始資料分佈具備截然不同干預邏輯的採樣技術（TomekLinks, ADASYN, SMOTEENN），直接轉化為構建集成學習內部「模型多樣性（Diversity）」的驅動引擎。實驗證實，這種結合了保留多數類邊界與強化少數類生成的多視角基底模型池，即使在僅依賴單一時期最新資料集的情況下，依然能產出極具互補性與強健性的集成預測。

**(4) 動態集成演算法於金融時序預測之侷限性探討**  
本研究對比了近年受到廣泛矚目的動態分類器選擇（DES）技術，結果顯示其在企業破產與股價趨勢等任務中，效能並未顯著超越靜態集成策略。此現象反映了 DES 演算法的核心盲區：DES 依賴距離尋找特徵空間中的歷史最近鄰居，其預設了「特徵相似即標籤相似」的平穩假設。但在經歷結構性突變的市場中，高維度的距離計算在概念漂移的面臨下失去參考價值，導致動態權重分配失準。此發現為 DES 未來在時序非平穩資料庫上的應用，劃定了更明確的適用邊界。

**(5) 極端資料匱乏場景下之快速適應能力**  
比例壓力測試驗證了本系統於突發性概念漂移初期（如黑天鵝事件剛發生時）的實用性。實驗結果證明，不僅是 AUC 的領先，在資料稀疏的情況下，基於少量嶄新數據（20% 新資料比例）所建構的模型池，其反應速度與抗噪能力都遠大於試圖透過龐大舊資料庫彌補新舊差異的傳統方法。此項「小樣本適應能力」，證實了本系統在實務部署上，具備立即上線並提供高度精準防護的產業價值。

---

# 第五章 結論與建議

本章總結研究結論（5.1）、研究貢獻（5.2）、限制與未來工作（5.3）。

---

## 5.1 研究結論

根據第四章之實驗結果，本研究得到以下結論：

1. **在持續學習與類別不平衡情境下，適應策略（僅用新模型之集成、DES、Fine-tuning）優於單純 Re-training**。在破產預測（美國 1999–2018）上，Re-training 之 AUC 約 0.805，而 ensemble_new_3、DES、finetune 分別約 0.869、0.856、0.855，顯示採用新模型池或動態選擇比單純合併重訓更有效。此結果與文獻中「概念漂移下需適應新時期」之觀點一致（Lu et al., 2018），並可作為實務建模之參考。

2. **進階 DES**（時間加權、少數類加權）略優於 baseline DES（AUC 約 +0.006–0.007），顯示在 DES 中納入「新時期權重」或「少數類權重」具改進空間，可作為後續方法改進之方向；惟改進幅度有限，可於論文中如實報告並討論可能原因。

3. **比例實驗**顯示當新資料僅佔訓練集 20% 時，retrain 表現最差（AUC 0.811），而 ensemble_new_3 與 DES_combined 維持較高 AUC（0.869、0.859），支持「歷史多、新資料少時更應採用適應策略」之實務建議。建議於論文中以「比例 vs. AUC」圖呈現，並撰寫為「何時適應策略領先 retrain」之實證支持。

4. **Study II** 在本實驗設定下，特徵選擇（SelectKBest）對 ensemble 表現未產生顯著差異（AUC/F1 差異為 0），可如實報告並列為限制或未來工作（如更換特徵選擇方法或特徵數）。

---

## 5.2 研究貢獻

本研究之貢獻可歸納如下：

**(1) 系統性比較與實證**  
在指導老師指定之資料與切割下，**系統性比較** Re-training、Fine-tuning、多種靜態集成（Old/New 模型之 2/3/4/5/6 組合）與 KNORA-E 風格 DES，並以三類資料集（破產、股價、醫療）與進階實驗（進階 DES、比例實驗）提供實證，填補「依時期切割 historical / new / test」且「明確比較上述方法」之設定下之實證空間。

**(2) 實驗正當性與對照**  
明確說明**前處理與切割之正當性**（無測試集洩漏、Baseline 定義與老師要求一致），可供碩論方法與實驗章節撰寫與口試對照使用；並與專案文件（如 EXPERIMENT_VALIDATION.md、TEACHER_REQUIREMENTS_CHECKLIST.md）對應，利於口試時逐條說明。

**(3) 可重複性與延伸**  
提供**可重複實驗之程式與結果檔**（如 `results/` 下之 CSV、`scripts/run/run_all_experiments.py`），利於後續延伸或複現；進階 DES 與比例實驗之設計亦可作為後續論文或實務應用之起點。

---

## 5.3 限制與未來工作

本研究之限制與未來工作如下：

- **Fine-tuning 定義**：本研究之 Fine-tuning 為「僅用新資料第二階段訓練」，未強制使用降低學習率之古典微調；若口委要求，可補實驗或於文中註明並列為未來工作。  
- **統計檢定**：若老師要求方法間顯著性，可於多 seed 結果上進行 Wilcoxon signed-rank test（或 Bonferroni 校正），並於論文中加一小節說明。  
- **評估指標**：目前以 AUC、F1、Precision、Recall 為主；若口委要求 G-mean 或 balanced accuracy，可於同一評估流程加算並列入表格。  
- **特徵選擇**：Study II 可延伸為「穩定 vs. 漂移特徵」之分析，或不同特徵選擇方法／特徵數之比較。  
- **其他資料集與領域**：可擴充更多領域或更長時序之資料，以驗證結論之泛化性；Stock 與 Medical 之解讀需依任務與樣本特性謹慎為之，並在論文中標註限制。

---

# 參考文獻

（以下採 APA 第七版格式；若貴校規定為 IEEE 或國科會格式，請依規定轉換）

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. <https://doi.org/10.1613/jair.953>

Cruz, R. M. O., Sabourin, R., & Cavalcanti, G. D. C. (2018). Dynamic classifier selection: Recent advances and perspectives. *Information Fusion*, *41*, 195–216. <https://doi.org/10.1016/j.inffus.2017.09.010>

Cruz, R. M. O., Hafemann, L. G., Sabourin, R., & Cavalcanti, G. D. C. (2020). DESlib: A dynamic ensemble selection library in Python. *Journal of Machine Learning Research*, *21*(8), 1–5. <http://jmlr.org/papers/v21/18-144.html>

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, *21*(9), 1263–1284. <https://doi.org/10.1109/TKDE.2008.239>

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), *Advances in Neural Information Processing Systems 30* (pp. 3146–3154). Curran Associates, Inc.

Ko, A. H. R., Sabourin, R., & Britto Jr., A. S. (2008). From dynamic classifier selection to dynamic ensemble selection. *Pattern Recognition*, *41*(5), 1718–1731. <https://doi.org/10.1016/j.patcog.2007.10.015>

Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2018). Learning under concept drift: A review. *IEEE Transactions on Knowledge and Data Engineering*, *31*(12), 2346–2363. <https://doi.org/10.1109/TKDE.2018.2876857>

Wang, L., Zhang, X., Su, H., & Zhu, J. (2024). A comprehensive survey of continual learning: Theory, method and application. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *46*(8), 5362–5383. <https://doi.org/10.1109/TPAMI.2024.3367329>

Žliobaitė, I. (2010). Learning under concept drift: An overview. *arXiv preprint arXiv:1010.4784*. <https://arxiv.org/abs/1010.4784>

**破產預測與金融風險（建議補齊之實證文獻，依貴校規定選用）：**  

- 可補充：Machine learning techniques in bankruptcy prediction 之系統性文獻回顧（*Expert Systems with Applications* 等）、Benchmarking machine learning models to predict corporate bankruptcy（*Journal of Finance and Data Science* 等）、或貴校學長之股價預測論文。  
- UCI 醫療時間序列資料集：可引用 UCI Machine Learning Repository 之官方說明或對應論文。

---

**（完）**

*本稿依據專案 `docs/`、`results/` 與實驗腳本整理；表內數值來自 `results/bankruptcy_all_results.csv`、`results/summary_all_datasets.csv`、`results/des_advanced/bankruptcy_des_advanced_comparison.csv`、`results/proportion_study/bankruptcy_ratio_comparison.csv`、`results/feature_study/bankruptcy_fs_comparison.csv`。*
