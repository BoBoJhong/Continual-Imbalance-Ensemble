# 專案相關文獻與引註指南

最後核對日期：2026-09-07

本文件依本專案實際使用的方法整理，目的不是羅列大量關鍵字相近的文章，而是提供可直接支撐研究動機、方法設計、實驗設定與結果討論的引用來源。完整 BibTeX 位於 [`references.bib`](references.bib)。

## 1. 文獻選擇原則

- **方法原始文獻優先**：即使原始方法發表於研討會，也應優先於後來的二手說明。
- **高影響力期刊／頂級會議補強**：優先採用 IEEE TKDE、IEEE TNNLS、IEEE TPAMI、JMLR、Pattern Recognition、Information Fusion、Nature、NeurIPS、ICLR、ICML、KDD 等來源。
- **資料集必須引用原始資料論文或官方 repository**：不能只引用 Kaggle、GitHub 鏡像或下載頁。
- **軟體實作與演算法概念分開引用**：例如 KNORA-E 引用 Ko et al. (2008)，使用 DESlib 時再引用 Cruz et al. (2020)。
- **「頂級」沒有跨領域唯一標準**：本文件不宣稱所有來源都是頂級期刊；對方法而言，「原始且可核驗」通常比期刊排名更重要。

## 2. 最優先引用清單

若篇幅有限，建議至少保留下列文獻。

| 專案內容 | 建議引用 | 性質 | 用途 |
| --- | --- | --- | --- |
| 美國企業破產資料 | Lombardo et al. (2022) | 資料集原始論文 | 78,682 firm-year observations、1999–2018、時間切割與標籤定義 |
| 經典破產預測 | Altman (1968); Ohlson (1980) | 領域奠基文獻 | 財務比率、判別分析與機率式破產模型 |
| 不平衡學習 | He and Garcia (2009) | IEEE TKDE 綜述 | 不平衡問題、資料層與演算法層處理方法 |
| SMOTE | Chawla et al. (2002) | 原始論文 | 合成少數類過採樣 |
| ADASYN | He et al. (2008) | 原始論文 | 依學習難度自適應產生少數類樣本 |
| SMOTEENN | Batista et al. (2004) | 原始／核心來源 | SMOTE 與 Edited Nearest Neighbours 的混合清理 |
| 概念漂移 | Gama et al. (2014); Lu et al. (2019) | ACM CS／IEEE TKDE 綜述 | 漂移定義、類型、偵測與調適分類 |
| 漂移與不平衡交集 | Gao et al. (2007); Wang et al. (2015) | 原始方法／IEEE TKDE | skewed drifting streams 與 online class imbalance |
| KNORA-E/U | Ko et al. (2008) | Pattern Recognition 原始論文 | 動態集成選擇與 KNORA 方法 |
| DES/DCS 整體架構 | Cruz et al. (2018) | Information Fusion 綜述 | region of competence、competence estimation、DCS/DES 分類 |
| XGBoost | Chen and Guestrin (2016) | KDD 原始論文 | 主要基學習器 |
| PR-AUC | Saito and Rehmsmeier (2015) | 指標研究 | 說明不平衡資料不能只看 ROC-AUC |
| 統計比較 | Demšar (2006) | JMLR 方法論 | paired Wilcoxon、Friedman 與多模型比較 |

## 3. 研究問題：概念漂移與類別不平衡

### 3.1 概念漂移

建議以 Gama et al. (2014) 或 Lu et al. (2019) 定義概念漂移，再用本專案的 Old/New 時間分段具體化。兩篇均屬高影響力綜述，適合文獻回顧；若介紹偵測器，則回到各方法原始文獻。

- **DDM**：Gama et al. (2004)。監控線上錯誤率及其標準差，對應 `experiments/phase4_drift/_detectors.py`。
- **ADWIN**：Bifet and Gavaldà (2007)。以自適應視窗檢測分布變化。
- **Page–Hinkley 類累積和檢定**：Page (1954)。本專案是工程化的單向均值漂移版本，論文中應寫「Page–Hinkley-style test」或清楚列出實作公式，不宜宣稱與特定套件完全等價。

可用論述：

> 概念漂移使資料分布或輸入與標籤間的關係隨時間改變，因此以固定歷史資料訓練的模型可能逐漸失效（Gama et al., 2014; Lu et al., 2019）。

### 3.2 漂移與不平衡同時存在

Gao et al. (2007) 直接處理 skewed、concept-drifting streams；Wang et al. (2015) 系統比較以 resampling 為基礎的 online ensemble。這兩篇最適合支撐本研究「不能只處理漂移或只處理不平衡」的研究缺口。

可用論述：

> 在少數類事件稀少且分布持續改變的情境中，模型必須同時處理類別偏斜與時間非平穩性；只針對其中一項設計的方法未必能維持少數類辨識能力（Gao et al., 2007; Wang et al., 2015）。

## 4. DAWCE／ROSS 的理論定位

`DAWCE`（Drift-Aware Weighted Continual Ensemble）與 `ROSS` 是本專案的研究命名，不是外部既有演算法名稱。因此：

- 不應寫成「依據某篇 DAWCE 原始論文實作」。
- 應將其定位為結合既有思想的新框架：時間分段、漂移邊界驗證、異質採樣模型池、validation-selected Old/New weighting，以及固定後在未見 test period 評估。
- 最接近的理論脈絡包括 SEA（Street & Kim, 2001）、概念漂移集成（Wang et al., 2003）、Dynamic Weighted Majority（Kolter & Maloof, 2007）與 Accuracy Updated Ensemble（Brzeziński & Stefanowski, 2014）。
- DAWCE 與上述線上方法並不等價。若目前以年度批次及 validation 搜尋權重，應稱為 **batch-adaptive continual ensemble**，不要直接宣稱是逐實例 online learning。

建議寫法：

> Inspired by performance-aware ensemble adaptation under concept drift (Wang et al., 2003; Kolter & Maloof, 2007; Brzeziński & Stefanowski, 2014), this study develops a batch-adaptive framework that explicitly separates historical and recent model pools and selects their relative weight using temporally prior validation data.

## 5. 不平衡資料處理

| 專案策略 | 主要文獻 | 引註重點 |
| --- | --- | --- |
| Random under/over-sampling | He and Garcia (2009) | 作為基本資料層方法，不需假稱由單一新論文提出 |
| Tomek Links | Tomek (1976) | 移除類別邊界上的近鄰對；原文是 condensed nearest-neighbour 的修改 |
| SMOTE | Chawla et al. (2002) | 在少數類近鄰間合成樣本 |
| ADASYN | He et al. (2008) | 對較難學習的少數類區域配置更多合成樣本 |
| SMOTEENN | Batista et al. (2004) | 合成後以 ENN 清理重疊與雜訊 |
| 不平衡學習整體理論 | He and Garcia (2009) | class skew、small disjuncts、overlap 與評估問題 |

注意：採樣只應在 training fold 內執行。若論文要說明資料洩漏防護，應同時描述實驗流程，不要只用採樣論文代替實作證據。

## 6. 動態分類器／集成選擇

- **KNORA-E／KNORA-U**：Ko et al. (2008) 是最直接的原始文獻。
- **OLA／LCA 與 DES/DCS taxonomy**：可用 Cruz et al. (2018) 作為統一說明來源。
- **FIRE-DES／Dynamic Frienemy Pruning**：Oliveira et al. (2017) 是原始框架；若實驗真的啟用 DFP，才應列為方法引用。
- **DESlib 軟體**：Cruz et al. (2020)。只有使用或對照 DESlib 實作時才引；本專案自己的 `src/ensemble/selector.py` 不能因介面相似就聲稱是 DESlib 原碼。
- **方法限制**：局部 competence 依賴 DSEL、距離尺度與 neighborhood 品質。若本研究發現 static ensemble 優於 DES/DCS，可引用 Cruz et al. (2018) 的 taxonomy 作為討論背景，但結果仍應由本研究實驗支持。

可用論述：

> Dynamic selection estimates classifier competence in a local region surrounding each query and selects either one classifier (DCS) or a subset of classifiers (DES) at prediction time (Cruz et al., 2018). KNORA-E/U operationalize this idea through the oracle behavior of classifiers in a k-nearest-neighbor competence region (Ko et al., 2008).

## 7. 基學習器與表格模型

| 模型 | 應引用來源 | 備註 |
| --- | --- | --- |
| XGBoost | Chen and Guestrin (2016) | 本專案主線模型；KDD 原始論文 |
| LightGBM | Ke et al. (2017) | NeurIPS 原始論文 |
| Random Forest | Breiman (2001) | Machine Learning 原始論文 |
| SVM | Cortes and Vapnik (1995) | Machine Learning 原始論文 |
| TabNet | Arik and Pfister (2021) | AAAI 原始論文 |
| FT-Transformer | Gorishniy et al. (2021) | NeurIPS；論文題名為 *Revisiting Deep Learning Models for Tabular Data* |
| TabR | Gorishniy et al. (2024) | ICLR 原始論文 |
| TabM | Gorishniy et al. (2025) | ICLR 原始論文 |
| TabICL | Qu et al. (2025) | ICML 原始論文；對應 `tabicl` backend |
| TabPFN fallback | Hollmann et al. (2025) | Nature；只有 wrapper 實際退回 `tabpfn` backend 時使用 |

重要實作辨識：`TabICLWrapper` 會優先載入 `TabICLClassifier`，失敗時才退回 `TabPFNClassifier`。研究結果表必須記錄實際 backend、套件版本與 checkpoint，不能把 TabICL 與 TabPFN 當成同一方法引用。

## 8. 特徵選擇與穩定性

- **Mutual-information feature selection**：Brown et al. (2012) 可提供資訊理論式特徵選擇的完整框架。
- **RFE**：Guyon et al. (2002) 是經典原始應用與方法來源。
- **SHAP**：Lundberg and Lee (2017) 是原始統一框架；本專案以 mean absolute SHAP 排序時，應明確寫出聚合方式。
- **Feature-selection stability**：Nogueira et al. (2018) 提供具統計性質的穩定度框架。若本專案報告的是 Jaccard，應稱為 Jaccard overlap/stability，不要誤稱為 Nogueira stability estimator。

## 9. 評估、時間驗證與統計檢定

### 9.1 指標

- ROC-AUC 可評估排序能力，但在高度不平衡資料上不應單獨使用。
- Saito and Rehmsmeier (2015) 可支撐 PR curve／average precision 對少數類評估的重要性。
- 建議同時報告 ROC-AUC、PR-AUC、F1、Recall、Precision、G-Mean、Balanced Accuracy，以及 Type-I／Type-II error；並清楚指定正類為 bankruptcy/failure。
- F1、Recall、Precision 等 threshold-dependent 指標的閾值必須只由 validation 選取。

### 9.2 時間切割

本專案的 bankruptcy 主線沿用 train 1999–2011、validation 2012–2014、test 2015–2018 的資料集原始基準，可引用 Lombardo et al. (2022)。時間相依資料的驗證不宜任意打亂；Bergmeir et al. (2018) 可用來說明一般 k-fold 在時間資料上的適用條件與限制，但該文聚焦 autoregressive forecasting，引用時不要擴張成所有 temporal classification 的定理。

### 9.3 統計比較

Demšar (2006) 建議兩個方法的配對比較可使用 Wilcoxon signed-rank test，多方法、多資料集比較可使用 Friedman 與適當 post-hoc procedure。本專案目前以多個年份邊界作為 paired units 時，應額外說明：

- 各 split 是否真正獨立；重疊年份窗會造成相依性。
- 報告 exact/approximate p-value、雙尾或單尾、樣本數、零差值處理及效果量。
- 多個方法或多個指標同時檢定時，考慮 Holm 等 multiplicity correction。
- `test-selected oracle` 只能作描述性上界，不能當成可部署方法或公平的顯著性比較對象。

## 10. 資料集引用

### 10.1 American Bankruptcy Dataset

主要引用：Lombardo et al. (2022)。該文同時提供資料來源、公司範圍、18 個會計變數、標籤邏輯及官方時間切割。本專案使用此資料時，應保留其原始定義並記錄任何重新命名或衍生欄位。

APA 7：

> Lombardo, G., Pellegrino, M., Adosoglou, G., Cagnoni, S., Pardalos, P. M., & Poggi, A. (2022). Machine learning for bankruptcy prediction in the American stock market: Dataset and benchmarks. *Future Internet, 14*(8), 244. https://doi.org/10.3390/fi14080244

### 10.2 Diabetes 130-US Hospitals

若使用 UCI 原始資料，資料集與研究論文應分別引用：

> Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

> Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International, 2014*, 781670. https://doi.org/10.1155/2014/781670

本專案的 `mortality` 與 `date` 若為後續加工欄位，不能直接說是 UCI 原始 target/time field；必須另述轉換規則並引用轉換腳本。

### 10.3 Stock SPX

目前 `stock_spx.csv` 看起來是專案衍生資料，尚未在 repository 文件中確認唯一原始論文或資料供應者。正式論文提交前至少補齊：價格來源、指數代碼、下載日期、調整價格規則、`Crash_Event` 與 `Future_Returns_20` 的明確公式。未補齊前，不應用一篇泛稱股災預測的論文冒充資料來源。

## 11. APA 第七版完整參考文獻

Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance, 23*(4), 589–609. https://doi.org/10.1111/j.1540-6261.1968.tb00843.x

Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. *Proceedings of the AAAI Conference on Artificial Intelligence, 35*(8), 6679–6687. https://doi.org/10.1609/aaai.v35i8.16826

Batista, G. E. A. P. A., Prati, R. C., & Monard, M. C. (2004). A study of the behavior of several methods for balancing machine learning training data. *ACM SIGKDD Explorations Newsletter, 6*(1), 20–29. https://doi.org/10.1145/1007730.1007735

Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis, 120*, 70–83. https://doi.org/10.1016/j.csda.2017.11.003

Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data with adaptive windowing. In *Proceedings of the 2007 SIAM International Conference on Data Mining* (pp. 443–448). SIAM. https://doi.org/10.1137/1.9781611972771.42

Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5–32. https://doi.org/10.1023/A:1010933404324

Brown, G., Pocock, A., Zhao, M.-J., & Luján, M. (2012). Conditional likelihood maximisation: A unifying framework for information theoretic feature selection. *Journal of Machine Learning Research, 13*, 27–66. https://www.jmlr.org/papers/v13/brown12a.html

Brzeziński, D., & Stefanowski, J. (2014). Reacting to different types of concept drift: The accuracy updated ensemble algorithm. *IEEE Transactions on Neural Networks and Learning Systems, 25*(1), 81–94. https://doi.org/10.1109/TNNLS.2013.2251352

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*, 273–297. https://doi.org/10.1007/BF00994018

Cruz, R. M. O., Hafemann, L. G., Sabourin, R., & Cavalcanti, G. D. C. (2020). DESlib: A dynamic ensemble selection library in Python. *Journal of Machine Learning Research, 21*(8), 1–5. https://www.jmlr.org/papers/v21/18-144.html

Cruz, R. M. O., Sabourin, R., & Cavalcanti, G. D. C. (2018). Dynamic classifier selection: Recent advances and perspectives. *Information Fusion, 41*, 195–216. https://doi.org/10.1016/j.inffus.2017.09.010

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1–30. https://www.jmlr.org/papers/v7/demsar06a.html

Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004). Learning with drift detection. In *Advances in Artificial Intelligence—SBIA 2004* (pp. 286–295). Springer. https://doi.org/10.1007/978-3-540-28645-5_29

Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys, 46*(4), Article 44. https://doi.org/10.1145/2523813

Gao, J., Fan, W., Han, J., & Yu, P. S. (2007). A general framework for mining concept-drifting data streams with skewed distributions. In *Proceedings of the 2007 SIAM International Conference on Data Mining* (pp. 3–14). SIAM. https://doi.org/10.1137/1.9781611972771.1

Gorishniy, Y., Kotelnikov, A., & Babenko, A. (2025). TabM: Advancing tabular deep learning with parameter-efficient ensembling. In *International Conference on Learning Representations*. https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html

Gorishniy, Y., Rubachev, I., Kartashev, N., Shlenskii, D., Kotelnikov, A., & Babenko, A. (2024). TabR: Tabular deep learning meets nearest neighbors. In *International Conference on Learning Representations*. https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html

Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. In *Advances in Neural Information Processing Systems* (Vol. 34). https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html

Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). Gene selection for cancer classification using support vector machines. *Machine Learning, 46*, 389–422. https://doi.org/10.1023/A:1012487302797

He, H., Bai, Y., Garcia, E. A., & Li, S. (2008). ADASYN: Adaptive synthetic sampling approach for imbalanced learning. In *2008 IEEE International Joint Conference on Neural Networks* (pp. 1322–1328). IEEE. https://doi.org/10.1109/IJCNN.2008.4633969

He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering, 21*(9), 1263–1284. https://doi.org/10.1109/TKDE.2008.239

Hollmann, N., Müller, S., Purucker, L., Krishnakumar, A., Körfer, M., Hoo, S. B., Schirrmeister, R. T., & Hutter, F. (2025). Accurate predictions on small data with a tabular foundation model. *Nature, 637*, 319–326. https://doi.org/10.1038/s41586-024-08328-6

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems* (Vol. 30). https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

Ko, A. H. R., Sabourin, R., & Britto, A. S., Jr. (2008). From dynamic classifier selection to dynamic ensemble selection. *Pattern Recognition, 41*(5), 1718–1731. https://doi.org/10.1016/j.patcog.2007.10.015

Kolter, J. Z., & Maloof, M. A. (2007). Dynamic weighted majority: An ensemble method for drifting concepts. *Journal of Machine Learning Research, 8*, 2755–2790. https://www.jmlr.org/papers/v8/kolter07a.html

Lombardo, G., Pellegrino, M., Adosoglou, G., Cagnoni, S., Pardalos, P. M., & Poggi, A. (2022). Machine learning for bankruptcy prediction in the American stock market: Dataset and benchmarks. *Future Internet, 14*(8), 244. https://doi.org/10.3390/fi14080244

Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019). Learning under concept drift: A review. *IEEE Transactions on Knowledge and Data Engineering, 31*(12), 2346–2363. https://doi.org/10.1109/TKDE.2018.2876857

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (Vol. 30). https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions

Nogueira, S., Sechidis, K., & Brown, G. (2018). On the stability of feature selection algorithms. *Journal of Machine Learning Research, 18*(174), 1–54. https://www.jmlr.org/papers/v18/17-514.html

Ohlson, J. A. (1980). Financial ratios and the probabilistic prediction of bankruptcy. *Journal of Accounting Research, 18*(1), 109–131. https://doi.org/10.2307/2490395

Oliveira, D. V. R., Cavalcanti, G. D. C., & Sabourin, R. (2017). Online pruning of base classifiers for dynamic ensemble selection. *Pattern Recognition, 72*, 44–58. https://doi.org/10.1016/j.patcog.2017.06.030

Page, E. S. (1954). Continuous inspection schemes. *Biometrika, 41*(1–2), 100–115. https://doi.org/10.1093/biomet/41.1-2.100

Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. (2025). TabICL: A tabular foundation model for in-context learning on large data. In *Proceedings of the 42nd International Conference on Machine Learning* (Vol. 267, pp. 50817–50847). PMLR. https://proceedings.mlr.press/v267/qu25d.html

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Street, W. N., & Kim, Y. (2001). A streaming ensemble algorithm (SEA) for large-scale classification. In *Proceedings of the Seventh ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 377–382). ACM. https://doi.org/10.1145/502512.502568

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records. *BioMed Research International, 2014*, 781670. https://doi.org/10.1155/2014/781670

Tomek, I. (1976). Two modifications of CNN. *IEEE Transactions on Systems, Man, and Cybernetics, SMC-6*(11), 769–772. https://doi.org/10.1109/TSMC.1976.4309452

Wang, H., Fan, W., Yu, P. S., & Han, J. (2003). Mining concept-drifting data streams using ensemble classifiers. In *Proceedings of the Ninth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 226–235). ACM. https://doi.org/10.1145/956750.956778

Wang, S., Minku, L. L., & Yao, X. (2015). Resampling-based ensemble methods for online class imbalance learning. *IEEE Transactions on Knowledge and Data Engineering, 27*(5), 1356–1368. https://doi.org/10.1109/TKDE.2014.2345380

## 12. 提交論文前的引用檢查

- [ ] 每個資料集都有原始來源、版本、下載日期與授權。
- [ ] 每個實際執行的模型都記錄 package version、參數與 random seed。
- [ ] TabICL／TabPFN 依實際 backend 分開引用。
- [ ] 自製 DAWCE／ROSS 清楚標示為本研究提出，不偽裝成外部既有方法。
- [ ] 每個方法名稱都能對應到程式、設定檔與原始文獻。
- [ ] test set 未參與特徵數、邊界、權重、threshold 或模型選擇。
- [ ] 統計檢定交代 paired unit、alternative、effect size 與多重比較處理。
- [ ] 所有 DOI 在最終排版前再以 Crossref／出版社頁核對一次。
