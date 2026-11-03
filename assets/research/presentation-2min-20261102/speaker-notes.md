# 2-Minute Individual Presentation

版本：2026-11-03 發布修訂。封面與四頁內容均為 16:9 PNG；第 2、3、4 頁以內建 image_gen 編輯並逐頁檢視。保留一份企業資料的事件代理分類案例，依最新論文交代最佳模型框架與目前測試完成範圍。

## Slide 1 - Introduction (0:00-0:30)

My research examines classification when data change over time and the important class is rare. Historical data provide more examples, but may be less relevant to later observations. Recent data may be more relevant, yet contain fewer positive cases. This creates a practical question: how should we combine historical and recent models while maintaining minority-class performance?

中文提示：時間變化、少數類別與歷史／近期資訊的取捨。

## Slide 2 - Research Question (0:30-1:00)

I ask three questions. How do historical and recent models differ? How do static ensembles compare with dynamic selection? Can feature selection reduce the input size while preserving performance? The current case study uses annual corporate records and a reconstructed bankruptcy-event proxy. It evaluates a two-stage model-pool update under shared comparison rules.

中文提示：依序說明期間、集成與特徵；事件時間是回溯重建代理。

## Slide 3 - Methodology (1:00-1:30)

The framework selects a base classifier using validation data, then builds three historical and three recent models through different sampling methods. Current ensemble results use an XGBoost pilot. I compare static and dynamic strategies, with and without feature selection. Ensemble choices and thresholds use validation only. Test evaluation reports ROC-AUC, F1 and recall across ten seeds.

中文提示：正式基礎分類器待共同驗證選定；目前是預設 XGBoost 流程測試。FS 只在 Old-fit 擬合、在採樣前執行。DSEL 與 Calibration 互斥。

## Slide 4 - Contributions (1:30-2:00)

The contribution is a reproducible comparison of training periods, ensemble strategies and feature reduction. The current evidence comes from one corporate dataset and one model-pool update. It helps assess whether historical models remain useful under the specified conditions. Next, I will complete shared classifier validation and rebuild the selected model pool. Broader generalization still requires further evidence.

中文提示：貢獻是共同流程與有界線的實證；不宣稱跨資料普遍優越性或長期多輪適應已完成。

## Claim ledger

| ID | 主張 | 來源 | 界線 |
|---|---|---|---|
| C1 | 三個研究問題與單一企業案例 | thesis/latex/chapters/01_introduction.tex；03_method.tex | 事件標籤是代理，非真實申請日期驗證 |
| C2 | 六模型池與共同驗證 | 03_method.tex 的模型池與選擇程序 | 正式分類器選擇仍待完成，目前為 XGBoost 測試 |
| C3 | 採樣前的 Old-only FS、互斥 DSEL／Calibration | 03_method.tex 的共同設定與實驗二 | 未執行多輪線上更新 |
| C4 | 十種子、AUC／F1／Recall | 03_method.tex；04_experiments.tex | 種子標準差不是獨立樣本信賴區間 |
| C5 | 比較流程與推論界線 | 01_introduction.tex；05_conclusion.tex | 不主張新演算法或跨資料普遍優越性 |

## 發布核對

舊版簡報僅保留於本地發布前備份。本資料夾圖片、講稿與兩個 ZIP 同步更新；ZIP 內圖片雜湊應與本資料夾一致。
