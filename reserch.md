# 非平穩環境下類別不平衡之持續學習集成框架

**Project Name**: Continual Learning Ensemble Framework for Class Imbalance in Non-Stationary Environments

---

## 1. 專案概述 (Project Overview)

本專案旨在解決在非平穩（Non-stationary）數據環境中，類別不平衡（Class Imbalance）對預測模型造成的效能衰退問題。我們提出一種結合**動態集成選擇 (Dynamic Ensemble Selection, DES)** 與 **混合採樣 (Hybrid Sampling)** 的持續學習框架，並探討**特徵選擇（Feature Selection）**對模型穩健性的影響。

---

## 2. 資料集需求 (Data Requirements)

本實驗需支援以下三種不同領域的時間序列資料集，且需具備「類別不平衡」特性。

| 資料集名稱 | 來源 | 描述/特性 | 備註 |
|-----------|------|----------|------|
| Bankruptcy Prediction | Kaggle | 1999~2018 年之企業破產數據 | 公開數據 |
| Stock Prediction | 私有數據 (學長論文) | 股市預測數據 | 需處理資料保密與格式轉換 |
| Time Series Medical | UCI Repository | 醫療時間序列數據 | 需確認具體數據集名稱 |

---

## 3. 實驗流程與資料切割 (Experimental Design & Partitioning)

系統需支援兩種資料切割模式，以驗證模型在不同時間跨度下的適應性。

### 3.1 切割模式 A：時間序列切割 (Chronological Split)

適用於明確定義時間點的長期數據（如 Bankruptcy）。

- **Historical Data (歷史數據)**: 1999 ~ 2011 (用於訓練初始模型池)
- **New Operating Data (新營運數據)**: 2012 ~ 2014 (用於更新模型/Fine-tuning)
- **Testing Data (測試數據)**: 2015 ~ 2018 (用於最終評估)

### 3.2 切割模式 B：區塊式交叉驗證 (Block-based 5-fold CV)

將數據按時間順序切分為 5 個連續區塊 (Folds)，定義如下：

- **Historical Data**: Fold 1 + Fold 2
- **New Operating Data**: Fold 3 + Fold 4
- **Testing Data**: Fold 5

---

## 4. 模型架構與訓練 (Model Architecture & Training)

系統需建立兩個主要的模型池 (Model Pools)，共計 **6 個基分類器 (Base Classifiers)**。

### 4.1 基分類器定義 (Base Classifiers)

所有基分類器需支援以下三種採樣策略以處理不平衡問題：

- **Under-sampling**: 隨機減少多數類樣本。
- **Over-sampling**: 生成或複製少數類樣本 (如 SMOTE)。
- **Hybrid-sampling**: 結合上述兩者 (如 SMOTE + ENN)。

### 4.2 模型池配置

#### Pool A: 'Old' Models (歷史模型)

**訓練數據**: Historical Data

- **Model 1**: Historical + Under-sampling
- **Model 2**: Historical + Over-sampling
- **Model 3**: Historical + Hybrid-sampling

#### Pool B: 'New' Models (新模型)

**訓練數據**: New Operating Data

- **Model 4**: New + Under-sampling
- **Model 5**: New + Over-sampling
- **Model 6**: New + Hybrid-sampling

---

## 5. 集成策略與比較 (Ensemble Strategy & Baselines)

### 5.1 基準模型 (Baselines)

作為對照組，系統需實作以下兩種傳統持續學習方法：

- **Re-training (重訓練)**: 將 Historical 與 New Operating Data 合併，重新訓練單一模型。
- **Fine-tuning (微調)**: 使用預訓練模型，僅以 New Operating Data 進行參數微調。

### 5.2 集成組合實驗 (Ensemble Combinations)

系統需支援**動態集成選擇 (DES)** 或**動態分類器選擇 (DCS)**，並測試以下組合：

- **2 Models**: 兩兩組合 (必須包含一個 'Old' 和一個 'New')。
- **3 Models**:
  - Type A: 2 'Old' + 1 'New'
  - Type B: 1 'Old' + 2 'New'
- **4 Models**: 任意 4 個模型的最佳組合。
- **5 Models**: 任意 5 個模型的最佳組合。
- **6 Models**: 全模型集成 (Models 1~6)。

---

## 6. 研究 II：特徵選擇 (Study II: Feature Selection)

本階段旨在分析特徵維度對集成模型效能的影響。

- **控制變因**: 是否開啟特徵選擇模組。

### 流程

1. 對 Historical Data 執行特徵選擇 (e.g., Information Gain, Chi-Square, Lasso)。
2. 記錄選出的 Key Features。
3. 使用選定特徵重新執行上述 (Section 4 & 5) 的所有實驗。
4. 比較 "With Feature Selection" vs. "Without Feature Selection" 的效能差異。

---

## 7. 評估指標 (Evaluation Metrics)

由於資料具備類別不平衡特性，**禁止僅使用 Accuracy**。系統需輸出以下指標：

- **AUC-ROC**: 評估分類器排序能力。
- **F1-Score**: 綜合 Precision 與 Recall。
- **G-Mean**: 評估多數類與少數類的平衡準確率。
- **Recall (Sensitivity)**: 特別針對少數類 (如破產、患病) 的抓取能力。

---

## 8. 技術棧建議 (Technical Stack)

### 語言

- Python 3.8+

### 核心庫

- **scikit-learn**: 用於分類器與評估。
- **imbalanced-learn**: 用於 SMOTE 等採樣方法。
- **deslib**: (選用) 用於實作 Dynamic Ensemble Selection。
- **pandas / numpy**: 數據處理。

### 版控

- Git (GitHub)

---

## 9. 交付成果 (Deliverables)

1. **Source Code**: 包含資料前處理、模型訓練、集成邏輯、評估腳本。
2. **Experimental Log**: 記錄所有組合 (Baselines vs. Ensembles) 在不同資料集上的指標數據。
3. **Result Plots**: 趨勢圖 (Time Series Performance) 與 比較長條圖。
4. **Final Report**: 總結最佳的模型組合策略與特徵選擇的影響。
