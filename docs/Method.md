# 方法筆記 Method Notes

> 本文件用於記錄研究中使用或參考的機器學習／深度學習方法，方便後續補充與查閱。

---

## TabNet

### 論文出處
- **Arik, S. O., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning.**
  *Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679–6687.*
  - arXiv: https://arxiv.org/abs/1908.07442
  - 發布單位：Google Cloud AI

### 核心概念
TabNet 是專為**結構化表格資料**設計的深度學習架構，主要創新在於引入 **Sequential Attention 機制**：

- 模型在每個決策步驟（step）只關注部分特徵（sparse feature selection）
- 使用 **Sparsemax** 而非 Softmax，使注意力更集中（大多數特徵的權重被推為 0）
- 每一步的特徵選擇結果可累加，形成可解釋的特徵重要性輸出

### 架構流程
```
輸入特徵 X
  → BN（Batch Normalization）
  → Step 1: 用 Prior Scale 控制哪些特徵還未被用過
       → Attention Transformer（選特徵）
       → Feature Transformer（轉換特徵）
       → 產生該步輸出 h_1
  → Step 2 ... Step N（重複上述，逐步累積 attention）
  → 最終輸出 = sum(h_1 ... h_N) → 分類/回歸頭
```

### 特點與優勢
| 特性 | 說明 |
|------|------|
| 無需特徵預處理 | 不需歸一化、不需手動特徵選擇 |
| 可解釋性 | 每步 attention mask 可視覺化哪些特徵被使用 |
| 增量訓練友好 | 支援 mini-batch SGD，可接 continual learning 框架 |
| 自監督預訓練 | 可用無標籤資料做 masked feature reconstruction 預訓練 |
| 結構化資料 SOTA | 在多個 UCI 與金融 benchmark 上超越 XGBoost 與 MLP |

### 與本研究的關聯
- 可直接替換 MLP，保留現有 sampling + year split 實驗框架
- 特徵重要性輸出有助於解釋哪些財務指標對破產預測最關鍵
- 若搭配 Focal Loss，可進一步改善 imbalanced 資料下的 Recall

### 實作資源
- PyTorch 實作（pytorch-tabnet）：https://github.com/dreamquark-ai/tabnet
  ```bash
  pip install pytorch-tabnet
  ```
- 核心類別：`TabNetClassifier`，API 設計類似 scikit-learn

### 個人補充筆記
> （請在此處填入你的理解、實驗觀察或延伸閱讀）

---

## FT-Transformer（Feature Tokenizer + Transformer）

### 論文出處
- **Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data.**
  *Advances in Neural Information Processing Systems (NeurIPS), 34, 18932–18943.*
  - arXiv: https://arxiv.org/abs/2106.11959
  - 發布單位：Yandex Research

### 核心概念
FT-Transformer 將 Transformer 架構應用於表格資料，核心思想是：**把每一個特徵（feature）當成一個 token**，再送入標準的 Transformer Encoder 做 self-attention。

#### Feature Tokenizer
- **數值特徵**：每個特徵 $x_j$ 乘上一個可學習的 embedding 向量 $W_j$，再加偏差 $b_j$
  $$T_j = x_j \cdot W_j + b_j, \quad W_j \in \mathbb{R}^d$$
- **類別特徵**：標準 lookup embedding（同 NLP 的 word embedding）
- 加上一個 `[CLS]` token 作為最終分類輸出

#### Transformer Encoder
- 輸入：$n_{features} + 1$ 個 token（含 CLS）
- 標準多頭自注意力（Multi-Head Self-Attention）+ FFN + LayerNorm
- 最終取 `[CLS]` token 的輸出做分類

### 架構示意
```
特徵 1 → Tokenizer → token_1 ┐
特徵 2 → Tokenizer → token_2 ├→ [token_1, token_2, ..., token_n, CLS]
...                           │         ↓
特徵 n → Tokenizer → token_n ┘   Transformer Encoder (L layers)
                                        ↓
                                 取 CLS 輸出 → Linear → 預測
```

### 特點與優勢
| 特性 | 說明 |
|------|------|
| 特徵交互建模 | Self-attention 可捕捉任意兩個特徵間的交互（XGBoost 靠 split、MLP 靠全連接） |
| 結構簡潔 | 直接沿用 NLP 的 Transformer，無需特殊設計 |
| 實驗驗證強 | 論文在 11 個表格 benchmark 上系統性比較，宣稱達到 tabular SOTA |
| 預訓練潛力 | 可用 BERT-style masked token prediction 做自監督預訓練 |

### 與 TabNet 的主要差異
| | TabNet | FT-Transformer |
|---|---|---|
| 特徵選擇方式 | Sequential attention（稀疏） | Dense self-attention（每個特徵都互相看） |
| 可解釋性 | 高（有稀疏 mask） | 中（attention map 較難解讀） |
| 訓練速度 | 較快 | 較慢（參數量大） |
| 小樣本表現 | 較好 | 需要較多資料 |
| 破產預測適用性 | 高（樣本量中等） | 中高（若樣本夠多） |

### 與本研究的關聯
- 適合探索財務特徵之間的非線性交互（例如：負債比率與現金流的聯合效果）
- 論文提供的 benchmark 包含金融類型資料集，有直接可比較的 baseline
- 建議在 phase4_feature 中作為 feature interaction 分析的方法之一

### 實作資源
- 官方實作（論文作者）：https://github.com/yandex-research/rtdl
  ```bash
  pip install rtdl
  ```
- 另有整合版：https://github.com/lucidrains/tab-transformer-pytorch

### 個人補充筆記
> （請在此處填入你的理解、實驗觀察或延伸閱讀）

---

## 延伸閱讀建議

| 方法 | 論文 | 重點 |
|------|------|------|
| NODE | Popov et al. (2019) arXiv:1909.06312 | 可微分 Decision Tree Ensemble |
| SAINT | Somepalli et al. (2021) arXiv:2106.01342 | 同時做 row-attention + column-attention |
| TabPFN | Hollmann et al. (2022) arXiv:2207.01848 | 用 meta-learning 的 in-context learning 做表格分類，推理極快 |
| AutoInt | Song et al. (2019) arXiv:1810.11921 | 用 multi-head self-attention 做特徵交互，來自推薦系統 |

---

## 本專案 `src/models` 實作清單

以下為 `src/models/` 資料夾中已實作的所有模型 wrapper，均提供統一介面（`fit` / `predict` / `predict_proba`），可搭配 `ImbalanceSampler` 與 `ModelPool` 使用。

---

## LightGBM（`lightgbm_wrapper.py`）

### 實作類別
`LightGBMWrapper`

### 方法出處
- **Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.**
  *Advances in Neural Information Processing Systems (NeurIPS), 30.*
  - 發布單位：Microsoft Research

### 核心概念
LightGBM 是基於 **Gradient Boosting Decision Tree（GBDT）** 的高效能實作，主要創新在於：

- **Histogram-based split finding**：將連續特徵離散成直方圖 bin，大幅降低記憶體與計算成本
- **Leaf-wise tree growth**（vs. level-wise）：每次展開損失減少最多的葉節點，在相同迭代次數下通常比 XGBoost 更深且準確
- **GOSS（Gradient-based One-Side Sampling）**：保留梯度大的樣本，對梯度小的樣本隨機抽樣，降低訓練成本
- **EFB（Exclusive Feature Bundling）**：將互斥的稀疏特徵合併，減少有效特徵數量

### 不平衡處理
- 設定 `is_unbalance=True` 或 `scale_pos_weight`，自動對少數類別提高權重
- 搭配 `ImbalanceSampler` 做前置採樣（undersampling / oversampling / hybrid）

### 本專案實作重點
| 參數 | 說明 |
|------|------|
| `use_imbalance` | 是否套用 `model_config.yaml` 的不平衡參數 |
| `fit(X_train, y_train, X_val, y_val)` | 支援驗證集早停 |
| `get_feature_importance()` | 回傳 DataFrame，包含特徵名稱與重要性分數 |

---

## XGBoost（`xgboost_wrapper.py`）

### 實作類別
`XGBoostWrapper`

### 方法出處
- **Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.**
  *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.*
  - arXiv: https://arxiv.org/abs/1603.02754
  - 發布單位：University of Washington

### 核心概念
XGBoost 是最廣泛使用的 GBDT 框架，以系統化的正則化與工程優化著稱：

- **二階泰勒展開**：同時利用梯度（一階）與 Hessian（二階）資訊進行最佳化，更新方向更精確
- **正則化（L1 + L2）**：在目標函數中加入葉子節點數與葉子分數的懲罰項，避免過擬合
- **Level-wise tree growth**：每一層同時展開所有節點，行為較穩定、對超參數不敏感
- **Column subsampling**：每次 boosting 輪次僅使用部分特徵，增加多樣性

### 不平衡處理
- `scale_pos_weight`：設為 `"auto"` 時由程式自動計算 `負類數 / 正類數`
- 搭配 `ImbalanceSampler` 做前置採樣

### 本專案實作重點
| 參數 | 說明 |
|------|------|
| `use_imbalance` | 是否套用不平衡參數 |
| `_calculate_scale_pos_weight(y)` | 自動計算類別權重比例 |
| `get_feature_importance()` | 回傳特徵重要性 DataFrame |

---

## Random Forest（`random_forest_wrapper.py`）

### 實作類別
`RandomForestWrapper`

### 方法出處
- **Breiman, L. (2001). Random Forests.**
  *Machine Learning, 45(1), 5–32.*
  - 發布單位：University of California, Berkeley

### 核心概念
Random Forest 是 **Bagging** 框架搭配隨機特徵選擇的整體學習法：

- **Bootstrap sampling**：每棵決策樹從訓練集有放回地採樣，形成各自的訓練子集
- **Random feature subspace**：每次節點分裂時，只從隨機選取的 $\sqrt{p}$ 個特徵中尋找最佳分割點（$p$ 為總特徵數），提升多樣性並降低樹間相關性
- **多數決投票**：最終預測由所有樹的輸出取多數決（分類）或平均（回歸）
- **Out-of-bag（OOB）估計**：未被 bootstrap 採到的樣本可作為驗證集，無需額外 hold-out

### 不平衡處理
- `class_weight="balanced"`：sklearn 會依類別頻率反比自動調整各類別的損失權重

### 本專案實作重點
| 參數 | 說明 |
|------|------|
| `n_estimators` | 決策樹棵數（預設 200） |
| `class_weight` | 預設 `"balanced"`，自動處理類別不平衡 |
| `get_feature_importances()` | 回傳 numpy array，為各棵樹 impurity decrease 的平均值 |
| `predict_proba(X)` | 回傳正類機率（shape: `(n,)`），介面與 XGBoostWrapper 一致 |

---

## MLP（`mlp_wrapper.py`）

### 實作類別
`MLPWrapper`

### 方法出處
- **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors.**
  *Nature, 323(6088), 533–536.*
- 實作基於 scikit-learn 的 `MLPClassifier`

### 核心概念
多層感知器（Multi-Layer Perceptron）是最基礎的前饋全連接神經網路：

- 每一層為線性變換 + 非線性激活函數（此處使用 ReLU）
- 透過反向傳播與 Adam 優化器更新權重
- 本專案使用固定架構：`Input → 128 → 64 → 32 → Sigmoid`

### 本專案架構
```
Input
  → Linear(128) + ReLU
  → Linear(64)  + ReLU
  → Linear(32)  + ReLU
  → Linear(1)   + Sigmoid
```

### 不平衡處理
- **不**在模型層設定 `class_weight`（避免與前置採樣雙重處理）
- 完全依賴 `ImbalanceSampler` 在訓練前平衡資料分佈
- 使用 `early_stopping=True` 防止過擬合

### 本專案實作重點
| 參數 | 說明 |
|------|------|
| `hidden_layer_sizes` | 預設 `(128, 64, 32)` |
| `max_iter` | 預設 300 epochs |
| `early_stopping` | 預設開啟，以 10% 資料作 validation |
| `predict_proba(X)` | 回傳正類機率（shape: `(n,)`） |

---

## TabNetWrapper（`tabnet_wrapper.py`）

> TabNet 的理論說明請見本文件上方 [TabNet](#tabnet) 章節，此處補充本專案的實作細節。

### 實作類別
`TabNetWrapper`

### 本專案架構與超參數
```
TabNetClassifier(
    n_d=32, n_a=32,    # 決策步驟特徵寬度
    n_steps=5,          # Sequential Attention 步驟數
    gamma=1.3,          # 先前步驟特徵的抑制係數
    lambda_sparse=1e-3, # 稀疏正則化強度
    max_epochs=200,
    patience=20         # early stopping
)
```

### 不平衡處理
- 主要依賴 `ImbalanceSampler` 前置採樣
- 可選傳入 `class_weight` 在損失函數層加權

### 本專案實作重點
| 參數 | 說明 |
|------|------|
| `n_d`, `n_a` | 決策步驟特徵維度（值越大模型越複雜） |
| `n_steps` | Attention 步驟數，影響特徵選擇的精細程度 |
| `lambda_sparse` | 稀疏正則化，值越大特徵選擇越集中 |
| `predict_proba(X)` | 回傳正類機率（shape: `(n,)`） |
| 安裝需求 | `pip install pytorch-tabnet` |

---

## ModelPool（`model_pool.py`）

### 實作類別
`ModelPool`

### 設計概念
`ModelPool` 為**本專案的 Ensemble 基礎架構**，負責管理多個基學習器（base learner）的建立、訓練與查詢，是 DCS／DES 動態選擇系統的上游模組。

核心思想：**以不同採樣策略訓練多個模型，形成具多樣性的模型池**，再交由後續動態選擇方法選出最適合當前樣本的模型。

### 模型池組成
每次呼叫 `create_pool()` 預設建立三個模型：

| 模型代號 | 採樣策略 |
|----------|----------|
| `{prefix}_under` | Undersampling（欠採樣） |
| `{prefix}_over` | Oversampling（過採樣，SMOTE） |
| `{prefix}_hybrid` | Hybrid（混合採樣） |

### 主要方法
| 方法 | 說明 |
|------|------|
| `create_pool(X_train, y_train, prefix)` | 建立完整三模型池 |
| `create_model_with_sampling(...)` | 建立單一特定採樣策略的模型 |
| `get_model(model_name)` | 以名稱取得已訓練模型 |

### 與整體框架的關聯
```
資料（年份切分） → ImbalanceSampler
                   ↓
              ModelPool（old / new）
                   ↓
          DCS / DES 動態選擇
                   ↓
               最終預測
```

---

*最後更新：2026-03-13*
