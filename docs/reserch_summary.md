# 研究結果完整摘要（依研究方法系統化整理）

> 撰寫日期：2026-03-03（更新：補入兩種切割方式的完整實驗結果）  
> 適合對象：剛接觸本研究的碩士生  
> **★ = 該欄位最佳值；每欄獨立標記。**

---

## 一、研究設計概覽

```
資料集 (3類)
  ├── Bankruptcy (US 1999~2018)
  ├── Medical (UCI)
  └── Stock (SPX / DJI / NDX)

切割方式 (2種)
  ├── a. Chronological：前→歷史 / 中→新營運 / 後→測試
  └── b. Block_CV：5折時序（折1,2→歷史 / 折3,4→新 / 折5→測試）

Baseline: Retrain / Finetune × none / undersampling / oversampling / hybrid
Ensemble: 6模型池（Old×3 + New×3），Soft Voting，C(6,k) 49種 + 8種命名組合
DCS/DES: OLA / LCA / _TW / KNORAE / time_weighted / minority_weighted / combined
Study II: 特徵選擇 ── no_fs / kbest_f / kbest_chi2 / lasso (r20/r50/r80)
```

---

## 二、資料集與指標

| 資料集 | 領域 | 正例比例 |
|---|---|---|
| Bankruptcy | 企業破產 (US 1999–2018) | ~3% |
| Medical | 醫療診斷 (UCI) | ~10% |
| Stock SPX/DJI/NDX | 股市下跌預測 | ~10% |

| 指標 | 方向 |
|---|---|
| AUC、F1、Recall、Precision、G-Mean | ↑ 越高越好 |
| FPR (Type1 Error)、FNR (Type2 Error) | ↓ 越低越好 |

> **切割方式命名**：本文以 **Chron** 代稱 chronological，**BCV** 代稱 block_cv。

---

## 三、Phase 1：Baseline —— Retrain vs Finetune

### 3.1 Bankruptcy

#### 3.1.1 Chronological

| 採樣 | AUC ↑ | F1 ↑ | Recall ↑ | FPR ↓ | FNR ↓ |
|---|---|---|---|---|---|
| **Retrain none** | **0.8645 ★** | **0.135 ★** | **0.812 ★** | 0.245 | 0.188 |
| Retrain under | 0.8633 | 0.134 | 0.798 | 0.242 | 0.202 |
| Retrain over | 0.7783 | 0.127 | 0.519 | **0.160 ★** | 0.481 |
| Retrain hybrid | 0.8047 | 0.134 | 0.613 | 0.181 | 0.387 |
| Finetune none | 0.8759 | **0.281 ★** | 0.516 | **0.052 ★** | 0.484 |
| **Finetune under** | **0.8794 ★** | 0.262 | 0.498 | 0.055 | 0.502 |
| Finetune over | 0.8265 | 0.184 | 0.460 | 0.085 | 0.540 |
| Finetune hybrid | 0.8552 | 0.194 | **0.568 ★** | 0.103 | **0.432 ★** |

#### 3.1.2 Block_CV

| 採樣 | AUC ↑ | F1 ↑ | Recall ↑ | Precision ↑ | FPR ↓ | FNR ↓ |
|---|---|---|---|---|---|---|
| **Retrain under** | **0.7394 ★** | 0.226 | 0.635 | 0.138 | 0.288 | 0.365 |
| Retrain none | 0.7375 | **0.235 ★** | **0.640 ★** | 0.144 | 0.276 | **0.360 ★** |
| Retrain over | 0.6821 | 0.215 | 0.378 | 0.150 | **0.155 ★** | 0.622 |
| Retrain hybrid | 0.7121 | 0.233 | 0.471 | **0.155 ★** | 0.187 | 0.529 |
| **Finetune none** | **0.6678 ★** | **0.199 ★** | **0.396 ★** | **0.133 ★** | 0.187 | **0.604 ★** |
| Finetune under | 0.6594 | 0.193 | 0.395 | 0.128 | 0.196 | 0.605 |
| Finetune over | 0.6141 | 0.158 | 0.253 | 0.115 | **0.141 ★** | 0.747 |
| Finetune hybrid | 0.6504 | 0.183 | 0.317 | 0.128 | 0.156 | 0.683 |

> - Chron vs BCV 差異顯著：Retrain AUC 0.8645 → 0.7375（−0.127）  
> - BCV 下 Retrain (0.739) 優於 Finetune (0.668)；Chron 下相反（Finetune 0.8794 最佳）

---

### 3.2 Medical

#### 3.2.1 Chronological

| 採樣 | AUC ↑ | F1 ↑ | Recall ↑ | FPR ↓ |
|---|---|---|---|---|
| **Retrain under** | **0.6658 ★** | **0.255 ★** | **0.601 ★** | 0.373 |
| Retrain none | 0.6647 | 0.253 | 0.600 | 0.370 |
| Retrain over | 0.6615 | 0.019 | 0.009 | **0.002 ★** |
| Retrain hybrid | 0.6615 | 0.167 | 0.108 | 0.023 |
| Finetune hybrid | 0.6529 | 0.183 | 0.126 | 0.028 |
| Finetune none | 0.6475 | 0.254 | 0.473 | 0.326 |
| Finetune under | 0.6439 | 0.251 | 0.472 | 0.333 |

#### 3.2.2 Block_CV

| 採樣 | AUC ↑ | F1 ↑ | Recall ↑ | FPR ↓ |
|---|---|---|---|---|
| **Retrain none** | **0.6651 ★** | 0.253 | 0.594 | 0.370 |
| Retrain under | 0.6647 | **0.256 ★** | **0.605 ★** | 0.373 |
| Retrain over | 0.6611 | 0.018 | 0.009 | **0.002 ★** |
| Retrain hybrid | 0.6621 | 0.172 | 0.112 | 0.023 |
| **Finetune hybrid** | **0.6649 ★** | 0.184 | 0.125 | 0.028 |
| Finetune none | 0.6606 | **0.259 ★** | **0.555 ★** | 0.326 |
| Finetune under | 0.6609 | 0.254 | 0.553 | 0.333 |

> - Chron Finetune AUC 整體低於 BCV（none: 0.6475 vs 0.6606，−0.013）  
> - **Recall 差距明顯**：Finetune_none Chron Recall=0.473 vs BCV=0.555（+0.082）  
> - Retrain 兩種切割差異小（Chron 0.6658 vs BCV 0.6651）

---

### 3.3 Stock SPX（S&P 500）

> ⚠️ SPX 所有方法 F1 = Recall = 0（無法識別下跌），僅以 AUC 排序。  
> 注：SPX/DJI/NDX 均無 oversampling 變體（僅 none / undersampling / hybrid）。

#### Chronological

| 採樣 | AUC ↑ | F1 ↑ | Recall ↑ | FPR ↓ |
|---|---|---|---|---|
| **Finetune hybrid** | **0.6069 ★** | 0.000 | 0.000 | 0.000 |
| Finetune none | 0.5765 | 0.000 | 0.000 | 0.000 |
| Retrain none | 0.5754 | 0.000 | 0.000 | 0.000 |
| Retrain hybrid | 0.5724 | 0.000 | 0.000 | 0.000 |
| Retrain under | 0.5567 | 0.000 | 0.000 | 0.000 |
| Finetune under | 0.5548 | 0.000 | 0.000 | 0.000 |

#### Block_CV

| 採樣 | AUC ↑ | F1 ↑ | Recall ↑ | FPR ↓ |
|---|---|---|---|---|
| **Finetune hybrid** | **0.6134 ★** | 0.000 | 0.000 | 0.000 |
| Finetune under | 0.5995 | 0.000 | 0.000 | 0.018 |
| Retrain none | 0.5980 | 0.000 | 0.000 | 0.000 |
| Finetune none | 0.5778 | 0.000 | 0.000 | 0.004 |
| Retrain under | 0.5546 | 0.000 | 0.000 | 0.000 |
| Retrain hybrid | 0.5529 | 0.000 | 0.000 | 0.000 |

> - 全部 12 種方法（兩切割各 6）F1 = Recall = 0，SPX 完全無法識別下跌  
> - BCV 略優於 Chron（Finetune_hybrid: 0.6134 vs 0.6069，差距 0.006）

---

### 3.4 Stock DJI（道瓊指數）

> ⚠️ DJI 切割方式影響最大；Chron 下多數方法 AUC < 0.5（低於隨機）。

#### Chronological

| 採樣 | AUC ↑ | F1 ↑ | Recall ↑ |
|---|---|---|---|
| **Finetune under** | **0.5387 ★** | 0.000 | 0.000 |
| Finetune hybrid | 0.5375 | 0.000 | 0.000 |
| Retrain hybrid | 0.5312 | 0.000 | 0.000 |
| Retrain none | 0.4982 | 0.000 | 0.000 |
| Retrain under | 0.4894 | 0.000 | 0.000 |
| Finetune none | 0.4801 | 0.000 | 0.000 |

#### Block_CV

| 採樣 | AUC ↑ | F1 ↑ | Recall ↑ |
|---|---|---|---|
| **Finetune none** | **0.6134 ★** | 0.000 | 0.000 |
| Finetune hybrid | 0.5845 | 0.000 | 0.000 |
| Finetune under | 0.5590 | 0.000 | 0.000 |
| Retrain none | 0.5067 | 0.000 | 0.000 |
| Retrain hybrid | 0.4866 | 0.000 | 0.000 |
| Retrain under | 0.4506 | 0.000 | 0.000 |

> - **切割方式影響最大**：BCV Finetune_none=0.6134 vs Chron=0.4801（差距 0.133！）  
> - Chron：3/6 方法 AUC < 0.5（Retrain_none=0.498、Retrain_under=0.489、Finetune_none=0.480）  
> - BCV Retrain_under=0.4506 也低於隨機（全資料集最低之一）  
> - F1 = Recall = 0 全部方法（兩切割）

---

### 3.5 Stock NDX（NASDAQ）

> NDX 是三支股市中唯一在部分方法出現正 F1/Recall 的資料集（兩種切割皆是）。

#### Chronological

| 採樣 | AUC ↑ | F1 ↑ | Recall ↑ | FPR ↓ |
|---|---|---|---|---|
| **Retrain under** | **0.5603 ★** | **0.158 ★** | 0.326 | 0.298 |
| Retrain hybrid | 0.5509 | 0.000 | 0.000 | 0.000 |
| Retrain none | 0.5236 | 0.078 | 0.084 | 0.113 |
| Finetune hybrid | 0.5473 | 0.000 | 0.000 | 0.000 |
| Finetune none | 0.5204 | 0.166 | **0.379 ★** | 0.340 |
| Finetune under | 0.5190 | 0.000 | 0.000 | 0.000 |

#### Block_CV

| 採樣 | AUC ↑ | F1 ↑ | Recall ↑ | FPR ↓ |
|---|---|---|---|---|
| **Retrain hybrid** | **0.5907 ★** | 0.000 | 0.000 | 0.015 |
| Finetune hybrid | 0.5754 | 0.000 | 0.000 | 0.000 |
| Retrain none | 0.5718 | 0.139 | 0.274 | 0.269 |
| Retrain under | 0.5616 | 0.155 | 0.390 | 0.367 |
| Finetune none | 0.5542 | **0.160 ★** | **0.421 ★** | 0.385 |
| Finetune under | 0.5494 | 0.149 | 0.379 | 0.371 |

> - BCV AUC 普遍優於 Chron（retrain_hybrid: 0.5907 vs 0.5509，+0.040）  
> - **Recall 偏好**：Chron 最高 finetune_none=0.379；BCV 最高 finetune_none=0.421  
> - BCV 多個方法有正 F1/Recall；Chron 僅 retrain_under 和 finetune_none 有值

---

### 3.6 Phase 1 小結

| 資料集 | 切割 | 最佳 AUC | 方法 | 最佳 Recall | 備註 |
|---|---|---|---|---|---|
| Bankruptcy | Chron | **0.8794** | Finetune_under | 0.812（Retrain_none）| FPR 5.5% vs 24.5% |
| Bankruptcy | BCV   | 0.7394 | Retrain_under | 0.640（Retrain_none）| Retrain > Finetune |
| Medical | Chron | 0.6658 | Retrain_under | 0.601（Retrain_under）| Finetune BCV 更好 |
| Medical | BCV   | 0.6651 | Retrain_none | 0.605（Retrain_under）| 差距小 |
| Stock SPX | Chron | 0.6069 | Finetune_hybrid | 0.000 | 無法識別下跌 |
| Stock SPX | BCV   | 0.6134 | Finetune_hybrid | 0.000 | BCV 略優 |
| Stock DJI | Chron | 0.5387 | Finetune_under | 0.000 | 多個 AUC < 0.5 |
| Stock DJI | BCV   | 0.6134 | Finetune_none | 0.000 | BCV 顯著優（+0.075）|
| Stock NDX | Chron | 0.5603 | Retrain_under | 0.379（Finetune_none）| F1/Recall 存在 |
| Stock NDX | BCV   | 0.5907 | Retrain_hybrid | 0.421（Finetune_none）| BCV 普遍優 |

---

## 四、Phase 2：集成模型組合（Ensemble Combinations）

> 模型池：Old1(under) / Old2(over) / Old3(hybrid)；New4(under) / New5(over) / New6(hybrid)  
> 集成策略：Soft Voting

### 4.1 命名組合對應

| 代號 | 說明 | 模型數 | 實驗名稱 |
|---|---|---|---|
| **a** | 1舊+1新 | 2 | `ensemble_2_old_hybrid_new_hybrid` |
| **b-TypeA** | 2舊+1新 | 3 | `ensemble_3_type_a` |
| **b-TypeB** | 1舊+2新 | 3 | `ensemble_3_type_b` |
| **c** | 2舊+2新 | 4 | `ensemble_4` |
| **d** | 2舊+3新 | 5 | `ensemble_5` |
| **e** | 3舊+3新 | 6 | `ensemble_all_6` |
| 對照舊 | 3舊 | 3 | `ensemble_old_3` |
| 對照新 | 3新 | 3 | `ensemble_new_3` |

---

### 4.2 Bankruptcy ── Chronological vs Block_CV

| 組合 | AUC_Chron ↑ | AUC_BCV ↑ | Recall_Chron ↑ | Recall_BCV ↑ |
|---|---|---|---|---|
| ensemble_old_3 | 0.8086 | 0.7286 | 0.592 | 0.458 |
| **ensemble_new_3** | **0.8693 ★** | 0.6545 | 0.519 | 0.305 |
| ensemble_all_6 | 0.8575 | 0.7150 | 0.578 | 0.385 |
| ensemble_2（a）| 0.8373 | 0.7047 | **0.603 ★** | 0.404 |
| ensemble_3_type_a（b-A）| 0.8490 | 0.7221 | 0.599 | 0.411 |
| ensemble_3_type_b（b-B）| 0.8384 | 0.6852 | 0.561 | 0.344 |
| ensemble_4（c）| 0.8492 | 0.7064 | 0.565 | 0.370 |
| ensemble_5（d）| 0.8402 | 0.7136 | 0.585 | **0.490 ★** |

> - **BCV 與 Chron 差距極大**：ensemble_new_3 從 0.8693 降至 0.6545（−0.215）  
> - BCV 下 ensemble_old_3 反而是 AUC 最高組合之一（0.7286）  
> - **選擇標準取決於切割方式**，不可混用

---

### 4.3 Medical ── Chronological vs Block_CV

| 組合 | AUC_Chron | AUC_BCV | Recall_Chron | Recall_BCV |
|---|---|---|---|---|
| ensemble_old_3 | 0.6579 | 0.6509 | **0.128 ★** | **0.132 ★** |
| **ensemble_new_3** | 0.6556 | **0.6665 ★** | 0.120 | 0.126 |
| **ensemble_all_6** | **0.6650 ★** | 0.6660 | 0.114 | 0.115 |
| ensemble_3_type_a | 0.6613 | 0.6610 | 0.126 | 0.120 |

> - Medical 兩種切割差異小（AUC ≤ 0.012），Recall 差異極小  
> - BCV 下 ensemble_new_3 最佳（0.6665 vs Chron 0.6556）

---

### 4.4 Stock SPX ── Chronological vs Block_CV

| 組合 | AUC_Chron | AUC_BCV | F1_Chron | F1_BCV | Recall_Chron | Recall_BCV |
|---|---|---|---|---|---|---|
| ensemble_old_3 | 0.5656 | 0.5160 | 0.000 | **0.126 ★** | 0.000 | **0.620 ★** |
| **ensemble_new_3** | **0.6002 ★** | **0.6043 ★** | 0.000 | 0.000 | 0.000 | 0.000 |
| ensemble_2（a）| 0.5435 | 0.5768 | 0.000 | 0.166 | 0.000 | 0.183 |
| ensemble_all_6 | 0.5655 | 0.5125 | 0.000 | 0.000 | 0.000 | 0.000 |

> - **Chron 下 F1/Recall 全為 0**（包含 ensemble_old_3），BCV 才有 Recall  
> - AUC 兩種切割差異小，ensemble_new_3 均最高（~0.60）

---

### 4.5 Stock DJI ── Chronological vs Block_CV

| 組合 | AUC_Chron | AUC_BCV | F1_Chron | F1_BCV | Recall_Chron | Recall_BCV |
|---|---|---|---|---|---|---|
| ensemble_old_3 | 0.4324 | **0.5456 ★** | 0.000 | **0.148 ★** | 0.000 | **0.554 ★** |
| **ensemble_new_3** | **0.5625 ★** | **0.5942 ★** | 0.000 | 0.000 | 0.000 | 0.000 |
| ensemble_3_type_a | 0.3975 | 0.5347 | 0.000 | 0.147 | 0.000 | 0.351 |
| ensemble_all_6 | 0.4324 | 0.5455 | 0.000 | 0.000 | 0.000 | 0.000 |

> - **DJI Chron 極差**：ensemble_old_3=0.4324、type_a=0.3975（均低於隨機）  
> - BCV 下所有組合 AUC 提升 0.10~0.16；且有 Recall

---

### 4.6 Stock NDX ── Chronological vs Block_CV

| 組合 | AUC_Chron | AUC_BCV | F1_Chron | F1_BCV | Recall_Chron | Recall_BCV |
|---|---|---|---|---|---|---|
| ensemble_old_3 | 0.5617 | 0.5490 | 0.189 | 0.167 | 0.800 | **1.000 ★** |
| ensemble_new_3 | 0.5566 | 0.5478 | 0.000 | 0.000 | 0.000 | 0.000 |
| **ensemble_2（a）** | **0.5808 ★** | **0.5912 ★** | 0.106 | **0.212 ★** | 0.095 | 0.863 |
| ensemble_3_type_a | 0.5555 | 0.5638 | 0.153 | 0.167 | **0.390 ★** | **1.000 ★** |

> - BCV 組合 a 在 AUC 與 F1 均最佳（0.5912 / 0.212），為 BCV 最均衡組合

---

### 4.7 系統性窮舉（C(6,k) 全部 49 種，各 Size 最佳）

| 資料集 | 切割 | 最佳固定組合 | 整體最佳 AUC | 備註 |
|---|---|---|---|---|
| Bankruptcy | Chron | Size 3（1舊+2新）| **0.8817** | 超過所有命名組合 |
| Bankruptcy | BCV | Size 3（2舊+1新）| 0.7363 | old_3=0.7286 次佳 |
| Medical | Chron | ensemble_all_6 | **0.6650** | 差距極小（<0.001）|
| Medical | BCV | ensemble_new_3 | **0.6665** | 差距極小 |
| Stock SPX | Chron | ensemble_new_3 | **0.6002** | 所有 F1=0 |
| Stock SPX | BCV | ensemble_new_3 | **0.6043** | combo_a 有 F1=0.166 |
| Stock DJI | Chron | ensemble_new_3 | **0.5625** | 多組 AUC＜0.5 |
| Stock DJI | BCV | ensemble_new_3 | **0.5942** | old_3 Recall 最高（0.554）|
| Stock NDX | Chron | ensemble_2（a）| **0.5808** | old_3 Recall=0.800 最高 |
| Stock NDX | BCV | ensemble_2（a）| **0.5912** | type_a Recall=1.000 |

> - 詳見 `results/phase2_ensemble/all_combinations_systematic.csv`  
> - Stock 三資料集最佳均為 ensemble_new_3 或 ensemble_2（AUC 優先）  
> - 若以 Recall 為主，Stock 偏好含 old_3 的組合

---

### 4.8 Phase 2 小結

1. **Chronological 切割**：ensemble_new_3 是最強固定組合（Bankruptcy 0.8693）
2. **Block_CV 切割**：BCV 整體 AUC 低 0.10~0.20；DJI Chron 甚至多組低於隨機
3. **Recall 偏好舊模型**：Stock 在 BCV 下 ensemble_old_3 Recall 最高，但 AUC 不如 new_3
4. **切割方式是系統行為的主因之一**，必須在相同切割基準下比較方法

---

## 五、Phase 3：動態選擇機制（DCS / DES）

> - **DCS**：每次只選 1 個分類器（OLA=Overall Local Accuracy；LCA=Local Class Accuracy；_TW=時間加權）  
> - **DES**：選出最佳子集成（KNORAE baseline；time_weighted；minority_weighted；combined）  
> ⚠️ **原始腳本均使用 Block_CV**；Chronological 結果為本次追加跑出。

---

### 5.1 Bankruptcy ── DCS

| 方法 | AUC_Chron ↑ | AUC_BCV ↑ | F1_Chron ↑ | F1_BCV ↑ | Recall_Chron ↑ | Recall_BCV ↑ |
|---|---|---|---|---|---|---|
| **DCS_OLA** | **0.8395 ★** | 0.6815 | 0.214 | 0.220 | **0.484 ★** | **0.319 ★** |
| DCS_LCA | 0.8372 | 0.6778 | **0.254 ★** | 0.173 | 0.387 | 0.161 |
| **DCS_OLA_TW** | **0.8415 ★** | 0.6774 | 0.220 | 0.209 | **0.499 ★** | 0.307 |
| DCS_LCA_TW | 0.8323 | 0.6762 | 0.235 | 0.173 | 0.429 | 0.180 |

### 5.2 Bankruptcy ── DES

| 方法 | AUC_Chron ↑ | AUC_BCV ↑ | Recall_Chron ↑ | Recall_BCV ↑ |
|---|---|---|---|---|
| DES_baseline (KNORAE)| 0.8560 | 0.7061 | **0.551 ★** | 0.377 |
| **DES_time_weighted** | **0.8626 ★** | 0.7105 | 0.530 | 0.358 |
| DES_minority_weighted | 0.8619 | **0.7138 ★** | 0.526 | **0.372 ★** |
| DES_combined | 0.8624 | 0.7126 | 0.530 | 0.366 |

> - **Chron DCS_OLA_TW=0.8415**，比 Block_CV（0.6774）高出 **0.165**；兩切割呈現完全不同數量級  
> - Chron 下 DCS 與 DES 差距縮小：DES_time_weighted（0.8626）vs DCS_OLA_TW（0.8415）= 只差 0.021  
> - BCV 下差距仍大：DES ~0.71 vs DCS ~0.68

---

### 5.3 Medical ── DCS 與 DES

| 方法 | AUC_Chron | AUC_BCV | F1_Chron | F1_BCV |
|---|---|---|---|---|
| DCS_OLA | 0.6059 | 0.6016 | 0.099 | 0.099 |
| **DCS_OLA_TW** | **0.6143 ★** | **0.6135 ★** | **0.100 ★** | **0.112 ★** |
| DES_baseline | 0.6349 | 0.6374 | 0.159 | 0.164 |
| **DES_time_weighted** | **0.6602 ★** | **0.6575 ★** | 0.136 | 0.141 |
| DES_combined | 0.6588 | 0.6567 | **0.157 ★** | **0.161 ★** |

> - Medical 兩種切割差異極小（<0.003 AUC）  
> - 所有 DCS/DES 仍低於 ensemble_new_3 BCV（0.6665）

---

### 5.4 Stock SPX ── DCS 與 DES

| 方法 | AUC_Chron | AUC_BCV | F1_BCV |
|---|---|---|---|
| DCS_OLA / OLA_TW | 0.5655 | 0.5079 | **0.109 ★** |
| **DCS_LCA / LCA_TW** | **0.5655 ★** | **0.5133 ★** | 0.000 |
| DES_baseline | 0.5655 | **0.5143 ★** | 0.000 |
| DES_time_weighted | 0.5655 | 0.4827 | 0.000 |

> - **Chron 所有方法 F1/Recall=0**，完全無鑑別力  
> - BCV DCS_OLA 有 F1=0.109，DES 全部 F1=0  
> - DES BCV 多組 AUC < 0.5，時間加權在股市有害

---

### 5.5 Stock DJI ── DCS 與 DES

| 方法 | AUC_Chron | AUC_BCV | F1_Chron | F1_BCV |
|---|---|---|---|---|
| **DCS_OLA / OLA_TW** | 0.4392 | **0.6046 ★** | 0.000 | **0.158 ★** |
| DCS_LCA / LCA_TW | 0.4473 | 0.5138 | 0.000 | 0.000 |
| DES_baseline | 0.4393 | **0.6053 ★** | 0.000 | 0.000 |
| DES_time_weighted | 0.4326 | 0.5464 | 0.000 | 0.000 |

> - **Chron 全軍覆沒**：all AUC < 0.45，低於隨機  
> - BCV DCS_OLA=0.6046 **高於 ensemble_new_3 BCV=0.5942**（+0.010）  
> - BCV DES_baseline=0.6053 **高於 ensemble_new_3**（+0.011）

---

### 5.6 Stock NDX ── DCS 與 DES

| 方法 | AUC_Chron | AUC_BCV | F1_BCV | Recall_BCV |
|---|---|---|---|---|
| DCS_OLA / OLA_TW | 0.5385 | 0.5493 | 0.149 | **0.379 ★** |
| **DCS_LCA / LCA_TW** | **0.5852 ★** | **0.5798 ★** | 0.000 | 0.000 |
| DES_baseline | 0.5386 | 0.5497 | 0.000 | 0.000 |
| **DES_minority_weighted** | 0.5282 | 0.5464 | **0.017 ★** | 0.011 |

---

### 5.7 各方法 × 兩種切割直接對比

| 資料集 | 切割 | 最佳純 Ensemble AUC | 最佳 DES AUC | DES vs Ens | 最佳 DCS AUC | DCS vs Ens |
|---|---|---|---|---|---|---|
| Bankruptcy | Chron | new_3: **0.8693** | DES_TW: **0.8626** | −0.007 | DCS_OLA_TW: **0.8415** | −0.028 |
| Bankruptcy | BCV | old_3: 0.7286 | DES_minor: 0.7138 | −0.015 | DCS_OLA: 0.6815 | −0.047 |
| Medical | Chron | all_6: 0.6650 | DES_TW: **0.6602** | −0.005 | DCS_OLA_TW: 0.6143 | −0.051 |
| Medical | BCV | new_3: **0.6665** | DES_TW: 0.6575 | −0.009 | DCS_OLA_TW: 0.6135 | −0.053 |
| Stock SPX | Chron | new_3: 0.6002 | DES_base: 0.5655 | −0.045 | DCS_LCA: 0.5655 | −0.035 |
| Stock SPX | BCV | new_3: **0.6043** | DES_base: 0.5143 | −0.090 | DCS_LCA: 0.5133 | −0.091 |
| Stock DJI | Chron | new_3: 0.5625 | DES_base: 0.4393 | −0.123 | DCS_LCA: 0.4473 | −0.115 |
| Stock DJI | BCV | new_3: **0.5942** | DES_base: **0.6053** | **+0.011** | DCS_OLA: **0.6046** | **+0.010** |
| Stock NDX | Chron | old_3: 0.5617 | DES_base: 0.5386 | −0.023 | DCS_LCA: **0.5852** | **+0.024** |
| Stock NDX | BCV | combo_a: **0.5912** | DES_base: 0.5497 | −0.042 | DCS_LCA: **0.5798** | −0.011 |

> - **Chronological 切割下 DCS 差距縮小**：Bankruptcy Chron DCS_OLA_TW=0.8415（原 BCV=0.6815 差距 0.188，現縮為 0.028）  
> - **BCV 下 DJI 是唯一 DES/DCS 同時超越 Ensemble 的案例**（+0.010~+0.011）  
> - **Chron DJI/SPX DES 全部 AUC < 0.57**，切割方式嚴重影響動態機制效能

---

### 5.8 Phase 3 小結

1. **切割方式是 DCS 表現的決定性因素**：Bankruptcy DCS Chron AUC=0.841 vs BCV=0.682（差距 0.159）
2. **Chron 下 DCS 接近 DES**（差距縮至 0.021），**BCV 下 DCS 仍顯著遜於 DES**（差距 0.032）
3. **DES 在 Bankruptcy 兩切割均不如 Finetune Baseline**（0.8626 vs 0.8794）
4. **BCV 下 Stock DJI/NDX DCS 有少數例外**超越 Ensemble，但絕對值仍低（~0.60）

---

## 六、Phase 4 & 5：特徵選擇與補充分析

### 6.1 Study II：特徵選擇對集成分類器的影響（Block_CV，ensemble_new_3 基準）

#### 6.1.1 Bankruptcy（基準 AUC=0.6545）

| 方法 | 保留比例 | AUC ↑ | F1 ↑ | Recall ↑ | AUC_Diff |
|---|---|---|---|---|---|
| no_fs（對照）| 100% | 0.6545 | 0.188 | 0.305 | — |
| kbest_f | r80 | 0.6551 | 0.198 | 0.341 | +0.001 |
| kbest_chi2 | r50 | 0.6591 | 0.191 | **0.374 ★** | +0.005 |
| **lasso** | r20/r50/r80 | **0.6656 ★** | **0.198 ★** | 0.353 | **+0.011** |

> LASSO AUC/F1 最佳；kbest_chi2 r50 Recall 最佳；整體提升有限（+0.011 max）

#### 6.1.2 Medical（基準 AUC=0.6665）

| 方法 | 保留比例 | AUC ↑ | F1 ↑ | AUC_Diff |
|---|---|---|---|---|
| no_fs（對照）| 100% | 0.6665 | 0.187 | — |
| kbest_f | r20 | 0.6142 | 0.156 | −0.052 |
| kbest_f | r50 | 0.6569 | 0.214 | −0.010 |
| **kbest_f** | **r80** | **0.6652 ★** | 0.192 | −0.001 |
| kbest_chi2 | r50 | 0.6602 | 0.210 | −0.006 |
| kbest_chi2 | r80 | 0.6620 | 0.167 | −0.005 |
| lasso | r20/r50/r80 | 0.6645 | **0.198 ★** | −0.002 |

> Medical：特徵選擇**無法提升** AUC（所有方法均低於 no_fs 基準）；保留 80% 特徵損失最小

#### 6.1.3 Stock SPX（基準 AUC=0.6043，F1=0）

| 方法 | 保留比例 | AUC ↑ | F1 ↑ | Recall ↑ | AUC_Diff |
|---|---|---|---|---|---|
| no_fs（對照）| 100% | 0.6043 | 0.000 | 0.000 | — |
| kbest_f | r20 | 0.5583 | 0.079 | **0.127 ★** | −0.046 |
| kbest_f | r80 | 0.5781 | 0.000 | 0.000 | −0.023 |
| **kbest_chi2** | **r80** | **0.5925 ★** | 0.000 | 0.000 | −0.012 |

> SPX：任何特徵選擇均使 AUC 降低；r20 反而出現 Recall=0.127（但 AUC 最差）

#### 6.1.4 Stock DJI（基準 AUC=0.5942，F1=0）

| 方法 | 保留比例 | AUC ↑ | F1 ↑ | Recall ↑ | AUC_Diff |
|---|---|---|---|---|---|
| no_fs（對照）| 100% | 0.5942 | 0.000 | 0.000 | — |
| kbest_f | r80 | 0.6317 | 0.000 | 0.000 | **+0.038** |
| **kbest_chi2** | **r80** | **0.6542 ★** | 0.000 | 0.000 | **+0.060** |
| lasso | r20/r50/r80 | 0.5609 | 0.000 | 0.000 | −0.033 |

> **DJI 是唯一特徵選擇有顯著提升的資料集**：kbest_chi2 r80 AUC=0.6542（+0.060！）

#### 6.1.5 Stock NDX（基準 AUC=0.5478，F1=0）

| 方法 | 保留比例 | AUC ↑ | F1 ↑ | Recall ↑ | AUC_Diff |
|---|---|---|---|---|---|
| no_fs（對照）| 100% | 0.5478 | 0.000 | 0.000 | — |
| **kbest_f** | **r50** | **0.5638 ★** | 0.154 | 0.400 | +0.016 |
| kbest_chi2 | r50 | 0.5583 | 0.154 | 0.389 | +0.011 |
| lasso | r20/r50/r80 | 0.5536 | **0.166 ★** | **0.411 ★** | +0.006 |

> NDX：kbest_f r50 AUC 最佳（+0.016）；lasso Recall 最高（0.411）；提升幅度小但一致

---

### 6.2 Base Learner 比較（Bankruptcy Chronological）

| 基分類器 | ensemble_new_3 AUC | ensemble_all_6 AUC |
|---|---|---|
| **LightGBM** | **0.8693 ★** | **0.8575 ★** |
| XGBoost | 0.8629 | 0.8440 |
| RandomForest | 0.8536 | 0.8414 |

---

### 6.3 新舊資料比例研究（Proportion Study，Block_CV）

> DES_baseline AUC 隨「使用新資料比例」的變化（ratio=新資料取用比例）

| 資料集 | ratio=10% | ratio=30% | ratio=50% | ratio=70% | ratio=100% |
|---|---|---|---|---|---|
| **Bankruptcy** | 0.6641 | 0.6724 | 0.6873 | 0.6944 | **0.7061 ★** |
| **Medical** | 0.6292 | 0.6323 | 0.6382 | 0.6339 | **0.6374 ★** |
| **Stock SPX** | 0.5159 | 0.5093 | 0.5163 | 0.6169 | 0.5143 |
| **Stock DJI** | 0.5284 | 0.5280 | 0.5277 | 0.4642 | **0.6053 ★** |
| **Stock NDX** | **0.5491 ★** | **0.5491 ★** | 0.5476 | 0.5495 | 0.5497 |

> - **Bankruptcy / Medical**：AUC 隨比例增加呈單調增加，「越多新資料越好」  
> - **Stock DJI / SPX**：比例 70% 出現異常下降（DJI: 0.4642），ratio=100% 反而最高  
>   → 股市非線性：部分新資料反而引入噪音，全量使用更穩定  
> - **Stock NDX**：各比例差異極小（0.5476~0.5497），對比例不敏感

---

### 6.4 切割方式比較（Split Comparison，Phase 5）

| 資料集 | 最佳_Chron | 方法_Chron | 最佳_BCV | 方法_BCV | Δ (Chron−BCV) |
|---|---|---|---|---|---|
| **Bankruptcy** | **0.8794** | finetune_under | 0.7394 | retrain_under | **−0.140** |
| Medical | 0.6658 | retrain_under | 0.6651 | retrain_none | ≈ 0 |
| Stock SPX | 0.6069 | finetune_hybrid | **0.6134** | finetune_hybrid | −0.007 |
| Stock DJI | 0.5387 | finetune_under | **0.6134** | finetune_none | **−0.075** |
| Stock NDX | 0.5603 | retrain_under | **0.5907** | retrain_hybrid | −0.030 |

> - **Bankruptcy** 切割差距最大（−0.140）；Chron 架構更能捕捉時間序列趨勢  
> - **Medical** 兩切割幾乎相同（差距 0.001），結構最穩健  
> - **DJI** 次大差距（−0.075）；BCV fold 可能涵蓋更多代表性時期  
> - **NDX** 中等差距（−0.030）：兩切割均有正 Recall，行為一致性最佳  
> - `results/phase4_analysis/medical_split_comparison.csv` / `stock_spx_split_comparison.csv` 已生成

---

## 七、整體比較（Bankruptcy Chronological 各方法排名）

| 排名 | 方法 | AUC | 所屬 Phase |
|---|---|---|---|
| 1 | combo_size3_best（1舊+2新，C(6,k)）| **0.8817** | P2 系統窮舉 |
| 2 | finetune_undersampling | 0.8794 | P1 Baseline |
| 3 | finetune_none | 0.8759 | P1 Baseline |
| 4 | ensemble_new_3 | 0.8693 | P2 Ensemble |
| 5 | retrain_none | 0.8645 | P1 Baseline |
| 6 | retrain_undersampling | 0.8633 | P1 Baseline |
| 7 | DES_time_weighted | 0.8626 | P3 DES |
| 8 | DES_combined | 0.8624 | P3 DES |
| 9 | DES_minority_weighted | 0.8619 | P3 DES |
| **10** | **DCS_OLA_TW（Chron）** | **0.8415** | P3 DCS |
| **11** | **DCS_OLA（Chron）** | **0.8395** | P3 DCS |
| 12 | ensemble_all_6 | 0.8575 | P2 Ensemble |
| 13 | DES_baseline (KNORAE) | 0.8560 | P3 DES |
| — | DCS_OLA_TW（BCV）| 0.6774 | P3 DCS BCV |

> - **DCS Chronological (0.84) 比 BCV (0.68) 高出 0.16**，切割方式改變 DCS 在排名中的位置  
> - Finetune Baseline 仍是最佳可部署方案（FPR=5.5%）

---

## 八、研究配置確認清單

| Phase | 實驗 | 資料集 | 切割方式 | 狀態 |
|---|---|---|---|---|
| P1 Retrain | × 4 採樣策略 | 全部 5 | chron + BCV | ✅ 完成（新 _chronological / _block_cv.csv）|
| P1 Finetune | × 4 採樣策略 | 全部 5 | chron + BCV | ✅ 完成 |
| P2 命名組合 a~e | 8 種 | 全部 5 | chron + BCV | ✅ 完成（新 _ensemble_results_{split}.csv）|
| P2 系統窮舉 C(6,k) | 49 種 × 2 切割 | 全部 5 | chron + BCV | ✅ 完成（490 rows）|
| P3 DCS | OLA/LCA/_TW | 全部 5 | chron + BCV | ✅ 完成（新 _dcs_comparison_{split}.csv）|
| P3 DES standard | KNORAE | 全部 5 | chron + BCV | ✅ 完成 |
| P3 DES advanced | TW/minor/combined | 全部 5 | chron + BCV | ✅ 完成（新 _des_advanced_{split}.csv）|
| P4 FS Study | 6 方法 | BK/Med/Stock | BCV | ✅ 完成 |
| P4 FS Sweep | r20/r50/r80 | BK/Med/Stock | BCV | ✅ 完成 |
| P5 Base Learner | LightGBM/XGB/RF | Bankruptcy | Chron | ✅ 完成 |
| P5 Split Comparison | Retrain+Finetune+DES | BK/SPX/Medical | chron vs BCV | ✅ 完成 |
| P5 Proportion Study | DES_base + combined | 全部 5 | BCV | ✅ 完成（新 _proportion_study.csv）|
| Multi-seed Test | BK/Med/Stock | 全部 | BCV | ✅ 完成 |

---

## 九、給碩士生的閱讀路線

```
第一步：了解研究動機
  → docs/研究方向.md

第二步：看實驗流程圖
  → docs/RE_flow.html

第三步：P1 Baseline（兩種切割）
  → results/phase1_baseline/bankruptcy_retrain_chronological.csv  （Chron）
  → results/phase1_baseline/bankruptcy_retrain_block_cv.csv       （BCV）
  → 同樣結構：medical_* / stock_spx_* / stock_dji_* / stock_ndx_*

第四步：P2 Ensemble
  → results/phase2_ensemble/bankruptcy_ensemble_results_chronological.csv
  → results/phase2_ensemble/all_combinations_systematic.csv  （49種 × 2切割）

第五步：P3 DCS / DES（注意切割差異！）
  → results/phase2_ensemble/dynamic/dcs/bankruptcy_dcs_comparison_chronological.csv  （AUC ≈ 0.84）
  → results/phase2_ensemble/dynamic/dcs/bankruptcy_dcs_comparison_block_cv.csv        （AUC ≈ 0.68）
  → results/phase2_ensemble/dynamic/des/bankruptcy_des_advanced_chronological.csv

第六步：特徵選擇
  → results/phase3_feature/bankruptcy_fs_sweep.csv

第七步：補充分析
  → results/phase4_analysis/bankruptcy_proportion_study.csv
  → results/phase4_analysis/stock_spx_split_comparison.csv

第八步：綜合比較
  → results/summary_all_datasets.csv
```
