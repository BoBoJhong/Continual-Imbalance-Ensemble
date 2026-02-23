# 實驗 09、10 效果解讀

---

## 一、實驗 09（進階 DES 比較）— **有效果，可寫進論文**

### 你目前的數字（US 1999–2018）

| 方法 | AUC | F1 | 解讀 |
|------|-----|-----|------|
| DES_baseline（KNORA-E） | 0.8560 | 0.222 | 對照組 |
| **DES_time_weighted**（新營運期權重 2.0） | **0.8626** | 0.224 | 略優於 baseline |
| DES_minority_weighted（少數類權重 2.0） | 0.8619 | 0.216 | 略優於 baseline |
| **DES_combined**（時間+少數類皆 2.0） | **0.8624** | 0.219 | 略優於 baseline |

### 效果結論

- **時間加權**、**少數類加權**、**combined** 都比 baseline DES 好一點：AUC 約 **+0.006～+0.007**（0.856 → 0.862）。
- **最佳**：DES_time_weighted（AUC 0.8626），combined 次之（0.8624）。
- 論文可寫：「在 continual + class imbalance 下，對 DSEL 做時間加權或 combined（時間+少數類）可略為提升 DES 的 AUC（約 +0.7%），支持『重視新營運期／少數類樣本』的選擇策略。」

**一句話**：09 有明顯、一致的小幅提升，效果可寫進碩論或後續論文。

---

## 二、實驗 10（比例實驗）— **已跑完，三種比例效果有分化**

### 你目前的數字（修正比例後重跑）

| 比例（new 佔訓練集） | retrain AUC | ensemble_new_3 AUC | DES_baseline AUC | DES_combined AUC |
|---------------------|-------------|--------------------|------------------|------------------|
| **20% new**         | 0.811       | **0.869**          | 0.850            | 0.859            |
| **50% new**         | 0.818       | **0.869**          | 0.839            | 0.851            |
| **80% new**         | 0.829       | **0.869**          | 0.847            | 0.851            |

### 效果解讀

1. **ensemble_new_3（只用新模型）**  
   - 三種比例下都是 **0.869**，表現最好且穩定。  
   - 因為 New 池一律用「全部 new」訓練，所以比例只影響 hist 大小，不影響 New 池，數字不變是預期的。

2. **retrain**  
   - 隨「new 比例」變高而變好：0.811（20%）→ 0.818（50%）→ 0.829（80%）。  
   - 新資料佔比愈高，合併重訓略有好處，但三種比例下都仍**低於** ensemble_new_3（0.869）。

3. **DES（baseline / combined）**  
   - 介於 retrain 與 ensemble_new_3 之間；DES_combined 多數略優於 DES_baseline。  
   - 20% new 時 DES_combined（0.859）明顯優於 retrain（0.811），顯示在「歷史多、新資料少」時，適應策略（DES／ensemble）比單純 retrain 有優勢。

4. **論文可寫的結論**  
   - 「在各種 historical vs new 比例下，ensemble_new_3 皆優於 retrain；當 new 僅佔 20% 時，retrain 最差（0.811），DES_combined 與 ensemble_new_3 可提升至 0.859、0.869，顯示在 continual 設定下適應策略（新模型池／動態選擇）優於合併重訓。」

### 可做的後續

- 用 **`results/proportion_study/bankruptcy_ratio_comparison.csv`** 畫「比例 vs AUC」折線圖（四條線：retrain、ensemble_new_3、DES_baseline、DES_combined），放進論文或簡報。

---

## 三、一句話總結

| 實驗 | 效果 | 建議 |
|------|------|------|
| **09 進階 DES** | 有：時間加權與 combined 略優於 baseline（AUC +0.006～0.007） | 可直接寫進論文 |
| **10 比例實驗** | 有：三種比例數字已分化；ensemble_new_3 最優且穩定，retrain 隨 new 比例升高略升但仍較差；20% new 時適應策略優勢最大 | 可畫「比例 vs AUC」圖、寫進論文 |
