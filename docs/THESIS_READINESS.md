# 碩論適用性說明

**結論：目前這些實驗可以作為你的碩論實驗部分。**

以下說明「為什麼可以」以及論文中建議寫清楚／補強的地方。

---

## 一、為什麼可以當碩論

| 項目 | 狀態 | 說明 |
|------|------|------|
| **符合老師方向** | ✅ | Class Imbalance 3 所列項目（三資料集、切割 a/b、baselines、ensemble、DES、Study II）皆有對應實作，見 **docs/TEACHER_REQUIREMENTS_CHECKLIST.md**。 |
| **實驗設計正當** | ✅ | 無測試集洩漏、切割與 baseline 定義清楚、評估僅在 test，見 **docs/EXPERIMENT_VALIDATION.md**。 |
| **有完整結果** | ✅ | Bankruptcy（US 1999–2018）、Stock、Medical 皆有 baseline / ensemble / DES 結果；Study II 有特徵選擇比較。 |
| **可重複** | ✅ | 固定 seed、有 run_multi_seed 可產 mean±std。 |

因此，**實驗設計與結果足以支撐一篇碩論的「方法」與「實驗」章節**，只要把動機、文獻、方法描述、結果分析與討論寫完整即可。

---

## 二、論文中建議寫清楚的幾點

1. **資料與切割**  
   - Bankruptcy：使用 US 1999–2018，**切割 a**（1999–2011 歷史、2012–2014 新營運、2015–2018 測試）。  
   - Stock / Medical：使用 **切割 b**（5-fold block CV：1+2 歷史、3+4 新營運、第 5 折測試）。  
   - 標準化、特徵選擇僅以 historical 擬合，再套用到 new 與 test。

2. **Baseline 定義**  
   - Re-training：合併 historical + new 訓練單一模型。  
   - Fine-tuning：先在 historical 訓練，再**僅用 new operating 資料**做第二階段訓練（sequential training on new data only）。

3. **結果解讀**  
   - **Bankruptcy**：ensemble_new_3、DES、finetune 優於 retrain；可討論「新營運期模型與動態選擇在 continual + imbalance 下的效益」。  
   - **Stock**：AUC 接近 1.0，論文中可註明「可能反映任務難度或資料特性，結果僅供參考」。  
   - **Medical**：樣本較少，可標註為「初步／探索性結果」。

4. **Study II**  
   - 在目前設定下，特徵選擇（SelectKBest）對 ensemble 表現影響不大（AUC_diff≈0），可如實寫為「本實驗中特徵選擇未顯著提升表現」。

---

## 三、若口委或老師要求可再補的項目

| 項目 | 目前狀態 | 若要補強 |
|------|----------|----------|
| **多 seed / mean±std** | 有腳本 `run_multi_seed.py`，可產 baseline 的 mean±std | 論文中若報「mean ± std」，需跑多 seed 並寫入結果表。 |
| **統計檢定** | 未內建 | 若老師要求「方法間要檢定」，可在多 seed 結果上做 Wilcoxon signed-rank test，在論文中加一小節說明。 |
| **G-mean / balanced accuracy** | 未計算 | 若口委要求，可在同一評估流程加算並列入表格。 |

這些都屬於「加分或依要求補強」，不影響「實驗能否當碩論」的結論。

---

## 四、論文章節對應（供撰寫參考）

- **第三章 研究方法**：對應 `docs/TEACHER_REQUIREMENTS_CHECKLIST.md`、`docs/EXPERIMENT_VALIDATION.md` 的設計說明（資料、切割、baselines、ensemble、DES、Study II）。  
- **第四章 實驗結果與分析**：對應 `results/summary_all_datasets.csv`、`results/bankruptcy_all_results.csv`、`results/feature_study/bankruptcy_fs_comparison.csv` 及上述「結果解讀」要點。  
- **限制與未來工作**：可寫 Stock/Medical 的解讀限制、fine-tuning 定義、以及若要做統計檢定或多指標的規劃。

---

**總結：目前實驗可作為你的碩論實驗部分；把上述幾點在論文中寫清楚，並視老師／口委要求決定是否補多 seed、統計檢定或額外指標即可。**
