# 實驗方向：可延伸的實質性新想法

**先說結論**：「突破」多半可遇不可求，但**在現有實驗上做一兩個「有辨識度的延伸」**，就有機會變成**實質貢獻**或**後續論文**。下面是可以具體做、且和你的方向一致的新想法。

---

## 一、你現在已經有的優勢

- **Continual learning + class imbalance**：同時考慮非平穩與不平衡，題目本身就有辨識度。
- **三資料集 + 切割 a/b + baselines + ensemble + DES + Study II**：架構完整，容易在上面「加一層」變成新貢獻。
- **結果已有故事**：適應新資料（finetune / 新模型 / DES）優於 retrain，可當 baseline 對比。

---

## 二、可做成「實質性新想法」的幾個方向

### 1. **時間／漂移感知的 DES（最推薦）**

**現狀**：KNORA-E 用 k-NN 找鄰居、看誰預測對，沒有區分「歷史 vs 新營運」的時序。

**新想法**：  
- **Time-weighted DES**：在 DSEL 裡對「新營運期樣本」給較高權重（或只取新營運期鄰居），讓選擇更偏向適應新分布。  
- **Drift-aware selection**：先估計 test 樣本較像 historical 還是 new（例如用簡單的 domain classifier 或距離到 hist/new 質心），再決定多用 Old 還是 New 模型。

**為什麼有實質性**：  
直接針對「continual = 分布會變」設計選擇策略，而不是用靜態的 KNORA-E，容易寫成「我們提出時間／漂移感知的 DES，實驗顯示在 … 優於 baseline DES」。

**實作提示**：在 `common_des.py` 的鄰居權重或候選模型篩選時，加入「樣本來自 hist 或 new」或「到 hist/new 質心距離」的權重。

---

### 2. **何時用靜態 ensemble、何時用 DES（決策層新想法）**

**現狀**：要嘛全用靜態 ensemble，要嘛全用 DES，沒有「依樣本或依時期選擇策略」。

**新想法**：  
- **Selective strategy**：對每個 test 樣本，先估計「不確定性」或「歷史/新分布相似度」；  
  - 不確定性低 → 用靜態 ensemble（省算力、穩定）；  
  - 不確定性高或像新分布 → 用 DES（適應性強）。  
- 或依**時期**：前段 test 用靜態、後段 test 用 DES，看能否提升整體。

**為什麼有實質性**：  
把「靜態 vs 動態」當成可學習或可決策的一步，而不是兩套獨立實驗，貢獻是「選擇策略」本身。

**實作提示**：用模型預測的 entropy 或 confidence 當不確定性；或用 k-NN 到 hist/new 的距離比當「像新分布」的 proxy。

---

### 3. **不平衡 + 新時期：少數類導向的評估與選擇**

**現狀**：DES 用「鄰居預測對錯」選模型，沒有顯式針對「少數類（破產）」。

**新想法**：  
- **Minority-focused DES**：在 DSEL 階段，只算或少數類樣本上的正確率來選模型；或對少數類鄰居權重加權。  
- **Cost-sensitive 或 threshold**：在 test 上做 threshold 搜尋（或 cost-sensitive 權重），讓 recall@少數類 或 F1 變目標，再比較各方法。

**為什麼有實質性**：  
class imbalance 是你的設定之一，顯式把「少數類表現」納入選擇或評估，就是一個清楚的貢獻點。

**實作提示**：在 `run_des` 裡，計算 dsel 正確率時只取 `y_dsel == 1` 的樣本，或對少數類權重乘上 class weight。

---

### 4. **特徵穩定性與 Study II 延伸（從「有無 FS」到「哪些特徵有用」）**

**現狀**：Study II 做「有/無特徵選擇」對 ensemble 的影響，結果是影響不大。

**新想法**：  
- **Stable vs shifting features**：用 historical 和 new 兩段資料算特徵重要性（例如 LightGBM feature_importances_），比較「兩段都重要」vs「只在某段重要」；  
  - 若某些特徵在 new 才重要，可設計「新時期專用特徵子集」或新時期專用模型，再與舊模型 ensemble。  
- **Feature-wise ensemble**：不同子集訓練不同模型，DES 時不只選「哪個模型」，還隱含「哪組特徵」在該鄰域有效。

**為什麼有實質性**：  
從「特徵選擇有沒用」升級到「哪些特徵在 continual 下穩定、哪些在變」，直接呼應 non-stationary，容易寫成一個小節或一篇短文。

**實作提示**：在 01/02 訓練後取 `model.model.feature_importances_`，對 hist 與 new 分別算，做差異分析或可視化。

---

### 5. **理論/實證：何時「舊+新」ensemble 會贏過 retrain**

**現狀**：你已經有實證「ensemble_new_3 / DES 優於 retrain」，但沒有系統性變因分析。

**新想法**：  
- **Controlled experiments**：固定總樣本數，改變「historical vs new 比例」（例如 8:2、5:5、2:8），看何時 ensemble/DES 領先 retrain 最多、何時差距縮小。  
- **簡短討論**：在論文中加一小節「為何適應策略在我們設定下有效」，用「分布偏移」「少數類在新時期更少」等直覺解釋，再配上上述比例實驗。

**為什麼有實質性**：  
給出「在什麼條件下你的方法特別有用」，比只報告數字更有「貢獻感」，也方便口委問「為什麼有效」。

**實作提示**：在 `common_bankruptcy.py` 或資料切完後，用 subsample 調整 hist/new 比例，再跑 01/02/03 比較。

---

## 三、怎麼選、怎麼寫進碩論

| 若你… | 建議 |
|--------|------|
| **只想穩穩畢業** | 現有實驗 + 清楚寫出「適應策略優於 retrain」即可；新想法可列在 **未來工作**（例如時間加權 DES、minority-focused DES）。 |
| **想有一點辨識度** | 挑 **1 或 2** 做一個小實驗（例如 time-weighted DES 或 minority-focused 選擇），在論文中多一節「延伸方法」+ 一組對比表。 |
| **想衝後續論文** | 以 **1 + 3** 或 **1 + 2** 為主軸，把「時間/漂移感知 + 不平衡導向」做成一個完整方法，再補比例實驗（5）當分析。 |

---

## 四、一句話總結

- **「實質性突破」**：多半不是單一點子，而是**在現有方向上加一個「有辨識度的設計」**（例如時間加權 DES、少數類導向選擇、或靜態/DES 決策）。  
- **你現在的方向已經有空間**：continual + imbalance + ensemble + DES，只要選一個延伸做深、寫清楚「問題→做法→實驗→討論」，就有機會變成實質貢獻或後續論文。  
- 若時間有限，**優先考慮「時間／漂移感知的 DES」**（方向 1）：和你的主軸最貼合、實作範圍可控、故事也最好講。

---

## 五、已實作（後續論文用）

以下已完整實作，可直接跑實驗與寫進論文／後續投稿：

| 項目 | 腳本 | 結果位置 |
|------|------|----------|
| **進階 DES**（時間加權 + 少數類加權 + combined） | `experiments/09_bankruptcy_des_advanced.py` | `results/des_advanced/bankruptcy_des_advanced_comparison.csv` |
| **比例實驗**（hist vs new 20% / 50% / 80%） | `experiments/10_bankruptcy_proportion_study.py` | `results/proportion_study/bankruptcy_ratio_comparison.csv` |

- **共用模組**：`experiments/common_des_advanced.py`（`run_des_advanced(..., time_weight_new=, minority_weight=)`）。
- **一鍵執行**：`python scripts/run_all_experiments.py` 會依序跑 01～10（含 09、10）。
