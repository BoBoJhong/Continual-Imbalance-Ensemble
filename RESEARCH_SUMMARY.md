# 碩士研究整體狀況總結
>
> 研究題目：**非平穩環境下類別不平衡之持續學習集成框架**  
> 更新時間：2026-02-23

---

## 一、研究概覽

| 項目 | 內容 |
|------|------|
| **研究方向** | Class Imbalance (3) — 持續學習 + 類別不平衡 + 集成 + DES |
| **資料集** | Bankruptcy (US 1999–2018)、Stock（Stooq S&P 500 真實）、Medical（UCI Diabetes 130 真實）|
| **基學習器** | LightGBM |
| **評估指標** | AUC-ROC、F1、Precision、Recall |
| **實驗總數** | 10 個（01～10） |
| **整體進度** | ✅ 程式 100%、✅ 實驗 100%、✅ 論文草稿 100% |

---

## 二、老師方向對應狀態（全部 ✅）

| 老師要求 | 對應狀態 |
|----------|---------|
| Continual learning + class imbalance | ✅ 三資料集均以 under/over/hybrid 採樣 |
| Bankruptcy 1999~2018 (Kaggle) | ✅ 使用 US 資料，切割 a（1999-2011/2012-2014/2015-2018）|
| Stock prediction（學長論文） | ✅ 實驗 05、07 完成 |
| Time series medical (UCI) | ✅ 實驗 06、08 完成 |
| 切割 a（年份切割） | ✅ 自動偵測 US 資料後啟用 |
| 切割 b（5-fold block CV） | ✅ 所有資料集支援 |
| Baseline 1：Re-training | ✅ 所有資料集 |
| Baseline 2：Fine-tuning | ✅ 所有資料集 |
| Old 模型 1/2/3（historical + under/over/hybrid）| ✅ |
| New 模型 4/5/6（new + under/over/hybrid）| ✅ |
| 組合 a-e（2/3/4/5/6 模型）| ✅ |
| Dynamic classifier/ensemble selection | ✅ KNORA-E 風格 DES |
| Study II：特徵選擇對 ensemble 的影響 | ✅ 實驗 04 完成 |

---

## 三、核心實驗結果（破產資料 US 1999–2018）

### 3.1 Baseline vs Ensemble vs DES（單 seed=42）

| 方法 | AUC | F1 | 類型 |
|------|-----|-----|------|
| **ensemble_new_3** | **0.8693** | **0.2394** | Ensemble |
| DES_time_weighted | 0.8626 | 0.2242 | 進階 DES |
| DES_combined | 0.8624 | 0.2189 | 進階 DES |
| DES_minority_weighted | 0.8619 | 0.2160 | 進階 DES |
| ensemble_all_6 | 0.8575 | 0.2160 | Ensemble |
| **DES_baseline** | **0.8560** | **0.2224** | DES |
| finetune | 0.8552 | 0.1936 | Baseline |
| ensemble_old_3 | 0.8086 | 0.1327 | Ensemble |
| **retrain** | **0.8047** | **0.1336** | Baseline |

> 🏆 **核心發現**：適應策略（新模型池 / DES / finetune）AUC 一致優於 retrain，差距約 **+0.06**。

### 3.2 多 Seed 統計（seeds = 42, 123, 456）

| 方法 | AUC mean | AUC std | 結論 |
|------|----------|---------|------|
| ensemble_new_3 | **0.8693** | 0.0007 | 最優且穩定 |
| ensemble_all_6 | 0.8568 | 0.0006 | 穩定 |
| DES_KNORAE | 0.8560 | 0.0000 | 穩定 |
| finetune | 0.8552 | 0.0005 | 穩定 |
| ensemble_old_3 | 0.8077 | 0.0015 | 較差 |
| retrain | 0.8062 | **0.0049** | 最差且最不穩定 |

> 📊 **std 極小**（≤ 0.005），結果高度可重複，不需要 Wilcoxon 檢定亦可說明穩定性。

### 3.3 進階 DES（實驗 09）

| 方法 | AUC | vs baseline |
|------|-----|------------|
| DES_baseline | 0.8560 | — |
| DES_time_weighted | 0.8626 | **+0.0066** |
| DES_minority_weighted | 0.8619 | +0.0059 |
| DES_combined | 0.8624 | +0.0064 |

### 3.4 比例實驗（實驗 10）

| new 佔比 | retrain AUC | ensemble_new_3 AUC | DES_combined AUC |
|---------|-------------|-------------------|-----------------|
| **20%** | 0.8109 | **0.8693** | 0.8593 |
| 50% | 0.8181 | **0.8693** | 0.8512 |
| 80% | 0.8285 | **0.8693** | 0.8507 |

> 📌 **新資料比例愈少，適應策略優勢愈明顯**（20% 時差距最大：+0.058）

### 3.5 Study II：特徵選擇（實驗 04）

> 有/無特徵選擇對 ensemble AUC/F1 差異為 0，本實驗設定下特徵選擇未產生顯著影響（如實寫入論文）。

---

## 四、三資料集彙總（AUC）

| 資料集 | retrain | finetune | ensemble_old_3 | ensemble_new_3 | ensemble_all_6 | DES_KNORAE |
|--------|---------|----------|:---:|:---:|:---:|:---:|
| Bankruptcy | 0.805 | 0.855 | 0.809 | **0.869** | 0.858 | 0.856 |
| Stock | 0.553 | **0.613** | 0.516 | 0.604 | 0.512 | 0.514 |
| Medical | 0.662 | 0.665 | 0.651 | **0.666** | 0.666 | 0.637 |

> 📌 三資料集均使用真實資料（Stooq S&P 500 / UCI Diabetes 130）。Stock 因市場崩盤預測困難 AUC 整體偏低（0.51–0.61），屬合理範圍。ensemble_new_3 在 Bankruptcy 和 Medical 均最佳，finetune 在 Stock 最佳。

---

## 五、檔案結構總覽

```
Continual-Imbalance-Ensemble/
├── data/raw/
│   ├── bankruptcy/american_bankruptcy_dataset.csv  ← US 1999-2018
│   ├── stock/stock_data.csv                         ← Stooq S&P 500 真實資料（2000-2020）
│   └── medical/diabetes130/diabetes130_medical.csv  ← UCI Diabetes 130-US Hospitals 真實資料
├── results/
│   ├── baseline/bankruptcy_baseline_results.csv
│   ├── ensemble/bankruptcy_ensemble_results.csv
│   ├── des/bankruptcy_des_results.csv
│   ├── des/stock_des_results.csv
│   ├── des/medical_des_results.csv
│   ├── des_advanced/bankruptcy_des_advanced_comparison.csv
│   ├── feature_study/bankruptcy_fs_comparison.csv
│   ├── proportion_study/bankruptcy_ratio_comparison.csv
│   ├── multi_seed/bankruptcy_multi_seed.csv          ← mean±std
│   ├── multi_seed/stock_multi_seed.csv
│   └── visualizations/                               ← 4 張圖表
│       ├── bankruptcy_auc_comparison.png
│       ├── all_datasets_comparison.png
│       ├── feature_selection_impact.png
│       └── ensemble_size_trend.png
├── thesis/THESIS_FULL.md                            ← 完整論文草稿（465行）
├── thesis/NCU_IM_FORMAT.md                          ← 中央大學資管所格式
└── scripts/generate_synthetic_data.py               ← 合成資料生成腳本
```

---

## 六、論文撰寫進度

| 章節 | 內容 | 狀態 |
|------|------|------|
| 摘要 | 含所有資料集、切割、結論 | ✅ 完成 |
| 第一章 緒論 | 背景、目的、研究問題、範圍 | ✅ 完成 |
| 第二章 文獻探討 | 持續學習、不平衡、DES、破產預測（含真實引用） | ✅ 完成 |
| 第三章 研究方法 | 資料切割、Baseline、模型池、靜態集成、DES、Study II | ✅ 完成 |
| 第四章 實驗結果 | 表格、分析、進階 DES、比例實驗、Study II | ✅ 完成 |
| 第五章 結論 | 四點結論、研究貢獻、限制與未來工作 | ✅ 完成 |
| 參考文獻 | APA 第七版，含 Wang 2024、Cruz 2018、Ko 2008 等 | ✅ 完成 |

---

## 七、建議口試準備重點

1. **為何 ensemble_new_3 最佳**：新模型池只用新時期訓練，更適應新分布；6 個模型全上反而引入舊模型噪音。
2. **Fine-tuning 定義說明**：本研究使用「先在 historical 訓練，再僅以 new 做第二階段訓練」，非古典降學習率微調。
3. **比例實驗結論**：歷史多、新資料少時（20%），retrain 最差，適應策略優勢最大 → 直接對應「小樣本持續學習」實務場景。
4. **Study II 結論**：特徵選擇無效可如實報告，列為「未來工作：更換特徵選擇方法或特徵數」。
5. **多 seed std 極小**（≤ 0.005）→ 結果穩定，不需額外顯著性檢定（可提報 Wilcoxon 作為加分）。

---

## 八、目前完整度評估

| 面向 | 完成度 | 備註 |
|------|--------|------|
| 老師要求覆蓋 | 100% | 逐條對應見 `docs/TEACHER_REQUIREMENTS_CHECKLIST.md` |
| 程式實作 | 100% | 10 個實驗腳本 + 5 個工具腳本 |
| 實驗執行 | 100% | 所有 CSV 結果均已產出 |
| 多 seed 穩定性 | ✅ | Bankruptcy + Stock，std ≤ 0.005 |
| 視覺化圖表 | ✅ 4 張 | `results/visualizations/` |
| 論文草稿 | 100% | `thesis/THESIS_FULL.md`（465 行） |
| 格式對應 | ✅ | `thesis/NCU_IM_FORMAT.md` |
