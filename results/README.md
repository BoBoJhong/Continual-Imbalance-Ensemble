# 實驗結果目錄 (Results)

本目錄存放由 `experiments/` 或 `scripts/` 自動生成的所有實驗數據 (.csv) 與圖表 (.png)。

> ⛔ **嚴禁手動編輯**
> 這裡的所有檔案都應該是程式「**自動輸出**」的結果。若內容有誤，請修改實驗腳本並重新執行，以確保數據的**可追溯性**與**真實性**。

## 📂 目錄結構與說明

```text
results/
├── *.csv                ← 各種橫跨資料集的「綜合比較報表」
│                          (由 scripts/analysis/compare_all_results.py 自動生成)
├── phase2_ensemble/     ← Phase 2 集成：`static/`（靜態）、`dynamic/des/`（DES）、`dynamic/dcs/`（DCS）；檔名前綴 `xgb_oldnew_ensemble_static_*`／`*_des_*`／`*_dcs_*`
├── phase3_feature/      ← Phase 3 特徵選擇（Study II）輸出
├── phase4_analysis/     ← Phase 4 補充分析（比例、split、閾值等）輸出
├── phase1_baseline/     ← Phase 1 baseline（含 xgb/、torch_mlp/ 等子目錄）
├── baseline/            ← (Bankruptcy) 實驗 01 的 Baseline 數據（歷史路徑）
├── ensemble/            ← (Bankruptcy) 實驗 02 的靜態集成數據
├── des/                 ← (Bankruptcy) 實驗 03 的動態集成數據
├── feature_study/       ← (Bankruptcy) 實驗 04 的特徵選取研究數據
├── des_advanced/        ← (Bankruptcy) 實驗 11 的進階 DES 分析數據
├── proportion_study/    ← (Bankruptcy) 實驗 12 的資料比例分析數據
├── stock/               ← 實驗 05, 06, 09 (Stock) 的專屬結果 (包含 SPX, DJI, NDX 及平均)
├── medical/             ← 實驗 07, 08, 10 (Medical) 的專屬結果
├── multi_seed/          ← 執行多次隨機數種子實驗後的穩定度評估數據
└── visualizations/      ← 統計長條圖、趨勢圖與顯著性熱力圖 (.png)
```

## 📊 CSV 欄位通用格式

大多數的實驗結果 CSV 皆包含以下核心評估指標：

- **AUC-ROC (預設排序依據)**：衡量模型在不平衡資料下，分辨正負樣本的能力。
- **F1-Score**：Precision 與 Recall 的調和平均，反映模型在少數類的精確度。
- **G-Mean**：多數類 (Specificity) 與少數類 (Sensitivity) 的平衡準確率。
- **Recall**：捕捉少數類 (例如：破產企業、患病者) 的靈敏度。

## 💡 如何閱讀綜合報表

根目錄底下的 `summary_all_datasets.csv` 彙整了三大資料集在核心模型 (Fine-tune, New 3, KNORA-E) 上的表現。
若要重新產生彙總表、P-Value 矩陣與全套視覺化圖表，請執行：

```powershell
python scripts\analysis\compare_all_results.py
```
