# 共用模組導航

本目錄提供共用 API；實驗協定與執行入口在 [experiments](../experiments/README.md)。模組存在不表示所有歷史入口使用同一套流程。

| 模組 | 內容 | 使用時注意 |
|---|---|---|
| [data](data/) | loader、splitter、preprocessor、sampler | chronological 與 block_cv 依實際資料欄位／列序解讀；block_cv 本身不保證時間順序 |
| [models](models/) | LR、RF、XGBoost、LightGBM、MLP、SVM 與選用深度學習 wrapper、model_pool | GPU／外部權重模型有額外環境需求；使用 run 實際超參數 |
| [ensemble/standard_dynamic.py](ensemble/standard_dynamic.py) | 官方 DESlib KNORA-E／KNORA-U／OLA／LCA 的適配與入口 | 最新固定年份實驗使用這一模組；DSEL 與 Calibration 分開 |
| [ensemble/selector.py](ensemble/selector.py) | 舊自訂 DES／DCS、EnsembleCombiner | src.ensemble 匯出的舊 selector 不等於上述官方標準實作 |
| [features/selector.py](features/selector.py) | kbest_f／chi2、LASSO、MI、SHAP、RFE、CART、GA 等共用選擇器 | 核心新 FS runner 使用自己的 fit-only 流程與共用函式，不能從此模組推論全部實驗的 fit 範圍 |
| [evaluation/metrics.py](evaluation/metrics.py) | 共用分類指標 | PR-AUC 等指標亦可能由入口額外計算；核對正類、門檻、pooled／年度口徑 |
| [utils](utils/) | 設定載入、logger、seed | seed 有助固定隨機程序；重現仍需資料、程式、參數與環境一致 |

目前固定年份與完整 Study 3B 的 under／over／hybrid 為 TomekLinks／ADASYN／SMOTEENN，以 manifest 的實際記錄為準。前處理、FS、採樣和模型訓練的可用資料由各入口協定控制。

## 現行共用邏輯

固定年份與完整 Study 3B 的共用模型／FS、採樣標記及 seed 已移到 [common_modeling.py](../experiments/_shared/common_modeling.py)，原子輸出與雜湊在 [common_artifacts.py](../experiments/_shared/common_artifacts.py)。兩支入口仍負責各自資料分區與研究協定；五個抽取函式已核對 AST 一致，固定年份入口可在滑動視窗主程式不可 import 時獨立載入。

維護清單與正式 run 見[實驗索引](../docs/實驗索引.md)。
