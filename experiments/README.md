# 實驗入口導航

所有指令從專案根目錄執行。實驗狀態、完整 run 路徑與協定見[實驗索引](../docs/實驗索引.md)，研究定義見[研究方向](../docs/研究方向.md)。

## 優先使用的入口

| 研究 | 入口 | 實際內容 |
|---|---|---|
| 核心 Study I | [bankruptcy_standard_dynamic.py](phase2_ensemble/dynamic/des/bankruptcy_standard_dynamic.py) | 固定年份六模型池、49 混合子集、官方 DESlib 動態選擇；預設 no-FS |
| 核心 Study II | [bankruptcy_standard_dynamic_fs.py](phase3_feature/bankruptcy_standard_dynamic_fs.py) | 同一協定；no-FS 與 MI／CART／SHAP／RFE 50%／80% |
| 擴充 Study 3B 基本比較 | [rolling_bankruptcy_overlap_ensemble.py](phase_flexible/rolling_bankruptcy_overlap_ensemble.py) | 近期窗口單模型與 FIFO 等權集成 |
| 擴充 Study 3B 完整流程 | [study3b_full_pipeline.py](phase_flexible/study3b_full_pipeline.py) | 六分類器 × 四採樣、46 集成候選、validation 篩選後 FS |
| B 窗口敏感度 | [study3b_window_sensitivity.py](phase_flexible/study3b_window_sensitivity.py) | 窗口 2／3／5／7 年 |
| 公平 A／B | [study3_ab_fair_comparison.py](phase_flexible/study3_ab_fair_comparison.py) | 同事件 target 下的探索比較 |
| 持續更新 replay | [study3b_continual_runtime.py](phase_flexible/study3b_continual_runtime.py) | 標籤 ledger、模型 registry、逐批預測；需本機 checkpoints |

核心實驗：Old-fit 1999–2011，New-fit 2012–2013，2014 互斥 DSEL／Calibration，Test 2015–2018。FS 僅擬合 Old-fit，採樣在 FS 後。完整滑動視窗流程則使用另一事件 target、年度 validation 與 FIFO，沒有 DES／DCS；兩者分開報告。

```powershell
python experiments/phase2_ensemble/dynamic/des/bankruptcy_standard_dynamic.py
python experiments/phase3_feature/bankruptcy_standard_dynamic_fs.py
python experiments/phase_flexible/study3b_full_pipeline.py
```

三個入口均支援 --resume RUN；重新執行前先查該 run manifest。固定年份結果共存於 results/phase2_ensemble/standard_dynamic/runs；Study II 並非另存至舊 phase3_feature 表格目錄。

## 研究範圍更新

原有 BRFSS 專用入口排除於發布快照；最新健康候選評估保留。詳見[停止與發布紀錄](../docs/BRFSS支線停止紀錄_20261103.md)。

## 既有目錄與主線的關係

| 目錄 | 定位 |
|---|---|
| phase1_baseline | 多分類器與選用深度學習基準；歷史內層驗證不同，須先對齊才作最新核心比較 |
| phase2_ensemble/static、dynamic | 同時包含現行標準入口與舊協定入口；按檔名與索引選擇 |
| phase3_feature | 現行標準 FS、獨立穩定性分析與舊 FS 實驗並存 |
| phase4_drift、phase5_weighted | ROSS、漂移、加權等擴充；oracle 入口須保持防護與標記 |
| phase_flexible | rolling、FIFO、完整 Study 3B 與 replay |
| _shared | 資料處理、共用模型／FS 與輸出工具；可重用邏輯，不直接執行 |
| [legacy](legacy/README.md) | 25 支歷史 Phase 2／3 入口與本輪來源快照；不列入維護 runner |

[一鍵 runner](../scripts/run/run_all_experiments.py)的 phase1 使用既有 retrain.py；study3b 僅使用基本 overlap 入口。先執行 --list，勿把所有維護入口視為同條件比較。

## 相依與封存

固定年份與完整滑動視窗入口現共用 [_shared/common_modeling.py](_shared/common_modeling.py)（模型、採樣標記、seed 與 FS）及 [common_artifacts.py](_shared/common_artifacts.py)（原子輸出與雜湊）。固定年份入口已解除對滑動視窗主程式的 import 相依。

25 支舊 Phase 2／3 入口已移至 [legacy](legacy/README.md)，原／新路徑與來源雜湊見 [搬移清冊](legacy/MIGRATION_MANIFEST.json)。歷史结果與共用 helper 保留原位，medical 輔助入口仍在原目錄；股票專用入口及共用分支已依指示移除；一鍵 runner 維持 17 支現行入口。現行兩支主程式改動前的原文也有來源快照；既有 run 不改 provenance，來源變更時 --resume 仍依原規則拒絕。
