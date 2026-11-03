# 結果導航與版本規則

此目錄保存研究證據與歷史版本。不同 target、切割、validation 或 seed 集合的結果分開引用；完整研究索引見[實驗索引](../docs/實驗索引.md)。

## 最新可定位的完成 run

| 研究 | 結果 | 狀態與閱讀順序 |
|---|---|---|
| 固定年份 Study I／II | [20261004T202331745995Z](phase2_ensemble/standard_dynamic/runs/20261004T202331745995Z/) | 先 manifest、partition_summary、audit_validation，再 summary／metrics／FS differences；10 seeds × 9 設定 |
| 滑動視窗完整 Study 3B | [20260928T185026381432Z](phase_flexible/study3b_full_pipeline/runs/20260928T185026381432Z/) | 先 manifest、label_audit、逐 seed complete，再 summary／summary_by_seed；十 seeds 已完成 |

固定年份 target 為公司最終狀態；Study 3B 為回溯重建事件 target。兩套結果不能合併排名。最新固定年份 run 不含同協定 Retrain。

## 研究範圍更新

原有 BRFSS 結果排除於發布快照，本地備份保留。其他研究結果與來源保持，詳見[停止與發布紀錄](../docs/BRFSS支線停止紀錄_20261103.md)。

## 舊表與目錄

| 範圍 | 用途／標記 |
|---|---|
| phase1_baseline | 多分類器、選用深度學習、試跑與 partial；須按來源和完成狀態篩選 |
| phase2_ensemble／phase3_feature 的舊 bankruptcy CSV 與 plots | 歷史協定；不等於官方標準動態／新 FS 結果 |
| phase4_drift／phase5_weighted、multi_seed、statistical_tests、thesis_tables | 既有擴充、統計與論文格式匯出；確認 oracle／post-hoc 和檢定單位 |
| phase_flexible | rolling、FIFO、消融、敏感度與完整流程；部分舊 run 混有已退出 B-data |
| 完整 Study 3B run 內 separated_tables／manual_spreadsheets／readout_tables | 三 seeds 歷史快照；不是十 seeds 全集 |
| 完整 run 內 analysis/static_vs_sliding_rf_under | 七 seeds 歷史快照 |
| phase_flexible/study3b_oldnew_ratio_sensitivity | 已退出主線的分支，保留沿革 |
| report_figures、research_audit | 各報告／審查時點輸出；依來源與日期引用 |
| summary_all_datasets.csv、RESULT_MANIFEST.json | 舊彙總與全域 provenance；不代表目前所有 run 已涵蓋 |

所有檔案的分類與 metadata 見[結果清冊](catalog/README.md)，包含正式 run、部分 seeds 快照、試跑、失敗、歷史協定、已退出範圍與待確認來源。位元組相同的檔案保留 seed／快照身份，另列重複群。

清冊由 [build_result_catalog.py](../scripts/analysis/build_result_catalog.py) 自動產生，政策見 [result_catalog_config.json](../config/result_catalog_config.json)。目錄 metadata 是生成當時的快照，不代表新成效或全部數值驗證；實驗數值檔仍保留原位。具體研究限制見[歷史清單](../docs/歷史與待整理項目.md)。

## 保存與新增結果

1. 原始預測、逐年／逐 seed 指標、manifest、來源雜湊與查核紀錄一併保存。
2. 新 run 與新匯出使用各自時間戳；不以十 seeds 新表覆寫三 seeds 快照。
3. pooled 與年度平均分開；引用附上 run、方法、target、seed 集合和選擇協定。
4. 生成檔由 owning script 重建；不得手改 CSV／JSON 中的實驗數值。
5. 清理前核對文件引用、恢復狀態與是否為同內容。分類為歷史不等於可刪除。

目前全域 RESULT_MANIFEST 產生於 2026-09-11。建立新版應透過 [generate_result_manifest.py](../scripts/analysis/generate_result_manifest.py)，先確認輸出位置並保留舊快照。本輪未重新產生。

## 程式封存與 provenance

25 支歷史 Phase 2／3 入口已搬至 [legacy](../experiments/legacy/README.md)，結果路徑保持原樣。現行固定年份與完整滑動視窗主程式改動前的原文，保存於 [來源快照](../experiments/legacy/source_snapshots/20261012/)；來源 SHA-256 與搬移對照供既有 manifest 追溯。新 run 會記錄新的共用模組。--resume 的來源一致性防護保持，舊 run 來源不同時不放寬檢查。
