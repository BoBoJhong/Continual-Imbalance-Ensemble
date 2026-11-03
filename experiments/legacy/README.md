# 歷史實驗入口

封存日期：2026-10-12。這裡保留舊 Phase 2／3 協定的可執行入口，供來源追溯；現行入口見[實驗索引](../../docs/實驗索引.md)。一鍵 runner 不包含本目錄。

搬移清冊見 [MIGRATION_MANIFEST.json](MIGRATION_MANIFEST.json)，記錄原／新路徑、原始與搬移後 SHA-256，以及 bootstrap 的 root 修正。算法、參數、歷史結果位置保持原樣；執行舊入口仍可能重建歷史輸出，應先核對其結果目標。

## Phase 2

| 舊位置（experiments/ 起算） | 新入口 |
|---|---|
| phase2_ensemble/static/all_combinations.py | [all_combinations.py](phase2_ensemble/static/all_combinations.py) |
| phase2_ensemble/static/undersampling.py | [undersampling.py](phase2_ensemble/static/undersampling.py) |
| phase2_ensemble/static/oversampling.py | [oversampling.py](phase2_ensemble/static/oversampling.py) |
| phase2_ensemble/static/hybrid.py | [hybrid.py](phase2_ensemble/static/hybrid.py) |
| phase2_ensemble/dynamic/des/standard.py | [standard.py](phase2_ensemble/dynamic/des/standard.py) |
| phase2_ensemble/dynamic/des/advanced.py | [advanced.py](phase2_ensemble/dynamic/des/advanced.py) |
| phase2_ensemble/dynamic/dcs/comparison.py | [comparison.py](phase2_ensemble/dynamic/dcs/comparison.py) |
| phase2_ensemble/static/xgb_oldnew_bankruptcy_year_splits_static.py | [xgb_oldnew_bankruptcy_year_splits_static.py](phase2_ensemble/static/xgb_oldnew_bankruptcy_year_splits_static.py) |
| phase2_ensemble/dynamic/des/xgb_oldnew_bankruptcy_year_splits_des.py | [xgb_oldnew_bankruptcy_year_splits_des.py](phase2_ensemble/dynamic/des/xgb_oldnew_bankruptcy_year_splits_des.py) |
| phase2_ensemble/dynamic/dcs/xgb_oldnew_bankruptcy_year_splits_dcs.py | [xgb_oldnew_bankruptcy_year_splits_dcs.py](phase2_ensemble/dynamic/dcs/xgb_oldnew_bankruptcy_year_splits_dcs.py) |

## Phase 3

| 舊位置（experiments/ 起算） | 新入口 |
|---|---|
| phase3_feature/fs_study.py | [fs_study.py](phase3_feature/fs_study.py) |
| phase3_feature/fs_sweep.py | [fs_sweep.py](phase3_feature/fs_sweep.py) |
| phase3_feature/xgb_bankruptcy_year_splits_fs.py | [xgb_bankruptcy_year_splits_fs.py](phase3_feature/xgb_bankruptcy_year_splits_fs.py) |
| phase3_feature/xgb_bankruptcy_year_splits_fs_advanced.py | [xgb_bankruptcy_year_splits_fs_advanced.py](phase3_feature/xgb_bankruptcy_year_splits_fs_advanced.py) |
| phase3_feature/xgb_bankruptcy_year_splits_fs_dcs.py | [xgb_bankruptcy_year_splits_fs_dcs.py](phase3_feature/xgb_bankruptcy_year_splits_fs_dcs.py) |
| phase3_feature/xgb_bankruptcy_year_splits_fs_full.py | [xgb_bankruptcy_year_splits_fs_full.py](phase3_feature/xgb_bankruptcy_year_splits_fs_full.py) |
| phase3_feature/xgb_bankruptcy_year_splits_rfe.py | [xgb_bankruptcy_year_splits_rfe.py](phase3_feature/xgb_bankruptcy_year_splits_rfe.py) |
| phase3_feature/mutual_info/fs_study.py | [fs_study.py](phase3_feature/mutual_info/fs_study.py) |
| phase3_feature/mutual_info/fs_sweep.py | [fs_sweep.py](phase3_feature/mutual_info/fs_sweep.py) |
| phase3_feature/mutual_info/xgb_bankruptcy_year_splits_fs.py | [xgb_bankruptcy_year_splits_fs.py](phase3_feature/mutual_info/xgb_bankruptcy_year_splits_fs.py) |
| phase3_feature/mutual_info/xgb_bankruptcy_year_splits_fs_advanced.py | [xgb_bankruptcy_year_splits_fs_advanced.py](phase3_feature/mutual_info/xgb_bankruptcy_year_splits_fs_advanced.py) |
| phase3_feature/mutual_info/xgb_bankruptcy_year_splits_fs_dcs.py | [xgb_bankruptcy_year_splits_fs_dcs.py](phase3_feature/mutual_info/xgb_bankruptcy_year_splits_fs_dcs.py) |
| phase3_feature/mutual_info/xgb_bankruptcy_year_splits_fs_full.py | [xgb_bankruptcy_year_splits_fs_full.py](phase3_feature/mutual_info/xgb_bankruptcy_year_splits_fs_full.py) |
| phase3_feature/rfe/xgb_bankruptcy_year_splits_rfe.py | [xgb_bankruptcy_year_splits_rfe.py](phase3_feature/rfe/xgb_bankruptcy_year_splits_rfe.py) |
| phase3_feature/shap/xgb_bankruptcy_year_splits_fs_advanced.py | [xgb_bankruptcy_year_splits_fs_advanced.py](phase3_feature/shap/xgb_bankruptcy_year_splits_fs_advanced.py) |

## 現行入口改動前的來源快照

[source_snapshots/20261012](source_snapshots/20261012/)保存本輪改動前的固定年份與完整滑動視窗主程式，內容與當時來源 SHA-256 對應。文字快照不用作新入口；新共享函式位於 experiments/_shared/common_modeling.py 與 common_artifacts.py。

既有 run manifest 不修改。因現行來源與來源清單已變更，原有 --resume 的 provenance 檢查仍會拒絕不一致來源；需要重跑時建立新 run，不將封存程序當成新的實驗成效。

2026-10-12 已依指示從歷史批次入口移除股票分支，目前只處理破產／醫療。原文來源快照維持歷史身份；搬移清冊 revisions 保留範圍調整前的雜湊。見[股票清理紀錄](../../docs/股票清理紀錄_20261012.md)。
