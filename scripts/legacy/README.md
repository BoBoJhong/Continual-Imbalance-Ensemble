# 歷史工具導航

封存日期：2026-10-12。這些工具綁定舊協定、舊結果表或一次性遷移；現行工具見 [scripts 導航](../README.md)。維護 runner 不包含本目錄。

| 類別 | 數量 | 用途與限制 |
|---|---|---|
| plots | 9 | 舊 Phase 2／3 圖表；部分用 Test 最佳列，只能按歷史／探索協定引用 |
| reports | 8 | 舊 FS 格式匯出、Excel、CART 重跑與結果搬移工具；明確執行仍可能寫入或搬移成果 |
| analysis | 5 | 三個舊論文格式匯出、舊 Wilcoxon 及 9 月收斂查核；後者保留 mean/std 模擬 fallback，不作正式顯著性證據 |
| run | 2 | 舊 LightGBM 多 seeds 與雙切割批次；不是目前固定年份核心入口 |

## 原／新路徑

[搬移清冊](MIGRATION_MANIFEST.json)記錄原／新路徑、原文及封存後 SHA-256、root 修正與 import 防護。23 支來源原文另保存在 [tool_sources.zip](source_snapshots/20261012/tool_sources.zip)，ZIP 內路徑仍是舊位置，供追溯使用。

| 舊位置 | 新入口 |
|---|---|
| scripts/plots/visualize_phase2_all_methods_complete.py | [visualize_phase2_all_methods_complete.py](plots/visualize_phase2_all_methods_complete.py) |
| scripts/plots/visualize_phase2_single_vs_ensemble.py | [visualize_phase2_single_vs_ensemble.py](plots/visualize_phase2_single_vs_ensemble.py) |
| scripts/plots/visualize_phase2_tuned_correct.py | [visualize_phase2_tuned_correct.py](plots/visualize_phase2_tuned_correct.py) |
| scripts/plots/visualize_phase2_xgb_ensemble_families.py | [visualize_phase2_xgb_ensemble_families.py](plots/visualize_phase2_xgb_ensemble_families.py) |
| scripts/plots/visualize_phase3_all_methods.py | [visualize_phase3_all_methods.py](plots/visualize_phase3_all_methods.py) |
| scripts/plots/visualize_phase3_delta_clean.py | [visualize_phase3_delta_clean.py](plots/visualize_phase3_delta_clean.py) |
| scripts/plots/visualize_phase3_evolution.py | [visualize_phase3_evolution.py](plots/visualize_phase3_evolution.py) |
| scripts/plots/visualize_phase3_evolution_complete.py | [visualize_phase3_evolution_complete.py](plots/visualize_phase3_evolution_complete.py) |
| scripts/plots/visualize_phase3_vs_phase2_delta.py | [visualize_phase3_vs_phase2_delta.py](plots/visualize_phase3_vs_phase2_delta.py) |
| scripts/reports/archive_bankruptcy_fs_triad.py | [archive_bankruptcy_fs_triad.py](reports/archive_bankruptcy_fs_triad.py) |
| scripts/reports/archive_fs_methods.py | [archive_fs_methods.py](reports/archive_fs_methods.py) |
| scripts/reports/export_cart_sweep_to_thesis.py | [export_cart_sweep_to_thesis.py](reports/export_cart_sweep_to_thesis.py) |
| scripts/reports/export_fs_method_static_dynamic.py | [export_fs_method_static_dynamic.py](reports/export_fs_method_static_dynamic.py) |
| scripts/reports/generate_advisor_excel.py | [generate_advisor_excel.py](reports/generate_advisor_excel.py) |
| scripts/reports/migrate_phase3_results_to_method_dirs.py | [migrate_phase3_results_to_method_dirs.py](reports/migrate_phase3_results_to_method_dirs.py) |
| scripts/reports/run_cart_full_to_thesis.py | [run_cart_full_to_thesis.py](reports/run_cart_full_to_thesis.py) |
| scripts/reports/sync_ga_phase3_outputs.py | [sync_ga_phase3_outputs.py](reports/sync_ga_phase3_outputs.py) |
| scripts/analysis/export_phase2_dcs_thesis_format.py | [export_phase2_dcs_thesis_format.py](analysis/export_phase2_dcs_thesis_format.py) |
| scripts/analysis/export_phase2_thesis_format.py | [export_phase2_thesis_format.py](analysis/export_phase2_thesis_format.py) |
| scripts/analysis/export_phase3_fs_thesis_format.py | [export_phase3_fs_thesis_format.py](analysis/export_phase3_fs_thesis_format.py) |
| scripts/analysis/statistical_test.py | [statistical_test.py](analysis/statistical_test.py) |
| scripts/run/run_multi_seed.py | [run_multi_seed.py](run/run_multi_seed.py) |
| scripts/run/_run_both_splits.py | [_run_both_splits.py](run/_run_both_splits.py) |

## 執行與保存

封存入口依檔案位置取得專案根目錄，保留原結果位置。六支繪圖工具的 mkdir 已移至實際執行函式；舊 Excel 的建表／讀檔／輸出區塊放入 main，import 不再啟動這些動作。

import 測試不等於歷史輸入都完整或 CLI 能重現所有成果。部分舊路徑已不存在，亦有既存文字亂碼與不同選擇協定；這輪保留原文快照，不修造歷史數字。明確執行前核對輸入與輸出；本輪未執行訓練、繪圖、匯出或結果遷移。

## 股票範圍已退出

2026-10-12 舊工具的股票批次入口、結果遷移清單與 Excel 股票分頁已移除；混合輸入表僅匯出破產／醫療列。原文 ZIP 保留整理前來源，搬移清冊的 revisions 記錄這次範圍調整與前一版雜湊；見[清理紀錄](../../docs/股票清理紀錄_20261012.md)。

## 9 月收斂查核移入

另有 [audit_thesis_convergence.py](analysis/audit_thesis_convergence.py) 移入；它使用已退出的比例研究輸入，缺少歷史 predictions 時先說明並停止。另份[搬移清冊](ANALYSIS_RETIREMENT_20261012.json)與原文文字快照保留来源；前輪 23 支原文 ZIP 不變。詳見[清理審核](../../docs/結果清理與過時項目審核_20261012.md)。
