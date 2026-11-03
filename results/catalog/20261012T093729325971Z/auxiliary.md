# 輔助資料／分析

跨領域與穩定性等獨立分析；適用範圍需交代。

[回分類首頁](INDEX.md)；[逐檔 JSON](catalog.json)

| 目錄 | 檔案數 |
|---|---|
| [results/multi_seed](../../multi_seed) | 1 |
| [results/phase1_baseline/lgbm/med_lgb_tuned48_20260514/tuned](../../phase1_baseline/lgbm/med_lgb_tuned48_20260514/tuned) | 1 |
| [results/phase1_baseline/logistic_regression/med_lr_cursor_20260514/tuned](../../phase1_baseline/logistic_regression/med_lr_cursor_20260514/tuned) | 1 |
| [results/phase1_baseline/medical_diabetic130_kfold/med_kfold_raw_full_20260514](../../phase1_baseline/medical_diabetic130_kfold/med_kfold_raw_full_20260514) | 3 |
| [results/phase1_baseline/medical_diabetic130_kfold/med_kfold_simple5_full_20260514](../../phase1_baseline/medical_diabetic130_kfold/med_kfold_simple5_full_20260514) | 3 |
| [results/phase1_baseline/random_forest/med_rf_cursor_20260514/tuned](../../phase1_baseline/random_forest/med_rf_cursor_20260514/tuned) | 1 |
| [results/phase1_baseline/tabicl/med_tabicl_gpu_20260514_temp075_ctx2048](../../phase1_baseline/tabicl/med_tabicl_gpu_20260514_temp075_ctx2048) | 1 |
| [results/phase1_baseline/xgb](../../phase1_baseline/xgb) | 1 |
| [results/phase1_baseline/xgb/med_tuned200_20260416](../../phase1_baseline/xgb/med_tuned200_20260416) | 1 |
| [results/phase1_baseline/xgb/plots](../../phase1_baseline/xgb/plots) | 1 |
| [results/phase1_baseline/xgb/tuned](../../phase1_baseline/xgb/tuned) | 1 |
| [results/phase2_ensemble/dynamic/dcs](../../phase2_ensemble/dynamic/dcs) | 2 |
| [results/phase2_ensemble/dynamic/des](../../phase2_ensemble/dynamic/des) | 8 |
| [results/phase2_ensemble/static](../../phase2_ensemble/static) | 8 |
| [results/phase3_feature/cart](../../phase3_feature/cart) | 2 |
| [results/phase3_feature/ga](../../phase3_feature/ga) | 3 |
| [results/phase3_feature/mutual_info](../../phase3_feature/mutual_info) | 5 |
| [results/phase3_feature/rfe](../../phase3_feature/rfe) | 4 |
| [results/phase3_feature/shap](../../phase3_feature/shap) | 3 |
| [results/phase3_feature/stability](../../phase3_feature/stability) | 5 |
| [results/phase3_feature/thesis](../../phase3_feature/thesis) | 5 |

## 逐檔清單

| 檔案 | bytes | 規則 |
|---|---|---|
| [results/multi_seed/medical_multi_seed.csv](../../multi_seed/medical_multi_seed.csv) | 1554 | auxiliary-domain |
| [results/phase1_baseline/lgbm/med_lgb_tuned48_20260514/tuned/medical_year_splits_lgb_raw.csv](../../phase1_baseline/lgbm/med_lgb_tuned48_20260514/tuned/medical_year_splits_lgb_raw.csv) | 18266 | auxiliary-domain |
| [results/phase1_baseline/logistic_regression/med_lr_cursor_20260514/tuned/medical_year_splits_lr_raw.csv](../../phase1_baseline/logistic_regression/med_lr_cursor_20260514/tuned/medical_year_splits_lr_raw.csv) | 9376 | auxiliary-domain |
| [results/phase1_baseline/medical_diabetic130_kfold/med_kfold_raw_full_20260514/kfold_metrics_per_fold.csv](../../phase1_baseline/medical_diabetic130_kfold/med_kfold_raw_full_20260514/kfold_metrics_per_fold.csv) | 3273 | auxiliary-domain |
| [results/phase1_baseline/medical_diabetic130_kfold/med_kfold_raw_full_20260514/kfold_summary_mean_std.csv](../../phase1_baseline/medical_diabetic130_kfold/med_kfold_raw_full_20260514/kfold_summary_mean_std.csv) | 1311 | auxiliary-domain |
| [results/phase1_baseline/medical_diabetic130_kfold/med_kfold_raw_full_20260514/run_meta.csv](../../phase1_baseline/medical_diabetic130_kfold/med_kfold_raw_full_20260514/run_meta.csv) | 265 | auxiliary-domain |
| [results/phase1_baseline/medical_diabetic130_kfold/med_kfold_simple5_full_20260514/kfold_metrics_per_fold.csv](../../phase1_baseline/medical_diabetic130_kfold/med_kfold_simple5_full_20260514/kfold_metrics_per_fold.csv) | 3271 | auxiliary-domain |
| [results/phase1_baseline/medical_diabetic130_kfold/med_kfold_simple5_full_20260514/kfold_summary_mean_std.csv](../../phase1_baseline/medical_diabetic130_kfold/med_kfold_simple5_full_20260514/kfold_summary_mean_std.csv) | 1349 | auxiliary-domain |
| [results/phase1_baseline/medical_diabetic130_kfold/med_kfold_simple5_full_20260514/run_meta.csv](../../phase1_baseline/medical_diabetic130_kfold/med_kfold_simple5_full_20260514/run_meta.csv) | 324 | auxiliary-domain |
| [results/phase1_baseline/random_forest/med_rf_cursor_20260514/tuned/medical_year_splits_rf_raw.csv](../../phase1_baseline/random_forest/med_rf_cursor_20260514/tuned/medical_year_splits_rf_raw.csv) | 18291 | auxiliary-domain |
| [results/phase1_baseline/tabicl/med_tabicl_gpu_20260514_temp075_ctx2048/medical_year_splits_tabicl_raw.csv](../../phase1_baseline/tabicl/med_tabicl_gpu_20260514_temp075_ctx2048/medical_year_splits_tabicl_raw.csv) | 5359 | auxiliary-domain |
| [results/phase1_baseline/xgb/med_tuned200_20260416/medical_year_splits_xgb_raw.csv](../../phase1_baseline/xgb/med_tuned200_20260416/medical_year_splits_xgb_raw.csv) | 17875 | auxiliary-domain |
| [results/phase1_baseline/xgb/medical_year_splits_xgb_raw.csv](../../phase1_baseline/xgb/medical_year_splits_xgb_raw.csv) | 5359 | auxiliary-domain |
| [results/phase1_baseline/xgb/plots/medical_year_splits_AUC_F1.png](../../phase1_baseline/xgb/plots/medical_year_splits_AUC_F1.png) | 175518 | auxiliary-domain |
| [results/phase1_baseline/xgb/tuned/medical_year_splits_xgb_raw.csv](../../phase1_baseline/xgb/tuned/medical_year_splits_xgb_raw.csv) | 17885 | auxiliary-domain |
| [results/phase2_ensemble/dynamic/dcs/xgb_oldnew_ensemble_dcs_by_sampling_raw_medical.csv](../../phase2_ensemble/dynamic/dcs/xgb_oldnew_ensemble_dcs_by_sampling_raw_medical.csv) | 1683 | auxiliary-domain |
| [results/phase2_ensemble/dynamic/dcs/xgb_oldnew_ensemble_dcs_thesis_format_medical.csv](../../phase2_ensemble/dynamic/dcs/xgb_oldnew_ensemble_dcs_thesis_format_medical.csv) | 1172 | auxiliary-domain |
| [results/phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_AUC_summary_wide_medical.csv](../../phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_AUC_summary_wide_medical.csv) | 1010 | auxiliary-domain |
| [results/phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_AUC_table_medical.csv](../../phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_AUC_table_medical.csv) | 3817 | auxiliary-domain |
| [results/phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_by_sampling_raw_medical.csv](../../phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_by_sampling_raw_medical.csv) | 5420 | auxiliary-domain |
| [results/phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_F1_summary_wide_medical.csv](../../phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_F1_summary_wide_medical.csv) | 1010 | auxiliary-domain |
| [results/phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_F1_table_medical.csv](../../phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_F1_table_medical.csv) | 3816 | auxiliary-domain |
| [results/phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_Recall_summary_wide_medical.csv](../../phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_Recall_summary_wide_medical.csv) | 1010 | auxiliary-domain |
| [results/phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_Recall_table_medical.csv](../../phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_Recall_table_medical.csv) | 3820 | auxiliary-domain |
| [results/phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_thesis_format_medical.csv](../../phase2_ensemble/dynamic/des/xgb_oldnew_ensemble_des_thesis_format_medical.csv) | 1155 | auxiliary-domain |
| [results/phase2_ensemble/static/xgb_oldnew_ensemble_static_AUC_summary_wide_medical.csv](../../phase2_ensemble/static/xgb_oldnew_ensemble_static_AUC_summary_wide_medical.csv) | 683 | auxiliary-domain |
| [results/phase2_ensemble/static/xgb_oldnew_ensemble_static_AUC_table_medical.csv](../../phase2_ensemble/static/xgb_oldnew_ensemble_static_AUC_table_medical.csv) | 2125 | auxiliary-domain |
| [results/phase2_ensemble/static/xgb_oldnew_ensemble_static_by_sampling_raw_medical.csv](../../phase2_ensemble/static/xgb_oldnew_ensemble_static_by_sampling_raw_medical.csv) | 3425 | auxiliary-domain |
| [results/phase2_ensemble/static/xgb_oldnew_ensemble_static_F1_summary_wide_medical.csv](../../phase2_ensemble/static/xgb_oldnew_ensemble_static_F1_summary_wide_medical.csv) | 683 | auxiliary-domain |
| [results/phase2_ensemble/static/xgb_oldnew_ensemble_static_F1_table_medical.csv](../../phase2_ensemble/static/xgb_oldnew_ensemble_static_F1_table_medical.csv) | 2124 | auxiliary-domain |
| [results/phase2_ensemble/static/xgb_oldnew_ensemble_static_Recall_summary_wide_medical.csv](../../phase2_ensemble/static/xgb_oldnew_ensemble_static_Recall_summary_wide_medical.csv) | 683 | auxiliary-domain |
| [results/phase2_ensemble/static/xgb_oldnew_ensemble_static_Recall_table_medical.csv](../../phase2_ensemble/static/xgb_oldnew_ensemble_static_Recall_table_medical.csv) | 2128 | auxiliary-domain |
| [results/phase2_ensemble/static/xgb_oldnew_static_thesis_format_medical.csv](../../phase2_ensemble/static/xgb_oldnew_static_thesis_format_medical.csv) | 1135 | auxiliary-domain |
| [results/phase3_feature/cart/xgb_medical_fs_full_des.csv](../../phase3_feature/cart/xgb_medical_fs_full_des.csv) | 2005 | auxiliary-domain |
| [results/phase3_feature/cart/xgb_medical_fs_full_static.csv](../../phase3_feature/cart/xgb_medical_fs_full_static.csv) | 1782 | auxiliary-domain |
| [results/phase3_feature/ga/_SOURCES.txt](../../phase3_feature/ga/_SOURCES.txt) | 847 | independent-fs |
| [results/phase3_feature/ga/bankruptcy_ga_fs_sweep.csv](../../phase3_feature/ga/bankruptcy_ga_fs_sweep.csv) | 360 | independent-fs |
| [results/phase3_feature/ga/medical_ga_fs_sweep.csv](../../phase3_feature/ga/medical_ga_fs_sweep.csv) | 360 | independent-fs |
| [results/phase3_feature/mutual_info/medical_fs_sweep.csv](../../phase3_feature/mutual_info/medical_fs_sweep.csv) | 1540 | auxiliary-domain |
| [results/phase3_feature/mutual_info/xgb_medical_fs_full_dcs.csv](../../phase3_feature/mutual_info/xgb_medical_fs_full_dcs.csv) | 3057 | auxiliary-domain |
| [results/phase3_feature/mutual_info/xgb_medical_fs_full_des.csv](../../phase3_feature/mutual_info/xgb_medical_fs_full_des.csv) | 3866 | auxiliary-domain |
| [results/phase3_feature/mutual_info/xgb_medical_fs_full_static.csv](../../phase3_feature/mutual_info/xgb_medical_fs_full_static.csv) | 3443 | auxiliary-domain |
| [results/phase3_feature/mutual_info/xgb_medical_fs_summary_diff.csv](../../phase3_feature/mutual_info/xgb_medical_fs_summary_diff.csv) | 528 | auxiliary-domain |
| [results/phase3_feature/rfe/xgb_medical_fs_full_dcs.csv](../../phase3_feature/rfe/xgb_medical_fs_full_dcs.csv) | 3085 | auxiliary-domain |
| [results/phase3_feature/rfe/xgb_medical_fs_full_des.csv](../../phase3_feature/rfe/xgb_medical_fs_full_des.csv) | 3897 | auxiliary-domain |
| [results/phase3_feature/rfe/xgb_medical_fs_full_static.csv](../../phase3_feature/rfe/xgb_medical_fs_full_static.csv) | 3474 | auxiliary-domain |
| [results/phase3_feature/rfe/xgb_medical_rfe_results.csv](../../phase3_feature/rfe/xgb_medical_rfe_results.csv) | 565 | auxiliary-domain |
| [results/phase3_feature/shap/xgb_medical_fs_full_dcs.csv](../../phase3_feature/shap/xgb_medical_fs_full_dcs.csv) | 3113 | auxiliary-domain |
| [results/phase3_feature/shap/xgb_medical_fs_full_des.csv](../../phase3_feature/shap/xgb_medical_fs_full_des.csv) | 3909 | auxiliary-domain |
| [results/phase3_feature/shap/xgb_medical_fs_full_static.csv](../../phase3_feature/shap/xgb_medical_fs_full_static.csv) | 3487 | auxiliary-domain |
| [results/phase3_feature/stability/bankruptcy_feature_stability_errors.csv](../../phase3_feature/stability/bankruptcy_feature_stability_errors.csv) | 91 | independent-fs |
| [results/phase3_feature/stability/bankruptcy_feature_stability_frequency.csv](../../phase3_feature/stability/bankruptcy_feature_stability_frequency.csv) | 11609 | independent-fs |
| [results/phase3_feature/stability/bankruptcy_feature_stability_pairwise_jaccard.csv](../../phase3_feature/stability/bankruptcy_feature_stability_pairwise_jaccard.csv) | 119030 | independent-fs |
| [results/phase3_feature/stability/bankruptcy_feature_stability_selected_features.csv](../../phase3_feature/stability/bankruptcy_feature_stability_selected_features.csv) | 25456 | independent-fs |
| [results/phase3_feature/stability/bankruptcy_feature_stability_summary.csv](../../phase3_feature/stability/bankruptcy_feature_stability_summary.csv) | 1211 | independent-fs |
| [results/phase3_feature/thesis/xgb_medical_CART_dynamic.csv](../../phase3_feature/thesis/xgb_medical_CART_dynamic.csv) | 1164 | auxiliary-domain |
| [results/phase3_feature/thesis/xgb_medical_CART_static.csv](../../phase3_feature/thesis/xgb_medical_CART_static.csv) | 954 | auxiliary-domain |
| [results/phase3_feature/thesis/xgb_medical_cart_sweep.csv](../../phase3_feature/thesis/xgb_medical_cart_sweep.csv) | 129 | auxiliary-domain |
| [results/phase3_feature/thesis/xgb_medical_fs_des_thesis_format.csv](../../phase3_feature/thesis/xgb_medical_fs_des_thesis_format.csv) | 1150 | auxiliary-domain |
| [results/phase3_feature/thesis/xgb_medical_fs_static_thesis_format.csv](../../phase3_feature/thesis/xgb_medical_fs_static_thesis_format.csv) | 1066 | auxiliary-domain |
