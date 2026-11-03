# 分析、查核與匯出

[回工具導航](../README.md)。使用前核對 run、target、seed 集合與選擇限制。

| 任務 | 入口 | 行為 |
|---|---|---|
| 程式與清理完整性 | [audit_project_maintenance.py](audit_project_maintenance.py) | 只讀審核，--output 可寫新 docs/audits JSON；不重算研究成效 |
| 逐檔分類 | [build_result_catalog.py](build_result_catalog.py) | 新建 results/catalog；不改原實驗數值 |
| 進度清點 | [study3b_pipeline_status.py](study3b_pipeline_status.py) | 讀 metadata；不判定程序活性 |
| 固定年份數值重播 | [audit_standard_dynamic.py](audit_standard_dynamic.py) | 更新指定 run 的 audit_validation.json |
| 固定年份公司群聚統計 | [fixed_company_statistics.py](fixed_company_statistics.py) | 90 設定逐筆配對，重算選定輸出；以公司重抽，165 項置中 bootstrap 近似檢定共同 Holm 校正；新建分析快照，不重訓或改原結果；--run 可指定事件標籤 run，輸出到 fixed_two_experiments_event |
| 固定年份見過／未見公司分層 | [fixed_seen_unseen_companies.py](fixed_seen_unseen_companies.py) | 依 Old_fit／New_fit 是否出現分組，重播 720 個已存輸出；組內按標籤分層重抽公司，僅報描述性 95% CI；不重訓、不作檢定；--run 可指定事件標籤 run。需與原統計相同的 numpy 2.4.0／numba 0.65.1 環境 |
| 切割審查 | [audit_phase123_split_contracts.py](audit_phase123_split_contracts.py) | 新建 docs/audits 時間戳 JSON，支援 --output |
| 格式／來源 | validate_result_artifacts.py、generate_result_manifest.py | 格式不等於方法有效性；後者會重建舊全域清冊 |
| B 完整流程匯出 | export_study3b_separated_tables.py／readout.py／study12_format.py | 新建表格快照；--update-thesis 另改正文 |
| B 消融與敏感度 | analyze_study3b_* | 各自分析快照、bootstrap 與圖表 |
| 舊加權擴充 | fair_weighted_ablation.py／weighted_split_validation.py／current_findings_* | 維護中的擴充，非最新固定年份公平對照 |
| 既有論文表 | [export_thesis_tables.py](export_thesis_tables.py) | runner 保留的 ROSS／加權歷史匯出 |

三個舊 Phase 2／3 格式匯出與 statistical_test.py 已移至 [legacy](../legacy/README.md)。舊 Wilcoxon 保留 mean/std 模擬 fallback，不將模擬輸出當成正式證據。

9 月收斂查核已移至 [legacy](../legacy/analysis/audit_thesis_convergence.py)，依賴的比例實驗結果已移除；不作現行數值重播入口。generate_result_manifest.py 仍預設覆寫舊全域清冊，待版本化改造；日常查詢先使用新結果清冊。
