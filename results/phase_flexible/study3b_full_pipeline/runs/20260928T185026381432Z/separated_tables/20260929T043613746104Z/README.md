# 滑動視窗分表結果

本快照納入完整完成的 3/10 個 seeds：42, 123, 456。Test 特徵年 2009–2018；每個方法、每個 seed 共 33,636 筆 Test 預測。

所有表皆為已完成 seeds 的平均。Pooled 先於每個 seed 合併年度預測計算指標，再取 seed 平均；年度平均則先對每個 seed 的年度指標等權平均，再取 seed 平均。二者分表，不將重複 seed 視為新的公司樣本。CSV 另保留 seed 標準差與各 seed 數值；標準差不是信賴區間。本次為執行中快照，不能當作十種子最終結論。

配置依名稱排列，不以 Test 排名重新選模型。FS 的基礎集成可能隨 seed／年度不同，FS 表比較的是同一年度已選配置上的 FS 規則。

- [單模型與採樣](01_single_models/README.md)
- [集成](02_ensembles/README.md)
- [特徵選擇](03_feature_selection/README.md)

每類資料夾再依分類器或集成機制拆開。原始逐筆預測仍保留在 run 的 seed 資料夾。
