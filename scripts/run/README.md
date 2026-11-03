# 維護 runner

[run_all_experiments.py](run_all_experiments.py)目前有 17 支維護入口。

```powershell
python scripts/run/run_all_experiments.py --list
```

--list 只核對與列清單；省略時會實際執行並更新結果，--phase 可限定階段。phase1 是既有 baseline，study3b 為基本 overlap／FIFO；完整六分類器流程須直接執行 study3b_full_pipeline.py，詳見[實驗索引](../../docs/實驗索引.md)。

run_multi_seed.py／_run_both_splits.py 已移至 [legacy](../legacy/README.md)。本輪 runner 程式與 17 支入口未改。
