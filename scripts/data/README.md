# 資料準備工具

[回工具導航](../README.md)。保留來源取得、衍生資料及 MOVER／ICU 準備工具，未因不在 runner 就封存。

| 類別 | 檔案範圍 | 核對項目 |
|---|---|---|
| 來源取得 | download_* | 來源版本、下載位置、真實／合成 fallback 與時間欄位 |
| 合成資料 | generate_synthetic_data.py | 僅產生醫療合成資料，明示合成身份 |
| 建表／準備 | build_icu_dataset.py、build_mover_dataset 相關流程、prepare_mover_modeling.py | 實際存在的入口、原始表與衍生 target |
| MOVER 開發 | _audit_mover.py、_qa_mover_labeled.py、_fix_mover_labeled.py | 仍有 baseline 提示引用；部分在執行／import 時直接讀寫資料，不作通用 library |

股票下載工具與合成股票生成器已移除。MOVER 的 build_mover_dataset.py 已存在；建表來源鏈與日期適用性仍須個別核對。這輪沒有重建資料；provenance 與日期適用性見[研究方向](../../docs/研究方向.md)。
