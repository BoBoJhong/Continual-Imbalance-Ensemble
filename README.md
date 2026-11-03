# Continual Imbalance Ensemble

研究非平穩、類別不平衡資料中的持續式集成學習。目前以破產資料為主、醫療資料為輔助，核心為固定年份 Old／New 模型池、靜態與動態選擇，以及特徵選擇；滑動視窗 FIFO、ROSS／DAWCE 與其他資料集分別保留為擴充或輔助研究。

2026-11-03 已依作者決定，從發布快照及新提交歷史排除原有 BRFSS 專用程式、資料、結果與舊說明。原件只保留在本地備份；最新 MEPS、KHP、RAND HRS 候選評估保留，詳見[停止與發布紀錄](docs/BRFSS支線停止紀錄_20261103.md)。

2026-10-12 已依指示移除股票專用資料、實驗、結果、日誌與工具；目前維護破產與醫療流程。混合歷史報告及原文快照保留當時範圍，詳細清單見[股票清理紀錄](docs/股票清理紀錄_20261012.md)。

另已移除 238 個退出範圍、失敗、試跑及重複結果；保留正式 run 與圖表來源。程式審核與剩餘歷史項目見[結果清理審核](docs/結果清理與過時項目審核_20261012.md)。

## 從這裡開始

| 需求 | 入口 |
|---|---|
| 找實驗、指令與正式結果版本 | [實驗索引](docs/實驗索引.md) |
| 確認最新研究範圍與待辦 | [研究方向](docs/研究方向.md) |
| 找報告、流程圖與審查 | [文件導航](docs/README.md) |
| 分辨歷史／部分快照／試跑 | [歷史與待整理項目](docs/歷史與待整理項目.md) |
| 查結果目錄與版本規則 | [結果導航](results/README.md) |
| 查所有結果檔、分類與重複群 | [逐檔結果清冊](results/catalog/README.md) |
| 論文編譯、正式結果同步、Overleaf／MCP | [論文環境與數據更新](docs/thesis_environment/README.md) |
| 查過往報告／審查 | [reports](docs/reports/README.md)／[audits](docs/audits/README.md) |

## 目前可引用的 run

- 固定年份 Study I／II：[20261004T202331745995Z](results/phase2_ensemble/standard_dynamic/runs/20261004T202331745995Z/)；10 seeds × 9 特徵設定，已完成並有獨立數值查核紀錄。Old-fit 1999–2011、New-fit 2012–2013、2014 DSEL／Calibration、Test 2015–2018。target 為原始公司最終狀態。
- 滑動視窗完整 Study 3B：[20260928T185026381432Z](results/phase_flexible/study3b_full_pipeline/runs/20260928T185026381432Z/)；六分類器 × 四採樣、集成與 FS，十 seeds 已完成。使用回溯重建事件 target，與固定年份結果分表呈現。

完成 run 不代表全部研究比較完成。最新固定年份 run 仍缺同協定 Old＋New Retrain，以及 Phase 1 驗證選出分類器的完整串接紀錄；真實標籤可用時點亦未驗證。限制與證據詳見[實驗索引](docs/實驗索引.md)。

## 安裝與查看入口

由專案根目錄執行；標準依賴以 [requirements.txt](requirements.txt) 為準。Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python scripts/run/run_all_experiments.py --list
```

選用的深度學習／GPU baseline 有各自額外依賴與權重需求，不在標準一鍵流程中。重現既存 run 時另核對其 manifest 的 Python、套件版本、參數與 source hashes；不要將目前安裝環境當作所有歷史 run 的環境。

## 執行核心實驗

以下指令會新建結果 run：

```powershell
# Study I：no-FS
python experiments/phase2_ensemble/dynamic/des/bankruptcy_standard_dynamic.py
# Study II：no-FS + MI/CART/SHAP/RFE 50%/80%
python experiments/phase3_feature/bankruptcy_standard_dynamic_fs.py
# 擴充：滑動視窗完整六分類器流程
python experiments/phase_flexible/study3b_full_pipeline.py
```

一鍵 runner 支援 --phase，但 phase1 是既有 baseline 協定，study3b 是基本 overlap／FIFO 入口。完整六分類器流程須使用上方直接入口。完整 runner 可能耗時數小時並更新結果；先用 --list 檢查選定流程。

## 目錄分工

| 目錄 | 放什麼 |
|---|---|
| [config](config/README.md) | 共用預設設定；run manifest 保存實際設定 |
| data | 原始與衍生資料、來源版本 |
| [experiments](experiments/README.md) | 現行實驗入口與流程協定；[_shared](experiments/_shared/) 提供共用函式 |
| [experiments/legacy](experiments/legacy/README.md) | 封存的歷史 Phase 2／3 入口、來源快照與搬移對照 |
| [results](results/README.md) | 原始結果、摘要、audit 與 provenance |
| [scripts](scripts/README.md) | 現行執行、分析、報告、資料與圖表工具 |
| [scripts/legacy](scripts/legacy/README.md) | 24 支歷史工具（原 23 支加 9 月查核）、搬移對照與來源快照 |
| [src](src/README.md) | 共用資料、模型、集成、特徵與評估模組 |
| tests | 核心契約、時間切割與資料洩漏防護測試 |
| [docs](docs/README.md) | 研究規格、索引、報告與審查 |
| [thesis](thesis/README.md) | 論文草稿與格式參考 |

原始資料與研究結果保留版本追溯。結果由 owning script 產生，避免手動修數字。25 支舊 Phase 2／3 程式已封存至 legacy；結果與報告位置保留。整理範圍與驗證見[整理紀錄](docs/整理紀錄_20261012.md)。
