# Continual Imbalance Ensemble

本專案研究非平穩、類別不平衡資料中的持續式集成學習，主線以美國企業
破產資料比較歷史模型、近期模型、重訓練、靜態集成、DES/DCS、特徵選擇、
ROSS、DAWCE 與 rolling walk-forward 選擇流程。

完整的研究問題、Study 流程、實驗結果、限制與待辦收錄於
[docs/研究方向.md](docs/研究方向.md)；供指導教授討論的成果摘要見
[研究成果與未來研究方向報告](docs/研究成果與未來研究方向報告.md)。`data/raw/` 與 `results/` 均刻意納入
Git，以保留來源與研究證據。

目前僅以破產資料為主線。研究正確性審查與修正進度見
[研究審查報告](docs/研究正確性與程式碼審查報告.md)。本地標籤可用時點尚未驗證；
保存的研究數字屬歷史探索性結果，不是本輪修正後重新訓練的結果。

## 安裝

標準環境支援 64-bit CPython 3.11–3.14，建議使用 Python 3.11 或 3.12。

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

安裝後驗證：

```bash
python -m pip check
python -m pytest -q
python -m compileall -q src experiments scripts
python scripts/run/run_all_experiments.py --list
python scripts/analysis/validate_result_artifacts.py results --strict
```

`requirements.txt` 是唯一的標準依賴清單，涵蓋目前維護中的 CPU 主流程、
特徵分析、統計、圖表、報表與測試。FT-Transformer、TabM、TabNet、TabR、
TabICL/TabPFN 與 Torch MLP 是選用的深度學習基準；它們的 GPU、模型權重與
PyTorch 需求互相不同，不納入標準環境，也不屬於一鍵主流程。

## 專案結構

```text
config/       共用模型、採樣、特徵與實驗設定
data/raw/     版本化來源資料
experiments/  Study 1–4 與 rolling 實驗
results/      版本化原始結果、摘要、統計與 provenance
scripts/      執行、分析、驗證與繪圖工具
src/          可重用資料、模型、集成與評估模組
tests/        核心契約與資料洩漏防護測試
docs/         研究規格、研究方向與教授報告
thesis/       論文草稿
```

## 執行

先驗證完整維護清單：

```bash
python scripts/run/run_all_experiments.py --list
```

按階段執行，例如：

```bash
python scripts/run/run_all_experiments.py --phase phase1
python scripts/run/run_all_experiments.py --phase phase3
python scripts/run/run_all_experiments.py --phase rolling --phase analysis
python scripts/run/run_all_experiments.py --phase study3b
```

完整執行會耗時數小時，且會更新 `results/`。結果不得手動修改，應由對應
實驗或分析腳本重新產生。

## 目前結論

目前保存的主要探索性證據來自 2009–2018 十個年度的 rolling walk-forward。
`New_under`（窗口由 ROSS 選擇，並非固定窗口）pooled AUC/F1 為 0.8403/0.3287；AdaptiveChoice 為
0.8221/0.3283。Holm 校正後，AdaptiveChoice 只在 AUC 上顯著優於 Equal6，
尚未顯著優於 `New_under`。因此目前支持「近期資料模型是強基準」及
「validation-guided weighting 可改善等權集成」，但不支持 DAWCE/ROSS
普遍優於強單模型。

Study 3 B 路線已有獨立實作：以衍生的一年期事件 target 比較 Recent3y、
B-data 等權與三模型 FIFO 等權集成，且每個模型保留自己的前處理。正式執行：

```bash
python experiments/phase_flexible/rolling_bankruptcy_overlap_ensemble.py --configured-seeds
```

目前已完成 2009–2018 × 10 configured seeds；因 TomekLinks 與目前 XGBoost
設定皆為決定性，10 seeds 的輸出完全相同，不能把零標準差解讀為真實世界
不確定性很低。此探索性結果中 B-model 的 pooled AP/AUC 為 0.1215/0.8576，
Recent3y 為 0.0966/0.8450。另已完成同一事件 target、特徵、sampling、模型、
年度界線及等權設定的公平 A/B run：Validation-selected A 的 AP/AUC 為
0.1056/0.8422；B-A 的 company-cluster bootstrap 區間仍跨 0，因此不可宣稱
B 顯著優於 A。

另已完成 B-model 的單因子消融。Tomek 相較不採樣的主要指標區間皆跨 0；
Pool3 相較 Pool1 的 AP/AUC/Recall 區間未跨 0，但未一致優於 Pool2/Pool5；
訓練窗口 1/3/5 年的 AP 為 0.1142/0.1215/0.1106，三年相較一年與五年的
AP/AUC 區間仍跨 0。因此目前支持「多模型保留相較單一近期模型有價值」，
不支持「Tomek、三年窗口、Pool3 是最佳組合」。

公平比較與 1,000 次公司 cluster bootstrap：

```bash
python experiments/phase_flexible/study3_ab_fair_comparison.py --seed 42 --bootstrap-replicates 1000
```

### Study 3B 持續更新 runtime

以下流程不必事先指定資料流的最終年度。`as-of-feature-year` 表示目前要處理／
預測的財報特徵年度；模型只可使用在該時點已通過年度 maturity gate 的標籤。

```bash
python experiments/phase_flexible/study3b_continual_runtime.py prepare-labels
python experiments/phase_flexible/study3b_continual_runtime.py initialize --as-of-feature-year 2009
python experiments/phase_flexible/study3b_continual_runtime.py predict --feature-year 2009
python experiments/phase_flexible/study3b_continual_runtime.py update --as-of-feature-year 2010
python experiments/phase_flexible/study3b_continual_runtime.py predict --feature-year 2010
```

模型與 registry 預設保存在本機 `checkpoints/`，不加入 Git；不含真實標籤的
逐批 predictions 與 manifest 保存在 `results/phase_flexible/study3b_live_predictions/`。
內建 label ledger 是由公開資料回溯重建的年度研究標籤，不等於已有真實
filing/report timestamp，因此 runtime 目前仍屬研究 replay，而非正式部署。

詳見 [研究成果與未來研究方向報告](docs/研究成果與未來研究方向報告.md)。
