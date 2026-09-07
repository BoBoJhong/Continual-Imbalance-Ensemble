# Continual Imbalance Ensemble

本專案研究非平穩、類別不平衡資料中的持續式集成學習，主線以美國企業
破產資料比較歷史模型、近期模型、重訓練、靜態集成、DES/DCS、特徵選擇、
ROSS、DAWCE 與 rolling walk-forward 選擇流程。

完整的研究問題、Study 流程、實驗結果、限制與待辦收錄於
[docs/研究方向.md](docs/研究方向.md)；供指導教授討論的成果摘要見
[研究成果與未來研究方向報告](docs/研究成果與未來研究方向報告.md)。`data/raw/` 與 `results/` 均刻意納入
Git，以保留來源與研究證據。

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
```

完整執行會耗時數小時，且會更新 `results/`。結果不得手動修改，應由對應
實驗或分析腳本重新產生。

## 目前結論

目前最可信的證據來自 2009–2018 十個年度的 rolling walk-forward。固定
`New_under` pooled AUC/F1 為 0.8403/0.3287；AdaptiveChoice 為
0.8221/0.3283。Holm 校正後，AdaptiveChoice 只在 AUC 上顯著優於 Equal6，
尚未顯著優於 `New_under`。因此目前支持「近期資料模型是強基準」及
「validation-guided weighting 可改善等權集成」，但不支持 DAWCE/ROSS
普遍優於強單模型。

詳見 [研究成果與未來研究方向報告](docs/研究成果與未來研究方向報告.md)。
