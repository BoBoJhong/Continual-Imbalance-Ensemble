# Scripts 說明

本目錄為**工具腳本**（非單一實驗），從專案根目錄執行。

| 腳本 | 用途 | 產出 |
|------|------|------|
| **run_all_experiments.py** | 依序執行 experiments/01~08 | results/、logs/ |
| **compare_baseline_ensemble.py** | Bankruptcy 合併 baseline + ensemble + DES | results/bankruptcy_all_results.csv |
| **compare_all_results.py** | 三資料集彙總、各資料集最佳 AUC | results/summary_all_datasets.csv |
| **run_multi_seed.py** | Bankruptcy baseline 多 seed → mean±std | results/baseline/bankruptcy_baseline_mean_std.csv |
| **download_medical_data.py** | 下載 Medical 資料（可選） | data/raw/ |
| **download_stock_data.py** | 下載 Stock 資料（可選） | data/raw/ |

單一實驗請直接執行 `experiments/01_*.py` 等，見 **docs/STRUCTURE.md** 與 **EXECUTION_GUIDE.md**。
