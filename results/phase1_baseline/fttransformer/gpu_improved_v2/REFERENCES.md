# FT-Transformer 改進版實驗（`gpu_improved_v2`）— 參考文獻與設定對照

本資料夾結果對應程式：`experiments/phase1_baseline/bankruptcy_year_splits_fttransformer.py`  
建議執行：`--device auto --results-subdir gpu_improved_v2`（有 NVIDIA GPU 時會用 CUDA + AMP）。

---

## 1. 核心模型（FT-Transformer）

**Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko**  
*Revisiting Deep Learning Models for Tabular Data*  
Advances in Neural Information Processing Systems (**NeurIPS**), **34**, 2021, 18932–18943.

- **arXiv：** <https://arxiv.org/abs/2106.11959>  
- **NeurIPS Proceedings：** <https://proceedings.neurips.cc/paper/2021/hash/9d86d5f2a7b543151d3c2f5a1e0a59eb-Abstract.html>  
- **官方程式碼（Yandex Research）：** <https://github.com/yandex-research/rtdl>  
- **本專案實作：** `pip install rtdl` → `rtdl.FTTransformer`（見 `src/models/fttransformer_wrapper.py`）

---

## 2. 混合精度訓練（AMP）

**Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu**  
*Mixed Precision Training*  
International Conference on Learning Representations (**ICLR**), **2018**.

- **arXiv：** <https://arxiv.org/abs/1710.03740>  

本專案在 **CUDA** 上預設使用 `torch.autocast` + `GradScaler`（對應現代 PyTorch AMP API）；執行時可加 `--no-amp` 關閉。

- **PyTorch Automatic Mixed Precision：** <https://pytorch.org/docs/stable/amp.html>

---

## 3. 優化器（AdamW）

**Ilya Loshchilov, Frank Hutter**  
*Decoupled Weight Decay Regularization*  
**ICLR** **2019**.

- **arXiv：** <https://arxiv.org/abs/1711.05101>  

本專案訓練迴圈使用 `torch.optim.AdamW`（與權重衰減 `weight_decay` 一併使用）。

---

## 4. 類別不平衡：二元 BCE 與正類權重

**PyTorch** `torch.nn.BCEWithLogitsLoss` 之 **`pos_weight`**：依少數類樣本數對正類損失加權（本專案預設 `pos_weight = n_neg / n_pos`，與 XGBoost `scale_pos_weight` 常見設定對齊概念）。

- **文件：** <https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html>

**（選讀）Gary King, Langche Zeng**  
*Logistic Regression in Rare Events Data*  
*Political Analysis*, **9** (2), 2001, 137–163.

- **論文連結：** <https://gking.harvard.edu/files/0s.pdf>  
  （稀有事件、加權與抽樣之統計討論；實務上常與 `pos_weight` / `scale_pos_weight` 並列參考。）

---

## 5. 資料載入與記憶體（GPU 友善批次）

驗證／推論採**分批**計算 loss 與 `predict_proba`，並在 CUDA 下使用 `DataLoader(..., pin_memory=True)` 等慣例，屬 PyTorch 官方建議之工程實務。

- **DataLoader：** <https://pytorch.org/docs/stable/data.html>

---

## 本輪實驗與舊版差異摘要

| 項目 | 說明 |
|------|------|
| 輸出目錄 | `fttransformer/gpu_improved_v2/`（與根目錄舊結果分開） |
| Finetune | **不執行**（僅 Old / New / Retrain；raw 列數見腳本註解） |
| 裝置 | `--device auto`：CUDA → MPS → CPU |
| 損失 | `BCEWithLogitsLoss`，`pos_weight` 依訓練集自動 |
| 其它 | AMP（可關）、驗證／推論分批、`pin_memory`（CUDA） |

---

## BibTeX（論文寫作用，可貼 Overleaf）

```bibtex
@inproceedings{gorishniy2021revisiting,
  title     = {Revisiting Deep Learning Models for Tabular Data},
  author    = {Gorishniy, Yury and Rubachev, Ivan and Khrulkov, Valentin and Babenko, Artem},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {34},
  pages     = {18932--18943},
  year      = {2021},
}

@inproceedings{micikevicius2018mixed,
  title     = {Mixed Precision Training},
  author    = {Micikevicius, Paulius and others},
  booktitle = {ICLR},
  year      = {2018},
}

@inproceedings{loshchilov2019decoupled,
  title     = {Decoupled Weight Decay Regularization},
  author    = {Loshchilov, Ilya and Hutter, Frank},
  booktitle = {ICLR},
  year      = {2019},
}
```

---

*此檔僅供寫作與重現對照；數值結果以本目錄內 CSV 為準。*
