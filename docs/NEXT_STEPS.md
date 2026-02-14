# 下一步實驗指南

## 🎉 恭喜！第一個實驗完成

您已經完成了 **Bankruptcy Baseline 實驗**！

---

## 📊 查看實驗結果

### 1. 結果檔案

```powershell
# 查看 CSV 結果
type results\baseline\bankruptcy_baseline_results.csv

# 或用 Excel/試算表打開
start results\baseline\bankruptcy_baseline_results.csv
```

### 2. 詳細日誌

```powershell
# 查看完整日誌
Get-ChildItem logs -Filter "*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | cat
```

### 3. 快速分析結果

創建一個簡單的結果視覺化：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 讀取結果
results = pd.read_csv('results/baseline/bankruptcy_baseline_results.csv', index_col=0)

# 顯示結果
print(results)

# 繪製比較圖
results.plot(kind='bar', figsize=(10, 6))
plt.title('Bankruptcy Baseline Methods Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('results/baseline/bankruptcy_comparison.png')
print("圖表已保存到: results/baseline/bankruptcy_comparison.png")
```

---

## 🚀 下一步選項

### 選項 A: 完成所有資料集的 Baseline（推薦）

#### 1. Stock 資料集 Baseline

```powershell
# 創建 Stock baseline 實驗
# （複製並修改 01_bankruptcy_baseline.py）
python experiments\02_stock_baseline.py
```

#### 2. Medical 資料集 Baseline

```powershell
python experiments\03_medical_baseline.py
```

**目的**: 驗證方法在不同領域的通用性

---

### 選項 B: 深入 Bankruptcy - Ensemble 實驗

#### 1. 不同模型組合

測試不同的 ensemble 組合：

- 2 models vs 3 models vs 6 models
- Old only vs New only vs Mixed
- 不同 sampling 策略組合

#### 2. 創建進階實驗腳本

```powershell
python experiments\04_bankruptcy_ensemble.py
```

**目的**: 找出最佳 ensemble 配置

---

### 選項 C: Feature Selection 研究

#### 1. 使用 mRMR 特徵選擇

```powershell
python experiments\05_bankruptcy_feature_selection.py
```

#### 2. 比較不同特徵數量

- Top 20 features
- Top 50 features  
- All features

**目的**: 減少特徵維度，提升效率

---

### 選項 D: 參數調整

#### 1. 調整 LightGBM 參數

編輯 `config/model_config.yaml`:

```yaml
lightgbm:
  base_params:
    learning_rate: 0.05  # 試試降低學習率
    num_leaves: 63       # 增加樹的複雜度
    max_depth: 8
```

#### 2. 調整 Sampling 策略

編輯 `config/sampling_config.yaml`

#### 3. 重新運行實驗

```powershell
python experiments\01_bankruptcy_baseline.py
```

**目的**: 優化模型性能

---

## 📋 推薦的實驗順序

### 第 1 週（當前）

1. ✅ Bankruptcy Baseline（已完成）
2. ⏳ Stock Baseline
3. ⏳ Medical Baseline
4. ⏳ 比較三個資料集的結果

**輸出**: 基準性能表格

### 第 2 週

1. Bankruptcy Ensemble 實驗
2. Stock Ensemble 實驗  
3. Medical Ensemble 實驗
4. 分析 Ensemble 效果

**輸出**: Ensemble vs Baseline 比較

### 第 3 週

1. Feature Selection 研究
2. 參數優化
3. 跨資料集分析

**輸出**: 最佳配置和完整結果

### 第 4 週

1. 撰寫論文實驗章節
2. 製作結果圖表
3. 統計檢定

**輸出**: 論文初稿

---

## 💡 立即行動（今天）

### 推薦路徑 1: 快速完成所有 Baseline

```powershell
# 1. 檢查 Bankruptcy 結果
type results\baseline\bankruptcy_baseline_results.csv

# 2. 創建 Stock baseline（複製腳本並修改）
# 修改資料載入路徑為 Stock

# 3. 執行 Stock baseline
python experiments\02_stock_baseline.py

# 4. 創建 Medical baseline
python experiments\03_medical_baseline.py
```

**時間**: ~30 分鐘
**成果**: 3 個資料集的基準結果

### 推薦路徑 2: 深入分析當前結果

```powershell
# 1. 視覺化結果
pip install matplotlib
python -c "import pandas as pd; import matplotlib.pyplot as plt; df = pd.read_csv('results/baseline/bankruptcy_baseline_results.csv', index_col=0); df.plot(kind='bar'); plt.savefig('results/baseline/comparison.png')"

# 2. 分析哪個方法最好
# 3. 調整參數再試一次
```

**時間**: ~1 小時
**成果**: 深入理解結果

---

## 🎯 建議

**我的推薦**: 選擇**推薦路徑 1**

**原因**:

1. 快速建立所有資料集的 baseline
2. 可以比較跨領域性能
3. 為論文提供完整的實驗數據
4. 之後可以針對性優化

**今天的目標**:

- ✅ Bankruptcy Baseline（已完成）
- ⏳ Stock Baseline
- ⏳ Medical Baseline
- ⏳ 創建結果比較表

---

## 📞 需要幫助？

如果您想：

1. 我幫您創建 Stock 和 Medical 的 baseline 腳本
2. 分析當前結果
3. 創建視覺化圖表
4. 調整配置參數

隨時告訴我！

---

**下一步建議**: 執行 Stock baseline 實驗 🚀
