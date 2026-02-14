# 立即可用的醫療資料集替代方案

## 🎯 推薦：無需等待 MIMIC-III 審核

您現在有 **3 個立即可用的選項**！

---

## ✨ 選項 1: 合成醫療資料（最快）⭐ 推薦

### 特點

- ✅ **立即可用** - 無需下載
- ✅ **時間序列** - 2010-2019 (10年)
- ✅ **真實特徵** - 生命徵象、實驗室檢驗
- ✅ **不平衡** - 死亡率約 15-20%

### 使用方式

```powershell
python scripts\download_medical_data.py
# 選擇 3（或直接按 Enter）
```

**1 秒內完成！**

### 資料內容

- 120 個月資料點
- 10 個特徵（年齡、心率、血壓、血糖等）
- 目標：死亡率預測
- 位置：`data/raw/medical/synthetic/synthetic_medical_data.csv`

---

## 📊 選項 2: Diabetes 130-US Hospitals

### 特點

- ✅ **真實資料** - 1999-2008 美國醫院
- ✅ **大型資料集** - 100,000+ 筆記錄
- ✅ **豐富特徵** - 50+ 個臨床特徵
- ✅ **時間序列** - 10 年期間

### 下載方式

```powershell
# 使用 Kaggle API
kaggle datasets download -d brandao/diabetes-130-us-hospitals-for-years-1999-2008

# 或使用腳本
python scripts\download_medical_data.py
# 選擇 1
```

### 資料資訊

- **樣本數**: ~100,000
- **目標**: 再入院預測
- **不平衡**: 約 35-40% 再入院率
- **來源**: UCI Machine Learning Repository

---

## ❤️ 選項 3: Heart Disease UCI

### 特點

- ✅ **經典資料集** - 廣泛使用於研究
- ✅ **高品質** - Cleveland Clinic 資料
- ✅ **適中大小** - 快速實驗
- ✅ **不平衡** - 心臟病患病率 45%

### 下載方式

```powershell
kaggle datasets download -d johnsmith88/heart-disease-dataset

# 或使用腳本
python scripts\download_medical_data.py
# 選擇 2
```

### 資料資訊

- **樣本數**: 303
- **特徵**: 13 個臨床特徵
- **目標**: 心臟病診斷
- **來源**: UCI Repository

---

## 📋 其他快速選項

### 4. Breast Cancer Wisconsin

```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
# 569 樣本, 30 特徵, 約 37% 惡性
```

### 5. Pima Indians Diabetes

```powershell
kaggle datasets download -d uciml/pima-indians-diabetes-database
# 768 樣本, 8 特徵, 約 35% 糖尿病
```

---

## 🆚 資料集比較

| 資料集 | 下載時間 | 樣本數 | 期間 | 不平衡比例 | 推薦度 |
|--------|---------|--------|------|-----------|--------|
| **合成資料** | 1 秒 | 120 | 10年 | 15-20% | ⭐⭐⭐⭐⭐ |
| Diabetes 130 | ~30 秒 | 100K+ | 10年 | 35-40% | ⭐⭐⭐⭐ |
| Heart Disease | ~5 秒 | 303 | N/A | 45% | ⭐⭐⭐ |
| MIMIC-III | 3-5天 | 53K+ | 12年 | 10-15% | ⭐⭐⭐⭐⭐ |

---

## 🚀 快速開始（推薦流程）

### 步驟 1: 創建合成資料（1 秒）

```powershell
python scripts\download_medical_data.py
# 按 Enter（選擇 3）
```

### 步驟 2: 立即開始實驗

```python
from src.data import DataLoader
import pandas as pd

# 載入合成醫療資料
medical_data = pd.read_csv('data/raw/medical/synthetic/synthetic_medical_data.csv')

X = medical_data.drop(['date', 'mortality'], axis=1)
y = medical_data['mortality']

print(f"資料大小: {X.shape}")
print(f"死亡率: {y.mean()*100:.2f}%")

# 開始訓練！
from src.models import LightGBMWrapper
model = LightGBMWrapper(name="medical_model")
model.fit(X, y.values)
```

---

## 💡 建議策略

### 方案 A: 快速驗證（今天）

1. ✅ 使用**合成資料**（1秒）
2. ✅ 驗證所有模型和流程
3. ✅ 確保代碼正確運行

### 方案 B: 真實資料（本週）

1. 下載 **Diabetes 130** 或 **Heart Disease**
2. 重複實驗驗證穩健性
3. 同時申請 MIMIC-III

### 方案 C: 完整研究（1-2週後）

1. MIMIC-III 審核通過
2. 使用三個真實資料集
3. 完整跨領域驗證

---

## 🎯 您現在擁有的資料集

| 資料集 | 狀態 | 位置 |
|--------|------|------|
| Bankruptcy | ✅ 就緒 | `data/raw/bankruptcy/` |
| Stock | ✅ 就緒 | `data/raw/stock/` |
| Medical (合成) | 🔄 1秒可用 | 執行腳本即可 |

---

## ⚡ 立即執行

```powershell
# 創建醫療合成資料
python scripts\download_medical_data.py

# 您就有完整的 3 個資料集了！
# Bankruptcy + Stock + Medical ✅
```

**總時間**: 1 秒
**結果**: 可立即開始完整實驗！

---

**推薦**: 先用合成資料完成所有實驗，同時申請 MIMIC-III。等審核通過後再用真實 MIMIC-III 資料重新驗證結果！這樣可以**立即開始**，不浪費等待時間。🚀
