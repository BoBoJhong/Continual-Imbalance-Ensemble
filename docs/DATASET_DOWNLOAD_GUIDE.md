# 資料集下載指南

## 📥 1. Bankruptcy 資料集 ✅ 已完成

### 狀態

✅ **已下載並解壓**

### 資料位置

```
data/raw/bankruptcy/data.csv
```

### 資料資訊

- **來源**: Kaggle - Company Bankruptcy Prediction
- **期間**: 1999-2009（台灣經濟期刊）
- **樣本數**: 6,819 家公司
- **特徵數**: 95 個財務指標
- **目標變數**: Bankrupt? (0=正常, 1=破產)
- **不平衡比例**: 約 3% 破產率

### 下一步

使用 DataLoader 載入：

```python
from src.data import DataLoader

loader = DataLoader()
X, y = loader.load_bankruptcy("data/raw/bankruptcy/data.csv")
```

---

## 📊 2. Stock 市場資料 🔄 準備中

### 下載方式

#### 方法 1: 自動下載（推薦）✨

```powershell
# 1. 安裝 yfinance
pip install yfinance

# 2. 執行下載腳本
python scripts\download_stock_data.py
```

**這會自動**：

- 下載 S&P 500 指數（2000-2020，20年資料）
- 計算技術指標（SMA, RSI, Volatility）
- 創建崩盤事件標籤
- 保存到 `data/raw/stock/stock_data.csv`

#### 方法 2: Kaggle 資料集

```powershell
# Stock Market Crash Prediction
kaggle datasets download -d paultimothymooney/stock-market-data

# 或其他股票資料集
kaggle datasets search "stock market crash"
```

### 資料特點

- **期間**: 2000-2020（可調整）
- **特徵**: 價格、成交量、技術指標
- **目標**: 崩盤事件（20天內跌幅>5%）
- **更新頻率**: 每日

---

## 🏥 3. MIMIC-III 醫療資料 ⚠️ 需要申請

### 申請流程

#### 步驟 1: 註冊 PhysioNet

1. 前往: <https://physionet.org/>
2. 點擊 "Sign up"
3. 填寫個人資訊（使用學校 email）

#### 步驟 2: 完成 CITI 訓練

1. 登入 PhysioNet 後，前往: <https://physionet.org/content/mimiciii/>
2. 點擊 "Request Access"
3. 完成 CITI "Data or Specimens Only Research" 課程
   - 約需 3-4 小時
   - 線上課程，可分次完成
   - 課程連結: <https://about.citiprogram.org/>

#### 步驟 3: 提交申請

1. 上傳 CITI 訓練證書
2. 填寫研究目的（可以寫：碩士論文研究）
3. 等待審核（通常 1-3 個工作天）

#### 步驟 4: 下載資料

審核通過後：

```bash
# 使用 wget 下載（Linux/Mac）
wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimiciii/1.4/

# Windows 可使用 Cloud 版本或 Google BigQuery
```

### 資料特點

- **樣本數**: 53,423 次 ICU 住院
- **期間**: 2001-2012
- **特徵**: 生命徵象、實驗室檢驗、用藥記錄
- **目標**: 死亡率預測
- **檔案大小**: 約 6 GB

### 替代方案（如果無法取得 MIMIC-III）

1. **MIMIC-III Demo**: 公開的示範資料（100位病人）

   ```powershell
   kaggle datasets download -d drscarlat/mimic3d
   ```

2. **其他醫療資料集**:
   - UCI Heart Disease Dataset
   - Diabetes 130-US hospitals dataset

---

## 📋 資料集摘要

| 資料集 | 狀態 | 大小 | 期間 | 不平衡比例 |
|--------|------|------|------|-----------|
| Bankruptcy | ✅ 完成 | 6,819 | 1999-2009 | 3% |
| Stock | 🔄 準備中 | ~5,000 | 2000-2020 | 5-10% |
| Medical | ⏳ 待申請 | 53,423 | 2001-2012 | 10-15% |

---

## 🚀 快速開始

### 立即可用（Bankruptcy）

```python
from src.data import DataLoader

loader = DataLoader()
X, y = loader.load_bankruptcy()
print(f"資料大小: {X.shape}")
print(f"類別分布: {y.value_counts()}")
```

### 下載 Stock 資料

```powershell
pip install yfinance
python scripts\download_stock_data.py
```

### Medical 資料

先完成 CITI 訓練並申請 MIMIC-III 訪問權限

---

## ⚠️ 注意事項

1. **Bankruptcy**: 資料已就緒，可立即開始實驗
2. **Stock**: 需要網路連線下載，約 1-2 分鐘
3. **Medical**: 需要 1-3 天審核時間，提前申請

## 📞 需要幫助？

- Bankruptcy 資料問題：檢查 `data/raw/bankruptcy/data.csv` 是否存在
- Stock 下載失敗：確認網路連線，或使用 Kaggle 替代方案
- Medical 申請問題：確保使用學校 email，研究目的寫清楚

---

**建議順序**: Bankruptcy (立即) → Stock (1-2分鐘) → Medical (1-3天後)
