# MIMIC-III 申請完整指南

## 📋 概述

MIMIC-III (Medical Information Mart for Intensive Care III) 是一個大型、公開的 ICU 資料集，包含約 40,000 名重症監護病患的去識別化健康資料。

## ⚠️ 重要事項

- **審核時間**: 1-3 個工作天
- **必要條件**: 完成 CITI 訓練課程（~3-4 小時）
- **建議**: **提前申請**，不要等到需要時才申請

---

## 🚀 申請流程（詳細步驟）

### 步驟 1: 註冊 PhysioNet 帳號

1. **前往註冊頁面**
   - URL: <https://physionet.org/register/>

2. **填寫註冊表單**
   - Email: **務必使用學校 email** (.edu.tw 或學校提供的郵箱)
   - First Name: 您的名字
   - Last Name: 您的姓氏
   - Organization: 您的學校全名（例如：National Taiwan University）
   - 勾選 "I agree to the terms and conditions"

3. **驗證 Email**
   - 檢查收件匣（可能在垃圾郵件）
   - 點擊驗證連結
   - 完成帳號啟用

---

### 步驟 2: 完成 CITI 訓練課程

這是**最重要**的步驟！

#### 2.1 註冊 CITI Program

1. 前往: <https://about.citiprogram.org/>
2. 點擊 "Register"
3. 選擇機構：
   - 搜尋您的學校
   - 如果找不到，選擇 "Independent Learner"

#### 2.2 選擇課程

必須完成以下課程：

- **課程名稱**: "Data or Specimens Only Research"
- **類型**: Biomedical Research
- **時間**: 約 3-4 小時
- **內容**: 研究倫理、資料保護、隱私規範

#### 2.3 完成課程

- 可分多次完成
- 需通過每個模組的測驗（80% 及格）
- 完成後下載證書（PDF）

#### 2.4 關鍵提示

✅ **重要**:

- 課程可以暫停，不需要一次完成
- 測驗可以重考
- 務必保存證書 PDF

---

### 步驟 3: 申請 MIMIC-III 訪問權限

1. **登入 PhysioNet**
   - URL: <https://physionet.org/login/>

2. **前往 MIMIC-III 頁面**
   - URL: <https://physionet.org/content/mimiciii/1.4/>

3. **點擊 "Request Access"**

4. **填寫申請表單**

   **必填欄位**:

   - **Course Completed**: 選擇 "Data or Specimens Only Research"
   - **Reference ID**: 輸入 CITI 證書上的 ID
   - **Completion Report**: 上傳 CITI 證書 PDF

   - **Research Summary** (研究摘要):

     ```
     標題: Continual Learning with Class Imbalance in Medical Time Series
     
     目的: This research aims to develop a dynamic ensemble selection framework 
     for handling class imbalance in non-stationary medical data, specifically 
     for ICU mortality prediction using continual learning approaches.
     
     方法: We will use MIMIC-III ICU data to evaluate our proposed ensemble 
     methods combining historical and new operating models with various 
     sampling strategies (SMOTEENN, ADASYN).
     
     預期成果: A robust machine learning framework for medical prediction 
     tasks under data drift and class imbalance conditions.
     ```

   - **Supervisor**: 填寫您的指導教授資訊
   - **Purpose**: 選擇 "Masters Thesis"

5. **提交申請**
   - 仔細檢查所有資訊
   - 點擊 "Submit"

---

### 步驟 4: 等待審核

- **審核時間**: 通常 1-3 個工作天
- **狀態查詢**: 登入後在 "Projects" 頁面查看
- **通過後**: 會收到 Email 通知

---

### 步驟 5: 下載資料

審核通過後，有幾種下載方式：

#### 方法 1: 使用 wget（Linux/Mac）

```bash
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
  https://physionet.org/files/mimiciii/1.4/
```

#### 方法 2: 使用 Google BigQuery（推薦 Windows 用戶）

```python
# 安裝套件
pip install google-cloud-bigquery pandas-gbq

# Python 腳本
from google.cloud import bigquery

client = bigquery.Client()

query = """
SELECT *
FROM `physionet-data.mimiciii_clinical.icustays`
LIMIT 1000
"""

df = client.query(query).to_dataframe()
```

#### 方法 3: 使用 MIMIC-III Demo（示範資料）

如果只是先測試，可以使用公開的示範資料：

```powershell
# 下載示範資料（100位病人）
kaggle datasets download -d drscarlat/mimic3d
```

---

## 📊 資料結構

MIMIC-III 包含多個表：

| 表名 | 內容 | 用途 |
|------|------|------|
| `patients` | 病患基本資訊 | 年齡、性別 |
| `admissions` | 住院記錄 | 入院時間、診斷 |
| `icustays` | ICU 記錄 | ICU 停留時間 |
| `chartevents` | 生命徵象 | 心率、血壓等 |
| `labevents` | 實驗室檢驗 | 血液、生化數值 |
| `prescriptions` | 用藥記錄 | 藥物、劑量 |

---

## 🎯 研究使用建議

### 適合您的研究任務

✅ **ICU 死亡率預測**

- 目標: `hospital_expire_flag` 或 `expire_flag`
- 不平衡比例: 約 10-15%
- 時間跨度: 2001-2012

### 資料提取流程

1. **選擇特徵**
   - 生命徵象（前 24 小時）
   - 實驗室檢驗結果
   - 人口統計資訊

2. **創建時間窗口**
   - Historical: 2001-2008
   - New Operating: 2009-2010
   - Testing: 2011-2012

3. **處理缺失值**
   - 使用前向填補或平均值

---

## ⏱️ 時間規劃

| 步驟 | 預計時間 |
|------|---------|
| 註冊 PhysioNet | 5 分鐘 |
| 完成 CITI 訓練 | 3-4 小時 |
| 填寫申請表 | 15 分鐘 |
| 等待審核 | 1-3 天 |
| 下載資料 | 1-2 小時 |

**總計**: 約 3-5 天（含審核時間）

---

## 💡 常見問題

### Q1: 一定要用學校 email 嗎？

**A**: 強烈建議，審核通過率較高。如使用個人 email，需額外說明研究背景。

### Q2: CITI 課程可以分次完成嗎？

**A**: 可以！課程進度會自動保存。

### Q3: 如果審核被拒絕怎麼辦？

**A**: 檢查 CITI 證書、研究摘要是否完整，修改後重新提交。

### Q4: 資料很大，下載很慢怎麼辦？

**A**:

- 使用 Google BigQuery（只下載需要的表）
- 或使用 MIMIC-III Demo 先測試

### Q5: 我可以公開使用這些資料嗎？

**A**:

- ✅ 可以發表研究成果
- ❌ 不可公開原始資料
- ✅ 可以分享處理後的特徵

---

## 📞 聯絡資訊

遇到問題？

- **PhysioNet Support**: <physionet-support@mit.edu>
- **CITI Support**: <https://support.citiprogram.org/>
- **MIMIC Community**: <https://github.com/MIT-LCP/mimic-code/issues>

---

## ✅ 檢查清單

申請前確認：

- [ ] 註冊 PhysioNet（使用學校 email）
- [ ] 完成 CITI 訓練
- [ ] 下載 CITI 證書 PDF
- [ ] 準備研究摘要（英文）
- [ ] 準備指導教授資訊
- [ ] 填寫完整申請表
- [ ] 提交申請

---

**建議**: 立即開始 CITI 訓練，這樣即使現在不需要 MIMIC-III，未來也隨時可用！

**祝申請順利！** 🎉
