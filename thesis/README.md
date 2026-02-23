# 碩士論文稿

本目錄包含依專案實驗整理之**完整碩士論文稿**，供繳交前依貴校格式調整使用。

## 檔案

- **THESIS_FULL.md**：完整論文內容（摘要、第一至五章、參考文獻）。  
  - **已納入真實文獻引用**（APA 第七版）：持續學習、概念漂移、類別不平衡/SMOTE、動態選擇/KNORA-E、破產預測、LightGBM 等均有對應引用；參考文獻列表為完整書目。  
  - **篇幅已擴充**：緒論含研究問題、文獻探討分小節並加引用、研究方法與實驗設計分節細化、結論與貢獻/限制擴寫；整體字數與章節長度較初稿明顯增加。  
  - 表內數值來自 `results/` 下之 CSV，與目前實驗結果一致。  
  - 可依貴校要求調整章節編號、頁首頁尾、參考文獻格式（APA / IEEE 等）。

## 中央大學資管格式

若為**國立中央大學資管所**，請參照同目錄 **`NCU_IM_FORMAT.md`**，內含學校常見裝訂順序、版面、頁碼、參考文獻與系所建議確認事項；最終以教務處與系所最新公告為準。

---

## 使用方式

1. **轉成 Word / PDF**  
   - 用 VS Code、Pandoc 或線上工具將 `THESIS_FULL.md` 轉成 `.docx` 或 `.pdf`。  
   - 例（Pandoc）：`pandoc thesis/THESIS_FULL.md -o thesis/THESIS_FULL.docx`

2. **依貴校規定調整**  
   - 字數／頁數：可精簡第二章文獻或第四章部分表格。  
   - 英文摘要：可將摘要翻譯後置於中文摘要之後。  
   - 圖表編號：統一為圖 4-1、表 4-1 等。  
   - 參考文獻：補齊卷期頁碼並改為貴校指定格式。

3. **對照與口試**  
   - 老師方向逐條對照：`docs/TEACHER_REQUIREMENTS_CHECKLIST.md`  
   - 實驗正當性：`docs/EXPERIMENT_VALIDATION.md`  
   - 碩論適用性：`docs/THESIS_READINESS.md`

## 資料來源

- 方法與切割：`docs/EXPERIMENT_VALIDATION.md`、`docs/TEACHER_REQUIREMENTS_CHECKLIST.md`  
- 結果表：`results/bankruptcy_all_results.csv`、`results/summary_all_datasets.csv`、`results/des_advanced/`、`results/proportion_study/`、`results/feature_study/`  
- 解讀：`docs/RESULTS_09_10_INTERPRETATION.md`、`docs/THESIS_READINESS.md`
