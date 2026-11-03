"""Apply the author's diagram comment: validation-selected best model, pilot XGBoost evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
C = "thesis/latex/chapters/"


def replace(name, old, new):
    p = ROOT / name
    t = p.read_text(encoding="utf-8")
    assert t.count(old) == 1, (name, old[:80], t.count(old))
    p.write_text(t.replace(old, new), encoding="utf-8")


replace(C + "03_method.tex", "以固定 XGBoost 建立 Old 三模型與 New 三模型", "以驗證選定的最佳模型建立 Old 三模型與 New 三模型")
replace(C + "01_introduction.tex", "本研究以固定分類器建立 Old／New 六模型池，使比較能集中於模型組成與選擇方式。", "本研究框架先依共同驗證準則選定表現最佳的基礎分類器，再以該分類器建立 Old／New 六模型池，使後續比較集中於模型組成與整合方式。")
replace(C + "01_introduction.tex", "研究問題二：固定 Old／New 模型池後，靜態與動態集成的表現是否不同？", "研究問題二：以驗證選定的最佳模型建立 Old／New 模型池後，靜態與動態集成的表現是否不同？")
replace(C + "01_introduction.tex", "使用相同分類器、六個基礎模型及固定驗證資料，", "使用驗證選定的分類器、共同六模型池及驗證資料，")
replace(C + "02_literature.tex", "固定年份實驗使用 XGBoost 作為共同基底，是為了讓後續比較集中於整合規則，並非宣稱其他分類器不適用。", "研究框架不預設某一分類器，先由共同驗證準則選出最佳基礎模型及參數，再比較其集成規則；目前測試暫用 XGBoost 檢查流程，不能將此測試設定當成最佳模型的選擇結論。")
replace(C + "02_literature.tex", "其證據範圍是本次固定年份、固定 XGBoost 模型池及單一公開資料下的實證比較。", "現有證據來自單一公開資料、固定年份與 XGBoost 的測試實作；研究框架的基礎模型仍由驗證表現選定，正式選模後須重新評估。")
replace(C + "03_method.tex", r"\subsubsection{驗證結果彙整與集成基底定位}", r"\subsubsection{驗證選模與目前測試實作}")
replace(C + "03_method.tex",
    "集成與 FS 實驗以 XGBoost 為固定基底。XGBoost 為樹提升方法\\autocite{chen2016xgboost}，其二元機率輸出可直接供靜態平均及動態選擇使用；固定基底也使六模型池只因訓練期間、採樣方式與特徵條件而改變，較容易判讀集成規則與 FS 的影響。此基底在 Baseline 完成前即已決定，並未依 Test 表現回選；本研究也不宣稱 XGBoost 是最佳分類器。",
    "研究框架先在相同資料切分與共同驗證規則下完成候選分類器及超參數比較，以預先指定的驗證 ROC-AUC 選定表現最佳的基礎模型，再用選定的分類器與參數建立 Old／New 六模型池；Test 不參與最佳模型、成員或門檻的選擇。最佳模型由資料與驗證表現決定，不預先限定為 XGBoost。\n\n目前已完成的集成與 FS 僅為 XGBoost 測試實作，用來檢查模型池、整合規則與特徵選取流程，尚未依完整六分類器的共同驗證結果重建最佳模型池。XGBoost 的機率輸出可供目前測試使用\\autocite{chen2016xgboost}，但這不表示它已被選為最佳分類器。TabM 與 TabICL 的事件數據亦待更新，正式選模與集成重跑應在候選結果齊備後完成。")
replace(C + "03_method.tex",
    "實驗一的集成部分固定基礎分類器種類，分析不同資料期間與採樣方式所形成模型的整合表現。本部分使用 XGBoost 建立 Old／New 模型池，比較靜態集成、DCS 與 DES，架構如圖~\\ref{fig:method-ensemble} 所示。",
    "實驗一的集成部分使用驗證選定的最佳基礎模型，分析不同資料期間與採樣方式所形成模型的整合表現。選定後，靜態集成、DCS 與 DES 共用同一分類器與六模型池，架構如圖~\\ref{fig:method-ensemble} 所示。下列數值設定與第四章現有結果來自暫以 XGBoost 執行的流程測試，正式最佳模型選定後再更新。")
replace(C + "03_method.tex", "六個模型都使用 XGBoost，差別在於訓練年度與採樣後的資料。", "框架中的六個模型使用同一個由驗證選定的分類器，差別在於訓練年度與採樣後的資料；目前測試實作暫以 XGBoost 建立這六個位置。")
replace(C + "03_method.tex", "實驗一的集成部分與實驗二使用相同的 XGBoost 和採樣設定，", "實驗一的集成部分與實驗二沿用驗證選定的模型；目前 XGBoost 測試使用的分類器與採樣設定，")
replace(C + "03_method.tex", "池內 XGBoost 使用預設參數，即 Baseline 搜尋中的候選 0，沒有讀取 Baseline 選定的超參數；", "目前測試池內 XGBoost 使用預設參數，即 Baseline 搜尋中的候選 0，尚未接入最佳分類器及其驗證選定參數；")
replace(C + "03_method.tex", r"\caption{集成模型之主要方法與參數}", r"\caption{目前集成測試實作之方法與參數}")
replace(C + "03_method.tex", "基礎分類器 & XGBoost；", "目前測試分類器 & XGBoost；")
replace(C + "03_method.tex", "資料分區、XGBoost 設定、採樣方式與集成方法沿用實驗一，", "資料分區、驗證選定的模型、採樣方式與集成方法沿用實驗一；現有結果暫沿用 XGBoost 測試設定，")
replace(C + "04_experiments.tex", "本章主要結果使用第~\\ref{sec:method-label}~節定義的事件標籤。", "本章先呈現目前的測試結果，集成與 FS 暫以 XGBoost 檢查研究流程，尚未依完整六分類器的驗證比較重建最佳模型池。主要結果使用第~\\ref{sec:method-label}~節定義的事件標籤。")
replace(C + "04_experiments.tex", "本研究不據此改換集成基底，也不宣稱 XGBoost 為最佳分類器。", "這份四分類器彙整是目前的部分驗證證據；完整候選尚未齊備，現有 XGBoost 測試不因此改寫為已完成最佳模型選擇，正式集成池應依共同驗證規則選模後重建。")
replace(C + "05_conclusion.tex", "依三項研究問題，結論整理如下。", "目前集成與 FS 的數據來自暫用 XGBoost 的流程測試，最佳基礎模型及參數仍須依完整候選的共同驗證結果決定。以下整理本次測試對三項研究問題的觀察，正式選模後須重新驗證。")
replace(C + "appendix.tex", "集成池的 XGBoost 使用預設參數，未讀入 Baseline 選定的參數。", "目前流程測試的集成池暫用預設 XGBoost，尚未接入框架要求的最佳分類器及其驗證選定參數。")
replace("thesis/latex/frontmatter/abstract_zh.tex", "再以 XGBoost 建立 Old／New 六模型池，", "框架依驗證表現選定最佳基礎模型，目前先以 XGBoost 建立 Old／New 六模型池測試流程，")
replace("thesis/latex/frontmatter/abstract_zh.tex", "結果顯示，近期資料的優勢依分類器而異。", "目前測試顯示，近期資料的優勢依分類器而異。")
replace("thesis/latex/frontmatter/abstract_en.tex", "then evaluates static and dynamic selection using a fixed six-model XGBoost pool.", "The framework selects its base model by validation performance; current pilot results use a six-model XGBoost pool to evaluate static and dynamic selection.")
print("Revised the overview, related method descriptions and pilot-result scope; no numeric changes.")
