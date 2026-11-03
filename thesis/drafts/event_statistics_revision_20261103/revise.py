"""Apply the authorized local revision; original source files are backed up beside this file."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[3]


def replace(name, old, new):
    p = ROOT / name
    text = p.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise ValueError((name, text.count(old), old[:100]))
    p.write_text(text.replace(old, new), encoding="utf-8")


chap = "thesis/latex/chapters/"
replace(chap + "01_introduction.tex",
    "結果同時列出合併四年、各年度與十個種子的分數。事件標籤的公司群聚配對檢定尚未完成，主要結果以描述性比較呈現；最終狀態標籤已完成以公司為重抽單位的配對檢定與多重比較校正，作為對照。",
    "結果同時列出合併四年、各年度與十個種子的分數。事件標籤與最終狀態標籤各自完成以公司為重抽單位的配對檢定及多重比較校正；前者為主要分析，後者為標籤對照。兩批分析均為固定模型與資料切分下的事後探索性分析，Baseline 不納入集成與 FS 的檢定範圍。")

replace(chap + "02_literature.tex",
    "本研究將 TabM 納入調參後的 Baseline 比較；其模型內部集成與後續 Old／New 模型池的整合屬於不同層次。",
    "TabM 的模型內部集成與本研究 Old／New 模型池的整合屬於不同層次。本文僅將其列為相關方法背景；事件標籤版本尚未執行，故未納入主要 Baseline 比較。")
replace(chap + "02_literature.tex",
    "本研究各分類器的調參預算並非完全相同，因此結果只作本資料下的描述性比較。TabICL 的既有探索紀錄未保存與其他分類器相同的驗證選參資訊，故不列入第四章的調參結果表；其補充設定及執行紀錄限制集中於附錄說明。",
    "本研究主要 Baseline 的四種分類器，在每個訓練策略與採樣組合下均評估 24 組候選參數；候選數相同，但搜尋空間與計算成本不同，不能解釋為各模型均已充分調參。先前 TabM 與 TabICL 的探索紀錄使用不同標籤或驗證方式，故不列入事件標籤結果表；執行範圍與限制於附錄說明。")
replace(chap + "02_literature.tex",
    "本研究將最終狀態標籤事後補充的 165 項比較放在同一校正範圍，完整保留增加、下降及未顯著的結果；",
    "本研究對事件標籤與最終狀態標籤分別進行事後分析，各將兩項實驗的 165 項比較放在同一校正範圍；兩種標籤的校正家族分開，完整保留增加、下降及未顯著的結果；")
replace(chap + "02_literature.tex",
    "上述文獻分別提供採樣、時間變化、動態選擇與 FS 的依據。本研究將這些既有方法放入兩項實驗，著重回答：在本資料與固定流程下，哪些分數差異能被觀察到，又有哪些差異仍缺乏足夠證據。實驗一以相同年份的 Baseline 作為單一分類器基準，另用固定六模型池比較集成；實驗二沿用同一設定加入 FS，檢查效能與特徵數的變化。研究定位是有控制條件的實證比較，不是宣稱已提出新的採樣、集成或解釋演算法。",
    "上述文獻分別說明採樣、時間變化、動態選擇與 FS 的原理，但這些原理本身不能回答本資料中的歷史模型是否仍有互補資訊，也不能保證縮減特徵後的局部選擇更可靠。本研究將待回答的實證問題具體化為三項對照：以 Old、New 與 Retrain 觀察資料期間的差異；讓靜態與動態方法共用六模型池、DSEL 及驗證決策規則；再於相同測試公司上比較 FS 與 no-FS。事件標籤與最終狀態標籤另行對照，以檢查方法排序是否依賴目標定義。\n\n本研究的補充價值在於將這些問題置於可追溯的共同流程，並以公司層級的配對不確定性限制結論。此定位不代表既有文獻從未結合上述方法，也不主張新的演算法或跨資料集的普遍優越性；其證據範圍是本次固定年份、固定 XGBoost 模型池及單一公開資料下的實證比較。")

replace(chap + "03_method.tex",
    "為評估方法間差異的不確定性，本研究針對最終狀態標籤的固定年份集成結果，補作以公司為重抽單位的配對群聚 bootstrap；事件標籤結果的相同分析尚未完成，第四章僅作描述性比較。",
    "為評估方法間差異的不確定性，本研究分別對事件標籤與最終狀態標籤的固定年份集成結果，補作以公司為重抽單位的配對群聚 bootstrap。兩種標籤各自分析與校正，事件標籤為主要分析，最終狀態標籤為對照。")
replace(chap + "03_method.tex",
    r"本研究使用 49,999 次重抽。為使統計分析可重現，bootstrap 重抽樣的隨機種子設為 20261027。",
    r"事件標籤分析使用 9,999 次重抽，最終狀態標籤對照使用 49,999 次，兩批皆保留既有分析規格，未因顯著性結果改變重抽次數。為使統計分析可重現，bootstrap 重抽樣的隨機種子皆設為 20261027。")
replace(chap + "03_method.tex", r"\qquad B=49{,}999.", r"\qquad B\in\{9{,}999,49{,}999\}.")
replace(chap + "03_method.tex",
    "分子與分母各加一，可避免報告零機率，最小可報值為 0.00002。",
    "分子與分母各加一，可避免報告零機率；事件標籤與最終狀態標籤的最小可報值分別為 0.0001 與 0.00002。尾端重抽次數很少時，校正後判斷可能對 Monte Carlo 誤差敏感。")
replace(chap + "03_method.tex", "兩項實驗共 165 個", "每一種標籤下的兩項實驗共 165 個")
# Repair a pre-existing split reference control sequence at the end of this section.
p = ROOT / (chap + "03_method.tex")
t = p.read_text(encoding="utf-8")
start = t.index("結果應限於本次資料與設定解讀，檢定結果於第~")
t = t[:start] + r"結果應限於本次資料與設定解讀。事件標籤的檢定於第~\ref{sec:results-event-statistics}~節摘要，兩種標籤的完整比較分列於附錄。" + "\n\\endgroup\n"
p.write_text(t, encoding="utf-8")

replace(chap + "04_experiments.tex",
    r"事件標籤下的公司群聚配對檢定尚未完成，因此第~\ref{sec:results-baseline}~至第~\ref{sec:results-experiment2}~節僅呈現描述性結果，不作顯著性判斷。第~\ref{sec:results-label}~節再以最終狀態標籤對照，說明主要結論是否受標籤定義影響；該標籤已完成的 165 項檢定列於附錄~\ref{sec:stats-full}。",
    r"事件標籤已完成以公司為單位、9,999 次重抽的配對分析，兩項實驗的 165 項檢定共同採 Holm 校正；摘要見第~\ref{sec:results-event-statistics}~節，完整結果見附錄~\ref{sec:event-stats-full}。Baseline 與集成的參數與驗證資料不同，其比較仍屬描述性。第~\ref{sec:results-label}~節另以最終狀態標籤對照，該標籤的 49,999 次重抽與 165 項校正另成一個分析家族，列於附錄~\ref{sec:stats-full}。")
replace(chap + "04_experiments.tex",
    "Logistic Regression 的驗證排名較前，但 Test AUC 最低，可見只有 33 筆事件的驗證資料不足以穩定區分分類器。",
    "Logistic Regression 的驗證排名較前，但 Test AUC 最低，顯示本次驗證排名未直接延續至測試期；驗證資料只有 33 筆事件，可能增加排名的不確定性，但本研究未分離其與模型、期間變化的影響。")
replace(chap + "04_experiments.tex",
    "綜合而言，在固定的六模型池中，「選擇哪些成員」比「是否整合全部成員」更影響結果：只用近期模型的 New3 與只選兩個成員的 V-static 表現較佳，全部等權平均與動態選擇則未帶來改善。這些差異尚未經事件標籤的配對檢定，屬描述性觀察。",
    r"綜合而言，在本次六模型池中，New3 的平均 AUC 與 V-static 的平均 F1-score 分別較高，納入全部成員與使用動態選擇未在這兩項指標上取得最高平均值。公司配對檢定支持兩者相對 Equal6 的 AUC 增加，但 V-static 的 F1-score 增加未通過 165 項 Holm 校正，詳見第~\ref{sec:results-event-statistics}~節。本研究沒有將成員品質與整合規則分開控制，因而不比較兩種因素的相對重要性。")
replace(chap + "04_experiments.tex", "調參後的單一模型也與集成相當，例如", "調參後的單一模型亦有接近或高於集成的描述性分數，例如")
replace(chap + "04_experiments.tex", "均不低於 V-static 的 0.8472。", "兩個描述性平均值均高於 V-static 的 0.8472；前者為四種採樣的平均，不能當成單一可部署模型，也未作此跨設定差異的顯著性或等效檢定。")
insert = r"""\section{事件標籤之公司群聚配對分析}\label{sec:results-event-statistics}

本節依第三章的既定比較，以 3,700 家測試公司為重抽單位；其中 119 家各有一筆事件。所有種子與方法共用相同公司重抽權重，模型與驗證門檻保持不變。信賴區間描述固定模型在重抽測試公司下的差值變化；顯著性以事件標籤全部 165 項比較共同校正後的 \(p_H\) 判斷，不能由逐項區間是否跨零單獨決定。

\subsection{集成方法相對 Equal6}

\input{generated/event-statistics-20261103/experiment1-auc-tests.tex}

表~\ref{tab:event-stats-exp1-auc} 顯示，New3 相對 Equal6 的 AUC 增加 0.0719，逐項 95\% 信賴區間為 [0.0381, 0.1054]；V-static 增加 0.0591，區間為 [0.0292, 0.0885]，兩者的 \(p_H\) 均為 0.0165。Old3 的 AUC 則顯著較低。其餘四種動態方法的 AUC 差異皆未通過校正；未達顯著不表示與 Equal6 等效。本分析未設定 New3 與 V-static 的直接檢定，因此不能由兩者相對共同基準的結果推論彼此顯著不同。

V-static 相對 Equal6 的 F1-score 增加 0.0754，逐項區間為 [0.0194, 0.1316]，但未校正 \(p=0.0093\)、\(p_H=0.9672\)，不構成校正後顯著改善。這說明最高平均 F1-score 與整體多重比較支持的改善是不同判斷。兩種 DCS 的 F1-score 顯著下降；New3 與四種動態方法的 Recall 均顯著增加，但不能忽略同時發生的誤報與 F1-score 變化。實驗一共 12 項檢定通過校正，其中 AUC、F1-score、Recall 分別為 3、3、6 項，方向包含增加與下降。

\subsection{FS 相對相同方法 no-FS}

實驗二的 48 項 AUC 比較中，沒有校正後顯著增加。V-static 在 RFE 80\% 的平均 AUC 增加雖為所有 FS 條件中該方法最大，仍不足以單憑平均值宣稱改善。唯一校正後顯著的 AUC 變化為 SHAP 50\% 的 V-static 下降 0.0587，逐項區間為 [-0.0895, -0.0288]、\(p_H=0.0254\)。但其 9,999 次重抽僅有一次尾端超越，未校正 \(p=0.0002\) 的 Monte Carlo 標準誤約為 0.00014，應視為對重抽精度敏感的結果，後續可增加重抽次數核對。

48 項 F1-score 比較皆未通過校正，不能將其解釋為 FS 與 no-FS 等效。Recall 有 32 項通過校正，涵蓋不同方向；Recall 的改變也包含各條件重新決定門檻的影響，並不代表整體辨識品質一致改善。完整差值、區間、原始與校正後 \(p\) 值見附錄~\ref{sec:event-stats-full}。整體證據支持保留 no-FS 作為必要對照，尚不足以主張特徵刪減能穩定提升效能。

"""
replace(chap + "04_experiments.tex", r"\section{標籤定義對結果的影響}\label{sec:results-label}", insert + r"\section{標籤定義對結果的影響}\label{sec:results-label}")
replace(chap + "04_experiments.tex",
    "研究問題二的答案是：在固定的預設參數模型池中，成員選擇比整合方式更重要。New3 與 V-static 優於 Equal6 及四種動態方法；兩者分別以 AUC 和 F1-score 較高。歷史模型在事件標籤下表現較弱，V-static 只保留最好的 Old 模型，Equal6 全部納入則降低排序能力。此結果支持保留簡單的單一期間與等權控制組，再以驗證資料決定是否加入歷史模型。",
    "研究問題二的答案是：在本次預設參數模型池中，New3 與 V-static 相對 Equal6 的 AUC 增加具有公司配對檢定支持。V-static 雖有最高平均 F1-score，其相對 Equal6 的增加未通過多重比較校正；與動態方法的高低僅依平均值描述，未另作直接檢定。這些結果支持保留單一期間與等權集成控制組，檢查加入歷史模型是否有益，但未分離成員品質與整合規則的效果。")
replace(chap + "04_experiments.tex", "不過，調參後的單一分類器與集成表現相當，集成池成員又未經調參。", "不過，調參後的單一分類器有接近或高於集成的描述性分數，集成池成員又未經調參，跨設定差異未作等效或顯著性檢定。")
replace(chap + "04_experiments.tex", "事件標籤結果尚未完成配對檢定，本章差異均為描述性。", "事件標籤的配對分析屬事後探索性，固定已訓練模型及驗證決策；有限重抽與多重比較仍限制推論，Baseline 及未事先列入的兩兩差異維持描述性解讀。")

replace(chap + "appendix.tex",
    r"事件標籤的公司群聚配對檢定尚未完成；已保存的一份檢定輸出缺少部分檔案，因此本文不引用其數值。最終狀態標籤的相同實驗與 165 項檢定已完成，列於附錄~\ref{sec:stats-full} 作為對照。",
    r"事件標籤與最終狀態標籤皆已完成公司群聚配對分析，各有 165 項檢定，分列於附錄~\ref{sec:event-stats-full} 與附錄~\ref{sec:stats-full}。兩者的重抽次數與 Holm 校正家族分開，不將對照標籤的顯著性延伸至主要標籤。")
event_app = r"""\section{事件標籤之公司群聚配對檢定}\label{sec:event-stats-full}

本節完整列出事件標籤實驗一的 21 項與實驗二的 144 項檢定。所有比較使用同一批 3,700 家測試公司與 9,999 組公司重抽權重，並共同採 Holm 校正；重抽種子為 20261027。原始預測重播、資料列配對及指標核對均通過。差值為候選方法減比較基準，區間為逐項 95\% 百分位區間，未作同時區間校正；顯著性以 \(p_H<0.05\) 判斷。

最小可報 \(p\) 為 0.0001，並非零。尾端次數很少時，多重比較可能放大 Monte Carlo 誤差的影響；例如 SHAP 50\% 的 V-static AUC 僅有一次尾端超越，\(p=2/10{,}000=0.0002\)、\(p_H=0.0254\)，應保留有限重抽精度的限制。這批分析固定原有模型與驗證決策，屬事後探索性，不包含重新抽訓練資料或未來年份的不確定性。

\subsection*{實驗一：各方法與 Equal6 比較}
\input{generated/event-statistics-20261103/appendix-experiment1.tex}

\subsection*{實驗二：FS 與相同方法 no-FS 比較}
\input{generated/event-statistics-20261103/appendix-experiment2-AUC.tex}
\input{generated/event-statistics-20261103/appendix-experiment2-F1.tex}
\input{generated/event-statistics-20261103/appendix-experiment2-Recall.tex}

\clearpage
"""
replace(chap + "appendix.tex", r"\section{最終狀態標籤之公司群聚配對檢定}\label{sec:stats-full}", event_app + r"\section{最終狀態標籤之公司群聚配對檢定}\label{sec:stats-full}")
replace(chap + "appendix.tex", "事件標籤的檢定尚未完成，不在此列。", r"事件標籤另列於附錄~\ref{sec:event-stats-full}，不與本節合併校正。")

(ROOT / (chap + "05_conclusion.tex")).write_text(r"""\chapter{結論與未來研究}\label{chap:conclusion}
\begingroup
\setlength{\emergencystretch}{2em}

\section{研究結論}

本研究以具有時間資訊的不平衡分類為對象，在美國企業年度財務資料上比較歷史與近期模型的運用方式。主要分析採用由最後觀測年度回溯重建的事件代理標籤，並以公司最終狀態標籤作為對照。依三項研究問題，結論整理如下。

第一，近期資料的優勢依分類器而異。描述性比較中，XGBoost 與 LightGBM 以近期資料模型的平均 AUC 較高，Random Forest 以合併重訓最高，Logistic Regression 三種策略相近。由於資料期間與筆數同時變動，此結果不單獨證明新近程度的效果，也未對四種分類器作跨設定的顯著性或等效檢定。

第二，在固定的 Old／New 模型池中，New3 與驗證選定靜態集成 V-static 相對 Equal6 的 AUC 增加，皆通過以公司為單位的配對檢定與 165 項 Holm 校正。New3 的平均 AUC 最高；V-static 的平均 F1-score 最高，但其相對 Equal6 的 F1-score 增加未通過校正。這些結果支持將單一期間與全部等權集成保留為控制組，檢查混合歷史與近期模型的效益；本研究未將成員品質與整合規則分開控制，不能比較兩者的相對重要性，也不能由共同基準檢定推論 New3 與 V-static 彼此顯著不同。

第三，特徵選取沒有帶來一致改善。48 項 FS 與同方法 no-FS 的 AUC 比較中，沒有校正後顯著增加；48 項 F1-score 比較亦皆未通過校正，不能將此解釋為等效。SHAP 50\% 的 V-static AUC 下降通過校正，但尾端重抽次數很少，需保留 Monte Carlo 精度限制。Recall 的變化依方法與門檻而異。FS 將特徵由 18 個減至 9 或 14 個，但本研究未證明這能穩定改善少數類辨識或節省實際運算成本。

此外，調參後的單一分類器有接近或高於集成的描述性分數，而集成池成員使用預設參數，兩者驗證資料亦不同，因此尚不足以判斷集成本身是否優於充分調參的單一模型。改用公司最終狀態標籤時，V-static 與 New3 的平均 AUC 順序改變，顯示結果的解讀依賴目標定義；兩種標籤的原始分數與顯著性不直接互相替代。

\section{研究限制}

第一，事件標籤以破產公司的最後觀測年度推定下一年度事件，未逐筆核對破產申請日與財報公開時間，故其效度限於回溯重建的代理目標。驗證資料只有 33 筆事件，DSEL 與 Calibration 分別只有 17 與 16 筆，選模與門檻決策的不確定性較大。

第二，兩批公司配對分析皆為事後探索性，固定原有模型、年份切分與驗證決策，未涵蓋訓練樣本、年份配置或跨公司共同衝擊的不確定性。事件標籤使用 9,999 次重抽，尾端次數較少的顯著性判斷可能受 Monte Carlo 誤差影響；未達顯著不表示方法等效。

第三，集成池成員未經調參，Baseline 與集成之間的分數差同時受參數、分類器與驗證資料影響。第四，所有實驗採固定年份切分，測試期間不更新模型，未驗證持續學習或線上適應，也沒有由年份差異單獨證明概念漂移。第五，實證只使用一份公開資料，同一公司可能跨訓練與測試年度出現，結論不延伸至未見公司或其他市場。

\section{未來研究方向}

後續可先增加事件標籤分析的重抽次數，核對尾端精度敏感的結果，並另行規劃具有明確比較家族的 New3 與 V-static 直接比較；新增分析應標示為後續探索，不回頭依顯著性修改模型。其次，以調參後的成員重建模型池，在相同驗證資料下比較集成與單一分類器，並設計能分離成員組成與整合規則的對照。

第三，採用滑動視窗或逐期更新，檢驗不同年份配置下的穩定性。第四，取得可核對的事件日期或使用其他具有明確事件時間的資料，再區分訓練期間出現過與未出現過的個體，評估推廣範圍。最後，若要主張 FS 降低成本，應以相同程序量測訓練與推論時間，並同時報告辨識效能的變化。
\endgroup
""", encoding="utf-8")

replace("thesis/latex/frontmatter/abstract_zh.tex",
    "調參後的單一分類器與集成表現相當，因此本研究的證據支持審慎選擇集成成員，而非主張集成必然優於單一模型。",
    "公司群聚配對分析顯示，New3 與驗證選定靜態集成相對六模型等權集成的 AUC 增加通過 165 項 Holm 校正，但後者的 F1-score 增加未通過校正。調參後的單一分類器亦有接近或高於集成的描述性分數；因參數與驗證資料不同，本研究尚不能主張集成優於單一模型。")
replace("thesis/latex/frontmatter/abstract_zh.tex", "以破產公司的最後觀測年度重建下一年度破產事件標籤，", "以破產公司的最後觀測年度回溯重建下一年度事件代理標籤，未逐筆核對申請日；")
replace("thesis/latex/frontmatter/abstract_en.tex",
    "A next-year bankruptcy event label is reconstructed from the last observed year of each failed company, so that positive cases account for only 0.77\\% of all records.",
    "A retrospective proxy for a next-year bankruptcy event is reconstructed from each failed company's last observed year, without individually verified filing dates; positive cases account for only 0.77\\% of all records.")
replace("thesis/latex/frontmatter/abstract_en.tex",
    "Tuned single classifiers perform comparably to the ensembles, so the evidence supports careful selection of ensemble members rather than a general superiority of ensembles.",
    "Paired company-cluster analyses show that New3 and the validation-selected static ensemble improve AUC over the six-model equal-weight ensemble after Holm correction across 165 tests, whereas the latter's F1-score increase does not pass this correction. Tuned single classifiers also obtain descriptive scores close to or above those of the ensembles. Differences in parameters and validation data prevent a conclusion that ensembling itself outperforms single classifiers.")

config_path = ROOT / "config/thesis_sources.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
legacy = next(x for x in config["analyses"] if x["id"] == "fixed_company_statistics")
legacy["source_snapshots"] = {"scripts/analysis/fixed_company_statistics.py": "thesis/drafts/event_statistics_revision_20261103/fixed_company_statistics.final_status.py"}
event_dir = "results/statistical_tests/fixed_two_experiments_event/20261031T120706072983Z"
config["analyses"].append({"id": "event_company_statistics", "directory": event_dir,
    "manifest_sha256": hashlib.sha256((ROOT / event_dir / "manifest.json").read_bytes()).hexdigest(),
    "expected_protocol": "fixed_bankruptcy_standard_deslib_fit_only_fs_event_target_v1"})
config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print("Revised chapters 1–5, appendices, abstracts and statistical bindings.")
