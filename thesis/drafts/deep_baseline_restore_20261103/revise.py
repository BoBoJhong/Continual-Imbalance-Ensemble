"""Restore the two neural baseline entries with visibly provisional historical results."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
CHAP = "thesis/latex/chapters/"


def replace(name, old, new):
    p = ROOT / name
    text = p.read_text(encoding="utf-8")
    assert text.count(old) == 1, (name, old[:60], text.count(old))
    p.write_text(text.replace(old, new), encoding="utf-8")


replace(CHAP + "01_introduction.tex", "整理 AUC、F1-score 與 Recall。", "整理 AUC、F1-score 與 Recall；另保留 TabM 與 TabICL 的既有結果，待事件標籤重跑後擴充為六分類器的同協定比較。")
replace(CHAP + "02_literature.tex", "本文僅將其列為相關方法背景；事件標籤版本尚未執行，故未納入主要 Baseline 比較。", r"本文將 TabM 保留為深度學習 Baseline，第三章補列已保存設定，第四章先呈現既有最終狀態標籤結果；事件標籤分數待重跑後更新。")
replace(CHAP + "02_literature.tex", "先前 TabM 與 TabICL 的探索紀錄使用不同標籤或驗證方式，故不列入事件標籤結果表；執行範圍與限制於附錄說明。", r"TabM 與 TabICL 的既有紀錄另列於第四章的暫列結果表，並保留為六分類器架構的兩個深度學習分支。由於其標籤與驗證方式不同，暫列數值不參與事件標籤的同條件排名；後端及執行範圍的限制於附錄說明。")

replace(CHAP + "03_method.tex", "Baseline 的流程如圖~\\ref{fig:method-baseline}。各分類器分別使用 Old、New 或合併資料訓練，並搭配四種採樣設定，作為單一模型的比較基準。", r"Baseline 的流程如圖~\ref{fig:method-baseline}。候選分類器涵蓋 XGBoost、LightGBM、Random Forest、Logistic Regression、TabM 與 TabICL 六種；各自使用 Old、New 或合併資料，搭配四種採樣。四種傳統分類器已完成事件標籤實驗，兩種深度學習分類器先保留既有結果，事件標籤數據待更新，兩批數值分表呈現。")
replace(CHAP + "03_method.tex", "TabM 的候選集合已定義，但事件標籤版本尚未執行，因此不列入比較。", r"TabM 與 TabICL 的方法及既有設定另見下節；其事件標籤版本尚未完成，暫列結果不套用本段的 24 候選與十種子完成狀態。")
replacement = r"""\subsubsection{深度學習分類器與既有設定}\label{sec:method-deep-reference}

TabM 與 TabICL 均保留在 Baseline 架構中。TabM 透過神經網路的多個預測分支形成表格分類模型；本研究既有紀錄保存了依驗證資料選出的網路寬度、區塊數、成員數及最佳化參數。表~\ref{tab:deep-reference-settings} 固定展示 Old 1999--2011 年、Hybrid 採樣的一組已保存設定，不依測試分數挑選展示情境，也不將單一設定當成所有期間的共同參數。

TabICL 以訓練樣本作為預測情境；本研究的既有入口支援不同套件後端，歷史 CSV 未保存足以確認實際後端與選參過程的資訊，因此本文沿用該入口名稱報告已有數值，並保留版本待核對的限制。兩分類器的既有結果使用公司最終狀態標籤，Old／New 在 1999--2014 年的訓練期間逐年移動分界，共 15 組重疊設定，Test 固定為 2015--2018 年；驗證方式與目前保留完整 2014 年的事件協定不同。

\input{generated/deep-baseline-reference-20261103/settings.tex}

第四章先呈現這兩個分類器的既有 AUC、F1-score 與 Recall，並標示為暫列。完成事件標籤重跑後，才可在相同年份、前處理、驗證規則與種子下更新六分類器比較；目前不以歷史數字取代事件結果。

\clearpage

\subsubsection{訓練流程與時間切分}"""
replace(CHAP + "03_method.tex", "\\clearpage\n\n\\subsubsection{訓練流程與時間切分}", replacement)
replace(CHAP + "03_method.tex", "圖~\\ref{fig:method-baseline} 中的深度學習分支對應 TabM，本次事件標籤版本尚未執行。", r"圖~\ref{fig:method-baseline} 中的深度學習分支涵蓋 TabM 與 TabICL；兩者目前呈現的歷史設定見第~\ref{sec:method-deep-reference}~節，事件數據待更新。")
replace(CHAP + "03_method.tex", "本部分的比較單位為「分類器", "已完成事件標籤部分的比較單位為「分類器")

replace(CHAP + "04_experiments.tex", "本章主要結果使用第~\\ref{sec:method-label}~節定義的事件標籤。", r"本章主要結果使用第~\ref{sec:method-label}~節定義的事件標籤。第~\ref{sec:results-deep-reference}~節另暫列 TabM 與 TabICL 的既有最終狀態標籤結果，其標籤、驗證方式及彙整口徑不同，並於表名與表下注明。")
deep_results = r"""\subsection{TabM 與 TabICL 既有結果（暫列）}\label{sec:results-deep-reference}

為保留六分類器的完整研究架構，表~\ref{tab:deep-reference-summary} 先放入 TabM 與 TabICL 的既有結果。這批數字預測的是公司最終狀態，使用 15 組 Old／New 訓練邊界與各自的舊驗證程序，並非前表的事件標籤實驗；事件數據待重跑後更新。

\input{generated/deep-baseline-reference-20261103/summary.tex}

Old 與 New 各彙整 60 筆設定紀錄，即 15 個邊界乘四種採樣；Retrain 僅有四筆，不複製為 15 組。這些設定共用測試期間，且訓練資料重疊，不是 60 次或 15 次獨立重複，也沒有可據以報告的十種子標準差。此處先保留三種策略及三項指標的紀錄，不將其與事件表的分數高低解讀為分類器優劣。各採樣明細與後端限制列於附錄~\ref{sec:deep-reference-details}。

"""
replace(CHAP + "04_experiments.tex", r"\section{實驗一：Old／New 集成比較}\label{sec:results-experiment1}", deep_results + r"\section{實驗一：Old／New 集成比較}\label{sec:results-experiment1}")
replace(CHAP + "05_conclusion.tex", "第三，集成池成員未經調參，Baseline 與集成之間的分數差同時受參數、分類器與驗證資料影響。", "第三，集成池成員未經調參，Baseline 與集成之間的分數差同時受參數、分類器與驗證資料影響。TabM 與 TabICL 先暫列既有最終狀態標籤結果，其事件數據尚待更新，本文結論不包含六分類器的同協定排名。")
replace(CHAP + "05_conclusion.tex", "其次，以調參後的成員重建模型池，", "其次，補齊 TabM 與 TabICL 的事件標籤實驗後更新六分類器比較，再以調參後的成員重建模型池，")

replace(CHAP + "appendix.tex", "TabM 的候選集合已定義，但事件標籤版本尚未執行；先前以最終狀態標籤執行的 TabM 與 TabICL 探索紀錄，因標籤與驗證方式不同，不納入本文比較。", r"TabM 與 TabICL 的事件標籤版本尚未完成；兩者先前最終狀態標籤的結果已暫列於第四章，設定與明細見第~\ref{sec:deep-reference-details}~節，不參與事件標籤的同條件比較。")
deep_app = r"""\clearpage
\section{TabM 與 TabICL 既有設定與採樣明細（暫列）}\label{sec:deep-reference-details}

本節補充第四章兩個深度學習分類器的歷史結果。原始 CSV 各有 124 筆設定：Old 與 New 各 60 筆，Retrain 四筆；欄位只保存各設定的測試指標，沒有獨立十種子結果。TabM 另保存驗證 AUC 及選定參數，TabICL 未保存相同欄位，不能將兩者寫成已完成同一搜尋預算的調參比較。

依保存的年份切分入口，訓練期間為 1999--2014 年，逐年改變 Old／New 分界，測試期間均為 2015--2018 年；舊驗證程序由各訓練期間內再切分驗證資料，與目前事件協定完整保留 2014 年的設計不同。這些數字僅作既有最終狀態標籤紀錄呈現；原執行環境與套件後端未完整留存，不能由目前程式或資料夾名稱反推當時全部設定。

尤其既有 TabICL wrapper 支援 TabICL 及 TabPFN 後端，結果 CSV 沒有記錄實際採用者。因此表中的 TabICL 保留既有入口名稱，不將這批分數視為已確認重現原 TabICL 論文的結果；更新時須同時保存套件、模型版本與實際後端。TabM 的單一展示情境已列於表~\ref{tab:deep-reference-settings}，其他設定仍應依各自保存的驗證參數判讀。

\input{generated/deep-baseline-reference-20261103/sampling.tex}

兩個分類器的事件標籤重跑與同協定比較尚待完成。完成後應同步更新本節、第四章暫列表、第三章設定及摘要的完成範圍，重新核對來源後替換數字。

\clearpage
"""
replace(CHAP + "appendix.tex", r"\section{事件標籤之公司群聚配對檢定}\label{sec:event-stats-full}", deep_app + r"\section{事件標籤之公司群聚配對檢定}\label{sec:event-stats-full}")

replace("thesis/latex/frontmatter/abstract_zh.tex", "再以 XGBoost 建立 Old／New 六模型池，", "另暫列 TabM 與 TabICL 的既有結果，待事件數據更新；再以 XGBoost 建立 Old／New 六模型池，")
replace("thesis/latex/frontmatter/abstract_en.tex", "Experiment 2 adds four feature selection methods at two retention levels.", "Historical TabM and TabICL scores are included provisionally, pending event-target reruns. Experiment 2 adds four feature selection methods at two retention levels.")

proof = "thesis/drafts/deep_baseline_restore_20261103/results-proof.json"
p = ROOT / "config/thesis_sources.json"
c = json.loads(p.read_text(encoding="utf-8"))
c["deep_baseline_reference"] = {"proof": proof, "sha256": hashlib.sha256((ROOT / proof).read_bytes()).hexdigest()}
p.write_text(json.dumps(c, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print("Restored both neural classifiers, provisional historical tables and matching scope statements.")
