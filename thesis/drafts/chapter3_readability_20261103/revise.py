from pathlib import Path
import hashlib
import json
import re

root = Path(__file__).resolve().parent
target = Path('thesis/latex/chapters/03_method.tex')
original_bytes = (root / '03_method.before.tex').read_bytes()
if hashlib.sha256(target.read_bytes()).hexdigest() != hashlib.sha256(original_bytes).hexdigest():
    raise SystemExit('The chapter changed after backup; inspect before applying.')
s = original_bytes.decode('utf-8').replace('\r\n', '\n')


def between(start, end):
    a = s.index(start)
    b = s.index(end, a)
    return s[a:b].strip()


def table(label):
    at = s.index('\\label{' + label + '}')
    a = s.rfind('\\begingroup', 0, at)
    b = s.index('\\endgroup', at) + len('\\endgroup')
    return s[a:b]


def figure(label):
    at = s.index('\\label{' + label + '}')
    a = s.rfind('\\begin{figure}', 0, at)
    b = s.index('\\end{figure}', at) + len('\\end{figure}')
    return s[a:b]


overview = figure('fig:method-overview')
dictionary = table('tab:method-feature-dictionary')
window = figure('fig:method-time-window')
baseline_figure = figure('fig:method-baseline')
search = table('tab:method-ml-search')
ensemble_figure = figure('fig:method-ensemble')
pool = table('tab:method-pool')
combinations = table('tab:method-combinations')
fs_figure = figure('fig:method-fs').replace(
    '各 FS 方法平行執行；SHAP 為模型解釋後的特徵排序，其定位另於正文說明。DSEL 與 Calibration 維持實驗一集成部分的分配。',
    '各 FS 方法平行執行，並在選取後分別重建模型池與動態選擇器。')
fs_table = table('tab:method-fs')
fs_params = table('tab:method-fs-parameters')

common_settings = r'''\begingroup
\small\setstretch{1.12}\setlength{\tabcolsep}{4pt}
\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.18\linewidth}>{\raggedright\arraybackslash}p{0.74\linewidth}@{}}
\caption{共同前處理與採樣設定}\label{tab:method-common-settings}\\
\toprule
項目 & 設定 \\
\midrule\endfirsthead
\multicolumn{2}{c}{表~\thetable（續）}\\\toprule
項目 & 設定 \\
\midrule\endhead
\midrule\multicolumn{2}{r}{續下頁}\\\endfoot
\bottomrule\endlastfoot
補值與縮放 & Old-fit 平均數補值，保留全空欄位的輸入維度；以補值後的 Old-fit 擬合 StandardScaler \\
None & 不採樣，保留擬合資料的類別分布 \\
Under & TomekLinks；\nolinkurl{sampling_strategy=auto} \\
Over & ADASYN；\nolinkurl{sampling_strategy=auto}、\nolinkurl{n_neighbors=5} \\
Hybrid & SMOTEENN；SMOTE 的 \nolinkurl{k_neighbors=5}；ENN 的 \nolinkurl{n_neighbors=3}、\nolinkurl{kind_sel=all} \\
\end{longtable}
\noindent{\footnotesize (資料來源：本研究實驗程式與固定版本套件設定整理)}
\endgroup'''

framework = r'''% 研究方法：共同設定集中說明，實驗依執行順序編排。
\chapter{研究方法}\label{chap:method}
\begingroup
\setlength{\emergencystretch}{2em}
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000

\section{研究架構}\label{sec:method-framework}

本章說明兩階段集成式持續學習的操作流程。實驗一先比較單一分類器的訓練期間與採樣方式，再建立 Old／New 模型池，比較靜態集成、DCS 與 DES；實驗二在相同條件下加入特徵選取，以完整特徵結果為對照。整體流程如圖~\ref{fig:method-overview}，實際資料與年份配置見第~\ref{sec:method-data}~節，評估與統計程序見第~\ref{sec:method-validation}~節。

輸入資料表示為 \(\{(x_i,y_i,t_i)\}_{i=1}^{n}\)，其中 \(x_i\) 為特徵向量、\(y_i\in\{0,1\}\) 為分類標籤、\(t_i\) 為觀測時間，\(n\) 為樣本數。同一個體的識別碼另供分區檢核與配對。以 \(\mathcal{O}\) 表示歷史模型池、\(\mathcal{N}\) 表示近期新增模型，更新由 \(\mathcal{O}\) 擴充至 \(\mathcal{O}\cup\mathcal{N}\)，再以驗證資料決定使用成員與分類門檻。本次採固定時間分區，在同一離線實驗內重建各設定，模擬一次模型池更新。
'''

data = r'''\section{實驗資料與配置}\label{sec:method-data}

本節交代兩項實驗共用的資料、時間分區與處理規則。各方法的原理已於第二章說明，以下著重實際輸入與執行設定。

\subsection{資料來源與標籤定義}\label{sec:method-label}

本研究使用 Lombardo 等人\autocite{lombardo2022bankruptcy}所提供、於 2024 年 4 月 10 日更新的公開資料\autocite{sowide-bankruptcy-data}，包含 1999--2018 年的 78,682 筆公司年度紀錄、8,971 個匿名公司識別碼與 18 個財務特徵。公開說明的 8,262 家公司與檔內識別碼數不同，本文依實際檔案計數；版本核對見附錄~\ref{sec:appendix-data-fs}。

原始欄位 \nolinkurl{status_label} 記錄公司的最終狀態：609 個識別碼為 \nolinkurl{failed}，其所有年度共 5,220 筆紀錄均標為破產，占 6.63\%。主要分析改用回溯重建的事件代理標籤：破產公司僅最後觀測年度記為 1，作為推定下一年度事件的代理；其餘紀錄記為 0，共有 609 筆代理事件，占 0.77\%。年度 \(t\) 的標籤可用時間設為 \(t+1\)，並據此檢查各分區標籤在測試開始時是否成熟。此為年度可用性假設；公開檔缺乏逐筆破產申請日期與財報公開時間，無法驗證實際資訊到達日。

最終狀態標籤保留為對照，使用相同資料列、分區及模型池結構，只改變目標欄位；結果見第~\ref{sec:results-label}~節。

\subsection{輸入欄位與資料品質}

觀測單位為公司年度紀錄，模型輸入限於表~\ref{tab:method-feature-dictionary} 的 X1--X18。識別碼、年度、標籤及產業分類均不作預測特徵，亦未新增比率、股價或落後期特徵。同一公司可跨擬合與測試年份出現，因此評估對象是較晚年度的紀錄，並非完全未見的公司。

載入時檢查公司與年度組合及財務欄位有效值，未發現重複公司年度紀錄、遺失值或無限值，故沒有因此刪除紀錄。各設定仍保留相同的補值器，完整處理規則見第~\ref{sec:method-common}~節。

2024 年公開版的欄位順序及數值與 2022 年版不同，本研究依公開研究附件的欄位名稱及跨版本數值對照整理\autocite{sowide-financial-supplement}，保留實際版本的數值與順序，未依舊版倍率換算。
'''

split = r'''\subsection{時間切分與驗證用途}\label{sec:method-time-window}

固定年份分區列於表~\ref{tab:method-partitions}。歷史／近期的切點及測試區間參考原資料研究\autocite{lombardo2022bankruptcy}，但另將 2014 年完整保留供驗證使用，使 New 實際只擬合 2012--2013 年。由於可用訓練資料與原文不同，不直接比較兩者分數；2011 年切點亦非由 Test 搜尋得到。

集成與 FS 將 2014 年以種子 42、依事件標籤分層等分為互不重疊的 DSEL 與 Calibration。DSEL 估計動態方法的局部分類能力；Calibration 選擇靜態成員與分類門檻，未另執行機率校準。兩者均不加入基礎模型擬合，亦無年內先後順序。Baseline 則合用這兩部分，以 2014 年全部 3,504 筆紀錄選擇超參數與門檻；三種訓練策略的擬合資料與集成一致。

\input{generated/event-results-20261103/event-partitions.tex}

2014 年驗證資料共 33 筆事件，分至 DSEL 與 Calibration 後各為 17 與 16 筆，故候選排名與門檻可能受少量事件影響。所有方法固定後才評估 2015--2018 年，各測試年度之間不重新訓練。各年度紀錄數另列於附錄~\ref{sec:appendix-data-fs}。

圖~\ref{fig:method-time-window} 對照一般視窗前移與本次固定配置；上半部僅為概念示意，多輪更新並非本次已執行的實驗。
'''

common = r'''\subsection{共同前處理、採樣與隨機種子}\label{sec:method-common}

兩項實驗均先以 Old-fit 擬合平均數補值與標準化參數，再將相同轉換套用於 New-fit、驗證與 Test，以維持共同特徵表示。Baseline 的 Retrain 合併轉換後的 Old-fit 與 New-fit。採樣只作用於各模型所屬的擬合資料，驗證與 Test 保留原始類別分布；實驗二另於採樣前執行特徵選取。

表~\ref{tab:method-common-settings} 列出共同設定。Baseline 包含四種採樣條件，集成模型池使用其中 Under、Over 與 Hybrid。第二章已說明各採樣方法的作用，本章沿用相同名稱。
'''
seeds = r'''模型、採樣及具有隨機性的特徵選取程序，使用 42、123、456、789、2024、7、21、84、168、999 共十個種子，可接收種子的元件依各次設定傳遞；2014 年的資料列分配則固定，不隨模型種子或 FS 條件改變。軟體環境與補充訓練控制見附錄~\ref{app:experimental-details}。以下十種子規則適用於已完成的事件實驗；深度模型歷史結果的彙整方式另於第~\ref{sec:method-deep-reference}~節交代。
'''

baseline = r'''\section{實驗一：Baseline 與集成分類器比較}\label{sec:method-experiment1}

本實驗包含單一分類器基準與模型池集成兩部分，前者比較資料期間、採樣及分類器，後者比較共同模型池的整合規則。

\subsection{Baseline 分類器比較}\label{sec:method-phase1}

Baseline 流程如圖~\ref{fig:method-baseline}，候選包含第二章介紹的 XGBoost、LightGBM、Random Forest、Logistic Regression、TabM 與 TabICL。四種傳統分類器已完成事件標籤實驗；兩種深度模型目前先列歷史結果，與事件結果分表呈現。
'''
baseline_training = r'''\subsubsection{訓練策略與執行流程}

各分類器分別建立三種模型：Old 使用 Old-fit，New 使用 New-fit，Retrain 合併兩者從頭訓練。New 亦為重新訓練，未延續 Old 模型權重。各策略搭配表~\ref{tab:method-common-settings} 的四種採樣，採用第~\ref{sec:method-common}~節的共同前處理。

對每個「分類器 \ensuremath{\times} 訓練策略 \ensuremath{\times} 採樣」組合，先依驗證 ROC-AUC 選出超參數，再以十個種子重新訓練，並分別依驗證 F1-score 決定門檻，最後計算 Test 指標。已完成部分共 \(4\times3\times4=48\) 個組合；深度模型的歷史設定不套用此完成範圍。

\subsubsection{超參數候選與選擇}\label{sec:method-phase1-params}

每個組合評估 24 組候選：候選 0 為目前集成測試採用的預設參數，其餘 23 組依固定亂數從表~\ref{tab:method-ml-search} 抽出，不窮舉全部排列。候選均以種子 42 擬合，驗證 AUC 相同時取候選編號較小者。選定參數後的十種子評估沿用前述流程，不依 Test 改選。
'''
search_note = r'''表中 None 與深度值 -1 均表示不設樹深度上限；Random Forest 的小數型 \nolinkurl{max_features} 表示每次分裂考量的特徵比例。各組合選中的候選編號保存於執行紀錄。

\subsubsection{深度模型的歷史設定}\label{sec:method-deep-reference}

TabM 的既有紀錄保存了驗證選定的網路寬度、區塊數、成員數及最佳化參數。表~\ref{tab:deep-reference-settings} 展示 Old 1999--2011 年、Hybrid 採樣的一組設定，並非所有情境共用的參數，展示情境亦未依測試分數挑選。TabICL 的既有入口支援不同套件後端，歷史 CSV 不足以確認實際後端及完整選參過程，故其模型版本仍待核對。

\input{generated/deep-baseline-reference-20261103/settings.tex}

兩者歷史結果採公司最終狀態標籤：Old／New 分界在 1999--2014 年間移動，共 15 組重疊設定，Test 固定為 2015--2018 年，驗證方式與本次事件協定不同。第四章暫列既有指標；完成事件標籤及共同驗證規則的重跑後，才能更新為六分類器的同條件比較。

\subsubsection{集成基礎模型的選擇與完成範圍}\label{sec:method-selection-status}

研究框架以共同驗證 ROC-AUC 選定基礎分類器及參數，再建立 Old／New 六模型池。第四章先彙整四種傳統分類器在六個池位置（Old／New \ensuremath{\times} Under、Over、Hybrid）的驗證表現；正式選模仍待完整候選齊備。現有集成與 FS 結果來自 XGBoost 預設參數（候選 0）的流程測試，尚未依完整六分類器驗證結果重建模型池。以下交代該測試的可重現設定，並以此界定第四章現有結果的範圍。
'''

ensemble = r'''\subsection{Old／New 集成分類器比較}\label{sec:method-phase2}

集成流程如圖~\ref{fig:method-ensemble}。各方法共用同一基礎分類器及六模型池，差異限於成員選擇與整合規則；實作狀態依第~\ref{sec:method-selection-status}~節說明。
'''
pool_intro = r'''\subsubsection{模型池建立與分類器設定}

依共同前處理完成轉換後，Old-fit 與 New-fit 各自採用三種採樣並訓練分類器，形成表~\ref{tab:method-pool} 的六個位置。實驗一使用完整 18 個特徵，基礎分類器不另加入類別權重。
'''
pool_settings = r'''目前測試的 XGBoost 設定為 \nolinkurl{objective=binary:logistic}、\nolinkurl{eval_metric=auc}、\nolinkurl{tree_method=hist}，其餘依固定版本預設值；未另搜尋池內超參數，亦未啟用早停。Baseline 與集成的參數及驗證用途不同，兩者分數差不單獨解釋為集成效果。Retrain 僅列於 Baseline，不納入此模型池。

\subsubsection{靜態成員與機率整合}

沿用第~\ref{sec:method-framework}~節的 \(\mathcal{O}\) 與 \(\mathcal{N}\)，跨期間混合候選須滿足：

\begin{equation}\label{eq:method-subsets}
S\subseteq\mathcal{O}\cup\mathcal{N},\qquad
S\cap\mathcal{O}\ne\varnothing,\qquad
S\cap\mathcal{N}\ne\varnothing.
\end{equation}

兩池各有三個模型，故混合候選共 \((2^3-1)(2^3-1)=49\) 種，依成員數分組如表~\ref{tab:method-combinations}。
'''
static = r'''對任一候選子集 \(S\)，以等權平均形成集成機率：

\begin{equation}\label{eq:method-probability}
\hat p_S(x)=\frac{1}{|S|}\sum_{i\in S}\hat p_i(x),
\end{equation}

其中，\(\hat p_i(x)\) 為第 \(i\) 個分類器對樣本 \(x\) 的正類機率，\(|S|\) 為成員數。另保留六個單模型，以及單一期間控制組 Old3、New3。完整六模型等權集成稱為 Equal6，已包含於 49 種混合候選；Old3 與 New3 則不在混合候選範圍內。

\subsubsection{動態選擇的實作設定}

依第二章所述機制，以完整六模型池比較 DCS 的 OLA、LCA，以及 DES 的 KNORA-E、KNORA-U。四種方法共用 DSEL，使用 DESlib 0.3.7，鄰居數固定為 \(k=7\)。DCS 採 \nolinkurl{best} 規則，局部能力平手時依固定模型順序選擇；DES 採 \nolinkurl{soft} 機率整合。額外動態剪枝與樣本難度分流均關閉，即 \nolinkurl{DFP=False}、\nolinkurl{with_IH=False}。

所有基礎分類器對某樣本判斷一致時，套件直接平均機率；因此 DCS 的輸出不一定每次都來自單一模型。其餘局部選擇與權重處理沿用該版本標準實作\autocite{deslib-ola-docs,deslib-lca-docs,deslib-knorae-docs,deslib-knorau-docs}，未另修改演算法。

\subsubsection{靜態候選與分類門檻的決定}

先在 Calibration 對 49 種混合候選計算 AP，選出最高者，記為 V-static（執行紀錄名稱為 ValidationSelectedStatic）；平手時依固定方法名稱的字典序決定。全部候選的 Test 表現僅供組成分析，不用來回選成員。

各單模型、控制組、靜態候選及動態方法，均在 Calibration 搜尋 F1-score 最高的門檻：

\begin{equation}\label{eq:method-threshold}
\mathcal{T}=\{0.01,0.02,\ldots,0.99\},\qquad
\tau_m^*=\underset{\tau\in\mathcal{T}}{\arg\max}\;
F1_{\mathrm{Calibration}}(m,\tau).
\end{equation}

平手取最小門檻，以 \(\hat p_m(x)\ge\tau_m^*\) 判為正類；V-static 沿用所選候選的門檻。成員與門檻固定後，才計算 Test 的整體與逐年指標。
'''

fs_intro = r'''\section{實驗二：特徵選取集成分類器比較}\label{sec:method-phase3}

實驗二沿用實驗一的集成設定，只在採樣前加入特徵選取，再重新訓練模型池，以檢查特徵精簡對分類表現的影響。流程如圖~\ref{fig:method-fs}。
'''
fs_flow = r'''\subsection{執行流程與特徵條件}

處理順序為「資料分區 \ensuremath{\rightarrow} 前處理 \ensuremath{\rightarrow} 特徵選取 \ensuremath{\rightarrow} 採樣 \ensuremath{\rightarrow} 模型訓練」。各選取器僅擬合轉換後、尚未採樣的 Old-fit；所選欄位及順序固定後，一致套用於 New-fit、DSEL、Calibration 與 Test。每個條件重新建立六模型池及動態選擇器，使分類器與局部鄰近區域使用相同輸入空間。

MI、CART、RFE 與 SHAP 排序分別獨立執行，不串接使用。各方法設定名義保留比例 \(r=0.5\) 與 \(r=0.8\)，依下式決定保留數：

\begin{equation}\label{eq:method-fs-count}
k_{\mathrm{FS}}=\max\{1,\lfloor d\times r\rfloor\}.
\end{equation}

原始特徵數 \(d=18\)，兩種條件分別保留 9 與 14 個，另以完整 18 特徵的 no-FS 為對照，共九種條件。兩個比例用於比較較大與較小幅度的刪減，並非由驗證或 Test 搜尋的最佳比例；80\% 為名義值，實際數量依向下取整計算。

\subsection{選取依據與固定參數}\label{sec:method-fs-parameters}

表~\ref{tab:method-fs} 摘要選取依據，表~\ref{tab:method-fs-parameters} 列出固定參數；方法原理與 XAI 定位沿用第二章。
'''
fs_note = r'''MI 與 RFE 由 scikit-learn 實作\autocite{sklearn-fs-docs}；SHAP 排序使用 LightGBM 的 \nolinkurl{pred_contrib=True} 輸出，排除基準值欄位後計算平均絕對貢獻\autocite{lightgbm-booster-docs}。CART、RFE 與 SHAP 的平衡類別權重只作用於選取器，不改變後續基礎分類器設定。未另指定的參數沿用表列套件版本預設值，亦不使用 Calibration 或 Test 調整選取器。
'''
fs_comparison = r'''\subsection{配對比較方式}

各條件沿用 49 種混合靜態候選、Old3、New3、Equal6 與四種動態方法，使用相同資料列分區、模型種子及 Test 樣本。Calibration 在各條件下重新選擇靜態候選與分類門檻；因此 V-static 的差值包含特徵與成員改變，須與維持成員組合的固定方法分開解讀。

對固定方法 \(m\)、指標 \(M\)、種子 \(s\) 與特徵條件 \(f\)，配對差值為：

\begin{equation}\label{eq:method-delta}
\Delta M_{s,m,f}=M_{s,m,f}-M_{s,m,\mathrm{no\_fs}}.
\end{equation}

正值表示高於 no-FS。即使成員組合固定，F1-score 與 Recall 差異仍包含重新訓練與各自門檻選擇的影響。各次所選欄位名稱、順序與數量均保存，十種子的入選次數列於附錄~\ref{sec:appendix-data-fs}，僅作一致性的描述，未另作排名顯著性檢定。
'''

metrics = between('\\subsection{分類效能評估指標}', '\\subsection{結果彙整與模型比較}')
metrics = metrics.replace(
    'AUC 為 ROC 曲線下的面積，用於衡量正負類樣本的排序區辨能力，數值越高表示整體區辨表現越佳。AUC 由預測機率計算，不依賴本研究所選的單一分類門檻；惟其無法單獨反映實際門檻下的正類樣本辨識情形，故另以 F1-score 與 Recall 補充。',
    'AUC 為 ROC 曲線下的面積，由預測機率計算，不使用分類門檻。')
metrics = metrics.replace(
    'F1-score 同時反映正類預測的精確程度與涵蓋程度。本研究以各方法在 Calibration 決定的門檻計算測試 F1-score；計算遇到未定義的零分母情形時，依統一的評估規則記為零。',
    'F1-score 與 Recall 使用各方法已固定的驗證門檻計算；Baseline 使用完整 2014 年，集成與 FS 使用 Calibration。計算遇到未定義的零分母時，統一記為零。')
metrics = metrics.replace(
    'Recall 與 TPR 相同，表示實際正類紀錄中被正確辨識的比例。較低的 Recall 代表較多正類紀錄被漏判；惟較高 Recall 亦可能伴隨較多誤報，因此須與 F1-score 及 AUC 共同解讀，不僅依 Recall 判定方法優劣。',
    'Recall 即式~\\ref{eq:recall} 的正類涵蓋比例，與 TPR 相同，依相同門檻另外報告。')
metrics = metrics.replace(
    '其中，\\(P_n\\) 與 \\(R_n\\) 分別為第 \\(n\\) 個操作點的 Precision 與 Recall。本研究以 Recall 增量加權 Precision；此計算不同於對 PR 曲線直接進行梯形積分，定義見 scikit-learn AP 文件\\autocite{sklearn-ap-docs}。AP 用於驗證選模，最終主文仍以 AUC、F1-score 與 Recall 為主要報告指標。',
    '其中，\\(P_n\\) 與 \\(R_n\\) 為第 \\(n\\) 個操作點的 Precision 與 Recall。本研究依 scikit-learn 定義計算 AP\\autocite{sklearn-ap-docs}，用於驗證選模；Test 主文指標為 AUC、F1-score 與 Recall。')
metrics = metrics.replace(
    '原始特徵數為 18，保留 9 個時精簡率為 50.00\\%，保留 14 個時為 22.22\\%；no-FS 的精簡率為 0\\%。名義保留比例為 80\\% 的設定，實際保留 14 個，即 77.78\\%，因此本文分別報告名義設定與實際數量。本研究未進行獨立的樣本選取實驗，不另套用樣本精簡率解釋過採樣或混合採樣的結果。',
    '保留 9 與 14 個特徵時，精簡率分別為 50.00\\% 與 22.22\\%；no-FS 為 0\\%。此指標只描述輸入維度的減少。')

summary = r'''\subsection{結果彙整}

整體結果先合併 2015--2018 年的測試標籤與預測，再計算指標，稱為合併四年結果（pooled），不等同各年度分數的平均。逐年結果另外計算，模型與門檻均沿用驗證決策。相同設定的十種子結果以平均值及標準差呈現；種子共用同一份 Test，標準差僅描述執行變異，不代表獨立資料的不確定性。歷史深度模型另依第~\ref{sec:method-deep-reference}~節的重疊設定彙整，不併入十種子比較。
'''
statistics = s[s.index('\\subsection{以公司為單位的統計比較}'):].strip()
old = between('兩個方法的分數不同，', '\\subsubsection{比較對象與估計量}')
new = r'''本研究補作公司群聚配對 bootstrap，以同一批重新抽取的測試公司比較方法差值。事件標籤為主要分析，最終狀態標籤另作對照，各自檢定與校正。此為既有實驗完成後制定的事後探索性分析，沿用保存的模型、特徵、成員與門檻，不因檢定結果重新選模。統計原理及群聚相關性的文獻依據已於第二章說明，以下列出實際估計量與重抽程序。'''
statistics = statistics.replace(old, new)
statistics = statistics.replace(
    '驗證選定靜態集成（以下簡稱 V-static）', 'V-static')
statistics = statistics.replace(
    'V-static 在各種子內依 Calibration AP 選擇混合成員，不使用 Test 排序結果選擇靜態方案。Baseline 與集成的參數及驗證資料不同，不納入此檢定範圍。',
    'Baseline 因參數及驗證用途不同，不納入此檢定。')
statistics = statistics.replace(
    'V-static 允許 FS 後依原有 Calibration 規則重新選擇成員，故該比較評估完整選模程序的變化；其他方法保持方法名稱與規則一致，但仍重新訓練模型並以 Calibration 固定各自門檻。',
    'V-static 與固定方法的比較界線依實驗二所述規則處理。')
statistics = statistics.replace(
    '同時比較很多方法時，較容易把偶然出現的分數差誤認為有效差異。因此，每一種標籤下的兩項實驗共 165 個 \\(p\\) 值一併使用 Holm 法\\autocite{holm1979multiple} 校正，以降低這類誤判。',
    '每種標籤下的兩項實驗共 165 個 \\(p\\) 值一併使用 Holm 法\\autocite{holm1979multiple} 校正。')
statistics = statistics.replace(
    '上述分析固定原有模型、資料切分與驗證決策，只改變抽到的測試公司。因此，它回答的是「換一批測試公司時，兩方法的分數差是否仍相近」，沒有評估換一批訓練資料後的變化。此外，不同公司也可能受到相同產業或景氣影響，這些關聯未另行估計。結果應限於本次資料與設定解讀。事件標籤的檢定於第~\\ref{sec:results-event-statistics}~節摘要，兩種標籤的完整比較分列於附錄。',
    '此重抽分析只衡量固定模型下測試公司組成的不確定性，未涵蓋重新取得訓練資料或跨公司產業、景氣關聯。事件檢定摘要見第~\\ref{sec:results-event-statistics}~節，兩種標籤的完整比較分列於附錄。')

parts = [framework, overview, r'\clearpage', data, dictionary, split, window,
         common, common_settings, seeds, r'\clearpage', baseline, baseline_figure,
         baseline_training, search, search_note, ensemble, ensemble_figure,
         pool_intro, pool, pool_settings, combinations, static, r'\clearpage',
         fs_intro, fs_figure, fs_flow, fs_table, fs_note, fs_params, fs_comparison,
         r'\section{實驗驗證準則}\label{sec:method-validation}',
         '本節統一定義指標、結果彙整與統計比較。資料用途與選擇程序依前述設定執行，驗證最高分不作為最終預測表現。',
         metrics, summary, statistics]
revised = '\n\n'.join(p.strip() for p in parts) + '\n'
target.write_text(revised, encoding='utf-8', newline='\n')
checks = {
    'before_chars': len(s), 'after_chars': len(revised),
    'before_sha256': hashlib.sha256(original_bytes).hexdigest(),
    'after_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'removed_labels': sorted(set(re.findall(r'\\label\{([^}]+)\}', s)) - set(re.findall(r'\\label\{([^}]+)\}', revised))),
    'preserved_inputs': re.findall(r'\\input\{([^}]+)\}', revised),
}
(root / 'revision.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(checks, ensure_ascii=False, indent=2))
