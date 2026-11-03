"""One-time, boundary-checked reorganization of the saved chapter."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / 'thesis/latex/chapters/03_method.tex'
s = P.read_text(encoding='utf-8')


def between(start, end):
    return s[s.index(start):s.index(end)]


header=s[:s.index(r'\section{研究架構}')]
framework=between(r'\section{研究架構}',r'\clearpage'+'\n'+r'\section{實驗一')
data=between(r'\subsection{資料來源與驗證設定}',r'\subsection{Baseline 分類器之超參數設定}')
baseline_params=between(r'\subsection{Baseline 分類器之超參數設定}',r'\subsection{實驗二之特徵選取參數}')
fs_params=between(r'\subsection{實驗二之特徵選取參數}',r'\subsection{集成模型之固定設定}')
ensemble_params=between(r'\subsection{集成模型之固定設定}',r'\section{實驗驗證準則}')
experiment1=between(r'\section{實驗一：Baseline',r'\section{實驗二：特徵選取集成分類器比較}')
experiment2=between(r'\section{實驗二：特徵選取集成分類器比較}',r'\section{實驗設定與參數}')
validation=s[s.index(r'\section{實驗驗證準則}'):]

data=data.replace(r'\subsection{資料來源與驗證設定}\label{sec:method-data}',
    '\\section{研究資料}\\label{sec:method-data}\n\n本章先說明實證資料與標籤，再介紹可套用於時間排序、類別不平衡資料的分析框架。本文以企業資料作為實證案例，資料領域與框架的操作步驟分別交代。\n\n\\subsection{資料來源與標籤定義}')
data=data.replace('載入資料時檢查', '\\subsection{欄位與資料品質}\n\n本研究以公司年度紀錄為觀測單位。公司識別碼、年度及標籤只供分區、配對與檢核，不作預測輸入。同一公司可能跨擬合與測試年份出現，因此本研究評估較晚年度的狀態分類，不將結果解釋為對未見公司的驗證。\n\n載入資料時檢查',1)
data=data.replace('各年度的紀錄數與標籤比例', '\\subsection{資料用途與驗證分區}\n\n各年度的紀錄數與標籤比例',1)

framework=framework.replace(r'\section{研究架構}',r'\section{通用研究架構}')
start=framework.index('企業的財務狀況')
end=framework.index('表~\\ref{tab:method-stages}')
framework=framework[:start]+'''本研究處理具有時間順序與類別不平衡的分類問題。以較早資料訓練的模型可以保留歷史資訊，較近期資料則反映新出現的樣本特性；兩者的樣本量與類別分布可能不同。框架將可用資料分為 Old 與 New，透過不同採樣建立模型池，再以靜態集成或動態選擇產生預測。特徵選取則作為平行條件，檢視調整輸入空間後的表現。

圖~\\ref{fig:method-overview} 將比較安排為兩項實驗。實驗一先建立單一分類器 Baseline，再固定基礎分類器比較 Old／New 模型的整合方式；實驗二在相同集成設定下加入特徵選取。流程中的正類可以是不同應用領域關注的少數類別，觀測單位、標籤與可用時間須依資料定義。本研究的具體資料已於第~\\ref{sec:method-data}~節交代，分類器與年份則在各實驗設定中列明。

'''+framework[end:]
framework=framework.replace('公司年度財務資料','具有時間資訊的分類資料').replace('定義破產正類；依年度劃分擬合、驗證與測試用途','定義觀測單位與正類；依時間劃分擬合、驗證與測試用途')
framework=framework.replace('以固定 XGBoost 設定','以固定基礎分類器設定').replace('評估 2015--2018 年測試資料','評估保留的較晚期間測試資料')
start=framework.index('本研究以公司年度財務紀錄為觀測單位')
end=framework.index('最終測試以',start)
framework=framework[:start]+framework[end:]
framework=framework.replace('破產類別','正類')
framework+='''\\subsection{時間視窗與本研究的固定配置}\\label{sec:method-time-window}

滑動視窗可用來描述框架在時間向前推進時的資料配置：對每一個預測時點，以較早期間建立 Old／New 模型，以獨立的驗證資料決定模型組合與門檻，再評估更晚的 Test；時點前移後，相同規則套用於下一個視窗。各視窗均須重新遵守資料可用時間與擬合邊界，不能把未來標籤提前用於模型建立。

圖~\\ref{fig:method-time-window} 上半部示範視窗等距前移，下半部列本研究實際採用的固定配置。這張圖用於說明時間安排，不代表兩項實驗已執行逐年更新模型的滑動視窗評估。本研究的集成與 FS 模型固定使用 Old-fit 1999--2011 年、New-fit 2012--2013 年及 2014 年驗證，測試 2015--2018 年；測試年度之間沒有重新訓練。Baseline 的 15 組設定則逐年移動 Old／New 分界，仍共用相同 Test，屬於分界敏感度比較。

\\begin{figure}[htbp]
\\centering
\\includegraphics[width=\\linewidth]{figures/time-window-concept.png}
\\caption{通用滑動視窗概念與本研究固定年份配置}\\label{fig:method-time-window}
\\par\\smallskip{\\footnotesize 資料來源：本研究。上半部為時間配置示意，區段寬度不表示本研究的年份數。下半部之 DSEL 與 Calibration 均來自 2014 年，兩者沒有年內先後順序。\\par}
\\end{figure}

'''

baseline_params=baseline_params.replace(r'\subsection{Baseline 分類器之超參數設定}',r'\subsubsection{分類器與超參數設定}')
# Keep the settings at the beginning of Baseline, before the procedural diagram.
position=experiment1.index(r'\begin{figure}')
experiment1=experiment1[:position]+baseline_params+'\n\\subsubsection{訓練流程與時間切分}\n\n'+experiment1[position:]
experiment1=experiment1.replace(r'\subsubsection{實驗資料與年份切分}', '')
experiment1=experiment1.replace('候選分類器涵蓋傳統機器學習與以神經網路為基礎的表格模型，包括 Logistic Regression、Random Forest、XGBoost、LightGBM、TabM 與 TabICL。主要超參數設定列於本章，補充訓練控制與後端版本說明見附錄~\\ref{app:experimental-details}。',
    '本研究使用 Logistic Regression、Random Forest、XGBoost、LightGBM、TabM 與 TabICL 標記版本作為候選。各模型的搜尋候選已列於本部分開頭；補充訓練控制與後端紀錄見附錄~\\ref{app:experimental-details}。')
ensemble_params=ensemble_params.replace(r'\subsection{集成模型之固定設定}',r'\subsubsection{集成模型之固定設定}')
position=experiment1.index(r'\subsubsection{靜態集成方法}')
experiment1=experiment1[:position]+ensemble_params+'\n'+experiment1[position:]
position=experiment2.index(r'\subsection{特徵選取與模型訓練流程}')
experiment2=experiment2[:position]+fs_params+'\n'+experiment2[position:]

# General process / metric language, with the real case retained in the data section.
for before,after in [('各財務特徵','各輸入特徵'),('18 個財務特徵','18 個輸入特徵'),('公司年度財務資料','具年度資訊的觀測資料')]:
    experiment1=experiment1.replace(before,after)
validation=validation.replace('本研究將 failed 定義為正類。TP 為實際破產且預測為破產的紀錄數；FN 為實際破產但預測為存續的紀錄數；FP 為實際存續但預測為破產的紀錄數；TN 為實際存續且預測為存續的紀錄數。',
    '令關注類別為正類。TP 與 TN 分別為正確分類的正類與負類紀錄數；FN 為正類被判為負類的紀錄數；FP 為負類被判為正類的紀錄數。實證案例的正類對應第~\\ref{sec:method-data}~節定義的標籤。')
validation=validation.replace('實際破產','實際正類').replace('預測為破產','預測為正類').replace('破產樣本','正類樣本').replace('破產紀錄','正類紀錄')
validation=validation.replace('本研究使用 49,999 次重抽，重抽種子固定為 20261027。',
    '本研究使用 49,999 次重抽。為使統計分析可重現，bootstrap 重抽樣的隨機種子設為 20261027。')
result=header+data+'\n\\clearpage\n'+framework+'\n\\clearpage\n'+experiment1+experiment2+validation
assert result.count(r'\chapter{')==1
assert result.count(r'\label{sec:method-phase1-params}')==1
assert result.count(r'\label{sec:method-data}')==1
assert result.index(r'\section{研究資料}') < result.index(r'\section{通用研究架構}') < result.index(r'\section{實驗一')
assert result.index(r'\caption{Baseline 傳統分類器') < result.index(r'\caption{實驗一之 Baseline')
P.write_text(result,encoding='utf-8')
print('Reorganized data, generic framework, settings and experiment procedures in place')
