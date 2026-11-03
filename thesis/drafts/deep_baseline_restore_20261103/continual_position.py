"""Apply the author's clarification: a limited two-stage continual ensemble setting."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHAP = "thesis/latex/chapters/"


def replace(name, old, new):
    p = ROOT / name
    t = p.read_text(encoding="utf-8")
    assert t.count(old) == 1, (name, old[:80], t.count(old))
    p.write_text(t.replace(old, new), encoding="utf-8")


replace("thesis/latex/frontmatter/abstract_zh.tex",
    "本研究以具有時間資訊的不平衡分類為對象，比較以歷史（Old）與近期（New）資料建立之模型的運用方式，並評估模型集成與特徵選取的影響。",
    "本研究以兩階段的集成更新模擬持續學習：保留歷史（Old）模型，整合近期（New）資料建立的新模型，並比較集成規則與特徵選取的影響。受資料可驗證範圍與本次配置限制，實證以固定 Old／New 分區評估一次模型池擴充。")
replace("thesis/latex/frontmatter/abstract_zh.tex", "關鍵字：}類別不平衡、", "關鍵字：}持續學習、類別不平衡、")
replace("thesis/latex/frontmatter/abstract_en.tex",
    "Temporal change and class imbalance create a trade-off between historical sample size and recent information. This study compares models trained on historical (Old) and recent (New) data, ensemble rules, and feature selection for time-stamped imbalanced classification.",
    "This study examines a two-stage continual ensemble setting: retaining historical (Old) models and integrating models trained on recent (New) data. Ensemble rules and feature selection are compared. Given the available evidence and experimental configuration, the study evaluates one pool expansion using fixed temporal partitions.")
replace("thesis/latex/frontmatter/abstract_en.tex", "Keywords:} class imbalance,", "Keywords:} continual learning, class imbalance,")
replace(CHAP + "01_introduction.tex",
    "本研究因此從歷史與近期資料的運用方式出發，比較分別建模、合併重訓及模型集成的表現。",
    "本研究因此從歷史與近期資料的運用方式出發，以保留歷史模型並整合近期新模型的兩階段流程，探討集成式持續學習中的知識保留與新資訊利用，並以分別建模及合併重訓作為對照。")
replace(CHAP + "01_introduction.tex",
    "本研究探討非穩態資料下，如何運用歷史與近期模型；但所有實驗都採固定年份切分，測試期間不更新模型，因此不驗證持續學習或線上適應的能力。",
    "本研究以模型池由 Old 擴充為 Old／New 混合池，呈現模型層級的兩階段持續學習情境。受資料可驗證範圍與本次實驗配置限制，主要分析採固定年份分區、離線模擬一次更新，再測試後續年度；尚未評估多輪持續更新、遺忘程度或長期線上適應。")
replace(CHAP + "02_literature.tex",
    "本研究主要事件標籤的 Baseline、集成與 FS 都固定一組年份配置，測試期間內沒有線上更新。第三章另以概念圖區分滑動配置與目前固定配置，目的在於讓讀者看清楚時間角色，而非將未執行的更新機制寫成成果。",
    "本研究主要事件標籤的 Baseline、集成與 FS 都固定一組年份配置，藉由 Old 模型池加入 New 模型，模擬一次模型層級的更新，再以後續期間評估；Test 期間未再更新。第三章的滑動概念圖用於說明多輪配置與本次有限兩階段配置的差別，不將未執行的長序列更新寫成成果。")
section = r"""\subsection{集成式持續學習與兩階段更新}

持續學習的更新可以發生在模型集成層級。Elwell 與 Polikar\autocite{elwell2011incremental} 的 Learn++.NSE 隨新資料批次訓練分類器，再以加權整合運用既有與新模型，提供批次增量集成的例子。此處引用其更新形式作為方法定位依據；本研究未重現其特定權重機制，也不沿用其效能結論。

本研究將此問題限縮為兩個時間階段：先以歷史資料建立 Old 模型，再以近期資料建立 New 模型，在保留 Old 候選的情況下擴充模型池，並決定如何整合。更新的是集成成員及其使用方式，不要求每個基礎分類器都延續舊模型權重。固定後續測試期用來評估這次更新的結果；它限制了可觀察的更新次數，並不取消 Old 到 Old／New 的兩階段設計。對多輪更新的遺忘、知識累積及長期穩定性，仍需其他評估。

"""
replace(CHAP + "02_literature.tex", r"\section{類別不平衡與採樣方法}", section + r"\section{類別不平衡與採樣方法}")
replace(CHAP + "03_method.tex",
    "本研究以具有時間資訊的不平衡分類資料為對象，比較歷史與近期資料的使用方式，並評估模型集成與特徵選取的影響。",
    "本研究以具有時間資訊的不平衡分類資料為對象，透過歷史模型保留與近期模型加入，探討兩階段集成式持續學習，並評估集成規則與特徵選取的影響。")
replace(CHAP + "03_method.tex", "Old／New 的期間長度、驗證方式與分類器可依應用設定。",
    r"Old／New 的期間長度、驗證方式與分類器可依應用設定。以 \(\mathcal{O}\) 表示歷史模型池、\(\mathcal{N}\) 表示近期新增模型，框架的更新由 \(\mathcal{O}\) 擴充至 \(\mathcal{O}\cup\mathcal{N}\)，再以驗證資料決定成員與門檻；Old3 與 New3 分別作為只保留單一期間的控制組。實作在同一離線實驗中重建各設定，以模擬這次模型池更新，沒有將多次線上部署或連續更新當成已觀察結果。")
replace(CHAP + "04_experiments.tex",
    "主要實驗也採固定年份、測試期間不更新模型，因此結果不代表對未見公司或持續學習情境的驗證。",
    "主要實驗以固定年份分區離線模擬 Old 到 Old／New 的一次模型池更新，Test 期間未再更新，因此結果限於兩階段持續集成情境，不能延伸為對未見公司、多輪遺忘或長期線上適應的驗證。")
replace(CHAP + "05_conclusion.tex",
    "本研究以具有時間資訊的不平衡分類為對象，在美國企業年度財務資料上比較歷史與近期模型的運用方式。",
    "本研究以具有時間資訊的不平衡分類為對象，在美國企業年度財務資料上，以 Old 模型池加入 New 模型的兩階段更新模擬集成式持續學習，並比較歷史與近期模型的運用方式。")
replace(CHAP + "05_conclusion.tex",
    "第四，所有實驗採固定年份切分，測試期間不更新模型，未驗證持續學習或線上適應，也沒有由年份差異單獨證明概念漂移。",
    "第四，受資料可驗證範圍與本次配置限制，主實驗只模擬 Old 到 Old／New 的一次模型池擴充；Test 期間沒有再更新，尚未評估多輪持續學習的遺忘或長期線上適應，也沒有由年份差異單獨證明概念漂移。")
replace(CHAP + "05_conclusion.tex", "第三，採用滑動視窗或逐期更新，檢驗不同年份配置下的穩定性。",
    "第三，將本次兩階段集成擴充為多輪滑動視窗或逐期更新，量測新資訊吸收、歷史知識保留與不同年份配置下的穩定性。")

p = ROOT / "thesis/latex/references.bib"
t = p.read_text(encoding="utf-8")
assert "elwell2011incremental" not in t
t += r"""
@article{elwell2011incremental,
  author = {Elwell, Ryan and Polikar, Robi},
  title = {Incremental Learning of Concept Drift in Nonstationary Environments},
  journal = {IEEE Transactions on Neural Networks},
  year = {2011},
  volume = {22},
  number = {10},
  pages = {1517--1531},
  doi = {10.1109/TNN.2011.2160459}
}
"""
p.write_text(t, encoding="utf-8")
print("Added continual-learning keywords and bounded two-stage ensemble positioning; no experiment changes.")
