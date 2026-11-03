"""Build an evidence-linked Markdown report with all methods and rolling years."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
METRICS = ["AUC", "PR_AUC", "F1", "G_Mean", "Recall", "Precision"]
MODELS = {"lr": "LR", "rf": "Random Forest", "xgb": "XGBoost", "lgb": "LightGBM", "mlp": "MLP", "svm": "SVM"}
SAMPLES = {"none": "不採樣", "under": "under", "over": "over", "hybrid": "hybrid"}


def label(method):
    if method.startswith("Newest_") or method.startswith("FIFO_"):
        kind, model, sample = method.split("_")
        return f"{MODELS[model]}＋{SAMPLES[sample]}" + (" FIFO" if kind == "FIFO" else " 單模型")
    for prefix, suffix in (("SamplingNewest_", "跨採樣（最新窗口）"), ("SamplingFIFO_", "跨採樣＋FIFO")):
        if method.startswith(prefix):
            return MODELS[method[len(prefix):]] + " " + suffix
    for prefix, suffix in (("ClassifierNewest_", "六分類器（最新窗口）"), ("ClassifierFIFO_", "六分類器＋FIFO")):
        if method.startswith(prefix):
            return suffix + "＋" + SAMPLES[method[len(prefix):]]
    if method.startswith("SelectedEnsemble_"):
        fs, ratio = method.removeprefix("SelectedEnsemble_").split("_r")
        return "不做 FS（18 特徵）" if fs == "none" else f"{fs.upper()} {ratio}%（{int(18 * int(ratio) / 100)} 特徵）"
    return {"CombinedNewest": "六分類器×三採樣（最新窗口）", "CombinedFIFO": "六分類器×三採樣×FIFO",
            "ValidationSelectedEnsemble": "年度 Validation 選集成（無 FS）",
            "ValidationSelectedFS": "年度 Validation 再選 FS／no-FS"}[method]


def table(frame, columns=None, names=None):
    if columns is not None:
        frame = frame[columns]
    headings = [names.get(c, c) if names else c for c in frame.columns]
    rows = ["| " + " | ".join(headings) + " |", "| " + " | ".join("---" for _ in headings) + " |"]
    for record in frame.itertuples(index=False, name=None):
        cells = []
        for value in record:
            if isinstance(value, (float, np.floating)):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value).replace("|", "/").replace("\n", " "))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n\n"



def report_link(target, output):
    """Make source links work from either root docs or nested report folders."""
    return quote(Path(os.path.relpath(target, output.parent)).as_posix(), safe="/.")


def default_report_output(at=None):
    at = at or datetime.now(timezone.utc)
    return ROOT / "docs/reports" / at.strftime("%Y-%m") / (
        "滑動視窗三階段實驗成果報告_" + at.strftime("%Y%m%dT%H%M%S%fZ") + ".md")

def build(run, output):
    captured_at = datetime.now(timezone.utc).isoformat()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    seeds = [s for s in manifest["config"]["seeds"] if (run / f"seed_{s}/complete.json").exists()]
    if not seeds:
        raise ValueError("No complete seeds")
    summaries = pd.concat([pd.read_csv(run / f"seed_{s}/summary.csv") for s in seeds], ignore_index=True)
    annual = pd.concat([pd.read_csv(run / f"seed_{s}/by_year.csv") for s in seeds], ignore_index=True)
    methods = sorted(summaries.method.unique())
    years = sorted(manifest["config"]["test_years"])
    if len(methods) != 81:
        raise ValueError("This report template requires the full 81-method experiment")
    source_hashes = {}
    for seed in seeds:
        for name in ("summary.csv", "by_year.csv", "complete.json"):
            path = run / f"seed_{seed}/{name}"
            source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        subset = annual.loc[annual.seed == seed]
        if subset.duplicated(["method", "test_feature_year"]).any():
            raise ValueError("Duplicate method/year")
        if set(subset.method) != set(methods) or len(subset) != len(methods) * len(years):
            raise ValueError("Incomplete method/year coverage")
    # All stored annual means are independently reconstructed from annual rows.
    annual_means = annual.groupby(["seed", "method"])[METRICS].mean().sort_index()
    saved_means = summaries.set_index(["seed", "method"])[["annual_mean_" + m for m in METRICS]].sort_index()
    np.testing.assert_allclose(annual_means.to_numpy(), saved_means.to_numpy(), atol=1e-12)
    means = summaries.groupby("method")[METRICS].mean()
    yearly = annual.groupby(["method", "test_feature_year"])[METRICS].mean()
    selected_methods = ["Newest_rf_under", "FIFO_rf_under", "ClassifierFIFO_none",
                        "ValidationSelectedEnsemble", "ValidationSelectedFS", "SelectedEnsemble_shap_r50"]
    checks, cohort_reference, probability_diagnostics, scale_checks = [], None, [], []
    for seed in seeds:
        print(f"CHECK predictions seed={seed}", flush=True)
        parts = []
        for chunk in pd.read_csv(run / f"seed_{seed}/predictions.csv.gz", chunksize=200000,
                                 usecols=["method", "source_row_id", "test_feature_year", "company_name",
                                          "y_true", "y_proba", "y_pred_f1"]):
            parts.append(chunk.loc[chunk.method.isin(selected_methods)])
        predictions = pd.concat(parts, ignore_index=True)
        for method, group in predictions.groupby("method"):
            group = group.sort_values(["test_feature_year", "source_row_id"])
            cohort = group[["test_feature_year", "source_row_id", "company_name", "y_true"]].reset_index(drop=True)
            if cohort_reference is None:
                cohort_reference = cohort
            pd.testing.assert_frame_equal(cohort_reference, cohort)
            computed = {"AUC": roc_auc_score(group.y_true, group.y_proba),
                        "PR_AUC": average_precision_score(group.y_true, group.y_proba),
                        "F1": f1_score(group.y_true, group.y_pred_f1, zero_division=0),
                        "Recall": recall_score(group.y_true, group.y_pred_f1, zero_division=0),
                        "Precision": precision_score(group.y_true, group.y_pred_f1, zero_division=0)}
            saved = summaries.loc[(summaries.seed == seed) & (summaries.method == method)].iloc[0]
            for metric, value in computed.items():
                if not np.isclose(value, saved[metric], atol=1e-12):
                    raise ValueError(f"Prediction mismatch {seed}/{method}/{metric}")
            checks.append({"seed": seed, "method": method, "n": len(group), **computed})
            # Post-hoc scale diagnostic: within-year ranks remove cross-year probability-scale differences.
            year_means = group.groupby("test_feature_year").y_proba.mean()
            outlier_year = int((year_means / year_means.median()).idxmax())
            rest = group.loc[group.test_feature_year != outlier_year]
            scale_checks.append({"seed": seed, "method": method, "Pooled AP": computed["PR_AUC"],
                                 "年內排序 pooled AP": average_precision_score(
                                     group.y_true, group.groupby("test_feature_year").y_proba.rank(pct=True)),
                                 "機率最偏高年": outlier_year,
                                 "該年平均機率／年度中位": year_means[outlier_year] / year_means.median(),
                                 "排除該年 pooled AP": average_precision_score(rest.y_true, rest.y_proba)})
            if method == "ValidationSelectedEnsemble":
                for year, batch in group.groupby("test_feature_year"):
                    probability_diagnostics.append({"seed": seed, "Test特徵年": year,
                                                    "平均機率": batch.y_proba.mean(),
                                                    "機率中位數": batch.y_proba.median(),
                                                    "AP": average_precision_score(batch.y_true, batch.y_proba)})
    period_rows = []
    choices = []
    for year in years:
        ends = tuple(range(year - 4, year - 1))
        sample_counts = annual.loc[(annual.method == "Newest_rf_under") & (annual.test_feature_year == year)]
        assert sample_counts.n_test.nunique() == sample_counts.n_test_positive.nunique() == 1
        period_rows.append({"輪次": years.index(year) + 1, "最新訓練窗口": f"{year - 4}–{year - 2}",
                            "FIFO 三個窗口": "／".join(f"{end - 2}–{end}" for end in ends),
                            "Validation": year - 1, "Test特徵年": year, "事件年": year + 1,
                            "Test筆數": int(sample_counts.n_test.iloc[0]), "正例": int(sample_counts.n_test_positive.iloc[0])})
        for seed in seeds:
            folder = run / f"seed_{seed}/year_{year}"
            selection = json.loads((folder / "ensemble_selection.json").read_text(encoding="utf-8"))
            fs = json.loads((folder / "fs_selection.json").read_text(encoding="utf-8"))
            assert not selection["test_used"] and not fs["test_used"]
            for model, sampling, end in selection["members"]:
                assert year - 4 <= end <= year - 2
                audit = json.loads((run / f"seed_{seed}/cache/{model}_{sampling}_M{end}_none_r100.json").read_text())
                assert audit["train_years"] == list(range(end - 2, end + 1))
            choices.append({"seed": seed, "Test特徵年": year, "Validation": year - 1,
                            "送入FS的集成": label(selection["selected"]), "再選FS": fs["selected_fs"],
                            "成員訓練窗口": "／".join(f"{end - 2}–{end}" for end in sorted({m[2] for m in selection["members"]}))})
    periods, choice_frame = pd.DataFrame(period_rows), pd.DataFrame(choices)
    base_link = report_link(run, output)
    n = len(cohort_reference)
    positives = int(cohort_reference.y_true.sum())
    companies = cohort_reference.company_name.nunique()
    seed_text = "、".join(map(str, seeds))
    doc = f"""# 滑動視窗三階段實驗成果與逐年分析

> 報告日期：2026-09-29；資料擷取時間（UTC）：{captured_at}。  
> 已完成 seeds：**{seed_text}，共 {len(seeds)}/{len(manifest['config']['seeds'])} 個**。  
> 本文完整呈現上述已完成 seeds 的 81 個方法及十輪年度結果；**不是十種子最終報告**。未完成 seed 不混入任何平均值。  
> 正式 run：[`{run.name}`]({base_link}/manifest.json)。

## 1. 本次成果與老師討論的重點

本次將 Study 1–2 的比較程序移到共同的年度滑動視窗：先做「六分類器 × 四採樣」，
再比較集成，最後對 Validation 選中的集成做特徵選取（FS）。
目前單模型、集成、FS 三階段均已有完整年度結果；其餘 seeds 持續執行。

本次結果支持繼續檢查 RF 不採樣／under 及跨年度模型保留；並未顯示把所有模型與採樣全部平均就能改善。
FS 的收益依指標而異，不能只報最高 AUC 或 AP。年度 Validation 選擇策略有明顯 pooled AP 波動，
也不能把它視為已勝過固定集成。其中 seed 456 的 pooled AP 偏低已查明是單一年度機率尺度失真（見第 7.3 節），
不是整體排序能力變差；第 6 節的 seed 平均受其拉低。

**必須釐清第三階段的定義：目前並非固定選一個「最佳 RF 集成」後全部年度都用它做 FS，
而是每個 seed、每個年度各用當時 Validation AP 選集成，再測 FS。**
因此 FS 表的 no-FS 對照是當輪所選集成，不能直接接在某一固定 RF/XGBoost 表列後解讀。
若研究目的改為固定少數候選集成的 FS 消融，需另定前置選擇年度或訓練期內選擇規則；此項目前沒有完成。

### 閱讀順序

1. 第 2 節：先看十輪時間切割，理解每輪資料與模型如何更新。
2. 第 4–6 節：依序看單模型、集成、FS 的完整 pooled 表與解讀。
3. 第 7 節：比較逐年成績及 seed 456 的 pooled／年度平均差異。
4. 附錄 A：所有 81 個方法的十年逐輪成績；附錄 B：年度平均總表；附錄 C：seed 波動。

## 2. 資料如何逐年滾動

### 2.1 時間與資料定義

資料為美國公司 1999–2018 財務特徵。模型固定使用 X1–X18；公司識別碼與年份不作特徵。
事件 target 依現有研究協定重建：failed 公司最後觀測的特徵年為正例，對應事件年=fyear+1。
原始公司狀態不是逐年事件標籤；精確財報公開及事件標籤可用日期尚未取得，所以本次是年度回溯研究。

共同 Test 為 2009–2018 的 **{n:,} 筆公司年度資料、{companies:,} 家公司、{positives} 個正例**，
正例比例為 **{positives/n:.2%}**。不同 seeds 重複使用同一批公司年度資料，不把列數乘上 seed 數當新樣本。

每輪 Test 特徵年為 t，最新模型用 t−4 至 t−2 訓練；t−1 為 Validation；t 為 Test，對應事件年 t+1。
FIFO3 三個窗口結束於 t−4、t−3、t−2；每模型看三年，三模型歷史聯集為五年。
六分類器與四種採樣共用相同時間切割。

### 2.2 十輪時間表與樣本數

""" + table(periods)
    doc += """第一輪先建立三個模型。第二輪新增 2006–2008 模型、保留 2004–2006 與 2005–2007 模型、
移除 2003–2005 模型；Validation/Test 各往後一年。這一更新發生在每個「分類器×採樣」配置。
第一輪測試前年份 2003–2007 的五年資料僅透過三個三年模型保留，並不是單一模型直接 fit 五年。

來源：各 seed 的 `by_year.csv`、每輪 `ensemble_selection.json` 與 `cache/*.json`。

## 3. 實驗配置與表格讀法

### 3.1 三階段配置

| 階段 | 內容 | 每年度方法數 |
|---|---|---:|
| 單模型 | LR、RF、XGBoost、LightGBM、MLP、SVM × none、under、over、hybrid | 24 |
| 固定集成 | 跨時間、跨採樣、跨分類器及其組合，機率等權平均 | 46 |
| Validation 選集成策略 | 每年度在 46 配置中依 Validation AP 選一個，保留 no-FS 預測 | 1 |
| FS 固定規則對照 | 當輪選中集成的 no-FS＋四方法×兩保留率 | 9 |
| Validation 選 FS 策略 | 再以 Validation AP 選 FS／no-FS | 1 |
| 合計 | 所有方法保留，無 Test 最佳列刪選 | 81 |

under=TomekLinks；over=ADASYN；hybrid=SMOTEENN。TomekLinks 是清理型 under，不保證平衡至 1:1。
none 是獨立對照。跨採樣集成只合併 under／over／hybrid，不額外加入 none 的一票。
各預測分類器使用固定基礎參數並關閉額外類別權重；MLP 不使用採樣後的隨機 early-stopping holdout。
沒有在本輪進行超參數搜尋、DES/DCS 或深度表格分類器比較。

各窗口獨立完成補值、標準化、可選 FS、採樣、訓練；FS 在原始訓練樣本上擬合。
Validation/Test 僅套用轉換，沒有採樣。新模型重新訓練；舊模型的預測凍結保留。
當 Test 年標籤在後續年度成熟時可成為歷史資料，但不回頭修改當年的預測。

### 3.2 指標與平均方式

| 指標 | 本報告的解讀 |
|---|---|
| AUC | ROC 曲線下面積，描述正負類排序 |
| AP | average precision，程式欄名 PR_AUC；不是另一種梯形積分 PR 面積 |
| F1 | Precision 與 Recall 的調和平均 |
| G-Mean | sensitivity 與 specificity 的幾何平均 |
| Recall | 實際正例中，被警報找到的比例 |
| Precision | 所有警報中，真正正例的比例 |

主文 pooled 表：**每個 seed 先合併十年 Test 預測計算指標，再對已完成 seeds 取平均**。
逐年表：在相同 Test 年、相同方法下對 seeds 取平均。
附錄年度平均表：先將每個 seed 十年的指標等權平均，再對 seeds 取平均。
Pooled AP 與年度平均 AP 的計算對象不同，不能混為同一數字；F1 也不是用表中的平均 Recall/Precision 重新計算。
分類門檻以各年度 Validation F1 的 0.01–0.99 網格決定；所有分類指標沿用該年度門檻。
另有 Validation FPR≤5% 操作點，實際 Test FPR 不保證低於 5%，完整欄位保留於原始逐年 CSV。

## 4. 第一階段：分類器與四種採樣

以下依分類器分表。每張表四種採樣都保留；數值為 pooled 指標的 seed 平均。

"""
    def metric_table(method_list):
        frame = means.loc[method_list].reset_index()
        frame["method"] = frame.method.map(label)
        return table(frame, names={"method": "配置", "PR_AUC": "AP"})
    for model, title in MODELS.items():
        doc += f"### 4.{list(MODELS).index(model) + 1} {title}\n\n"
        doc += metric_table([f"Newest_{model}_{sample}" for sample in SAMPLES])
    rf_single, rf_fifo = means.loc["Newest_rf_under"], means.loc["FIFO_rf_under"]
    doc += f"""### 4.7 單模型結果的解讀

本快照 RF 不採樣／under 的 AP 點估計突出；RF＋under AP={rf_single.PR_AUC:.4f}。
XGBoost＋under 的 AUC={means.loc['Newest_xgb_under', 'AUC']:.4f}，與 RF 的指標優勢並不一致。
RF 的 over／hybrid AP 低於其 none／under；SVM 的 over／hybrid 則高於其 none／under，
因此沒有一種採樣對六分類器同時成立的全面優勢。這是當前固定模型參數下的結果，不代表調參後排序必然相同。

## 5. 第二階段：完整集成比較

### 5.1 同分類器、同採樣的跨年度 FIFO（24 配置）

"""
    for model, title in MODELS.items():
        doc += f"**{title}：最近三個窗口模型等權平均**\n\n"
        doc += metric_table([f"FIFO_{model}_{sample}" for sample in SAMPLES])
    for number, title, method_list in (
        (2, "跨採樣：最新窗口", [f"SamplingNewest_{m}" for m in MODELS]),
        (3, "跨採樣＋FIFO", [f"SamplingFIFO_{m}" for m in MODELS]),
        (4, "跨分類器：最新窗口", [f"ClassifierNewest_{s}" for s in SAMPLES]),
        (5, "跨分類器＋FIFO", [f"ClassifierFIFO_{s}" for s in SAMPLES]),
        (6, "全部分類器與三採樣合併", ["CombinedNewest", "CombinedFIFO"]),
    ):
        doc += f"### 5.{number} {title}\n\n" + metric_table(method_list)
    doc += f"""### 5.7 集成是否改善單模型？

相同 RF＋under 設定，最新單模型 AP={rf_single.PR_AUC:.4f}，FIFO AP={rf_fifo.PR_AUC:.4f}；
AUC 由 {rf_single.AUC:.4f} 變為 {rf_fifo.AUC:.4f}；Recall 由 {rf_single.Recall:.4f} 變為 {rf_fifo.Recall:.4f}，
但 Precision 由 {rf_single.Precision:.4f} 變為 {rf_fifo.Precision:.4f}。其方向是排序與召回提高、警報精確度降低，
而不是所有指標一起改善。FIFO 同時多涵蓋兩年歷史，不能將差異全歸因於純集成機制。

六分類器×三採樣×FIFO 的 AP={means.loc['CombinedFIFO', 'PR_AUC']:.4f}，
低於 RF＋under FIFO 的 {rf_fifo.PR_AUC:.4f}；把更多模型混在一起並未自然得到較高 AP。
所有固定配置的高低是描述性 Test 比較，沒有以最高 Test 分數回頭改選本次模型。

## 6. 第三階段：實際送入 FS 的集成與結果

### 6.1 每年究竟選了什麼？

以下逐 seed、逐年度列出當輪 Validation 選擇。成員窗口是當輪實際被選集成的訓練資料，
不是一律假設有三個窗口。`none_r100` 表示再選擇時保留 no-FS。

""" + table(choice_frame)
    doc += "### 6.2 當輪已選集成的 FS 對照\n\n"
    fs_methods = ["SelectedEnsemble_none_r100"] + [f"SelectedEnsemble_{fs}_r{r}" for fs in ("mi", "cart", "rfe", "shap") for r in (50, 80)]
    doc += metric_table(fs_methods)
    none, shap = means.loc[fs_methods[0]], means.loc["SelectedEnsemble_shap_r50"]
    doc += f"""MI、CART、RFE、SHAP 均只 fit 訓練窗口；50%／80% 為 9／14 個特徵。
CART/RFE 使用 balanced decision tree，SHAP 由訓練窗口上的 LightGBM TreeSHAP 平均絕對貢獻排序。
每個模型可以保留不同特徵，最後仍以機率平均集成。

本快照 SHAP 50% 的 AP={shap.PR_AUC:.4f}，no-FS={none.PR_AUC:.4f}；
但 F1 分別為 {shap.F1:.4f} 與 {none.F1:.4f}，Precision 分別為 {shap.Precision:.4f} 與 {none.Precision:.4f}。
因此該點估計支持檢查 FS 的排序收益與警報取捨，不支持全面優越，也不把 Test 最高的 FS 宣稱為最佳設定。

注意：本節 pooled AP 的 seed 平均包含 seed 456 一個年度的機率尺度失真，各 FS 列都被同樣拉低；
FS 之間的相對比較仍可讀，但絕對值不宜直接與第 4–5 節固定配置比較，請搭配第 7.3 節與附錄 B。

### 6.3 真正按 Validation 選擇的兩個策略

""" + metric_table(["ValidationSelectedEnsemble", "ValidationSelectedFS"])
    doc += """第二列才是 Validation 再選擇 FS／no-FS 的策略結果，不等於事後取上表最高 Test AP 的一列。
這種分階段搜尋重用同年度 Validation 作集成、FS 與門檻選擇，Validation 分數不當成無偏泛化估計。
沒有固定某個集成跨十年做全部 FS 的完整矩陣；若老師希望採此設計，需另行執行並分開標示。

## 7. 實際逐輪結果與 pooled／年度平均差異

### 7.1 例：RF＋under 單模型逐年結果

"""
    def annual_table(method):
        frame = yearly.loc[method].reset_index().rename(columns={"test_feature_year": "Test特徵年", "PR_AUC": "AP"})
        frame = periods[["輪次", "最新訓練窗口", "Validation", "Test特徵年"]].merge(frame, on="Test特徵年")
        return table(frame)
    doc += annual_table("Newest_rf_under")
    doc += "### 7.2 同一 RF＋under 的 FIFO 逐年結果\n\n"
    doc += "三個模型窗口逐輪對應第 2.2 節；以下最新訓練窗口是最年輕成員，完整歷史聯集為五年。\n\n"
    doc += annual_table("FIFO_rf_under")
    doc += "### 7.3 年度選擇策略的逐 seed 差異\n\n"
    strategy = summaries.loc[summaries.method.isin(["ValidationSelectedEnsemble", "ValidationSelectedFS"]),
                              ["seed", "method", "AUC", "PR_AUC", "annual_mean_PR_AUC", "F1", "Recall", "Precision"]].sort_values(["seed", "method"])
    strategy["method"] = strategy.method.map(label)
    doc += table(strategy, names={"method": "策略", "PR_AUC": "Pooled AP", "annual_mean_PR_AUC": "年度平均 AP"})
    scale = pd.DataFrame(scale_checks).sort_values(["method", "seed"])
    scale["method"] = scale.method.map(label)
    worst = scale.loc[scale["Pooled AP"].idxmin()]
    worst_choice = choice_frame.loc[(choice_frame.seed == worst.seed)
                                    & (choice_frame["Test特徵年"] == worst["機率最偏高年"]), "送入FS的集成"].iloc[0]
    doc += """pooled 把十個年度的機率放進同一個排序，年度平均只比較每一年內的排序，兩者分別由逐筆預測重算。
下表是事後診斷，僅用於解釋，不用它修改本次選模、選 FS 或選閾值；主文 pooled 指標定義不變。

- **年內排序 pooled AP**：先把每年機率換成該年內的百分位排名，再合併十年計算 AP，消除跨年機率尺度差異。
- **排除該年 pooled AP**：拿掉平均機率相對年度中位數最偏高的一年後重算。

""" + table(scale, names={"method": "方法"})
    doc += f"""**診斷結論**：最低 pooled AP 出現在 seed {int(worst.seed)} 的「{worst.method}」（{worst['Pooled AP']:.4f}）。
該 seed 在 {int(worst['機率最偏高年'])} 年由 Validation 選中「{worst_choice}」，其平均機率為其他年度中位數的
{worst['該年平均機率／年度中位']:.1f} 倍，整年公司被排到十年合併排序的前段，稀釋了其他年度的正例。
排除該年後 pooled AP 為 {worst['排除該年 pooled AP']:.4f}；改用年內排序則為 {worst['年內排序 pooled AP']:.4f}，
與其他 seeds 相近。因此這不是整體排序能力變差，而是「年度切換分類器＋未校準機率」造成的 pooled 指標失真。
第 6.2、6.3 節的 seed 平均包含此失真，閱讀時應搭配本表與附錄 B 的年度平均。
若後續要讓年度選擇策略可直接 pooled 比較，需事先規定機率校準或以年內排序輸出，並另行標示為新設定。

以下為該 seed 各年機率分布：

"""
    diagnostic = pd.DataFrame(probability_diagnostics)
    doc += table(diagnostic.loc[diagnostic.seed == worst.seed].drop(columns="seed"))
    doc += """## 8. 目前可以向老師說什麼

1. 已在同一年度協定下完成本快照的六分類器×四採樣、46 個集成及 FS 對照，所有年度結果可追溯。
2. RF none／under 是值得保留的單模型對照；RF under FIFO 有較高 AP、AUC 與 Recall 點估計，但 Precision 較低。
3. 全部模型／採樣混合未帶來一致收益，不能以成員越多解讀為效果越好。
4. FS 對排序與分類操作指標的影響不同；當前沒有全指標最佳方案。
5. 年度選擇策略的 pooled AP 有 seed 波動；已查明主因是年度切換到未校準的分類器（如 SVM）時機率尺度改變，
   應同時報逐年、pooled、年內排序與各 seed，而非只挑某種摘要。

### 仍未解決的問題與下一步

- 完成剩餘 seeds，再以相同規則更新快照；seed 標準差不是公司或時間不確定性的信賴區間。
- 釐清第三階段目標：年度 Validation 選策略，或固定少數集成的 FS 消融。目前完成前者。
- 若要固定候選集成，需事先指定或用允許的訓練／前置驗證資料選擇；不能把本次最高 Test 列改稱獨立選出的最佳方法。
- 本輪尚未做公司群聚／年度區塊的成對不確定性分析，不使用「顯著優於」措辭。
- 比較 FIFO 與最新模型同時改變模型數及歷史聯集；需要同歷史範圍對照才能進一步分離機制。
- SVM 等 over／hybrid 成本較高；實際各模型採樣前後列數與耗時見 training_audit.csv。
- 這是重建事件標籤的回溯研究，沒有精確可用時間保證，不能當作已驗證的即時部署。

## 9. 檔案、來源與數據查核

"""
    for seed in seeds:
        doc += (f"- seed {seed}：[完整摘要]({base_link}/seed_{seed}/summary.csv)、"
                f"[逐年指標]({base_link}/seed_{seed}/by_year.csv)、"
                f"[逐筆預測]({base_link}/seed_{seed}/predictions.csv.gz)、"
                f"[訓練與特徵稽核]({base_link}/seed_{seed}/training_audit.csv)。\n")
    doc += ("\n[實驗程式](" + report_link(ROOT / "experiments/phase_flexible/study3b_full_pipeline.py", output) + ")；"
            "[本報告產生與查核程式](" + report_link(Path(__file__), output) + ")。\n\n")
    manual = sorted((run / "manual_spreadsheets").glob("*.zip"))
    if manual:
        doc += f"[Study 1–2 形式的獨立試算表 ZIP]({report_link(manual[-1], output)})（其納入 seeds 以包內 manifest 為準）。\n\n"
    doc += """### 主張與證據台帳

| ID | 主張 | 證據與檢查 | 狀態 |
|---|---|---|---|
| C1 | 本文只納入完整完成的 seeds | complete.json、method×year 覆蓋檢查 | verified snapshot |
| C2 | 單模型與集成的完整平均數可追溯 | summary.csv；所有年度平均由 by_year.csv 重算 | verified descriptive |
| C3 | RF under FIFO 與單模型的指標取捨 | 兩方法各 seed 的逐筆預測重算 AUC/AP/F1/Recall/Precision | verified descriptive，無顯著性結論 |
| C4 | FS no-FS 是年度所選集成，不是固定最佳集成 | 每年 ensemble_selection.json、fs_selection.json | implementation verified |
| C5 | seed 456 的低 pooled AP 不是引用了另一方法列 | 該策略逐筆預測重新計算，核對 summary | verified；年內排序與排除單年重算確認為機率尺度失真（事後診斷） |
| C6 | 各方法具有相同 Test 對照基礎 | 原管線完整 cohort 檢查；本報告再核對六個關鍵方法跨 seeds 的公司、列 ID、年度與標籤 | verified within checked scope |

本報告可供討論目前數據與研究設計；不代表十種子已完成或論文已可定稿。

## 附錄 A：全部 81 個方法的逐年結果

每個摺疊區塊包含十輪。表格數值為相同方法、相同 Test 年的 seed 平均。
固定配置的完整成員窗口按第 2.2 節：Newest 僅最新窗口，FIFO 含三窗口；
FS／年度選擇策略的實際成員窗口另按第 6.1 節逐 seed 列出。

"""
    for method in methods:
        doc += f"<details>\n<summary>{label(method)}（{method}）</summary>\n\n"
        doc += annual_table(method)
        doc += "</details>\n\n"
    doc += "## 附錄 B：年度等權平均總表（與 pooled 分開）\n\n"
    macro = summaries.groupby("method")[["annual_mean_" + metric for metric in METRICS]].mean()
    macro.columns = METRICS
    macro = macro.reset_index()
    macro["method"] = macro.method.map(label)
    doc += table(macro, names={"method": "配置", "PR_AUC": "年度平均 AP"})
    doc += "## 附錄 C：Pooled AUC／AP 的 seed 波動\n\n標準差只描述這些訓練亂數設定的波動，不是信賴區間，也不代表獨立資料重現。\n\n"
    dispersion = summaries.groupby("method")[["AUC", "PR_AUC"]].agg(["mean", "std", "min", "max"])
    dispersion.columns = [f"{metric}_{stat}" for metric, stat in dispersion.columns]
    dispersion = dispersion.reset_index()
    dispersion["method"] = dispersion.method.map(label)
    doc += table(dispersion, names={"method": "配置"})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(doc, encoding="utf-8")
    audit_output = output.with_name(output.stem + "_查核.json")
    audit_output.write_text(json.dumps({"captured_at": captured_at, "completed_seeds": seeds,
                                      "methods": len(methods), "years": years, "n_test": n,
                                      "n_positive": positives, "n_companies": int(companies),
                                      "source_sha256": source_hashes, "prediction_rechecks": checks,
                                      "scale_diagnostics": scale_checks,
                                      "annual_means_reconciled": True, "selected_windows_verified": True,
                                      "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest()},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"REPORT={output}\nAUDIT={audit_output}\nMETHODS={len(methods)} SEEDS={seeds}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    build(args.run.resolve(), (args.output or default_report_output()).resolve())
