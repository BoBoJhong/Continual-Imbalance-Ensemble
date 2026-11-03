"""Export completed seeds as separate baseline, ensemble and FS tables.

Pooled and annual-macro estimates are never mixed. No training or selection is
changed; the export is an immutable snapshot of completed seed artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["AUC", "PR_AUC", "F1", "G_Mean", "Recall", "Precision"]
HEADERS = ["AUC", "AP", "F1", "G-Mean", "Recall", "Precision"]
CLASSIFIERS = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
               "lgb": "LightGBM", "mlp": "MLP", "svm": "SVM"}
FAMILIES = {"temporal": "跨年度 FIFO", "sampling": "跨採樣（最新窗口）",
            "sampling_temporal": "跨採樣＋FIFO", "classifier": "跨分類器（最新窗口）",
            "classifier_temporal": "跨分類器＋FIFO", "combined": "跨分類器＋跨採樣（最新窗口）",
            "combined_temporal": "跨分類器＋跨採樣＋FIFO"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def markdown_table(frame):
    lines = ["| 配置 | " + " | ".join(HEADERS) + " |", "|---|" + "---:|" * len(METRICS)]
    for row in frame.itertuples(index=False):
        values = [f"{getattr(row, metric):.4f}" for metric in METRICS]
        lines.append(f"| `{row.method}` | " + " | ".join(values) + " |")
    return "\n".join(lines)


def export(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    seeds = [seed for seed in manifest["config"]["seeds"] if (run / f"seed_{seed}/complete.json").exists()]
    if not seeds:
        raise ValueError("No completed seed; refusing to mix incomplete methods")
    summaries, annual, sources = [], [], {}
    for seed in seeds:
        for filename in ("summary.csv", "by_year.csv", "complete.json"):
            path = run / f"seed_{seed}/{filename}"
            sources[str(path.relative_to(run))] = sha256(path)
        summary = pd.read_csv(run / f"seed_{seed}/summary.csv")
        yearly = pd.read_csv(run / f"seed_{seed}/by_year.csv")
        if set(summary.seed) != {seed} or set(yearly.seed) != {seed}:
            raise ValueError("Seed provenance mismatch")
        if summary.method.duplicated().any() or yearly.duplicated(["method", "test_feature_year"]).any():
            raise ValueError("Duplicate method/year")
        summaries.append(summary)
        annual.append(yearly)
    summary, yearly = pd.concat(summaries, ignore_index=True), pd.concat(annual, ignore_index=True)
    methods = set(summary.method)
    expected_years = set(manifest["config"]["test_years"])
    for seed in seeds:
        if set(summary.loc[summary.seed == seed, "method"]) != methods:
            raise ValueError("Unequal candidate sets across seeds")
        for method, group in yearly.loc[yearly.seed == seed].groupby("method"):
            if set(group.test_feature_year) != expected_years:
                raise ValueError(f"Missing years: {seed}/{method}")
    if summary.n_predictions.nunique() != 1:
        raise ValueError("Unequal test observation counts")
    # Independently reconcile each stored annual mean with yearly source rows.
    for row in summary.itertuples(index=False):
        data = yearly.loc[(yearly.seed == row.seed) & (yearly.method == row.method)]
        for metric in METRICS:
            if not np.isclose(data[metric].mean(), getattr(row, "annual_mean_" + metric), atol=1e-12):
                raise ValueError(f"Annual mean mismatch: {row.method}/{metric}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = run / "separated_tables" / timestamp
    output.mkdir(parents=True, exist_ok=False)
    groups = []
    for classifier in manifest["config"]["classifiers"]:
        groups.append(("01_single_models", classifier, CLASSIFIERS[classifier],
                       {m for m in methods if m.startswith(f"Newest_{classifier}_")}))
    for family, title in FAMILIES.items():
        selected = set(yearly.loc[(yearly.stage == "ensemble") & (yearly.family == family), "method"])
        if selected:
            groups.append(("02_ensembles", family, title, selected))
    groups.append(("02_ensembles", "validation_selected", "Validation 選擇集成策略", {"ValidationSelectedEnsemble"}))
    groups.append(("03_feature_selection", "fs_comparison", "選定集成的 FS 對照（含 no-FS）",
                   {m for m in methods if m.startswith("SelectedEnsemble_")}))
    groups.append(("03_feature_selection", "validation_selected", "Validation 再選擇 FS／no-FS 策略", {"ValidationSelectedFS"}))
    assigned = [method for _, _, _, members in groups for method in members]
    if len(assigned) != len(set(assigned)) or set(assigned) != methods:
        raise ValueError("Groups must partition all methods exactly once")
    phase_titles = {"01_single_models": "單模型與採樣", "02_ensembles": "集成", "03_feature_selection": "特徵選擇"}
    context = (f"本快照納入完整完成的 {len(seeds)}/{len(manifest['config']['seeds'])} 個 seeds："
               f"{', '.join(map(str, seeds))}。Test 特徵年 {min(expected_years)}–{max(expected_years)}；"
               f"每個方法、每個 seed 共 {int(summary.n_predictions.iloc[0]):,} 筆 Test 預測。\n\n"
               "所有表皆為已完成 seeds 的平均。Pooled 先於每個 seed 合併年度預測計算指標，再取 seed 平均；"
               "年度平均則先對每個 seed 的年度指標等權平均，再取 seed 平均。"
               "二者分表，不將重複 seed 視為新的公司樣本。CSV 另保留 seed 標準差與各 seed 數值；"
               "標準差不是信賴區間。本次為執行中快照，不能當作十種子最終結論。\n\n"
               "配置依名稱排列，不以 Test 排名重新選模型。FS 的基礎集成可能隨 seed／年度不同，"
               "FS 表比較的是同一年度已選配置上的 FS 規則。\n\n")
    phase_docs = {phase: f"# {title}\n\n{context}" for phase, title in phase_titles.items()}
    thesis_parts = []
    counts = {}
    for phase, key, title, members in groups:
        folder = output / phase / key
        folder.mkdir(parents=True)
        selected = summary.loc[summary.method.isin(members)].sort_values(["method", "seed"])
        counts[f"{phase}/{key}"] = len(members)
        means = selected.groupby("method", sort=True)[METRICS].mean().reset_index()
        annual_cols = ["annual_mean_" + metric for metric in METRICS]
        annual_selected = selected[["method", "seed"] + annual_cols].rename(columns=dict(zip(annual_cols, METRICS, strict=True)))
        annual_means = annual_selected.groupby("method", sort=True)[METRICS].mean().reset_index()
        for filename, data in (("pooled_mean", means), ("annual_mean", annual_means)):
            source = selected if filename == "pooled_mean" else annual_selected
            sd = source.groupby("method")[METRICS].std().add_suffix("_seed_sd")
            export_frame = data.merge(sd, left_on="method", right_index=True)
            export_frame["n_seeds"] = len(seeds)
            export_frame.to_csv(folder / f"{filename}.csv", index=False, encoding="utf-8-sig")
        selected.drop(columns=[c for c in selected if c.startswith("annual_mean_")]).to_csv(
            folder / "pooled_by_seed.csv", index=False, encoding="utf-8-sig")
        annual_selected.to_csv(folder / "annual_mean_by_seed.csv", index=False, encoding="utf-8-sig")
        yearly.loc[yearly.method.isin(members)].sort_values(["method", "seed", "test_feature_year"]).to_csv(
            folder / "by_year.csv", index=False, encoding="utf-8-sig")
        phase_docs[phase] += (f"## {title}\n\n### Pooled\n\n{markdown_table(means)}\n\n"
                              f"### 年度平均\n\n{markdown_table(annual_means)}\n\n"
                              f"[Pooled CSV]({key}/pooled_mean.csv) · [年度平均 CSV]({key}/annual_mean.csv) · "
                              f"[逐年數據]({key}/by_year.csv)\n\n")
        thesis_parts.append((phase, title, markdown_table(means)))
    for phase, text in phase_docs.items():
        (output / phase / "README.md").write_text(text, encoding="utf-8")
    index = "# 滑動視窗分表結果\n\n" + context
    for phase, title in phase_titles.items():
        index += f"- [{title}]({phase}/README.md)\n"
    index += "\n每類資料夾再依分類器或集成機制拆開。原始逐筆預測仍保留在 run 的 seed 資料夾。\n"
    (output / "README.md").write_text(index, encoding="utf-8")
    snapshot = {"created_at": datetime.now(timezone.utc).isoformat(), "run": str(run.resolve()),
                "completed_seeds": seeds, "planned_seeds": manifest["config"]["seeds"],
                "source_sha256": sources, "method_counts": counts, "methods_total": len(methods),
                "checks": {"unique_partition": True, "equal_method_coverage": True,
                           "equal_year_coverage": True, "annual_means_reconciled": True},
                "status": "interim" if len(seeds) < len(manifest["config"]["seeds"]) else "complete"}
    (output / "snapshot_manifest.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    block = "### 4.5.8 多分類器、採樣、集成與 FS 的分階段結果（執行中快照）\n\n" + context
    relative = "../" + output.relative_to(Path(__file__).resolve().parents[2]).as_posix()
    block += f"資料來源：[分表索引]({relative}/README.md)、[快照來源與查核]({relative}/snapshot_manifest.json)。\n\n"
    block += "以下主文僅列 pooled 表；年度平均另於各類分表文件呈現，以免兩種摘要混用。\n\n"
    last_phase = None
    for phase, title, table in thesis_parts:
        if phase != last_phase:
            block += f"#### 4.5.8.{list(phase_titles).index(phase) + 1} {phase_titles[phase]}\n\n"
            last_phase = phase
        block += f"**{title}（Pooled）**\n\n{table}\n\n"
    block += ("上述比較保留全部預定義配置，沒有將最高 Test 分數當作可部署的選擇規則。"
              "只有 ValidationSelectedEnsemble 與 ValidationSelectedFS 代表逐年以 Validation AP 選擇的策略；"
              "固定配置的 Test 差異僅作描述。FS 的 no-FS 列重用相同已選集成，"
              "可作該階段的配對對照。尚未完成全部 seeds 與本輪差異的不確定性分析，"
              "不據此更新研究假設的最終判定、摘要或宣稱方法全面優越。\n\n")
    (output / "thesis_section.md").write_text(block, encoding="utf-8")
    print(json.dumps({"output": str(output), "seeds": seeds, "methods": len(methods), "groups": counts}, ensure_ascii=False))
    return output


def sync_thesis(output):
    root = Path(__file__).resolve().parents[2]
    thesis = root / "thesis/THESIS_FULL.md"
    text = thesis.read_text(encoding="utf-8")
    method = """### 3.10.7 多分類器、採樣、集成與 FS 的分階段協定

本擴充沿用三年 fit 窗口、FIFO3 及 2009–2018 年度 Test。分類器為 Logistic Regression、
Random Forest、XGBoost、LightGBM、MLP 與 RBF SVM，分別搭配 none、TomekLinks（under）、
ADASYN（over）及 SMOTEENN（hybrid），共 24 個單模型配置。各預測分類器不另加類別權重，
使用既有 wrapper 的基礎參數；MLP 停用採樣後的隨機 early-stopping holdout。
每個窗口獨立擬合均值補值與標準化，採樣只作用於訓練資料。

集成候選事先固定為跨時間 FIFO、跨採樣、跨分類器，以及其組合，共 46 種等權機率平均配置。
none 保留為獨立對照，不併入 under／over／hybrid 的跨採樣三票。
各年度只用前一年度 Validation AP 選一個集成，同分依配置名稱排序，沒有以跨年度 Test 排名回選。
最新版單模型直接重用當年的最新窗口模型。每個 seed 的十二個不同窗口共訓練 288 個 no-FS 基模型。

FS 階段針對該年度選中的集成配置，比較 no-FS 與 MI、CART、RFE、SHAP，各保留 50%／80%
（18 維對應 9／14 維）。FS 只使用各成員的原始訓練窗口，次序為補值、標準化、FS、採樣及訓練；
每個成員保存自己的特徵欄位。CART／RFE 使用 balanced decision tree；SHAP 以訓練窗口上的
100 棵樹 balanced LightGBM 計算平均絕對 TreeSHAP contribution，排除 bias 欄。
FS 的 no-FS 對照直接重用原集成，另以同年度 Validation AP 選 FS／no-FS 規則。

各年度 Validation 也決定 F1 門檻及 FPR 操作點，Test 不參與任何選擇。
此分階段搜尋重用 Validation，故其分數不作無偏效能；它亦不搜尋完整分類器×集成×FS 全組合。
輸出將單模型、各類集成、FS 分表，pooled 與年度等權平均分開，另保存各 seed 與逐年明細。
預定十個 seeds；未完成時只報所有方法均完成的 seeds，不混入部分年度。
本輪保存凍結模型的未來機率快取以供研究 replay，不等同 §3.10.2 的部署模型 checkpoint。
新流程尚未包含 DES/DCS 或深度表格模型，也未以 Test 最佳表現更改既定候選集合。

完整設定、來源版本與執行限制見[實驗規格](../docs/Study3B_多分類器採樣集成FS流程.md)及 §4.5.8 的快照來源。
"""
    results = (output / "thesis_section.md").read_text(encoding="utf-8")
    for name, block, anchor in (("METHOD", method, "# 第四章 實驗設計與結果"),
                                ("RESULTS", results, "## 4.6 Study 4：成本敏感分析")):
        start, end = f"<!-- STUDY3B_FULL_{name}_START -->", f"<!-- STUDY3B_FULL_{name}_END -->"
        replacement = f"{start}\n{block}\n{end}\n\n"
        if start in text:
            text = re.sub(re.escape(start) + r".*?" + re.escape(end) + r"\n*", lambda _, value=replacement: value,
                          text, flags=re.S)
        else:
            if text.count(anchor) != 1:
                raise ValueError(f"Ambiguous thesis insertion anchor: {anchor}")
            text = text.replace(anchor, replacement + anchor, 1)
    thesis.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--update-thesis", action="store_true")
    args = parser.parse_args()
    output = export(args.run.resolve())
    if args.update_thesis:
        sync_thesis(output)
