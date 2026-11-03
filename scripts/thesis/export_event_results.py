"""Export event-target chapter tables and figures; descriptive only, never select on Test.

Bankruptcy main results use the reconstructed next-year event target (same rows,
years, pool and Calibration rules as the final-status run). Paired company
bootstrap tests for the event target are not exported here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd

from export_fixed_statistics import FS_LABELS, METHOD_LABELS, longtable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis/latex/generated/event-results-20261103"
FIGURES = ROOT / "thesis/latex/figures"
PROOF = ROOT / "thesis/drafts/event_label_revision_20261103/results-proof.json"
EVENT_RUN = "results/phase2_ensemble/standard_dynamic_event/runs/20261031T114653242210Z"
FINAL_RUN = "results/phase2_ensemble/standard_dynamic/runs/20261004T202331745995Z"
BASELINE_RUN = "results/phase1_baseline/fixed_event/runs/20261031T175308731604Z"
PROTOCOLS = {EVENT_RUN: "fixed_bankruptcy_standard_deslib_fit_only_fs_event_target_v1",
             FINAL_RUN: "fixed_bankruptcy_standard_deslib_fit_only_fs_v1"}
SEEDS = [42, 123, 456, 789, 2024, 7, 21, 84, 168, 999]
METRICS = ("AUC", "F1", "Recall")
METHODS = list(METHOD_LABELS)
POOL = ("Old_under", "Old_over", "Old_hybrid", "New_under", "New_over", "New_hybrid")
CLASSIFIERS = {"xgb": "XGBoost", "lgb": "LightGBM", "rf": "Random Forest", "lr": "Logistic Regression"}
PERIODS = ("Old", "New", "Retrain")
SAMPLING = {"none": "None", "under": "Under", "over": "Over", "hybrid": "Hybrid"}
YEARS = ("2015", "2016", "2017", "2018")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emphasized(value, maximum):
    """Highlight all ties at the displayed precision, without selecting a model."""
    shown = f"{value:.4f}"
    return r"\textbf{" + shown + "}" if shown == f"{maximum:.4f}" else shown


def with_sd(mean, sd, maximum=None):
    shown = emphasized(mean, maximum) if maximum is not None else f"{mean:.4f}"
    return r"\shortstack[l]{" + shown + r"\\{\scriptsize(\(\pm\)" + f"{sd:.4f}" + r")}}"


def load_run(relative, remember):
    """Seed-level metrics verified against each completion record."""
    run = ROOT / relative
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and manifest["protocol"] == PROTOCOLS[relative]
    assert manifest["config"]["seeds"] == SEEDS
    remember(run / "manifest.json")
    frames = []
    for seed in SEEDS:
        for fs in sorted(p.name for p in (run / f"seed_{seed}").iterdir() if p.is_dir()):
            p = run / f"seed_{seed}/{fs}/metrics.csv"
            complete = p.parent / "complete.json"
            remember(p)
            remember(complete)
            assert sha(p) == json.loads(complete.read_text(encoding="utf-8"))["metrics_sha256"]
            d = pd.read_csv(p, dtype={"period": str})
            assert set(d.seed) == {seed} and set(d.fs) == {fs}
            frames.append(d)
    metrics = pd.concat(frames, ignore_index=True)
    assert not metrics.duplicated(["seed", "fs", "method", "period"]).any()
    partitions = pd.read_csv(run / "partition_summary.csv")
    remember(run / "partition_summary.csv")
    return metrics, partitions


def seed_summary(pooled, fs, methods, metrics=METRICS):
    rows = {}
    for method in methods:
        g = pooled[pooled.fs.eq(fs) & pooled.method.eq(method)]
        assert len(g) == len(SEEDS), (fs, method)
        rows[method] = {m: (g[m].mean(), g[m].std(ddof=1)) for m in metrics}
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        proof = json.loads(PROOF.read_text(encoding="utf-8"))
        for mapping in (proof["source_sha256"], proof["generated_sha256"]):
            for name, expected in mapping.items():
                assert sha(ROOT / name) == expected, name
        print("PASS: event-target Baseline, ensemble, FS, label comparison and figures")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    PROOF.parent.mkdir(parents=True, exist_ok=True)
    sources, generated = {}, {}

    def remember(p):
        sources[Path(p).relative_to(ROOT).as_posix()] = sha(p)

    def write(name, caption, label, columns, widths, rows, note, merge_columns=()):
        (OUT / name).write_text(longtable(caption, label, columns, widths, rows, note, merge_columns), encoding="utf-8")

    remember(Path(__file__).resolve())
    remember(ROOT / "scripts/thesis/export_fixed_statistics.py")
    event, event_parts = load_run(EVENT_RUN, remember)
    final, final_parts = load_run(FINAL_RUN, remember)
    pooled = event[event.period.eq("pooled")]
    final_pooled = final[final.period.eq("pooled")]
    assert pooled.n_test.eq(12282).all() and pooled.n_test_positive.eq(119).all()

    # Partitions under both labels: same rows, different positives.
    order = [("Old_fit", "Old-fit", "Old 模型訓練；前處理與特徵選取擬合"), ("New_fit", "New-fit", "New 模型訓練"),
             ("DSEL", "DSEL", "動態選擇的局部能力估計"), ("Calibration", "Calibration", "靜態候選選擇與分類門檻"),
             ("Test", "Test", "設定固定後的最終評估")]
    ep, fp = event_parts.set_index("partition"), final_parts.set_index("partition")
    rows, records = [], []
    for key, name, use in order:
        e, f = ep.loc[key], fp.loc[key]
        assert e.n_rows == f.n_rows and e.first_year == f.first_year and e.last_year == f.last_year
        years = f"{e.first_year}" if e.first_year == e.last_year else f"{e.first_year}–{e.last_year}"
        rows.append([name, years, f"{e.n_rows:,}", f"{e.n_positive:,}（{100*e.n_positive/e.n_rows:.2f}\\%）",
                     f"{f.n_positive:,}", use])
        records.append(dict(partition=key, years=years, n_rows=int(e.n_rows), event_positive=int(e.n_positive),
                            final_status_positive=int(f.n_positive), label_available_year=int(e.last_label_available)))
    pd.DataFrame(records).to_csv(OUT / "event-partitions.csv", index=False)
    write("event-partitions.tex", "固定年份資料分區與兩種標籤之正類紀錄數", "tab:method-partitions",
          ["分區", "年份", "紀錄數", "事件標籤正類", "最終狀態正類", "主要用途"],
          ["0.12", "0.11", "0.10", "0.17", "0.12", "0.29"], rows,
          r"資料來源：本研究實驗資料分區彙整。數值為公司年度紀錄數，不是獨立公司數。事件標籤只把破產公司最後一個觀測年度記為正類；最終狀態標籤把破產公司所有年度記為正類。兩種標籤使用相同資料列與分區。")

    # Experiment 1 Baseline: same rows, event target, Old/New/Retrain, 2014 validation.
    base = ROOT / BASELINE_RUN
    bman = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    assert bman["status"] == "completed" and bman["protocol"] == "fixed_bankruptcy_event_target_baselines_v1"
    assert bman["config"]["seeds"] == SEEDS and bman["config"]["classifiers"] == list(CLASSIFIERS)
    for name in ("manifest.json", "metrics.csv", "tuning.csv", "classifier_ranking_pool_slots.csv", "partition_summary.csv"):
        remember(base / name)
    bm = pd.read_csv(base / "metrics.csv", dtype={"test_period": str})
    bp = bm[bm.test_period.eq("pooled")]
    assert len(bp) == len(CLASSIFIERS) * 3 * 4 * len(SEEDS) and bp.n_test_positive.eq(119).all()
    assert not bp.duplicated(["classifier", "period", "sampling", "seed"]).any()
    cols = ["validation_AUC"] + list(METRICS)
    summary, sampling_rows = [], []
    for c, label in CLASSIFIERS.items():
        for period in PERIODS:
            g = bp[bp.classifier.eq(c) & bp.period.eq(period)]
            assert set(g.sampling) == set(SAMPLING) and len(g) == 4 * len(SEEDS)
            summary.append(dict(classifier=label, method=period, **g[cols].mean().to_dict()))
            for s in SAMPLING:
                h = g[g.sampling.eq(s)]
                sampling_rows.append(dict(classifier=label, method=period, sampling=s,
                                          chosen_candidate=int(h.chosen_candidate.iloc[0]),
                                          **{k: h[k].mean() for k in cols},
                                          **{k + "_sd": h[k].std(ddof=1) for k in METRICS}))
    pd.DataFrame(summary).to_csv(OUT / "event-baseline-summary.csv", index=False)
    pd.DataFrame(sampling_rows).to_csv(OUT / "event-baseline-sampling.csv", index=False)
    best = {c: {k: max(r[k] for r in summary if r["classifier"] == c) for k in cols} for c in CLASSIFIERS.values()}
    write("event-baseline-summary.tex", "事件標籤下各分類器三種訓練策略之結果", "tab:results-baseline",
          ["分類器", "策略", "驗證 AUC", "AUC", "F1-score", "Recall"], ["0.24", "0.11", "0.14", "0.14", "0.14", "0.14"],
          [[r["classifier"], r["method"]] + [emphasized(r[k], best[r["classifier"]][k]) for k in cols] for r in summary],
          r"資料來源：本研究事件標籤 Baseline 實驗結果。驗證 AUC 為 2014 年驗證資料，其餘為 2015--2018 年 Test 之合併四年結果；各值為四種採樣與十個種子的平均。粗體表示同一分類器三種策略中該欄最高值，不代表統計顯著。",
          merge_columns=(0,))
    sbest = {}
    for r in sampling_rows:
        key = (r["classifier"], r["method"])
        sbest.setdefault(key, {k: -1.0 for k in cols})
        for k in cols:
            sbest[key][k] = max(sbest[key][k], r[k])
    write("event-baseline-sampling.tex", "事件標籤 Baseline 各採樣設定之結果", "tab:results-baseline-sampling",
          ["分類器", "策略", "採樣", "驗證 AUC", "AUC", "F1-score", "Recall"],
          ["0.20", "0.10", "0.10", "0.125", "0.125", "0.125", "0.125"],
          [[r["classifier"], r["method"], SAMPLING[r["sampling"]]] +
           [emphasized(r[k], sbest[(r["classifier"], r["method"])][k]) for k in cols] for r in sampling_rows],
          r"資料來源：本研究事件標籤 Baseline 實驗結果。各值為十個種子平均；超參數依 2014 年驗證 AUC 由 24 組候選選出，候選 0 為集成池使用的預設參數。粗體比較同一分類器與策略下的四種採樣。Under、Over、Hybrid 分別為 TomekLinks、ADASYN、SMOTEENN。",
          merge_columns=(0, 1))
    ranking = pd.read_csv(base / "classifier_ranking_pool_slots.csv")
    assert list(ranking.validation_rank) == [1, 2, 3, 4] and set(ranking.classifier) == set(CLASSIFIERS)
    write("event-classifier-validation.tex", "六個模型池位置之分類器驗證結果", "tab:results-classifier-validation",
          ["驗證排名", "分類器", "驗證 AUC", "驗證 PR-AUC"], ["0.16", "0.36", "0.20", "0.20"],
          [[str(r.validation_rank), CLASSIFIERS[r.classifier], f"{r.validation_AUC:.4f}", f"{r.validation_PR_AUC:.4f}"]
           for r in ranking.itertuples()],
          r"資料來源：本研究事件標籤 Baseline 實驗結果。各值為 Old／New 與 Under、Over、Hybrid 共六個模型池位置、十個種子的 2014 年驗證平均，依驗證 AUC 排序；本表不使用 Test 分數。2014 年驗證資料只有 33 筆正類，排名差距須審慎解讀。")

    # Fairness check: the ensemble pool uses default XGBoost (candidate 0).
    rows, gap = [], []
    for slot in POOL:
        period, sampling = slot.split("_")
        d = pooled[pooled.fs.eq("no_fs") & pooled.method.eq(slot)]
        t = bp[bp.classifier.eq("xgb") & bp.period.eq(period) & bp.sampling.eq(sampling)]
        assert len(d) == len(t) == len(SEEDS)
        cand = int(t.chosen_candidate.iloc[0])
        assert t.chosen_candidate.nunique() == 1
        gap.append(dict(slot=slot, tuned_candidate=cand, **{f"default_{m}": d[m].mean() for m in METRICS},
                        **{f"tuned_{m}": t[m].mean() for m in METRICS}))
        rows.append([slot.replace("_", r"\_"), "預設（集成池）"] + [f"{d[m].mean():.4f}" for m in METRICS])
        rows.append(["", f"調參（候選 {cand}）"] + [f"{t[m].mean():.4f}" for m in METRICS])
    pd.DataFrame(gap).to_csv(OUT / "event-tuning-gap.csv", index=False)
    write("event-tuning-gap.tex", "集成池預設 XGBoost 與 Baseline 調參 XGBoost 之對照", "tab:results-tuning-gap",
          ["模型池位置", "XGBoost 參數", "AUC", "F1-score", "Recall"], ["0.20", "0.26", "0.16", "0.16", "0.16"], rows,
          r"資料來源：本研究事件標籤集成與 Baseline 實驗結果。兩者使用相同資料列、標籤、擬合期間與採樣；預設列的門檻由 Calibration（2014 年半數）決定，調參列的參數與門檻由完整 2014 年驗證資料決定。各值為十個種子的合併四年 Test 平均，僅供說明參數條件的差異。")

    # Experiment 1 ensemble.
    rows = []
    pool_rows = seed_summary(pooled, "no_fs", POOL)
    maxima = {m: max(v[m][0] for v in pool_rows.values()) for m in METRICS}
    for slot in POOL:
        rows.append([slot.replace("_", r"\_")] + [emphasized(pool_rows[slot][m][0], maxima[m]) for m in METRICS])
    write("event-pool-models.tex", "事件標籤下六個基礎模型之測試結果", "tab:results-pool-models",
          ["基礎模型", "AUC", "F1-score", "Recall"], ["0.30", "0.21", "0.21", "0.21"], rows,
          r"資料來源：本研究事件標籤集成實驗結果。完整 18 特徵、十個種子平均；under、over、hybrid 分別為 TomekLinks、ADASYN、SMOTEENN。粗體為六模型中該指標最高值，不表示差異達統計顯著；門檻由 Calibration 決定。")
    static = pooled[pooled.fs.eq("no_fs") & pooled.method.str.startswith("Static_")].copy()
    static["size"] = static.method.str.count(r"\+") + 1
    rows = []
    for size, count in {2: 9, 3: 18, 4: 15, 5: 6, 6: 1}.items():
        g = static[static["size"].eq(size)]
        assert g.groupby("seed").size().eq(count).all() and len(g) == count * len(SEEDS)
        seed_means = g.groupby("seed")[list(METRICS)].mean()
        rows.append([str(size), str(count)] + [seed_means[m].mean() for m in METRICS])
    maxima = [max(r[j] for r in rows) for j in (2, 3, 4)]
    write("event-static-size.tex", "事件標籤下不同靜態成員數之候選平均結果", "tab:results-static-size",
          ["模型數", "候選數", "AUC", "F1-score", "Recall"], ["0.15", "0.15", "0.21", "0.21", "0.21"],
          [r[:2] + [emphasized(r[j], maxima[j - 2]) for j in (2, 3, 4)] for r in rows],
          r"資料來源：本研究事件標籤集成實驗結果。每個種子先平均同一成員數全部候選的 Test 分數，再取十個種子平均。粗體僅供比較，不依此選成員數，也不表示顯著差異。")
    ens_metrics = ("AUC", "PR_AUC", "F1", "Recall")
    ens = seed_summary(pooled, "no_fs", METHODS, ens_metrics)
    maxima = {m: max(v[m][0] for v in ens.values()) for m in ens_metrics}
    pd.DataFrame([dict(method=k, **{f"{m}_mean": v[m][0] for m in ens_metrics}, **{f"{m}_sd": v[m][1] for m in ens_metrics})
                  for k, v in ens.items()]).to_csv(OUT / "event-ensemble.csv", index=False)
    write("event-ensemble.tex", "事件標籤下未進行 FS 之集成方法結果", "tab:results-ensemble",
          ["方法", "AUC", "PR-AUC", "F1-score", "Recall"], ["0.16", "0.18", "0.18", "0.18", "0.18"],
          [[METHOD_LABELS[k]] + [with_sd(ens[k][m][0], ens[k][m][1], maxima[m]) for m in ens_metrics] for k in METHODS],
          r"資料來源：本研究事件標籤集成實驗結果。各格上列為十個種子之合併四年 Test 平均，下列為種子標準差；Test 共 12,282 筆紀錄、119 筆事件。粗體為八種方法中該指標最高值，不表示差異達統計顯著。")
    selected = pooled[pooled.method.eq("ValidationSelectedStatic")].groupby("fs").selected_static.agg(
        lambda s: json.dumps(s.value_counts().to_dict(), ensure_ascii=False))
    selected.to_csv(OUT / "event-vstatic-selection.csv")
    yearly = event[event.fs.eq("no_fs") & event.method.isin(METHODS) & event.period.isin(YEARS)]
    assert len(yearly) == len(SEEDS) * len(METHODS) * len(YEARS)
    yr = []
    for method in METHODS:
        for year in YEARS:
            g = yearly[yearly.method.eq(method) & yearly.period.eq(year)]
            assert len(g) == len(SEEDS) and g.n_test.nunique() == g.n_test_positive.nunique() == 1
            yr.append(dict(method=method, year=year, n_test=int(g.n_test.iloc[0]),
                           positive=int(g.n_test_positive.iloc[0]), **g[list(METRICS)].mean().to_dict()))
    pd.DataFrame(yr).to_csv(OUT / "event-ensemble-yearly.csv", index=False)
    positives = {r["year"]: r["positive"] for r in yr}
    ybest = {y: {m: max(r[m] for r in yr if r["year"] == y) for m in METRICS} for y in YEARS}
    write("event-ensemble-yearly.tex", "事件標籤下固定年份集成之逐年測試結果", "tab:results-ensemble-yearly",
          ["方法", "年度", "AUC", "F1-score", "Recall"], ["0.28", "0.12", "0.18", "0.18", "0.18"],
          [[METHOD_LABELS[r["method"]], r["year"]] + [emphasized(r[m], ybest[r["year"]][m]) for m in METRICS] for r in yr],
          r"資料來源：本研究事件標籤集成實驗結果。各值為十個種子的年度指標平均；粗體表示同一年度八種方法中該指標最高值。2015--2018 年事件數依序為 " +
          "、".join(str(positives[y]) for y in YEARS) + r"；各年沿用 Calibration 固定的門檻。",
          merge_columns=(0,))

    # Experiment 2 FS.
    fs_methods = METHODS[2:]
    fs_order = ["no_fs"] + list(FS_LABELS)
    fs_desc = {fs: seed_summary(pooled, fs, fs_methods) for fs in fs_order}
    pd.DataFrame([dict(fs=fs, method=m, metric=k, mean=v[k][0], seed_sd=v[k][1])
                  for fs, d in fs_desc.items() for m, v in d.items() for k in METRICS]).to_csv(OUT / "event-fs.csv", index=False)
    for metric, metric_label in [("AUC", "AUC"), ("F1", "F1-score"), ("Recall", "Recall")]:
        maxima = {m: max(fs_desc[fs][m][metric][0] for fs in fs_order) for m in fs_methods}
        rows = [[("no-FS（18）" if fs == "no_fs" else FS_LABELS[fs])] +
                [with_sd(*fs_desc[fs][m][metric], maxima[m]) for m in fs_methods] for fs in fs_order]
        write(f"event-fs-{metric.lower()}.tex", f"事件標籤下實驗二各特徵條件之 {metric_label}", f"tab:results-fs-{metric.lower()}",
              ["FS"] + [METHOD_LABELS[m].replace("DCS-", "") for m in fs_methods], ["0.16"] + ["0.123"] * 6, rows,
              r"資料來源：本研究事件標籤 FS 集成實驗結果。各格上列為十個種子的合併四年 Test 平均，下列為種子標準差。粗體表示同一方法（同一欄）九種特徵條件中最高的平均分數，不代表已選定最佳設定或差異達統計顯著。50\%、80\% 分別保留 9、14 個特徵；V-static 依各條件的 Calibration 重新選成員。")

    # Feature selection frequency across seeds, from each event-target audit record.
    features = [f"X{i}" for i in range(1, 19)]
    counts = {fs: dict.fromkeys(features, 0) for fs in FS_LABELS}
    for seed in SEEDS:
        for fs in FS_LABELS:
            p = ROOT / EVENT_RUN / f"seed_{seed}/{fs}/audit.json"
            remember(p)
            audit = json.loads(p.read_text(encoding="utf-8"))
            chosen = audit["selected_features"]
            assert audit["fs"] == fs and len(chosen) == audit["n_features"] == (9 if fs.endswith("50") else 14)
            for name in chosen:
                counts[fs][name] += 1
    pd.DataFrame(counts).T.to_csv(OUT / "event-fs-frequency.csv")
    for part, names in ((1, features[:9]), (2, features[9:])):
        write(f"event-fs-frequency-{part}.tex", f"實驗二特徵入選種子數（{names[0]}--{names[-1]}）",
              f"tab:method-fs-frequency-{part}", ["FS 設定"] + names, ["0.20"] + ["0.075"] * 9,
              [[FS_LABELS[fs]] + [str(counts[fs][n]) for n in names] for fs in FS_LABELS],
              r"資料來源：本研究事件標籤各種子特徵選取結果。數值為十個種子中入選的次數；此表僅描述已選特徵，不據此重新挑選 Test 最佳子集。")

    # Label comparison: same rows, pool and rules; only the target differs.
    fin = seed_summary(final_pooled, "no_fs", METHODS)
    rows, recs = [], []
    for k in METHODS:
        for name, d in (("最終狀態", fin), ("事件", ens)):
            rows.append([METHOD_LABELS[k] if name == "最終狀態" else "", name] + [f"{d[k][m][0]:.4f}" for m in METRICS])
            recs.append(dict(method=k, label=name, **{m: d[k][m][0] for m in METRICS}))
    pd.DataFrame(recs).to_csv(OUT / "label-comparison.csv", index=False)
    write("label-comparison.tex", "兩種標籤定義下未進行 FS 之集成結果", "tab:results-label-comparison",
          ["方法", "標籤", "AUC", "F1-score", "Recall"], ["0.22", "0.18", "0.18", "0.18", "0.18"], rows,
          r"資料來源：本研究固定年份集成實驗結果。兩種標籤使用相同資料列、分區、模型池、十個種子與選擇規則；Test 正類分別為 287 筆（最終狀態）與 119 筆（事件）。正類比例不同，原始分數只比較方向，不比較高低。")

    # Figures.
    fonts = ROOT / "thesis/latex/fonts"
    fontManager.addfont(str(fonts / "kaiu.ttf"))
    cjk = FontProperties(fname=str(fonts / "kaiu.ttf"))
    plt.rcParams.update({"font.family": cjk.get_name(), "font.size": 12, "axes.unicode_minus": False,
                         "axes.linewidth": .8, "svg.fonttype": "path", "savefig.facecolor": "white"})

    def save(fig, stem):
        for extension in ("png", "svg"):
            path = FIGURES / f"{stem}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.12,
                        metadata={"Creator": "Continual-Imbalance-Ensemble"})
            generated[path.relative_to(ROOT).as_posix()] = sha(path)
        plt.close(fig)

    labels = list(CLASSIFIERS.values())
    frame = pd.DataFrame(summary)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.2), sharey=True)
    for ax, metric in zip(axes, METRICS):
        for method, color, marker, offset in [("Old", "#9B6500", "o", -.19), ("New", "#235A88", "s", 0),
                                              ("Retrain", "#555555", "^", .19)]:
            r = frame[frame.method.eq(method)].set_index("classifier").loc[labels]
            ax.scatter(r[metric], np.arange(len(labels)) + offset, color=color, marker=marker, s=39,
                       label=method, edgecolors="white", linewidths=.4, zorder=3)
        ax.set_title("F1-score" if metric == "F1" else metric, fontsize=14)
        ax.set_xlim((.7, .9) if metric == "AUC" else (0, .25) if metric == "F1" else (0, .6))
        ax.grid(axis="x", color=".88", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[1].set_xlabel("測試分數（各指標使用各自刻度）", labelpad=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(.6, 1.02), frameon=False)
    fig.subplots_adjust(left=.255, right=.99, wspace=.24, top=.84, bottom=.18)
    save(fig, "event-baseline-strategies")

    fig, axes = plt.subplots(1, 3, figsize=(10.3, 5.6), sharey=True)
    for ax, metric in zip(axes, METRICS):
        for y, k in enumerate(METHODS):
            mean, sd = ens[k][metric]
            ax.errorbar(mean, y, xerr=sd, fmt="o", markersize=5, color="#235A88", capsize=3, linewidth=1.2, zorder=3)
        ax.set_title("F1-score" if metric == "F1" else metric, fontsize=14)
        ax.set_xlim((.68, .88) if metric == "AUC" else (0, .22) if metric == "F1" else (0, .75))
        ax.xaxis.set_major_locator(MultipleLocator(.05 if metric != "Recall" else .15))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="x", color=".88", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(METHODS)), [METHOD_LABELS[k] for k in METHODS])
    axes[0].invert_yaxis()
    axes[1].set_xlabel("十個種子的平均值 ± 標準差", labelpad=12)
    fig.subplots_adjust(left=.155, right=.99, wspace=.19, top=.89, bottom=.17)
    save(fig, "event-ensemble-comparison")

    matrix = np.array([[fs_desc[fs][m]["AUC"][0] - fs_desc["no_fs"][m]["AUC"][0] for m in fs_methods]
                       for fs in FS_LABELS])
    limit = max(.05, float(np.ceil(np.abs(matrix).max() * 100) / 100))
    fig, ax = plt.subplots(figsize=(10.2, 6.3))
    im = ax.imshow(matrix, cmap="PuOr", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(np.arange(6), [METHOD_LABELS[m] for m in fs_methods])
    ax.set_yticks(np.arange(8), [v.replace("\\%", "%") for v in FS_LABELS.values()])
    ax.tick_params(length=0, pad=8)
    for i in range(8):
        for j in range(6):
            ax.text(j, i, f"{matrix[i, j]:+.4f}", ha="center", va="center", fontsize=12,
                    color="white" if abs(matrix[i, j]) > .65 * limit else "#171717")
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("相同集成方法下的 FS - no-FS（十種子平均）", labelpad=15)
    fig.colorbar(im, ax=ax, fraction=.04, pad=.035).set_label("AUC 差值", labelpad=10)
    fig.subplots_adjust(left=.15, right=.94, top=.97, bottom=.17)
    save(fig, "event-fs-auc-differences")

    for p in OUT.iterdir():
        if p.is_file():
            generated[p.relative_to(ROOT).as_posix()] = sha(p)
    proof = dict(status="passed", source_sha256=sources, generated_sha256=generated,
                 runs=dict(event=EVENT_RUN, final_status=FINAL_RUN, baseline=BASELINE_RUN),
                 scope="descriptive Test reporting only; no model selection, no new training, no significance tests")
    PROOF.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in proof.items() if not k.endswith("sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
