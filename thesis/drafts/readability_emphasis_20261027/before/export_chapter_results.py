"""Export descriptive results for chapters 4 / appendix; never select on Test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from export_fixed_statistics import FS_LABELS, METHOD_LABELS, longtable, verified_analysis
from resolve_method_clarifications import BASELINES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis/latex/generated/chapter-results-20261027"
PROOF = ROOT / "thesis/drafts/framework_results_revision_20261027/results-proof.json"
METRICS = ("AUC", "F1", "Recall")
SAMPLING = {"none": "None", "undersampling": "Under", "oversampling": "Over", "hybrid": "Hybrid"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate_baseline(frame):
    """Boundary means, then equal boundary weights; Retrain is not replicated."""
    d = frame[frame.method.isin(["Old", "New", "Retrain"])].copy()
    if d.duplicated(["split", "method", "sampling"]).any():
        raise ValueError("Duplicate Baseline keys")
    expected = {f"split_{i}+{16-i}" for i in range(1, 16)}
    for method in ("Old", "New", "Retrain"):
        g = d[d.method.eq(method)]
        if set(g.sampling) != set(SAMPLING):
            raise ValueError("Incomplete sampling set")
        counts = g.groupby("sampling").size()
        if not counts.eq(1 if method == "Retrain" else 15).all():
            raise ValueError("Incomplete Baseline boundaries")
        if method != "Retrain" and set(g.split) != expected:
            raise ValueError("Unexpected boundary set")
    scores = d[list(METRICS)].to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid Baseline scores")
    overall, by_sampling = [], []
    for method in ("Old", "New", "Retrain"):
        g = d[d.method.eq(method)]
        means = g.groupby("split")[list(METRICS)].mean().mean()
        overall.append(dict(method=method, n_rows=len(g), **means.to_dict()))
        for sampling in SAMPLING:
            part = g[g.sampling.eq(sampling)]
            by_sampling.append(dict(method=method, sampling=sampling, n_boundaries=len(part),
                                    **part[list(METRICS)].mean().to_dict()))
    return overall, by_sampling


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        proof = json.loads(PROOF.read_text(encoding="utf-8"))
        for mapping in (proof["source_sha256"], proof["generated_sha256"]):
            for name, expected in mapping.items():
                assert sha(ROOT / name) == expected, name
        print("PASS: Baseline, full FS scores, yearly results and model-pool summaries")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    PROOF.parent.mkdir(parents=True, exist_ok=True)
    sources = {}

    def remember(p):
        sources[p.relative_to(ROOT).as_posix()] = sha(p)

    def write(name, caption, label, columns, widths, rows, note):
        (OUT / name).write_text(longtable(caption, label, columns, widths, rows, note), encoding="utf-8")

    for p in [Path(__file__).resolve(), ROOT / "scripts/thesis/export_fixed_statistics.py",
              ROOT / "scripts/thesis/resolve_method_clarifications.py"]:
        remember(p)
    analysis, manifest = verified_analysis()
    remember(analysis / "manifest.json")
    remember(analysis / "descriptive.csv")
    baseline, sampling_rows = [], []
    for candidate, relative in BASELINES:
        p = ROOT / relative
        remember(p)
        a, b = aggregate_baseline(pd.read_csv(p))
        baseline.extend(dict(candidate=candidate, source=relative, **r) for r in a)
        sampling_rows.extend(dict(candidate=candidate, source=relative, **r) for r in b)
    pd.DataFrame(baseline).to_csv(OUT / "baseline-summary.csv", index=False)
    pd.DataFrame(sampling_rows).to_csv(OUT / "baseline-sampling.csv", index=False)
    write("baseline-summary.tex", "實驗一 Baseline 各分類器與訓練策略之測試結果", "tab:results-baseline",
          ["分類器／版本", "策略", "AUC", "F1-score", "Recall"], ["0.34", "0.13", "0.15", "0.15", "0.15"],
          [[r['candidate'], r['method']] + [f"{r[m]:.4f}" for m in METRICS] for r in baseline],
          r"Old／New 各先平均同一邊界的四種採樣，再平均 15 組邊界；Retrain 僅平均四種採樣，沒有複製成 15 次。均為 2015--2018 測試指標。不同歷史版本的調參預算、前處理與紀錄完整性未完全一致；本表為描述性彙整，不作演算法公平排名或與固定年份集成的顯著性比較。TabICL 為結果標記，後端限制見附錄。")
    write("baseline-sampling.tex", "實驗一 Baseline 各採樣設定之測試結果", "tab:results-baseline-sampling",
          ["分類器／版本", "策略", "採樣", "AUC", "F1-score", "Recall"], ["0.28", "0.10", "0.13", "0.14", "0.14", "0.14"],
          [[r['candidate'], r['method'], SAMPLING[r['sampling']]] + [f"{r[m]:.4f}" for m in METRICS] for r in sampling_rows],
          r"Old／New 各採樣平均 15 組邊界，Retrain 每種採樣僅一筆；不是獨立重複試驗平均。Under、Over、Hybrid 為低採樣、過採樣與混合採樣。版本差異與解讀範圍同正文 Baseline 表。")
    run = ROOT / manifest['specification']['run']
    all_metrics = []
    for seed in manifest['specification']['seeds']:
        p = run / f"seed_{seed}/no_fs/metrics.csv"
        complete = p.parent / "complete.json"
        remember(p); remember(complete)
        assert sha(p) == json.loads(complete.read_text(encoding='utf-8'))['metrics_sha256']
        d = pd.read_csv(p, dtype={'period': str})
        assert set(d.seed) == {seed}
        all_metrics.append(d)
    metrics = pd.concat(all_metrics, ignore_index=True)
    assert not metrics.duplicated(['seed', 'method', 'period']).any()
    desc = pd.read_csv(analysis / 'descriptive.csv')
    methods = list(METHOD_LABELS)
    yearly = metrics[metrics.method.isin(methods) & metrics.period.ne('pooled')]
    assert len(yearly) == 10 * 8 * 4
    rows, yearly_records = [], []
    for method in methods:
        for year in ('2015', '2016', '2017', '2018'):
            g = yearly[yearly.method.eq(method) & yearly.period.eq(year)]
            assert len(g) == 10 and g.n_test.nunique() == g.n_test_positive.nunique() == 1
            means = g[list(METRICS)].mean()
            rows.append([METHOD_LABELS[method], year] + [f"{means[m]:.4f}" for m in METRICS])
            yearly_records.append(dict(method=method, year=year, n_test=int(g.n_test.iloc[0]),
                                       positive=int(g.n_test_positive.iloc[0]), **means.to_dict()))
    pd.DataFrame(yearly_records).to_csv(OUT / 'ensemble-yearly.csv', index=False)
    write('ensemble-yearly.tex', '實驗一固定年份集成之逐年測試結果', 'tab:results-ensemble-yearly',
          ['方法', '年度', 'AUC', 'F1-score', 'Recall'], ['0.28', '0.12', '0.18', '0.18', '0.18'], rows,
          r"各值為十個種子的年度指標平均；不是 pooled 指標。2015--2018 年正類紀錄數依序為 111、83、57、36，所有年度沿用 Calibration 固定的門檻。")
    pool = metrics[metrics.period.eq('pooled')]
    rows = []
    for method in ('Old_under', 'Old_over', 'Old_hybrid', 'New_under', 'New_over', 'New_hybrid'):
        g = pool[pool.method.eq(method)]
        assert len(g) == 10
        rows.append([method.replace('_', r'\_')] + [f"{g[m].mean():.4f}" for m in METRICS])
    write('pool-models.tex', '實驗一六個基礎模型之測試結果', 'tab:results-pool-models',
          ['基礎模型', 'AUC', 'F1-score', 'Recall'], ['0.30', '0.21', '0.21', '0.21'], rows,
          r"完整 18 特徵、固定年份與相同十個種子；under、over、hybrid 分別代表 TomekLinks、ADASYN、SMOTEENN。各模型門檻由 Calibration 決定。這些模型屬固定年份協定，不代替歷史 Baseline 的同名訓練策略。")
    static = pool[pool.method.str.startswith('Static_')].copy()
    static['size'] = static.method.str.count(r'\+') + 1
    sizes = {2: 9, 3: 18, 4: 15, 5: 6, 6: 1}
    rows = []
    for size, count in sizes.items():
        g = static[static['size'].eq(size)]
        assert g.groupby('seed').size().eq(count).all() and len(g) == count * 10
        seed_means = g.groupby('seed')[list(METRICS)].mean()
        rows.append([str(size), str(count)] + [f"{seed_means[m].mean():.4f}" for m in METRICS])
    write('static-size.tex', '實驗一不同靜態成員數之候選平均結果', 'tab:results-static-size',
          ['模型數', '候選數', 'AUC', 'F1-score', 'Recall'], ['0.15', '0.15', '0.21', '0.21', '0.21'], rows,
          r"每個種子先等權平均相同成員數的全部候選之 Test 指標，再跨十個種子平均；未按 Test 挑選候選，也未將候選再次集成。這是模型組成的描述性比較，不是成員數效果的因果或顯著性檢定。")
    fs_methods = methods[2:]
    for metric, metric_label in [('AUC', 'AUC'), ('F1', 'F1-score'), ('Recall', 'Recall')]:
        rows = []
        for fs, label in [('no_fs', 'no-FS（18）')] + list(FS_LABELS.items()):
            row = [label]
            for method in fs_methods:
                g = desc[desc.fs.eq(fs) & desc.method.eq(method) & desc.metric.eq(metric)]
                assert len(g) == 1 and g.n_seeds.iloc[0] == 10
                r = g.iloc[0]
                row.append(r"\shortstack[l]{" + f"{r['mean']:.4f}" + r"\\{\scriptsize(\(\pm\)" + f"{r.seed_sd:.4f}" + r")}}")
            rows.append(row)
        write(f'fs-{metric.lower()}-absolute.tex', f'實驗二各特徵條件之 pooled {metric_label}', f'tab:results-fs-{metric.lower()}',
              ['FS'] + [METHOD_LABELS[m].replace('DCS-', '') for m in fs_methods], ['0.16'] + ['0.123'] * 6, rows,
              r"各格上列為十個種子的平均值，下列括號為種子標準差；50\%、80\% 分別保留 9、14 個特徵。no-FS 使用 18 個特徵，作為同方法對照。V-static 可依各條件的 Calibration 重新選成員。表內不依測試分數標記最佳條件；顯著性另依配對檢定。")
    proof = dict(status='passed', source_sha256=sources,
                 generated_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in OUT.iterdir() if p.is_file()},
                 baseline_sources=len(BASELINES), baseline_summary_rows=len(baseline), baseline_sampling_rows=len(sampling_rows),
                 yearly_rows=len(yearly_records), fs_settings=9, fs_methods=6,
                 scope='descriptive Test reporting only; no model selection or additional training')
    PROOF.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in proof.items() if not k.endswith('sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
