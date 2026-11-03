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


def emphasized(value, maximum):
    """Highlight all ties at the displayed precision, without selecting a model."""
    shown = f"{value:.4f}"
    return r"\textbf{" + shown + "}" if shown == f"{maximum:.4f}" else shown


def group_maxima(records, keys):
    groups = {}
    for row in records:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, {m: -float('inf') for m in METRICS})
        for metric in METRICS:
            groups[key][metric] = max(groups[key][metric], row[metric])
    return groups


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

    def write(name, caption, label, columns, widths, rows, note, merge_columns=()):
        (OUT / name).write_text(longtable(caption, label, columns, widths, rows, note, merge_columns), encoding="utf-8")

    for p in [Path(__file__).resolve(), ROOT / "scripts/thesis/export_fixed_statistics.py",
              ROOT / "scripts/thesis/resolve_method_clarifications.py"]:
        remember(p)
    analysis, manifest = verified_analysis()
    remember(analysis / "manifest.json")
    remember(analysis / "descriptive.csv")
    baseline, sampling_rows, selected_rows = [], [], []
    for candidate, relative in BASELINES:
        p = ROOT / relative
        remember(p)
        raw = pd.read_csv(p)
        a, b = aggregate_baseline(raw)
        baseline.extend(dict(candidate=candidate, source=relative, **r) for r in a)
        sampling_rows.extend(dict(candidate=candidate, source=relative, **r) for r in b)
        # A fixed illustrative context, declared independently of Test scores.
        chosen = raw[raw.split.eq('split_13+3') & raw.method.eq('Old') & raw.sampling.eq('hybrid')]
        assert len(chosen) == 1
        if 'tune_best_params' in chosen and pd.notna(chosen.tune_best_params.iloc[0]):
            r = chosen.iloc[0]
            selected_rows.append(dict(candidate=candidate, source=relative, split=r.split,
                                      method=r.method, sampling=r.sampling,
                                      validation_auc=float(r.tune_val_auc),
                                      parameters=json.loads(r.tune_best_params)))
    pd.DataFrame(baseline).to_csv(OUT / "baseline-summary.csv", index=False)
    pd.DataFrame(sampling_rows).to_csv(OUT / "baseline-sampling.csv", index=False)
    pd.DataFrame([{**r, 'parameters': json.dumps(r['parameters'], ensure_ascii=False, sort_keys=True)}
                  for r in selected_rows]).to_csv(OUT / 'validation-selected-parameters.csv', index=False)
    parameter_rows = []
    for r in selected_rows:
        values = []
        for key, value in sorted(r['parameters'].items()):
            shown = str(value).replace('_', r'\_')
            values.append(r'\nolinkurl{' + key + r'} = \textbf{' + shown + '}')
        parameter_rows.append([r['candidate'], '；'.join(values)])
    parameter_text = longtable('Baseline 指定情境之驗證選定超參數', 'tab:validation-selected-parameters',
        ['分類器', '驗證選定值（粗體）'], ['0.25', '0.71'], parameter_rows,
        r'資料來源：本研究已保存的驗證選參紀錄。固定展示 Old 1999--2011 年、Hybrid 採樣。粗體表示該情境中依驗證 AUC 選定的參數；其他年份或採樣可能選出不同設定，集成部分另用固定參數。')
    (OUT / 'validation-selected-parameters.tex').write_text(
        parameter_text.replace(r'\small\setstretch', r'\footnotesize\setstretch'), encoding='utf-8')
    baseline_best = group_maxima(baseline, ['candidate'])
    sampling_best = group_maxima(sampling_rows, ['candidate', 'method'])
    write("baseline-summary.tex", "超參數調整後各分類器之測試結果", "tab:results-baseline",
          ["分類器", "策略", "AUC", "F1-score", "Recall"], ["0.30", "0.13", "0.16", "0.16", "0.16"],
          [[r['candidate'], r['method']] + [emphasized(r[m], baseline_best[(r['candidate'],)][m]) for m in METRICS] for r in baseline],
          r"粗體表示同一分類器三種策略中該指標的最高值，不代表統計顯著。", merge_columns=(0,))
    write("baseline-sampling.tex", "實驗一 Baseline 各採樣設定之測試結果", "tab:results-baseline-sampling",
          ["分類器", "策略", "採樣", "AUC", "F1-score", "Recall"], ["0.25", "0.10", "0.13", "0.15", "0.15", "0.15"],
          [[r['candidate'], r['method'], SAMPLING[r['sampling']]] + [emphasized(r[m], sampling_best[(r['candidate'], r['method'])][m]) for m in METRICS] for r in sampling_rows],
          r"Old／New 各採樣平均 15 組邊界，Retrain 每種採樣只有一筆。粗體比較同一分類器與策略下的四種採樣，表示該指標最高值。Under、Over、Hybrid 分別為低採樣、過採樣與混合採樣。", merge_columns=(0,1))
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
    year_best = group_maxima(yearly_records, ['year'])
    rows = [[METHOD_LABELS[r['method']], r['year']] +
            [emphasized(r[m], year_best[(r['year'],)][m]) for m in METRICS] for r in yearly_records]
    write('ensemble-yearly.tex', '實驗一固定年份集成之逐年測試結果', 'tab:results-ensemble-yearly',
          ['方法', '年度', 'AUC', 'F1-score', 'Recall'], ['0.28', '0.12', '0.18', '0.18', '0.18'], rows,
          r"各值為十個種子的年度指標平均。粗體表示同一年度八種方法中該指標最高值。2015--2018 年正類紀錄數依序為 111、83、57、36；各年沿用 Calibration 固定的門檻。", merge_columns=(0,))
    pool = metrics[metrics.period.eq('pooled')]
    rows = []
    for method in ('Old_under', 'Old_over', 'Old_hybrid', 'New_under', 'New_over', 'New_hybrid'):
        g = pool[pool.method.eq(method)]
        assert len(g) == 10
        rows.append([method.replace('_', r'\_')] + [f"{g[m].mean():.4f}" for m in METRICS])
    maxima = [max(float(row[j]) for row in rows) for j in (1, 2, 3)]
    rows = [[r[0]] + [emphasized(float(r[j]), maxima[j-1]) for j in (1, 2, 3)] for r in rows]
    write('pool-models.tex', '實驗一六個基礎模型之測試結果', 'tab:results-pool-models',
          ['基礎模型', 'AUC', 'F1-score', 'Recall'], ['0.30', '0.21', '0.21', '0.21'], rows,
          r"完整 18 特徵、固定年份與相同十個種子；under、over、hybrid 分別為 TomekLinks、ADASYN、SMOTEENN。粗體為六模型中該指標最高值，不表示差異達統計顯著。各模型門檻由 Calibration 決定；本表與歷史 Baseline 分開解讀。")
    static = pool[pool.method.str.startswith('Static_')].copy()
    static['size'] = static.method.str.count(r'\+') + 1
    sizes = {2: 9, 3: 18, 4: 15, 5: 6, 6: 1}
    rows = []
    for size, count in sizes.items():
        g = static[static['size'].eq(size)]
        assert g.groupby('seed').size().eq(count).all() and len(g) == count * 10
        seed_means = g.groupby('seed')[list(METRICS)].mean()
        rows.append([str(size), str(count)] + [f"{seed_means[m].mean():.4f}" for m in METRICS])
    maxima = [max(float(row[j]) for row in rows) for j in (2, 3, 4)]
    rows = [[r[0], r[1]] + [emphasized(float(r[j]), maxima[j-2]) for j in (2, 3, 4)] for r in rows]
    write('static-size.tex', '實驗一不同靜態成員數之候選平均結果', 'tab:results-static-size',
          ['模型數', '候選數', 'AUC', 'F1-score', 'Recall'], ['0.15', '0.15', '0.21', '0.21', '0.21'], rows,
          r"每個種子先平均同一成員數全部候選的測試分數，再取十個種子平均。粗體為各成員數中該指標最高值，僅供比較，不依此選成員數，也不表示顯著差異。本表沒有把候選再次組成新模型。")
    fs_methods = methods[2:]
    for metric, metric_label in [('AUC', 'AUC'), ('F1', 'F1-score'), ('Recall', 'Recall')]:
        rows = []
        maxima = desc[desc.metric.eq(metric) & desc.method.isin(fs_methods)].groupby('method')['mean'].max()
        for fs, label in [('no_fs', 'no-FS（18）')] + list(FS_LABELS.items()):
            row = [label]
            for method in fs_methods:
                g = desc[desc.fs.eq(fs) & desc.method.eq(method) & desc.metric.eq(metric)]
                assert len(g) == 1 and g.n_seeds.iloc[0] == 10
                r = g.iloc[0]
                row.append(r"\shortstack[l]{" + emphasized(r['mean'], maxima.loc[method]) + r"\\{\scriptsize(\(\pm\)" + f"{r.seed_sd:.4f}" + r")}}")
            rows.append(row)
        write(f'fs-{metric.lower()}-absolute.tex', f'實驗二各特徵條件之 pooled {metric_label}', f'tab:results-fs-{metric.lower()}',
              ['FS'] + [METHOD_LABELS[m].replace('DCS-', '') for m in fs_methods], ['0.16'] + ['0.123'] * 6, rows,
              r"各格上列為十個種子的平均值，下列為種子標準差。粗體表示同一方法（同一欄）九種特徵條件中最高的平均分數，四位小數相同者並列；不代表已選定最佳設定或差異達統計顯著。50\%、80\% 分別保留 9、14 個特徵，no-FS 為 18 個；V-static 可依各條件的 Calibration 重新選成員。")
    proof = dict(status='passed', source_sha256=sources,
                 generated_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in OUT.iterdir() if p.is_file()},
                 baseline_sources=len(BASELINES), baseline_summary_rows=len(baseline), baseline_sampling_rows=len(sampling_rows),
                 yearly_rows=len(yearly_records), fs_settings=9, fs_methods=6,
                 selected_parameter_context=dict(split='split_13+3', method='Old', sampling='hybrid', selection='saved validation AUC'),
                 selected_parameter_rows=len(selected_rows),
                 scope='descriptive Test reporting only; no model selection or additional training')
    PROOF.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in proof.items() if not k.endswith('sha256')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
