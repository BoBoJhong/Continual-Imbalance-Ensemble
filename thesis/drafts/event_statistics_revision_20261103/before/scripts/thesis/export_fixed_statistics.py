"""Export verified, explicitly pinned statistical analysis to thesis tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METHOD_LABELS = {"Old3": "Old3", "New3": "New3", "Equal6": "Equal6", "ValidationSelectedStatic": "V-static",
                 "DCS_OLA": "DCS-OLA", "DCS_LCA": "DCS-LCA", "KNORA_E": "KNORA-E", "KNORA_U": "KNORA-U"}
FS_LABELS = {"mi_r50": "MI 50\\%", "mi_r80": "MI 80\\%", "cart_r50": "CART 50\\%", "cart_r80": "CART 80\\%",
             "shap_r50": "SHAP 50\\%", "shap_r80": "SHAP 80\\%", "rfe_r50": "RFE 50\\%", "rfe_r80": "RFE 80\\%"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_analysis(root=ROOT):
    config = json.loads((root / "config/thesis_sources.json").read_text(encoding="utf-8-sig"))
    binding = next(a for a in config["analyses"] if a["id"] == "fixed_company_statistics")
    directory = (root / binding["directory"]).resolve()
    assert directory.is_relative_to(root / "results/statistical_tests")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert sha(directory / "manifest.json") == binding["manifest_sha256"]
    assert manifest["specification"]["protocol"] == binding["expected_protocol"]
    assert manifest["specification"]["n_tests"] == 165
    for path, expected in manifest["source_sha256"].items():
        assert sha(root / path) == expected, f"Statistical source changed: {path}"
    for path, expected in manifest["output_sha256"].items():
        assert sha(directory / path) == expected, f"Statistical output changed: {path}"
    return directory, manifest


def grouped_cells(rows, merge_columns):
    """Merge labels within their parent group and keep outer groups on one page."""
    result = [list(row) for row in rows]
    for column in sorted(merge_columns):
        start = 0
        while start < len(rows):
            end = start + 1
            while end < len(rows) and rows[end][:column+1] == rows[start][:column+1]:
                end += 1
            if end-start > 1 and rows[start][column]:
                result[start][column] = r"\multirow[t]{" + str(end-start) + r"}{=}{" + rows[start][column] + "}"
                for index in range(start+1, end):
                    result[index][column] = ""
            start = end
    outer = min(merge_columns) if merge_columns else None
    ends = {i for i in range(len(rows)) if outer is None or i == len(rows)-1
            or rows[i][:outer+1] != rows[i+1][:outer+1]}
    return result, ends


def source_note(note):
    """Put the source credit in parentheses, before reading notes."""
    if "資料來源" not in note:
        note = r"資料來源：本研究實驗結果整理。\par " + note
    match = re.match(r"資料來源：([^。]+)。(.*)", note, re.S)
    if not match:
        raise ValueError("Separate the source credit from explanatory notes")
    tail = match.group(2).strip()
    if tail.startswith(r"\par"):
        tail = tail[4:].lstrip()
    return "(資料來源：" + match.group(1) + ")" + (r"\par " + tail if tail else "")


def longtable(caption, label, columns, widths, rows, note, merge_columns=()):
    note = source_note(note)
    spec = "@{}" + "".join(r">{\raggedright\arraybackslash}p{" + w + r"\linewidth}" for w in widths) + "@{}"
    header = " & ".join(columns) + r" \\"
    text = [r"\begingroup", r"\small\setstretch{1.12}\setlength{\tabcolsep}{3pt}", r"\setlength{\LTleft}{\fill}\setlength{\LTright}{\fill}",
            r"\begin{longtable}{" + spec + "}", r"\caption{" + caption + r"}\label{" + label + r"}\\",
            r"\toprule", header, r"\midrule\endfirsthead", r"\multicolumn{" + str(len(columns)) + r"}{c}{表~\thetable（續）}\\",
            r"\toprule", header, r"\midrule\endhead", r"\midrule\multicolumn{" + str(len(columns)) + r"}{r}{續下頁}\\\endfoot",
            r"\bottomrule\endlastfoot"]
    # Short result tables should not leave one method or a few rows on the next page.
    if len(rows) <= 16 and not any(r'\nolinkurl' in cell for row in rows for cell in row):
        units = sum(2 if any(r'\shortstack' in cell for cell in row) else 1 for row in rows) + 8
        text.insert(3, r'\Needspace{' + str(units) + r'\baselineskip}')
    cells, ends = grouped_cells(rows, merge_columns)
    for i, row in enumerate(cells):
        text.append(" & ".join(row) + (r" \\" if i in ends else r" \\*"))
        if merge_columns and i in ends and i < len(rows)-1:
            text.append(r"\addlinespace[4pt]")
    text += [r"\end{longtable}", r"\noindent{\footnotesize " + note + "}", r"\endgroup", ""]
    return "\n".join(text)


def number(value, signed=False):
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    directory, manifest = verified_analysis()
    out = ROOT / "thesis/latex/generated" / ("statistics-" + directory.name)
    if args.check:
        state = json.loads((out / "export-manifest.json").read_text(encoding="utf-8"))
        assert sha(Path(__file__).resolve()) == state["exporter_sha256"]
        assert sha(directory / "manifest.json") == state["analysis_manifest_sha256"]
        for path, expected in state["generated_sha256"].items():
            assert sha(out / path) == expected, f"Generated statistical table changed: {path}"
        print("PASS: all 165 tests, prediction sources and exported statistical tables are current")
        return
    out.mkdir(parents=True, exist_ok=True)
    tests = pd.read_csv(directory / "paired_tests.csv")
    descriptive = pd.read_csv(directory / "descriptive.csv")
    note = (r"差值為候選方法減比較基準；信賴區間為公司群聚 bootstrap 的逐項 95\% 區間，非同時區間。"
            r"\(p_H\) 為兩項實驗全部 165 項比較共同採 Holm 校正之雙尾近似 \(p\) 值；星號表示 \(p_H<0.05\)。"
            r"所有比較使用相同公司重抽權重，模型與驗證門檻固定。")
    methods = list(METHOD_LABELS)
    maxima = descriptive[descriptive.fs.eq('no_fs') & descriptive.method.isin(methods)].groupby('metric')['mean'].max()
    rows = []
    for method in methods:
        group = descriptive[descriptive.fs.eq("no_fs") & descriptive.method.eq(method)].set_index("metric")
        cells = []
        for m in ('AUC', 'F1', 'Recall'):
            shown = number(group.loc[m, 'mean'])
            if shown == number(maxima.loc[m]):
                shown = r'\textbf{' + shown + '}'
            cells.append(shown + r'\(\pm\)' + number(group.loc[m, 'seed_sd']))
        rows.append([METHOD_LABELS[method]] + cells)
    (out / "experiment1-descriptive.tex").write_text(longtable(
        "實驗一固定年份集成之 pooled 結果", "tab:stats-exp1-descriptive", ["方法", "AUC", "F1-score", "Recall"],
        ["0.22", "0.24", "0.24", "0.24"], rows,
        r"數值為十個種子的平均值與標準差；標準差說明十次執行的分數差異。粗體為同一指標（同一欄）的最高平均值，四位小數相同者並列；不代表差異達統計顯著。V-static 由 Calibration AP 選成員，Equal6 使用全部六模型等權平均。"), encoding="utf-8")
    exp1 = tests[tests.experiment.eq("experiment1")]
    rows = []
    for row in exp1[exp1.metric.eq("AUC")].itertuples():
        rows.append([METHOD_LABELS[row.method_a], number(row.difference, True),
                     "[" + number(row.ci_low, True) + ", " + number(row.ci_high, True) + "]",
                     number(row.p_holm) + (r"\(^{*}\)" if row.reject_holm_005 else "")])
    (out / "experiment1-auc-tests.tex").write_text(longtable(
        "實驗一各集成方法相對 Equal6 之 AUC 配對檢定", "tab:stats-exp1-auc", ["方法", r"\(\Delta\mathrm{AUC}\)", "95\\% 信賴區間", r"\(p_H\)"],
        ["0.25", "0.17", "0.33", "0.17"], rows, note), encoding="utf-8")
    exp2 = tests[tests.experiment.eq("experiment2")]
    fs_methods = methods[2:]
    rows = []
    for fs, fs_label in FS_LABELS.items():
        row = [fs_label]
        for method in fs_methods:
            test = exp2[exp2.fs_a.eq(fs) & exp2.method_a.eq(method) & exp2.metric.eq("AUC")].iloc[0]
            row.append(number(test.difference, True) + (r"\(^{*}\)" if test.reject_holm_005 else ""))
        rows.append(row)
    (out / "experiment2-auc-differences.tex").write_text(longtable(
        "實驗二各 FS 設定相對同方法 no-FS 之 AUC 差值", "tab:stats-exp2-auc", ["FS"] + [METHOD_LABELS[m].replace("DCS-", "") for m in fs_methods],
        ["0.16"] + ["0.123"] * 6, rows,
        r"50\% 與 80\% 設定實際保留 9 與 14 個特徵。正值表示 AUC 增加。星號定義與表~\ref{tab:stats-exp1-auc} 相同；完整區間、F1-score、Recall 與校正值見附錄~\ref{sec:stats-full}。"), encoding="utf-8")
    rows = []
    for row in exp1.itertuples():
        rows.append([METHOD_LABELS[row.method_a], row.metric, number(row.difference, True),
                     "[" + number(row.ci_low, True) + ", " + number(row.ci_high, True) + "]", f"{row.p_raw:.5f}", f"{row.p_holm:.5f}"])
    (out / "appendix-experiment1.tex").write_text(longtable(
        "實驗一完整配對檢定：各方法減 Equal6", "tab:stats-all-exp1", ["方法", "指標", "差值", "95\\% 信賴區間", r"\(p\)", r"\(p_H\)"],
        ["0.19", "0.09", "0.13", "0.30", "0.10", "0.10"], rows, note, merge_columns=(0,)), encoding="utf-8")
    for metric in ("AUC", "F1", "Recall"):
        rows = []
        for row in exp2[exp2.metric.eq(metric)].itertuples():
            rows.append([FS_LABELS[row.fs_a], METHOD_LABELS[row.method_a], number(row.difference, True),
                         "[" + number(row.ci_low, True) + ", " + number(row.ci_high, True) + "]", f"{row.p_raw:.5f}", f"{row.p_holm:.5f}"])
        (out / f"appendix-experiment2-{metric}.tex").write_text(longtable(
            "實驗二完整 " + metric + " 配對檢定：FS 減同方法 no-FS", "tab:stats-all-exp2-" + metric.lower(),
            ["FS", "方法", "差值", "95\\% 信賴區間", r"\(p\)", r"\(p_H\)"],
            ["0.14", "0.19", "0.13", "0.29", "0.08", "0.08"], rows, note, merge_columns=(0,)), encoding="utf-8")
    # Exact results used by the authored narrative, kept alongside the tables.
    summary = dict(replicates=manifest["specification"]["replicates"], n_tests=len(tests),
                   rejected_by_experiment=tests.groupby("experiment").reject_holm_005.sum().to_dict(),
                   rejected_by_experiment_metric=tests.groupby(["experiment", "metric"]).reject_holm_005.sum().astype(int).to_dict())
    summary["rejected_by_experiment_metric"] = {"/".join(k): v for k, v in summary["rejected_by_experiment_metric"].items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    state = dict(analysis_directory=directory.relative_to(ROOT).as_posix(), analysis_manifest_sha256=sha(directory / "manifest.json"),
                 exporter_sha256=sha(Path(__file__).resolve()), generated_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != "export-manifest.json"})
    (out / "export-manifest.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
