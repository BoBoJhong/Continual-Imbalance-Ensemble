"""Export the pinned event-label cluster tests without refitting or recomputing inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from export_fixed_statistics import METHOD_LABELS, FS_LABELS, longtable, number, sha, verified_analysis

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis/latex/generated/event-statistics-20261103"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    directory, manifest = verified_analysis(ROOT, "event_company_statistics")
    spec = manifest["specification"]
    assert spec["replicates"] == 9999 and spec["unit"] == "company_name"
    validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    assert validation["status"] == "passed" and validation["n_tests"] == 165
    assert validation["n_test_positive"] == 119 and validation["missing_configurations"] == 0
    if args.check:
        state = json.loads((OUT / "export-manifest.json").read_text(encoding="utf-8"))
        assert state["exporter_sha256"] == sha(Path(__file__).resolve())
        assert state["table_formatter_sha256"] == sha(ROOT / "scripts/thesis/export_fixed_statistics.py")
        assert state["analysis_manifest_sha256"] == sha(directory / "manifest.json")
        for name, expected in state["generated_sha256"].items():
            assert sha(OUT / name) == expected, name
        print("PASS: 165 event-label tests, upstream predictions and exported tables")
        return
    tests = pd.read_csv(directory / "paired_tests.csv")
    assert len(tests) == 165 and not tests.duplicated(["experiment", "fs_a", "method_a", "metric"]).any()
    OUT.mkdir(parents=True, exist_ok=True)
    note = (r"資料來源：本研究事件標籤公司群聚配對分析。\par "
            r"差值為候選方法減比較基準；區間為逐項 95\% 百分位區間，非同時區間。"
            r"9,999 次公司重抽；\(p_H\) 為本標籤兩項實驗共 165 項雙尾近似檢定的 Holm 校正值。"
            r"星號表示 \(p_H<0.05\)，未達顯著不表示等效；模型、種子及驗證決策固定。")
    exp1 = tests[tests.experiment.eq("experiment1")]
    rows = [[METHOD_LABELS[r.method_a], number(r.difference, True),
             "[" + number(r.ci_low, True) + ", " + number(r.ci_high, True) + "]",
             f"{r.p_holm:.4f}" + (r"\(^{*}\)" if r.reject_holm_005 else "")]
            for r in exp1[exp1.metric.eq("AUC")].itertuples()]
    (OUT / "experiment1-auc-tests.tex").write_text(longtable(
        "事件標籤下各集成方法相對 Equal6 之 AUC 配對檢定", "tab:event-stats-exp1-auc",
        ["方法", r"\(\Delta\mathrm{AUC}\)", "95\\% 信賴區間", r"\(p_H\)"],
        ["0.25", "0.17", "0.33", "0.17"], rows, note), encoding="utf-8")
    groups = [("appendix-experiment1.tex", exp1, "實驗一：各方法減 Equal6", "tab:event-stats-all-exp1", False)]
    groups += [(f"appendix-experiment2-{metric}.tex",
                tests[tests.experiment.eq("experiment2") & tests.metric.eq(metric)],
                "實驗二 " + metric + "：FS 減同方法 no-FS", "tab:event-stats-all-exp2-" + metric.lower(), True)
               for metric in ("AUC", "F1", "Recall")]
    for name, group, caption, label, fs in groups:
        rows = []
        for r in group.itertuples():
            rows.append(([FS_LABELS[r.fs_a], METHOD_LABELS[r.method_a]] if fs else [METHOD_LABELS[r.method_a], r.metric]) +
                        [number(r.difference, True), "[" + number(r.ci_low, True) + ", " + number(r.ci_high, True) + "]",
                         f"{r.p_raw:.5f}", f"{r.p_holm:.5f}"])
        (OUT / name).write_text(longtable(
            "事件標籤" + caption, label,
            (["FS", "方法"] if fs else ["方法", "指標"]) + ["差值", "95\\% 信賴區間", r"\(p\)", r"\(p_H\)"],
            ["0.14", "0.19", "0.13", "0.29", "0.08", "0.08"] if fs else ["0.19", "0.09", "0.13", "0.30", "0.10", "0.10"],
            rows, note, merge_columns=(0,)), encoding="utf-8")
    summary = {"analysis_directory": directory.relative_to(ROOT).as_posix(), "replicates": spec["replicates"],
               "n_tests": len(tests), "rejected_by_experiment_metric": {
                   "/".join(k): int(v) for k, v in tests.groupby(["experiment", "metric"]).reject_holm_005.sum().items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    state = {"analysis_manifest_sha256": sha(directory / "manifest.json"),
             "exporter_sha256": sha(Path(__file__).resolve()),
             "table_formatter_sha256": sha(ROOT / "scripts/thesis/export_fixed_statistics.py"),
             "generated_sha256": {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != "export-manifest.json"}}
    (OUT / "export-manifest.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    proof = {"status": "passed", "source_sha256": {
        "scripts/thesis/export_event_statistics.py": sha(Path(__file__).resolve()),
        "scripts/thesis/export_fixed_statistics.py": sha(ROOT / "scripts/thesis/export_fixed_statistics.py"),
        (directory / "manifest.json").relative_to(ROOT).as_posix(): sha(directory / "manifest.json")},
        "generated_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in OUT.iterdir() if p.is_file()}}
    (ROOT / "thesis/drafts/event_statistics_revision_20261103/statistics-proof.json").write_text(
        json.dumps(proof, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
