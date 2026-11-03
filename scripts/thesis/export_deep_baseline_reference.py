"""Restore explicitly pinned historical neural baseline results as provisional references.

These final-status CSVs are not event-target results or independent seed repeats.
No fitting, test-based configuration selection or significance testing occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from export_fixed_statistics import longtable, sha
from export_chapter_results import aggregate_baseline, SAMPLING

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis/latex/generated/deep-baseline-reference-20261103"
PROOF = ROOT / "thesis/drafts/deep_baseline_restore_20261103/results-proof.json"
SOURCES = {
    "TabM": "results/phase1_baseline/tabm/tuned_light_full_20260410/tuned/bankruptcy_year_splits_tabm_raw.csv",
    "TabICL": "results/phase1_baseline/tabicl/tabicl_gpu_20260506_temp075_ctx2048/bankruptcy_year_splits_tabicl_raw.csv",
}
PERIODS = ("Old", "New", "Retrain")
METRICS = ("AUC", "F1", "Recall")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        proof = json.loads(PROOF.read_text(encoding="utf-8"))
        assert proof["status"] == "passed" and proof["event_results_pending"]
        for key in ("source_sha256", "generated_sha256"):
            for name, expected in proof[key].items():
                assert sha(ROOT / name) == expected, name
        print("PASS: provisional TabM/TabICL sources, aggregation and exported tables")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {Path(__file__).resolve().relative_to(ROOT).as_posix(): sha(Path(__file__).resolve())}
    for name in ("scripts/thesis/export_fixed_statistics.py", "scripts/thesis/export_chapter_results.py",
                 "experiments/phase1_baseline/bankruptcy_year_splits_tabm.py",
                 "experiments/phase1_baseline/bankruptcy_year_splits_tabicl.py",
                 "experiments/_shared/common_bankruptcy.py", "src/models/tabicl_wrapper.py"):
        sources[name] = sha(ROOT / name)
    summaries, samplings, selected = [], [], []
    for classifier, relative in SOURCES.items():
        p = ROOT / relative
        sources[relative] = sha(p)
        d = pd.read_csv(p)
        assert len(d) == 124 and set(d.method) == set(PERIODS)
        overall, by_sampling = aggregate_baseline(d)
        summaries += [dict(classifier=classifier, target="final_status", status="provisional_reference", **row) for row in overall]
        samplings += [dict(classifier=classifier, target="final_status", **row) for row in by_sampling]
        if classifier == "TabM":
            context = d[d.split.eq("split_13+3") & d.method.eq("Old") & d.sampling.eq("hybrid")]
            assert len(context) == 1 and pd.notna(context.iloc[0].tune_best_params)
            selected = json.loads(context.iloc[0].tune_best_params)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(samplings).to_csv(OUT / "sampling.csv", index=False)
    note = (r"資料來源：本研究既有最終狀態標籤之 TabM 與 TabICL 執行紀錄，暫列待更新。\par "
            r"本表使用公司最終狀態標籤與舊驗證方式，事件數據待更新。Old／New 平均 15 個邊界與四採樣，Retrain 僅四採樣；"
            r"紀錄數不是獨立種子數，不與事件模型作同條件排名。TabICL 沿用既有入口名稱，歷史後端待核對。")
    rows = [[r["classifier"], r["method"], str(r["n_rows"])] + [f"{r[m]:.4f}" for m in METRICS] for r in summaries]
    (OUT / "summary.tex").write_text(longtable(
        "TabM 與 TabICL 既有最終狀態標籤結果（暫列，待事件數據更新）", "tab:deep-reference-summary",
        ["分類器", "策略", "紀錄數", "AUC", "F1-score", "Recall"],
        ["0.20", "0.12", "0.09", "0.16", "0.16", "0.16"], rows, note, merge_columns=(0,)), encoding="utf-8")
    rows = [[r["classifier"], r["method"], SAMPLING[r["sampling"]]] + [f"{r[m]:.4f}" for m in METRICS] for r in samplings]
    (OUT / "sampling.tex").write_text(longtable(
        "TabM 與 TabICL 既有各採樣結果（最終狀態標籤，暫列）", "tab:deep-reference-sampling",
        ["分類器", "策略", "採樣", "AUC", "F1-score", "Recall"],
        ["0.16", "0.12", "0.17", "0.14", "0.14", "0.14"], rows,
        note + r"\par 各採樣列的 Old／New 平均 15 個邊界，Retrain 各僅一筆；同一分類器分組不跨頁。",
        merge_columns=(0, 1)), encoding="utf-8")
    settings = [["TabM", "展示情境", "Old 1999--2011、Hybrid；由已保存驗證選參紀錄取值"]]
    settings += [["TabM", r"\nolinkurl{" + key + "}", str(value)] for key, value in selected.items()]
    settings += [["TabICL", "選參與後端", "原 CSV 未保存選定參數及可確認的後端版本"],
                 ["兩分類器", "事件標籤", "待以相同年份、驗證規則與種子重跑後更新"]]
    (OUT / "settings.tex").write_text(longtable(
        "深度學習分類器之既有設定紀錄與更新狀態", "tab:deep-reference-settings",
        ["分類器", "項目", "保存內容"], ["0.16", "0.28", "0.48"], settings,
        r"資料來源：本研究既有原始結果欄位與執行入口核對。\par "
        r"TabM 僅展示預先固定情境的一組已選參數，不代表所有邊界皆使用相同設定；不由所選值反推完整歷史搜尋預算。"
        r"未保存的 TabICL 設定不依目錄名稱或測試分數補造。", merge_columns=(0,)), encoding="utf-8")
    proof = dict(status="passed", source_sha256=sources,
                 generated_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in OUT.iterdir() if p.is_file()},
                 classifier_sources=SOURCES, target="company final status; historical reference only", event_results_pending=True,
                 n_raw_rows_per_classifier=124, aggregation="four sampling means per boundary, then 15 boundary means; Retrain four rows only",
                 seed_scope="CSV does not record independent multi-seed repeats; no seed SD inferred",
                 selection="explicitly pinned existing archives; no comparison of Test scores across variants to choose a source",
                 historical_backend="TabICL entry supports tabicl or tabpfn; historical backend not confirmed from CSV")
    PROOF.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
