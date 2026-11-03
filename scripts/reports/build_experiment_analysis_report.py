"""Generate the data-analyst style tables used by the all-experiment report.

Writes Markdown table snippets for the multi-classifier Study 3B pipeline from
completed seeds only. Metrics are restricted to AUC, F1 and Recall.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase_flexible import rolling_bankruptcy_overlap_ensemble as base  # noqa: E402

RUN = ROOT / "results/phase_flexible/study3b_full_pipeline/runs/20260928T185026381432Z"
M = ["AUC", "F1", "Recall"]
YEARS = list(range(2009, 2019))
CLS = {"rf": "RF", "xgb": "XGBoost", "lgb": "LightGBM", "lr": "LR", "mlp": "MLP", "svm": "SVM"}
SMP = {"none": "none", "under": "under", "over": "over", "hybrid": "hybrid"}
f = "{:.3f}".format


def table(frame, bold=True):
    cols = list(frame.columns)
    out = "| " + " | ".join(str(c) for c in [frame.index.name or ""] + cols) + " |\n"
    out += "|---|" + "---:|" * len(cols) + "\n"
    best = {c: frame[c].max() for c in cols if np.issubdtype(frame[c].dtype, np.number)}
    for idx, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"**{f(v)}**" if bold and v == best[c] else f(v))
            else:
                cells.append(str(v))
        out += f"| {idx} | " + " | ".join(cells) + " |\n"
    return out + "\n"


def main(output):
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    seeds = [s for s in manifest["config"]["seeds"] if (RUN / f"seed_{s}/complete.json").exists()]
    sm = pd.concat([pd.read_csv(RUN / f"seed_{s}/summary.csv") for s in seeds])
    by = pd.concat([pd.read_csv(RUN / f"seed_{s}/by_year.csv") for s in seeds])
    avg = sm.groupby("method")[M].mean()
    yearly = by.groupby(["method", "test_feature_year"])[M].mean()
    parts = {"seeds": ", ".join(map(str, seeds)), "n_seeds": len(seeds)}

    rows = []
    for c, cn in CLS.items():
        for s in SMP:
            a, b = avg.loc[f"Newest_{c}_{s}"], avg.loc[f"FIFO_{c}_{s}"]
            rows.append({"配置": f"{cn}＋{s}", "單模型 AUC": a.AUC, "單模型 F1": a.F1, "單模型 Recall": a.Recall,
                         "FIFO AUC": b.AUC, "FIFO F1": b.F1, "FIFO Recall": b.Recall})
    parts["single_fifo"] = table(pd.DataFrame(rows).set_index("配置"))

    rows = []
    for c, cn in CLS.items():
        rows.append((f"{cn} 跨採樣（最新）", f"SamplingNewest_{c}"))
        rows.append((f"{cn} 跨採樣＋FIFO", f"SamplingFIFO_{c}"))
    for s in SMP:
        rows.append((f"六分類器＋{s}（最新）", f"ClassifierNewest_{s}"))
        rows.append((f"六分類器＋{s}＋FIFO", f"ClassifierFIFO_{s}"))
    rows += [("全部混合（最新）", "CombinedNewest"), ("全部混合＋FIFO", "CombinedFIFO"),
             ("每年 Validation 選集成", "ValidationSelectedEnsemble")]
    frame = pd.DataFrame([{"集成方式": n, **avg.loc[m].to_dict()} for n, m in rows]).set_index("集成方式")
    parts["ensembles"] = table(frame)

    rows = [("不做 FS（18 特徵）", "SelectedEnsemble_none_r100")]
    rows += [(f"{x.upper()} {r}%（{int(18 * r / 100)} 特徵）", f"SelectedEnsemble_{x}_r{r}")
             for x in ("mi", "cart", "rfe", "shap") for r in (50, 80)]
    rows += [("每年 Validation 選 FS", "ValidationSelectedFS")]
    parts["fs"] = table(pd.DataFrame([{"FS": n, **avg.loc[m].to_dict()} for n, m in rows]).set_index("FS"))

    fixed = [m for m in avg.index if not m.startswith(("Selected", "Validation"))]
    rank = avg.loc[fixed].rank(ascending=False)
    top = avg.loc[fixed].assign(**{"平均名次": rank.mean(axis=1), "最差名次": rank.max(axis=1)})
    top = top.sort_values("平均名次").head(8)
    names = {m: m for m in top.index}
    parts["ranking"] = "| 配置 | AUC | F1 | Recall | 三指標平均名次 | 最差名次 |\n|---|---:|---:|---:|---:|---:|\n" + "".join(
        f"| {names[m]} | {f(r.AUC)} | {f(r.F1)} | {f(r.Recall)} | {r['平均名次']:.1f} | {r['最差名次']:.1f} |\n"
        for m, r in top.iterrows()) + "\n"
    seed_rank = (sm[sm.method.isin(fixed)].set_index(["seed", "method"])[M]
                 .groupby(level=0).rank(ascending=False).xs("FIFO_rf_under", level=1))
    parts["seed_rank"] = "| seed | AUC 名次 | F1 名次 | Recall 名次 |\n|---|---:|---:|---:|\n" + "".join(
        f"| {s} | {r.AUC:.0f} | {r.F1:.0f} | {r.Recall:.1f} |\n" for s, r in seed_rank.iterrows()) + "\n"

    yr = pd.DataFrame({y: yearly.loc[("FIFO_rf_under", y)] for y in YEARS}).T
    single = pd.DataFrame({y: yearly.loc[("Newest_rf_under", y)] for y in YEARS}).T
    frame = pd.DataFrame({"單模型 AUC": single.AUC, "FIFO AUC": yr.AUC, "單模型 F1": single.F1, "FIFO F1": yr.F1,
                          "單模型 Recall": single.Recall, "FIFO Recall": yr.Recall})
    frame.index.name = "Test 年"
    parts["rf_yearly"] = table(frame, bold=False)
    parts["fifo_auc_wins"] = int((yr.AUC > single.AUC).sum())
    parts["fifo_better"] = {k: int(sum(avg.loc[f"FIFO_{c}_{s}", k] > avg.loc[f"Newest_{c}_{s}", k]
                                      for c in CLS for s in SMP)) for k in M}
    output.write_text(json.dumps(parts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"OUTPUT={output} SEEDS={seeds}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
