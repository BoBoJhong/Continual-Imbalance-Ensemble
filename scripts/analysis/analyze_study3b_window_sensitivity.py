"""Validate and summarize the fixed 2/3/5/7-year experiment.

Chart contract: compare four window categories and two model strategies using
grouped bars (annual-macro AP / pooled AP) in a static thesis PNG/PDF. Two roots,
blue/orange and hatch distinction, zero baselines; eight configuration rows.
Annual detail, denominators, and companion metrics remain in source CSVs.
Company intervals condition on seed 42; seed variability is reported separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.phase_flexible import rolling_bankruptcy_overlap_ensemble as base
from experiments.phase_flexible.study3_ab_fair_comparison import paired_company_cluster_bootstrap
from experiments.phase_flexible.study3b_window_sensitivity import WIDTHS, sha256
from src.evaluation import compute_metrics

METRICS = ["PR_AUC", "AUC", "F1", "Recall", "Precision", "G_Mean",
           "Recall_at_5pctFPR", "Realized_FPR_at_5pctFPR", "Type1_Error"]


def summarize(yearly, pooled):
    rows = []
    for (width, method), group in yearly.groupby(["window_years", "method"]):
        seed_macros = group.groupby("seed")[METRICS].mean()
        annual = group.groupby("test_feature_year")[METRICS].mean()
        pool = pooled.loc[(pooled.window_years == width) & (pooled.method == method)]
        row = {"window_years": width, "method": method,
               "n_seeds": group.seed.nunique(), "n_years": len(annual),
               "annual_AP_sd": annual.PR_AUC.std(ddof=1),
               "annual_AP_min": annual.PR_AUC.min(), "annual_AP_max": annual.PR_AUC.max()}
        for metric in METRICS:
            row[f"macro_{metric}"] = seed_macros[metric].mean()
            row[f"macro_{metric}_seed_sd"] = seed_macros[metric].std(ddof=1)
            row[f"pooled_{metric}"] = pool[metric].mean()
            row[f"pooled_{metric}_seed_sd"] = pool[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def annual_block_sensitivity(yearly, replicates=2000):
    """Circular moving blocks of annual macro AP differences, never row IID."""
    annual = yearly.groupby(["window_years", "method", "test_feature_year"]).PR_AUC.mean()
    comparisons = [(3, "FIFO3", w, "FIFO3") for w in WIDTHS if w != 3]
    comparisons += [(w, "FIFO3", w, "Newest") for w in WIDTHS]
    rows = []
    for block_length in (2, 3):
        rng = np.random.default_rng(20260916)
        n = yearly.test_feature_year.nunique()
        starts = rng.integers(0, n, size=(replicates, int(np.ceil(n / block_length))))
        indices = ((starts[:, :, None] + np.arange(block_length)) % n).reshape(replicates, -1)[:, :n]
        for rw, rm, cw, cm in comparisons:
            difference = (annual.loc[rw, rm] - annual.loc[cw, cm]).to_numpy()
            values = difference[indices].mean(axis=1)
            rows.append({"reference": f"W{rw}_{rm}", "comparison": f"W{cw}_{cm}",
                         "metric": "annual_macro_AP", "block_length": block_length,
                         "difference": difference.mean(), "ci95_low": np.quantile(values, .025),
                         "ci95_high": np.quantile(values, .975), "n_years": n,
                         "replicates": replicates,
                         "scope": "exploratory circular annual blocks; no retraining; not joint company-time inference"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed" or 42 not in manifest["seeds"]:
        raise ValueError("Require completed run including seed 42")
    years, seeds = manifest["test_years"], manifest["seeds"]
    yearly, pooled, audits = [], [], []
    reference_predictions = None
    checks = []
    seed_predictions_identical = True
    for seed in seeds:
        folder = run / f"seed_{seed}"
        y = pd.read_csv(folder / "by_year.csv")
        p = pd.read_csv(folder / "pooled.csv")
        a = pd.read_csv(folder / "training_audit.csv")
        pred = pd.read_csv(folder / "predictions.csv.gz")
        keys = ["window_years", "method", "test_feature_year", "source_row_id"]
        pred = pred.sort_values(keys).reset_index(drop=True)
        assert not pred.duplicated(keys).any()
        expected = {(w, m, t) for w in WIDTHS for m in ("Newest", "FIFO3") for t in years}
        assert set(map(tuple, y[["window_years", "method", "test_feature_year"]].to_numpy())) == expected
        assert len(y) == len(expected)
        assert set(pred.seed) == {seed}
        assert np.isfinite(pred.y_proba).all() and pred.y_proba.between(0, 1).all()
        assert np.array_equal((pred.y_proba >= pred.f1_threshold).astype(int), pred.y_pred_f1)
        assert np.array_equal((pred.y_proba >= pred.fpr5_threshold).astype(int), pred.y_pred_fpr5)
        assert (a.train_end < a.added_for_test_year - 1).all()
        assert (a.train_end - a.train_start + 1 == a.window_years).all()
        assert len(a) == len(WIDTHS) * (len(years) + 2)
        identity = None
        for (width, method), group in pred.groupby(["window_years", "method"]):
            ids = group[["test_feature_year", "source_row_id", "company_name", "y_true"]].reset_index(drop=True)
            if identity is not None:
                pd.testing.assert_frame_equal(ids, identity)
            identity = ids
            recomputed = base._pooled_summary(group).iloc[0]
            saved = p.loc[(p.window_years == width) & (p.method == method)].iloc[0]
            np.testing.assert_allclose(recomputed[METRICS].astype(float), saved[METRICS].astype(float), atol=1e-12)
        if seed == 42:
            reference_predictions = pred.copy()
            for (width, method, year), group in pred.groupby(["window_years", "method", "test_feature_year"]):
                recomputed = compute_metrics(group.y_true, group.y_proba, y_pred=group.y_pred_f1)
                saved = y.loc[(y.window_years == width) & (y.method == method)
                              & (y.test_feature_year == year)].iloc[0]
                for metric, value in recomputed.items():
                    np.testing.assert_allclose(value, saved[metric], atol=1e-12)
        yearly.append(y)
        pooled.append(p)
        audits.append(a)
        checks.append({"seed": seed, "status": "passed", "prediction_rows": len(pred),
                       "fitted_models": len(a), "paired_test_rows": len(identity),
                       "test_events": int(identity.y_true.sum())})
    assert reference_predictions is not None
    for seed in seeds:
        pred = pd.read_csv(run / f"seed_{seed}/predictions.csv.gz").sort_values(keys).reset_index(drop=True)
        cols = keys + ["company_name", "y_true", "y_proba", "y_pred_f1", "y_pred_fpr5", "f1_threshold", "fpr5_threshold"]
        seed_predictions_identical &= reference_predictions[cols].equals(pred[cols])
    yearly, pooled, audits = pd.concat(yearly), pd.concat(pooled), pd.concat(audits)
    summary = summarize(yearly, pooled)
    output = run / "analysis" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    print(f"ANALYSIS_DIR={output}", flush=True)
    summary.to_csv(output / "summary.csv", index=False)
    yearly.to_csv(output / "by_year_all_seeds.csv", index=False)
    pooled.to_csv(output / "pooled_all_seeds.csv", index=False)
    cost = audits.groupby(["window_years", "seed"], as_index=False).agg(
        fitted_models=("model_id", "size"), total_fit_rows=("n_train", "sum"),
        fit_seconds=("fit_seconds", "sum"), min_positive_per_window=("n_positive_before", "min"),
        max_positive_per_window=("n_positive_before", "max"))
    cost.to_csv(output / "cost_by_seed.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "validation_checks.csv", index=False)
    intervals = []
    ref = reference_predictions.copy()
    ref["method"] = "W" + ref.window_years.astype(str) + "_" + ref.method
    groups = [(ref.loc[ref.method.str.endswith("FIFO3")], "W3_FIFO3")]
    groups += [(ref.loc[ref.window_years == w], f"W{w}_FIFO3") for w in WIDTHS]
    for selected, reference_method in groups:
        print(f"Bootstrap {reference_method}, {selected.method.nunique()} configurations", flush=True)
        intervals.append(paired_company_cluster_bootstrap(
            selected, reference_method=reference_method,
            replicates=args.bootstrap_replicates, seed=20260916))
    company_intervals = pd.concat(intervals).assign(model_seed=42)
    company_intervals.to_csv(output / "company_bootstrap.csv", index=False)
    temporal_intervals = annual_block_sensitivity(yearly)
    temporal_intervals.to_csv(output / "annual_block_sensitivity.csv", index=False)
    # This manuscript-ready table is generated from the reviewed numeric artifacts.
    table = ["| 年數 | 方法 | 年度平均 AP | 年度 AP SD | pooled AP | AUC | F1 | Recall | Precision |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        table.append(f"| {row.window_years} | {row.method} | {row.macro_PR_AUC:.4f} | "
                     f"{row.annual_AP_sd:.4f} | {row.pooled_PR_AUC:.4f} | {row.pooled_AUC:.4f} | "
                     f"{row.pooled_F1:.4f} | {row.pooled_Recall:.4f} | {row.pooled_Precision:.4f} |")
    table.extend(["", "年度 AP SD 為十個年度 AP 的樣本標準差，不是種子標準差或信賴區間。", "",
                  "| 差值方向 | pooled AP 差 | 公司群聚 95% CI | 年度平均 AP 差 | 年度區塊 2 年 CI | 年度區塊 3 年 CI |",
                  "|---|---:|---|---:|---|---|"])
    for row in company_intervals.loc[company_intervals.metric == "AP"].itertuples(index=False):
        blocks = temporal_intervals.loc[(temporal_intervals.reference == row.reference_method)
                                       & (temporal_intervals.comparison == row.comparison_method)]
        b2, b3 = blocks.loc[blocks.block_length == 2].iloc[0], blocks.loc[blocks.block_length == 3].iloc[0]
        table.append(f"| {row.reference_method} − {row.comparison_method} | "
                     f"{row.observed_reference_minus_comparison:.4f} | [{row.ci95_low:.4f}, {row.ci95_high:.4f}] | "
                     f"{b2.difference:.4f} | [{b2.ci95_low:.4f}, {b2.ci95_high:.4f}] | "
                     f"[{b3.ci95_low:.4f}, {b3.ci95_high:.4f}] |")
    table.extend(["", "公司區間採 seed 42 固定預測；年度區塊區間採各 seed 年度 AP 的平均差。兩者統計量與不確定性範圍不同。", "",
                  "| 年數 | 每 seed 訓練模型數 | 累積 fitting rows | 每窗口正例範圍 | 訓練秒數中位數 |",
                  "|---:|---:|---:|---|---:|"])
    for width, group in cost.groupby("window_years"):
        assert group.total_fit_rows.nunique() == 1
        table.append(f"| {width} | {int(group.fitted_models.iloc[0])} | {int(group.total_fit_rows.iloc[0]):,} | "
                     f"{int(group.min_positive_per_window.min())}–{int(group.max_positive_per_window.max())} | "
                     f"{group.fit_seconds.median():.2f} |")
    (output / "thesis_tables.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    colors = {"Newest": "#2F6B9A", "FIFO3": "#D97706"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric, title in zip(axes, ("macro_PR_AUC", "pooled_PR_AUC"),
                               ("Annual macro average precision", "Pooled average precision"), strict=True):
        for index, method in enumerate(("Newest", "FIFO3")):
            frame = summary.loc[summary.method == method].sort_values("window_years")
            bars = ax.bar(np.arange(4) + (index - .5) * .36, frame[metric], .36,
                          label=method, color=colors[method], hatch="" if index == 0 else "//")
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set_xticks(np.arange(4), [str(w) for w in WIDTHS])
        ax.set_xlabel("Training window (years)")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, summary[metric].max() * 1.25)
        ax.set_ylabel("Average precision (AP)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5E7EB")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Training-window sensitivity | 2009-2018 test feature years", fontsize=13)
    fig.text(.5, .015, "10 seeds; same test observations. Macro weights each year equally; pooled combines rows.\n"
             "Exploratory point estimates; uncertainty and annual variability are reported separately.",
             ha="center", fontsize=8, color="#4B5563")
    fig.tight_layout(rect=(0, .08, 1, .94))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"window_sensitivity.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    analysis_manifest = {"status": "completed", "source_run": str(run),
                         "source_manifest_sha256": sha256(run / "manifest.json"),
                         "script_sha256": sha256(Path(__file__)),
                         "all_seed_predictions_identical": bool(seed_predictions_identical),
                         "company_bootstrap_model_seed": 42,
                         "company_bootstrap_replicates": args.bootstrap_replicates,
                         "company_bootstrap_scope": "fixed predictions, company clusters, pooled metric differences",
                         "temporal_scope": "annual macro AP differences; circular blocks 2/3; 2000 replicates; 10 years only",
                         "multiple_comparisons": "descriptive marginal intervals, no confirmatory significance claims",
                         "validation": checks,
                         "artifacts": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / "analysis_manifest.json").write_text(json.dumps(analysis_manifest, indent=2), encoding="utf-8")
    print(summary[["window_years", "method", "macro_PR_AUC", "annual_AP_sd", "pooled_PR_AUC",
                   "pooled_Recall", "pooled_Precision"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
