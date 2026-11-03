"""Reproducible, source-checked Phase 1-3 bankruptcy result figures.

No training, result mutation, test-selected winners, or protocol mixing.
Algorithm names remain in the source key rather than the figure panels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets/research/experiment-results/bankruptcy"
RUN = ROOT / "results/phase2_ensemble/standard_dynamic/runs/20261004T202331745995Z"
INVENTORY = ROOT / "docs/audits/2026-10/phase123_split_audit_20261006.json"
METRICS = {"AUC": "AUC", "F1": "F1-score", "Recall": "Recall"}
SCORE_DOMAINS = {"AUC": (.45, 1.0), "F1": (0.0, .35), "Recall": (0.0, 1.0)}
SCORE_TICKS = {"AUC": [.5, .75, 1.0], "F1": [0, .1, .2, .3], "Recall": [0, .25, .5, .75, 1.0]}
BLUE, GOLD, GRAY = "#2F6FAE", "#B48632", "#555B61"
STYLE = {
    "Old": (BLUE, "o", "-"), "New": (GOLD, "s", "-"),
    "Retrain": (GRAY, None, "--"),
    "Equal6": (GRAY, "o", "-"),
    "ValidationSelectedStatic": ("#8A9096", "s", "--"),
    "DCS_OLA": (BLUE, "^", "-"), "DCS_LCA": ("#5B91C4", "v", "--"),
    "KNORA_E": (GOLD, "D", "-"), "KNORA_U": ("#D5AB61", "P", "--"),
}
LABELS = {
    "Old3": "Old-only", "New3": "New-only", "Equal6": "Equal-six",
    "ValidationSelectedStatic": "V-static", "DCS_OLA": "DCS-1",
    "DCS_LCA": "DCS-2", "KNORA_E": "DES-1", "KNORA_U": "DES-2",
}
METHODS = tuple(LABELS)
FIXED = ("Equal6", "DCS_OLA", "DCS_LCA", "KNORA_E", "KNORA_U")
ANNUAL = ("Equal6", "ValidationSelectedStatic", "DCS_OLA", "DCS_LCA", "KNORA_E", "KNORA_U")
FS_KEY = {
    "mi_r50": ("FS1", "MI", 9), "mi_r80": ("FS2", "MI", 14),
    "cart_r50": ("FS3", "CART", 9), "cart_r80": ("FS4", "CART", 14),
    "shap_r50": ("FS5", "SHAP", 9), "shap_r80": ("FS6", "SHAP", 14),
    "rfe_r50": ("FS7", "RFE", 9), "rfe_r80": ("FS8", "RFE", 14),
}
CANDIDATES = (
    ("C1", "Traditional ML", "XGBoost", "results/phase1_baseline/xgb/tuned/bankruptcy_year_splits_xgb_raw.csv"),
    ("C2", "Traditional ML", "LightGBM", "results/phase1_baseline/lgbm/bk_lgb_tuned48_20260421/tuned/bankruptcy_year_splits_lgb_raw.csv"),
    ("C3", "Traditional ML", "Random Forest", "results/phase1_baseline/random_forest/bankruptcy_year_splits_rf_raw.csv"),
    ("C4", "Traditional ML", "Logistic Regression", "results/phase1_baseline/logistic_regression/bk_lr_tuned4_20260421/tuned/bankruptcy_year_splits_lr_raw.csv"),
    ("C5", "Deep learning", "TabM", "results/phase1_baseline/tabm/tuned_light_full_20260410/tuned/bankruptcy_year_splits_tabm_raw.csv"),
    ("C6", "Deep learning", "TabICL-labelled entry; inspect recorded backend", "results/phase1_baseline/tabicl/tabicl_gpu_20260506_setA_temp075_ctx2048/bankruptcy_year_splits_tabicl_raw.csv"),
)
CHARTS, SOURCES, CHECKS = [], [], {}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def source(path, kind):
    row = {"path": rel(path), "sha256": sha256(path), "kind": kind}
    SOURCES.append(row)
    return row


def validate_scores(frame):
    values = frame[list(METRICS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Metrics must be finite proportions in [0, 1]")


def load_baselines():
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    known = {x["path"].replace("\\", "/"): x["sha256"] for x in inventory["baseline_result_inventory"]}
    frames = []
    for code, family, algorithm, filename in CANDIDATES:
        path = ROOT / filename
        if sha256(path) != known[filename]:
            raise ValueError(f"Historical artifact changed since inventory: {filename}")
        source(path, "historical baseline; incomplete training provenance")
        raw = pd.read_csv(path)
        raw.insert(0, "source_row", np.arange(len(raw)))
        raw = raw[raw.method.isin(["Old", "New", "Retrain"])].copy()
        if raw.duplicated(["split", "method", "sampling"]).any():
            raise ValueError(f"Duplicate baseline keys: {filename}")
        counts = raw.groupby(["method", "sampling"]).size()
        for sampling in ("none", "undersampling", "oversampling", "hybrid"):
            if any(counts.get((method, sampling), 0) != count for method, count in (("Old", 15), ("New", 15), ("Retrain", 1))):
                raise ValueError(f"Incomplete baseline grid: {filename}, {sampling}")
        raw["old_year_count"] = raw["split"].str.extract(r"split_(\d+)\+\d+")[0].astype(int)
        raw["old_end_year"] = raw.old_year_count + 1998
        raw["new_start_year"] = raw.old_end_year + 1
        raw["test_period"] = "2015-2018"
        raw["candidate_code"], raw["model_family"] = code, family
        raw["source_path"] = filename
        raw["display_role"] = np.where(raw.method.eq("Retrain"), "single full-data reference", "boundary-specific model")
        raw.loc[raw.method.eq("Retrain"), ["old_year_count", "old_end_year", "new_start_year"]] = np.nan
        validate_scores(raw)
        frames.append(raw)
    result = pd.concat(frames, ignore_index=True)
    CHECKS["baseline_rows_all_sampling"] = len(result)
    CHECKS["baseline_rows_hybrid"] = int(result.sampling.eq("hybrid").sum())
    CHECKS["baseline_provenance_limit"] = "Artifact hashes and coverage checked; historical models not retrained; six files have different tuning provenance."
    return result


def load_standard():
    manifest_path, audit_path = RUN / "manifest.json", RUN / "audit_validation.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    if manifest["status"] != "completed" or audit["status"] != "verified":
        raise ValueError("Completed run and independent audit required")
    if sha256(manifest_path) != audit["manifest_sha256"]:
        raise ValueError("Audit does not match manifest")
    source(manifest_path, "completed run metadata")
    source(audit_path, "existing independent numeric audit")
    source(RUN / "metrics.csv", "pooled and annual test metrics")
    source(RUN / "fs_paired_differences.csv", "official same-seed FS differences")
    frames = []
    for seed in manifest["config"]["seeds"]:
        for method, ratio in manifest["config"]["fs_configs"]:
            tag = "no_fs" if method == "none" else f"{method}_r{round(ratio * 100)}"
            directory = RUN / f"seed_{seed}" / tag
            complete = json.loads((directory / "complete.json").read_text())
            if complete["status"] != "completed" or sha256(directory / "metrics.csv") != complete["metrics_sha256"]:
                raise ValueError(f"Invalid configuration: {directory}")
            frames.append(pd.read_csv(directory / "metrics.csv"))
    data = pd.read_csv(RUN / "metrics.csv")
    key = ["seed", "fs", "method", "period"]
    if data.duplicated(key).any() or len(data) != 28350:
        raise ValueError("Unexpected or duplicate standard-run grid")
    rebuilt = pd.concat(frames, ignore_index=True)
    pd.testing.assert_frame_equal(data.sort_values(key).reset_index(drop=True), rebuilt.sort_values(key).reset_index(drop=True), check_exact=False, rtol=1e-12, atol=1e-12)
    validate_scores(data)
    if data.groupby(["seed", "fs"]).size().ne(315).any():
        raise ValueError("Each seed/FS configuration must contain 63 methods x 5 periods")
    if data.groupby("period")[["n_test", "n_test_positive"]].nunique().ne(1).any().any():
        raise ValueError("Methods must evaluate the same sample counts")
    CHECKS.update(standard_configurations=len(frames), standard_metric_rows=len(data), standard_seeds=int(data.seed.nunique()), standard_per_config_checksums=True, aggregate_reproduced_from_configuration_metrics=True, source_audit_at=audit["at"])
    return data, manifest


def axes_style(ax, metric=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#92979C")
    ax.grid(axis="y", color="#E4E6E8", linewidth=.6)
    ax.set_axisbelow(True)
    if metric:
        ax.set_ylim(*SCORE_DOMAINS[metric])
        ax.set_yticks(SCORE_TICKS[metric])
        ax.set_ylabel(METRICS[metric])


def header(fig, title, subtitle):
    fig.suptitle(title, x=.055, y=.98, ha="left", fontsize=13, fontweight="bold")
    fig.text(.055, .934, subtitle, ha="left", va="top", fontsize=9, color="#444B51")


def save(fig, phase, name, data, question, family, takeaway, notes, fields):
    directory = OUT / phase
    directory.mkdir(parents=True, exist_ok=True)
    png, svg, csv = [directory / f"{name}.{ext}" for ext in ("png", "svg", "csv")]
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(svg, facecolor="white", metadata={"Date": None})
    data.to_csv(csv, index=False, float_format="%.12g")
    contract = {
        "id": name, "phase": phase, "analytical_question": question,
        "takeaway": takeaway, "family": family, "renderer": "Matplotlib standalone static",
        "surface": "user-owned thesis figure; omit provider branding",
        "data_rows": len(data), "fields": fields, "notes": notes,
        "palette_policy": "at most two approved roots (blue/gold) plus neutrals; markers/line styles distinguish series",
        "exports": [rel(png), rel(svg)], "data": rel(csv),
        "figure_inches": [round(float(v), 3) for v in fig.get_size_inches()],
        "minimum_font_pt": 9, "metrics": list(METRICS.values()),
        "absolute_score_domains": SCORE_DOMAINS,
    }
    fig.canvas.draw()
    canvas = fig.bbox
    hidden_ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lower, upper = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lower - 1e-12 <= tick.get_loc() <= upper + 1e-12:
                    hidden_ticks.update((id(tick.label1), id(tick.label2)))
    for obj in fig.findobj(matplotlib.text.Text):
        if id(obj) in hidden_ticks or not obj.get_visible() or not obj.get_text():
            continue
        box = obj.get_window_extent(fig.canvas.get_renderer())
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > canvas.width + 1 or box.y1 > canvas.height + 1):
            raise ValueError(f"Text outside figure: {name}, {obj.get_text()}")
    renderer = fig.canvas.get_renderer()
    footers = [t.get_window_extent(renderer) for t in fig.texts if t.get_position()[1] < .1]
    for ax in fig.axes:
        if ax.xaxis.label.get_text() and any(ax.xaxis.label.get_window_extent(renderer).overlaps(b) for b in footers):
            raise ValueError(f"Axis label overlaps footer: {name}")
        ticks = [tick.label1 for tick in ax.xaxis.get_major_ticks() if id(tick.label1) not in hidden_ticks and tick.label1.get_visible() and tick.label1.get_text()]
        boxes = [t.get_window_extent(renderer) for t in ticks]
        if any(a.overlaps(b) for a, b in zip(boxes, boxes[1:])):
            raise ValueError(f"Adjacent x tick labels overlap: {name}")
    CHARTS.append(contract)
    plt.close(fig)


def phase1_figures(baselines):
    data = baselines[baselines.sampling.eq("hybrid")].copy()
    for metric in METRICS:
        fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.4), sharex=True, sharey=True)
        header(fig, f"Phase 1: {METRICS[metric]} by historical/new boundary", "Hybrid sampling | 15 boundaries | Test 2015-2018 | Historical snapshots")
        fig.subplots_adjust(left=.10, right=.98, top=.84, bottom=.10, hspace=.37, wspace=.15)
        for ax, (code, family, _, _) in zip(axes.flat, CANDIDATES):
            frame = data[data.candidate_code.eq(code)]
            for method in ("Old", "New"):
                values = frame[frame.method.eq(method)].sort_values("old_end_year")
                color, marker, line = STYLE[method]
                ax.plot(values.old_end_year, values[metric], color=color, marker=marker, markersize=3.2, linewidth=1.3, linestyle=line)
            reference = frame.loc[frame.method.eq("Retrain"), metric].item()
            ax.axhline(reference, color=GRAY, linestyle="--", linewidth=1.3)
            ax.set_title(f"{code} | {family}", loc="left", fontsize=10)
            axes_style(ax, metric)
            ax.set_xticks([1999, 2003, 2007, 2011, 2013])
            ax.set_xlim(1998.6, 2013.4)
        for ax in axes[-1]:
            ax.set_xlabel("Old-window end year")
        handles = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], linestyle=STYLE[m][2], label=m) for m in STYLE if m in ("Old", "New", "Retrain")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .895), ncol=3, frameon=False)
        fig.text(.055, .024, "Boundary sensitivity, not a forecast-time trend. Retrain is one fixed full-data reference.\nC1-C6 and the exact source snapshots are keyed in README; no test-best classifier is selected.", fontsize=9)
        save(fig, "phase1", f"phase1-hybrid-boundary-{metric.lower()}", data,
             f"How does {METRICS[metric]} vary over 15 Old/New boundaries under fixed hybrid sampling?",
             "line / six faceted parameter-sensitivity panels", "Describe the observed Old/New boundary sensitivity separately for each historical candidate.",
             "No uncertainty estimate for historical rows; no ranking or cross-phase effect attribution. Retrain is plotted as a reference, not 15 independent retrain runs.",
             ["candidate_code", "model_family", "method", "old_end_year", "sampling", "AUC", "F1", "Recall", "source_path", "source_row"])


def phase2_figures(metrics):
    no_fs = metrics[metrics.fs.eq("no_fs")]
    pooled = no_fs[no_fs.period.eq("pooled") & no_fs.method.isin(METHODS)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.2), sharey=True)
    header(fig, "Phase 2: pooled ensemble performance", "No FS | Test 2015-2018: 12,282 rows, 287 failed | Mean +/- SD of 10 seeds")
    fig.subplots_adjust(left=.165, right=.98, top=.81, bottom=.21, wspace=.22)
    for ax, metric in zip(axes, METRICS):
        summary = pooled.groupby("method")[metric].agg(["mean", "std"]).reindex(METHODS)
        y = np.arange(len(METHODS))
        ax.errorbar(summary["mean"], y, xerr=summary["std"], fmt="o", color=BLUE, markersize=4, capsize=2, linewidth=1.1)
        ax.set_yticks(y, [LABELS[m] for m in METHODS])
        ax.invert_yaxis(); ax.set_xlim(*SCORE_DOMAINS[metric]); ax.set_xticks(SCORE_TICKS[metric])
        ax.set_title(METRICS[metric], fontsize=10); ax.set_xlabel("Score")
        axes_style(ax)
        ax.grid(axis="x", color="#E4E6E8", linewidth=.6)
    fig.text(.055, .035, "V-static is selected on Calibration AP; classification thresholds are fixed on Calibration F1.\nSeed SD describes run variability, not an independent-data confidence interval.", fontsize=9)
    save(fig, "phase2", "phase2-pooled-ensemble-comparison", pooled, "How do fixed ensemble strategies compare on the same pooled Test rows?", "faceted dot and SD interval", "Compare the eight prespecified strategies using the same Test and ten seeds.", "Fixed run only; no same-protocol Retrain exists here. Algorithm code key in README.", list(pooled.columns))

    mixed = no_fs[no_fs.period.eq("pooled") & no_fs.method.str.startswith("Static_")].copy()
    mixed["ensemble_size"] = mixed.method.str.removeprefix("Static_").str.count(r"\+") + 1
    expected = {2: 9, 3: 18, 4: 15, 5: 6, 6: 1}
    if any(mixed[mixed.ensemble_size.eq(size)].groupby("seed").size().ne(count).any() for size, count in expected.items()):
        raise ValueError("Incorrect mixed-subset cardinality")
    per_seed = mixed.groupby(["seed", "ensemble_size"])[list(METRICS)].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.0))
    header(fig, "Phase 2: mixed-subset performance by pool size", "All same-size mixed subsets averaged within seed | Mean +/- SD of 10 seeds")
    fig.subplots_adjust(left=.09, right=.98, top=.78, bottom=.26, wspace=.35)
    for ax, metric in zip(axes, METRICS):
        s = per_seed.groupby("ensemble_size")[metric].agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], color=BLUE, marker="o", linewidth=1.3, markersize=4, capsize=2)
        axes_style(ax, metric)
        ax.set_xticks(list(expected), [f"{size}\n({count})" for size, count in expected.items()])
        ax.set_xlabel("Models (subsets per seed)")
    fig.text(.055, .035, "The plotted score averages subset metrics; it is not an extra ensemble's prediction.\nAll subsets are retained. Different sizes have different compositions; no test-best subset is selected.", fontsize=9)
    CHECKS["static_subsets_per_seed"] = expected
    save(fig, "phase2", "phase2-static-subset-size", mixed, "How do average candidate metrics vary with mixed-subset size?", "line / ordered model-count sensitivity", "Show descriptive metric averages for all subsets at each size, without selecting a Test winner.", "Seed SD is calculated after averaging within seed. This does not isolate a causal size effect.", list(mixed.columns))

    annual = no_fs[no_fs.period.ne("pooled") & no_fs.method.isin(ANNUAL)].copy()
    annual["year"] = annual.period.astype(int)
    counts = annual.groupby("year")[["n_test", "n_test_positive"]].first()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.8))
    header(fig, "Phase 2: annual test performance", "Four observed years | No FS | Mean +/- SD of 10 seeds | Positive class: failed")
    fig.subplots_adjust(left=.09, right=.98, top=.72, bottom=.29, wspace=.35)
    for ax, metric in zip(axes, METRICS):
        for i, method in enumerate(ANNUAL):
            s = annual[annual.method.eq(method)].groupby("year")[metric].agg(["mean", "std"])
            color, marker, _ = STYLE[method]
            ax.errorbar(s.index + (i - 2.5) * .06, s["mean"], yerr=s["std"], fmt=marker, color=color, markersize=4, capsize=1.5, linewidth=.9)
        axes_style(ax, metric)
        ax.set_xlim(2014.65, 2018.35)
        ax.set_xticks(counts.index, [f"{year}\n{int(row.n_test)}\n{int(row.n_test_positive)}" for year, row in counts.iterrows()])
        ax.set_xlabel("Year / n / failed")
    handles = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], linestyle="", label=LABELS[m]) for m in ANNUAL]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .855), ncol=3, frameon=False, columnspacing=1.5)
    fig.text(.055, .033, "Annual points are discrete comparisons; four years do not establish a longer trend.\nThe failed-class sample count varies by year. Codes are keyed in README.", fontsize=9)
    CHECKS["annual_temporal_points"] = 4
    save(fig, "phase2", "phase2-annual-test-comparison", annual, "How do prespecified strategies differ within each of four Test years?", "faceted grouped dot and SD interval", "Display annual snapshots and failed-class denominators without interpolating a long-term trend.", "Finer Test timestamps are unavailable in this run; do not add synthetic monthly points or older protocols.", list(annual.columns))


def phase3_figures(metrics):
    chosen = metrics[metrics.period.eq("pooled") & metrics.method.isin(FIXED)].copy()
    reference = chosen[chosen.fs.eq("no_fs")][["seed", "method", *METRICS]].rename(columns={m: f"{m}_no_fs" for m in METRICS})
    paired = chosen[chosen.fs.ne("no_fs")].merge(reference, on=["seed", "method"], validate="many_to_one")
    for metric in METRICS:
        paired[f"{metric}_delta"] = paired[metric] - paired[f"{metric}_no_fs"]
    official = pd.read_csv(RUN / "fs_paired_differences.csv")
    comparison = paired.merge(official[["seed", "fs", "method", "period", *[m + "_delta" for m in METRICS]]], on=["seed", "fs", "method", "period"], suffixes=("", "_official"), validate="one_to_one")
    if len(paired) != 400 or len(comparison) != 400:
        raise ValueError("Incomplete paired FS comparison")
    for metric in METRICS:
        np.testing.assert_allclose(comparison[metric + "_delta"], comparison[metric + "_delta_official"], rtol=1e-10, atol=1e-12)
    paired["fs_code"] = paired.fs.map(lambda v: FS_KEY[v][0])
    means = paired.groupby(["fs", "method"])[[m + "_delta" for m in METRICS]].mean()
    bound = max(.01, np.ceil(np.abs(means.to_numpy()).max() * 100) / 100)
    cmap = LinearSegmentedColormap.from_list("gold_zero_blue", [GOLD, "#FFFFFF", BLUE])
    for metric in METRICS:
        matrix = means[metric + "_delta"].unstack("method").reindex(index=list(FS_KEY), columns=FIXED)
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        header(fig, f"Phase 3: paired change in {METRICS[metric]}", "FS minus no FS | Same method, seed and Test rows | Mean difference of 10 seeds")
        fig.subplots_adjust(left=.17, right=.87, top=.80, bottom=.16)
        im = ax.imshow(matrix.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
        ax.set_xticks(range(len(FIXED)), [LABELS[m] for m in FIXED])
        ax.set_yticks(range(len(FS_KEY)), [f"{v[0]} ({v[2]})" for v in FS_KEY.values()])
        ax.set_xlabel("Fixed ensemble method"); ax.set_ylabel("FS code (features retained)")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix.iloc[i, j]
                label = "0.000" if round(value, 3) == 0 else f"{value:+.3f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=10, color="white" if value > bound * .65 else "#20262B")
        bar = fig.colorbar(im, ax=ax, fraction=.045, pad=.035)
        bar.set_ticks(np.linspace(-bound, bound, 5))
        bar.set_label("Score difference")
        fig.text(.055, .035, "Blue: higher score; gold: lower score. Shared color limits across the three metric panels.\nCodes are keyed in README. Paired seed differences do not establish statistical significance.", fontsize=9)
        save(fig, "phase3", f"phase3-fs-paired-{metric.lower()}", paired,
             f"What is the same-seed change in {METRICS[metric]} after feature selection?", "annotated paired-difference heatmap",
             "Describe positive, negative and null mean differences for fixed methods without a superiority claim.",
             "Each cell averages ten paired seed differences; training and threshold fitting are repeated per FS setting. V-static policy is excluded.", list(paired.columns))
    CHECKS.update(fs_paired_rows=400, fs_delta_cells_per_metric=40, fs_differences_match_official=True, fs_heatmap_shared_absolute_limit=float(bound))

    policy = metrics[metrics.period.eq("pooled") & metrics.method.eq("ValidationSelectedStatic")].copy()
    order = ["no_fs", *FS_KEY]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 5.4), sharey=True)
    header(fig, "Phase 3: validation-selected static policy", "Same Test | Static composition may change by FS setting | Mean +/- SD of 10 seeds")
    fig.subplots_adjust(left=.17, right=.98, top=.80, bottom=.21, wspace=.22)
    labels = ["No FS (18)"] + [f"{v[0]} ({v[2]})" for v in FS_KEY.values()]
    for ax, metric in zip(axes, METRICS):
        s = policy.groupby("fs")[metric].agg(["mean", "std"]).reindex(order)
        y = np.arange(len(order))
        ax.errorbar(s["mean"], y, xerr=s["std"], fmt="o", color=BLUE, markersize=4, capsize=2, linewidth=1.1)
        ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(*SCORE_DOMAINS[metric]); ax.set_xticks(SCORE_TICKS[metric])
        ax.set_title(METRICS[metric], fontsize=10); ax.set_xlabel("Score")
        axes_style(ax); ax.grid(axis="x", color="#E4E6E8", linewidth=.6)
    fig.text(.055, .035, "Selection uses Calibration AP and threshold uses Calibration F1 for every setting.\nThis compares the whole selection policy, not an isolated feature-selection effect.", fontsize=9)
    save(fig, "phase3", "phase3-fs-selected-static-policy", policy,
         "How does the validation-selected static policy perform across all FS configurations?", "faceted dot and SD interval",
         "Show all prespecified FS configurations, keeping policy comparisons separate from fixed-method deltas.",
         "No FS is an analytical comparator in result charts, not an extra Phase 3 methodology-diagram node.", list(policy.columns))


def write_metadata(baselines, manifest):
    run_ref = rel(RUN)
    report = {
        "scope": "bankruptcy Phase 1-3 only", "run": run_ref,
        "formats": ["PNG 300 dpi", "SVG with editable text"],
        "metrics": list(METRICS.values()), "primary_metric": "AUC", "positive_class": "failed",
        "generator": {"path": rel(Path(__file__)), "sha256": sha256(__file__)},
        "source_manifest_config": manifest["config"], "sources": SOURCES,
        "checks": CHECKS, "charts": CHARTS,
        "limitations": [
            "Phase 1 is historical and does not share the fixed-year validation/tuning protocol of Phase 2-3.",
            "Historical baseline artifacts do not independently establish the best classifier or complete training provenance.",
            "No same-protocol Retrain run is added to Phase 2-3.",
            "Seed SD is descriptive run variability, not independent-data confidence or statistical significance.",
            "Final company status is the retrospective target, not a verified one-year-ahead bankruptcy event target.",
            "Result plotting does not modify data, original results, training settings, or methodology diagrams.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chart-contracts.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    visual_status = "pending"
    review_path = OUT / "visual-review.json"
    if review_path.exists():
        review = json.loads(review_path.read_text(encoding="utf-8"))
        reviewed = review.get("png_sha256", {})
        current = [path for chart in CHARTS for path in chart["exports"] if path.endswith(".png")]
        if review.get("status") == "passed" and len(current) == len(reviewed) and all(reviewed.get(path) == sha256(ROOT / path) for path in current):
            visual_status = "passed_for_matching_png_exports"
    (OUT / "quality-checks.json").write_text(json.dumps({"status": "numeric_and_layout_checks_passed", "checks": CHECKS, "chart_count": len(CHARTS), "export_count": len(CHARTS) * 2, "visual_review": visual_status}, indent=2) + "\n", encoding="utf-8")
    baselines.to_csv(OUT / "phase1/baseline-source-grid.csv", index=False, float_format="%.12g")
    lines = [
        "# 破產預測實驗結果圖", "",
        "本目錄收錄 Phase 1–3 的十張結果圖，各有 PNG、SVG 與作圖 CSV；同名檔直接覆寫，不另存版本。研究方法流程圖仍位於 `assets/research/diagrams`。", "",
        "AUC 為主要指標，F1-score／Recall 以 failed 為正類。圖中延續概括標示；具体算法、設定代碼與來源如下。這些是結果圖，不新增訓練或成效實驗。", "",
        "## 圖檔", "", "| 階段 | 圖 | 用途 |", "|---|---|---|",
    ]
    for chart in CHARTS:
        base = f"{chart['phase']}/{chart['id']}"
        lines.append(f"| {chart['phase']} | [PNG]({base}.png)／[SVG]({base}.svg)／[CSV]({base}.csv) | {chart['analytical_question']} |")
    lines += [
        "", "## Phase 1：歷史切割敏感度", "",
        "固定使用 hybrid 採樣，不按 Test 選最佳採樣。Old／New 各有十五組邊界；Old-window end year 是切割參數，不能把折線當成部署年度的成效趨勢。Retrain 每個候選與採樣僅有一筆全資料結果，以固定水平參考線呈現，不複製成十五次獨立實驗。",
        "", "候選來源依已盤點且完整涵蓋十五組邊界的既有檔案固定指定，未依 Test 排名挑選版本。各檔調參與執行來源不同，圖面只作描述性比較，不能據此宣稱某分類器已被驗證為最佳。舊 CSV 的獨立 Finetune 分支未混入使用者所定義的 New／Fine-tune。", "",
        "| 圖中代碼 | 類別 | 實作標籤 | 固定來源 |", "|---|---|---|---|",
    ]
    for code, family, algorithm, filename in CANDIDATES:
        lines.append(f"| {code} | {family} | {algorithm} | `{filename}` |")
    lines += [
        "", "完整三策略×四採樣來源另存 [baseline-source-grid.csv](phase1/baseline-source-grid.csv)，含所有原始列定位及調參欄位。三張折線主圖僅使用 hybrid 固定切片。", "",
        "## Phase 2：同協定集成比較", "",
        f"來源固定為 `{run_ref}`，completed 且有既存獨立 audit。作圖時重新核對 90 份配置 metrics 的 checksum，並重建總 metrics；本次不重訓模型或重跑全套機率 audit。",
        "", "Old-fit 1999–2011、New-fit 2012–2013、DSEL／Calibration 為 2014 年互斥子集、Test 2015–2018。Test 共有 12,282 列，其中 287 列 failed。",
        "", "- pooled 點圖：八個預先指定方法，同一 Test、十 seeds 的平均值±標準差。",
        "- 模型數折線：先在每個 seed 內平均該模型數的所有靜態混合子集指標，再跨十 seeds 計算平均值與標準差。每個 seed 的 2／3／4／5／6 模型子集數為 9／18／15／6／1；不是挑 Test 最佳組合，也不是把各子集再次組成一個新模型。",
        "- 年度點圖：只有四個測試年度，已核對來源沒有更細時間粒度，故用離散點比較，不補造月資料或外插年度趨勢。2015／2016／2017／2018 的 failed 列數為 111／83／57／36，年樣本數顯示於圖中。",
        "", "| 圖中名稱 | 原始 method | 說明 |", "|---|---|---|",
        "| Old-only／New-only／Equal-six | Old3／New3／Equal6 | 群組或完整六模型等權集成 |",
        "| V-static | ValidationSelectedStatic | Calibration AP 選靜態候選；Calibration F1 選門檻 |",
        "| DCS-1／DCS-2 | DCS_OLA／DCS_LCA | 官方 OLA／LCA |",
        "| DES-1／DES-2 | KNORA_E／KNORA_U | 官方 KNORA-E／KNORA-U |",
        "", "## Phase 3：固定方法差值與選模政策分開", "",
        "三張熱圖顯示 FS 減 no-FS 的同 seed、同方法、同 Test 差值，取十 seeds 平均；以 Equal-six、DCS-1／2、DES-1／2 五個固定方法比較。重新計算的 400 筆 paired rows 與正式差值 CSV 一致。三圖使用相同、以零為中心的色階；正值代表指標較高，負值代表較低，不能當成統計顯著性。FS 每組重新訓練及依驗證資料重新固定門檻，所以差值反映整個 FS 訓練流程，不是獨立的特徵刪減因果效應。",
        "", "另有一張 V-static 點圖，候選組合可隨 FS 變動，故是整體選模政策比較，不混入固定方法的純配對圖。No FS 僅為結果分析對照，不加入第三階段方法流程圖。", "",
        "| 圖中代碼 | 原始 fs | 實作 | 特徵數 |", "|---|---|---|---|",
    ]
    for tag, (code, method, count) in FS_KEY.items():
        lines.append(f"| {code} | {tag} | {method} | {count} |")
    lines += [
        "", "SHAP 是模型解釋後的特徵排序，不能僅因方法主圖概括為 Filter／Wrapper／Embedded，就改稱純 Embedded。具体算法保留於本對照表。", "",
        "## 使用界線與重建", "",
        "- Phase 1 與最新 Phase 2／3 分開呈現；不能把三階段分數差直接當成集成或 FS 的改善量。",
        "- 誤差線是 seeds 的樣本標準差，不是資料獨立重抽樣的信賴區間。相同 Test 列與相關的模型不能視為獨立觀測。",
        "- 沒有補作同協定 Retrain、選最佳分類器或統計顯著性檢定。",
        "- 原始公司最終 status 標籤不等於已驗證的提前一年破產事件預測。",
        "- SVG 保留文字與向量；PNG 為 300 dpi。圖寬 7.2 英吋，縮至 180 mm 時約為原尺寸 98%，仍須依論文版心檢查。",
        "- 絕對指標圖統一使用 AUC 0.45–1、F1-score 0–0.35、Recall 0–1 的明示刻度，各指標在所有圖保持相同範圍。熱圖使用差值刻度，不能與絕對分數混讀。",
        "", "重建入口：`scripts/plots/build_bankruptcy_experiment_figures.py`；從專案根目錄執行：", "",
        "```powershell", ".venv/Scripts/python.exe scripts/plots/build_bankruptcy_experiment_figures.py", "```", "",
        "[圖表契約、來源雜湊與限制](chart-contracts.json)；[數值與版面檢查](quality-checks.json)；[視覺檢視紀錄](visual-review.json)。",
        "", "## 證據帳", "", "| 主張 | 證據 | 狀態 |", "|---|---|---|",
        "| 六個歷史候選含完整三策略×四採樣網格 | 原 CSV、盤點 SHA-256、source-grid | 檔案身份與覆蓋已核對；歷史執行未重建 |",
        "| 固定年份圖來源涵蓋 90 配置與 28,350 列指標 | manifest、配置 checksum、重建 metrics | 本次來源與作圖核對通過 |",
        "| 固定方法 FS 差值可由同 seed 原指標重算 | 400 paired rows、正式差值 CSV | 三項指標重算一致 |",
        "| 圖面不足以證明最佳分類器、跨階段效益或顯著性 | 不同協定／seed 相依性／未補執行 | 明確限定，仍需補同協定證據 |",
    ]
    (OUT / "README.md").write_text("\n".join(lines).replace("具体", "具體").replace("来源", "來源") + "\n", encoding="utf-8")


def main():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "svg.fonttype": "none", "savefig.dpi": 300,
    })
    baselines = load_baselines()
    metrics, manifest = load_standard()
    phase1_figures(baselines)
    phase2_figures(metrics)
    phase3_figures(metrics)
    write_metadata(baselines, manifest)
    print(json.dumps({"output": rel(OUT), "charts": len(CHARTS), "formats": ["PNG", "SVG", "CSV"], "checks": CHECKS}, ensure_ascii=True))


if __name__ == "__main__":
    main()
