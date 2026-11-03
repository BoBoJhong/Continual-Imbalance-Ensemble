"""Draw chapter 4 figures from audited exports, without Test-based selection."""
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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis/latex/figures"
PROOF = ROOT / "thesis/drafts/front_four_revision_20261027/figures-proof.json"
CHAPTER_PROOF = ROOT / "thesis/drafts/framework_results_revision_20261027/results-proof.json"
ANALYSIS = ROOT / "results/statistical_tests/fixed_two_experiments/20261027T051346477136Z"
METRICS = ["AUC", "F1", "Recall"]
METHODS = ["Old3", "New3", "Equal6", "ValidationSelectedStatic", "DCS_OLA", "DCS_LCA", "KNORA_E", "KNORA_U"]
METHOD_LABELS = ["Old3", "New3", "Equal6", "V-static", "DCS-OLA", "DCS-LCA", "KNORA-E", "KNORA-U"]
FS_ORDER = ["mi_50", "mi_80", "cart_50", "cart_80", "shap_50", "shap_80", "rfe_50", "rfe_80"]
FS_LABELS = ["MI 50%", "MI 80%", "CART 50%", "CART 80%", "SHAP 50%", "SHAP 80%", "RFE 50%", "RFE 80%"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Source changed: {path}")


def draw():
    binding = json.loads((ROOT / "config/thesis_sources.json").read_text(encoding="utf-8"))
    require_hash(CHAPTER_PROOF, binding["chapter_results"]["sha256"])
    chapter = json.loads(CHAPTER_PROOF.read_text(encoding="utf-8"))
    if chapter["status"] != "passed":
        raise ValueError("Chapter export is not verified")
    for key in ("source_sha256", "generated_sha256"):
        for name, expected in chapter[key].items():
            require_hash(ROOT / name, expected)
    manifest_path = ANALYSIS / "manifest.json"
    expected = next(x for x in binding["analyses"] if x["id"] == "fixed_company_statistics")
    require_hash(manifest_path, expected["manifest_sha256"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected_hash in manifest["output_sha256"].items():
        require_hash(ANALYSIS / name, expected_hash)

    baseline_path = ROOT / "thesis/latex/generated/chapter-results-20261027/baseline-summary.csv"
    descriptive_path = ANALYSIS / "descriptive.csv"
    paired_path = ANALYSIS / "paired_tests.csv"
    baseline = pd.read_csv(baseline_path)
    descriptive = pd.read_csv(descriptive_path)
    paired = pd.read_csv(paired_path)
    fonts = ROOT / "thesis/latex/fonts"
    fontManager.addfont(str(fonts / "kaiu.ttf"))
    cjk = FontProperties(fname=str(fonts / "kaiu.ttf"))
    plt.rcParams.update({"font.family": cjk.get_name(), "font.size": 12,
                         "axes.unicode_minus": False, "axes.linewidth": .8,
                         "svg.fonttype": "path", "savefig.facecolor": "white"})
    OUT.mkdir(parents=True, exist_ok=True)
    generated = {}

    def save(fig, stem):
        for extension in ("png", "svg"):
            path = OUT / f"{stem}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.12,
                        metadata={"Creator": "Continual-Imbalance-Ensemble"})
            generated[path.relative_to(ROOT).as_posix()] = sha(path)
        plt.close(fig)

    # Preserve the exporter order; it is not a Test-score ranking.
    candidates = list(baseline.candidate.drop_duplicates())
    labels = ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression", "TabM"]
    assert candidates == labels and len(baseline) == 15
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 4.7), sharey=True)
    series = [("Old", "#9B6500", "o", -.19), ("New", "#235A88", "s", 0),
              ("Retrain", "#555555", "^", .19)]
    for ax, metric in zip(axes, METRICS):
        for method, color, marker, offset in series:
            rows = baseline[baseline.method.eq(method)].set_index("candidate").loc[candidates]
            ax.scatter(rows[metric], np.arange(len(candidates)) + offset, color=color, marker=marker,
                       s=39, label=method, edgecolors="white", linewidths=.4, zorder=3)
        ax.set_title("F1-score" if metric == "F1" else metric, fontsize=14)
        ax.set_xlim((.55, .95) if metric == "AUC" else (0, .7 if metric == "Recall" else .35))
        ax.grid(axis="x", color=".88", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(candidates)), labels)
    axes[0].invert_yaxis()
    axes[1].set_xlabel("測試分數（各指標使用各自刻度）", labelpad=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
               bbox_to_anchor=(.6, 1.01), frameon=False)
    fig.subplots_adjust(left=.255, right=.99, wspace=.15, top=.86, bottom=.17)
    save(fig, "results-baseline-strategies")

    no_fs = descriptive[descriptive.fs.eq("no_fs")]
    assert len(no_fs) == 24 and no_fs.n_seeds.eq(10).all()
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 5.6), sharey=True)
    for ax, metric in zip(axes, METRICS):
        d = no_fs[no_fs.metric.eq(metric)].set_index("method").loc[METHODS]
        for y, (method, row) in enumerate(d.iterrows()):
            ax.errorbar(row["mean"], y, xerr=row.seed_sd, fmt="o", markersize=5,
                        color="#235A88", capsize=3, linewidth=1.2, zorder=3)
        ax.set_title("F1-score" if metric == "F1" else metric, fontsize=14)
        ax.set_xlim((.67, .86) if metric == "AUC" else (.08, .29) if metric == "F1" else (.12, .4))
        ax.xaxis.set_major_locator(MultipleLocator(.05))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="x", color=".88", linewidth=.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(8), METHOD_LABELS)
    axes[0].invert_yaxis()
    axes[1].set_xlabel("十個種子的平均值 ± 標準差", labelpad=12)
    fig.subplots_adjust(left=.155, right=.99, wspace=.19, top=.89, bottom=.17)
    save(fig, "results-ensemble-comparison")

    # Require the exact archived conditions rather than infer a label from scores.
    cells = paired[paired.experiment.eq("experiment2") & paired.metric.eq("AUC")]
    mapping = {fs: fs.replace("_", "_r") for fs in FS_ORDER}
    assert set(cells.fs_a) == set(mapping.values())
    matrix = np.empty((8, 6))
    plot_methods = METHODS[2:]
    for i, fs in enumerate(FS_ORDER):
        for j, method in enumerate(plot_methods):
            row = cells[cells.fs_a.eq(mapping[fs]) & cells.method_a.eq(method)]
            assert len(row) == 1 and row.fs_b.eq("no_fs").all()
            matrix[i, j] = row.iloc[0].difference
    limit = max(.05, float(np.ceil(np.abs(matrix).max() * 100) / 100))
    fig, ax = plt.subplots(figsize=(10.2, 6.3))
    im = ax.imshow(matrix, cmap="PuOr", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set_xticks(np.arange(6), METHOD_LABELS[2:])
    ax.set_yticks(np.arange(8), FS_LABELS)
    ax.tick_params(length=0, pad=8)
    for i in range(8):
        for j in range(6):
            ax.text(j, i, f"{matrix[i,j]:+.4f}", ha="center", va="center", fontsize=12,
                    color="white" if abs(matrix[i,j]) > .65 * limit else "#171717")
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("相同集成方法下的 FS - no-FS（十種子平均）", labelpad=15)
    cbar = fig.colorbar(im, ax=ax, fraction=.04, pad=.035)
    cbar.set_label("AUC 差值", labelpad=10)
    fig.subplots_adjust(left=.15, right=.94, top=.97, bottom=.17)
    save(fig, "results-fs-auc-differences")

    source_paths = [Path(__file__), CHAPTER_PROOF, baseline_path, descriptive_path, paired_path, manifest_path]
    proof = {"status": "passed", "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths},
             "generated_sha256": generated,
             "rules": {"baseline": "Fixed exporter order; equal boundary/sampling weights; no error bars",
                       "ensemble": "Pooled four-year scores; mean +/- sample SD across ten seeds, not CI",
                       "fs": "48 matched AUC mean differences; no significance encoded; no Test selection"}}
    PROOF.parent.mkdir(parents=True, exist_ok=True)
    PROOF.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS: three figures (PNG/SVG), 15 tuned Baseline rows, 24 ensemble cells, 48 FS differences")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        proof = json.loads(PROOF.read_text(encoding="utf-8"))
        for key in ("source_sha256", "generated_sha256"):
            for name, expected in proof[key].items():
                require_hash(ROOT / name, expected)
        print("PASS: chapter 4 figure evidence and outputs")
    else:
        draw()


if __name__ == "__main__":
    main()
