"""
Statistical tests for the current bankruptcy findings.

Unlike scripts/analysis/statistical_test.py, this script does not simulate
per-seed scores from mean/std. It uses existing split-level outputs and runs
paired Wilcoxon signed-rank tests across the same year splits.

Outputs:
    results/statistical_tests/current_findings/bankruptcy_current_findings_wilcoxon.csv
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

OUT_DIR = project_root / "results" / "statistical_tests" / "current_findings"
METRICS = ("AUC", "F1", "Recall")
ALPHA = 0.05


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm adjustment within one pre-specified family of hypotheses."""
    valid = p_values.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    running_max = 0.0
    total = len(valid)
    for rank, (index, p_value) in enumerate(valid.items()):
        running_max = max(running_max, min(1.0, (total - rank) * float(p_value)))
        adjusted.loc[index] = running_max
    return adjusted


def _wilcoxon_pair(
    *,
    values_a: Iterable[float],
    values_b: Iterable[float],
    alternative: str,
) -> float:
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]
    if len(a) == 0:
        return np.nan
    diff = a - b
    if np.allclose(diff, 0.0):
        return 1.0
    return float(wilcoxon(a, b, alternative=alternative, zero_method="wilcox").pvalue)


def _append_test(
    rows: list[dict],
    *,
    family: str,
    comparison: str,
    metric: str,
    method_a: str,
    method_b: str,
    values_a: Iterable[float],
    values_b: Iterable[float],
    alternative: str = "greater",
) -> None:
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]
    if len(a) == 0:
        return

    p_directional = _wilcoxon_pair(values_a=a, values_b=b, alternative=alternative)
    p_two_sided = _wilcoxon_pair(values_a=a, values_b=b, alternative="two-sided")
    rows.append(
        {
            "family": family,
            "comparison": comparison,
            "metric": metric,
            "method_a": method_a,
            "method_b": method_b,
            "alternative": alternative,
            "n_pairs": len(a),
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff_a_minus_b": float(np.mean(a - b)),
            "median_diff_a_minus_b": float(np.median(a - b)),
            "n_a_better": int(np.sum(a > b)),
            "n_b_better": int(np.sum(a < b)),
            "n_equal": int(np.sum(np.isclose(a, b))),
            "p_directional": p_directional,
            "p_two_sided": p_two_sided,
            "significant_directional": bool(p_directional < ALPHA),
            "significant_two_sided": bool(p_two_sided < ALPHA),
        }
    )


def add_phase2_old_new_tests(rows: list[dict]) -> None:
    path = (
        project_root
        / "results"
        / "phase2_ensemble"
        / "static"
        / "xgb_oldnew_ensemble_static_by_sampling_raw_bankruptcy.csv"
    )
    df = pd.read_csv(path)
    base = df[df["ensemble"].isin(["Old", "New", "OldNew"])].copy()

    for sampling_type in ("under", "over", "hybrid"):
        sub = base[base["type"] == sampling_type]
        for metric in METRICS:
            pivot = sub.pivot(index="split", columns="ensemble", values=metric)
            if {"New", "Old"}.issubset(pivot.columns):
                _append_test(
                    rows,
                    family="phase2_static_old_new",
                    comparison=f"{sampling_type}: New > Old",
                    metric=metric,
                    method_a=f"New_{sampling_type}",
                    method_b=f"Old_{sampling_type}",
                    values_a=pivot["New"],
                    values_b=pivot["Old"],
                )
            if {"New", "OldNew"}.issubset(pivot.columns):
                _append_test(
                    rows,
                    family="phase2_static_old_new",
                    comparison=f"{sampling_type}: New > OldNew",
                    metric=metric,
                    method_a=f"New_{sampling_type}",
                    method_b=f"OldNew_{sampling_type}",
                    values_a=pivot["New"],
                    values_b=pivot["OldNew"],
                )


def add_phase3_feature_tests(rows: list[dict]) -> None:
    path = project_root / "results" / "phase3_feature" / "combined" / "xgb_bankruptcy_fs_full_static.csv"
    df = pd.read_csv(path)

    for ensemble in ("New_3", "All_6"):
        sub = df[df["ensemble"] == ensemble]
        for metric in METRICS:
            pivot = sub.pivot(index="split", columns="fs", values=metric)
            if "no_fs" not in pivot.columns:
                continue
            for fs_variant in ("mi_r80", "shap_r80", "rfe_r80"):
                if fs_variant not in pivot.columns:
                    continue
                _append_test(
                    rows,
                    family="phase3_feature_selection",
                    comparison=f"{ensemble}: {fs_variant} > no_fs",
                    metric=metric,
                    method_a=f"{ensemble}_{fs_variant}",
                    method_b=f"{ensemble}_no_fs",
                    values_a=pivot[fs_variant],
                    values_b=pivot["no_fs"],
                )


def add_stability_tests(rows: list[dict]) -> None:
    path = (
        project_root
        / "results"
        / "phase3_feature"
        / "stability"
        / "bankruptcy_feature_stability_pairwise_jaccard.csv"
    )
    df = pd.read_csv(path)

    for (scope, method), group in df.groupby(["train_scope", "fs_method"]):
        pivot = group.pivot(index=["split_a", "split_b"], columns="fs_ratio", values="jaccard")
        if 0.5 not in pivot.columns or 0.8 not in pivot.columns:
            continue
        _append_test(
            rows,
            family="feature_stability",
            comparison=f"{scope}/{method}: r80 > r50 Jaccard",
            metric="Jaccard",
            method_a=f"{scope}_{method}_r80",
            method_b=f"{scope}_{method}_r50",
            values_a=pivot[0.8],
            values_b=pivot[0.5],
        )


def add_dynamic_static_tests(rows: list[dict]) -> None:
    path = (
        project_root
        / "results"
        / "phase2_ensemble"
        / "model_comparison"
        / "plots"
        / "bankruptcy_phase2_xgb_static_des_dcs_points.csv"
    )
    df = pd.read_csv(path)

    for metric in METRICS:
        sub = df[df["metric"] == metric]
        pivot = sub.pivot(index="split", columns="family", values="value")
        for dynamic_family in ("DES", "DCS"):
            if {"Static", dynamic_family}.issubset(pivot.columns):
                _append_test(
                    rows,
                    family="phase2_dynamic_vs_static",
                    comparison=f"{dynamic_family} > Static",
                    metric=metric,
                    method_a=dynamic_family,
                    method_b="Static",
                    values_a=pivot[dynamic_family],
                    values_b=pivot["Static"],
                )
                _append_test(
                    rows,
                    family="phase2_dynamic_vs_static",
                    comparison=f"Static > {dynamic_family}",
                    metric=metric,
                    method_a="Static",
                    method_b=dynamic_family,
                    values_a=pivot["Static"],
                    values_b=pivot[dynamic_family],
                )
        if {"DES", "DCS"}.issubset(pivot.columns):
            _append_test(
                rows,
                family="phase2_dynamic_vs_static",
                comparison="DES > DCS",
                metric=metric,
                method_a="DES",
                method_b="DCS",
                values_a=pivot["DES"],
                values_b=pivot["DCS"],
            )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    add_phase2_old_new_tests(rows)
    add_phase3_feature_tests(rows)
    add_stability_tests(rows)
    add_dynamic_static_tests(rows)

    result = pd.DataFrame(rows).sort_values(
        ["significant_directional", "p_directional", "family", "comparison", "metric"],
        ascending=[False, True, True, True, True],
    )
    result["p_directional_holm"] = result.groupby("family", group_keys=False)[
        "p_directional"
    ].apply(_holm_adjust)
    result["p_two_sided_holm"] = result.groupby("family", group_keys=False)[
        "p_two_sided"
    ].apply(_holm_adjust)
    result["significant_directional_holm"] = result["p_directional_holm"] < ALPHA
    result["significant_two_sided_holm"] = result["p_two_sided_holm"] < ALPHA
    out_path = OUT_DIR / "bankruptcy_current_findings_wilcoxon.csv"
    result.to_csv(out_path, index=False, float_format="%.8f")

    sig = result[result["significant_directional"]]
    print(f"Saved: {out_path}")
    print(f"Total tests: {len(result)}")
    print(f"Directional significant tests (p < {ALPHA}): {len(sig)}")
    print(
        "Directional significant tests after within-family Holm correction "
        f"(p < {ALPHA}): {int(result['significant_directional_holm'].sum())}"
    )
    if not sig.empty:
        print(
            sig[
                [
                    "family",
                    "comparison",
                    "metric",
                    "n_pairs",
                    "mean_a",
                    "mean_b",
                    "p_directional",
                    "p_two_sided",
                ]
            ].head(30).to_string(index=False)
        )


if __name__ == "__main__":
    main()
