"""Export independent Excel-readable CSVs following Study 1/2 conventions."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

METRICS = ("AUC", "PR_AUC", "F1", "G_Mean", "Recall", "Precision")
SAMPLING = {"hybrid": "hybrid", "none": "none", "over": "oversampling", "under": "undersampling"}
ORDER = list(SAMPLING.values())


def write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.10g")


def export(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    seeds = [s for s in manifest["config"]["seeds"] if (run / f"seed_{s}/complete.json").exists()]
    if not seeds:
        raise ValueError("No completed seeds")
    summaries, annual, sources = [], [], {}
    for seed in seeds:
        for name in ("summary.csv", "by_year.csv"):
            source = run / f"seed_{seed}/{name}"
            sources[str(source.relative_to(run))] = hashlib.sha256(source.read_bytes()).hexdigest()
        summaries.append(pd.read_csv(run / f"seed_{seed}/summary.csv"))
        annual.append(pd.read_csv(run / f"seed_{seed}/by_year.csv"))
    summary, yearly = pd.concat(summaries, ignore_index=True), pd.concat(annual, ignore_index=True)
    method_set = set(summary.method)
    expected_years = set(manifest["config"]["test_years"])
    for seed in seeds:
        if set(summary.loc[summary.seed == seed, "method"]) != method_set:
            raise ValueError("Unequal method coverage")
        subset = yearly.loc[yearly.seed == seed]
        if subset.duplicated(["method", "test_feature_year"]).any():
            raise ValueError("Duplicate method/year")
        if any(set(g.test_feature_year) != expected_years for _, g in subset.groupby("method")):
            raise ValueError("Unequal year coverage")
    averages = summary.groupby("method").mean(numeric_only=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = run / "manual_spreadsheets" / stamp
    output.mkdir(parents=True, exist_ok=False)
    mapping, covered = [], []

    def details(folder, prefix, methods):
        methods = list(methods)
        covered.extend(methods)
        selected = summary.loc[summary.method.isin(methods)]
        pool = selected[["method", "seed", "n_predictions", *METRICS]]
        write(pool.sort_values(["method", "seed"]), folder / f"{prefix}_pooled_by_seed.csv")
        raw = yearly.loc[yearly.method.isin(methods)].copy()
        # Study 1/2 split rows become explicit annual rolling origins here.
        raw.insert(0, "split", "test_" + raw.test_feature_year.astype(str))
        raw.insert(1, "dataset_name", "Bankruptcy_US")
        raw.insert(2, "newest_train_window", (raw.test_feature_year - manifest["config"]["window_years"] - 1).astype(str)
                   + "~" + (raw.test_feature_year - 2).astype(str))
        raw.insert(3, "validation_window", raw.test_feature_year - 1)
        raw.insert(4, "test_window", raw.test_feature_year)
        raw = raw.sort_values(["test_feature_year", "method", "seed"])
        write(raw, folder / f"{prefix}_by_year_raw.csv")
        for method in methods:
            mapping.append({"method": method, "folder": str(folder.relative_to(output)), "file_prefix": prefix})
        return raw

    def standard_table(folder, prefix, methods):
        """One independent result family, with raw, pooled, annual and metric-only files."""
        raw = details(folder, prefix, methods)
        for label, source_prefix in (("pooled", ""), ("annual_mean", "annual_mean_")):
            frame = averages.loc[sorted(methods), [source_prefix + m for m in METRICS]].copy()
            frame.columns = METRICS
            frame = frame.reset_index()
            write(frame, folder / f"{prefix}_{label}_summary.csv")
        year_means = raw.groupby(["split", "method"], sort=True)[list(METRICS)].mean().reset_index()
        write(year_means, folder / f"{prefix}_thesis_format.csv")
        for metric in METRICS:
            wide = year_means.pivot(index="split", columns="method", values=metric).reset_index()
            write(wide, folder / f"{prefix}_{metric}_only.csv")

    def sampling_table(folder, prefix, classifier, temporal=False):
        model = "FIFO" if temporal else "Newest"
        methods = [f"{model}_{classifier}_{sample}" for sample in SAMPLING]
        raw = details(folder, prefix, methods)
        rows, annual_rows = [], []
        for metric in METRICS:
            row = {"metric": metric, "method": model}
            annual_row = {"metric": metric, "method": model}
            for sampling, label in SAMPLING.items():
                method = f"{model}_{classifier}_{sampling}"
                row[label] = averages.loc[method, metric]
                annual_row[label] = averages.loc[method, "annual_mean_" + metric]
            rows.append(row)
            annual_rows.append(annual_row)
            write(pd.DataFrame([row]).drop(columns="metric"), folder / f"{prefix}_compact_{metric}_only.csv")
            for sample, label in SAMPLING.items():
                method = f"{model}_{classifier}_{sample}"
                expected = summary.loc[summary.method == method, metric].mean()
                if not np.isclose(row[label], expected, atol=1e-13):
                    raise AssertionError("Sampling pivot changed a metric")
        write(pd.DataFrame(rows)[["metric", "method", *ORDER]], folder / f"{prefix}_compact_summary.csv")
        write(pd.DataFrame(annual_rows)[["metric", "method", *ORDER]], folder / f"{prefix}_annual_mean_summary.csv")
        for sampling, label in SAMPLING.items():
            method = f"{model}_{classifier}_{sampling}"
            write(raw.loc[raw.method == method], folder / f"{prefix}_{label}_by_year.csv")

    for classifier in manifest["config"]["classifiers"]:
        sampling_table(output / "01_baseline" / classifier, f"bk_{classifier}", classifier)
        sampling_table(output / "02_ensemble" / "fifo" / classifier, f"bk_{classifier}_fifo", classifier, True)
    families = {
        "sampling": "sampling_newest", "sampling_temporal": "sampling_fifo",
        "classifier": "classifier_newest", "classifier_temporal": "classifier_fifo",
        "combined": "combined_newest", "combined_temporal": "combined_fifo",
    }
    for family, name in families.items():
        methods = sorted(set(yearly.loc[(yearly.stage == "ensemble") & (yearly.family == family), "method"]))
        if methods:
            standard_table(output / "02_ensemble" / name, f"bk_{name}", methods)
    standard_table(output / "02_ensemble" / "validation_selected", "bk_validation_selected_ensemble", ["ValidationSelectedEnsemble"])
    for fs in ("none", *manifest["config"]["fs_methods"]):
        methods = sorted(m for m in method_set if m.startswith(f"SelectedEnsemble_{fs}_"))
        standard_table(output / "03_fs" / fs, f"bk_{fs.upper()}_static", methods)
    standard_table(output / "03_fs" / "validation_selected", "bk_validation_selected_fs", ["ValidationSelectedFS"])
    if len(covered) != len(method_set) or set(covered) != method_set:
        raise AssertionError("Each method must appear in exactly one result group")
    write(pd.DataFrame(mapping), output / "file_index.csv")
    readme = f"""# Study 1–2 形式的獨立結果試算表

納入 seeds：{', '.join(map(str, seeds))}（{len(seeds)}/{len(manifest['config']['seeds'])} 完成）。
全部檔案為 UTF-8 BOM CSV，可直接用 Excel 開啟；不是把所有結果塞入一張表。

- `01_baseline/<分類器>/`：每個分類器獨立檔案，四採樣並排，沿用 `bk_*_compact_summary.csv` 與 `compact_AUC_only` 等命名。
- `02_ensemble/fifo/<分類器>/`：各分類器 FIFO 獨立檔案，四採樣並排。
- `02_ensemble/<集成方式>/`：跨採樣、跨分類器及合併集成分別存檔。
- `03_fs/<FS方法>/`：MI、CART、RFE、SHAP 分開存檔，50%／80% 保留率列為不同配置；none 為獨立對照。

`compact_summary`／`pooled_summary` 是先對每個 seed 合併 Test 計算，再對已完成 seeds 取平均。
`annual_mean_summary` 是年度等權平均，另檔保存。`pooled_by_seed` 保存各 seed 原值。
`by_year_raw` 保存完整逐年原值；每個採樣另有獨立 `*_by_year.csv`。
集成與 FS 的 `thesis_format` 為各 Test 年度的 seed 平均，`AUC_only` 等為年度×配置寬表。
所有結果保留 AUC、AP（欄名 PR_AUC）、F1、G-Mean、Recall、Precision，不挑 Test 最佳列。

資料源：`{run}`。原始預測與既有論文快照不變；本次僅另行匯出試算表。
FS 基礎集成依年度／seed 的 Validation 選擇不同，查 `by_year_raw` 的 `selected_config` 與 `members`。
檔案總覽見 `file_index.csv`。本匯出為固定快照，不把未完成 seeds 混入。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    audit = {"completed_seeds": seeds, "methods": len(method_set), "source_sha256": sources,
             "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "unique_method_partition": True, "sampling_pivots_verified": True,
             "csv_files": len(list(output.rglob("*.csv")))}
    (output / "export_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    archive = output.with_suffix(".zip")
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, Path("study3b_spreadsheets") / path.relative_to(output))
    print(json.dumps({"output": str(output), "zip": str(archive), **audit}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    export(args.run.resolve())
