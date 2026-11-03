"""Three plain result tables for a human readout of completed pipeline seeds."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MODELS = {"lr": "LR", "rf": "Random Forest", "xgb": "XGBoost", "lgb": "LightGBM", "mlp": "MLP", "svm": "SVM"}
SAMPLES = {"none": "不採樣", "under": "under", "over": "over", "hybrid": "hybrid"}
METRICS = ["AUC", "PR_AUC", "F1", "Recall", "Precision"]


def export(run):
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    seeds = [s for s in manifest["config"]["seeds"] if (run / f"seed_{s}/complete.json").exists()]
    if not seeds:
        raise ValueError("No completed seeds")
    source = pd.concat([pd.read_csv(run / f"seed_{s}/summary.csv") for s in seeds], ignore_index=True)
    means = source.groupby("method")[METRICS].mean()
    output = run / "readout_tables" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    tables = {"01_單模型_分類器與四種採樣.csv": [], "02_全部集成結果.csv": [], "03_FS前後比較.csv": []}
    for model, label in MODELS.items():
        for sample, title in SAMPLES.items():
            method = f"Newest_{model}_{sample}"
            tables["01_單模型_分類器與四種採樣.csv"].append({"分類器": label, "採樣": title, **means.loc[method].to_dict()})
    for method, row in means.iterrows():
        if method.startswith(("FIFO_", "Sampling", "Classifier", "Combined")):
            tables["02_全部集成結果.csv"].append({"配置": method, **row.to_dict()})
    for fs, title in (("none", "不做FS"), ("mi", "MI"), ("cart", "CART"), ("rfe", "RFE"), ("shap", "SHAP")):
        for ratio in ([100] if fs == "none" else [50, 80]):
            tables["03_FS前後比較.csv"].append({"FS方法": title, "保留百分比": ratio, "特徵數": int(18 * ratio / 100),
                                               **means.loc[f"SelectedEnsemble_{fs}_r{ratio}"].to_dict()})
    assert [len(rows) for rows in tables.values()] == [24, 46, 9]
    for filename, rows in tables.items():
        pd.DataFrame(rows).rename(columns={"PR_AUC": "AP"}).to_csv(output / filename, index=False,
                                                                 encoding="utf-8-sig", float_format="%.10g")
    choices = []
    for seed in seeds:
        for path in sorted((run / f"seed_{seed}").glob("year_*/ensemble_selection.json")):
            choice = json.loads(path.read_text(encoding="utf-8"))
            choices.append({"seed": seed, "test_year": int(path.parent.name.split("_")[1]),
                            "送入FS的集成": choice["selected"], "selection_metric": "Validation AP"})
    pd.DataFrame(choices).to_csv(output / "04_實際送入FS的逐年配置.csv", index=False, encoding="utf-8-sig")
    selected = source[source.method.isin(["ValidationSelectedEnsemble", "ValidationSelectedFS"])]
    selected[["seed", "method", *METRICS, "annual_mean_PR_AUC"]].to_csv(
        output / "05_年度選擇策略_各seed.csv", index=False, encoding="utf-8-sig")
    (output / "README.md").write_text(
        f"# 本次三階段成果\n\n已完成 seeds：{seeds}。所有表為 pooled 指標的 seed 平均，非年度平均。\n\n"
        "01 為六分類器×四採樣；02 為全部46個集成；03 為年度Validation所選集成的FS比較。\n"
        "03 的基礎集成可能逐年、逐seed改變，實際配置見04；不能將其當作固定RF或固定XGB的FS結果。\n"
        "05保留逐年選擇策略的各seed pooled與年度平均AP，避免只看平均掩蓋波動。\n"
        "目前實驗尚未全部完成，不用Test排名更換候選或宣稱最優。\n", encoding="utf-8")
    print("OUTPUT", output)
    print(selected[["seed", "method", "AUC", "PR_AUC", "annual_mean_PR_AUC", "F1"]].to_string(index=False))
    print("CHOICES", pd.DataFrame(choices).groupby("送入FS的集成").size().to_dict())
    print("SELECTED_MEAN", means.loc[["ValidationSelectedEnsemble", "ValidationSelectedFS"]].to_string())
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    export(args.run.resolve())
