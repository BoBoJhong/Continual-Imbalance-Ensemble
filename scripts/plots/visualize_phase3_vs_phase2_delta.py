import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
P1_RAW_PATH = ROOT / "results/phase1_baseline/xgb/rerun_20260406_gridplus_n24/tuned/bankruptcy_year_splits_xgb_raw.csv"
P2_STATIC_PATH = ROOT / "results/phase2_ensemble/static/xgb_oldnew_ensemble_static_by_sampling_raw_bankruptcy.csv"
P2_DES_PATH = ROOT / "results/phase2_ensemble/dynamic/des/tuned_des_rerun_20260406/xgb_oldnew_ensemble_des_thesis_format_bankruptcy.csv"
P2_DCS_PATH = ROOT / "results/phase2_ensemble/dynamic/dcs/xgb_oldnew_ensemble_dcs_thesis_format_bankruptcy.csv"
P3_ADV_PATH = ROOT / "results/phase3_feature/xgb_bankruptcy_fs_advanced_raw.csv"
P3_DES_PATH = ROOT / "results/phase3_feature/xgb_bankruptcy_fs_full_des.csv"
OUT_PATH = ROOT / "results/phase3_feature/plots/xgb_phase3_vs_phase2_delta.png"

SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")


def parse_split_order(split_id: str):
    m = SPLIT_RE.match(str(split_id).strip())
    if not m:
        return (99, 99)
    return (int(m.group(1)), int(m.group(2)))


def p1_val_selected(df: pd.DataFrame, metrics):
    selected = df.dropna(subset=["tune_val_auc"]).copy()
    idx = selected.groupby("split", sort=False)["tune_val_auc"].idxmax()
    return selected.loc[idx, ["split"] + metrics]


def build_delta_plot():
    metrics = ["AUC", "F1"]

    p1_raw = pd.read_csv(P1_RAW_PATH)
    p1_val = p1_val_selected(p1_raw, metrics)
    p1_val = p1_val.rename(columns={m: f"{m}_p1_val" for m in metrics})

    p2_static = pd.read_csv(P2_STATIC_PATH)
    p2_static = p2_static[
        (p2_static["ensemble"] == "6models") & (p2_static["type"] == "k_subsets_mean")
    ][["split"] + metrics]
    p2_static = p2_static.rename(columns={m: f"{m}_p2_static" for m in metrics})

    p2_des = pd.read_csv(P2_DES_PATH).rename(columns={"split_id": "split"})[["split"] + metrics]
    p2_des = p2_des.rename(columns={m: f"{m}_p2_des" for m in metrics})

    p2_dcs = pd.read_csv(P2_DCS_PATH).rename(columns={"split_id": "split"})[["split"] + metrics]
    p2_dcs = p2_dcs.rename(columns={m: f"{m}_p2_dcs" for m in metrics})

    p3_adv = pd.read_csv(P3_ADV_PATH).groupby("split", as_index=False)[metrics].max()
    p3_adv = p3_adv.rename(columns={m: f"{m}_p3_adv" for m in metrics})

    p3_dyn = pd.read_csv(P3_DES_PATH).groupby("split", as_index=False)[metrics].max()
    p3_dyn = p3_dyn.rename(columns={m: f"{m}_p3_dyn" for m in metrics})

    df = p3_adv.merge(p3_dyn, on="split", how="outer")
    df = (
        df.merge(p2_des, on="split", how="left")
        .merge(p2_dcs, on="split", how="left")
        .merge(p2_static, on="split", how="left")
        .merge(p1_val, on="split", how="left")
    )
    df = df.sort_values("split", key=lambda s: s.map(parse_split_order)).reset_index(drop=True)

    for m in metrics:
        df[f"{m}_adv_vs_p2_des"] = df[f"{m}_p3_adv"] - df[f"{m}_p2_des"]
        df[f"{m}_adv_vs_p2_static"] = df[f"{m}_p3_adv"] - df[f"{m}_p2_static"]
        df[f"{m}_dyn_vs_p2_des"] = df[f"{m}_p3_dyn"] - df[f"{m}_p2_des"]
        df[f"{m}_dyn_vs_p2_dcs"] = df[f"{m}_p3_dyn"] - df[f"{m}_p2_dcs"]
        df[f"{m}_dyn_vs_p2_static"] = df[f"{m}_p3_dyn"] - df[f"{m}_p2_static"]
        df[f"{m}_p2_des_vs_p1"] = df[f"{m}_p2_des"] - df[f"{m}_p1_val"]
        df[f"{m}_p2_dcs_vs_p1"] = df[f"{m}_p2_dcs"] - df[f"{m}_p1_val"]
        df[f"{m}_p2_static_vs_p1"] = df[f"{m}_p2_static"] - df[f"{m}_p1_val"]
        df[f"{m}_adv_vs_p1"] = df[f"{m}_p3_adv"] - df[f"{m}_p1_val"]
        df[f"{m}_dyn_vs_p1"] = df[f"{m}_p3_dyn"] - df[f"{m}_p1_val"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    x = range(len(df))

    colors = {
        "adv_vs_p2_des": "#1F77B4",
        "adv_vs_p2_static": "#17BECF",
        "dyn_vs_p2_des": "#D62728",
        "dyn_vs_p2_dcs": "#8C564B",
        "dyn_vs_p2_static": "#FF9896",
        "p2_des_vs_p1": "#7F7F7F",
        "p2_dcs_vs_p1": "#BCBD22",
        "p2_static_vs_p1": "#1F77B4",
        "adv_vs_p1": "#2CA02C",
        "dyn_vs_p1": "#9467BD",
    }

    for i, m in enumerate(metrics):
        ax = axes[i]
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.plot(x, df[f"{m}_adv_vs_p2_des"], marker="o", linewidth=2, color=colors["adv_vs_p2_des"], label="FS+Advanced - P2 DES")
        ax.plot(x, df[f"{m}_adv_vs_p2_static"], marker="o", linewidth=1.8, linestyle=":", color=colors["adv_vs_p2_static"], label="FS+Advanced - P2 Static")
        ax.plot(x, df[f"{m}_dyn_vs_p2_des"], marker="s", linewidth=2, color=colors["dyn_vs_p2_des"], label="FS+Dynamic - P2 DES")
        ax.plot(x, df[f"{m}_dyn_vs_p2_dcs"], marker="s", linewidth=1.8, linestyle="-.", color=colors["dyn_vs_p2_dcs"], label="FS+Dynamic - P2 DCS")
        ax.plot(x, df[f"{m}_dyn_vs_p2_static"], marker="s", linewidth=1.8, linestyle=":", color=colors["dyn_vs_p2_static"], label="FS+Dynamic - P2 Static")
        ax.plot(x, df[f"{m}_p2_des_vs_p1"], marker="x", linewidth=1.5, linestyle="--", color=colors["p2_des_vs_p1"], label="P2 DES - P1 original (val-selected)")
        ax.plot(x, df[f"{m}_p2_dcs_vs_p1"], marker="x", linewidth=1.4, linestyle="-.", color=colors["p2_dcs_vs_p1"], label="P2 DCS - P1 original (val-selected)")
        ax.plot(x, df[f"{m}_p2_static_vs_p1"], marker="x", linewidth=1.4, linestyle=":", color=colors["p2_static_vs_p1"], label="P2 Static - P1 original (val-selected)")
        ax.plot(x, df[f"{m}_adv_vs_p1"], marker="^", linewidth=1.8, color=colors["adv_vs_p1"], label="FS+Advanced - P1 original")
        ax.plot(x, df[f"{m}_dyn_vs_p1"], marker="v", linewidth=1.8, color=colors["dyn_vs_p1"], label="FS+Dynamic - P1 original")
        ax.set_ylabel(f"Delta {m}", fontweight="bold")
        ax.set_title(f"Phase3 / Phase2 vs P1 Original Delta ({m})", fontweight="bold")
        ax.legend(loc="best", fontsize=9)

    plt.xticks(x, df["split"], rotation=40, ha="right")
    plt.xlabel("Year Splits", fontweight="bold")
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=200)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    build_delta_plot()
