"""
Export clean thesis tables from raw result CSVs.
Output directory: results/thesis_tables/
Usage: python scripts/analysis/export_thesis_tables.py
"""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT  = ROOT / "results" / "thesis_tables"
OUT.mkdir(parents=True, exist_ok=True)


def _save(df: pd.DataFrame, path: Path) -> None:
    """Save CSV; skip gracefully if file is locked (e.g. open in Excel)."""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  Saved -> {path.name}")
    except PermissionError:
        print(f"  WARNING: {path.name} is locked (close in Excel?), skipping")


# ─────────────────────────────────────────────
# Study 3-A  ROSS 候選邊界掃描（論文用）
# ─────────────────────────────────────────────
def table_ross_candidates():
    src = ROOT / "results/phase4_drift/bk_validation_ross_candidates.csv"
    df  = pd.read_csv(src)

    keep = {
        "drift_start_year": "Drift Start Year",
        "old_window":        "Old Period",
        "new_window":        "New Period",
        "old_n":             "Old N",
        "new_n":             "New N",
        "old_val_AUC":       "Old Val AUC",
        "new_val_AUC":       "New Val AUC",
        "gap_new_minus_old_val_AUC": "Gap (New-Old)",
        "new_val_F1":        "New Val F1",
        "rank_by_new_val_auc": "AUC Rank",
    }
    out = df[list(keep.keys())].rename(columns=keep)

    for col in ["Old Val AUC", "New Val AUC", "Gap (New-Old)", "New Val F1"]:
        out[col] = out[col].map(lambda x: f"{x:.4f}")

    out = out.sort_values("AUC Rank").reset_index(drop=True)
    out.insert(0, "Selected", out["Drift Start Year"].apply(lambda y: "★" if y == 2009 else ""))

    path = OUT / "study3a_ross_candidates.csv"
    print("[Study3-A] ROSS candidates")
    _save(out, path)
    print(out.to_string(index=False))
    print()
    return out


# ─────────────────────────────────────────────
# Study 3-B  DAWCE 最佳配置摘要（論文用）
# ─────────────────────────────────────────────
def table_dawce_summary():
    src = ROOT / "results/phase5_weighted/bk_validation_ross_weight_summary.csv"
    df  = pd.read_csv(src)

    keep = {
        "method":             "Method",
        "drift_start_year":   "Drift Year",
        "old_window":         "Old Period",
        "new_window":         "New Period",
        "fs_variant":         "FS",
        "w_old":              "w_old",
        "w_new":              "w_new",
        "AUC":                "AUC",
        "F1":                 "F1",
        "Recall":             "Recall",
        "Precision":          "Precision",
        "Type1_Error":        "Type1 Err",
        "Type2_Error":        "Type2 Err",
        "delta_vs_fixed_equal_AUC": "dAUC vs Equal",
        "delta_vs_fixed_equal_F1":  "dF1 vs Equal",
    }
    out = df[list(keep.keys())].rename(columns=keep)

    for col in ["AUC","F1","Recall","Precision","Type1 Err","Type2 Err"]:
        out[col] = out[col].map(lambda x: f"{x:.4f}")
    out["dAUC vs Equal"] = df["delta_vs_fixed_equal_AUC"].map(lambda x: f"{x:+.4f}")
    out["dF1 vs Equal"]  = df["delta_vs_fixed_equal_F1"].map(lambda x: f"{x:+.4f}")
    for col in ["w_old", "w_new"]:
        out[col] = out[col].map(lambda x: f"{x:.2f}")

    f1_vals  = df["F1"].values
    best_idx = f1_vals.argmax()
    out.insert(0, "Best", ["★" if i == best_idx else "" for i in range(len(out))])

    path = OUT / "study3b_dawce_summary.csv"
    print("[Study3-B] DAWCE summary")
    _save(out, path)
    print(out.to_string(index=False))
    print()
    return out


# ─────────────────────────────────────────────
# Study 3-C  多切割 Wilcoxon（論文用）
# ─────────────────────────────────────────────
def table_wilcoxon_weighted():
    src = ROOT / "results/phase5_weighted/bk_year_split_weight_wilcoxon.csv"
    df  = pd.read_csv(src)

    keep = {
        "fs_variant":                          "FS",
        "metric":                              "Metric",
        "n_pairs":                             "N Pairs",
        "selected_mean":                       "Selected Mean",
        "equal_mean":                          "Equal Mean",
        "mean_diff_selected_minus_equal":      "Mean Diff",
        "n_selected_better":                   "N Selected Better",
        "p_two_sided":                         "p (two-sided)",
        "significant_two_sided":               "Significant (p<0.05)",
    }
    out = df[list(keep.keys())].rename(columns=keep)

    for col in ["Selected Mean", "Equal Mean", "Mean Diff"]:
        out[col] = out[col].map(lambda x: f"{x:.4f}")
    out["p (two-sided)"] = out["p (two-sided)"].map(lambda x: f"{x:.6f}")
    out["Mean Diff"] = df["mean_diff_selected_minus_equal"].map(lambda x: f"{x:+.4f}")

    path = OUT / "study3c_wilcoxon_weighted.csv"
    print("[Study3-C] Wilcoxon weighted ensemble")
    _save(out, path)
    print(out.to_string(index=False))
    print()
    return out


# ─────────────────────────────────────────────
# Study 3-D  漂移偵測器對照（論文用）
# ─────────────────────────────────────────────
def table_drift_detector_compare():
    src_pts  = ROOT / "results/phase4_drift/bk_drift_detection_points.csv"
    src_traj = ROOT / "results/phase4_drift/bk_drift_auc_signal_detection.csv"
    src_cmp  = ROOT / "results/phase4_drift/bk_drift_auc_signal_comparison.csv"

    df_pts = pd.read_csv(src_pts)
    out_pts = df_pts.copy()
    out_pts.columns = ["Detector", "Drift Year Detected"]
    path_pts = OUT / "study3d_detector_trigger.csv"
    print("[Study3-D] Detector trigger points")
    _save(out_pts, path_pts)
    print(out_pts.to_string(index=False))
    print()

    df_traj  = pd.read_csv(src_traj)
    out_traj = df_traj[["year","n","AUC","signal_1_minus_AUC"]].copy()
    out_traj.columns = ["Year", "N", "AUC", "1-AUC (Signal)"]
    out_traj["AUC"] = out_traj["AUC"].map(lambda x: f"{x:.4f}")
    out_traj["1-AUC (Signal)"] = out_traj["1-AUC (Signal)"].map(lambda x: f"{x:.4f}")
    out_traj.insert(0, "Burn-in Ref (1-AUC)", ["0.3904" if i == 0 else "" for i in range(len(out_traj))])

    path_traj = OUT / "study3d_auc_signal_trajectory.csv"
    print("[Study3-D] AUC signal trajectory")
    _save(out_traj, path_traj)
    print(out_traj.to_string(index=False))
    print()

    df_cmp = pd.read_csv(src_cmp)
    out_cmp = df_cmp[["boundary_method","ensemble","AUC","F1","Recall","Precision"]].copy()
    out_cmp.columns = ["Boundary Method", "Ensemble", "AUC", "F1", "Recall", "Precision"]
    for col in ["AUC","F1","Recall","Precision"]:
        out_cmp[col] = out_cmp[col].map(lambda x: f"{x:.4f}")
    out_cmp_best = out_cmp.sort_values("F1", ascending=False).groupby("Boundary Method").first().reset_index()
    path_cmp = OUT / "study3d_boundary_comparison.csv"
    print("[Study3-D] Boundary comparison (best per method)")
    _save(out_cmp_best, path_cmp)
    print(out_cmp_best.to_string(index=False))
    print()


# ─────────────────────────────────────────────
# Study 3-E  AWE vs DAWCE vs Equal-weight（論文用）
# ─────────────────────────────────────────────
def table_awe_comparison():
    src = ROOT / "results/phase5_weighted/bk_awe_comparison.csv"
    if not src.exists():
        print(f"[Study3-E] File not found, skipping: {src}")
        return

    df = pd.read_csv(src)
    keep = {
        "config":    "Config",
        "method":    "Method",
        "AUC":       "AUC",
        "F1":        "F1",
        "Recall":    "Recall",
        "Precision": "Precision",
        "w_old":     "w_old",
        "w_new":     "w_new",
    }
    out = df[list(keep.keys())].rename(columns=keep)

    for col in ["AUC", "F1", "Recall", "Precision"]:
        out[col] = out[col].map(lambda x: f"{x:.4f}")
    out["w_old"] = out["w_old"].map(lambda x: f"{x:.3f}")
    out["w_new"] = out["w_new"].map(lambda x: f"{x:.3f}")

    best_mask = df.groupby("config")["F1"].transform("max") == df["F1"]
    out.insert(0, "Best", ["★" if v else "" for v in best_mask])

    path = OUT / "study3e_awe_comparison.csv"
    print("[Study3-E] AWE vs DAWCE comparison")
    _save(out, path)
    print(out.to_string(index=False))
    print()


# ─────────────────────────────────────────────
# Study 3-F  k=1 vs k=2 Multi-boundary ROSS（論文用）
# ─────────────────────────────────────────────
def table_multi_boundary_ross():
    src_k1  = ROOT / "results/phase4_drift/bk_multi_boundary_k1_candidates.csv"
    src_k2  = ROOT / "results/phase4_drift/bk_multi_boundary_k2_candidates.csv"
    src_cmp = ROOT / "results/phase4_drift/bk_multi_boundary_k_comparison.csv"

    for src in [src_k1, src_k2, src_cmp]:
        if not src.exists():
            print(f"[Study3-F] File not found, skipping: {src}")
            return

    # k=1 top 5
    k1     = pd.read_csv(src_k1)
    out_k1 = k1[["b1", "periods", "w1", "w2", "val_F1", "val_AUC", "test_F1", "test_AUC"]].head(5).copy()
    out_k1.columns = ["Boundary (b*)", "Periods", "w_old", "w_new", "Val F1", "Val AUC", "Test F1", "Test AUC"]
    for col in ["Val F1", "Val AUC", "Test F1", "Test AUC"]:
        out_k1[col] = out_k1[col].map(lambda x: f"{x:.4f}")
    for col in ["w_old", "w_new"]:
        out_k1[col] = out_k1[col].map(lambda x: f"{x:.2f}")
    path_k1 = OUT / "study3f_multi_boundary_k1_top5.csv"
    print("[Study3-F] k=1 top 5")
    _save(out_k1, path_k1)
    print(out_k1.to_string(index=False))
    print()

    # k=2 top 5
    k2     = pd.read_csv(src_k2)
    out_k2 = k2[["b1", "b2", "periods", "w1", "w2", "w3",
                  "val_F1", "val_AUC", "test_F1", "test_AUC"]].head(5).copy()
    out_k2.columns = ["b1*", "b2*", "Periods", "w_P1", "w_P2", "w_P3",
                      "Val F1", "Val AUC", "Test F1", "Test AUC"]
    for col in ["Val F1", "Val AUC", "Test F1", "Test AUC"]:
        out_k2[col] = out_k2[col].map(lambda x: f"{x:.4f}")
    for col in ["w_P1", "w_P2", "w_P3"]:
        out_k2[col] = out_k2[col].map(lambda x: f"{x:.2f}")
    path_k2 = OUT / "study3f_multi_boundary_k2_top5.csv"
    print("[Study3-F] k=2 top 5")
    _save(out_k2, path_k2)
    print(out_k2.to_string(index=False))
    print()

    # k comparison
    cmp     = pd.read_csv(src_cmp)
    out_cmp = cmp.copy()
    out_cmp["selected"] = out_cmp["selected"].map(lambda x: "★ Selected" if x else "")
    for col in ["val_F1", "val_AUC", "test_F1", "test_AUC", "delta_val_F1"]:
        out_cmp[col] = out_cmp[col].map(lambda x: f"{x:.4f}")
    path_cmp = OUT / "study3f_multi_boundary_k_comparison.csv"
    print("[Study3-F] k comparison")
    _save(out_cmp, path_cmp)
    print(out_cmp.to_string(index=False))
    print()


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Thesis table export")
    print("=" * 60)
    table_ross_candidates()
    table_dawce_summary()
    table_wilcoxon_weighted()
    table_drift_detector_compare()
    table_awe_comparison()
    table_multi_boundary_ross()
    print(f"\nAll tables saved to: {OUT}")
