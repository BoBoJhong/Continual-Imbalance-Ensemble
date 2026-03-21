"""
彙總所有實驗結果：Bankruptcy / Stock / Medical 的 baseline、ensemble、DES、Study II。
產出 results/summary_all_datasets.csv 與終端摘要。
"""
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
results_dir = project_root / "results"


def load_if_exists(path, dataset_tag=""):
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    if dataset_tag:
        df["dataset"] = dataset_tag
    return df


def main():
    rows = []

    # Bankruptcy
    for name, tag in [
        ("baseline/bankruptcy_baseline_results.csv", "bankruptcy"),
        ("ensemble/bankruptcy_ensemble_results.csv", "bankruptcy"),
        ("des/bankruptcy_des_results.csv", "bankruptcy"),
    ]:
        df = load_if_exists(results_dir / name, tag)
        if df is not None:
            df["experiment"] = name.split("/")[0]
            rows.append(df)
    # feature_study 結構不同（AUC_no_fs/AUC_with_fs），不納入本摘要

    # Stock
    for name, tag in [
        ("ensemble/stock_ensemble_results.csv", "stock"),       # 統一路徑（優先）
        ("stock/stock_baseline_ensemble_results.csv", "stock"),
        ("des/stock_des_results.csv", "stock"),
    ]:
        df = load_if_exists(results_dir / name, tag)
        if df is not None:
            df["experiment"] = name.split("/")[0]
            rows.append(df)

    # Medical
    for name, tag in [
        ("ensemble/medical_ensemble_results.csv", "medical"),   # 統一路徑（優先）
        ("medical/medical_baseline_ensemble_results.csv", "medical"),
        ("des/medical_des_results.csv", "medical"),
    ]:
        df = load_if_exists(results_dir / name, tag)
        if df is not None:
            df["experiment"] = name.split("/")[0]
            rows.append(df)

    if not rows:
        print("未找到任何結果檔。請先執行實驗。")
        return

    # 只取有 AUC 的表格，合併成 (method, dataset, AUC, F1, etc)
    summary_rows = []
    for df in rows:
        if "AUC" not in df.columns:
            continue
        for idx in df.index:
            try:
                row = {
                    "dataset": df.loc[idx].get("dataset", ""),
                    "method": str(idx),
                    "AUC": float(df.loc[idx, "AUC"]) if "AUC" in df.columns else None,
                    "F1": float(df.loc[idx, "F1"]) if "F1" in df.columns else None,
                    "Precision": float(df.loc[idx, "Precision"]) if "Precision" in df.columns else None,
                    "Recall": float(df.loc[idx, "Recall"]) if "Recall" in df.columns else None,
                    "Type1_Error(FPR)": float(df.loc[idx, "Type 1 Error (FPR)"]) if "Type 1 Error (FPR)" in df.columns else (float(df.loc[idx, "Type1_Error"]) if "Type1_Error" in df.columns else None),
                    "Type2_Error(FNR)": float(df.loc[idx, "Type 2 Error (FNR)"]) if "Type 2 Error (FNR)" in df.columns else (float(df.loc[idx, "Type2_Error"]) if "Type2_Error" in df.columns else None),
                }
                summary_rows.append(row)
            except Exception:
                continue
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        print("無 AUC 欄位可彙總")
        return
    
    # 針對比較需求：顯示各方法的優劣
    summary = summary.sort_values(by=["dataset", "AUC"], ascending=[True, False])

    out_csv = results_dir / "summary_all_datasets_detailed.csv"
    try:
        summary.to_csv(out_csv, index=False)
        print(f"已保存: {out_csv}")
    except PermissionError:
        out_csv = results_dir / "summary_all_datasets_detailed_new.csv"
        summary.to_csv(out_csv, index=False)
        print(f"警告：原檔案被鎖定，已儲存至新檔: {out_csv}")

    # 同時輸出 summary_all_datasets.csv（含所有指標，方便視覺化腳本使用）
    out_simple = results_dir / "summary_all_datasets.csv"
    simple_cols = ["dataset", "method", "AUC", "F1", "Precision", "Recall",
                   "Type1_Error(FPR)", "Type2_Error(FNR)"]
    simple_cols_exist = [c for c in simple_cols if c in summary.columns]
    try:
        summary[simple_cols_exist].to_csv(out_simple, index=False)
        print(f"已保存: {out_simple}")
    except PermissionError:
        print(f"警告：{out_simple} 被鎖定，跳過")

    print("\n各資料集最佳 AUC：")
    for ds in summary["dataset"].dropna().unique():
        sub = summary[summary["dataset"] == ds]
        best = sub.loc[sub["AUC"].idxmax()]
        t1 = best.get("Type1_Error(FPR)", float("nan"))
        t1_str = f"{t1:.4f}" if pd.notna(t1) else "N/A"
        print(f"  {ds}: {best['method']} AUC={best['AUC']:.4f}, F1={best['F1']:.4f}, Type1_Error={t1_str}")


if __name__ == "__main__":
    main()
