"""
彙總所有實驗結果：Bankruptcy / Stock / Medical 的 baseline、ensemble、DES、Study II。
產出 results/summary_all_datasets.csv 與終端摘要。
"""
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
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
        ("stock/stock_baseline_ensemble_results.csv", "stock"),
        ("des/stock_des_results.csv", "stock"),
    ]:
        df = load_if_exists(results_dir / name, tag)
        if df is not None:
            df["experiment"] = name.split("/")[0]
            rows.append(df)

    # Medical
    for name, tag in [
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

    # 只取有 AUC 的表格，合併成 (method, dataset, AUC, F1)
    summary_rows = []
    for df in rows:
        if "AUC" not in df.columns:
            continue
        for idx in df.index:
            try:
                row = {
                    "dataset": df.loc[idx].get("dataset", ""),
                    "method": str(idx),
                    "AUC": float(df.loc[idx, "AUC"]),
                    "F1": float(df.loc[idx, "F1"]) if "F1" in df.columns else None,
                }
                summary_rows.append(row)
            except Exception:
                continue
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        print("無 AUC 欄位可彙總")
        return
    out_csv = results_dir / "summary_all_datasets.csv"
    summary.to_csv(out_csv, index=False)
    print(f"已保存: {out_csv}")
    print("\n各資料集最佳 AUC：")
    for ds in summary["dataset"].dropna().unique():
        sub = summary[summary["dataset"] == ds]
        best = sub.loc[sub["AUC"].idxmax()]
        print(f"  {ds}: {best['method']} AUC={best['AUC']:.4f}")


if __name__ == "__main__":
    main()
