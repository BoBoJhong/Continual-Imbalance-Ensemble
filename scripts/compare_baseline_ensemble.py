"""
比較實驗 1 (Baseline)、實驗 2 (Ensemble)、實驗 3 (DES) 的結果，
輸出合併表與簡單圖表。
"""
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent
baseline_path = project_root / "results/baseline/bankruptcy_baseline_results.csv"
ensemble_path = project_root / "results/ensemble/bankruptcy_ensemble_results.csv"
des_path = project_root / "results/des/bankruptcy_des_results.csv"
output_dir = project_root / "results"
output_dir.mkdir(parents=True, exist_ok=True)


def main():
    # 讀取結果
    df_b = pd.read_csv(baseline_path, index_col=0)
    df_e = pd.read_csv(ensemble_path, index_col=0)
    df_b["實驗"] = "baseline"
    df_e["實驗"] = "ensemble"
    parts = [df_b, df_e]

    if des_path.exists():
        df_d = pd.read_csv(des_path, index_col=0)
        df_d["實驗"] = "des"
        parts.append(df_d)

    # 合併
    df_all = pd.concat(parts, axis=0)
    df_all = df_all.sort_values("AUC", ascending=False)

    # 儲存合併表
    out_csv = output_dir / "bankruptcy_all_results.csv"
    df_all.to_csv(out_csv)
    print(f"已儲存合併結果: {out_csv}\n")

    # 終端輸出摘要
    print("=" * 60)
    print("Bankruptcy 實驗結果比較 (依 AUC 排序)")
    print("=" * 60)
    print(df_all[["AUC", "F1", "實驗"]].to_string())
    print("\n最佳 AUC 方法:", df_all["AUC"].idxmax(), f"({df_all['AUC'].max():.4f})")

    # 簡單長條圖（若環境有 matplotlib）
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        df_plot = df_all.sort_values("AUC", ascending=True)
        ax.barh(df_plot.index.astype(str), df_plot["AUC"], color=["#2ecc71" if e == "baseline" else "#3498db" for e in df_plot["實驗"]])
        ax.set_xlabel("AUC")
        ax.set_title("Bankruptcy: Baseline vs Ensemble 比較")
        ax.set_xlim(0.9, 1.0)
        plt.tight_layout()
        out_fig = output_dir / "bankruptcy_auc_comparison.png"
        plt.savefig(out_fig, dpi=150)
        plt.close()
        print(f"已儲存圖表: {out_fig}")
    except Exception as e:
        print("(略過圖表:", e, ")")


if __name__ == "__main__":
    main()
