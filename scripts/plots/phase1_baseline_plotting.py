"""
Phase 1 baseline 共用繪圖：年份切割折線圖 + compact 熱力圖。
各 visualize_phase1_* 腳本使用目前 baseline 主線方法順序（Old / New / Retrain）。

折線圖橫軸預設為 Split index（1…n）；若要顯示 old+new 年數刻度，請設定環境變數
PHASE1_PLOT_SPLIT_AXIS=window 後再執行 visualize 腳本。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    raise ImportError(
        "Phase 1 baseline plots need matplotlib and seaborn. "
        "Run: pip install matplotlib seaborn "
        "(full requirements.txt includes them; requirements-core.txt does not.)"
    ) from e

import numpy as np
import pandas as pd

SPLIT_RE = re.compile(r"^split_(\d+)\+(\d+)$")
SAMPLING_ORDER = ["none", "undersampling", "oversampling", "hybrid"]

# 年份折線圖橫軸：預設 "index"（僅 1…n，較不擁擠）；「window」顯示 old+new 訓練年數（如 2+14）。
# 覆寫：環境變數 PHASE1_PLOT_SPLIT_AXIS=window
_SPLIT_AXIS_ENV = os.environ.get("PHASE1_PLOT_SPLIT_AXIS", "index").strip().lower()
SPLIT_X_AXIS_STYLE = _SPLIT_AXIS_ENV if _SPLIT_AXIS_ENV in ("index", "window") else "index"

# 僅在某一個 split 有數值時（例如 Retrain 全資料只訓練一次），改畫橫跨橫軸的參考線，避免單點折線。
DEFAULT_GLOBAL_REFERENCE_METHODS = frozenset({"Retrain"})

METHOD_ORDER_XGB = ["Old", "New", "Retrain"]
METHOD_ORDER_NO_FINETUNE = ["Old", "New", "Retrain"]
METHOD_COLORS_XGB = {
    "Old": "#5B8DB8",
    "New": "#E07B54",
    "Retrain": "#6DBF8E",
}

# Torch MLP / TabNet 破產實驗已與 XGB 對齊為三策略。
METHOD_ORDER_MLP = ["Old", "New", "Retrain"]
METHOD_COLORS_MLP = {
    "Old": "#5B8DB8",
    "New": "#E07B54",
    "Retrain": "#6DBF8E",
    # 舊版 sklearn MLP 腳本或過期 raw 若仍含「Old+New」：
    "Old+New": "#3D9970",
}

# sklearn MLP（medical、stock、舊版 bankruptcy MLP）等三策略 raw。
METHOD_ORDER_LEGACY_THREE = ["Old", "Old+New", "New"]
METHOD_COLORS_LEGACY_THREE = {
    "Old": "#5B8DB8",
    "Old+New": "#3D9970",
    "New": "#E07B54",
}


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


_style()


def parse_new_years(split_id: str) -> int | None:
    m = SPLIT_RE.match(str(split_id).strip())
    if not m:
        return None
    return int(m.group(2))


def split_tick_label(split_id: str) -> str:
    """橫軸刻度：舊視窗年數 + 新視窗年數（與 split_2+14 對齊）。"""
    m = SPLIT_RE.match(str(split_id).strip())
    if m:
        return f"{m.group(1)}+{m.group(2)}"
    return str(split_id)


def dataset_label_from_raw_stem(stem: str, raw_suffix: str) -> str:
    if stem.startswith("bankruptcy"):
        return "Bankruptcy"
    if stem.startswith("medical"):
        return "Medical"
    if stem.startswith("stock"):
        return "Stock"
    return stem.replace(raw_suffix, "").replace("_", " ").title()


def plot_year_split_lines(
    df: pd.DataFrame,
    dataset_title: str,
    metrics: list[str],
    out_path: Path,
    *,
    model_title: str,
    method_order: list[str],
    method_colors: dict[str, str],
    global_reference_methods: frozenset[str] | None = None,
) -> None:
    if global_reference_methods is None:
        global_reference_methods = DEFAULT_GLOBAL_REFERENCE_METHODS

    df = df.copy()
    df["_ny"] = df["split"].map(parse_new_years)
    df = df.dropna(subset=["_ny"])
    splits_sorted = df.sort_values("_ny", ascending=False)["split"].unique()

    x_pos = np.arange(len(splits_sorted))
    split_to_x = {s: i for i, s in enumerate(splits_sorted)}

    if SPLIT_X_AXIS_STYLE == "window":
        x_labels = [split_tick_label(s) for s in splits_sorted]
        x_axis_label = "Window split (old + new train years)"
    else:
        x_labels = [str(i + 1) for i in range(len(splits_sorted))]
        x_axis_label = "Split index"

    n_splits = len(x_pos)
    n_met = len(metrics)
    n_samp = len(SAMPLING_ORDER)
    # 多個 window split 時橫軸易重疊：依點數加寬圖、斜向刻度並預留下緣。
    base_w = 3.6 * n_met + 1.0
    min_w_per_split = 0.34 if SPLIT_X_AXIS_STYLE == "index" else 0.36
    fig_w = max(base_w, min_w_per_split * max(n_splits, 1) + 2.0)
    fig_h = 2.95 * n_samp + 0.65 + min(0.35 * n_samp, 0.008 * n_splits * n_samp)
    tick_fs = max(7, min(9, 11 - n_splits // 4))
    if SPLIT_X_AXIS_STYLE == "window":
        x_rot = 52 if n_splits > 6 else (35 if n_splits > 4 else 0)
    else:
        x_rot = 45 if n_splits > 14 else 0

    # sharex=True 會讓整張圖共用單一 x 軸，多欄時通常只顯示一欄的刻度 → 橫軸年份看不到。
    fig, axes = plt.subplots(
        n_samp,
        n_met,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex="col",
    )

    for si, sampling in enumerate(SAMPLING_ORDER):
        sub_s = df[df["sampling"] == sampling]
        for mi, metric in enumerate(metrics):
            ax = axes[si, mi]
            if metric not in sub_s.columns:
                ax.set_visible(False)
                continue
            # 全寬參考線須最後繪製，否則會被後續折線蓋住（RF/XGB 常發生）。
            deferred_global_ref: list[tuple[str, np.ndarray, np.ndarray, str]] = []
            for method in method_order:
                msub = sub_s[sub_s["method"] == method].drop_duplicates(subset=["split"])
                if msub.empty:
                    continue
                xs, ys = [], []
                for _, row in msub.iterrows():
                    s = row["split"]
                    if s in split_to_x:
                        xs.append(split_to_x[s])
                        ys.append(float(row[metric]))
                if not xs:
                    continue
                order = np.argsort(xs)
                xs = np.array(xs)[order]
                ys = np.array(ys)[order]
                color = method_colors.get(method, "#333333")
                # Retrain 等「全量只 fit 一次」：raw 常只帶一個 split，單點連線語意不清 → 畫全寬水平參考線
                if (
                    method in global_reference_methods
                    and len(xs) == 1
                    and len(x_pos) > 1
                ):
                    deferred_global_ref.append((method, xs, ys, color))
                    continue
                ax.plot(
                    xs,
                    ys,
                    "o-",
                    color=color,
                    linewidth=1.6,
                    markersize=4,
                    label=method,
                    zorder=2,
                )
            for method, xs, ys, color in deferred_global_ref:
                y0 = float(ys[0])
                ax.plot(
                    [x_pos[0], x_pos[-1]],
                    [y0, y0],
                    linestyle="--",
                    color=color,
                    linewidth=2.0,
                    label=method,
                    zorder=5,
                )
                ax.plot(
                    float(xs[0]),
                    y0,
                    "o",
                    color=color,
                    markersize=5,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    zorder=6,
                    label="_nolegend_",
                )
            ax.set_xticks(x_pos)
            if si == n_samp - 1:
                ax.set_xticklabels(
                    x_labels,
                    rotation=x_rot,
                    ha="right" if x_rot else "center",
                    fontsize=tick_fs,
                )
                ax.tick_params(axis="x", pad=2)
                ax.set_xlabel(x_axis_label)
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(metric)
            ax.set_title(f"{sampling}")
            if si == 0 and mi == n_met - 1:
                ax.legend(loc="lower right", fontsize=7, framealpha=0.9)

    fig.suptitle(
        f"Phase 1 {model_title} — {dataset_title} (year splits)",
        fontsize=12,
        y=1.02,
    )
    bottom_reserve = 0.11 + min(0.22, 0.012 * n_splits)
    if x_rot:
        bottom_reserve = max(bottom_reserve, 0.14 + min(0.2, 0.015 * n_splits))
    fig.tight_layout(rect=(0.02, bottom_reserve, 0.98, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_compact_heatmaps(
    summary_path: Path,
    out_prefix: Path,
    *,
    model_title: str,
    method_order: list[str],
    project_root: Path,
) -> None:
    df = pd.read_csv(summary_path)
    if not {"metric", "method"}.issubset(df.columns):
        return
    sampling_cols = [c for c in df.columns if c not in ("metric", "method")]
    metrics = df["metric"].unique()
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1, 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        sub = df[df["metric"] == metric].set_index("method")[sampling_cols]
        sub = sub.reindex([m for m in method_order if m in sub.index])
        col_order = [c for c in ["hybrid", "none", "oversampling", "undersampling"] if c in sub.columns]
        sub = sub[col_order]
        sns.heatmap(
            sub.astype(float),
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0 if metric not in ("Type1_Error", "Type2_Error") else None,
            ax=ax,
            cbar_kws={"shrink": 0.6},
        )
        ax.set_title(metric)
        ax.set_xlabel("Sampling")
        ax.set_ylabel("Method")
    fig.suptitle(
        f"Phase 1 {model_title} — mean over splits (compact summary)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out = out_prefix.parent / f"{out_prefix.name}_compact_heatmaps.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {out.relative_to(project_root)}")


def run_phase1_baseline_visualization(
    project_root: Path,
    metrics: list[str],
    *,
    result_subdir: str,
    raw_glob: str,
    raw_suffix: str,
    model_title: str,
    method_order: list[str],
    method_colors: dict[str, str],
    compact_summary_name: str | None = None,
) -> bool:
    """
    讀取 results/phase1_baseline/<result_subdir>/ 下 raw CSV，輸出折線圖至 plots/；
    若存在 compact_summary_name 則另產熱力圖。
    成功處理至少一個 raw 檔時回傳 True；找不到 raw 時回傳 False。
    """
    base_dir = project_root / "results" / "phase1_baseline" / result_subdir
    plots_dir = base_dir / "plots"
    raw_files = sorted(base_dir.glob(raw_glob)) if base_dir.is_dir() else []

    if not raw_files:
        print(f"找不到 raw CSV：{base_dir}（glob: {raw_glob}）")
        return False

    print(f"Phase 1 {model_title} 圖表輸出…")
    for path in raw_files:
        df = pd.read_csv(path)
        need = {"split", "method", "sampling", *metrics}
        missing = need - set(df.columns)
        if missing:
            print(f"  [SKIP] {path.name} 缺少欄位: {missing}")
            continue
        label = dataset_label_from_raw_stem(path.stem, raw_suffix)
        out = plots_dir / f"{path.stem.replace(raw_suffix, '')}_year_splits_{'_'.join(metrics)}.png"
        plot_year_split_lines(
            df,
            label,
            metrics,
            out,
            model_title=model_title,
            method_order=method_order,
            method_colors=method_colors,
        )
        print(f"  [SAVED] {out.relative_to(project_root)}")

    if compact_summary_name:
        summary = base_dir / compact_summary_name
        if summary.exists():
            print("\nCompact summary 熱力圖…")
            stem = summary.stem
            hprefix = (
                stem[: -len("_compact_summary")]
                if stem.endswith("_compact_summary")
                else stem
            )
            plot_compact_heatmaps(
                summary,
                plots_dir / hprefix,
                model_title=model_title,
                method_order=method_order,
                project_root=project_root,
            )

    print(f"\n完成。圖表目錄: {plots_dir.relative_to(project_root)}")
    return True
