"""Add the two extra checks to the XGBoost workbook of the 15 year-split bankruptcy study.

Input: the folder written by scripts/experiments/year_splits_extra_checks.py and the thesis
XGBoost run. The workbook is rebuilt with the unchanged thesis exporter (its 9 sheets are
identical to the delivered workbook), then these sheets are appended with the same layout:

  排除2018_基線_*, 排除2018_集成_*   Test restricted to 2015-2017; models, thresholds, V-static unchanged
  排除2018_結論比較                  spec family C (69 Wilcoxon tests, Holm) on 2015-2018 vs 2015-2017
  修正前_集成_*                      DESlib-default protocol of the fixed-split base run (no correction)
  修正前對修正後_*                   mean over splits per column, corrected vs before correction

Family C on 2015-2017 is also written as CSV next to the thesis statistics. Numbers are only
averaged and rearranged; nothing is refitted or reselected here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font


def fake_run(folder: Path, baseline: pd.DataFrame, ensemble: pd.DataFrame, seeds) -> Path:
    """Write rows in the thesis run layout so the thesis table and test code can read them."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(
        {"status": "completed", "classifier": "xgb", "config": {"seeds": seeds}}), encoding="utf-8")
    baseline.to_csv(folder / "baseline_metrics.csv", index=False)
    ensemble.loc[ensemble.period.astype(str).eq("pooled")].to_csv(folder / "ensemble_metrics_pooled.csv", index=False)
    return folder


def add_baseline_sheets(wb, xlsx, tables, baseline, prefix, title, scope):
    sampling = list(tables.SAMPLING.values())
    for metric in tables.METRICS:
        frame = tables.baseline_table(baseline, metric)
        frame = frame.loc[frame.years.ne("avg")].drop(columns="avg")
        ws, top = xlsx.new_sheet(wb, f"{prefix}基線_{metric}", f"{title}基線 {metric}", [
            scope, "由上到下：Old 1yr→15yr（離測試最遠→近）、16yr(retrain)、New 15yr→1yr（最下面離測試最近）。", xlsx.LEGEND])
        xlsx.write_block(ws, top, frame, ["years"], sampling, "avg", sampling)
        xlsx.finish(ws, 6, top)


def add_ensemble_sheets(wb, xlsx, tables, ensemble, baseline, prefix, title, scope, extra):
    for metric in tables.METRICS:
        frame = tables.ensemble_table(ensemble, baseline, metric)
        frame = frame.loc[frame.split.ne("avg")]
        values = [c for c in frame.columns if c not in tables.KEYS]
        methods = [c for c in values if c not in tables.REFERENCES]
        ws, top = xlsx.new_sheet(wb, f"{prefix}集成_{metric}", f"{title}集成 {metric}", [
            scope, *extra, "一列一組切割（1+15 最上、15+1 最下）。前兩欄是基線對照；2–5models 是該數量所有組合的平均。"
                           "avg(集成) 不含兩欄基線對照。", xlsx.LEGEND, xlsx.GROUP_LEGEND])
        short = {c: xlsx.family(c)[1] for c in values}
        xlsx.write_block(ws, top + 1, frame.rename(columns=short), list(tables.KEYS), [short[c] for c in values],
                         "avg(集成)", [short[c] for c in methods])
        xlsx.write_group_band(ws, top, len(tables.KEYS) + 1, values)
        xlsx.finish(ws, len(tables.KEYS) + len(values) + 1, top + 1, first_width=9)


def header_row(ws, xlsx, row, names):
    for j, text in enumerate(names, 1):
        cell = ws.cell(row=row, column=j, value=text)
        cell.font, cell.fill, cell.border = Font(name=xlsx.FONT, bold=True), xlsx.HEADER_FILL, xlsx.BORDER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 30


def body_cell(ws, xlsx, row, column, value, bold=False, number=False):
    cell = ws.cell(row=row, column=column, value=value)
    cell.font, cell.border = Font(name=xlsx.FONT, bold=bold), xlsx.BORDER
    if number:
        cell.number_format = "0.0000"
        cell.alignment = Alignment(horizontal="center")
    return cell


def add_mean_comparison(wb, xlsx, tables, left, right, metric, name, notes, labels):
    """Mean over splits per ensemble column for two versions, their difference and split counts."""
    ws = wb.create_sheet(name)
    for i, text in enumerate(notes, 1):
        ws.cell(row=i, column=1, value=text).font = Font(name=xlsx.FONT, bold=i == 1, size=12 if i == 1 else 9)
    top = len(notes) + 2
    header_row(ws, xlsx, top, ["欄位", labels[0], labels[1], f"差（{labels[1]}−{labels[0]}）", f"{labels[1]}較高的切割數"])
    columns = [c for c in left.columns if c not in tables.KEYS]
    for i, column in enumerate(columns, top + 1):
        a, b = left[column].astype(float), right[column].astype(float)
        higher = int((b.round(4) > a.round(4)).sum())
        body_cell(ws, xlsx, i, 1, column)
        better = 2 if round(a.mean(), 4) > round(b.mean(), 4) else 3 if round(b.mean(), 4) > round(a.mean(), 4) else 0
        for j, value in ((2, a.mean()), (3, b.mean()), (4, b.mean() - a.mean())):
            cell = body_cell(ws, xlsx, i, j, float(value), bold=j == better, number=True)
            if j == better:
                cell.fill = xlsx.COLUMN_BEST_FILL
        body_cell(ws, xlsx, i, 5, f"{higher} / {len(b)}").alignment = Alignment(horizontal="center")
    ws.column_dimensions["A"].width = 24
    for letter in "BCDE":
        ws.column_dimensions[letter].width = 17
    ws.freeze_panes = ws.cell(row=top + 1, column=2)


def add_conclusion_sheet(wb, xlsx, full, restricted):
    ws = wb.create_sheet("排除2018_結論比較")
    notes = ["排除 2018 年：統計檢定家族 C（集成，69 項）的結論比較",
             "左：論文用的 Test 2015–2018；右：Test 只留 2015–2017。兩邊都是 15 組切割的 Wilcoxon 檢定，Holm 在 69 項內校正。",
             "模型、門檻與 V-static 的選擇都不變（都只用驗證資料決定），只有計分的 Test 年份不同。",
             "「結論相同」＝顯著與否相同，且平均差方向相同。"]
    for i, text in enumerate(notes, 1):
        ws.cell(row=i, column=1, value=text).font = Font(name=xlsx.FONT, bold=i == 1, size=12 if i == 1 else 9)
    top = len(notes) + 2
    names = ["比較組", "A", "B", "指標", "2015–2018 平均差", "2015–2018 Holm p", "2015–2018 顯著",
             "2015–2017 平均差", "2015–2017 Holm p", "2015–2017 顯著", "方向相同", "結論相同"]
    header_row(ws, xlsx, top, names)
    keys = ["group", "A", "B", "metric"]
    joined = full.merge(restricted, on=keys, suffixes=("_full", "_1517"))
    if len(joined) != len(full) or len(joined) != len(restricted):
        raise ValueError("Family C rows differ between the two Test windows")
    for i, row in enumerate(joined.itertuples(index=False), top + 1):
        same_direction = np.sign(row.mean_diff_full) == np.sign(row.mean_diff_1517)
        same = same_direction and row.significant_full == row.significant_1517
        values = [row.group, row.A, row.B, row.metric, row.mean_diff_full, row.p_holm_full, row.significant_full,
                  row.mean_diff_1517, row.p_holm_1517, row.significant_1517,
                  "是" if same_direction else "否", "是" if same else "否"]
        for j, value in enumerate(values, 1):
            cell = body_cell(ws, xlsx, i, j, float(value) if isinstance(value, (float, np.floating)) else value,
                             number=isinstance(value, (float, np.floating)), bold=j == 12 and not same)
            if j == 12 and not same:
                cell.fill = xlsx.COLUMN_BEST_FILL
    widths = [18, 12, 14, 8] + [13] * 8
    for j, width in enumerate(widths, 1):
        ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = width
    ws.freeze_panes = ws.cell(row=top + 1, column=5)
    return joined


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkout", type=Path, required=True, help="experiment checkout (exporter and test code)")
    parser.add_argument("--xgb-run", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True, help="output folder of year_splits_extra_checks.py")
    parser.add_argument("--out", type=Path, required=True, help="XGBoost workbook to write")
    parser.add_argument("--stats-out", type=Path, required=True, help="folder for the 2015-2017 family C CSV")
    args = parser.parse_args()

    sys.path.insert(0, str((args.checkout / "scripts/analysis").resolve()))
    import export_year_splits_event_xlsx as xlsx
    import year_splits_event_statistics as tests
    tables = xlsx.tables

    checks = args.checks.resolve()
    verification = json.loads((checks / "verification.json").read_text(encoding="utf-8"))
    if any(value != 0.0 for key, value in verification.items() if "max_abs_diff" in key):
        raise ValueError(f"Refitted pool differs from the thesis run: {verification}")
    seeds = json.loads((checks / "manifest.json").read_text(encoding="utf-8"))["seeds"]
    run_1517 = fake_run(checks / "tables_2015_2017", pd.read_csv(checks / "baseline_1517.csv"),
                        pd.read_csv(checks / "ensemble_1517.csv"), seeds)
    run_default = fake_run(checks / "tables_before_correction", pd.read_csv(checks / "baseline_full.csv"),
                           pd.read_csv(checks / "ensemble_default.csv"), seeds)

    xlsx.main(["--run", str(args.xgb_run), "--out", str(args.out)])
    wb = load_workbook(args.out)
    _, thesis_base, thesis_ens = tables.load_run(args.xgb_run.resolve())
    thesis_ens = thesis_ens.loc[thesis_ens.fs.eq("no_fs")]
    _, base_1517, ens_1517 = tables.load_run(run_1517)
    _, _, ens_default = tables.load_run(run_default)
    seed_text = f"{len(seeds)} 個種子平均"

    scope = f"XGBoost；{seed_text}；Test 2015–2017 合併（排除 2018）；四位小數"
    add_baseline_sheets(wb, xlsx, tables, base_1517, "排除2018_", "排除 2018：", scope)
    add_ensemble_sheets(wb, xlsx, tables, ens_1517, base_1517, "排除2018_", "排除 2018：", scope,
                        ["模型、門檻與 V-static 的選擇與論文相同（都只用驗證資料決定），只把 Test 限縮到 2015–2017 年。"])
    full_c = tests.finish(tests.family_c(tests.ensemble_table(args.xgb_run.resolve()),
                                         tests.baseline_table(args.xgb_run.resolve())))
    c_1517 = tests.finish(tests.family_c(tests.ensemble_table(run_1517), tests.baseline_table(run_1517)))
    joined = add_conclusion_sheet(wb, xlsx, full_c, c_1517)

    scope = f"XGBoost；{seed_text}；Test 2015–2018 合併；四位小數"
    add_ensemble_sheets(wb, xlsx, tables, ens_default, thesis_base, "修正前_", "修正前（套件預設設定）：", scope,
                        ["與論文相同的 6 個模型，但不做三項修正：成員機率不校正、成員以 0.5 判定、DCS 平手取第一個、"
                         "KNORA-U 用套件原本的加權。靜態集成直接平均未校正的機率。這是固定切割版論文「修正前」的做法。"])
    for metric in tables.METRICS:
        corrected = tables.ensemble_table(thesis_ens, thesis_base, metric)
        default = tables.ensemble_table(ens_default, thesis_base, metric)
        corrected, default = corrected.loc[corrected.split.ne("avg")], default.loc[default.split.ne("avg")]
        add_mean_comparison(wb, xlsx, tables, default, corrected, metric, f"修正前對修正後_{metric}", [
            f"修正前 對 修正後（論文用的版本）：{metric}",
            f"各欄為 15 組切割的平均（{seed_text}；Test 2015–2018 合併；四位小數）。沒有特徵選取。",
            "差＝修正後 − 修正前（正數表示修正後較高）。基線_ 開頭的兩列是單一模型，不受修正影響，兩邊相同。"],
            ("修正前", "修正後"))
    for metric in tables.METRICS:
        full = tables.ensemble_table(thesis_ens, thesis_base, metric)
        part = tables.ensemble_table(ens_1517, base_1517, metric)
        full, part = full.loc[full.split.ne("avg")], part.loc[part.split.ne("avg")]
        add_mean_comparison(wb, xlsx, tables, full, part, metric, f"排除2018對全部_{metric}", [
            f"Test 2015–2018（論文）對 Test 2015–2017（排除 2018）：{metric}",
            f"各欄為 15 組切割的平均（{seed_text}；四位小數）。沒有特徵選取。",
            "差＝2015–2017 − 2015–2018。"], ("2015–2018", "2015–2017"))
    wb.save(args.out)

    stats_out = args.stats_out.resolve()
    (stats_out / "未四捨五入").mkdir(parents=True, exist_ok=True)
    c_1517.to_csv(stats_out / "未四捨五入" / "C_集成_排除2018.csv", index=False, encoding="utf-8-sig")
    tests.readable(c_1517).to_csv(stats_out / "C_集成_排除2018.csv", index=False, encoding="utf-8-sig")
    same = (np.sign(joined.mean_diff_full) == np.sign(joined.mean_diff_1517)) & \
        (joined.significant_full == joined.significant_1517)
    print(f"WROTE {args.out}")
    print(f"family C: significant 2015-2018 {int((full_c.p_holm < 0.05).sum())}/69, "
          f"2015-2017 {int((c_1517.p_holm < 0.05).sum())}/69, same conclusion {int(same.sum())}/69")


if __name__ == "__main__":
    main()
