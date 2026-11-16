"""Random Forest workbook with baseline and (supplementary) ensemble sheets, plus an XGBoost comparison.

Uses the unchanged year-split xlsx exporter of the dataset's checkout for the baseline and
ensemble sheets (same layout and marks as the thesis workbooks). Baseline sheets use the RF
baseline run reported in the thesis; ensemble sheets use the RF pool run. The RF supplement has no
feature selection, so the exporter's feature-selection sheets are not written. One extra sheet
per metric compares the XGBoost pool (thesis run, no_fs part) with the RF pool: the mean over
splits of every ensemble column, the difference RF - XGBoost, and in how many splits RF is higher.
Numbers are averaged only; nothing is refitted or reselected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font

MODULES = {"bankruptcy": "export_year_splits_event_xlsx", "natality": "export_year_splits_natality_xlsx"}
TEST = {"bankruptcy": "Test 2015–2018 合併", "natality": "Test 2022–2023 出生年合併"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=MODULES, required=True)
    parser.add_argument("--checkout", type=Path, required=True, help="checkout holding the dataset's exporter")
    parser.add_argument("--rf-run", type=Path, required=True, help="RF pool ensemble run")
    parser.add_argument("--baseline-run", type=Path, required=True, help="RF baseline-only run used in the thesis")
    parser.add_argument("--xgb-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    sys.path.insert(0, str((args.checkout / "scripts/analysis").resolve()))
    xlsx = __import__(MODULES[args.dataset])
    tables = xlsx.tables

    # Baseline numbers come from the RF baseline run the thesis reports; ensemble numbers from the
    # RF pool run. (The pool run refits the same baselines; they agree to four decimals.)
    rf_manifest, _, rf_ens = tables.load_run(args.rf_run.resolve())
    base_manifest, rf_base, _ = tables.load_run(args.baseline_run.resolve())
    if (rf_manifest["classifier"], base_manifest["classifier"]) != ("rf", "rf") or \
            rf_manifest["config"]["seeds"] != base_manifest["config"]["seeds"]:
        raise ValueError("Need the RF pool run and the RF baseline run with the same seeds")
    load_run = tables.load_run
    tables.load_run = lambda run: (rf_manifest, rf_base, rf_ens)
    # Sheets 基線 and 集成 come from the thesis exporter; its 特徵選取 sheets are skipped.
    tables.fs_table = lambda *a, **k: pd.DataFrame(columns=["method", "split"])
    xlsx.main(["--run", str(args.rf_run), "--out", str(args.out)])
    tables.load_run = load_run
    wb = load_workbook(args.out)
    for name in [n for n in wb.sheetnames if n.startswith("特徵選取_")]:
        del wb[name]

    xgb_manifest, xgb_base, xgb_ens = tables.load_run(args.xgb_run.resolve())
    if xgb_manifest["classifier"] != "xgb" or rf_ens is None or xgb_ens is None:
        raise ValueError("Need the XGBoost ensemble run and the RF ensemble run")
    xgb_ens = xgb_ens.loc[xgb_ens.fs.eq("no_fs")]

    bold = Font(name=xlsx.FONT, bold=True)
    for metric in tables.METRICS:
        x = tables.ensemble_table(xgb_ens, xgb_base, metric)
        r = tables.ensemble_table(rf_ens, rf_base, metric)
        x, r = x.loc[x.split.ne("avg")], r.loc[r.split.ne("avg")]
        if list(x.split) != list(r.split):
            raise ValueError("Split rows differ between the two runs")
        columns = [c for c in x.columns if c not in tables.KEYS]
        ws = wb.create_sheet(f"XGB對RF_{metric}")
        notes = [f"XGBoost 模型池 對 Random Forest 模型池：{metric}",
                 f"各欄為所有切割的平均（10 個種子平均；{TEST[args.dataset]}；四位小數）。兩者都沒有特徵選取。",
                 "差＝RF − XGBoost（正數表示 RF 較高）；「RF 較高的切割數」以四位小數比較。",
                 "基線_ 開頭的兩列是單一模型對照，不是集成。"]
        for i, text in enumerate(notes, 1):
            ws.cell(row=i, column=1, value=text).font = bold if i == 1 else Font(name=xlsx.FONT)
        header = ["方法", "XGBoost 模型池", "RF 模型池", "差 (RF−XGB)", "RF 較高的切割數"]
        top = len(notes) + 2
        for j, text in enumerate(header, 1):
            cell = ws.cell(row=top, column=j, value=text)
            cell.font, cell.fill, cell.border = bold, xlsx.HEADER_FILL, xlsx.BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for i, column in enumerate(columns, top + 1):
            xv, rv = x[column].astype(float), r[column].astype(float)
            higher = int((rv.round(4) > xv.round(4)).sum())
            values = [column, xv.mean(), rv.mean(), rv.mean() - xv.mean(), f"{higher} / {len(rv)}"]
            for j, value in enumerate(values, 1):
                cell = ws.cell(row=i, column=j, value=value)
                cell.font, cell.border = Font(name=xlsx.FONT), xlsx.BORDER
                if isinstance(value, float):
                    cell.number_format = "0.0000"
                    cell.alignment = Alignment(horizontal="center")
            better = 2 if round(xv.mean(), 4) > round(rv.mean(), 4) else 3 if round(rv.mean(), 4) > round(xv.mean(), 4) else None
            if better:
                ws.cell(row=i, column=better).font = bold
                ws.cell(row=i, column=better).fill = xlsx.COLUMN_BEST_FILL
        ws.column_dimensions["A"].width = 26
        for letter in "BCDE":
            ws.column_dimensions[letter].width = 16
        ws.freeze_panes = ws.cell(row=top + 1, column=2)
    wb.save(args.out)
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
