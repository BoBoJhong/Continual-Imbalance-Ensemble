"""Export explicitly pinned result CSVs to versioned LaTeX snapshots, without training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inside(base: Path, relative: str) -> Path:
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base.resolve()):
        raise ValueError(f"Path escapes its allowed directory: {relative}")
    return candidate


def latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_\allowbreak{}", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "/": r"/\allowbreak{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def table_tex(table: dict, rows: list[dict]) -> str:
    metrics = table["metrics"]
    widths = r">{\raggedright\arraybackslash}p{0.28\textwidth}" + (
        r">{\centering\arraybackslash}p{0.12\textwidth}" * len(metrics)
    )
    heading = "方法 & " + " & ".join(latex(m["label"]) for m in metrics) + r" \\"
    lines = [r"\begingroup", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\setlength{\LTleft}{\fill}", r"\setlength{\LTright}{\fill}",
             r"\setlength{\LTcapwidth}{\textwidth}",
             r"\begin{longtable}{" + widths + "}",
             r"\caption{" + latex(table["caption"]) + r"}\label{tab:" + table["id"] + r"}\\",
             r"\toprule", heading, r"\midrule", r"\endfirsthead",
             r"\toprule", heading, r"\midrule", r"\endhead",
             r"\bottomrule", r"\endfoot"]
    for row in rows:
        cells = [latex(row[table["row_key"]])]
        cells.extend(f'{float(row[m["mean"]]):.4f}' + r"\(\pm\)" +
                     f'{float(row[m["std"]]):.4f}' for m in metrics)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    return "\n".join(lines)


def prepare(root: Path, config_path: Path) -> tuple[dict, list[dict]]:
    data = config_path.read_bytes()
    config = json.loads(data.decode("utf-8-sig"))
    tables = config["tables"]
    if not tables:
        raise ValueError("No result table configured")
    sources = {config_path.relative_to(root).as_posix(): sha(data)}
    exporter = Path(__file__).resolve()
    if exporter.is_relative_to(root):
        sources[exporter.relative_to(root).as_posix()] = sha(exporter.read_bytes())
    prepared = []
    ids = set()
    for table in tables:
        ident = table["id"]
        if not re.fullmatch(r"[a-z][a-z0-9_]*", ident) or ident in ids:
            raise ValueError("Invalid or duplicate table id")
        ids.add(ident)
        run = inside(root / "results", str(Path(table["run"]).relative_to("results")))
        manifest_file = run / "manifest.json"
        manifest_data = manifest_file.read_bytes()
        manifest = json.loads(manifest_data.decode("utf-8-sig"))
        if manifest.get("status") != "completed":
            raise ValueError(f"Run is not completed: {table['run']}")
        if manifest.get("protocol") != table["expected_protocol"]:
            raise ValueError(f"Protocol mismatch: {ident}")
        seeds = manifest["config"]["seeds"]
        if sorted(seeds) != sorted(table["expected_seeds"]) or len(seeds) != len(set(seeds)):
            raise ValueError(f"Seed set mismatch: {ident}")
        sources[manifest_file.relative_to(root).as_posix()] = sha(manifest_data)
        for seed in seeds:
            complete_path = inside(run, table["complete_pattern"].format(seed=seed))
            complete_data = complete_path.read_bytes()
            complete = json.loads(complete_data.decode("utf-8-sig"))
            if complete.get("status") != "completed":
                raise ValueError(f"Incomplete seed: {ident}/{seed}")
            sources[complete_path.relative_to(root).as_posix()] = sha(complete_data)
            if "metrics_sha256" in complete:
                metrics_path = complete_path.parent / "metrics.csv"
                checksum = sha(metrics_path.read_bytes())
                if checksum != complete["metrics_sha256"]:
                    raise ValueError(f"Seed metrics checksum mismatch: {ident}/{seed}")
                sources[metrics_path.relative_to(root).as_posix()] = checksum
        source_path = inside(run, table["csv"])
        source_data = source_path.read_bytes()
        sources[source_path.relative_to(root).as_posix()] = sha(source_data)
        reader = csv.DictReader(io.StringIO(source_data.decode("utf-8-sig")))
        columns = reader.fieldnames or []
        needed = {table["row_key"], *table["filters"]}
        if not 1 <= len(table["metrics"]) <= 5:
            raise ValueError("Configure one to five metrics per table")
        needed.update(key for m in table["metrics"] for key in (m["mean"], m["std"]))
        if not needed.issubset(columns):
            raise ValueError(f"Missing CSV columns: {sorted(needed - set(columns))}")
        rows = [row for row in reader if all(row[k] == v for k, v in table["filters"].items())]
        if not rows:
            raise ValueError(f"Filters select no rows: {ident}")
        keys = [row[table["row_key"]] for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate row key after filtering: {ident}")
        for row in rows:
            if None in row or any(row.get(column) is None for column in columns):
                raise ValueError(f"Malformed CSV row: {ident}")
            for metric in table["metrics"]:
                mean, std = float(row[metric["mean"]]), float(row[metric["std"]])
                if not math.isfinite(mean) or not 0 <= mean <= 1 or not math.isfinite(std) or std < 0:
                    raise ValueError(f"Invalid metric: {ident}/{row[table['row_key']]}")
        rows.sort(key=lambda row: row[table["row_key"]])
        prepared.append({"config": table, "rows": rows, "columns": columns,
                         "original_csv": source_data, "seeds": seeds})
    # Statistical analyses bind their raw predictions and derived results too.
    # The table exporter remains generic; it must not silently ignore this evidence.
    for analysis in config.get("analyses", []):
        directory = inside(root / "results", str(Path(analysis["directory"]).relative_to("results")))
        manifest_path = directory / "manifest.json"
        if sha(manifest_path.read_bytes()) != analysis["manifest_sha256"]:
            raise ValueError(f"Statistical analysis manifest changed: {analysis['id']}")
        proof = json.loads(manifest_path.read_text(encoding="utf-8"))
        if proof["status"] != "completed" or proof["specification"]["protocol"] != analysis["expected_protocol"]:
            raise ValueError(f"Statistical analysis protocol mismatch: {analysis['id']}")
        sources[manifest_path.relative_to(root).as_posix()] = analysis["manifest_sha256"]
        snapshots = analysis.get("source_snapshots", {})
        if not set(snapshots).issubset(proof["source_sha256"]):
            raise ValueError("Unknown statistical source snapshot")
        for path, expected in proof["source_sha256"].items():
            source = inside(root, snapshots.get(path, path))
            if path in snapshots and (not path.endswith(".py") or source.suffix != ".py"
                    or not source.is_relative_to(root / "thesis/drafts")):
                raise ValueError("Only archived analysis code may replace a live source")
            if sha(source.read_bytes()) != expected:
                raise ValueError(f"Statistical prediction source changed: {path}")
            sources[source.relative_to(root).as_posix()] = expected
        for path, expected in proof["output_sha256"].items():
            output = inside(directory, path)
            if sha(output.read_bytes()) != expected:
                raise ValueError(f"Statistical result changed: {path}")
            sources[output.relative_to(root).as_posix()] = expected
    for binding_key in ("method_inventory", "method_clarifications", "chapter_results", "chapter_figures",
                        "event_results", "event_statistics", "deep_baseline_reference"):
        inventory_binding = config.get(binding_key)
        if not inventory_binding:
            continue
        proof_path = inside(root, inventory_binding["proof"])
        if sha(proof_path.read_bytes()) != inventory_binding["sha256"]:
            raise ValueError("Method inventory proof changed")
        inventory = json.loads(proof_path.read_text(encoding="utf-8"))
        if inventory["status"] != "passed":
            raise ValueError("Method inventory is not verified")
        sources[inventory_binding["proof"]] = inventory_binding["sha256"]
        for mapping in (inventory["source_sha256"], inventory["generated_sha256"]):
            for path, expected in mapping.items():
                if sha(inside(root, path).read_bytes()) != expected:
                    raise ValueError(f"Method inventory source changed: {path}")
                sources[path] = expected
    # Refuse a snapshot whose sources changed while being read.
    for path, expected in sources.items():
        if sha((root / path).read_bytes()) != expected:
            raise ValueError(f"Source changed during export: {path}")
    return {"schema_version": 1, "source_sha256": sources}, prepared


def synchronize(root: Path = ROOT, config_path: Path | None = None, check: bool = False) -> dict:
    config_path = config_path or root / "config/thesis_sources.json"
    root = root.resolve()
    config_path = config_path.resolve()
    state, prepared = prepare(root, config_path)
    generated = root / "thesis/latex/generated"
    pointer = generated / "current.json"
    if check:
        if not pointer.is_file():
            raise ValueError("No current snapshot; run scripts/thesis/sync_results.py first")
        previous = json.loads(pointer.read_text(encoding="utf-8"))
        if previous["source_sha256"] != state["source_sha256"]:
            raise ValueError("Result sources changed; regenerate the thesis result snapshot")
        for path, expected in previous["generated_sha256"].items():
            if sha(inside(generated, path).read_bytes()) != expected:
                raise ValueError(f"Generated snapshot has changed: {path}")
        print("PASS: pinned sources and generated thesis tables are current")
        return previous
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = generated / timestamp
    snapshot.mkdir(parents=True, exist_ok=False)
    notes = [r"\section*{來源與指標口徑}",
             "此為來源核對預覽；方法按字母排序，不依 Test 結果選擇方法。",
             "指標以四位小數顯示，原始精度保留於同一快照 CSV；誤差為 seed 標準差。", ""]
    state.update(snapshot=timestamp, generated_at_utc=datetime.now(timezone.utc).isoformat(), tables=[])
    includes = [r"\input{generated/" + timestamp + r"/source-notes.tex}"]
    for item in prepared:
        table = item["config"]
        ident = table["id"]
        (snapshot / (ident + ".tex")).write_text(table_tex(table, item["rows"]),encoding="utf-8")
        (snapshot / (ident + ".source.csv")).write_bytes(item["original_csv"])
        with (snapshot / (ident + ".selected.csv")).open("w",encoding="utf-8",newline="") as f:
            writer = csv.DictWriter(f, fieldnames=item["columns"])
            writer.writeheader()
            writer.writerows(item["rows"])
        state["tables"].append({"id": ident, "run": table["run"], "protocol": table["expected_protocol"],
                                "filters": table["filters"], "metrics": table["metrics"],
                                "seeds": item["seeds"], "row_count": len(item["rows"])})
        notes += [r"\subsection*{" + latex(table["caption"]) + "}", latex(table["context"]),
                  r"\par Run：\texttt{" + latex(Path(table["run"]).name) + r"}；seeds：" +
                  latex(", ".join(map(str, item["seeds"]))) + r"。\par", ""]
        includes.append(r"\input{generated/" + timestamp + "/" + ident + ".tex}")
        print(f"Exported {ident}: {len(item['rows'])} rows, {len(item['seeds'])} seeds")
    (snapshot / "source-notes.tex").write_text("\n".join(notes), encoding="utf-8")
    (generated / "current.tex").write_text("\n".join(includes) + "\n", encoding="utf-8")
    state["generated_sha256"] = {p.relative_to(generated).as_posix(): sha(p.read_bytes())
                                 for p in snapshot.iterdir() if p.is_file()}
    state["generated_sha256"]["current.tex"] = sha((generated / "current.tex").read_bytes())
    (snapshot / "manifest.json").write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    pointer.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if sources or generated tables changed")
    parser.add_argument("--config", type=Path, default=ROOT / "config/thesis_sources.json")
    args = parser.parse_args()
    try:
        synchronize(config_path=args.config,check=args.check)
    except (ValueError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
