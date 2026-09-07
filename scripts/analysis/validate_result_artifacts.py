"""Audit CSV result artifacts for readability, shape, and reproducibility fields."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def audit_results(root: Path) -> dict:
    """Return a compact, JSON-serializable profile of all CSV files under root."""
    root = root.resolve()
    csv_files = sorted(root.rglob("*.csv"))
    schemas: Counter[tuple[str, ...]] = Counter()
    issues: list[dict] = []
    total_rows = 0
    files_with_nulls = 0
    files_with_duplicates = 0
    files_with_seed = 0

    for path in csv_files:
        relative_path = str(path.relative_to(root))
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            issues.append({"path": relative_path, "issue": "empty_csv"})
            continue
        except Exception as exc:
            issues.append(
                {
                    "path": relative_path,
                    "issue": "unreadable_csv",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        total_rows += len(frame)
        schemas[tuple(str(column) for column in frame.columns)] += 1
        if frame.isna().any().any():
            files_with_nulls += 1
        if frame.duplicated().any():
            files_with_duplicates += 1
        if {str(column).lower() for column in frame.columns} & {"seed", "random_seed"}:
            files_with_seed += 1

    top_schemas = [
        {"files": count, "columns": list(columns)}
        for columns, count in schemas.most_common(10)
    ]
    return {
        "root": str(root),
        "csv_files": len(csv_files),
        "total_rows": total_rows,
        "schema_count": len(schemas),
        "files_with_nulls": files_with_nulls,
        "files_with_exact_duplicates": files_with_duplicates,
        "files_with_seed_column": files_with_seed,
        "issues": issues,
        "top_schemas": top_schemas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when an empty or unreadable CSV is found",
    )
    args = parser.parse_args()

    report = audit_results(args.root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if args.strict and report["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
