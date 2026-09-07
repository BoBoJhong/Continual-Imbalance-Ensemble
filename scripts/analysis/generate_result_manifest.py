"""Generate a provenance and checksum manifest for tracked result artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = RESULTS_ROOT / "RESULT_MANIFEST.json"
PACKAGES = (
    "imbalanced-learn",
    "lightgbm",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "xgboost",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={PROJECT_ROOT.as_posix()}",
            *args,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _artifact(path: Path, root: Path) -> dict:
    record = {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            record["csv"] = {"readable": False, "reason": "empty_csv"}
        except Exception as exc:
            record["csv"] = {
                "readable": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        else:
            record["csv"] = {
                "readable": True,
                "rows": len(frame),
                "columns": [str(column) for column in frame.columns],
                "null_cells": int(frame.isna().sum().sum()),
                "exact_duplicate_rows": int(frame.duplicated().sum()),
            }
    return record


def build_manifest() -> dict:
    result_files = sorted(
        path
        for path in RESULTS_ROOT.rglob("*")
        if path.is_file() and path.resolve() != OUTPUT_PATH.resolve()
    )
    raw_files = sorted(path for path in DATA_ROOT.rglob("*") if path.is_file())
    status = _git("status", "--porcelain")
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/analysis/generate_result_manifest.py",
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(status) if status is not None else None,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: importlib.metadata.version(package)
                for package in PACKAGES
                if _package_exists(package)
            },
        },
        "raw_data": [_artifact(path, DATA_ROOT) for path in raw_files],
        "results": [_artifact(path, RESULTS_ROOT) for path in result_files],
    }


def _package_exists(package: str) -> bool:
    try:
        importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    OUTPUT_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {OUTPUT_PATH} with {len(manifest['results'])} result files "
        f"and {len(manifest['raw_data'])} raw-data files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
