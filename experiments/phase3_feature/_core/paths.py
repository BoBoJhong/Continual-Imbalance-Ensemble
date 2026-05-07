"""Phase 3 — repo root and per-method results layout (mutual_info / shap / rfe)."""
from __future__ import annotations

from pathlib import Path

METHOD_SUBDIRS = ("mutual_info", "shap", "rfe")


def repo_root_from(file: Path) -> Path:
    """Repo root from a file under experiments/phase3_feature/<method>/script.py."""
    return file.resolve().parent.parent.parent.parent


def results_phase3(project_root: Path) -> Path:
    return project_root / "results" / "phase3_feature"


def results_method_dir(project_root: Path, method: str) -> Path:
    if method not in METHOD_SUBDIRS:
        raise ValueError(f"method must be one of {METHOD_SUBDIRS}, got {method!r}")
    d = results_phase3(project_root) / method
    d.mkdir(parents=True, exist_ok=True)
    return d
