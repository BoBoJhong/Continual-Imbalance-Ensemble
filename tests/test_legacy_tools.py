"""Historical tools import without reading research data or writing artifacts."""
import hashlib
import json
import runpy
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "scripts/legacy/MIGRATION_MANIFEST.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("entry", MANIFEST["entries"], ids=lambda entry: entry["original_path"])
def test_archived_tool_import_is_read_only_and_resolves_project_root(entry, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Import must not read experiment tables, write files, or move artifacts")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    monkeypatch.setattr(pd.DataFrame, "to_csv", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Workbook, "save", forbidden)
    monkeypatch.setattr(plt, "savefig", forbidden)
    monkeypatch.setattr(Figure, "savefig", forbidden)
    monkeypatch.setattr(shutil, "move", forbidden)
    monkeypatch.setattr(shutil, "copy2", forbidden)
    archived = ROOT / entry["archived_path"]
    assert not (ROOT / entry["original_path"]).exists()
    namespace = runpy.run_path(str(archived), run_name="legacy_import_check")
    assert namespace["LEGACY_PROJECT_ROOT"] == ROOT
    for name in ("ROOT", "root", "project_root", "PROJECT_ROOT"):
        if name in namespace:
            assert Path(namespace[name]).resolve() == ROOT
    for name in ("OUTPUT_DIR", "OUT_DIR", "OUT_PATH", "out_dir", "RESULT_DIR", "TMP_OUT", "THESIS_OUT"):
        if name in namespace:
            assert Path(namespace[name]).resolve().is_relative_to(ROOT / "results")


def test_original_tool_source_zip_matches_every_migration_hash():
    with zipfile.ZipFile(ROOT / MANIFEST["original_source_archive"]) as archive:
        assert set(archive.namelist()) == {e["original_path"] for e in MANIFEST["entries"]}
        for entry in MANIFEST["entries"]:
            assert hashlib.sha256(archive.read(entry["original_path"])).hexdigest() == entry["original_sha256"]
