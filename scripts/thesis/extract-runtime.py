"""Extract the fixed official TinyTeX archive into this project's private tool directory."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

root = Path(__file__).resolve().parents[2]
lock = json.loads((root / "config/thesis_environment.lock.json").read_text(encoding="utf-8"))
tools = root / ".tools/thesis"
archive = tools / "TinyTeX-v2026.10.zip"
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
if digest != lock["windows_tinytex_sha256"]:
    raise SystemExit("TinyTeX snapshot checksum mismatch")
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        relative = Path(info.filename)
        target = (tools / relative).resolve()
        if relative.is_absolute() or not relative.parts or relative.parts[0] != "TinyTeX":
            raise SystemExit("Unexpected archive root")
        if not target.is_relative_to(tools.resolve()):
            raise SystemExit("Archive path escapes tool directory")
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(info) as source, target.open("wb") as dest:
            shutil.copyfileobj(source, dest)
(tools / "TinyTeX/.snapshot-sha256").write_text(digest + "\n", encoding="ascii")
print("Fixed TinyTeX snapshot extracted into", tools)
