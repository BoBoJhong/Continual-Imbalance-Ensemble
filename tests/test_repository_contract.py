import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _locked_requirements() -> set[str]:
    return {
        line.strip().lower()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_pyproject_dependencies_are_locked_in_requirements() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    locked = _locked_requirements()
    declared = {dependency.lower() for dependency in project["dependencies"]}
    for group in project.get("optional-dependencies", {}).values():
        declared.update(dependency.lower() for dependency in group)

    assert declared <= locked
    assert project["requires-python"] == ">=3.11,<3.15"


def test_single_requirements_entry_point() -> None:
    assert sorted(path.name for path in ROOT.glob("requirements*.txt")) == [
        "requirements.txt"
    ]


def test_required_research_documents_exist() -> None:
    docs = ROOT / "docs"
    assert (docs / "研究方向.md").is_file()
    assert (docs / "研究成果與未來研究方向報告.md").is_file()
