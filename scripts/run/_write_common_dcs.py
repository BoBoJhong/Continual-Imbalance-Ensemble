"""Deprecated compatibility check for the shared DCS implementation.

The canonical module is maintained directly at
``experiments/_shared/common_dcs.py``. This former source generator is kept as
a harmless check so old instructions do not overwrite reviewed source code.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = PROJECT_ROOT / "experiments" / "_shared" / "common_dcs.py"


def main() -> int:
    if not TARGET.is_file():
        raise FileNotFoundError(f"Canonical DCS helper is missing: {TARGET}")
    print(f"DCS helper is maintained directly: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
