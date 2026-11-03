#!/usr/bin/env sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$script_dir/thesis.py" build --backend "${1:-local}" --fonts "${2:-strict}" --document "${3:-main}"
