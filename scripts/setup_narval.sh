#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
module load python/3.13.2
python_bin="$(command -v python)"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
if ! command -v uv >/dev/null 2>&1; then
  "$python_bin" -m pip install --user --only-binary=:all: uv
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --locked --no-build --no-managed-python --no-python-downloads --python "$python_bin"
uv pip check --python .venv/bin/python
echo "Environment ready: $PWD/.venv"
