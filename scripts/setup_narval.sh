#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
module load python/3.13.2
# Alliance's custom PYTHONPATH can suppress external manylinux wheel tags.
unset PYTHONPATH
python_bin="$(command -v python)"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
else
  # Alliance sets pip to --no-index and its CVMFS wheelhouse has no uv wheel.
  # Override that setting for this one bootstrap install on the login node.
  if ! env -u PIP_NO_INDEX -u PIP_FIND_LINKS -u PIP_EXTRA_INDEX_URL \
    PIP_CONFIG_FILE=/dev/null "$python_bin" -m pip install --user \
    --only-binary=:all: --index-url https://pypi.org/simple 'uv==0.12.10'; then
    echo "Could not install uv from PyPI. Check PyPI access on this login node." >&2
    exit 1
  fi
  user_base="$("$python_bin" -m site --user-base)"
  uv_bin="$user_base/bin/uv"
  if [[ ! -x "$uv_bin" ]]; then
    echo "uv installed, but its executable was not found at $uv_bin" >&2
    exit 1
  fi
fi
"$uv_bin" sync --locked --no-build --no-managed-python --no-python-downloads --python "$python_bin"
"$uv_bin" pip check --python .venv/bin/python
echo "Environment ready: $PWD/.venv"
