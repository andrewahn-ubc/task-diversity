#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
module load python/3.13.2
# Alliance's custom PYTHONPATH can suppress external manylinux wheel tags.
unset PYTHONPATH
python_bin="$(command -v python)"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
if [[ -z "${SCRATCH:-}" || ! -d "$SCRATCH" ]]; then
  echo 'SCRATCH is unavailable. Run this on Narval with your scratch directory mounted.' >&2
  exit 1
fi
repo_key="$(printf '%s' "$PWD" | sha256sum | cut -c1-12)"
state_dir="$SCRATCH/task-diversity/$repo_key"
export PYTHONUSERBASE="$state_dir/python-user"
export UV_CACHE_DIR="$state_dir/uv-cache"
export UV_PROJECT_ENVIRONMENT="$state_dir/venv"
export TMPDIR="$state_dir/tmp"
mkdir -p "$PYTHONUSERBASE" "$UV_CACHE_DIR" "$TMPDIR"

# Keep the virtual environment, datasets, checkpoints, and Slurm logs off HOME.
# The project still uses its usual relative paths through these symlinks.
link_to_scratch() {
  local name="$1" target="$2"
  mkdir -p "$(dirname "$target")"
  if [[ "$name" != .venv ]]; then
    mkdir -p "$target"
  fi
  if [[ -L "$name" ]]; then
    if [[ "$(readlink "$name")" != "$target" ]]; then
      echo "$name points elsewhere. Move its data to $target and update the link." >&2
      exit 1
    fi
  elif [[ -e "$name" ]]; then
    echo "$name already exists in the repository. Move it to $target, then link it there." >&2
    exit 1
  else
    ln -s "$target" "$name"
  fi
}
link_to_scratch .venv "$UV_PROJECT_ENVIRONMENT"
link_to_scratch outputs "$state_dir/outputs"
link_to_scratch logs "$state_dir/logs"

uv_bin="$PYTHONUSERBASE/bin/uv"
if [[ ! -x "$uv_bin" ]]; then
  # Alliance's pip defaults to its CVMFS wheelhouse, which has no uv wheel.
  # Install the pinned PyPI wheel into SCRATCH, without touching HOME's pip cache.
  if ! env -u PIP_NO_INDEX -u PIP_FIND_LINKS -u PIP_EXTRA_INDEX_URL \
    PIP_CONFIG_FILE=/dev/null "$python_bin" -m pip install --user --no-cache-dir \
    --only-binary=:all: --index-url https://pypi.org/simple 'uv==0.12.10'; then
    echo "uv bootstrap failed. Check the error above and free space in $SCRATCH if needed." >&2
    exit 1
  fi
  if [[ ! -x "$uv_bin" ]]; then
    echo "uv installed, but its executable was not found at $uv_bin" >&2
    exit 1
  fi
fi
"$uv_bin" sync --locked --no-build --no-managed-python --no-python-downloads --python "$python_bin"
"$uv_bin" pip check --python .venv/bin/python
echo "Environment ready: $PWD/.venv (stored in $UV_PROJECT_ENVIRONMENT)"
echo "Experiment outputs and logs: $state_dir"
