#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-all}"
TRAIN_CHUNK_TIME="02:00:00"
case "$MODE" in
  all|--preflight-only|--smoke-only|--resume-main) ;;
  *)
    echo "Usage: bash scripts/narval_submit.sh [--preflight-only|--smoke-only|--resume-main]" >&2
    exit 2
    ;;
esac

if ! command -v module >/dev/null 2>&1; then
  echo "The Alliance 'module' command is unavailable. Run this script on Narval." >&2
  exit 2
fi
if ! command -v sbatch >/dev/null 2>&1 && [[ "$MODE" != "--preflight-only" ]]; then
  echo "The 'sbatch' command is unavailable. Run this script on a Narval login node." >&2
  exit 2
fi

# Do not let a caller's user packages or project paths contaminate the Narval
# module stack. This script runs in a subprocess, so these changes do not alter
# the caller's interactive shell.
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
while IFS='=' read -r variable _; do
  case "$variable" in
    PIP_*|VIRTUALENV_*) unset "$variable" ;;
  esac
done < <(env)
export PYTHONNOUSERSITE=1

module --force purge
module load StdEnv/2023 python/3.11

BASE_PYTHON="$(command -v python)"
"$BASE_PYTHON" - <<'PY'
import os
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"Expected Python 3.11 after loading the Narval module, got {sys.version}"
    )
executable = os.path.realpath(sys.executable)
if not executable.startswith("/cvmfs/soft.computecanada.ca/"):
    raise SystemExit(f"Expected Alliance CVMFS Python, got {executable}")
PY

echo "Verifying pinned wheels in Narval's active StdEnv/2023 + Python 3.11 environment..."
avail_wheels -r requirements-narval.txt --not-available

VENV_DIR="$REPO_ROOT/.venv-narval"
VIRTUALENV_APP_DATA="$REPO_ROOT/.virtualenv-app-data"
VENV_STAMP="$VENV_DIR/.banyan-environment-schema"
ENVIRONMENT_SCHEMA="$("$BASE_PYTHON" - requirements-narval.txt <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()
print(f"narval-bootstrap-v2:{digest}")
PY
)"

venv_is_usable() {
  [[ -x "$VENV_DIR/bin/python" ]] &&
    [[ -f "$VENV_STAMP" ]] &&
    [[ "$(<"$VENV_STAMP")" == "$ENVIRONMENT_SCHEMA" ]] &&
    "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import sys
import pip

raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
}

if ! venv_is_usable; then
  echo "Creating or repairing the project-local Python environment..."
  # An isolated app-data directory avoids stale/corrupt virtualenv seed caches
  # in ~/.local/share. --clear also repairs a partially created environment.
  "$BASE_PYTHON" -m virtualenv \
    --no-download \
    --no-periodic-update \
    --app-data "$VIRTUALENV_APP_DATA" \
    --reset-app-data \
    --clear \
    "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INDEX=1
export PIP_NO_CACHE_DIR=1
export PIP_REQUIRE_VIRTUALENV=1

mkdir -p results/environment logs
export MPLCONFIGDIR="$REPO_ROOT/results/environment/matplotlib-cache"
mkdir -p "$MPLCONFIGDIR"

# Resolve the complete dependency closure afresh even when the environment is
# already populated. The JSON report lets us reject an sdist, a user-local
# wheel, or any source outside the Alliance CVMFS wheelhouse before installing.
python -m pip install \
  --dry-run \
  --ignore-installed \
  --no-index \
  --only-binary=:all: \
  --report results/environment/offline-resolution.json \
  --requirement requirements-narval.txt

PYTHONPATH="$REPO_ROOT/src" python -m banyan_pilot.dependency_audit \
  results/environment/offline-resolution.json

python -m pip install \
  --no-index \
  --only-binary=:all: \
  --requirement requirements-narval.txt
python -m pip check

python -m pip freeze --local > results/environment/pip-freeze.txt
python -m pip inspect --local > results/environment/pip-inspect.json
python -m pip config debug > results/environment/pip-config.txt
export PYTHONPATH="$REPO_ROOT/src"
python -m banyan_pilot.preflight \
  --config configs/main.toml \
  --require-python-311 \
  --require-narval-runtime \
  --output results/environment/preflight-login.json
printf '%s\n' "$ENVIRONMENT_SCHEMA" > "$VENV_STAMP"

if [[ "$MODE" == "--preflight-only" ]]; then
  echo "Narval dependency and CPU preflight passed. No jobs submitted."
  exit 0
fi

if [[ "$MODE" == "--resume-main" ]]; then
  MAIN_JOB="$(sbatch --parsable --account=def-mijungp --array=0-24 --time="$TRAIN_CHUNK_TIME" \
    --export=ALL,BANYAN_MODE=main,BANYAN_CONFIG=configs/main.toml,BANYAN_OUTPUT=results/raw-cbp \
    slurm/train_array.sbatch)"
  AGG_JOB="$(sbatch --parsable --account=def-mijungp --dependency="afterok:${MAIN_JOB}" \
    slurm/aggregate.sbatch)"
  echo "Submitted/resumed main array: $MAIN_JOB"
  echo "Submitted dependent aggregation: $AGG_JOB"
  echo "Monitor with: squeue -u $USER -j ${MAIN_JOB},${AGG_JOB}"
  exit 0
fi

SMOKE_JOB="$(sbatch --parsable --account=def-mijungp --array=0-4 --time="$TRAIN_CHUNK_TIME" \
  --export=ALL,BANYAN_MODE=smoke,BANYAN_CONFIG=configs/main.toml,BANYAN_OUTPUT=results/raw-cbp \
  slurm/train_array.sbatch)"
echo "Submitted smoke array: $SMOKE_JOB"

if [[ "$MODE" == "--smoke-only" ]]; then
  echo "Monitor with: squeue -u $USER -j $SMOKE_JOB"
  exit 0
fi

GATE_JOB="$(sbatch --parsable --account=def-mijungp --dependency="afterok:${SMOKE_JOB}" \
  slurm/smoke_gate.sbatch)"
MAIN_JOB="$(sbatch --parsable --account=def-mijungp --array=0-24 --time="$TRAIN_CHUNK_TIME" \
  --dependency="afterok:${GATE_JOB}" \
  --export=ALL,BANYAN_MODE=main,BANYAN_CONFIG=configs/main.toml,BANYAN_OUTPUT=results/raw-cbp \
  slurm/train_array.sbatch)"
AGG_JOB="$(sbatch --parsable --account=def-mijungp --dependency="afterok:${MAIN_JOB}" \
  slurm/aggregate.sbatch)"

cat <<EOF
Submitted smoke gate: $GATE_JOB
Submitted full 25-run array (held until gate passes): $MAIN_JOB
Submitted final aggregation (held until all runs pass): $AGG_JOB
Monitor with: squeue -u $USER -j ${SMOKE_JOB},${GATE_JOB},${MAIN_JOB},${AGG_JOB}
The smoke jobs are the first phase of the five seed-0 main runs; their checkpoints are reused.
Training allocations are two-hour chunks; unfinished array tasks checkpoint and requeue themselves.
If the smoke gate fails, inspect results/raw-cbp/smoke-gate.json; the main budget is not changed automatically.
EOF
