#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-all}"
TRAIN_CHUNK_TIME="01:00:00"
SMOKE_CHUNK_COUNT=8
MAIN_CHUNK_COUNT=21
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
if [[ "$MODE" != "--preflight-only" ]]; then
  for slurm_command in sbatch scancel; do
    if ! command -v "$slurm_command" >/dev/null 2>&1; then
      echo "The '$slurm_command' command is unavailable. Run this script on a Narval login node." >&2
      exit 2
    fi
  done
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

PIPELINE_JOB_IDS=()
SUBMISSION_COMPLETE=0
CHAIN_LAST_JOB=""
CHAIN_JOB_IDS=""
SINGLE_JOB=""

cleanup_partial_submission() {
  local status=$?
  if ((status != 0 && SUBMISSION_COMPLETE == 0 && ${#PIPELINE_JOB_IDS[@]} > 0)); then
    echo "Submission failed; cancelling the partially submitted pipeline: ${PIPELINE_JOB_IDS[*]}" >&2
    scancel "${PIPELINE_JOB_IDS[@]}" || true
  fi
}
trap cleanup_partial_submission EXIT

submit_training_chain() {
  local training_mode="$1"
  local array_spec="$2"
  local chunk_count="$3"
  local initial_dependency="$4"
  local previous_job="$initial_dependency"
  local chunk raw_job job_id final_chunk dependency_kind
  local -a chain_ids=()
  local -a submit_args

  for ((chunk = 1; chunk <= chunk_count; chunk++)); do
    final_chunk=0
    if ((chunk == chunk_count)); then
      final_chunk=1
    fi
    submit_args=(
      --parsable
      --account=def-mijungp
      --array="$array_spec"
      --time="$TRAIN_CHUNK_TIME"
      --job-name="banyan-${training_mode}-c${chunk}"
    )
    if [[ -n "$previous_job" ]]; then
      dependency_kind=aftercorr
      if ((chunk == 1)); then
        dependency_kind=afterok
      fi
      submit_args+=(--dependency="${dependency_kind}:${previous_job}")
    fi
    raw_job="$(sbatch "${submit_args[@]}" \
      --export="ALL,BANYAN_MODE=${training_mode},BANYAN_CONFIG=configs/main.toml,BANYAN_OUTPUT=results/raw-cbp,BANYAN_CHUNK_INDEX=${chunk},BANYAN_CHUNK_COUNT=${chunk_count},BANYAN_FINAL_CHUNK=${final_chunk}" \
      slurm/train_array.sbatch)"
    job_id="${raw_job%%;*}"
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
      echo "Could not parse sbatch job ID from: $raw_job" >&2
      exit 2
    fi
    chain_ids+=("$job_id")
    PIPELINE_JOB_IDS+=("$job_id")
    previous_job="$job_id"
  done

  CHAIN_LAST_JOB="$previous_job"
  local IFS=,
  CHAIN_JOB_IDS="${chain_ids[*]}"
}

submit_single_job() {
  local dependency="$1"
  local script="$2"
  local raw_job job_id
  raw_job="$(sbatch --parsable --account=def-mijungp \
    --dependency="afterok:${dependency}" "$script")"
  job_id="${raw_job%%;*}"
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Could not parse sbatch job ID from: $raw_job" >&2
    exit 2
  fi
  PIPELINE_JOB_IDS+=("$job_id")
  SINGLE_JOB="$job_id"
}

monitor_command() {
  local IFS=,
  printf 'squeue -u %s -j %s\n' "$USER" "${PIPELINE_JOB_IDS[*]}"
}

if [[ "$MODE" == "--resume-main" ]]; then
  submit_training_chain main 0-24 "$MAIN_CHUNK_COUNT" ""
  MAIN_JOB_IDS="$CHAIN_JOB_IDS"
  MAIN_LAST_JOB="$CHAIN_LAST_JOB"
  submit_single_job "$MAIN_LAST_JOB" slurm/aggregate.sbatch
  AGG_JOB="$SINGLE_JOB"
  echo "Submitted $MAIN_CHUNK_COUNT one-hour main arrays: $MAIN_JOB_IDS"
  echo "Submitted dependent aggregation: $AGG_JOB"
  echo "Monitor with: $(monitor_command)"
  SUBMISSION_COMPLETE=1
  exit 0
fi

submit_training_chain smoke 0-4 "$SMOKE_CHUNK_COUNT" ""
SMOKE_JOB_IDS="$CHAIN_JOB_IDS"
SMOKE_LAST_JOB="$CHAIN_LAST_JOB"
echo "Submitted $SMOKE_CHUNK_COUNT one-hour smoke arrays: $SMOKE_JOB_IDS"

if [[ "$MODE" == "--smoke-only" ]]; then
  echo "Monitor with: $(monitor_command)"
  SUBMISSION_COMPLETE=1
  exit 0
fi

submit_single_job "$SMOKE_LAST_JOB" slurm/smoke_gate.sbatch
GATE_JOB="$SINGLE_JOB"
submit_training_chain main 0-24 "$MAIN_CHUNK_COUNT" "$GATE_JOB"
MAIN_JOB_IDS="$CHAIN_JOB_IDS"
MAIN_LAST_JOB="$CHAIN_LAST_JOB"
submit_single_job "$MAIN_LAST_JOB" slurm/aggregate.sbatch
AGG_JOB="$SINGLE_JOB"

cat <<EOF
Submitted smoke gate: $GATE_JOB
Submitted $MAIN_CHUNK_COUNT one-hour full-sweep arrays (held until the gate passes): $MAIN_JOB_IDS
Submitted final aggregation (held until the final main chunk succeeds): $AGG_JOB
Monitor with: $(monitor_command)
The eight smoke chunks allow 6 hours before checkpoint signals, exceeding the 5-hour-30-minute 1.5x target.
The twenty-one main chunks allow 15 hours 45 minutes before checkpoint signals, exceeding the 15-hour 1.5x target.
Every chunk is a distinct SLURM array job with a maximum one-hour time limit.
The smoke jobs are the first phase of the five seed-0 main runs; their checkpoints are reused.
If the smoke gate fails, inspect results/raw-cbp/smoke-gate.json; the main budget is not changed automatically.
EOF
SUBMISSION_COMPLETE=1
