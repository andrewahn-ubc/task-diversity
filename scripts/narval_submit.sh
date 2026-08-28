#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-all}"
TRAIN_CHUNK_TIME="01:00:00"
LEARNABILITY_CHUNK_COUNT=4
GPU_ACCOUNT="rrg-mijungp_gpu"
CPU_ACCOUNT="def-mijungp_cpu"
case "$MODE" in
  all|--preflight-only|--resume) ;;
  *)
    echo "Usage: bash scripts/narval_submit.sh [--preflight-only|--resume]" >&2
    exit 2
    ;;
esac

if ! command -v module >/dev/null 2>&1; then
  echo "The Alliance 'module' command is unavailable. Run this script on Narval." >&2
  exit 2
fi
if ! command -v sshare >/dev/null 2>&1; then
  echo "The 'sshare' command is unavailable. Run this script on a Narval login node." >&2
  exit 2
fi
if [[ "$MODE" != "--preflight-only" ]]; then
  for slurm_command in sbatch scancel; do
    if ! command -v "$slurm_command" >/dev/null 2>&1; then
      echo "The '$slurm_command' command is unavailable. Run this script on Narval." >&2
      exit 2
    fi
  done
fi

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
    raise SystemExit(f"Expected Python 3.11, got {sys.version}")
if not os.path.realpath(sys.executable).startswith("/cvmfs/soft.computecanada.ca/"):
    raise SystemExit(f"Expected Alliance CVMFS Python, got {sys.executable}")
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
import pip
import sys

raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
}

if ! venv_is_usable; then
  echo "Creating or repairing the project-local Python environment..."
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
  --config configs/learnability.toml \
  --require-python-311 \
  --require-narval-runtime \
  --output results/environment/preflight-login.json
printf '%s\n' "$ENVIRONMENT_SCHEMA" > "$VENV_STAMP"

ACCOUNT_REPORT="$(sshare -l -U "$USER")"
for required_account in "$GPU_ACCOUNT" "$CPU_ACCOUNT"; do
  if ! awk -v expected="$required_account" \
    '$1 == expected { found = 1 } END { exit(found ? 0 : 1) }' \
    <<<"$ACCOUNT_REPORT"; then
    echo "Required Narval association '$required_account' is unavailable for $USER." >&2
    echo "$ACCOUNT_REPORT" >&2
    exit 2
  fi
done
echo "Verified Narval accounts: GPU=$GPU_ACCOUNT CPU=$CPU_ACCOUNT"

if [[ "$MODE" == "--preflight-only" ]]; then
  echo "Narval dependency and CPU preflight passed. No jobs submitted."
  exit 0
fi

PIPELINE_JOB_IDS=()
SUBMISSION_COMPLETE=0
cleanup_partial_submission() {
  local status=$?
  if ((status != 0 && SUBMISSION_COMPLETE == 0 && ${#PIPELINE_JOB_IDS[@]} > 0)); then
    echo "Submission failed; cancelling partial pipeline: ${PIPELINE_JOB_IDS[*]}" >&2
    scancel "${PIPELINE_JOB_IDS[@]}" || true
  fi
}
trap cleanup_partial_submission EXIT

previous_job=""
chain_ids=()
for ((chunk = 1; chunk <= LEARNABILITY_CHUNK_COUNT; chunk++)); do
  final_chunk=0
  if ((chunk == LEARNABILITY_CHUNK_COUNT)); then
    final_chunk=1
  fi
  submit_args=(
    --parsable
    --account="$GPU_ACCOUNT"
    --array=0-15
    --time="$TRAIN_CHUNK_TIME"
    --job-name="banyan-learn-c${chunk}"
  )
  if [[ -n "$previous_job" ]]; then
    submit_args+=(--dependency="aftercorr:${previous_job}")
  fi
  raw_job="$(sbatch "${submit_args[@]}" \
    --export="ALL,BANYAN_CONFIG=configs/learnability.toml,BANYAN_CHUNK_INDEX=${chunk},BANYAN_CHUNK_COUNT=${LEARNABILITY_CHUNK_COUNT},BANYAN_FINAL_CHUNK=${final_chunk}" \
    slurm/learnability_array.sbatch)"
  job_id="${raw_job%%;*}"
  if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Could not parse sbatch job ID from: $raw_job" >&2
    exit 2
  fi
  PIPELINE_JOB_IDS+=("$job_id")
  chain_ids+=("$job_id")
  previous_job="$job_id"
done

raw_aggregate="$(sbatch --parsable --account="$CPU_ACCOUNT" \
  --dependency="afterok:${previous_job}" slurm/learnability_aggregate.sbatch)"
aggregate_job="${raw_aggregate%%;*}"
if [[ ! "$aggregate_job" =~ ^[0-9]+$ ]]; then
  echo "Could not parse aggregation job ID from: $raw_aggregate" >&2
  exit 2
fi
PIPELINE_JOB_IDS+=("$aggregate_job")

IFS=,
cat <<EOF
Submitted ${LEARNABILITY_CHUNK_COUNT} one-hour learnability arrays: ${chain_ids[*]}
Submitted dependent analysis: $aggregate_job
Monitor with: squeue -u $USER -j ${PIPELINE_JOB_IDS[*]}
This runs 16 independent one-distribution conditions and launches no continual jobs.
Completed checkpoints are idempotent; --resume safely submits the same finite chain.
EOF
SUBMISSION_COMPLETE=1
