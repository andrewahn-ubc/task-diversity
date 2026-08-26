#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-all}"
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

module --force purge
module load StdEnv/2023 python/3.11

echo "Verifying pinned wheels in Narval's active StdEnv/2023 + Python 3.11 environment..."
avail_wheels -r requirements-narval.txt

if [[ ! -x .venv-narval/bin/python ]]; then
  virtualenv --no-download .venv-narval
fi
source .venv-narval/bin/activate
python -m pip install --no-index --requirement requirements-narval.txt
python -m pip check

mkdir -p results/environment logs
python -m pip freeze --local > results/environment/pip-freeze.txt
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m banyan_pilot.preflight \
  --config configs/main.toml \
  --require-python-311 \
  --output results/environment/preflight-login.json

if [[ "$MODE" == "--preflight-only" ]]; then
  echo "Narval dependency and CPU preflight passed. No jobs submitted."
  exit 0
fi

if [[ "$MODE" == "--resume-main" ]]; then
  MAIN_JOB="$(sbatch --parsable --account=def-mijungp --array=0-14 --time=10:00:00 \
    --export=ALL,BANYAN_MODE=main,BANYAN_CONFIG=configs/main.toml,BANYAN_OUTPUT=results/raw \
    slurm/train_array.sbatch)"
  AGG_JOB="$(sbatch --parsable --account=def-mijungp --dependency="afterok:${MAIN_JOB}" \
    slurm/aggregate.sbatch)"
  echo "Submitted/resumed main array: $MAIN_JOB"
  echo "Submitted dependent aggregation: $AGG_JOB"
  echo "Monitor with: squeue -u $USER -j ${MAIN_JOB},${AGG_JOB}"
  exit 0
fi

SMOKE_JOB="$(sbatch --parsable --account=def-mijungp --array=0-2 --time=01:00:00 \
  --export=ALL,BANYAN_MODE=smoke,BANYAN_CONFIG=configs/smoke.toml,BANYAN_OUTPUT=results/smoke \
  slurm/train_array.sbatch)"
echo "Submitted smoke array: $SMOKE_JOB"

if [[ "$MODE" == "--smoke-only" ]]; then
  echo "Monitor with: squeue -u $USER -j $SMOKE_JOB"
  exit 0
fi

GATE_JOB="$(sbatch --parsable --account=def-mijungp --dependency="afterok:${SMOKE_JOB}" \
  slurm/smoke_gate.sbatch)"
MAIN_JOB="$(sbatch --parsable --account=def-mijungp --array=0-14 --time=10:00:00 \
  --dependency="afterok:${GATE_JOB}" \
  --export=ALL,BANYAN_MODE=main,BANYAN_CONFIG=configs/main.toml,BANYAN_OUTPUT=results/raw \
  slurm/train_array.sbatch)"
AGG_JOB="$(sbatch --parsable --account=def-mijungp --dependency="afterok:${MAIN_JOB}" \
  slurm/aggregate.sbatch)"

cat <<EOF
Submitted smoke gate: $GATE_JOB
Submitted full 15-run array (held until gate passes): $MAIN_JOB
Submitted final aggregation (held until all runs pass): $AGG_JOB
Monitor with: squeue -u $USER -j ${SMOKE_JOB},${GATE_JOB},${MAIN_JOB},${AGG_JOB}
If the smoke gate fails, inspect results/smoke/gate.json; the main budget is not changed automatically.
EOF
