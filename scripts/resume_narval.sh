#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"
n="${2:-}"
seed="${3:-}"
variant="${4:-base}"
if [[ "$mode" != pilot && "$mode" != full || -z "$n" || -z "$seed" ]]; then
  echo "Usage: bash scripts/resume_narval.sh pilot|full N SEED [VARIANT]" >&2
  exit 2
fi
if [[ "$mode" == full ]]; then variant=base; fi
cd "$(dirname "$0")/.."
export REPO_DIR="$PWD"
mkdir -p logs
sbatch --parsable --export="ALL,REPO_DIR=$REPO_DIR,MODE=$mode,N=$n,SEED=$seed,VARIANT=$variant" \
  scripts/narval_train.sbatch
