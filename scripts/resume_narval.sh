#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"
n="${2:-}"
seed="${3:-}"
variant="${4:-}"
if [[ "$mode" != pilot && "$mode" != full || -z "$n" || -z "$seed" ]]; then
  echo "Usage: bash scripts/resume_narval.sh pilot|full N SEED [VARIANT]" >&2
  exit 2
fi
if [[ -z "$variant" ]]; then
  if [[ "$mode" == full ]]; then
    variant=selected
  else
    variant=base
  fi
fi
if [[ "$variant" != base && "$variant" != lower_lr &&
      "$variant" != more_entropy && "$variant" != more_cbp &&
      ! ( "$mode" == full && "$variant" == selected ) ]]; then
  echo "Invalid $mode variant: $variant" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
export REPO_DIR="$PWD"
mkdir -p logs
sbatch --parsable --export="ALL,REPO_DIR=$REPO_DIR,MODE=$mode,N=$n,SEED=$seed,VARIANT=$variant" \
  scripts/narval_train.sbatch
