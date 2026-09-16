#!/usr/bin/env bash
# Submit pilot or full experiment. Each training trajectory chains sub-hour jobs.
set -euo pipefail
mode="${1:-}"
if [[ "$mode" != pilot && "$mode" != full ]]; then
  echo "Usage: bash scripts/submit_narval.sh pilot|full" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
export REPO_DIR="$PWD"
module load python/3.13.2
unset PYTHONPATH
if [[ ! -x .venv/bin/python ]]; then
  echo "Run bash scripts/setup_narval.sh first" >&2
  exit 1
fi
if [[ "$mode" == full ]]; then
  ./.venv/bin/python scripts/select_pilot.py
  ns=(1 4 16 64 256)
  seeds=(0 1 2)
  variants=(base)
else
  ns=(4 64)
  seeds=(0 1)
  variants=(base lower_lr more_entropy more_cbp)
fi
mkdir -p logs
preflight_id=$(sbatch --parsable --export="ALL,REPO_DIR=$REPO_DIR" \
  scripts/narval_preflight.sbatch)
echo "GPU dependency preflight: $preflight_id"
for n in "${ns[@]}"; do
  prep_id=$(sbatch --parsable --dependency="afterok:$preflight_id" \
    --export="ALL,REPO_DIR=$REPO_DIR,MODE=$mode,N=$n" \
    scripts/narval_prepare.sbatch)
  echo "Prepared-data job n=$n: $prep_id"
  for variant in "${variants[@]}"; do
    for seed in "${seeds[@]}"; do
      train_id=$(sbatch --parsable --dependency="afterok:$prep_id" \
        --export="ALL,REPO_DIR=$REPO_DIR,MODE=$mode,N=$n,SEED=$seed,VARIANT=$variant" \
        scripts/narval_train.sbatch)
      echo "Training $mode n=$n variant=$variant seed=$seed: $train_id"
    done
  done
done
