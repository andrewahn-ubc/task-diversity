#!/usr/bin/env bash
#SBATCH --account=rrg-mijungp_gpu
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
#SBATCH --signal=B:USR1@1200
#SBATCH --output=logs/train-%j.out
set -euo pipefail
cd "${REPO_DIR:?}"
module load python/3.13.2
unset PYTHONPATH
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
./.venv/bin/python scripts/run_continual.py \
  --mode "${MODE:?}" --n "${N:?}" --seed "${SEED:?}" \
  --variant "${VARIANT:?}" --max-minutes 40
run_dir="outputs/continual/runs/$MODE/n$(printf '%04d' "$N")/$VARIANT/seed$SEED"
complete=$(./.venv/bin/python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["complete"]))' "$run_dir/progress.json")
if [[ "$complete" == 0 ]]; then
  sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
    --export="ALL,REPO_DIR=$REPO_DIR,MODE=$MODE,N=$N,SEED=$SEED,VARIANT=$VARIANT" \
    scripts/narval_train.sbatch
fi
