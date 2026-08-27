#!/usr/bin/env bash
#SBATCH --job-name=banyan-pilot
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$REPO_ROOT"
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1
module --force purge
module load StdEnv/2023 python/3.11
source "$REPO_ROOT/.venv-narval/bin/activate"
export PYTHONPATH="$REPO_ROOT/src"
export MPLCONFIGDIR="${SLURM_TMPDIR}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

python -m banyan_pilot.preflight \
  --config "${BANYAN_CONFIG:?}" \
  --require-python-311 \
  --require-narval-runtime \
  --require-cuda

DIVERSITIES=(1 4 16 64 256)
if [[ "${BANYAN_MODE:?}" == "smoke" ]]; then
  DIVERSITY="${DIVERSITIES[$SLURM_ARRAY_TASK_ID]}"
  SEED=0
  EXTRA_TRAIN_ARGS=(--stop-after-distributions 1)
else
  DIVERSITY_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
  SEED=$((SLURM_ARRAY_TASK_ID % 5))
  DIVERSITY="${DIVERSITIES[$DIVERSITY_INDEX]}"
  EXTRA_TRAIN_ARGS=()
fi

set +e
srun --ntasks=1 python -m banyan_pilot.train \
  --config "$BANYAN_CONFIG" \
  --diversity "$DIVERSITY" \
  --seed "$SEED" \
  --output-root "${BANYAN_OUTPUT:?}" \
  --device cuda \
  "${EXTRA_TRAIN_ARGS[@]}"
STATUS=$?
set -e

if [[ "$STATUS" -eq 75 ]]; then
  echo "Checkpoint saved; requeuing ${SLURM_JOB_ID}."
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$STATUS"
