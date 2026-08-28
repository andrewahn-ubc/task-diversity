#!/usr/bin/env bash
#SBATCH --job-name=banyan-learn
#SBATCH --account=rrg-mijungp_gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --signal=USR1@900
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

CONFIG="${BANYAN_CONFIG:-configs/learnability.toml}"
python -m banyan_pilot.preflight \
  --config "$CONFIG" \
  --require-python-311 \
  --require-narval-runtime \
  --require-cuda

ALGORITHMS=(ppo ppo_cbp)
CATALOG_SEEDS=(260600880 260600881)
DIVERSITIES=(1 4)
TRAINING_SEEDS=(0 1)

INDEX="$SLURM_ARRAY_TASK_ID"
TRAINING_SEED="${TRAINING_SEEDS[$((INDEX % ${#TRAINING_SEEDS[@]}))]}"
INDEX=$((INDEX / ${#TRAINING_SEEDS[@]}))
DIVERSITY="${DIVERSITIES[$((INDEX % ${#DIVERSITIES[@]}))]}"
INDEX=$((INDEX / ${#DIVERSITIES[@]}))
CATALOG_SEED="${CATALOG_SEEDS[$((INDEX % ${#CATALOG_SEEDS[@]}))]}"
INDEX=$((INDEX / ${#CATALOG_SEEDS[@]}))
ALGORITHM="${ALGORITHMS[$INDEX]}"

OUTPUT_ROOT="results/learnability/raw/${ALGORITHM}/catalog_${CATALOG_SEED}"
set +e
srun --ntasks=1 python -m banyan_pilot.train \
  --config "$CONFIG" \
  --diversity "$DIVERSITY" \
  --seed "$TRAINING_SEED" \
  --catalog-seed "$CATALOG_SEED" \
  --algorithm "$ALGORITHM" \
  --vary-layout \
  --output-root "$OUTPUT_ROOT" \
  --device cuda
STATUS=$?
set -e

if [[ "$STATUS" -eq 75 ]]; then
  if [[ "${BANYAN_FINAL_CHUNK:?}" == "1" ]]; then
    echo "Checkpoint saved, but the finite ${BANYAN_CHUNK_COUNT:?}-chunk budget was exhausted." >&2
    exit 75
  fi
  echo "Checkpoint saved at chunk ${BANYAN_CHUNK_INDEX:?}/${BANYAN_CHUNK_COUNT:?}; the next corresponding array element will resume it."
  exit 0
fi
exit "$STATUS"
