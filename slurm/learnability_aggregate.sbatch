#!/usr/bin/env bash
#SBATCH --job-name=banyan-learn-analysis
#SBATCH --account=def-mijungp_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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
python -m banyan_pilot.learnability \
  --config configs/learnability.toml \
  --root results/learnability/raw \
  --output results/learnability/analysis
