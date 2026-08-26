#!/usr/bin/env bash
#SBATCH --job-name=banyan-aggregate
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$REPO_ROOT"
module --force purge
module load StdEnv/2023 python/3.11
source .venv-narval/bin/activate
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${SLURM_TMPDIR}/matplotlib"
mkdir -p "$MPLCONFIGDIR"
python -m banyan_pilot.aggregate --root results/raw --output results
