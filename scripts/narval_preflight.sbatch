#!/usr/bin/env bash
#SBATCH --account=rrg-mijungp_gpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/preflight-%j.out
set -euo pipefail
cd "${REPO_DIR:?}"
module load python/3.13.2
unset PYTHONPATH
export JAX_PLATFORMS=cuda
./.venv/bin/python scripts/check_narval.py
