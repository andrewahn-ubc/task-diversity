#!/usr/bin/env bash
#SBATCH --account=rrg-mijungp_gpu
#SBATCH --time=00:59:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/prepare-%j.out
set -euo pipefail
cd "${REPO_DIR:?}"
module load python/3.13.2
unset PYTHONPATH
./.venv/bin/python scripts/prepare_continual.py --mode "${MODE:?}" --n "${N:?}"
