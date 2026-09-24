#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail
: "${PACKAGE_ROOT:?}"
: "${DATASET:?}"
: "${FEATURE_CACHE_DIR:?}"
: "${DATA_ROOT:?}"
source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-cu
export PYTHONPATH="${PACKAGE_ROOT}:${PYTHONPATH:-}"
python "${PACKAGE_ROOT}/scripts/prepare_feature_cache.py" \
  --dataset "${DATASET}" \
  --data-root "${DATA_ROOT}" \
  --output-dir "${FEATURE_CACHE_DIR}" \
  --train-subset 5000 --test-subset 1000 \
  --batch-size 64 --num-workers 4 --download
