#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="${PACKAGE_ROOT:-/scratch/sr7849/QC_June28/ICASSP_multiseed_new_datasets}"
for dataset in cifar10 kmnist svhn; do
  echo "===== ${dataset} ====="
  bash "${PACKAGE_ROOT}/scripts/submit_dataset.sh" "${dataset}"
done
