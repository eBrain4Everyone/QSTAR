#!/usr/bin/env bash
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
: "${MODE:?MODE required}"
: "${PACKAGE_ROOT:?PACKAGE_ROOT required}"
exec bash "${PACKAGE_ROOT}/scripts/qstar_worker.sh" "${MODE}"
