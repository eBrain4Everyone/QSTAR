#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: resume_to_full_10epochs.sh DATASET RUN_ROOT}"
RUN_ROOT="${2:?usage: resume_to_full_10epochs.sh DATASET RUN_ROOT}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/sr7849/QC_June28}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/ICASSP_multiseed_new_datasets}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_ROOT}/ICASSP_multiseed_new_datasets}"
PARTITION="${PARTITION:-compute}"
SEEDS_CSV="${SEEDS_CSV:-1:42:48:550:2026}"
FEATURE_CACHE_DIR="${WORK_ROOT}/cache/${DATASET}"
LOG_DIR="${RUN_ROOT}/logs_resume_full"
ARRAY_SBATCH="${PACKAGE_ROOT}/slurm/qstar_array.sbatch"
mkdir -p "${LOG_DIR}"
base_export="ALL,PACKAGE_ROOT=${PACKAGE_ROOT},PROJECT_ROOT=${PROJECT_ROOT},RUN_ROOT=${RUN_ROOT},DATASET=${DATASET},FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR},SEEDS_CSV=${SEEDS_CSV}"

submit_array() {
  local name="$1" mode="$2" array="$3" time="$4" mem="$5"
  sbatch --parsable --job-name="${name}" --partition="${PARTITION}" --array="${array}" --time="${time}" --mem="${mem}" \
    --output="${LOG_DIR}/${name}-%A_%a.out" --error="${LOG_DIR}/${name}-%A_%a.err" \
    --export="${base_export},MODE=${mode},FORCE_FULL=1,DISABLE_FLASHJT=1,DISABLE_EARLY_STOPPING=1" "${ARRAY_SBATCH}"
}

j1=$(submit_array "qstar-${DATASET}-full-classical" classical "3-4%2" 04:00:00 16G)
j2=$(submit_array "qstar-${DATASET}-full-standard" standard "24-39%8" 1-12:00:00 32G)
j3=$(submit_array "qstar-${DATASET}-full-ketgpt" ketgpt "3-4%2" 2-12:00:00 32G)
j4=$(submit_array "qstar-${DATASET}-full-a180" adaptive180 "3-4%2" 2-00:00:00 32G)
j5=$(submit_array "qstar-${DATASET}-full-a160" adaptive160 "3-4%2" 2-00:00:00 32G)
dep="${j1}:${j2}:${j3}:${j4}:${j5}"
summary=$(sbatch --parsable --job-name="qstar-${DATASET}-final-summary" --partition="${PARTITION}" --time=00:30:00 --cpus-per-task=1 --mem=8G \
  --dependency="afterok:${dep}" --output="${LOG_DIR}/summary-%j.out" --error="${LOG_DIR}/summary-%j.err" \
  --export="${base_export},MODE=summarize" "${ARRAY_SBATCH}")

echo "Resume jobs: ${dep}"
echo "Final summary: ${summary}"
