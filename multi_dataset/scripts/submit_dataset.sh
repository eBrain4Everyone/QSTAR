#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: submit_dataset.sh DATASET [RUN_TAG]}"
RUN_TAG="${2:-run_$(date -u +%Y%m%d_%H%M%S)}"
case "${DATASET}" in cifar10|kmnist|svhn|fashionmnist) ;; *) echo "Unsupported dataset ${DATASET}" >&2; exit 2;; esac

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/sr7849/QC_June28}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/ICASSP_multiseed_new_datasets}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_ROOT}/ICASSP_multiseed_new_datasets}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
PARTITION="${PARTITION:-compute}"
SEEDS_CSV="${SEEDS_CSV:-1:42:48:550:2026}"
RUN_ROOT="${WORK_ROOT}/runs/${DATASET}/${RUN_TAG}"
FEATURE_CACHE_DIR="${WORK_ROOT}/cache/${DATASET}"
LOG_DIR="${RUN_ROOT}/logs"
ARRAY_SBATCH="${PACKAGE_ROOT}/slurm/qstar_array.sbatch"
CACHE_SBATCH="${PACKAGE_ROOT}/slurm/prepare_cache.sbatch"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}/manifests" "${RUN_ROOT}/summary" \
  "${RUN_ROOT}/table1/classical" "${RUN_ROOT}/table1/standard_qtl" "${RUN_ROOT}/table1/ketgpt" \
  "${RUN_ROOT}/table2/adaptive160" "${RUN_ROOT}/table2/adaptive180" "${RUN_ROOT}/checkpoints"

base_export="ALL,PACKAGE_ROOT=${PACKAGE_ROOT},PROJECT_ROOT=${PROJECT_ROOT},RUN_ROOT=${RUN_ROOT},DATASET=${DATASET},FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR},SEEDS_CSV=${SEEDS_CSV},DATA_ROOT=${DATA_ROOT}"

cache_job=$(sbatch --parsable --job-name="qstar-${DATASET}-cache" --partition="${PARTITION}" \
  --output="${LOG_DIR}/cache-%j.out" --error="${LOG_DIR}/cache-%j.err" \
  --export="${base_export}" "${CACHE_SBATCH}")

submit_array() {
  local name="$1" mode="$2" array="$3" time="$4" mem="$5" dep="$6"
  local dep_args=()
  [[ -n "${dep}" ]] && dep_args=(--dependency="afterok:${dep}")
  sbatch --parsable --job-name="${name}" --partition="${PARTITION}" --array="${array}" --time="${time}" --mem="${mem}" \
    --output="${LOG_DIR}/${name}-%A_%a.out" --error="${LOG_DIR}/${name}-%A_%a.err" \
    "${dep_args[@]}" --export="${base_export},MODE=${mode},FORCE_FULL=0,DISABLE_FLASHJT=0,DISABLE_EARLY_STOPPING=0" "${ARRAY_SBATCH}"
}

# Phase A: three guaranteed full-history seeds. FlashJT code is present but cannot stop these seed IDs.
p1_classical=$(submit_array "qstar-${DATASET}-p1-classical" classical "0-2%3" 04:00:00 16G "${cache_job}")
p1_standard=$(submit_array "qstar-${DATASET}-p1-standard" standard "0-23%8" 1-12:00:00 32G "${cache_job}")
p1_ketgpt=$(submit_array "qstar-${DATASET}-p1-ketgpt" ketgpt "0-2%3" 2-12:00:00 32G "${cache_job}")
p1_a180=$(submit_array "qstar-${DATASET}-p1-a180" adaptive180 "0-2%3" 2-00:00:00 32G "${cache_job}")
p1_a160=$(submit_array "qstar-${DATASET}-p1-a160" adaptive160 "0-2%3" 2-00:00:00 32G "${cache_job}")

phase1_dep="${p1_classical}:${p1_standard}:${p1_ketgpt}:${p1_a180}:${p1_a160}"

# Phase B: seeds 550/2026. They start only after all Phase-A histories exist.
p2_classical=$(submit_array "qstar-${DATASET}-p2-classical" classical "3-4%2" 04:00:00 16G "${phase1_dep}")
p2_standard=$(submit_array "qstar-${DATASET}-p2-standard" standard "24-39%8" 1-12:00:00 32G "${phase1_dep}")
p2_ketgpt=$(submit_array "qstar-${DATASET}-p2-ketgpt" ketgpt "3-4%2" 2-12:00:00 32G "${phase1_dep}")
p2_a180=$(submit_array "qstar-${DATASET}-p2-a180" adaptive180 "3-4%2" 2-00:00:00 32G "${phase1_dep}")
p2_a160=$(submit_array "qstar-${DATASET}-p2-a160" adaptive160 "3-4%2" 2-00:00:00 32G "${phase1_dep}")

phase2_dep="${p2_classical}:${p2_standard}:${p2_ketgpt}:${p2_a180}:${p2_a160}"
summary_job=$(sbatch --parsable --job-name="qstar-${DATASET}-summary" --partition="${PARTITION}" --time=00:30:00 --cpus-per-task=1 --mem=8G \
  --dependency="afterok:${phase2_dep}" --output="${LOG_DIR}/summary-%j.out" --error="${LOG_DIR}/summary-%j.err" \
  --export="${base_export},MODE=summarize" "${ARRAY_SBATCH}")

cat > "${RUN_ROOT}/run_manifest.txt" <<EOF
project_root=${PROJECT_ROOT}
package_root=${PACKAGE_ROOT}
run_root=${RUN_ROOT}
dataset=${DATASET}
feature_cache_dir=${FEATURE_CACHE_DIR}
seeds=${SEEDS_CSV}
full_history_seeds=1:42:48
flashjt_assisted_seeds=550:2026
ketgpt_candidates=160:180
standard_qtl_grid=q4d1:q4d2:q4d4:q6d1:q6d2:q6d4:q8d1:q8d2
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
std_definition=sample standard deviation (ddof=1)
cache_job=${cache_job}
phase1_classical=${p1_classical}
phase1_standard=${p1_standard}
phase1_ketgpt=${p1_ketgpt}
phase1_adaptive180=${p1_a180}
phase1_adaptive160=${p1_a160}
phase2_classical=${p2_classical}
phase2_standard=${p2_standard}
phase2_ketgpt=${p2_ketgpt}
phase2_adaptive180=${p2_a180}
phase2_adaptive160=${p2_a160}
summary_job=${summary_job}
EOF

echo "RUN_ROOT=${RUN_ROOT}"
echo "CACHE=${cache_job}"
echo "PHASE1=${phase1_dep}"
echo "PHASE2=${phase2_dep}"
echo "SUMMARY=${summary_job}"
echo "Monitor: squeue -j ${cache_job},${p1_classical},${p1_standard},${p1_ketgpt},${p1_a180},${p1_a160},${p2_classical},${p2_standard},${p2_ketgpt},${p2_a180},${p2_a160},${summary_job}"
