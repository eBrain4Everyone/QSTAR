#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: submit_dataset_localdeps.sh DATASET [RUN_TAG]}"
RUN_TAG="${2:-run_$(date -u +%Y%m%d_%H%M%S)}"
case "${DATASET}" in kmnist|svhn|cifar10|fashionmnist) ;; *) echo "Unsupported dataset ${DATASET}" >&2; exit 2;; esac

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/sr7849/QC_June28}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/ICASSP_multiseed_new_datasets}"
WORK_ROOT="${WORK_ROOT:-${PROJECT_ROOT}/ICASSP_multiseed_new_datasets}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
PARTITION="${PARTITION:-compute}"
SEEDS_CSV="${SEEDS_CSV:-1:42:48:550:2026}"
FLASHJT_MIN_HISTORY="${FLASHJT_MIN_HISTORY:-20}"
RUN_ROOT="${WORK_ROOT}/runs/${DATASET}/${RUN_TAG}"
FEATURE_CACHE_DIR="${WORK_ROOT}/cache/${DATASET}"
LOG_DIR="${RUN_ROOT}/logs"
ARRAY_SBATCH="${PACKAGE_ROOT}/slurm/qstar_array_localdeps.sbatch"
CACHE_SBATCH="${PACKAGE_ROOT}/slurm/prepare_cache.sbatch"

# Keep this architecture CPU-only so it cannot consume A100s reserved in practice for VLM work.
if [[ "${PARTITION}" != "compute" && "${ALLOW_NONCOMPUTE:-0}" != "1" ]]; then
  echo "Refusing PARTITION=${PARTITION}. Set ALLOW_NONCOMPUTE=1 only if intentional." >&2
  exit 2
fi

# Per-experiment array throttles. These are intentionally conservative.
STD_P1_THROTTLE="${STD_P1_THROTTLE:-1}"
STD_P2_THROTTLE="${STD_P2_THROTTLE:-2}"
KET_P1_THROTTLE="${KET_P1_THROTTLE:-1}"
KET_P2_THROTTLE="${KET_P2_THROTTLE:-2}"
ADAPT_P1_THROTTLE="${ADAPT_P1_THROTTLE:-1}"
ADAPT_P2_THROTTLE="${ADAPT_P2_THROTTLE:-2}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}/manifests" "${RUN_ROOT}/summary" \
  "${RUN_ROOT}/table1/classical" "${RUN_ROOT}/table1/standard_qtl" "${RUN_ROOT}/table1/ketgpt" \
  "${RUN_ROOT}/table1/ketgpt_parts/candidate_160" "${RUN_ROOT}/table1/ketgpt_parts/candidate_180" \
  "${RUN_ROOT}/table2/adaptive160" "${RUN_ROOT}/table2/adaptive180" "${RUN_ROOT}/checkpoints"

base_export="ALL,PACKAGE_ROOT=${PACKAGE_ROOT},PROJECT_ROOT=${PROJECT_ROOT},RUN_ROOT=${RUN_ROOT},DATASET=${DATASET},FEATURE_CACHE_DIR=${FEATURE_CACHE_DIR},SEEDS_CSV=${SEEDS_CSV},DATA_ROOT=${DATA_ROOT},FLASHJT_MIN_HISTORY=${FLASHJT_MIN_HISTORY}"

cache_job=$(sbatch --parsable --job-name="qstar-${DATASET}-cache" --partition="${PARTITION}" \
  --output="${LOG_DIR}/cache-%j.out" --error="${LOG_DIR}/cache-%j.err" \
  --export="${base_export}" "${CACHE_SBATCH}")

submit_array() {
  local name="$1" mode="$2" array="$3" time="$4" mem="$5" dep="$6" extra_export="${7:-}"
  local dep_args=()
  [[ -n "${dep}" ]] && dep_args=(--dependency="afterok:${dep}")
  local export_arg="${base_export},MODE=${mode},FORCE_FULL=0,DISABLE_FLASHJT=0,DISABLE_EARLY_STOPPING=0"
  [[ -n "${extra_export}" ]] && export_arg="${export_arg},${extra_export}"
  sbatch --parsable --job-name="${name}" --partition="${PARTITION}" --array="${array}" --time="${time}" --mem="${mem}" \
    --output="${LOG_DIR}/${name}-%A_%a.out" --error="${LOG_DIR}/${name}-%A_%a.err" \
    "${dep_args[@]}" --export="${export_arg}" "${ARRAY_SBATCH}"
}

submit_single() {
  local name="$1" mode="$2" time="$3" mem="$4" dep="$5"
  sbatch --parsable --job-name="${name}" --partition="${PARTITION}" --time="${time}" --mem="${mem}" \
    --dependency="afterok:${dep}" --output="${LOG_DIR}/${name}-%j.out" --error="${LOG_DIR}/${name}-%j.err" \
    --export="${base_export},MODE=${mode},FORCE_FULL=0,DISABLE_FLASHJT=0,DISABLE_EARLY_STOPPING=0" "${ARRAY_SBATCH}"
}

# Classical is intentionally kept as one tiny family job because Linear+MLP complete in seconds.
p1_classical=$(submit_array "qstar-${DATASET}-hist-classical" classical "0-2%3" 04:00:00 16G "${cache_job}")
p2_classical=$(submit_array "qstar-${DATASET}-assist-classical" classical "3-4%2" 04:00:00 16G "${p1_classical}")

# Each Standard-QTL configuration gets an independent 1/42/48 -> 550/2026 gate.
declare -a std_p1_jobs=()
declare -a std_p2_jobs=()
for spec in "4 1" "4 2" "4 4" "6 1" "6 2" "6 4" "8 1" "8 2"; do
  read -r q d <<< "${spec}"
  tag="q${q}d${d}"
  p1=$(submit_array "qstar-${DATASET}-hist-${tag}" standard_cfg "0-2%${STD_P1_THROTTLE}" 1-12:00:00 32G "${cache_job}" "QUBITS=${q},DEPTH=${d}")
  p2=$(submit_array "qstar-${DATASET}-assist-${tag}" standard_cfg "3-4%${STD_P2_THROTTLE}" 1-12:00:00 32G "${p1}" "QUBITS=${q},DEPTH=${d}")
  std_p1_jobs+=("${tag}=${p1}")
  std_p2_jobs+=("${tag}=${p2}")
done

# Candidate #160 and #180 are independent gates (unlike the original bundled KetGPT job).
p1_k160=$(submit_array "qstar-${DATASET}-hist-k160" ketgpt160 "0-2%${KET_P1_THROTTLE}" 2-12:00:00 32G "${cache_job}")
p2_k160=$(submit_array "qstar-${DATASET}-assist-k160" ketgpt160 "3-4%${KET_P2_THROTTLE}" 2-12:00:00 32G "${p1_k160}")
p1_k180=$(submit_array "qstar-${DATASET}-hist-k180" ketgpt180 "0-2%${KET_P1_THROTTLE}" 2-12:00:00 32G "${cache_job}")
p2_k180=$(submit_array "qstar-${DATASET}-assist-k180" ketgpt180 "3-4%${KET_P2_THROTTLE}" 2-12:00:00 32G "${p1_k180}")

# Adaptive candidates are already naturally candidate-specific.
p1_a160=$(submit_array "qstar-${DATASET}-hist-a160" adaptive160 "0-2%${ADAPT_P1_THROTTLE}" 2-00:00:00 32G "${cache_job}")
p2_a160=$(submit_array "qstar-${DATASET}-assist-a160" adaptive160 "3-4%${ADAPT_P2_THROTTLE}" 2-00:00:00 32G "${p1_a160}")
p1_a180=$(submit_array "qstar-${DATASET}-hist-a180" adaptive180 "0-2%${ADAPT_P1_THROTTLE}" 2-00:00:00 32G "${cache_job}")
p2_a180=$(submit_array "qstar-${DATASET}-assist-a180" adaptive180 "3-4%${ADAPT_P2_THROTTLE}" 2-00:00:00 32G "${p1_a180}")

# Convert standard P2 key=value list into a dependency list of job IDs.
std_p2_dep=""
for kv in "${std_p2_jobs[@]}"; do
  jid="${kv#*=}"
  [[ -z "${std_p2_dep}" ]] && std_p2_dep="${jid}" || std_p2_dep="${std_p2_dep}:${jid}"
done

merge_job=$(submit_single "qstar-${DATASET}-merge-ketgpt" merge_ketgpt 00:20:00 8G "${p2_k160}:${p2_k180}")
all_final_dep="${p2_classical}:${std_p2_dep}:${p2_k160}:${p2_k180}:${p2_a160}:${p2_a180}:${merge_job}"
summary_job=$(submit_single "qstar-${DATASET}-summary" summarize 00:30:00 8G "${all_final_dep}")

{
  echo "architecture=experiment_local_dependencies_v1"
  echo "project_root=${PROJECT_ROOT}"
  echo "package_root=${PACKAGE_ROOT}"
  echo "run_root=${RUN_ROOT}"
  echo "dataset=${DATASET}"
  echo "partition=${PARTITION}"
  echo "feature_cache_dir=${FEATURE_CACHE_DIR}"
  echo "seeds=${SEEDS_CSV}"
  echo "full_history_seeds=1:42:48"
  echo "flashjt_assisted_seeds=550:2026"
  echo "flashjt_min_history=${FLASHJT_MIN_HISTORY}"
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "cache_job=${cache_job}"
  echo "hist_classical=${p1_classical}"
  echo "assist_classical=${p2_classical}"
  for kv in "${std_p1_jobs[@]}"; do echo "hist_standard_${kv}"; done
  for kv in "${std_p2_jobs[@]}"; do echo "assist_standard_${kv}"; done
  echo "hist_ketgpt160=${p1_k160}"
  echo "assist_ketgpt160=${p2_k160}"
  echo "hist_ketgpt180=${p1_k180}"
  echo "assist_ketgpt180=${p2_k180}"
  echo "hist_adaptive160=${p1_a160}"
  echo "assist_adaptive160=${p2_a160}"
  echo "hist_adaptive180=${p1_a180}"
  echo "assist_adaptive180=${p2_a180}"
  echo "merge_ketgpt_job=${merge_job}"
  echo "summary_job=${summary_job}"
} > "${RUN_ROOT}/run_manifest.txt"

cat <<PRINT
RUN_ROOT=${RUN_ROOT}
CACHE=${cache_job}
CLASSICAL  history=${p1_classical} assisted=${p2_classical}
KETGPT160  history=${p1_k160} assisted=${p2_k160}
KETGPT180  history=${p1_k180} assisted=${p2_k180}
ADAPT160   history=${p1_a160} assisted=${p2_a160}
ADAPT180   history=${p1_a180} assisted=${p2_a180}
MERGE=${merge_job}
SUMMARY=${summary_job}

Standard-QTL local gates:
PRINT
for i in "${!std_p1_jobs[@]}"; do
  echo "  ${std_p1_jobs[$i]} -> ${std_p2_jobs[$i]}"
done

echo
echo "All jobs are on partition=${PARTITION}; this launcher requests NO GPU/GRES."
echo "Monitor: python scripts/status.py ${RUN_ROOT}"
echo "Manifest: ${RUN_ROOT}/run_manifest.txt"
