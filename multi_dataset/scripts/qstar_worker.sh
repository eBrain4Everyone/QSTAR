#!/usr/bin/env bash
set -euo pipefail

: "${PACKAGE_ROOT:?}"
: "${PROJECT_ROOT:?}"
: "${RUN_ROOT:?}"
: "${DATASET:?}"
: "${FEATURE_CACHE_DIR:?}"
: "${SEEDS_CSV:?}"
mode="${1:?mode required}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch-cu
export PYTHONPATH="${PACKAGE_ROOT}:${PYTHONPATH:-}"

IFS=':' read -r -a seeds <<< "${SEEDS_CSV}"
RESUME_FLAG="--resume"
FORCE_FLAG=""
FLASH_FLAG=""
EARLY_FLAG=""
if [[ "${FORCE_FULL:-0}" == "1" ]]; then FORCE_FLAG="--force-full"; fi
if [[ "${DISABLE_FLASHJT:-0}" == "1" ]]; then FLASH_FLAG="--disable-flashjt"; fi
if [[ "${DISABLE_EARLY_STOPPING:-0}" == "1" ]]; then EARLY_FLAG="--disable-early-stopping"; fi

stage_ketgpt() {
  local tag="$1"
  local dest="${RUN_ROOT}/pennylane_data/${tag}/task_${SLURM_ARRAY_TASK_ID:-0}"
  mkdir -p "${dest}/ketgpt"
  local source_h5="${KETGPT_SOURCE_H5:-${PROJECT_ROOT}/tables34_multiseed_runs/run_20260816_151916/pennylane_data/table4/seed_1_candidate_160/ketgpt/ketgpt.h5}"
  if [[ ! -s "${dest}/ketgpt/ketgpt.h5" && -s "${source_h5}" ]]; then
    echo "Staging KetGPT HDF5 from ${source_h5}" >&2
    cp --reflink=auto "${source_h5}" "${dest}/ketgpt/ketgpt.h5"
    printf '%s\n' "${source_h5}" > "${dest}/SOURCE_H5.txt"
  fi
  export QSTAR_KETGPT_DATA_DIR="${dest}"
}

case "${mode}" in
  classical)
    seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"
    out="${RUN_ROOT}/table1/classical/seed_${seed}.csv"
    ckpt="${RUN_ROOT}/checkpoints/table1/classical/seed_${seed}"
    mkdir -p "$(dirname "${out}")" "${ckpt}"
    cd "${PACKAGE_ROOT}"
    python run_qstar_baselines_new.py \
      --dataset "${DATASET}" --feature-cache-dir "${FEATURE_CACHE_DIR}" \
      --run-root "${RUN_ROOT}" --checkpoint-root "${ckpt}" \
      --models linear matched_mlp --epochs 10 --batch_size 16 \
      --compressor_dim 8 --mlp_hidden_dim 8 --n_qubits 8 --q_layers 2 --shots 100 \
      --seed "${seed}" ${RESUME_FLAG} ${FORCE_FLAG} ${FLASH_FLAG} ${EARLY_FLAG} --out_csv "${out}"
    ;;

  standard)
    q=(4 4 4 6 6 6 8 8)
    depth=(1 2 4 1 2 4 1 2)
    seed_index=$((SLURM_ARRAY_TASK_ID / 8))
    config_index=$((SLURM_ARRAY_TASK_ID % 8))
    seed="${seeds[${seed_index}]}"
    nq="${q[${config_index}]}"; nd="${depth[${config_index}]}"
    out="${RUN_ROOT}/table1/standard_qtl/seed_${seed}_q${nq}_d${nd}.csv"
    ckpt="${RUN_ROOT}/checkpoints/table1/standard_qtl/seed_${seed}_q${nq}_d${nd}"
    mkdir -p "$(dirname "${out}")" "${ckpt}"
    cd "${PACKAGE_ROOT}"
    python run_qstar_baselines_new.py \
      --dataset "${DATASET}" --feature-cache-dir "${FEATURE_CACHE_DIR}" \
      --run-root "${RUN_ROOT}" --checkpoint-root "${ckpt}" \
      --models quantum --epochs 10 --batch_size 16 \
      --compressor_dim "${nq}" --mlp_hidden_dim 8 --n_qubits "${nq}" --q_layers "${nd}" --shots 100 \
      --seed "${seed}" ${RESUME_FLAG} ${FORCE_FLAG} ${FLASH_FLAG} ${EARLY_FLAG} --out_csv "${out}"
    ;;

  ketgpt)
    seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"
    stage_ketgpt "table1_ketgpt"
    out="${RUN_ROOT}/table1/ketgpt/seed_${seed}.csv"
    candidates="${RUN_ROOT}/table1/ketgpt/seed_${seed}_candidates.csv"
    ckpt="${RUN_ROOT}/checkpoints/table1/ketgpt/seed_${seed}"
    mkdir -p "$(dirname "${out}")" "${ckpt}"
    cd "${PACKAGE_ROOT}"
    python run_qstar_ketgpt_new.py \
      --dataset "${DATASET}" --feature-cache-dir "${FEATURE_CACHE_DIR}" \
      --run-root "${RUN_ROOT}" --checkpoint-root "${ckpt}" \
      --candidate-ids 160 180 --epochs 10 --batch_size 16 --compressor_dim 8 --shots 100 \
      --seed "${seed}" ${RESUME_FLAG} ${FORCE_FLAG} ${FLASH_FLAG} ${EARLY_FLAG} \
      --candidate_csv "${candidates}" --out_csv "${out}"
    ;;

  adaptive180|adaptive160)
    seed="${seeds[${SLURM_ARRAY_TASK_ID}]}"
    cid="${mode#adaptive}"
    stage_ketgpt "${mode}"
    out="${RUN_ROOT}/table2/${mode}/seed_${seed}.csv"
    ckpt="${RUN_ROOT}/checkpoints/table2/${mode}/seed_${seed}"
    mkdir -p "$(dirname "${out}")" "${ckpt}"
    cd "${PACKAGE_ROOT}"
    python run_qstar_adaptive_new.py \
      --dataset "${DATASET}" --feature-cache-dir "${FEATURE_CACHE_DIR}" \
      --run-root "${RUN_ROOT}" --checkpoint-root "${ckpt}" \
      --epochs 10 --batch_size 16 --compressor_dim 8 --mlp_hidden_dim 8 \
      --n_qubits 8 --q_layers 2 --shots 100 --ketgpt_id "${cid}" \
      --thresholds 0.70 0.80 0.90 --seed "${seed}" ${RESUME_FLAG} ${FORCE_FLAG} ${FLASH_FLAG} ${EARLY_FLAG} \
      --out_csv "${out}"
    ;;

  summarize)
    cd "${PACKAGE_ROOT}"
    python scripts/summarize_dataset.py "${RUN_ROOT}" --dataset "${DATASET}" --seeds "${SEEDS_CSV}" --target-epochs 10
    ;;

  *) echo "Unknown mode: ${mode}" >&2; exit 2;;
esac
