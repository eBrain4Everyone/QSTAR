#!/bin/bash
#SBATCH --job-name=qtl_missing
#SBATCH --output=logs/qtl_missing_%j.out
#SBATCH --error=logs/qtl_missing_%j.err
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate torch-cu

OUT=results_qtl_missing_resource_ablation.csv

for CONFIG in "4 1" "4 4" "6 1" "6 2" "6 4" "8 1"
do
  set -- $CONFIG
  Q=$1
  DEPTH=$2

  echo "============================================"
  echo "Running Q=${Q}, compressor_dim=${Q}, depth=${DEPTH}, shots=100"
  echo "============================================"

  python run_fmnist_qtl_baselines.py \
    --models quantum \
    --epochs 10 \
    --batch_size 16 \
    --train_subset 5000 \
    --test_subset 1000 \
    --compressor_dim ${Q} \
    --mlp_hidden_dim 8 \
    --n_qubits ${Q} \
    --q_layers ${DEPTH} \
    --shots 100 \
    --seed 42 \
    --out_csv ${OUT}

  echo "Finished Q=${Q}, depth=${DEPTH}"
done
