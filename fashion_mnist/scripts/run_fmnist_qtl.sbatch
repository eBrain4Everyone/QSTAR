#!/bin/bash
#SBATCH --job-name=fmnist_qtl
#SBATCH --output=logs/fmnist_qtl_%j.out
#SBATCH --error=logs/fmnist_qtl_%j.err
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

mkdir -p logs

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate torch-cu

python run_fmnist_qtl_baselines.py \
  --models quantum \
  --epochs 10 \
  --batch_size 16 \
  --train_subset 5000 \
  --test_subset 1000 \
  --compressor_dim 8 \
  --mlp_hidden_dim 8 \
  --n_qubits 8 \
  --q_layers 2 \
  --shots 100 \
  --seed 42 \
  --out_csv results_fmnist_qtl_5000_q8.csv