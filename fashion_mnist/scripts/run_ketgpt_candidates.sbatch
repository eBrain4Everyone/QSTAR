#!/bin/bash
#SBATCH --job-name=ketgpt_qtl
#SBATCH --output=logs/ketgpt_qtl_%j.out
#SBATCH --error=logs/ketgpt_qtl_%j.err
#SBATCH --time=32:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

mkdir -p logs

source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate torch-cu

python run_fmnist_qtl_ketgpt_search_resume.py \
  --epochs 10 \
  --batch_size 16 \
  --train_subset 5000 \
  --test_subset 1000 \
  --compressor_dim 8 \
  --target_qubits 8 \
  --shots 100 \
  --min_params 20 \
  --max_params 80 \
  --max_gates 150 \
  --max_candidates 5 \
  --seed 42 \
  --candidate_csv ketgpt_filtered_candidates.csv \
  --out_csv results_fmnist_ketgpt_candidates.csv
