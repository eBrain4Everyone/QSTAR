# QSTAR: Quantum Selective Transfer with Adaptive Routing

Official implementation and experimental results for QSTAR.

QSTAR retains high-confidence classical predictions and routes low-confidence samples to a selected quantum fallback.

## Repository structure

- `model/` — core QSTAR implementation and final model entry points.
- `scripts/hpc/` — final Jubail/Slurm execution pipeline.
- `scripts/aggregation/` — result aggregation and table-generation utilities.
- `results/fashion_mnist/` — Fashion-MNIST fixed-head, routing, and resource-ablation results.
- `results/cifar10/` — CIFAR-10 experiment outputs.
- `results/kmnist/` — KMNIST experiment outputs.
- `results/svhn/` — SVHN experiment outputs.
- `results/aggregates/` — final aggregated CSVs used to construct paper tables.
- `extra/` — valid QSTAR-related experiments and utilities not used as primary claims in the submitted paper.
- `provenance/` — experiment specifications and source hashes.

## Main paper experiments

The submitted QSTAR study contains:

1. Fashion-MNIST fixed-head classical, standard-QTL, and KetGPT comparisons.
2. Fashion-MNIST low-confidence and full adaptive routing with Candidate #160.
3. Fashion-MNIST KetGPT resource ablation.
4. Cross-dataset generalization on CIFAR-10, KMNIST, and SVHN.

For cross-dataset generalization, Candidate #160 is reported at tau = 0.90.

## Extra experiments

The `extra/` directory contains experiments performed as part of the broader QSTAR project but not used as primary result tables in the submitted paper, including shot/qubit ablations, validation studies, FlashJT utilities, and development tools.

## Data

Raw benchmark datasets and large PennyLane/KetGPT HDF5 files are intentionally excluded from Git.

KetGPT circuit data: https://www.kaggle.com/datasets/boranapak/ketgpt-data
