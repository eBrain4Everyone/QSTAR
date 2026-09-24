QSTAR local-dependency patch

Adds new files only; it does not replace the existing CIFAR launcher/worker:
  scripts/qstar_worker_localdeps.sh
  scripts/submit_dataset_localdeps.sh
  scripts/merge_ketgpt_parts.py
  scripts/patch_flashjt_forecast_gate.py
  slurm/qstar_array_localdeps.sbatch

Architecture:
- cache first
- full-history seeds 1/42/48
- each experiment/config independently releases 550/2026
- FlashJT remains restricted to seeds 550/2026
- default FlashJT minimum full-history pool remains 20
- all jobs default to Slurm partition=compute and request no GPU/GRES

The forecast-gate patch is recommended before launching new datasets. It is a
small correction to qstar_new/common.py so the minimum-history requirement is
checked at epoch 3 when the saved forecast is created, not only at epoch 6.
