# QSTAR — ICASSP Multiseed New Datasets

Run-ready extension of the audited QSTAR Table I / Table II pipeline for:

- CIFAR-10
- KMNIST
- SVHN

Target installation directory:

`/scratch/sr7849/QC_June28/ICASSP_multiseed_new_datasets`

The output tables intentionally match the current QSTAR Table 1 / Table 2 row structure:

**Table I:** Classical Linear, Classical MLP, the eight original Standard-QTL configurations, KetGPT #160, KetGPT #180.

**Table II:** for each threshold 0.70/0.80/0.90: low-confidence MLP, KetGPT #180, KetGPT #160, Adaptive Classical, Adaptive KetGPT #180, Adaptive KetGPT #160.

## What is preserved

- Frozen ResNet18 representation feeding a 512 -> compressor bottleneck.
- compressor dimension = qubit count for Standard QTL; 8 for classical/KetGPT adaptive experiments.
- batch size 16, Adam LR 1e-3, weight decay 1e-5.
- first 5,000 training examples and first 1,000 test examples.
- seed-specific 15% validation split using seeds 1, 42, 48, 550, 2026.
- 100-shot quantum execution.
- Standard-QTL grid: q4/d1,d2,d4; q6/d1,d2,d4; q8/d1,d2. No q8/d4.
- candidate #160 and #180 KetGPT circuits.
- adaptive thresholds 0.70, 0.80, 0.90.
- validation-accuracy model selection.

## Acceleration/recovery additions

### Feature cache

The cache is built once per dataset at the frozen ResNet18 output and reused by every model/seed. The train/validation split is still performed per seed *after* feature extraction.

Important: the original source froze ResNet weights but allowed BatchNorm modules to enter training mode because `FullModel.train()` propagated into the backbone. Static feature caching is not mathematically equivalent to that behavior. This implementation makes the backbone genuinely frozen by keeping BatchNorm in eval mode. See `SOURCE_REVIEW.md`.

For clean cross-dataset comparison, use this same implementation for all new datasets. `fashionmnist` is also supported if an apples-to-apples rerun against the new strict-frozen protocol is desired.

### Checkpoints

Each trained head stores:

- `checkpoint_latest.pt`
- `checkpoint_best.pt`
- `checkpoint_epoch003.pt`
- `checkpoint_epoch006.pt`
- `checkpoint_epoch010.pt` when reached
- `checkpoint_final.pt`
- `trajectory.json`

Model, optimizer, validation history, cumulative training time, and Python/NumPy/PyTorch RNG state are retained.

### FlashJT

Seeds 1, 42, 48 are guaranteed full 10-epoch runs and form the history pool.

Seeds 550 and 2026 are FlashJT-assisted:

1. epoch 3: predict validation metrics for epochs 4..10;
2. epochs 4,5,6 run normally;
3. epoch 6: compare the saved epoch-3 predictions to actual epochs 4/5/6;
4. stop only if every monitored relative error is <=10% and mean relative error is <=5%;
5. otherwise continue normally.

Monitored values are training loss plus validation Accuracy/Precision/Recall/F1/AUC. FlashJT predictions are saved for audit and are **not substituted for test-set results**.

### Ordinary early stopping

Only assisted seeds can early-stop. It begins after epoch 6 with patience 3 and min-delta 0.001 on validation accuracy. The three history seeds always run all 10 epochs.


### Runtime accounting

`Train (h)` in the generated Table-I files is the actual per-model optimization/runtime after the shared feature cache has been built. The cache manifest separately records train/test feature-extraction wall time. This avoids charging the same frozen-ResNet computation repeatedly to every model, but means the new cached Train(h) values should not be numerically compared to the old uncached Fashion-MNIST Train(h) column without noting the accounting difference.

## Preview versus publication-ready tables

The summary job always writes complete-format preview tables after all five seed jobs exist:

- `summary/table1_preview_mean_std.csv/.tex`
- `summary/table2_preview_mean_std.csv/.tex`

If any contributing seed stopped before epoch 10, the preview is marked non-publication-ready. The normal `table1_paper_mean_std.*` and `table2_paper_mean_std.*` filenames are created **only when all contributing runs have executed all 10 epochs**.

This avoids silently treating FlashJT-shortened training as identical to the original 10-epoch protocol.

To finish shortened seeds later, resume from their checkpoints using `scripts/resume_to_full_10epochs.sh`.

## Install on Jubail

From `/scratch/sr7849/QC_June28`, extract the supplied archive so this directory becomes:

```text
/scratch/sr7849/QC_June28/ICASSP_multiseed_new_datasets/
```

Then:

```bash
cd /scratch/sr7849/QC_June28/ICASSP_multiseed_new_datasets
conda activate torch-cu
python tests/smoke_test.py
```

Expected final line:

`SMOKE TEST PASS`

## Recommended first run

Start with one dataset, not all three simultaneously:

```bash
cd /scratch/sr7849/QC_June28/ICASSP_multiseed_new_datasets
bash scripts/submit_dataset.sh cifar10
```

The command prints the run root. Save it. Monitor with the `squeue` command printed by the launcher.

Check progress:

```bash
python scripts/status.py /scratch/sr7849/QC_June28/ICASSP_multiseed_new_datasets/runs/cifar10/<RUN_TAG>
```

After the summary finishes:

```bash
cat /scratch/sr7849/QC_June28/ICASSP_multiseed_new_datasets/runs/cifar10/<RUN_TAG>/summary/publication_status.json
```

If `publication_ready` is false because FlashJT/early stopping shortened seeds 550 or 2026, complete them from checkpoints:

```bash
bash scripts/resume_to_full_10epochs.sh cifar10 \
  /scratch/sr7849/QC_June28/ICASSP_multiseed_new_datasets/runs/cifar10/<RUN_TAG>
```

That resubmits only the two assisted seeds, disables both stop mechanisms, resumes checkpoints, and regenerates the final paper tables after epoch 10.

## Other datasets

```bash
bash scripts/submit_dataset.sh kmnist
bash scripts/submit_dataset.sh svhn
```

All three can be submitted with `scripts/submit_all_three.sh`, but one dataset first is safer for validating cluster/runtime behavior.

## KetGPT HDF5 handling

The audited QSTAR project already has a validated KetGPT HDF5 file under the Tables-3/4 run tree. Each quantum array task stages its own file copy into a unique task directory before invoking PennyLane. This deliberately avoids the HDF5 locking collision seen when multiple jobs use one writable PennyLane dataset path.

Default source:

`/scratch/sr7849/QC_June28/tables34_multiseed_runs/run_20260816_151916/pennylane_data/table4/seed_1_candidate_160/ketgpt/ketgpt.h5`

Override with `KETGPT_SOURCE_H5=/path/to/ketgpt.h5` if needed.

## Main files

- `run_qstar_baselines_new.py` — Classical + Standard QTL.
- `run_qstar_ketgpt_new.py` — fixed-head #160/#180.
- `run_qstar_adaptive_new.py` — adaptive routing for one candidate per invocation.
- `qstar_new/common.py` — cache loader, checkpoints, FlashJT, early stopping.
- `scripts/prepare_feature_cache.py` — deterministic dataset/backbone cache.
- `scripts/summarize_dataset.py` — exact 12-row Table I and 18-row Table II completeness logic.
- `scripts/submit_dataset.sh` — two-phase scheduler (3 full seeds, then 2 assisted seeds).
- `scripts/resume_to_full_10epochs.sh` — converts an accelerated preview run into a full 10-epoch five-seed run.

## FlashJT audit

After assisted seeds have reached epoch 6, collect the predictor verification decisions with:

```bash
python scripts/flashjt_report.py <RUN_ROOT>
```

This writes `summary/flashjt_audit.csv` with the history count, per-run stop decision, maximum/mean relative error, and final stop reason.
