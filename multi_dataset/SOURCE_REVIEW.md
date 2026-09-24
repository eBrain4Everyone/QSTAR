# Source review and integration decisions

Source capture reviewed: `qstar_table12_source_capture_20260921_195639.tar.gz`.

## Original August Table-I/II runner

The captured `qstar_multiseed/qstar_worker.sh` confirms:

- five seeds: 1, 42, 48, 550, 2026;
- Table I classical: Linear + matched MLP;
- Standard-QTL: exactly eight configurations (40 files total across five seeds), with no q8/d4;
- original Table-I KetGPT sweep selected IDs #22 and #180;
- original Table-II adaptive run used #180;
- original worker already used a unique PennyLane `QSTAR_KETGPT_DATA_DIR` per array task to avoid cache/HDF5 collisions.

## Why the new runner uses #160 and #180

The current QSTAR tables shown for this extension have moved beyond the original August runner: the fixed-head table contains #160 and #180, and the adaptive table contains #160 and #180 alongside the shared classical rows.

The captured source already supports arbitrary `--ketgpt_id` in `Step5_Full_Adaptive/step5_full_adaptive_ketgpt.py`. The new runner therefore runs #160 and #180 explicitly rather than reproducing the old #22/#180 filter.

Candidate metadata used as a consistency check:

- #160: 8 qubits, 10 trainable quantum parameters, 9 gates.
- #180: 8 qubits, 58 trainable quantum parameters, 49 gates.

## Dataset changes

The original data loader was FashionMNIST-only and always applied `Grayscale(num_output_channels=3)`.

For the new datasets:

- KMNIST keeps grayscale -> 3-channel conversion.
- CIFAR-10 and SVHN retain native RGB.
- all datasets are resized to 224x224 and use the same ImageNet normalization before ResNet18.
- the first-5000 / first-1000 subset convention and seed-specific 15% validation split are retained.

Applying the FashionMNIST grayscale transform literally to CIFAR-10/SVHN would discard their color information, so that part is dataset-aware rather than copied blindly.

## Frozen-backbone/cache issue found in the original source

The original `FrozenResNet18` sets `requires_grad=False` but is a normal child of `FullModel`. During each epoch the code calls `model.train()`, which also puts the ResNet BatchNorm layers in training mode. Thus the original backbone weights are frozen, but its BatchNorm behavior/state is not strictly frozen.

A one-time static feature cache cannot reproduce batch-dependent training-mode BatchNorm behavior. The new implementation therefore makes the backbone genuinely frozen (`eval()` BatchNorm) before extracting reusable 512-D features.

This is a deliberate correctness change required by the requested cache. If direct numeric comparison to the existing FashionMNIST numbers is important, rerun FashionMNIST through this new cached pipeline as an additional reference rather than assuming the old and new backbone semantics are identical.

## Checkpoint correction

The original baseline/KetGPT scripts had no resumable checkpointing. The adaptive script saved a single latest checkpoint but did not preserve the best-model state or RNG state robustly for a later resume.

The new system stores latest, best, milestone, final, optimizer, trajectory, and RNG state. Best-model selection remains validation accuracy, matching the original intent.

## Table completeness rule

The new summarizer enforces the exact displayed table shapes:

- Table I: 12 aggregate rows, each n=5.
- Table II: 18 aggregate rows, each n=5.

Preview tables can be produced after accelerated seeds stop. Publication-named files are withheld until all contributing training paths have actually executed 10 epochs. This makes checkpoint resume the final step if FlashJT was used to save time during exploration.
