#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import pennylane as qml

from qstar_new.common import (
    TrainingController,
    FeatureInputModel,
    canonical_dataset_name,
    dataset_display_name,
    load_feature_cache,
    collect_probs,
    compute_metrics,
    count_trainable_params,
    count_head_params,
    set_seed,
)


class LinearHead(nn.Module):
    def __init__(self, in_dim=8, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=8, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        return self.net(x)


def get_ketgpt_candidate_ops(target_id):
    if "QSTAR_KETGPT_DATA_DIR" not in os.environ:
        raise RuntimeError("QSTAR_KETGPT_DATA_DIR is not set")
    [ds] = qml.data.load("ketgpt", folder_path=os.environ["QSTAR_KETGPT_DATA_DIR"], num_threads=1)
    return ds.circuits[int(target_id)]


def count_ketgpt_params(circuit_ops):
    total = 0
    for op in circuit_ops:
        name = op.name.lower()
        if len(op.parameters) > 0 and name in ["u2", "ry", "u1", "rz", "rx"]:
            total += 2 if name == "u2" else 1
    return int(total)


class KetGPTQuantumHead(nn.Module):
    def __init__(self, ketgpt_id=180, n_qubits=8, num_classes=10, shots=100):
        super().__init__()
        self.ketgpt_id = int(ketgpt_id)
        self.circuit_ops = get_ketgpt_candidate_ops(self.ketgpt_id)
        self.n_qubits = int(n_qubits)
        self.shots = int(shots)
        self.num_params = count_ketgpt_params(self.circuit_ops)
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            weight_idx = 0
            for op in self.circuit_ops:
                name = op.name.lower()
                if len(op.parameters) > 0 and name in ["u2", "ry", "u1", "rz", "rx"]:
                    wires = list(op.wires)
                    if name == "u2":
                        qml.U2(weights[weight_idx], weights[weight_idx + 1], wires=wires)
                        weight_idx += 2
                    elif name == "ry":
                        qml.RY(weights[weight_idx], wires=wires)
                        weight_idx += 1
                    elif name == "rx":
                        qml.RX(weights[weight_idx], wires=wires)
                        weight_idx += 1
                    else:
                        qml.RZ(weights[weight_idx], wires=wires)
                        weight_idx += 1
                else:
                    try:
                        qml.apply(op)
                    except Exception:
                        pass
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.weight_shapes = {"weights": (self.num_params,)}
        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)
        self.readout = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        return self.readout(torch.stack([self.qlayer(x[i]) for i in range(x.shape[0])], dim=0))


def count_quantum_params(model):
    if isinstance(model.head, KetGPTQuantumHead):
        return int(sum(p.numel() for name, p in model.head.named_parameters() if "qlayer" in name and p.requires_grad))
    return 0


def qfields(model, shots, train_loader, epochs_executed):
    is_q = isinstance(model.head, KetGPTQuantumHead)
    if not is_q:
        return {
            "num_qubits": 0,
            "circuit_depth": 0,
            "num_gates_approx": 0,
            "shots": 0,
            "train_circuit_evaluations_approx": 0,
            "test_circuit_evaluations_approx": 0,
        }
    qp = count_quantum_params(model)
    return {
        "num_qubits": model.head.n_qubits,
        "circuit_depth": "ketgpt_variable",
        "num_gates_approx": len(model.head.circuit_ops),
        "shots": shots,
        "train_circuit_evaluations_approx": (1 + 2 * qp) * len(train_loader) * int(epochs_executed),
        "test_circuit_evaluations_approx": 0,
    }


def adaptive_probs(conf_probs, fallback_probs, threshold):
    confidence = np.max(conf_probs, axis=1)
    low_mask = confidence < threshold
    final_probs = conf_probs.copy()
    final_probs[low_mask] = fallback_probs[low_mask]
    return final_probs, low_mask


def status_fields(status, prefix=""):
    if not prefix:
        return status
    return {f"{prefix}{k}": v for k, v in status.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["fashionmnist", "cifar10", "kmnist", "svhn"])
    p.add_argument("--feature-cache-dir", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--train_subset", type=int, default=5000)  # provenance-only, cache already fixed
    p.add_argument("--test_subset", type=int, default=1000)   # provenance-only, cache already fixed
    p.add_argument("--compressor_dim", type=int, default=8)
    p.add_argument("--mlp_hidden_dim", type=int, default=8)
    p.add_argument("--n_qubits", type=int, default=8)
    p.add_argument("--q_layers", type=int, default=2)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--ketgpt_id", type=int, choices=[160, 180], required=True)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.70, 0.80, 0.90])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-full", action="store_true")
    p.add_argument("--disable-flashjt", action="store_true")
    p.add_argument("--disable-early-stopping", action="store_true")
    p.add_argument("--flashjt-min-history", type=int, default=20)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    out_path = Path(args.out_csv)
    if out_path.exists() and not args.force_full:
        print(f"Output already exists; leaving audited run unchanged: {out_path}", flush=True)
        return

    dataset = canonical_dataset_name(args.dataset)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, cache_manifest = load_feature_cache(
        args.feature_cache_dir, args.batch_size, args.seed, num_workers=0
    )
    print("Device:", device, flush=True)
    print(json.dumps(vars(args), indent=2), flush=True)

    # Preserve original construction order: confidence -> MLP -> quantum.
    models = {
        "confidence_linear": FeatureInputModel(LinearHead(args.compressor_dim, 10), args.compressor_dim),
        "matched_mlp": FeatureInputModel(MLPHead(args.compressor_dim, args.mlp_hidden_dim, 10), args.compressor_dim),
        "ketgpt_quantum": FeatureInputModel(
            KetGPTQuantumHead(args.ketgpt_id, args.n_qubits, 10, args.shots), args.compressor_dim
        ),
    }

    statuses = {}
    train_times = {}
    test_probs = {}
    test_times = {}
    y_test_ref = None

    for name, model in models.items():
        meta = {
            "dataset": dataset,
            "table": "table2",
            "model_family": name,
            "config_id": f"adaptive_candidate{args.ketgpt_id}_{name}",
            "candidate_id": args.ketgpt_id if name == "ketgpt_quantum" else 0,
            "num_qubits": args.n_qubits if name == "ketgpt_quantum" else 0,
            "circuit_depth": len(model.head.circuit_ops) if name == "ketgpt_quantum" else 0,
            "shots": args.shots if name == "ketgpt_quantum" else 0,
            "lr": args.lr,
            "batch_size": args.batch_size,
        }
        controller = TrainingController(
            run_root=args.run_root,
            checkpoint_dir=Path(args.checkpoint_root) / name,
            metadata=meta,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            resume=args.resume,
            force_full=args.force_full,
            enable_flashjt=not args.disable_flashjt,
            enable_early_stopping=not args.disable_early_stopping,
            flashjt_min_history=args.flashjt_min_history,
        )
        status = controller.train(model, train_loader, val_loader, device)
        statuses[name] = status
        train_times[name] = status["train_time_sec"]
        y_test, probs, infer_time = collect_probs(model, test_loader, device)
        test_probs[name] = probs
        test_times[name] = infer_time
        if y_test_ref is None:
            y_test_ref = y_test

    rows = []
    confidence_probs = test_probs["confidence_linear"]
    mlp_probs = test_probs["matched_mlp"]
    quantum_probs = test_probs["ketgpt_quantum"]
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2

    # Fixed rows retained for audit parity with the original Step5 script.
    for name in ["confidence_linear", "matched_mlp", "ketgpt_quantum"]:
        model = models[name]
        metrics = compute_metrics(y_test_ref, test_probs[name])
        is_q = name == "ketgpt_quantum"
        status = statuses[name]
        rows.append({
            "dataset": dataset_display_name(dataset),
            "dataset_key": dataset,
            "task": "multiclass_10_class",
            "backbone": "Frozen ResNet18 (strict eval-mode cache)",
            "cache_manifest_sha256": cache_manifest.get("manifest_sha256", ""),
            "candidate_id": args.ketgpt_id,
            "seed": args.seed,
            "experiment_type": "fixed",
            "model_name": "ketgpt_quantum" if is_q else name,
            "threshold": np.nan,
            "low_count": np.nan,
            "low_pct": np.nan,
            "routed_to_classical_pct": 0.0 if is_q else 100.0,
            "routed_to_quantum_pct": 100.0 if is_q else 0.0,
            "avg_shots_per_sample": args.shots if is_q else 0,
            "train_time_sec": train_times[name],
            "inference_time_sec_total": test_times[name],
            "inference_time_ms_per_sample": 1000 * test_times[name] / len(y_test_ref),
            "total_trainable_params": count_trainable_params(model),
            "head_trainable_params": count_head_params(model),
            "quantum_params": count_quantum_params(model),
            **qfields(model, args.shots, train_loader, status["epochs_executed"]),
            "memory_usage_mb": memory_mb,
            **metrics,
            **status_fields(status),
        })

    conf_score = np.max(confidence_probs, axis=1)
    for threshold in args.thresholds:
        low_mask = conf_score < threshold
        low_count = int(low_mask.sum())
        low_pct = 100.0 * low_count / len(y_test_ref)
        low_mlp_metrics = compute_metrics(y_test_ref[low_mask], mlp_probs[low_mask]) if low_count else {}
        low_q_metrics = compute_metrics(y_test_ref[low_mask], quantum_probs[low_mask]) if low_count else {}
        adaptive_classical_probs, _ = adaptive_probs(confidence_probs, mlp_probs, threshold)
        adaptive_q_probs, _ = adaptive_probs(confidence_probs, quantum_probs, threshold)
        adaptive_classical_metrics = compute_metrics(y_test_ref, adaptive_classical_probs)
        adaptive_q_metrics = compute_metrics(y_test_ref, adaptive_q_probs)

        mlp_status = statuses["matched_mlp"]
        q_status = statuses["ketgpt_quantum"]
        conf_status = statuses["confidence_linear"]

        rows.append({
            "dataset": dataset_display_name(dataset), "dataset_key": dataset, "task": "multiclass_10_class",
            "backbone": "Frozen ResNet18 (strict eval-mode cache)", "cache_manifest_sha256": cache_manifest.get("manifest_sha256", ""),
            "candidate_id": args.ketgpt_id, "seed": args.seed,
            "experiment_type": "low_confidence_only", "model_name": "matched_mlp_on_low_conf",
            "threshold": threshold, "low_count": low_count, "low_pct": low_pct,
            "routed_to_classical_pct": low_pct, "routed_to_quantum_pct": 0.0, "avg_shots_per_sample": 0,
            "train_time_sec": train_times["matched_mlp"], "inference_time_sec_total": test_times["matched_mlp"],
            "inference_time_ms_per_sample": 1000 * test_times["matched_mlp"] / len(y_test_ref),
            "total_trainable_params": count_trainable_params(models["matched_mlp"]),
            "head_trainable_params": count_head_params(models["matched_mlp"]), "quantum_params": 0,
            **qfields(models["matched_mlp"], args.shots, train_loader, mlp_status["epochs_executed"]),
            "memory_usage_mb": memory_mb, **low_mlp_metrics,
            "epochs_executed": min(conf_status["epochs_executed"], mlp_status["epochs_executed"]),
            "best_epoch": min(conf_status["best_epoch"], mlp_status["best_epoch"]),
            "stop_reason": f"confidence={conf_status['stop_reason']};mlp={mlp_status['stop_reason']}",
        })

        rows.append({
            "dataset": dataset_display_name(dataset), "dataset_key": dataset, "task": "multiclass_10_class",
            "backbone": "Frozen ResNet18 (strict eval-mode cache)", "cache_manifest_sha256": cache_manifest.get("manifest_sha256", ""),
            "candidate_id": args.ketgpt_id, "seed": args.seed,
            "experiment_type": "low_confidence_only", "model_name": "ketgpt_quantum_on_low_conf",
            "threshold": threshold, "low_count": low_count, "low_pct": low_pct,
            "routed_to_classical_pct": 0.0, "routed_to_quantum_pct": low_pct,
            "avg_shots_per_sample": args.shots * low_pct / 100.0,
            "train_time_sec": train_times["ketgpt_quantum"], "inference_time_sec_total": test_times["ketgpt_quantum"],
            "inference_time_ms_per_sample": 1000 * test_times["ketgpt_quantum"] / len(y_test_ref),
            "total_trainable_params": count_trainable_params(models["ketgpt_quantum"]),
            "head_trainable_params": count_head_params(models["ketgpt_quantum"]), "quantum_params": count_quantum_params(models["ketgpt_quantum"]),
            **qfields(models["ketgpt_quantum"], args.shots, train_loader, q_status["epochs_executed"]),
            "memory_usage_mb": memory_mb, **low_q_metrics,
            "epochs_executed": min(conf_status["epochs_executed"], q_status["epochs_executed"]),
            "best_epoch": min(conf_status["best_epoch"], q_status["best_epoch"]),
            "stop_reason": f"confidence={conf_status['stop_reason']};quantum={q_status['stop_reason']}",
        })

        rows.append({
            "dataset": dataset_display_name(dataset), "dataset_key": dataset, "task": "multiclass_10_class",
            "backbone": "Frozen ResNet18 (strict eval-mode cache)", "cache_manifest_sha256": cache_manifest.get("manifest_sha256", ""),
            "candidate_id": args.ketgpt_id, "seed": args.seed,
            "experiment_type": "adaptive_full", "model_name": "adaptive_classical",
            "threshold": threshold, "low_count": low_count, "low_pct": low_pct,
            "routed_to_classical_pct": low_pct, "routed_to_quantum_pct": 0.0, "avg_shots_per_sample": 0,
            "train_time_sec": train_times["confidence_linear"] + train_times["matched_mlp"],
            "inference_time_sec_total": test_times["confidence_linear"] + (low_pct / 100.0) * test_times["matched_mlp"],
            "inference_time_ms_per_sample": 1000 * (test_times["confidence_linear"] + (low_pct / 100.0) * test_times["matched_mlp"]) / len(y_test_ref),
            "total_trainable_params": count_trainable_params(models["confidence_linear"]) + count_trainable_params(models["matched_mlp"]),
            "head_trainable_params": count_head_params(models["confidence_linear"]) + count_head_params(models["matched_mlp"]), "quantum_params": 0,
            "num_qubits": 0, "circuit_depth": 0, "num_gates_approx": 0, "shots": 0,
            "train_circuit_evaluations_approx": 0, "test_circuit_evaluations_approx": 0,
            "memory_usage_mb": memory_mb, **adaptive_classical_metrics,
            "epochs_executed": min(conf_status["epochs_executed"], mlp_status["epochs_executed"]),
            "best_epoch": min(conf_status["best_epoch"], mlp_status["best_epoch"]),
            "stop_reason": f"confidence={conf_status['stop_reason']};mlp={mlp_status['stop_reason']}",
        })

        qf = qfields(models["ketgpt_quantum"], args.shots, train_loader, q_status["epochs_executed"])
        rows.append({
            "dataset": dataset_display_name(dataset), "dataset_key": dataset, "task": "multiclass_10_class",
            "backbone": "Frozen ResNet18 (strict eval-mode cache)", "cache_manifest_sha256": cache_manifest.get("manifest_sha256", ""),
            "candidate_id": args.ketgpt_id, "seed": args.seed,
            "experiment_type": "adaptive_full", "model_name": "adaptive_ketgpt_qtl",
            "threshold": threshold, "low_count": low_count, "low_pct": low_pct,
            "routed_to_classical_pct": 100.0 - low_pct, "routed_to_quantum_pct": low_pct,
            "avg_shots_per_sample": args.shots * low_pct / 100.0,
            "train_time_sec": train_times["confidence_linear"] + train_times["ketgpt_quantum"],
            "inference_time_sec_total": test_times["confidence_linear"] + (low_pct / 100.0) * test_times["ketgpt_quantum"],
            "inference_time_ms_per_sample": 1000 * (test_times["confidence_linear"] + (low_pct / 100.0) * test_times["ketgpt_quantum"]) / len(y_test_ref),
            "total_trainable_params": count_trainable_params(models["confidence_linear"]) + count_trainable_params(models["ketgpt_quantum"]),
            "head_trainable_params": count_head_params(models["confidence_linear"]) + count_head_params(models["ketgpt_quantum"]),
            "quantum_params": count_quantum_params(models["ketgpt_quantum"]),
            **qf, "test_circuit_evaluations_approx": int((low_pct / 100.0) * len(test_loader)),
            "memory_usage_mb": memory_mb, **adaptive_q_metrics,
            "epochs_executed": min(conf_status["epochs_executed"], q_status["epochs_executed"]),
            "best_epoch": min(conf_status["best_epoch"], q_status["best_epoch"]),
            "stop_reason": f"confidence={conf_status['stop_reason']};quantum={q_status['stop_reason']}",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
