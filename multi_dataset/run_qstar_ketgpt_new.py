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
    evaluate_model,
    count_trainable_params,
    count_head_params,
    replace_or_append_csv_row,
    set_seed,
)


def count_trainable_params_from_ops(circuit_ops):
    total = 0
    for op in circuit_ops:
        name = op.name.lower()
        if len(op.parameters) > 0 and name in ["u2", "ry", "u1", "rz", "rx"]:
            total += 2 if name == "u2" else 1
    return int(total)


def get_num_qubits(circuit_ops):
    wires = sorted(set(int(w) for op in circuit_ops for w in op.wires))
    return max(wires) + 1 if wires else 0


def apply_ketgpt_ops(circuit_ops, weights):
    weight_idx = 0
    for op in circuit_ops:
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
            elif name in ["u1", "rz"]:
                qml.RZ(weights[weight_idx], wires=wires)
                weight_idx += 1
        else:
            try:
                qml.apply(op)
            except Exception:
                pass


def load_candidates(candidate_ids):
    if "QSTAR_KETGPT_DATA_DIR" not in os.environ:
        raise RuntimeError("QSTAR_KETGPT_DATA_DIR must point to a staged PennyLane KetGPT dataset directory")
    [ds] = qml.data.load(
        "ketgpt",
        folder_path=os.environ["QSTAR_KETGPT_DATA_DIR"],
        num_threads=1,
    )
    result = []
    for cid in candidate_ids:
        ops = ds.circuits[int(cid)]
        q = get_num_qubits(ops)
        params = count_trainable_params_from_ops(ops)
        gates = len(ops)
        if q != 8:
            raise RuntimeError(f"Candidate {cid} expected 8 qubits, found {q}")
        result.append({
            "ketgpt_id": int(cid),
            "num_qubits": q,
            "num_params": params,
            "gate_count": gates,
            "ops": ops,
        })
    return result


class KetGPTQuantumHead(nn.Module):
    def __init__(self, circuit_ops, n_qubits=8, num_params=10, num_classes=10, shots=100):
        super().__init__()
        self.circuit_ops = circuit_ops
        self.n_qubits = n_qubits
        self.num_params = num_params
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            apply_ketgpt_ops(circuit_ops, weights)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.weight_shapes = {"weights": (num_params,)}
        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)
        self.readout = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        q_out = torch.stack([self.qlayer(x[i]) for i in range(x.shape[0])], dim=0)
        return self.readout(q_out)


def quantum_params(model):
    return int(sum(p.numel() for name, p in model.head.named_parameters() if "qlayer" in name and p.requires_grad))


def existing_complete(out_csv, candidate_id, seed, epochs):
    p = Path(out_csv)
    if not p.exists():
        return False
    try:
        df = pd.read_csv(p)
        rows = df[(df.ketgpt_id.astype(int) == int(candidate_id)) & (df.seed.astype(int) == int(seed))]
        return len(rows) == 1 and int(rows.iloc[0].get("epochs_executed", 0)) >= int(epochs)
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["fashionmnist", "cifar10", "kmnist", "svhn"])
    p.add_argument("--feature-cache-dir", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--candidate-ids", nargs="+", type=int, default=[160, 180])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--compressor_dim", type=int, default=8)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-full", action="store_true")
    p.add_argument("--disable-flashjt", action="store_true")
    p.add_argument("--disable-early-stopping", action="store_true")
    p.add_argument("--flashjt-min-history", type=int, default=20)
    p.add_argument("--candidate_csv", required=True)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    dataset = canonical_dataset_name(args.dataset)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, cache_manifest = load_feature_cache(
        args.feature_cache_dir, args.batch_size, args.seed, num_workers=0
    )
    print("Device:", device, flush=True)
    print(json.dumps(vars(args), indent=2), flush=True)

    candidates = load_candidates(args.candidate_ids)
    pd.DataFrame([
        {k: c[k] for k in ["ketgpt_id", "num_qubits", "num_params", "gate_count"]}
        for c in candidates
    ]).to_csv(args.candidate_csv, index=False)

    for c in candidates:
        cid = int(c["ketgpt_id"])
        if existing_complete(args.out_csv, cid, args.seed, args.epochs) and not args.force_full:
            print(f"SKIP complete candidate {cid} seed={args.seed}", flush=True)
            continue

        head = KetGPTQuantumHead(
            circuit_ops=c["ops"],
            n_qubits=c["num_qubits"],
            num_params=c["num_params"],
            num_classes=10,
            shots=args.shots,
        )
        model = FeatureInputModel(head, args.compressor_dim)
        ckpt_dir = Path(args.checkpoint_root) / f"candidate_{cid}"
        metadata = {
            "dataset": dataset,
            "table": "table1",
            "model_family": "ketgpt",
            "config_id": f"ketgpt_{cid}",
            "candidate_id": cid,
            "num_qubits": c["num_qubits"],
            "circuit_depth": _stable_depth_marker(c["gate_count"]),
            "shots": args.shots,
            "lr": args.lr,
            "batch_size": args.batch_size,
        }
        controller = TrainingController(
            run_root=args.run_root,
            checkpoint_dir=ckpt_dir,
            metadata=metadata,
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
        test_metrics = evaluate_model(model, test_loader, device)
        qparams = quantum_params(model)
        circuit_evals_per_batch = 1 + 2 * qparams
        row = {
            "dataset": dataset_display_name(dataset),
            "dataset_key": dataset,
            "task": "multiclass_10_class",
            "backbone": "Frozen ResNet18 (strict eval-mode cache)",
            "cache_manifest_sha256": cache_manifest.get("manifest_sha256", ""),
            "model_name": "ketgpt_quantum",
            "candidate_rank": args.candidate_ids.index(cid),
            "ketgpt_id": cid,
            "seed": args.seed,
            "compressor_dim": args.compressor_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            **status,
            **test_metrics,
            "total_trainable_params": count_trainable_params(model),
            "head_trainable_params": count_head_params(model),
            "quantum_params": qparams,
            "num_qubits": c["num_qubits"],
            "circuit_depth": "ketgpt_variable",
            "num_gates_approx": c["gate_count"],
            "shots": args.shots,
            "train_circuit_evaluations_approx": circuit_evals_per_batch * len(train_loader) * status["epochs_executed"],
            "test_circuit_evaluations_approx": len(test_loader),
            "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        }
        replace_or_append_csv_row(args.out_csv, row, ["ketgpt_id", "seed"])
        print(json.dumps(row, indent=2), flush=True)


def _stable_depth_marker(gate_count: int) -> int:
    # FlashJT only needs a numeric architecture descriptor. Paper output remains "Var.".
    return int(gate_count)


if __name__ == "__main__":
    main()
