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


class LinearHead(nn.Module):
    def __init__(self, in_dim=8, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MatchedMLPHead(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=8, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        return self.net(x)


class QuantumHead(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, num_classes=10, shots=100):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)
        self.readout = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        if self.shots is not None:
            q_out = torch.stack([self.qlayer(x[i]) for i in range(x.shape[0])], dim=0)
        else:
            q_out = self.qlayer(x)
        return self.readout(q_out)


def build_model(model_name, args):
    if model_name == "linear":
        head = LinearHead(args.compressor_dim, 10)
    elif model_name == "matched_mlp":
        head = MatchedMLPHead(args.compressor_dim, args.mlp_hidden_dim, 10)
    elif model_name == "quantum":
        head = QuantumHead(args.n_qubits, args.q_layers, 10, args.shots)
    else:
        raise ValueError(model_name)
    return FeatureInputModel(head, args.compressor_dim)


def quantum_params(model):
    if not isinstance(model.head, QuantumHead):
        return 0
    return int(sum(p.numel() for name, p in model.head.named_parameters() if "qlayer" in name and p.requires_grad))


def resource_info(model, args, train_loader, test_loader, epochs_executed):
    qparams = quantum_params(model)
    if isinstance(model.head, QuantumHead):
        n_qubits = model.head.n_qubits
        circuit_depth = model.head.n_layers
        num_gates = model.head.n_layers * model.head.n_qubits * 3
        shots = int(args.shots or 0)
        train_evals = (1 + 2 * qparams) * len(train_loader) * int(epochs_executed)
        test_evals = len(test_loader)
    else:
        n_qubits = circuit_depth = num_gates = shots = train_evals = test_evals = 0
    return {
        "total_trainable_params": count_trainable_params(model),
        "head_trainable_params": count_head_params(model),
        "quantum_params": qparams,
        "num_qubits": n_qubits,
        "circuit_depth": circuit_depth,
        "num_gates_approx": num_gates,
        "shots": shots,
        "train_circuit_evaluations_approx": train_evals,
        "test_circuit_evaluations_approx": test_evals,
        "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / 1024**2,
    }


def existing_complete(out_csv, model_name, seed, epochs):
    p = Path(out_csv)
    if not p.exists():
        return False
    try:
        df = pd.read_csv(p)
        rows = df[(df.model_name == model_name) & (df.seed.astype(int) == int(seed))]
        return len(rows) == 1 and int(rows.iloc[0].get("epochs_executed", 0)) >= int(epochs)
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["fashionmnist", "cifar10", "kmnist", "svhn"])
    p.add_argument("--feature-cache-dir", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--checkpoint-root", required=True)
    p.add_argument("--models", nargs="+", default=["linear", "matched_mlp", "quantum"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--compressor_dim", type=int, default=8)
    p.add_argument("--mlp_hidden_dim", type=int, default=8)
    p.add_argument("--n_qubits", type=int, default=8)
    p.add_argument("--q_layers", type=int, default=2)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-full", action="store_true")
    p.add_argument("--disable-flashjt", action="store_true")
    p.add_argument("--disable-early-stopping", action="store_true")
    p.add_argument("--flashjt-min-history", type=int, default=20)
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

    for model_name in args.models:
        if existing_complete(args.out_csv, model_name, args.seed, args.epochs) and not args.force_full:
            print(f"SKIP complete row: {model_name} seed={args.seed}", flush=True)
            continue

        model = build_model(model_name, args)
        config_id = model_name
        if model_name == "quantum":
            config_id = f"q{args.n_qubits}_d{args.q_layers}"
        ckpt_dir = Path(args.checkpoint_root) / config_id
        metadata = {
            "dataset": dataset,
            "table": "table1",
            "model_family": "standard_qtl" if model_name == "quantum" else "classical",
            "config_id": config_id,
            "candidate_id": 0,
            "num_qubits": args.n_qubits if model_name == "quantum" else 0,
            "circuit_depth": args.q_layers if model_name == "quantum" else 0,
            "shots": args.shots if model_name == "quantum" else 0,
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
        train_status = controller.train(model, train_loader, val_loader, device)
        test_metrics = evaluate_model(model, test_loader, device)
        resources = resource_info(model, args, train_loader, test_loader, train_status["epochs_executed"])
        row = {
            "dataset": dataset_display_name(dataset),
            "dataset_key": dataset,
            "task": "multiclass_10_class",
            "backbone": "Frozen ResNet18 (strict eval-mode cache)",
            "cache_manifest_sha256": cache_manifest.get("manifest_sha256", ""),
            "model_name": model_name,
            "seed": args.seed,
            "compressor_dim": args.compressor_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            **train_status,
            **test_metrics,
            **resources,
        }
        replace_or_append_csv_row(args.out_csv, row, ["model_name", "seed"])
        print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
