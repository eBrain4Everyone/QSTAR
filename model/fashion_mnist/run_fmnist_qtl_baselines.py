import argparse, time, json, os, psutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import pennylane as qml


# -------------------------
# Data
# -------------------------
def get_loaders(batch_size, train_subset=None, test_subset=None, seed=42):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_full = datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)
    test_set = datasets.FashionMNIST("./data", train=False, download=True, transform=tfm)

    if train_subset:
        train_full = Subset(train_full, list(range(train_subset)))
    if test_subset:
        test_set = Subset(test_set, list(range(test_subset)))

    val_size = int(0.15 * len(train_full))
    train_size = len(train_full) - val_size

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [train_size, val_size], generator=gen)

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4),
    )


# -------------------------
# Backbone
# -------------------------
class FrozenResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(model.children())[:-1])

        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            x = torch.flatten(x, 1)
        return x


# -------------------------
# Heads
# -------------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim=4, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MatchedMLPHead(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=8, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


class QuantumHead(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2, num_classes=10, shots=None):
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
        # Finite-shot PennyLane sampling is more stable sample-by-sample
        # than batched execution on default.qubit.
        if self.shots is not None:
            q_outputs = []
            for i in range(x.shape[0]):
                xi = x[i]
                qi = self.qlayer(xi)
                q_outputs.append(qi)
            q_out = torch.stack(q_outputs, dim=0)
        else:
            q_out = self.qlayer(x)

        return self.readout(q_out)


class FullModel(nn.Module):
    def __init__(self, head, compressor_dim=4):
        super().__init__()
        self.backbone = FrozenResNet18()
        self.compressor = nn.Sequential(
            nn.Linear(512, compressor_dim),
            nn.Tanh()
        )
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.compressor(x)
        return self.head(x)


# -------------------------
# Metrics
# -------------------------
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_head_params(model):
    return sum(p.numel() for p in model.head.parameters() if p.requires_grad)


def count_quantum_params(model):
    if isinstance(model.head, QuantumHead):
        return sum(p.numel() for name, p in model.head.named_parameters()
                   if "qlayer" in name and p.requires_grad)
    return 0


def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs = [], [], []

    start = time.time()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()

            probs.append(prob)
            preds.extend(np.argmax(prob, axis=1))
            ys.extend(y.numpy())

    infer_time = time.time() - start

    ys = np.array(ys)
    preds = np.array(preds)
    probs = np.vstack(probs)

    metrics = {
        "accuracy": accuracy_score(ys, preds),
        "precision_macro": precision_score(ys, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(ys, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(ys, preds, average="macro", zero_division=0),
        "auc_ovr_macro": roc_auc_score(ys, probs, multi_class="ovr", average="macro"),
        "inference_time_sec_total": infer_time,
        "inference_time_ms_per_sample": 1000 * infer_time / len(ys),
    }

    return metrics


def train(model, train_loader, val_loader, args, device):
    model.to(device)

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    best_val = -1
    best_state = None

    start = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss {total_loss/len(train_loader):.4f} | "
            f"Val Acc {val_metrics['accuracy']:.4f} | "
            f"Val F1 {val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    train_time = time.time() - start

    model.load_state_dict(best_state)
    return train_time


def get_resource_info(model, args, train_loader, test_loader):
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024**2

    is_quantum = isinstance(model.head, QuantumHead)

    quantum_params = count_quantum_params(model)

    if is_quantum:
        n_qubits = model.head.n_qubits
        n_layers = model.head.n_layers
        circuit_depth = n_layers
        num_gates = n_layers * n_qubits * 3
        shots = args.shots if args.shots is not None else 0

        # Approximate parameter-shift cost per batch.
        circuit_evals_per_batch = 1 + 2 * quantum_params
        total_train_batches = len(train_loader) * args.epochs
        train_circuit_evals = circuit_evals_per_batch * total_train_batches
        infer_circuit_evals = len(test_loader)
    else:
        n_qubits = 0
        circuit_depth = 0
        num_gates = 0
        shots = 0
        train_circuit_evals = 0
        infer_circuit_evals = 0

    return {
        "total_trainable_params": count_trainable_params(model),
        "head_trainable_params": count_head_params(model),
        "quantum_params": quantum_params,
        "num_qubits": n_qubits,
        "circuit_depth": circuit_depth,
        "num_gates_approx": num_gates,
        "shots": shots,
        "train_circuit_evaluations_approx": train_circuit_evals,
        "test_circuit_evaluations_approx": infer_circuit_evals,
        "memory_usage_mb": memory_mb,
    }


def build_model(model_name, args):
    if model_name == "linear":
        head = LinearHead(args.compressor_dim, 10)
    elif model_name == "matched_mlp":
        head = MatchedMLPHead(args.compressor_dim, args.mlp_hidden_dim, 10)
    elif model_name == "quantum":
        head = QuantumHead(args.n_qubits, args.q_layers, 10, args.shots)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return FullModel(head, args.compressor_dim)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", nargs="+", default=["linear", "matched_mlp", "quantum"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--compressor_dim", type=int, default=4)
    parser.add_argument("--mlp_hidden_dim", type=int, default=8)

    parser.add_argument("--n_qubits", type=int, default=4)
    parser.add_argument("--q_layers", type=int, default=2)
    parser.add_argument("--shots", type=int, default=None)

    parser.add_argument("--train_subset", type=int, default=3000)
    parser.add_argument("--test_subset", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="results_fmnist_qtl.csv")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_loaders(
        args.batch_size,
        args.train_subset,
        args.test_subset,
        args.seed
    )

    all_results = []

    for model_name in args.models:
        print(f"\n==============================")
        print(f"Running model: {model_name}")
        print(f"==============================")

        model = build_model(model_name, args)

        train_time = train(model, train_loader, val_loader, args, device)
        test_metrics = evaluate(model, test_loader, device)
        resource_metrics = get_resource_info(model, args, train_loader, test_loader)

        row = {
            "dataset": "FashionMNIST",
            "task": "multiclass_10_class",
            "backbone": "Frozen ResNet18",
            "model_name": model_name,
            "seed": args.seed,
            "compressor_dim": args.compressor_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "train_time_sec": train_time,
            **test_metrics,
            **resource_metrics,
        }

        all_results.append(row)

        print(json.dumps(row, indent=2))

    df = pd.DataFrame(all_results)

    if os.path.exists(args.out_csv):
        old = pd.read_csv(args.out_csv)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved results to: {args.out_csv}")


if __name__ == "__main__":
    main()