import argparse, time, json, os, psutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import pennylane as qml


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


class QuantumHead(nn.Module):
    def __init__(
        self,
        n_qubits=8,
        n_layers=2,
        num_classes=10,
        shots=100,
        encoding="angle_y",
        ansatz="strongly_entangling",
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.encoding = encoding
        self.ansatz = ansatz

        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        if ansatz == "real_amplitudes":
            weight_shapes = {"weights": (n_layers, n_qubits)}

        elif ansatz == "efficient_su2":
            weight_shapes = {"weights": (n_layers, n_qubits, 2)}

        elif ansatz == "iqp_style":
            weight_shapes = {"weights": (n_layers, n_qubits)}

        elif ansatz == "qaoa_style":
            weight_shapes = {"weights": (n_layers, n_qubits, 2)}

        elif ansatz == "qft_style":
            weight_shapes = {"weights": (n_layers, n_qubits)}

        elif ansatz == "ghz_feature":
            weight_shapes = {"weights": (n_layers, n_qubits)}

        else:
            raise ValueError(f"Unknown ansatz: {ansatz}")

        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, weights):
            # -------------------------
            # Encodings
            # -------------------------
            if encoding == "angle_y":
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

            elif encoding == "angle_x":
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")

            elif encoding == "angle_z":
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Z")

            elif encoding == "angle_xyz":
                for i in range(n_qubits):
                    qml.RX(inputs[i], wires=i)
                    qml.RY(inputs[i], wires=i)
                    qml.RZ(inputs[i], wires=i)

            else:
                raise ValueError(f"Unknown encoding: {encoding}")

            # -------------------------
            # MQTBench-inspired ansatz variants
            # -------------------------
            if ansatz == "real_amplitudes":
                for l in range(n_layers):
                    for q in range(n_qubits):
                        qml.RY(weights[l, q], wires=q)
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])

            elif ansatz == "efficient_su2":
                for l in range(n_layers):
                    for q in range(n_qubits):
                        qml.RY(weights[l, q, 0], wires=q)
                        qml.RZ(weights[l, q, 1], wires=q)
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                    qml.CNOT(wires=[n_qubits - 1, 0])

            elif ansatz == "iqp_style":
                for l in range(n_layers):
                    for q in range(n_qubits):
                        qml.Hadamard(wires=q)
                    for q in range(n_qubits):
                        qml.RZ(inputs[q] + weights[l, q], wires=q)
                    for q in range(n_qubits - 1):
                        qml.CZ(wires=[q, q + 1])
                    qml.CZ(wires=[n_qubits - 1, 0])

            elif ansatz == "qaoa_style":
                for l in range(n_layers):
                    gamma = weights[l, :, 0]
                    beta = weights[l, :, 1]

                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
                        qml.RZ(gamma[q], wires=q + 1)
                        qml.CNOT(wires=[q, q + 1])

                    for q in range(n_qubits):
                        qml.RX(beta[q], wires=q)

            elif ansatz == "qft_style":
                for l in range(n_layers):
                    for q in range(n_qubits):
                        qml.RY(weights[l, q], wires=q)

                    for i in range(n_qubits):
                        qml.Hadamard(wires=i)
                        for j in range(i + 1, n_qubits):
                            qml.ControlledPhaseShift(np.pi / (2 ** (j - i)), wires=[j, i])

            elif ansatz == "ghz_feature":
                qml.Hadamard(wires=0)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])

                for l in range(n_layers):
                    for q in range(n_qubits):
                        qml.RY(inputs[q] + weights[l, q], wires=q)
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.weight_shapes = weight_shapes
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.readout = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        q_outputs = []
        for i in range(x.shape[0]):
            qi = self.qlayer(x[i])
            q_outputs.append(qi)
        q_out = torch.stack(q_outputs, dim=0)
        return self.readout(q_out)


class FullModel(nn.Module):
    def __init__(self, head, compressor_dim=8):
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


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_head_params(model):
    return sum(p.numel() for p in model.head.parameters() if p.requires_grad)


def count_quantum_params(model):
    return sum(
        p.numel()
        for name, p in model.head.named_parameters()
        if "qlayer" in name and p.requires_grad
    )


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

    return {
        "accuracy": accuracy_score(ys, preds),
        "precision_macro": precision_score(ys, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(ys, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(ys, preds, average="macro", zero_division=0),
        "auc_ovr_macro": roc_auc_score(ys, probs, multi_class="ovr", average="macro"),
        "inference_time_sec_total": infer_time,
        "inference_time_ms_per_sample": 1000 * infer_time / len(ys),
    }


def train(model, train_loader, val_loader, args, device):
    model.to(device)

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
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
            f"Val F1 {val_metrics['f1_macro']:.4f}",
            flush=True,
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

    quantum_params = count_quantum_params(model)
    circuit_evals_per_batch = 1 + 2 * quantum_params
    total_train_batches = len(train_loader) * args.epochs

    return {
        "total_trainable_params": count_trainable_params(model),
        "head_trainable_params": count_head_params(model),
        "quantum_params": quantum_params,
        "num_qubits": args.n_qubits,
        "circuit_depth": args.q_layers,
        "num_gates_approx": estimate_gates(args),
        "shots": args.shots,
        "train_circuit_evaluations_approx": circuit_evals_per_batch * total_train_batches,
        "test_circuit_evaluations_approx": len(test_loader),
        "memory_usage_mb": memory_mb,
    }


def estimate_gates(args):
    q = args.n_qubits
    l = args.q_layers

    if args.ansatz == "real_amplitudes":
        return l * (q + (q - 1))

    if args.ansatz == "efficient_su2":
        return l * ((2 * q) + q)

    if args.ansatz == "iqp_style":
        return l * (q + q + q)

    if args.ansatz == "qaoa_style":
        return l * ((2 * (q - 1) + (q - 1)) + q)

    if args.ansatz == "qft_style":
        return l * (q + q + (q * (q - 1) // 2))

    if args.ansatz == "ghz_feature":
        return (1 + (q - 1)) + l * (q + (q - 1))

    return -1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--compressor_dim", type=int, default=8)
    parser.add_argument("--n_qubits", type=int, default=8)
    parser.add_argument("--q_layers", type=int, default=2)
    parser.add_argument("--shots", type=int, default=100)

    parser.add_argument("--encoding", type=str, default="angle_y",
                        choices=["angle_y", "angle_x", "angle_z", "angle_xyz"])

    parser.add_argument("--ansatz", type=str, default="real_amplitudes",
                        choices=[
                            "real_amplitudes",
                            "efficient_su2",
                            "iqp_style",
                            "qaoa_style",
                            "qft_style",
                            "ghz_feature",
                        ])

    parser.add_argument("--train_subset", type=int, default=5000)
    parser.add_argument("--test_subset", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="results_fmnist_qtl_stronger_quantum.csv")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(json.dumps(vars(args), indent=2), flush=True)

    train_loader, val_loader, test_loader = get_loaders(
        args.batch_size,
        args.train_subset,
        args.test_subset,
        args.seed,
    )

    head = QuantumHead(
        n_qubits=args.n_qubits,
        n_layers=args.q_layers,
        num_classes=10,
        shots=args.shots,
        encoding=args.encoding,
        ansatz=args.ansatz,
    )

    model = FullModel(head, compressor_dim=args.compressor_dim)

    train_time = train(model, train_loader, val_loader, args, device)
    test_metrics = evaluate(model, test_loader, device)
    resource_metrics = get_resource_info(model, args, train_loader, test_loader)

    row = {
        "dataset": "FashionMNIST",
        "task": "multiclass_10_class",
        "backbone": "Frozen ResNet18",
        "model_name": "quantum",
        "ansatz": args.ansatz,
        "encoding": args.encoding,
        "seed": args.seed,
        "compressor_dim": args.compressor_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_time_sec": train_time,
        **test_metrics,
        **resource_metrics,
    }

    print(json.dumps(row, indent=2), flush=True)

    df = pd.DataFrame([row])

    if os.path.exists(args.out_csv):
        old = pd.read_csv(args.out_csv)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(args.out_csv, index=False)
    print(f"Saved results to: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()