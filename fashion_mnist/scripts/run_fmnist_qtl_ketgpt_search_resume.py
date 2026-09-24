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

        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),

        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0),

        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0),

    )





# -------------------------

# KetGPT Candidate Filtering

# -------------------------

def count_trainable_params_from_ops(circuit_ops):

    param_gates = []

    for op in circuit_ops:

        name = op.name.lower()

        if len(op.parameters) > 0 and name in ["u2", "ry", "u1", "rz", "rx"]:

            param_gates.append(op)



    num_params = 0

    for op in param_gates:

        name = op.name.lower()

        if name == "u2":

            num_params += 2

        else:

            num_params += 1



    return num_params





def get_num_qubits(circuit_ops):

    wires = sorted(set(int(w) for op in circuit_ops for w in op.wires))

    if not wires:

        return 0

    return max(wires) + 1





def apply_ketgpt_ops(circuit_ops, weights, n_qubits):

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





def filter_ketgpt_candidates(

    target_qubits=8,

    min_params=20,

    max_params=80,

    max_gates=150,

    max_candidates=10,

):

    print("Loading KetGPT dataset...", flush=True)

    [ds] = qml.data.load("ketgpt")



    print(f"Total KetGPT circuits: {len(ds.circuits)}", flush=True)



    candidates = []



    for idx, circuit_ops in enumerate(ds.circuits):

        try:

            num_qubits = get_num_qubits(circuit_ops)



            if num_qubits != target_qubits:

                continue



            num_params = count_trainable_params_from_ops(circuit_ops)

            gate_count = len(circuit_ops)



            if num_params < min_params:

                continue

            if num_params > max_params:

                continue

            if gate_count > max_gates:

                continue



            dev = qml.device("default.qubit", wires=num_qubits)



            @qml.qnode(dev, interface="autograd")

            def test_circuit(test_weights):

                for i in range(num_qubits):

                    qml.RX(0.1, wires=i)



                apply_ketgpt_ops(circuit_ops, test_weights, num_qubits)



                for i in range(num_qubits - 1):

                    qml.CNOT(wires=[i, i + 1])



                return qml.expval(qml.PauliZ(0))



            test_weights = np.random.normal(0, 0.01, size=(num_params,))

            test_output = test_circuit(test_weights)



            if test_output is None or not np.isfinite(test_output):

                continue



            candidates.append({

                "candidate_rank": len(candidates),

                "ketgpt_id": idx,

                "num_qubits": num_qubits,

                "num_params": num_params,

                "gate_count": gate_count,

                "ops": circuit_ops,

            })



            print(

                f"Accepted KetGPT circuit {idx}: "

                f"qubits={num_qubits}, params={num_params}, gates={gate_count}",

                flush=True,

            )



            if len(candidates) >= max_candidates:

                break



        except Exception as e:

            print(f"Skipping circuit {idx}: {e}", flush=True)



    print(f"\nSelected candidates: {len(candidates)}", flush=True)



    for c in candidates:

        if c["ketgpt_id"] == 22:
            print("Skipping completed candidate 22", flush=True)
            continue


        print(

            f"Rank {c['candidate_rank']} | "

            f"KetGPT ID {c['ketgpt_id']} | "

            f"Qubits {c['num_qubits']} | "

            f"Params {c['num_params']} | "

            f"Gates {c['gate_count']}",

            flush=True,

        )



    return candidates





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

# KetGPT Quantum Head

# -------------------------

class KetGPTQuantumHead(nn.Module):

    def __init__(self, circuit_ops, n_qubits=8, num_params=48, num_classes=10, shots=100):

        super().__init__()



        self.circuit_ops = circuit_ops

        self.n_qubits = n_qubits

        self.num_params = num_params

        self.shots = shots



        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)



        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")

        def circuit(inputs, weights):

            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")



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



            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]



        self.weight_shapes = {"weights": (num_params,)}

        self.qlayer = qml.qnn.TorchLayer(circuit, self.weight_shapes)

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





# -------------------------

# Metrics

# -------------------------

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

            f"Val F1 {val_metrics['f1_macro']:.4f}",

            flush=True,

        )



        if val_metrics["accuracy"] > best_val:

            best_val = val_metrics["accuracy"]

            best_state = {k: v.cpu() for k, v in model.state_dict().items()}



    train_time = time.time() - start



    model.load_state_dict(best_state)

    return train_time





def append_result(row, out_csv):

    df = pd.DataFrame([row])



    if os.path.exists(out_csv):

        old = pd.read_csv(out_csv)

        df = pd.concat([old, df], ignore_index=True)



    df.to_csv(out_csv, index=False)





def main():

    parser = argparse.ArgumentParser()



    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--weight_decay", type=float, default=1e-5)



    parser.add_argument("--compressor_dim", type=int, default=8)

    parser.add_argument("--target_qubits", type=int, default=8)

    parser.add_argument("--shots", type=int, default=100)



    parser.add_argument("--min_params", type=int, default=20)

    parser.add_argument("--max_params", type=int, default=80)

    parser.add_argument("--max_gates", type=int, default=150)

    parser.add_argument("--max_candidates", type=int, default=5)



    parser.add_argument("--train_subset", type=int, default=5000)

    parser.add_argument("--test_subset", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)



    parser.add_argument("--out_csv", type=str, default="results_fmnist_ketgpt_candidates.csv")

    parser.add_argument("--candidate_csv", type=str, default="ketgpt_filtered_candidates.csv")



    args = parser.parse_args()



    torch.manual_seed(args.seed)

    np.random.seed(args.seed)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}", flush=True)

    print(json.dumps(vars(args), indent=2), flush=True)



    candidates = filter_ketgpt_candidates(

        target_qubits=args.target_qubits,

        min_params=args.min_params,

        max_params=args.max_params,

        max_gates=args.max_gates,

        max_candidates=args.max_candidates,

    )



    candidate_rows = [

        {

            "candidate_rank": c["candidate_rank"],

            "ketgpt_id": c["ketgpt_id"],

            "num_qubits": c["num_qubits"],

            "num_params": c["num_params"],

            "gate_count": c["gate_count"],

        }

        for c in candidates

    ]



    pd.DataFrame(candidate_rows).to_csv(args.candidate_csv, index=False)

    print(f"Saved filtered candidates to {args.candidate_csv}", flush=True)



    train_loader, val_loader, test_loader = get_loaders(

        args.batch_size,

        args.train_subset,

        args.test_subset,

        args.seed,

    )



    for c in candidates:

        if int(c["ketgpt_id"]) == 22:
            print("Skipping completed candidate 22 before training", flush=True)
            continue

        print("\n==========================================", flush=True)

        print(

            f"Training KetGPT candidate rank={c['candidate_rank']} "

            f"id={c['ketgpt_id']} params={c['num_params']} gates={c['gate_count']}",

            flush=True,

        )

        print("==========================================", flush=True)



        head = KetGPTQuantumHead(

            circuit_ops=c["ops"],

            n_qubits=args.target_qubits,

            num_params=c["num_params"],

            num_classes=10,

            shots=args.shots,

        )



        model = FullModel(head, compressor_dim=args.compressor_dim)



        train_time = train(model, train_loader, val_loader, args, device)

        test_metrics = evaluate(model, test_loader, device)



        quantum_params = count_quantum_params(model)

        process = psutil.Process(os.getpid())

        memory_mb = process.memory_info().rss / 1024**2



        circuit_evals_per_batch = 1 + 2 * quantum_params

        total_train_batches = len(train_loader) * args.epochs



        row = {

            "dataset": "FashionMNIST",

            "task": "multiclass_10_class",

            "backbone": "Frozen ResNet18",

            "model_name": "ketgpt_quantum",

            "candidate_rank": c["candidate_rank"],

            "ketgpt_id": c["ketgpt_id"],

            "seed": args.seed,

            "compressor_dim": args.compressor_dim,

            "epochs": args.epochs,

            "batch_size": args.batch_size,

            "lr": args.lr,

            "train_time_sec": train_time,

            **test_metrics,

            "total_trainable_params": count_trainable_params(model),

            "head_trainable_params": count_head_params(model),

            "quantum_params": quantum_params,

            "num_qubits": args.target_qubits,

            "circuit_depth": "ketgpt_variable",

            "num_gates_approx": c["gate_count"],

            "shots": args.shots,

            "train_circuit_evaluations_approx": circuit_evals_per_batch * total_train_batches,

            "test_circuit_evaluations_approx": len(test_loader),

            "memory_usage_mb": memory_mb,

        }



        print(json.dumps(row, indent=2), flush=True)

        append_result(row, args.out_csv)

        print(f"Appended result to {args.out_csv}", flush=True)





if __name__ == "__main__":

    main()
