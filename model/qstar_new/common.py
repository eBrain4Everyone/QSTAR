from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FULL_HISTORY_SEEDS = {1, 42, 48}
FLASHJT_ASSISTED_SEEDS = {550, 2026}
MONITORED_METRICS = [
    "train_loss",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "auc_ovr_macro",
]


def dataset_display_name(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "")
    mapping = {
        "fashionmnist": "Fashion-MNIST",
        "fmnist": "Fashion-MNIST",
        "cifar10": "CIFAR-10",
        "kmnist": "KMNIST",
        "svhn": "SVHN",
    }
    if key not in mapping:
        raise ValueError(f"Unknown dataset: {name}")
    return mapping[key]


def canonical_dataset_name(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "")
    mapping = {
        "fashionmnist": "fashionmnist",
        "fmnist": "fashionmnist",
        "cifar10": "cifar10",
        "kmnist": "kmnist",
        "svhn": "svhn",
    }
    if key not in mapping:
        raise ValueError(f"Unknown dataset: {name}")
    return mapping[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y_true, probs):
    try:
        return roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except Exception:
        return np.nan


def compute_metrics(y_true, probs) -> Dict[str, float]:
    preds = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision_macro": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "auc_ovr_macro": float(safe_auc(y_true, probs)),
    }


def evaluate_model(model, loader, device) -> Dict[str, float]:
    model.eval()
    ys, probs = [], []
    start = time.time()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs.append(prob)
            ys.extend(y.numpy())
    infer_time = time.time() - start
    ys = np.asarray(ys)
    probs = np.vstack(probs)
    metrics = compute_metrics(ys, probs)
    metrics["inference_time_sec_total"] = float(infer_time)
    metrics["inference_time_ms_per_sample"] = float(1000.0 * infer_time / len(ys))
    return metrics


def collect_probs(model, loader, device):
    model.eval()
    ys, probs = [], []
    start = time.time()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
            ys.extend(y.numpy())
    return np.asarray(ys), np.vstack(probs), float(time.time() - start)


def atomic_json_dump(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_torch_save(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def load_feature_cache(cache_dir: str | Path, batch_size: int, seed: int, num_workers: int = 0):
    """
    Cache stores deterministic frozen-ResNet features for the first N training/test examples.
    The original QSTAR split policy is then reproduced: a seed-specific 15% random validation split.
    """
    cache_dir = Path(cache_dir)
    train_blob = torch.load(cache_dir / "train.pt", map_location="cpu", weights_only=False)
    test_blob = torch.load(cache_dir / "test.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((cache_dir / "manifest.json").read_text())

    train_ds = TensorDataset(train_blob["features"].float(), train_blob["labels"].long())
    test_ds = TensorDataset(test_blob["features"].float(), test_blob["labels"].long())

    val_size = int(0.15 * len(train_ds))
    train_size = len(train_ds) - val_size
    split_gen = torch.Generator().manual_seed(int(seed))
    train_set, val_set = random_split(train_ds, [train_size, val_size], generator=split_gen)

    # Match the original pipeline's shuffle semantics. Global RNG is checkpointed/restored.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, manifest


class FeatureInputModel(nn.Module):
    """Same trainable QSTAR compressor/head, but receives cached 512-D frozen-backbone features."""
    def __init__(self, head: nn.Module, compressor_dim: int):
        super().__init__()
        self.compressor = nn.Sequential(nn.Linear(512, compressor_dim), nn.Tanh())
        self.head = head

    def forward(self, x):
        return self.head(self.compressor(x))


def count_trainable_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_head_params(model) -> int:
    return int(sum(p.numel() for p in model.head.parameters() if p.requires_grad))


@dataclass
class FlashJTDecision:
    allow_stop: bool
    reason: str
    max_relative_error: Optional[float] = None
    mean_relative_error: Optional[float] = None
    checked_values: int = 0

    def to_dict(self):
        return asdict(self)


def _stable_text_number(value: Any) -> float:
    s = str(value)
    h = 2166136261
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h / 0xFFFFFFFF


def _trajectory_feature_vector(epochs: Dict[int, Dict[str, float]], metadata: Dict[str, Any], metrics: List[str]):
    feats: List[float] = []
    for key in [
        "dataset", "table", "model_family", "config_id", "candidate_id",
        "num_qubits", "circuit_depth", "shots", "lr", "batch_size",
    ]:
        value = metadata.get(key, 0)
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            feats.append(float(value))
        else:
            feats.append(_stable_text_number(value))
    for metric in metrics:
        vals = [float(epochs[e][metric]) for e in (1, 2, 3)]
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("nonfinite early metric")
        feats.extend(vals)
        feats.append(vals[1] - vals[0])
        feats.append(vals[2] - vals[1])
    return np.asarray(feats, dtype=np.float64)


def _trajectory_target_vector(epochs: Dict[int, Dict[str, float]], metrics: List[str], end_epoch=10):
    vals: List[float] = []
    for e in range(4, end_epoch + 1):
        for metric in metrics:
            v = float(epochs[e][metric])
            if not math.isfinite(v):
                raise ValueError("nonfinite target metric")
            vals.append(v)
    return np.asarray(vals, dtype=np.float64)


class FlashJT:
    """Small-data trajectory predictor trained only on fully observed 10-epoch QSTAR runs."""
    def __init__(
        self,
        monitored_metrics: Optional[List[str]] = None,
        min_history_runs: int = 20,
        max_individual_error: float = 0.10,
        max_mean_error: float = 0.05,
        epsilon: float = 1e-6,
    ):
        self.monitored_metrics = monitored_metrics or list(MONITORED_METRICS)
        self.min_history_runs = int(min_history_runs)
        self.max_individual_error = float(max_individual_error)
        self.max_mean_error = float(max_mean_error)
        self.epsilon = float(epsilon)
        self.model = None
        self.history_count = 0

    def fit_from_run_root(self, run_root: str | Path) -> int:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.multioutput import MultiOutputRegressor

        paths = sorted(Path(run_root).glob("checkpoints/**/trajectory.json"))
        X, Y = [], []
        for path in paths:
            try:
                record = json.loads(path.read_text())
                if not record.get("completed_full_run", False):
                    continue
                epoch_list = record.get("epochs", [])
                epochs = {int(x["epoch"]): x["metrics"] for x in epoch_list}
                if not set(range(1, 11)).issubset(epochs):
                    continue
                X.append(_trajectory_feature_vector(epochs, record.get("metadata", {}), self.monitored_metrics))
                Y.append(_trajectory_target_vector(epochs, self.monitored_metrics, 10))
            except Exception:
                continue
        self.history_count = len(X)
        if not X:
            self.model = None
            return 0
        base = ExtraTreesRegressor(
            n_estimators=256,
            random_state=2026,
            n_jobs=1,
        )
        self.model = MultiOutputRegressor(base, n_jobs=1)
        self.model.fit(np.vstack(X), np.vstack(Y))
        return self.history_count

    def predict(self, early_epochs: Dict[int, Dict[str, float]], metadata: Dict[str, Any]):
        if self.model is None:
            raise RuntimeError("FlashJT is not fitted")
        x = _trajectory_feature_vector(early_epochs, metadata, self.monitored_metrics).reshape(1, -1)
        y = self.model.predict(x).reshape(-1)
        out: Dict[int, Dict[str, float]] = {}
        i = 0
        for e in range(4, 11):
            out[e] = {}
            for metric in self.monitored_metrics:
                out[e][metric] = float(y[i])
                i += 1
        return out

    def verify(self, seed: int, forecast: Dict[int, Dict[str, float]], actual: Dict[int, Dict[str, float]]):
        if int(seed) not in FLASHJT_ASSISTED_SEEDS:
            return FlashJTDecision(False, "seed_not_flashjt_assisted")
        if self.history_count < self.min_history_runs:
            return FlashJTDecision(False, f"insufficient_history:{self.history_count}<{self.min_history_runs}")
        errors: List[float] = []
        for e in (4, 5, 6):
            if e not in forecast or e not in actual:
                return FlashJTDecision(False, f"missing_epoch:{e}")
            for metric in self.monitored_metrics:
                if metric not in forecast[e] or metric not in actual[e]:
                    return FlashJTDecision(False, f"missing_metric:{metric}@{e}")
                p = float(forecast[e][metric])
                a = float(actual[e][metric])
                if not (math.isfinite(p) and math.isfinite(a)):
                    return FlashJTDecision(False, f"nonfinite:{metric}@{e}")
                err = abs(p - a) / max(abs(a), self.epsilon)
                errors.append(err)
                if err > self.max_individual_error:
                    return FlashJTDecision(
                        False,
                        f"individual_error_failed:{metric}@{e}",
                        float(max(errors)),
                        float(np.mean(errors)),
                        len(errors),
                    )
        max_err = float(max(errors)) if errors else None
        mean_err = float(np.mean(errors)) if errors else None
        if mean_err is None or mean_err > self.max_mean_error:
            return FlashJTDecision(False, "mean_error_failed", max_err, mean_err, len(errors))
        return FlashJTDecision(True, "flashjt_validated", max_err, mean_err, len(errors))


class TrainingController:
    """
    Common training loop for all QSTAR heads.

    Full-history seeds (1,42,48) always run 10 epochs.
    Assisted seeds (550,2026) may stop at epoch 6 after FlashJT verification,
    otherwise conservative validation-accuracy early stopping can act after epoch 6.
    """
    def __init__(
        self,
        *,
        run_root: str | Path,
        checkpoint_dir: str | Path,
        metadata: Dict[str, Any],
        seed: int,
        epochs: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        resume: bool = True,
        force_full: bool = False,
        enable_flashjt: bool = True,
        enable_early_stopping: bool = True,
        early_patience: int = 3,
        early_min_delta: float = 0.001,
        flashjt_min_history: int = 20,
    ):
        self.run_root = Path(run_root)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.resume = bool(resume)
        self.force_full = bool(force_full)
        self.enable_flashjt = bool(enable_flashjt) and not self.force_full
        self.enable_early_stopping = bool(enable_early_stopping) and not self.force_full
        self.early_patience = int(early_patience)
        self.early_min_delta = float(early_min_delta)
        self.flashjt = FlashJT(min_history_runs=flashjt_min_history)
        self.latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        self.best_path = self.checkpoint_dir / "checkpoint_best.pt"
        self.final_path = self.checkpoint_dir / "checkpoint_final.pt"
        self.trajectory_path = self.checkpoint_dir / "trajectory.json"
        self.forecast_path = self.checkpoint_dir / "flashjt_forecast_epoch3.json"
        self.verification_path = self.checkpoint_dir / "flashjt_verification_epoch6.json"

    def _save_trajectory(self, history: List[Dict[str, Any]], completed_full_run: bool, stop_reason: str):
        atomic_json_dump({
            "metadata": self.metadata,
            "seed": self.seed,
            "epochs": history,
            "completed_full_run": bool(completed_full_run),
            "stop_reason": stop_reason,
        }, self.trajectory_path)

    def train(self, model, train_loader, val_loader, device):
        model.to(device)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        start_epoch = 1
        best_val = -float("inf")
        best_epoch = 0
        early_best = -float("inf")
        history: List[Dict[str, Any]] = []
        cumulative_train_sec = 0.0
        stop_reason = "completed"
        bad_epochs = 0
        forecast = None

        if self.resume and self.latest_path.exists():
            ckpt = torch.load(self.latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            restore_rng_state(ckpt.get("rng_state", {}))
            start_epoch = int(ckpt["epoch"]) + 1
            best_val = float(ckpt.get("best_val", -float("inf")))
            best_epoch = int(ckpt.get("best_epoch", 0))
            early_best = float(ckpt.get("early_best", best_val))
            history = ckpt.get("history", [])
            cumulative_train_sec = float(ckpt.get("cumulative_train_sec", 0.0))
            bad_epochs = int(ckpt.get("bad_epochs", 0))
            print(f"Resumed from epoch {start_epoch}", flush=True)

        if self.forecast_path.exists():
            try:
                forecast = json.loads(self.forecast_path.read_text()).get("forecast")
                if forecast is not None:
                    forecast = {int(k): v for k, v in forecast.items()}
            except Exception:
                forecast = None

        session_start = time.time()
        actual_epoch_metrics: Dict[int, Dict[str, float]] = {
            int(x["epoch"]): x["metrics"] for x in history if "epoch" in x and "metrics" in x
        }

        for epoch in range(start_epoch, self.epochs + 1):
            model.train()
            total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

            train_loss = total_loss / max(len(train_loader), 1)
            val_metrics = evaluate_model(model, val_loader, device)
            epoch_metrics = {
                "train_loss": float(train_loss),
                "accuracy": float(val_metrics["accuracy"]),
                "precision_macro": float(val_metrics["precision_macro"]),
                "recall_macro": float(val_metrics["recall_macro"]),
                "f1_macro": float(val_metrics["f1_macro"]),
                "auc_ovr_macro": float(val_metrics["auc_ovr_macro"]),
            }
            actual_epoch_metrics[epoch] = epoch_metrics
            history = [x for x in history if int(x.get("epoch", -1)) != epoch]
            history.append({"epoch": epoch, "metrics": epoch_metrics})
            history.sort(key=lambda x: int(x["epoch"]))

            val_acc = epoch_metrics["accuracy"]
            checkpoint_improved = val_acc > best_val
            if checkpoint_improved:
                best_val = val_acc
                best_epoch = epoch

            early_improved = val_acc > (early_best + self.early_min_delta)
            if early_improved:
                early_best = val_acc
                bad_epochs = 0
            elif epoch > 6:
                bad_epochs += 1

            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "best_epoch": best_epoch,
                "early_best": early_best,
                "history": history,
                "metadata": self.metadata,
                "rng_state": capture_rng_state(),
                "cumulative_train_sec": cumulative_train_sec + (time.time() - session_start),
                "bad_epochs": bad_epochs,
            }
            atomic_torch_save(payload, self.latest_path)
            if checkpoint_improved:
                atomic_torch_save(payload, self.best_path)
            if epoch in (3, 6, 10):
                atomic_torch_save(payload, self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pt")

            self._save_trajectory(history, completed_full_run=(epoch >= self.epochs), stop_reason="running")

            print(
                f"Epoch {epoch}/{self.epochs} | Loss {train_loss:.4f} | "
                f"Val Acc {val_acc:.4f} | Val F1 {epoch_metrics['f1_macro']:.4f}",
                flush=True,
            )

            # FlashJT is intentionally disabled for the three full-history seeds.
            assisted = self.seed in FLASHJT_ASSISTED_SEEDS
            if epoch == 3 and self.enable_flashjt and assisted:
                count = self.flashjt.fit_from_run_root(self.run_root)
                enough_history = count >= self.flashjt.min_history_runs
                if self.flashjt.model is not None and enough_history:
                    early = {e: actual_epoch_metrics[e] for e in (1, 2, 3)}
                    forecast = self.flashjt.predict(early, self.metadata)
                    atomic_json_dump({
                        "seed": self.seed,
                        "history_count": count,
                        "minimum_history_runs": self.flashjt.min_history_runs,
                        "metadata": self.metadata,
                        "forecast": forecast,
                    }, self.forecast_path)
                    print(f"FlashJT forecast saved (history={count})", flush=True)
                else:
                    reason = (
                        f"insufficient_history_at_forecast:{count}<{self.flashjt.min_history_runs}"
                        if self.flashjt.model is not None and not enough_history
                        else "no_fitted_history"
                    )
                    forecast = None
                    atomic_json_dump({
                        "seed": self.seed,
                        "history_count": count,
                        "minimum_history_runs": self.flashjt.min_history_runs,
                        "metadata": self.metadata,
                        "forecast": None,
                        "reason": reason,
                    }, self.forecast_path)
                    print(f"FlashJT forecast skipped: {reason}", flush=True)

            if epoch == 6 and self.enable_flashjt and assisted and forecast is not None:
                # Refit only to recover history_count; verification uses the saved epoch-3 forecast unchanged.
                self.flashjt.fit_from_run_root(self.run_root)
                actual = {e: actual_epoch_metrics[e] for e in (4, 5, 6)}
                decision = self.flashjt.verify(self.seed, forecast, actual)
                atomic_json_dump({
                    **decision.to_dict(),
                    "forecast_source": str(self.forecast_path),
                    "actual_epochs": actual,
                }, self.verification_path)
                print(f"FlashJT verification: {decision.to_dict()}", flush=True)
                if decision.allow_stop:
                    stop_reason = "flashjt_validated"
                    break

            if (
                self.enable_early_stopping
                and assisted
                and epoch > 6
                and bad_epochs >= self.early_patience
            ):
                stop_reason = "early_stopping"
                break

        cumulative_train_sec += time.time() - session_start
        epochs_executed = max([int(x["epoch"]) for x in history], default=0)
        completed_full = epochs_executed >= self.epochs
        if completed_full:
            stop_reason = "completed"

        # Load the genuine best checkpoint selected by validation accuracy.
        selected_path = self.best_path if self.best_path.exists() else self.latest_path
        selected = torch.load(selected_path, map_location=device, weights_only=False)
        model.load_state_dict(selected["model_state_dict"])

        final_payload = {
            **selected,
            "selected_best_epoch": int(selected.get("best_epoch", best_epoch)),
            "epochs_executed": epochs_executed,
            "stop_reason": stop_reason,
            "cumulative_train_sec": cumulative_train_sec,
            "history": history,
        }
        atomic_torch_save(final_payload, self.final_path)
        self._save_trajectory(history, completed_full_run=completed_full, stop_reason=stop_reason)

        return {
            "train_time_sec": float(cumulative_train_sec),
            "epochs_executed": int(epochs_executed),
            "best_epoch": int(final_payload.get("selected_best_epoch", 0)),
            "stop_reason": stop_reason,
            "flashjt_forecast_path": str(self.forecast_path) if self.forecast_path.exists() else "",
            "flashjt_verification_path": str(self.verification_path) if self.verification_path.exists() else "",
            "checkpoint_latest": str(self.latest_path),
            "checkpoint_best": str(self.best_path) if self.best_path.exists() else "",
            "checkpoint_final": str(self.final_path),
        }


def replace_or_append_csv_row(out_csv: str | Path, row: Dict[str, Any], key_fields: List[str]):
    import pandas as pd

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        if len(old):
            mask = np.ones(len(old), dtype=bool)
            for key in key_fields:
                if key not in old.columns:
                    mask &= False
                    continue
                target = row.get(key)
                if pd.isna(target):
                    mask &= old[key].isna().to_numpy()
                else:
                    mask &= (old[key].astype(str) == str(target)).to_numpy()
            old = old.loc[~mask]
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out_csv, index=False)
