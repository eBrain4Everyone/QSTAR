#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from qstar_new.common import canonical_dataset_name, dataset_display_name, IMAGENET_MEAN, IMAGENET_STD, atomic_json_dump, atomic_torch_save


def transform_for(dataset: str):
    dataset = canonical_dataset_name(dataset)
    ops = [transforms.Resize((224, 224))]
    # Preserve grayscale->3ch behavior for the two grayscale datasets.
    # Keep native RGB information for CIFAR-10/SVHN.
    if dataset in {"fashionmnist", "kmnist"}:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(ops)


def load_dataset(dataset: str, data_root: Path, train: bool, download: bool):
    dataset = canonical_dataset_name(dataset)
    tfm = transform_for(dataset)
    if dataset == "fashionmnist":
        return datasets.FashionMNIST(data_root, train=train, download=download, transform=tfm)
    if dataset == "kmnist":
        return datasets.KMNIST(data_root, train=train, download=download, transform=tfm)
    if dataset == "cifar10":
        return datasets.CIFAR10(data_root, train=train, download=download, transform=tfm)
    if dataset == "svhn":
        return datasets.SVHN(data_root, split="train" if train else "test", download=download, transform=tfm)
    raise ValueError(dataset)


class FrozenResNet18Features(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        # Strictly frozen means BatchNorm stays in eval mode. This is what makes
        # static feature caching mathematically consistent.
        super().train(False)
        self.features.eval()
        return self

    def forward(self, x):
        self.features.eval()
        with torch.no_grad():
            return torch.flatten(self.features(x), 1)


def extract(model, dataset, batch_size, device, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            feats.append(model(x).cpu())
            labels.append(torch.as_tensor(y).long().cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def sha256_text(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["fashionmnist", "cifar10", "kmnist", "svhn"])
    p.add_argument("--data-root", default="/scratch/sr7849/QC_June28/data")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-subset", type=int, default=5000)
    p.add_argument("--test-subset", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--download", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    dataset_name = canonical_dataset_name(args.dataset)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    transform_desc = (
        "Resize(224,224); "
        + ("Grayscale(3); " if dataset_name in {"fashionmnist", "kmnist"} else "native-RGB; ")
        + "ToTensor; ImageNet normalization"
    )
    manifest_core = {
        "cache_version": 2,
        "dataset": dataset_name,
        "dataset_display": dataset_display_name(dataset_name),
        "train_subset": int(args.train_subset),
        "test_subset": int(args.test_subset),
        "subset_policy": "first_N_examples_before_seed_specific_15pct_validation_split",
        "transform": transform_desc,
        "backbone": "torchvision resnet18 ResNet18_Weights.DEFAULT",
        "backbone_feature_dim": 512,
        "strict_frozen_backbone": True,
        "batchnorm_mode": "eval",
    }
    manifest_core["manifest_sha256"] = sha256_text(json.dumps(manifest_core, sort_keys=True))

    manifest_path = out / "manifest.json"
    if not args.force and manifest_path.exists() and (out / "train.pt").exists() and (out / "test.pt").exists():
        old = json.loads(manifest_path.read_text())
        if old.get("manifest_sha256") == manifest_core["manifest_sha256"]:
            print(f"VALID CACHE: {out}")
            print(json.dumps(old, indent=2))
            return

    train_full = load_dataset(dataset_name, data_root, train=True, download=args.download)
    test_full = load_dataset(dataset_name, data_root, train=False, download=args.download)
    if args.train_subset:
        train_full = Subset(train_full, list(range(min(args.train_subset, len(train_full)))))
    if args.test_subset:
        test_full = Subset(test_full, list(range(min(args.test_subset, len(test_full)))))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Preparing {dataset_display_name(dataset_name)} features on {device}", flush=True)
    backbone = FrozenResNet18Features().to(device)
    t0 = time.time()
    train_features, train_labels = extract(backbone, train_full, args.batch_size, device, args.num_workers)
    train_extract_sec = time.time() - t0
    t1 = time.time()
    test_features, test_labels = extract(backbone, test_full, args.batch_size, device, args.num_workers)
    test_extract_sec = time.time() - t1

    atomic_torch_save({"features": train_features, "labels": train_labels}, out / "train.pt")
    atomic_torch_save({"features": test_features, "labels": test_labels}, out / "test.pt")
    manifest = {
        **manifest_core,
        "train_examples": int(len(train_labels)),
        "test_examples": int(len(test_labels)),
        "train_tensor_shape": list(train_features.shape),
        "test_tensor_shape": list(test_features.shape),
        "train_feature_extraction_time_sec": float(train_extract_sec),
        "test_feature_extraction_time_sec": float(test_extract_sec),
        "total_feature_extraction_time_sec": float(train_extract_sec + test_extract_sec),
    }
    atomic_json_dump(manifest, manifest_path)
    print(f"CACHE READY: {out}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
