#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd
import numpy as np

if len(sys.argv) != 2:
    raise SystemExit(
        "usage: python scripts/preview_selected_aggregates.py RUN_ROOT"
    )

root = Path(sys.argv[1])
rows = []

def add_classical(seed):
    p = root / f"table1/classical/seed_{seed}.csv"
    if not p.exists():
        return

    df = pd.read_csv(p)

    for _, r in df.iterrows():
        model = str(r.get("model_name", ""))

        if model == "linear":
            label = "Classical Linear"
        elif model == "matched_mlp":
            label = "MLP"
        else:
            continue

        epochs = int(r.get("epochs_executed", 10))
        if epochs < 10:
            continue

        rows.append({
            "model": label,
            "seed": seed,
            "accuracy": float(r["accuracy"]),
            "f1": float(r["f1_macro"]),
            "auc": float(r["auc_ovr_macro"]),
            "train_time_sec": float(r.get("train_time_sec", np.nan)),
        })


def add_ketgpt(seed):
    p = root / f"table1/ketgpt/seed_{seed}.csv"
    if not p.exists():
        return

    df = pd.read_csv(p)

    for _, r in df.iterrows():
        cid = r.get("ketgpt_id", None)

        if pd.isna(cid):
            # fallback if candidate encoded in model/config string
            text = " ".join(str(x) for x in r.values)
            if "160" in text:
                cid = 160
            elif "180" in text:
                cid = 180
            else:
                continue

        cid = int(float(cid))

        if cid not in (160, 180):
            continue

        epochs = int(r.get("epochs_executed", 10))
        if epochs < 10:
            continue

        rows.append({
            "model": f"KetGPT #{cid}",
            "seed": seed,
            "accuracy": float(r["accuracy"]),
            "f1": float(r["f1_macro"]),
            "auc": float(r["auc_ovr_macro"]),
            "train_time_sec": float(r.get("train_time_sec", np.nan)),
        })


for seed in [1, 42, 48, 550, 2026]:
    add_classical(seed)
    add_ketgpt(seed)

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No completed selected results found.")

print("\nPER-SEED RESULTS")
print("=" * 100)

show = df.copy()
show["accuracy"] *= 100
show["train_h"] = show["train_time_sec"] / 3600

print(
    show[
        ["model", "seed", "accuracy", "f1", "auc", "train_h"]
    ].to_string(
        index=False,
        formatters={
            "accuracy": lambda x: f"{x:.2f}",
            "f1": lambda x: f"{x:.4f}",
            "auc": lambda x: f"{x:.4f}",
            "train_h": lambda x: f"{x:.4f}",
        }
    )
)

print("\nCURRENT AGGREGATES")
print("=" * 100)
print(
    f"{'Model':<22}"
    f"{'n':>4}"
    f"{'Seeds':>22}"
    f"{'Accuracy (%)':>20}"
    f"{'F1':>18}"
    f"{'AUC':>18}"
    f"{'Train (h)':>18}"
)
print("-" * 100)

order = [
    "Classical Linear",
    "MLP",
    "KetGPT #160",
    "KetGPT #180",
]

for model in order:
    g = df[df.model == model].sort_values("seed")

    if len(g) == 0:
        continue

    n = len(g)
    ddof = 1 if n > 1 else 0

    acc_mean = g.accuracy.mean() * 100
    acc_std = g.accuracy.std(ddof=ddof) * 100

    f1_mean = g.f1.mean()
    f1_std = g.f1.std(ddof=ddof)

    auc_mean = g.auc.mean()
    auc_std = g.auc.std(ddof=ddof)

    t = g.train_time_sec / 3600
    t_mean = t.mean()
    t_std = t.std(ddof=ddof)

    seeds = ",".join(str(x) for x in g.seed.tolist())

    print(
        f"{model:<22}"
        f"{n:>4}"
        f"{seeds:>22}"
        f"{acc_mean:>8.2f} ± {acc_std:<7.2f}"
        f"{f1_mean:>8.3f} ± {f1_std:<6.3f}"
        f"{auc_mean:>8.3f} ± {auc_std:<6.3f}"
        f"{t_mean:>8.4f} ± {t_std:<6.4f}"
    )

print("\nNOTE")
print("  n=3 rows are PREVIEW statistics over seeds 1,42,48.")
print("  Final paper values should use all five seeds.")
print("  Standard deviation is sample SD (ddof=1).")
print("  Cached-feature Train(h) is head-training time, not original end-to-end ResNet time.")
