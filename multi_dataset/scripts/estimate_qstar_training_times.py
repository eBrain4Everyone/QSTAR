#!/usr/bin/env python3

from pathlib import Path
import json
import re
import math
from datetime import datetime

import numpy as np
import pandas as pd


ROOT = Path(
    "/scratch/sr7849/QC_June28/"
    "ICASSP_multiseed_new_datasets"
)

RUNS = {
    "cifar10": ROOT / "runs/cifar10/run_20260921_164946",
    "kmnist": ROOT / "runs/kmnist/run_20260922_213814",
    "svhn": ROOT / "runs/svhn/run_20260922_213846",
}

SEEDS = {1, 42, 48, 550, 2026}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def infer_seed(path):
    s = str(path)

    for pat in [
        r"seed[_-](\d+)",
        r"/seed(\d+)(?:/|_|$)",
    ]:
        m = re.search(pat, s, re.I)
        if m:
            seed = int(m.group(1))
            if seed in SEEDS:
                return seed

    return None


def identify(path, runroot):
    s = str(path.relative_to(runroot)).lower()

    if "table1/classical/linear" in s:
        return "classical/linear"

    if "table1/classical/matched_mlp" in s:
        return "classical/matched_mlp"

    m = re.search(
        r"table1/standard_qtl/q(\d+)[_-]d(\d+)",
        s
    )
    if m:
        return f"standard_qtl/q{m.group(1)}_d{m.group(2)}"

    m = re.search(
        r"table1/ketgpt/(?:candidate_)?(160|180)",
        s
    )
    if m:
        return f"ketgpt/candidate_{m.group(1)}"

    return None


def get_epoch(path):
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return np.nan

    values = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if str(k).lower() in {
                    "epoch",
                    "current_epoch",
                    "epochs_executed",
                    "completed_epoch",
                    "final_epoch",
                }:
                    try:
                        values.append(int(v))
                    except Exception:
                        pass
                walk(v)

        elif isinstance(x, list):
            for y in x:
                walk(y)

    walk(obj)

    return max(values) if values else np.nan


def candidate_dir(traj):
    """
    Find the smallest useful experiment/seed directory containing
    trajectory.json.
    """
    p = traj.parent

    # trajectory is often under checkpoints/ or similar.
    while p != p.parent:
        name = p.name.lower()

        if (
            re.search(r"seed[_-]?\d+", name)
            or name.startswith("candidate_")
        ):
            return p

        if len(p.parts) <= 3:
            break

        p = p.parent

    return traj.parent


def artifact_wallclock(directory):
    """
    Estimate elapsed span from files actually created/modified inside
    the experiment directory.

    We avoid using directories themselves because they may have been
    created well before execution.
    """
    files = []

    for p in directory.rglob("*"):
        if not p.is_file():
            continue

        # Ignore very generic metadata where possible.
        if p.name.startswith("."):
            continue

        try:
            t = p.stat().st_mtime
        except Exception:
            continue

        files.append((t, p))

    if len(files) < 2:
        return np.nan, "", ""

    files.sort(key=lambda x: x[0])

    start_t, start_file = files[0]
    end_t, end_file = files[-1]

    elapsed = end_t - start_t

    if elapsed < 0:
        return np.nan, "", ""

    return elapsed, str(start_file), str(end_file)


# ------------------------------------------------------------
# Main scan
# ------------------------------------------------------------

rows = []

for dataset, runroot in RUNS.items():

    if not runroot.exists():
        continue

    for traj in runroot.rglob("trajectory.json"):

        method = identify(traj, runroot)
        seed = infer_seed(traj)

        if method is None or seed is None:
            continue

        epoch = get_epoch(traj)

        exp_dir = candidate_dir(traj)

        elapsed_sec, first_file, last_file = artifact_wallclock(exp_dir)

        rows.append({
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "epoch": epoch,
            "estimated_wallclock_sec": elapsed_sec,
            "estimated_wallclock_min": (
                elapsed_sec / 60.0
                if not pd.isna(elapsed_sec)
                else np.nan
            ),
            "estimated_wallclock_h": (
                elapsed_sec / 3600.0
                if not pd.isna(elapsed_sec)
                else np.nan
            ),
            "time_source": "artifact_timestamp_span",
            "experiment_dir": str(exp_dir),
            "first_artifact": first_file,
            "last_artifact": last_file,
            "trajectory_file": str(traj),
        })


df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("No Table-I trajectories found.")

# Remove duplicate trajectory discoveries if any.
df = (
    df.sort_values(
        ["dataset", "method", "seed", "estimated_wallclock_sec"]
    )
    .drop_duplicates(
        ["dataset", "method", "seed"],
        keep="last"
    )
)

outdir = ROOT / "results/runtime_audit"
outdir.mkdir(parents=True, exist_ok=True)

seed_out = outdir / "table1_runtime_estimates_by_seed.csv"
df.to_csv(seed_out, index=False)


# ------------------------------------------------------------
# Aggregate over whatever seeds currently exist
# ------------------------------------------------------------

agg_rows = []

for (dataset, method), g in df.groupby(
    ["dataset", "method"]
):

    vals = pd.to_numeric(
        g["estimated_wallclock_h"],
        errors="coerce"
    ).dropna()

    agg_rows.append({
        "dataset": dataset,
        "method": method,
        "n_timed_seeds": len(vals),
        "seeds": "|".join(
            map(str, sorted(g["seed"].unique()))
        ),
        "epochs": "|".join(
            f"{int(r.seed)}:{int(r.epoch)}"
            for r in g.itertuples()
            if not pd.isna(r.epoch)
        ),
        "estimated_train_time_h_mean": (
            vals.mean() if len(vals) else np.nan
        ),
        "estimated_train_time_h_std": (
            vals.std(ddof=1)
            if len(vals) > 1
            else np.nan
        ),
        "estimated_train_time_h_min": (
            vals.min() if len(vals) else np.nan
        ),
        "estimated_train_time_h_max": (
            vals.max() if len(vals) else np.nan
        ),
        "time_source": "artifact_timestamp_span",
    })


agg = pd.DataFrame(agg_rows)

agg_out = outdir / "table1_runtime_estimates_aggregate.csv"
agg.to_csv(agg_out, index=False)


print()
print("=" * 100)
print("PER-SEED RUNTIME ESTIMATES")
print("=" * 100)

show = df[
    [
        "dataset",
        "method",
        "seed",
        "epoch",
        "estimated_wallclock_h",
    ]
].copy()

print(show.to_string(index=False))

print()
print("=" * 100)
print("AGGREGATE ESTIMATED RUNTIME")
print("=" * 100)

display = agg.copy()

display["time_h"] = display.apply(
    lambda r:
        (
            f"{r['estimated_train_time_h_mean']:.2f} ± "
            f"{r['estimated_train_time_h_std']:.2f}"
        )
        if (
            not pd.isna(r["estimated_train_time_h_mean"])
            and not pd.isna(r["estimated_train_time_h_std"])
        )
        else (
            f"{r['estimated_train_time_h_mean']:.2f}"
            if not pd.isna(
                r["estimated_train_time_h_mean"]
            )
            else ""
        ),
    axis=1,
)

print(
    display[
        [
            "dataset",
            "method",
            "n_timed_seeds",
            "seeds",
            "time_h",
        ]
    ].to_string(index=False)
)

print()
print("Saved:")
print(seed_out)
print(agg_out)
