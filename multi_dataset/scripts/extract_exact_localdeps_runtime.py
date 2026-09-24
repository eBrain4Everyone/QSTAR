#!/usr/bin/env python3

from pathlib import Path
import subprocess
import re
import pandas as pd
import numpy as np


ROOT = Path(
    "/scratch/sr7849/QC_June28/"
    "ICASSP_multiseed_new_datasets"
)

RUNS = {
    "kmnist": ROOT / "runs/kmnist/run_20260922_213814",
    "svhn": ROOT / "runs/svhn/run_20260922_213846",
}

TASK_TO_SEED = {
    0: 1,
    1: 42,
    2: 48,
    3: 550,
    4: 2026,
}


def read_manifest(path):
    vals = {}

    with open(path) as f:
        for line in f:
            line = line.strip()

            if not line or "=" not in line:
                continue

            k, v = line.split("=", 1)
            vals[k] = v

    return vals


def method_from_manifest_key(key):
    # We intentionally do not attribute classical bundle wall time
    # to Linear or MLP individually.
    if key in {"hist_classical", "assist_classical"}:
        return "classical_bundle"

    m = re.match(
        r"(?:hist|assist)_standard_q(\d+)d(\d+)",
        key
    )
    if m:
        return f"standard_qtl/q{m.group(1)}_d{m.group(2)}"

    m = re.match(
        r"(?:hist|assist)_ketgpt(160|180)",
        key
    )
    if m:
        return f"ketgpt/candidate_{m.group(1)}"

    m = re.match(
        r"(?:hist|assist)_adaptive(160|180)",
        key
    )
    if m:
        return f"adaptive/candidate_{m.group(1)}"

    return None


def stage_from_key(key):
    if key.startswith("hist_"):
        return "FULL_HISTORY"
    if key.startswith("assist_"):
        return "FLASHJT_ELIGIBLE"
    return ""


# ------------------------------------------------------------
# Read manifests and create job -> method mapping
# ------------------------------------------------------------

job_map = {}

for dataset, runroot in RUNS.items():
    manifest_path = runroot / "run_manifest.txt"

    if not manifest_path.exists():
        raise SystemExit(
            f"Missing manifest: {manifest_path}"
        )

    manifest = read_manifest(manifest_path)

    for key, value in manifest.items():
        if not (
            key.startswith("hist_")
            or key.startswith("assist_")
        ):
            continue

        method = method_from_manifest_key(key)

        if method is None:
            continue

        try:
            parent_job = int(value)
        except ValueError:
            continue

        job_map[parent_job] = {
            "dataset": dataset,
            "manifest_key": key,
            "method": method,
            "seed_role": stage_from_key(key),
            "run_root": str(runroot),
        }


# ------------------------------------------------------------
# Query exact Slurm allocation timing
# ------------------------------------------------------------

cmd = [
    "sacct",
    "-X",
    "-n",
    "-P",
    "-S",
    "2026-09-20",
    "-u",
    __import__("os").environ["USER"],
    "--format="
    "JobID,JobIDRaw,JobName,State,ElapsedRaw,Elapsed,"
    "Start,End,ExitCode,NodeList",
]

raw = subprocess.check_output(
    cmd,
    text=True,
)

rows = []

for line in raw.splitlines():
    p = line.split("|")

    if len(p) < 10:
        continue

    (
        jobid,
        jobid_raw,
        jobname,
        state,
        elapsed_raw,
        elapsed,
        start,
        end,
        exitcode,
        node,
    ) = p[:10]

    # Human-readable Slurm array IDs look like:
    # 18097675_0
    # Parse JobID, not JobIDRaw.
    m = re.match(r"^(\d+)_(\d+)$", jobid)

    if not m:
        continue

    parent = int(m.group(1))
    task = int(m.group(2))

    if parent not in job_map:
        continue

    if task not in TASK_TO_SEED:
        continue

    meta = job_map[parent]
    seed = TASK_TO_SEED[task]

    # Validate role/task consistency.
    if meta["seed_role"] == "FULL_HISTORY":
        if seed not in {1, 42, 48}:
            continue

    if meta["seed_role"] == "FLASHJT_ELIGIBLE":
        if seed not in {550, 2026}:
            continue

    try:
        sec = float(elapsed_raw)
    except Exception:
        sec = np.nan

    rows.append({
        "dataset": meta["dataset"],
        "method": meta["method"],
        "seed": seed,
        "seed_role": meta["seed_role"],
        "manifest_key": meta["manifest_key"],
        "parent_job_id": parent,
        "array_task_id": task,
        "job_id": jobid,
        "job_id_raw": jobid_raw,
        "job_name": jobname,
        "state": state,
        "elapsed_raw_sec": sec,
        "elapsed": elapsed,
        "elapsed_min": (
            sec / 60.0
            if not pd.isna(sec)
            else np.nan
        ),
        "elapsed_h": (
            sec / 3600.0
            if not pd.isna(sec)
            else np.nan
        ),
        "start": start,
        "end": end,
        "exit_code": exitcode,
        "node": node,
        "time_source": "SLURM_ELAPSEDRAW_EXACT",
    })


df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit(
        "No matching KMNIST/SVHN Slurm array tasks found."
    )


# ------------------------------------------------------------
# Mark whether timing is suitable as a per-method Table-I time
# ------------------------------------------------------------

df["table1_time_usable"] = (
    ~df["method"].eq("classical_bundle")
)

df["timing_note"] = np.where(
    df["method"].eq("classical_bundle"),
    (
        "Bundled allocation contains multiple classical heads; "
        "use script-reported per-model train_time_sec instead."
    ),
    (
        "Exact Slurm elapsed wall-clock for this "
        "dataset/method/seed allocation."
    ),
)


# ------------------------------------------------------------
# Save per-seed file
# ------------------------------------------------------------

outdir = ROOT / "results/runtime_audit"
outdir.mkdir(parents=True, exist_ok=True)

seed_file = (
    outdir /
    "table1_exact_slurm_runtime_kmnist_svhn_by_seed.csv"
)

df.to_csv(
    seed_file,
    index=False,
)


# ------------------------------------------------------------
# Aggregate only completed usable Table-I allocations
# ------------------------------------------------------------

done = df[
    df["table1_time_usable"]
    & df["state"].str.startswith("COMPLETED")
].copy()

agg_rows = []

for (dataset, method), g in done.groupby(
    ["dataset", "method"]
):
    vals = pd.to_numeric(
        g["elapsed_h"],
        errors="coerce",
    ).dropna()

    seeds = sorted(
        int(x)
        for x in g["seed"].unique()
    )

    agg_rows.append({
        "dataset": dataset,
        "method": method,
        "n_completed_timed_seeds": len(vals),
        "completed_seeds": "|".join(
            map(str, seeds)
        ),
        "mean_wallclock_h": (
            vals.mean()
            if len(vals)
            else np.nan
        ),
        "std_wallclock_h": (
            vals.std(ddof=1)
            if len(vals) > 1
            else np.nan
        ),
        "min_wallclock_h": (
            vals.min()
            if len(vals)
            else np.nan
        ),
        "max_wallclock_h": (
            vals.max()
            if len(vals)
            else np.nan
        ),
        "time_source": "SLURM_ELAPSEDRAW_EXACT",
    })

agg = pd.DataFrame(agg_rows)

agg_file = (
    outdir /
    "table1_exact_slurm_runtime_kmnist_svhn_aggregate.csv"
)

agg.to_csv(
    agg_file,
    index=False,
)


# ------------------------------------------------------------
# Console
# ------------------------------------------------------------

print()
print("=" * 110)
print("EXACT KMNIST / SVHN SLURM RUNTIME BY SEED")
print("=" * 110)

show = df[
    [
        "dataset",
        "method",
        "seed",
        "seed_role",
        "state",
        "elapsed_h",
        "job_id",
    ]
].copy()

print(show.to_string(index=False))

print()
print("=" * 110)
print("COMPLETED TABLE-I RUNTIME AGGREGATES")
print("=" * 110)

if agg.empty:
    print("No completed usable allocations.")
else:
    out = agg.copy()

    out["wallclock_entry"] = out.apply(
        lambda r:
            (
                f"{r['mean_wallclock_h']:.2f} ± "
                f"{r['std_wallclock_h']:.2f} h"
            )
            if not pd.isna(r["std_wallclock_h"])
            else f"{r['mean_wallclock_h']:.2f} h",
        axis=1,
    )

    print(
        out[
            [
                "dataset",
                "method",
                "n_completed_timed_seeds",
                "completed_seeds",
                "wallclock_entry",
            ]
        ].to_string(index=False)
    )

print()
print("Saved:")
print(seed_file)
print(agg_file)
