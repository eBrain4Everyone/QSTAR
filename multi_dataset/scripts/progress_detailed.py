#!/usr/bin/env python3
import json
import sys
import subprocess
from pathlib import Path
from collections import Counter

SEEDS = [1, 42, 48, 550, 2026]
QCFG = [(4,1),(4,2),(4,4),(6,1),(6,2),(6,4),(8,1),(8,2)]

if len(sys.argv) != 2:
    raise SystemExit("usage: python scripts/progress_detailed.py RUN_ROOT")

root = Path(sys.argv[1])

def traj(path):
    if not path.exists():
        return 0, "waiting", False
    try:
        r = json.loads(path.read_text())
        epochs = r.get("epochs", [])
        ep = max([int(x["epoch"]) for x in epochs], default=0)
        reason = r.get("stop_reason", "unknown")
        full = bool(r.get("completed_full_run", False))
        return ep, reason, full
    except Exception:
        return 0, "bad_json", False

rows = []

# ------------------------------------------------------------
# Table I: Classical
# ------------------------------------------------------------
for seed in SEEDS:
    base = root / "checkpoints/table1/classical" / f"seed_{seed}"
    out = root / f"table1/classical/seed_{seed}.csv"

    for model, label in [
        ("linear", "Classical Linear"),
        ("matched_mlp", "MLP"),
    ]:
        ep, reason, full = traj(base / model / "trajectory.json")
        rows.append(("T1", label, seed, ep, reason, full, out.exists()))

# ------------------------------------------------------------
# Table I: Standard QTL
# ------------------------------------------------------------
for seed in SEEDS:
    for q,d in QCFG:
        base = (
            root / "checkpoints/table1/standard_qtl"
            / f"seed_{seed}_q{q}_d{d}"
            / f"q{q}_d{d}"
        )
        out = root / f"table1/standard_qtl/seed_{seed}_q{q}_d{d}.csv"
        ep, reason, full = traj(base / "trajectory.json")
        rows.append((
            "T1", f"Standard QTL q{q} d{d}",
            seed, ep, reason, full, out.exists()
        ))

# ------------------------------------------------------------
# Table I: KetGPT #160 / #180
# ------------------------------------------------------------
for seed in SEEDS:
    base = root / "checkpoints/table1/ketgpt" / f"seed_{seed}"
    out = root / f"table1/ketgpt/seed_{seed}.csv"

    for cid in [160,180]:
        ep, reason, full = traj(
            base / f"candidate_{cid}" / "trajectory.json"
        )
        rows.append((
            "T1", f"KetGPT #{cid}",
            seed, ep, reason, full, out.exists()
        ))

# ------------------------------------------------------------
# Table II: each adaptive candidate trains:
# confidence -> matched MLP -> KetGPT
# ------------------------------------------------------------
for cid in [160,180]:
    for seed in SEEDS:
        base = (
            root / "checkpoints/table2"
            / f"adaptive{cid}"
            / f"seed_{seed}"
        )
        out = root / f"table2/adaptive{cid}/seed_{seed}.csv"

        for model,label in [
            ("confidence_linear", f"Conf. Linear [#{cid}]"),
            ("matched_mlp",       f"Fallback MLP [#{cid}]"),
            ("ketgpt_quantum",    f"Fallback KetGPT #{cid}"),
        ]:
            ep, reason, full = traj(base / model / "trajectory.json")
            rows.append((
                "T2", label,
                seed, ep, reason, full, out.exists()
            ))

def state(ep, reason, full):
    if full or ep >= 10:
        return "DONE"
    if reason == "flashjt_validated":
        return "FLASHJT"
    if reason == "early_stopping":
        return "EARLY"
    if ep > 0:
        return "RUN"
    return "WAIT"

print()
print("QSTAR DETAILED PROGRESS")
print("RUN:", root)
print("=" * 108)
print(
    f"{'Tbl':<4}"
    f"{'Experiment':<29}"
    f"{'Seed':>6}"
    f"{'Epoch':>9}"
    f"{'Status':>11}"
    f"{'Stop reason':>23}"
    f"{'CSV':>8}"
)
print("-" * 108)

counts = Counter()

for table, name, seed, ep, reason, full, csv_exists in rows:
    s = state(ep, reason, full)
    counts[s] += 1
    print(
        f"{table:<4}"
        f"{name:<29}"
        f"{seed:>6}"
        f"{str(ep) + '/10':>9}"
        f"{s:>11}"
        f"{reason:>23}"
        f"{('YES' if csv_exists else '-'):>8}"
    )

print("=" * 108)

total = len(rows)
done = counts["DONE"]
started = sum(1 for r in rows if r[3] > 0)
flash = counts["FLASHJT"]
early = counts["EARLY"]

print(
    f"MODEL RUNS: {total} total | "
    f"{done} full-10 | "
    f"{started} started | "
    f"{flash} FlashJT stops | "
    f"{early} early stops | "
    f"{counts['WAIT']} waiting"
)

# ------------------------------------------------------------
# Scheduler failure / timeout check
# ------------------------------------------------------------
manifest = root / "run_manifest.txt"
jobs = []

if manifest.exists():
    for line in manifest.read_text().splitlines():
        if "=" not in line:
            continue
        k,v = line.split("=",1)
        if (
            k.startswith("phase1_")
            or k.startswith("phase2_")
            or k in {"cache_job","summary_job"}
        ):
            if v.strip().isdigit():
                jobs.append(v.strip())

if jobs:
    print("\nSLURM JOB/ARRAY STATUS")
    print("-" * 108)
    subprocess.run([
        "sacct",
        "-X",
        "-j", ",".join(jobs),
        "--format=JobID,JobName%28,State,Elapsed,ExitCode"
    ])

print()
print("Legend:")
print("  DONE    = actual 10 epochs completed")
print("  RUN     = at least one epoch recorded; run not final")
print("  FLASHJT = stopped after validated FlashJT decision")
print("  EARLY   = ordinary early stopping")
print("  WAIT    = no epoch recorded yet")
print()
