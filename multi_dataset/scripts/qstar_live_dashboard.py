#!/usr/bin/env python3

import os
import re
import json
import glob
import subprocess
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

ROOT = Path("/scratch/sr7849/QC_June28/ICASSP_multiseed_new_datasets")
RUNS_ROOT = ROOT / "runs"

DATASETS = ["cifar10", "kmnist", "svhn"]

# How far back sacct should search.
SACCT_LOOKBACK = "2026-09-20"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def run_cmd(cmd):
    try:
        return subprocess.check_output(
            cmd,
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def latest_run(dataset):
    d = RUNS_ROOT / dataset
    if not d.exists():
        return None

    runs = sorted(
        [p for p in d.glob("run_*") if p.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    return runs[0] if runs else None


def normalize_state(state):
    state = state.strip().upper()

    # sacct may emit things like CANCELLED+ or FAILED+
    state = state.rstrip("+")

    if state.startswith("COMPLETED"):
        return "COMPLETED"
    if state.startswith("RUNNING"):
        return "RUNNING"
    if state.startswith("PENDING"):
        return "PENDING"
    if state.startswith("FAILED"):
        return "FAILED"
    if state.startswith("CANCELLED"):
        return "CANCELLED"
    if state.startswith("TIMEOUT"):
        return "TIMEOUT"
    if "OUT_OF_MEMORY" in state:
        return "OUT_OF_MEMORY"
    if state.startswith("PREEMPTED"):
        return "PREEMPTED"
    if state.startswith("NODE_FAIL"):
        return "NODE_FAIL"

    return state


def state_icon(state):
    return {
        "COMPLETED": "DONE",
        "RUNNING": "RUN ",
        "PENDING": "WAIT",
        "FAILED": "FAIL",
        "CANCELLED": "CANC",
        "TIMEOUT": "TIME",
        "OUT_OF_MEMORY": "OOM ",
        "PREEMPTED": "PRE ",
        "NODE_FAIL": "NODE",
    }.get(state, state[:4])


# ------------------------------------------------------------
# Slurm accounting
# ------------------------------------------------------------

def get_sacct():
    """
    Use -X to avoid .batch/.extern job-step noise.
    Returns actual QSTAR allocations/tasks known to Slurm.
    """

    cmd = (
        f"sacct -X -n -P "
        f"-S {SACCT_LOOKBACK} "
        f"-u {os.environ.get('USER', '')} "
        f"--format=JobIDRaw,JobName,State,Elapsed,ExitCode,NodeList"
    )

    txt = run_cmd(cmd)

    jobs = []

    for line in txt.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue

        jobid, name, state, elapsed, exitcode, nodelist = parts[:6]

        if not name.startswith("qstar-"):
            continue

        # Ignore job steps. Keep:
        # 18064331
        # 18064326_32
        if not re.match(r"^\d+(?:_\d+)?$", jobid):
            continue

        jobs.append({
            "jobid": jobid,
            "name": name,
            "state": normalize_state(state),
            "elapsed": elapsed,
            "exitcode": exitcode,
            "node": nodelist,
        })

    return jobs


def print_scheduler_summary(jobs):
    print("=" * 96)
    print("SLURM / SACCT — ALL QSTAR JOB ALLOCATIONS")
    print("=" * 96)

    counts = Counter(j["state"] for j in jobs)

    preferred = [
        "COMPLETED",
        "RUNNING",
        "PENDING",
        "FAILED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "CANCELLED",
        "PREEMPTED",
        "NODE_FAIL",
    ]

    print(f"Tracked QSTAR allocations/tasks: {len(jobs)}")
    print()

    for st in preferred:
        if counts[st]:
            print(f"{st:<15}: {counts[st]:>4}")

    extras = sorted(set(counts) - set(preferred))
    for st in extras:
        print(f"{st:<15}: {counts[st]:>4}")

    bad_states = {
        "FAILED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "CANCELLED",
        "PREEMPTED",
        "NODE_FAIL",
    }

    bad = [j for j in jobs if j["state"] in bad_states]

    if bad:
        print()
        print("PROBLEM JOBS")
        print("-" * 96)
        print(
            f"{'JOBID':<18} {'STATE':<15} {'NAME':<28} "
            f"{'ELAPSED':<12} {'EXIT':<10} {'NODE'}"
        )
        for j in bad:
            print(
                f"{j['jobid']:<18} "
                f"{j['state']:<15} "
                f"{j['name']:<28} "
                f"{j['elapsed']:<12} "
                f"{j['exitcode']:<10} "
                f"{j['node']}"
            )


# ------------------------------------------------------------
# Current queue
# ------------------------------------------------------------

def get_squeue():
    fmt = "%i|%P|%j|%T|%M|%R"
    txt = run_cmd(
        f"squeue -h -u {os.environ.get('USER', '')} -o '{fmt}'"
    )

    rows = []

    for line in txt.splitlines():
        p = line.split("|")
        if len(p) != 6:
            continue

        jobid, partition, name, state, elapsed, reason = p

        if not name.startswith("qstar-"):
            continue

        rows.append({
            "jobid": jobid,
            "partition": partition,
            "name": name,
            "state": normalize_state(state),
            "elapsed": elapsed,
            "reason": reason,
        })

    return rows


def print_active_queue(rows):
    print()
    print("=" * 96)
    print("CURRENT QSTAR QUEUE")
    print("=" * 96)

    running = [x for x in rows if x["state"] == "RUNNING"]
    pending = [x for x in rows if x["state"] == "PENDING"]

    print(f"RUNNING: {len(running)}")
    print(f"PENDING: {len(pending)}")

    print()
    print("RUNNING NOW")
    print("-" * 96)
    print(
        f"{'JOBID':<22} {'PARTITION':<10} {'NAME':<24} "
        f"{'TIME':<12} {'NODE'}"
    )

    if not running:
        print("(none)")
    else:
        for j in running:
            print(
                f"{j['jobid']:<22} "
                f"{j['partition']:<10} "
                f"{j['name']:<24} "
                f"{j['elapsed']:<12} "
                f"{j['reason']}"
            )


# ------------------------------------------------------------
# Trajectory parsing
# ------------------------------------------------------------

EPOCH_KEYS = {
    "epoch",
    "current_epoch",
    "epochs_executed",
    "completed_epoch",
    "last_epoch",
}


def recursively_collect_epochs(obj):
    vals = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()

            if lk in EPOCH_KEYS:
                try:
                    vals.append(int(v))
                except Exception:
                    pass

            vals.extend(recursively_collect_epochs(v))

    elif isinstance(obj, list):
        for x in obj:
            vals.extend(recursively_collect_epochs(x))

    return vals


def get_epoch(path):
    try:
        with open(path, "r") as f:
            obj = json.load(f)
    except Exception:
        return None

    vals = recursively_collect_epochs(obj)

    # A common trajectory format is a list with one object per epoch.
    if isinstance(obj, list) and obj:
        vals.append(len(obj))

    # Another common format: {"history": [...]}
    if isinstance(obj, dict):
        for key in [
            "history",
            "epochs",
            "trajectory",
            "epoch_history",
            "training_history",
        ]:
            v = obj.get(key)
            if isinstance(v, list):
                vals.append(len(v))

    vals = [x for x in vals if isinstance(x, int) and 0 <= x <= 1000]

    return max(vals) if vals else None


def infer_seed(path):
    s = str(path)

    patterns = [
        r"seed[_-](\d+)",
        r"/seed(\d+)(?:/|_|$)",
    ]

    for pat in patterns:
        m = re.search(pat, s, re.I)
        if m:
            return int(m.group(1))

    return None


def clean_experiment_name(path, runroot):
    """
    Build a compact identifier from the trajectory's relative path.
    """
    try:
        rel = path.relative_to(runroot)
    except Exception:
        rel = path

    parts = list(rel.parts[:-1])

    remove = {
        "checkpoints",
        "checkpoint",
        "trajectories",
        "trajectory",
        "history",
        "logs",
    }

    cleaned = []

    for p in parts:
        if p.lower() in remove:
            continue

        if re.match(r"seed[_-]?\d+", p, re.I):
            continue

        cleaned.append(p)

    # Last few components are usually enough to identify model.
    if len(cleaned) > 4:
        cleaned = cleaned[-4:]

    return "/".join(cleaned) or "(unknown)"


def find_result_csv_for_seed(runroot, seed):
    if seed is None:
        return False

    patterns = [
        f"**/seed_{seed}.csv",
        f"**/seed-{seed}.csv",
        f"**/seed{seed}.csv",
    ]

    for pat in patterns:
        if list(runroot.glob(pat)):
            return True

    return False


def collect_trajectories(runroot):
    rows = []

    files = list(runroot.rglob("trajectory.json"))

    for f in files:
        epoch = get_epoch(f)
        seed = infer_seed(f)
        exp = clean_experiment_name(f, runroot)

        rows.append({
            "experiment": exp,
            "seed": seed,
            "epoch": epoch,
            "path": f,
        })

    rows.sort(
        key=lambda x: (
            x["experiment"],
            x["seed"] if x["seed"] is not None else -1,
        )
    )

    return rows


def expected_end(seed):
    # Historical full seeds always target 10.
    # Assisted seeds may validly stop at epoch 6 if FlashJT gate passes.
    if seed in {1, 42, 48}:
        return "10"
    if seed in {550, 2026}:
        return "6/10"
    return "10"


def trajectory_status(row):
    ep = row["epoch"]
    seed = row["seed"]

    if ep is None:
        return "?"

    if seed in {1, 42, 48}:
        if ep >= 10:
            return "DONE"
        return "RUN"

    if seed in {550, 2026}:
        if ep >= 10:
            return "DONE"
        if ep >= 6:
            return "6+"
        return "RUN"

    if ep >= 10:
        return "DONE"

    return "RUN"


def print_dataset_progress(dataset):
    rr = latest_run(dataset)

    print()
    print("=" * 96)
    print(f"{dataset.upper()} — MODEL / EPOCH PROGRESS")
    print("=" * 96)

    if rr is None:
        print("No run directory found.")
        return

    print(f"Run root: {rr}")

    rows = collect_trajectories(rr)

    if not rows:
        print("No trajectory.json files found yet.")
        return

    counts = Counter(trajectory_status(x) for x in rows)

    print(
        f"Trajectories: {len(rows)}  |  "
        f"DONE={counts['DONE']}  "
        f"RUN={counts['RUN']}  "
        f"6+={counts['6+']}  "
        f"UNKNOWN={counts['?']}"
    )

    print()
    print(
        f"{'STATUS':<7} "
        f"{'SEED':<7} "
        f"{'EPOCH':<10} "
        f"{'EXPERIMENT'}"
    )
    print("-" * 96)

    for r in rows:
        seed = str(r["seed"]) if r["seed"] is not None else "?"
        ep = str(r["epoch"]) if r["epoch"] is not None else "?"
        target = expected_end(r["seed"])
        st = trajectory_status(r)

        print(
            f"{st:<7} "
            f"{seed:<7} "
            f"{ep + '/' + target:<10} "
            f"{r['experiment']}"
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print("\033[2J\033[H", end="")

    print("=" * 96)
    print("QSTAR LIVE HPC DASHBOARD")
    print("=" * 96)
    print("Updated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Root:", ROOT)

    jobs = get_sacct()
    print_scheduler_summary(jobs)

    q = get_squeue()
    print_active_queue(q)

    for ds in DATASETS:
        print_dataset_progress(ds)

    print()
    print("=" * 96)
    print("NOTES")
    print("=" * 96)
    print(
        "DONE/RUN epoch status comes from trajectory.json, while scheduler state "
        "comes from Slurm."
    )
    print(
        "For seeds 550/2026, epoch 6 may be a valid FlashJT-assisted stopping "
        "point; 10 means the run continued to full training."
    )


if __name__ == "__main__":
    main()
