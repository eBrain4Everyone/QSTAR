#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(
    "/scratch/sr7849/QC_June28/"
    "ICASSP_multiseed_new_datasets"
)

# ============================================================
# These are ONLY the rows currently missing from the manuscript
# ============================================================

TABLE1_MISSING = [
    # CIFAR-10
    ("cifar10", "standard_qtl/q4_d4", "Standard QTL q4 d4"),
    ("cifar10", "standard_qtl/q6_d4", "Standard QTL q6 d4"),
    ("cifar10", "standard_qtl/q8_d2", "Standard QTL q8 d2"),

    # KMNIST
    ("kmnist", "standard_qtl/q4_d4", "Standard QTL q4 d4"),
    ("kmnist", "standard_qtl/q6_d2", "Standard QTL q6 d2"),
    ("kmnist", "standard_qtl/q6_d4", "Standard QTL q6 d4"),
    ("kmnist", "standard_qtl/q8_d2", "Standard QTL q8 d2"),
    ("kmnist", "ketgpt/candidate_180", "KetGPT #180"),

    # SVHN
    ("svhn", "standard_qtl/q4_d4", "Standard QTL q4 d4"),
    ("svhn", "standard_qtl/q6_d2", "Standard QTL q6 d2"),
    ("svhn", "standard_qtl/q6_d4", "Standard QTL q6 d4"),
    ("svhn", "standard_qtl/q8_d2", "Standard QTL q8 d2"),
    ("svhn", "ketgpt/candidate_180", "KetGPT #180"),
]


# ============================================================
# Helpers
# ============================================================

def safe_num(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def fmt_pm(mean, std, digits):
    mean = safe_num(mean)
    std = safe_num(std)

    if pd.isna(mean):
        return "--"

    if pd.isna(std):
        return f"{mean:.{digits}f}"

    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def metric_pm(row, metric, digits=3, percentage=False):
    mean = safe_num(row.get(f"{metric}_mean", np.nan))
    std = safe_num(row.get(f"{metric}_std", np.nan))

    if pd.isna(mean):
        # fall back to existing formatted entry
        entry = row.get(f"{metric}_entry", "")
        if pd.notna(entry) and str(entry).strip():
            return str(entry)
        return "--"

    if percentage and abs(mean) <= 1.5:
        mean *= 100.0
        if not pd.isna(std):
            std *= 100.0

    return fmt_pm(mean, std, digits)


def table1_time(row, dataset, method, runtime_df):
    """
    Prefer exact Slurm wall-clock for KMNIST/SVHN if all five
    allocations are completed. Otherwise use Table-I recorded time.
    """

    if runtime_df is not None:
        q = runtime_df[
            (runtime_df["dataset"].astype(str).str.lower() == dataset)
            & (runtime_df["method"].astype(str) == method)
        ]

        if len(q):
            rr = q.iloc[-1]

            n = safe_int(
                rr.get("n_completed_timed_seeds", 0)
            )

            if n == 5:
                mean = safe_num(
                    rr.get("mean_wallclock_h", np.nan)
                )
                std = safe_num(
                    rr.get("std_wallclock_h", np.nan)
                )

                if not pd.isna(mean):
                    return (
                        fmt_pm(mean, std, 2)
                        + " h [Slurm]"
                    )

    # Exporter-created hour columns.
    mean = safe_num(
        row.get("train_time_hours_mean", np.nan)
    )
    std = safe_num(
        row.get("train_time_hours_std", np.nan)
    )

    if not pd.isna(mean):
        return fmt_pm(mean, std, 2) + " h"

    # Raw seconds if hour columns did not exist.
    mean_sec = safe_num(
        row.get("train_time_sec_mean", np.nan)
    )
    std_sec = safe_num(
        row.get("train_time_sec_std", np.nan)
    )

    if not pd.isna(mean_sec):
        return fmt_pm(
            mean_sec / 3600.0,
            (
                std_sec / 3600.0
                if not pd.isna(std_sec)
                else np.nan
            ),
            2,
        ) + " h"

    # Last fallback.
    entry = row.get("train_time_hours_entry", "")

    if pd.notna(entry) and str(entry).strip():
        return str(entry) + " h"

    return "--"


# ============================================================
# Find newest Table-I export
# ============================================================

export_root = ROOT / "results/table_exports"

exports = sorted(
    export_root.glob("export_*"),
    key=lambda p: p.stat().st_mtime,
)

if not exports:
    raise SystemExit(
        "ERROR: no results/table_exports/export_* directory found."
    )

EXPORT = exports[-1]

t1_file = EXPORT / "table1_all_aggregates.csv"

if not t1_file.exists():
    raise SystemExit(
        f"ERROR: missing {t1_file}"
    )

t1 = pd.read_csv(t1_file)


# ============================================================
# Exact KMNIST/SVHN Slurm runtime
# ============================================================

runtime_file = (
    ROOT
    / "results/runtime_audit/"
      "table1_exact_slurm_runtime_kmnist_svhn_aggregate.csv"
)

runtime_df = (
    pd.read_csv(runtime_file)
    if runtime_file.exists()
    else None
)


# ============================================================
# TABLE I
# ============================================================

print()
print("=" * 120)
print("TABLE I — ROWS CURRENTLY MISSING FROM PAPER")
print("=" * 120)

ready_count = 0

for dataset, method, label in TABLE1_MISSING:

    q = t1[
        (t1["dataset"].astype(str).str.lower() == dataset)
        & (t1["method"].astype(str) == method)
    ]

    print()
    print(f"[{dataset.upper()}] {label}")

    if q.empty:
        print("  STATUS   : NOT DISCOVERED")
        continue

    r = q.iloc[-1]

    n = safe_int(r.get("n_final_seeds", 0))

    readiness = str(
        r.get("readiness", "")
    )

    protocol = str(
        r.get("training_protocol", "")
    )

    if n < 5:
        print(f"  STATUS   : STILL INCOMPLETE ({n}/5)")

        missing = r.get(
            "missing_final_seeds",
            ""
        )
        progress = r.get(
            "seed_epoch_progress",
            ""
        )

        if pd.notna(missing) and str(missing).strip():
            print(f"  MISSING  : {missing}")

        if pd.notna(progress) and str(progress).strip():
            print(f"  PROGRESS : {progress}")

        continue

    ready_count += 1

    acc = metric_pm(
        r,
        "accuracy",
        digits=1,
        percentage=True,
    )

    f1 = metric_pm(
        r,
        "f1_macro",
        digits=3,
        percentage=False,
    )

    auc = metric_pm(
        r,
        "auc_ovr_macro",
        digits=3,
        percentage=False,
    )

    train = table1_time(
        r,
        dataset,
        method,
        runtime_df,
    )

    print("  STATUS   : 5/5 COMPLETE")
    print(f"  CHECK    : {readiness}")
    print(f"  PROTOCOL : {protocol}")
    print(f"  Acc (%)  : {acc}")
    print(f"  F1       : {f1}")
    print(f"  AUC      : {auc}")
    print(f"  Train    : {train}")

print()
print(
    f"NEW TABLE-I ROWS READY: {ready_count}"
)


# ============================================================
# TABLE II
# ============================================================

t2_file = (
    ROOT
    / "results/table_exports/"
      "table2_new_datasets_aggregates.csv"
)

print()
print("=" * 120)
print("TABLE II — MISSING KETGPT #180 ROWS")
print("=" * 120)

if not t2_file.exists():
    print("ERROR: Table-II aggregate file not found:")
    print(t2_file)
    raise SystemExit(0)

t2 = pd.read_csv(t2_file)

# Current paper already has Candidate #160 + baselines.
# Only Candidate #180 for KMNIST/SVHN needs updating.

for dataset in ["kmnist", "svhn"]:

    q = t2[
        (t2["dataset"].astype(str).str.lower() == dataset)
        & (
            pd.to_numeric(
                t2["candidate"],
                errors="coerce",
            ) == 180
        )
        & (
            t2["model"].astype(str)
            == "KetGPT #180"
        )
    ].copy()

    print()
    print("-" * 120)
    print(dataset.upper())
    print("-" * 120)

    if q.empty:
        print("Candidate #180: no routing rows discovered")
        continue

    nmax = safe_int(
        pd.to_numeric(
            q["n_seeds"],
            errors="coerce",
        ).max()
    )

    # Show seed provenance if available.
    seeds = set()

    if "seeds" in q.columns:
        for x in q["seeds"].dropna():
            for s in str(x).split("|"):
                s = s.strip()
                if s:
                    seeds.add(s)

    if nmax < 5:
        print(
            f"Candidate #180 STATUS: "
            f"STILL INCOMPLETE ({nmax}/5)"
        )

        if seeds:
            print(
                "Completed seeds:",
                ", ".join(
                    sorted(
                        seeds,
                        key=lambda x: int(x)
                    )
                )
            )

        if "missing_seeds" in q.columns:
            missing = set()

            for x in q["missing_seeds"].dropna():
                for s in str(x).split("|"):
                    s = s.strip()
                    if s:
                        missing.add(s)

            if missing:
                print(
                    "Missing seeds:",
                    ", ".join(
                        sorted(
                            missing,
                            key=lambda x: int(x)
                        )
                    )
                )

        continue

    print("Candidate #180 STATUS: 5/5 COMPLETE")
    print()
    print(
        "Exp.       Thr.   Rout.(%)       Params  "
        "Acc.(%)       Prec.           Rec.            "
        "F1              AUC"
    )

    exp_order = {
        "Low-conf.": 0,
        "Adaptive": 1,
    }

    q["_order"] = (
        q["exp"]
        .map(exp_order)
        .fillna(99)
    )

    q = q.sort_values(
        ["_order", "threshold"]
    )

    for _, r in q.iterrows():

        exp = str(r["exp"])
        thr = safe_num(r["threshold"])
        params = safe_int(r["params"])

        routed = str(
            r.get("routed_entry", "--")
        )

        acc = str(
            r.get("accuracy_entry", "--")
        )

        prec = str(
            r.get("precision_macro_entry", "--")
        )

        rec = str(
            r.get("recall_macro_entry", "--")
        )

        f1 = str(
            r.get("f1_macro_entry", "--")
        )

        auc = str(
            r.get("auc_ovr_macro_entry", "--")
        )

        print(
            f"{exp:<10} "
            f"{thr:>4.2f}   "
            f"{routed:<14} "
            f"{params:<6d}  "
            f"{acc:<13} "
            f"{prec:<15} "
            f"{rec:<15} "
            f"{f1:<15} "
            f"{auc}"
        )

print()
print("=" * 120)
print("END")
print("=" * 120)
