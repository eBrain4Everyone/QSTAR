#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(
    "/scratch/sr7849/QC_June28/"
    "ICASSP_multiseed_new_datasets"
)

# ============================================================
# Rows that are CURRENTLY blank in Table I
# ============================================================

T1_MISSING = [
    # CIFAR-10
    ("cifar10", "standard_qtl/q4_d4", "QTL q4 d4"),
    ("cifar10", "standard_qtl/q6_d4", "QTL q6 d4"),
    ("cifar10", "standard_qtl/q8_d2", "QTL q8 d2"),

    # KMNIST
    ("kmnist", "standard_qtl/q4_d4", "QTL q4 d4"),
    ("kmnist", "standard_qtl/q6_d2", "QTL q6 d2"),
    ("kmnist", "standard_qtl/q6_d4", "QTL q6 d4"),
    ("kmnist", "standard_qtl/q8_d2", "QTL q8 d2"),
    ("kmnist", "ketgpt/candidate_180", "KetGPT #180"),

    # SVHN
    ("svhn", "standard_qtl/q4_d4", "QTL q4 d4"),
    ("svhn", "standard_qtl/q6_d2", "QTL q6 d2"),
    ("svhn", "standard_qtl/q6_d4", "QTL q6 d4"),
    ("svhn", "standard_qtl/q8_d2", "QTL q8 d2"),
    ("svhn", "ketgpt/candidate_180", "KetGPT #180"),
]


def num(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def integer(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def pm(mean, std, digits):
    mean = num(mean)
    std = num(std)

    if pd.isna(mean):
        return "--"

    if pd.isna(std):
        return f"{mean:.{digits}f}"

    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def metric(row, name, digits=3, percent=False):
    mean = num(row.get(f"{name}_mean", np.nan))
    std = num(row.get(f"{name}_std", np.nan))

    if pd.isna(mean):
        return "--"

    if percent and abs(mean) <= 1.5:
        mean *= 100.0
        if not pd.isna(std):
            std *= 100.0

    return pm(mean, std, digits)


# ============================================================
# Locate latest Table-I export
# ============================================================

exports = sorted(
    (ROOT / "results/table_exports").glob("export_*"),
    key=lambda p: p.stat().st_mtime,
)

if not exports:
    raise SystemExit("No export_* directory found.")

EXPORT = exports[-1]

t1_path = EXPORT / "table1_all_aggregates.csv"

if not t1_path.exists():
    raise SystemExit(f"Missing: {t1_path}")

t1 = pd.read_csv(t1_path)


# ============================================================
# Exact Slurm timing for KMNIST/SVHN
# ============================================================

rt_path = (
    ROOT /
    "results/runtime_audit/"
    "table1_exact_slurm_runtime_kmnist_svhn_aggregate.csv"
)

runtime = (
    pd.read_csv(rt_path)
    if rt_path.exists()
    else pd.DataFrame()
)


def get_time(dataset, method, row):
    """
    Current average runtime over completed seeds.

    KMNIST/SVHN:
        prefer exact Slurm completed-job timings.

    CIFAR:
        use train-time aggregate recorded by result exporter.
    """

    if not runtime.empty:
        q = runtime[
            (runtime["dataset"].astype(str).str.lower() == dataset)
            & (runtime["method"].astype(str) == method)
        ]

        if len(q):
            r = q.iloc[-1]

            ntime = integer(
                r.get("n_completed_timed_seeds", 0)
            )

            mean = num(
                r.get("mean_wallclock_h", np.nan)
            )
            std = num(
                r.get("std_wallclock_h", np.nan)
            )

            if not pd.isna(mean):
                return (
                    pm(mean, std, 2)
                    + f" h [{ntime} timed]"
                )

    mean = num(
        row.get("train_time_hours_mean", np.nan)
    )
    std = num(
        row.get("train_time_hours_std", np.nan)
    )

    if not pd.isna(mean):
        return pm(mean, std, 2) + " h"

    mean = num(
        row.get("train_time_sec_mean", np.nan)
    )
    std = num(
        row.get("train_time_sec_std", np.nan)
    )

    if not pd.isna(mean):
        return pm(
            mean / 3600.0,
            std / 3600.0 if not pd.isna(std) else np.nan,
            2,
        ) + " h"

    return "--"


# ============================================================
# TABLE I — partial averages
# ============================================================

print()
print("=" * 125)
print("TABLE I — CURRENT PARTIAL AVERAGES FOR ROWS BLANK IN PAPER")
print("=" * 125)

print(
    f"{'Dataset':<10} "
    f"{'Configuration':<18} "
    f"{'N':<6} "
    f"{'Acc (%)':<17} "
    f"{'F1':<17} "
    f"{'AUC':<17} "
    f"Train"
)

print("-" * 125)

for dataset, method_name, label in T1_MISSING:

    q = t1[
        (t1["dataset"].astype(str).str.lower() == dataset)
        & (t1["method"].astype(str) == method_name)
    ]

    if q.empty:
        print(
            f"{dataset:<10} "
            f"{label:<18} "
            f"{'0/5':<6} "
            f"{'--':<17} "
            f"{'--':<17} "
            f"{'--':<17} "
            f"--"
        )
        continue

    r = q.iloc[-1]

    n = integer(r.get("n_final_seeds", 0))

    acc = metric(
        r,
        "accuracy",
        digits=1,
        percent=True,
    )

    f1 = metric(
        r,
        "f1_macro",
        digits=3,
    )

    auc = metric(
        r,
        "auc_ovr_macro",
        digits=3,
    )

    tr = get_time(
        dataset,
        method_name,
        r,
    )

    print(
        f"{dataset:<10} "
        f"{label:<18} "
        f"{str(n) + '/5':<6} "
        f"{acc:<17} "
        f"{f1:<17} "
        f"{auc:<17} "
        f"{tr}"
    )


# ============================================================
# TABLE II — currently missing rows only
# ============================================================

t2_path = (
    ROOT /
    "results/table_exports/"
    "table2_new_datasets_aggregates.csv"
)

print()
print()
print("=" * 140)
print("TABLE II — CURRENT PARTIAL AVERAGES FOR BLANK CANDIDATE #180 ROWS")
print("=" * 140)

if not t2_path.exists():
    print("Missing:", t2_path)
    raise SystemExit(0)

t2 = pd.read_csv(t2_path)

print(
    f"{'Dataset':<9} "
    f"{'Exp.':<10} "
    f"{'Thr.':<6} "
    f"{'N':<6} "
    f"{'Routed (%)':<17} "
    f"{'Params':<8} "
    f"{'Acc (%)':<17} "
    f"{'Prec.':<17} "
    f"{'Rec.':<17} "
    f"{'F1':<17} "
    f"AUC"
)

print("-" * 160)

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

    if q.empty:
        print(dataset, "Candidate #180 not found")
        continue

    order = {
        "Low-conf.": 0,
        "Adaptive": 1,
    }

    q["_ord"] = q["exp"].map(order).fillna(99)

    q = q.sort_values(
        ["_ord", "threshold"]
    )

    for _, r in q.iterrows():

        n = integer(r.get("n_seeds", 0))
        exp = str(r["exp"])
        thr = num(r["threshold"])

        routed = pm(
            r.get(
                "routed_to_quantum_pct_mean",
                np.nan,
            ),
            r.get(
                "routed_to_quantum_pct_std",
                np.nan,
            ),
            1,
        )

        # Fallback in case the aggregate uses only routed_entry.
        if routed == "--":
            x = r.get("routed_entry", "--")
            routed = (
                str(x)
                if pd.notna(x)
                else "--"
            )

        params = integer(
            r.get("params", 0)
        )

        acc = pm(
            r.get("accuracy_mean", np.nan),
            r.get("accuracy_std", np.nan),
            1,
        )

        # Accuracy is usually stored 0--1.
        acc_mean = num(
            r.get("accuracy_mean", np.nan)
        )
        acc_std = num(
            r.get("accuracy_std", np.nan)
        )

        if not pd.isna(acc_mean) and abs(acc_mean) <= 1.5:
            acc = pm(
                acc_mean * 100.0,
                (
                    acc_std * 100.0
                    if not pd.isna(acc_std)
                    else np.nan
                ),
                1,
            )

        prec = pm(
            r.get("precision_macro_mean", np.nan),
            r.get("precision_macro_std", np.nan),
            3,
        )

        rec = pm(
            r.get("recall_macro_mean", np.nan),
            r.get("recall_macro_std", np.nan),
            3,
        )

        f1 = pm(
            r.get("f1_macro_mean", np.nan),
            r.get("f1_macro_std", np.nan),
            3,
        )

        auc = pm(
            r.get("auc_ovr_macro_mean", np.nan),
            r.get("auc_ovr_macro_std", np.nan),
            3,
        )

        print(
            f"{dataset:<9} "
            f"{exp:<10} "
            f"{thr:<6.2f} "
            f"{str(n) + '/5':<6} "
            f"{routed:<17} "
            f"{params:<8} "
            f"{acc:<17} "
            f"{prec:<17} "
            f"{rec:<17} "
            f"{f1:<17} "
            f"{auc}"
        )

print()
print("=" * 140)
print("These are PARTIAL averages whenever N < 5.")
print("Do not use them as final paper values until N = 5.")
print("=" * 140)
