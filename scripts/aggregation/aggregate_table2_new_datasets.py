#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(
    "/scratch/sr7849/QC_June28/"
    "ICASSP_multiseed_new_datasets"
)

RUNS = {
    "cifar10": ROOT / "runs/cifar10/run_20260921_164946",
    "kmnist": ROOT / "runs/kmnist/run_20260922_213814",
    "svhn": ROOT / "runs/svhn/run_20260922_213846",
}

EXPECTED_SEEDS = [1, 42, 48, 550, 2026]
CANDIDATES = [160, 180]

METRICS = [
    "low_pct",
    "routed_to_classical_pct",
    "routed_to_quantum_pct",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "auc_ovr_macro",
    "train_time_sec",
    "inference_time_sec_total",
    "inference_time_ms_per_sample",
]


def fmt(mean, std, digits=3):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def fmt_acc(mean, std):
    if pd.isna(mean):
        return ""

    if abs(mean) <= 1.5:
        mean *= 100.0
        if not pd.isna(std):
            std *= 100.0

    if pd.isna(std):
        return f"{mean:.1f}"

    return f"{mean:.1f} ± {std:.1f}"


def fmt_pct(mean, std):
    if pd.isna(mean):
        return ""

    if pd.isna(std):
        return f"{mean:.1f}"

    return f"{mean:.1f} ± {std:.1f}"


def classify_row(row):
    """
    Turn raw model_name/experiment_type into Table-II paper labels.
    """

    exp = str(row["experiment_type"]).strip().lower()
    model = str(row["model_name"]).strip().lower()

    # Fixed component rows are not paper routing rows.
    if exp == "fixed":
        return None, None

    # Preserve experiment type.
    if "low" in exp:
        paper_exp = "Low-conf."
    elif "adaptive" in exp:
        paper_exp = "Adaptive"
    else:
        paper_exp = str(row["experiment_type"])

    # Normalize model.
    if "matched_mlp" in model or model == "mlp":
        paper_model = "MLP"

    elif "classical" in model:
        paper_model = "Classical"

    elif "ketgpt" in model or "quantum" in model:
        paper_model = "KetGPT"

    else:
        paper_model = str(row["model_name"])

    return paper_exp, paper_model


all_seed_rows = []
all_aggregate_rows = []

for dataset, runroot in RUNS.items():

    print()
    print("=" * 110)
    print(dataset.upper())
    print("Run:", runroot)
    print("=" * 110)

    for candidate in CANDIDATES:

        directory = (
            runroot
            / "table2"
            / f"adaptive{candidate}"
        )

        files = sorted(directory.glob("seed_*.csv"))

        if not files:
            print(
                f"Candidate {candidate}: "
                "no seed CSVs found"
            )
            continue

        combined = []

        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print("ERROR:", f, e)
                continue

            if df.empty:
                continue

            combined.append(df)

        if not combined:
            continue

        df = pd.concat(
            combined,
            ignore_index=True,
        )

        # Keep only expected seeds.
        df = df[
            df["seed"].isin(EXPECTED_SEEDS)
        ].copy()

        # Add normalized Table-II labels.
        labels = df.apply(
            classify_row,
            axis=1,
            result_type="expand",
        )

        df["paper_exp"] = labels[0]
        df["paper_model_base"] = labels[1]

        # Remove fixed component-training rows.
        routing = df[
            df["paper_exp"].notna()
        ].copy()

        if routing.empty:
            print(
                f"Candidate {candidate}: "
                "no routing rows detected"
            )
            print(
                "experiment_type values:",
                sorted(df["experiment_type"]
                       .dropna()
                       .astype(str)
                       .unique())
            )
            print(
                "model_name values:",
                sorted(df["model_name"]
                       .dropna()
                       .astype(str)
                       .unique())
            )
            continue

        # Candidate is relevant only for quantum fallback rows.
        def final_model_name(r):
            base = r["paper_model_base"]

            if base == "KetGPT":
                return f"KetGPT #{candidate}"

            return base

        routing["paper_model"] = routing.apply(
            final_model_name,
            axis=1,
        )

        # Store raw seed-level rows.
        for _, r in routing.iterrows():
            out = {
                "dataset": dataset,
                "candidate": candidate,
                "seed": int(r["seed"]),
                "exp": r["paper_exp"],
                "model": r["paper_model"],
                "threshold": r["threshold"],
                "epochs_executed": r.get(
                    "epochs_executed",
                    np.nan,
                ),
                "best_epoch": r.get(
                    "best_epoch",
                    np.nan,
                ),
                "stop_reason": r.get(
                    "stop_reason",
                    "",
                ),
                "total_trainable_params": r.get(
                    "total_trainable_params",
                    np.nan,
                ),
            }

            for m in METRICS:
                out[m] = r.get(m, np.nan)

            all_seed_rows.append(out)

        # ----------------------------------------------------
        # Aggregate each paper row
        # ----------------------------------------------------

        group_cols = [
            "paper_exp",
            "paper_model",
            "threshold",
        ]

        for keys, g in routing.groupby(
            group_cols,
            dropna=False,
        ):
            exp, model, threshold = keys

            seeds = sorted(
                int(x)
                for x in g["seed"].unique()
            )

            missing = [
                s for s in EXPECTED_SEEDS
                if s not in seeds
            ]

            row = {
                "dataset": dataset,
                "candidate": candidate,
                "exp": exp,
                "model": model,
                "threshold": threshold,
                "n_seeds": len(seeds),
                "seeds": "|".join(
                    map(str, seeds)
                ),
                "missing_seeds": "|".join(
                    map(str, missing)
                ),
                "status": (
                    "COMPLETE_5_SEED"
                    if len(seeds) == 5
                    else "INCOMPLETE"
                ),
            }

            # Params should normally be constant.
            params = pd.to_numeric(
                g["total_trainable_params"],
                errors="coerce",
            ).dropna()

            row["params"] = (
                int(round(params.iloc[0]))
                if len(params)
                else np.nan
            )

            for metric in METRICS:
                vals = pd.to_numeric(
                    g[metric],
                    errors="coerce",
                ).dropna()

                row[f"{metric}_n"] = len(vals)

                row[f"{metric}_mean"] = (
                    vals.mean()
                    if len(vals)
                    else np.nan
                )

                row[f"{metric}_std"] = (
                    vals.std(ddof=1)
                    if len(vals) > 1
                    else np.nan
                )

            # Use routed-to-fallback quantity.
            #
            # Low-conf MLP / Classical:
            # routed_to_classical_pct is the relevant fallback routing.
            #
            # KetGPT:
            # routed_to_quantum_pct is the relevant fallback routing.
            if model.startswith("KetGPT"):
                route_metric = "routed_to_quantum_pct"
            else:
                route_metric = "routed_to_classical_pct"

            row["route_metric"] = route_metric

            row["routed_entry"] = fmt_pct(
                row.get(
                    f"{route_metric}_mean",
                    np.nan,
                ),
                row.get(
                    f"{route_metric}_std",
                    np.nan,
                ),
            )

            row["accuracy_entry"] = fmt_acc(
                row.get("accuracy_mean", np.nan),
                row.get("accuracy_std", np.nan),
            )

            for metric in [
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "auc_ovr_macro",
            ]:
                row[f"{metric}_entry"] = fmt(
                    row.get(
                        f"{metric}_mean",
                        np.nan,
                    ),
                    row.get(
                        f"{metric}_std",
                        np.nan,
                    ),
                    digits=3,
                )

            all_aggregate_rows.append(row)


seed_df = pd.DataFrame(all_seed_rows)
agg_df = pd.DataFrame(all_aggregate_rows)

outdir = ROOT / "results/table_exports"
outdir.mkdir(
    parents=True,
    exist_ok=True,
)

seed_file = outdir / "table2_new_datasets_seed_rows.csv"
agg_file = outdir / "table2_new_datasets_aggregates.csv"

seed_df.to_csv(
    seed_file,
    index=False,
)

agg_df.to_csv(
    agg_file,
    index=False,
)


# ============================================================
# Console paper view
# ============================================================

if agg_df.empty:
    print("\nNo routing aggregates produced.")
    raise SystemExit(0)

# Logical paper ordering.
exp_order = {
    "Low-conf.": 0,
    "Adaptive": 1,
}

model_order = {
    "MLP": 0,
    "Classical": 0,
    "KetGPT #180": 1,
    "KetGPT #160": 2,
}

agg_df["_exp_order"] = (
    agg_df["exp"]
    .map(exp_order)
    .fillna(99)
)

agg_df["_model_order"] = (
    agg_df["model"]
    .map(model_order)
    .fillna(99)
)

agg_df = agg_df.sort_values(
    [
        "dataset",
        "_exp_order",
        "threshold",
        "_model_order",
        "candidate",
    ]
)

print()
print("=" * 140)
print("TABLE II AGGREGATES")
print("=" * 140)

cols = [
    "dataset",
    "exp",
    "model",
    "threshold",
    "n_seeds",
    "status",
    "routed_entry",
    "params",
    "accuracy_entry",
    "precision_macro_entry",
    "recall_macro_entry",
    "f1_macro_entry",
    "auc_ovr_macro_entry",
]

print(
    agg_df[cols]
    .to_string(index=False)
)

print()
print("Saved:")
print(seed_file)
print(agg_file)
