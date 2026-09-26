#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(
    "/scratch/sr7849/QC_June28/"
    "ICASSP_multiseed_new_datasets"
)

# Use the actual experiment roots, not newest-by-timestamp,
# because CIFAR has an extra empty run directory.
RUNS = {
    "cifar10": ROOT / "runs/cifar10/run_20260921_164946",
    "kmnist":  ROOT / "runs/kmnist/run_20260922_213814",
    "svhn":    ROOT / "runs/svhn/run_20260922_213846",
}

EXPECTED_SEEDS = [1, 42, 48, 550, 2026]
CANDIDATES = [160, 180]


# ============================================================
# Helpers
# ============================================================

def mean_sd(series):
    vals = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if len(vals) == 0:
        return np.nan, np.nan, 0

    mean = vals.mean()

    sd = (
        vals.std(ddof=1)
        if len(vals) > 1
        else np.nan
    )

    return mean, sd, len(vals)


def fmt(mean, sd, digits=3):
    if pd.isna(mean):
        return "--"

    if pd.isna(sd):
        return f"{mean:.{digits}f}"

    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def fmt_acc(mean, sd):
    if pd.isna(mean):
        return "--"

    # Results are normally stored as [0,1].
    if abs(mean) <= 1.5:
        mean *= 100.0
        if not pd.isna(sd):
            sd *= 100.0

    return fmt(mean, sd, 1)


def fmt_pct(mean, sd):
    return fmt(mean, sd, 1)


def classify(row):
    exp_raw = str(
        row.get("experiment_type", "")
    ).strip().lower()

    model_raw = str(
        row.get("model_name", "")
    ).strip().lower()

    # Ignore fixed component rows.
    if exp_raw == "fixed":
        return None, None

    if "low" in exp_raw:
        exp = "Low-conf."
    elif "adaptive" in exp_raw:
        exp = "Adaptive"
    else:
        return None, None

    if (
        "matched_mlp" in model_raw
        or model_raw == "mlp"
    ):
        model = "MLP"

    elif "classical" in model_raw:
        model = "Classical"

    elif (
        "ketgpt" in model_raw
        or "quantum" in model_raw
    ):
        model = "KetGPT"

    else:
        return None, None

    return exp, model


# ============================================================
# Read all currently available Table-II seed results
# ============================================================

raw_rows = []

for dataset, runroot in RUNS.items():

    for candidate in CANDIDATES:

        d = (
            runroot
            / "table2"
            / f"adaptive{candidate}"
        )

        for f in sorted(d.glob("seed_*.csv")):

            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(
                    f"WARNING: failed reading {f}: {e}"
                )
                continue

            if df.empty:
                continue

            for _, r in df.iterrows():

                exp, model = classify(r)

                if exp is None:
                    continue

                thr = pd.to_numeric(
                    r.get("threshold", np.nan),
                    errors="coerce",
                )

                if pd.isna(thr):
                    continue

                seed = pd.to_numeric(
                    r.get("seed", np.nan),
                    errors="coerce",
                )

                if pd.isna(seed):
                    continue

                seed = int(seed)

                if seed not in EXPECTED_SEEDS:
                    continue

                # Keep source candidate explicit.
                if model == "KetGPT":
                    paper_model = f"KetGPT #{candidate}"
                else:
                    paper_model = model

                raw_rows.append({
                    "dataset": dataset,
                    "source_candidate": candidate,
                    "seed": seed,
                    "exp": exp,
                    "model": paper_model,
                    "threshold": float(thr),

                    "routed_to_classical_pct":
                        r.get(
                            "routed_to_classical_pct",
                            np.nan,
                        ),

                    "routed_to_quantum_pct":
                        r.get(
                            "routed_to_quantum_pct",
                            np.nan,
                        ),

                    "accuracy":
                        r.get("accuracy", np.nan),

                    "precision_macro":
                        r.get(
                            "precision_macro",
                            np.nan,
                        ),

                    "recall_macro":
                        r.get(
                            "recall_macro",
                            np.nan,
                        ),

                    "f1_macro":
                        r.get(
                            "f1_macro",
                            np.nan,
                        ),

                    "auc_ovr_macro":
                        r.get(
                            "auc_ovr_macro",
                            np.nan,
                        ),

                    "total_trainable_params":
                        r.get(
                            "total_trainable_params",
                            np.nan,
                        ),

                    "epochs_executed":
                        r.get(
                            "epochs_executed",
                            np.nan,
                        ),

                    "stop_reason":
                        r.get(
                            "stop_reason",
                            "",
                        ),

                    "source_file": str(f),
                })


raw = pd.DataFrame(raw_rows)

if raw.empty:
    raise SystemExit(
        "No Table-II routing rows found."
    )


# Avoid accidental duplicate rows for the same seed/setting.
raw = (
    raw.sort_values(
        [
            "dataset",
            "source_candidate",
            "seed",
            "exp",
            "model",
            "threshold",
        ]
    )
    .drop_duplicates(
        [
            "dataset",
            "source_candidate",
            "seed",
            "exp",
            "model",
            "threshold",
        ],
        keep="last",
    )
)


# ============================================================
# Aggregate over however many seeds currently exist
# ============================================================

agg_rows = []

GROUPS = [
    "dataset",
    "source_candidate",
    "exp",
    "model",
    "threshold",
]

for keys, g in raw.groupby(
    GROUPS,
    dropna=False,
):

    (
        dataset,
        source_candidate,
        exp,
        model,
        threshold,
    ) = keys

    seeds = sorted(
        int(x)
        for x in g["seed"].unique()
    )

    missing = [
        x
        for x in EXPECTED_SEEDS
        if x not in seeds
    ]

    row = {
        "dataset": dataset,
        "source_candidate": int(source_candidate),
        "exp": exp,
        "model": model,
        "threshold": float(threshold),
        "n_seeds": len(seeds),
        "seed_status": f"{len(seeds)}/5",
        "seeds": "|".join(map(str, seeds)),
        "missing": "|".join(map(str, missing)),
    }

    # Correct routing quantity for the fallback branch.
    route_col = (
        "routed_to_quantum_pct"
        if model.startswith("KetGPT")
        else "routed_to_classical_pct"
    )

    for metric in [
        route_col,
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "auc_ovr_macro",
    ]:
        mean, sd, n = mean_sd(g[metric])

        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = sd
        row[f"{metric}_n"] = n

    params = pd.to_numeric(
        g["total_trainable_params"],
        errors="coerce",
    ).dropna()

    row["params"] = (
        int(round(params.iloc[0]))
        if len(params)
        else np.nan
    )

    row["route_col"] = route_col

    row["routed_entry"] = fmt_pct(
        row[f"{route_col}_mean"],
        row[f"{route_col}_std"],
    )

    row["accuracy_entry"] = fmt_acc(
        row["accuracy_mean"],
        row["accuracy_std"],
    )

    row["precision_entry"] = fmt(
        row["precision_macro_mean"],
        row["precision_macro_std"],
        3,
    )

    row["recall_entry"] = fmt(
        row["recall_macro_mean"],
        row["recall_macro_std"],
        3,
    )

    row["f1_entry"] = fmt(
        row["f1_macro_mean"],
        row["f1_macro_std"],
        3,
    )

    row["auc_entry"] = fmt(
        row["auc_ovr_macro_mean"],
        row["auc_ovr_macro_std"],
        3,
    )

    # Accuracy in percentage points for easier ranking/output.
    acc = row["accuracy_mean"]

    row["accuracy_pct_mean"] = (
        acc * 100.0
        if not pd.isna(acc) and abs(acc) <= 1.5
        else acc
    )

    agg_rows.append(row)


agg = pd.DataFrame(agg_rows)


# ============================================================
# Sort logically
# ============================================================

dataset_order = {
    "cifar10": 0,
    "kmnist": 1,
    "svhn": 2,
}

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

agg["_dataset_order"] = (
    agg["dataset"]
    .map(dataset_order)
)

agg["_exp_order"] = (
    agg["exp"]
    .map(exp_order)
)

agg["_model_order"] = (
    agg["model"]
    .map(model_order)
    .fillna(99)
)

agg = agg.sort_values(
    [
        "_dataset_order",
        "_exp_order",
        "threshold",
        "_model_order",
        "source_candidate",
    ]
)


# ============================================================
# Save files
# ============================================================

outdir = (
    ROOT
    / "results"
    / "table_exports"
)

outdir.mkdir(
    parents=True,
    exist_ok=True,
)

raw_file = (
    outdir
    / "table2_current_seed_rows.csv"
)

agg_file = (
    outdir
    / "table2_current_partial_aggregates.csv"
)

raw.to_csv(
    raw_file,
    index=False,
)

agg.drop(
    columns=[
        "_dataset_order",
        "_exp_order",
        "_model_order",
    ]
).to_csv(
    agg_file,
    index=False,
)


# ============================================================
# 1. Print ALL aggregate results
# ============================================================

print()
print("=" * 155)
print(
    "TABLE II — CURRENT AGGREGATES "
    "(MEAN ± SAMPLE SD OVER AVAILABLE SEEDS)"
)
print("=" * 155)

show_cols = [
    "dataset",
    "source_candidate",
    "exp",
    "model",
    "threshold",
    "seed_status",
    "routed_entry",
    "params",
    "accuracy_entry",
    "precision_entry",
    "recall_entry",
    "f1_entry",
    "auc_entry",
]

print(
    agg[show_cols]
    .rename(
        columns={
            "dataset": "Dataset",
            "source_candidate": "SrcCand",
            "exp": "Exp",
            "model": "Model",
            "threshold": "Thr",
            "seed_status": "N",
            "routed_entry": "Routed%",
            "params": "Params",
            "accuracy_entry": "Acc%",
            "precision_entry": "Prec",
            "recall_entry": "Rec",
            "f1_entry": "F1",
            "auc_entry": "AUC",
        }
    )
    .to_string(index=False)
)


# ============================================================
# 2. Best threshold for each model
# ============================================================

print()
print()
print("=" * 130)
print(
    "BEST THRESHOLD FOR EACH MODEL "
    "(RANKED BY CURRENT MEAN ACCURACY)"
)
print("=" * 130)

best_per_model = (
    agg.sort_values(
        "accuracy_pct_mean",
        ascending=False,
    )
    .groupby(
        [
            "dataset",
            "source_candidate",
            "exp",
            "model",
        ],
        as_index=False,
    )
    .first()
)

best_per_model["_dataset_order"] = (
    best_per_model["dataset"]
    .map(dataset_order)
)

best_per_model["_exp_order"] = (
    best_per_model["exp"]
    .map(exp_order)
)

best_per_model = best_per_model.sort_values(
    [
        "_dataset_order",
        "_exp_order",
        "source_candidate",
        "model",
    ]
)

print(
    best_per_model[
        [
            "dataset",
            "source_candidate",
            "exp",
            "model",
            "threshold",
            "seed_status",
            "routed_entry",
            "accuracy_entry",
            "f1_entry",
            "auc_entry",
        ]
    ]
    .rename(
        columns={
            "dataset": "Dataset",
            "source_candidate": "SrcCand",
            "exp": "Exp",
            "model": "Model",
            "threshold": "BestThr",
            "seed_status": "N",
            "routed_entry": "Routed%",
            "accuracy_entry": "Acc%",
            "f1_entry": "F1",
            "auc_entry": "AUC",
        }
    )
    .to_string(index=False)
)


# ============================================================
# 3. Strongest QUANTUM row per dataset + experiment type
# ============================================================

print()
print()
print("=" * 130)
print(
    "STRONGEST QUANTUM CONFIGURATION PER DATASET / EXPERIMENT"
)
print(
    "Criterion: highest CURRENT mean accuracy; "
    "N/5 is shown so partial runs are obvious."
)
print("=" * 130)

quantum = agg[
    agg["model"].str.startswith(
        "KetGPT",
        na=False,
    )
].copy()

for dataset in [
    "cifar10",
    "kmnist",
    "svhn",
]:
    for exp in [
        "Low-conf.",
        "Adaptive",
    ]:

        q = quantum[
            (quantum["dataset"] == dataset)
            & (quantum["exp"] == exp)
        ].copy()

        if q.empty:
            continue

        q = q.sort_values(
            [
                "accuracy_pct_mean",
                "f1_macro_mean",
                "auc_ovr_macro_mean",
            ],
            ascending=False,
        )

        winner = q.iloc[0]

        print()
        print(
            f"{dataset.upper():<8} | "
            f"{exp:<10} | "
            f"{winner['model']:<12} | "
            f"thr={winner['threshold']:.2f} | "
            f"N={winner['seed_status']} | "
            f"routed={winner['routed_entry']}% | "
            f"Acc={winner['accuracy_entry']}% | "
            f"F1={winner['f1_entry']} | "
            f"AUC={winner['auc_entry']}"
        )


# ============================================================
# 4. Full quantum ranking for each dataset
# ============================================================

print()
print()
print("=" * 150)
print(
    "QUANTUM ROUTING RANKING — ALL THRESHOLDS "
    "(DESCENDING CURRENT MEAN ACCURACY)"
)
print("=" * 150)

for dataset in [
    "cifar10",
    "kmnist",
    "svhn",
]:

    print()
    print("-" * 150)
    print(dataset.upper())
    print("-" * 150)

    q = quantum[
        quantum["dataset"] == dataset
    ].sort_values(
        [
            "accuracy_pct_mean",
            "f1_macro_mean",
            "auc_ovr_macro_mean",
        ],
        ascending=False,
    )

    print(
        q[
            [
                "exp",
                "model",
                "threshold",
                "seed_status",
                "routed_entry",
                "accuracy_entry",
                "f1_entry",
                "auc_entry",
            ]
        ]
        .rename(
            columns={
                "exp": "Exp",
                "model": "Model",
                "threshold": "Thr",
                "seed_status": "N",
                "routed_entry": "Routed%",
                "accuracy_entry": "Acc%",
                "f1_entry": "F1",
                "auc_entry": "AUC",
            }
        )
        .to_string(index=False)
    )


print()
print("=" * 150)
print("NOTES")
print("=" * 150)
print(
    "1. N gives the number of currently completed seeds out of 5."
)
print(
    "2. Means/SDs use ONLY currently available completed seed rows."
)
print(
    "3. A 1/5 row has no sample SD; it is not directly comparable "
    "in reliability to a 5/5 row."
)
print(
    "4. 'Strongest' above means highest current mean accuracy. "
    "F1 and AUC are shown alongside it."
)
print(
    "5. SrcCand identifies which routing family generated the "
    "baseline/routing threshold, preventing Candidate-160 and "
    "Candidate-180 baselines from being mixed."
)
print()
print("Saved:")
print(raw_file)
print(agg_file)
