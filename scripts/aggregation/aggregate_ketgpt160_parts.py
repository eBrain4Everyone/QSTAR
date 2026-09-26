#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import re


ROOT = Path(
    "/scratch/sr7849/QC_June28/"
    "ICASSP_multiseed_new_datasets"
)

DATASETS = ["kmnist", "svhn"]
EXPECTED = [1, 42, 48, 550, 2026]

ALIASES = {
    "accuracy": [
        "accuracy", "acc",
        "test_accuracy", "test_acc",
    ],
    "precision_macro": [
        "precision_macro",
        "macro_precision",
        "precision",
    ],
    "recall_macro": [
        "recall_macro",
        "macro_recall",
        "recall",
    ],
    "f1_macro": [
        "f1_macro",
        "macro_f1",
        "f1",
        "f1_score",
    ],
    "auc_ovr_macro": [
        "auc_ovr_macro",
        "roc_auc_ovr_macro",
        "roc_auc_macro",
        "roc_auc",
        "auc",
    ],
    "train_time_sec": [
        "train_time_sec",
        "training_time_sec",
        "runtime_sec",
    ],
}


def norm(s):
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def getval(row, names):
    mapping = {
        norm(c): c
        for c in row.index
    }

    for name in names:
        n = norm(name)

        if n in mapping:
            v = row[mapping[n]]

            if not pd.isna(v):
                try:
                    return float(v)
                except Exception:
                    return v

    return np.nan


def seed_from_file(path):
    m = re.search(
        r"seed[_-]?(\d+)",
        path.name,
        re.I,
    )

    return int(m.group(1)) if m else None


all_rows = []

for dataset in DATASETS:

    run = sorted(
        (ROOT / "runs" / dataset).glob("run_*")
    )[-1]

    files = sorted(
        run.glob(
            "table1/ketgpt_parts/"
            "candidate_160/*.csv"
        )
    )

    print()
    print("=" * 90)
    print(dataset.upper())
    print("=" * 90)
    print("Run:", run)

    dataset_rows = []

    for f in files:

        seed = seed_from_file(f)

        if seed not in EXPECTED:
            continue

        df = pd.read_csv(f)

        if df.empty:
            continue

        # Candidate-part file should contain the final row.
        row = df.iloc[-1]

        out = {
            "dataset": dataset,
            "method": "KetGPT #160",
            "seed": seed,
            "source_file": str(f),
        }

        for metric, aliases in ALIASES.items():
            out[metric] = getval(
                row,
                aliases,
            )

        dataset_rows.append(out)
        all_rows.append(out)

    d = pd.DataFrame(dataset_rows)

    if d.empty:
        print("No usable result rows found.")
        continue

    print()
    print("PER-SEED RESULTS")
    print(
        d[
            [
                "seed",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "auc_ovr_macro",
            ]
        ].to_string(index=False)
    )

    print()

    seeds = sorted(d["seed"].tolist())

    print(
        "Completed seeds:",
        seeds,
        f"({len(seeds)}/5)",
    )

    print()
    print("AGGREGATE")

    for metric in [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "auc_ovr_macro",
    ]:

        vals = pd.to_numeric(
            d[metric],
            errors="coerce",
        ).dropna()

        if len(vals) == 0:
            print(
                f"{metric:<18}: unavailable"
            )
            continue

        mean = vals.mean()
        sd = (
            vals.std(ddof=1)
            if len(vals) > 1
            else np.nan
        )

        # Print Accuracy as %
        if metric == "accuracy":
            if mean <= 1.5:
                mean *= 100
                sd *= 100

            print(
                f"{metric:<18}: "
                f"{mean:.2f} ± {sd:.2f} %"
            )

        else:
            print(
                f"{metric:<18}: "
                f"{mean:.4f} ± {sd:.4f}"
            )


outdir = ROOT / "results/table_exports"
outdir.mkdir(
    parents=True,
    exist_ok=True,
)

raw = pd.DataFrame(all_rows)

outfile = (
    outdir /
    "ketgpt160_kmnist_svhn_seed_results.csv"
)

raw.to_csv(
    outfile,
    index=False,
)

print()
print("Saved:", outfile)
