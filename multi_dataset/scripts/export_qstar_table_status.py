#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("/scratch/sr7849/QC_June28/ICASSP_multiseed_new_datasets")
RUNS_ROOT = ROOT / "runs"

DEFAULT_DATASETS = ["cifar10", "kmnist", "svhn"]
EXPECTED_SEEDS = [1, 42, 48, 550, 2026]
HISTORY_SEEDS = [1, 42, 48]
ASSISTED_SEEDS = [550, 2026]

TABLE1_METHODS = [
    "classical/linear",
    "classical/matched_mlp",
    "ketgpt/candidate_160",
    "ketgpt/candidate_180",
    "standard_qtl/q4_d1",
    "standard_qtl/q4_d2",
    "standard_qtl/q4_d4",
    "standard_qtl/q6_d1",
    "standard_qtl/q6_d2",
    "standard_qtl/q6_d4",
    "standard_qtl/q8_d1",
    "standard_qtl/q8_d2",
]

TABLE2_COMPONENTS = [
    "confidence_linear",
    "matched_mlp",
    "ketgpt_quantum",
]

PERFORMANCE_METRICS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "auc_ovr_macro",
]

OTHER_METRICS = [
    "train_time_sec",
    "inference_time_sec",
    "inference_time_ms",
    "memory_mb",
    "peak_memory_mb",
    "trainable_params",
    "total_params",
    "head_params",
    "gate_count",
    "qwc_groups",
    "shots",
]

METRIC_ALIASES = {
    "accuracy": [
        "accuracy", "acc", "test_accuracy", "test_acc"
    ],
    "precision_macro": [
        "precision_macro", "macro_precision", "precision"
    ],
    "recall_macro": [
        "recall_macro", "macro_recall", "recall"
    ],
    "f1_macro": [
        "f1_macro", "macro_f1", "f1", "f1_score"
    ],
    "auc_ovr_macro": [
        "auc_ovr_macro", "roc_auc_ovr_macro",
        "roc_auc_macro", "roc_auc", "auc"
    ],
    "train_time_sec": [
        "train_time_sec", "training_time_sec", "runtime_sec",
        "train_seconds", "elapsed_sec"
    ],
    "inference_time_sec": [
        "inference_time_sec", "infer_time_sec"
    ],
    "inference_time_ms": [
        "inference_time_ms", "infer_time_ms"
    ],
    "memory_mb": [
        "memory_mb", "gpu_memory_mb"
    ],
    "peak_memory_mb": [
        "peak_memory_mb", "max_memory_mb"
    ],
    "trainable_params": [
        "trainable_params", "n_trainable_params"
    ],
    "total_params": [
        "total_params", "params", "n_params"
    ],
    "head_params": [
        "head_params"
    ],
    "gate_count": [
        "gate_count", "n_gates", "gates"
    ],
    "qwc_groups": [
        "qwc_groups", "measurement_groups", "n_qwc_groups"
    ],
    "shots": [
        "shots", "n_shots"
    ],
}

SEED_ALIASES = ["seed", "random_seed", "run_seed"]
EPOCH_ALIASES = [
    "epochs_executed",
    "completed_epoch",
    "final_epoch",
    "current_epoch",
    "epoch",
]
THRESHOLD_ALIASES = [
    "threshold", "confidence_threshold", "tau", "routing_threshold"
]


# ============================================================
# BASIC HELPERS
# ============================================================

def norm_col(x):
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def numeric(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def latest_run(dataset):
    base = RUNS_ROOT / dataset
    if not base.exists():
        return None

    candidates = [p for p in base.glob("run_*") if p.is_dir()]
    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[0]


def first_existing(row, aliases):
    normalized = {norm_col(c): c for c in row.index}

    for a in aliases:
        a = norm_col(a)
        if a in normalized:
            val = row[normalized[a]]
            if not pd.isna(val):
                return val
    return None


def infer_seed_from_path(path):
    s = str(path)

    for pat in [
        r"seed[_-](\d+)",
        r"/seed(\d+)(?:/|_|\.|$)",
    ]:
        m = re.search(pat, s, re.I)
        if m:
            return int(m.group(1))

    return None


def infer_seed(row, path):
    x = first_existing(row, SEED_ALIASES)
    if x is not None:
        try:
            return int(float(x))
        except Exception:
            pass

    return infer_seed_from_path(path)


def max_epoch_from_obj(obj):
    vals = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = norm_col(k)

            if lk in {
                "epoch",
                "current_epoch",
                "epochs_executed",
                "completed_epoch",
                "last_epoch",
                "final_epoch",
            }:
                try:
                    vals.append(int(v))
                except Exception:
                    pass

            vals.extend(max_epoch_from_obj(v))

        for k in [
            "history",
            "epochs",
            "trajectory",
            "epoch_history",
            "training_history",
        ]:
            v = obj.get(k)
            if isinstance(v, list):
                vals.append(len(v))

    elif isinstance(obj, list):
        vals.append(len(obj))
        for x in obj:
            vals.extend(max_epoch_from_obj(x))

    return vals


def allow_stop_from_obj(obj):
    """
    Recursively search a trajectory object for an allow_stop=True flag.
    Returns a boolean.
    """

    def as_bool(v):
        if isinstance(v, bool):
            return v

        if isinstance(v, (int, float)):
            return bool(v)

        if isinstance(v, str):
            return v.strip().lower() in {
                "true", "1", "yes", "y", "on"
            }

        return False

    if isinstance(obj, dict):
        for k, v in obj.items():

            if norm_col(k) == "allow_stop" and as_bool(v):
                return True

            if allow_stop_from_obj(v):
                return True

    elif isinstance(obj, list):
        for x in obj:
            if allow_stop_from_obj(x):
                return True

    return False


# ============================================================
# TRAJECTORIES
# ============================================================

def trajectory_identity(path, runroot):
    rel = str(path.relative_to(runroot)).lower()

    seed = infer_seed_from_path(path)

    # Table I classical
    m = re.search(
        r"table1/classical/(linear|matched_mlp)",
        rel
    )
    if m:
        return {
            "table": "I",
            "method": f"classical/{m.group(1)}",
            "component": "",
            "seed": seed,
        }

    # Table I standard QTL
    m = re.search(
        r"table1/standard_qtl/q(\d+)[_-]d(\d+)",
        rel
    )
    if m:
        return {
            "table": "I",
            "method": f"standard_qtl/q{m.group(1)}_d{m.group(2)}",
            "component": "",
            "seed": seed,
        }

    # Table I KetGPT
    m = re.search(
        r"table1/ketgpt/(?:candidate_)?(160|180)",
        rel
    )
    if not m:
        m = re.search(
            r"table1/ketgpt/.*candidate_(160|180)",
            rel
        )

    if m:
        return {
            "table": "I",
            "method": f"ketgpt/candidate_{m.group(1)}",
            "component": "",
            "seed": seed,
        }

    # Table II components
    m = re.search(
        r"table2/adaptive(160|180)/(confidence_linear|matched_mlp|ketgpt_quantum)",
        rel
    )
    if m:
        return {
            "table": "II_COMPONENT",
            "method": f"candidate_{m.group(1)}",
            "component": m.group(2),
            "seed": seed,
        }

    return None


def collect_trajectories(dataset, runroot):
    rows = []

    for path in runroot.rglob("trajectory.json"):
        ident = trajectory_identity(path, runroot)

        if ident is None:
            continue

        try:
            obj = json.loads(path.read_text())
        except Exception:
            obj = {}

        vals = max_epoch_from_obj(obj)
        vals = [
            int(x) for x in vals
            if isinstance(x, (int, float))
            and 0 <= int(x) <= 1000
        ]

        epoch = max(vals) if vals else np.nan
        allow_stop = allow_stop_from_obj(obj)

        rows.append({
            "dataset": dataset,
            **ident,
            "epoch": epoch,
            "flashjt_allow_stop_seen": allow_stop,
            "trajectory_file": str(path),
        })

    return pd.DataFrame(rows)


# ============================================================
# RESULT CSV DISCOVERY
# ============================================================

def candidate_id_from_path_or_row(path_s, row):
    m = re.search(r"(?:candidate_|adaptive)(160|180)", path_s)
    if m:
        return m.group(1)

    for c in [
        "ketgpt_id",
        "candidate_id",
        "candidate",
    ]:
        x = first_existing(row, [c])
        if x is not None:
            sx = str(x)
            m = re.search(r"(160|180)", sx)
            if m:
                return m.group(1)

    return None


def q_and_depth(path_s, row):
    q = first_existing(
        row,
        ["n_qubits", "num_qubits", "qubits", "q"]
    )
    d = first_existing(
        row,
        ["depth", "layers", "n_layers", "circuit_depth"]
    )

    try:
        q = int(float(q))
    except Exception:
        q = None

    try:
        d = int(float(d))
    except Exception:
        d = None

    if q is None or d is None:
        m = re.search(r"q(\d+)[_-]d(\d+)", path_s)
        if m:
            q = int(m.group(1))
            d = int(m.group(2))

    return q, d


def extract_metrics(row):
    out = {}

    for metric, aliases in METRIC_ALIASES.items():
        x = first_existing(row, aliases)
        out[metric] = numeric(x)

    return out


def table1_identity(path_s, row):
    if "/classical/" in path_s:
        model = first_existing(
            row,
            ["model_name", "model", "head", "method"]
        )

        if model is not None:
            model = norm_col(model)

            if "linear" in model and "confidence" not in model:
                return "classical/linear"

            if "mlp" in model:
                return "classical/matched_mlp"

        # fallback from filename/path
        if "matched_mlp" in path_s or "/mlp" in path_s:
            return "classical/matched_mlp"

        if "linear" in path_s:
            return "classical/linear"

        return None

    if "/standard_qtl/" in path_s:
        q, d = q_and_depth(path_s, row)
        if q is not None and d is not None:
            return f"standard_qtl/q{q}_d{d}"
        return None

    if "/ketgpt/" in path_s:
        cid = candidate_id_from_path_or_row(path_s, row)
        if cid:
            return f"ketgpt/candidate_{cid}"

    return None


def table2_identity(path_s, row):
    cid = candidate_id_from_path_or_row(path_s, row)
    if cid is None:
        return None

    threshold = first_existing(row, THRESHOLD_ALIASES)

    try:
        threshold = float(threshold)
    except Exception:
        threshold = np.nan

    # Try to identify a paper-level route/method name.
    label = first_existing(
        row,
        [
            "policy",
            "routing_policy",
            "route",
            "method",
            "model_name",
            "fallback",
            "head",
            "component",
        ]
    )

    label = norm_col(label) if label is not None else ""

    # Component output rather than Table-II routing output
    component = None

    for comp in TABLE2_COMPONENTS:
        if comp in path_s or comp == label:
            component = comp
            break

    if not math.isnan(threshold):
        kind = "ROUTING_RESULT"

        if not label:
            label = f"adaptive_candidate_{cid}"

        key = (
            f"candidate_{cid}"
            f"|threshold_{threshold:.2f}"
            f"|{label}"
        )

        return {
            "kind": kind,
            "method": key,
            "candidate": int(cid),
            "threshold": threshold,
            "component": "",
        }

    if component:
        return {
            "kind": "COMPONENT_RESULT",
            "method": f"candidate_{cid}",
            "candidate": int(cid),
            "threshold": np.nan,
            "component": component,
        }

    # It may still be a final candidate result without an explicit threshold.
    return {
        "kind": "CANDIDATE_RESULT",
        "method": f"candidate_{cid}",
        "candidate": int(cid),
        "threshold": np.nan,
        "component": label,
    }


def scan_result_csvs(dataset, runroot):
    discovered_schema = []
    candidates = []

    skip_terms = [
        "aggregate",
        "summary",
        "table_exports",
        "preview_selected",
    ]

    for path in runroot.rglob("*.csv"):
        rel = str(path.relative_to(runroot))
        ps = "/" + rel.lower()

        if "/table1/" not in ps and "/table2/" not in ps:
            continue

        if any(x in path.name.lower() for x in skip_terms):
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        discovered_schema.append({
            "dataset": dataset,
            "file": rel,
            "rows": len(df),
            "columns": "|".join(map(str, df.columns)),
        })

        if df.empty:
            continue

        for idx, row in df.iterrows():
            seed = infer_seed(row, path)

            if seed not in EXPECTED_SEEDS:
                continue

            metrics = extract_metrics(row)

            result_epoch = first_existing(row, EPOCH_ALIASES)
            try:
                result_epoch = int(float(result_epoch))
            except Exception:
                result_epoch = np.nan

            base = {
                "dataset": dataset,
                "seed": seed,
                "result_epoch": result_epoch,
                "source_file": rel,
                "source_row": idx,
                "mtime": path.stat().st_mtime,
                **metrics,
            }

            if "/table1/" in ps:
                method = table1_identity(ps, row)
                if method is None:
                    continue

                candidates.append({
                    **base,
                    "table": "I",
                    "kind": "TABLE1_RESULT",
                    "method": method,
                    "candidate": np.nan,
                    "threshold": np.nan,
                    "component": "",
                })

            elif "/table2/" in ps:
                ident = table2_identity(ps, row)
                if ident is None:
                    continue

                candidates.append({
                    **base,
                    "table": "II",
                    **ident,
                })

    cdf = pd.DataFrame(candidates)
    sdf = pd.DataFrame(discovered_schema)

    if cdf.empty:
        return cdf, sdf

    # Prefer rows with more populated metrics, then newest file.
    metric_cols = PERFORMANCE_METRICS + OTHER_METRICS

    cdf["_metric_count"] = (
        cdf[metric_cols]
        .notna()
        .sum(axis=1)
    )

    dedup_cols = [
        "dataset",
        "table",
        "kind",
        "method",
        "component",
        "seed",
    ]

    # Include threshold for Table-II routing results.
    cdf["_threshold_key"] = cdf["threshold"].fillna(-999.0)
    dedup_cols.append("_threshold_key")

    cdf = (
        cdf.sort_values(
            ["_metric_count", "mtime"],
            ascending=[False, False]
        )
        .drop_duplicates(
            subset=dedup_cols,
            keep="first"
        )
        .drop(columns=["_metric_count", "_threshold_key"])
        .reset_index(drop=True)
    )

    return cdf, sdf


# ============================================================
# TRAJECTORY LOOKUPS / PROTOCOL CLASSIFICATION
# ============================================================

def trajectory_map(tdf):
    out = {}

    if tdf.empty:
        return out

    for _, r in tdf.iterrows():
        key = (
            r["table"],
            r["method"],
            r["component"],
            int(r["seed"]) if not pd.isna(r["seed"]) else None,
        )

        out[key] = {
            "epoch": r["epoch"],
            "allow_stop": bool(r["flashjt_allow_stop_seen"]),
            "trajectory_file": r["trajectory_file"],
        }

    return out


def table1_epoch(tmap, method, seed):
    x = tmap.get(("I", method, "", seed))
    if not x:
        return np.nan, False

    return x["epoch"], x["allow_stop"]


def candidate_component_epochs(tmap, candidate, seed):
    vals = []
    allows = []

    for component in TABLE2_COMPONENTS:
        x = tmap.get(
            ("II_COMPONENT", f"candidate_{candidate}", component, seed)
        )

        if x:
            if not pd.isna(x["epoch"]):
                vals.append(int(x["epoch"]))
            allows.append(bool(x["allow_stop"]))

    return vals, any(allows)


def classify_protocol(final_epochs, allow_flags):
    """
    final_epochs: dict seed -> epoch for final result seeds.
    Called only when all 5 final seed rows exist.
    """

    hist = [final_epochs.get(s, np.nan) for s in HISTORY_SEEDS]
    assisted = [final_epochs.get(s, np.nan) for s in ASSISTED_SEEDS]

    if any(pd.isna(x) for x in hist + assisted):
        return (
            "COMPLETE_EPOCH_UNKNOWN",
            "All five final results exist; one or more final epochs could not be resolved."
        )

    if not all(x >= 10 for x in hist):
        return (
            "REVIEW_HISTORY_EPOCH",
            "All five result rows exist, but a historical seed ended before epoch 10."
        )

    if all(x >= 10 for x in assisted):
        return (
            "FULL_TRAINING",
            "All five seeds have final results and both assisted seeds reached epoch 10."
        )

    early = [x for x in assisted if x < 10]
    full = [x for x in assisted if x >= 10]

    if early and not full:
        if all(x == 6 for x in early):
            return (
                "FLASHJT_ASSISTED",
                "All five seeds are final; seeds 550 and 2026 terminated at the FlashJT validation point."
            )

        return (
            "EARLY_FINAL_REVIEW",
            "All five seeds are final, but an assisted seed ended between epochs 7 and 9; review stopping provenance."
        )

    if early and full:
        if all(x == 6 for x in early):
            return (
                "HYBRID_FULL_FLASHJT",
                "All five seeds are final; one assisted run used the FlashJT stop while another continued to epoch 10."
            )

        return (
            "HYBRID_EARLY_FINAL_REVIEW",
            "All five seeds are final, but at least one assisted seed ended at an unusual epoch."
        )

    return "COMPLETE_OTHER", "All five final results exist."


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_metrics(rows):
    out = {}

    for metric in PERFORMANCE_METRICS + OTHER_METRICS:
        if metric not in rows.columns:
            continue

        vals = pd.to_numeric(
            rows[metric],
            errors="coerce"
        ).dropna()

        if len(vals) == 0:
            out[f"{metric}_n"] = 0
            out[f"{metric}_mean"] = np.nan
            out[f"{metric}_std"] = np.nan
            continue

        out[f"{metric}_n"] = len(vals)
        out[f"{metric}_mean"] = vals.mean()

        out[f"{metric}_std"] = (
            vals.std(ddof=1)
            if len(vals) > 1
            else np.nan
        )

    return out


def metric_entry(mean, std, performance=False):
    if pd.isna(mean):
        return ""

    scale = 1.0

    # Most classification metrics are stored 0..1.
    if performance and abs(mean) <= 1.5:
        scale = 100.0

    mean = mean * scale
    std = std * scale if not pd.isna(std) else np.nan

    if pd.isna(std):
        return f"{mean:.2f}"

    return f"{mean:.2f} ± {std:.2f}"


def final_epoch_for_table1(method, seed, row, tmap):
    if not pd.isna(row.get("result_epoch", np.nan)):
        return int(row["result_epoch"])

    ep, _ = table1_epoch(tmap, method, seed)
    return ep


def build_table1(dataset, results, tdf):
    tmap = trajectory_map(tdf)

    t1 = results[
        (results["table"] == "I")
        & (results["kind"] == "TABLE1_RESULT")
    ].copy()

    aggregate_rows = []
    seed_progress_rows = []

    result_lookup = {}

    for _, r in t1.iterrows():
        result_lookup[(r["method"], int(r["seed"]))] = r

    for method in TABLE1_METHODS:
        final_rows = []

        seed_epochs = {}
        allow_flags = {}
        final_seed_list = []

        for seed in EXPECTED_SEEDS:
            key = (method, seed)
            result = result_lookup.get(key)

            ep, allow = table1_epoch(tmap, method, seed)
            result_ready = result is not None

            if result_ready:
                final_ep = final_epoch_for_table1(
                    method, seed, result, tmap
                )
                final_rows.append(result)
                final_seed_list.append(seed)
            else:
                final_ep = ep

            seed_epochs[seed] = final_ep
            allow_flags[seed] = allow

            if result_ready:
                status = "FINAL_RESULT_AVAILABLE"
            elif not pd.isna(ep):
                status = "IN_PROGRESS"
            else:
                status = "NOT_STARTED_OR_WAITING"

            seed_progress_rows.append({
                "dataset": dataset,
                "table": "I",
                "method": method,
                "seed": seed,
                "seed_role": (
                    "FULL_HISTORY"
                    if seed in HISTORY_SEEDS
                    else "FLASHJT_ELIGIBLE"
                ),
                "epoch": final_ep,
                "result_ready": result_ready,
                "status": status,
                "flashjt_allow_stop_seen": allow,
                "source_file": (
                    result["source_file"]
                    if result_ready
                    else ""
                ),
            })

        final_seed_list = sorted(final_seed_list)
        missing = [
            s for s in EXPECTED_SEEDS
            if s not in final_seed_list
        ]

        complete = len(final_seed_list) == 5

        if complete:
            protocol, protocol_doc = classify_protocol(
                seed_epochs,
                allow_flags
            )
            readiness = "TABLE_READY"
            aggregate_scope = "FINAL_5_SEED"
        else:
            protocol = "INCOMPLETE"
            protocol_doc = (
                "Provisional aggregate over currently available final seed results only; "
                "do not use as a final paper-table value."
            )
            readiness = "PROVISIONAL_ONLY"
            aggregate_scope = "AVAILABLE_FINAL_SEEDS_ONLY"

        rows_df = (
            pd.DataFrame(final_rows)
            if final_rows
            else pd.DataFrame()
        )

        agg = (
            aggregate_metrics(rows_df)
            if not rows_df.empty
            else {}
        )

        progress = ";".join(
            f"{s}:{'NA' if pd.isna(seed_epochs[s]) else int(seed_epochs[s])}"
            + ("F" if s in final_seed_list else "R")
            for s in EXPECTED_SEEDS
        )

        row = {
            "dataset": dataset,
            "table": "I",
            "method": method,
            "readiness": readiness,
            "training_protocol": protocol,
            "protocol_documentation": protocol_doc,
            "aggregate_scope": aggregate_scope,
            "n_final_seeds": len(final_seed_list),
            "expected_seeds": "1|42|48|550|2026",
            "available_final_seeds": "|".join(map(str, final_seed_list)),
            "missing_final_seeds": "|".join(map(str, missing)),
            "seed_epoch_progress": progress,
            **agg,
        }

        for metric in PERFORMANCE_METRICS:
            row[f"{metric}_entry"] = metric_entry(
                row.get(f"{metric}_mean", np.nan),
                row.get(f"{metric}_std", np.nan),
                performance=True,
            )

        # ----------------------------------------------------
        # Training-time reporting for Table I
        # ----------------------------------------------------
        train_mean_sec = row.get("train_time_sec_mean", np.nan)
        train_std_sec = row.get("train_time_sec_std", np.nan)

        if not pd.isna(train_mean_sec):
            row["train_time_hours_mean"] = train_mean_sec / 3600.0
            row["train_time_hours_std"] = (
                train_std_sec / 3600.0
                if not pd.isna(train_std_sec)
                else np.nan
            )

            row["train_time_hours_entry"] = metric_entry(
                row["train_time_hours_mean"],
                row["train_time_hours_std"],
                performance=False,
            )
        else:
            row["train_time_hours_mean"] = np.nan
            row["train_time_hours_std"] = np.nan
            row["train_time_hours_entry"] = ""

        for metric in OTHER_METRICS:
            row[f"{metric}_entry"] = metric_entry(
                row.get(f"{metric}_mean", np.nan),
                row.get(f"{metric}_std", np.nan),
                performance=False,
            )

        aggregate_rows.append(row)

    return (
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(seed_progress_rows),
    )


def build_table2_component_progress(dataset, results, tdf):
    """
    Component readiness is diagnostic.
    It is NOT automatically a Table-II paper row.
    """

    tmap = trajectory_map(tdf)

    comp_results = results[
        (results["table"] == "II")
        & (results["kind"] == "COMPONENT_RESULT")
    ].copy()

    lookup = {}

    for _, r in comp_results.iterrows():
        lookup[
            (
                int(r["candidate"]),
                str(r["component"]),
                int(r["seed"]),
            )
        ] = r

    rows = []

    for candidate in [160, 180]:
        for component in TABLE2_COMPONENTS:
            for seed in EXPECTED_SEEDS:
                tr = tmap.get(
                    (
                        "II_COMPONENT",
                        f"candidate_{candidate}",
                        component,
                        seed,
                    )
                )

                ep = tr["epoch"] if tr else np.nan
                allow = tr["allow_stop"] if tr else False

                rr = lookup.get(
                    (candidate, component, seed)
                )

                ready = rr is not None

                if ready and not pd.isna(rr["result_epoch"]):
                    ep = int(rr["result_epoch"])

                if ready:
                    status = "FINAL_COMPONENT_RESULT"
                elif not pd.isna(ep):
                    status = "IN_PROGRESS"
                else:
                    status = "NOT_STARTED_OR_WAITING"

                rows.append({
                    "dataset": dataset,
                    "table": "II_COMPONENT",
                    "candidate": candidate,
                    "component": component,
                    "seed": seed,
                    "seed_role": (
                        "FULL_HISTORY"
                        if seed in HISTORY_SEEDS
                        else "FLASHJT_ELIGIBLE"
                    ),
                    "epoch": ep,
                    "result_ready": ready,
                    "status": status,
                    "flashjt_allow_stop_seen": allow,
                    "source_file": (
                        rr["source_file"]
                        if ready else ""
                    ),
                })

    return pd.DataFrame(rows)


def build_table2_component_aggregates(dataset, results, component_progress):
    """
    Diagnostic-only aggregates for the three constituent models.
    These are NOT labelled as final Table-II routing entries.
    """

    comp_results = results[
        (results["table"] == "II")
        & (results["kind"] == "COMPONENT_RESULT")
    ].copy()

    rows = []

    for candidate in [160, 180]:
        for component in TABLE2_COMPONENTS:
            part = comp_results[
                (comp_results["candidate"] == candidate)
                & (comp_results["component"] == component)
            ].copy()

            seeds = sorted(
                set(
                    int(x)
                    for x in part["seed"].dropna().tolist()
                    if int(x) in EXPECTED_SEEDS
                )
            )

            agg = aggregate_metrics(part) if not part.empty else {}

            rows.append({
                "dataset": dataset,
                "candidate": candidate,
                "component": component,
                "readiness": (
                    "ALL_5_COMPONENT_RESULTS"
                    if len(seeds) == 5
                    else "INCOMPLETE_COMPONENT"
                ),
                "usage": "DIAGNOSTIC_ONLY_NOT_TABLE_II_ROUTING_ROW",
                "n_final_seeds": len(seeds),
                "available_final_seeds": "|".join(map(str, seeds)),
                "missing_final_seeds": "|".join(
                    str(s) for s in EXPECTED_SEEDS
                    if s not in seeds
                ),
                **agg,
            })

    return pd.DataFrame(rows)


def table2_candidate_epoch_map(tdf, candidate):
    tmap = trajectory_map(tdf)
    out = {}
    allow = {}

    for seed in EXPECTED_SEEDS:
        vals, allow_stop = candidate_component_epochs(
            tmap,
            candidate,
            seed,
        )

        # For a candidate-level endpoint, the slowest constituent matters.
        out[seed] = max(vals) if vals else np.nan
        allow[seed] = allow_stop

    return out, allow


def build_table2_routing_aggregates(dataset, results, tdf):
    """
    Actual Table-II result rows require discovered routing/threshold rows.

    If the current codebase does not write such rows, this file will be empty
    instead of inventing Table-II values from component-training metrics.
    """

    rr = results[
        (results["table"] == "II")
        & (results["kind"] == "ROUTING_RESULT")
    ].copy()

    if rr.empty:
        return pd.DataFrame()

    output = []

    grouping = [
        "method",
        "candidate",
        "threshold",
    ]

    for keys, part in rr.groupby(grouping, dropna=False):
        method, candidate, threshold = keys
        candidate = int(candidate)

        part = (
            part.sort_values("mtime")
            .drop_duplicates("seed", keep="last")
        )

        seeds = sorted(
            int(x)
            for x in part["seed"].dropna().unique()
            if int(x) in EXPECTED_SEEDS
        )

        missing = [
            s for s in EXPECTED_SEEDS
            if s not in seeds
        ]

        complete = len(seeds) == 5

        epochs, allows = table2_candidate_epoch_map(
            tdf, candidate
        )

        if complete:
            protocol, doc = classify_protocol(
                epochs, allows
            )
            readiness = "TABLE_READY"
            scope = "FINAL_5_SEED"
        else:
            protocol = "INCOMPLETE"
            doc = (
                "Provisional Table-II aggregate over currently available "
                "final routing rows only."
            )
            readiness = "PROVISIONAL_ONLY"
            scope = "AVAILABLE_FINAL_SEEDS_ONLY"

        agg = aggregate_metrics(part)

        row = {
            "dataset": dataset,
            "table": "II",
            "method": method,
            "candidate": candidate,
            "threshold": threshold,
            "readiness": readiness,
            "training_protocol": protocol,
            "protocol_documentation": doc,
            "aggregate_scope": scope,
            "n_final_seeds": len(seeds),
            "available_final_seeds": "|".join(map(str, seeds)),
            "missing_final_seeds": "|".join(map(str, missing)),
            **agg,
        }

        for metric in PERFORMANCE_METRICS:
            row[f"{metric}_entry"] = metric_entry(
                row.get(f"{metric}_mean", np.nan),
                row.get(f"{metric}_std", np.nan),
                performance=True,
            )

        output.append(row)

    return pd.DataFrame(output)


# ============================================================
# OUTPUT
# ============================================================

def safe_concat(dfs):
    good = [x for x in dfs if x is not None and not x.empty]
    return pd.concat(good, ignore_index=True) if good else pd.DataFrame()


def write_readme(outdir):
    txt = """QSTAR TABLE EXPORTS
===================

FILES
-----

seed_progress.csv
    One row per Table-I method/seed.
    Shows epoch, seed role, result availability and current state.

table1_all_aggregates.csv
    All Table-I methods.
    Finished rows use all five seeds.
    Incomplete rows aggregate only currently available FINAL results.

table1_ready_entries.csv
    Only fully finished five-seed Table-I rows.
    These are the rows eligible for paper-table use.

table1_incomplete_preview.csv
    Incomplete Table-I experiments.
    Metrics are provisional and are for monitoring only.

table2_component_progress.csv
    Progress of confidence_linear, matched_mlp and ketgpt_quantum
    components for adaptive candidate 160 and 180.

table2_component_aggregates.csv
    Diagnostic component aggregates.
    These are NOT automatically Table-II routing rows.

table2_all_aggregates.csv
    Threshold/routing aggregates discovered from seed-level Table-II
    output files, if such files exist.

table2_ready_entries.csv
    Fully finished five-seed Table-II routing rows only.

table2_incomplete_preview.csv
    Provisional Table-II routing aggregates.

raw_discovered_result_rows.csv
    Deduplicated seed-level rows found in the experiment CSVs.

discovered_csv_schemas.csv
    File/column inventory used to audit result discovery.

status_dictionary.csv
    Meaning of the training/readiness labels.


TRAINING PROTOCOL LABELS
------------------------

FULL_TRAINING
    All five final seed results exist.
    Historical seeds are full histories and both assisted seeds reached epoch 10.

FLASHJT_ASSISTED
    All five final results exist.
    Historical seeds are full histories.
    Assisted seeds terminated through the validated FlashJT stopping point.

HYBRID_FULL_FLASHJT
    All five final results exist.
    One assisted seed stopped through FlashJT while another continued to epoch 10.

INCOMPLETE
    Fewer than five final result rows exist.
    Any aggregate is provisional and MUST NOT be quoted as a final five-seed table entry.

EARLY_FINAL_REVIEW / HYBRID_EARLY_FINAL_REVIEW
    A final assisted run appears to end at an unusual epoch (for example 7-9).
    Inspect provenance before using it in the paper.


IMPORTANT
---------

An epoch >= 6 by itself is NOT treated as a completed FlashJT run.

The script uses final result rows as the primary completion signal.
The trajectory epoch is then used to document whether the final result came
from full training or an assisted/early endpoint.

For Table II, component-training metrics and final routing metrics are kept
separate deliberately. The script will not fabricate a Table-II routing row
from component metrics.
"""
    (outdir / "README.txt").write_text(txt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
    )
    parser.add_argument(
        "--out",
        default=None,
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    outdir = (
        Path(args.out)
        if args.out
        else ROOT / "results" / "table_exports" / f"export_{stamp}"
    )

    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_schema = []
    all_t1 = []
    all_progress = []
    all_t2_progress = []
    all_t2_compagg = []
    all_t2 = []

    run_manifest = []

    for dataset in args.datasets:
        runroot = latest_run(dataset)

        if runroot is None:
            print(f"[WARN] No run found for {dataset}")
            continue

        print(f"[SCAN] {dataset}: {runroot}")

        run_manifest.append({
            "dataset": dataset,
            "run_root": str(runroot),
        })

        trajectories = collect_trajectories(
            dataset,
            runroot,
        )

        results, schema = scan_result_csvs(
            dataset,
            runroot,
        )

        if not results.empty:
            all_results.append(results)

        if not schema.empty:
            all_schema.append(schema)

        t1, progress = build_table1(
            dataset,
            results,
            trajectories,
        )

        all_t1.append(t1)
        all_progress.append(progress)

        t2progress = build_table2_component_progress(
            dataset,
            results,
            trajectories,
        )
        all_t2_progress.append(t2progress)

        t2comp = build_table2_component_aggregates(
            dataset,
            results,
            t2progress,
        )
        all_t2_compagg.append(t2comp)

        t2 = build_table2_routing_aggregates(
            dataset,
            results,
            trajectories,
        )

        if not t2.empty:
            all_t2.append(t2)

    raw_results = safe_concat(all_results)
    schema = safe_concat(all_schema)
    t1 = safe_concat(all_t1)
    progress = safe_concat(all_progress)
    t2progress = safe_concat(all_t2_progress)
    t2comp = safe_concat(all_t2_compagg)
    t2 = safe_concat(all_t2)

    # --------------------------------------------------------
    # Table I outputs
    # --------------------------------------------------------

    if not t1.empty:
        t1.to_csv(
            outdir / "table1_all_aggregates.csv",
            index=False,
        )

        t1[
            t1["readiness"] == "TABLE_READY"
        ].to_csv(
            outdir / "table1_ready_entries.csv",
            index=False,
        )

        t1[
            t1["readiness"] != "TABLE_READY"
        ].to_csv(
            outdir / "table1_incomplete_preview.csv",
            index=False,
        )

    if not progress.empty:
        progress.to_csv(
            outdir / "seed_progress.csv",
            index=False,
        )

    # --------------------------------------------------------
    # Table II outputs
    # --------------------------------------------------------

    if not t2progress.empty:
        t2progress.to_csv(
            outdir / "table2_component_progress.csv",
            index=False,
        )

    if not t2comp.empty:
        t2comp.to_csv(
            outdir / "table2_component_aggregates.csv",
            index=False,
        )

    if not t2.empty:
        t2.to_csv(
            outdir / "table2_all_aggregates.csv",
            index=False,
        )

        t2[
            t2["readiness"] == "TABLE_READY"
        ].to_csv(
            outdir / "table2_ready_entries.csv",
            index=False,
        )

        t2[
            t2["readiness"] != "TABLE_READY"
        ].to_csv(
            outdir / "table2_incomplete_preview.csv",
            index=False,
        )
    else:
        # Make empty files so absence is explicit.
        pd.DataFrame().to_csv(
            outdir / "table2_all_aggregates.csv",
            index=False,
        )
        pd.DataFrame().to_csv(
            outdir / "table2_ready_entries.csv",
            index=False,
        )
        pd.DataFrame().to_csv(
            outdir / "table2_incomplete_preview.csv",
            index=False,
        )

    if not raw_results.empty:
        raw_results.to_csv(
            outdir / "raw_discovered_result_rows.csv",
            index=False,
        )

    if not schema.empty:
        schema.to_csv(
            outdir / "discovered_csv_schemas.csv",
            index=False,
        )

    pd.DataFrame(run_manifest).to_csv(
        outdir / "run_roots.csv",
        index=False,
    )

    dictionary = pd.DataFrame([
        {
            "label": "TABLE_READY",
            "meaning": "All five expected seed result rows are available.",
        },
        {
            "label": "PROVISIONAL_ONLY",
            "meaning": "Incomplete experiment; aggregate uses only currently final seeds.",
        },
        {
            "label": "FULL_TRAINING",
            "meaning": "Five final seeds; assisted seeds also reached epoch 10.",
        },
        {
            "label": "FLASHJT_ASSISTED",
            "meaning": "Five final seeds; assisted seeds used the validated FlashJT stopping point.",
        },
        {
            "label": "HYBRID_FULL_FLASHJT",
            "meaning": "Five final seeds; one assisted seed stopped through FlashJT and another ran to epoch 10.",
        },
        {
            "label": "INCOMPLETE",
            "meaning": "Not all five final result rows are available.",
        },
        {
            "label": "EARLY_FINAL_REVIEW",
            "meaning": "Final assisted result appears at an unusual epoch; inspect provenance before publication.",
        },
    ])

    dictionary.to_csv(
        outdir / "status_dictionary.csv",
        index=False,
    )

    write_readme(outdir)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("EXPORT COMPLETE")
    print("=" * 100)
    print(outdir)
    print()

    if not t1.empty:
        print("TABLE I")
        print("-" * 100)

        cols = [
            "dataset",
            "method",
            "readiness",
            "training_protocol",
            "n_final_seeds",
            "available_final_seeds",
            "missing_final_seeds",
            "accuracy_entry",
            "f1_macro_entry",
            "auc_ovr_macro_entry",
        ]

        print(
            t1[cols]
            .to_string(index=False)
        )

    print()

    if not t2.empty:
        print("TABLE II ROUTING RESULTS")
        print("-" * 100)

        cols = [
            "dataset",
            "method",
            "readiness",
            "training_protocol",
            "n_final_seeds",
            "accuracy_entry",
            "f1_macro_entry",
            "auc_ovr_macro_entry",
        ]

        print(
            t2[cols]
            .to_string(index=False)
        )
    else:
        print(
            "No explicit threshold/routing-level Table-II result rows "
            "were discovered."
        )
        print(
            "Component progress/aggregates were still exported separately."
        )

    print()
    print("Paper-ready Table I:")
    print(outdir / "table1_ready_entries.csv")

    print("Incomplete Table I preview:")
    print(outdir / "table1_incomplete_preview.csv")

    print("Table II component progress:")
    print(outdir / "table2_component_progress.csv")

    print("Paper-ready Table II:")
    print(outdir / "table2_ready_entries.csv")


if __name__ == "__main__":
    main()
