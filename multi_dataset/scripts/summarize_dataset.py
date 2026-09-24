#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from qstar_new.common import canonical_dataset_name, dataset_display_name, atomic_json_dump


def pm(mean, std, decimals):
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def load_files(root: Path, pattern: str, expected_count: int):
    paths = [p for p in sorted(glob.glob(str(root / pattern))) if not p.endswith("_candidates.csv")]
    if len(paths) != expected_count:
        raise RuntimeError(f"{pattern}: expected {expected_count} files, found {len(paths)}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = os.path.relpath(path, root)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_seeds(frame, expected, label):
    found = sorted(frame.seed.astype(int).unique().tolist())
    if found != expected:
        raise RuntimeError(f"{label}: expected seeds {expected}, found {found}")


def latex_table1(frame, path: Path, dataset: str, preview: bool):
    note = " PREVIEW: accelerated seeds may have stopped before epoch 10." if preview else ""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Fixed-head comparison on {dataset} with a frozen ResNet18 backbone. Results are mean $\pm$ sample standard deviation over five seeds; quantum models use 100 shots. Train time excludes the one-time shared frozen-feature cache construction.{note}}}",
        rf"\label{{tab:{dataset.lower().replace('-', '').replace(' ', '_')}_fixed_head}}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Model & Head & Qub. & Dep. & Head Par. & Acc. (\%) & F1 & AUC & Train (h) \\",
        r"\midrule",
    ]
    for _, r in frame.iterrows():
        vals = [r["Model"], r["Head"], r["Qub."], r["Dep."], r["Head Par."], r["Acc. (%)"], r["F1"], r["AUC"], r["Train (h)"]]
        lines.append(" & ".join(str(x).replace("±", r"$\pm$") for x in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines))


def latex_table2(frame, path: Path, dataset: str, preview: bool):
    note = " PREVIEW: accelerated seeds may have stopped before epoch 10." if preview else ""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Adaptive routing results on {dataset} using KetGPT Candidates \#180 and \#160 as quantum fallbacks. Results are mean $\pm$ sample standard deviation over five seeds.{note}}}",
        rf"\label{{tab:{dataset.lower().replace('-', '').replace(' ', '_')}_adaptive}}",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Exp. & Model & Thr. & Rout. (\%) & Params & Acc. (\%) & Prec. & Rec. & F1 & AUC \\",
        r"\midrule",
    ]
    for _, r in frame.iterrows():
        vals = [r["Exp."], r["Model"], r["Thr."], r["Rout. (%)"], r["Params"], r["Acc. (%)"], r["Prec."], r["Rec."], r["F1"], r["AUC"]]
        lines.append(" & ".join(str(x).replace("#", r"\#").replace("±", r"$\pm$") for x in vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", default="1:42:48:550:2026")
    ap.add_argument("--target-epochs", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.run_root)
    dataset_key = canonical_dataset_name(args.dataset)
    dataset = dataset_display_name(dataset_key)
    expected_seeds = sorted(int(x) for x in args.seeds.split(":"))
    out = root / "summary"
    out.mkdir(parents=True, exist_ok=True)

    classical = load_files(root, "table1/classical/seed_*.csv", 5)
    standard = load_files(root, "table1/standard_qtl/seed_*_q*_d*.csv", 40)
    ketgpt = load_files(root, "table1/ketgpt/seed_*.csv", 5)
    adaptive180 = load_files(root, "table2/adaptive180/seed_*.csv", 5)
    adaptive160 = load_files(root, "table2/adaptive160/seed_*.csv", 5)

    for frame, label in [(classical,"classical"),(standard,"standard"),(ketgpt,"ketgpt"),(adaptive180,"adaptive180"),(adaptive160,"adaptive160")]:
        check_seeds(frame, expected_seeds, label)

    # ---------- Table I ----------
    classical = classical.copy()
    classical["family"] = "Classical"
    classical["paper_model"] = classical.model_name.map({"linear":"Linear", "matched_mlp":"MLP"})
    classical["paper_head"] = classical["paper_model"]
    classical["paper_qubits"] = "–"
    classical["paper_depth"] = "–"
    classical["paper_params"] = classical.head_trainable_params.astype(int)

    standard = standard.copy()
    standard["family"] = "Standard QTL"
    standard["paper_model"] = "Standard QTL"
    standard["paper_head"] = "Var."
    standard["paper_qubits"] = standard.num_qubits.astype(int).astype(str)
    standard["paper_depth"] = standard.circuit_depth.astype(int).astype(str)
    standard["paper_params"] = standard.quantum_params.astype(int)

    ketgpt = ketgpt[ketgpt.ketgpt_id.astype(int).isin([160,180])].copy()
    expected_ket_rows = 10
    if len(ketgpt) != expected_ket_rows:
        raise RuntimeError(f"Table I KetGPT expected {expected_ket_rows} seed rows (#160/#180), found {len(ketgpt)}")
    ketgpt["family"] = "KetGPT"
    ketgpt["paper_model"] = "KetGPT #" + ketgpt.ketgpt_id.astype(int).astype(str)
    ketgpt["paper_head"] = "Cand. #" + ketgpt.ketgpt_id.astype(int).astype(str)
    ketgpt["paper_qubits"] = ketgpt.num_qubits.astype(int).astype(str)
    ketgpt["paper_depth"] = "Var."
    ketgpt["paper_params"] = ketgpt.quantum_params.astype(int)

    t1 = pd.concat([classical, standard, ketgpt], ignore_index=True)
    t1["accuracy_pct"] = 100.0 * t1.accuracy
    t1["train_time_h"] = t1.train_time_sec / 3600.0
    t1.to_csv(out / "table1_all_seed_results.csv", index=False)

    group1 = ["family","paper_model","paper_head","paper_qubits","paper_depth","paper_params"]
    s1 = t1.groupby(group1, sort=False).agg(
        n=("seed","count"), accuracy_mean=("accuracy_pct","mean"), accuracy_std=("accuracy_pct","std"),
        f1_mean=("f1_macro","mean"), f1_std=("f1_macro","std"),
        auc_mean=("auc_ovr_macro","mean"), auc_std=("auc_ovr_macro","std"),
        train_h_mean=("train_time_h","mean"), train_h_std=("train_time_h","std"),
        min_epochs=("epochs_executed","min"), max_epochs=("epochs_executed","max"),
    ).reset_index()
    family_order = {"Classical":0,"Standard QTL":1,"KetGPT":2}
    model_order = {"Linear":0,"MLP":1,"KetGPT #160":0,"KetGPT #180":1}
    s1["_family"] = s1.family.map(family_order)
    s1["_model"] = s1.paper_model.map(model_order).fillna(0)
    s1["_q"] = pd.to_numeric(s1.paper_qubits, errors="coerce").fillna(-1)
    s1["_d"] = pd.to_numeric(s1.paper_depth, errors="coerce").fillna(-1)
    s1 = s1.sort_values(["_family","_model","_q","_d"]).drop(columns=["_family","_model","_q","_d"])
    if len(s1) != 12 or not (s1.n == 5).all():
        raise RuntimeError("Table I must contain exactly 12 five-seed rows")
    s1.to_csv(out / "table1_mean_std_numeric.csv", index=False)
    display_model = []
    previous_family = None
    for family in s1.family.tolist():
        display_model.append(family if family != previous_family else "")
        previous_family = family
    paper1 = pd.DataFrame({
        "Model":display_model, "Head":s1.paper_head, "Qub.":s1.paper_qubits, "Dep.":s1.paper_depth,
        "Head Par.":s1.paper_params.astype(int),
        "Acc. (%)":[pm(m,s,1) for m,s in zip(s1.accuracy_mean,s1.accuracy_std)],
        "F1":[pm(m,s,3) for m,s in zip(s1.f1_mean,s1.f1_std)],
        "AUC":[pm(m,s,3) for m,s in zip(s1.auc_mean,s1.auc_std)],
        "Train (h)":[pm(m,s,2) for m,s in zip(s1.train_h_mean,s1.train_h_std)],
        "n":s1.n,
    })

    # ---------- Table II ----------
    # Use the #180 run as the reference source for the shared MLP/Classical rows,
    # and candidate-specific rows from each candidate run. This mirrors the current QSTAR table.
    a180 = adaptive180[adaptive180.threshold.notna()].copy()
    a160 = adaptive160[adaptive160.threshold.notna()].copy()

    pieces = []
    for thr in [0.70,0.80,0.90]:
        def one(frame, model):
            x = frame[(np.isclose(frame.threshold.astype(float), thr)) & (frame.model_name == model)].copy()
            if len(x) != 5:
                raise RuntimeError(f"Table II {thr} {model}: expected 5 seed rows, found {len(x)}")
            return x
        x = one(a180, "matched_mlp_on_low_conf"); x["Exp."]="Low-conf."; x["Model"]="MLP"; x["_ord"]=0; pieces.append(x)
        x = one(a180, "ketgpt_quantum_on_low_conf"); x["Exp."]="Low-conf."; x["Model"]="KetGPT #180"; x["_ord"]=1; pieces.append(x)
        x = one(a160, "ketgpt_quantum_on_low_conf"); x["Exp."]="Low-conf."; x["Model"]="KetGPT #160"; x["_ord"]=2; pieces.append(x)
        x = one(a180, "adaptive_classical"); x["Exp."]="Adaptive"; x["Model"]="Classical"; x["_ord"]=3; pieces.append(x)
        x = one(a180, "adaptive_ketgpt_qtl"); x["Exp."]="Adaptive"; x["Model"]="KetGPT #180"; x["_ord"]=4; pieces.append(x)
        x = one(a160, "adaptive_ketgpt_qtl"); x["Exp."]="Adaptive"; x["Model"]="KetGPT #160"; x["_ord"]=5; pieces.append(x)
    t2 = pd.concat(pieces, ignore_index=True)
    t2["accuracy_pct"] = 100.0 * t2.accuracy
    t2.to_csv(out / "table2_all_seed_results.csv", index=False)
    s2 = t2.groupby(["threshold","Exp.","Model","_ord"], sort=False).agg(
        n=("seed","count"), route_mean=("low_pct","mean"), route_std=("low_pct","std"),
        params=("total_trainable_params","first"), params_nunique=("total_trainable_params","nunique"),
        accuracy_mean=("accuracy_pct","mean"), accuracy_std=("accuracy_pct","std"),
        precision_mean=("precision_macro","mean"), precision_std=("precision_macro","std"),
        recall_mean=("recall_macro","mean"), recall_std=("recall_macro","std"),
        f1_mean=("f1_macro","mean"), f1_std=("f1_macro","std"),
        auc_mean=("auc_ovr_macro","mean"), auc_std=("auc_ovr_macro","std"),
        min_epochs=("epochs_executed","min"), max_epochs=("epochs_executed","max"),
    ).reset_index().sort_values(["threshold","_ord"])
    if len(s2) != 18 or not (s2.n == 5).all() or not (s2.params_nunique == 1).all():
        raise RuntimeError("Table II must contain exactly 18 five-seed rows with stable parameter counts")
    s2.to_csv(out / "table2_mean_std_numeric.csv", index=False)
    paper2 = pd.DataFrame({
        "Exp.":s2["Exp."], "Model":s2["Model"], "Thr.":s2.threshold.map(lambda x:f"{x:.2f}"),
        "Rout. (%)":[pm(m,s,2 if ("160" in model) else 1) for m,s,model in zip(s2.route_mean,s2.route_std,s2["Model"])],
        "Params":s2.params.astype(int),
        "Acc. (%)":[pm(m,s,2) for m,s in zip(s2.accuracy_mean,s2.accuracy_std)],
        "Prec.":[pm(m,s,3) for m,s in zip(s2.precision_mean,s2.precision_std)],
        "Rec.":[pm(m,s,3) for m,s in zip(s2.recall_mean,s2.recall_std)],
        "F1":[pm(m,s,3) for m,s in zip(s2.f1_mean,s2.f1_std)],
        "AUC":[pm(m,s,3) for m,s in zip(s2.auc_mean,s2.auc_std)],
        "n":s2.n,
    })

    # Audit accelerated/full status before producing publication filenames.
    audit_cols = [c for c in ["dataset","seed","model_name","ketgpt_id","candidate_id","epochs_executed","best_epoch","stop_reason","source_file"] if c in t1.columns]
    audit1 = t1[audit_cols].copy()
    audit2_cols = [c for c in ["dataset","seed","experiment_type","model_name","candidate_id","threshold","epochs_executed","best_epoch","stop_reason","source_file"] if c in t2.columns]
    audit2 = t2[audit2_cols].copy()
    audit1.to_csv(out / "table1_training_audit.csv", index=False)
    audit2.to_csv(out / "table2_training_audit.csv", index=False)

    full1 = bool((t1.epochs_executed.astype(int) >= args.target_epochs).all())
    full2 = bool((t2.epochs_executed.astype(int) >= args.target_epochs).all())
    publication_ready = full1 and full2

    paper1.to_csv(out / "table1_preview_mean_std.csv", index=False)
    paper2.to_csv(out / "table2_preview_mean_std.csv", index=False)
    latex_table1(paper1, out / "table1_preview_mean_std.tex", dataset, preview=not publication_ready)
    latex_table2(paper2, out / "table2_preview_mean_std.tex", dataset, preview=not publication_ready)

    if publication_ready:
        paper1.to_csv(out / "table1_paper_mean_std.csv", index=False)
        paper2.to_csv(out / "table2_paper_mean_std.csv", index=False)
        latex_table1(paper1, out / "table1_paper_mean_std.tex", dataset, preview=False)
        latex_table2(paper2, out / "table2_paper_mean_std.tex", dataset, preview=False)
    else:
        for name in ["table1_paper_mean_std.csv","table2_paper_mean_std.csv","table1_paper_mean_std.tex","table2_paper_mean_std.tex"]:
            (out / name).unlink(missing_ok=True)

    status = {
        "dataset": dataset,
        "dataset_key": dataset_key,
        "expected_seeds": expected_seeds,
        "table1_rows": 12,
        "table2_rows": 18,
        "table1_all_seeds_full_10_epochs": full1,
        "table2_all_seeds_full_10_epochs": full2,
        "publication_ready": publication_ready,
        "preview_tables_written": True,
        "paper_tables_written": publication_ready,
    }
    atomic_json_dump(status, out / "publication_status.json")
    print(json.dumps(status, indent=2))
    print("Preview tables:", out / "table1_preview_mean_std.csv", out / "table2_preview_mean_std.csv", sep="\n")
    if publication_ready:
        print("PUBLICATION TABLES READY")
    else:
        print("PREVIEW COMPLETE; resume accelerated seeds to 10 epochs before using paper filenames.")


if __name__ == "__main__":
    main()
