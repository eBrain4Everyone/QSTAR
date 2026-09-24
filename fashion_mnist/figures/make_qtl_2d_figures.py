import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 400,
    "axes.linewidth": 0.8,
})

COLORS = {
    "Classical": "#4C78A8",
    "Standard QTL": "#E45756",
    "KetGPT": "#54A24B",
    "Adaptive Classical": "#F58518",
    "Adaptive KetGPT": "#7B61FF",
}

MARKERS = {
    "Classical": "o",
    "Standard QTL": "^",
    "KetGPT": "D",
    "Adaptive Classical": "s",
    "Adaptive KetGPT": "P",
}

PATHS = {
    "standard": "../results_qtl_missing_resource_ablation.csv",
    "ketgpt": "../KetGPT_Qubit_Ablation/results_ketgpt_qubit_ablation.csv",
    "step5": "../Step5_Full_Adaptive/step5_full_adaptive_ketgpt_results.csv",
    "shots_ab": "../Shot_Ablation/results_ketgpt180_shot_ablation_antsbees.csv",
}

def load(path):
    if not os.path.exists(path):
        print("Missing:", path)
        return pd.DataFrame()
    return pd.read_csv(path)

def clean_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save(name):
    path = os.path.join(OUTDIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0.04)
    print("Saved:", path)
    plt.close()

def pareto_frontier(x, y):
    pts = sorted(zip(x, y), key=lambda t: t[0])
    frontier = []
    best_y = -1
    for xi, yi in pts:
        if yi > best_y:
            frontier.append((xi, yi))
            best_y = yi
    return zip(*frontier) if frontier else ([], [])

def build_tradeoff_df():
    rows = []

    std = load(PATHS["standard"])
    if not std.empty:
        for _, r in std.iterrows():
            rows.append({
                "family": "Standard QTL",
                "label": f"{int(r.num_qubits)}Q-D{r.circuit_depth}",
                "accuracy": r.accuracy * 100,
                "f1": r.f1_macro,
                "train_time": r.train_time_sec,
                "inference": r.inference_time_ms_per_sample,
                "params": r.total_trainable_params,
                "quantum_params": r.quantum_params,
                "gates": r.num_gates_approx,
                "qubits": r.num_qubits,
                "circuit_evals": r.train_circuit_evaluations_approx,
            })

    kg = load(PATHS["ketgpt"])
    if not kg.empty:
        for _, r in kg.iterrows():
            rows.append({
                "family": "KetGPT",
                "label": f"#{int(r.ketgpt_id)}",
                "accuracy": r.accuracy * 100,
                "f1": r.f1_macro,
                "train_time": r.train_time_sec,
                "inference": r.inference_time_ms_per_sample,
                "params": r.total_trainable_params,
                "quantum_params": r.quantum_params,
                "gates": r.num_gates_approx,
                "qubits": r.num_qubits,
                "circuit_evals": r.train_circuit_evaluations_approx,
            })

    st5 = load(PATHS["step5"])
    if not st5.empty:
        use = st5[st5["experiment_type"].isin(["fixed", "adaptive_full"])].copy()
        for _, r in use.iterrows():
            name = str(r.model_name)
            if "adaptive_ketgpt" in name:
                fam = "Adaptive KetGPT"
                label = f"AQ-{float(r.threshold):.1f}"
            elif "adaptive_classical" in name:
                fam = "Adaptive Classical"
                label = f"AC-{float(r.threshold):.1f}"
            elif "ketgpt" in name:
                fam = "KetGPT"
                label = "Fixed KG"
            else:
                fam = "Classical"
                label = name.replace("confidence_linear", "Linear").replace("matched_mlp", "MLP")
            rows.append({
                "family": fam,
                "label": label,
                "accuracy": r.accuracy * 100,
                "f1": r.f1_macro,
                "train_time": r.train_time_sec,
                "inference": r.inference_time_ms_per_sample,
                "params": r.total_trainable_params,
                "quantum_params": r.quantum_params,
                "gates": r.num_gates_approx,
                "qubits": r.num_qubits,
                "circuit_evals": r.train_circuit_evaluations_approx,
                "threshold": r.threshold,
                "qroute": r.routed_to_quantum_pct,
                "avg_shots": r.avg_shots_per_sample,
            })

    return pd.DataFrame(rows)

# ------------------------------------------------------------
# FIGURE 1: Performance-cost tradeoff, like benchmark style
# ------------------------------------------------------------
def fig_performance_cost():
    df = build_tradeoff_df()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 3.9))

    for fam in ["Classical", "Standard QTL", "KetGPT", "Adaptive Classical", "Adaptive KetGPT"]:
        sub = df[df.family == fam]
        if sub.empty:
            continue
        ax.scatter(
            sub.train_time,
            sub.accuracy,
            c=COLORS[fam],
            marker=MARKERS[fam],
            s=55,
            edgecolor="black",
            linewidth=0.45,
            label=fam,
            alpha=0.9,
        )
        for _, r in sub.iterrows():
            if fam in ["KetGPT", "Adaptive KetGPT"] or r.label in ["Linear", "MLP"]:
                ax.text(r.train_time * 1.04, r.accuracy + 0.25, r.label, fontsize=7)

    fx, fy = pareto_frontier(df.train_time.values, df.accuracy.values)
    ax.plot(list(fx), list(fy), "--", color="black", linewidth=1.0, label="Pareto frontier")

    ax.set_xscale("log")
    ax.set_xlabel("Training Time (s, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Performance--Cost Trade-off on Fashion-MNIST")
    ax.set_ylim(min(34, df.accuracy.min() - 3), max(84, df.accuracy.max() + 2))
    clean_axes(ax)

    ax.annotate(
        "Compact KetGPT\nnear Pareto-optimal",
        xy=(df[df.label=="#160"].train_time.iloc[0], df[df.label=="#160"].accuracy.iloc[0]) if "#160" in set(df.label) else (3000, 81),
        xytext=(700, 78),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.25", fc="#eef6e8", ec="black", lw=0.6),
        fontsize=8,
    )

    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.34), frameon=True)
    save("fig1_performance_cost_tradeoff.png")

# ------------------------------------------------------------
# FIGURE 2: Adaptive routing curve
# ------------------------------------------------------------
def fig_adaptive_routing():
    df = load(PATHS["step5"])
    if df.empty:
        return

    ad = df[df.experiment_type == "adaptive_full"].copy()
    ad["acc_pct"] = ad.accuracy * 100

    ac = ad[ad.model_name == "adaptive_classical"].sort_values("threshold")
    aq = ad[ad.model_name == "adaptive_ketgpt_qtl"].sort_values("threshold")

    fig, ax1 = plt.subplots(figsize=(6.7, 3.7))

    ax1.plot(ac.threshold, ac.acc_pct, marker="s", color=COLORS["Adaptive Classical"],
             linewidth=1.6, label="Adaptive Classical")
    ax1.plot(aq.threshold, aq.acc_pct, marker="P", color=COLORS["Adaptive KetGPT"],
             linewidth=1.6, label="Adaptive KetGPT-QTL")

    fixed_linear = df[(df.experiment_type=="fixed") & (df.model_name=="confidence_linear")]
    fixed_kg = df[(df.experiment_type=="fixed") & (df.model_name=="ketgpt_quantum")]
    if not fixed_linear.empty:
        ax1.axhline(fixed_linear.accuracy.iloc[0] * 100, color=COLORS["Classical"],
                    linestyle="--", linewidth=1.0, label="Fixed Linear")
    if not fixed_kg.empty:
        ax1.axhline(fixed_kg.accuracy.iloc[0] * 100, color=COLORS["KetGPT"],
                    linestyle=":", linewidth=1.2, label="Fixed KetGPT")

    ax2 = ax1.twinx()
    ax2.plot(aq.threshold, aq.routed_to_quantum_pct, marker="o", color="black",
             linewidth=1.2, label="Routed to Quantum (%)")
    ax2.set_ylabel("Routed to Quantum (%)")
    ax2.set_ylim(0, 80)

    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Adaptive Routing Operating Points")
    ax1.set_xticks([0.70, 0.80, 0.90])
    ax1.set_ylim(76.5, 81.5)
    clean_axes(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.42), frameon=True)

    ax1.annotate(
        "Best adaptive point",
        xy=(0.70, aq[aq.threshold==0.70].acc_pct.iloc[0]),
        xytext=(0.735, 81.1),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.25", fc="#efe8ff", ec="black", lw=0.6),
        fontsize=8,
    )

    save("fig2_adaptive_routing_curve.png")

# ------------------------------------------------------------
# FIGURE 3: KetGPT search landscape, 2D bubble/scatter
# ------------------------------------------------------------
def fig_ketgpt_search():
    df = load(PATHS["ketgpt"])
    if df.empty:
        return

    df = df.copy()
    df["acc_pct"] = df.accuracy * 100
    df["train_h"] = df.train_time_sec / 3600

    fig, ax = plt.subplots(figsize=(6.7, 3.7))

    q_colors = {4: "#6A5ACD", 6: "#54A24B", 8: "#F58518"}
    q_markers = {4: "o", 6: "^", 8: "D"}

    for q, sub in df.groupby("num_qubits"):
        ax.scatter(
            sub.num_gates_approx,
            sub.acc_pct,
            s=60 + sub.train_h * 6,
            c=q_colors.get(int(q), "gray"),
            marker=q_markers.get(int(q), "o"),
            edgecolor="black",
            linewidth=0.45,
            label=f"{int(q)} qubits",
            alpha=0.9,
        )
        for _, r in sub.iterrows():
            ax.text(r.num_gates_approx + 1.5, r.acc_pct + 0.25, f"#{int(r.ketgpt_id)}", fontsize=8)

    ax.set_xlabel("Gate Count")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("KetGPT Circuit Search Landscape")
    clean_axes(ax)

    ax.annotate(
        "Best accuracy with\ncompact circuit",
        xy=(9, 81.9),
        xytext=(28, 76.5),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.25", fc="#eef6e8", ec="black", lw=0.6),
        fontsize=8,
    )

    ax.legend(loc="lower right", frameon=True)
    save("fig3_ketgpt_search_landscape.png")

# ------------------------------------------------------------
# FIGURE 4: Shot ablation
# ------------------------------------------------------------
def fig_shot_ablation():
    df = load(PATHS["shots_ab"])
    if df.empty:
        return

    df = df.sort_values("shots")
    fig, ax1 = plt.subplots(figsize=(6.7, 3.7))

    ax1.plot(df.shots, df.accuracy * 100, marker="o", color=COLORS["KetGPT"],
             linewidth=1.6, label="Accuracy")
    ax1.plot(df.shots, df.f1_macro * 100, marker="D", color=COLORS["Adaptive KetGPT"],
             linewidth=1.6, label="F1-score")

    ax2 = ax1.twinx()
    ax2.plot(df.shots, df.inference_time_ms_per_sample, marker="s", color="#444444",
             linewidth=1.2, linestyle="--", label="Inference time")
    ax2.set_ylabel("Inference Time (ms/sample)")

    ax1.set_xlabel("Measurement Shots")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Shot-number Ablation on Ants & Bees")
    ax1.set_xticks([25, 50, 100, 200])
    ax1.set_ylim(89.5, 95.0)
    clean_axes(ax1)

    best = df.loc[df.accuracy.idxmax()]
    ax1.annotate(
        "Highest accuracy\nat 25 shots",
        xy=(best.shots, best.accuracy * 100),
        xytext=(65, 94.3),
        arrowprops=dict(arrowstyle="->", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.25", fc="#eef6e8", ec="black", lw=0.6),
        fontsize=8,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.32), frameon=True)

    save("fig4_shot_ablation_antsbees.png")

# ------------------------------------------------------------
# FIGURE 5: Low-confidence branch comparison
# ------------------------------------------------------------
def fig_low_confidence():
    df = load(PATHS["step5"])
    if df.empty:
        return

    low = df[df.experiment_type == "low_confidence_only"].copy()
    low["acc_pct"] = low.accuracy * 100
    low["branch"] = low.model_name.apply(lambda x: "KetGPT #180" if "ketgpt" in str(x) else "Matched MLP")

    pivot = low.pivot_table(index="threshold", columns="branch", values="acc_pct").sort_index()

    fig, ax = plt.subplots(figsize=(6.7, 3.7))
    x = np.arange(len(pivot.index))
    width = 0.34

    ax.bar(x - width/2, pivot["Matched MLP"], width,
           color=COLORS["Adaptive Classical"], edgecolor="black", linewidth=0.5, label="Matched MLP")
    ax.bar(x + width/2, pivot["KetGPT #180"], width,
           color=COLORS["KetGPT"], edgecolor="black", linewidth=0.5, label="KetGPT #180")

    for i, thr in enumerate(pivot.index):
        gain = pivot.loc[thr, "KetGPT #180"] - pivot.loc[thr, "Matched MLP"]
        ax.text(i, max(pivot.loc[thr]) + 0.9, f"+{gain:.1f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in pivot.index])
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Low-confidence Accuracy (%)")
    ax.set_title("Quantum Branch Utility on Low-confidence Samples")
    ax.set_ylim(50, 74)
    clean_axes(ax)
    ax.legend(frameon=True)
    save("fig5_low_confidence_branch_comparison.png")

if __name__ == "__main__":
    fig_performance_cost()
    fig_adaptive_routing()
    fig_ketgpt_search()
    fig_shot_ablation()
    fig_low_confidence()
