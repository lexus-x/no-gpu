"""
generate_figures.py — Paper-ready figures from real benchmark data.

Outputs:
  figures/fig1_success_rate.png     — grouped bar chart (main result)
  figures/fig2_scaling_curve.png    — N vs success rate (the surprising result)
  figures/fig3_component_ablation.png — what matters in scoring
  figures/fig4_trajectory.png       — example trajectory comparison
  results/table1_main.md            — markdown table
  results/table2_ablation.md        — ablation table
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
from pathlib import Path

Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

COLORS = {
    "baseline": "#4a4a4a",
    "random_N5": "#a0a0a0",
    "smooth_N5": "#7eb5d6",
    "full_N5": "#4a90d9",
    "adaptive_full": "#e74c3c",
}


def load(path):
    return pd.read_csv(path)


# ── Figure 1: Main result bar chart ───────────────────────────────

def fig1_main_result(df, out="figures/fig1_success_rate.png"):
    tasks = df["task"].unique()
    methods = df["method"].unique()

    fig, axes = plt.subplots(1, len(tasks), figsize=(5 * len(tasks), 5), sharey=True)
    if len(tasks) == 1:
        axes = [axes]

    labels = {
        "baseline": "Baseline\n(1 sample)",
        "random_N5": "Random\n(N=5)",
        "smooth_N5": "Smooth\n(N=5)",
        "full_N5": "Full Score\n(N=5)",
        "adaptive_full": "Ours: Adaptive\n(N=5→15)",
    }

    for ax, task in zip(axes, tasks):
        tdf = df[df["task"] == task].set_index("method").reindex(methods)
        bars = ax.bar(
            range(len(methods)),
            tdf["success_rate"],
            color=[COLORS.get(m, "#999") for m in methods],
            edgecolor="white", linewidth=1.5,
        )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([labels.get(m, m) for m in methods], fontsize=8)
        ax.set_title(task.replace("-", " ").title(), fontsize=14, fontweight="bold")
        ax.set_ylabel("Success Rate (%)" if ax == axes[0] else "")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, tdf["success_rate"]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                       f"{val:.0f}%", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Test-Time Multi-Sampling Improves VLA Success Rate",
                fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 2: Scaling curve ───────────────────────────────────────

def fig2_scaling(df, out="figures/fig2_scaling_curve.png"):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(df["n_samples"], df["success_rate"], "o-",
             color="#e74c3c", markersize=10, linewidth=2.5, label="Success Rate")
    ax1.set_xlabel("Number of Samples (N)", fontsize=12)
    ax1.set_ylabel("Success Rate (%)", fontsize=12, color="#e74c3c")
    ax1.tick_params(axis="y", labelcolor="#e74c3c")

    ax2 = ax1.twinx()
    ax2.plot(df["n_samples"], df["avg_latency_ms"], "s--",
             color="#4a90d9", markersize=8, linewidth=2, label="Latency")
    ax2.set_ylabel("Avg Latency (ms)", fontsize=12, color="#4a90d9")
    ax2.tick_params(axis="y", labelcolor="#4a90d9")

    # Mark the sweet spot
    best_idx = df["success_rate"].idxmax()
    best_n = df.loc[best_idx, "n_samples"]
    best_sr = df.loc[best_idx, "success_rate"]
    ax1.annotate(f"Best: N={int(best_n)}",
                xy=(best_n, best_sr),
                xytext=(best_n + 2, best_sr - 5),
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))

    ax1.set_title("Inference Scaling: Success Rate vs Latency Cost",
                 fontsize=13, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 3: Component ablation ──────────────────────────────────

def fig3_ablation(df, out="figures/fig3_component_ablation.png"):
    fig, ax = plt.subplots(figsize=(8, 5))

    variant_labels = {
        "full": "Full (all components)",
        "no_consistency": "w/o consistency",
        "no_progress": "w/o progress",
        "consistency_only": "Consistency only",
        "smoothness_only": "Smoothness only",
    }

    colors = []
    for v in df["variant"]:
        if v == "full":
            colors.append("#2ecc71")
        elif "only" in v:
            colors.append("#95a5a6")
        else:
            colors.append("#e67e22")

    bars = ax.barh(
        range(len(df)),
        df["success_rate"],
        color=colors,
        edgecolor="white", linewidth=1.5,
    )
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([variant_labels.get(v, v) for v in df["variant"]], fontsize=10)
    ax.set_xlabel("Success Rate (%)", fontsize=12)
    ax.set_title("Ablation: Which Scoring Components Matter?", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, df["success_rate"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Figure 4: Trajectory visualization ────────────────────────────

def fig4_trajectory(traj_data, out="figures/fig4_trajectory.png"):
    """
    Show action trajectories for a successful vs failed episode.
    traj_data: dict with "success" and "failed" trajectory arrays.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for ax, (label, traj) in zip(axes, traj_data.items()):
        traj = np.array(traj)
        dims = min(traj.shape[1], 4)  # plot first 4 action dims
        for d in range(dims):
            ax.plot(traj[:, d], label=f"Dim {d}", alpha=0.8)
        ax.set_ylabel("Action Value")
        ax.set_title(f"{'Successful' if 'success' in label else 'Failed'} Episode", fontsize=12)
        ax.legend(fontsize=8, ncol=dims)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Timestep")
    fig.suptitle("Action Trajectory Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Tables ────────────────────────────────────────────────────────

def table1(df, out="results/table1_main.md"):
    lines = ["| Task | Method | Success Rate (%) | Avg Reward | Latency (ms) | Avg N |",
             "|------|--------|:----------------:|:----------:|:------------:|:-----:|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['task']} | {r['method']} | {r['success_rate']:.1f} "
                    f"| {r['avg_reward']:.1f} | {r['avg_latency_ms']:.0f} "
                    f"| {r.get('avg_samples_per_step', 1):.1f} |")
    text = "\n".join(lines)
    with open(out, "w") as f:
        f.write(text)
    print(f"\n{text}")


def table2(df, out="results/table2_ablation.md"):
    lines = ["| Variant | Success Rate (%) |",
             "|---------|:----------------:|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['variant']} | {r['success_rate']:.1f} |")
    text = "\n".join(lines)
    with open(out, "w") as f:
        f.write(text)
    print(f"\n{text}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="results/benchmark.csv")
    parser.add_argument("--scaling", default="results/ablation_scaling.csv")
    parser.add_argument("--components", default="results/ablation_components.csv")
    args = parser.parse_args()

    if Path(args.benchmark).exists():
        df = load(args.benchmark)
        fig1_main_result(df)
        table1(df)
    else:
        print(f"SKIP: {args.benchmark} not found")

    if Path(args.scaling).exists():
        df = load(args.scaling)
        fig2_scaling(df)
    else:
        print(f"SKIP: {args.scaling} not found")

    if Path(args.components).exists():
        df = load(args.components)
        fig3_ablation(df)
        table2(df)
    else:
        print(f"SKIP: {args.components} not found")


if __name__ == "__main__":
    main()
