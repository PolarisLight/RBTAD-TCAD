from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    }
)


COLORS = {
    "head_bg": "#DDEFE4",
    "tail_bg": "#F7E8C8",
    "head": "#117A8B",
    "tail": "#D8902F",
    "neutral": "#8C8C8C",
    "zero": "#4F4F4F",
    "gain": "#117A8B",
    "drop": "#B75D69",
}


def save_all(fig, name):
    base = OUT / name
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", bbox_inches="tight", dpi=600)
    fig.savefig(f"{base}.tiff", bbox_inches="tight", dpi=600)
    plt.close(fig)


def fig_intro_longtail():
    counts = np.array([46, 28, 19, 15, 11, 9, 8, 7, 6, 5])
    tasks = np.arange(1, 11)

    fig, ax = plt.subplots(figsize=(3.45, 2.05), constrained_layout=True)
    ax.axvspan(0.5, 3.5, color=COLORS["head_bg"], zorder=0)
    ax.axvspan(3.5, 10.5, color=COLORS["tail_bg"], zorder=0)
    ax.bar(
        tasks,
        counts,
        color=[COLORS["head"]] * 3 + [COLORS["tail"]] * 7,
        edgecolor="white",
        linewidth=0.6,
        width=0.72,
    )
    ax.axvline(3.5, color=COLORS["neutral"], lw=0.8, ls="--")
    ax.text(2.0, 49.0, "head", ha="center", va="top", color="#4A4A4A", fontsize=6.5)
    ax.text(7.0, 49.0, "tail", ha="center", va="top", color="#4A4A4A", fontsize=6.5)
    for x, c in zip(tasks, counts):
        ax.text(x, c + 1.2, str(c), ha="center", va="bottom", fontsize=5.8)
    ax.set_xlim(0.35, 10.65)
    ax.set_ylim(0, 52)
    ax.set_xticks(tasks)
    ax.set_xlabel("Task index")
    ax.set_ylabel("Demonstrations")
    ax.set_title("LIBERO-Core-LT task-frequency distribution")
    save_all(fig, "fig1_intro_longtail_distribution")


def fig_corefull_delta():
    baseline = np.array([60, 80, 60, 30, 34, 40, 56, 28, 38, 2], dtype=float)
    tcad = np.array([68, 78, 56, 58, 82, 30, 40, 36, 50, 4], dtype=float)
    delta = tcad - baseline
    tasks = np.arange(1, 11)

    fig = plt.figure(figsize=(5.15, 2.45), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    colors = np.where(delta >= 0, COLORS["gain"], COLORS["drop"])
    ax.axhline(0, color=COLORS["zero"], lw=0.8)
    ax.axvspan(-2, 40, color="#F1F1F1", zorder=0)
    ax.scatter(baseline, delta, c=colors, s=36, edgecolor="white", linewidth=0.5, zorder=3)
    for t, x, y in zip(tasks, baseline, delta):
        ax.text(x + 1.8, y + (1.4 if y >= 0 else -2.6), f"T{t}", fontsize=5.8, va="center")
    ax.text(20, 43, "low-baseline\nregime", ha="center", va="top", fontsize=6.2, color="#555555")
    ax.set_xlim(-2, 86)
    ax.set_ylim(-22, 54)
    ax.set_xlabel("BC task success (%)")
    ax.set_ylabel("TCAD - BC (points)")
    ax.set_title("Task-level gain is largest where BC is weak")

    ax2 = fig.add_subplot(gs[0, 1])
    head_idx = np.array([0, 1, 2])
    hard_idx = np.array([3, 4, 5, 6, 7, 8, 9])
    group_names = ["T1-T3", "T4-T10"]
    group_delta = [delta[head_idx].mean(), delta[hard_idx].mean()]
    ax2.bar(
        [0, 1],
        group_delta,
        color=[COLORS["neutral"], COLORS["gain"]],
        edgecolor="white",
        width=0.58,
    )
    ax2.axhline(0, color=COLORS["zero"], lw=0.8)
    for i, d in enumerate(group_delta):
        ax2.text(i, d + (1.4 if d >= 0 else -2.2), f"{d:+.1f}", ha="center", va="center", fontsize=6.5)
    ax2.set_xticks([0, 1], group_names)
    ax2.set_ylabel("Mean delta (points)")
    ax2.set_ylim(-5, 15)
    ax2.set_title("Grouped effect")

    save_all(fig, "fig3_corefull_delta_analysis")


if __name__ == "__main__":
    fig_intro_longtail()
    fig_corefull_delta()
    print(f"Wrote revised figures to {OUT}")
