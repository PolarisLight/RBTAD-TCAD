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
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    }
)


COLORS = {
    "neutral": "#8C8C8C",
    "neutral_dark": "#545454",
    "baseline": "#A8A8A8",
    "resampling": "#C7C7C7",
    "apa": "#D8902F",
    "ours": "#117A8B",
    "ours_light": "#8FD0D8",
    "tail_bg": "#F7E8C8",
    "head_bg": "#DDEFE4",
    "bad": "#B75D69",
    "good": "#4F8A63",
}


def panel_label(ax, label):
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=8,
        va="top",
        ha="left",
    )


def save_all(fig, name):
    base = OUT / name
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", bbox_inches="tight", dpi=600)
    fig.savefig(f"{base}.tiff", bbox_inches="tight", dpi=600)
    plt.close(fig)


def fig_problem_diagnostic():
    counts = np.array([46, 28, 19, 15, 11, 9, 8, 7, 6, 5])
    tasks = np.arange(1, 11)
    pos_rates = np.array([35.5, 18.2])
    non_pos = 100 - pos_rates
    groups = ["All states\nn=200", "Relation-tail\nn=44"]
    mean_margin = [-0.4787, -1.0950]

    fig = plt.figure(figsize=(7.1, 2.55), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.24)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axvspan(0.5, 3.5, color=COLORS["head_bg"], zorder=0)
    ax0.axvspan(3.5, 10.5, color=COLORS["tail_bg"], zorder=0)
    bar_colors = [COLORS["ours"]] * 3 + [COLORS["apa"]] * 7
    ax0.bar(tasks, counts, color=bar_colors, edgecolor="white", linewidth=0.7)
    ax0.axvline(3.5, color="#777777", lw=0.8, ls="--")
    ax0.text(2.0, 48.5, "head tasks", ha="center", va="top", color=COLORS["neutral_dark"])
    ax0.text(7.0, 48.5, "tail tasks", ha="center", va="top", color=COLORS["neutral_dark"])
    for x, c in zip(tasks, counts):
        ax0.text(x, c + 1.2, str(c), ha="center", va="bottom", fontsize=5.8)
    ax0.set_xlim(0.35, 10.65)
    ax0.set_ylim(0, 52)
    ax0.set_xticks(tasks)
    ax0.set_xlabel("LIBERO-Core-LT task index")
    ax0.set_ylabel("Demonstrations")
    ax0.set_title("Long-tail simulation split")
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(groups))
    ax1.barh(y, non_pos, color=COLORS["bad"], edgecolor="white", label="Non-positive margin")
    ax1.barh(y, pos_rates, left=non_pos, color=COLORS["good"], edgecolor="white", label="Positive margin")
    for yi, pr, mr in zip(y, pos_rates, mean_margin):
        ax1.text(100.8, yi, f"{pr:.1f}% positive\nmean {mr:.3f}", va="center", fontsize=6.2)
    ax1.set_yticks(y, groups)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 126)
    ax1.set_xlabel("Instruction-swap samples (%)")
    ax1.set_title("Baseline often prefers no correct task boundary")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.49, -0.36), ncol=2, frameon=False)
    panel_label(ax1, "b")

    save_all(fig, "fig1_problem_diagnostic")


def fig_main_results():
    rows = [
        ("BC", "Original distribution", 26.5),
        ("Re-sampling", "q = 0.75", 25.1),
        ("Re-sampling", "q = 0.50", 25.1),
        ("Re-sampling", "q = 0.25", 27.1),
        ("APA ablation", "Formatting only", 26.0),
        ("APA ablation", "Augmentation only", 26.9),
        ("APA", "Formatting + augmentation", 36.1),
        ("Ours", "RBTAD", 40.0),
    ]
    labels = [f"{family}: {method}" for family, method, _ in rows]
    values = np.array([v for _, _, v in rows])
    colors = []
    for family, _, _ in rows:
        if family == "Ours":
            colors.append(COLORS["ours"])
        elif family == "APA":
            colors.append(COLORS["apa"])
        elif family == "APA ablation":
            colors.append("#E4BE81")
        elif family == "Re-sampling":
            colors.append(COLORS["resampling"])
        else:
            colors.append(COLORS["baseline"])

    fig, ax = plt.subplots(figsize=(7.1, 3.15), constrained_layout=True)
    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, edgecolor="white", linewidth=0.7)
    ax.axvline(26.5, color=COLORS["neutral_dark"], lw=0.9, ls=":", label="BC")
    ax.axvline(36.1, color=COLORS["apa"], lw=0.9, ls="--", label="APA")
    for yi, value in zip(y, values):
        weight = "bold" if value == values.max() else "normal"
        ax.text(value + 0.7, yi, f"{value:.1f}", va="center", fontsize=6.5, fontweight=weight)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 45)
    ax.set_xlabel("Average success rate on LIBERO-Core-LT (%)")
    ax.set_title("RBTAD improves over the reported baseline family without generated trajectories")
    ax.legend(loc="lower right", frameon=False, ncol=2)
    panel_label(ax, "a")
    save_all(fig, "fig4_main_results")


def fig_controlled_corefull():
    tasks = np.arange(1, 11)
    baseline = np.array([0.60, 0.80, 0.60, 0.30, 0.34, 0.40, 0.56, 0.28, 0.38, 0.02]) * 100
    tcad = np.array([0.68, 0.78, 0.56, 0.58, 0.82, 0.30, 0.40, 0.36, 0.50, 0.04]) * 100

    fig = plt.figure(figsize=(7.1, 2.75), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.75, 1.65], wspace=0.26)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar([0, 1], [43.0, 50.0], width=0.62, color=[COLORS["baseline"], COLORS["ours"]], edgecolor="white")
    ax0.text(0, 44.2, "43.0", ha="center", va="bottom", fontsize=6.6)
    ax0.text(1, 51.2, "50.0", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax0.set_xticks([0, 1], ["BC", "TCAD"])
    ax0.set_ylabel("Average success rate (%)")
    ax0.set_ylim(0, 60)
    ax0.set_title("Within-pipeline control")
    panel_label(ax0, "a")

    ax1 = fig.add_subplot(gs[0, 1])
    for x, b, t in zip(tasks, baseline, tcad):
        color = COLORS["ours"] if t >= b else COLORS["neutral"]
        ax1.plot([x, x], [b, t], color=color, lw=1.0, alpha=0.85)
    ax1.scatter(tasks - 0.06, baseline, s=22, color=COLORS["baseline"], edgecolor="white", linewidth=0.4, label="BC")
    ax1.scatter(tasks + 0.06, tcad, s=26, color=COLORS["ours"], edgecolor="white", linewidth=0.4, label="TCAD")
    for x, b, t in zip(tasks, baseline, tcad):
        delta = t - b
        if abs(delta) >= 10:
            ax1.text(x + 0.15, max(b, t) + 3.0, f"{delta:+.0f}", fontsize=5.8, ha="center")
    ax1.set_xlim(0.4, 10.6)
    ax1.set_ylim(0, 90)
    ax1.set_xticks(tasks)
    ax1.set_xlabel("LIBERO-Core-Full task index")
    ax1.set_ylabel("Success rate (%)")
    ax1.set_title("Per-task changes are concentrated in difficult tasks")
    ax1.legend(loc="upper right", frameon=False, ncol=2)
    panel_label(ax1, "b")

    save_all(fig, "fig5_corefull_control")


def fig_iterations():
    rows = [
        ("Tail-only\nrescue", 34.0, "post-hoc"),
        ("RCTAD", 35.0, "training"),
        ("Relation-anchor\nnegatives", 37.0, "training"),
        ("RBTAD", 40.0, "training"),
        ("Selective projector\nmerge", 42.0, "diagnostic"),
    ]
    labels = [r[0] for r in rows]
    values = np.array([r[1] for r in rows])
    types = [r[2] for r in rows]
    colors = [
        COLORS["ours"] if lab == "RBTAD" else "#B8C8CC" if typ == "training" else "#D8D8D8"
        for lab, typ in zip(labels, types)
    ]
    colors[-1] = "#C9A56B"

    fig, ax = plt.subplots(figsize=(5.2, 2.55), constrained_layout=True)
    x = np.arange(len(rows))
    ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.7)
    ax.axhline(36.1, color=COLORS["apa"], lw=0.9, ls="--", label="APA reported")
    for xi, val, typ in zip(x, values, types):
        ax.text(xi, val + 0.7, f"{val:.0f}", ha="center", va="bottom", fontsize=6.2)
        if typ == "diagnostic":
            ax.text(xi, 2.2, "diagnostic\nnot main", ha="center", va="bottom", fontsize=5.8, color=COLORS["neutral_dark"])
    ax.set_ylim(0, 47)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Internal iterations motivate the end-to-end RBTAD choice")
    ax.legend(loc="upper left", frameon=False)
    panel_label(ax, "a")
    save_all(fig, "figS_iterations")


def main():
    fig_problem_diagnostic()
    fig_main_results()
    fig_controlled_corefull()
    fig_iterations()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
