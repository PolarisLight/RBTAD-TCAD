from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path(__file__).resolve().parent


def add_box(ax, xy, wh, title, lines, edge, fill):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + 0.03, y + h - 0.08, title, fontsize=11.6, fontweight="bold", va="top")
    ax.text(x + 0.03, y + h - 0.17, "\n".join(lines), fontsize=10.0, va="top", linespacing=1.25)


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )
    fig, ax = plt.subplots(figsize=(13.5, 4.9), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    gray_edge, gray_fill = "#686868", "#f3f3f3"
    teal_edge, teal_fill = "#087982", "#e8f6f5"
    orange_edge, orange_fill = "#c96712", "#fff2e7"
    blue_edge, blue_fill = "#1d5f8f", "#eaf3fb"

    add_box(
        ax,
        (0.035, 0.22),
        (0.255, 0.58),
        "1. Behavior cloning",
        [
            r"Training sample: $(o_t,\ell_i,a_t)$",
            "",
            r"$\mathcal{L}_{BC}=-\log p_\theta(a_t\mid o_t,\ell_i)$",
            "",
            "Learns expert action likelihood",
            "under the given task instruction.",
        ],
        gray_edge,
        gray_fill,
    )

    add_box(
        ax,
        (0.365, 0.18),
        (0.285, 0.66),
        "2. Counterfactual instruction pair",
        [
            r"Keep the same observation and action: $(o_t,a_t)$",
            "",
            r"Correct instruction $\ell_i$",
            r"$s^+=\log p_\theta(a_t\mid o_t,\ell_i)$",
            "",
            r"Wrong-target instruction $\ell_i^-$",
            r"$s^-=\log p_\theta(a_t\mid o_t,\ell_i^-)$",
            "",
            "Negative changes the task target,",
            "not the image or expert action.",
        ],
        teal_edge,
        teal_fill,
    )

    pos_box = FancyBboxPatch(
        (0.39, 0.48),
        0.235,
        0.12,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.2,
        edgecolor=blue_edge,
        facecolor=blue_fill,
    )
    neg_box = FancyBboxPatch(
        (0.39, 0.325),
        0.235,
        0.12,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.2,
        edgecolor=orange_edge,
        facecolor=orange_fill,
    )
    ax.add_patch(pos_box)
    ax.add_patch(neg_box)

    add_box(
        ax,
        (0.71, 0.18),
        (0.265, 0.66),
        "3. Training objective",
        [
            r"$\mathcal{L}=\mathbb{E}[w_i\mathcal{L}_{BC}]$",
            r"$+\lambda[m-(s^+-s^-)]_+$",
            "",
            r"Rare tasks use capped",
            r"positive weight $w_i$.",
            "TCAD applies only to eligible pairs.",
            "",
            "Inference unchanged:",
            "same VLA policy, no generator,",
            "detector, or extra module.",
        ],
        orange_edge,
        orange_fill,
    )

    for start, end in [((0.292, 0.51), (0.36, 0.51)), ((0.652, 0.51), (0.705, 0.51))]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=18,
                linewidth=1.8,
                color="#555555",
            )
        )

    fig.suptitle(
        "RBTAD: rare-balanced task-conditioned action discrimination",
        fontsize=14,
        fontweight="bold",
        y=0.96,
    )
    ax.text(
        0.5,
        0.055,
        "Training-only regularization: compare correct and plausible wrong instructions for the same observation/action pair; keep inference cost unchanged.",
        ha="center",
        fontsize=10.5,
        color="#333333",
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"fig2_method_overview.{ext}", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main()
