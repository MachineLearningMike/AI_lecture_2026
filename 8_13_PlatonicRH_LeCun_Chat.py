"""
Platonic Representation Hypothesis (PRH) with JEPAs

Paper read: https://arxiv.org/pdf/2610.39302.pdf
Title: "Proving and Improving the Platonic Representation Hypothesis with JEPAs"
Authors: Randall Balestriero, Yann LeCun, et al.

Lecture arc:
  1. What is the PRH?
  2. Why JEPAs are the right formalism
  3. The central theorem: good views make modalities agree up to rotation
  4. Bad views, bad positives, and next-token prediction
  5. Sample efficiency and the scaling law of alignment
  6. Takeaways for representation learning
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_AXIS,
    FS_TICK,
    FS_ANNOT,
    FS_LEGEND,
    FIGURE_DPI,
    SCREEN_FRACTION,
    apply_rcparams,
    add_step_footer,
    run_slide_deck,
)

apply_rcparams(plt)

np.random.seed(13)
RNG = np.random.default_rng(13)

C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_OLD = "#fce4d6"
C_FUN = "#6c3483"
C_WARN = "#f9e79f"
C_GRAY = "#b0b0b0"


def screen_figsize(fraction=SCREEN_FRACTION, dpi=FIGURE_DPI):
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return (w / dpi * fraction, h / dpi * fraction)
    except Exception:
        return (16 * fraction / 0.95, 9 * fraction / 0.95)


def new_figure(*args, **kwargs):
    kwargs.setdefault("figsize", screen_figsize())
    kwargs.setdefault("dpi", FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


def blank(ax, xlim=(0, 12), ylim=(0, 8)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(rect)
    return rect


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=2.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=18),
    )


def make_world_plot(ax, title, color="tab:blue"):
    z = RNG.normal(size=(300, 2))
    z[::2] += np.array([1.5, 0.5])
    z[1::2] += np.array([-1.2, 0.8])
    ax.scatter(z[:, 0], z[:, 1], s=12, c=color, alpha=0.75)
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


# -------------------------------------------------------
# Step 1: the question
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle("Step 1: Do different modalities converge to one world model?", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank(ax)

    round_box(ax, 0.5, 5.6, 3.4, 1.4, C_NEW, C_DARK, lw=1.6)
    ax.text(2.2, 6.2, "Vision", ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK)
    ax.text(2.2, 5.9, "photographs of a scene", ha="center", fontsize=FS_TICK)

    round_box(ax, 8.1, 5.6, 3.4, 1.4, C_OLD, C_ACCENT, lw=1.6)
    ax.text(9.8, 6.2, "Language", ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_ACCENT)
    ax.text(9.8, 5.9, "captions describing it", ha="center", fontsize=FS_TICK)

    round_box(ax, 3.5, 2.2, 5.0, 1.8, C_GOAL, C_OK, lw=1.8)
    ax.text(6.0, 3.2, "Platonic Representation Hypothesis (PRH)", ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#145a32")
    ax.text(6.0, 2.8, "As models scale, different modalities may converge to the same latent picture of reality.", ha="center", fontsize=FS_TICK, color="0.2")

    arrow(ax, 3.9, 6.15, 8.1, 6.15, C_DARK, 2.2)
    ax.text(6.0, 6.65, "same underlying world state z", ha="center", fontsize=FS_TICK, fontweight="bold")

    ax.text(
        6.0, 0.75,
        "The paper asks: when do separately trained encoders provably agree, up to rotation?",
        ha="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )
    return fig


# -------------------------------------------------------
# Step 2: the formal object
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle("Step 2: JEPAs make the question precise", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank(ax)

    round_box(ax, 0.6, 5.6, 10.8, 1.35, C_DARK, C_DARK, lw=0)
    ax.text(6.0, 6.2, "Joint-Embedding Predictive Architecture (JEPA)", ha="center", va="center", fontsize=FS_ANNOT + 2, color="#f9e79f", fontweight="bold")

    round_box(ax, 0.8, 2.0, 10.4, 3.0, C_MUTE, "0.45", lw=1.7)
    ax.text(
        6.0, 4.0,
        r"$L_m(f) = \mathbb{E}_{(z,z')\sim \pi_m} \ell(f(g_m(z)), f(g_m(z'))) + \Omega(f)$"
        "\n\n"
        r"Positive pairs $(g_m(z), g_m(z'))$ are treated as two views of the same world state, "
        r"and the anti-collapse term $\Omega$ keeps the representation from degenerating.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax, 0.8, 0.5, 4.6, 1.35, C_GOAL, C_OK, lw=1.6)
    ax.text(3.1, 1.15, "Good views", ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#145a32")
    ax.text(3.1, 0.8, "two observations of one scene", ha="center", fontsize=FS_TICK)

    round_box(ax, 6.9, 0.5, 4.6, 1.35, C_OLD, C_ACCENT, lw=1.6)
    ax.text(9.1, 1.15, "Bad views", ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color=C_ACCENT)
    ax.text(9.1, 0.8, "corruptions of one observation", ha="center", fontsize=FS_TICK)

    ax.text(6.0, 1.8, "The paper proves: good views make the solution unique up to rotation; bad views only partially align.", ha="center", fontsize=FS_TICK, fontweight="bold")
    return fig


# -------------------------------------------------------
# Step 3: key theorem (Platonic Representation Theorem)
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle("Step 3: The main theorem: shared good positives make worlds align", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank(ax)

    round_box(ax, 0.45, 6.2, 11.1, 1.3, C_DARK, C_DARK, lw=0)
    ax.text(6.0, 6.8, r"Theorem 1: If several modalities share the same positive-pair structure, their encoders agree up to one rotation.", ha="center", va="center", fontsize=FS_ANNOT + 1, color="#f9e79f", fontweight="bold")

    round_box(ax, 0.7, 2.7, 10.6, 2.8, C_MUTE, "0.45", lw=1.8)
    ax.text(
        6.0, 4.55,
        r"$f_m^* \circ g_m = Q_m h^*$  for some $Q_m \in O(k)$"
        "\n\n"
        r"Therefore, $f_1^* \approx Q_{12} f_2^*$ after a single orthogonal transform."
        "\n"
        r"The world representation $h^*$ is shared; only the coordinate system differs.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )

    round_box(ax, 0.7, 0.45, 2.8, 1.5, C_NEW, C_DARK, lw=1.7)
    ax.text(2.1, 1.15, "image world", ha="center", fontsize=FS_TICK + 1, fontweight="bold")
    ax.text(2.1, 0.75, "pixels & geometry", ha="center", fontsize=FS_TICK)

    round_box(ax, 4.7, 0.45, 2.8, 1.5, C_OLD, C_ACCENT, lw=1.7)
    ax.text(6.1, 1.15, "text world", ha="center", fontsize=FS_TICK + 1, fontweight="bold")
    ax.text(6.1, 0.75, "tokens & semantics", ha="center", fontsize=FS_TICK)

    round_box(ax, 8.7, 0.45, 2.8, 1.5, C_GOAL, C_OK, lw=1.7)
    ax.text(10.1, 1.15, "shared latent", ha="center", fontsize=FS_TICK + 1, fontweight="bold", color="#145a32")
    ax.text(10.1, 0.75, r"$h^*(z)$", ha="center", fontsize=FS_TICK + 2, fontweight="bold")

    arrow(ax, 3.5, 1.15, 4.7, 1.15, C_DARK, 2.0)
    arrow(ax, 7.5, 1.15, 8.7, 1.15, C_DARK, 2.0)
    ax.text(6.0, 2.15, "same underlying scene, different observables", ha="center", fontsize=FS_TICK, fontweight="bold")
    return fig


# -------------------------------------------------------
# Step 4: good-view vs bad-view JAEPAs
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle("Step 4: A good view is not a corrupted view", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank(ax)

    round_box(ax, 0.7, 4.9, 4.9, 2.1, C_GOAL, C_OK, lw=1.8)
    ax.text(3.15, 6.2, "Good-view JEPA", ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#145a32")
    ax.text(3.15, 5.5, "two photos of the same scene", ha="center", fontsize=FS_TICK)
    ax.text(3.15, 5.0, "or two captions describing the same object", ha="center", fontsize=FS_TICK)
    ax.text(3.15, 4.4, "The encoder learns the scene, not the nuisance.", ha="center", fontsize=FS_TICK, fontweight="bold")

    round_box(ax, 6.4, 4.9, 4.9, 2.1, C_OLD, C_ACCENT, lw=1.8)
    ax.text(9.05, 6.2, "Bad-view JEPA", ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_ACCENT)
    ax.text(9.05, 5.5, "two crops of the same image", ha="center", fontsize=FS_TICK)
    ax.text(9.05, 5.0, "or a prefix + continuation in one document", ha="center", fontsize=FS_TICK)
    ax.text(9.05, 4.4, "The encoder spends dimensions on nuisance factors.", ha="center", fontsize=FS_TICK, fontweight="bold")

    ax.text(
        6.0, 2.7,
        "The crucial distinction is whether positives are linked through the world state or through one single observation.",
        ha="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )

    ax.text(
        6.0, 1.0,
        r"This is why the paper argues that next-token prediction is a bad-view JEPA: prefix and completion share a document, but not a full world state.",
        ha="center", fontsize=FS_TICK, fontweight="bold", color="0.2",
    )
    return fig


# -------------------------------------------------------
# Step 5: theory of misalignment
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle("Step 5: What if positives are not perfectly matched?", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank(ax)

    round_box(ax, 0.55, 5.8, 10.9, 1.4, C_DARK, C_DARK, lw=0)
    ax.text(6.0, 6.45, r"Theorem 2: kernel mismatch bounds the residual misalignment", ha="center", va="center", fontsize=FS_ANNOT + 2, color="#f9e79f", fontweight="bold")

    round_box(ax, 0.8, 2.3, 10.4, 2.7, C_MUTE, "0.45", lw=1.7)
    ax.text(
        6.0, 4.0,
        r"$\min_{Q\in O(k)} \|\Phi_1(z) - Q\Phi_2(z)\|^2 \leq \|K_{\pi_1}-K_{\pi_2}\|_F^2 / \lambda_k$"
        "\n\n"
        r"The smaller the mismatch between the two positive-pair kernels, the smaller the alignment gap."
        "\n"
        r"In other words: the same world is aligned only if the two JEPAs' positives relate the same structure.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    x = np.logspace(-4, 3, 200)
    y = 0.18 * x
    ax.plot(x, y, color=C_ACCENT, lw=2.5, alpha=0.9)
    ax.scatter([0.01, 0.1, 1.0, 10.0], [0.0018, 0.018, 0.18, 1.8], s=30, color=C_OK, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-4, 1e3)
    ax.set_ylim(1e-4, 1e3)
    ax.set_xlabel("kernel mismatch", fontsize=FS_AXIS)
    ax.set_ylabel("alignment error", fontsize=FS_AXIS)
    ax.set_title("Residual error rises with mismatch", fontsize=FS_AX_TITLE)
    ax.grid(alpha=0.15)
    ax.text(0.002, 0.0007, "good overlap", fontsize=FS_TICK, color=C_OK, fontweight="bold")
    ax.text(20, 50, "bad overlap", fontsize=FS_TICK, color=C_ACCENT, fontweight="bold")
    return fig


# -------------------------------------------------------
# Step 6: scaling and sample efficiency
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle("Step 6: The paper’s empirical punchline: alignment with far less scale", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank(ax)

    # Approximate scaling-law curves from the paper.
    x = np.array([2e7, 5e7, 1e8, 3e8, 9e8])
    good = np.array([0.058, 0.082, 0.109, 0.152, 0.184])
    bad = np.array([0.028, 0.034, 0.042, 0.072, 0.111])

    ax.plot(x, good, color=C_OK, lw=3, marker="o", markersize=7, label="good-view JEPA")
    ax.plot(x, bad, color=C_ACCENT, lw=3, marker="s", markersize=7, label="next-token / bad-view")
    ax.set_xscale("log")
    ax.set_xlim(1.5e7, 1.5e9)
    ax.set_ylim(0.02, 0.2)
    ax.set_xlabel("training samples seen", fontsize=FS_AXIS)
    ax.set_ylabel("alignment with DINOv2", fontsize=FS_AXIS)
    ax.set_title("Sample-efficiency advantage of good views", fontsize=FS_AX_TITLE)
    ax.legend(frameon=True, loc="upper right", fontsize=FS_LEGEND)

    ax.text(2.3e7, 0.102, "21.6M", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax.text(2.3e7, 0.052, "matches bad-view at 900M", fontsize=FS_TICK, color=C_ACCENT, fontweight="bold")
    ax.text(4e8, 0.19, "0.184", fontsize=FS_TICK + 1, color=C_OK, fontweight="bold")
    ax.text(4e8, 0.13, "above Qwen2.5-72B", fontsize=FS_TICK, color=C_DARK, fontweight="bold")
    return fig


# -------------------------------------------------------
# Step 7: role of LeCun
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle("Step 7: What the paper adds to the modern AI story", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank(ax)

    round_box(ax, 0.7, 5.2, 3.8, 1.5, C_NEW, C_DARK, lw=1.8)
    ax.text(2.6, 6.0, "Key idea", ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK)
    ax.text(2.6, 5.5, "PLATONIC = shared positive pairs\nnot just more scale", ha="center", fontsize=FS_TICK)

    round_box(ax, 4.25, 5.2, 3.5, 1.5, C_GOAL, C_OK, lw=1.8)
    ax.text(6.0, 6.0, "Theory", ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#145a32")
    ax.text(6.0, 5.5, "good-view JEPA\nlearns world geometry", ha="center", fontsize=FS_TICK)

    round_box(ax, 8.1, 5.2, 3.2, 1.5, C_OLD, C_ACCENT, lw=1.8)
    ax.text(9.7, 6.0, "Practice", ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_ACCENT)
    ax.text(9.7, 5.5, "alignment with\nfar fewer samples", ha="center", fontsize=FS_TICK)

    round_box(ax, 1.0, 1.2, 10.0, 2.5, C_MUTE, "0.45", lw=1.8)
    ax.text(
        6.0, 2.8,
        "Takeaway: the PRH is not only a conjecture anymore. In the JEPA setting, it becomes a theorem when positives are world-consistent,\nwhile bad positives and next-token prediction explain why alignment is weak or slow. The paper turns the question into a recipe: build better positive views, not just bigger nets.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )
    return fig


# -------------------------------------------------------
# Main deck
# -------------------------------------------------------
if __name__ == "__main__":
    builders = [
        build_step_1,
        build_step_2,
        build_step_3,
        build_step_4,
        build_step_5,
        build_step_6,
        build_step_7,
    ]
    run_slide_deck(builders)
