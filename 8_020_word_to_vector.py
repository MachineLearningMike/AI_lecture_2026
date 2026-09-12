"""
How to train word-to-vector models (Word2Vec-style).

 builds the intuition ("company it keeps").
This lecture is the training recipe: windows -> pairs -> loss -> SGD.

Sketch:
  1. Goal -- learn a vector for every word type
  2. Training data -- sliding context windows
  3. Skip-gram -- predict context from center (formula)
  4. CBOW -- predict center from context (formula)
  5. Full softmax is expensive -- why
  6. Negative sampling loss (practical Word2Vec)
  7. Visual intuition -- window becomes (center, context) pairs
  8. Visual intuition -- one update moves vectors
  9. Training loop + knobs
  10. GloVe glance -- count matrix factorization cousin
  11. Static vectors vs contextual (BERT / GPT bridge)
  12. Takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
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
from matplotlib.patches import FancyBboxPatch
import numpy as np

np.random.seed(941)
RNG = np.random.default_rng(941)


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


def show_maximized(fig):
    fig.set_dpi(FIGURE_DPI)
    fig.set_size_inches(screen_figsize(), forward=True)
    fig.canvas.draw_idle()
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        try:
            manager.window.state("zoomed")
        except Exception:
            try:
                manager.window.showMaximized()
            except Exception:
                try:
                    manager.frame.Maximize(True)
                except Exception:
                    pass
    add_step_footer(fig)
    plt.show()


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


SENTENCE = ["the", "cat", "sat", "on", "the", "mat"]
VOCAB = ["the", "cat", "sat", "on", "mat", "dog", "ran"]


# -------------------------------------------------------
# FIGURE 1: Goal
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 1: Goal -- train a dense vector for every word type",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax1)

    round_box(ax1, 0.4, 6.45, 11.2, 1.25, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 7.05,
        r"Input: raw text corpus.   Output: embedding table $E\in\mathbb{R}^{|V|\times d}$.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax1, 0.45, 3.4, 3.6, 2.7, C_MUTE, "0.45", lw=1.5)
    ax1.text(2.25, 5.7, "Before", ha="center", fontsize=FS_TICK, fontweight="bold")
    ax1.text(
        2.25, 4.4,
        "word = discrete ID\n"
        "cat ≠ dog in any\n"
        "geometric sense\n\n"
        "one-hot / dictionary",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax1, 4.3, 3.4, 3.4, 2.7, C_WARN, "#b7950b", lw=1.6)
    ax1.text(6.0, 5.7, "Train", ha="center", fontsize=FS_TICK, fontweight="bold", color="#7d6608")
    ax1.text(
        6.0, 4.4,
        "predict a word\n"
        "from its neighbors\n"
        "(or neighbors from it)\n\n"
        "self-supervised",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax1, 7.95, 3.4, 3.6, 2.7, C_GOAL, C_OK, lw=1.8)
    ax1.text(9.75, 5.7, "After", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax1.text(
        9.75, 4.4,
        r"$v_{\mathrm{cat}}\approx v_{\mathrm{dog}}$"
        "\n"
        "in direction / neighbors"
        "\n\n"
        "usable geometry",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax1.text(
        6.0, 1.7,
        "Mikolov et al. (2013): Word2Vec — efficient training of word embeddings.\n"
        "This lecture: the data, the losses, and the practical negative-sampling trick.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_NEW, edgecolor=C_DARK, alpha=0.95),
    )
    ax1.text(
        6.0, 0.45,
        "Intuition playground: 0790_word_vector.py   |   embedding table idea: 0780_Embedding.py",
        ha="center", fontsize=FS_TICK, style="italic", color="0.35",
    )
    return fig1


# -------------------------------------------------------
# FIGURE 2: Sliding windows
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        r"Step 2: Training data -- a sliding window of width $2c$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax2)

    round_box(ax2, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax2.text(
        6.0, 7.1,
        r"For each center word $w_t$, context = $\{w_{t-c},\ldots,w_{t-1},w_{t+1},\ldots,w_{t+c}\}$.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    # sentence tokens
    xs = np.linspace(1.0, 11.0, len(SENTENCE))
    center_i = 2  # "sat"
    for i, (x, tok) in enumerate(zip(xs, SENTENCE)):
        face = C_ACCENT if i == center_i else (C_GOAL if abs(i - center_i) <= 2 and i != center_i else C_NEW)
        edge = C_ACCENT if i == center_i else "0.4"
        round_box(ax2, x - 0.7, 4.5, 1.4, 1.2, face, edge, lw=1.8)
        ax2.text(x, 5.1, tok, ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
                 color="white" if i == center_i else C_DARK)

    ax2.text(xs[center_i], 4.1, "center", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
    ax2.annotate(
        "", xy=(xs[0] - 0.2, 4.35), xytext=(xs[4] + 0.2, 4.35),
        arrowprops=dict(arrowstyle="<->", color=C_OK, lw=2.0),
    )
    ax2.text(6.0, 3.7, r"context window ($c=2$ here)", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_OK)

    round_box(ax2, 0.5, 0.5, 11.0, 2.8, C_MUTE, "0.45", lw=1.5)
    ax2.text(
        6.0, 1.9,
        "Corpus pass = many overlapping windows.\n"
        r"Skip-gram emits pairs $(w_t, w_{t+j})$ for each context offset $j\neq 0$."
        "\n"
        r"CBOW uses the bag $\{w_{t+j}\}_{j\neq 0}$ to predict $w_t$."
        "\n\n"
        "No human labels — the text supervises itself.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig2


# -------------------------------------------------------
# FIGURE 3: Skip-gram
# -------------------------------------------------------
def build_step_3():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        "Step 3: Skip-gram -- center word predicts each context word",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax3)

    round_box(ax3, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax3.text(
        6.0, 7.0,
        r"$\max \; \sum_t \sum_{0<|j|\leq c}"
        r"\log p(w_{t+j}\mid w_t)$",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#f9e79f",
    )

    round_box(ax3, 0.5, 3.2, 11.0, 2.85, C_MUTE, "0.45", lw=1.6)
    ax3.text(
        6.0, 4.6,
        r"With vectors $v_w$ (input / center) and $u_w$ (output / context):"
        "\n\n"
        r"$p(w_O\mid w_I)"
        r"= \dfrac{\exp(u_{w_O}^\top v_{w_I})}"
        r"{\sum_{w=1}^{|V|}\exp(u_w^\top v_{w_I})}$"
        "\n\n"
        "Score = dot product; normalize with softmax over the whole vocabulary.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax3, 0.45, 0.45, 5.4, 2.45, C_GOAL, C_OK, lw=1.6)
    ax3.text(
        3.15, 1.65,
        "Per positive pair $(w_I,w_O)$:\n"
        r"$L = -\log p(w_O\mid w_I)$"
        "\n\n"
        "Push $u_{w_O}$ toward $v_{w_I}$.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#145a32",
    )

    round_box(ax3, 6.15, 0.45, 5.4, 2.45, C_OLD, C_ACCENT, lw=1.6)
    ax3.text(
        8.85, 1.65,
        "Two tables are common:\n"
        r"$V$ (center) and $U$ (context)."
        "\n\n"
        "Often keep $V$ (or average)\n"
        "as the published embedding.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig3


# -------------------------------------------------------
# FIGURE 4: CBOW
# -------------------------------------------------------
def build_step_4():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        "Step 4: CBOW -- context bag predicts the center word",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax4)

    round_box(ax4, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax4.text(
        6.0, 7.0,
        r"$\max \; \sum_t \log p\!\left(w_t\mid \{w_{t+j}\}_{0<|j|\leq c}\right)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax4, 0.5, 3.35, 11.0, 2.7, C_MUTE, "0.45", lw=1.6)
    ax4.text(
        6.0, 4.7,
        r"Context vector (mean of input embeddings):"
        "\n"
        r"$\bar v_t = \dfrac{1}{2c}\sum_{0<|j|\leq c} v_{w_{t+j}}$"
        "\n\n"
        r"$p(w_t\mid \mathrm{context})"
        r"= \dfrac{\exp(u_{w_t}^\top \bar v_t)}"
        r"{\sum_{w}\exp(u_w^\top \bar v_t)}$",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax4, 0.45, 0.45, 5.4, 2.55, C_NEW, C_DARK, lw=1.6)
    ax4.text(3.15, 2.5, "CBOW vibe", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax4.text(
        3.15, 1.4,
        "Faster (one prediction\n"
        "per window).\n"
        "Good on frequent words.\n"
        "Smoothes context.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax4, 6.15, 0.45, 5.4, 2.55, "#f5eef8", C_FUN, lw=1.6)
    ax4.text(8.85, 2.5, "Skip-gram vibe", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_FUN)
    ax4.text(
        8.85, 1.4,
        "More pairs / window.\n"
        "Often stronger on\n"
        "rare words &\n"
        "analogies.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig4


# -------------------------------------------------------
# FIGURE 5: Softmax cost
# -------------------------------------------------------
def build_step_5():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        "Step 5: Full softmax denominator is the bottleneck",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax5)

    round_box(ax5, 0.4, 5.6, 11.2, 2.05, C_MUTE, "0.45", lw=1.6)
    ax5.text(
        6.0, 6.6,
        r"Every positive pair needs "
        r"$\sum_{w=1}^{|V|}\exp(u_w^\top v)$"
        "\n\n"
        r"$|V|$ can be $10^5$–$10^6$.  That sum every step is too slow.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    round_box(ax5, 0.45, 2.7, 5.4, 2.55, C_OLD, C_ACCENT, lw=1.6)
    ax5.text(3.15, 4.8, "Historical fixes", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
    ax5.text(
        3.15, 3.6,
        "• Hierarchical softmax\n"
        "  (binary tree over V)\n"
        "• Negative sampling\n"
        "  (the practical default)\n"
        "• Sampled softmax",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax5, 6.15, 2.7, 5.4, 2.55, C_GOAL, C_OK, lw=1.8)
    ax5.text(8.85, 4.8, "Training idea shift", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax5.text(
        8.85, 3.6,
        "Do not normalize over all V.\n"
        "Classify: true neighbor\n"
        "vs a few noise words.\n\n"
        "Logistic / BCE style loss.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax5.text(
        6.0, 1.2,
        "Next slide: the negative-sampling objective actually used in Word2Vec.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    return fig5


# -------------------------------------------------------
# FIGURE 6: Negative sampling
# -------------------------------------------------------
def build_step_6():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        "Step 6: Negative sampling -- the practical Word2Vec loss",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax6)

    round_box(ax6, 0.4, 6.25, 11.2, 1.45, C_DARK, C_DARK, lw=0)
    ax6.text(
        6.0, 6.95,
        r"$L_{\mathrm{NEG}}"
        r"= -\log\sigma(u_{w_O}^\top v_{w_I})"
        r" - \sum_{k=1}^{K}\log\sigma(-u_{w_k}^\top v_{w_I})$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax6, 0.45, 3.35, 5.5, 2.6, C_GOAL, C_OK, lw=1.6)
    ax6.text(3.2, 5.55, "Positive term", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax6.text(
        3.2, 4.3,
        r"$\sigma(u_{w_O}^\top v_{w_I})\to 1$"
        "\n\n"
        "True context word:\n"
        "raise the dot product.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax6, 6.15, 3.35, 5.4, 2.6, C_OLD, C_ACCENT, lw=1.6)
    ax6.text(8.85, 5.55, "Negative terms", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
    ax6.text(
        8.85, 4.3,
        r"$w_k\sim P_n(w)$  (noise)"
        "\n\n"
        r"Push $u_{w_k}^\top v_{w_I}$ down."
        "\n"
        r"Typical $K=5$–$20$.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax6.text(
        6.0, 1.7,
        r"Noise distribution: often $P_n(w)\propto f(w)^{3/4}$ (unigram dampened)."
        "\n"
        r"$\sigma(x)=1/(1+e^{-x})$.  Minimize $L_{\mathrm{NEG}}$ with SGD / Adagrad."
        "\n\n"
        "Same spirit for CBOW: replace $v_{w_I}$ by the context mean $\\bar v$.",
        ha="center", fontsize=FS_TICK + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )
    return fig6


# -------------------------------------------------------
# FIGURE 7: Visual — pairs from window
# -------------------------------------------------------
def build_step_7():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        "Step 7: Visual intuition -- one window becomes training pairs",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax7)

    round_box(ax7, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax7.text(
        6.0, 7.1,
        r'Window: [the, cat, sat, on, the] with center "sat", $c=2$',
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    pairs = [
        ("sat", "the"),
        ("sat", "cat"),
        ("sat", "on"),
        ("sat", "the"),
    ]
    for i, (a, b) in enumerate(pairs):
        x = 0.7 + i * 2.85
        round_box(ax7, x, 4.0, 2.6, 2.0, C_NEW, C_DARK, lw=1.5)
        ax7.text(x + 1.3, 5.35, "skip-gram pair", ha="center",
                 fontsize=FS_TICK - 1, color="0.35")
        ax7.text(x + 1.3, 4.75, rf"$({a},\;{b})$", ha="center",
                 fontsize=FS_ANNOT, fontweight="bold")

    round_box(ax7, 0.5, 0.5, 11.0, 3.1, C_MUTE, "0.45", lw=1.5)
    ax7.text(
        6.0, 2.05,
        "For each pair, sample $K$ negatives, e.g. dog, ran, mat, ...\n"
        "One SGD step updates $v_{\\mathrm{sat}}$ and the $u$'s of the positive/negative words.\n\n"
        "Millions of such micro-tasks → geometry where co-occurring words align.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig7


# -------------------------------------------------------
# FIGURE 8: Visual — vectors move
# -------------------------------------------------------
def build_step_8():
    fig8, (ax8a, ax8b) = new_figure(1, 2)
    fig8.suptitle(
        "Step 8: Visual intuition -- positive pull, negative push",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # before
    ax8a.set_xlim(-2.5, 2.5)
    ax8a.set_ylim(-2.2, 2.2)
    ax8a.set_aspect("equal")
    ax8a.set_title("Before update", fontsize=FS_AX_TITLE - 2)
    pts = {
        "sat": np.array([0.1, 0.2]),
        "cat": np.array([-1.2, 0.9]),
        "on": np.array([1.0, 0.8]),
        "dog": np.array([-0.2, -1.4]),
        "ran": np.array([1.5, -1.0]),
    }
    colors = {"sat": C_ACCENT, "cat": C_OK, "on": C_OK, "dog": C_FUN, "ran": C_FUN}
    for w, p in pts.items():
        ax8a.scatter([p[0]], [p[1]], s=120, c=colors[w], edgecolors="k", zorder=3)
        ax8a.text(p[0] + 0.08, p[1] + 0.12, w, fontsize=FS_TICK, fontweight="bold")
    ax8a.grid(True, alpha=0.25)
    ax8a.set_xticks([])
    ax8a.set_yticks([])

    # after: positives closer, negatives farther
    pts2 = {
        "sat": np.array([0.15, 0.35]),
        "cat": np.array([-0.55, 0.55]),
        "on": np.array([0.55, 0.6]),
        "dog": np.array([-0.5, -1.7]),
        "ran": np.array([1.85, -1.35]),
    }
    ax8b.set_xlim(-2.5, 2.5)
    ax8b.set_ylim(-2.2, 2.2)
    ax8b.set_aspect("equal")
    ax8b.set_title("After a few NEG steps", fontsize=FS_AX_TITLE - 2)
    for w, p in pts2.items():
        ax8b.scatter([p[0]], [p[1]], s=120, c=colors[w], edgecolors="k", zorder=3)
        ax8b.text(p[0] + 0.08, p[1] + 0.12, w, fontsize=FS_TICK, fontweight="bold")
    # arrows showing motion
    for w in pts:
        ax8b.annotate(
            "", xy=pts2[w], xytext=pts[w],
            arrowprops=dict(arrowstyle="->", color="0.45", lw=1.4, alpha=0.7),
        )
    ax8b.grid(True, alpha=0.25)
    ax8b.set_xticks([])
    ax8b.set_yticks([])

    fig8.text(
        0.5, 0.02,
        r"Green = true contexts (attract).  Purple = negatives (repel).  Center $v_{\mathrm{sat}}$ relocates.",
        ha="center", fontsize=FS_TICK, fontweight="bold",
    )
    fig8.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig8


# -------------------------------------------------------
# FIGURE 9: Training loop
# -------------------------------------------------------
def build_step_9():
    fig9, ax9 = new_figure()
    fig9.suptitle(
        "Step 9: Training loop and the knobs that matter",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax9)

    steps = [
        (7.0, r"1. Build vocab (min count); optionally subsample frequent words", C_MUTE),
        (6.05, r"2. Stream text; for each center emit context pairs (skip-gram) or bags (CBOW)", C_NEW),
        (5.1, r"3. For each positive: draw $K$ negatives from $P_n$", C_OLD),
        (4.15, r"4. Descend $L_{\mathrm{NEG}}$; update only touched rows of $V,U$", "#f5eef8"),
        (3.2, r"5. Epochs over corpus (often 1–5 passes on large data)", C_GOAL),
    ]
    for y, text, face in steps:
        round_box(ax9, 0.45, y - 0.4, 11.1, 0.85, face, "0.45", lw=1.2)
        ax9.text(6.0, y, text, ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

    round_box(ax9, 0.45, 0.4, 11.1, 2.35, C_WARN, "#b7950b", lw=1.6)
    ax9.text(6.0, 2.3, "Knobs", ha="center", fontsize=FS_TICK, fontweight="bold", color="#7d6608")
    ax9.text(
        6.0, 1.25,
        r"$d$ (50–300) · window $c$ (2–10) · $K$ negatives · min-count · subsample $t$"
        "\n"
        r"learning rate schedule · CBOW vs skip-gram · whether to share / average $U,V$"
        "\n\n"
        "More data + sensible subsample usually beats exotic architecture tweaks.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig9


# -------------------------------------------------------
# FIGURE 10: GloVe glance
# -------------------------------------------------------
def build_step_10():
    fig10, ax10 = new_figure()
    fig10.suptitle(
        "Step 10: Cousin -- GloVe trains on a co-occurrence matrix",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax10)

    round_box(ax10, 0.4, 6.4, 11.2, 1.3, C_DARK, C_DARK, lw=0)
    ax10.text(
        6.0, 7.05,
        "Same goal (static word vectors).  Different supervision: global counts, not streamed pairs only.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax10, 0.5, 3.35, 11.0, 2.75, C_MUTE, "0.45", lw=1.6)
    ax10.text(
        6.0, 4.7,
        r"Build $X_{ij}=$ how often word $j$ appears in context of $i$."
        "\n\n"
        r"$L_{\mathrm{GloVe}}"
        r"= \sum_{i,j} f(X_{ij})"
        r"\left("
        r"u_j^\top v_i + b_i + c_j - \log X_{ij}"
        r"\right)^2$",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax10, 0.45, 0.45, 5.4, 2.55, C_NEW, C_DARK, lw=1.6)
    ax10.text(
        3.15, 1.7,
        "Word2Vec:\n"
        "online windows,\n"
        "local prediction /\n"
        "NEG classification.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax10, 6.15, 0.45, 5.4, 2.55, C_GOAL, C_OK, lw=1.6)
    ax10.text(
        8.85, 1.7,
        "GloVe:\n"
        "factorize log-counts\n"
        "with a weighted\n"
        "least-squares loss.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#145a32",
    )
    return fig10


# -------------------------------------------------------
# FIGURE 11: Bridge to contextual
# -------------------------------------------------------
def build_step_11():
    fig11, ax11 = new_figure()
    fig11.suptitle(
        "Step 11: After Word2Vec -- static table vs contextual encoder",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax11)

    round_box(ax11, 0.4, 4.2, 5.5, 3.4, C_MUTE, "0.45", lw=1.6)
    ax11.text(3.15, 7.2, "Word2Vec / GloVe", ha="center",
              fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax11.text(
        3.15, 5.5,
        "One vector per type.\n"
        r'"bank" is one row.'
        "\n\n"
        "Fast, strong baseline,\n"
        "no sentence context\n"
        "at lookup time.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax11, 6.1, 4.2, 5.5, 3.4, "#f5eef8", C_FUN, lw=1.8)
    ax11.text(8.85, 7.2, "BERT / GPT embeddings", ha="center",
              fontsize=FS_TICK, fontweight="bold", color=C_FUN)
    ax11.text(
        8.85, 5.5,
        "Vector depends on\n"
        "the whole sentence.\n\n"
        "Trained with MLM /\n"
        "next-token loss on\n"
        "Transformers (0950/0960).",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax11.text(
        6.0, 2.2,
        "Word2Vec teaches the geometry idea and the self-supervised window trick.\n"
        "Modern LLMs keep self-supervision, replace the shallow table with deep context.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    ax11.text(
        6.0, 0.7,
        "Still useful: initialize, analyze, retrieve, teach analogy / similarity first.",
        ha="center", fontsize=FS_TICK, style="italic", color="0.35",
    )
    return fig11


# -------------------------------------------------------
# FIGURE 12: Takeaway
# -------------------------------------------------------
def build_step_12():
    fig12, ax12 = new_figure()
    fig12.suptitle(
        "Step 12: Takeaway -- how to train word-to-vector models",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax12)

    round_box(ax12, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax12.text(
        6.0, 7.0,
        r"Skip-gram + NEG:  "
        r"$L=-\log\sigma(u_{O}^\top v_{I})"
        r"-\sum_{k=1}^{K}\log\sigma(-u_{k}^\top v_{I})$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    bullets = [
        r"1. Slide a window over text; form center–context supervision (self-supervised).",
        r"2. Skip-gram: $p(w_O\mid w_I)$ via dots; CBOW: predict center from mean context.",
        r"3. Replace full softmax with negative sampling (or hierarchical softmax).",
        r"4. SGD on $L_{\mathrm{NEG}}$ moves $V,U$ so co-occurring words align.",
        r"5. GloVe is the count-matrix cousin; Transformers later make vectors contextual.",
    ]
    y = 5.35
    for line in bullets:
        round_box(ax12, 0.5, y - 0.35, 11.0, 0.75, C_MUTE, "0.5", lw=1.1)
        ax12.text(6.0, y, line, ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")
        y -= 0.9

    ax12.text(
        6.0, 0.45,
        "Binding lens: windows bind words that share company into nearby vectors.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
    )
    return fig12


run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
    build_step_6,
    build_step_7,
    build_step_8,
    build_step_9,
    build_step_10,
    build_step_11,
    build_step_12,
])

print("\n" + "=" * 60)
print("0941 WORD-TO-VECTOR -- TRAINING")
print("=" * 60)
print("""
Data:
  sliding windows -> (center, context) pairs or CBOW bags

Skip-gram:
  max sum_t sum_{0<|j|<=c} log p(w_{t+j} | w_t)
  p(w_O|w_I) = softmax( U w_O · V w_I )   [full form]

Negative sampling (practical):
  L_NEG = -log sigma(u_O · v_I) - sum_{k=1..K} log sigma(-u_k · v_I)
  w_k ~ P_n(w)  (often unigram^{3/4})

CBOW:
  predict w_t from mean of context input vectors; same NEG trick

GloVe:
  weighted least squares on log co-occurrence counts

, , , 
""")
