"""
How ChatGPT-class models are trained -- mechanistically.

OpenAI-style pipeline (conceptual):
  data preparation -> pre-training -> post-training -> evaluation / improvement

Sketch:
  1. Pipeline at a glance
  2. Pre-training: what is actually learned? (next-token loss)
  3. What the Transformer actually learns (f_theta)
  4. Generation is then almost embarrassingly simple
  5. Post-training: shape which parts of the distribution are preferred
  6. The open question: what does post-training change?
  7. Fine-tuning starts from theta_0 -- same mechanism, new parameters
  8. SFT: still next-token prediction; different training distribution
  9. What SFT does to probability mass
  10. SFT limitation: many valid answers
  11. Preference data: A succeeds B
  12. Reward model R_phi evaluates; policy generates
  13. RL updates the LLM from sequence reward
  14. Token-level action + sequence-level reward
  15. SFT vs REINFORCE / PPO formulas
  16. KL constraint + L^PPO
  17. Three stages + binding framework
  18. Key RL question: sequence reward -> token policy
  19. Policy-gradient / REINFORCE identity
  20. Credit assignment (why vanilla REINFORCE shares R)
  21. Advantages A_t and PPO's clipped rho A
  22. LLM: same token actions, sequence reward
  23. Score function and 1/pi weighting
  24. Softmax coupling: where probability mass comes from
  25. R/pi is reward-equivalent weighting -- not literal r_t
  26. Token-wise responsibility vs token-wise reward

Note: OpenAI has not published a complete current recipe; we use the
public InstructGPT / ChatGPT post-training pattern as the teaching model.
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

C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_GOAL = "#d4edda"
C_WARN = "#f8d7da"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_FUN = "#8e44ad"
C_OLD = "#fce4d6"
C_BIN = "#d68910"


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


def blank(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=2.2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=14),
    )


# -------------------------------------------------------
# FIGURE 1: Pipeline at a glance
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 1: How ChatGPT-class models are trained — the pipeline",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax1)

    round_box(ax1, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 7.1,
        "Mechanistic view — not a brochure.\n"
        "Data preparation → pre-training → post-training → evaluation / improvement",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    stages = [
        (0.4, "1. Data", "Clean / filter /\ntokenize text+code", C_MUTE, "0.45"),
        (3.3, "2. Pre-train", "Next-token loss\non huge corpora", C_NEW, "#2874a6"),
        (6.2, "3. Post-train", "SFT + preference /\nRLHF-style shaping", "#f5eef8", C_FUN),
        (9.1, "4. Evaluate", "Measure, fix,\nimprove, redeploy", C_GOAL, C_OK),
    ]
    for x, title, body, face, edge in stages:
        round_box(ax1, x, 3.0, 2.6, 3.0, face, edge, lw=2.0)
        ax1.text(x + 1.3, 5.4, title, ha="center", fontsize=FS_ANNOT, fontweight="bold", color=edge if edge != "0.45" else C_DARK)
        ax1.text(x + 1.3, 4.1, body, ha="center", va="center", fontsize=FS_TICK + 1)
    for x in (2.95, 5.85, 8.75):
        arrow(ax1, x, 4.5, x + 0.3, 4.5, C_DARK, 2.0)

    ax1.text(
        6.0, 1.6,
        "This lecture focuses on the learning core: pre-training, then what post-training changes.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9e79f", edgecolor="#b7950b", alpha=0.95),
    )
    return fig1


# -------------------------------------------------------
# FIGURE 2: Pre-training -- next token
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        "Step 2: Pre-training — what is actually learned?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax2)

    round_box(ax2, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax2.text(
        6.0, 7.1,
        r"Token sequence $x_1,x_2,\ldots,x_T$.  At position $t$, the target is $x_{t+1}$.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    # Visual token strip
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    for i, tok in enumerate(tokens):
        x = 0.7 + i * 1.7
        face = C_NEW if i < 4 else C_OLD
        edge = "#2874a6" if i < 4 else C_BIN
        round_box(ax2, x, 4.7, 1.5, 1.2, face, edge, lw=1.8)
        ax2.text(x + 0.75, 5.3, tok, ha="center", va="center",
                 fontsize=FS_ANNOT, fontweight="bold", color=edge)
    ax2.text(4.1, 4.35, r"context $x_{\leq t}$", ha="center", fontsize=FS_TICK, color="#2874a6", fontweight="bold")
    ax2.text(9.2, 4.35, r"target $x_{t+1}$", ha="center", fontsize=FS_TICK, color=C_BIN, fontweight="bold")
    arrow(ax2, 7.4, 5.3, 7.7, 5.3, C_DARK, 2.0)

    round_box(ax2, 0.5, 1.4, 11.0, 2.6, C_MUTE, "0.45", lw=1.6)
    ax2.text(
        6.0, 2.9,
        r"Transformer outputs a full distribution over the vocabulary $V$:"
        "\n"
        r"$P_\theta(v\mid x_{\leq t}),\quad v\in V$"
        "\n\n"
        r"Loss (cross-entropy / NLL):   "
        r"$L(\theta)=-\sum_t \log P_\theta(x_{t+1}\mid x_{\leq t})$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )
    ax2.text(
        6.0, 0.7,
        r"Core operation:   data $\rightarrow$ complete conditional probability distribution",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
    )
    return fig2


# -------------------------------------------------------
# FIGURE 3: What is learned -- f_theta
# -------------------------------------------------------
def build_step_3():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        r"Step 3: What the Transformer actually learns — $f_\theta$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax3)

    round_box(ax3, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax3.text(
        6.0, 7.1,
        "Not a lookup table  context → next word.\n"
        r"It learns parameters $\theta$ of a gigantic function $f_\theta$.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax3, 0.5, 3.6, 5.2, 2.5, C_WARN, C_ACCENT, lw=2.0)
    ax3.text(3.1, 5.5, "Not this", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT)
    ax3.text(
        3.1, 4.4,
        "Huge dictionary of\n"
        r"$\mathrm{context}\mapsto\mathrm{next\ word}$"
        "\n"
        "(memorized pairs)",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax3, 6.3, 3.6, 5.2, 2.5, C_GOAL, C_OK, lw=2.2)
    ax3.text(8.9, 5.5, "This", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax3.text(
        8.9, 4.4,
        r"$f_\theta(x_{\leq t})=P_\theta(\cdot\mid x_{\leq t})$"
        "\n\n"
        r"context space $\rightarrow$ distribution space",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    round_box(ax3, 0.5, 0.7, 11.0, 2.5, C_MUTE, "0.45", lw=1.6)
    ax3.text(
        6.0, 1.95,
        "Parameters move by gradient descent on the next-token loss:\n"
        r"$\theta \leftarrow \theta - \eta\,\nabla_\theta L$"
        "\n"
        "That is where the model's statistical structure is acquired.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )
    return fig3


# -------------------------------------------------------
# FIGURE 4: Generation loop
# -------------------------------------------------------
def build_step_4():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        "Step 4: Generation is then almost embarrassingly simple",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax4)

    round_box(ax4, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax4.text(
        6.0, 7.1,
        r"Once $f_\theta$ is learned, sampling is a short loop — not the source of intelligence.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    # Loop diagram
    boxes = [
        (0.6, r"$x_{\leq t}$", C_MUTE),
        (3.5, r"$P_\theta(\cdot\mid x_{\leq t})$", C_NEW),
        (7.0, r"sample $x_{t+1}$", C_OLD),
        (9.7, r"$x_{\leq t+1}$", C_GOAL),
    ]
    for x, title, face in boxes:
        round_box(ax4, x, 4.0, 2.4, 1.6, face, "0.4", lw=1.8)
        ax4.text(x + 1.2, 4.8, title, ha="center", va="center",
                 fontsize=FS_ANNOT, fontweight="bold")
    for x in (3.05, 5.95, 9.45):
        arrow(ax4, x, 4.8, x + 0.4, 4.8, C_DARK, 2.0)

    # Feedback arrow
    ax4.annotate(
        "", xy=(1.8, 3.85), xytext=(10.9, 3.85),
        arrowprops=dict(arrowstyle="->", color=C_FUN, lw=2.2,
                        connectionstyle="arc3,rad=0.35"),
    )
    ax4.text(6.0, 3.15, "repeat", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_FUN)

    round_box(ax4, 0.5, 0.6, 11.0, 2.2, C_MUTE, "0.45", lw=1.6)
    ax4.text(
        6.0, 1.7,
        "The recurrence itself is not intelligence.\n"
        r"The sophisticated object is the conditional distribution  $P_\theta(\cdot\mid\mathrm{context})$.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )
    return fig4


# -------------------------------------------------------
# FIGURE 5: Post-training
# -------------------------------------------------------
def build_step_5():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        "Step 5: Then comes post-training — a different story",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax5)

    round_box(ax5, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax5.text(
        6.0, 7.1,
        "Historically: supervised fine-tuning on human demonstrations,\n"
        "then preference data + RLHF-style reinforcement learning.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax5, 0.45, 3.2, 5.4, 2.9, C_NEW, "#2874a6", lw=2.2)
    ax5.text(3.15, 5.6, "Pre-training", ha="center", fontsize=FS_AX_TITLE - 2,
             fontweight="bold", color="#2874a6")
    ax5.text(
        3.15, 4.3,
        "Learn the world / language\ndistribution.\n\n"
        "Prediction, reasoning patterns,\nproblem-solving capabilities.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax5, 6.15, 3.2, 5.4, 2.9, "#f5eef8", C_FUN, lw=2.2)
    ax5.text(8.85, 5.6, "Post-training", ha="center", fontsize=FS_AX_TITLE - 2,
             fontweight="bold", color=C_FUN)
    ax5.text(
        8.85, 4.3,
        "Shape which parts of that\ndistribution are preferred.\n\n"
        "Alignment, usefulness, safety,\ninstruction-following.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax5, 0.45, 0.55, 11.1, 2.3, C_MUTE, "0.45", lw=1.6)
    ax5.text(
        6.0, 1.7,
        "Conceptual separation:\n"
        r"Pre-training: learn $P_\theta(\cdot\mid\mathrm{context})$ from the corpus."
        "\n"
        "Post-training: tilt that distribution toward preferred behaviors.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )
    return fig5


# -------------------------------------------------------
# FIGURE 6: Open question
# -------------------------------------------------------
def build_step_6():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        "Step 6: The interesting question",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax6)

    round_box(ax6, 0.6, 4.4, 10.8, 2.8, C_DARK, C_DARK, lw=0)
    ax6.text(
        6.0, 5.8,
        "If pre-training already teaches an enormous\n"
        "conditional probability distribution,",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#d6eaf8",
    )
    ax6.text(
        6.0, 4.9,
        "what exactly does post-training change in that distribution?",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#f9e79f",
    )

    round_box(ax6, 0.6, 1.0, 10.8, 2.9, C_GOAL, C_OK, lw=2.2)
    ax6.text(
        6.0, 2.45,
        "Deeper than: \"fine-tuning makes ChatGPT follow instructions.\"\n\n"
        "Pre-training builds capability.\n"
        "Post-training selects and amplifies preferred modes of that capability.\n\n"
        "Next steps answer this mechanistically: SFT, then preference / RL.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#145a32",
    )
    return fig6


# -------------------------------------------------------
# FIGURE 7: Start from pretrained theta_0
# -------------------------------------------------------
def build_step_7():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        r"Step 7: Fine-tuning starts from $\theta_0$ — same mechanism, new parameters",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax7)

    round_box(ax7, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax7.text(
        6.0, 7.1,
        "Qualification: OpenAI has not published a complete current recipe.\n"
        "We teach the public InstructGPT / ChatGPT post-training pattern.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax7, 0.5, 3.5, 5.2, 2.6, C_NEW, "#2874a6", lw=2.0)
    ax7.text(3.1, 5.5, r"After pre-training", ha="center", fontsize=FS_ANNOT, fontweight="bold", color="#2874a6")
    ax7.text(
        3.1, 4.4,
        r"parameters $\theta_0$"
        "\n"
        r"policy $\pi_{\theta_0}(x_t\mid x_{<t})$"
        "\n\n"
        r"$P_{\theta_0}(y\mid x)=\prod_t P_{\theta_0}(y_t\mid x,y_{<t})$",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax7, 6.3, 3.5, 5.2, 2.6, "#f5eef8", C_FUN, lw=2.0)
    ax7.text(8.9, 5.5, "Fine-tuning does", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_FUN)
    ax7.text(
        8.9, 4.4,
        r"$\theta_0\rightarrow\theta_1$"
        "\n\n"
        r"$P_{\theta_0}(\cdot\mid\mathrm{ctx})"
        r"\;\rightarrow\;"
        r"P_{\theta_1}(\cdot\mid\mathrm{ctx})$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    ax7.text(
        6.0, 1.8,
        "It does not replace next-token prediction.\n"
        "It changes which conditional distributions the parameters implement.",
        ha="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9e79f", edgecolor="#b7950b", alpha=0.95),
    )
    return fig7


# -------------------------------------------------------
# FIGURE 8: SFT
# -------------------------------------------------------
def build_step_8():
    fig8, ax8 = new_figure()
    fig8.suptitle(
        "Step 8: Supervised fine-tuning (SFT) — still next-token prediction",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax8)

    round_box(ax8, 0.4, 6.45, 11.2, 1.25, C_DARK, C_DARK, lw=0)
    ax8.text(
        6.0, 7.05,
        r"Human (or curated) demonstration: $(x,y^*)$ with $y^*=(y_1^*,\ldots,y_T^*)$."
        "\n"
        r"$L_{\mathrm{SFT}}=-\sum_t\log P_\theta(y_t^*\mid x,y_{<t}^*)$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax8, 0.5, 3.5, 5.2, 2.5, C_MUTE, "0.45", lw=1.6)
    ax8.text(3.1, 5.4, "Unchanged", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax8.text(
        3.1, 4.3,
        "Same Transformer.\n"
        r"Same map: context $\rightarrow P_\theta(\mathrm{next}\mid\mathrm{ctx})$."
        "\n"
        "Same gradient descent on parameters.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax8, 6.3, 3.5, 5.2, 2.5, C_GOAL, C_OK, lw=2.2)
    ax8.text(8.9, 5.4, "What changed", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax8.text(
        8.9, 4.3,
        "The training distribution.\n\n"
        "Pre-train: internet / data distribution\n"
        "SFT: desired human-response distribution",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    ax8.text(
        6.0, 1.9,
        "That distinction is huge: same mechanism, different data binding.",
        ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK,
    )
    ax8.text(
        6.0, 0.9,
        "InstructGPT: collect human demonstrations → supervised fine-tune GPT-3.",
        ha="center", fontsize=FS_TICK + 1, style="italic", color="0.35",
    )
    return fig8


# -------------------------------------------------------
# FIGURE 9: SFT reshapes probability
# -------------------------------------------------------
def build_step_9():
    fig9, ax9 = new_figure()
    fig9.suptitle(
        "Step 9: What SFT does to probability",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax9)

    round_box(ax9, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax9.text(
        6.0, 7.1,
        r'Prompt $x$: "Explain quantum mechanics to a child."'
        "\n"
        r"Pre-trained $P_{\theta_0}(y\mid x)$ is broad over many styles of continuation.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    # Bar-like modes under theta_0
    modes = [
        (0.5, "technical", 2.2, C_MUTE),
        (2.5, "conversational", 1.6, C_MUTE),
        (4.5, "incomplete", 1.1, C_MUTE),
        (6.5, "verbose", 1.4, C_MUTE),
        (8.5, "unsafe /\nirrelevant", 0.9, C_WARN),
    ]
    ax9.text(6.0, 5.85, r"Before SFT — many modes have mass under $P_{\theta_0}$",
             ha="center", fontsize=FS_TICK, fontweight="bold", color="0.35")
    for x, lab, h, face in modes:
        round_box(ax9, x, 3.5, 1.7, h, face, "0.45", lw=1.2)
        ax9.text(x + 0.85, 3.5 + h / 2, lab, ha="center", va="center", fontsize=FS_TICK - 2)

    # After SFT highlight
    round_box(ax9, 0.5, 0.55, 11.0, 2.5, C_GOAL, C_OK, lw=2.0)
    ax9.text(
        6.0, 2.0,
        "Human demonstrates: \"Imagine that everything around us is made of tiny...\"\n\n"
        r"$P_{\theta_1}(y^*\mid x) > P_{\theta_0}(y^*\mid x)$  on training examples"
        "\n"
        "SFT = distribution shaping through externally supplied examples.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#145a32",
    )
    return fig9


# -------------------------------------------------------
# FIGURE 10: SFT limitation
# -------------------------------------------------------
def build_step_10():
    fig10, ax10 = new_figure()
    fig10.suptitle(
        "Step 10: SFT limitation — many valid answers",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax10)

    round_box(ax10, 0.4, 5.8, 11.2, 1.7, C_DARK, C_DARK, lw=0)
    ax10.text(
        6.0, 6.65,
        r"Suppose $y_A$ and $y_B$ are both perfectly acceptable."
        "\n"
        r"The demonstration contains only $y_A$.  SFT says: raise $P(y_A)$."
        "\n"
        r"It does not directly say: \"$y_B$ is also good.\"",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax10, 0.6, 2.8, 5.0, 2.5, C_NEW, "#2874a6", lw=1.8)
    ax10.text(3.1, 4.7, r"Demonstration $y_A$", ha="center", fontsize=FS_ANNOT, fontweight="bold", color="#2874a6")
    ax10.text(3.1, 3.6, "SFT increases its\ntoken-sequence probability.",
              ha="center", va="center", fontsize=FS_ANNOT)

    round_box(ax10, 6.4, 2.8, 5.0, 2.5, C_MUTE, "0.45", lw=1.8)
    ax10.text(8.9, 4.7, r"Also-valid $y_B$", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax10.text(8.9, 3.6, "No direct credit.\nHuman behavior is multimodal.",
              ha="center", va="center", fontsize=FS_ANNOT)

    ax10.text(
        6.0, 1.4,
        "That is one reason preference training becomes useful.",
        ha="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9e79f", edgecolor="#b7950b", alpha=0.95),
    )
    return fig10


# -------------------------------------------------------
# FIGURE 11: Preference data
# -------------------------------------------------------
def build_step_11():
    fig11, ax11 = new_figure()
    fig11.suptitle(
        "Step 11: Preference data changes the learning signal",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax11)

    round_box(ax11, 0.4, 6.45, 11.2, 1.25, C_DARK, C_DARK, lw=0)
    ax11.text(
        6.0, 7.05,
        r"Not $(x,y^*)$ only — now $(x,y_A,y_B)$ with a human judgment $y_A\succ y_B$.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax11, 0.5, 3.5, 5.2, 2.5, C_MUTE, "0.45", lw=1.6)
    ax11.text(3.1, 5.4, "SFT signal", ha="center", fontsize=FS_ANNOT, fontweight="bold")
    ax11.text(3.1, 4.3, "\"Generate exactly\nthis token sequence.\"",
              ha="center", va="center", fontsize=FS_ANNOT + 1)

    round_box(ax11, 6.3, 3.5, 5.2, 2.5, "#f5eef8", C_FUN, lw=2.2)
    ax11.text(8.9, 5.4, "Preference signal", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_FUN)
    ax11.text(8.9, 4.3, "\"Between these behaviors,\nI prefer this one.\"",
              ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold")

    ax11.text(
        6.0, 2.0,
        "Example: explain the moon landing to a six-year-old.\n"
        "A = good explanation;  B = technically correct but incomprehensible;  human: A ≻ B.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax11.text(
        6.0, 0.85,
        "InstructGPT: compare multiple model outputs to train a reward model.",
        ha="center", fontsize=FS_TICK + 1, style="italic", color="0.35",
    )
    return fig11


# -------------------------------------------------------
# FIGURE 12: Reward model
# -------------------------------------------------------
def build_step_12():
    fig12, ax12 = new_figure()
    fig12.suptitle(
        r"Step 12: Reward model $R_\phi$ — evaluate, do not generate",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax12)

    round_box(ax12, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax12.text(
        6.0, 7.1,
        r"Train $R_\phi(x,y)$ on comparisons $(x,y_A,y_B,\,y_A\succ y_B)$"
        " so that "
        r"$R_\phi(x,y_A)>R_\phi(x,y_B)$.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax12, 0.6, 3.3, 5.0, 2.7, C_NEW, "#2874a6", lw=2.2)
    ax12.text(3.1, 5.4, "Policy LLM", ha="center", fontsize=FS_AX_TITLE - 2, fontweight="bold", color="#2874a6")
    ax12.text(3.1, 4.2, "Generates response\n"
              r"$y\sim\pi_\theta(\cdot\mid x)$",
              ha="center", va="center", fontsize=FS_ANNOT + 1)

    round_box(ax12, 6.4, 3.3, 5.0, 2.7, C_OLD, C_BIN, lw=2.2)
    ax12.text(8.9, 5.4, "Reward model", ha="center", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_BIN)
    ax12.text(8.9, 4.2, "Evaluates response\n"
              r"$r=R_\phi(x,y)$",
              ha="center", va="center", fontsize=FS_ANNOT + 1)

    ax12.text(
        6.0, 1.7,
        "Two neural models, two jobs: generate vs evaluate.\n"
        "That separation is conceptually essential for preference / RL post-training.",
        ha="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9e79f", edgecolor="#b7950b", alpha=0.95),
    )
    return fig12


# -------------------------------------------------------
# FIGURE 13: RL loop
# -------------------------------------------------------
def build_step_13():
    fig13, ax13 = new_figure()
    fig13.suptitle(
        "Step 13: RL changes the LLM using the reward",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax13)

    round_box(ax13, 0.4, 6.55, 11.2, 1.15, C_DARK, C_DARK, lw=0)
    ax13.text(
        6.0, 7.1,
        r"$x \rightarrow \pi_\theta \rightarrow y \rightarrow R_\phi(x,y) \rightarrow$ update $\theta$"
        "   |   InstructGPT / ChatGPT-style recipes used PPO.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    nodes = [
        (0.5, "prompt $x$", C_MUTE),
        (3.1, r"sample $y\sim\pi_\theta$", C_NEW),
        (6.2, r"score $r=R_\phi$", C_OLD),
        (9.1, r"update $\theta$", C_GOAL),
    ]
    for x, title, face in nodes:
        round_box(ax13, x, 4.55, 2.5, 1.35, face, "0.4", lw=1.8)
        ax13.text(x + 1.25, 5.2, title, ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")
    for x in (3.05, 5.65, 8.75):
        arrow(ax13, x, 5.2, x + 0.35, 5.2, C_DARK, 2.0)

    round_box(ax13, 0.4, 2.35, 5.5, 1.9, C_MUTE, "0.45", lw=1.4)
    ax13.text(3.15, 3.85, "REINFORCE (baseline form)", ha="center",
              fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax13.text(
        3.15, 3.0,
        r"$\nabla_\theta J=\mathbb{E}\!\left[R(y)\sum_t\nabla_\theta\log\pi_\theta(y_t|s_t)\right]$",
        ha="center", va="center", fontsize=FS_TICK, fontweight="bold",
    )

    round_box(ax13, 6.1, 2.35, 5.5, 1.9, "#f5eef8", C_FUN, lw=1.6)
    ax13.text(8.85, 3.95, "PPO (clipped surrogate)", ha="center",
              fontsize=FS_TICK, fontweight="bold", color=C_FUN)
    ax13.text(
        8.85, 2.95,
        r"$\rho_t=\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}$"
        "\n"
        r"$L^{\mathrm{CLIP}}=\mathbb{E}_t\!\left["
        r"\min\!\left("
        r"\rho_t A_t,"
        r"\,\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,A_t"
        r"\right)\right]$",
        ha="center", va="center", fontsize=FS_TICK - 1, fontweight="bold",
    )

    ax13.text(
        6.0, 1.2,
        "Higher-reward responses become more probable under $\\pi_\\theta$.\n"
        r"Same Transformer — different learning signal than SFT.  "
        r"(+ KL to $\pi_{\mathrm{SFT}}$ on later slides.)",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax13.text(
        6.0, 0.35,
        "Public pattern: SFT then RL with rewards; full current OpenAI recipe undisclosed.",
        ha="center", fontsize=FS_TICK - 1, style="italic", color="0.35",
    )
    return fig13


# -------------------------------------------------------
# FIGURE 14: Token action + sequence reward
# -------------------------------------------------------
def build_step_14():
    fig14, ax14 = new_figure()
    fig14.suptitle(
        "Step 14: Token-level action + sequence-level reward",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax14)

    round_box(ax14, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax14.text(
        6.0, 7.1,
        "No contradiction with next-token generation.\n"
        "Many small actions; one evaluation of the resulting trajectory.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    # Token actions
    for i, t in enumerate(["y1", "y2", "y3", "...", "yT"]):
        x = 1.0 + i * 1.9
        round_box(ax14, x, 4.3, 1.6, 1.2, C_NEW, "#2874a6", lw=1.6)
        ax14.text(x + 0.8, 4.9, rf"$a_t={t}$" if t != "..." else "...",
                  ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#2874a6")
    ax14.text(6.0, 3.85, "token-level actions (still next-token sampling)",
              ha="center", fontsize=FS_TICK, fontweight="bold", color="#2874a6")

    arrow(ax14, 6.0, 4.2, 6.0, 3.2, C_BIN, 2.2)
    round_box(ax14, 2.5, 1.3, 7.0, 1.7, C_OLD, C_BIN, lw=2.0)
    ax14.text(
        6.0, 2.15,
        r"sequence-level reward  $R(x,y_1,\ldots,y_T)$"
        "\n"
        "(often after the full response)",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#6e2c00",
    )
    return fig14


# -------------------------------------------------------
# FIGURE 15: SFT vs RL signals
# -------------------------------------------------------
def build_step_15():
    fig15, ax15 = new_figure()
    fig15.suptitle(
        "Step 15: SFT vs REINFORCE / PPO — formulas, not slogans",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax15)

    round_box(ax15, 0.35, 4.15, 5.6, 3.4, C_GOAL, C_OK, lw=2.2)
    ax15.text(3.15, 7.15, "SFT", ha="center", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    ax15.text(
        3.15, 5.55,
        "\"Here is the correct token sequence.\"\n\n"
        r"$L_{\mathrm{SFT}}=-\sum_t\log\pi_\theta(y_t^*|x,y_{<t}^*)$"
        "\n\n"
        "Specifies what to raise at each position.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax15, 6.15, 4.15, 5.5, 3.4, "#f5eef8", C_FUN, lw=2.2)
    ax15.text(8.9, 7.15, "REINFORCE / PPO-family", ha="center",
              fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_FUN)
    ax15.text(
        8.9, 5.55,
        "\"You generated this; reward was R.\"\n\n"
        r"$\nabla J=\mathbb{E}\!\left[R\sum_t\nabla\log\pi(y_t|s_t)\right]$"
        "\n\n"
        "Rates the trajectory — not each gold token.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax15, 0.35, 0.45, 11.3, 3.3, C_MUTE, "0.45", lw=1.5)
    ax15.text(6.0, 3.25, "PPO (clipped surrogate; schematic)", ha="center",
              fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax15.text(
        6.0, 2.0,
        r"$\rho_t=\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}$"
        "\n"
        r"$L^{\mathrm{CLIP}}(\theta)"
        r"=\mathbb{E}_t\!\left["
        r"\min\!\left("
        r"\rho_t A_t,\;"
        r"\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,A_t"
        r"\right)"
        r"\right]$"
        "\n"
        "Same policy class as SFT — different supervision.  Clip keeps updates from jumping too far.",
        ha="center", va="center", fontsize=FS_TICK, fontweight="bold",
    )
    return fig15


# -------------------------------------------------------
# FIGURE 16: KL constraint
# -------------------------------------------------------
def build_step_16():
    fig16, ax16 = new_figure()
    fig16.suptitle(
        "Step 16: KL constraint — do not only maximize reward",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax16)

    round_box(ax16, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax16.text(
        6.0, 7.1,
        "If you only maximize R, the policy may exploit the reward model.\n"
        "InstructGPT: KL penalty toward the SFT model (often per-token).",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax16, 0.45, 3.85, 11.1, 2.35, C_MUTE, "0.45", lw=1.6)
    ax16.text(6.0, 5.75, "RL objective with KL (InstructGPT-style)", ha="center",
              fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax16.text(
        6.0, 4.7,
        r"$\max_\theta\;"
        r"\mathbb{E}_{y\sim\pi_\theta}[R(x,y)]"
        r"\;-\;"
        r"\beta\,D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{SFT}})$"
        "\n"
        r"(equivalently reward shaping: $\tilde R=R-\beta\log\frac{\pi_\theta}{\pi_{\mathrm{SFT}}}$)",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    round_box(ax16, 0.45, 1.55, 11.1, 2.0, "#f5eef8", C_FUN, lw=1.6)
    ax16.text(6.0, 3.1, "PPO keeps updates local (clip) + optional value / entropy terms",
              ha="center", fontsize=FS_TICK, fontweight="bold", color=C_FUN)
    ax16.text(
        6.0, 2.25,
        r"$L^{\mathrm{PPO}}\approx L^{\mathrm{CLIP}}-c_1 L^{\mathrm{VF}}+c_2\,H[\pi_\theta]$"
        "\n"
        r"with $L^{\mathrm{CLIP}}$ as in Step 15 (advantage-weighted clipped ratio).",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax16.text(
        6.0, 0.7,
        "KL / clip: raise preferred behavior without wandering far from the SFT language model.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
    )
    return fig16


# -------------------------------------------------------
# FIGURE 17: Summary table + binding
# -------------------------------------------------------
def build_step_17():
    fig17, ax17 = new_figure()
    fig17.suptitle(
        "Step 17: What \"fine-tuning\" actually means",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax17)

    # Table header
    headers = [("Stage", 0.5), ("Learning signal", 3.2), ("What is learned", 7.6)]
    round_box(ax17, 0.4, 6.35, 11.2, 0.85, "#e8eef8", "0.45", lw=1.4)
    for lab, x in headers:
        ax17.text(x + 1.2, 6.75, lab, ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK)

    rows = [
        ("Pre-training", "observed next token", "general data / language distribution", C_NEW),
        ("SFT", "desired response tokens", "preferred response distribution", C_GOAL),
        ("Preference / RL", "comparative / quality reward", "preference over trajectories", "#f5eef8"),
    ]
    y = 5.35
    for stage, sig, learned, face in rows:
        round_box(ax17, 0.4, y - 0.35, 11.2, 0.95, face, "0.5", lw=1.2)
        ax17.text(1.7, y + 0.1, stage, ha="center", fontsize=FS_TICK, fontweight="bold")
        ax17.text(4.7, y + 0.1, sig, ha="center", fontsize=FS_TICK)
        ax17.text(9.3, y + 0.1, learned, ha="center", fontsize=FS_TICK)
        y -= 1.05

    round_box(ax17, 0.4, 0.45, 11.2, 2.0, C_DARK, C_DARK, lw=0)
    ax17.text(
        6.0, 1.45,
        r"predict what occurs  $\rightarrow$  imitate what is demonstrated  $\rightarrow$  prefer what is rewarded"
        "\n\n"
        "Binding lens: pre-train = data-internal; SFT = external demos; preference/RL = external reward.\n"
        r"Underlying machine stays $P_\theta(\mathrm{next\ token}\mid\mathrm{context})$ — targets change.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#f9e79f",
    )
    return fig17


# -------------------------------------------------------
# FIGURE 18: Key RL question
# -------------------------------------------------------
def build_step_18():
    fig18, ax18 = new_figure()
    fig18.suptitle(
        "Step 18: The key RL question — sequence reward → token policy",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax18)

    round_box(ax18, 0.4, 5.9, 11.2, 1.7, C_DARK, C_DARK, lw=0)
    ax18.text(
        6.0, 6.75,
        "RL has an action policy and (often) delayed rewards.\n"
        "How can a model learn a token-level action policy from a sequence-level reward?",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax18, 0.5, 2.8, 11.0, 2.7, C_GOAL, C_OK, lw=2.2)
    ax18.text(
        6.0, 4.15,
        "Short answer:\n"
        "RL need not be told which action was correct.\n"
        "It estimates how much each action contributed to the eventual reward,\n"
        "then raises the probability of actions that tend to produce higher reward.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#145a32",
    )

    ax18.text(
        6.0, 1.5,
        r"Setup: $y=(y_1,\ldots,y_5)$ from $\pi_\theta(y_t\mid s_t)$, $s_t=(x,y_{<t})$."
        "\n"
        r"End reward only: $R=+1$.  No labels $y_1=\mathrm{good},\ y_2=\mathrm{good},\ldots$",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    return fig18


# -------------------------------------------------------
# FIGURE 19: Policy gradient / REINFORCE
# -------------------------------------------------------
def build_step_19():
    fig19, ax19 = new_figure()
    fig19.suptitle(
        "Step 19: The policy-gradient / REINFORCE identity",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax19)

    round_box(ax19, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax19.text(
        6.0, 7.0,
        r"Objective $J(\theta)=\mathbb{E}_{y\sim\pi_\theta}[R(y)]$."
        "\n"
        "Heart of the answer — the policy-gradient identity:",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax19, 0.6, 3.5, 10.8, 2.5, C_MUTE, "0.45", lw=1.6)
    ax19.text(
        6.0, 4.75,
        r"$\nabla_\theta J"
        r"= \mathbb{E}\!\left["
        r"R(y)\sum_t \nabla_\theta\log\pi_\theta(y_t\mid s_t)"
        r"\right]$",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold",
    )

    round_box(ax19, 0.5, 0.55, 11.0, 2.6, C_GOAL, C_OK, lw=2.0)
    ax19.text(
        6.0, 1.85,
        "High $R$: positive learning signal on sampled actions\n"
        r"$R\,\nabla_\theta\log\pi_\theta(y_t\mid s_t)$  →  raise those action probabilities."
        "\n"
        "Low $R$: the opposite — suppress actions from poor trajectories.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#145a32",
    )
    return fig19


# -------------------------------------------------------
# FIGURE 20: Credit assignment
# -------------------------------------------------------
def build_step_20():
    fig20, ax20 = new_figure()
    fig20.suptitle(
        "Step 20: Credit assignment — the subtle problem",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax20)

    round_box(ax20, 0.4, 6.55, 11.2, 1.15, C_DARK, C_DARK, lw=0)
    ax20.text(
        6.0, 7.1,
        r"If the whole sequence gets the same $R=1$, REINFORCE initially treats"
        "\n"
        "all tokens in that trajectory similarly — it does not know which token \"caused\" success.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax20, 0.5, 4.55, 11.0, 1.7, C_MUTE, "0.45", lw=1.5)
    ax20.text(
        6.0, 5.4,
        r"Vanilla REINFORCE:  "
        r"$\nabla_\theta J"
        r"=\mathbb{E}\!\left["
        r"R(y)\sum_t\nabla_\theta\log\pi_\theta(y_t\mid s_t)"
        r"\right]$"
        "\n"
        r"Same scalar $R(y)$ multiplies every token's $\nabla\log\pi$ — that is the credit-assignment gap.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax20, 0.5, 2.55, 5.3, 1.7, C_WARN, C_ACCENT, lw=2.0)
    ax20.text(3.15, 3.85, "Does not know", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT)
    ax20.text(3.15, 3.15, "\"Token 37 was responsible\nfor the good answer.\"",
              ha="center", va="center", fontsize=FS_TICK + 1)

    round_box(ax20, 6.2, 2.55, 5.3, 1.7, C_MUTE, "0.45", lw=1.8)
    ax20.text(8.85, 3.85, "Only knows", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax20.text(8.85, 3.15, "\"This trajectory was good.\"",
              ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

    ax20.text(
        6.0, 1.2,
        "RL is fundamentally credit assignment.\n"
        r"Tools: reward-to-go, $V/Q$, advantages $A_t$, GAE, PPO's clipped $\rho_t A_t$, ...",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9e79f", edgecolor="#b7950b", alpha=0.95),
    )
    return fig20


# -------------------------------------------------------
# FIGURE 21: Advantages
# -------------------------------------------------------
def build_step_21():
    fig21, ax21 = new_figure()
    fig21.suptitle(
        r"Step 21: Advantages $A_t$ — better than raw $R$ on every token",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax21)

    round_box(ax21, 0.5, 5.55, 11.0, 2.1, C_MUTE, "0.45", lw=1.6)
    ax21.text(
        6.0, 6.6,
        r"$A_t = Q(s_t,a_t) - V(s_t)$"
        "\n"
        r"REINFORCE with advantage:  "
        r"$\nabla_\theta J \;\approx\; \sum_t A_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )

    round_box(ax21, 0.5, 3.15, 11.0, 2.1, "#f5eef8", C_FUN, lw=1.6)
    ax21.text(
        6.0, 4.2,
        "PPO plugs the same $A_t$ into a clipped probability ratio"
        "\n"
        r"$L^{\mathrm{CLIP}}"
        r"=\mathbb{E}_t\!\left["
        r"\min\!\left("
        r"\rho_t A_t,\;"
        r"\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,A_t"
        r"\right)\right]$"
        "  with  "
        r"$\rho_t=\pi_\theta/\pi_{\theta_{\mathrm{old}}}$",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax21, 0.5, 0.45, 11.0, 2.4, C_GOAL, C_OK, lw=2.0)
    ax21.text(
        6.0, 1.65,
        "Now the model is not only told \"the whole answer was good.\"\n"
        "It gets an estimated advantage for each action in its state.\n"
        r"$Q$: how good is taking $a_t$ in $s_t$?   "
        r"$V$: how good is $s_t$ on average?   "
        r"$A_t$: better/worse than average.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#145a32",
    )
    return fig21


# -------------------------------------------------------
# FIGURE 22: LLM concrete view
# -------------------------------------------------------
def build_step_22():
    fig22, ax22 = new_figure()
    fig22.suptitle(
        "Step 22: Especially concrete for an LLM",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax22)

    round_box(ax22, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax22.text(
        6.0, 7.1,
        r"Action space remains $a_t=y_t$.  A response is a trajectory $a_1,\ldots,a_T$."
        "\n"
        r"Example reward: $R(\text{entire response})=0.9$.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax22, 0.5, 3.4, 11.0, 2.7, C_NEW, "#2874a6", lw=2.0)
    ax22.text(
        6.0, 4.75,
        "RL asks:\n"
        "Which token-level actions, in their contexts,\n"
        "tend to lead to high-reward trajectories?",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#1a5276",
    )

    round_box(ax22, 0.5, 0.6, 11.0, 2.4, C_MUTE, "0.45", lw=1.6)
    ax22.text(
        6.0, 1.8,
        "Action: token    |    Reward: potentially entire sequence    |    Learning: credit assignment"
        "\n\n"
        "Not \"teach the correct next token\" (that is SFT).\n"
        "RL: raise the tendency to make actions like these in comparable states.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )
    return fig22


# -------------------------------------------------------
# FIGURE 23: Score function / 1/pi
# -------------------------------------------------------
def build_step_23():
    fig23, ax23 = new_figure()
    fig23.suptitle(
        r"Step 23: Score function and $1/\pi$ weighting",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax23)

    round_box(ax23, 0.4, 5.9, 11.2, 1.7, C_DARK, C_DARK, lw=0)
    ax23.text(
        6.0, 6.75,
        r"$R(y)\,\nabla_\theta\log\pi_\theta(y_t|s_t)"
        r"= R(y)\,\frac{1}{\pi_\theta(y_t|s_t)}\,\nabla_\theta\pi_\theta(y_t|s_t)$"
        "\n"
        r"Equivalently:  $\nabla_\theta J"
        r"=\mathbb{E}[R(y)\sum_t \frac{1}{\pi_\theta(y_t|s_t)}\nabla_\theta\pi_\theta(y_t|s_t)]$.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax23, 0.5, 3.2, 5.3, 2.3, C_MUTE, "0.45", lw=1.6)
    ax23.text(3.15, 4.8, r"Already likely action", ha="center", fontsize=FS_ANNOT, fontweight="bold")
    ax23.text(3.15, 3.9, r"$\pi$ large $\Rightarrow 1/\pi$ smaller"
              "\nweaker amplification",
              ha="center", va="center", fontsize=FS_ANNOT)

    round_box(ax23, 6.2, 3.2, 5.3, 2.3, "#f5eef8", C_FUN, lw=2.0)
    ax23.text(8.85, 4.8, "Surprising sampled action", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_FUN)
    ax23.text(8.85, 3.9, r"$\pi$ small $\Rightarrow 1/\pi$ larger"
              "\nstronger amplification",
              ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold")

    ax23.text(
        6.0, 1.7,
        r"Same trajectory reward $R(y)$:"
        "\n"
        r"positive $R$ reinforces the sampled action; negative $R$ suppresses it."
        "\n"
        r"$1/\pi$ is a likelihood-ratio correction — not a separate environmental reward.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    return fig23


# -------------------------------------------------------
# FIGURE 24: Softmax coupling
# -------------------------------------------------------
def build_step_24():
    fig24, ax24 = new_figure()
    fig24.suptitle(
        "Step 24: Softmax coupling — where probability mass comes from",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax24)

    round_box(ax24, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax24.text(
        6.0, 7.0,
        "Negative updates on competitors do not come from $1/\\pi$ alone.\n"
        "They come from normalization: probability mass has to come from somewhere.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax24, 0.6, 3.3, 10.8, 2.7, C_MUTE, "0.45", lw=1.6)
    ax24.text(
        6.0, 4.65,
        r"$\pi_i=\frac{e^{z_i}}{\sum_j e^{z_j}}$"
        "\n\n"
        r"$\frac{\partial \log\pi_k}{\partial z_i}=\delta_{ik}-\pi_i$",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold",
    )

    ax24.text(
        6.0, 1.8,
        "If action $k$ is reinforced: its logit tends to rise;\n"
        "competing logits receive negative gradients via $-\\pi_i$.\n"
        "That is the coupling behind \"some actions get downgraded.\"",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9e79f", edgecolor="#b7950b", alpha=0.95),
    )
    return fig24


# -------------------------------------------------------
# FIGURE 25: R/pi formulation
# -------------------------------------------------------
def build_step_25():
    fig25, ax25 = new_figure()
    fig25.suptitle(
        r"Step 25: Is $R(y)/\pi(y_t\mid s_t)$ the action reward?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax25)

    round_box(ax25, 0.4, 5.9, 11.2, 1.7, C_DARK, C_DARK, lw=0)
    ax25.text(
        6.0, 6.75,
        "Intuition: yes, it amplifies the sequence reward in an action-dependent way.\n"
        "Precision: call it reward-equivalent weighting — not the environmental reward itself.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax25, 0.5, 3.2, 11.0, 2.4, C_MUTE, "0.45", lw=1.6)
    ax25.text(
        6.0, 4.4,
        r"REINFORCE contribution for sampled $y_t$:"
        "\n"
        r"$R(y)\,\nabla_\theta\log\pi_\theta(y_t|s_t)"
        r"= \left(\frac{R(y)}{\pi_\theta(y_t|s_t)}\right)"
        r"\nabla_\theta\pi_\theta(y_t|s_t)$"
        "\n"
        r"(factor in parentheses = action-dependent amplification)",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    round_box(ax25, 0.5, 0.55, 11.0, 2.3, C_GOAL, C_OK, lw=2.0)
    ax25.text(
        6.0, 1.7,
        r"Actual task/environment reward remains $R(y)$."
        "\n"
        r"The factor $1/\pi$ is the likelihood-ratio term that makes the estimator unbiased."
        "\n"
        "Surprising successful actions get stronger policy-gradient pressure relative to $\\pi$.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#145a32",
    )
    return fig25


# -------------------------------------------------------
# FIGURE 26: Responsibility vs reward
# -------------------------------------------------------
def build_step_26():
    fig26, ax26 = new_figure()
    fig26.suptitle(
        "Step 26: Token-wise responsibility — not necessarily token-wise reward",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax26)

    round_box(ax26, 0.4, 6.45, 11.2, 1.25, C_DARK, C_DARK, lw=0)
    ax26.text(
        6.0, 7.05,
        "Without token-wise assessment, how can you make a token-wise update?\n"
        "The gradient itself performs that assessment implicitly.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax26, 0.45, 3.55, 5.4, 2.55, C_NEW, "#2874a6", lw=2.0)
    ax26.text(3.15, 5.55, "Token-wise assessment", ha="center",
              fontsize=FS_ANNOT, fontweight="bold", color="#2874a6")
    ax26.text(
        3.15, 4.4,
        "How did $y_t$ affect the\nprobability of this trajectory?\n\n"
        r"$\nabla_\theta\log\pi_\theta(y_t\mid s_t)$",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax26, 6.15, 3.55, 5.4, 2.55, C_OLD, C_BIN, lw=2.0)
    ax26.text(8.85, 5.55, "Token-wise reward $r_t$", ha="center",
              fontsize=FS_ANNOT, fontweight="bold", color=C_BIN)
    ax26.text(
        8.85, 4.4,
        "How good was token $y_t$ itself?\n\n"
        "REINFORCE need not construct\nan explicit $r_t$.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax26, 0.45, 0.45, 11.1, 2.8, C_MUTE, "0.45", lw=1.6)
    ax26.text(
        6.0, 1.85,
        r"$\nabla_\theta J"
        r"=\mathbb{E}[\sum_t "
        r"R(y)\,"
        r"\nabla_\theta\log\pi_\theta(y_t|s_t)"
        r"]$"
        "\n"
        r"$R(y)$ = sequence reward;   "
        r"$\nabla\log\pi$ = token-specific sensitivity"
        "\n\n"
        r"Token-specific learning contribution:  $g_t=R(y)\nabla_\theta\log\pi_\theta(y_t|s_t)$."
        "\n"
        r"$R(y)\neq r_t$,  but  $R(y)\nabla\log\pi$  is a token-wise learning signal."
        "\n"
        "Reward says whether the trajectory was good; the gradient says where that can change the policy.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig26


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
    build_step_13,
    build_step_14,
    build_step_15,
    build_step_16,
    build_step_17,
    build_step_18,
    build_step_19,
    build_step_20,
    build_step_21,
    build_step_22,
    build_step_23,
    build_step_24,
    build_step_25,
    build_step_26,
])

print("\n" + "=" * 60)
print("HOW CHATGPT-CLASS MODELS ARE TRAINED")
print("=" * 60)
print("""
Pipeline: data prep -> pre-train -> post-train -> evaluate/improve

Pre-training / SFT / preference-RL as before.

How sequence reward trains token policy (REINFORCE):
  J(theta) = E_{y~pi}[R(y)]
  grad J = E[ R(y) * sum_t grad log pi(y_t | s_t) ]

  High R reinforces sampled tokens; low R suppresses them.
  Same R on all tokens => credit-assignment problem.
  Advantages A_t = Q - V give better per-action signals:
    grad J ~ sum_t A_t grad log pi(a_t | s_t)

PPO (clipped surrogate used in InstructGPT-style recipes):
  rho_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
  L^CLIP = E_t[ min(rho_t A_t, clip(rho_t, 1-eps, 1+eps) A_t) ]
  L^PPO  ~ L^CLIP - c1 L^VF + c2 H[pi]
  plus KL toward pi_SFT:  max E[R] - beta D_KL(pi || pi_SFT)

Score function / 1/pi:
  R * grad log pi = (R/pi) * grad pi
  R/pi = action-dependent amplification (reward-equivalent weighting),
  not a separate environmental reward r_t.

Softmax:
  d log pi_k / d z_i = 1_{i=k} - pi_i
  Reinforcing k pulls mass from competitors.

Deep point:
  RL need not build explicit token rewards.
  It builds token-wise responsibility for the sequence reward via grad log pi.
""")
