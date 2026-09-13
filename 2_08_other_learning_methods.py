"""
2_08_other_learning_methods.py

Is MLE the only method of learning?  No.

MLE is one answer to: given data, how should I choose the model?
Many other objectives exist -- and sometimes they coincide with MLE.

Sketch:
  1. MLE is one way, not the way
  2. Reminder: what MLE answers
  3. A table of major learning methods
  4. Subtlety: MLE and loss minimization often coincide
  5. Least squares as Gaussian MLE
  6. Cross-entropy as conditional MLE
  7. Some learning is fundamentally different -- RL
  8. Three deeper kinds of "learning"
  9. Modern AI combines all three
  10. Conceptual map / takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_AXIS,
    FS_TICK,
    FS_ANNOT,
    FS_LEGEND,
    FIGURE_DPI,
    apply_rcparams,
    screen_figsize,
    run_slide_deck,
)
apply_rcparams(plt)
import numpy as np

np.random.seed(35)


def new_figure(*args, **kwargs):
    kwargs.setdefault("figsize", screen_figsize())
    kwargs.setdefault("dpi", FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_WARN = "#f8d7da"
C_FUN = "#8e44ad"
C_OLD = "#fce4d6"
C_LOOP = "#2874a6"
C_CODE = "#fbfcf8"


# -------------------------------------------------------
# Step 1
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: Is MLE the only method of learning?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        "No.  MLE is one way of learning -- not the only way.",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.5, 2.6, 11.0, 2.5, C_NEW, C_LOOP, lw=2.2)
    ax.text(
        6.0, 4.4,
        "Machine learning is a collection of answers to:",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 3.4,
        "Given data (or experience), how should I choose the model?",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
    )

    round_box(ax, 0.5, 0.45, 11.0, 1.8, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "MLE is one particular answer:\n"
        "choose parameters that make the observed data most likely.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: MLE reminder
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: Reminder -- what MLE answers",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 1.0, 4.6, 10.0, 2.8, C_NEW, C_LOOP, lw=2.4)
    ax.text(
        6.0, 6.5,
        "Maximum Likelihood Estimation",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 5.5,
        r"$\hat\theta=\arg\max_\theta P(D\mid\theta)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color=C_DARK,
    )

    trio = [
        (0.5, 0.7, "Data fixed", "We observed D.\nIt does not move."),
        (4.25, 0.7, "Parameter varies", r"Try different $\theta$"
         "\nand score each."),
        (8.0, 0.7, "Pick the best", r"The $\theta$ that" + "\nmakes D most likely."),
    ]
    for x, y, title, body in trio:
        round_box(ax, x, y, 3.5, 3.4, C_MUTE, "0.45", lw=1.8)
        ax.text(x + 1.75, y + 2.6, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.75, y + 1.3, body, ha="center", va="center",
                fontsize=FS_TICK + 1)
    return fig


# -------------------------------------------------------
# Step 3: table of methods
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: Major approaches (a partial map)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    rows = [
        ("MLE", "Make observed data most likely"),
        ("MAP", "MLE + prior belief about parameters"),
        ("Bayesian inference", "Keep a distribution over parameters"),
        ("Least squares", "Minimize squared prediction error"),
        ("Maximum margin", "Largest-margin decision boundary"),
        ("Empirical risk min.", "Minimize average training loss"),
        ("Reinforcement learning", "Maximize expected future reward"),
        ("Self-supervised", "Create signals from the data itself"),
        ("Contrastive learning", "Pull related reps together"),
        ("Adversarial / GAN", "Generator vs discriminator game"),
        ("Evolutionary methods", "Mutation / selection search"),
    ]
    # header
    round_box(ax, 0.35, 7.05, 3.5, 0.55, C_DARK, C_DARK, lw=0)
    round_box(ax, 3.95, 7.05, 7.7, 0.55, C_DARK, C_DARK, lw=0)
    ax.text(2.1, 7.3, "Method", ha="center", va="center",
            fontsize=FS_TICK - 1, fontweight="bold", color="white")
    ax.text(7.8, 7.3, "Basic idea", ha="center", va="center",
            fontsize=FS_TICK - 1, fontweight="bold", color="white")

    for i, (method, idea) in enumerate(rows):
        y = 6.4 - i * 0.55
        face = C_NEW if i % 2 == 0 else C_MUTE
        round_box(ax, 0.35, y, 3.5, 0.5, face, "0.55", lw=1.0)
        round_box(ax, 3.95, y, 7.7, 0.5, face, "0.55", lw=1.0)
        ax.text(2.1, y + 0.25, method, ha="center", va="center",
                fontsize=FS_TICK - 2, fontweight="bold", color=C_DARK)
        ax.text(7.8, y + 0.25, idea, ha="center", va="center",
                fontsize=FS_TICK - 2)

    ax.text(
        6.0, 0.25,
        "Same question -- different objectives.",
        ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK,
    )
    return fig


# -------------------------------------------------------
# Step 4: subtlety
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: Subtlety -- MLE and loss minimization often coincide",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.3, 11.0, 2.2, C_WARN, C_ACCENT, lw=2.2)
    ax.text(
        6.0, 6.4,
        "Many familiar losses are MLE in disguise.",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color="#641e16",
    )

    round_box(ax, 0.5, 2.5, 5.3, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 4.3, "Gaussian noise", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 3.3,
        "MLE  ⇔  least squares / MSE",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax, 6.2, 2.5, 5.3, 2.4, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 4.3, "Classifier probabilities", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 3.3,
        "MLE  ⇔  cross-entropy / NLL",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax, 0.5, 0.4, 11.0, 1.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.25,
        "So MLE is hiding under a surprisingly large amount of conventional ML.\n"
        "The next slides make those equivalences explicit.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 5: least squares = Gaussian MLE
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: Least squares as Gaussian MLE",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 6.85,
        r"$y_i = f_\theta(x_i)+\varepsilon_i$"
        "      with      "
        r"$\varepsilon_i\sim\mathcal{N}(0,\sigma^2)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color=C_DARK,
    )

    round_box(ax, 0.5, 2.7, 11.0, 2.4, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 4.35,
        "Maximizing "
        r"$P(D\mid\theta)$"
        " turns into minimizing",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 3.4,
        r"$\sum_i (y_i-f_\theta(x_i))^2$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color="#1c2833",
    )

    round_box(ax, 0.5, 0.4, 11.0, 1.9, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.35,
        "Ordinary least-squares regression = MLE under Gaussian noise.\n"
        "(Same MLE story as earlier steps.)",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 6: cross-entropy = conditional MLE
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Cross-entropy as conditional MLE",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.3, 11.0, 2.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.4,
        r"Train with  $-\sum_i\log P_\theta(y_i\mid x_i)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 3,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.5, 2.5, 11.0, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.7,
        "That is maximum-likelihood estimation of the conditional model\n"
        r"$P_\theta(y\mid x)$."
        "\n\n"
        "Cross-entropy loss  ≈  negative log-likelihood.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )

    round_box(ax, 0.5, 0.4, 11.0, 1.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.25,
        "Classifier training with CE is not a different religion from MLE --\n"
        "it is MLE for a predictive distribution over labels.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 7: RL is different
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: Some learning is fundamentally different -- RL",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_WARN, C_ACCENT, lw=2.0)
    ax.text(
        6.0, 6.5,
        "There may be no dataset of correct answers saying:\n"
        r'"For state $s$, the correct action is $a$."',
        ha="center", va="center", fontsize=FS_ANNOT, color="#641e16",
    )

    round_box(ax, 0.5, 3.15, 11.0, 1.95, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 4.1,
        r"$s_t \rightarrow a_t \rightarrow r_t \rightarrow s_{t+1}$"
        "      objective      "
        r"$\max_\theta\,\mathbb{E}_\pi[\sum_t\gamma^t r_t]$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )

    round_box(ax, 0.5, 0.4, 11.0, 2.4, C_MUTE, "0.45", lw=1.6)
    # Leading + body text moved down ~1 cm (~0.66 data units).
    ax.text(
        6.0, 1.49,
        r"REINFORCE:  $\nabla_\theta J="
        r"\mathbb{E}[R\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)]$"
        "\n\n"
        "Learn a policy for high-reward behavior --\n"
        "not parameters that make a fixed dataset maximally likely.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 8: three kinds of learning
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        'Step 8: Three deeper kinds of "learning"',
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    cards = [
        (0.4, 2.6, "1. Parameter estimation",
         '"What parameters best\nexplain the data?"\n\n'
         "MLE, MAP, Bayesian,\nleast squares, ...",
         C_NEW, C_LOOP),
        (4.2, 2.6, "2. Function / representation",
         '"What useful function\nor representation?"\n\n'
         "Nets, SSL, contrastive,\nautoencoders, ...",
         C_GOAL, C_OK),
        (8.0, 2.6, "3. Behavior / policy",
         '"What should I do\nto achieve a goal?"\n\n'
         "RL, imitation,\nagent learning, ...",
         C_OLD, "#e67e22"),
    ]
    for x, y, title, body, face, edge in cards:
        round_box(ax, x, y, 3.6, 4.8, face, edge, lw=2.2)
        ax.text(x + 1.8, y + 4.1, title, ha="center",
                fontsize=FS_TICK, fontweight="bold",
                color=edge if edge != "#e67e22" else "#a04000")
        ax.text(x + 1.8, y + 2.0, body, ha="center", va="center",
                fontsize=FS_TICK + 1)

    round_box(ax, 0.4, 0.35, 11.2, 1.9, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 1.3,
        "People call all three \"learning.\"\n"
        "Keeping them separate clarifies what an algorithm is optimizing.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Step 9: modern AI combines them
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Modern AI often combines all three",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    stages = [
        (0.5, 4.6, "1", "Pretrain ≈ MLE",
         "Next-token likelihood\n(parameter estimation)"),
        (4.25, 4.6, "2", "Align / RL",
         "Preferences or rewards\n(behavior learning)"),
        (8.0, 4.6, "3", "Act as agent",
         "Tools + feedback loop\n(policy in the world)"),
    ]
    for x, y, num, title, body in stages:
        round_box(ax, x, y, 3.5, 2.9, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 0.45, y + 2.35, num, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_LOOP)
        ax.text(x + 1.9, y + 2.35, title, ha="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        ax.text(x + 1.75, y + 1.1, body, ha="center", va="center",
                fontsize=FS_TICK + 1)

    round_box(ax, 0.5, 0.45, 11.0, 3.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 2.9,
        "Example: an LLM may\n"
        "• learn parameters by next-token MLE,\n"
        "• then align with preference optimization / RL,\n"
        "• then operate as an agent that learns from environmental feedback.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 10: takeaway
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: Conceptual map",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.3, 11.0, 2.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.4,
        "MLE is a fundamental learning principle,\n"
        "but definitely not the only one.",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.5, 2.7, 11.0, 2.2, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.8,
        "Deeper taxonomy question",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 3.15,
        "What are the fundamental objectives by which a machine can learn?",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    tips = [
        (0.5, 0.4, "Often coincides", "MSE, CE, many\nlosses = MLE stories"),
        (4.25, 0.4, "Sometimes differs", "RL, adversarial,\nevolutionary search"),
        (8.0, 0.4, "Best map", "Objectives first;\nalgorithms second"),
    ]
    for x, y, title, body in tips:
        round_box(ax, x, y, 3.5, 1.95, C_GOAL, C_OK, lw=1.8)
        ax.text(x + 1.75, y + 1.45, title, ha="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_OK)
        ax.text(x + 1.75, y + 0.65, body, ha="center", va="center",
                fontsize=FS_TICK)
    return fig


# -------------------------------------------------------
# Deck
# -------------------------------------------------------
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
])


print("\n" + "=" * 60)
print("OTHER LEARNING METHODS -- BEYOND MLE")
print("=" * 60)
print("""
MLE is one answer to: how should I choose the model from data?

Many other objectives exist (MAP, Bayesian, margin, ERM, RL, SSL, ...).

Often equivalent:
  Gaussian noise  =>  MLE = least squares / MSE
  Classifier CE   =>  MLE of P_theta(y|x)

Fundamentally different:
  RL optimizes expected reward along trajectories, not dataset likelihood.

Three kinds of "learning":
  1) parameter estimation
  2) function / representation learning
  3) behavior / policy learning

Modern systems often combine all three.
""")
