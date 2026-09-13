"""
6_02_Markov_Decision_Process.py

The MDP assumption in reinforcement learning -- what it is, why theory needs it,
and where real problems only approximate it.

Sketch:
  1. RL needs a formal stage
  2. The five MDP pieces (S, A, P, R, gamma)
  3. The Markov assumption
  4. Why Markov unlocks Bellman
  5. Episode cartoon -- history collapses into s_t
  6. Policies and the induced process
  7. What is NOT assumed (yet)
  8. When reality is only approximately Markov
  9. Takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
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

np.random.seed(605)


def new_figure(*args, **kwargs):
    kwargs.setdefault("figsize", screen_figsize())
    kwargs.setdefault("dpi", FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


def blank(ax, xlim=(0, 12), ylim=(0, 8)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
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
C_WARN = "#fdebd0"
C_FUN = "#6c3483"
C_RL = "#fdebd0"
C_LOOP = "#2874a6"


# -------------------------------------------------------
# Step 1
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: RL theory needs a formal stage",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 5.7, 11.2, 1.9, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.65,
        "Before Bellman, policy iteration, or Q-learning,\n"
        "we need a precise model of sequential decision making.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.4, 2.7, "Agent", "chooses actions"),
        (4.2, 2.7, "Environment", "returns next state\nand reward"),
        (8.0, 2.7, "Goal", "maximize expected\nreturn over time"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.5, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 1.8, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 0.95, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.35, 11.2, 2.0, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "The standard mathematical stage for that loop is an MDP.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: Five pieces
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 2: An MDP is the tuple $(S,A,P,R,\gamma)$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    pieces = [
        (0.35, r"$S$", "States", "Where the agent can be.\nCan be discrete or continuous."),
        (2.65, r"$A$", "Actions", "What the agent can do.\nMay depend on $s$."),
        (4.95, r"$P$", "Transitions", r"$p(s'|s,a)$" + "\nhow the world moves."),
        (7.25, r"$R$", "Rewards", r"$r(s,a,s')$ or $r(s,a)$" + "\nimmediate feedback."),
        (9.55, r"$\gamma$", "Discount", r"$\gamma\in[0,1)$" + "\npatience / horizon."),
    ]
    for x, sym, name, body in pieces:
        round_box(ax, x, 3.55, 2.15, 4.0, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 1.075, 6.95, sym, ha="center",
                fontsize=FS_ANNOT + 3, fontweight="bold", color=C_DARK)
        ax.text(x + 1.075, 6.15, name, ha="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_LOOP)
        ax.text(x + 1.075, 4.85, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")

    round_box(ax, 0.35, 0.4, 11.3, 2.8, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.8,
        "Together these define a controlled stochastic process:\n"
        "pick $a_t$ in $s_t$ → sample $(s_{t+1}, r_{t+1})$ from $P,R$ → repeat.\n"
        r"Return $G_t=\sum_{k\geq 0}\gamma^k r_{t+k+1}$ is what we want large in expectation.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 3: Markov assumption
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: The Markov assumption",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 5.85, 11.2, 1.8, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.75,
        r"$p(s_{t+1}, r_{t+1}\mid s_t, a_t,\;\mathrm{history})$"
        r"  $=\; p(s_{t+1}, r_{t+1}\mid s_t, a_t)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.0, 5.5, 2.5, C_GOAL, C_OK, lw=2.0)
    ax.text(3.15, 5.05, "Meaning", ha="center",
            fontsize=FS_ANNOT - 1, fontweight="bold", color=C_OK)
    ax.text(
        3.15, 3.95,
        "The present state is a\n"
        "sufficient statistic of the past.\n\n"
        "Everything you need to predict\n"
        "the future is already in $s_t$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 6.1, 3.0, 5.5, 2.5, C_WARN, "#d68910", lw=2.0)
    ax.text(8.85, 5.05, "Not claiming", ha="center",
            fontsize=FS_ANNOT - 1, fontweight="bold", color="#9a7d0a")
    ax.text(
        8.85, 3.95,
        "Not: \"the world has no memory.\"\n\n"
        "Rather: we chose a state\n"
        "representation that already\n"
        "includes the relevant memory.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#6e2c00",
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.35, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.5,
        "If $s_t$ forgets something important (hidden velocity, inventory, opponent intent),\n"
        "the process is only partially observed -- a POMDP, not a pure MDP.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 4: Why Markov unlocks Bellman
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: Why the Markov assumption unlocks Bellman",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 5.5, 11.2, 2.15, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 6.55,
        "If tomorrow depends only on (state, action) now,\n"
        "then the value of \"now\" can be written using only\n"
        "reward-now + value-of-next -- no full trajectory history.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#1a5276",
    )

    round_box(ax, 0.4, 2.7, 5.5, 2.5, C_GOAL, C_OK, lw=2.0)
    ax.text(
        3.15, 4.55, "With Markov",
        ha="center", fontsize=FS_ANNOT - 1, fontweight="bold", color=C_OK,
    )
    ax.text(
        3.15, 3.55,
        r"$V(s)=$"
        "\n"
        r"$\mathbb{E}[r+\gamma V(s')\mid s]$"
        "\n\n"
        "One state is enough\n"
        "to start the recursion.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 6.1, 2.7, 5.5, 2.5, "#f8d7da", C_ACCENT, lw=2.0)
    ax.text(
        8.85, 4.55, "Without Markov",
        ha="center", fontsize=FS_ANNOT - 1, fontweight="bold", color=C_ACCENT,
    )
    ax.text(
        8.85, 3.55,
        "Value would depend on the\n"
        "whole history $h_t$.\n\n"
        "Bellman-style backup over\n"
        "histories is usually intractable.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#641e16",
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.05, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "This is why textbooks start with MDPs: recursion needs a Markov state.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 5: History cartoon
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: History collapses into the current state",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    # timeline of history
    ys = 5.8
    for i, lab in enumerate([r"$s_0$", r"$a_0$", r"$r_1$", r"$s_1$", r"$a_1$", r"$r_2$", r"$s_2$", r"$\cdots$", r"$s_t$"]):
        x = 0.7 + i * 1.2
        face = C_GOAL if "s" in lab else (C_WARN if "a" in lab else C_NEW)
        round_box(ax, x - 0.45, ys - 0.45, 0.9, 0.9, face, "0.45", lw=1.4)
        ax.text(x, ys, lab, ha="center", va="center", fontsize=FS_TICK, fontweight="bold")
    ax.annotate(
        "", xy=(10.6, 4.6), xytext=(10.6, 5.2),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2.0),
    )
    ax.text(10.6, 4.2, "Markov says:\nthis is enough", ha="center",
            fontsize=FS_TICK, fontweight="bold", color=C_DARK)

    round_box(ax, 2.5, 1.8, 7.0, 1.8, C_GOAL, C_OK, lw=2.2)
    ax.text(
        6.0, 2.7,
        r"Predict $(s_{t+1}, r_{t+1})$ from $(s_t, a_t)$ alone.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#145a32",
    )

    ax.text(
        6.0, 0.8,
        "Good state design packs the useful past into $s_t$\n"
        "(positions + velocities, board + whose turn, chat + memory buffer, ...).",
        ha="center", fontsize=FS_TICK + 1, color="#333333",
        bbox=dict(boxstyle="round", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )
    return fig


# -------------------------------------------------------
# Step 6: Policies
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 6: A policy $\pi$ turns an MDP into a stochastic process over states",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 5.6, 11.2, 2.0, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 6.6,
        r"$\pi(a\mid s)$  -- distribution over actions in state $s$"
        "\n"
        r"(deterministic policies are the special case $\pi(a\mid s)=\mathbf{1}_{a=\pi(s)}$).",
        ha="center", va="center", fontsize=FS_ANNOT, color="#1a5276",
    )

    cards = [
        (0.4, 2.55, "Induced dynamics",
         r"$p_\pi(s'|s)=\sum_a\pi(a|s)p(s'|s,a)$"),
        (4.2, 2.55, "Induced rewards",
         r"$r_\pi(s)=\sum_a\pi(a|s)\,r(s,a)$"),
        (8.0, 2.55, "Objective",
         r"$J(\pi)=\mathbb{E}_\pi[\sum_t\gamma^t r_t]$"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.7, C_MUTE, "0.5", lw=1.6)
        ax.text(x + 1.8, y + 2.0, title, ha="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 1.0, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.35, 11.2, 1.9, C_WARN, "#d68910", lw=1.8)
    ax.text(
        6.0, 1.3,
        "Finding a good $\\pi$ is the RL problem. MDPs define the arena;\n"
        "algorithms (PI, VI, Q-learning, policy gradients) search that arena.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#6e2c00",
    )
    return fig


# -------------------------------------------------------
# Step 7: What is not assumed
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: What the MDP assumption does NOT require (yet)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    items = [
        (6.2, "Known model?", "Classical DP assumes $P,R$ known.\n"
         "Model-free RL learns without them."),
        (4.7, "Small discrete $S,A$?", "Theory often starts finite.\n"
         "Practice uses function approximation."),
        (3.2, "Single agent?", "MDPs are one agent vs environment.\n"
         "Games need Markov games / MARL."),
        (1.7, "Perfect observation?", "MDP assumes $s_t$ is observed.\n"
         "Partial observability → POMDP."),
        (0.35, "Stationary $P,R$?", "Standard MDP: fixed dynamics.\n"
         "Nonstationary worlds need extra care."),
    ]
    for y, title, body in items:
        round_box(ax, 0.4, y, 11.2, 1.25, C_NEW, C_LOOP, lw=1.5)
        ax.text(2.3, y + 0.62, title, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        ax.text(7.5, y + 0.62, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")
    return fig


# -------------------------------------------------------
# Step 8: Approximate Markov
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: Reality is often only approximately Markov",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    cards = [
        (0.4, 4.0, "Robotics", "Need velocities, contacts,\nhistory windows in the state."),
        (4.2, 4.0, "Dialogue / LLM agents", "Need memory / scratchpads\nbecause raw tokens forget."),
        (8.0, 4.0, "Finance / users", "Hidden intents and lags;\nfeatures approximate the state."),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 3.4, C_MUTE, "0.5", lw=1.8)
        ax.text(x + 1.8, y + 2.55, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 1.4, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.4, 11.2, 3.2, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 2.0,
        "Engineering move: enrich the observation until the Markov assumption\n"
        "is good enough for the decisions you care about.\n\n"
        "Theory still starts with an exact MDP -- then we relax.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#145a32",
    )
    return fig


# -------------------------------------------------------
# Step 9: Takeaway
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Takeaway -- MDP is the default RL stage",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    rows = [
        (6.1, "Tuple", r"$(S,A,P,R,\gamma)$ defines states, actions, dynamics, rewards, discount."),
        (4.7, "Markov", r"Future $\perp$ history given $(s_t,a_t)$ -- state is a sufficient summary."),
        (3.3, "Why", "Markov makes value recursive → Bellman equations become possible."),
        (1.9, "Practice", "Build features / memory so the assumption is approximately true."),
        (0.5, "Next", "Bellman · policy / value iteration · policy gradients."),
    ]
    for y, title, body in rows:
        round_box(ax, 0.4, y, 11.2, 1.2, C_NEW, C_LOOP, lw=1.6)
        ax.text(2.0, y + 0.6, title, ha="center", va="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(7.3, y + 0.6, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")
    return fig


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
])


print("\n" + "=" * 60)
print("MARKOV DECISION PROCESS")
print("=" * 60)
print("""
MDP = (S, A, P, R, gamma)
Markov: p(s',r | s,a, history) = p(s',r | s,a)
This is the stage on which Bellman / PI / VI are built.
""")
