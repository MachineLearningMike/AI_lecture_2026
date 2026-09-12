"""
6_030_policy_iteration.py

Policy iteration in detail -- evaluate a policy, improve it, repeat.

Sits after 6_020_Bellman_equality.py (Bellman / PE / PI glance) and before
6_040_policy_gradient_methods.py.

Sketch:
  1. What policy iteration is
  2. Reminder: MDP, V^pi, Q^pi
  3. Policy evaluation -- Bellman expectation
  4. Iterative policy evaluation (sweeps)
  5. Policy improvement -- greedy w.r.t. Q^pi
  6. Policy improvement theorem
  7. The full loop
  8. Toy gridworld setup
  9. Demo: values + arrows across iterations
  10. Slider: scrub PI iterations
  11. Value iteration -- Bellman optimality backups
  12. Policy iteration vs value iteration
  13. Takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.widgets import Slider
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

np.random.seed(615)
RNG = np.random.default_rng(615)


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
C_ACTIVE = "#f5b7b1"
C_LOOP = "#2874a6"


# -------------------------------------------------------
# Tiny gridworld for the demo
# -------------------------------------------------------
# 4x4 grid, actions: 0N 1E 2S 3W
# Terminals: (0,0) and (3,3). Reward -1 each step; 0 on terminal self-loop.
N_ROWS, N_COLS = 4, 4
ACTIONS = [( -1, 0), (0, 1), (1, 0), (0, -1)]  # N E S W
ACTION_MARK = ["↑", "→", "↓", "←"]
GAMMA = 0.9
THETA = 1e-8


def state_id(r, c):
    return r * N_COLS + c


def rc_of(s):
    return divmod(s, N_COLS)


def is_terminal(s):
    r, c = rc_of(s)
    return (r, c) in ((0, 0), (N_ROWS - 1, N_COLS - 1))


def step(s, a):
    """Deterministic transition. Stay in place on bump / terminal."""
    if is_terminal(s):
        return s, 0.0
    r, c = rc_of(s)
    dr, dc = ACTIONS[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < N_ROWS and 0 <= nc < N_COLS):
        nr, nc = r, c
    sp = state_id(nr, nc)
    return sp, -1.0


def q_from_v(V, s, a):
    sp, r = step(s, a)
    return r + GAMMA * V[sp]


def policy_evaluation(pi, V0=None, theta=THETA, max_iters=10_000):
    """Exact iterative PE for a deterministic policy pi[s] in {0,1,2,3}."""
    nS = N_ROWS * N_COLS
    V = np.zeros(nS) if V0 is None else V0.copy()
    for _ in range(max_iters):
        delta = 0.0
        V_new = V.copy()
        for s in range(nS):
            if is_terminal(s):
                V_new[s] = 0.0
                continue
            v = q_from_v(V, s, int(pi[s]))
            delta = max(delta, abs(v - V[s]))
            V_new[s] = v
        V = V_new
        if delta < theta:
            break
    return V


def policy_improvement(V):
    """Greedy deterministic policy w.r.t. V. Returns (pi, stable)."""
    nS = N_ROWS * N_COLS
    pi = np.zeros(nS, dtype=int)
    for s in range(nS):
        if is_terminal(s):
            pi[s] = 0
            continue
        qs = [q_from_v(V, s, a) for a in range(4)]
        pi[s] = int(np.argmax(qs))
    return pi


def run_policy_iteration(pi0=None, max_iters=20):
    """Return list of (pi, V) after each PE, including the start policy's V."""
    nS = N_ROWS * N_COLS
    if pi0 is None:
        # Start: always try to go East (a bad but simple habit).
        pi = np.ones(nS, dtype=int)  # E
    else:
        pi = pi0.copy()
    history = []
    for _ in range(max_iters):
        V = policy_evaluation(pi)
        history.append((pi.copy(), V.copy()))
        pi_new = policy_improvement(V)
        if np.array_equal(pi_new, pi):
            break
        pi = pi_new
    return history


PI_HISTORY = run_policy_iteration()


def draw_grid_policy(ax, V, pi, title=""):
    """Heatmap of V with policy arrows."""
    Vm = V.reshape(N_ROWS, N_COLS)
    im = ax.imshow(Vm, cmap="Blues", vmin=V.min(), vmax=max(V.max(), 1e-6))
    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = state_id(r, c)
            ax.text(
                c, r, f"{V[s]:.1f}", ha="center", va="center",
                fontsize=FS_TICK - 1, color="k", fontweight="bold",
            )
            if is_terminal(s):
                ax.add_patch(Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, fill=False,
                    edgecolor="#c0392b", linewidth=2.5,
                ))
                continue
            a = int(pi[s])
            dr, dc = ACTIONS[a]
            ax.annotate(
                "", xy=(c + 0.30 * dc, r + 0.30 * dr),
                xytext=(c - 0.05 * dc, r - 0.05 * dr),
                arrowprops=dict(arrowstyle="->", color="#1a5276", lw=2.0),
            )
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, N_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_ROWS, 1), minor=True)
    ax.grid(which="minor", color="0.7", linestyle="-", linewidth=0.8)
    return im


# -------------------------------------------------------
# Step 1: Big picture
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: Policy iteration -- score the habit, then fix the habit",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 7.0,
        r"Alternate:  evaluate $\pi$  $\rightarrow$  improve $\pi$  $\rightarrow$  repeat"
        "\n"
        "until the greedy policy stops changing.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.4, 3.2, "Policy evaluation",
         r"Fix $\pi$. Compute $V^\pi$"
         "\n"
         "(how good is this habit?)."),
        (4.2, 3.2, "Policy improvement",
         r"Act greedily w.r.t. $V^\pi$"
         "\n"
         r"(or $Q^\pi$). Get $\pi'$."),
        (8.0, 3.2, "Repeat",
         r"Set $\pi\leftarrow\pi'$."
         "\n"
         "Stop when $\\pi'=\\pi$."),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.8, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 2.05, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 1.05, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.35, 11.2, 2.5, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.6,
        "Classical dynamic-programming route to an optimal policy in a known MDP.\n"
        "No samples yet -- we assume the transition model $p(s',r|s,a)$ is given.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: Reminder
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 2: Reminder -- $V^\pi$ and $Q^\pi$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 5.5, 5.5, 2.15, C_GOAL, C_OK, lw=2.0)
    ax.text(
        3.15, 7.15, r"State value",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    ax.text(
        3.15, 6.2,
        r"$V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]$"
        "\n"
        "expected return from $s$ under $\\pi$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 6.1, 5.5, 5.5, 2.15, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        8.85, 7.15, r"Action value",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        8.85, 6.2,
        r"$Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]$"
        "\n"
        "do $a$ once, then follow $\\pi$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 2.55, 11.2, 2.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 3.85,
        r"Bellman expectation (one-step):"
        "\n"
        r"$V^\pi(s)=\sum_a\pi(a|s)\sum_{s',r}"
        r"p(s',r|s,a)\,[r+\gamma V^\pi(s')]$"
        "\n"
        r"$Q^\pi(s,a)=\sum_{s',r}p(s',r|s,a)\,[r+\gamma V^\pi(s')]$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
        linespacing=1.35,
    )

    round_box(ax, 0.4, 0.35, 11.2, 1.9, C_WARN, "#d68910", lw=1.8)
    ax.text(
        6.0, 1.3,
        "Policy iteration needs both: evaluate gives $V^\\pi$;\n"
        "improve compares $Q^\\pi(s,\\cdot)$ and picks the best action.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#6e2c00",
    )
    return fig


# -------------------------------------------------------
# Step 3: Policy evaluation
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 3: Policy evaluation -- solve for $V^\pi$ with $\pi$ fixed",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 7.0,
        r"Hold $\pi$ fixed. The Bellman expectation equations are linear in $V^\pi$.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.25, 5.5, 2.85, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 5.7, "As a linear system", ha="center",
            fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 4.3,
        r"$|S|$ unknowns $V(s)$."
        "\n"
        "One equation per state."
        "\n\n"
        "Can solve exactly with\n"
        "sparse linear algebra\n"
        r"when $|S|$ is small.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 3.25, 5.5, 2.85, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 5.7, "As iteration", ha="center",
            fontsize=FS_ANNOT - 1, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 4.3,
        r"$V_{k+1}(s)\leftarrow$"
        "\n"
        r"$\mathbb{E}_\pi[r+\gamma V_k(s')]$"
        "\n\n"
        "Sweep all states."
        "\n"
        "Stop when max change $<\\theta$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.65,
        "Output: a report card $V^\\pi$ for the current behavior.\n"
        "No new policy yet -- only \"how good is what we already do?\"",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 4: Iterative PE picture
# -------------------------------------------------------
def build_step_4():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 4: Iterative policy evaluation -- Bellman backups until stable",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    # Run a few PE sweeps on the initial East policy and show residual decay.
    pi0 = np.ones(N_ROWS * N_COLS, dtype=int)
    V = np.zeros(N_ROWS * N_COLS)
    residuals = []
    snapshots = [V.copy()]
    for k in range(40):
        delta = 0.0
        V_new = V.copy()
        for s in range(N_ROWS * N_COLS):
            if is_terminal(s):
                V_new[s] = 0.0
                continue
            v = q_from_v(V, s, int(pi0[s]))
            delta = max(delta, abs(v - V[s]))
            V_new[s] = v
        V = V_new
        residuals.append(delta)
        if k in (0, 2, 8, 39):
            snapshots.append(V.copy())

    ax = axes[0]
    ax.semilogy(residuals, color=C_LOOP, lw=2.4)
    ax.set_xlabel("PE sweep", fontsize=FS_AXIS)
    ax.set_ylabel(r"max $|V_{k+1}-V_k|$", fontsize=FS_AXIS)
    ax.set_title("Bellman residual vs sweep", fontsize=FS_AX_TITLE - 1)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3, which="both")
    ax.text(
        0.98, 0.95,
        "Fixed $\\pi$ = always East.\n"
        "Residual drops to ~0\n"
        "=> $V$ satisfies Bellman.",
        transform=ax.transAxes, ha="right", va="top", fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )

    ax = axes[1]
    draw_grid_policy(ax, snapshots[-1], pi0, title=r"$V^\pi$ after PE (arrows = $\pi$)")
    fig.colorbar(axes[1].images[0], ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 5: Policy improvement
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 5: Policy improvement -- greedy one-step lookahead",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 7.0,
        r"$\pi'(s)\in\mathrm{arg\,max}_a Q^\pi(s,a)"
        r"=\mathrm{arg\,max}_a\sum_{s',r}p(s',r|s,a)\,[r+\gamma V^\pi(s')]$",
        ha="center", va="center", fontsize=FS_ANNOT - 1,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.2, 5.5, 2.9, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 5.7, "Procedure", ha="center",
            fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 4.3,
        "For each state $s$:\n"
        "  try every action $a$ for one step,\n"
        "  then follow $\\pi$ thereafter.\n\n"
        "Keep the $a$ with largest\n"
        "one-step lookahead $Q^\\pi(s,a)$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 3.2, 5.5, 2.9, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 5.7, "Deterministic case", ha="center",
            fontsize=FS_ANNOT - 1, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 4.3,
        "If ties: any argmax is fine.\n\n"
        "If $\\pi'$ equals $\\pi$\n"
        "at every state, stop --\n"
        "policy iteration has\n"
        "found a fixed point.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.55, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.6,
        "Intuition: if a one-step detour looks better than your current habit,\n"
        "and afterward you behave at least as well as before, the future improves.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 6: Improvement theorem
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Policy improvement theorem",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 6.2, 11.2, 1.45, C_GOAL, C_OK, lw=2.2)
    ax.text(
        6.0, 6.9,
        r"If $\pi'$ is greedy w.r.t. $Q^\pi$, then"
        "\n"
        r"$V^{\pi'}(s)\geq V^\pi(s)$  for every state $s$.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#145a32",
    )

    steps = [
        (4.85, "1. By greediness",
         r"$Q^\pi(s,\pi'(s))\geq Q^\pi(s,\pi(s))=V^\pi(s)$."),
        (3.35, "2. Unroll the inequality",
         r"Following $\pi'$ for one step, then $\pi$, is already $\geq V^\pi$."),
        (1.85, "3. Repeat the argument",
         r"Replace the trailing $\pi$ by $\pi'$ again and again."),
        (0.35, "4. Limit",
         r"The infinite rollout under $\pi'$ yields $V^{\pi'}\geq V^\pi$."),
    ]
    for i, (y, title, body) in enumerate(steps):
        face = C_NEW if i % 2 == 0 else C_MUTE
        edge = C_LOOP if i % 2 == 0 else "0.5"
        round_box(ax, 0.4, y, 11.2, 1.25, face, edge, lw=1.6)
        ax.text(2.1, y + 0.62, title, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        ax.text(7.4, y + 0.62, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")
    return fig


# -------------------------------------------------------
# Step 7: Full loop
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: Policy iteration -- the full loop",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    stages = [
        (0.5, 5.6, "1. Init", r"Pick any $\pi_0$"),
        (3.3, 5.6, "2. Evaluate", r"Compute $V^{\pi_k}$"),
        (6.1, 5.6, "3. Improve", r"$\pi_{k+1}=$ greedy"),
        (8.9, 5.6, "4. Check", r"If $\pi_{k+1}=\pi_k$\nstop; else $k\leftarrow k+1$"),
    ]
    for x, y, title, body in stages:
        round_box(ax, x, y, 2.6, 2.0, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.3, y + 1.45, title, ha="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(x + 1.3, y + 0.7, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")
    for x in (2.95, 5.75, 8.55):
        ax.annotate(
            "", xy=(x + 0.25, 6.6), xytext=(x - 0.25, 6.6),
            arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2.0),
        )

    round_box(ax, 0.5, 2.85, 11.0, 2.35, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 4.0,
        "Finite MDPs with finite actions: only finitely many deterministic policies.\n"
        "Each strict improvement raises $V$ somewhere and never decreases it anywhere.\n"
        "=> policy iteration reaches an optimal policy in finitely many iterations.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 0.5, 0.35, 11.0, 2.2, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.45,
        "Cost per iteration: a full policy evaluation (many Bellman sweeps)\n"
        "plus one greedy sweep. Often few outer iterations, expensive inner PE.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 8: Gridworld setup
# -------------------------------------------------------
def build_step_8():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 8: Toy gridworld for the demo",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank = np.zeros((N_ROWS, N_COLS))
    ax.imshow(blank, cmap="Greys", vmin=-1, vmax=1)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = state_id(r, c)
            if is_terminal(s):
                ax.add_patch(Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#f5b7b1", edgecolor="#c0392b", lw=2.0,
                ))
                ax.text(c, r, "T", ha="center", va="center",
                        fontsize=FS_ANNOT, fontweight="bold", color="#641e16")
            else:
                ax.text(c, r, f"({r},{c})", ha="center", va="center",
                        fontsize=FS_TICK - 1, color="0.35")
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.set_xticks(np.arange(-0.5, N_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_ROWS, 1), minor=True)
    ax.grid(which="minor", color="0.6", lw=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title("States + terminals", fontsize=FS_AX_TITLE - 1)

    ax = axes[1]
    ax.axis("off")
    ax.set_title("Rules", fontsize=FS_AX_TITLE - 1)
    ax.text(
        0.5, 0.55,
        f"Actions: N E S W\n"
        f"Reward: -1 each step (0 on terminal)\n"
        f"Discount: gamma = {GAMMA}\n"
        f"Transitions: deterministic; bump = stay\n\n"
        "Start policy: always East\n"
        "(a bad habit -- walks into walls).\n\n"
        "Goal of PI: arrows that lead to a terminal\n"
        "as efficiently as the discounted objective wants.",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor=C_MUTE, edgecolor="0.5"),
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 9: Demo across iterations
# -------------------------------------------------------
def build_step_9():
    n_show = min(4, len(PI_HISTORY))
    fig, axes = new_figure(1, n_show)
    if n_show == 1:
        axes = [axes]
    fig.suptitle(
        f"Step 9: Policy iteration on the grid -- {len(PI_HISTORY)} outer iteration(s)",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    for i in range(n_show):
        # show evenly spaced including last
        idx = int(round(i * (len(PI_HISTORY) - 1) / max(n_show - 1, 1)))
        pi, V = PI_HISTORY[idx]
        draw_grid_policy(axes[i], V, pi, title=f"after PE #{idx}")
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 10: Slider over iterations
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: Scrub policy-iteration iterations",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.20, left=0.08, right=0.95, top=0.88)
    ax.set_navigate(False)

    pi0, V0 = PI_HISTORY[0]
    im = draw_grid_policy(ax, V0, pi0, title="iteration 0")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FS_TICK)

    ax_sl = fig.add_axes([0.22, 0.08, 0.55, 0.04])
    ax_sl.set_navigate(False)
    ax_sl.set_zorder(20)
    n_it = len(PI_HISTORY) - 1
    sl = Slider(ax_sl, "PI iter", 0, max(n_it, 0), valinit=0, valstep=1, color=C_LOOP)

    def _update(_=None):
        k = int(sl.val)
        pi, V = PI_HISTORY[k]
        ax.clear()
        draw_grid_policy(ax, V, pi, title=f"after PE #{k}")
        fig.canvas.draw_idle()

    sl.on_changed(_update)
    fig._sliders = (sl,)
    return fig


def value_iteration(max_iters=200, theta=THETA):
    """Bellman optimality iteration. Returns (V_history, final greedy pi)."""
    nS = N_ROWS * N_COLS
    V = np.zeros(nS)
    history = [V.copy()]
    for _ in range(max_iters):
        delta = 0.0
        V_new = V.copy()
        for s in range(nS):
            if is_terminal(s):
                V_new[s] = 0.0
                continue
            qs = [q_from_v(V, s, a) for a in range(4)]
            v = max(qs)
            delta = max(delta, abs(v - V[s]))
            V_new[s] = v
        V = V_new
        history.append(V.copy())
        if delta < theta:
            break
    pi = policy_improvement(V)
    return history, pi


VI_HISTORY, VI_PI = value_iteration()


# -------------------------------------------------------
# Step 11: Value iteration
# -------------------------------------------------------
def build_step_11():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 11: Value iteration -- $V(s)\leftarrow\max_a Q(s,a)$ each sweep",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    ax = axes[0]
    ax.axis("off")
    ax.set_title("Algorithm", fontsize=FS_AX_TITLE - 1, fontweight="bold")
    ax.text(
        0.5, 0.55,
        "Initialize V(s) arbitrarily (0 OK).\n\n"
        "Repeat sweeps over all states:\n"
        r"  $V(s)\leftarrow\max_a\sum_{s',r}"
        r"p(s',r|s,a)[r+\gamma V(s')]$"
        "\n"
        "until the max change $<\\theta$.\n\n"
        "Then recover a greedy policy:\n"
        r"  $\pi(s)\in\mathrm{arg\,max}_a Q(s,a)$."
        "\n\n"
        "No explicit policy during the backups --\n"
        "only at the end.",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=FS_ANNOT - 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor=C_GOAL, edgecolor=C_OK),
    )

    ax = axes[1]
    # show late VI value + greedy arrows
    V = VI_HISTORY[-1]
    draw_grid_policy(
        ax, V, VI_PI,
        title=f"VI result after {len(VI_HISTORY)-1} sweeps",
    )
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 12: vs value iteration
# -------------------------------------------------------
def build_step_12():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 12: Policy iteration vs value iteration",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 4.35, 5.5, 3.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 7.2, "Policy iteration", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 5.6,
        "Outer loop over policies.\n"
        "Each iteration: full PE\n"
        "(many Bellman sweeps)\n"
        "+ one greedy improve.\n\n"
        "Few outer iters typically.\n"
        "Expensive accurate PE.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 4.35, 5.5, 3.3, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 7.2, "Value iteration", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 5.6,
        "No explicit policy until the end.\n"
        "Each sweep: Bellman optimality\n"
        r"backup $V(s)\leftarrow\max_a Q(s,a)$."
        "\n\n"
        "Cheaper per sweep.\n"
        "Policy is recovered by\n"
        "a final greedy w.r.t. $V^*$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 0.4, 0.4, 11.2, 3.5, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 2.15,
        "Both converge to an optimal policy in finite MDPs.\n"
        "PI: stabilize the policy, then score it exactly (or nearly).\n"
        "VI: improve the value table with greedy backups every sweep.\n\n"
        "Truncated PI / optimistic PE sit in between:\n"
        "do only a few PE sweeps, then improve.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 13: Takeaway
# -------------------------------------------------------
def build_step_13():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 13: Takeaway",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank(ax)

    rows = [
        (6.2, "Evaluate", r"Fix $\pi$; solve Bellman expectation for $V^\pi$ / $Q^\pi$."),
        (4.85, "Improve", r"Greedy one-step lookahead: $\pi'(s)=\arg\max_a Q^\pi(s,a)$."),
        (3.5, "Guarantee", r"Policy improvement theorem: $V^{\pi'}\geq V^\pi$ everywhere."),
        (2.15, "Value iteration", r"Or skip the policy: backup $V(s)\leftarrow\max_a Q(s,a)$ each sweep."),
        (0.8, "Next", "When the model is unknown: learn from samples -- TD, Q-learning, policy gradients."),
    ]
    for y, title, body in rows:
        round_box(ax, 0.4, y, 11.2, 1.15, C_NEW, C_LOOP, lw=1.6)
        ax.text(2.0, y + 0.55, title, ha="center", va="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(7.3, y + 0.55, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")
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
    build_step_11,
    build_step_12,
    build_step_13,
])


print("\n" + "=" * 60)
print("POLICY ITERATION")
print("=" * 60)
print(f"""
Loop:
  1) Policy evaluation: compute V^pi (solve / iterate Bellman expectation)
  2) Policy improvement: pi <- greedy w.r.t. Q^pi
  3) Stop when pi does not change

Gridworld demo: {len(PI_HISTORY)} outer iteration(s), gamma={GAMMA}.
""")
