"""
6_06_policy_value_iteration.py

Policy iteration in detail -- evaluate a policy, improve it, repeat.

Sketch:
  1. What policy iteration is
  2. Reminder: MDP, V^pi, Q^pi
  3. Policy evaluation -- Bellman expectation
  4. Toy gridworld setup
  5. Iterative policy evaluation (sweeps)
  6. Policy improvement -- greedy w.r.t. Q^pi
  7. Policy improvement theorem
  8. The full loop
  9. Policy iteration: algorithm and demonstration
  10. Value iteration: algorithm and demonstration
  11. Policy iteration vs value iteration
  12. Takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.widgets import Button, Slider
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
    """Return list of (pi, V) after each policy-evaluation.

    history[0] = (π₀, V^{π₀}) after evaluating the *initial* policy
    (always East). It is NOT an arbitrary zero vector -- PE has already run.
    history[k] = (π_k, V^{π_k}) after the k-th evaluate step of PI.
    """
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


def pi_frame_title(k: int) -> str:
    """Explain what board k shows (especially k=0). Two lines for tight layouts."""
    if k == 0:
        # Values are computed by PE from V=0 -- not observed / not given.
        return "PE #0: computed $V^{\\pi_0}$\nfrom $V$=0 (only terminals known)"
    return f"PE #{k}: computed $V^{{\\pi_{k}}}$\nimprove $\\rightarrow$ re-evaluate"


def build_pi_scrub_frames(max_iters=20):
    """Fine-grained PI timeline for Step 10: Init -> PE_k -> PI_k -> PE_{k+1} -> ...

    Each frame is a dict: {pi, V, title, note, kind, k}.
    """
    nS = N_ROWS * N_COLS
    pi = np.ones(nS, dtype=int)  # always East
    V = np.zeros(nS)
    frames = [{
        "pi": pi.copy(),
        "V": V.copy(),
        "kind": "init",
        "k": 0,
        "title": "Init: $\\pi_0$=always East\n$V\\leftarrow 0$ (only terminals known)",
        "note": r"Start of the algorithm: policy fixed, values not yet evaluated.",
    }]
    for k in range(max_iters):
        V = policy_evaluation(pi)
        frames.append({
            "pi": pi.copy(),
            "V": V.copy(),
            "kind": "PE",
            "k": k,
            "title": (
                f"Just did PE #{k}\n"
                f"computed $V^{{\\pi_{k}}}$ (arrows=$\\pi_{k}$)"
            ),
            "note": (
                f"PE #{k}: Bellman backups for fixed $\\pi_{k}$ "
                "produced these values."
            ),
        })
        pi_new = policy_improvement(V)
        if np.array_equal(pi_new, pi):
            frames.append({
                "pi": pi.copy(),
                "V": V.copy(),
                "kind": "stop",
                "k": k,
                "title": (
                    f"Just did PI #{k}: $\\pi$ unchanged\n"
                    "stop -- optimal policy found"
                ),
                "note": (
                    f"PI #{k} left $\\pi_{k}$ unchanged "
                    "$\\Rightarrow$ policy iteration stops."
                ),
            })
            break
        pi = pi_new
        frames.append({
            "pi": pi.copy(),
            "V": V.copy(),  # still V^{π_k}; PE of new π comes next
            "kind": "PI",
            "k": k,
            "title": (
                f"Just did PI #{k}\n"
                f"new $\\pi_{{{k+1}}}$ (values still $V^{{\\pi_{k}}}$)"
            ),
            "note": (
                f"PI #{k}: greedy improve w.r.t. $V^{{\\pi_{k}}}$. "
                "Arrows changed; values wait for the next PE."
            ),
        })
    return frames


PI_SCRUB_FRAMES = build_pi_scrub_frames()


def _cm_to_data_y(ax, cm: float) -> float:
    """Convert a vertical length in cm to imshow data units on this axes."""
    bbox = ax.get_position()
    fig_h_in = ax.figure.get_size_inches()[1]
    ax_h_in = max(bbox.height * fig_h_in, 0.1)
    # imshow shows about N_ROWS data units vertically.
    return (cm / 2.54) * (N_ROWS / ax_h_in)


def _shrink_axes(ax, scale: float = 0.85) -> None:
    """Shrink an axes box by ``scale`` while keeping its center fixed."""
    pos = ax.get_position()
    w, h = pos.width * scale, pos.height * scale
    ax.set_position([
        pos.x0 + (pos.width - w) / 2,
        pos.y0 + (pos.height - h) / 2,
        w,
        h,
    ])


def draw_grid_policy(ax, V, pi=None, title="", up_extra_down_cm: float = 0.0,
                     show_policy: bool = True):
    """Heatmap of V; optional policy arrows (off during VI sweeps)."""
    Vm = V.reshape(N_ROWS, N_COLS)
    im = ax.imshow(Vm, cmap="Blues", vmin=V.min(), vmax=max(V.max(), 1e-6))
    tile = 1.0  # each cell is 1x1 in data coords
    # Shift arrows down (imshow origin=upper) so they clear the value text.
    arrow_down = 0.22 * tile
    up_extra = _cm_to_data_y(ax, up_extra_down_cm) if up_extra_down_cm else 0.0
    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = state_id(r, c)
            ax.text(
                c, r, f"{V[s]:.1f}", ha="center", va="center",
                fontsize=FS_TICK - 1, color="k",
            )
            if is_terminal(s):
                ax.add_patch(Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, fill=False,
                    edgecolor="#c0392b", linewidth=2.5,
                ))
                continue
            if not show_policy or pi is None:
                continue
            a = int(pi[s])
            dr, dc = ACTIONS[a]
            # Action 0 = North (up on screen); optional extra downward nudge.
            extra = up_extra if a == 0 else 0.0
            ax.annotate(
                "",
                xy=(c + 0.30 * dc, r + 0.30 * dr + arrow_down + extra),
                xytext=(c - 0.05 * dc, r - 0.05 * dr + arrow_down + extra),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.4),
            )
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Two-line titles need a bit more pad so neighboring boards do not collide.
    title_pad = 10 if "\n" in (title or "") else 4
    ax.set_title(title, fontsize=FS_AX_TITLE - 3, fontweight="bold", pad=title_pad)
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

    # +0.5 cm between lines; +2 cm box height (from previous Step-2 sizing).
    _, h_in = fig.get_size_inches()
    scale = 8.0 / max(h_in * 0.75, 1.0)  # blank y: 0..8
    dy = 0.42 + (0.5 / 2.54) * scale
    dh_2cm = (2.0 / 2.54) * scale
    prev_mid_h = max(2.6 - dh_2cm, 1.5)  # height after the earlier -2 cm change
    mid_h = prev_mid_h + dh_2cm            # extend +2 cm from that
    mid_lines = [
        r"Bellman expectation (one-step):",
        r"$V^\pi(s)=\sum_a\pi(a|s)\sum_{s',r}"
        r"p(s',r|s,a)\,[r+\gamma V^\pi(s')]$",
        r"$Q^\pi(s,a)=\sum_{s',r}p(s',r|s,a)\,[r+\gamma V^\pi(s')]$",
    ]
    mid_y = 3.85 - mid_h / 2
    round_box(ax, 0.4, mid_y, 11.2, mid_h, C_MUTE, "0.45", lw=1.6)
    y_top = 3.85 + dy
    for k, line in enumerate(mid_lines):
        ax.text(
            6.0, y_top - k * dy, line,
            ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
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
# Step 5: Iterative PE picture
# -------------------------------------------------------
def build_step_4():
    # Top: one worked backup with numbers. Bottom: V=0 -> 1 sweep -> converged.
    w_in, h_in = screen_figsize()
    fig = plt.figure(figsize=(w_in, h_in), dpi=FIGURE_DPI)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.35], hspace=0.38, wspace=0.28)
    fig.suptitle(
        "Step 5: How PE computes the numbers (arrows = fixed $\\pi$, not greedy on $V$)",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    pi0 = np.ones(N_ROWS * N_COLS, dtype=int)  # always East
    V = np.zeros(N_ROWS * N_COLS)
    V0 = V.copy()

    # One concrete cell for a worked example: (1,0) -- East goes to (1,1).
    ex_r, ex_c = 1, 0
    ex_s = state_id(ex_r, ex_c)
    ex_a = int(pi0[ex_s])  # East
    sp0, r0 = step(ex_s, ex_a)
    spr, spc = rc_of(sp0)

    def one_sweep(V_in):
        V_new = V_in.copy()
        delta = 0.0
        for s in range(N_ROWS * N_COLS):
            if is_terminal(s):
                V_new[s] = 0.0
                continue
            v = q_from_v(V_in, s, int(pi0[s]))
            delta = max(delta, abs(v - V_in[s]))
            V_new[s] = v
        return V_new, delta

    V1, _ = one_sweep(V0)
    # Continue to convergence for the final board
    V = V1.copy()
    for _ in range(200):
        V, delta = one_sweep(V)
        if delta < THETA:
            break

    # Worked numeric backup using V after sweep 0 (all zeros)
    v_before = float(V0[ex_s])
    v_after_1 = float(V1[ex_s])  # -1 + gamma * V0[sp] = -1
    v_sp_before = float(V0[sp0])
    backup_1 = r0 + GAMMA * v_sp_before

    ax_ex = fig.add_subplot(gs[0, :])
    ax_ex.axis("off")
    ax_ex.text(
        0.5, 0.55,
        "We only know terminal values (= 0). Everything else is computed.\n\n"
        rf"Fixed policy $\pi_0$: always East (red arrows). "
        rf"Pick state ({ex_r},{ex_c})."
        "\n"
        rf"$\pi_0$ says: go East $\rightarrow$ land in ({spr},{spc}), reward $r={r0:.0f}$."
        "\n\n"
        rf"Bellman backup:  $V({ex_r},{ex_c})"
        rf"\;\leftarrow\; r + \gamma V({spr},{spc})"
        rf"\;=\; {r0:.0f} + {GAMMA:g}\cdot {v_sp_before:.1f}"
        rf"\;=\; {backup_1:.1f}$"
        "\n"
        rf"(from the all-zero board). After many sweeps this settles to "
        rf"$V^{{\pi_0}}({ex_r},{ex_c})={V[ex_s]:.1f}$."
        "\n\n"
        "Important: arrows show the fixed $\\pi$ being evaluated -- "
        "they need NOT point toward higher $V$ neighbors yet.\n"
        "Pointing uphill in $V$ is policy improvement (next steps).",
        transform=ax_ex.transAxes, ha="center", va="center",
        fontsize=FS_TICK + 1,
        bbox=dict(boxstyle="round,pad=0.55", facecolor=C_NEW, edgecolor=C_LOOP),
    )

    ax0 = fig.add_subplot(gs[1, 0])
    draw_grid_policy(ax0, V0, pi0, title=r"Start $V$=0 (only T known)")
    # Highlight the example cell
    ax0.add_patch(Rectangle(
        (ex_c - 0.5, ex_r - 0.5), 1, 1, fill=False,
        edgecolor="#f1c40f", linewidth=3.0, zorder=5,
    ))

    ax1 = fig.add_subplot(gs[1, 1])
    draw_grid_policy(
        ax1, V1, pi0,
        title=rf"After 1 sweep: ({ex_r},{ex_c})$\rightarrow${v_after_1:.1f}",
    )
    ax1.add_patch(Rectangle(
        (ex_c - 0.5, ex_r - 0.5), 1, 1, fill=False,
        edgecolor="#f1c40f", linewidth=3.0, zorder=5,
    ))

    ax2 = fig.add_subplot(gs[1, 2])
    draw_grid_policy(
        ax2, V, pi0,
        title=rf"After PE: $V^{{\pi_0}}$ (still $\pi$=East)",
    )
    ax2.add_patch(Rectangle(
        (ex_c - 0.5, ex_r - 0.5), 1, 1, fill=False,
        edgecolor="#f1c40f", linewidth=3.0, zorder=5,
    ))
    fig.colorbar(ax2.images[0], ax=ax2, fraction=0.046, pad=0.04)
    fig.subplots_adjust(left=0.05, right=0.96, top=0.88, bottom=0.06)
    return fig


# -------------------------------------------------------
# Step 6: Policy improvement
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 6: Policy improvement -- greedy one-step lookahead",
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
# Step 7: Improvement theorem
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: Policy improvement theorem",
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
# Step 8: Full loop
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: Policy iteration -- the full loop",
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
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2.4),
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
# Step 4: Gridworld setup
# -------------------------------------------------------
def build_step_8():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 4: Toy gridworld for the demo",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    # Same Blues board background as value/policy grids in later steps.
    blank = np.zeros((N_ROWS, N_COLS))
    ax.imshow(blank, cmap="Blues", vmin=0.0, vmax=1.0)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = state_id(r, c)
            if is_terminal(s):
                ax.add_patch(Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#f5b7b1", edgecolor="#c0392b", lw=2.5,
                ))
                ax.text(c, r, "T", ha="center", va="center",
                        fontsize=FS_ANNOT, fontweight="bold", color="#641e16")
            else:
                ax.text(c, r, f"({r},{c})", ha="center", va="center",
                        fontsize=FS_TICK - 1, color="#1a5276")
    ax.set_xticks(range(N_COLS))
    ax.set_yticks(range(N_ROWS))
    ax.set_xticks(np.arange(-0.5, N_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_ROWS, 1), minor=True)
    ax.grid(which="minor", color="0.25", linestyle="-", linewidth=1.4)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.2")
        spine.set_linewidth(1.6)
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
        "Known a priori: terminal V = 0 only.\n"
        "Other V(s) are unknown -- PE computes them.\n\n"
        "Start policy: always East\n"
        "(a bad habit -- walks into walls).\n\n"
        "Goal of PI: arrows that lead to a terminal\n"
        "as efficiently as the discounted objective wants.",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round,pad=0.55", facecolor=C_MUTE, edgecolor="0.5"),
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig



# -------------------------------------------------------
# Step 9: Step through PI
# -------------------------------------------------------
def build_step_10():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 9: Policy iteration: algorithm and demonstration",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.20, left=0.05, right=0.96, top=0.88, wspace=0.28)

    frames = PI_SCRUB_FRAMES

    # Left: algorithm reminder
    ax_alg = axes[0]
    ax_alg.axis("off")
    ax_alg.set_navigate(False)
    ax_alg.set_title("Policy iteration algorithm", fontsize=FS_AX_TITLE - 1, fontweight="bold")
    alg = (
        r"Initialize $\pi_0$ (here: always East)." "\n"
        r"Initialize $V \leftarrow 0$ (terminals stay 0)." "\n"
        "\n"
        r"for $k = 0, 1, 2, \ldots$" "\n"
        r"    1. Policy evaluation (PE #$k$):" "\n"
        r"           repeat for fixed $\pi_k$:" "\n"
        r"               $V(s) \leftarrow \sum_{s',r}"
        r" p(s',r|s,\pi_k(s))\,[r+\gamma V(s')]$" "\n"
        r"           until max change $< \theta$" "\n"
        "\n"
        r"    2. Policy improvement (PI #$k$):" "\n"
        r"           $\pi_{k+1}(s) \in \mathrm{arg\,max}_a Q(s,a)$" "\n"
        r"           using the $V^{\pi_k}$ just computed" "\n"
        "\n"
        r"    3. if $\pi_{k+1} = \pi_k$: stop" "\n"
        "\n"
        r"Next / Prev / slider:" "\n"
        r"    frame 0 = Init" "\n"
        r"    then PE #0, PI #0, PE #1, PI #1, ..."
    )
    # Anchor the algorithm box directly under the axes title.
    ax_alg.text(
        0.06, 0.97, alg,
        transform=ax_alg.transAxes, ha="left", va="top",
        fontsize=FS_TICK,
        family="monospace", linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.55", facecolor=C_NEW, edgecolor=C_LOOP),
    )

    # Right: interactive board -- start at Init (V=0, π₀)
    ax = axes[1]
    ax.set_navigate(False)
    fr0 = frames[0]
    im = draw_grid_policy(
        ax, fr0["V"], fr0["pi"], title=fr0["title"],
        up_extra_down_cm=0.5,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FS_TICK)
    note = fig.text(
        0.72, 0.145, fr0["note"],
        ha="center", va="center", fontsize=FS_TICK,
        color="#333333",
    )

    # Controls under the right panel
    ax_prev = fig.add_axes([0.52, 0.07, 0.08, 0.055])
    ax_sl = fig.add_axes([0.62, 0.08, 0.22, 0.04])
    ax_next = fig.add_axes([0.86, 0.07, 0.08, 0.055])
    for a in (ax_prev, ax_sl, ax_next):
        a.set_navigate(False)
        a.set_zorder(20)

    n_fr = len(frames) - 1
    sl = Slider(ax_sl, "step", 0, max(n_fr, 0), valinit=0, valstep=1, color=C_LOOP)
    btn_prev = Button(ax_prev, "Prev", color="#f5f5f5", hovercolor="#f5b7b1")
    btn_next = Button(ax_next, "Next", color="#f5f5f5", hovercolor="#f5b7b1")
    btn_prev.label.set_fontsize(FS_TICK)
    btn_next.label.set_fontsize(FS_TICK)

    def _update(_=None):
        i = int(sl.val)
        fr = frames[i]
        ax.clear()
        draw_grid_policy(
            ax, fr["V"], fr["pi"], title=fr["title"],
            up_extra_down_cm=0.5,
        )
        note.set_text(fr["note"])
        fig.canvas.draw_idle()

    def _step(delta):
        i = int(np.clip(sl.val + delta, 0, max(n_fr, 0)))
        sl.set_val(i)

    sl.on_changed(_update)
    btn_prev.on_clicked(lambda _e: _step(-1))
    btn_next.on_clicked(lambda _e: _step(+1))
    fig._sliders = (sl,)
    fig._retained_widgets = [sl, btn_prev, btn_next]

    # Keep left panel top fixed under its title; only grow downward if needed.
    # Shift right board left; shrink the grid board (smaller cells).
    w_in, h_in = fig.get_size_inches()
    dh_left = (2.0 / 2.54) / max(h_in, 0.1)
    dx_right = (5.5 / 2.54) / max(w_in, 0.1)
    pos = ax_alg.get_position()
    ax_alg.set_position([
        pos.x0,
        pos.y0 - dh_left,
        pos.width,
        pos.height + dh_left,
    ])
    for a in (ax, cbar.ax):
        pos = a.get_position()
        a.set_position([
            pos.x0 - dx_right,
            pos.y0,
            pos.width,
            pos.height,
        ])
    # Smaller grid cells (~40% smaller board).
    _shrink_axes(ax, 0.60)
    _shrink_axes(cbar.ax, 0.60)
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


def build_vi_scrub_frames():
    """Fine-grained VI timeline: Init V=0, then each optimality sweep.

    Policy arrows appear ONLY on the final frame -- matching the algorithm
    (VI updates V only; pi is recovered after convergence).
    """
    frames = []
    for i, V in enumerate(VI_HISTORY):
        if i == 0:
            frames.append({
                "V": V.copy(),
                "pi": None,
                "show_policy": False,
                "title": "Init: $V\\leftarrow 0$\n(no policy yet)",
                "note": (
                    "Start of value iteration: only $V$ exists. "
                    "No policy arrows until the end."
                ),
            })
        else:
            frames.append({
                "V": V.copy(),
                "pi": None,
                "show_policy": False,
                "title": (
                    f"Just did VI sweep #{i}\n"
                    r"$V(s)\leftarrow\max_a Q(s,a)$  (still no $\pi$)"
                ),
                "note": (
                    f"Sweep #{i}: only values change. "
                    r"Policy is NOT updated during VI sweeps."
                ),
            })
    frames.append({
        "V": VI_HISTORY[-1].copy(),
        "pi": VI_PI.copy(),
        "show_policy": True,
        "title": (
            f"VI done after {len(VI_HISTORY)-1} sweeps\n"
            r"NOW recover greedy $\pi$ from $V^*$"
        ),
        "note": (
            r"Only now: $\pi(s)\in\mathrm{arg\,max}_a Q(s,a)$ "
            "from the converged $V$. Red arrows appear."
        ),
    })
    return frames


VI_SCRUB_FRAMES = build_vi_scrub_frames()


# -------------------------------------------------------
# Step 10: Value iteration (algorithm + demonstration)
# -------------------------------------------------------
def build_step_11():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 10: Value iteration: algorithm and demonstration",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.20, left=0.05, right=0.96, top=0.88, wspace=0.28)

    frames = VI_SCRUB_FRAMES

    # Left: algorithm
    ax_alg = axes[0]
    ax_alg.axis("off")
    ax_alg.set_navigate(False)
    ax_alg.set_title("Value iteration algorithm", fontsize=FS_AX_TITLE - 1, fontweight="bold")
    alg = (
        r"There is NO policy during value iteration." "\n"
        r"(Unlike policy iteration, we do not store $\pi$.)" "\n"
        "\n"
        r"Initialize array $V$ arbitrarily (here: 0)." "\n"
        r"Keep $V(\mathrm{terminal})=0$." "\n"
        "\n"
        r"repeat:                         # one sweep" "\n"
        r"    $\Delta \leftarrow 0$" "\n"
        r"    for each state $s$:" "\n"
        r"        $v \leftarrow V(s)$" "\n"
        r"        $V(s) \leftarrow \max_a \sum_{s',r}"
        r" p(s',r|s,a)\,[r+\gamma V(s')]$" "\n"
        r"        $\Delta \leftarrow \max(\Delta, |v-V(s)|)$" "\n"
        r"until $\Delta < \theta$" "\n"
        "\n"
        r"ONLY THEN output a greedy policy:" "\n"
        r"    $\pi(s) \leftarrow \mathrm{arg\,max}_a"
        r" \sum_{s',r} p(s',r|s,a)\,[r+\gamma V(s')]$" "\n"
        "\n"
        r"What $\max_a$ does: pick the best action" "\n"
        r"inside the backup -- it does not store $\pi$." "\n"
        "\n"
        r"Next / Prev / slider:" "\n"
        r"    Init / sweeps = $V$ only (no arrows)" "\n"
        r"    last frame = finally extract $\pi$"
    )
    # Anchor the algorithm box directly under the axes title.
    ax_alg.text(
        0.06, 0.97, alg,
        transform=ax_alg.transAxes, ha="left", va="top",
        fontsize=FS_TICK,
        family="monospace", linespacing=1.45,
        bbox=dict(boxstyle="round,pad=0.55", facecolor=C_GOAL, edgecolor=C_OK),
    )

    # Right: interactive board
    ax = axes[1]
    ax.set_navigate(False)
    fr0 = frames[0]
    im = draw_grid_policy(
        ax, fr0["V"], fr0["pi"], title=fr0["title"],
        up_extra_down_cm=1.5,
        show_policy=fr0["show_policy"],
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FS_TICK)
    note = fig.text(
        0.72, 0.145, fr0["note"],
        ha="center", va="center", fontsize=FS_TICK,
        color="#333333",
    )

    ax_prev = fig.add_axes([0.52, 0.07, 0.08, 0.055])
    ax_sl = fig.add_axes([0.62, 0.08, 0.22, 0.04])
    ax_next = fig.add_axes([0.86, 0.07, 0.08, 0.055])
    for a in (ax_prev, ax_sl, ax_next):
        a.set_navigate(False)
        a.set_zorder(20)

    n_fr = len(frames) - 1
    sl = Slider(ax_sl, "step", 0, max(n_fr, 0), valinit=0, valstep=1, color=C_OK)
    btn_prev = Button(ax_prev, "Prev", color="#f5f5f5", hovercolor="#d5f5e3")
    btn_next = Button(ax_next, "Next", color="#f5f5f5", hovercolor="#d5f5e3")
    btn_prev.label.set_fontsize(FS_TICK)
    btn_next.label.set_fontsize(FS_TICK)

    def _update(_=None):
        i = int(sl.val)
        fr = frames[i]
        ax.clear()
        draw_grid_policy(
            ax, fr["V"], fr["pi"], title=fr["title"],
            up_extra_down_cm=1.5,
            show_policy=fr["show_policy"],
        )
        note.set_text(fr["note"])
        fig.canvas.draw_idle()

    def _step(delta):
        i = int(np.clip(sl.val + delta, 0, max(n_fr, 0)))
        sl.set_val(i)

    sl.on_changed(_update)
    btn_prev.on_clicked(lambda _e: _step(-1))
    btn_next.on_clicked(lambda _e: _step(+1))
    fig._sliders = (sl,)
    fig._retained_widgets = [sl, btn_prev, btn_next]

    # Keep left panel top fixed under its title; only grow downward.
    # Shift right board left; shrink the grid board (smaller cells).
    w_in, h_in = fig.get_size_inches()
    dh_left = (2.0 / 2.54) / max(h_in, 0.1)
    dx_right = (2.5 / 2.54) / max(w_in, 0.1)
    pos = ax_alg.get_position()
    ax_alg.set_position([
        pos.x0,
        pos.y0 - dh_left,
        pos.width,
        pos.height + dh_left,
    ])
    for a in (ax, cbar.ax):
        pos = a.get_position()
        a.set_position([
            pos.x0 - dx_right,
            pos.y0,
            pos.width,
            pos.height,
        ])
    # Smaller grid cells (~40% smaller board).
    _shrink_axes(ax, 0.60)
    _shrink_axes(cbar.ax, 0.60)
    return fig


# -------------------------------------------------------
# Step 11: vs value iteration
# -------------------------------------------------------
def build_step_12():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 11: Policy iteration vs value iteration",
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
# Step 12: Takeaway
# -------------------------------------------------------
def build_step_13():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 12: Takeaway",
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
    build_step_8,  # define the toy gridworld first
    build_step_4,  # then run PE on it
    build_step_5,
    build_step_6,
    build_step_7,
    build_step_10,  # step-through Init / PE / PI (was Step 10, now Step 9)
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
