"""
The Bellman equality -- intuitively.

After : RL cares about return over time.
Bellman says: the value of "now" equals reward now plus (discounted) value of "next".

Sketch:
  1. MDP assumption -- the usual RL theory setting
  2. Reminder -- return is a sum over the future
  3. Value of a state -- how good is it to be here?
  4. The recursive idea -- today = today\'s reward + tomorrow
  5. Bellman equality for V^pi (the formula)
  6. Tiny path cartoon -- unroll one step
  7. Visual -- grid values that already obey Bellman
  8. Action values Q -- same idea, one action deeper
  9. Why gamma?  (patience / horizon)
  10. Bootstrapping -- learn by trusting your own next estimate
  11. Optimality glance -- Bellman optimality (max)
  12. Policy evaluation -- score a fixed pi with Bellman
  13. Policy improvement -- act greedily w.r.t. V^pi / Q^pi
  14. Policy iteration -- alternate evaluate and improve
  15. Visual intuition -- arrows flip toward better actions
  16. Takeaway
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
import matplotlib.patches as patches
import numpy as np

np.random.seed(301)
RNG = np.random.default_rng(301)


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
C_RL = "#fdebd0"


# -------------------------------------------------------
# FIGURE 0 / Step 1: MDP assumption
# -------------------------------------------------------
def build_step_1():
    fig0, ax0 = new_figure()
    fig0.suptitle(
        "Step 1: RL theory usually assumes an MDP",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax0)

    round_box(ax0, 0.4, 6.45, 11.2, 1.25, C_DARK, C_DARK, lw=0)
    ax0.text(
        6.0, 7.05,
        r"Markov Decision Process:  $(S, A, P, R, \gamma)$"
        "  —  the standard stage for Bellman equations.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    # five MDP pieces
    pieces = [
        (0.4, r"$S$", "states", C_NEW),
        (2.7, r"$A$", "actions", C_RL),
        (5.0, r"$P$", r"$p(s'|s,a)$", C_MUTE),
        (7.3, r"$R$", "rewards", C_GOAL),
        (9.6, r"$\gamma$", "discount", C_OLD),
    ]
    for x, sym, name, face in pieces:
        round_box(ax0, x, 4.35, 2.1, 1.75, face, "0.4", lw=1.5)
        ax0.text(x + 1.05, 5.55, sym, ha="center", fontsize=FS_ANNOT + 2, fontweight="bold")
        ax0.text(x + 1.05, 4.85, name, ha="center", fontsize=FS_TICK, fontweight="bold")

    round_box(ax0, 0.4, 2.35, 11.2, 1.7, "#f5eef8", C_FUN, lw=1.8)
    ax0.text(
        6.0, 3.2,
        r"Markov assumption:  $p(s_{t+1}, r_{t+1}\mid s_t, a_t,\;\mathrm{history})$"
        r"  $=\; p(s_{t+1}, r_{t+1}\mid s_t, a_t)$."
        "\n"
        "The present state is a sufficient summary — the past is already baked into $s_t$.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax0.text(
        6.0, 1.15,
        "Why it matters for Bellman: if tomorrow depends only on (state, action) now,\n"
        "then value(now) can be written using only reward-now + value(next) — no full history.\n"
        "Real life is often only approximately Markov; theory still starts here.",
        ha="center", fontsize=FS_TICK + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    return fig0


# -------------------------------------------------------
# FIGURE 1: Return reminder
# -------------------------------------------------------
def build_step_2():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 2: RL scores a whole future -- not one label",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax1)

    round_box(ax1, 0.4, 6.45, 11.2, 1.25, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 7.05,
        r"Return from time $t$:   "
        r"$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    # timeline
    for i, lab in enumerate([r"$s_t$", r"$a_t$", r"$R_{t+1}$", r"$s_{t+1}$", r"$a_{t+1}$", r"$R_{t+2}$", r"$\cdots$"]):
        x = 0.7 + i * 1.55
        face = C_GOAL if "R" in lab else (C_RL if "a" in lab else C_NEW)
        round_box(ax1, x, 3.9, 1.4, 1.5, face, "0.4", lw=1.5)
        ax1.text(x + 0.7, 4.65, lab, ha="center", va="center",
                 fontsize=FS_TICK + 1, fontweight="bold")

    ax1.text(
        6.0, 2.4,
        "Supervised learning: one target per input.\n"
        "RL: credit is delayed — a good move may pay off many steps later.\n\n"
        "Bellman\'s question: can we rewrite that long sum as something local?",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )
    ax1.text(
        6.0, 0.55,
        "Companion: (agent / environment / reward loop).",
        ha="center", fontsize=FS_TICK, style="italic", color="0.35",
    )
    return fig1


# -------------------------------------------------------
# FIGURE 2: What is value
# -------------------------------------------------------
def build_step_3():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        r"Step 3: State value -- expected return from being in $s$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax2)

    round_box(ax2, 0.4, 6.4, 11.2, 1.3, C_DARK, C_DARK, lw=0)
    ax2.text(
        6.0, 7.05,
        r"$V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$"
        "   —  \"how good is state $s$, if I follow policy $\\pi$?\"",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax2, 0.45, 3.3, 5.5, 2.8, C_GOAL, C_OK, lw=1.8)
    ax2.text(3.2, 5.7, "High $V$", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax2.text(
        3.2, 4.4,
        "Near the goal,\n"
        "safe corridor,\n"
        "rich future rewards.\n\n"
        "Worth being here.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax2, 6.15, 3.3, 5.4, 2.8, C_OLD, C_ACCENT, lw=1.8)
    ax2.text(8.85, 5.7, "Low $V$", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
    ax2.text(
        8.85, 4.4,
        "Near a pit,\n"
        "dead end,\n"
        "long unpaid path.\n\n"
        "Better leave soon.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax2.text(
        6.0, 1.5,
        "Value is not the immediate reward alone.\n"
        "It is a forecast of the whole discounted future under $\\pi$.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    return fig2


# -------------------------------------------------------
# FIGURE 3: Recursive idea
# -------------------------------------------------------
def build_step_4():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        "Step 4: The recursive idea -- today includes tomorrow",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax3)

    round_box(ax3, 0.4, 6.45, 11.2, 1.25, C_DARK, C_DARK, lw=0)
    ax3.text(
        6.0, 7.05,
        r"$G_t = R_{t+1} + \gamma G_{t+1}$"
        "     (exact algebra of returns)",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#f9e79f",
    )

    # three boxes
    round_box(ax3, 0.5, 3.5, 3.4, 2.5, C_OLD, C_ACCENT, lw=1.6)
    ax3.text(2.2, 5.5, "Now", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
    ax3.text(2.2, 4.4, r"get $R_{t+1}$" "\n" "right after acting",
             ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

    round_box(ax3, 4.3, 3.5, 3.4, 2.5, C_NEW, C_DARK, lw=1.6)
    ax3.text(6.0, 5.5, "Then", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax3.text(6.0, 4.4, "land in $s'$" "\n" "and keep going",
             ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

    round_box(ax3, 8.1, 3.5, 3.4, 2.5, C_GOAL, C_OK, lw=1.6)
    ax3.text(9.8, 5.5, "Later", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax3.text(9.8, 4.4, r"all future return $G_{t+1}$" "\n" "already summarized",
             ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

    arrow(ax3, 3.9, 4.75, 4.3, 4.75, C_DARK, 2.0)
    arrow(ax3, 7.7, 4.75, 8.1, 4.75, C_DARK, 2.0)

    ax3.text(
        6.0, 1.6,
        "If return itself splits into \"reward now + discounted return next\",\n"
        "then the *expectation* of return — the value — must split the same way.\n"
        "That consistency is the Bellman equality.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )
    return fig3


# -------------------------------------------------------
# FIGURE 4: Bellman equality for V^pi
# -------------------------------------------------------
def build_step_5():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        r"Step 5: Bellman equality for $V^\pi$ -- the consistency law",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax4)

    round_box(ax4, 0.4, 5.85, 11.2, 1.85, C_DARK, C_DARK, lw=0)
    ax4.text(
        6.0, 6.75,
        r"$V^\pi(s)"
        r"= \mathbb{E}_\pi\!\left["
        r"R_{t+1} + \gamma V^\pi(S_{t+1})"
        r"\;|\; S_t=s"
        r"\right]$",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#f9e79f",
    )

    round_box(ax4, 0.45, 2.9, 5.5, 2.6, C_MUTE, "0.45", lw=1.5)
    ax4.text(3.2, 5.1, "In words", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax4.text(
        3.2, 3.9,
        "Value(now)\n"
        "= expected [\n"
        "   reward you get next\n"
        " + γ · value(next state)\n"
        "] under policy π.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax4, 6.15, 2.9, 5.4, 2.6, C_GOAL, C_OK, lw=1.8)
    ax4.text(8.85, 5.1, "Why \"equality\"?", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax4.text(
        8.85, 3.9,
        "Not a wish — an identity.\n"
        "If $V^\\pi$ is the true value\n"
        "of $\\pi$, both sides match.\n\n"
        "A wrong table of numbers\n"
        "will violate it.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax4.text(
        6.0, 1.3,
        r"Expanded with policy & transitions:"
        "\n"
        r"$V^\pi(s)=\sum_a\pi(a|s)\sum_{s',r}p(s',r|s,a)\,[r+\gamma V^\pi(s')]$",
        ha="center", fontsize=FS_TICK + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    return fig4


# -------------------------------------------------------
# FIGURE 5: One-step cartoon
# -------------------------------------------------------
def build_step_6():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        "Step 6: Cartoon -- one step of the equality",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax5)

    # states
    round_box(ax5, 0.8, 4.2, 3.0, 2.4, C_NEW, C_DARK, lw=2.0)
    ax5.text(2.3, 6.1, r"state $s$", ha="center", fontsize=FS_TICK, fontweight="bold")
    ax5.text(2.3, 5.2, r"$V(s)=?$", ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK)

    round_box(ax5, 8.2, 4.2, 3.0, 2.4, C_GOAL, C_OK, lw=2.0)
    ax5.text(9.7, 6.1, r"next $s'$", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax5.text(9.7, 5.2, r"$V(s')$ known", ha="center", fontsize=FS_ANNOT, fontweight="bold")

    arrow(ax5, 3.9, 5.4, 8.1, 5.4, C_ACCENT, 2.5)
    ax5.text(6.0, 6.0, r"action $a\sim\pi$", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
    ax5.text(6.0, 4.7, r"reward $r$", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT)

    round_box(ax5, 1.5, 1.0, 9.0, 2.5, C_MUTE, "0.45", lw=1.5)
    ax5.text(
        6.0, 2.25,
        r"Bellman check for this step:"
        "\n"
        r"$V(s) \;\;\approx\;\; r + \gamma V(s')$  ?"
        "   (or average, if next state is random)"
        "\n\n"
        "Intuition: the price of a house ≈ rent this year + discounted price next year.\n"
        "Value is a consistent pricing of states.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig5


# -------------------------------------------------------
# FIGURE 6: Grid visual
# -------------------------------------------------------
def build_step_7():
    fig6, (ax6a, ax6b) = new_figure(1, 2)
    fig6.suptitle(
        r"Step 7: Visual -- values that already satisfy Bellman ($\gamma=0.9$)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # Tiny 1D corridor: states 0,1,2,3 with reward +1 only at leaving 3 toward goal
    # Deterministic rightward policy. V(3)=1, V(2)=0.9, V(1)=0.81, V(0)=0.729
    vals = [0.73, 0.81, 0.90, 1.00]
    labels = ["s0", "s1", "s2", "s3 (near goal)"]

    ax6a.set_xlim(-0.5, 3.5)
    ax6a.set_ylim(-0.5, 1.5)
    ax6a.set_aspect("equal")
    ax6a.axis("off")
    ax6a.set_title("Corridor toward +1", fontsize=FS_AX_TITLE - 2)
    for i, (v, lab) in enumerate(zip(vals, labels)):
        face = plt.cm.YlGn(0.35 + 0.6 * v)
        ax6a.add_patch(patches.FancyBboxPatch(
            (i - 0.4, 0.15), 0.8, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=face, edgecolor="0.3", lw=1.5,
        ))
        ax6a.text(i, 0.75, f"{v:.2f}", ha="center", va="center",
                  fontsize=FS_ANNOT, fontweight="bold")
        ax6a.text(i, -0.15, lab, ha="center", va="top", fontsize=FS_TICK)
    for i in range(3):
        ax6a.annotate("", xy=(i + 0.55, 0.6), xytext=(i + 0.45, 0.6),
                      arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2))
    ax6a.text(1.5, 1.25, r"always move right; $R=+1$ entering goal after s3",
              ha="center", fontsize=FS_TICK)

    # Right: show one equality
    ax6b.axis("off")
    ax6b.set_xlim(0, 10)
    ax6b.set_ylim(0, 10)
    ax6b.set_title("Check one state", fontsize=FS_AX_TITLE - 2)
    round_box(ax6b, 0.5, 2.5, 9.0, 6.0, C_MUTE, "0.45", lw=1.5)
    ax6b.text(
        5.0, 7.2,
        r"At $s_1$ with $\gamma=0.9$:"
        "\n\n"
        r"$V(s_1)=0.81$"
        "\n"
        r"$R_{next}+\gamma V(s_2)=0 + 0.9\times 0.90=0.81$"
        "\n\n"
        "Sides match → Bellman equality holds.\n"
        "Each cell is priced consistently with its neighbor.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    fig6.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig6


# -------------------------------------------------------
# FIGURE 7: Q values
# -------------------------------------------------------
def build_step_8():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        r"Step 8: Same idea for actions -- $Q^\pi(s,a)$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax7)

    round_box(ax7, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax7.text(
        6.0, 7.0,
        r"$Q^\pi(s,a)"
        r"= \mathbb{E}_\pi\!\left["
        r"R_{t+1}+\gamma V^\pi(S_{t+1})"
        r"\;|\; S_t=s,\,A_t=a"
        r"\right]$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax7, 0.45, 3.2, 5.5, 2.8, C_NEW, C_DARK, lw=1.6)
    ax7.text(3.2, 5.6, r"$V$ = \"how good is here?\"", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax7.text(
        3.2, 4.3,
        "Averages over actions\n"
        r"the policy would take:"
        "\n\n"
        r"$V^\pi(s)=\sum_a\pi(a|s)\,Q^\pi(s,a)$",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax7, 6.15, 3.2, 5.4, 2.8, C_RL, "#d68910", lw=1.6)
    ax7.text(8.85, 5.6, r"$Q$ = \"how good is this move?\"", ha="center",
             fontsize=FS_TICK, fontweight="bold", color="#6e2c00")
    ax7.text(
        8.85, 4.3,
        "Fixes the first action,\n"
        "then follows $\\pi$.\n\n"
        "Natural for control:\n"
        "compare left vs right.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax7.text(
        6.0, 1.4,
        "Bellman for Q is the same story: immediate reward + discounted next value.\n"
        "Q-learning / SARSA / actor–critic all lean on this backup.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    return fig7


# -------------------------------------------------------
# FIGURE 8: Gamma
# -------------------------------------------------------
def build_step_9():
    fig8, (ax8a, ax8b) = new_figure(1, 2)
    fig8.suptitle(
        r"Step 9: What $\gamma$ does -- patience of the value",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    steps = np.arange(0, 12)
    for ax, gamma, title, color in (
        (ax8a, 0.5, r"$\gamma=0.5$  (myopic)", C_ACCENT),
        (ax8b, 0.95, r"$\gamma=0.95$  (patient)", C_OK),
    ):
        w = gamma ** steps
        ax.bar(steps, w, color=color, edgecolor="k", alpha=0.85)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("steps into the future", fontsize=FS_AXIS - 1)
        ax.set_ylabel("weight on that reward", fontsize=FS_AXIS - 1)
        ax.set_title(title, fontsize=FS_AX_TITLE - 2)
        ax.tick_params(labelsize=FS_TICK - 1)
        ax.grid(True, axis="y", alpha=0.3)

    fig8.text(
        0.5, 0.02,
        r"Small $\gamma$: care about now.  Large $\gamma$: distant rewards still matter in $V$.",
        ha="center", fontsize=FS_TICK, fontweight="bold",
    )
    fig8.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig8


# -------------------------------------------------------
# FIGURE 9: Bootstrapping
# -------------------------------------------------------
def build_step_10():
    fig9, ax9 = new_figure()
    fig9.suptitle(
        "Step 10: Why Bellman helps learning -- bootstrap from the next estimate",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax9)

    round_box(ax9, 0.4, 6.45, 11.2, 1.25, C_DARK, C_DARK, lw=0)
    ax9.text(
        6.0, 7.05,
        "You do not need the whole future trajectory every time — only a target built from $r$ and $V(s')$.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax9, 0.45, 3.35, 5.5, 2.75, C_OLD, C_ACCENT, lw=1.6)
    ax9.text(3.2, 5.7, "Monte Carlo style", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
    ax9.text(
        3.2, 4.4,
        "Wait until the episode ends.\n"
        "Average full returns $G_t$.\n\n"
        "Unbiased, but high variance\n"
        "and slow credit.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax9, 6.15, 3.35, 5.4, 2.75, C_GOAL, C_OK, lw=1.8)
    ax9.text(8.85, 5.7, "Bellman / TD style", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax9.text(
        8.85, 4.4,
        "Target ≈ $r + \\gamma V(s')$\n"
        "using current estimate.\n\n"
        "Update now; trust yourself\n"
        "a little (bootstrapping).",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax9.text(
        6.0, 1.5,
        r"TD idea:  $V(s) \leftarrow V(s) + \alpha\,[r+\gamma V(s')-V(s)]$"
        "\n"
        "The term in brackets is the Bellman residual — how wrong the equality currently is.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )
    return fig9


# -------------------------------------------------------
# FIGURE 10: Optimality glance
# -------------------------------------------------------
def build_step_11():
    fig10, ax10 = new_figure()
    fig10.suptitle(
        "Step 11: Glance -- Bellman optimality (best possible values)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax10)

    round_box(ax10, 0.4, 5.7, 11.2, 2.0, C_MUTE, "0.45", lw=1.6)
    ax10.text(
        6.0, 6.7,
        r"For the optimal $V^*$ (best policy):"
        "\n\n"
        r"$V^*(s)"
        r"= \max_a \sum_{s',r}p(s',r|s,a)\,[r+\gamma V^*(s')]$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    round_box(ax10, 0.45, 2.7, 5.5, 2.7, C_NEW, C_DARK, lw=1.6)
    ax10.text(3.2, 4.95, r"Equality for a fixed $\pi$", ha="center",
              fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax10.text(
        3.2, 3.7,
        r"$V^\pi$: average over $\pi$'s actions."
        "\n"
        "Evaluate how good a\n"
        "given behavior is.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax10, 6.15, 2.7, 5.4, 2.7, "#f5eef8", C_FUN, lw=1.8)
    ax10.text(8.85, 4.95, "Optimality equation", ha="center",
              fontsize=FS_TICK, fontweight="bold", color=C_FUN)
    ax10.text(
        8.85, 3.7,
        r"$V^*$: take the best action"
        "\n"
        "(the max)."
        "\n\n"
        "Control: improve the policy.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax10.text(
        6.0, 1.2,
        "Same recursive spine.  Evaluation uses expectation under π; optimality uses a max.\n"
        "Next: turn that into an algorithm — evaluate a policy, then improve it.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    return fig10


# -------------------------------------------------------
# FIGURE 12: Policy evaluation
# -------------------------------------------------------
def build_step_12():
    fig12, ax12 = new_figure()
    fig12.suptitle(
        r"Step 12: Policy evaluation -- score a fixed $\pi$ with Bellman",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax12)

    round_box(ax12, 0.4, 6.4, 11.2, 1.3, C_DARK, C_DARK, lw=0)
    ax12.text(
        6.0, 7.05,
        r"Hold $\pi$ fixed.  Solve the Bellman equality for the unknown table $V^\pi$.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax12, 0.45, 3.35, 5.5, 2.75, C_MUTE, "0.45", lw=1.6)
    ax12.text(3.2, 5.7, "Linear system", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax12.text(
        3.2, 4.4,
        r"For every $s$:"
        "\n"
        r"$V(s)=\sum_a\pi(a|s)\sum_{s',r}"
        r"p(s',r|s,a)[r+\gamma V(s')]$"
        "\n\n"
        r"$|S|$ equations, $|S|$ unknowns.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax12, 6.15, 3.35, 5.4, 2.75, C_GOAL, C_OK, lw=1.8)
    ax12.text(8.85, 5.7, "Or iterate", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax12.text(
        8.85, 4.4,
        r"$V_{k+1}(s) \leftarrow$"
        "\n"
        r"$\mathbb{E}_\pi[r+\gamma V_k(s')]$"
        "\n\n"
        "Sweep states until the\n"
        "Bellman residual is small.\n"
        "(iterative policy evaluation)",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax12.text(
        6.0, 1.5,
        "Output of evaluation: a report card $V^\\pi$ (or $Q^\\pi$) for the current behavior.\n"
        "No new policy yet — only \"how good is what we are already doing?\"",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    return fig12


# -------------------------------------------------------
# FIGURE 13: Policy improvement
# -------------------------------------------------------
def build_step_13():
    fig13, ax13 = new_figure()
    fig13.suptitle(
        r"Step 13: Policy improvement -- act greedily w.r.t. $V^\pi$ / $Q^\pi$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax13)

    round_box(ax13, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax13.text(
        6.0, 7.0,
        r"$\pi'(s) \in \mathrm{arg\,max}_a Q^\pi(s,a)"
        r"= \mathrm{arg\,max}_a \sum_{s',r}p(s',r|s,a)\,[r+\gamma V^\pi(s')]$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax13, 0.45, 3.2, 5.5, 2.8, C_NEW, C_DARK, lw=1.6)
    ax13.text(3.2, 5.55, "Idea", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax13.text(
        3.2, 4.25,
        "You already know $V^\\pi$.\n"
        "At each state, try every action\n"
        "for one step, then follow $\\pi$.\n\n"
        "Keep the action with the\n"
        "best one-step lookahead.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax13, 6.15, 3.2, 5.4, 2.8, C_GOAL, C_OK, lw=1.8)
    ax13.text(8.85, 5.55, "Policy improvement theorem", ha="center",
              fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax13.text(
        8.85, 4.25,
        r"If $\pi'$ is greedy w.r.t. $V^\pi$,"
        "\n"
        r"then $V^{\pi'}(s)\geq V^\pi(s)$"
        "\n"
        "for every $s$."
        "\n\n"
        "Greedy never hurts\n"
        "(and often helps).",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax13.text(
        6.0, 1.4,
        "Intuition: if a detour looks better than your current habit for one step,\n"
        "and afterward you behave at least as well as before, the whole future improves.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )
    return fig13


# -------------------------------------------------------
# FIGURE 14: Policy iteration
# -------------------------------------------------------
def build_step_14():
    fig14, ax14 = new_figure()
    fig14.suptitle(
        "Step 14: Policy iteration -- evaluate, improve, repeat",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax14)

    round_box(ax14, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax14.text(
        6.0, 7.1,
        r"$\pi_0 \rightarrow V^{\pi_0}$ (eval)  "
        r"$\rightarrow \pi_1$ (improve)  "
        r"$\rightarrow \cdots \rightarrow \pi^*$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    steps = [
        (5.9, r"1. Start with any policy $\pi$  (even random).", C_MUTE),
        (4.95, r"2. Policy evaluation: compute $V^\pi$ (solve / iterate Bellman).", C_NEW),
        (4.0, r"3. Policy improvement: $\pi \leftarrow$ greedy w.r.t. $V^\pi$ (or $Q^\pi$).", C_GOAL),
        (3.05, r"4. If $\pi$ did not change: stop (optimal).  Else go to 2.", C_OLD),
    ]
    for y, text, face in steps:
        round_box(ax14, 0.5, y - 0.4, 11.0, 0.85, face, "0.45", lw=1.3)
        ax14.text(6.0, y, text, ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

    round_box(ax14, 0.5, 0.4, 11.0, 2.15, "#f5eef8", C_FUN, lw=1.6)
    ax14.text(
        6.0, 1.45,
        "Finite MDPs: only finitely many deterministic policies,\n"
        "and each strict improve raises value — so the loop reaches $\\pi^*$ in finite steps.\n\n"
        r"Value iteration merges the two steps: backup with $\max_a$ every sweep "
        r"(approximate improve+eval).",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig14


# -------------------------------------------------------
# FIGURE 15: Visual intuition
# -------------------------------------------------------
def build_step_15():
    fig15, (ax15a, ax15b) = new_figure(1, 2)
    fig15.suptitle(
        "Step 15: Visual intuition -- improvement flips action arrows",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # Tiny 2-state cartoon: Left (L) and Right (R) with actions stay/go
    def _draw_two_state(ax, title, arrows_from_L, arrows_from_R, vL, vR, note):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, fontsize=FS_AX_TITLE - 2)
        # states
        for x, name, v, face in (
            (2.2, "L", vL, C_NEW),
            (7.8, "R", vR, C_GOAL),
        ):
            round_box(ax, x - 1.1, 4.2, 2.2, 2.4, face, "0.35", lw=1.8)
            ax.text(x, 5.9, name, ha="center", fontsize=FS_ANNOT, fontweight="bold")
            ax.text(x, 5.0, f"V={v}", ha="center", fontsize=FS_TICK + 1, fontweight="bold")
        # arrows: list of (x1,y1,x2,y2,label,color)
        for x1, y1, x2, y2, lab, col in arrows_from_L + arrows_from_R:
            ax.annotate(
                "", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=col, lw=2.2, mutation_scale=16),
            )
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.35, lab,
                    ha="center", fontsize=FS_TICK - 1, color=col, fontweight="bold")
        ax.text(5.0, 1.3, note, ha="center", fontsize=FS_TICK, fontweight="bold")


    _draw_two_state(
        ax15a,
        r"Before: $\pi$ always stays",
        arrows_from_L=[(2.2, 4.1, 2.2, 2.6, "stay", C_ACCENT)],
        arrows_from_R=[(7.8, 4.1, 7.8, 2.6, "stay", C_ACCENT)],
        vL="1",
        vR="2",
        note="Safe but mediocre values",
    )
    # self-loop visual as downward arrow is ok

    _draw_two_state(
        ax15b,
        r"After improve: go toward better $V$",
        arrows_from_L=[(3.4, 5.2, 6.6, 5.2, "go→R", C_OK)],
        arrows_from_R=[(7.8, 6.7, 7.8, 7.8, "stay*", C_OK)],
        vL="1",
        vR="2",
        note=r"Greedy w.r.t. $Q^\pi$ flips L's action",
    )

    fig15.text(
        0.5, 0.02,
        r"Evaluation writes the numbers; improvement rewires the policy arrows; iteration repeats.",
        ha="center", fontsize=FS_TICK, fontweight="bold",
    )
    fig15.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig15


# -------------------------------------------------------
# FIGURE 16: Takeaway
# -------------------------------------------------------
def build_step_16():
    fig16, ax16 = new_figure()
    fig16.suptitle(
        "Step 16: Takeaway -- Bellman, evaluation, improvement",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax16)

    round_box(ax16, 0.4, 6.55, 11.2, 1.15, C_DARK, C_DARK, lw=0)
    ax16.text(
        6.0, 7.1,
        r"$V^\pi(s)=\mathbb{E}_\pi[R_{t+1}+\gamma V^\pi(S_{t+1})\mid S_t=s]$"
        r"   $\;\;$ then $\;\;$"
        r"$\pi'(s)\in\mathrm{arg\,max}_a Q^\pi(s,a)$",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#f9e79f",
    )

    bullets = [
        r"0. MDP: Markov $(s,a)$ is enough — Bellman can ignore full history.",
        r"1. Bellman equality: value(now) = reward soon + discounted value(next).",
        r"2. Policy evaluation: solve / iterate Bellman for a fixed $\pi$ → $V^\pi$.",
        r"3. Policy improvement: greedy one-step lookahead w.r.t. $V^\pi$ / $Q^\pi$.",
        r"4. Policy iteration: evaluate ↔ improve until the policy stops changing ($\pi^*$).",
        r"5. Improvement theorem: greedy $\pi'$ satisfies $V^{\pi'}\geq V^\pi$ everywhere.",
    ]
    y = 5.55
    for line in bullets:
        round_box(ax16, 0.5, y - 0.28, 11.0, 0.62, C_MUTE, "0.5", lw=1.1)
        ax16.text(6.0, y, line, ha="center", va="center", fontsize=FS_TICK, fontweight="bold")
        y -= 0.78

    ax16.text(
        6.0, 0.35,
        "Binding lens: evaluation binds worth to a policy; improvement rebinds actions to that worth.",
        ha="center", fontsize=FS_TICK + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
    )
    return fig16


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
])

print("\n" + "=" * 60)
print("0301 BELLMAN EQUALITY")
print("=" * 60)
print("""
MDP assumption (usual RL theory stage):
  (S, A, P, R, gamma)
  Markov: next state/reward depend on (s,a) only -- not full history.

Bellman equality:
  V^pi(s) = E_pi[ R_{t+1} + gamma V^pi(S_{t+1}) | S_t = s ]

Policy evaluation:
  Fix pi; solve/iterate Bellman for V^pi.

Policy improvement:
  pi'(s) in argmax_a Q^pi(s,a)
  Theorem: V^{pi'} >= V^pi for all s.

Policy iteration:
  evaluate -> improve -> repeat until pi stable (= optimal in finite MDPs).

""")
