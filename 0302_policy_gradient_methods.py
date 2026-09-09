"""
0302_policy_gradient_methods.py

Policy gradient methods -- REINFORCE, baselines, actor-critic, PPO, and the family map.

After 0301_Bellman_equality.py (values / policy iteration), this lecture
optimizes a parameterized policy π_θ directly.

Sketch:
  1. Value-based vs policy-based
  2. Parameterized policy π_θ(a|s)
  3. Objective J(θ) = expected return
  4. Policy gradient theorem (full statement)
  5. REINFORCE (Monte Carlo policy gradient)
  6. Visual -- raise probability of actions on good trajectories
  7. Baselines -- subtract a critic of the state
  8. Advantages A_t = Q - V
  9. Actor-critic architecture
  10. Generalized Advantage Estimation (GAE) formulas
  11. GAE visual -- lambda mixes TD and Monte Carlo
  12. Why trust the update size -- TRPO vibe
  13. PPO clipped surrogate (formulas)
  14. Continuous actions / Gaussian policies
  15. Family map -- REINFORCE → AC → PPO / SAC / DDPG
  16. Bridge to LLM RLHF
  17. Takeaway

Companion to 0301_Bellman_equality.py, 0173_train_ChatGPT.py (REINFORCE/PPO in LLMs).
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
)
apply_rcparams(plt)
from matplotlib.patches import FancyBboxPatch
import numpy as np

np.random.seed(302)
RNG = np.random.default_rng(302)


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
# FIGURE 1: Value-based vs policy-based
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    "Step 1: Two ways to get a better policy",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax1)

round_box(ax1, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
ax1.text(
    6.0, 7.1,
    "0301: improve π via values (policy iteration).   "
    "Here: optimize π_θ with gradients on expected return.",
    ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
)

round_box(ax1, 0.45, 3.3, 5.5, 2.9, C_MUTE, "0.45", lw=1.6)
ax1.text(3.2, 5.8, "Value-based", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK)
ax1.text(
    3.2, 4.4,
    "Learn $V$ or $Q$.\n"
    "Policy is derived:\n"
    r"$\pi(s)=\mathrm{arg\,max}_a Q(s,a)$."
    "\n\n"
    "DQN, SARSA, tabular PI.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

round_box(ax1, 6.15, 3.3, 5.4, 2.9, "#f5eef8", C_FUN, lw=1.8)
ax1.text(8.85, 5.8, "Policy-based", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_FUN)
ax1.text(
    8.85, 4.4,
    r"Learn $\pi_\theta(a|s)$ directly."
    "\n"
    r"Ascend $\nabla_\theta J(\theta)$."
    "\n\n"
    "REINFORCE, A2C, PPO,\n"
    "natural for continuous /\n"
    "stochastic policies.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

ax1.text(
    6.0, 1.4,
    "Actor-critic methods combine both: a policy (actor) + a value helper (critic).",
    ha="center", fontsize=FS_ANNOT, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Parameterized policy
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    r"Step 2: A policy is a parameterized distribution $\pi_\theta(a\mid s)$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax2)

round_box(ax2, 0.4, 6.4, 11.2, 1.3, C_DARK, C_DARK, lw=0)
ax2.text(
    6.0, 7.05,
    r"Neural net (or softmax table) maps state $\rightarrow$ action probabilities / Gaussian params.",
    ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
)

round_box(ax2, 0.45, 3.35, 5.5, 2.75, C_NEW, C_DARK, lw=1.6)
ax2.text(3.2, 5.7, "Discrete actions", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK)
ax2.text(
    3.2, 4.4,
    r"$\pi_\theta(a|s)=\mathrm{softmax}(z_\theta(s))_a$"
    "\n\n"
    "logits $z$ from a network.\n"
    "Sample $a\\sim\\pi_\\theta(\\cdot|s)$.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

round_box(ax2, 6.15, 3.35, 5.4, 2.75, C_GOAL, C_OK, lw=1.6)
ax2.text(8.85, 5.7, "Continuous actions", ha="center",
         fontsize=FS_TICK, fontweight="bold", color=C_OK)
ax2.text(
    8.85, 4.4,
    r"$\pi_\theta(a|s)=\mathcal{N}(\mu_\theta(s),\sigma_\theta^2(s))$"
    "\n\n"
    "or squashed Gaussian (SAC).\n"
    "Sample + reparameterize.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

ax2.text(
    6.0, 1.5,
    "Training moves $\\theta$ so high-return trajectories become more probable under $\\pi_\\theta$.",
    ha="center", fontsize=FS_ANNOT, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.28", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Objective
# -------------------------------------------------------
fig3, ax3 = new_figure()
fig3.suptitle(
    r"Step 3: Objective -- expected return $J(\theta)$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax3)

round_box(ax3, 0.4, 5.9, 11.2, 1.8, C_DARK, C_DARK, lw=0)
ax3.text(
    6.0, 6.8,
    r"$J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)]"
    r"= \mathbb{E}_{\tau\sim\pi_\theta}\!\left[\sum_{t=0}^{T}\gamma^t R_{t+1}\right]$",
    ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
)

round_box(ax3, 0.45, 2.7, 5.5, 2.9, C_MUTE, "0.45", lw=1.5)
ax3.text(3.2, 5.2, "Trajectory", ha="center", fontsize=FS_TICK, fontweight="bold")
ax3.text(
    3.2, 3.9,
    r"$\tau=(s_0,a_0,r_1,s_1,\ldots)$"
    "\n\n"
    r"sampled by acting $a_t\sim\pi_\theta(\cdot|s_t)$"
    "\n"
    "in the MDP.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

round_box(ax3, 6.15, 2.7, 5.4, 2.9, C_GOAL, C_OK, lw=1.8)
ax3.text(8.85, 5.2, "Goal", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
ax3.text(
    8.85, 3.9,
    r"Find $\theta^\star=\mathrm{arg\,max}_\theta J(\theta)$."
    "\n\n"
    "No need for $\\mathrm{arg\,max}_a Q$\n"
    "at decision time — the net\n"
    "already outputs $\\pi_\\theta$.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

ax3.text(
    6.0, 1.2,
    "Problem: $J$ depends on $\\theta$ through the sampling distribution — not a fixed dataset.",
    ha="center", fontsize=FS_ANNOT, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
)
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Policy gradient theorem
# -------------------------------------------------------
fig4, ax4 = new_figure()
fig4.suptitle(
    "Step 4: Policy gradient theorem",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax4)

round_box(ax4, 0.35, 6.15, 11.3, 1.55, C_DARK, C_DARK, lw=0)
ax4.text(
    6.0, 6.9,
    r"$\nabla_\theta J(\theta)"
    r"= \mathbb{E}_{s\sim d^{\pi_\theta},\,a\sim\pi_\theta}"
    r"\!\left["
    r"\nabla_\theta\log\pi_\theta(a|s)\, Q^{\pi_\theta}(s,a)"
    r"\right]$",
    ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
)

round_box(ax4, 0.35, 3.55, 11.3, 2.35, C_MUTE, "0.45", lw=1.5)
ax4.text(6.0, 5.5, "Equivalent practical forms", ha="center",
         fontsize=FS_TICK, fontweight="bold", color=C_DARK)
ax4.text(
    6.0, 4.35,
    r"$\nabla_\theta J"
    r"= \mathbb{E}_{\tau\sim\pi_\theta}"
    r"\!\left[\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t)\, G_t\right]$"
    "     (REINFORCE / MC)"
    "\n"
    r"$\nabla_\theta J"
    r"= \mathbb{E}"
    r"\!\left[\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t)\, A^{\pi_\theta}(s_t,a_t)\right]$"
    "     (advantage form)"
    "\n"
    r"Score identity:  "
    r"$\nabla_\theta\log\pi_\theta(a|s)"
    r"= \pi_\theta(a|s)^{-1}\nabla_\theta\pi_\theta(a|s)$",
    ha="center", va="center", fontsize=FS_TICK, fontweight="bold",
)

round_box(ax4, 0.35, 0.35, 5.5, 2.9, C_GOAL, C_OK, lw=1.6)
ax4.text(3.1, 2.85, "What it says", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
ax4.text(
    3.1, 1.5,
    "Average, under the policy's\n"
    "state distribution $d^\\pi$,\n"
    "the score $\\nabla\\log\\pi$\n"
    "weighted by how good the\n"
    "action was ($Q$ or $A$ or $G$).",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

round_box(ax4, 6.15, 0.35, 5.5, 2.9, C_OLD, C_ACCENT, lw=1.6)
ax4.text(8.9, 2.85, "What you do not need", ha="center",
         fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
ax4.text(
    8.9, 1.5,
    "No derivative through $P$\n"
    "or the reward function.\n"
    "Sample trajectories, then\n"
    "backprop only through\n"
    "$\\log\\pi_\\theta(a_t|s_t)$.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: REINFORCE
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    "Step 5: REINFORCE -- Monte Carlo policy gradient",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax5)

round_box(ax5, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
ax5.text(
    6.0, 7.0,
    r"$\nabla_\theta J"
    r"\approx \sum_t G_t\,\nabla_\theta\log\pi_\theta(a_t|s_t)$"
    "    with  "
    r"$G_t=\sum_{k\geq t}\gamma^{k-t}R_{k+1}$",
    ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
)

steps = [
    (5.35, r"1. Roll out episode(s) with current $\pi_\theta$.", C_MUTE),
    (4.4, r"2. Compute returns $G_t$ from rewards (Monte Carlo).", C_NEW),
    (3.45, r"3. Ascend $\theta$ with $\sum_t G_t\nabla\log\pi_\theta(a_t|s_t)$  (or batch average).", C_GOAL),
    (2.5, r"4. Repeat.  High $G_t$ reinforces taken actions; low $G_t$ suppresses them.", C_OLD),
]
for y, text, face in steps:
    round_box(ax5, 0.5, y - 0.4, 11.0, 0.85, face, "0.45", lw=1.2)
    ax5.text(6.0, y, text, ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

ax5.text(
    6.0, 0.85,
    "Pros: simple, unbiased (with proper returns).   "
    "Cons: high variance; needs full episodes; slow credit.",
    ha="center", fontsize=FS_TICK + 1, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.25", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
)
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Visual REINFORCE
# -------------------------------------------------------
fig6, (ax6a, ax6b) = new_figure(1, 2)
fig6.suptitle(
    "Step 6: Visual -- good trajectories push probability mass onto their actions",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

actions = ["L", "stay", "R"]
p_before = np.array([0.25, 0.50, 0.25])
p_after = np.array([0.10, 0.25, 0.65])  # reinforced "R" after high return

ax6a.bar(actions, p_before, color=[C_NEW, C_MUTE, C_NEW], edgecolor="k")
ax6a.set_ylim(0, 1.0)
ax6a.set_ylabel(r"$\pi(a|s)$", fontsize=FS_AXIS)
ax6a.set_title(r"Before: at state $s$", fontsize=FS_AX_TITLE - 2)
ax6a.tick_params(labelsize=FS_TICK)
ax6a.grid(True, axis="y", alpha=0.3)
for i, p in enumerate(p_before):
    ax6a.text(i, p + 0.03, f"{p:.2f}", ha="center", fontsize=FS_TICK)

ax6b.bar(actions, p_after, color=[C_MUTE, C_MUTE, C_OK], edgecolor="k")
ax6b.set_ylim(0, 1.0)
ax6b.set_ylabel(r"$\pi(a|s)$", fontsize=FS_AXIS)
ax6b.set_title(r"After high-$G$ episode that went R", fontsize=FS_AX_TITLE - 2)
ax6b.tick_params(labelsize=FS_TICK)
ax6b.grid(True, axis="y", alpha=0.3)
for i, p in enumerate(p_after):
    ax6b.text(i, p + 0.03, f"{p:.2f}", ha="center", fontsize=FS_TICK)

fig6.text(
    0.5, 0.02,
    r"Update $\propto G\cdot\nabla\log\pi$: success on R increases $\pi(R|s)$; competitors lose mass via softmax.",
    ha="center", fontsize=FS_TICK, fontweight="bold",
)
fig6.tight_layout(rect=[0, 0.05, 1, 0.92])
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Baselines
# -------------------------------------------------------
fig7, ax7 = new_figure()
fig7.suptitle(
    "Step 7: Baselines -- subtract something that does not depend on $a$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax7)

round_box(ax7, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
ax7.text(
    6.0, 7.0,
    r"$\nabla_\theta J"
    r"= \mathbb{E}\!\left["
    r"\sum_t \nabla_\theta\log\pi_\theta(a_t|s_t)\,(G_t - b(s_t))"
    r"\right]$",
    ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
)

round_box(ax7, 0.45, 3.2, 5.5, 2.8, C_MUTE, "0.45", lw=1.6)
ax7.text(3.2, 5.55, "Why it is valid", ha="center", fontsize=FS_TICK, fontweight="bold")
ax7.text(
    3.2, 4.25,
    r"$b(s)$ independent of $a$"
    "\n"
    r"$\Rightarrow$ "
    r"$\mathbb{E}_{a\sim\pi}[\nabla\log\pi\cdot b(s)]=0$."
    "\n\n"
    "Unbiased gradient,\n"
    "often much less variance.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

round_box(ax7, 6.15, 3.2, 5.4, 2.8, C_GOAL, C_OK, lw=1.8)
ax7.text(8.85, 5.55, "Common choices", ha="center",
         fontsize=FS_TICK, fontweight="bold", color=C_OK)
ax7.text(
    8.85, 4.25,
    r"$b(s)=\hat V(s)$  (learned)"
    "\n"
    "moving average of returns\n"
    "batch mean return\n\n"
    "Leads naturally to\n"
    "advantages / actor-critic.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

ax7.text(
    6.0, 1.3,
    "Raw REINFORCE treats every step with the same noisy $G_t$.  "
    "A baseline asks: better than usual for this state?",
    ha="center", fontsize=FS_ANNOT, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
)
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Advantages
# -------------------------------------------------------
fig8, ax8 = new_figure()
fig8.suptitle(
    r"Step 8: Advantages $A_t=Q(s_t,a_t)-V(s_t)$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax8)

round_box(ax8, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
ax8.text(
    6.0, 7.0,
    r"$\nabla_\theta J"
    r"\approx \mathbb{E}\!\left["
    r"\sum_t A_t\,\nabla_\theta\log\pi_\theta(a_t|s_t)"
    r"\right]$",
    ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#f9e79f",
)

round_box(ax8, 0.45, 2.9, 3.6, 3.1, C_NEW, C_DARK, lw=1.5)
ax8.text(2.25, 5.6, r"$Q$", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
ax8.text(2.25, 4.3, "value of\ntaking this\naction, then\nfollowing π",
         ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

round_box(ax8, 4.2, 2.9, 3.6, 3.1, C_MUTE, "0.45", lw=1.5)
ax8.text(6.0, 5.6, r"$V$", ha="center", fontsize=FS_ANNOT, fontweight="bold")
ax8.text(6.0, 4.3, "value of the\nstate on average\n(over π's actions)",
         ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

round_box(ax8, 7.95, 2.9, 3.6, 3.1, C_GOAL, C_OK, lw=1.8)
ax8.text(9.75, 5.6, r"$A=Q-V$", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
ax8.text(9.75, 4.3, "how much better\n/ worse was this\naction than usual?",
         ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

ax8.text(
    6.0, 1.3,
    "Positive $A$: reinforce.  Negative $A$: suppress.  Zero-mean signal → cleaner learning.",
    ha="center", fontsize=FS_ANNOT, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
)
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 9: Actor-critic
# -------------------------------------------------------
fig9, ax9 = new_figure()
fig9.suptitle(
    "Step 9: Actor-critic -- policy + value learning together",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax9)

round_box(ax9, 0.5, 4.0, 4.8, 3.5, "#f5eef8", C_FUN, lw=2.0)
ax9.text(2.9, 7.05, "Actor  $\\pi_\\theta$", ha="center",
         fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_FUN)
ax9.text(
    2.9, 5.5,
    "Chooses actions.\n"
    r"Update with $A\,\nabla\log\pi$."
    "\n\n"
    "Policy parameters $\\theta$.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

round_box(ax9, 6.7, 4.0, 4.8, 3.5, C_GOAL, C_OK, lw=2.0)
ax9.text(9.1, 7.05, "Critic  $V_\\phi$ / $Q_\\phi$", ha="center",
         fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_OK)
ax9.text(
    9.1, 5.5,
    "Scores states / actions.\n"
    "Trained by TD / Bellman\n"
    "regression (MSE).\n\n"
    "Provides $A_t$ to the actor.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

arrow(ax9, 5.4, 5.7, 6.6, 5.7, C_DARK, 2.2)
ax9.text(6.0, 6.15, r"$A_t$", ha="center", fontsize=FS_ANNOT, fontweight="bold")

ax9.text(
    6.0, 2.2,
    "One-step actor-critic target:  "
    r"$A_t \approx r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$"
    "\n"
    "(TD error).  Actor uses it; critic regresses toward the same Bellman backup.",
    ha="center", fontsize=FS_ANNOT, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.3", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
)
ax9.text(
    6.0, 0.55,
    "Names: Actor-Critic, A2C (synchronous), A3C (async workers) — same spine.",
    ha="center", fontsize=FS_TICK, style="italic", color="0.35",
)
show_maximized(fig9)


# -------------------------------------------------------
# FIGURE 10: GAE formulas
# -------------------------------------------------------
fig10, ax10 = new_figure()
fig10.suptitle(
    "Step 10: Generalized Advantage Estimation (GAE)",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax10)

round_box(ax10, 0.35, 6.25, 11.3, 1.45, C_DARK, C_DARK, lw=0)
ax10.text(
    6.0, 6.95,
    r"TD residual:  "
    r"$\delta_t = R_{t+1} + \gamma V(s_{t+1}) - V(s_t)$"
    "\n"
    r"$A_t^{\mathrm{GAE}(\gamma,\lambda)}"
    r"= \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}$",
    ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
)

round_box(ax10, 0.35, 3.35, 5.6, 2.6, C_MUTE, "0.45", lw=1.5)
ax10.text(3.15, 5.55, "Recursive form (what code uses)", ha="center",
          fontsize=FS_TICK, fontweight="bold", color=C_DARK)
ax10.text(
    3.15, 4.3,
    r"$A_t^{\mathrm{GAE}}"
    r"= \delta_t + \gamma\lambda\, A_{t+1}^{\mathrm{GAE}}$"
    "\n\n"
    "Scan backward from the end\n"
    "of a rollout / episode.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

round_box(ax10, 6.2, 3.35, 5.45, 2.6, "#f5eef8", C_FUN, lw=1.8)
ax10.text(8.9, 5.55, "Special cases", ha="center",
          fontsize=FS_TICK, fontweight="bold", color=C_FUN)
ax10.text(
    8.9, 4.3,
    r"$\lambda=0$:  $A_t=\delta_t$  (1-step TD)"
    "\n"
    r"$\lambda=1$:  $A_t=G_t-V(s_t)$  (MC)"
    "\n\n"
    r"$\lambda\in(0,1)$: exponential"
    "\n"
    "mix of multi-step advantages.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

ax10.text(
    6.0, 1.55,
    r"GAE $\approx$ exponentially weighted average of $k$-step advantage estimators."
    "\n"
    r"Lower variance than pure MC; less bias than pure 1-step TD when $\lambda$ is mid-range.",
    ha="center", fontsize=FS_TICK + 1, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
)
ax10.text(
    6.0, 0.4,
    "Schulman et al. (2016).  Standard advantage input to PPO / TRPO actors.",
    ha="center", fontsize=FS_TICK, style="italic", color="0.35",
)
show_maximized(fig10)


# -------------------------------------------------------
# FIGURE 11: GAE visual -- lambda
# -------------------------------------------------------
fig11, (ax11a, ax11b) = new_figure(1, 2)
fig11.suptitle(
    r"Step 11: GAE visual -- $\lambda$ mixes short TD with long Monte Carlo",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

# Left: weight on future deltas for different lambda
lags = np.arange(0, 12)
for lam, col, lab in (
    (0.0, C_ACCENT, r"$\lambda=0$ (TD)"),
    (0.5, C_FUN, r"$\lambda=0.5$"),
    (0.95, C_OK, r"$\lambda=0.95$ (near MC)"),
):
    w = (0.99 * lam) ** lags  # gamma*lambda with gamma~0.99 toy
    ax11a.plot(lags, w, "-o", color=col, lw=2.2, ms=5, label=lab)
ax11a.set_xlabel(r"lag $l$ in $(\gamma\lambda)^l\delta_{t+l}$", fontsize=FS_AXIS - 1)
ax11a.set_ylabel("weight", fontsize=FS_AXIS - 1)
ax11a.set_title("How far GAE looks ahead", fontsize=FS_AX_TITLE - 2)
ax11a.legend(fontsize=FS_LEGEND - 1, loc="upper right")
ax11a.grid(True, alpha=0.3)
ax11a.tick_params(labelsize=FS_TICK - 1)

# Right: bias-variance cartoon
ax11b.set_xlim(0, 10)
ax11b.set_ylim(0, 10)
ax11b.axis("off")
ax11b.set_title(r"Bias–variance dial", fontsize=FS_AX_TITLE - 2)
round_box(ax11b, 0.5, 6.2, 9.0, 2.8, C_OLD, C_ACCENT, lw=1.6)
ax11b.text(5.0, 8.2, r"$\lambda \rightarrow 0$", ha="center",
           fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
ax11b.text(5.0, 7.2, "low variance, more bias\n(trust $V$ every step)",
           ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")
round_box(ax11b, 0.5, 2.6, 9.0, 2.8, C_GOAL, C_OK, lw=1.6)
ax11b.text(5.0, 4.6, r"$\lambda \rightarrow 1$", ha="center",
           fontsize=FS_TICK, fontweight="bold", color=C_OK)
ax11b.text(5.0, 3.6, "low bias, more variance\n(trust the full return)",
           ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")
ax11b.text(5.0, 1.2, r"Practice: $\lambda\approx 0.9$–$0.98$ with $\gamma\approx 0.99$",
           ha="center", fontsize=FS_TICK, fontweight="bold")

fig11.tight_layout(rect=[0, 0.02, 1, 0.92])
show_maximized(fig11)


# -------------------------------------------------------
# FIGURE 12: Trust region
# -------------------------------------------------------
fig12, ax12 = new_figure()
fig12.suptitle(
    "Step 12: Large policy steps can destroy performance -- trust the region",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax12)

round_box(ax12, 0.4, 6.4, 11.2, 1.3, C_DARK, C_DARK, lw=0)
ax12.text(
    6.0, 7.05,
    "Vanilla PG / AC can take a step that collapses π.  Constrain how far π_new moves from π_old.",
    ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
)

round_box(ax12, 0.45, 3.2, 5.5, 2.9, C_OLD, C_ACCENT, lw=1.6)
ax12.text(3.2, 5.7, "TRPO idea", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
ax12.text(
    3.2, 4.35,
    r"Maximize a surrogate"
    "\n"
    r"s.t.  $D_{\mathrm{KL}}(\pi_{\mathrm{old}}\|\pi_\theta)\leq\delta$."
    "\n\n"
    "Natural gradient /\n"
    "conjugate gradients.\n"
    "Strong but heavy.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

round_box(ax12, 6.15, 3.2, 5.4, 2.9, C_GOAL, C_OK, lw=1.8)
ax12.text(8.85, 5.7, "PPO idea", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
ax12.text(
    8.85, 4.35,
    "Same spirit, simpler:\n"
    "clip the probability ratio\n"
    "so updates stay local.\n\n"
    "First-order SGD/Adam.\n"
    "Default in many labs\n"
    "+ LLM RLHF recipes.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

ax12.text(
    6.0, 1.4,
    r"Importance ratio  $\rho_t=\pi_\theta(a_t|s_t)/\pi_{\theta_{\mathrm{old}}}(a_t|s_t)$  "
    "measures how much the new policy changed on this action.",
    ha="center", fontsize=FS_ANNOT, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.28", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
)
show_maximized(fig12)


# -------------------------------------------------------
# FIGURE 13: PPO
# -------------------------------------------------------
fig13, ax13 = new_figure()
fig13.suptitle(
    "Step 13: PPO -- clipped surrogate objective",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax13)

round_box(ax13, 0.4, 6.25, 11.2, 1.45, C_DARK, C_DARK, lw=0)
ax13.text(
    6.0, 6.95,
    r"$L^{\mathrm{CLIP}}(\theta)"
    r"= \mathbb{E}_t\!\left["
    r"\min\!\left("
    r"\rho_t A_t,\;"
    r"\mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\,A_t"
    r"\right)"
    r"\right]$",
    ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
)

round_box(ax13, 0.45, 3.25, 5.5, 2.7, C_MUTE, "0.45", lw=1.6)
ax13.text(3.2, 5.55, "If $A_t>0$", ha="center", fontsize=FS_TICK, fontweight="bold")
ax13.text(
    3.2, 4.25,
    "Want larger $\\rho_t$\n"
    "(more probability on\n"
    "the good action),\n"
    "but clip stops $\\rho$\n"
    r"above $1+\epsilon$.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

round_box(ax13, 6.15, 3.25, 5.4, 2.7, C_OLD, C_ACCENT, lw=1.6)
ax13.text(8.85, 5.55, "If $A_t<0$", ha="center",
          fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
ax13.text(
    8.85, 4.25,
    "Want smaller $\\rho_t$\n"
    "(less probability),\n"
    "but clip stops $\\rho$\n"
    r"below $1-\epsilon$.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

ax13.text(
    6.0, 1.55,
    r"Full loss often:  $L^{\mathrm{PPO}}\approx L^{\mathrm{CLIP}}-c_1 L^{\mathrm{VF}}+c_2 H[\pi_\theta]$"
    "\n"
    r"with $L^{\mathrm{VF}}$ = critic value error,  $H$ = entropy bonus,  $\epsilon\sim 0.1$–$0.3$.",
    ha="center", fontsize=FS_TICK + 1, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.28", facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
)
ax13.text(
    6.0, 0.4,
    "Multiple epochs on the same batch are OK because clipping limits over-optimism.",
    ha="center", fontsize=FS_TICK, style="italic", color="0.35",
)
show_maximized(fig13)


# -------------------------------------------------------
# FIGURE 14: Continuous / Gaussian
# -------------------------------------------------------
fig14, (ax14a, ax14b) = new_figure(1, 2)
fig14.suptitle(
    "Step 14: Continuous control -- Gaussian policies still use $\\nabla\\log\\pi$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

x = np.linspace(-3, 3, 200)
pdf_old = np.exp(-0.5 * ((x - 0.0) / 0.8) ** 2) / (0.8 * np.sqrt(2 * np.pi))
pdf_new = np.exp(-0.5 * ((x - 0.9) / 0.55) ** 2) / (0.55 * np.sqrt(2 * np.pi))
ax14a.fill_between(x, pdf_old, alpha=0.35, color=C_FUN, label=r"old $\pi$")
ax14a.plot(x, pdf_old, color=C_FUN, lw=2)
ax14a.fill_between(x, pdf_new, alpha=0.35, color=C_OK, label=r"new $\pi$")
ax14a.plot(x, pdf_new, color=C_OK, lw=2)
ax14a.axvline(1.2, color=C_ACCENT, ls="--", lw=1.8, label="sampled a")
ax14a.set_title(r"$\mathcal{N}(\mu,\sigma^2)$ shifts after +A", fontsize=FS_AX_TITLE - 2)
ax14a.legend(fontsize=FS_LEGEND - 1, loc="upper left")
ax14a.set_xlabel("action", fontsize=FS_AXIS - 1)
ax14a.set_yticks([])
ax14a.grid(True, alpha=0.25)

ax14b.axis("off")
ax14b.set_xlim(0, 10)
ax14b.set_ylim(0, 10)
round_box(ax14b, 0.4, 1.5, 9.2, 7.0, C_MUTE, "0.45", lw=1.5)
ax14b.text(
    5.0, 6.5,
    "Same PG / PPO math.\n\n"
    r"$\log\pi(a|s)=-\frac{(a-\mu)^2}{2\sigma^2}-\log\sigma+c$"
    "\n\n"
    "Off-policy cousins:\n"
    "DDPG / TD3 (deterministic +\n"
    "Q critic), SAC (max entropy\n"
    "stochastic actor).\n\n"
    "Still actor + critic —\n"
    "different data / objectives.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

fig14.tight_layout(rect=[0, 0.02, 1, 0.92])
show_maximized(fig14)


# -------------------------------------------------------
# FIGURE 15: Family map
# -------------------------------------------------------
fig15, ax15 = new_figure()
fig15.suptitle(
    "Step 15: Family map -- from REINFORCE to modern actors",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax15)

rows = [
    (6.4, "REINFORCE", "MC return x grad log pi", "unbiased, high variance", C_MUTE),
    (5.35, "+ baseline / advantage", "G - V  or  A = Q - V", "variance down", C_NEW),
    (4.3, "Actor-Critic / A2C", "TD / n-step critic", "online, on-policy", C_GOAL),
    (3.25, "GAE + TRPO/PPO", "lambda-advantage + trust/clip", "stable deep RL default", "#f5eef8"),
    (2.2, "Off-policy actors", "DDPG, TD3, SAC", "replay buffer, continuous", C_OLD),
]
for y, name, mech, note, face in rows:
    round_box(ax15, 0.4, y - 0.4, 3.4, 0.9, face, "0.45", lw=1.3)
    ax15.text(2.1, y, name, ha="center", va="center", fontsize=FS_TICK, fontweight="bold")
    round_box(ax15, 4.0, y - 0.4, 4.0, 0.9, C_MUTE, "0.5", lw=1.1)
    ax15.text(6.0, y, mech, ha="center", va="center", fontsize=FS_TICK)
    round_box(ax15, 8.2, y - 0.4, 3.4, 0.9, C_MUTE, "0.5", lw=1.1)
    ax15.text(9.9, y, note, ha="center", va="center", fontsize=FS_TICK)

ax15.text(
    6.0, 0.7,
    r"Shared DNA: score function $\nabla\log\pi$ · (advantage-like signal).  Wrappers differ.",
    ha="center", fontsize=FS_ANNOT, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.25", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
)
show_maximized(fig15)


# -------------------------------------------------------
# FIGURE 16: LLM bridge
# -------------------------------------------------------
fig16, ax16 = new_figure()
fig16.suptitle(
    "Step 16: Same math in LLM post-training (RLHF)",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax16)

round_box(ax16, 0.4, 6.4, 11.2, 1.3, C_DARK, C_DARK, lw=0)
ax16.text(
    6.0, 7.05,
    r"Tokens = actions.  Prompt = state.  Sequence reward / preference = return signal.",
    ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
)

round_box(ax16, 0.45, 3.3, 5.5, 2.8, C_NEW, C_DARK, lw=1.6)
ax16.text(3.2, 5.7, "Classic PG view", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK)
ax16.text(
    3.2, 4.4,
    r"REINFORCE: $R\sum_t\nabla\log\pi(y_t|s_t)$"
    "\n\n"
    "PPO on LLM policy with\n"
    "KL toward SFT reference\n"
    "(InstructGPT-style).",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

round_box(ax16, 6.15, 3.3, 5.4, 2.8, "#f5eef8", C_FUN, lw=1.8)
ax16.text(8.85, 5.7, "Lecture bridge", ha="center",
          fontsize=FS_TICK, fontweight="bold", color=C_FUN)
ax16.text(
    8.85, 4.4,
    "Formulas unpacked in\n"
    "0173_train_ChatGPT.py\n"
    "(REINFORCE, clip, KL).\n\n"
    "Here: the RL backbone\n"
    "those slides assume.",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
)

ax16.text(
    6.0, 1.5,
    "DPO later bypasses explicit RL sampling — but still comes from a preference / PG lineage.",
    ha="center", fontsize=FS_ANNOT, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
)
show_maximized(fig16)


# -------------------------------------------------------
# FIGURE 17: Takeaway
# -------------------------------------------------------
fig17, ax17 = new_figure()
fig17.suptitle(
    "Step 17: Takeaway -- policy gradients in one page",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank(ax17)

round_box(ax17, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
ax17.text(
    6.0, 7.1,
    r"$\nabla J=\mathbb{E}[\sum_t A_t\nabla\log\pi_\theta(a_t|s_t)]$"
    r"   $\;\;$"
    r"$A_t^{\mathrm{GAE}}=\sum_l(\gamma\lambda)^l\delta_{t+l}$",
    ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#f9e79f",
)

bullets = [
    r"1. Policy gradient theorem: $\nabla J=\mathbb{E}[\nabla\log\pi\cdot Q]$ (or $G_t$ / $A_t$).",
    r"2. REINFORCE: Monte Carlo $G_t\cdot\nabla\log\pi$ — simple, noisy.",
    r"3. Baselines / advantages cut variance; critic estimates $V$ or $Q$.",
    r"4. GAE: $A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$ mixes TD ($\lambda=0$) and MC ($\lambda=1$).",
    r"5. Actor-critic + PPO clip keep deep policy updates stable.",
    r"6. Same spine in continuous control and LLM RLHF.",
]
y = 5.55
for line in bullets:
    round_box(ax17, 0.5, y - 0.28, 11.0, 0.62, C_MUTE, "0.5", lw=1.1)
    ax17.text(6.0, y, line, ha="center", va="center", fontsize=FS_TICK, fontweight="bold")
    y -= 0.78

ax17.text(
    6.0, 0.35,
    "Binding lens: advantages bind \"was this action better than usual?\" into π updates.",
    ha="center", fontsize=FS_TICK + 1, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.22", facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
)
show_maximized(fig17)


print("\n" + "=" * 60)
print("0302 POLICY GRADIENT METHODS")
print("=" * 60)
print("""
Objective:
  J(theta) = E_{tau ~ pi_theta}[ return(tau) ]

Policy gradient theorem:
  grad J = E_{s~d^pi, a~pi}[ grad log pi_theta(a|s) * Q^pi(s,a) ]
  = E[ sum_t grad log pi * G_t ]           (REINFORCE)
  = E[ sum_t grad log pi * A_t ]           (advantage form)

GAE (Generalized Advantage Estimation):
  delta_t = R_{t+1} + gamma V(s_{t+1}) - V(s_t)
  A_t^GAE = sum_{l>=0} (gamma*lambda)^l * delta_{t+l}
  A_t = delta_t + gamma*lambda * A_{t+1}   (backward recursion)
  lambda=0 -> 1-step TD;  lambda=1 -> MC advantage

Actor-critic / PPO:
  Actor: ascent on A * grad log pi
  Critic: fit V (TD / GAE targets)
  L_CLIP = E[ min(rho A, clip(rho, 1-eps, 1+eps) A) ]

See also: 0301_Bellman_equality.py, 0173_train_ChatGPT.py
""")
