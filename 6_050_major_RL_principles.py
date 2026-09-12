"""
6_050_major_RL_principles.py

Major principles that organize modern RL -- without re-deriving MDPs,
Bellman equations, policy/value iteration, or policy-gradient formulas
(those live in 6_005 / 6_010 / 6_015 / 6_020).

Sketch:
  1. Map of the big tensions
  2. Exploration vs exploitation
  3. On-policy vs off-policy
  4. Model-based vs model-free
  5. Credit assignment -- who gets the blame/reward?
  6. Bootstrapping vs Monte Carlo
  7. The deadly triad (warning)
  8. Sample efficiency vs stability
  9. Online interaction vs offline / batch RL
  10. Evaluation principles
  11. Takeaway
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
C_LOOP = "#2874a6"


# -------------------------------------------------------
# Step 1
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: Major RL principles -- the tensions that organize algorithms",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank(ax)

    tiles = [
        (0.4, 5.5, "Explore vs\nexploit"),
        (3.3, 5.5, "On vs off\npolicy"),
        (6.2, 5.5, "Model-based\nvs free"),
        (9.1, 5.5, "Bootstrap\nvs MC"),
        (0.4, 2.7, "Credit\nassignment"),
        (3.3, 2.7, "Efficiency\nvs stability"),
        (6.2, 2.7, "Online vs\noffline"),
        (9.1, 2.7, "How to\nevaluate"),
    ]
    for x, y, title in tiles:
        round_box(ax, x, y, 2.6, 2.3, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 1.3, y + 1.15, title, ha="center", va="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)

    round_box(ax, 0.4, 0.35, 11.2, 2.0, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "These are design axes -- not algorithms.\n"
        "Every major method picks a corner of this map.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: Explore vs exploit
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: Exploration vs exploitation",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 4.5, 5.5, 3.1, C_WARN, "#d68910", lw=2.0)
    ax.text(3.15, 7.1, "Explore", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color="#9a7d0a")
    ax.text(
        3.15, 5.7,
        "Try unfamiliar actions / states\n"
        "to gather information.\n\n"
        "Pays later; costs now.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#6e2c00",
    )

    round_box(ax, 6.1, 4.5, 5.5, 3.1, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 7.1, "Exploit", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 5.7,
        "Use the current best guess\n"
        "to harvest reward.\n\n"
        "Pays now; may miss better options.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 0.4, 0.4, 11.2, 3.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 2.25,
        "Mechanisms you will see:\n"
        "epsilon-greedy · Boltzmann / softmax · UCB / optimism · entropy bonuses · noise in policies\n\n"
        "Without exploration, learning collapses to whatever it saw first.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 3: On vs off policy
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: On-policy vs off-policy learning",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 4.35, 5.5, 3.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 7.15, "On-policy", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 5.6,
        "Improve the same policy\n"
        "that generated the data.\n\n"
        "Examples: SARSA vibe,\n"
        "vanilla REINFORCE / PPO\n"
        "(with fresh rollouts).",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 4.35, 5.5, 3.3, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 7.15, "Off-policy", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 5.6,
        "Learn about one policy\n"
        "from data of another.\n\n"
        "Examples: Q-learning,\n"
        "DQN replay buffers,\n"
        "many offline RL methods.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 0.4, 0.4, 11.2, 3.5, C_WARN, "#d68910", lw=1.8)
    ax.text(
        6.0, 2.15,
        "Off-policy reuses data (sample efficient) but needs corrections / care\n"
        "(importance weights, clipped objectives, conservative backups).\n"
        "On-policy is stabler to analyze, hungrier for new interaction.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#6e2c00",
    )
    return fig


# -------------------------------------------------------
# Step 4: Model-based vs free
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: Model-based vs model-free",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 4.35, 5.5, 3.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 7.15, "Model-based", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 5.6,
        "Learn or use $P,R$\n"
        "(or a world model),\n"
        "then plan / imagine.\n\n"
        "DP with a known model;\n"
        "Dyna; MuZero-style\n"
        "learned dynamics.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 4.35, 5.5, 3.3, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 7.15, "Model-free", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 5.6,
        "Directly learn values\n"
        "and/or a policy from\n"
        "interaction samples.\n\n"
        "TD, Q-learning, DQN,\n"
        "policy gradients, PPO.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 0.4, 0.4, 11.2, 3.5, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 2.15,
        "Models help transfer and planning; they can also be wrong and mislead.\n"
        "Model-free methods skip the dynamics, but may need more environment steps.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 5: Credit assignment
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: Credit assignment -- which decisions earned the return?",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 5.7, 11.2, 1.9, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.65,
        "Rewards often arrive late. Learning must assign credit\n"
        "backward through time to the actions that mattered.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.4, 2.55, "Returns / MC", "Sum future rewards\nalong a whole episode."),
        (4.2, 2.55, "TD / bootstrap", "One-step target uses\ncurrent value estimates."),
        (8.0, 2.55, "Eligibility /\nadvantages", "Emphasize actions that\nbeat the expected baseline."),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.8, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 2.0, title, ha="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 1.0, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.35, 11.2, 1.9, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.3,
        "Hard environments are hard largely because credit is sparse or delayed.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 6: Bootstrap vs MC
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Bootstrapping vs Monte Carlo targets",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 4.35, 5.5, 3.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 7.15, "Monte Carlo", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 5.6,
        "Target = complete return $G_t$.\n"
        "Unbiased (under the policy),\n"
        "high variance, needs episodes\n"
        "to finish.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 4.35, 5.5, 3.3, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 7.15, "Bootstrapping (TD)", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 5.6,
        r"Target mixes reward with"
        "\n"
        r"an estimate $V(s')$ or $Q$."
        "\n"
        "Lower variance, biased by\n"
        "current errors, online-friendly.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 0.4, 0.4, 11.2, 3.5, C_WARN, "#d68910", lw=1.8)
    ax.text(
        6.0, 2.15,
        r"$n$-step returns and $\lambda$-returns interpolate the two."
        "\n"
        "More bootstrap → faster, more coupled to function-approx errors.\n"
        "More MC → cleaner targets, noisier updates.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#6e2c00",
    )
    return fig


# -------------------------------------------------------
# Step 7: Deadly triad
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: Warning -- the deadly triad",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 6.2, 11.2, 1.45, "#f8d7da", C_ACCENT, lw=2.2)
    ax.text(
        6.0, 6.9,
        "Function approximation + bootstrapping + off-policy sampling\n"
        "can diverge even on simple linear problems.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#641e16",
    )

    cards = [
        (0.4, 3.1, "1. FA", "Values/policies are\napproximate (neural nets)."),
        (4.2, 3.1, "2. Bootstrap", "Targets depend on\nthe same approximator."),
        (8.0, 3.1, "3. Off-policy", "Data distribution ≠\nthe target policy."),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.7, C_MUTE, "0.5", lw=1.8)
        ax.text(x + 1.8, y + 1.95, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT)
        ax.text(x + 1.8, y + 1.05, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.35, 11.2, 2.4, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.55,
        "Practical responses: replay + target nets, clipped objectives,\n"
        "on-policy trust regions, conservative backups, careful evaluation.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 8: Efficiency vs stability
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: Sample efficiency vs training stability",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 4.4, 5.5, 3.2, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 7.1, "Push efficiency", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 5.6,
        "Reuse data (replay)\n"
        "Off-policy corrections\n"
        "World models / planning\n"
        "Better exploration bonuses",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 4.4, 5.5, 3.2, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 7.1, "Push stability", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 5.6,
        "Trust-region / clipped updates\n"
        "Target networks\n"
        "Normalize advantages / obs\n"
        "Smaller steps, more on-policy data",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 0.4, 0.4, 11.2, 3.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 2.2,
        "You rarely maximize both at once.\n"
        "PPO leans stability; classic DQN leans reuse; model-based methods lean imagination.\n"
        "Pick the corner that matches your environment cost and failure mode.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 9: Online vs offline
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Online interaction vs offline / batch RL",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 4.35, 5.5, 3.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 7.15, "Online", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 5.6,
        "Agent keeps acting,\n"
        "collecting fresh data,\n"
        "updating continually.\n\n"
        "Can explore; can also\n"
        "damage a real system.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 4.35, 5.5, 3.3, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 7.15, "Offline / batch", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 5.6,
        "Learn from a fixed dataset\n"
        "logged by some behavior.\n\n"
        "No new interaction --\n"
        "must fight distribution\n"
        "shift and overestimation.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )

    round_box(ax, 0.4, 0.4, 11.2, 3.5, C_WARN, "#d68910", lw=1.8)
    ax.text(
        6.0, 2.15,
        "Many production settings are offline-first (logs, simulators, human demos).\n"
        "Then: conservative objectives, uncertainty penalties, careful OPE.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#6e2c00",
    )
    return fig


# -------------------------------------------------------
# Step 10: Evaluation
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: Evaluation principles",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    rows = [
        (6.2, "Separate train / eval seeds", "Report mean ± variability over runs, not one lucky curve."),
        (4.85, "Match the deployment policy", "Eval on-policy return for the policy you will ship."),
        (3.5, "Watch for overfitting to env", "Fixed seeds / brittle sims can fake progress."),
        (2.15, "Offline? Use OPE carefully", "Importance sampling, fitted Q-eval, lower bounds -- with diagnostics."),
        (0.8, "Safety metrics matter", "Constraint violations, cost rates, tail risk -- not only mean return."),
    ]
    for y, title, body in rows:
        round_box(ax, 0.4, y, 11.2, 1.15, C_NEW, C_LOOP, lw=1.5)
        ax.text(2.6, y + 0.55, title, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        ax.text(7.7, y + 0.55, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")
    return fig


# -------------------------------------------------------
# Step 11: Takeaway
# -------------------------------------------------------
def build_step_11():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 11: Takeaway -- pick your corner deliberately",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    rows = [
        (6.2, "Explore / exploit", "Information now vs reward now."),
        (4.85, "On / off policy", "Fresh matched data vs reusable mismatched data."),
        (3.5, "Model / free", "Plan with dynamics vs learn values/policies directly."),
        (2.15, "MC / bootstrap", "Clean complete returns vs fast bootstrapped targets."),
        (0.8, "Ship checklist", "Stability, evaluation honesty, and safety constraints beat a pretty training curve."),
    ]
    for y, title, body in rows:
        round_box(ax, 0.4, y, 11.2, 1.15, C_GOAL, C_OK, lw=1.6)
        ax.text(2.4, y + 0.55, title, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_OK)
        ax.text(7.6, y + 0.55, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#145a32")
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
    build_step_10,
    build_step_11,
])


print("\n" + "=" * 60)
print("MAJOR RL PRINCIPLES")
print("=" * 60)
print("""
Axes (not algorithms):
  explore/exploit, on/off policy, model-based/free,
  bootstrap/MC, online/offline, efficiency/stability, evaluation.
""")
