"""
1_09_frequentist_vs_bayesian.py

Bayesian statistics through the Frequentist scaffold:
population, experiment, trial, outcome, sample space, event, probability.

Thesis: Bayesian work uses the same experimental framework; what changes is
that probability is also assigned over hypotheses / parameters about the
population -- not only over experimental outcomes.

Sketch:
  1. Thesis -- same experiment, different use of probability
  2. Population and unknown theta
  3. Experiment
  4. Trial
  5. Outcome
  6. Sample space
  7. Event (subset)
  8. Probability under a model
  9. Bayesian move -- priors on competing hypotheses
  10. Evidence D and P(D | M_i)
  11. Posterior update (worked coin example)
  12. Core chain -- likelihood bound iff posterior loses to complement
  13. What did not change
  14. Two probability spaces: outcomes vs hypotheses
  15. Likelihood vs posterior over theta
  16. Partition caveat -- cover the hypothesis space
  17. Takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_ANNOT,
    FS_TICK,
    FIGURE_DPI,
    apply_rcparams,
    screen_figsize,
    run_slide_deck,
)
apply_rcparams(plt)


def new_figure(*args, **kwargs):
    # Keep figure kwargs off the Axes path -- some Matplotlib builds forward
    # unexpected keywords from subplots(**kwargs) into Axes.set().
    figsize = kwargs.pop("figsize", screen_figsize())
    dpi = kwargs.pop("dpi", FIGURE_DPI)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if args or kwargs:
        ax = fig.add_subplot(*args, **kwargs)
        return fig, ax
    ax = fig.add_subplot()
    return fig, ax


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


def section_banner(ax, text, y=6.45, h=1.15):
    round_box(ax, 0.35, y, 11.3, h, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, y + h / 2, text,
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_LOOP = "#2874a6"
C_OK = "#1e8449"
C_GOAL = "#d4edda"
C_MUTE = "#f4f6f7"
C_OLD = "#fce4d6"
C_OLD_EDGE = "#d35400"
C_ACCENT = "#c0392b"
C_WARN = "#f8d7da"
C_FUN = "#8e44ad"
C_FUN_FACE = "#f5eef8"

# Worked example: D has 7 heads, 3 tails.
# P(D|M1)=0.5**10, P(D|M2)=0.7**7*0.3**3, P(D|M3)=0.9**7*0.1**3
# priors 0.5, 0.3, 0.2  -->  posteriors ~ 0.390, 0.533, 0.076
_PRIOR = (0.5, 0.3, 0.2)
_LIK = (
    0.5 ** 10,
    (0.7 ** 7) * (0.3 ** 3),
    (0.9 ** 7) * (0.1 ** 3),
)
_UNNORM = tuple(p * L for p, L in zip(_PRIOR, _LIK))
_Z = sum(_UNNORM)
_POST = tuple(u / _Z for u in _UNNORM)


# -------------------------------------------------------
# Step 1: Thesis
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: Bayesian statistics through the Frequentist scaffold",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 5.85, 11.2, 1.65, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.675,
        "Bayesian statistics does not require a different kind of experiment.\n"
        "It uses the same framework -- and interprets probability more broadly.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    row1 = [
        (0.5, "Population"),
        (3.5, "Experiment"),
        (6.5, "Trial"),
        (9.5, "Outcome"),
    ]
    row2 = [
        (0.5, r"Sample space $\Omega$"),
        (4.5, "Event (subset)"),
        (8.5, r"Probability $P$"),
    ]
    for x, label in row1:
        round_box(ax, x, 4.35, 2.7, 1.2, C_NEW, C_LOOP, lw=1.6)
        ax.text(x + 1.35, 4.95, label, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
    for x, label in row2:
        round_box(ax, x, 2.9, 3.5, 1.2, C_NEW, C_LOOP, lw=1.6)
        ax.text(x + 1.75, 3.5, label, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)

    round_box(ax, 0.4, 0.4, 11.2, 2.2, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.5,
        "Frequentist scaffold: probability over experimental outcomes.\n"
        "Bayesian extension: also assign probability over hypotheses\n"
        "about the population (models / parameters).",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#145a32",
    )
    return fig


# -------------------------------------------------------
# Step 2: Population
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: Population -- the complete system under study",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, "A population is the complete system or collection under study.")

    round_box(ax, 0.4, 3.55, 5.5, 2.55, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 5.55, "Running example", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 4.45,
        "A coin is tossed repeatedly\n"
        "under the same conditions.\n"
        "We want to know whether it is fair.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 3.55, 5.5, 2.55, C_OLD, C_OLD_EDGE, lw=2.0)
    ax.text(8.85, 5.55, "Unknown property", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OLD_EDGE)
    ax.text(
        8.85, 4.45,
        r"$\theta = P(\mathrm{Heads})$" + "\n\n"
        "A property of the coin's behavior.\n"
        r"We do not know $\theta$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.45, 11.2, 2.7, C_MUTE, "0.55", lw=1.6)
    ax.text(
        6.0, 1.8,
        "The population here is the coin's toss-generating process --\n"
        "the full system whose unknown fairness we want to learn.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 3: Experiment
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: Experiment -- an abstract, repeatable procedure",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        "An experiment defines the possible observations and the rules that produce them.",
    )

    round_box(ax, 1.5, 2.4, 9.0, 3.5, C_NEW, C_LOOP, lw=2.2)
    ax.text(6.0, 5.2, "Example experiment", ha="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
    ax.text(
        6.0, 3.7,
        "Toss the coin once and record\n"
        "whether it lands Heads or Tails.\n\n"
        "Abstract, repeatable, fully specified.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.45, 11.2, 1.55, C_MUTE, "0.55", lw=1.5)
    ax.text(
        6.0, 1.225,
        "Same definition as in the Frequentist scaffold -- Bayesian work reuses it.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 4: Trial
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: Trial -- one execution of the experiment",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, "Each trial produces exactly one outcome.")

    trials = [
        (0.7, "Trial 1", "Heads", C_GOAL, C_OK),
        (4.35, "Trial 2", "Tails", C_OLD, C_OLD_EDGE),
        (8.0, "Trial 3", "Heads", C_GOAL, C_OK),
    ]
    for x, title, body, face, edge in trials:
        round_box(ax, x, 2.6, 3.3, 3.2, face, edge, lw=2.0)
        ax.text(x + 1.65, 5.15, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=edge)
        ax.text(x + 1.65, 3.7, body, ha="center", va="center",
                fontsize=FS_AX_TITLE, fontweight="bold", color=C_DARK)

    round_box(ax, 0.4, 0.45, 11.2, 1.7, C_MUTE, "0.55", lw=1.5)
    ax.text(
        6.0, 1.3,
        "A trial is not a draw of one unit from a list of people.\n"
        "It is one run of the whole procedure on the system under study.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 5: Outcome
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: Outcome -- the result of a trial",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, r"For a single coin toss, the possible results are $H$ or $T$.")

    round_box(ax, 2.0, 2.5, 8.0, 3.4, C_NEW, C_LOOP, lw=2.2)
    ax.text(6.0, 5.2, "One-toss outcomes", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        6.0, 3.7,
        r"$\Omega = \{H,\, T\}$" + "\n\n"
        "The outcome of a trial is exactly one element of this set.",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2, color="#333333",
    )

    round_box(ax, 0.4, 0.45, 11.2, 1.6, C_MUTE, "0.55", lw=1.5)
    ax.text(
        6.0, 1.25,
        "Outcomes are what the experiment emits. Hypotheses about the coin are something else.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 6: Sample space
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Sample space -- all possible outcomes",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, "The sample space is the set of every outcome the experiment can produce.")

    round_box(ax, 0.4, 3.4, 5.5, 2.7, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 5.5, "One toss", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 4.3,
        r"$\Omega = \{H,\, T\}$" + "\n\n"
        "2 possible outcomes.",
        ha="center", va="center", fontsize=FS_TICK + 2, color="#333333",
    )

    round_box(ax, 6.1, 3.4, 5.5, 2.7, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 5.5, "Ten tosses (whole sequence)", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 4.3,
        r"$\Omega = \{H,\, T\}^{10}$" + "\n\n"
        r"$2^{10} = 1024$ sequences.",
        ha="center", va="center", fontsize=FS_TICK + 2, color="#333333",
    )

    round_box(ax, 0.4, 0.45, 11.2, 2.5, C_OLD, C_OLD_EDGE, lw=2.0)
    ax.text(
        6.0, 1.7,
        "If the experiment is \"toss ten times and record the sequence,\"\n"
        "then one trial yields one full length-10 string -- one point in "
        r"$\{H,T\}^{10}$.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 7: Event (subset)
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: Event -- a subset of the sample space",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, "An event collects the outcomes we care about as a yes/no question.")

    round_box(ax, 1.2, 2.3, 9.6, 3.6, C_NEW, C_LOOP, lw=2.2)
    ax.text(6.0, 5.3, "Ten-toss example", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        6.0, 3.7,
        r"$A = \{\text{sequences with exactly 7 heads}\}$" + "\n\n"
        "A is a subset of "
        r"$\Omega = \{H,T\}^{10}$" + ".\n"
        "It contains every length-10 string with seven H's.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.45, 11.2, 1.5, C_MUTE, "0.55", lw=1.5)
    ax.text(
        6.0, 1.2,
        "Probability will be assigned to events -- subsets -- not only to single outcomes.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 8: Probability under a model
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: Probability -- a measure on events, given a model",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        "A probability measure assigns a number in [0, 1] to each event.",
    )

    round_box(ax, 0.4, 3.35, 11.2, 2.75, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 4.7,
        r"$P(A) = P(\text{exactly 7 heads in 10 tosses})$" + "\n\n"
        "If the coin is fair "
        r"$(\theta = 0.5)$" + ":\n"
        r"$P(A) = \binom{10}{7}(0.5)^{10}$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.45, 11.2, 2.5, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.7,
        "These probabilities are fixed once we fix the population model "
        r"$(\theta)$" + ".\n"
        "That is the classical Frequentist reading: probability of outcomes\n"
        "under a stated data-generating process.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#145a32",
    )
    return fig


# -------------------------------------------------------
# Step 9: Bayesian move
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: The Bayesian move -- probability over hypotheses",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        "We may not know whether the coin is fair. Consider competing models.",
    )

    models = [
        (0.5, r"$M_1$", r"$\theta = 0.5$", "fair"),
        (4.25, r"$M_2$", r"$\theta = 0.7$", "biased"),
        (8.0, r"$M_3$", r"$\theta = 0.9$", "strongly biased"),
    ]
    for x, name, theta, gloss in models:
        round_box(ax, x, 3.35, 3.5, 2.7, C_FUN_FACE, C_FUN, lw=2.0)
        ax.text(x + 1.75, 5.45, name, ha="center",
                fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_FUN)
        ax.text(x + 1.75, 4.55, theta, ha="center",
                fontsize=FS_ANNOT + 1, color=C_DARK)
        ax.text(x + 1.75, 3.8, gloss, ha="center",
                fontsize=FS_TICK, color="0.35")

    round_box(ax, 0.4, 0.4, 11.2, 2.55, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 1.7,
        "A Bayesian assigns probabilities to these hypotheses:\n"
        r"$P(M_1),\ P(M_2),\ P(M_3)$"
        "  -- uncertainty about which model describes the population.\n"
        "This is the essential Bayesian move.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Step 10: Evidence and likelihoods
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: The experiment produces evidence",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, "Ten trials; seven heads. One observed sequence D.")

    round_box(ax, 0.4, 4.55, 11.2, 1.55, C_OLD, C_OLD_EDGE, lw=2.0)
    ax.text(
        6.0, 5.325,
        r"$D = (H,H,T,H,H,T,H,H,H,T)$" + "   (7 heads, 3 tails)",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#333333",
    )

    cards = [
        (0.4, r"$M_1$", r"$P(D\mid M_1)=0.5^{10}$", f"= {_LIK[0]:.4g}"),
        (4.25, r"$M_2$", r"$P(D\mid M_2)=0.7^{7}(0.3)^{3}$", f"= {_LIK[1]:.4g}"),
        (8.0, r"$M_3$", r"$P(D\mid M_3)=0.9^{7}(0.1)^{3}$", f"= {_LIK[2]:.4g}"),
    ]
    for x, name, formula, val in cards:
        round_box(ax, x, 1.55, 3.5, 2.6, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 1.75, 3.6, name, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.75, 2.75, formula, ha="center",
                fontsize=FS_TICK, color="#333333")
        ax.text(x + 1.75, 2.0, val, ha="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_LOOP)

    round_box(ax, 0.4, 0.35, 11.2, 0.95, C_MUTE, "0.55", lw=1.4)
    ax.text(
        6.0, 0.825,
        "These are probabilities of the observed outcome, conditional on each hypothesis.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 11: Posterior update
# -------------------------------------------------------
def build_step_11():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 11: Posterior probability of each hypothesis",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.35, 6.35, 11.3, 1.25, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.975,
        r"$P(M_i\mid D)=\dfrac{P(D\mid M_i)\,P(M_i)}"
        r"{\sum_j P(D\mid M_j)\,P(M_j)}$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2, color="#f9e79f",
    )

    round_box(ax, 0.4, 4.55, 11.2, 1.45, C_MUTE, "0.55", lw=1.5)
    ax.text(
        6.0, 5.275,
        r"Priors:  $P(M_1)=0.5$,   $P(M_2)=0.3$,   $P(M_3)=0.2$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
    )

    posts = [
        (0.4, r"$M_1$", _PRIOR[0], _POST[0], C_NEW, C_LOOP),
        (4.25, r"$M_2$", _PRIOR[1], _POST[1], C_GOAL, C_OK),
        (8.0, r"$M_3$", _PRIOR[2], _POST[2], C_FUN_FACE, C_FUN),
    ]
    for x, name, prior, post, face, edge in posts:
        round_box(ax, x, 1.55, 3.5, 2.65, face, edge, lw=2.0)
        ax.text(x + 1.75, 3.65, name, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=edge)
        ax.text(
            x + 1.75, 2.55,
            f"prior {prior:.1f}\n"
            f"posterior {post:.3f}",
            ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
        )

    round_box(ax, 0.4, 0.35, 11.2, 0.95, C_OLD, C_OLD_EDGE, lw=1.6)
    ax.text(
        6.0, 0.825,
        r"$M_2$ rises: seven heads in ten tosses favor $\theta=0.7$ over fair or extreme.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 12: Core chain -- likelihood bound iff H loses to not-H
# -------------------------------------------------------
def build_step_12():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 12: When does $H$ lose to $\bar H$ after seeing $D$?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        r"Assume $P(H)>0$, $P(D)>0$.  An equivalence -- not a one-way implication.",
        y=6.5, h=1.05,
    )

    round_box(ax, 0.35, 5.15, 11.3, 1.15, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 5.725,
        r"$P(D\mid H)<\dfrac{P(D)}{2P(H)}"
        r"\;\Longleftrightarrow\;"
        r"P(H\mid D)<P(\bar H\mid D)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 3, color="#f9e79f",
    )

    links = [
        (4.05, C_NEW, C_LOOP,
         r"$P(D\mid H)<\dfrac{P(D)}{2P(H)}$",
         r"multiply by $P(H)>0$"),
        (2.95, C_MUTE, "0.55",
         r"$P(D\cap H)<\dfrac{P(D)}{2}$",
         r"$P(D)=P(D\cap H)+P(D\cap\bar H)$"),
        (1.85, C_OLD, C_OLD_EDGE,
         r"$P(D\cap H)<P(D\cap\bar H)$",
         r"divide by $P(D)>0$"),
        (0.75, C_GOAL, C_OK,
         r"$P(H\mid D)<P(\bar H\mid D)$",
         r"$H$ loses to $\bar H$ given $D$"),
    ]
    for i, (y, face, edge, formula, hint) in enumerate(links):
        round_box(ax, 0.35, y, 7.6, 0.95, face, edge, lw=1.7)
        ax.text(4.15, y + 0.475, formula, ha="center", va="center",
                fontsize=FS_ANNOT, color="#333333")
        round_box(ax, 8.15, y, 3.5, 0.95, C_MUTE, "0.65", lw=1.2)
        ax.text(9.9, y + 0.475, hint, ha="center", va="center",
                fontsize=FS_TICK, color="0.35")
        if i < len(links) - 1:
            ax.text(4.15, y - 0.08, "\u21d5", ha="center", va="top",
                    fontsize=FS_ANNOT + 2, color=C_LOOP, fontweight="bold")

    return fig


# -------------------------------------------------------
# Step 13: What did not change
# -------------------------------------------------------
def build_step_13():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 13: The experiment did not change",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, "Bayesian updating reuses the same empirical machinery.")

    same = [
        (0.5, 3.55, "Experiment", "Same toss procedure"),
        (4.25, 3.55, "Trials", "Same ten executions"),
        (8.0, 3.55, "Outcomes", "Same observed D"),
        (0.5, 1.55, "Sample space", r"Still $\{H,T\}^{10}$"),
        (4.25, 1.55, "Events / $P$", "Same likelihood\nmachine under each $M_i$"),
    ]
    for x, y, title, body in same:
        round_box(ax, x, y, 3.5, 1.7, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 1.75, y + 1.2, title, ha="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(x + 1.75, y + 0.55, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")

    round_box(ax, 8.0, 1.55, 3.5, 1.7, C_GOAL, C_OK, lw=2.2)
    ax.text(9.75, 2.75, "What changed", ha="center",
            fontsize=FS_ANNOT - 1, fontweight="bold", color=C_OK)
    ax.text(
        9.75, 2.1,
        "Probabilities on\nhypotheses "
        r"$M_i$",
        ha="center", va="center", fontsize=FS_TICK, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 14: Two probability spaces
# -------------------------------------------------------
def build_step_14():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 14: Two probability spaces to keep apart",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 3.35, 5.5, 4.15, C_NEW, C_LOOP, lw=2.2)
    ax.text(3.15, 7.0, "A. Outcome space", ha="center",
            fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 5.35,
        r"$\Omega = \{H,T\}^{10}$" + "\n\n"
        "Question:\n"
        "How likely is a particular\n"
        "outcome under a given model?\n\n"
        r"$P(D\mid M_2)$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 3.35, 5.5, 4.15, C_FUN_FACE, C_FUN, lw=2.2)
    ax.text(8.85, 7.0, "B. Hypothesis space", ha="center",
            fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_FUN)
    ax.text(
        8.85, 5.35,
        r"$\mathcal{M} = \{M_1,M_2,M_3\}$" + "\n\n"
        "Question:\n"
        "Given the observed outcome,\n"
        "how plausible is each hypothesis?\n\n"
        r"$P(M_2\mid D)$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.4, 11.2, 2.55, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 1.675,
        "Bayesian inference applies probability to uncertainty over hypotheses\n"
        "or parameters -- not merely to the outcomes of repeated trials.\n"
        "That is the fundamental extension.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Step 15: Likelihood vs posterior over theta
# -------------------------------------------------------
def build_step_15():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 15: Frequentist estimate vs Bayesian posterior over $\theta$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        "Same data D; two different objects placed on the parameter.",
    )

    round_box(ax, 0.4, 2.55, 5.5, 3.5, C_OLD, C_OLD_EDGE, lw=2.0)
    ax.text(3.15, 5.5, "Frequentist tools", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OLD_EDGE)
    ax.text(
        3.15, 3.9,
        r"$\hat{\theta} = 7/10$" + "\n\n"
        r"$L(\theta; D) = P(D\mid\theta)$" + "\n\n"
        "The likelihood is a score for "
        r"$\theta$" + ",\n"
        "not a probability distribution over "
        r"$\theta$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 2.55, 5.5, 3.5, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 5.5, "Bayesian update", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 3.9,
        "Combine likelihood with a prior:\n\n"
        r"$p(\theta\mid D)\;\propto\;"
        r"p(D\mid\theta)\,p(\theta)$" + "\n\n"
        "Now uncertainty over "
        r"$\theta$" + " is a\n"
        "genuine probability distribution.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.4, 11.2, 1.8, C_MUTE, "0.55", lw=1.5)
    ax.text(
        6.0, 1.3,
        "Experiment → outcomes.  Likelihood → scores models.\n"
        "Bayesian probability → uncertainty over those models.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 16: Partition caveat
# -------------------------------------------------------
def build_step_16():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 16: Cover the hypothesis space -- not the sample space",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        r"The sum in $P(D)=\sum_i P(D\mid M_i)P(M_i)$ needs a partition of $\mathcal{M}$.",
    )

    round_box(ax, 0.4, 3.45, 5.5, 2.6, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 5.5, "Sample space", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 4.35,
        r"$\Omega = \{H,T\}^{10}$" + "\n"
        "What outcomes can the\nexperiment produce?",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 3.45, 5.5, 2.6, C_FUN_FACE, C_FUN, lw=2.0)
    ax.text(8.85, 5.5, "Hypothesis / parameter space", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_FUN)
    ax.text(
        8.85, 4.35,
        r"$\Theta = [0,1]$" + "  (all coin biases)\n"
        "What population models\nare we entertaining?",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.75, C_WARN, C_ACCENT, lw=2.0)
    ax.text(
        6.0, 1.725,
        r"$M_1,M_2,M_3$ do not cover $[0,1]$." + "\n"
        r"Setting $P(M_1)+P(M_2)+P(M_3)=1$ deliberately restricts the model."
        + "\n"
        "Discrete case: sum over all hypotheses in the chosen "
        r"$\mathcal{M}$." + "\n"
        r"Continuous case: $p(\theta\mid D)="
        r"\dfrac{p(D\mid\theta)p(\theta)}"
        r"{\int_\Theta p(D\mid\theta')p(\theta')\,d\theta'}$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 17: Takeaway
# -------------------------------------------------------
def build_step_17():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 17: Takeaway",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    cards = [
        (0.4, 4.15, 3.7, 3.2, C_NEW, C_LOOP,
         "Same scaffold",
         "Population, experiment,\ntrial, outcome, "
         r"$\Omega$" + ",\nevent, and "
         r"$P(\cdot\mid M)$"
         "\nare shared."),
        (4.15, 4.15, 3.7, 3.2, C_FUN_FACE, C_FUN,
         "Bayesian extension",
         "Also put probability on\nhypotheses / parameters\nabout the population,\nthen update with data."),
        (7.9, 4.15, 3.7, 3.2, C_GOAL, C_OK,
         "Keep spaces straight",
         "Outcome space "
         r"$\Omega$" + "\n"
         "vs hypothesis space "
         r"$\mathcal{M}$" + " / "
         r"$\Theta$" + ".\n"
         "Partition the latter."),
    ]
    for x, y, w, h, face, edge, title, body in cards:
        round_box(ax, x, y, w, h, face, edge, lw=2.0)
        ax.text(x + w / 2, y + h - 0.55, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=edge)
        ax.text(x + w / 2, y + 1.35, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")

    round_box(ax, 0.4, 0.45, 11.2, 3.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 2.05,
        "The experiment generates outcomes.\n"
        "The likelihood evaluates those outcomes under possible population models.\n"
        "Bayesian probability expresses uncertainty over those models.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#f9e79f",
    )
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
    build_step_12,
    build_step_13,
    build_step_14,
    build_step_15,
    build_step_16,
    build_step_17,
])
