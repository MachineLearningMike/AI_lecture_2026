"""
1_020_probability_statistics_terms.py

A speaking / reference bank for core probability and statistics terms.

Sketch:
  1. Title
  2. Map of the term families
  3-5. Foundations (population, sample, parameter, ...)
  6-8. Randomness and probability
  9-12. Distributions and describing data
  13-15. Independence and relationships
  16-20. Inference: sample to population
  21. Types of data and variables
  22. Big-picture branches
  23. Takeaway

Sits near early Chapter 1: after why distributions matter (1_010),
before / beside density formalisms (1_015) and concrete families.
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
C_OLD = "#fce4d6"
C_LOOP = "#2874a6"


def section_banner(ax, text, y=6.45, h=1.2):
    round_box(ax, 0.35, y, 11.3, h, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, y + h / 2, text,
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )


def draw_term_grid(ax, cards, *, cols=2, y_top=6.15, row_h=2.7, gap=0.25):
    """Place term/definition cards in a grid. cards: (term, body)."""
    card_w = (11.3 - gap * (cols - 1)) / cols
    for i, (term, body) in enumerate(cards):
        r, c = divmod(i, cols)
        x = 0.35 + c * (card_w + gap)
        y = y_top - row_h - r * (row_h + gap)
        face = C_NEW if (r + c) % 2 == 0 else C_GOAL
        edge = C_LOOP if (r + c) % 2 == 0 else C_OK
        round_box(ax, x, y, card_w, row_h, face, edge, lw=1.8)
        ax.text(
            x + card_w / 2, y + row_h - 0.45, term,
            ha="center", va="center", fontsize=FS_ANNOT,
            fontweight="bold", color=C_DARK if edge == C_LOOP else C_OK,
        )
        ax.text(
            x + card_w / 2, y + row_h / 2 - 0.15, body,
            ha="center", va="center", fontsize=FS_TICK,
            color="#333333", linespacing=1.2,
        )


def draw_term_rows(ax, cards, *, y0=5.9, row_h=1.15, gap=0.18):
    """Full-width term rows: term on left, definition on right."""
    for i, (term, body) in enumerate(cards):
        y = y0 - i * (row_h + gap)
        round_box(ax, 0.35, y, 11.3, row_h, C_NEW if i % 2 == 0 else C_MUTE,
                  C_LOOP if i % 2 == 0 else "0.55", lw=1.6)
        ax.text(
            1.7, y + row_h / 2, term,
            ha="center", va="center", fontsize=FS_ANNOT - 1,
            fontweight="bold", color=C_DARK,
        )
        ax.text(
            7.2, y + row_h / 2, body,
            ha="center", va="center", fontsize=FS_TICK,
            color="#333333", linespacing=1.15,
        )


# -------------------------------------------------------
# Step 1: Title
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5, 0.62,
        "Probability &\nStatistics Terms",
        ha="center", va="center",
        fontsize=FS_FIGURE_TITLE + 10, fontweight="bold", color=C_OK,
    )
    ax.text(
        0.5, 0.28,
        "A shared vocabulary for Chapter 1\n"
        "Foundations → randomness → distributions → inference",
        ha="center", va="center", fontsize=FS_ANNOT, color="0.35",
    )
    return fig


# -------------------------------------------------------
# Step 2: Map
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: Seven families of terms",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    families = [
        (0.4, 5.5, "1. Foundations", "population, sample,\nparameter, statistic"),
        (4.2, 5.5, "2. Randomness", "experiment, event,\nprobability, RV"),
        (8.0, 5.5, "3. Distributions", "PMF / PDF / CDF,\nmean, variance"),
        (0.4, 2.6, "4. Relationships", "independence,\nBayes, correlation"),
        (4.2, 2.6, "5. Inference", "CI, tests,\np-values, errors"),
        (8.0, 2.6, "6–7. Data & branches", "scales; descriptive\nvs inferential"),
    ]
    for x, y, title, body in families:
        round_box(ax, x, y, 3.6, 2.4, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 1.75, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 0.85, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")
    round_box(ax, 0.4, 0.35, 11.2, 1.9, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 1.3,
        "Do not memorize as a dictionary dump.\n"
        "Learn each term by the question it answers.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Steps 3-5: Foundations
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: Foundations -- what we are studying",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        "Population vs sample. Parameter vs statistic. How we select data.",
    )
    points = [
        (0.5, 4.55, "Population", "The whole world of interest."),
        (4.25, 4.55, "Sample", "The part we actually see."),
        (8.0, 4.55, "Goal", "Use the sample to learn\nabout the population."),
    ]
    for x, y, title, body in points:
        round_box(ax, x, y, 3.5, 2.0, C_GOAL, C_OK, lw=2.0)
        ax.text(x + 1.75, y + 1.4, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
        ax.text(x + 1.75, y + 0.7, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")
    round_box(ax, 0.5, 0.45, 11.0, 3.6, C_OLD, "#d35400", lw=2.0)
    ax.text(
        6.0, 2.25,
        "Parameter = truth about the population (usually unknown).\n"
        "Statistic = number computed from the sample (known once data arrive).\n\n"
        "Most of statistics is: invent good statistics that track parameters.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#6e2c00",
    )
    return fig


def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: Foundations -- population, sample, parameter, statistic",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_grid(ax, [
        ("Population",
         "Entire set of individuals / items /\n"
         "events you want conclusions about.\n"
         "Ex: all adult humans;\nall bolts from a factory."),
        ("Sample",
         "A subset of the population you\n"
         "actually observe or measure."),
        ("Parameter",
         "A numerical characteristic of a\n"
         "population (true mean $\\mu$,\n"
         "true proportion $p$).\n"
         "Usually unknown and fixed."),
        ("Statistic",
         "A numerical characteristic of a\n"
         "sample (sample mean $\\bar{x}$,\n"
         "sample proportion $\\hat{p}$).\n"
         "Computed from data; estimates\nparameters."),
    ], cols=2, y_top=6.35, row_h=2.85, gap=0.22)
    return fig


def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: Foundations -- census and sampling",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_grid(ax, [
        ("Census",
         "Measure the whole population.\n"
         "Rare and often impractical."),
        ("Sampling",
         "The process of selecting a sample."),
    ], cols=2, y_top=6.35, row_h=2.4, gap=0.25)
    round_box(ax, 0.35, 0.4, 11.3, 3.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.2, "Key sampling types",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 1.7,
        "random  ·  stratified  ·  cluster  ·  systematic  ·  convenience\n\n"
        "Random / stratified / cluster / systematic aim for representativeness.\n"
        "Convenience is easy -- and often biased.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
        linespacing=1.25,
    )
    return fig


# -------------------------------------------------------
# Steps 6-8: Randomness & Probability
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Randomness & probability -- the stage",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, "Experiment → outcomes → sample space → events → probability")
    flow = [
        (0.4, 3.5, "Experiment\n/ Trial", "Uncertain process"),
        (3.1, 3.5, "Outcome", "One result"),
        (5.5, 3.5, "Sample space\n$S$ or $\\Omega$", "All outcomes"),
        (8.5, 3.5, "Event", "A subset of $\\Omega$"),
    ]
    for x, y, title, body in flow:
        round_box(ax, x, y, 2.4, 2.4, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 1.2, y + 1.55, title, ha="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        ax.text(x + 1.2, y + 0.65, body, ha="center",
                fontsize=FS_TICK, color="#333333")
    round_box(ax, 0.4, 0.4, 11.2, 2.7, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.75,
        r"Probability assigns each event a number in $[0,1]$."
        + "\n"
        "It measures how likely that event is.\n"
        r"A random variable turns outcomes into numbers.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#145a32",
    )
    return fig


def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: Randomness -- experiment to probability",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_rows(ax, [
        ("Experiment / Trial",
         "A process with uncertain outcomes."),
        ("Outcome",
         "A single possible result of the experiment."),
        ("Sample space ($S$, $\\Omega$)",
         "The set of all possible outcomes."),
        ("Event",
         "A subset of the sample space."),
        ("Probability",
         r"A number in $[0,1]$ measuring how likely an event is."),
    ], y0=6.05, row_h=1.05, gap=0.16)
    return fig


def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: Random variables -- discrete vs continuous",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_grid(ax, [
        ("Random variable",
         "A variable whose value depends\n"
         "on a random outcome.\n"
         "Ex: number of heads in 3 flips."),
        ("Discrete",
         "Countable values.\n"
         "Ex: counts, die faces,\ntoken ids."),
        ("Continuous",
         "Any value in a range.\n"
         "Ex: height, time, loss."),
        ("Why it matters",
         "Discrete → PMF / sticks.\n"
         "Continuous → PDF / areas.\n"
         "Both share a CDF."),
    ], cols=2, y_top=6.35, row_h=2.85, gap=0.22)
    return fig


# -------------------------------------------------------
# Steps 9-12: Distributions & describing data
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Distributions -- how probability is arranged",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, "Distribution = the full pattern of uncertainty over values")
    cards = [
        (0.4, 3.35, "Probability\ndistribution",
         "How probabilities are\nspread over possible values."),
        (4.2, 3.35, "PMF",
         "Probability mass function\nfor discrete variables."),
        (8.0, 3.35, "PDF",
         "Probability density function\nfor continuous variables."),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.6, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 1.85, title, ha="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 0.85, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")
    round_box(ax, 0.4, 0.4, 11.2, 2.6, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.7,
        r"CDF (cumulative distribution function):  $F(x)=P(X\leq x)$."
        + "\n"
        "One object that works for both discrete and continuous $X$.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#145a32",
    )
    return fig


def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: PMF, PDF, CDF",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_grid(ax, [
        ("Probability distribution",
         "How probabilities are spread\nover the possible values of $X$."),
        ("PMF",
         "Probability mass function.\n"
         r"$p(x)=P(X=x)$ for discrete $X$."
         + "\nSticks; they sum to 1."),
        ("PDF",
         "Probability density function.\n"
         "For continuous $X$; height is\n"
         "not a probability -- areas are."),
        ("CDF",
         r"$F(x)=P(X\leq x)$."
         + "\nCumulative: piles probability\n"
         r"from $-\infty$ up to $x$."),
    ], cols=2, y_top=6.35, row_h=2.85, gap=0.22)
    return fig


def build_step_11():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 11: Center -- mean, median, mode",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_grid(ax, [
        ("Mean / Expected value",
         r"$\mu$, $E[X]$ -- the average"
         + "\nor long-run center of $X$."),
        ("Median",
         "The middle value\n(50th percentile)."),
        ("Mode",
         "The most frequent value\n(or highest density region)."),
        ("Quick contrast",
         "Mean feels pull from tails.\n"
         "Median resists outliers.\n"
         "Mode is \"most typical\"."),
    ], cols=2, y_top=6.35, row_h=2.85, gap=0.22)
    return fig


def build_step_12():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 12: Spread and shape",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_rows(ax, [
        ("Variance ($\\sigma^2$)",
         "Average squared distance from the mean; measures spread."),
        ("Standard deviation ($\\sigma$)",
         "Square root of variance; spread in the original units."),
        ("Skewness / Kurtosis",
         "Asymmetry (skew) and tailedness / peakedness (kurtosis)."),
        ("Percentile / Quartile",
         "Position in a distribution (e.g. Q1, Q3, IQR = Q3 − Q1)."),
    ], y0=5.7, row_h=1.2, gap=0.2)
    return fig


# -------------------------------------------------------
# Steps 13-15: Independence & relationships
# -------------------------------------------------------
def build_step_13():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 13: Independence & relationships -- overview",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, "When do two things move together -- or not?")
    cards = [
        (0.4, 3.4, "Independence",
         "Neither influences\nthe other."),
        (4.2, 3.4, "Conditioning",
         r"$P(A\mid B)$ updates"
         + "\nA after seeing B."),
        (8.0, 3.4, "Dependence\nsummary",
         "Covariance /\ncorrelation."),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.5, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 1.75, title, ha="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 0.8, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")
    round_box(ax, 0.4, 0.4, 11.2, 2.6, C_WARN, C_ACCENT, lw=2.0)
    ax.text(
        6.0, 1.7,
        "Bayes' theorem is the disciplined way to update beliefs\n"
        "when new evidence arrives.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#641e16",
    )
    return fig


def build_step_14():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 14: Probability relationships",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_rows(ax, [
        ("Independence",
         "Two events / variables do not influence each other."),
        ("Conditional probability",
         r"$P(A\mid B)$ -- probability of $A$ given that $B$ occurred."),
        ("Joint probability",
         r"$P(A\cap B)$ -- probability that $A$ and $B$ both occur."),
        ("Marginal probability",
         "Probability of one variable after ignoring the others."),
    ], y0=5.7, row_h=1.2, gap=0.2)
    return fig


def build_step_15():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 15: Covariance, correlation, Bayes",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_grid(ax, [
        ("Covariance",
         "How two variables vary\ntogether (signed, units mix)."),
        ("Correlation ($r$)",
         "Standardized covariance.\n"
         r"Always in $[-1,1]$."),
        ("Bayes' theorem",
         "Updates a probability\ngiven new evidence.\n"
         r"$P(H\mid E)\propto P(E\mid H)P(H)$."),
        ("Reading tip",
         "Correlation ≠ causation.\n"
         "Bayes needs a prior;\n"
         "evidence reweights it."),
    ], cols=2, y_top=6.35, row_h=2.85, gap=0.22)
    return fig


# -------------------------------------------------------
# Steps 16-20: Inference
# -------------------------------------------------------
def build_step_16():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 16: Inference -- from sample to population",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, "Statistical inference: conclude about a population from a sample")
    cards = [
        (0.4, 3.35, "Estimate",
         "Guess a parameter\nwith a statistic."),
        (4.2, 3.35, "Interval",
         "Report a range likely\nto cover the parameter."),
        (8.0, 3.35, "Test",
         "Decide between claims\nusing data."),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.6, C_GOAL, C_OK, lw=2.0)
        ax.text(x + 1.8, y + 1.85, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
        ax.text(x + 1.8, y + 0.9, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")
    round_box(ax, 0.4, 0.4, 11.2, 2.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.7,
        "Because samples bounce, every statistic has a sampling distribution.\n"
        "Standard error / CLT explain that bounce.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


def build_step_17():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 17: Estimators, bias, sampling distribution",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_rows(ax, [
        ("Statistical inference",
         "Drawing conclusions about a population from a sample."),
        ("Estimator / Estimate",
         "A rule (estimator) / the number it returns (estimate) for a parameter."),
        ("Bias",
         "Systematic error: expected value of the estimator ≠ true parameter."),
        ("Sampling distribution",
         "Distribution of a statistic across many repeated samples."),
        ("Standard error",
         "Standard deviation of a sampling distribution."),
    ], y0=6.05, row_h=1.05, gap=0.14)
    return fig


def build_step_18():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 18: CLT and confidence intervals",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_grid(ax, [
        ("Central Limit Theorem",
         "Sample means approach a\n"
         "normal distribution as $n$\n"
         "grows (mild conditions)."),
        ("Confidence interval",
         "A range likely containing\n"
         "the true parameter\n"
         "(e.g. a 95% CI)."),
        ("What 95% means",
         "In repeated sampling, about\n"
         "95% of such intervals cover\n"
         "the true parameter."),
        ("Link",
         "CLT justifies normal-based\n"
         "SEs and many CI formulas\n"
         "for means / proportions."),
    ], cols=2, y_top=6.35, row_h=2.85, gap=0.22)
    return fig


def build_step_19():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 19: Hypothesis tests -- the claims",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_rows(ax, [
        ("Hypothesis test",
         "A procedure for deciding between claims using data."),
        ("Null hypothesis ($H_0$)",
         "Default / no-effect claim (the one we tentatively assume)."),
        ("Alternative ($H_1$)",
         "The claim you are testing for (effect / difference / signal)."),
        ("p-value",
         r"Probability of data this extreme if $H_0$ is true."),
        ("Significance level ($\\alpha$)",
         r"Threshold for rejecting $H_0$ (often $0.05$)."),
    ], y0=6.05, row_h=1.05, gap=0.14)
    return fig


def build_step_20():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 20: Errors and power",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_grid(ax, [
        ("Type I error",
         r"False positive: reject a true $H_0$."
         + "\n"
         r"Rate controlled by $\alpha$."),
        ("Type II error",
         r"False negative: fail to reject"
         + "\n"
         r"a false $H_0$."),
        ("Power",
         r"Probability of correctly rejecting"
         + "\n"
         r"a false $H_0$."
         + "\n(= 1 − Type II rate)."),
        ("Design tradeoff",
         r"Smaller $\alpha$ → fewer false"
         + "\nalarms, often less power.\n"
         "Larger $n$ raises power."),
    ], cols=2, y_top=6.35, row_h=2.85, gap=0.22)
    return fig


# -------------------------------------------------------
# Step 21: Types of data
# -------------------------------------------------------
def build_step_21():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 21: Types of data & variables",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_rows(ax, [
        ("Qualitative / Categorical",
         "Labels (color, yes/no, class id) -- not arithmetic magnitudes."),
        ("Quantitative / Numerical",
         "Numbers that measure magnitude."),
        ("Discrete vs continuous",
         "Counts vs measurements on a continuum (again)."),
        ("Nominal / Ordinal",
         "Named categories / ordered categories (weakest scales)."),
        ("Interval / Ratio",
         "Numeric with meaningful differences / also a true zero (strongest)."),
    ], y0=6.05, row_h=1.05, gap=0.14)
    return fig


# -------------------------------------------------------
# Step 22: Big-picture branches
# -------------------------------------------------------
def build_step_22():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 22: Big-picture branches",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    draw_term_grid(ax, [
        ("Descriptive statistics",
         "Summarize the data you have\n(mean, charts, tables)."),
        ("Inferential statistics",
         "Generalize from sample\nto population."),
        ("Probability theory",
         "The mathematical foundation\nunder both."),
        ("Bayesian vs Frequentist",
         "Two major philosophies of\ninference (priors & posteriors\nvs long-run procedures)."),
    ], cols=2, y_top=6.35, row_h=2.85, gap=0.22)
    return fig


# -------------------------------------------------------
# Step 23: Takeaway
# -------------------------------------------------------
def build_step_23():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 23: Takeaway -- questions each family answers",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)
    rows = [
        ("Foundations", "What world? What did we see? What is unknown vs computed?"),
        ("Randomness", "What can happen, and how likely is each event?"),
        ("Distributions", "How is uncertainty shaped? Where is center / spread?"),
        ("Relationships", "Does A change with B? How do we update on evidence?"),
        ("Inference", "What does the sample say about the population?"),
    ]
    for i, (title, body) in enumerate(rows):
        y = 6.15 - i * 1.15
        round_box(ax, 0.4, y, 11.2, 1.0, C_NEW if i % 2 == 0 else C_MUTE,
                  C_LOOP if i % 2 == 0 else "0.5", lw=1.5)
        ax.text(2.0, y + 0.5, title, ha="center", va="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(7.3, y + 0.5, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")
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
])


print("\n" + "=" * 60)
print("PROBABILITY & STATISTICS TERMS")
print("=" * 60)
print("""
Families:
  Foundations -- population, sample, parameter, statistic, census, sampling
  Randomness  -- experiment, outcome, Omega, event, probability, RV
  Distributions -- PMF / PDF / CDF, mean / median / mode, variance / sd
  Relationships -- independence, conditional / joint / marginal, Bayes
  Inference -- estimators, SE, CLT, CI, tests, p-values, errors, power
  Data types -- categorical / numeric; nominal to ratio
  Branches -- descriptive, inferential, probability; Bayesian vs Frequentist
""")
