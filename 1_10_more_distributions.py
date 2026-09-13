"""
1_10_more_distributions.py

Major probability distributions used in ML -- formulas and when to use them.

Beyond the Gaussian, this lecture surveys the rest of
the core toolkit:

  Discrete:  Bernoulli, Binomial, Categorical, Multinomial, Poisson
  Continuous: Uniform, Exponential, Beta
  Simplex:   Dirichlet (conjugate to Categorical / Multinomial)

Sketch:
  1. Map of the toolkit
  2. Bernoulli
  3. Binomial
  4. Categorical / Multinomial
  5. Uniform
  6. Exponential
  7. Poisson
  8. Beta
  9. Dirichlet
  10. Cheat sheet -- which distribution when
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
from scipy.stats import beta as beta_dist
from scipy.stats import binom, expon, poisson, uniform

np.random.seed(25)


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
C_PURPLE = "#8e44ad"


def formula_header(ax, lines: str):
    round_box(ax, 0.4, 6.2, 11.2, 1.5, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.95, lines,
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )


def when_box(ax, text: str, y=0.35, h=1.5):
    round_box(ax, 0.4, y, 11.2, h, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, y + h / 2, text,
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )


def nudge_right_panel_left(fig, axes, cm: float = 1.0) -> None:
    """Widen the left axes and shift the right axes left by ~cm centimeters."""
    plt.tight_layout()
    dx = (cm / 2.54) / fig.get_size_inches()[0]
    p0 = axes[0].get_position()
    p1 = axes[1].get_position()
    axes[0].set_position([p0.x0, p0.y0, p0.width + dx, p0.height])
    axes[1].set_position([p1.x0 - dx, p1.y0, p1.width, p1.height])


# -------------------------------------------------------
# Step 1: Map
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: A map of major distributions (beyond Gaussian)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    formula_header(
        ax,
        "Same job as a Gaussian: a named family with a formula and a role.\n"
        "Pick the family that matches the kind of randomness you have.",
    )

    cards = [
        (0.4, 3.4, "Discrete",
         "Bernoulli\nBinomial\nCategorical\nMultinomial\nPoisson"),
        (4.2, 3.4, "Continuous",
         "Uniform\nExponential\nBeta\n(+ Gaussian)"),
        (8.0, 3.4, "On the simplex",
         "Dirichlet\n\n(prior over\nCategorical probs)"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.5, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 2.05, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 0.95, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    when_box(
        ax,
        "Each next slide: formula  ·  mean / variance (when simple)  ·  when ML uses it.",
        y=0.45, h=1.4,
    )
    return fig


# -------------------------------------------------------
# Step 2: Bernoulli
# -------------------------------------------------------
def build_step_2():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 2: Bernoulli($p$) -- one coin flip",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    round_box(ax, 0.1, 5.5, 11.8, 2.1, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.55,
        r"$X\in\{0,1\}$"
        + "\n"
        r"$P(X=1)=p$"
        + r"$,\quad P(X=0)=1-p$"
        + "\n"
        r"$P(X=x)=p^x(1-p)^{1-x}$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.1, 2.6, 11.8, 2.5, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.85,
        r"$\mathbb{E}[X]=p$"
        + "\n"
        r"$\mathrm{Var}(X)=p(1-p)$"
        + "\n\n"
        "Binary label, click / no-click,\nsuccess / failure.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.1, 0.4, 11.8, 1.8, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.3,
        "Building block of logistic regression\n"
        "and binary cross-entropy.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    p = 0.7
    ax.bar([0, 1], [1 - p, p], color=["#85929e", C_LOOP], edgecolor="0.2", width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=FS_TICK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("probability", fontsize=FS_AXIS)
    ax.set_title(rf"$p={p:g}$", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, axis="y", alpha=0.3)

    nudge_right_panel_left(fig, axes, cm=1.0)
    return fig


# -------------------------------------------------------
# Step 3: Binomial
# -------------------------------------------------------
def build_step_3():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 3: Binomial($n,p$) -- count of successes",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    round_box(ax, 0.1, 5.3, 11.8, 2.3, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.45,
        r"$X=\#$ successes in $n$ i.i.d. Bernoulli($p$) trials"
        + "\n"
        r"$P(X=k)=\binom{n}{k} p^k(1-p)^{n-k}$"
        + r"$,\quad k=0,\ldots,n$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.1, 2.5, 11.8, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.7,
        r"$\mathbb{E}[X]=np$"
        + "\n"
        r"$\mathrm{Var}(X)=np(1-p)$"
        + "\n\n"
        "Sum of independent Bernoullis.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.1, 0.4, 11.8, 1.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.25,
        "A/B test conversions out of n users;\n"
        "number of correct classifications.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    n, p = 20, 0.35
    ks = np.arange(0, n + 1)
    ax.bar(ks, binom.pmf(ks, n, p), color=C_LOOP, edgecolor="0.2", width=0.8)
    ax.set_xlabel(r"$k$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$P(X=k)$", fontsize=FS_AXIS)
    ax.set_title(rf"$n={n},\ p={p:g}$", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, axis="y", alpha=0.3)

    nudge_right_panel_left(fig, axes, cm=1.0)
    return fig


# -------------------------------------------------------
# Step 4: Categorical / Multinomial
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: Categorical / Multinomial -- K-way outcomes",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.15, 11.2, 1.55, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.9,
        r"Categorical($\mathbf{p}$): one draw among $K$ classes"
        + "\n"
        r"$P(X=k)=p_k$"
        + r"$,\quad \mathbf{p}=(p_1,\ldots,p_K)$"
        + r"$,\ \sum_k p_k=1$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.55, 5.4, 2.35, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        3.1, 5.45,
        "Categorical",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        3.1, 4.35,
        "One-hot label / next-token id.\n"
        r"Softmax output $\approx \mathbf{p}$"
        + "\n"
        "(Softmax turns scores into a\n"
        r"probability vector $\mathbf{p}$)."
        + "\n"
        "Cross-entropy loss.",
        ha="center", va="center", fontsize=FS_TICK, color="#333333",
        linespacing=1.2,
    )

    round_box(ax, 6.2, 3.55, 5.4, 2.35, C_OLD, "#d35400", lw=2.0)
    ax.text(
        8.9, 5.45,
        "Multinomial",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color="#d35400",
    )
    ax.text(
        8.9, 4.35,
        r"$n$ = sample size (# draws)"
        + "\n"
        r"$\mathbf{c}=(c_1,\ldots,c_K)$: counts"
        + "\n"
        r"($c_k$ = how often class $k$; $\sum_k c_k=n$)"
        + "\n"
        r"$P(\mathbf{c})=\frac{n!}{c_1!\cdots c_K!}"
        r"\prod_k p_k^{c_k}$",
        ha="center", va="center", fontsize=FS_TICK, color="#333333",
        linespacing=1.25,
    )

    when_box(
        ax,
        r"Bernoulli = Categorical with $K=2$.   Binomial = Multinomial with $K=2$."
        + "\n"
        r"Read $\mathbf{c}$ as the count vector; $n$ is the sample size.",
        y=0.35, h=2.85,
    )
    return fig


# -------------------------------------------------------
# Step 5: Uniform
# -------------------------------------------------------
def build_step_5():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 5: Uniform($a,b$) -- flat on an interval",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    round_box(ax, 0.1, 5.3, 11.8, 2.3, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.45,
        r"$f(x)=\dfrac{1}{b-a}$  for  $x\in[a,b]$"
        + "\n"
        r"$f(x)=0$ otherwise",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.1, 2.5, 11.8, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.7,
        r"$\mathbb{E}[X]=\dfrac{a+b}{2}$"
        + "\n"
        r"$\mathrm{Var}(X)=\dfrac{(b-a)^2}{12}$"
        + "\n\n"
        "Maximum-entropy on a bounded interval.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.1, 0.4, 11.8, 1.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.25,
        "Default noise / random init on a box;\n"
        "proposal with no preferred value.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    a, b = -2.0, 3.0
    xs = np.linspace(a - 1.5, b + 1.5, 400)
    ax.plot(xs, uniform.pdf(xs, loc=a, scale=b - a), color="seagreen", lw=2.8)
    ax.fill_between(xs, uniform.pdf(xs, loc=a, scale=b - a), color="seagreen", alpha=0.25)
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$f(x)$", fontsize=FS_AXIS)
    ax.set_title(rf"$a={a:g},\ b={b:g}$", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)

    nudge_right_panel_left(fig, axes, cm=1.0)
    return fig


# -------------------------------------------------------
# Step 6: Exponential
# -------------------------------------------------------
def build_step_6():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 6: Exponential($\lambda$) -- waiting times",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    round_box(ax, 0.1, 5.3, 11.8, 2.3, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.45,
        r"$f(x)=\lambda e^{-\lambda x}\quad(x\geq 0)$"
        + "\n"
        r"$F(x)=1-e^{-\lambda x}$",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.1, 2.5, 11.8, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.7,
        r"$\mathbb{E}[X]=1/\lambda$"
        + "\n"
        r"$\mathrm{Var}(X)=1/\lambda^2$"
        + "\n\n"
        "Memoryless continuous lifetime.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.1, 0.4, 11.8, 1.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.25,
        "Inter-arrival times; positive skewed noise;\n"
        "ReLU-era intuition for tails.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    xs = np.linspace(0, 6, 400)
    for lam, c in [(0.5, "#5dade2"), (1.0, "darkorange"), (2.0, "seagreen")]:
        ax.plot(xs, expon.pdf(xs, scale=1 / lam), color=c, lw=2.5,
                label=rf"$\lambda={lam:g}$")
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$f(x)$", fontsize=FS_AXIS)
    ax.set_title("right-skewed, support $[0,\infty)$", fontsize=FS_AX_TITLE - 2,
                 fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, alpha=0.3)

    nudge_right_panel_left(fig, axes, cm=1.0)
    return fig


# -------------------------------------------------------
# Step 7: Poisson
# -------------------------------------------------------
def build_step_7():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 7: Poisson($\lambda$) -- counts in a fixed window",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    round_box(ax, 0.3, 5.55, 11.4, 2.05, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.55,
        r"$X\in\{0,1,2,\ldots\}$"
        + "\n"
        r"$P(X=k)=\dfrac{\lambda^k e^{-\lambda}}{k!}"
        r"\,,\quad\lambda>0$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.3, 3.15, 11.4, 2.15, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 4.2,
        r"$\mathbb{E}[X]=\lambda"
        r"\,,\quad\mathrm{Var}(X)=\lambda$"
        + "\n"
        "Mean equals variance."
        + "\n"
        r"Limit of Binomial($n,p$) with $np=\lambda$ fixed, $n\to\infty$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color=C_DARK,
    )
    round_box(ax, 0.3, 0.35, 11.4, 2.55, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.6,
        "Counts of rare / random events in a window:\n"
        "clicks per hour, arrivals, typos, defects.\n\n"
        r"Linked to Exponential: if inter-arrival times are Exp($\lambda$),"
        + "\n"
        r"the number of arrivals in time 1 is Poisson($\lambda$).",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    ax = axes[1]
    ks = np.arange(0, 18)
    for lam, c in [(2.0, "#5dade2"), (5.0, "darkorange"), (10.0, "seagreen")]:
        ax.plot(ks, poisson.pmf(ks, lam), "o-", color=c, lw=2.0, ms=5,
                label=rf"$\lambda={lam:g}$")
    ax.set_xlabel(r"$k$  (count)", fontsize=FS_AXIS)
    ax.set_ylabel(r"$P(X=k)$", fontsize=FS_AXIS)
    ax.set_title(r"Poisson PMF for several $\lambda$", fontsize=FS_AX_TITLE - 2,
                 fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98, 0.95,
        r"larger $\lambda$" + "\nshifts right",
        transform=ax.transAxes, ha="right", va="top", fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    nudge_right_panel_left(fig, axes, cm=1.0)
    return fig


# -------------------------------------------------------
# Step 8: Beta
# -------------------------------------------------------
def build_step_8():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 8: Beta($\alpha,\beta$) -- distribution on $(0,1)$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    round_box(ax, 0.1, 5.2, 11.8, 2.4, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.4,
        r"$f(x)=\dfrac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$"
        + r"$\quad (0<x<1)$"
        + "\n"
        r"$B(\alpha,\beta)=\int_0^1 t^{\alpha-1}(1-t)^{\beta-1}\,dt$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.1, 2.45, 11.8, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.65,
        r"$\mathbb{E}[X]=\dfrac{\alpha}{\alpha+\beta}$"
        + "\n\n"
        "Conjugate prior for Bernoulli / Binomial $p$.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.1, 0.35, 11.8, 1.75, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.2,
        "Belief over a probability:\n"
        "click-through rate, coin bias, calibration.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    xs = np.linspace(0.001, 0.999, 400)
    for a, b, c in [(0.5, 0.5, "#5dade2"), (2, 5, "darkorange"),
                    (5, 2, "seagreen"), (2, 2, C_PURPLE)]:
        ax.plot(xs, beta_dist.pdf(xs, a, b), color=c, lw=2.4,
                label=rf"$\alpha={a:g},\beta={b:g}$")
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$f(x)$", fontsize=FS_AXIS)
    ax.set_title("shapes on the unit interval", fontsize=FS_AX_TITLE - 2,
                 fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper center")
    ax.grid(True, alpha=0.3)

    nudge_right_panel_left(fig, axes, cm=1.0)
    return fig


# -------------------------------------------------------
# Step 9: Dirichlet
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 9: Dirichlet($\boldsymbol{\alpha}$) -- distribution on the simplex",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.05, 11.2, 1.65, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.85,
        r"$\mathbf{p}=(p_1,\ldots,p_K)$"
        + r"$\ $ with $\ p_k>0,\ \sum_k p_k=1$"
        + "\n"
        r"$f(\mathbf{p})\propto \prod_{k=1}^K p_k^{\alpha_k-1}$"
        + r"$,\quad \alpha_k>0$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.25, 5.4, 2.55, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        3.1, 5.35,
        "Role",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        3.1, 4.15,
        "Prior over Categorical /\nMultinomial probabilities.\n\n"
        r"Beta = Dirichlet with $K=2$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.2, 3.25, 5.4, 2.55, C_GOAL, C_OK, lw=2.0)
    ax.text(
        8.9, 5.35,
        "Mean",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    ax.text(
        8.9, 4.15,
        r"$\mathbb{E}[p_k]="
        r"\dfrac{\alpha_k}{\sum_j \alpha_j}$"
        + "\n\n"
        r"Larger $\sum\alpha$ → sharper"
        + "\naround the mean.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.4, 11.2, 2.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.7,
        "Topic models, Dirichlet priors in Bayesian softmax / LDA-style models,\n"
        "smoothing a discrete distribution before seeing much data.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 10: Cheat sheet
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: Which distribution when?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    rows = [
        (6.85, "Bernoulli / Binomial", "binary outcome or success count"),
        (5.95, "Categorical / Multinomial", "K-way label or token; bag-of-counts"),
        (5.05, "Gaussian", "real-valued noise; least-squares world"),
        (4.15, "Uniform", "flat on a box; uninformative bounded noise"),
        (3.25, "Exponential", "waiting / inter-arrival times"),
        (2.35, "Poisson", "event counts in a fixed window"),
        (1.45, "Beta / Dirichlet", "uncertainty over probabilities / simplex"),
    ]
    for y, left, right in rows:
        round_box(ax, 0.4, y - 0.35, 11.2, 0.9, C_NEW, C_LOOP, lw=1.5)
        ax.text(2.7, y + 0.1, left, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        ax.text(7.8, y + 0.1, right, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.25, 11.2, 0.85, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 0.67,
        r"Takeaway: averages settle; the distribution family still shapes the noise.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
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
])


print("\n" + "=" * 60)
print("MAJOR DISTRIBUTIONS -- FORMULAS")
print("=" * 60)
print("""
Bernoulli(p):     P(X=x)=p^x (1-p)^{1-x}
Binomial(n,p):    P(X=k)=C(n,k) p^k (1-p)^{n-k}
Categorical(p):   P(X=k)=p_k
Multinomial:      n! / (c1!...cK!) * prod p_k^{c_k}
Uniform(a,b):     f(x)=1/(b-a) on [a,b]
Exponential(l):   f(x)=l e^{-l x}  (x>=0)
Poisson(l):       P(X=k)=l^k e^{-l} / k!
Beta(a,b):        f(x) ~ x^{a-1}(1-x)^{b-1} on (0,1)
Dirichlet(a):     f(p) ~ prod p_k^{a_k-1} on the simplex

""")
