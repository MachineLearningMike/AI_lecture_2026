"""
1_22_sum_of_randomity.py

Sum of random variables -- formulas and proofs.

Core results (always true for finite moments):

    E[X + Y] = E[X] + E[Y]

    Var(X + Y) = Var(X) + Var(Y) + 2 Cov(X, Y)

If X and Y are uncorrelated (in particular, independent), Cov=0 and variances add.

Sketch:
  1. What "sum of randomity" means
  2. The two formulas on one board
  3. Proof: linearity of expectation
  4. Proof: variance expansion with covariance
  5. Independent / uncorrelated special case
  6. Demo: independent Gaussians -- sum gets wider
  7. Demo: correlated pair -- Cov term matters
  8. Many variables
  9. Bridge to the average M = S / n
  10. Takeaway
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
from scipy.stats import norm

np.random.seed(67)
RNG = np.random.default_rng(67)


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
C_GRAY = "#85929e"
C_LOOP = "#2874a6"

MU, SIGMA = 0.0, 1.0
N = 20000


# -------------------------------------------------------
# Step 1: Idea
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: Sum of randomity -- a new random variable from old ones",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 5.7, 11.2, 1.9, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.65,
        r"Given random variables $X$ and $Y$, their sum"
        + "\n"
        r"$S = X + Y$  is again a random variable.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.4, 2.6, "One experiment", "Draw $X$, draw $Y$,\nadd the numbers."),
        (4.2, 2.6, "New randomness", "Uncertainty of $X$ and $Y$\ncombine into $S$."),
        (8.0, 2.6, "Questions", r"Where is $S$ centered?"
         + "\n"
         r"How wide is $S$?"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.7, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 2.0, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 1.0, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.35, 11.2, 1.9, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.3,
        "This lecture proves how means and variances transform under addition.\n"
        "Products are different. Averages are sums scaled by 1/n.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: Formula board
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: The two formulas to prove",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 4.35, 11.2, 3.2, C_GOAL, C_OK, lw=2.2)
    ax.text(
        6.0, 6.85, "Expectation always adds",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    ax.text(
        6.0, 5.5,
        r"$E[X+Y]=E[X]+E[Y]$"
        + "\n\n"
        "No independence needed. Finite means are enough.",
        ha="center", va="center", fontsize=FS_ANNOT + 2, color="#145a32",
    )

    round_box(ax, 0.4, 0.4, 11.2, 3.6, C_NEW, C_LOOP, lw=2.2)
    ax.text(
        6.0, 3.45, "Variance needs a covariance correction",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 1.85,
        r"$\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)+2\,\mathrm{Cov}(X,Y)$"
        + "\n\n"
        r"If $\mathrm{Cov}(X,Y)=0$ (uncorrelated / independent), variances add."
        + "\n"
        r"$\mathrm{Cov}(X,Y)=E[(X-E X)(Y-E Y)]=E[XY]-E[X]E[Y]$.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 3: Proof of E[X+Y]
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 3: Proof -- $E[X+Y]=E[X]+E[Y]$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.2, 11.2, 1.4, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.9,
        "Linearity is a property of the integral / sum that defines expectation.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    # Continuous proof
    round_box(ax, 0.4, 3.15, 5.5, 2.8, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        3.15, 5.5, "Continuous case",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        3.15, 4.2,
        r"$E[X+Y]=\iint (x+y)\,f_{X,Y}(x,y)\,dx\,dy$"
        + "\n"
        r"$=\iint x\,f\,dx\,dy+\iint y\,f\,dx\,dy$"
        + "\n"
        r"$=E[X]+E[Y]$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
        linespacing=1.45,
    )

    # Discrete proof
    round_box(ax, 6.1, 3.15, 5.5, 2.8, C_GOAL, C_OK, lw=2.0)
    ax.text(
        8.85, 5.5, "Discrete case",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    ax.text(
        8.85, 4.2,
        r"$E[X+Y]=\sum_{x,y}(x+y)\,P(X=x,Y=y)$"
        + "\n"
        r"$=\sum_{x,y} x\,P+\sum_{x,y} y\,P$"
        + "\n"
        r"$=E[X]+E[Y]$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
        linespacing=1.45,
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.5, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.6,
        "Key point: we never factored the joint into $f_X f_Y$.\n"
        "Dependence is allowed. Linearity does not require independence.\n"
        r"Same proof gives $E[aX+bY]=aE[X]+bE[Y]$.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 4: Proof of Var expansion
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 4: Proof -- $\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)+2\mathrm{Cov}(X,Y)$",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    steps = [
        (6.9, "1. Start from the definition",
         r"$\mathrm{Var}(X+Y)=E\left[((X+Y)-E[X+Y])^2\right]$"),
        (5.35, "2. Center using linearity",
         r"$E[X+Y]=E[X]+E[Y]$, so the centered sum is"
         + "\n"
         r"$(X-EX)+(Y-EY)$."),
        (3.55, "3. Expand the square",
         r"$((X-EX)+(Y-EY))^2$"
         + "\n"
         r"$=(X-EX)^2+(Y-EY)^2+2(X-EX)(Y-EY)$"),
        (1.45, "4. Take expectation",
         r"$E[(X-EX)^2]=\mathrm{Var}(X)$,  "
         r"$E[(Y-EY)^2]=\mathrm{Var}(Y)$,"
         + "\n"
         r"$E[(X-EX)(Y-EY)]=\mathrm{Cov}(X,Y)$."),
    ]
    for i, (y, title, body) in enumerate(steps):
        face = C_NEW if i % 2 == 0 else C_GOAL
        edge = C_LOOP if i % 2 == 0 else C_OK
        round_box(ax, 0.35, y - 0.05, 11.3, 1.45, face, edge, lw=1.7)
        ax.text(2.05, y + 0.95, title, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        ax.text(7.35, y + 0.65, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333", linespacing=1.25)
    return fig


# -------------------------------------------------------
# Step 5: Independent special case
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: When variances simply add",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 5.5, 11.2, 2.1, C_GOAL, C_OK, lw=2.2)
    ax.text(
        6.0, 6.55,
        r"If $X\perp Y$ (independent), then $E[XY]=E[X]E[Y]$,"
        + "\n"
        r"hence $\mathrm{Cov}(X,Y)=0$, and"
        + "\n"
        r"$\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)$.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#145a32",
    )

    round_box(ax, 0.4, 2.85, 5.5, 2.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        3.15, 4.55, "Uncorrelated is enough",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        3.15, 3.6,
        r"$\mathrm{Cov}=0$ already kills the"
        + "\n"
        "cross term. Independence is"
        + "\n"
        "stronger than uncorrelated.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 2.85, 5.5, 2.3, C_WARN, C_ACCENT, lw=2.0)
    ax.text(
        8.85, 4.55, "Positive dependence",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT,
    )
    ax.text(
        8.85, 3.6,
        r"$\mathrm{Cov}>0$ makes the sum"
        + "\n"
        "wider than the independent case."
        + "\n"
        r"$\mathrm{Cov}<0$ makes it narrower.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#641e16",
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.15, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.4,
        r"For i.i.d. $X_i$ with variance $\sigma^2$:"
        + "\n"
        r"$\mathrm{Var}(X_1+\cdots+X_n)=n\sigma^2$."
        + "  That is why averages shrink like $1/n$ in variance.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 6: Independent Gaussian demo
# -------------------------------------------------------
def build_step_6():
    fig, axes = new_figure(1, 3)
    fig.suptitle(
        r"Step 6: Independent Gaussians -- $S=X+Y$ is wider",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    X = RNG.normal(MU, SIGMA, size=N)
    Y = RNG.normal(MU, SIGMA, size=N)
    S = X + Y
    panels = [
        (axes[0], X, MU, SIGMA, r"$X$", "steelblue", r"$\mathrm{Var}=1$"),
        (axes[1], Y, MU, SIGMA, r"$Y$ (indep.)", "#5dade2", r"$\mathrm{Var}=1$"),
        (axes[2], S, 2 * MU, SIGMA * np.sqrt(2), r"$S=X+Y$", "seagreen",
         r"$\mathrm{Var}=2$"),
    ]
    for ax, data, mean, std, title, color, note in panels:
        ax.hist(data, bins=55, density=True, alpha=0.75, color=color,
                edgecolor="white", linewidth=0.4)
        xs = np.linspace(mean - 4 * std, mean + 4 * std, 400)
        ax.plot(xs, norm.pdf(xs, mean, std), "r-", lw=2.3, label="theory")
        ax.axvline(mean, color="k", ls="--", alpha=0.55)
        ax.set_title(title, fontsize=FS_AX_TITLE - 1)
        ax.set_xlabel("value", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.98, 0.95,
            note + f"\nemp var={np.var(data, ddof=1):.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=FS_TICK,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
        )
        ax.legend(fontsize=FS_LEGEND - 2, loc="upper left")
    axes[0].set_ylabel("Density", fontsize=FS_AXIS)
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 7: Correlated demo
# -------------------------------------------------------
def build_step_7():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 7: Correlation changes $\mathrm{Var}(X+Y)$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # Build correlated standard normals via Gaussian copula-style linear mix
    Z1 = RNG.normal(size=N)
    Z2 = RNG.normal(size=N)
    rho = 0.8
    X = Z1
    Y = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
    S = X + Y
    var_theory = 1.0 + 1.0 + 2.0 * rho  # = 2 + 2 rho

    ax = axes[0]
    ax.scatter(X[::10], Y[::10], s=8, alpha=0.35, c=C_LOOP, edgecolors="none")
    ax.set_xlabel(r"$X$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$Y$", fontsize=FS_AXIS)
    ax.set_title(fr"joint cloud, $\rho={rho:g}$", fontsize=FS_AX_TITLE - 1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.98,
        fr"Cov$(X,Y)\approx{np.cov(X,Y,ddof=1)[0,1]:.2f}$"
        + "\n"
        fr"theory $\rho={rho:g}$",
        transform=ax.transAxes, ha="left", va="top", fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )

    ax = axes[1]
    ax.hist(S, bins=55, density=True, alpha=0.75, color="seagreen",
            edgecolor="white", linewidth=0.4, label="samples of S")
    std_s = np.sqrt(var_theory)
    xs = np.linspace(S.mean() - 4 * std_s, S.mean() + 4 * std_s, 400)
    ax.plot(xs, norm.pdf(xs, S.mean(), std_s), "r-", lw=2.4,
            label=fr"theory $N(0,{var_theory:.1f})$")
    # Independent reference
    xs0 = np.linspace(-6, 6, 400)
    ax.plot(xs0, norm.pdf(xs0, 0, np.sqrt(2)), "b--", lw=2.0, alpha=0.7,
            label=r"if indep: $N(0,2)$")
    ax.set_title(r"$S=X+Y$ is wider when $\rho>0$", fontsize=FS_AX_TITLE - 1)
    ax.set_xlabel("value", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND - 2, loc="upper right")
    ax.text(
        0.02, 0.02,
        fr"theory Var$(S)=2+2\rho={var_theory:.1f}$"
        + "\n"
        fr"emp Var$(S)={np.var(S, ddof=1):.2f}$",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 8: Many variables
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 8: Many variables -- $S_n=X_1+\cdots+X_n$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 5.55, 11.2, 2.05, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.55,
        r"$E[S_n]=\sum_{i=1}^n E[X_i]$"
        + "\n"
        r"$\mathrm{Var}(S_n)=\sum_i\mathrm{Var}(X_i)"
        r"+2\sum_{i<j}\mathrm{Cov}(X_i,X_j)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 2.85, 5.5, 2.4, C_GOAL, C_OK, lw=2.0)
    ax.text(
        3.15, 4.7, "i.i.d. case",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    ax.text(
        3.15, 3.7,
        r"$E[S_n]=n\mu$"
        + "\n"
        r"$\mathrm{Var}(S_n)=n\sigma^2$"
        + "\n"
        r"(all Cov terms vanish)",
        ha="center", va="center", fontsize=FS_ANNOT, color="#145a32",
    )

    round_box(ax, 6.1, 2.85, 5.5, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        8.85, 4.7, "Proof idea",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        8.85, 3.7,
        "Expand "
        r"$E[(\sum_i (X_i-E X_i))^2]$."
        + "\n"
        "Diagonal terms = variances."
        + "\n"
        "Off-diagonal = covariances.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.15, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.4,
        "Same algebra as the two-variable proof -- just more cross terms.\n"
        "Independence / uncorrelated pairs delete the double sum.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 9: Bridge to average
# -------------------------------------------------------
def build_step_9():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 9: From sum to average -- $M_n=S_n/n$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # i.i.d. N(0,1), compare S and M for n=4
    n = 4
    X = RNG.normal(MU, SIGMA, size=(N, n))
    S = X.sum(axis=1)
    M = S / n

    ax = axes[0]
    ax.hist(S, bins=55, density=True, alpha=0.75, color="seagreen",
            edgecolor="white", linewidth=0.4)
    std_s = SIGMA * np.sqrt(n)
    xs = np.linspace(-4 * std_s, 4 * std_s, 400)
    ax.plot(xs, norm.pdf(xs, 0, std_s), "r-", lw=2.3)
    ax.set_title(fr"$S_4$: Var $= {n}\sigma^2={n}$", fontsize=FS_AX_TITLE - 1)
    ax.set_xlabel("value", fontsize=FS_AXIS)
    ax.set_ylabel("Density", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(M, bins=55, density=True, alpha=0.75, color="darkorange",
            edgecolor="white", linewidth=0.4)
    std_m = SIGMA / np.sqrt(n)
    xs = np.linspace(-4 * std_m, 4 * std_m, 400)
    ax.plot(xs, norm.pdf(xs, 0, std_m), "r-", lw=2.3, label="theory")
    ax.plot(xs, norm.pdf(xs, 0, SIGMA), "b--", lw=1.8, alpha=0.6, label=r"one $X$")
    ax.set_title(fr"$M_4=S_4/4$: Var $=\sigma^2/{n}$", fontsize=FS_AX_TITLE - 1)
    ax.set_xlabel("value", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")

    fig.text(
        0.5, 0.02,
        r"Scaling: $E[aZ]=aE[Z]$,  $\mathrm{Var}(aZ)=a^2\mathrm{Var}(Z)$."
        r"  So $M_n=S_n/n$ keeps the mean and divides variance by $n^2$.",
        ha="center", va="bottom", fontsize=FS_ANNOT, color="0.25",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 10: Takeaway
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: Takeaway -- sums add means; variances need Cov",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    rows = [
        (5.9, "Expectation", r"$E[X+Y]=E[X]+E[Y]$ always (linearity)."),
        (4.5, "Variance", r"$\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)+2\mathrm{Cov}(X,Y)$."),
        (3.1, "Independence", r"$\mathrm{Cov}=0$ $\Rightarrow$ variances add."),
        (1.7, "Related ideas", "Averages shrink spread. Products are different."),
    ]
    for y, title, body in rows:
        round_box(ax, 0.4, y, 11.2, 1.15, C_NEW if title != "Related ideas" else C_MUTE,
                  C_LOOP if title != "Related ideas" else "0.5", lw=1.6)
        ax.text(2.1, y + 0.55, title, ha="center", va="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(7.4, y + 0.55, body, ha="center", va="center",
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
])


print("\n" + "=" * 60)
print("SUM OF RANDOMITY")
print("=" * 60)
print("""
Formulas:
  E[X+Y] = E[X] + E[Y]                         (always)
  Var(X+Y) = Var(X) + Var(Y) + 2 Cov(X,Y)

Uncorrelated / independent => Cov=0 => variances add.

iid sum of n:
  E[S_n] = n mu
  Var(S_n) = n sigma^2
  average M_n = S_n/n has Var = sigma^2 / n
""")
