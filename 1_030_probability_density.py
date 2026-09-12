"""
1_030_probability_density.py

Random variables, probability, CDF, and PDF -- definitions with pictures.

Sketch:
  1. What is a random variable?
  2. Discrete vs continuous outcomes
  3. Formal probability function P(·)
  4. Terminology: CDF and PDF
  5. Formal CDF function F_X
  6. Visualize a CDF
  7. Slider: CDF value vs x
  8. Animation: CDF sweeps with x
  9. Formal PDF / PMF functions
  10. Visualize a PDF (area = probability)
  11. How CDF and PDF connect
  12. Rules a PDF / CDF must obey
  13. Tiny discrete + continuous comparison
  14. Takeaway

Sits after 1_010_distribution.py (why distributions matter) and before
1_040_gaussian.py / 1_050_more_distributions.py (concrete families).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch, Polygon
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
from scipy.stats import norm

np.random.seed(15)


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


# Shared continuous example: N(0,1) for CDF/PDF pictures
XS = np.linspace(-3.5, 3.5, 500)
PDF = norm.pdf(XS)
CDF = norm.cdf(XS)


# -------------------------------------------------------
# Step 1: Random variable
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: What is a random variable?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        "A random variable is a number that comes from a random experiment.\n"
        "It assigns a numerical value to each possible outcome.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.5, 2.5, "Experiment", "toss a die\nmeasure height\nnext word"),
        (4.25, 2.5, "Outcome", "face shows 4\n172 cm\n\"the\""),
        (8.0, 2.5, "Random variable X", "X = 4\nX = 172\nX = token id"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.5, 2.6, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.75, y + 2.05, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.75, y + 0.95, body, ha="center", va="center",
                fontsize=FS_TICK + 1)

    round_box(ax, 0.5, 0.4, 11.0, 1.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.25,
        r"We write capital $X$ for the random variable, lowercase $x$ for a realized value.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: discrete vs continuous
# -------------------------------------------------------
def build_step_2():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 2: Discrete vs continuous random variables",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    ax = axes[0]
    faces = np.arange(1, 7)
    probs = np.ones(6) / 6
    ax.bar(faces, probs, color="#5dade2", edgecolor="0.2", width=0.7)
    ax.set_title("Discrete: die face", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$P(X=x)$", fontsize=FS_AXIS)
    ax.set_xticks(faces)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(0, 0.35)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(
        0.5, 0.76, "Point masses on a countable set",
        transform=ax.transAxes, ha="center", fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )

    ax = axes[1]
    ax.plot(XS, PDF, color="seagreen", lw=2.8)
    ax.fill_between(XS, PDF, color="seagreen", alpha=0.25)
    ax.set_title(r"Continuous: e.g. $X\sim\mathcal{N}(0,1)$", fontsize=FS_AX_TITLE - 2,
                 fontweight="bold")
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"density $f(x)$", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.5, 0.76, "Smooth continuum of values",
        transform=ax.transAxes, ha="center", fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 3: Formal probability function
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 3: Formal probability -- the function $P(\cdot)$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.25, 11.2, 1.45, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.95,
        r"$P$ is a function from events to numbers in $[0,1]$."
        + "\n"
        r"An event $A$ is a set of outcomes;  $P(A)$ is its probability.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.35, 5.4, 2.65, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        3.1, 5.55,
        "Kolmogorov axioms",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        3.1, 4.35,
        r"$P(A)\geq 0$"
        + "\n"
        r"$P(\Omega)=1$"
        + "\n\n"
        r"If $A_i$ pairwise disjoint:"
        + "\n"
        r"$P(\cup_i A_i)=\sum_i P(A_i)$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.2, 3.35, 5.4, 2.65, C_GOAL, C_OK, lw=2.0)
    ax.text(
        8.9, 5.55,
        "Through a random variable",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    ax.text(
        8.9, 4.35,
        r"$P(X\in B)"
        r"=P(\{\omega: X(\omega)\in B\})$"
        + "\n\n"
        r"Examples: $P(X=4)$, $P(a\leq X\leq b)$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.4, 11.2, 2.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.75,
        r"Discrete: probability can sit on points, e.g. $P(X=4)=\frac{1}{6}$."
        + "\n"
        r"Continuous: usually $P(X=x)=0$; mass lives on intervals."
        + "\n\n"
        r"$P$ is the root object -- CDF and PDF are built from it.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 4: Terminology -- CDF and PDF
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: Terminology -- CDF and PDF",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 7.0,
        "Two names you will see constantly. Memorize what the letters stand for.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.25, 5.4, 2.85, C_NEW, C_LOOP, lw=2.2)
    ax.text(
        3.1, 5.7, "CDF",
        ha="center", fontsize=FS_AX_TITLE, fontweight="bold", color=C_DARK,
    )
    ax.text(
        3.1, 5.05,
        "Cumulative Distribution Function",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_LOOP,
    )
    ax.text(
        3.1, 4.05,
        r"$F_X(x)=P(X\leq x)$"
        + "\n\n"
        "Cumulative: piles probability\n"
        "from $-\infty$ up to $x$.\n"
        "Distribution Function: a full\n"
        "map from thresholds to probs.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.2, 3.25, 5.4, 2.85, C_GOAL, C_OK, lw=2.2)
    ax.text(
        8.9, 5.7, "PDF",
        ha="center", fontsize=FS_AX_TITLE, fontweight="bold", color=C_OK,
    )
    ax.text(
        8.9, 5.05,
        "Probability Density Function",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color="#1e8449",
    )
    ax.text(
        8.9, 4.05,
        r"$f_X(x)=F_X'(x)$  (when smooth)"
        + "\n\n"
        "Density: height is not itself\n"
        "a probability -- areas are.\n"
        "Function of $x$ for continuous $X$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.65,
        "Say the full words once, then the acronym is fine.\n"
        r"CDF answers: how much probability lies at most at this $x$?"
        + "\n"
        r"PDF answers: how densely is probability packed near this $x$?",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 5: Formal CDF function
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 5: Formal CDF -- the function $F_X$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.25, 11.2, 1.45, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.95,
        r"$F_X:\mathbb{R}\to[0,1]$"
        + "    with    "
        r"$F_X(x)=P(X\leq x)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.35, 5.4, 2.65, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        3.1, 5.55,
        "What it does",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        3.1, 4.35,
        "At each real number $x$,\n"
        r"$F_X(x)$ returns the probability"
        + "\n"
        "that $X$ lands at most $x$.\n\n"
        r"Accumulates mass from $-\infty$ up to $x$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.2, 3.35, 5.4, 2.65, C_GOAL, C_OK, lw=2.0)
    ax.text(
        8.9, 5.55,
        "Useful identities",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    ax.text(
        8.9, 4.35,
        r"$P(a<X\leq b)=F_X(b)-F_X(a)$"
        + "\n\n"
        r"$\lim_{x\to-\infty}F_X(x)=0$"
        + "\n"
        r"$\lim_{x\to+\infty}F_X(x)=1$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.4, 11.2, 2.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.75,
        r"$F_X$ is nondecreasing and right-continuous."
        + "\n"
        "Defined for every random variable -- discrete or continuous.\n"
        "A PDF need not exist; the CDF always does.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 6: visualize CDF
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 6: Visualize $F(x)=P(X\leq x)$  (standard normal)",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    ax.plot(XS, CDF, color=C_LOOP, lw=3.0, label=r"$F(x)$")
    x0 = 1.0
    f0 = float(norm.cdf(x0))
    ax.axvline(x0, color=C_ACCENT, ls="--", lw=1.8)
    ax.axhline(f0, color=C_ACCENT, ls=":", lw=1.5)
    ax.scatter([x0], [f0], s=80, c=C_ACCENT, zorder=5)
    ax.annotate(
        rf"$F({x0:g})={f0:.2f}$",
        xy=(x0, f0), xytext=(x0 + 0.6, f0 - 0.25),
        fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT,
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.5),
    )
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$F(x)$", fontsize=FS_AXIS)
    ax.set_title("CDF rises from 0 to 1", fontsize=FS_AX_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc="lower right")
    ax.text(
        0.02, 0.98,
        "Read: probability of landing\nto the left of this x.",
        transform=ax.transAxes, ha="left", va="top", fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 7: Slider -- CDF value vs x
# -------------------------------------------------------
def build_step_7():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 7: Drag $x$ -- watch the CDF value $F(x)$ change",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.18, left=0.08, right=0.98, top=0.88, wspace=0.28)

    ax_pdf, ax_cdf = axes
    x0 = 0.5

    ax_pdf.plot(XS, PDF, color="seagreen", lw=2.6)
    fill_pdf = ax_pdf.fill_between(XS[XS <= x0], PDF[XS <= x0],
                                   color="seagreen", alpha=0.35)
    vline_pdf = ax_pdf.axvline(x0, color=C_ACCENT, ls="--", lw=1.8)
    ax_pdf.set_title("PDF: mass to the left of $x$", fontsize=FS_AX_TITLE - 2,
                     fontweight="bold")
    ax_pdf.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax_pdf.set_ylabel(r"$f(x)$", fontsize=FS_AXIS)
    ax_pdf.tick_params(labelsize=FS_TICK)
    ax_pdf.grid(True, alpha=0.3)
    txt_pdf = ax_pdf.text(
        0.02, 0.95, "", transform=ax_pdf.transAxes, ha="left", va="top",
        fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )

    ax_cdf.plot(XS, CDF, color=C_LOOP, lw=2.8, label=r"$F(x)$")
    hline = ax_cdf.axhline(float(norm.cdf(x0)), color=C_ACCENT, ls=":", lw=1.5)
    vline_cdf = ax_cdf.axvline(x0, color=C_ACCENT, ls="--", lw=1.8)
    pt = ax_cdf.scatter([x0], [float(norm.cdf(x0))], s=70, c=C_ACCENT, zorder=5)
    ax_cdf.set_title("CDF readout", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax_cdf.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax_cdf.set_ylabel(r"$F(x)$", fontsize=FS_AXIS)
    ax_cdf.set_ylim(-0.05, 1.05)
    ax_cdf.tick_params(labelsize=FS_TICK)
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.legend(fontsize=FS_LEGEND - 1, loc="center right")
    txt_cdf = ax_cdf.text(
        0.98, 0.05, "", transform=ax_cdf.transAxes, ha="right", va="bottom",
        fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )

    ax_sl = fig.add_axes([0.20, 0.06, 0.60, 0.04])
    sl_x = Slider(ax_sl, r"$x$", -3.0, 3.0, valinit=x0, valstep=0.05, color=C_LOOP)

    def _draw(x):
        nonlocal fill_pdf
        x = float(x)
        fx = float(norm.cdf(x))
        fill_pdf.remove()
        fill_pdf = ax_pdf.fill_between(XS[XS <= x], PDF[XS <= x],
                                       color="seagreen", alpha=0.35)
        vline_pdf.set_xdata([x, x])
        vline_cdf.set_xdata([x, x])
        hline.set_ydata([fx, fx])
        pt.set_offsets([[x, fx]])
        txt_pdf.set_text(rf"shaded area $= F({x:.2f}) = {fx:.3f}$")
        txt_cdf.set_text(rf"$F({x:.2f})={fx:.3f}$")
        fig.canvas.draw_idle()

    def _on_slide(_=None):
        _draw(sl_x.val)

    sl_x.on_changed(_on_slide)
    fig._sl_x = sl_x  # keep slider alive
    _draw(x0)
    return fig


# -------------------------------------------------------
# Step 8: Animation -- CDF sweeps with x
# -------------------------------------------------------
def build_step_8():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 8: Animation -- $x$ sweeps, $F(x)$ updates",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.14, left=0.08, right=0.98, top=0.88, wspace=0.28)

    ax_pdf, ax_cdf = axes
    xs_anim = np.linspace(-3.0, 3.0, 81)

    ax_pdf.plot(XS, PDF, color="seagreen", lw=2.6)
    fill_pdf = ax_pdf.fill_between(XS[XS <= xs_anim[0]], PDF[XS <= xs_anim[0]],
                                   color="seagreen", alpha=0.35)
    vline_pdf = ax_pdf.axvline(xs_anim[0], color=C_ACCENT, ls="--", lw=1.8)
    ax_pdf.set_title("PDF area up to $x$", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax_pdf.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax_pdf.set_ylabel(r"$f(x)$", fontsize=FS_AXIS)
    ax_pdf.tick_params(labelsize=FS_TICK)
    ax_pdf.grid(True, alpha=0.3)
    txt_pdf = ax_pdf.text(
        0.02, 0.95, "", transform=ax_pdf.transAxes, ha="left", va="top",
        fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )

    ax_cdf.plot(XS, CDF, color=C_LOOP, lw=2.8)
    trail_x, trail_y = [], []
    trail_line, = ax_cdf.plot([], [], color=C_ACCENT, lw=2.0, alpha=0.85,
                              label="path of $(x,F(x))$")
    hline = ax_cdf.axhline(float(norm.cdf(xs_anim[0])), color=C_ACCENT, ls=":", lw=1.4)
    vline_cdf = ax_cdf.axvline(xs_anim[0], color=C_ACCENT, ls="--", lw=1.8)
    pt = ax_cdf.scatter([xs_anim[0]], [float(norm.cdf(xs_anim[0]))],
                        s=70, c=C_ACCENT, zorder=5)
    ax_cdf.set_title("Moving point on the CDF", fontsize=FS_AX_TITLE - 2,
                     fontweight="bold")
    ax_cdf.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax_cdf.set_ylabel(r"$F(x)$", fontsize=FS_AXIS)
    ax_cdf.set_ylim(-0.05, 1.05)
    ax_cdf.tick_params(labelsize=FS_TICK)
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.legend(fontsize=FS_LEGEND - 1, loc="center right")
    txt_cdf = ax_cdf.text(
        0.98, 0.05, "", transform=ax_cdf.transAxes, ha="right", va="bottom",
        fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )

    state = {"fill": fill_pdf, "i": 0}

    def _frame(i):
        x = float(xs_anim[i])
        fx = float(norm.cdf(x))
        state["fill"].remove()
        state["fill"] = ax_pdf.fill_between(XS[XS <= x], PDF[XS <= x],
                                            color="seagreen", alpha=0.35)
        vline_pdf.set_xdata([x, x])
        vline_cdf.set_xdata([x, x])
        hline.set_ydata([fx, fx])
        pt.set_offsets([[x, fx]])
        if i == 0:
            trail_x.clear()
            trail_y.clear()
        trail_x.append(x)
        trail_y.append(fx)
        trail_line.set_data(trail_x, trail_y)
        txt_pdf.set_text(rf"area $= F({x:.2f}) = {fx:.3f}$")
        txt_cdf.set_text(rf"$x={x:.2f}$   $F(x)={fx:.3f}$")
        return (vline_pdf, vline_cdf, hline, pt, trail_line, txt_pdf, txt_cdf)

    anim = animation.FuncAnimation(
        fig, _frame, frames=len(xs_anim), interval=70, blit=False, repeat=True,
    )
    fig._cdf_anim = anim  # prevent garbage-collection of the animation
    fig.text(
        0.5, 0.02,
        r"Auto-play: $x$ walks left to right; shaded PDF area equals the CDF height.",
        ha="center", va="bottom", fontsize=FS_TICK, color="0.35",
    )
    return fig


# -------------------------------------------------------
# Step 9: Formal PDF / PMF functions
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 9: Formal PDF / PMF -- the functions $f_X$ and $p_X$",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)
    # Prior down-shift (~5 cm), then top+middle back up ~3 cm.
    dy_bottom = -3.3
    dy_up = 2.86
    dy_top = dy_bottom + dy_up
    # Middle taller again; ~0.5 cm extra between body lines (linespacing≈2.2).
    mid_h = 2.85 + 1.91 + 0.95
    ax.set_ylim(-3.5, 8)

    round_box(ax, 0.4, 6.35 + dy_top, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 7.0 + dy_top,
        "Two cousins: a density for continuous $X$, a mass function for discrete $X$.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    # Keep middle top under the banner; extra height grows downward.
    mid_top = 3.25 + 2.85 + dy_top
    mid_y = mid_top - mid_h
    round_box(ax, 0.4, mid_y, 5.4, mid_h, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        3.1, mid_y + mid_h - 0.55,
        r"Continuous PDF  $f_X$",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        3.1, mid_y + mid_h / 2 - 0.25,
        r"$f_X:\mathbb{R}\to[0,\infty)$"
        + "\n"
        r"$F_X(x)=\int_{-\infty}^{x} f_X(t)\,dt$"
        + "\n"
        r"$f_X(x)=F_X'(x)$  (when smooth)"
        + "\n"
        r"$P(a\leq X\leq b)=\int_a^b f_X(x)\,dx$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
        linespacing=2.2,
    )

    round_box(ax, 6.2, mid_y, 5.4, mid_h, C_OLD, "#d35400", lw=2.0)
    ax.text(
        8.9, mid_y + mid_h - 0.55,
        r"Discrete PMF  $p_X$",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color="#d35400",
    )
    ax.text(
        8.9, mid_y + mid_h / 2 - 0.25,
        r"$p_X(x)=P(X=x)$"
        + "\n"
        r"$F_X(x)=\sum_{u\leq x} p_X(u)$"
        + "\n"
        r"$\sum_x p_X(x)=1$"
        + "\n"
        r"$0\leq p_X(x)\leq 1$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
        linespacing=2.2,
    )

    round_box(ax, 0.4, 0.4 + dy_bottom, 11.2, 2.6, C_WARN, C_ACCENT, lw=2.0)
    ax.text(
        6.0, 1.7 + dy_bottom,
        r"$f_X(x)$ is not a probability -- only areas under $f_X$ are."
        + "\n"
        r"Height $\times$ tiny width $\approx P(X\in[x,x+dx])$."
        + "\n"
        r"By contrast, each stick $p_X(x)$ already is a probability.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#641e16",
    )
    return fig


# -------------------------------------------------------
# Step 10: visualize PDF with area
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 10: Visualize PDF -- shaded area is probability",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    a, b = -0.5, 1.2
    ax.plot(XS, PDF, color="seagreen", lw=3.0, label=r"$f(x)$")
    mask = (XS >= a) & (XS <= b)
    ax.fill_between(XS[mask], PDF[mask], color="seagreen", alpha=0.45,
                     label=rf"$P({a:g}\leq X\leq {b:g})$")
    # polygon outline for clarity
    xs_band = XS[mask]
    ys_band = PDF[mask]
    verts = np.column_stack([
        np.concatenate([[xs_band[0]], xs_band, [xs_band[-1]]]),
        np.concatenate([[0], ys_band, [0]]),
    ])
    ax.add_patch(Polygon(verts, closed=True, facecolor="none",
                         edgecolor=C_OK, lw=2.0))

    p_ab = float(norm.cdf(b) - norm.cdf(a))
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$f(x)$", fontsize=FS_AXIS)
    ax.set_title(r"Standard normal density", fontsize=FS_AX_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc="upper right")
    ax.text(
        0.02, 0.96,
        rf"Area ≈ {p_ab:.2f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=FS_ANNOT + 1, fontweight="bold", color=C_OK,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 11: CDF <-> PDF
# -------------------------------------------------------
def build_step_11():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 11: How CDF and PDF connect",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(XS, CDF, color=C_LOOP, lw=2.8)
    ax.set_title(r"$F(x)=\int_{-\infty}^{x} f(t)\,dt$", fontsize=FS_AX_TITLE - 2,
                 fontweight="bold")
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$F(x)$", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.08, "CDF accumulates the PDF",
            transform=ax.transAxes, ha="center", fontsize=FS_TICK,
            bbox=dict(boxstyle="round", facecolor=C_NEW, alpha=0.95))

    ax = axes[1]
    ax.plot(XS, PDF, color="seagreen", lw=2.8)
    ax.set_title(r"$f(x)=F'(x)$  (where smooth)", fontsize=FS_AX_TITLE - 2,
                 fontweight="bold")
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$f(x)$", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.08, "PDF is the slope of the CDF",
            transform=ax.transAxes, ha="center", fontsize=FS_TICK,
            bbox=dict(boxstyle="round", facecolor=C_GOAL, alpha=0.95))

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 12: rules
# -------------------------------------------------------
def build_step_12():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 12: Rules a PDF / CDF must obey",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    # Top panels moved down ~3 cm (~2.0 data units on this axes).
    dy_top = -2.0

    left = [
        (0.4, 5.3 + dy_top, "PDF rules",
         r"$f(x)\geq 0$"
         "\n\n"
         r"$\int_{-\infty}^{\infty} f(x)\,dx = 1$"
         "\n\n"
         r"$P(a\leq X\leq b)=\int_a^b f$"),
        (0.4, 0.5, "Not a probability",
         "Height $f(x)$ can be $>1$.\n"
         "Only areas are probabilities."),
    ]
    right = [
        (6.2, 5.3 + dy_top, "CDF rules",
         r"$0\leq F(x)\leq 1$"
         "\n\n"
         r"$F$ nondecreasing"
         "\n\n"
         r"$F(-\infty)=0,\ F(+\infty)=1$"),
        (6.2, 0.5, "Interval from CDF",
         r"$P(a<X\leq b)=F(b)-F(a)$"),
    ]
    for x, y, title, body in left + right:
        h = 4.4 if y > 2 else 2.0
        face = C_NEW if y > 2 else C_MUTE
        edge = C_LOOP if y > 2 else "0.45"
        round_box(ax, x, y, 5.4, h, face, edge, lw=2.0)
        ax.text(x + 2.7, y + h - 0.55, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold",
                color=C_DARK if y > 2 else "#333333")
        ax.text(x + 2.7, y + h / 2 - 0.2, body, ha="center", va="center",
                fontsize=FS_TICK + 1)
    return fig


# -------------------------------------------------------
# Step 13: discrete PMF vs continuous PDF
# -------------------------------------------------------
def build_step_13():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 13: Discrete PMF vs continuous PDF",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    ax.vlines([1, 2, 3, 4], 0, [0.1, 0.4, 0.3, 0.2], colors=C_LOOP, lw=4)
    ax.scatter([1, 2, 3, 4], [0.1, 0.4, 0.3, 0.2], s=70, c=C_LOOP, zorder=3)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(0, 0.55)
    ax.set_title(r"PMF: $p(x)=P(X=x)$", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel("probability", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, axis="y", alpha=0.3)
    ax.text(0.5, 0.92, "Sticks sum to 1", transform=ax.transAxes,
            ha="center", fontsize=FS_TICK,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95))

    ax = axes[1]
    ax.plot(XS, PDF, color="seagreen", lw=2.8)
    ax.fill_between(XS, PDF, color="seagreen", alpha=0.25)
    ax.set_title(r"PDF: $f(x)$ density", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel("density", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.text(0.5, 0.92, "Area under curve = 1", transform=ax.transAxes,
            ha="center", fontsize=FS_TICK,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95))

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 14: takeaway
# -------------------------------------------------------
def build_step_14():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 14: Takeaway",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    points = [
        (0.5, 5.9, "Probability", r"$P(\cdot)$ maps events to $[0,1]$ (axioms)."),
        (0.5, 4.5, "CDF", r"Cumulative Distribution Function: $F_X(x)=P(X\leq x)$."),
        (0.5, 3.1, "PDF", r"Probability Density Function: $f_X=F_X'$; areas are probs."),
        (0.5, 1.7, "PMF", r"$p_X(x)=P(X=x)$; discrete sticks sum to 1."),
    ]
    for x, y, title, body in points:
        round_box(ax, x, y, 11.0, 1.2, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 2.2, y + 0.6, title, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 7.2, y + 0.6, body, ha="center", va="center",
                fontsize=FS_ANNOT, color="#333333")

    round_box(ax, 0.5, 0.3, 11.0, 1.1, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 0.85,
        r"$F' = f$   ·   area under $f$ = probability   ·   next: Gaussians",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
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
])


print("\n" + "=" * 60)
print("RANDOM VARIABLES, PROBABILITY, CDF, PDF")
print("=" * 60)
print("""
Random variable X: numerical outcome of a random experiment.

Probability:  P maps events to [0,1]  (nonnegative, P(Omega)=1, additive)

Terminology:
  CDF = Cumulative Distribution Function
  PDF = Probability Density Function

CDF:  F_X : R -> [0,1],   F_X(x) = P(X <= x)
      P(a < X <= b) = F_X(b) - F_X(a)

PDF:  f_X : R -> [0, inf),  F_X(x) = integral_{-inf}^{x} f_X
      f_X = F_X' when smooth
      P(a<=X<=b) = integral_a^b f_X

PMF:  p_X(x) = P(X=x),   F_X(x) = sum_{u<=x} p_X(u)

Interactive: drag x (Step 7) or watch the sweep (Step 8) to see F(x) move.

Rules: f >= 0 and integrates to 1; F rises from 0 to 1.
""")
