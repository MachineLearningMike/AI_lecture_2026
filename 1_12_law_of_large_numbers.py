"""
Law of Large Numbers (LLN) -- the sample average settles on the true mean.

  As n grows,  bar{X}_n = (X_1 + ... + X_n) / n   approaches  E[X]
  (in probability / almost surely, under mild conditions).

CLT (090) is about the *shape* of the sampling distribution of the mean.
LLN is about the mean *going to the right place*.

Sketch:
  1. Formal statement (WLLN / SLLN)
  2. One long stream of random draws
  3. Running average settles while raw draws keep jumping
  4. Many paths -- they all head toward mu
  5. Interactive: drag n, watch the average
  6. Error shrinks like 1/sqrt(n) (typical scale)
  7. LLN vs CLT -- two different promises
  8. Takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Slider
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
import numpy as np

np.random.seed(85)
RNG = np.random.default_rng(85)


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


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_FUN = "#8e44ad"
C_BIN = "#d68910"
C_GRAY = "#85929e"


# Coin / Bernoulli(p) as the workhorse example
P_TRUE = 0.65
MU = P_TRUE  # E[X] for Bernoulli
SIGMA = np.sqrt(P_TRUE * (1 - P_TRUE))


def bernoulli_stream(n, rng=None):
    rng = RNG if rng is None else rng
    return rng.random(n) < P_TRUE


# -------------------------------------------------------
# FIGURE 0 / Step 1: Formal statement
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: Law of Large Numbers -- formal statement",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.25, 11.2, 1.45, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.95,
        r"Setup: i.i.d. $X_1,X_2,\ldots$ with finite mean $\mu = \mathbb{E}[X_i]$"
        + "\n"
        + r"Sample mean:  $\bar X_n = (X_1 + \cdots + X_n)/n$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.35, 5.4, 2.65, C_NEW, "#2874a6", lw=2.0)
    ax.text(
        3.1, 5.55,
        "Weak LLN",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color="#2874a6",
    )
    ax.text(
        3.1, 4.35,
        r"$\bar{X}_n \;\overset{P}{\longrightarrow}\; \mu$"
        + "\n\n"
        "convergence in probability:\n"
        r"for every $\varepsilon>0$,"
        + "\n"
        r"$P(|\bar{X}_n - \mu| > \varepsilon) \to 0$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.2, 3.35, 5.4, 2.65, C_GOAL, C_OK, lw=2.0)
    ax.text(
        8.9, 5.55,
        "Strong LLN",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    ax.text(
        8.9, 4.35,
        r"$\bar{X}_n \;\overset{\mathrm{a.s.}}{\longrightarrow}\; \mu$"
        + "\n\n"
        "almost-sure convergence:\n"
        r"if $\mathbb{E}|X_i| < \infty$, then"
        + "\n"
        r"$P(\bar{X}_n \to \mu) = 1$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.4, 11.2, 2.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.75,
        "In plain language: with enough independent samples,\n"
        r"the average settles on the true mean $\mu$."
        + "\n\n"
        "LLN says where the average goes.  "
        r"(CLT says how the leftover error is shaped.)",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# FIGURE 1: Raw stream of draws
# -------------------------------------------------------
def build_step_2():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 2: Single draws keep jumping -- they do not settle",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    n1 = 120
    x1 = bernoulli_stream(n1).astype(float)
    ax1.scatter(np.arange(1, n1 + 1), x1, s=28, c=C_DARK, alpha=0.75, label="draws X_i")
    ax1.axhline(MU, color=C_ACCENT, ls="--", lw=2.2, label=fr"true mean $\mu={MU:g}$")
    ax1.set_xlabel("draw index i", fontsize=FS_AXIS)
    ax1.set_ylabel(r"$X_i$ (Bernoulli)", fontsize=FS_AXIS)
    ax1.set_ylim(-0.15, 1.25)
    ax1.legend(fontsize=FS_LEGEND, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(fr"Parent: Bernoulli($p={P_TRUE:g}$)", fontsize=FS_AX_TITLE - 2)
    fig1.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig1


# -------------------------------------------------------
# FIGURE 2: Running average settles
# -------------------------------------------------------
def build_step_3():
    fig2, axes2 = new_figure(1, 2)
    fig2.suptitle(
        r"Step 3: The running average $\bar X_n$ settles toward $\mu$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    n2 = 800
    x2 = bernoulli_stream(n2).astype(float)
    idx = np.arange(1, n2 + 1)
    run_mean = np.cumsum(x2) / idx

    axes2[0].scatter(idx[:200], x2[:200], s=16, c=C_GRAY, alpha=0.55, label="draws")
    axes2[0].plot(idx[:200], run_mean[:200], color=C_OK, lw=2.4, label=r"$\bar X_n$")
    axes2[0].axhline(MU, color=C_ACCENT, ls="--", lw=2.0)
    axes2[0].set_title("early: average still wobbles", fontsize=FS_AX_TITLE - 2)
    axes2[0].set_xlabel("n", fontsize=FS_AXIS)
    axes2[0].set_ylabel("value", fontsize=FS_AXIS)
    axes2[0].legend(fontsize=FS_LEGEND - 1, loc="upper right")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].set_ylim(-0.05, 1.15)

    axes2[1].plot(idx, run_mean, color=C_OK, lw=2.0, label=r"$\bar X_n$")
    axes2[1].axhline(MU, color=C_ACCENT, ls="--", lw=2.0, label=fr"$\mu={MU:g}$")
    axes2[1].set_title("later: closer and closer to the mean", fontsize=FS_AX_TITLE - 2)
    axes2[1].set_xlabel("n", fontsize=FS_AXIS)
    axes2[1].legend(fontsize=FS_LEGEND - 1, loc="upper right")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].set_ylim(0.35, 0.95)
    fig2.tight_layout(rect=[0, 0.12, 1, 0.92])
    fig2.text(
        0.5, 0.045,
        "It's probability convergence, not of sequence.",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold",
        color=C_ACCENT,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fadbd8",
                  edgecolor=C_ACCENT, linewidth=2.0),
    )
    return fig2


# -------------------------------------------------------
# FIGURE 3: Many independent paths
# -------------------------------------------------------
def build_step_4():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        "Step 4: Many experiments -- almost every path heads toward the same $\\mu$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    n3 = 400
    n_paths = 40
    for k in range(n_paths):
        path = bernoulli_stream(n3, rng=np.random.default_rng(85 + k)).astype(float)
        rm = np.cumsum(path) / np.arange(1, n3 + 1)
        ax3.plot(rm, color=C_DARK, alpha=0.22, lw=1.2)
    # highlight one path
    path0 = bernoulli_stream(n3, rng=np.random.default_rng(0)).astype(float)
    rm0 = np.cumsum(path0) / np.arange(1, n3 + 1)
    ax3.plot(rm0, color=C_OK, lw=2.6, label="one sample path", zorder=3)
    ax3.axhline(MU, color=C_ACCENT, ls="--", lw=2.4, label=fr"$\mu={MU:g}$", zorder=4)
    ax3.set_xlabel("n", fontsize=FS_AXIS)
    ax3.set_ylabel(r"$\bar X_n$", fontsize=FS_AXIS)
    ax3.set_ylim(0.25, 1.0)
    ax3.legend(fontsize=FS_LEGEND, loc="upper right")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig3


# -------------------------------------------------------
# FIGURE 4: Interactive n
# -------------------------------------------------------
def build_step_5():
    fig4, (ax4a, ax4b) = new_figure(1, 2)
    fig4.suptitle(
        "Step 5: Drag n -- the sample average approaches the true mean",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig4.subplots_adjust(bottom=0.18, wspace=0.28)

    # Fixed long stream for reproducible slider
    STREAM = bernoulli_stream(2000).astype(float)
    CUMMEAN = np.cumsum(STREAM) / np.arange(1, len(STREAM) + 1)


    def draw_n(n):
        ax4a.clear(); ax4b.clear()
        n = int(n)
        ax4a.plot(np.arange(1, n + 1), CUMMEAN[:n], color=C_OK, lw=2.2)
        ax4a.axhline(MU, color=C_ACCENT, ls="--", lw=2.0)
        ax4a.scatter([n], [CUMMEAN[n - 1]], s=70, c=C_FUN, zorder=3)
        ax4a.set_xlim(1, len(STREAM))
        ax4a.set_ylim(0.3, 1.0)
        ax4a.set_xlabel("n", fontsize=FS_AXIS)
        ax4a.set_ylabel(r"$\bar X_n$", fontsize=FS_AXIS)
        ax4a.set_title(fr"$\bar X_{{{n}}} = {CUMMEAN[n - 1]:.3f}$   (true $\mu={MU:g}$)",
                       fontsize=FS_AX_TITLE - 2)
        ax4a.grid(True, alpha=0.3)

        # histogram of many length-n averages
        means = []
        rng = np.random.default_rng(7)
        for _ in range(600):
            means.append(bernoulli_stream(n, rng=rng).mean())
        means = np.asarray(means)
        ax4b.hist(means, bins=28, density=True, color=C_NEW, edgecolor=C_DARK, alpha=0.9)
        ax4b.axvline(MU, color=C_ACCENT, ls="--", lw=2.2, label=r"$\mu$")
        ax4b.axvline(means.mean(), color=C_OK, lw=2.0, label="mean of means")
        ax4b.set_title(f"distribution of $\\bar X_n$ across repeats (n={n})",
                       fontsize=FS_AX_TITLE - 3)
        ax4b.set_xlabel(r"$\bar X_n$", fontsize=FS_AXIS)
        ax4b.set_ylabel("density", fontsize=FS_AXIS)
        ax4b.legend(fontsize=FS_LEGEND - 1)
        ax4b.grid(True, alpha=0.3)


    ax_n = fig4.add_axes([0.25, 0.06, 0.50, 0.04])
    sl_n = Slider(ax_n, "n", 5, 500, valinit=30, valstep=5, color=C_FUN)


    def _u4(_=None):
        draw_n(sl_n.val)
        fig4.canvas.draw_idle()


    sl_n.on_changed(_u4)
    _u4()
    print("Step 5: drag n.")
    return fig4


# -------------------------------------------------------
# FIGURE 5: Typical error scale 1/sqrt(n)
# -------------------------------------------------------
def build_step_6():
    fig5, (ax5a, ax5b) = new_figure(1, 2)
    fig5.suptitle(
        r"Step 6: Typical fluctuation of $\bar X_n$ shrinks like $1/\sqrt{n}$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ns = np.array([5, 10, 20, 40, 80, 160, 320])
    abs_err = []
    rng = np.random.default_rng(11)
    for n in ns:
        errs = [abs(bernoulli_stream(int(n), rng=rng).mean() - MU) for _ in range(400)]
        abs_err.append(np.mean(errs))
    abs_err = np.asarray(abs_err)
    theory = SIGMA * np.sqrt(2 / (np.pi * ns))  # E|Z| * sigma/sqrt(n) for Gaussian approx

    ax5a.plot(ns, abs_err, "o-", color=C_DARK, lw=2.2, label=r"mean $| \bar X_n-\mu |$")
    ax5a.plot(ns, theory, "--", color=C_ACCENT, lw=2.0, label=r"~$\sigma\sqrt{2/(\pi n)}$")
    ax5a.set_xlabel("n", fontsize=FS_AXIS)
    ax5a.set_ylabel("typical absolute error", fontsize=FS_AXIS)
    ax5a.set_title("errors fall as n grows", fontsize=FS_AX_TITLE - 2)
    ax5a.legend(fontsize=FS_LEGEND - 1)
    ax5a.grid(True, alpha=0.3)

    ax5b.loglog(ns, abs_err, "o-", color=C_DARK, lw=2.2, label="simulation")
    ax5b.loglog(ns, theory, "--", color=C_ACCENT, lw=2.0, label=r"$1/\sqrt{n}$ scale")
    ax5b.set_xlabel("n (log)", fontsize=FS_AXIS)
    ax5b.set_ylabel("error (log)", fontsize=FS_AXIS)
    ax5b.set_title("same story on log-log axes", fontsize=FS_AX_TITLE - 2)
    ax5b.legend(fontsize=FS_LEGEND - 1)
    ax5b.grid(True, alpha=0.3, which="both")
    fig5.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig5


# -------------------------------------------------------
# FIGURE 6: LLN vs CLT
# -------------------------------------------------------
def build_step_7():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        "Step 7: LLN vs CLT -- two different promises about the average",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax6)

    round_box(ax6, 0.5, 2.2, 5.3, 5.0, C_NEW, "#2874a6", lw=2.2)
    ax6.text(3.15, 6.5, "Law of Large Numbers", ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color="#2874a6")
    ax6.text(
        3.15, 4.3,
        r"$\bar X_n \to \mu$" + "\n\n"
        "The average goes to\n"
        "the right place.\n\n"
        "Convergence of a number.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )

    round_box(ax6, 6.2, 2.2, 5.3, 5.0, C_GOAL, C_OK, lw=2.2)
    ax6.text(8.85, 6.5, "Central Limit Theorem", ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    ax6.text(
        8.85, 4.3,
        r"$\sqrt{n}(\bar X_n-\mu)$" + "\nbecomes ~ Gaussian\n\n"
        "The average's fluctuation\nhas a predictable shape.\n\n"
        "Convergence of a distribution.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )

    round_box(ax6, 0.5, 0.35, 11.0, 1.55, C_MUTE, "0.45", lw=1.6)
    ax6.text(
        6.0, 1.1,
        "LLN: where the average goes.   CLT: how the remaining error is distributed.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )
    return fig6


# -------------------------------------------------------
# FIGURE 7: Takeaway
# -------------------------------------------------------
def build_step_8():
    fig7, axes7 = new_figure(1, 3)
    fig7.suptitle(
        "Step 8: More independent samples -- a stabler average",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    panels = [
        (axes7[0], "Noise does not vanish", "Each new draw\nstill jumps.", "#fadbd8", C_ACCENT),
        (axes7[1], "The average settles", r"$\bar X_n$ hugs $\mu$" + "\nas n grows.", C_NEW, "#2874a6"),
        (axes7[2], "Rate is slowish", r"Error ~ $1/\sqrt{n}$" + "\n(typical scale).", C_GOAL, C_OK),
    ]
    for ax, title, body, face, edge in panels:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        round_box(ax, 0.5, 1.5, 9.0, 7.0, face, edge, lw=2.2)
        ax.text(5.0, 6.8, title, ha="center", fontsize=FS_AX_TITLE - 3,
                fontweight="bold", color=edge)
        ax.text(5.0, 4.2, body, ha="center", va="center", fontsize=FS_ANNOT + 1)
    return fig7


run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
    build_step_6,
    build_step_7,
    build_step_8,
])

print("\n" + "=" * 60)
print("LAW OF LARGE NUMBERS")
print("=" * 60)
print(rf"""
LLN:  as n grows, sample average -> true mean mu
Demo: Bernoulli(p={P_TRUE:g}), mu={MU:g}

CLT (separate lecture): fluctuations of the average become Gaussian.
LLN: the average converges to the right value.
""")
