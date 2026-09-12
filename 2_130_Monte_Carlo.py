"""
2_130_Monte_Carlo.py

Monte Carlo methods -- estimate expectations (and integrals) by sampling.

Bayesian learning, evaluation, and (later) RL returns.

Sketch:
  1. The idea -- replace hard averages by sample averages
  2. Formal MC estimator
  3. Why it works -- LLN (pointer to 1_030)
  4. Demo: estimate an integral / area
  5. Error shrinks like 1/sqrt(N)
  6. Slider: watch the estimate concentrate
  7. Hit-or-miss picture (estimate pi)
  8. Importance sampling (when plain sampling is wasteful)
  9. Where MC shows up in ML
  10. Takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
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

np.random.seed(120)
RNG = np.random.default_rng(120)


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


# Shared demo integrand: E[g(X)] for X ~ Uniform(0,1), g(x)=4*sqrt(1-x^2)
# (quarter-circle area => pi)
def g_quarter(x):
    return 4.0 * np.sqrt(np.maximum(0.0, 1.0 - x ** 2))


TRUE_PI = np.pi


# -------------------------------------------------------
# Step 1
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: Monte Carlo -- hard averages by easy samples",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 5.7, 11.2, 1.9, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.65,
        "When an expectation or integral is hard to compute exactly,\n"
        "draw random samples and average a simple function of them.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.4, 2.7, "Want", r"$I=\mathbb{E}[g(X)]$"),
        (4.2, 2.7, "Do", r"Draw $X_1,\ldots,X_N$"
         "\n"
         r"average $g(X_i)$"),
        (8.0, 2.7, "Get", r"$\hat I_N=\frac{1}{N}\sum_i g(X_i)$"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.6, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 1.9, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 1.0, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.35, 11.2, 2.0, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "Named after the casino -- randomness as a computational engine.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: Formal estimator
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: The Monte Carlo estimator",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 5.5, 11.2, 2.15, C_GOAL, C_OK, lw=2.2)
    ax.text(
        6.0, 6.55,
        r"If $X_i$ are i.i.d. $\sim p$  and  "
        r"$\hat I_N=\frac{1}{N}\sum_{i=1}^N g(X_i)$,"
        "\n"
        r"then $\mathbb{E}[\hat I_N]=\mathbb{E}[g(X)]$"
        "  (unbiased, when the expectation exists).",
        ha="center", va="center", fontsize=FS_ANNOT, color="#145a32",
    )

    round_box(ax, 0.4, 2.7, 5.5, 2.5, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 4.75, "Integral view", ha="center",
            fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 3.6,
        r"$I=\int g(x)\,p(x)\,dx$"
        "\n\n"
        "Sampling from $p$ turns the\n"
        "integral into an average.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 2.7, 5.5, 2.5, C_WARN, "#d68910", lw=2.0)
    ax.text(8.85, 4.75, "Not magic", ha="center",
            fontsize=FS_ANNOT - 1, fontweight="bold", color="#9a7d0a")
    ax.text(
        8.85, 3.6,
        "You still need samples from $p$\n"
        "(or a proposal -- Step 8).\n\n"
        "Bad $g$ or bad $p$ => huge variance.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#6e2c00",
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.05, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "Same pattern as a sample mean -- because it is a sample mean of $Y_i=g(X_i)$.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 3: LLN link
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: Why it works -- law of large numbers",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 5.6, 11.2, 2.05, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.6,
        r"If $\mathbb{E}|g(X)|<\infty$, then"
        "\n"
        r"$\hat I_N \rightarrow \mathbb{E}[g(X)]$  as $N\to\infty$  (LLN).",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.4, 2.55, "Consistency", "More samples => estimate\napproaches the truth."),
        (4.2, 2.55, "Unbiasedness", r"For every finite $N$,"
         "\n"
         r"$\mathbb{E}[\hat I_N]=I$."),
        (8.0, 2.55, "", "1_030 Law of Large Numbers\n"
         "1_040 Central Limit Theorem"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.7, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 1.95, title, ha="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 1.0, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.35, 11.2, 1.9, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.3,
        "Monte Carlo is LLN turned into an algorithm.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 4: Integral demo
# -------------------------------------------------------
def build_step_4():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 4: Demo -- estimate $\pi$ by averaging $g(U)$, $U\sim\mathrm{Unif}(0,1)$",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    xs = np.linspace(0, 1, 400)
    ax = axes[0]
    ax.fill_between(xs, g_quarter(xs), color="#d6eaf8", alpha=0.8)
    ax.plot(xs, g_quarter(xs), color=C_LOOP, lw=2.5, label=r"$g(x)=4\sqrt{1-x^2}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$g(x)$", fontsize=FS_AXIS)
    ax.set_title(r"Integrand -- area $=\pi$", fontsize=FS_AX_TITLE - 1)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98, 0.05,
        r"$\int_0^1 g(x)\,dx=\pi$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )

    N = 2000
    u = RNG.random(N)
    vals = g_quarter(u)
    running = np.cumsum(vals) / np.arange(1, N + 1)
    ax = axes[1]
    ax.plot(running, color=C_OK, lw=2.0, label=r"running $\hat\pi_N$")
    ax.axhline(TRUE_PI, color=C_ACCENT, ls="--", lw=2.0, label=r"true $\pi$")
    ax.set_xlabel(r"$N$", fontsize=FS_AXIS)
    ax.set_ylabel("estimate", fontsize=FS_AXIS)
    ax.set_title("Monte Carlo running average", fontsize=FS_AX_TITLE - 1)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2.5, 3.8)
    ax.text(
        0.98, 0.05,
        f"final N={N}: {running[-1]:.4f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 5: 1/sqrt(N) rate
# -------------------------------------------------------
def build_step_5():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 5: Typical error shrinks like $1/\sqrt{N}$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # Empirical RMSE vs N from many replications
    Ns = np.array([10, 20, 50, 100, 200, 500, 1000, 2000])
    n_rep = 400
    rmse = []
    for n in Ns:
        errs = []
        for _ in range(n_rep):
            u = RNG.random(n)
            errs.append(np.mean(g_quarter(u)) - TRUE_PI)
        rmse.append(np.sqrt(np.mean(np.square(errs))))
    rmse = np.asarray(rmse)

    # Theoretical asymptotic: Var(g(U))/N
    # Estimate Var(g(U)) once
    u_big = RNG.random(50_000)
    var_g = float(np.var(g_quarter(u_big), ddof=1))
    theory = np.sqrt(var_g / Ns)

    ax = axes[0]
    ax.loglog(Ns, rmse, "o-", color=C_LOOP, lw=2.2, label="empirical RMSE")
    ax.loglog(Ns, theory, "--", color=C_ACCENT, lw=2.2, label=r"$\sqrt{\mathrm{Var}(g)/N}$")
    ax.set_xlabel(r"$N$", fontsize=FS_AXIS)
    ax.set_ylabel("RMSE", fontsize=FS_AXIS)
    ax.set_title("Error vs number of samples", fontsize=FS_AX_TITLE - 1)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1]
    ax.axis("off")
    ax.set_title("Message", fontsize=FS_AX_TITLE - 1)
    ax.text(
        0.5, 0.55,
        r"If $\mathrm{Var}(g(X))=\sigma^2<\infty$,"
        "\n"
        r"CLT says $\sqrt{N}(\hat I_N-I)\to\mathcal{N}(0,\sigma^2)$."
        "\n\n"
        "So standard error ~ "
        r"$\sigma/\sqrt{N}$."
        "\n\n"
        "100× smaller error needs about\n"
        "10,000× more samples -- expensive.\n\n"
        "Variance reduction matters.",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor=C_MUTE, edgecolor="0.5"),
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 6: Slider
# -------------------------------------------------------
def build_step_6():
    fig, (ax, ax_hist) = new_figure(1, 2)
    fig.suptitle(
        "Step 6: Drag N -- watch the Monte Carlo estimate concentrate",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    fig.subplots_adjust(bottom=0.22, left=0.08, right=0.98, wspace=0.28)

    # Precompute many short runs for histogram at each N bookmark
    bookmarks = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000])
    n_rep = 300
    store = {}
    for n in bookmarks:
        est = np.array([np.mean(g_quarter(RNG.random(n))) for _ in range(n_rep)])
        store[int(n)] = est

    n0 = 200
    est0 = store[n0]
    ax.axvline(TRUE_PI, color=C_ACCENT, ls="--", lw=2.2, label=r"true $\pi$")
    _, _, patches_h = ax.hist(
        est0, bins=24, density=True, color=C_LOOP, alpha=0.75,
        edgecolor="white", label="MC estimates",
    )
    ax.set_xlim(2.6, 3.6)
    ax.set_xlabel(r"$\hat\pi_N$", fontsize=FS_AXIS)
    ax.set_ylabel("density", fontsize=FS_AXIS)
    ax.set_title("Sampling distribution of the estimator", fontsize=FS_AX_TITLE - 2)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax_hist.axis("off")
    hud = ax_hist.text(
        0.5, 0.55, "", transform=ax_hist.transAxes,
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor=C_GOAL, edgecolor=C_OK),
    )

    ax_sl = fig.add_axes([0.20, 0.08, 0.60, 0.04])
    ax_sl.set_navigate(False)
    ax_sl.set_zorder(20)
    # slider indexes into bookmarks
    sl = Slider(ax_sl, "N index", 0, len(bookmarks) - 1, valinit=3, valstep=1, color=C_LOOP)

    def _update(_=None):
        k = int(sl.val)
        n = int(bookmarks[k])
        est = store[n]
        ax.cla()
        ax.axvline(TRUE_PI, color=C_ACCENT, ls="--", lw=2.2, label=r"true $\pi$")
        ax.hist(est, bins=24, density=True, color=C_LOOP, alpha=0.75,
                edgecolor="white", label="MC estimates")
        ax.set_xlim(2.6, 3.6)
        ax.set_xlabel(r"$\hat\pi_N$", fontsize=FS_AXIS)
        ax.set_ylabel("density", fontsize=FS_AXIS)
        ax.set_title(f"N = {n}", fontsize=FS_AX_TITLE - 2)
        ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")
        ax.grid(True, alpha=0.3)
        se = float(np.std(est, ddof=1))
        hud.set_text(
            f"N = {n}\n\n"
            f"mean estimate = {est.mean():.4f}\n"
            f"std across runs = {se:.4f}\n\n"
            "Larger N => tighter pile\n"
            "around true pi."
        )
        fig.canvas.draw_idle()

    sl.on_changed(_update)
    fig._sliders = (sl,)
    _update()
    return fig


# -------------------------------------------------------
# Step 7: Hit or miss
# -------------------------------------------------------
def build_step_7():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 7: Hit-or-miss -- throw darts to estimate $\pi$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    N = 1500
    pts = RNG.random((N, 2)) * 2 - 1  # in [-1,1]^2
    hit = (pts[:, 0] ** 2 + pts[:, 1] ** 2) <= 1.0
    pi_hat = 4.0 * hit.mean()

    ax = axes[0]
    ax.add_patch(Rectangle((-1, -1), 2, 2, fill=False, edgecolor="0.3", lw=1.5))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=C_LOOP, lw=2.2))
    ax.scatter(pts[hit, 0], pts[hit, 1], s=8, c=C_OK, alpha=0.55, label="hit")
    ax.scatter(pts[~hit, 0], pts[~hit, 1], s=8, c=C_ACCENT, alpha=0.55, label="miss")
    ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Unit disk inside the square", fontsize=FS_AX_TITLE - 1)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")

    ax = axes[1]
    ax.axis("off")
    ax.set_title("Estimator", fontsize=FS_AX_TITLE - 1)
    ax.text(
        0.5, 0.55,
        "Area(square) = 4.\n"
        "Area(disk) = pi.\n\n"
        "P(hit) = pi / 4.\n\n"
        r"$\hat\pi = 4\times(\mathrm{hits}/N)$"
        "\n\n"
        f"N = {N}\n"
        f"estimate = {pi_hat:.4f}\n"
        f"true pi = {TRUE_PI:.4f}",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor=C_MUTE, edgecolor="0.5"),
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 8: Importance sampling
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: Importance sampling -- sample where it matters",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    round_box(ax, 0.4, 5.7, 11.2, 1.9, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.65,
        r"If sampling from $p$ is hard or wasteful, sample from proposal $q$"
        "\n"
        r"and reweight:  $\mathbb{E}_p[g(X)]=\mathbb{E}_q\!\left[g(X)\frac{p(X)}{q(X)}\right]$.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 2.85, 5.5, 2.55, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 4.95, "Estimator", ha="center",
            fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 3.8,
        r"$\hat I_N=\frac{1}{N}\sum_i g(X_i)\,w_i$"
        "\n"
        r"$w_i=p(X_i)/q(X_i)$"
        "\n\n"
        r"$X_i\sim q$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 2.85, 5.5, 2.55, C_WARN, "#d68910", lw=2.0)
    ax.text(8.85, 4.95, "Design rule", ha="center",
            fontsize=FS_ANNOT - 1, fontweight="bold", color="#9a7d0a")
    ax.text(
        8.85, 3.8,
        "Choose $q$ to put mass\n"
        "where $|g|p$ is large.\n\n"
        "Bad $q$ => huge weights\n"
        "and awful variance.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#6e2c00",
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.2, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.45,
        "Used constantly in rare-event simulation, Bayesian inference,\n"
        "and off-policy RL corrections.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 9: Where MC appears in ML
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Where Monte Carlo shows up in ML",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax)

    cards = [
        (0.4, 5.35, "Expectations", "Risk, calibration, expected\ngradients under noise."),
        (4.2, 5.35, "Bayesian ML", "Posterior expectations\nvia MCMC / variational MC."),
        (8.0, 5.35, "Generative models", "Sample from $p_\\theta(x)$;\n"
         "MC estimates of likelihoods."),
        (0.4, 2.55, "Evaluation", "BLEU/FID-style metrics\noften average over samples."),
        (4.2, 2.55, "RL returns", "Monte Carlo returns $G_t$\n"
         "average complete trajectories."),
        (8.0, 2.55, "Dropout MC", "Stochastic forward passes\nestimate predictive uncertainty."),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.5, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 1.8, y + 1.8, title, ha="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 0.95, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")

    round_box(ax, 0.4, 0.35, 11.2, 1.9, C_GOAL, C_OK, lw=1.8)
    ax.text(
        6.0, 1.3,
        "Whenever the math says \"expectation\" and the integral is ugly -- sample.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#145a32",
    )
    return fig


# -------------------------------------------------------
# Step 10: Takeaway
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: Takeaway",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank(ax)

    rows = [
        (6.2, "Idea", r"Replace $\mathbb{E}[g(X)]$ by a sample average of $g(X_i)$."),
        (4.85, "Guarantee", "LLN => consistency; CLT => $1/\\sqrt{N}$ error rate."),
        (3.5, "Cost", "Halving error needs ~4× samples unless you reduce variance."),
        (2.15, "Upgrade", "Importance sampling / control variates / stratification."),
        (0.8, "In ML", "Bayesian averages, generative sampling, RL returns, uncertainty."),
    ]
    for y, title, body in rows:
        round_box(ax, 0.4, y, 11.2, 1.15, C_NEW, C_LOOP, lw=1.6)
        ax.text(2.0, y + 0.55, title, ha="center", va="center",
                fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK)
        ax.text(7.3, y + 0.55, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")
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
print("MONTE CARLO METHODS")
print("=" * 60)
print("""
Estimator:  I_hat = (1/N) sum g(X_i),   X_i ~ p
Unbiased if E|g(X)| < infinity; consistent by LLN; SE ~ sigma/sqrt(N).
""")
