"""
0620_probability_vs_statistics.py

Probability vs statistics -- two directions of the same relationship.

  Probability:  known world  -->  what data look like
  Statistics:   observed data -->  what the world might be

Bayes is the bridge: update beliefs about the world given data.
MLE is a statistical workhorse: pick the world that makes the data most likely.

Sketch:
  1. Same ingredients, opposite arrows
  2. The difference -- what each field asks
  3. The relation -- shared likelihood object P(data|model)
  4. Forward demo: fix a coin, simulate flips
  5. Inverse demo: see flips, infer the coin
  6. Interactive: probability mode (set p, watch samples)
  7. Interactive: statistics mode (see data, scan likelihood / estimate)
  8. Bayes as the bridge
  9. Where this shows up in modeling (generative vs discriminative; MLE)
  10. Takeaway: difference + relation

Companion to 0600_law_of_large_numbers.py, 0610_central_limit_theorem.py,
0630_Bayesian_theorem.py, 0470_maximum_likelihood.py,
0740_discriminative_generative.py.
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
)
apply_rcparams(plt)
import numpy as np

np.random.seed(115)
RNG = np.random.default_rng(115)


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
C_WARN = "#f8d7da"
C_FUN = "#8e44ad"
C_BIN = "#d68910"
C_GRAY = "#85929e"


def bernoulli_lik(p, k, n):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (p ** k) * ((1 - p) ** (n - k))


# -------------------------------------------------------
# FIGURE 1: Opposite arrows
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    "Step 1: Probability theory and statistics -- difference and relation",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax1)

round_box(ax1, 0.6, 5.0, 4.4, 2.4, C_NEW, "#2874a6", lw=2.2)
ax1.text(2.8, 6.55, "WORLD / MODEL", ha="center",
         fontsize=FS_ANNOT + 1, fontweight="bold", color="#2874a6")
ax1.text(2.8, 5.7, r"parameter $p$, distribution $P$",
         ha="center", fontsize=FS_ANNOT)

round_box(ax1, 7.0, 5.0, 4.4, 2.4, C_GOAL, C_OK, lw=2.2)
ax1.text(9.2, 6.55, "DATA", ha="center",
         fontsize=FS_ANNOT + 1, fontweight="bold", color=C_OK)
ax1.text(9.2, 5.7, r"samples $x_1,\ldots,x_n$",
         ha="center", fontsize=FS_ANNOT)

ax1.annotate("", xy=(7.0, 6.2), xytext=(5.1, 6.2),
             arrowprops=dict(arrowstyle="->", color="#2874a6", lw=3.0, mutation_scale=18))
ax1.text(6.05, 6.7, "Probability", ha="center", fontsize=FS_ANNOT,
         fontweight="bold", color="#2874a6")

ax1.annotate("", xy=(5.1, 5.4), xytext=(7.0, 5.4),
             arrowprops=dict(arrowstyle="->", color=C_OK, lw=3.0, mutation_scale=18))
ax1.text(6.05, 4.85, "Statistics", ha="center", fontsize=FS_ANNOT,
         fontweight="bold", color=C_OK)

round_box(ax1, 0.6, 0.6, 5.1, 3.6, C_WARN, C_ACCENT, lw=2.0)
ax1.text(3.15, 3.5, "DIFFERENCE", ha="center",
         fontsize=FS_ANNOT + 1, fontweight="bold", color=C_ACCENT)
ax1.text(
    3.15, 2.1,
    "Opposite given / unknown.\n"
    "Opposite direction of travel.",
    ha="center", va="center", fontsize=FS_ANNOT,
)

round_box(ax1, 6.3, 0.6, 5.1, 3.6, C_GOAL, C_OK, lw=2.0)
ax1.text(8.85, 3.5, "RELATION", ha="center",
         fontsize=FS_ANNOT + 1, fontweight="bold", color=C_OK)
ax1.text(
    8.85, 2.1,
    "Same two objects:\n"
    "model and data.\n"
    "Linked by likelihood.",
    ha="center", va="center", fontsize=FS_ANNOT,
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: The difference
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    "Step 2: The difference -- which side is given?",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax2)

round_box(ax2, 0.4, 1.2, 5.4, 6.0, C_NEW, "#2874a6", lw=2.2)
ax2.text(3.1, 6.6, "Probability theory", ha="center",
         fontsize=FS_AX_TITLE - 2, fontweight="bold", color="#2874a6")
ax2.text(
    3.1, 4.0,
    "GIVEN: a model / world\n"
    r"(e.g. $p=0.7$, or $N(\mu,\sigma^2)$)" + "\n\n"
    "ASK: what will data look like?\n"
    "P(60 heads in 100 flips)?\n\n"
    "Forward: model → data.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)

round_box(ax2, 6.2, 1.2, 5.4, 6.0, C_GOAL, C_OK, lw=2.2)
ax2.text(8.9, 6.6, "Statistics", ha="center",
         fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
ax2.text(
    8.9, 4.0,
    "GIVEN: observed data\n"
    "(e.g. 60 heads in 100 flips)\n\n"
    "ASK: what is the world?\n"
    "Estimate p? Uncertainty?\n\n"
    "Inverse: data → model.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 2b: The relation
# -------------------------------------------------------
fig2b, ax2b = new_figure()
fig2b.suptitle(
    r"Step 3: The relation -- both use $P(\mathrm{data}\mid\mathrm{model})$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax2b)

round_box(ax2b, 3.0, 5.2, 6.0, 2.2, "#f5eef8", C_FUN, lw=2.4)
ax2b.text(
    6.0, 6.3,
    r"Likelihood / sampling model" + "\n"
    r"$L = P(\mathrm{data}\mid\mathrm{model})$",
    ha="center", va="center", fontsize=FS_ANNOT + 2,
    fontweight="bold", color=C_FUN,
)

round_box(ax2b, 0.5, 1.0, 5.2, 3.6, C_NEW, "#2874a6", lw=2.0)
ax2b.text(3.1, 3.9, "In probability", ha="center",
          fontsize=FS_ANNOT + 1, fontweight="bold", color="#2874a6")
ax2b.text(
    3.1, 2.4,
    "Model is known.\n"
    "Evaluate / simulate\n"
    "this probability.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)

round_box(ax2b, 6.3, 1.0, 5.2, 3.6, C_GOAL, C_OK, lw=2.0)
ax2b.text(8.9, 3.9, "In statistics", ha="center",
          fontsize=FS_ANNOT + 1, fontweight="bold", color=C_OK)
ax2b.text(
    8.9, 2.4,
    "Data are known.\n"
    "Vary the model to\n"
    "maximize / update L.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)

ax2b.annotate("", xy=(6.0, 5.2), xytext=(3.1, 4.7),
              arrowprops=dict(arrowstyle="->", color="#2874a6", lw=2.2))
ax2b.annotate("", xy=(6.0, 5.2), xytext=(8.9, 4.7),
              arrowprops=dict(arrowstyle="->", color=C_OK, lw=2.2))
show_maximized(fig2b)


# -------------------------------------------------------
# FIGURE 3: Forward demo
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    r"Step 4: Probability mode -- fix $p$, generate data",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

p_true = 0.7
n_flips = 40
n_reps = 500
counts = RNG.binomial(n_flips, p_true, size=n_reps)
axes3[0].hist(counts, bins=np.arange(0, n_flips + 2) - 0.5, density=True,
              color="#2874a6", edgecolor="white", alpha=0.85)
axes3[0].axvline(p_true * n_flips, color=C_ACCENT, ls="--", lw=2.2,
                 label=fr"$np={p_true * n_flips:g}$")
axes3[0].set_xlabel(f"number of heads in {n_flips} flips", fontsize=FS_AXIS)
axes3[0].set_ylabel("density", fontsize=FS_AXIS)
axes3[0].set_title(fr"known coin $p={p_true:g}$", fontsize=FS_AX_TITLE - 2)
axes3[0].legend(fontsize=FS_LEGEND - 1)
axes3[0].grid(True, alpha=0.3)

# one realized sequence
seq = (RNG.random(n_flips) < p_true).astype(float)
axes3[1].scatter(np.arange(1, n_flips + 1), seq, c=C_DARK, s=36)
axes3[1].set_ylim(-0.2, 1.3)
axes3[1].set_xlabel("flip index", fontsize=FS_AXIS)
axes3[1].set_ylabel("outcome (1=H)", fontsize=FS_AXIS)
axes3[1].set_title("one random sample path", fontsize=FS_AX_TITLE - 2)
axes3[1].grid(True, alpha=0.3)
fig3.tight_layout(rect=[0, 0.02, 1, 0.92])
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Inverse demo
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    r"Step 5: Statistics mode -- see data, infer $p$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

# Observed data from unknown p (we pretend not to know)
obs = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1])
k_obs, n_obs = int(obs.sum()), len(obs)
p_hat = k_obs / n_obs
ps = np.linspace(0.01, 0.99, 400)
lik = bernoulli_lik(ps, k_obs, n_obs)
lik /= lik.max()

axes4[0].scatter(np.arange(1, n_obs + 1), obs, c=C_OK, s=44)
axes4[0].set_ylim(-0.2, 1.3)
axes4[0].set_xlabel("flip index", fontsize=FS_AXIS)
axes4[0].set_ylabel("observed outcome", fontsize=FS_AXIS)
axes4[0].set_title(fr"data only: {k_obs} heads / {n_obs} flips", fontsize=FS_AX_TITLE - 2)
axes4[0].grid(True, alpha=0.3)

axes4[1].plot(ps, lik, color=C_OK, lw=2.6)
axes4[1].axvline(p_hat, color=C_ACCENT, ls="--", lw=2.2, label=fr"MLE $\hat p={p_hat:.2f}$")
axes4[1].set_xlabel(r"candidate $p$", fontsize=FS_AXIS)
axes4[1].set_ylabel("likelihood (scaled)", fontsize=FS_AXIS)
axes4[1].set_title(r"which $p$ makes these data most likely?", fontsize=FS_AX_TITLE - 2)
axes4[1].legend(fontsize=FS_LEGEND - 1)
axes4[1].grid(True, alpha=0.3)
fig4.tight_layout(rect=[0, 0.02, 1, 0.92])
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Interactive probability mode
# -------------------------------------------------------
fig5, (ax5a, ax5b) = new_figure(1, 2)
fig5.suptitle(
    r"Step 6: Probability toy -- drag true $p$, watch samples",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
fig5.subplots_adjust(bottom=0.18, wspace=0.28)

N_FLIP = 60


def draw_prob(p):
    ax5a.clear(); ax5b.clear()
    p = float(p)
    reps = RNG.binomial(N_FLIP, p, size=800)
    ax5a.hist(reps, bins=np.arange(0, N_FLIP + 2) - 0.5, density=True,
              color="#2874a6", edgecolor="white", alpha=0.85)
    ax5a.axvline(p * N_FLIP, color=C_ACCENT, ls="--", lw=2.2)
    ax5a.set_xlim(-1, N_FLIP + 1)
    ax5a.set_xlabel("heads count", fontsize=FS_AXIS)
    ax5a.set_ylabel("density", fontsize=FS_AXIS)
    ax5a.set_title(fr"P(data | $p$={p:.2f})", fontsize=FS_AX_TITLE - 2)
    ax5a.grid(True, alpha=0.3)

    seq = (RNG.random(N_FLIP) < p).astype(float)
    ax5b.scatter(np.arange(1, N_FLIP + 1), seq, c=C_DARK, s=28)
    ax5b.axhline(p, color=C_ACCENT, ls="--", lw=1.8, label=fr"$p={p:.2f}$")
    ax5b.set_ylim(-0.2, 1.35)
    ax5b.set_xlabel("flip", fontsize=FS_AXIS)
    ax5b.set_title(fr"one sample  (mean={seq.mean():.2f})", fontsize=FS_AX_TITLE - 2)
    ax5b.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    ax5b.grid(True, alpha=0.3)


ax_p5 = fig5.add_axes([0.25, 0.06, 0.50, 0.04])
sl_p5 = Slider(ax_p5, r"true $p$", 0.05, 0.95, valinit=0.70, valstep=0.01, color="#2874a6")


def _u5(_=None):
    draw_prob(sl_p5.val)
    fig5.canvas.draw_idle()


sl_p5.on_changed(_u5)
_u5()
print("Step 6: probability mode -- drag p.")
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Interactive statistics mode
# -------------------------------------------------------
fig6, (ax6a, ax6b) = new_figure(1, 2)
fig6.suptitle(
    r"Step 7: Statistics toy -- fixed data; scan candidate $p$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
fig6.subplots_adjust(bottom=0.18, wspace=0.28)

# Fixed observed dataset (unknown true p for the student view)
DATA6 = (RNG.random(80) < 0.62).astype(int)
K6, N6 = int(DATA6.sum()), len(DATA6)
PS6 = np.linspace(0.01, 0.99, 400)
LIK6 = bernoulli_lik(PS6, K6, N6)
LIK6 /= LIK6.max()
PHAT6 = K6 / N6


def draw_stat(p_cand):
    ax6a.clear(); ax6b.clear()
    p_cand = float(p_cand)
    ax6a.scatter(np.arange(1, N6 + 1), DATA6, c=C_OK, s=22, alpha=0.85)
    ax6a.set_ylim(-0.2, 1.35)
    ax6a.set_xlabel("flip", fontsize=FS_AXIS)
    ax6a.set_title(fr"observed data: $\hat p_{{\mathrm{{MLE}}}}={PHAT6:.2f}$",
                   fontsize=FS_AX_TITLE - 2)
    ax6a.grid(True, alpha=0.3)

    ax6b.plot(PS6, LIK6, color=C_OK, lw=2.4, label="likelihood")
    ax6b.axvline(PHAT6, color=C_ACCENT, ls="--", lw=2.0, label="MLE")
    ax6b.axvline(p_cand, color=C_FUN, lw=2.2, label=fr"candidate $p={p_cand:.2f}$")
    y_c = bernoulli_lik(p_cand, K6, N6) / bernoulli_lik(PHAT6, K6, N6)
    ax6b.scatter([p_cand], [y_c], s=70, c=C_FUN, zorder=3)
    ax6b.set_xlabel(r"candidate $p$", fontsize=FS_AXIS)
    ax6b.set_ylabel("likelihood (scaled)", fontsize=FS_AXIS)
    ax6b.set_title(fr"$L(p)$ at candidate = {y_c:.3f} (rel. to MLE)",
                   fontsize=FS_AX_TITLE - 3)
    ax6b.legend(fontsize=FS_LEGEND - 1, loc="upper left")
    ax6b.grid(True, alpha=0.3)
    ax6b.set_ylim(-0.05, 1.15)


ax_p6 = fig6.add_axes([0.25, 0.06, 0.50, 0.04])
sl_p6 = Slider(ax_p6, r"candidate $p$", 0.05, 0.95, valinit=0.40, valstep=0.01, color=C_FUN)


def _u6(_=None):
    draw_stat(sl_p6.val)
    fig6.canvas.draw_idle()


sl_p6.on_changed(_u6)
_u6()
print("Step 7: statistics mode -- drag candidate p.")
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Bayes as bridge
# -------------------------------------------------------
fig7, ax7 = new_figure()
fig7.suptitle(
    r"Step 8: Bayes bridges the directions -- $P(\mathrm{world}\mid\mathrm{data})$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax7)

round_box(ax7, 0.5, 5.2, 3.4, 2.3, C_NEW, "#2874a6", lw=2.0)
ax7.text(2.2, 6.35, "Prior", ha="center", fontsize=FS_ANNOT + 1,
         fontweight="bold", color="#2874a6")
ax7.text(2.2, 5.6, r"$P(\mathrm{world})$", ha="center", fontsize=FS_ANNOT)

round_box(ax7, 4.3, 5.2, 3.4, 2.3, "#f5eef8", C_FUN, lw=2.0)
ax7.text(6.0, 6.35, "Likelihood", ha="center", fontsize=FS_ANNOT + 1,
         fontweight="bold", color=C_FUN)
ax7.text(6.0, 5.6, r"$P(\mathrm{data}\mid\mathrm{world})$", ha="center", fontsize=FS_ANNOT)

round_box(ax7, 8.1, 5.2, 3.4, 2.3, C_GOAL, C_OK, lw=2.0)
ax7.text(9.8, 6.35, "Posterior", ha="center", fontsize=FS_ANNOT + 1,
         fontweight="bold", color=C_OK)
ax7.text(9.8, 5.6, r"$P(\mathrm{world}\mid\mathrm{data})$", ha="center", fontsize=FS_ANNOT)

ax7.annotate("", xy=(4.2, 6.35), xytext=(4.0, 6.35),
             arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2))
ax7.annotate("", xy=(8.0, 6.35), xytext=(7.8, 6.35),
             arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2))

round_box(ax7, 0.5, 0.6, 11.0, 4.0, C_MUTE, "0.45", lw=1.6)
ax7.text(
    6.0, 2.6,
    "Likelihood is a probability statement (forward).\n"
    "Posterior is a statistical belief (inverse).\n\n"
    "Bayes multiplies them -- then renormalizes by P(data).\n"
    "MLE keeps only the likelihood peak (a statistical shortcut).",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Modeling vibes
# -------------------------------------------------------
fig8, ax8 = new_figure()
fig8.suptitle(
    "Step 9: In modeling -- generative stories vs discriminative maps",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax8)

round_box(ax8, 0.4, 2.4, 5.4, 4.8, C_NEW, "#2874a6", lw=2.2)
ax8.text(3.1, 6.6, "Generative vibe", ha="center",
         fontsize=FS_AX_TITLE - 2, fontweight="bold", color="#2874a6")
ax8.text(
    3.1, 4.3,
    "Model how data are born:\n"
    r"$P(x)$, or $P(x\mid y)$, $P(y)$" + "\n\n"
    "Probability-first.\n"
    "Can simulate new x.\n"
    "Bayes / MLE natural tools.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)

round_box(ax8, 6.2, 2.4, 5.4, 4.8, C_GOAL, C_OK, lw=2.2)
ax8.text(8.9, 6.6, "Discriminative vibe", ha="center",
         fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
ax8.text(
    8.9, 4.3,
    "Model the decision map:\n"
    r"$P(y\mid x)$ or a score $f(x)$" + "\n\n"
    "Statistics-first for labels.\n"
    "Train on observed pairs.\n"
    "Judge on unobserved cases.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)

round_box(ax8, 0.4, 0.35, 11.2, 1.7, C_MUTE, "0.45", lw=1.6)
ax8.text(
    6.0, 1.2,
    "Both need the probability↔statistics handshake: score data under a model, then update the model from data.",
    ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
)
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 9: Takeaway
# -------------------------------------------------------
fig9, axes9 = new_figure(1, 3)
fig9.suptitle(
    "Step 10: Difference and relation -- remember both",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

panels = [
    (axes9[0], "DIFFERENCE",
     "Probability: model given,\npredict data.\n\n"
     "Statistics: data given,\ninfer model.",
     C_WARN, C_ACCENT),
    (axes9[1], "RELATION",
     "Both speak through\n"
     r"$P(\mathrm{data}\mid\mathrm{model})$."
     "\n\nSame link, opposite use.",
     C_GOAL, C_OK),
    (axes9[2], "BRIDGE",
     "Bayes: posterior beliefs.\n"
     "MLE: maximize likelihood.\n\n"
     "Statistics uses\nprobability tools.",
     "#f5eef8", C_FUN),
]
for ax, title, body, face, edge in panels:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    round_box(ax, 0.5, 1.5, 9.0, 7.0, face, edge, lw=2.2)
    ax.text(5.0, 7.2, title, ha="center", fontsize=FS_AX_TITLE - 2,
            fontweight="bold", color=edge)
    ax.text(5.0, 4.0, body, ha="center", va="center", fontsize=FS_ANNOT + 1)
show_maximized(fig9)


print("\n" + "=" * 60)
print("PROBABILITY vs STATISTICS")
print("=" * 60)
print("""
DIFFERENCE
  Probability theory: given a model, reason about data (forward).
  Statistics: given data, reason about a model (inverse).

RELATION
  Both use P(data | model).
  Probability evaluates it; statistics optimizes / updates over models.

BRIDGE
  Bayes posterior and MLE are statistical goals built from probability tools.
""")
