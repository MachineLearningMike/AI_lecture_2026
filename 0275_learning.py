"""
0275_learning.py

What "learning" means in this course: from observed pairs to a model
that predicts the not-yet-observed.

Sketch:
  1. What are observed data, for example?
  2. A quiet note on noise
  3. What is the task?
  4. What to do? -- MLE enters
  5. Past and future are not always the same
  6. Essential assumptions
  7. How MLE fits the world -- Gaussian <=> MSE
  8. Other working combinations
  9. Alternatives to MLE
  10. Different distributions of Y need different objectives?

Sits between 0270_torch_training.py and 0280 / 0470_maximum_likelihood.py.
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
    SCREEN_FRACTION,
    apply_rcparams,
    add_step_footer,
)
apply_rcparams(plt)
import numpy as np

np.random.seed(27)


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


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=2.4):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=16),
    )


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_WARN = "#f8d7da"
C_FUN = "#8e44ad"
C_OLD = "#fce4d6"
C_CODE = "#fbfcf8"


# -------------------------------------------------------
# FIGURE 1: What are observed data?
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    "Step 1: What are observed data, for example?",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax1)

round_box(ax1, 0.4, 6.35, 11.2, 1.25, C_DARK, C_DARK, lw=0)
ax1.text(
    6.0, 6.95,
    "Observed data are what we actually recorded -- not the hidden clean truth.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
    fontweight="bold", color="#f9e79f",
)

# Pipeline cards
pipeline = [
    (0.4, 3.55, "X_org", "latent clean\ncause"),
    (2.7, 3.55, "X_input", "noisy input\nto the process"),
    (5.0, 3.55, "process(...)", "unknown map\n+ more noise"),
    (7.5, 3.55, "Y path", "output before\nobservation noise"),
    (9.7, 3.55, "observed", "X_observed,\nY_observed"),
]
for x, y, title, body in pipeline:
    round_box(ax1, x, y, 2.0, 2.5, C_NEW, "#2874a6", lw=1.8)
    ax1.text(x + 1.0, y + 1.95, title, ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax1.text(x + 1.0, y + 0.95, body, ha="center", va="center",
             fontsize=FS_TICK - 1, color="#333333")
for x1, x2 in [(2.4, 2.7), (4.7, 5.0), (7.0, 7.5), (9.5, 9.7)]:
    arrow(ax1, x1, 4.8, x2, 4.8, C_DARK, lw=2.0)

round_box(ax1, 0.4, 0.25, 11.2, 3.05, C_CODE, "0.55", lw=1.5)
ax1.text(
    6.0, 2.85,
    "A concrete generative sketch (noise everywhere)",
    ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
)
ax1.text(
    6.0, 1.45,
    "X_org = randn(N, 1) + 1.0\n"
    "X_input = X_org + 0.1 * randn(N, 1)\n"
    "others = 0.1 * randn(N, 1)\n"
    "process(X, others) = 2*(X + 0.3*others)^3 + 3 + 0.2*randn(...)\n"
    "X_observed = X_input + 0.1*randn(...)\n"
    "Y_observed = process(X_input, others) + 0.1*randn(...)",
    ha="center", va="center", fontsize=FS_TICK - 1, family="monospace",
    color="#1c2833",
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Note on noise
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    "Step 2: A quiet note on noise",
    fontsize=FS_FIGURE_TITLE, fontweight="bold",
)
blank_slide(ax2)

round_box(ax2, 1.2, 3.2, 9.6, 3.2, C_WARN, C_ACCENT, lw=2.4)
ax2.text(
    6.0, 5.3,
    "It would be a heaven if all noises\nhad normal distributions.",
    ha="center", va="center", fontsize=FS_AX_TITLE,
    fontweight="bold", color="#641e16",
)
ax2.text(
    6.0, 3.85,
    "Reality mixes measurement error, omitted variables,\n"
    "rounding, and regime quirks. Gaussian noise is a\n"
    "convenient working hypothesis -- not a birthright.",
    ha="center", va="center", fontsize=FS_ANNOT + 1, color="#641e16",
)

round_box(ax2, 1.2, 0.8, 9.6, 1.9, C_MUTE, "0.45", lw=1.6)
ax2.text(
    6.0, 1.75,
    "We still start with Gaussian thinking, because it yields\n"
    "clean mathematics and a familiar objective: squared error.\n"
    "Later we ask what changes when the noise is not Gaussian.",
    ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: What is the task?
# -------------------------------------------------------
fig3, ax3 = new_figure()
fig3.suptitle(
    "Step 3: What is the task?",
    fontsize=FS_FIGURE_TITLE, fontweight="bold",
)
blank_slide(ax3)

round_box(ax3, 0.5, 5.55, 11.0, 2.0, C_DARK, C_DARK, lw=0)
ax3.text(
    6.0, 6.55,
    "Given a newly observed x, predict the corresponding y\n"
    "that exists but is not observed -- yet, or for some other reason.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
    fontweight="bold", color="#f9e79f",
)

trio = [
    (0.5, 2.35, "Input",
     "x_newly_observed\n\n"
     "Seen now;\nnever used as a\ntraining label."),
    (4.25, 2.35, "Target",
     "y_not_observed\n\n"
     "Exists in the world.\nHidden by time,\ncost, or access."),
    (8.0, 2.35, "Goal",
     "Learn Model such that\n\n"
     "Model(x_new)\n≈ y_not_observed"),
]
for x, y, title, body in trio:
    round_box(ax3, x, y, 3.5, 2.9, C_GOAL, C_OK, lw=2.0)
    ax3.text(x + 1.75, y + 2.4, title, ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    ax3.text(x + 1.75, y + 1.1, body, ha="center", va="center",
             fontsize=FS_TICK + 1)

round_box(ax3, 0.5, 0.35, 11.0, 1.7, C_NEW, "#2874a6", lw=2.0)
ax3.text(
    6.0, 1.2,
    "Objective (example): \"near\" means small Euclidean distance.\n"
    "Every observation carries equal weight of importance.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
    fontweight="bold", color=C_DARK,
)
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: What to do? MLE enters
# -------------------------------------------------------
fig4, ax4 = new_figure()
fig4.suptitle(
    "Step 4: What to do?  --  MLE enters",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax4)

round_box(ax4, 0.5, 5.7, 11.0, 1.8, C_WARN, C_ACCENT, lw=2.2)
ax4.text(
    6.0, 6.6,
    "Many candidate Models can look plausible.\n"
    "Goal and objective are still too ambiguous to crown a unique winner.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
    fontweight="bold", color="#641e16",
)

round_box(ax4, 0.5, 2.7, 11.0, 2.6, C_NEW, "#2874a6", lw=2.2)
ax4.text(
    6.0, 4.55,
    "Maximum Likelihood Estimate",
    ha="center", va="center", fontsize=FS_AX_TITLE - 1,
    fontweight="bold", color=C_DARK,
)
ax4.text(
    6.0, 3.5,
    "A model that best explains the observed pairs\n"
    "(X_observed, Y_observed) under a stated probability story.\n\n"
    "MLE = \"which candidate makes these data most probable?\"",
    ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
)

cols = [
    (0.5, 0.35, "Not unique",
     "Even among MLEs,\nseveral Models may\ntie or nearly tie."),
    (4.25, 0.35, "Selection remains",
     "We still choose\namong MLE candidates\nby structure / prior."),
    (8.0, 0.35, "Why it helps",
     "Turns a vague goal\ninto a concrete score\non the observations."),
]
for x, y, title, body in cols:
    round_box(ax4, x, y, 3.5, 2.05, C_MUTE, "0.45", lw=1.6)
    ax4.text(x + 1.75, y + 1.55, title, ha="center",
             fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax4.text(x + 1.75, y + 0.7, body, ha="center", va="center",
             fontsize=FS_TICK)
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Past vs future
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    "Step 5: Observed = past, to-be-observed = future?  Not always.",
    fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
)
blank_slide(ax5)

round_box(ax5, 0.5, 5.4, 5.3, 2.1, C_OLD, "#e67e22", lw=2.0)
ax5.text(3.15, 6.85, "Often true", ha="center",
         fontsize=FS_AX_TITLE - 2, fontweight="bold", color="#a04000")
ax5.text(
    3.15, 5.95,
    "Train on yesterday's logs.\n"
    "Predict tomorrow's outcome.\n"
    "Time hid the label.",
    ha="center", va="center", fontsize=FS_ANNOT,
)

round_box(ax5, 6.2, 5.4, 5.3, 2.1, C_NEW, "#2874a6", lw=2.0)
ax5.text(8.85, 6.85, "Often false", ha="center",
         fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
ax5.text(
    8.85, 5.95,
    "Label exists now, but is costly,\n"
    "private, or measured later.\n"
    "\"Future\" is a metaphor for hidden.",
    ha="center", va="center", fontsize=FS_ANNOT,
)

round_box(ax5, 0.5, 0.7, 11.0, 4.2, C_MUTE, "0.45", lw=1.8)
ax5.text(
    6.0, 4.3,
    "The learning claim is not about the calendar.",
    ha="center", va="center", fontsize=FS_AX_TITLE - 2,
    fontweight="bold", color=C_DARK,
)
ax5.text(
    6.0, 2.5,
    "It is about transfer:\n\n"
    "patterns estimated from the pairs we hold\n"
    "should remain useful for pairs we do not hold --\n"
    "whether they arrive later in time, elsewhere in space,\n"
    "or simply behind a curtain we cannot yet lift.",
    ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
)
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Essential assumptions
# -------------------------------------------------------
fig6, ax6 = new_figure()
fig6.suptitle(
    "Step 6: What is the essential assumption?",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax6)

round_box(ax6, 0.5, 5.5, 11.0, 2.0, C_DARK, C_DARK, lw=0)
ax6.text(
    6.0, 6.5,
    "1. Observed and to-be-observed come from the same process.\n"
    "2. That process is largely stationary, and continuous in the inputs.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
    fontweight="bold", color="#f9e79f",
)

round_box(ax6, 0.5, 2.85, 5.3, 2.3, C_WARN, C_ACCENT, lw=2.2)
ax6.text(3.15, 4.55, "Stationarity breaks", ha="center",
         fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT)
ax6.text(
    3.15, 3.6,
    "What if the slot machine\nis changed overnight?\n\n"
    "Yesterday's MLE describes\na machine that no longer exists.",
    ha="center", va="center", fontsize=FS_TICK + 1,
)

round_box(ax6, 6.2, 2.85, 5.3, 2.3, C_OLD, "#e67e22", lw=2.2)
ax6.text(8.85, 4.55, "Continuity breaks", ha="center",
         fontsize=FS_ANNOT, fontweight="bold", color="#a04000")
ax6.text(
    8.85, 3.6,
    "What if the machine has a\nsharp leap at one input?\n\n"
    "Local smoothness fails;\n"
    "nearby x no longer informs y.",
    ha="center", va="center", fontsize=FS_TICK + 1,
)

round_box(ax6, 0.5, 0.4, 11.0, 2.1, C_GOAL, C_OK, lw=2.0)
ax6.text(
    6.0, 1.45,
    "Learning is licensed by sameness of process.\n"
    "When the world jumps -- in time or in input --\n"
    "the license expires, no matter how elegant the estimator.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
    fontweight="bold", color=C_OK,
)
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Gaussian MLE <=> MSE
# -------------------------------------------------------
fig7, ax7 = new_figure()
fig7.suptitle(
    r"Step 7: How does MLE fit the world?  Gaussian $\Leftrightarrow$ MSE",
    fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
)
blank_slide(ax7)

round_box(ax7, 0.4, 5.35, 11.2, 2.2, C_NEW, "#2874a6", lw=2.0)
ax7.text(
    6.0, 6.85,
    r"Model the residual:   $Y = f_\theta(X) + \varepsilon$"
    "\n"
    r"with  $\varepsilon_i \stackrel{iid}{\sim} \mathcal{N}(0,\sigma^2)$",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
    fontweight="bold", color=C_DARK,
)
ax7.text(
    6.0, 5.75,
    "Then the likelihood of the observed pairs factors nicely.",
    ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
)

round_box(ax7, 0.4, 2.35, 11.2, 2.7, C_MUTE, "0.45", lw=1.6)
ax7.text(
    6.0, 4.45,
    "Proof sketch (iid Gaussian noise)",
    ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
)
ax7.text(
    6.0, 3.35,
    r"$L(\theta)=\prod_i (2\pi\sigma^2)^{-1/2}"
    r"\exp\!\big(-(y_i-f_\theta(x_i))^2/(2\sigma^2)\big)$"
    "\n\n"
    r"$\log L(\theta) = \mathrm{const}"
    r" - \frac{1}{2\sigma^2}\sum_i (y_i-f_\theta(x_i))^2$"
    "\n\n"
    r"Maximizing $\log L$  $\Longleftrightarrow$  minimizing"
    r"  $\sum_i (y_i-f_\theta(x_i))^2$  (MSE).",
    ha="center", va="center", fontsize=FS_TICK + 1, color="#1c2833",
)

round_box(ax7, 0.4, 0.35, 11.2, 1.7, C_GOAL, C_OK, lw=2.0)
ax7.text(
    6.0, 1.2,
    "(MLE, Gaussian noise, MSE) is a working combination.\n"
    "Train with squared error  =  do Gaussian maximum likelihood.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
    fontweight="bold", color=C_OK,
)
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Other combinations
# -------------------------------------------------------
fig8, ax8 = new_figure()
fig8.suptitle(
    "Step 8: (MLE, other distributions, other losses) -- find them",
    fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
)
blank_slide(ax8)

pairs = [
    (0.4, 4.35, "Gaussian", "Squared error\n(MSE / L2)", C_GOAL, C_OK),
    (4.2, 4.35, "Laplace", "Absolute error\n(MAE / L1)", C_NEW, "#2874a6"),
    (8.0, 4.35, "Bernoulli", "Binary cross-entropy", C_OLD, "#e67e22"),
    (0.4, 1.55, "Categorical", "Categorical\ncross-entropy", "#f5eef8", C_FUN),
    (4.2, 1.55, "Poisson", "Poisson deviance\n/ count NLL", C_MUTE, "0.45"),
    (8.0, 1.55, "Huber-ish", "Robust losses ≈\ncontaminated noise", C_WARN, C_ACCENT),
]
for x, y, dist, loss, face, edge in pairs:
    round_box(ax8, x, y, 3.5, 2.4, face, edge, lw=2.0)
    ax8.text(x + 1.75, y + 1.8, dist, ha="center",
             fontsize=FS_ANNOT, fontweight="bold", color=edge if edge != "0.45" else C_DARK)
    ax8.text(x + 1.75, y + 0.9, loss, ha="center", va="center",
             fontsize=FS_TICK + 1, color="#333333")

ax8.text(
    6.0, 0.55,
    "Choose the noise story first; the loss is then not a fashion --\n"
    "it is the negative log-likelihood (NLL) of that story.",
    ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
)
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 9: Alternatives to MLE
# -------------------------------------------------------
fig9, ax9 = new_figure()
fig9.suptitle(
    "Step 9: What are alternatives to MLE?",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax9)

round_box(ax9, 0.5, 5.6, 11.0, 1.9, C_WARN, C_ACCENT, lw=2.0)
ax9.text(
    6.0, 6.55,
    "Do you agree on MLE?\n"
    "It is powerful -- and it is still only one way to pin down an ambiguous goal.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
    fontweight="bold", color="#641e16",
)

alts = [
    (0.5, 2.55, "Bayesian",
     "Put a prior on Model.\n"
     "Predict with the\nposterior, not a point."),
    (4.25, 2.55, "Risk / decision",
     "Minimize expected cost\nunder an explicit\nloss and deployment."),
    (8.0, 2.55, "Robust / minimax",
     "Prepare for worst\nplausible noise or\nshift, not the MLE fit."),
]
for x, y, title, body in alts:
    round_box(ax9, x, y, 3.5, 2.7, C_NEW, "#2874a6", lw=2.0)
    ax9.text(x + 1.75, y + 2.15, title, ha="center",
             fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax9.text(x + 1.75, y + 1.0, body, ha="center", va="center",
             fontsize=FS_TICK + 1)

round_box(ax9, 0.5, 0.35, 11.0, 1.85, C_MUTE, "0.45", lw=1.6)
ax9.text(
    6.0, 1.25,
    "Other routes to the same vague goal: method of moments, scoring rules,\n"
    "contrastive objectives, adversarial criteria, human preference feedback.\n"
    "Each replaces \"most likely under a noise model\" with a different virtue.",
    ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
)
show_maximized(fig9)


# -------------------------------------------------------
# FIGURE 10: Different Y distributions, different objectives
# -------------------------------------------------------
fig10, ax10 = new_figure()
fig10.suptitle(
    "Step 10: Different distributions of Y_observed need different objectives?",
    fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
)
blank_slide(ax10)

round_box(ax10, 0.5, 5.5, 11.0, 2.0, C_DARK, C_DARK, lw=0)
ax10.text(
    6.0, 6.5,
    "Yes -- if you stay inside the MLE family.\n"
    "The distribution of Y (given X) names the negative log-likelihood.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
    fontweight="bold", color="#f9e79f",
)

rows = [
    (0.5, 3.55, "Continuous, light tails", "Gaussian story → MSE"),
    (6.2, 3.55, "Continuous, heavy tails", "Laplace / Huber → MAE-like"),
    (0.5, 1.75, "Binary labels", "Bernoulli → BCE"),
    (6.2, 1.75, "Many classes", "Categorical → CCE"),
]
for x, y, title, body in rows:
    round_box(ax10, x, y, 5.3, 1.5, C_GOAL, C_OK, lw=1.8)
    ax10.text(x + 2.65, y + 1.05, title, ha="center",
              fontsize=FS_TICK + 1, fontweight="bold", color=C_OK)
    ax10.text(x + 2.65, y + 0.45, body, ha="center",
              fontsize=FS_TICK + 1, color="#333333")

round_box(ax10, 0.5, 0.25, 11.0, 1.25, C_MUTE, "0.45", lw=1.6)
ax10.text(
    6.0, 0.85,
    "Mismatch the noise story and the loss, and you optimize the wrong virtue:\n"
    "a precise answer to a question the data did not ask.",
    ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
)
show_maximized(fig10)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("LEARNING -- FROM OBSERVED PAIRS TO A MODEL")
print("=" * 60)
print("""
Observed data:
  Noisy records of inputs and outputs from a hidden process.
  Clean X_org is usually never seen.

Task:
  Predict y_not_observed from x_newly_observed.
  \"Near\" may mean small Euclidean distance; equal weights.

MLE:
  Turns an ambiguous goal into a score on (X_obs, Y_obs).
  Still not unique; we choose among MLE candidates.

Assumptions:
  Same process; largely stationary; continuous in inputs.
  Slot machines that change overnight -- or leap in input -- break the license.

Gaussian MLE <=> MSE:
  Under iid N(0, sigma^2) residual noise, max log L == min squared error.
  Other distributions pair with other losses (Laplace/MAE, Bernoulli/BCE, ...).

Alternatives:
  Bayesian, decision-theoretic risk, robust/minimax, and many modern surrogates.
""")
