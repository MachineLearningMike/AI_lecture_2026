"""
Train / validate / test / deploy (+ monitor).

One coherent arc from fitting on observed cases to serving in the field.

Glossary:
  Train    = fit on observed cases reserved for updating.
  Validate = choose designs on a held-out validation set.
  Test     = trust with one locked score on a test set.
  Deploy   = serve on unobserved field cases (+ monitor).
  Cases    = observed vs unobserved (field at deploy).
  Loss/MSE = numeric stand-in for quality of choices.

Slogan:
  Train to fit. Validate to choose. Test to trust. Deploy to serve -- then watch.

Sketch:
  1. The world is uneven -- local noise, global drift
  2. Mechanism is not assessment
  3. Hold out unobserved cases (two piles; later validate+test)
  4. Why not train-only / why not one holdout (contamination)
  5. Three roles: fit / choose / trust
  6. Visual three-way split
  7. Interactive: model capacity (overfit)
  8. Interactive: early stopping epoch
  9. Interactive: train / val / test fractions
  10. Full arc including deploy
  11. Stage seeker (Prev / Next)
  12. A green test is not sufficient
  13. Monitor / retrain loop
  14. Takeaway map (conceptual labels only)
  15. Sets suffer sampling noise and distribution shift
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import Button, Slider
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

np.random.seed(22)
RNG = np.random.default_rng(22)


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


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=2.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=18),
    )


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
C_ACTIVE = "#f9e79f"
C_TRAIN = "#2874a6"


# -------------------------------------------------------
# Shared toy data
# -------------------------------------------------------
def make_data(n=60, noise=0.35, seed=22):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1.6, 1.6, size=n))
    y_true = np.sin(2.2 * x) + 0.35 * x
    y = y_true + rng.normal(0, noise, size=n)
    return x, y, y_true


def poly_fit_predict(x_train, y_train, x_query, degree):
    degree = int(max(0, degree))
    coef = np.polyfit(x_train, y_train, degree)
    return np.polyval(coef, x_query), coef


def mse(y_hat, y):
    return float(np.mean((y_hat - y) ** 2))


def truth_curve(xx):
    return np.sin(2.2 * xx) + 0.35 * xx


X_ALL, Y_ALL, Y_TRUE = make_data()
PERM2 = RNG.permutation(len(X_ALL))


def split_xy(frac_train=0.7):
    n = len(X_ALL)
    n_tr = int(np.clip(round(frac_train * n), 5, n - 5))
    idx_tr = np.sort(PERM2[:n_tr])
    idx_ho = np.sort(PERM2[n_tr:])
    return (
        X_ALL[idx_tr], Y_ALL[idx_tr],
        X_ALL[idx_ho], Y_ALL[idx_ho],
        idx_tr, idx_ho,
    )


# Shared across Steps 3/7/8 (builders run independently)
xx = np.linspace(-1.8, 1.8, 200)
x_tr6, y_tr6, x_va6, y_va6, _, _ = split_xy(0.7)


# Larger population for three-pile demos
N = 100
X = np.sort(RNG.uniform(-1.5, 1.5, N))
Y = truth_curve(X) + RNG.normal(0, 0.3, N)
PERM = RNG.permutation(N)


def three_way_split(frac_tr=0.60, frac_va=0.20):
    n_tr = int(np.clip(round(frac_tr * N), 10, N - 20))
    n_va = int(np.clip(round(frac_va * N), 5, N - n_tr - 5))
    n_te = N - n_tr - n_va
    idx_tr = np.sort(PERM[:n_tr])
    idx_va = np.sort(PERM[n_tr:n_tr + n_va])
    idx_te = np.sort(PERM[n_tr + n_va:])
    return idx_tr, idx_va, idx_te, n_tr, n_va, n_te


STAGES = [
    ("TRAIN", "Fit the rule\non observed cases", C_NEW, C_TRAIN),
    ("VALIDATE", "Choose the rule\nwithout flattery", "#f5eef8", C_FUN),
    ("TEST", "Trust once --\nhonest passport", C_GOAL, C_OK),
    ("DEPLOY", "Serve unobserved\nfield cases", C_WARN, C_ACCENT),
]


# -------------------------------------------------------
# FIGURE 0: The world is uneven
# -------------------------------------------------------
def build_step_1():
    fig0, ax0 = new_figure()
    fig0.suptitle(
        "Step 1: The world is uneven.",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )

    n_uneven = 240
    x_pts = np.arange(1, n_uneven + 1)
    # Slow global drift (smooth, large-scale) -- larger amplitude
    global_drift = 2.4 * (
        0.55 * np.sin(2 * np.pi * x_pts / n_uneven)
        + 0.35 * (x_pts / n_uneven)
        - 0.25 * np.cos(4 * np.pi * x_pts / n_uneven)
    )
    # Local distribution: its own medium-scale drift; 2x local variance
    # (neighborhood mean and volatility both wander along x)
    local_mean_drift = 0.85 * np.sin(2 * np.pi * x_pts / (n_uneven / 6.0) + 0.4)
    local_scale = 2.0 * (0.45 + 1.15 * (0.5 + 0.5 * np.sin(2 * np.pi * x_pts / (n_uneven / 3.5) + 1.1)))
    # True local mean level: global slow drift + medium local mean wander
    local_mean = global_drift + local_mean_drift

    # Clouds = samples from the local distribution at each t (centered on true local mean)
    n_cloud = 14
    cloud_x, cloud_y = [], []
    for xi, mu_loc, s_loc in zip(x_pts, local_mean, local_scale):
        cloud_x.append(xi + RNG.normal(0.0, 0.55, size=n_cloud))
        cloud_y.append(mu_loc + RNG.normal(0.0, 0.22 * float(s_loc), size=n_cloud))
    cloud_x = np.concatenate(cloud_x)
    cloud_y = np.concatenate(cloud_y)
    ax0.scatter(cloud_x, cloud_y, s=8, c=C_TRAIN, alpha=0.28, edgecolors="none",
                label="local sample clouds", zorder=2)
    ax0.plot(x_pts, global_drift, color=C_ACCENT, lw=2.6, label="global drift", zorder=3)
    ax0.plot(x_pts, local_mean, color=C_FUN, lw=1.8, ls="--",
             alpha=0.85, label="local mean (global + local)", zorder=3)

    # Call out a calm-ish window vs a wild local-variance window
    win0, win1 = 55, 95
    win2, win3 = 155, 200
    ax0.axvspan(win0, win1, color=C_FUN, alpha=0.10, zorder=1)
    ax0.axvspan(win2, win3, color=C_BIN, alpha=0.12, zorder=1)
    ax0.annotate(
        "locally: large, wandering noise",
        xy=(0.5 * (win0 + win1), float(local_mean[win0:win1].max() + 0.4)),
        xytext=(15, float(np.nanmax(cloud_y) * 0.92)),
        fontsize=FS_ANNOT, color=C_FUN, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_FUN, lw=1.8),
    )
    ax0.annotate(
        "local variance also drifts",
        xy=(0.5 * (win2 + win3), float(local_mean[win2:win3].mean())),
        xytext=(125, float(np.nanmin(cloud_y) * 0.85)),
        fontsize=FS_ANNOT, color=C_BIN, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_BIN, lw=1.8),
    )
    ax0.annotate(
        "globally: a slow drift",
        xy=(210, float(global_drift[209])),
        xytext=(165, float(global_drift[209] - 1.1)),
        fontsize=FS_ANNOT, color=C_ACCENT, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.8),
    )

    ax0.set_xlabel("time", fontsize=FS_AXIS)
    ax0.set_ylabel("sample distribution", fontsize=FS_AXIS)
    ax0.set_title("local noise is large and drifting; a slower drift sits underneath",
                  fontsize=FS_AX_TITLE - 2)
    ax0.legend(fontsize=FS_LEGEND - 1, loc="lower left")
    ax0.grid(True, alpha=0.3)
    fig0.tight_layout(rect=[0, 0.12, 1, 0.92])
    fig0.text(
        0.5, 0.045,
        "We assume the data are i.i.d. (independent and identically distributed) from a stationary distribution.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=C_MUTE, edgecolor="0.45", linewidth=1.6),
    )
    return fig0


# -------------------------------------------------------
# FIGURE 1: Mechanism is not assessment
# -------------------------------------------------------
def build_step_2():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 2: The training loop is machinery, not a verdict",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax1)

    round_box(ax1, 0.5, 5.2, 11.0, 2.3, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 6.35,
        "A falling training loss proves you can fit observed cases.\n"
        "It does not prove quality of choices on unobserved cases.\n"
        "(Loss / MSE is only a numeric stand-in for that quality.)",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.5, 1.2, C_WARN, C_ACCENT, "Practice score", "quality of choices\non observed cases"),
        (4.25, 1.2, C_GOAL, C_OK, "Real exam", "quality of choices\non unobserved cases"),
        (8.0, 1.2, C_NEW, C_TRAIN, "Bridge", "hold out data\nthen validate / test"),
    ]
    for x0, y0, face, edge, title, body in cards:
        round_box(ax1, x0, y0, 3.5, 3.4, face, edge, lw=2.2)
        ax1.text(x0 + 1.75, y0 + 2.5, title, ha="center",
                 fontsize=FS_AX_TITLE - 2, fontweight="bold", color=edge)
        ax1.text(x0 + 1.75, y0 + 1.3, body, ha="center", va="center",
                 fontsize=FS_ANNOT + 1)
    return fig1


# -------------------------------------------------------
# FIGURE 2: Hold out (two piles)
# -------------------------------------------------------
def build_step_3():
    fig2, axes2 = new_figure(1, 2)
    fig2.suptitle(
        "Step 3: Hold out unobserved cases -- then you can assess honestly",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    x_tr, y_tr, x_ho, y_ho, _, _ = split_xy(0.7)
    axes2[0].scatter(X_ALL, Y_ALL, s=40, c=C_GRAY, edgecolors="white", label="all data")
    axes2[0].plot(xx, truth_curve(xx), color=C_DARK, lw=2.0, label="truth")
    axes2[0].set_title("before the split -- one pile", fontsize=FS_AX_TITLE - 2)
    axes2[0].legend(fontsize=FS_LEGEND - 1, loc="upper left")
    axes2[0].grid(True, alpha=0.3)

    axes2[1].scatter(x_tr, y_tr, s=45, c=C_TRAIN, edgecolors="white", label="train (observed)")
    axes2[1].scatter(x_ho, y_ho, s=55, c=C_ACCENT, marker="s", edgecolors="white", label="held-out")
    axes2[1].plot(xx, truth_curve(xx), color=C_DARK, lw=2.0, alpha=0.7)
    axes2[1].set_title("after the split -- train vs held-out", fontsize=FS_AX_TITLE - 2)
    axes2[1].legend(fontsize=FS_LEGEND - 1, loc="upper left")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].text(
        0.5, -0.18,
        "Next: split held-out into VALIDATE (choose) and TEST (trust).",
        transform=axes2[1].transAxes, ha="center", fontsize=FS_TICK, color=C_DARK,
    )
    for ax in axes2:
        ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
        ax.set_ylabel(r"$y$", fontsize=FS_AXIS)
    fig2.tight_layout(rect=[0, 0.06, 1, 0.92])
    return fig2


# -------------------------------------------------------
# FIGURE 3: Why not train-only / why not one holdout
# -------------------------------------------------------
def build_step_4():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        "Step 4: Why not train-only -- and why not one holdout pile?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax3)

    round_box(ax3, 0.5, 5.5, 11.0, 2.0, C_WARN, C_ACCENT, lw=2.2)
    ax3.text(
        6.0, 6.5,
        "Train-only never measures unobserved quality.\n"
        "One shared holdout, used both to choose and to publish, becomes contaminated.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        color="#641e16",
    )

    cols = [
        (0.5, 1.0, C_WARN, C_ACCENT, "Train-only",
         "One loss. One hope.\nNo independent check.\nOptimism is unmeasured."),
        (4.25, 1.0, "#f5eef8", C_FUN, "One holdout, many peeks",
         "You try knobs, pick the\nbest held-out score --\nnow that score is biased."),
        (8.0, 1.0, C_GOAL, C_OK, "Split the jobs",
         "VALIDATE = choose\nTEST = trust once\nDEPLOY waits for the claim."),
    ]
    for x0, y0, face, edge, title, body in cols:
        round_box(ax3, x0, y0, 3.5, 4.0, face, edge, lw=2.0)
        ax3.text(x0 + 1.75, y0 + 3.2, title, ha="center",
                 fontsize=FS_ANNOT + 1, fontweight="bold", color=edge)
        ax3.text(x0 + 1.75, y0 + 1.5, body, ha="center", va="center",
                 fontsize=FS_ANNOT)
    return fig3


# -------------------------------------------------------
# FIGURE 4: Three roles
# -------------------------------------------------------
def build_step_5():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        "Step 5: Three roles -- fit, choose, trust",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax4)

    roles = [
        (0.4, 2.2, C_NEW, C_TRAIN, "TRAIN\nfit",
         "Question:\nHow do I update?\n\n"
         "Use observed labels\nto change parameters."),
        (4.25, 2.2, "#f5eef8", C_FUN, "VALIDATE\nchoose",
         "Question:\nWhich design?\n\n"
         "Compare candidates;\nstop / select knobs."),
        (8.1, 2.2, C_GOAL, C_OK, "TEST\ntrust",
         "Question:\nHow good is it?\n\n"
         "One locked score\nbefore deploy."),
    ]
    for x0, y0, face, edge, title, body in roles:
        round_box(ax4, x0, y0, 3.5, 4.8, face, edge, lw=2.2)
        ax4.text(x0 + 1.75, y0 + 4.05, title, ha="center",
                 fontsize=FS_AX_TITLE - 2, fontweight="bold", color=edge)
        ax4.text(x0 + 1.75, y0 + 1.7, body, ha="center", va="center",
                 fontsize=FS_ANNOT + 1)

    round_box(ax4, 0.4, 0.3, 11.2, 1.6, C_MUTE, "0.45", lw=1.6)
    ax4.text(
        6.0, 1.1,
        "Separation of powers: fitting, choosing, and trusting must not share the same peeks.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )
    return fig4


# -------------------------------------------------------
# FIGURE 5: Visual three-way split
# -------------------------------------------------------
def build_step_6():
    fig5, (ax5a, ax5b) = new_figure(1, 2)
    fig5.suptitle(
        "Step 6: Shuffle, then cut into three locked piles",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    idx_tr, idx_va, idx_te, n_tr, n_va, n_te = three_way_split()
    ax5a.scatter(X[idx_tr], Y[idx_tr], s=28, c=C_TRAIN, marker="o", label="train", zorder=3)
    ax5a.scatter(X[idx_va], Y[idx_va], s=40, c=C_FUN, marker="D", label="validate", zorder=3)
    ax5a.scatter(X[idx_te], Y[idx_te], s=45, c=C_ACCENT, marker="s", label="test", zorder=3)
    xx5 = np.linspace(-1.7, 1.7, 200)
    ax5a.plot(xx5, truth_curve(xx5), color=C_DARK, lw=1.6, ls="--", alpha=0.8)
    ax5a.set_title("points colored by role", fontsize=FS_AX_TITLE - 2)
    ax5a.legend(fontsize=FS_LEGEND - 1, loc="upper left")
    ax5a.grid(True, alpha=0.3)
    ax5a.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax5a.set_ylabel(r"$y$", fontsize=FS_AXIS)

    ax5b.barh([0], [n_tr], color=C_TRAIN, height=0.45, label="train")
    ax5b.barh([0], [n_va], left=n_tr, color=C_FUN, height=0.45, label="validate")
    ax5b.barh([0], [n_te], left=n_tr + n_va, color=C_ACCENT, height=0.45, label="test")
    ax5b.set_xlim(0, N)
    ax5b.set_yticks([])
    ax5b.set_xlabel("number of examples", fontsize=FS_AXIS)
    ax5b.set_title(f"60 / 20 / 20  (n={N})", fontsize=FS_AX_TITLE - 2)
    ax5b.legend(fontsize=FS_LEGEND - 1, loc="upper center", ncol=3)
    ax5b.grid(True, alpha=0.3, axis="x")
    fig5.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig5


# -------------------------------------------------------
# FIGURE 6: Interactive capacity / overfitting
# -------------------------------------------------------
def build_step_7():
    fig6, (ax6a, ax6b) = new_figure(1, 2)
    fig6.suptitle(
        "Step 7: Drag capacity -- train may improve while validation worsens",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig6.subplots_adjust(bottom=0.18, wspace=0.28)

    deg0 = 3


    def draw_capacity(degree):
        ax6a.clear(); ax6b.clear()
        degree = int(degree)
        yhat_xx, _ = poly_fit_predict(x_tr6, y_tr6, xx, degree)
        yhat_tr, _ = poly_fit_predict(x_tr6, y_tr6, x_tr6, degree)
        yhat_va, _ = poly_fit_predict(x_tr6, y_tr6, x_va6, degree)
        tr_l = mse(yhat_tr, y_tr6)
        va_l = mse(yhat_va, y_va6)

        ax6a.scatter(x_tr6, y_tr6, s=35, c=C_TRAIN, marker="o", label="train", zorder=3)
        ax6a.scatter(x_va6, y_va6, s=45, c=C_FUN, marker="D", label="validate", zorder=3)
        ax6a.plot(xx, yhat_xx, color=C_FUN, lw=2.6, label=f"poly deg {degree}")
        ax6a.plot(xx, truth_curve(xx), color=C_DARK, lw=1.5, ls="--", alpha=0.8)
        ax6a.set_ylim(-2.8, 2.8)
        ax6a.set_title("fit on train only", fontsize=FS_AX_TITLE - 2)
        ax6a.legend(fontsize=FS_LEGEND - 1, loc="upper left")
        ax6a.grid(True, alpha=0.3)
        ax6a.set_xlabel(r"$x$", fontsize=FS_AXIS)
        ax6a.set_ylabel(r"$y$", fontsize=FS_AXIS)

        degrees = np.arange(0, 13)
        tr_losses, va_losses = [], []
        for d in degrees:
            ht, _ = poly_fit_predict(x_tr6, y_tr6, x_tr6, d)
            hv, _ = poly_fit_predict(x_tr6, y_tr6, x_va6, d)
            tr_losses.append(mse(ht, y_tr6))
            va_losses.append(mse(hv, y_va6))
        ax6b.plot(degrees, tr_losses, color=C_TRAIN, lw=2.4, marker="o", label="train MSE")
        ax6b.plot(degrees, va_losses, color=C_FUN, lw=2.4, marker="D", label="validation MSE")
        ax6b.axvline(degree, color=C_BIN, ls="--", lw=1.8)
        ax6b.scatter([degree], [va_l], s=80, c=C_BIN, zorder=3)
        ax6b.set_xlabel("polynomial degree (capacity)", fontsize=FS_AXIS)
        ax6b.set_ylabel("MSE (stand-in)", fontsize=FS_AXIS)
        ax6b.set_title(f"train {tr_l:.3f}   |   validation {va_l:.3f}", fontsize=FS_AX_TITLE - 2)
        ax6b.legend(fontsize=FS_LEGEND - 1, loc="upper left")
        ax6b.grid(True, alpha=0.3)
        tip = "underfit" if degree <= 1 else ("sweet spot?" if va_l <= min(va_losses) * 1.15 else "overfit risk")
        ax6b.text(0.98, 0.05, tip, transform=ax6b.transAxes, ha="right", va="bottom",
                  fontsize=FS_ANNOT, fontweight="bold",
                  color=C_OK if "sweet" in tip else (C_BIN if "under" in tip else C_ACCENT))


    ax_deg = fig6.add_axes([0.25, 0.06, 0.50, 0.04])
    sl_deg = Slider(ax_deg, "degree", 0, 12, valinit=deg0, valstep=1, color=C_FUN)


    def _u6(_=None):
        draw_capacity(sl_deg.val)
        fig6.canvas.draw_idle()


    sl_deg.on_changed(_u6)
    _u6()
    print("Step 7: drag polynomial degree -- train vs val.")
    return fig6


# -------------------------------------------------------
# FIGURE 7: Interactive early stopping
# -------------------------------------------------------
def build_step_8():
    fig7, (ax7a, ax7b) = new_figure(1, 2)
    fig7.suptitle(
        "Step 8: Early stopping -- pick the epoch where validation is best",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig7.subplots_adjust(bottom=0.18, wspace=0.28)

    path_tr, path_va, path_pred = [], [], []
    for k in range(31):
        t = k / 30.0
        low_tr, _ = poly_fit_predict(x_tr6, y_tr6, x_tr6, 1)
        high_tr, _ = poly_fit_predict(x_tr6, y_tr6, x_tr6, 9)
        low_va, _ = poly_fit_predict(x_tr6, y_tr6, x_va6, 1)
        high_va, _ = poly_fit_predict(x_tr6, y_tr6, x_va6, 9)
        low_xx, _ = poly_fit_predict(x_tr6, y_tr6, xx, 1)
        high_xx, _ = poly_fit_predict(x_tr6, y_tr6, xx, 9)
        s = t ** 1.4
        ptr = (1 - s) * low_tr + s * high_tr
        pva = (1 - s) * low_va + s * high_va
        pxx = (1 - s) * low_xx + s * high_xx
        path_tr.append(mse(ptr, y_tr6))
        path_va.append(mse(pva, y_va6) + 0.25 * max(0.0, s - 0.5) ** 2)
        path_pred.append(pxx)

    path_tr = np.array(path_tr)
    path_va = np.array(path_va)
    best_ep = int(np.argmin(path_va))
    ep0 = best_ep


    def draw_early(ep):
        ax7a.clear(); ax7b.clear()
        ep = int(ep)
        ax7a.plot(path_tr, color=C_TRAIN, lw=2.4, label="train")
        ax7a.plot(path_va, color=C_FUN, lw=2.4, label="validation")
        ax7a.axvline(ep, color=C_BIN, ls="--", lw=2.0, label=f"chosen epoch {ep}")
        ax7a.axvline(best_ep, color=C_OK, ls=":", lw=1.8, label=f"best validation {best_ep}")
        ax7a.scatter([ep], [path_va[ep]], s=70, c=C_BIN, zorder=3)
        ax7a.set_xlabel("epoch", fontsize=FS_AXIS)
        ax7a.set_ylabel("MSE (stand-in)", fontsize=FS_AXIS)
        ax7a.set_title("stop when validation bottoms out", fontsize=FS_AX_TITLE - 2)
        ax7a.legend(fontsize=FS_LEGEND - 1, loc="upper right")
        ax7a.grid(True, alpha=0.3)

        ax7b.scatter(x_tr6, y_tr6, s=30, c=C_TRAIN, marker="o", alpha=0.8, label="train")
        ax7b.scatter(x_va6, y_va6, s=40, c=C_FUN, marker="D", alpha=0.9, label="validate")
        ax7b.plot(xx, path_pred[ep], color=C_BIN, lw=2.6, label="model @ epoch")
        ax7b.plot(xx, truth_curve(xx), color=C_DARK, ls="--", lw=1.5)
        ax7b.set_ylim(-2.8, 2.8)
        ax7b.set_title(
            f"train {path_tr[ep]:.3f}  |  validation {path_va[ep]:.3f}",
            fontsize=FS_AX_TITLE - 2,
        )
        ax7b.legend(fontsize=FS_LEGEND - 1, loc="upper left")
        ax7b.grid(True, alpha=0.3)
        ax7b.set_xlabel(r"$x$", fontsize=FS_AXIS)


    ax_ep = fig7.add_axes([0.25, 0.06, 0.50, 0.04])
    sl_ep = Slider(ax_ep, "epoch", 0, len(path_va) - 1, valinit=ep0, valstep=1, color=C_FUN)


    def _u7(_=None):
        draw_early(sl_ep.val)
        fig7.canvas.draw_idle()


    sl_ep.on_changed(_u7)
    _u7()
    print("Step 8: drag epoch -- early stopping playground.")
    return fig7


# -------------------------------------------------------
# FIGURE 8: Interactive train / val / test fractions
# -------------------------------------------------------
def build_step_9():
    fig8, (ax8a, ax8b) = new_figure(1, 2)
    fig8.suptitle(
        "Step 9: Drag the split -- more train can help fit; less test weakens the verdict",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    fig8.subplots_adjust(bottom=0.20, wspace=0.28)


    def poly_mse(x_tr, y_tr, x_ev, y_ev, deg=4):
        coef = np.polyfit(x_tr, y_tr, deg)
        return float(np.mean((np.polyval(coef, x_ev) - y_ev) ** 2))


    def draw_split(frac_tr, frac_va):
        ax8a.clear(); ax8b.clear()
        frac_tr = float(frac_tr); frac_va = float(frac_va)
        if frac_tr + frac_va >= 0.95:
            frac_va = max(0.05, 0.95 - frac_tr)
        idx_tr, idx_va, idx_te, n_tr, n_va, n_te = three_way_split(frac_tr, frac_va)
        ax8a.scatter(X[idx_tr], Y[idx_tr], s=26, c=C_TRAIN, marker="o", label=f"train ({n_tr})")
        ax8a.scatter(X[idx_va], Y[idx_va], s=36, c=C_FUN, marker="D", label=f"val ({n_va})")
        ax8a.scatter(X[idx_te], Y[idx_te], s=40, c=C_ACCENT, marker="s", label=f"test ({n_te})")
        ax8a.set_title("current split", fontsize=FS_AX_TITLE - 2)
        ax8a.legend(fontsize=FS_LEGEND - 1, loc="upper left")
        ax8a.grid(True, alpha=0.3)
        ax8a.set_xlabel(r"$x$", fontsize=FS_AXIS)
        ax8a.set_ylabel(r"$y$", fontsize=FS_AXIS)

        tr_l = poly_mse(X[idx_tr], Y[idx_tr], X[idx_tr], Y[idx_tr])
        va_l = poly_mse(X[idx_tr], Y[idx_tr], X[idx_va], Y[idx_va])
        te_l = poly_mse(X[idx_tr], Y[idx_tr], X[idx_te], Y[idx_te])
        ax8b.bar(["train", "val", "test"], [tr_l, va_l, te_l],
                 color=[C_TRAIN, C_FUN, C_ACCENT], edgecolor="white", width=0.6)
        ax8b.set_ylabel("MSE of train-fit poly(4)", fontsize=FS_AXIS)
        ax8b.set_title("fit on train; score each pile", fontsize=FS_AX_TITLE - 2)
        ax8b.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate([tr_l, va_l, te_l]):
            ax8b.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=FS_TICK, fontweight="bold")


    ax_tr = fig8.add_axes([0.15, 0.08, 0.30, 0.04])
    ax_va = fig8.add_axes([0.55, 0.08, 0.30, 0.04])
    sl_tr = Slider(ax_tr, "train frac", 0.40, 0.80, valinit=0.60, valstep=0.05, color=C_TRAIN)
    sl_va = Slider(ax_va, "val frac", 0.10, 0.35, valinit=0.20, valstep=0.05, color=C_FUN)


    def _u8(_=None):
        draw_split(sl_tr.val, sl_va.val)
        fig8.canvas.draw_idle()


    sl_tr.on_changed(_u8)
    sl_va.on_changed(_u8)
    _u8()
    print("Step 9: drag train/val fractions.")
    return fig8


# -------------------------------------------------------
# FIGURE 9: Full arc including deploy
# -------------------------------------------------------
def build_step_10():
    fig9, ax9 = new_figure()
    fig9.suptitle(
        "Step 10: The full arc -- fit, choose, trust, then serve",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax9)

    for i, (name, body, face, edge) in enumerate(STAGES):
        x0 = 0.4 + i * 2.9
        round_box(ax9, x0, 3.6, 2.7, 3.5, face, edge, lw=2.2)
        ax9.text(x0 + 1.35, 6.4, f"{i + 1}. {name}", ha="center",
                 fontsize=FS_ANNOT + 1, fontweight="bold", color=edge)
        ax9.text(x0 + 1.35, 4.9, body, ha="center", va="center", fontsize=FS_ANNOT)
        if i < 3:
            arrow(ax9, x0 + 2.8, 5.3, x0 + 2.8 + 0.05, 5.3, C_DARK, lw=2.0)

    round_box(ax9, 0.4, 0.5, 11.2, 2.6, C_DARK, C_DARK, lw=0)
    ax9.text(
        6.0, 1.8,
        "Training builds a candidate.\n"
        "Validate chooses; test trusts -- then ask if it deserves the field.\n"
        "Deploy is not a slide -- it is a commitment to serve unobserved cases.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )
    return fig9


# -------------------------------------------------------
# FIGURE 10: Stage seeker
# -------------------------------------------------------
def build_step_11():
    fig10, (ax10a, ax10b) = new_figure(1, 2)
    fig10.suptitle(
        "Step 11: Stage seeker -- walk the arc (Prev / Next)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig10.subplots_adjust(bottom=0.18, wspace=0.25)

    STAGE_DETAIL = [
        ("TRAIN", "Fit: update weights on train only.\nLoss may fall by memorizing.\nOutput: candidate model.",
         ["fit", "gradients", "checkpoints"]),
        ("VALIDATE", "Choose: score candidates on validation.\nPick capacity / epoch.\nOutput: selected design.",
         ["choose", "early stop", "abort"]),
        ("TEST", "Trust: freeze choices.\nScore once on test.\nOutput: honest claim.",
         ["lock", "trust", "gate"]),
        ("DEPLOY", "Serve on unobserved field cases.\nWatch live metrics.\nOutput: field behavior.",
         ["serve", "monitor", "rollback"]),
    ]


    def draw_stage(k):
        ax10a.clear(); ax10b.clear()
        k = int(np.clip(k, 0, 3))
        ax10a.set_xlim(0, 12); ax10a.set_ylim(0, 8); ax10a.axis("off")
        ax10a.set_title("where we are", fontsize=FS_AX_TITLE - 2, fontweight="bold")
        for i, (name, _body, face, edge) in enumerate(STAGES):
            x0 = 0.5 + i * 2.85
            active = i == k
            round_box(ax10a, x0, 3.0, 2.6, 3.2,
                      C_ACTIVE if active else face,
                      edge, lw=2.6 if active else 1.6)
            ax10a.text(x0 + 1.3, 5.2 if not active else 5.35, name, ha="center",
                       fontsize=FS_ANNOT + (1 if active else 0),
                       fontweight="bold", color=edge)
            if i < 3:
                arrow(ax10a, x0 + 2.7, 4.6, x0 + 2.75, 4.6, C_DARK, lw=1.6)
        ax10a.text(6.0, 1.5, f"Stage {k + 1} / 4", ha="center",
                   fontsize=FS_ANNOT + 2, fontweight="bold", color=C_DARK)

        name, blurb, tags = STAGE_DETAIL[k]
        ax10b.set_xlim(0, 10); ax10b.set_ylim(0, 10); ax10b.axis("off")
        _, _, face, edge = STAGES[k]
        round_box(ax10b, 0.5, 1.0, 9.0, 8.0, face, edge, lw=2.2)
        ax10b.text(5.0, 8.0, name, ha="center", fontsize=FS_AX_TITLE,
                   fontweight="bold", color=edge)
        ax10b.text(5.0, 5.5, blurb, ha="center", va="center", fontsize=FS_ANNOT + 1)
        for j, tag in enumerate(tags):
            round_box(ax10b, 1.2 + j * 2.6, 1.6, 2.3, 1.2, "white", edge, lw=1.4)
            ax10b.text(2.35 + j * 2.6, 2.2, tag, ha="center", va="center",
                       fontsize=FS_ANNOT, fontweight="bold", color=edge)


    ax_sl = fig10.add_axes([0.30, 0.06, 0.40, 0.04])
    ax_p = fig10.add_axes([0.235, 0.058, 0.035, 0.028])
    ax_n = fig10.add_axes([0.730, 0.058, 0.035, 0.028])
    sl_st = Slider(ax_sl, "stage", 0, 3, valinit=0, valstep=1, color=C_FUN)
    btn_p = Button(ax_p, "\u2190", color=C_MUTE, hovercolor=C_NEW)
    btn_n = Button(ax_n, "\u2192", color=C_MUTE, hovercolor=C_GOAL)
    btn_p.label.set_fontsize(8)
    btn_n.label.set_fontsize(8)


    def _set_st(v):
        v = int(np.clip(v, 0, 3))
        if int(sl_st.val) == v:
            draw_stage(v); fig10.canvas.draw_idle()
        else:
            sl_st.set_val(v)


    def _u10(_=None):
        draw_stage(sl_st.val); fig10.canvas.draw_idle()


    def _prev(_=None):
        _set_st(int(sl_st.val) - 1)


    def _next(_=None):
        _set_st(int(sl_st.val) + 1)


    def _key(event):
        if event.key in ("right", " ", "n", "enter"):
            _next()
        elif event.key in ("left", "p", "backspace"):
            _prev()


    sl_st.on_changed(_u10)
    btn_p.on_clicked(_prev)
    btn_n.on_clicked(_next)
    fig10.canvas.mpl_connect("key_press_event", _key)
    _u10()
    print("Step 11: stage seeker.")
    return fig10


# -------------------------------------------------------
# FIGURE 11: Green test not sufficient
# -------------------------------------------------------
def build_step_12():
    fig11, ax11 = new_figure()
    fig11.suptitle(
        "Step 12: A green test is necessary -- not sufficient",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax11)

    risks = [
        (0.4, 4.3, "Distribution shift", "Field x differs from\nlab x."),
        (4.25, 4.3, "Label shift / abuse", "Users game the system;\ndefinitions drift."),
        (8.1, 4.3, "Silent failures", "Rare harms hide under\na fine average."),
        (0.4, 0.7, "Latency / cost", "A correct model that\nis too slow still fails."),
        (4.25, 0.7, "Monitoring gaps", "No one watches;\ndrift goes unnoticed."),
        (8.1, 0.7, "Ownership", "Who rolls back?\nWho pays the cost?"),
    ]
    for x0, y0, title, body in risks:
        round_box(ax11, x0, y0, 3.5, 3.1, C_WARN, C_ACCENT, lw=1.8)
        ax11.text(x0 + 1.75, y0 + 2.35, title, ha="center",
                  fontsize=FS_ANNOT + 1, fontweight="bold", color=C_ACCENT)
        ax11.text(x0 + 1.75, y0 + 1.15, body, ha="center", va="center",
                  fontsize=FS_ANNOT)
    return fig11


# -------------------------------------------------------
# FIGURE 12: Monitor / retrain loop
# -------------------------------------------------------
def build_step_13():
    fig12, ax12 = new_figure()
    fig12.suptitle(
        "Step 13: After deploy -- monitor, learn, sometimes retrain",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax12)

    loop = [
        (1.0, 5.2, "Deploy"),
        (4.5, 5.2, "Monitor"),
        (8.0, 5.2, "Detect drift"),
        (8.0, 2.0, "Collect new labels"),
        (4.5, 2.0, "Retrain / validate / test"),
        (1.0, 2.0, "Redeploy"),
    ]
    for i, (x0, y0, title) in enumerate(loop):
        face = C_WARN if i == 0 else (C_GOAL if i in (1, 5) else C_NEW)
        edge = C_ACCENT if i == 0 else (C_OK if i in (1, 5) else C_TRAIN)
        round_box(ax12, x0, y0, 2.8, 1.7, face, edge, lw=2.0)
        ax12.text(x0 + 1.4, y0 + 0.85, title, ha="center", va="center",
                  fontsize=FS_ANNOT, fontweight="bold", color=edge)

    arrow(ax12, 3.9, 6.05, 4.4, 6.05, C_DARK, lw=2.0)
    arrow(ax12, 7.4, 6.05, 7.9, 6.05, C_DARK, lw=2.0)
    arrow(ax12, 9.4, 5.1, 9.4, 3.8, C_DARK, lw=2.0)
    arrow(ax12, 7.9, 2.85, 7.4, 2.85, C_DARK, lw=2.0)
    arrow(ax12, 4.4, 2.85, 3.9, 2.85, C_DARK, lw=2.0)
    arrow(ax12, 2.4, 3.8, 2.4, 5.1, C_DARK, lw=2.0)

    ax12.text(
        6.0, 0.55,
        "Deploy is not the end of learning -- it is learning under higher stakes.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )
    return fig12


# -------------------------------------------------------
# FIGURE 13: Takeaway map (conceptual labels only)
# -------------------------------------------------------
def build_step_14():
    fig13, ax13 = new_figure()
    fig13.suptitle(
        "Step 14: Map -- discipline before courage",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax13)

    round_box(ax13, 0.5, 5.5, 11.0, 2.0, C_DARK, C_DARK, lw=0)
    ax13.text(
        6.0, 6.5,
        "Courage is deploying.\nDiscipline is earning the right to deploy.",
        ha="center", va="center", fontsize=FS_ANNOT + 2,
        fontweight="bold", color="#f9e79f",
    )

    map_items = [
        (0.5, 2.8, "The loop", "how to train"),
        (3.4, 2.8, "Hold out", "why generalize"),
        (6.3, 2.8, "Three piles", "fit / choose / trust"),
        (9.2, 2.8, "Deploy", "serve + monitor"),
    ]
    for x0, y0, title, body in map_items:
        round_box(ax13, x0, y0, 2.6, 2.3, C_GOAL, C_OK, lw=2.0)
        ax13.text(x0 + 1.3, y0 + 1.7, title, ha="center",
                  fontsize=FS_ANNOT + 1, fontweight="bold", color=C_OK)
        ax13.text(x0 + 1.3, y0 + 0.85, body, ha="center", va="center", fontsize=FS_ANNOT)

    round_box(ax13, 0.5, 0.4, 11.0, 2.0, C_MUTE, "0.45", lw=1.6)
    ax13.text(
        6.0, 1.4,
        "Train to fit. Validate to choose. Test to trust. Deploy to serve -- then watch.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )
    return fig13


# -------------------------------------------------------
# FIGURE 14: Sampling noise and distribution shift among sets
# -------------------------------------------------------
def build_step_15():
    fig14, ax14 = new_figure()
    fig14.suptitle(
        "Step 15: Train, validation, test, and deploy sets suffer\n"
        "sampling noise and distribution shift",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax14)
    # Room below the default [0, 8] frame so the deploy box can sit 0.5 inch lower
    ax14.set_ylim(-0.7, 8.0)

    # Three overlapping circles (train large; validate & test same smaller size)
    # Coordinates in the blank_slide frame (0..12, 0..8)
    c_train = (5.2, 4.15)
    c_val = (7.15, 4.55)
    c_test = (6.35, 2.85)
    r_train = 2.35
    r_small = 1.55
    alpha_c = 0.38

    ax14.add_patch(Circle(
        c_train, r_train,
        facecolor=C_TRAIN, edgecolor=C_TRAIN, linewidth=2.2, alpha=alpha_c, zorder=2,
    ))
    ax14.add_patch(Circle(
        c_val, r_small,
        facecolor=C_FUN, edgecolor=C_FUN, linewidth=2.2, alpha=alpha_c, zorder=3,
    ))
    ax14.add_patch(Circle(
        c_test, r_small,
        facecolor=C_ACCENT, edgecolor=C_ACCENT, linewidth=2.2, alpha=alpha_c, zorder=3,
    ))

    # Outside annotations (not inside the circle cluster)
    ax14.annotate(
        "Train set",
        xy=(c_train[0] - 0.85 * r_train, c_train[1] + 0.55 * r_train),
        xytext=(1.1, 6.6),
        fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_TRAIN,
        arrowprops=dict(arrowstyle="->", color=C_TRAIN, lw=2.0),
        ha="left", va="center",
    )
    ax14.annotate(
        "Validate set",
        xy=(c_val[0] + 0.55 * r_small, c_val[1] + 0.55 * r_small),
        xytext=(9.3, 6.5),
        fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_FUN,
        arrowprops=dict(arrowstyle="->", color=C_FUN, lw=2.0),
        ha="left", va="center",
    )
    ax14.annotate(
        "Test set",
        xy=(c_test[0] + 0.35 * r_small, c_test[1] - 0.65 * r_small),
        xytext=(9.3, 1.85),
        fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_ACCENT,
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2.0),
        ha="left", va="center",
    )

    # Pull deploy callout down by ~0.5 inch (~0.55 data units on this slide)
    round_box(ax14, 0.6, -0.20, 7.4, 1.35, C_WARN, C_ACCENT, lw=2.0)
    ax14.text(
        4.3, 0.47,
        "Deploy set will make it more complex.",
        ha="center", va="center", fontsize=FS_ANNOT + 2,
        fontweight="bold", color="#641e16",
    )

    ax14.text(
        6.0, 7.35,
        "Overlaps: shared structure  |  Non-overlaps: sampling noise & shift",
        ha="center", va="center", fontsize=FS_TICK + 1, color=C_DARK,
    )
    return fig14


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
])

print("\n" + "=" * 60)
print("TRAIN / VALIDATE / TEST / DEPLOY")
print("=" * 60)
print("""
Arc:
  TRAIN     -> fit a candidate rule on observed cases
  VALIDATE  -> choose among candidates without contaminating the claim
  TEST      -> trust once (honest passport stamp)
  DEPLOY    -> serve on unobserved field cases; monitor afterward

Why three piles?
  Train-only never measures unobserved quality.
  One shared holdout used for both choose and trust becomes contaminated.

Signal of overfitting:
  train loss keeps falling; validation loss rises.

Slogan:
  Train to fit. Validate to choose. Test to trust. Deploy to serve -- then watch.

A green test is necessary, not sufficient.
Deploy raises the stakes; feedback may force retrain.
""")
