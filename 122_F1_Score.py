"""
122_F_Score.py

Binary classification metrics from a confusion matrix.

Sketch:
  1. Why counts matter -- accuracy alone can lie
  2. Confusion matrix -- TP, TN, FP, FN
  3. Rates -- TPR, TNR, FPR, FNR
  4. Recall and Precision -- two ways to be careful
  5. Drag the decision threshold -- watch Precision / Recall move
  6. F-score -- harmonic mean of Precision and Recall
  7. Imbalance demo -- accuracy looks fine; F1 does not
  8. Takeaway

Companion to 035_regression_classification.py, 190_loss_functions.py,
194_Loss_Binary_Cross_Entropy.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
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

np.random.seed(122)
RNG = np.random.default_rng(122)


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


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def rates_from_counts(tp, tn, fp, fn):
    tpr = tp / max(tp + fn, 1)          # recall / sensitivity
    tnr = tn / max(tn + fp, 1)          # specificity
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    precision = tp / max(tp + fp, 1)
    recall = tpr
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return dict(
        TPR=tpr, TNR=tnr, FPR=fpr, FNR=fnr,
        Precision=precision, Recall=recall, F1=f1, Accuracy=acc,
    )


def draw_confusion(ax, tp, tn, fp, fn, title="Confusion matrix"):
    """Draw a labeled 2x2 confusion matrix in axes data coords 0..10."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=FS_AX_TITLE - 1)

    cells = [
        (1.3, 5.2, tn, "TN", "True Negative", "#d5f5e3"),
        (5.3, 5.2, fp, "FP", "False Positive", "#f5b7b1"),
        (1.3, 1.5, fn, "FN", "False Negative", "#fad7a0"),
        (5.3, 1.5, tp, "TP", "True Positive", "#a9dfbf"),
    ]
    for x, y, val, short, long_name, face in cells:
        ax.add_patch(FancyBboxPatch(
            (x, y), 3.2, 2.9,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=face, edgecolor="0.3", linewidth=1.8,
        ))
        ax.text(x + 1.6, y + 2.15, short, ha="center", va="center",
                fontsize=FS_ANNOT + 2, fontweight="bold")
        ax.text(x + 1.6, y + 1.35, long_name, ha="center", va="center",
                fontsize=FS_ANNOT - 1, color="0.25")
        ax.text(x + 1.6, y + 0.55, str(val), ha="center", va="center",
                fontsize=FS_AX_TITLE - 1, fontweight="bold")

    ax.text(2.9, 8.55, r"pred $= 0$", ha="center",
            fontsize=FS_ANNOT + 1, fontweight="bold")
    ax.text(6.9, 8.55, r"pred $= 1$", ha="center",
            fontsize=FS_ANNOT + 1, fontweight="bold")
    ax.text(0.45, 6.65, r"true $= 0$", ha="center", va="center",
            rotation=90, fontsize=FS_ANNOT + 1, fontweight="bold")
    ax.text(0.45, 2.95, r"true $= 1$", ha="center", va="center",
            rotation=90, fontsize=FS_ANNOT + 1, fontweight="bold")


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_BIN = "#d68910"
C_PREC = "#2874a6"
C_REC = "#8e44ad"


# -------------------------------------------------------
# Shared toy binary problem for demos
# -------------------------------------------------------
# Class-conditional scores (higher => more likely positive).
n0, n1 = 60, 40
scores_neg = RNG.normal(-0.4, 0.9, size=n0)
scores_pos = RNG.normal(1.1, 0.9, size=n1)
scores = np.concatenate([scores_neg, scores_pos])
y_true = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])
THR0 = 0.50


def predict_at(threshold):
    return (scores >= threshold).astype(int)


y_hat0 = predict_at(THR0)
TP0, TN0, FP0, FN0 = confusion_counts(y_true, y_hat0)
R0 = rates_from_counts(TP0, TN0, FP0, FN0)


# -------------------------------------------------------
# FIGURE 1: Why these metrics
# -------------------------------------------------------
fig0, ax0 = new_figure()
fig0.suptitle(
    "Step 1: Accuracy is not enough -- we need a language for mistakes",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax0)

round_box(ax0, 0.5, 5.3, 11.0, 2.3, C_DARK, C_DARK, lw=0)
ax0.text(
    6.0, 6.45,
    "Binary classifier: every example is labeled 0 or 1,\n"
    "and the model predicts 0 or 1.",
    ha="center", va="center", fontsize=FS_ANNOT + 2,
    fontweight="bold", color="#f9e79f",
)

cards = [
    (0.5, 2.6, C_NEW, "#2874a6", "Two truths",
     "Actual positive\nActual negative"),
    (4.25, 2.6, C_GOAL, C_OK, "Two predictions",
     "Predicted positive\nPredicted negative"),
    (8.0, 2.6, "#fdebd0", C_BIN, "Four outcomes",
     "TP  TN\nFP  FN"),
]
for x0, y0, face, edge, title, body in cards:
    round_box(ax0, x0, y0, 3.5, 2.4, face, edge, lw=2.0)
    ax0.text(x0 + 1.75, y0 + 1.75, title, ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=edge)
    ax0.text(x0 + 1.75, y0 + 0.85, body, ha="center", va="center",
             fontsize=FS_ANNOT + 1)

round_box(ax0, 0.5, 0.35, 11.0, 1.95, C_MUTE, "0.45", lw=1.8)
ax0.text(
    6.0, 1.3,
    "Precision, Recall, and F-score are how we summarize those four counts.\n"
    "They answer different questions about the same 2x2 table.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)
show_maximized(fig0)


# -------------------------------------------------------
# FIGURE 2: Confusion matrix
# -------------------------------------------------------
fig1, (ax1a, ax1b) = new_figure(1, 2)
fig1.suptitle(
    "Step 2: Confusion matrix -- TP, TN, FP, FN",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

draw_confusion(ax1a, TP0, TN0, FP0, FN0,
               title=f"Counts at threshold = {THR0:.2f}")

ax1b.set_xlim(0, 10)
ax1b.set_ylim(0, 10)
ax1b.axis("off")
ax1b.set_title("Read each cell out loud", fontsize=FS_AX_TITLE - 1)
rows = [
    (C_GOAL, C_OK, "TP", "true=1, pred=1", "Caught a real positive."),
    (C_NEW, "#2874a6", "TN", "true=0, pred=0", "Correctly rejected a negative."),
    ("#f5b7b1", C_ACCENT, "FP", "true=0, pred=1", "False alarm (Type I)."),
    ("#fad7a0", C_BIN, "FN", "true=1, pred=0", "Missed a positive (Type II)."),
]
for i, (face, edge, name, rule, meaning) in enumerate(rows):
    y = 7.6 - i * 1.85
    round_box(ax1b, 0.5, y, 9.0, 1.6, face, edge, lw=1.8)
    ax1b.text(1.4, y + 0.8, name, ha="center", va="center",
              fontsize=FS_ANNOT + 3, fontweight="bold", color=edge)
    ax1b.text(5.4, y + 1.05, rule, ha="center", va="center",
              fontsize=FS_ANNOT + 1, fontweight="bold")
    ax1b.text(5.4, y + 0.45, meaning, ha="center", va="center",
              fontsize=FS_ANNOT)
fig1.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 3: Rates TPR TNR FPR FNR
# -------------------------------------------------------
fig2, (ax2a, ax2b) = new_figure(1, 2)
fig2.suptitle(
    "Step 3: Rates from the matrix -- TPR, TNR, FPR, FNR",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

draw_confusion(ax2a, TP0, TN0, FP0, FN0, title="Same counts")
# highlight row meanings
ax2a.text(
    5.0, 0.35,
    "TPR/FNR split the true-positive row.\n"
    "TNR/FPR split the true-negative row.",
    ha="center", va="center", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)

ax2b.set_xlim(0, 10)
ax2b.set_ylim(0, 10)
ax2b.axis("off")
ax2b.set_title("Definitions (with current numbers)", fontsize=FS_AX_TITLE - 1)

rate_cards = [
    (r"$TPR = \frac{TP}{TP+FN}$", f"= {R0['TPR']:.2f}", "True Positive Rate\n= Recall / Sensitivity"),
    (r"$TNR = \frac{TN}{TN+FP}$", f"= {R0['TNR']:.2f}", "True Negative Rate\n= Specificity"),
    (r"$FPR = \frac{FP}{FP+TN}$", f"= {R0['FPR']:.2f}", "False Positive Rate\n= 1 - TNR"),
    (r"$FNR = \frac{FN}{FN+TP}$", f"= {R0['FNR']:.2f}", "False Negative Rate\n= 1 - TPR"),
]
for i, (form, val, gloss) in enumerate(rate_cards):
    x = 0.4 + (i % 2) * 4.8
    y = 5.2 - (i // 2) * 4.4
    round_box(ax2b, x, y, 4.5, 4.0, C_MUTE if i % 2 == 0 else C_NEW, "0.45", lw=1.6)
    ax2b.text(x + 2.25, y + 3.1, form, ha="center", va="center",
              fontsize=FS_ANNOT + 1, fontweight="bold")
    ax2b.text(x + 2.25, y + 2.15, val, ha="center", va="center",
              fontsize=FS_AX_TITLE - 1, fontweight="bold", color=C_DARK)
    ax2b.text(x + 2.25, y + 1.0, gloss, ha="center", va="center",
              fontsize=FS_ANNOT)
fig2.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 4: Recall and Precision
# -------------------------------------------------------
fig3, ax3 = new_figure()
fig3.suptitle(
    "Step 4: Recall and Precision -- two questions about positives",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax3)

round_box(ax3, 0.4, 4.35, 5.5, 3.2, "#f5eef8", C_REC, lw=2.2)
ax3.text(3.15, 7.0, "Recall  (= TPR)", ha="center",
         fontsize=FS_AX_TITLE - 1, fontweight="bold", color=C_REC)
ax3.text(
    3.15, 5.7,
    r"$Recall=\frac{TP}{TP+FN}$"
    "\n\n"
    "Of all actual positives,\n"
    "what fraction did we catch?\n\n"
    "Fear: missing a real case (FN).",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)

round_box(ax3, 6.1, 4.35, 5.5, 3.2, C_NEW, C_PREC, lw=2.2)
ax3.text(8.85, 7.0, "Precision", ha="center",
         fontsize=FS_AX_TITLE - 1, fontweight="bold", color=C_PREC)
ax3.text(
    8.85, 5.7,
    r"$Precision=\frac{TP}{TP+FP}$"
    "\n\n"
    "Of all predicted positives,\n"
    "what fraction were real?\n\n"
    "Fear: crying wolf (FP).",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)

round_box(ax3, 0.4, 0.4, 11.2, 3.5, C_MUTE, "0.45", lw=1.8)
ax3.text(
    6.0, 2.15,
    "Same TP in both formulas -- different denominators.\n"
    "Recall looks down the true-positive row (TP+FN).\n"
    "Precision looks down the predicted-positive column (TP+FP).\n\n"
    f"At threshold {THR0:.2f}:  Recall = {R0['Recall']:.2f},  "
    f"Precision = {R0['Precision']:.2f}",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 5: Threshold slider -- Precision / Recall tradeoff
# -------------------------------------------------------
fig4, (ax4a, ax4b) = new_figure(1, 2)
fig4.suptitle(
    "Step 5: Drag the threshold -- Precision and Recall trade off",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
fig4.subplots_adjust(bottom=0.22, wspace=0.28)

# Left: score histograms
bins = np.linspace(scores.min() - 0.3, scores.max() + 0.3, 28)
ax4a.hist(scores_neg, bins=bins, alpha=0.55, color="#85929e",
          label="true 0", edgecolor="white")
ax4a.hist(scores_pos, bins=bins, alpha=0.65, color=C_OK,
          label="true 1", edgecolor="white")
thr_line = ax4a.axvline(THR0, color=C_ACCENT, lw=2.6, label="threshold")
ax4a.set_xlabel("model score", fontsize=FS_AXIS)
ax4a.set_ylabel("count", fontsize=FS_AXIS)
ax4a.set_title("Higher score => more likely positive", fontsize=FS_AX_TITLE - 2)
ax4a.legend(fontsize=FS_LEGEND - 1, loc="upper left")
ax4a.grid(True, axis="y", alpha=0.3)

ax4b.set_xlim(0, 10)
ax4b.set_ylim(0, 10)
ax4b.axis("off")
round_box(ax4b, 0.5, 1.2, 9.0, 7.5, C_MUTE, "0.45", lw=1.8)
hud4 = ax4b.text(5.0, 5.2, "", ha="center", va="center", fontsize=FS_ANNOT + 1)

ax_sl = fig4.add_axes([0.20, 0.07, 0.60, 0.04])
slider_thr = Slider(
    ax_sl, r"threshold", float(scores.min()), float(scores.max()),
    valinit=THR0, valstep=0.05,
)


def _update_thr(_=None):
    thr = float(slider_thr.val)
    thr_line.set_xdata([thr, thr])
    yp = predict_at(thr)
    tp, tn, fp, fn = confusion_counts(y_true, yp)
    r = rates_from_counts(tp, tn, fp, fn)
    hud4.set_text(
        f"threshold = {thr:.2f}\n\n"
        f"TP={tp}   FP={fp}\n"
        f"FN={fn}   TN={tn}\n\n"
        f"Recall    = TPR = {r['Recall']:.2f}\n"
        f"Precision         = {r['Precision']:.2f}\n"
        f"FPR               = {r['FPR']:.2f}\n"
        f"F1                = {r['F1']:.2f}\n\n"
        "Lower threshold: catch more (Recall up),\n"
        "but more false alarms (Precision down)."
    )
    fig4.canvas.draw_idle()


slider_thr.on_changed(_update_thr)
_update_thr()
print("Step 5: drag threshold; watch Precision / Recall.")
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 6: F-score
# -------------------------------------------------------
fig5, (ax5a, ax5b) = new_figure(1, 2)
fig5.suptitle(
    r"Step 6: F-score -- harmonic mean of Precision and Recall",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

# Left: F1 as function of one variable when the other is fixed
p_grid = np.linspace(0.02, 1.0, 200)
for rec_fixed, color in [(0.3, "#f5b7b1"), (0.6, C_BIN), (0.9, C_OK)]:
    f1 = 2 * p_grid * rec_fixed / (p_grid + rec_fixed)
    ax5a.plot(p_grid, f1, lw=2.5, color=color,
              label=rf"Recall$={rec_fixed:.1f}$")
ax5a.set_xlabel("Precision", fontsize=FS_AXIS)
ax5a.set_ylabel("F1", fontsize=FS_AXIS)
ax5a.set_title(r"$F_1=2\cdot\frac{Precision\cdot Recall}{Precision+Recall}$",
               fontsize=FS_AX_TITLE - 2)
ax5a.set_xlim(0, 1.02)
ax5a.set_ylim(0, 1.05)
ax5a.grid(True, alpha=0.3)
ax5a.legend(fontsize=FS_LEGEND - 1, loc="lower right")
ax5a.text(
    0.02, 0.98,
    "Harmonic mean stays low\n"
    "if either side is low.",
    transform=ax5a.transAxes, ha="left", va="top", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)

# Right: arithmetic vs harmonic intuition
ax5b.set_xlim(0, 10)
ax5b.set_ylim(0, 10)
ax5b.axis("off")
ax5b.set_title("Why harmonic, not average?", fontsize=FS_AX_TITLE - 1)
round_box(ax5b, 0.5, 5.3, 9.0, 3.8, C_NEW, "#2874a6", lw=2.0)
ax5b.text(
    5.0, 7.2,
    "Example: Precision=1.00, Recall=0.10\n\n"
    "Arithmetic mean = 0.55  (looks OK)\n"
    "F1 = 0.18               (harsh, honest)\n\n"
    "You almost never find the positives.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)
round_box(ax5b, 0.5, 0.7, 9.0, 4.2, C_GOAL, C_OK, lw=2.0)
ax5b.text(
    5.0, 2.8,
    r"General $F_\beta$ weights Recall by $\beta^2$:"
    "\n\n"
    r"$F_\beta=(1+\beta^2)\frac{Precision\cdot Recall}"
    r"{(\beta^2\cdot Precision)+Recall}$"
    "\n\n"
    r"$\beta=1$ is the usual F1 (balanced)."
    "\n"
    r"$\beta>1$ cares more about Recall.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)
fig5.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 7: Imbalance -- accuracy lies, F1 complains
# -------------------------------------------------------
fig6, (ax6a, ax6b) = new_figure(1, 2)
fig6.suptitle(
    "Step 7: Class imbalance -- Accuracy can look great while F1 collapses",
    fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
)

# Rare positive class
y_imb = np.array([0] * 95 + [1] * 5)
# Lazy classifier: always predict 0
y_lazy = np.zeros_like(y_imb)
# Decent classifier: catch 4/5 positives with 3 false alarms
y_decent = np.array([0] * 92 + [1] * 3 + [1] * 4 + [0] * 1)

tp_l, tn_l, fp_l, fn_l = confusion_counts(y_imb, y_lazy)
tp_d, tn_d, fp_d, fn_d = confusion_counts(y_imb, y_decent)
r_lazy = rates_from_counts(tp_l, tn_l, fp_l, fn_l)
r_decent = rates_from_counts(tp_d, tn_d, fp_d, fn_d)

names = ["Always predict 0", "Useful detector"]
accs = [r_lazy["Accuracy"], r_decent["Accuracy"]]
f1s = [r_lazy["F1"], r_decent["F1"]]
x = np.arange(2)
w = 0.35
ax6a.bar(x - w / 2, accs, width=w, color="#85929e", edgecolor="white", label="Accuracy")
ax6a.bar(x + w / 2, f1s, width=w, color=C_REC, edgecolor="white", label="F1")
ax6a.set_xticks(x)
ax6a.set_xticklabels(names, fontsize=FS_TICK)
ax6a.set_ylim(0, 1.15)
ax6a.set_ylabel("score", fontsize=FS_AXIS)
ax6a.set_title("95 negatives, 5 positives", fontsize=FS_AX_TITLE - 1)
ax6a.legend(fontsize=FS_LEGEND, loc="upper right")
ax6a.grid(True, axis="y", alpha=0.3)
for i, (a, f) in enumerate(zip(accs, f1s)):
    ax6a.text(i - w / 2, a + 0.03, f"{a:.2f}", ha="center", fontsize=FS_ANNOT)
    ax6a.text(i + w / 2, f + 0.03, f"{f:.2f}", ha="center", fontsize=FS_ANNOT)

ax6b.set_xlim(0, 10)
ax6b.set_ylim(0, 10)
ax6b.axis("off")
round_box(ax6b, 0.5, 1.2, 9.0, 7.5, "#fdebd0", C_BIN, lw=2.0)
ax6b.text(
    5.0, 5.2,
    "Always-negative classifier\n"
    f"Accuracy = {r_lazy['Accuracy']:.2f}   F1 = {r_lazy['F1']:.2f}\n"
    f"Recall = {r_lazy['Recall']:.2f}   Precision = {r_lazy['Precision']:.2f}\n\n"
    "Useful detector\n"
    f"Accuracy = {r_decent['Accuracy']:.2f}   F1 = {r_decent['F1']:.2f}\n"
    f"Recall = {r_decent['Recall']:.2f}   Precision = {r_decent['Precision']:.2f}\n\n"
    "Accuracy celebrates the majority class.\n"
    "F1 asks: did you find the rare positives\n"
    "without flooding false alarms?",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)
fig6.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 8: Takeaway
# -------------------------------------------------------
fig7, ax7 = new_figure()
fig7.suptitle(
    "Step 8: F-score family in one breath",
    fontsize=FS_FIGURE_TITLE, fontweight="bold",
)
blank_slide(ax7)

cards8 = [
    (0.4, 4.35, C_NEW, "#2874a6", "Matrix",
     "TP TN\nFP FN\nfour atomic outcomes"),
    (4.2, 4.35, "#f5eef8", C_REC, "Rates",
     "TPR=Recall\nTNR, FPR, FNR\nrow-normalized"),
    (8.0, 4.35, C_GOAL, C_OK, "Scores",
     "Precision\nRecall\nF1 harmonic mean"),
]
for x0, y0, face, edge, title, body in cards8:
    round_box(ax7, x0, y0, 3.5, 2.9, face, edge, lw=2.2)
    ax7.text(x0 + 1.75, y0 + 2.2, title, ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=edge)
    ax7.text(x0 + 1.75, y0 + 1.05, body, ha="center", va="center",
             fontsize=FS_ANNOT + 1)

round_box(ax7, 0.4, 0.4, 11.2, 3.5, C_MUTE, "0.45", lw=1.8)
ax7.text(
    6.0, 2.15,
    "Start from the confusion matrix. Derive rates. Then Precision / Recall / F1.\n"
    "Pick the metric that matches the cost you fear (misses vs false alarms).\n\n"
    "See also: 035 (classification), 190 (0-1 error), 194 (BCE trains soft scores\n"
    "that you later threshold into this hard 2x2 table).",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)
show_maximized(fig7)


print("\n" + "=" * 60)
print("122 F-SCORE / PRECISION / RECALL")
print("=" * 60)
print(f"""
Confusion: TP={TP0}, TN={TN0}, FP={FP0}, FN={FN0}  (demo threshold={THR0})
TPR/Recall={R0['TPR']:.3f}  TNR={R0['TNR']:.3f}  FPR={R0['FPR']:.3f}  FNR={R0['FNR']:.3f}
Precision={R0['Precision']:.3f}  F1={R0['F1']:.3f}  Accuracy={R0['Accuracy']:.3f}

Steps:
  1. Why counts beat bare accuracy
  2. TP / TN / FP / FN
  3. TPR / TNR / FPR / FNR
  4. Recall and Precision
  5. Threshold slider tradeoff
  6. F1 harmonic mean (+ F_beta)
  7. Imbalance: Accuracy vs F1
  8. Takeaway

See also: 035_regression_classification.py, 190_loss_functions.py,
          194_Loss_Binary_Cross_Entropy.py
""")
