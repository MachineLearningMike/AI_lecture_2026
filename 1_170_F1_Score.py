"""
Binary classification metrics from a confusion matrix.

Sketch:
  1. Why counts matter -- accuracy alone can lie
  2. Confusion matrix -- TP, TN, FP, FN
  3. Rates -- TPR, TNR, FPR, FNR
  4. Confusion Table and scores -- live drag mosaic + rates
  5. Recall and Precision -- two ways to be careful
  6. Drag the decision threshold -- watch Precision / Recall move
  7. F-score -- harmonic mean of Precision and Recall
  8. Imbalance demo -- accuracy looks fine; F1 does not
  9. Takeaway
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
    run_slide_deck,
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
    eps = 1e-12
    tpr = tp / max(tp + fn, eps)          # recall / sensitivity
    tnr = tn / max(tn + fp, eps)          # specificity
    fpr = fp / max(fp + tn, eps)
    fnr = fn / max(fn + tp, eps)
    precision = tp / max(tp + fp, eps)
    recall = tpr
    if precision + recall <= eps:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    acc = (tp + tn) / max(tp + tn + fp + fn, eps)
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
def build_step_1():
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
    return fig0


# -------------------------------------------------------
# FIGURE 2: Confusion matrix
# -------------------------------------------------------
def build_step_2():
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
    return fig1


# -------------------------------------------------------
# FIGURE 3: Rates TPR TNR FPR FNR
# -------------------------------------------------------
def build_step_3():
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
    return fig2


# -------------------------------------------------------
# FIGURE 4: Interactive area mosaic (drag boundaries)
# -------------------------------------------------------
def build_step_4():
    """Interactive confusion-table mosaic."""
    _fw_m, _fh_m = screen_figsize()
    _left_margin, _right_margin = 0.06, 0.98
    _usable_w = _fw_m * (_right_margin - _left_margin)
    _left_in = _usable_w * 1.2 / (1.2 + 1.0)
    _right_in = _usable_w * 1.0 / (1.2 + 1.0)
    _left_in += 1.5 + 1.0 + 1.0   # confusion table width bumps
    _right_in += 1.0 + 1.0 + 0.5 + 1.0  # right panel width bumps (+1.0 now)
    _fh_m_tall = _fh_m + 0.8 + 1.0 + 1.0 + 0.5  # table height bumps + room to shift left panel up
    _fw_m_wide = (_left_in + _right_in) / (_right_margin - _left_margin)
    _LEFT_PANEL_UP_IN = 0.5 + 0.5  # prior 0.5 + this request 0.5; applied after maximize
    fig_mos = plt.figure(figsize=(_fw_m_wide, _fh_m_tall), dpi=FIGURE_DPI)
    fig_mos.suptitle(
        "Step 4: Confusion Table and scores",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    gs_mos = fig_mos.add_gridspec(
        1, 2,
        width_ratios=[_left_in, _right_in],
        wspace=0.28,
    )
    # Left: top spacer so column titles sit below the figure title, then mosaic.
    gs_left = gs_mos[0, 0].subgridspec(2, 1, height_ratios=[0.22, 1.0], hspace=0.08)
    ax_mos_hdr = fig_mos.add_subplot(gs_left[0, 0])
    ax_mos = fig_mos.add_subplot(gs_left[1, 0])
    ax_mos_hdr.set_xlim(0, 1)
    ax_mos_hdr.set_ylim(0, 1)
    ax_mos_hdr.axis("off")
    gs_right = gs_mos[0, 1].subgridspec(2, 1, height_ratios=[0.75, 1.55], hspace=0.22)
    ax_mos_bars = fig_mos.add_subplot(gs_right[0, 0])
    ax_mos_form = fig_mos.add_subplot(gs_right[1, 0])
    fig_mos.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.86)

    # State in area fractions of the unit square.
    mos = {
        "pos": 0.40,   # width of Actual Positive column
        "tpr": 0.75,   # top fraction of col1 = TP / (TP+FN)
        "fpr": 0.20,   # top fraction of col2 = FP / (FP+TN)
    }
    _MIN_SPLIT = 0.06
    _drag = {"mode": None}  # 'v', 'h1', 'h2'


    def _mos_areas(state):
        pos = float(state["pos"])
        tpr = float(state["tpr"])
        fpr = float(state["fpr"])
        tp = pos * tpr
        fn = pos * (1.0 - tpr)
        fp = (1.0 - pos) * fpr
        tn = (1.0 - pos) * (1.0 - fpr)
        return tp, tn, fp, fn


    def _mos_metrics(state):
        tp, tn, fp, fn = _mos_areas(state)
        return rates_from_counts(tp, tn, fp, fn), (tp, tn, fp, fn)


    ax_mos.set_xlim(0, 1)
    ax_mos.set_ylim(0, 1)
    ax_mos.set_aspect("equal")
    ax_mos.set_xticks([])
    ax_mos.set_yticks([])
    for spine in ax_mos.spines.values():
        spine.set_linewidth(1.8)

    # Four area patches + center labels + three drag handles
    pat_tp = Rectangle((0, 0), 1, 1, facecolor="#a9dfbf", edgecolor="0.25", lw=1.5)
    pat_fn = Rectangle((0, 0), 1, 1, facecolor="#fad7a0", edgecolor="0.25", lw=1.5)
    pat_fp = Rectangle((0, 0), 1, 1, facecolor="#f5b7b1", edgecolor="0.25", lw=1.5)
    pat_tn = Rectangle((0, 0), 1, 1, facecolor="#d5f5e3", edgecolor="0.25", lw=1.5)
    for p in (pat_fn, pat_tn, pat_tp, pat_fp):
        ax_mos.add_patch(p)

    txt_tp = ax_mos.text(0.5, 0.5, "", ha="center", va="center",
                         fontsize=FS_ANNOT, fontweight="bold")
    txt_fn = ax_mos.text(0.5, 0.5, "", ha="center", va="center",
                         fontsize=FS_ANNOT, fontweight="bold")
    txt_fp = ax_mos.text(0.5, 0.5, "", ha="center", va="center",
                         fontsize=FS_ANNOT, fontweight="bold")
    txt_tn = ax_mos.text(0.5, 0.5, "", ha="center", va="center",
                         fontsize=FS_ANNOT, fontweight="bold")
    # Full names in gray, placed to the right of TP/FP/FN/TN.
    _name_kw = dict(
        ha="left", va="center", fontsize=max(FS_ANNOT - 1, 8),
        color="0.45", clip_on=True,
    )
    txt_tp_name = ax_mos.text(0.5, 0.5, "True Positive", **_name_kw)
    txt_fn_name = ax_mos.text(0.5, 0.5, "False Negative", **_name_kw)
    txt_fp_name = ax_mos.text(0.5, 0.5, "False Positive", **_name_kw)
    txt_tn_name = ax_mos.text(0.5, 0.5, "True Negative", **_name_kw)

    handle_v, = ax_mos.plot([mos["pos"], mos["pos"]], [0, 1], color=C_DARK, lw=3.0,
                            solid_capstyle="round", zorder=5)
    handle_h1, = ax_mos.plot([0, mos["pos"]], [1.0 - mos["tpr"], 1.0 - mos["tpr"]],
                             color=C_DARK, lw=3.0, solid_capstyle="round", zorder=5)
    handle_h2, = ax_mos.plot([mos["pos"], 1], [1.0 - mos["fpr"], 1.0 - mos["fpr"]],
                             color=C_DARK, lw=3.0, solid_capstyle="round", zorder=5)
    _mos_handles = {"v": handle_v, "h1": handle_h1, "h2": handle_h2}
    _hover_labels = {
        "v": "Ready: drag class boundary (Actual + / Actual -)",
        "h1": "Ready: drag TP / FN split",
        "h2": "Ready: drag FP / TN split",
    }
    hover_badge = ax_mos.text(
        0.02, 0.98, "",
        transform=ax_mos.transAxes, ha="left", va="top", fontsize=FS_ANNOT,
        fontweight="bold", color=C_ACCENT, zorder=6,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fdebd0",
                  edgecolor=C_ACCENT, alpha=0.95),
    )
    hover_badge.set_visible(False)

    ax_mos_hdr.text(0.20, 0.35, "Actual Positive", ha="center", va="center",
                    fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax_mos_hdr.text(0.72, 0.35, "Actual Negative", ha="center", va="center",
                    fontsize=FS_ANNOT, fontweight="bold", color="#85929e")
    # Bottom hint is part of the left panel; keep it in axes coords so it moves with ax_mos.
    mos_hint = ax_mos.text(
        0.5, -0.10,
        "Hover a bold boundary until it highlights, then drag",
        ha="center", va="top", fontsize=FS_ANNOT,
        transform=ax_mos.transAxes,
    )

    # Right bars
    mos_bar_names = [
        "Actual\nPositive", "Actual\nNegative",
        "TPR", "FNR", "TNR", "FPR",
        "Recall", "Precision", "F1",
    ]
    mos_bar_colors = [
        C_OK, "#85929e",
        C_REC, C_BIN, C_OK, C_ACCENT,
        C_REC, C_PREC, "#1a5276",
    ]
    mos_bars = ax_mos_bars.bar(
        np.arange(len(mos_bar_names)), np.zeros(len(mos_bar_names)),
        color=mos_bar_colors, edgecolor="white", width=0.72,
    )
    ax_mos_bars.set_xticks(np.arange(len(mos_bar_names)))
    ax_mos_bars.set_xticklabels(mos_bar_names, fontsize=max(FS_TICK - 2, 7))
    ax_mos_bars.set_ylim(0, 1.18)
    ax_mos_bars.set_ylabel("value", fontsize=FS_AXIS)
    ax_mos_bars.set_title("Live rates from areas", fontsize=FS_AX_TITLE - 1)
    ax_mos_bars.grid(True, axis="y", alpha=0.3)
    mos_bar_labs = [
        ax_mos_bars.text(i, 0.05, "", ha="center", va="bottom",
                         fontsize=max(FS_ANNOT - 2, 7), fontweight="bold")
        for i in range(len(mos_bar_names))
    ]

    ax_mos_form.set_xlim(0, 10)
    ax_mos_form.set_ylim(0, 10)
    ax_mos_form.axis("off")
    round_box(ax_mos_form, 0.2, 0.3, 9.6, 9.4, C_MUTE, "0.45", lw=1.6)
    # One text artist per formula; vertical gap between formulas = 0.1 inch.
    _N_FORM_LINES = 9
    mos_form_lines = [
        ax_mos_form.text(
            5.0, 5.0, "", ha="center", va="center", fontsize=FS_ANNOT,
        )
        for _ in range(_N_FORM_LINES)
    ]
    _form_gap = {"step": None}  # data-units between formula baselines


    def _formula_baseline_step():
        """Baseline spacing = formula line height + 0.1 inch gap."""
        if _form_gap["step"] is not None:
            return _form_gap["step"]
        pos = ax_mos_form.get_position()
        h_in = max(fig_mos.get_figheight() * pos.height, 1e-6)
        gap_data = 0.1 / h_in * 10.0          # 0.1 inch -> data y in [0,10]
        line_data = (FS_ANNOT / 72.0) * 1.45 / h_in * 10.0
        _form_gap["step"] = gap_data + line_data
        return _form_gap["step"]


    def _place_formula_lines(lines):
        step = _formula_baseline_step()
        total = (_N_FORM_LINES - 1) * step
        y_top = 5.0 + 0.5 * total
        for i, (artist, text) in enumerate(zip(mos_form_lines, lines)):
            artist.set_text(text)
            artist.set_position((5.0, y_top - i * step))


    def _redraw_mosaic(_=None):
        pos = mos["pos"]
        tpr = mos["tpr"]
        fpr = mos["fpr"]
        # Column 1: Actual Positive
        # FN bottom [0, 1-tpr], TP top [1-tpr, 1]
        y_split1 = 1.0 - tpr
        pat_fn.set_xy((0.0, 0.0))
        pat_fn.set_width(pos)
        pat_fn.set_height(y_split1)
        pat_tp.set_xy((0.0, y_split1))
        pat_tp.set_width(pos)
        pat_tp.set_height(tpr)
        # Column 2: Actual Negative
        y_split2 = 1.0 - fpr
        pat_tn.set_xy((pos, 0.0))
        pat_tn.set_width(1.0 - pos)
        pat_tn.set_height(y_split2)
        pat_fp.set_xy((pos, y_split2))
        pat_fp.set_width(1.0 - pos)
        pat_fp.set_height(fpr)

        r, (tp, tn, fp, fn) = _mos_metrics(mos)
        # Abbreviation + area centered; full name in gray to the right of the abbrev.
        cx_pos = 0.5 * pos
        cx_neg = pos + 0.5 * (1.0 - pos)
        cy_tp = y_split1 + 0.5 * tpr
        cy_fn = 0.5 * y_split1
        cy_fp = y_split2 + 0.5 * fpr
        cy_tn = 0.5 * y_split2
        # Nudge names right of the centered "TP"/"FN"/... on the upper text line.
        name_dx = 0.055
        name_dy = 0.055

        txt_tp.set_position((cx_pos, cy_tp))
        txt_fn.set_position((cx_pos, cy_fn))
        txt_fp.set_position((cx_neg, cy_fp))
        txt_tn.set_position((cx_neg, cy_tn))
        txt_tp.set_text(f"TP\n{tp:.2f}")
        txt_fn.set_text(f"FN\n{fn:.2f}")
        txt_fp.set_text(f"FP\n{fp:.2f}")
        txt_tn.set_text(f"TN\n{tn:.2f}")
        txt_tp_name.set_position((cx_pos + name_dx, cy_tp + name_dy))
        txt_fn_name.set_position((cx_pos + name_dx, cy_fn + name_dy))
        txt_fp_name.set_position((cx_neg + name_dx, cy_fp + name_dy))
        txt_tn_name.set_position((cx_neg + name_dx, cy_tn + name_dy))

        handle_v.set_data([pos, pos], [0, 1])
        handle_h1.set_data([0, pos], [y_split1, y_split1])
        handle_h2.set_data([pos, 1], [y_split2, y_split2])

        values = [
            pos, 1.0 - pos,
            r["TPR"], r["FNR"], r["TNR"], r["FPR"],
            r["Recall"], r["Precision"], r["F1"],
        ]
        for bar, lab, val in zip(mos_bars, mos_bar_labs, values):
            bar.set_height(val)
            lab.set_y(val + 0.03)
            lab.set_text(f"{val:.2f}")

        _place_formula_lines([
            f"Actual Positive = True Positive + False Negative = {pos:.2f}",
            f"Actual Negative = False Positive + True Negative = {1.0 - pos:.2f}",
            f"TPR = True Positive / Actual Positive = {r['TPR']:.2f}",
            f"TNR = True Negative / Actual Negative = {r['TNR']:.2f}",
            f"FPR = False Positive / Actual Negative = {r['FPR']:.2f}",
            f"FNR = False Negative / Actual Positive = {r['FNR']:.2f}",
            f"Recall = True Positive / Actual Positive = {r['Recall']:.2f}",
            f"Precision = True Positive / (TP+FP = Positive-predicted) = {r['Precision']:.2f}",
            f"F1 = 2 * Precision * Recall / (Precision + Recall) = {r['F1']:.2f}",
        ])
        fig_mos.canvas.draw_idle()


    def _hit_mode(x, y):
        """Which boundary is nearest to the click (in axes data coords)."""
        pos, tpr, fpr = mos["pos"], mos["tpr"], mos["fpr"]
        y1 = 1.0 - tpr
        y2 = 1.0 - fpr
        # Thresholds in data units (unit square).
        tol = 0.04
        candidates = []
        candidates.append(("v", abs(x - pos)))
        if 0.0 <= x <= pos:
            candidates.append(("h1", abs(y - y1)))
        if pos <= x <= 1.0:
            candidates.append(("h2", abs(y - y2)))
        mode, dist = min(candidates, key=lambda t: t[1])
        return mode if dist <= tol else None


    def _set_mos_cursor(kind):
        """kind: None, 'ew' (vertical boundary), 'ns' (horizontal boundary)."""
        try:
            from matplotlib.backend_bases import cursors
            mapping = {
                None: cursors.POINTER,
                "ew": cursors.RESIZE_HORIZONTAL,
                "ns": cursors.RESIZE_VERTICAL,
            }
            fig_mos.canvas.set_cursor(mapping.get(kind, cursors.POINTER))
        except Exception:
            pass


    def _style_mos_handles(active=None):
        for key, handle in _mos_handles.items():
            if key == active:
                handle.set_color(C_ACCENT)
                handle.set_linewidth(5.8)
                handle.set_alpha(1.0)
            else:
                handle.set_color(C_DARK)
                handle.set_linewidth(3.0)
                handle.set_alpha(0.95)
        if active is None:
            hover_badge.set_visible(False)
        else:
            hover_badge.set_text(_hover_labels[active])
            hover_badge.set_visible(True)


    def _on_press(event):
        if event.inaxes != ax_mos or event.xdata is None or event.ydata is None:
            return
        mode = _hit_mode(event.xdata, event.ydata)
        _drag["mode"] = mode
        if mode is not None:
            _style_mos_handles(mode)
            fig_mos.canvas.draw_idle()


    def _on_release(event):
        _drag["mode"] = None
        # Re-evaluate hover under the cursor after release.
        if event.inaxes == ax_mos and event.xdata is not None and event.ydata is not None:
            mode = _hit_mode(event.xdata, event.ydata)
            _style_mos_handles(mode)
            _set_mos_cursor(
                None if mode is None else ("ew" if mode == "v" else "ns")
            )
        else:
            _style_mos_handles(None)
            _set_mos_cursor(None)
        fig_mos.canvas.draw_idle()


    def _on_motion(event):
        # Dragging a picked boundary.
        if _drag["mode"] is not None:
            if event.inaxes != ax_mos or event.xdata is None or event.ydata is None:
                return
            x = float(np.clip(event.xdata, _MIN_SPLIT, 1.0 - _MIN_SPLIT))
            y = float(np.clip(event.ydata, _MIN_SPLIT, 1.0 - _MIN_SPLIT))
            mode = _drag["mode"]
            if mode == "v":
                mos["pos"] = x
            elif mode == "h1":
                # top fraction tpr = 1 - y_split
                mos["tpr"] = float(np.clip(1.0 - y, _MIN_SPLIT, 1.0 - _MIN_SPLIT))
            elif mode == "h2":
                mos["fpr"] = float(np.clip(1.0 - y, _MIN_SPLIT, 1.0 - _MIN_SPLIT))
            _style_mos_handles(mode)
            _redraw_mosaic()
            return

        # Hover acknowledgement: highlight handle + cursor + badge.
        if event.inaxes != ax_mos or event.xdata is None or event.ydata is None:
            _style_mos_handles(None)
            _set_mos_cursor(None)
            fig_mos.canvas.draw_idle()
            return
        mode = _hit_mode(event.xdata, event.ydata)
        _style_mos_handles(mode)
        _set_mos_cursor(None if mode is None else ("ew" if mode == "v" else "ns"))
        fig_mos.canvas.draw_idle()


    fig_mos.canvas.mpl_connect("button_press_event", _on_press)
    fig_mos.canvas.mpl_connect("button_release_event", _on_release)
    fig_mos.canvas.mpl_connect("motion_notify_event", _on_motion)
    print("Step 4: hover boundaries to arm drag; drag to update areas and bars.")
    # Keep the widened/tall figsize (do not reset via show_maximized).
    fig_mos.set_dpi(FIGURE_DPI)
    fig_mos.set_size_inches(_fw_m_wide, _fh_m_tall, forward=True)
    fig_mos.canvas.draw_idle()
    _form_gap["step"] = None
    _redraw_mosaic()
    # Apply AFTER maximize so it is not undone. Moves header + table + bottom hint.
    _dy_left_up = _LEFT_PANEL_UP_IN / max(fig_mos.get_figheight(), 1e-6)
    for _ax in (ax_mos_hdr, ax_mos):
        _p = _ax.get_position()
        _ax.set_position([_p.x0, _p.y0 + _dy_left_up, _p.width, _p.height])
    fig_mos.canvas.draw_idle()
    fig_mos._mos_state = mos
    return fig_mos


# -------------------------------------------------------
# FIGURE 5: Recall and Precision
# -------------------------------------------------------
def build_step_5():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        "Step 5: Recall and Precision -- two questions about positives",
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
    return fig3


# -------------------------------------------------------
# FIGURE 6: Threshold slider -- everything live
# -------------------------------------------------------
def build_step_6():
    """Threshold slider with live confusion metrics."""
    _CM_GROW_IN = 3.0  # inches added to the confusion-table panel
    fig4 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig4.suptitle(
        "Step 6: How to interpret model scores? Learnable?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    gs4 = fig4.add_gridspec(
        2, 2,
        height_ratios=[1.05, 1.15],
        width_ratios=[1.35, 1.0],
        hspace=0.38, wspace=0.28,
    )
    ax4_hist = fig4.add_subplot(gs4[0, :])
    ax4_cm = fig4.add_subplot(gs4[1, 0])
    ax4_met = fig4.add_subplot(gs4[1, 1])  # bar graph stays in original right slot
    fig4.subplots_adjust(bottom=0.16, top=0.88, left=0.07, right=0.98)

    # Top: score histograms + live threshold + predicted-positive shade
    bins = np.linspace(scores.min() - 0.3, scores.max() + 0.3, 28)
    ax4_hist.hist(scores_neg, bins=bins, alpha=0.55, color="#85929e",
                  label="true 0", edgecolor="white")
    ax4_hist.hist(scores_pos, bins=bins, alpha=0.65, color=C_OK,
                  label="true 1", edgecolor="white")
    thr_line = ax4_hist.axvline(THR0, color=C_ACCENT, lw=2.6, label="threshold")
    ax4_hist.set_xlabel("model score", fontsize=FS_AXIS)
    ax4_hist.set_ylabel("count", fontsize=FS_AXIS)
    ax4_hist.set_title("Higher score => more likely positive", fontsize=FS_AX_TITLE - 2)
    ax4_hist.grid(True, axis="y", alpha=0.3)
    x_left_hist, x_right_hist = float(scores.min()) - 0.3, float(scores.max()) + 0.3
    ax4_hist.set_xlim(x_left_hist, x_right_hist)
    # Placeholder shade; geometry filled in _update_thr after ylim is known.
    pred_span = Rectangle(
        (THR0, 0.0), max(x_right_hist - THR0, 0.0), 1.0,
        facecolor=C_ACCENT, edgecolor="none", alpha=0.12,
        label="predict 1 (score >= thr)", zorder=0,
    )
    ax4_hist.add_patch(pred_span)
    ax4_hist.legend(fontsize=FS_LEGEND - 1, loc="upper left")

    # Bottom-left: live confusion table (same area layout as Step 4)
    ax4_cm.set_xlim(0, 1)
    ax4_cm.set_ylim(0, 1)
    ax4_cm.set_aspect("equal")
    ax4_cm.set_xticks([])
    ax4_cm.set_yticks([])
    for spine in ax4_cm.spines.values():
        spine.set_linewidth(1.8)

    s6_pat_tp = Rectangle((0, 0), 1, 1, facecolor="#a9dfbf", edgecolor="0.25", lw=1.5)
    s6_pat_fn = Rectangle((0, 0), 1, 1, facecolor="#fad7a0", edgecolor="0.25", lw=1.5)
    s6_pat_fp = Rectangle((0, 0), 1, 1, facecolor="#f5b7b1", edgecolor="0.25", lw=1.5)
    s6_pat_tn = Rectangle((0, 0), 1, 1, facecolor="#d5f5e3", edgecolor="0.25", lw=1.5)
    for _p in (s6_pat_fn, s6_pat_tn, s6_pat_tp, s6_pat_fp):
        ax4_cm.add_patch(_p)

    s6_txt_tp = ax4_cm.text(0.5, 0.5, "", ha="center", va="center",
                            fontsize=FS_ANNOT, fontweight="bold")
    s6_txt_fn = ax4_cm.text(0.5, 0.5, "", ha="center", va="center",
                            fontsize=FS_ANNOT, fontweight="bold")
    s6_txt_fp = ax4_cm.text(0.5, 0.5, "", ha="center", va="center",
                            fontsize=FS_ANNOT, fontweight="bold")
    s6_txt_tn = ax4_cm.text(0.5, 0.5, "", ha="center", va="center",
                            fontsize=FS_ANNOT, fontweight="bold")
    _s6_name_kw = dict(
        ha="left", va="center", fontsize=max(FS_ANNOT - 1, 8),
        color="0.45", clip_on=True,
    )
    s6_txt_tp_name = ax4_cm.text(0.5, 0.5, "True Positive", **_s6_name_kw)
    s6_txt_fn_name = ax4_cm.text(0.5, 0.5, "False Negative", **_s6_name_kw)
    s6_txt_fp_name = ax4_cm.text(0.5, 0.5, "False Positive", **_s6_name_kw)
    s6_txt_tn_name = ax4_cm.text(0.5, 0.5, "True Negative", **_s6_name_kw)

    ax4_cm.text(0.20, 1.06, "Actual Positive", ha="center", va="bottom",
                fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
                transform=ax4_cm.transAxes)
    ax4_cm.text(0.72, 1.06, "Actual Negative", ha="center", va="bottom",
                fontsize=FS_ANNOT, fontweight="bold", color="#85929e",
                transform=ax4_cm.transAxes)

    # Bottom-right: live bars for all rates
    metric_names = ["TPR\nRecall", "TNR", "FPR", "FNR", "Precision", "F1", "Accuracy"]
    metric_keys = ["TPR", "TNR", "FPR", "FNR", "Precision", "F1", "Accuracy"]
    metric_colors = [C_REC, C_OK, C_ACCENT, C_BIN, C_PREC, "#1a5276", "#85929e"]
    bars_met = ax4_met.bar(
        np.arange(len(metric_names)), np.zeros(len(metric_names)),
        color=metric_colors, edgecolor="white", width=0.72,
    )
    ax4_met.set_xticks(np.arange(len(metric_names)))
    ax4_met.set_xticklabels(metric_names, fontsize=FS_TICK - 1)
    ax4_met.set_ylim(0, 1.18)
    ax4_met.set_ylabel("rate", fontsize=FS_AXIS)
    ax4_met.set_title("All rates live", fontsize=FS_AX_TITLE - 1)
    ax4_met.grid(True, axis="y", alpha=0.3)
    bar_labels = [
        ax4_met.text(i, 0.05, "", ha="center", va="bottom",
                     fontsize=FS_ANNOT - 1, fontweight="bold")
        for i in range(len(metric_names))
    ]
    # ~5 cm left of panel center (≈0.45 in axes x on this metrics panel).
    ax4_met.text(
        0.05, -0.22,
        "I, Grok, spent five hours wrestling with this slide layout—completely in vain.",
        transform=ax4_met.transAxes, ha="center", va="top", fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fdebd0",
                  edgecolor=C_BIN, alpha=0.95),
    )

    ax_sl = fig4.add_axes([0.18, 0.04, 0.64, 0.035])
    slider_thr = Slider(
        ax_sl, r"threshold", float(scores.min()), float(scores.max()),
        valinit=THR0, valstep=0.05,
    )


    def _update_thr(_=None):
        thr = float(slider_thr.val)
        thr_line.set_xdata([thr, thr])
        # Live predicted-positive region (score >= threshold)
        y0, y1 = ax4_hist.get_ylim()
        pred_span.set_x(thr)
        pred_span.set_width(max(x_right_hist - thr, 0.0))
        pred_span.set_y(y0)
        pred_span.set_height(max(y1 - y0, 0.0))
        yp = predict_at(thr)
        tp, tn, fp, fn = confusion_counts(y_true, yp)
        r = rates_from_counts(tp, tn, fp, fn)
        n_tot = max(tp + tn + fp + fn, 1)
        # Area fractions for Step-4-style table (columns = actual class).
        pos = (tp + fn) / n_tot
        neg = 1.0 - pos
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        y_split1 = 1.0 - tpr
        y_split2 = 1.0 - fpr

        s6_pat_fn.set_xy((0.0, 0.0))
        s6_pat_fn.set_width(pos)
        s6_pat_fn.set_height(y_split1)
        s6_pat_tp.set_xy((0.0, y_split1))
        s6_pat_tp.set_width(pos)
        s6_pat_tp.set_height(tpr)
        s6_pat_tn.set_xy((pos, 0.0))
        s6_pat_tn.set_width(neg)
        s6_pat_tn.set_height(y_split2)
        s6_pat_fp.set_xy((pos, y_split2))
        s6_pat_fp.set_width(neg)
        s6_pat_fp.set_height(fpr)

        cx_pos = 0.5 * pos
        cx_neg = pos + 0.5 * neg
        cy_tp = y_split1 + 0.5 * tpr
        cy_fn = 0.5 * y_split1
        cy_fp = y_split2 + 0.5 * fpr
        cy_tn = 0.5 * y_split2
        name_dx, name_dy = 0.055, 0.055
        # Show count and area fraction, matching Step 4 cell notation style.
        s6_txt_tp.set_position((cx_pos, cy_tp))
        s6_txt_fn.set_position((cx_pos, cy_fn))
        s6_txt_fp.set_position((cx_neg, cy_fp))
        s6_txt_tn.set_position((cx_neg, cy_tn))
        s6_txt_tp.set_text(f"TP\n{tp} ({tp / n_tot:.2f})")
        s6_txt_fn.set_text(f"FN\n{fn} ({fn / n_tot:.2f})")
        s6_txt_fp.set_text(f"FP\n{fp} ({fp / n_tot:.2f})")
        s6_txt_tn.set_text(f"TN\n{tn} ({tn / n_tot:.2f})")
        s6_txt_tp_name.set_position((cx_pos + name_dx, cy_tp + name_dy))
        s6_txt_fn_name.set_position((cx_pos + name_dx, cy_fn + name_dy))
        s6_txt_fp_name.set_position((cx_neg + name_dx, cy_fp + name_dy))
        s6_txt_tn_name.set_position((cx_neg + name_dx, cy_tn + name_dy))

        values = [r[k] for k in metric_keys]
        for bar, lab, val in zip(bars_met, bar_labels, values):
            bar.set_height(val)
            lab.set_y(val + 0.03)
            lab.set_text(f"{val:.2f}")

        fig4.canvas.draw_idle()


    slider_thr.on_changed(_update_thr)
    _update_thr()
    print("Step 6: drag threshold; matrix + all rates update live.")


    def _grow_step6_cm_plus_3():
        """Grow figure; confusion table = (side0+3) wide, 0.5 in shorter; shift left 2 in.
        Bar graph stays on the right and moves up 1 in."""
        fig4.set_dpi(FIGURE_DPI)
        fig4.set_size_inches(screen_figsize(), forward=True)
        fig4.subplots_adjust(bottom=0.16, top=0.88, left=0.07, right=0.98)
        fig4.canvas.draw()
        fw0, fh0 = fig4.get_size_inches()
        pcm = ax4_cm.get_position()
        pmet = ax4_met.get_position()
        phist = ax4_hist.get_position()
        side0 = min(pcm.width * fw0, pcm.height * fh0)
        side1 = side0 + _CM_GROW_IN
        cm_w = max(side1 - 1.0 - 1.0, 1.0)     # prior -1 in, then all-panels -1 in
        cm_h = max(side1 - 0.5 - 0.5 - 1.0, 1.0)
        # Apply -2 in left first (may hit the floor), THEN +0.4 in right for all panels.
        left_in = max(pcm.x0 * fw0 - 2.0, 0.05) + 0.4
        bot_in = max(pcm.y0 * fh0 - 0.5, 0.05)   # table + bars down 0.5 in
        met_w = max(pmet.width * fw0 - 1.0, 1.5)  # all panels -1 in
        met_h = max(pmet.height * fh0 - 1.0, 1.5)
        met_bot = bot_in + 1.0 - 0.5          # was +1.0 up; then -0.5; shares table downshift
        hist_h = max(phist.height * fh0 - 1.0, 1.2)
        gap = 0.28
        top_in = max((1.0 - phist.y0 - phist.height) * fh0, 0.40)
        fw1 = left_in + cm_w + gap + met_w + 0.20
        fh1 = max(bot_in + cm_h, met_bot + met_h) + gap + hist_h + top_in
        fig4.set_size_inches(fw1, fh1, forward=True)
        fig4.canvas.draw()
        # Fill the requested box (not letterboxed by equal-aspect).
        ax4_cm.set_aspect("auto")
        ax4_cm.set_position([
            left_in / fw1,
            bot_in / fh1,
            cm_w / fw1,
            cm_h / fh1,
        ])
        # Bar graph on the right, raised by 1 inch.
        ax4_met.set_position([
            (left_in + cm_w + gap) / fw1,
            met_bot / fh1,
            met_w / fw1,
            met_h / fh1,
        ])
        ax4_hist.set_position([
            left_in / fw1,
            (max(bot_in + cm_h, met_bot + met_h) + gap) / fh1,
            (cm_w + gap + met_w) / fw1,
            hist_h / fh1,
        ])
        fig4.canvas.draw()
        p2 = ax4_cm.get_position()
        fw2, fh2 = fig4.get_size_inches()
        print(
            f"Step 6: cm x0={p2.x0 * fw2:.2f} in (includes +0.4 right), "
            f"size {p2.width * fw2:.2f}x{p2.height * fh2:.2f}; "
            f"bars {met_w:.2f}x{met_h:.2f}, hist_h {hist_h:.2f}"
        )


    # Grow first (real inches), then maximize for full screen.
    _grow_step6_cm_plus_3()
    # Re-assert size after maximize (some backends reset layout).
    _grow_step6_cm_plus_3()
    ax_sl.set_navigate(False)
    ax_sl.set_zorder(20)
    fig4._sliders = (slider_thr,)
    return fig4


# -------------------------------------------------------
# FIGURE 7: F-score
# -------------------------------------------------------
def build_step_7():
    fig5, (ax5a, ax5b) = new_figure(1, 2)
    fig5.suptitle(
        r"Step 7: F-score -- harmonic mean of Precision and Recall",
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
    return fig5


# -------------------------------------------------------
# FIGURE 8: Imbalance -- accuracy lies, F1 complains
# -------------------------------------------------------
def build_step_8():
    fig6, (ax6a, ax6b) = new_figure(1, 2)
    fig6.suptitle(
        "Step 8: Class imbalance -- Accuracy can look great while F1 collapses",
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
    return fig6


# -------------------------------------------------------
# FIGURE 9: Takeaway
# -------------------------------------------------------
def build_step_9():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        "Step 9: F-score family in one breath",
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
        "035 (classification), 190 (0-1 error), 194 (BCE trains soft scores\n"
        "that you later threshold into this hard 2x2 table).",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )
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
    build_step_9,
])

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
  4. Drag area mosaic (live bars)
  5. Recall and Precision
  6. Threshold slider tradeoff
  7. F1 harmonic mean (+ F_beta)
  8. Imbalance: Accuracy vs F1
  9. Takeaway

, ,
""")
