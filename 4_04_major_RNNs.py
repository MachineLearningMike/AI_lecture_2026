"""
4_04_major_RNNs.py

Major RNN architectures -- intuition visualizations + core formulas.

  Vanilla RNN -> LSTM -> GRU
  (+ biRNN / seq2seq notes)

Sketch:
  1. Why sequences need memory
  2. Vanilla RNN recurrence (formula + unroll)
  3. Hidden-state intuition (live slider)
  4. Vanishing / exploding gradients
  5. LSTM gates (intuition diagram)
  6. GRU (simpler gated memory)
  7. Bidirectional + encoder-decoder (brief)
  8. Side-by-side comparison
  9. Takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
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

np.random.seed(430)


def new_figure(*args, **kwargs):
    kwargs.setdefault("figsize", screen_figsize())
    kwargs.setdefault("dpi", FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0, zorder=2):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder,
    ))


def arrow(ax, x0, y0, x1, y1, color=None, lw=2.0):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color or C_DARK, lw=lw),
        zorder=3,
    )


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_WARN = "#f8d7da"
C_OLD = "#fce4d6"
C_LOOP = "#2874a6"
C_PURPLE = "#d7bde2"
C_TEAL = "#a3e4d7"
C_PINK = "#f5b7b1"
C_GATE = "#5dade2"
C_CELL = "#f9e79f"


# -------------------------------------------------------
# Step 1: Why sequences need memory
# -------------------------------------------------------
def build_step_1():
    fig, axes = new_figure(1, 2)
    # Each of the two panels ~3 cm wider.
    w_in, h_in = fig.get_size_inches()
    fig.set_size_inches(w_in + 2 * (3.0 / 2.54), h_in, forward=True)
    fig.suptitle(
        "Step 1: Sequences need memory -- not one shot",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Feedforward habit", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    for i, tok in enumerate(tokens):
        x = 0.7 + i * 1.85
        round_box(ax, x, 4.6, 1.6, 1.4, C_MUTE, "0.55", lw=1.5)
        ax.text(x + 0.8, 5.3, tok, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold")
        round_box(ax, x, 2.2, 1.6, 1.4, C_NEW, C_LOOP, lw=1.5)
        ax.text(x + 0.8, 2.9, "f(x)", ha="center", va="center",
                fontsize=FS_TICK, color=C_DARK)
        arrow(ax, x + 0.8, 4.5, x + 0.8, 3.7)
    round_box(ax, 0.5, 0.4, 11.0, 1.3, C_WARN, C_ACCENT, lw=1.8)
    ax.text(
        6.0, 1.05,
        "Each token judged alone -- no carry of \"who did what\".",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_ACCENT,
        fontweight="bold",
    )

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("Recurrent habit", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    y_tok, y_h = 6.2, 3.3
    for i, tok in enumerate(tokens):
        x = 0.7 + i * 1.85
        round_box(ax, x, y_tok, 1.6, 1.1, C_MUTE, "0.55", lw=1.4)
        ax.text(x + 0.8, y_tok + 0.55, tok, ha="center", va="center",
                fontsize=FS_TICK, fontweight="bold")
        round_box(ax, x, y_h, 1.6, 1.4, C_GOAL, C_OK, lw=1.8)
        ax.text(x + 0.8, y_h + 0.7, f"h{i+1}", ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_OK)
        arrow(ax, x + 0.8, y_tok, x + 0.8, y_h + 1.45)
        if i > 0:
            arrow(ax, x - 0.15, y_h + 0.7, x, y_h + 0.7, color=C_ACCENT, lw=2.2)
    ax.text(6.0, 1.7, "Memory rides along as hidden state h_t",
            ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    round_box(ax, 0.5, 0.35, 11.0, 1.0, C_MUTE, "0.45", lw=1.4)
    ax.text(
        6.0, 0.85,
        "RNN idea: update a state, then reuse it on the next step.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 2: Vanilla RNN formula + unroll
# -------------------------------------------------------
def build_step_2():
    w_in, h_in = screen_figsize()
    extra = 3.0 / 2.54  # left panel +3 cm
    fig, axes = plt.subplots(
        1, 2,
        figsize=(w_in + extra, h_in),
        dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [w_in / 2 + extra, w_in / 2]},
    )
    fig.suptitle(
        "Step 2: Vanilla RNN -- one shared cell, reused in time",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Formula", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.4, 11.2, 2.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        r"$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$"
        + "\n"
        r"$y_t = W_{hy} h_t + b_y$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.7, 11.2, 2.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.85,
        "Same W for every t  (parameter sharing in time).\n"
        "h_{t-1} is the compressed past.\n"
        "tanh keeps values in (-1, 1).",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.4, 0.4, 11.2, 1.9, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "Intuition: h is a whiteboard the network\n"
        "erases and rewrites each step.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Unrolled in time", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    xs = [1.3, 3.8, 6.3, 8.8]
    labels = [r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$"]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        round_box(ax, x - 0.7, 7.4, 1.4, 1.1, C_MUTE, "0.5", lw=1.4)
        ax.text(x, 7.95, lab, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold")
        round_box(ax, x - 0.85, 4.2, 1.7, 1.8, C_TEAL, C_DARK, lw=2.0)
        ax.text(x, 5.1, "RNN\ncell", ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold")
        arrow(ax, x, 7.3, x, 6.1)
        round_box(ax, x - 0.7, 1.5, 1.4, 1.1, C_GOAL, C_OK, lw=1.6)
        ax.text(x, 2.05, rf"$h_{i+1}$", ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
        arrow(ax, x, 4.1, x, 2.7)
        if i > 0:
            arrow(ax, xs[i - 1] + 0.9, 5.1, x - 0.9, 5.1, color=C_ACCENT, lw=2.3)
    ax.text(5.0, 0.55, "One cell, copied across time -- arrows carry memory",
            ha="center", fontsize=FS_TICK + 1, color="#333333")

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 3: Hidden-state intuition with slider
# -------------------------------------------------------
def build_step_3():
    fig, axes = new_figure(1, 2)
    fig.subplots_adjust(bottom=0.18)
    fig.suptitle(
        "Step 3: Intuition -- how much of the past survives?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # Toy 1D RNN: h_t = tanh(a * x_t + b * h_{t-1})
    T = 24
    t = np.arange(T)
    x = np.zeros(T)
    x[3] = 1.0
    x[10] = -0.8
    x[17] = 0.6

    ax0 = axes[0]
    ax0.set_title("Input pulses", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax0.stem(t, x, linefmt="C0-", markerfmt="C0o", basefmt="k-")
    ax0.set_xlabel("time t", fontsize=FS_AXIS)
    ax0.set_ylabel(r"$x_t$", fontsize=FS_AXIS)
    ax0.set_ylim(-1.3, 1.3)
    ax0.grid(True, alpha=0.3)
    ax0.tick_params(labelsize=FS_TICK)

    ax1 = axes[1]
    ax1.set_title("Hidden state response", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    b0 = 0.85
    h = np.zeros(T)
    for i in range(1, T):
        h[i] = np.tanh(0.9 * x[i] + b0 * h[i - 1])
    (line_h,) = ax1.plot(t, h, color=C_OK, lw=2.8, label=r"$h_t$")
    ax1.axhline(0, color="0.5", lw=1)
    ax1.set_xlabel("time t", fontsize=FS_AXIS)
    ax1.set_ylabel(r"$h_t$", fontsize=FS_AXIS)
    ax1.set_ylim(-1.2, 1.2)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=FS_TICK)
    note = ax1.text(
        0.02, 0.98,
        rf"recurrence $b$ = {b0:.2f}  (drag slider)",
        transform=ax1.transAxes, ha="left", va="top",
        fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )

    ax_sl = fig.add_axes([0.20, 0.06, 0.60, 0.04])
    sl = Slider(ax_sl, r"memory $b$", 0.0, 0.98, valinit=b0, valstep=0.02, color=C_LOOP)

    def update(_=None):
        b = float(sl.val)
        hh = np.zeros(T)
        for i in range(1, T):
            hh[i] = np.tanh(0.9 * x[i] + b * hh[i - 1])
        line_h.set_ydata(hh)
        note.set_text(rf"recurrence $b$ = {b:.2f}  (drag slider)")
        fig.canvas.draw_idle()

    sl.on_changed(update)
    fig._retained_widgets = getattr(fig, "_retained_widgets", [])
    fig._retained_widgets.append(sl)
    return fig


# -------------------------------------------------------
# Step 4: Vanishing / exploding
# -------------------------------------------------------
def build_step_4():
    w_in, h_in = screen_figsize()
    extra = 3.0 / 2.54  # left panel +3 cm
    fig, axes = plt.subplots(
        1, 2,
        figsize=(w_in + extra, h_in),
        dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [w_in / 2 + extra, w_in / 2]},
    )
    fig.suptitle(
        "Step 4: Long credit assignment is hard for vanilla RNNs",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Gradient through time", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.5, 11.2, 2.1, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.55,
        r"$\dfrac{\partial h_t}{\partial h_{t-k}}"
        r"\approx \prod_{j=0}^{k-1} W_{hh}^{\top}"
        r"\mathrm{diag}(\tanh'(z_{t-j}))$",
        ha="center", va="center", fontsize=FS_ANNOT - 1,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.9, 5.4, 2.2, C_WARN, C_ACCENT, lw=2.0)
    ax.text(
        3.1, 4.0,
        "If factors < 1:\nvanishing\n(past is forgotten)",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color=C_ACCENT,
    )
    round_box(ax, 6.2, 2.9, 5.4, 2.2, C_OLD, "#d35400", lw=2.0)
    ax.text(
        8.9, 4.0,
        "If factors > 1:\nexploding\n(training blows up)",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#d35400",
    )
    round_box(ax, 0.4, 0.4, 11.2, 2.1, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.45,
        "Intuition: multiplying many soft \"dimmers\" either kills the signal\n"
        "or amplifies noise -- hard to learn long-range deps.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    ax.set_title("Product of gains over depth", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    k = np.arange(1, 31)
    for gain, color, label in [
        (0.9, C_ACCENT, "gain 0.9 → vanish"),
        (1.0, C_LOOP, "gain 1.0 → stable-ish"),
        (1.1, "#d35400", "gain 1.1 → explode"),
    ]:
        ax.semilogy(k, gain ** k, lw=2.6, color=color, label=label)
    ax.set_xlabel("steps back k", fontsize=FS_AXIS)
    ax.set_ylabel(r"$|\mathrm{gain}|^k$", fontsize=FS_AXIS)
    ax.legend(fontsize=FS_LEGEND, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 5: LSTM intuition
# -------------------------------------------------------
def build_step_5():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 5: LSTM -- a conveyor belt for memory",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    w_in, h_in = fig.get_size_inches()
    # Approximate panel size after 1x2 split (for cm -> data conversion).
    left_h_in = max(h_in * 0.75, 1.0)
    right_w_in = max(w_in * 0.45, 1.0)
    dy_left_3cm = (3.0 / 2.54) / left_h_in * 8.0   # blank_slide y: 0..8
    dx_right_3cm = (3.0 / 2.54) / right_w_in * 10.0  # right panel x: 0..10

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Gate formulas (compact)", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.35, 0.4, 11.3, 7.2, C_MUTE, "0.45", lw=1.5)
    # Move formula text down by ~3 cm inside the left textbox.
    ax.text(
        6.0, 6.7 - dy_left_3cm,
        r"$f_t=\sigma(W_f[x_t,h_{t-1}]+b_f)$   forget"
        + "\n"
        r"$i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)$   input"
        + "\n"
        r"$\tilde{c}_t=\tanh(W_c[x_t,h_{t-1}]+b_c)$   candidate"
        + "\n"
        r"$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$"
        + "\n"
        r"$o_t=\sigma(W_o[x_t,h_{t-1}]+b_o)$   output"
        + "\n"
        r"$h_t = o_t \odot \tanh(c_t)$",
        ha="center", va="center", fontsize=FS_TICK + 2,
        fontweight="bold", color=C_DARK,
        linespacing=1.55,
    )

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Intuition diagram", fontsize=FS_AX_TITLE - 2, fontweight="bold")

    # cell highway
    round_box(ax, 0.6, 6.6, 8.8, 1.5, C_CELL, "#b7950b", lw=2.2)
    ax.text(5.0, 7.35, r"cell state $c_t$  (conveyor belt)", ha="center",
            va="center", fontsize=FS_ANNOT, fontweight="bold")
    arrow(ax, 0.3, 7.35, 0.55, 7.35, color="#b7950b", lw=2.5)
    arrow(ax, 9.5, 7.35, 9.85, 7.35, color="#b7950b", lw=2.5)
    ax.text(0.15, 7.9, r"$c_{t-1}$", fontsize=FS_TICK, color="#7d6608")
    ax.text(9.2, 7.9, r"$c_t$", fontsize=FS_TICK, color="#7d6608")

    # gates
    gates = [
        (1.2, "forget\n× f", C_WARN, C_ACCENT),
        (4.0, "input\n+ i·ĉ", C_GOAL, C_OK),
        (6.8, "output\n× o", C_NEW, C_LOOP),
    ]
    for x, label, face, edge in gates:
        round_box(ax, x, 4.5, 2.0, 1.5, face, edge, lw=2.0)
        ax.text(x + 1.0, 5.25, label, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=edge)
        arrow(ax, x + 1.0, 6.05, x + 1.0, 6.55, color=edge, lw=2.0)

    # Bottom textbox: wider by ~3 cm, keep centered.
    box_w = 5.0 + dx_right_3cm
    box_x = 5.0 - box_w / 2
    round_box(ax, box_x, 1.5, box_w, 1.8, C_PURPLE, "#7d3c98", lw=2.0)
    ax.text(
        5.0, 2.4,
        r"hidden $h_t$ = what we expose now",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#7d3c98",
    )
    arrow(ax, 7.8, 4.5, 6.5, 3.4, color=C_LOOP, lw=2.0)

    ax.text(
        5.0, 0.55,
        "Forget what is stale · write what is new · reveal only what is needed",
        ha="center", fontsize=FS_TICK + 1, color="#333333",
    )

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 6: GRU
# -------------------------------------------------------
def build_step_6():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 6: GRU -- fewer gates, same spirit",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    w_in, _ = fig.get_size_inches()
    right_w_in = max(w_in * 0.45, 1.0)
    dx_bottom_5cm = (5.0 / 2.54) / right_w_in * 10.0  # right panel x: 0..10

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Formulas", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 4.8, 11.2, 2.8, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.2,
        r"$z_t=\sigma(W_z[x_t,h_{t-1}])$   update"
        + "\n"
        r"$r_t=\sigma(W_r[x_t,h_{t-1}])$   reset"
        + "\n"
        r"$\tilde{h}_t=\tanh(W_h[x_t, r_t\odot h_{t-1}])$"
        + "\n"
        r"$h_t=(1-z_t)\odot h_{t-1} + z_t\odot \tilde{h}_t$",
        ha="center", va="center", fontsize=FS_TICK + 2,
        fontweight="bold", color="#f9e79f", linespacing=1.5,
    )
    round_box(ax, 0.4, 0.5, 11.2, 3.8, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 2.4,
        "No separate cell state -- h is the memory.\n\n"
        "z mixes old h and new proposal.\n"
        "r controls how much old h enters the proposal.\n\n"
        "Often similar accuracy to LSTM, fewer parameters.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )

    ax = axes[1]
    # Bottom textbox wider by ~5 cm; expand x-range if needed so it is not clipped.
    bottom_w = 8.8 + dx_bottom_5cm
    bottom_x = 5.0 - bottom_w / 2
    ax.set_xlim(min(0.0, bottom_x - 0.15), max(10.0, bottom_x + bottom_w + 0.15))
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Intuition: soft interpolate", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.8, 6.5, 3.4, 2.0, C_OLD, "#d35400", lw=2.0)
    ax.text(2.5, 7.5, "$h_{t-1}$\nold memory", ha="center", va="center",
            fontsize=FS_ANNOT, fontweight="bold", color="#d35400")
    round_box(ax, 5.8, 6.5, 3.4, 2.0, C_GOAL, C_OK, lw=2.0)
    ax.text(7.5, 7.5, r"$\tilde{h}_t$" + "\nnew proposal", ha="center", va="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    arrow(ax, 2.5, 6.4, 4.2, 4.3, color="#d35400", lw=2.2)
    arrow(ax, 7.5, 6.4, 5.8, 4.3, color=C_OK, lw=2.2)
    round_box(ax, 2.8, 2.5, 4.4, 2.0, C_PURPLE, "#7d3c98", lw=2.2)
    ax.text(
        5.0, 3.5,
        r"$h_t = (1-z)\,h_{t-1} + z\,\tilde{h}_t$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#7d3c98",
    )
    ax.text(
        5.0, 1.2,
        "z ≈ 0 keep past · z ≈ 1 overwrite",
        ha="center", fontsize=FS_ANNOT, color="#333333",
    )
    round_box(ax, bottom_x, 0.3, bottom_w, 0.7, C_MUTE, "0.5", lw=1.3)
    ax.text(5.0, 0.65, r"reset $r$ decides whether the proposal may peek at the past",
            ha="center", va="center", fontsize=FS_TICK + 1)

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 7: biRNN + seq2seq
# -------------------------------------------------------
def build_step_7():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 7: After the cell -- directions and encoder/decoder",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Bidirectional RNN", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    xs = [1.5, 3.5, 5.5, 7.5, 9.5]
    for i, x in enumerate(xs):
        round_box(ax, x - 0.55, 5.8, 1.1, 1.0, C_MUTE, "0.5", lw=1.3)
        ax.text(x, 6.3, rf"$x_{i+1}$", ha="center", va="center",
                fontsize=FS_TICK, fontweight="bold")
        round_box(ax, x - 0.55, 3.7, 1.1, 1.1, C_NEW, C_LOOP, lw=1.6)
        ax.text(x, 4.25, "→", ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_LOOP)
        round_box(ax, x - 0.55, 1.7, 1.1, 1.1, C_OLD, "#d35400", lw=1.6)
        ax.text(x, 2.25, "←", ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color="#d35400")
        arrow(ax, x, 5.7, x, 4.9)
        arrow(ax, x, 3.6, x, 2.9)
        if i > 0:
            arrow(ax, xs[i - 1] + 0.6, 4.25, x - 0.6, 4.25, color=C_LOOP, lw=1.8)
            arrow(ax, x - 0.6, 2.25, xs[i - 1] + 0.6, 2.25, color="#d35400", lw=1.8)
    round_box(ax, 1.0, 0.35, 10.0, 0.95, C_MUTE, "0.45", lw=1.3)
    ax.text(
        6.0, 0.82,
        "Each position sees left context and right context.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("Encoder → Decoder (seq2seq)", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.5, 4.6, 4.8, 2.8, C_NEW, C_LOOP, lw=2.0)
    ax.text(2.9, 6.7, "Encoder RNN", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(2.9, 5.5, "read source\ninto a state\n(or states)", ha="center",
            va="center", fontsize=FS_TICK + 1)
    round_box(ax, 6.7, 4.6, 4.8, 2.8, C_GOAL, C_OK, lw=2.0)
    ax.text(9.1, 6.7, "Decoder RNN", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(9.1, 5.5, "emit target\nstep by step\nfrom memory", ha="center",
            va="center", fontsize=FS_TICK + 1)
    arrow(ax, 5.4, 6.0, 6.6, 6.0, color=C_ACCENT, lw=2.8)
    ax.text(6.0, 6.55, "state", ha="center", fontsize=FS_TICK,
            fontweight="bold", color=C_ACCENT)
    round_box(ax, 0.5, 0.5, 11.0, 3.5, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 2.25,
        "Classic neural MT / summarization pattern:\n"
        "compress the source, then generate the target.\n\n"
        "Attention (later idea) lets the decoder peek at\n"
        "encoder states instead of one bottleneck vector.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 8: Comparison
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: Side-by-side -- what each design contributed",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    header = ["Model", "Signature idea", "Watch for"]
    rows = [
        ["Vanilla RNN", "shared cell + h_t memory", "vanish / explode"],
        ["LSTM", "cell belt + f/i/o gates", "more params"],
        ["GRU", "update/reset on h only", "no separate c"],
        ["BiRNN", "forward + backward pass", "needs full sequence"],
        ["Seq2seq", "encode then decode", "bottleneck without attn"],
    ]
    xs = [0.4, 2.6, 7.2]
    ws = [2.1, 4.5, 4.4]
    y = 6.7
    for x, w, htext in zip(xs, ws, header):
        round_box(ax, x, y, w, 0.85, C_DARK, C_DARK, lw=0)
        ax.text(x + w / 2, y + 0.42, htext, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color="#f9e79f")
    y = 5.7
    for i, row in enumerate(rows):
        face = C_NEW if i % 2 == 0 else C_MUTE
        for x, w, text in zip(xs, ws, row):
            round_box(ax, x, y, w, 0.9, face, "0.55", lw=1.2)
            ax.text(x + w / 2, y + 0.45, text, ha="center", va="center",
                    fontsize=FS_TICK + 1, color="#333333",
                    fontweight="bold" if x < 1 else "normal")
        y -= 1.0

    round_box(ax, 0.4, 0.35, 11.2, 0.95, C_GOAL, C_OK, lw=1.8)
    ax.text(
        6.0, 0.82,
        "Progression: memory → gated memory → lean gates → richer use of context.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 9: Takeaway
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Takeaway",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    points = [
        (5.9, "RNN core", r"$h_t = f(x_t, h_{t-1})$ with shared weights"),
        (4.5, "Pain point", "long products of gains vanish or explode"),
        (3.1, "LSTM / GRU", "gates learn what to keep, write, reveal"),
        (1.7, "Systems", "bi-direction and encoder/decoder reuse the cell"),
    ]
    for y, left, right in points:
        round_box(ax, 0.5, y - 0.5, 11.0, 1.2, C_NEW, C_LOOP, lw=1.8)
        ax.text(2.5, y + 0.1, left, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(7.6, y + 0.1, right, ha="center", va="center",
                fontsize=FS_ANNOT, color="#333333")

    round_box(ax, 0.5, 0.25, 11.0, 1.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 0.75,
        "An RNN family = how you store and update a running story over time.",
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
])


print("\n" + "=" * 60)
print("MAJOR RNNs")
print("=" * 60)
print(
    """
Vanilla RNN: shared recurrent cell, hidden state memory.
LSTM: conveyor-belt cell state + forget/input/output gates.
GRU: lean gated update on h alone.
Then: bidirectional context and encoder/decoder reuse.

Interactive: Step 3 -- drag recurrence b to see memory linger or die.
"""
)
print("=" * 60)
