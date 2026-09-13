"""
4_02_major_CNNs.py

Major CNN architectures -- formulas and visual block structure.

  LeNet -> AlexNet -> VGG -> Inception -> ResNet
  (+ short DenseNet / EfficientNet notes)

Sketch:
  1. Shared CNN building block (convolution formula)
  2. LeNet-5
  3. AlexNet
  4. VGG
  5. Inception (GoogLeNet)
  6. ResNet (residual learning)
  7. DenseNet / EfficientNet (brief)
  8. Side-by-side comparison
  9. Takeaway

structured stacks of local linear maps + nonlinearity + pooling.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
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

np.random.seed(420)


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


def arrow(ax, x0, y0, x1, y1):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2.0),
        zorder=3,
    )


def stack_blocks(ax, x, y0, widths_labels, colors, h=0.7, gap=0.12):
    """Draw a vertical stack of labeled blocks upward from y0. Returns top y."""
    y = y0
    for (w, label), c in zip(widths_labels, colors):
        round_box(ax, x - w / 2, y, w, h, c, C_DARK, lw=1.6)
        ax.text(x, y + h / 2, label, ha="center", va="center",
                fontsize=FS_TICK, fontweight="bold", color="#1c2833", zorder=4)
        y += h + gap
    return y


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


# -------------------------------------------------------
# Step 1: Shared conv formula
# -------------------------------------------------------
def build_step_1():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 1: Shared building block -- convolution",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Formula", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.3, 11.2, 2.3, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.45,
        r"$(K * X)_{c',i,j}"
        r"=\sum_{c,u,v} K_{c',c,u,v}\, X_{c,\,i+u,\,j+v}"
        r"+ b_{c'}$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.6, 11.2, 2.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.75,
        "Then nonlinearity (ReLU / GELU) and often pooling / stride.\n\n"
        r"Output size (same pad, stride $s$, kernel $k$):"
        + "\n"
        r"$H'=\lfloor(H-k)/s\rfloor+1$",
        ha="center", va="center", fontsize=FS_TICK + 1, color=C_DARK,
    )
    round_box(ax, 0.4, 0.4, 11.2, 1.8, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.3,
        "Every major CNN is a particular wiring of these local filters\n"
        "plus how gradients flow (plain stack vs residual vs dense).",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Local receptive field", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    # input grid
    for i in range(6):
        for j in range(6):
            ax.add_patch(Rectangle((1 + j * 0.7, 3.5 + i * 0.7), 0.65, 0.65,
                                   facecolor="#eaf2f8", edgecolor=C_DARK, lw=1))
    # kernel highlight
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((1 + (j + 1) * 0.7, 3.5 + (i + 2) * 0.7), 0.65, 0.65,
                                   facecolor="#f9e79f", edgecolor=C_ACCENT, lw=2))
    ax.text(3.1, 2.6, "input X", ha="center", fontsize=FS_TICK, fontweight="bold")
    # kernel
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((6.5 + j * 0.7, 5.5 + i * 0.7), 0.65, 0.65,
                                   facecolor=C_OLD, edgecolor=C_DARK, lw=1.5))
    ax.text(7.55, 4.7, "kernel K", ha="center", fontsize=FS_TICK, fontweight="bold")
    ax.annotate("", xy=(6.3, 6.5), xytext=(5.3, 6.5),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2))

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 2: LeNet
# -------------------------------------------------------
def build_step_2():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 2: LeNet-5 -- the prototype CNN (1998)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Idea", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.5, 11.2, 2.1, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.55,
        "Alternating convolution + pooling, then fully connected head.\n"
        "Designed for small grayscale digits (28x28 / 32x32).",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.7, 11.2, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.9,
        r"Conv $\to$ pool $\to$ Conv $\to$ pool $\to$ FC $\to$ FC"
        + "\n\n"
        "Learns local stroke detectors, then combines them.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.4, 0.4, 11.2, 1.9, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "Historical baseline: proves end-to-end learned filters beat hand features.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Block stack", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    labels = [
        (3.2, "input 32x32"),
        (4.0, "Conv 6@5x5"),
        (4.0, "Pool 2x2"),
        (4.6, "Conv 16@5x5"),
        (4.0, "Pool 2x2"),
        (4.6, "FC 120"),
        (4.6, "FC 84"),
        (3.6, "out 10"),
    ]
    colors = [C_MUTE, C_NEW, C_TEAL, C_NEW, C_TEAL, C_OLD, C_OLD, C_GOAL]
    stack_blocks(ax, 5.0, 0.6, labels, colors, h=0.85, gap=0.15)

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 3: AlexNet
# -------------------------------------------------------
def build_step_3():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 3: AlexNet -- deep + GPU + ReLU (2012)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("What changed", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    cards = [
        (0.4, 5.3, "ReLU", "faster training\nthan sigmoid/tanh"),
        (4.2, 5.3, "Dropout", "regularize huge\nFC layers"),
        (8.0, 5.3, "Scale", "ImageNet-sized\nRGB inputs"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.2, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 1.6, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 0.8, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 2.5, 11.2, 2.3, C_OLD, "#d35400", lw=2.0)
    ax.text(
        6.0, 3.65,
        "5 conv layers + 3 FC.\n"
        "Large early kernels (11x11, stride 4) then smaller ones.\n"
        "Local Response Norm (historical; BatchNorm later replaces this idea).",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )
    round_box(ax, 0.4, 0.4, 11.2, 1.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.25,
        "Formula unchanged (conv+ReLU+pool); engineering unlocked depth.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Schematic depth", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    labels = [
        (3.4, "224 RGB"),
        (4.4, "Conv 11x11 /4"),
        (3.6, "Pool"),
        (4.2, "Conv 5x5"),
        (3.6, "Pool"),
        (4.0, "Conv 3x3 x3"),
        (3.6, "Pool"),
        (4.2, "FC 4096 x2"),
        (3.4, "Softmax"),
    ]
    colors = [C_MUTE, C_NEW, C_TEAL, C_NEW, C_TEAL, C_NEW, C_TEAL, C_OLD, C_GOAL]
    stack_blocks(ax, 5.0, 0.35, labels, colors, h=0.78, gap=0.1)

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 4: VGG
# -------------------------------------------------------
def build_step_4():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 4: VGG -- stacks of small 3x3 convolutions",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Design rule", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.4, 11.2, 2.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        "Prefer many 3x3 convs over one large kernel.\n"
        "Same receptive field, more nonlinearity, fewer params.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.7, 11.2, 2.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.85,
        r"Two $3\times 3$ (stride 1) $\approx$ one $5\times 5$ field"
        + "\n"
        r"Three $3\times 3$ $\approx$ one $7\times 7$ field"
        + "\n\n"
        "Channels double after each pooling stage (64→128→256→512).",
        ha="center", va="center", fontsize=FS_TICK + 1, color=C_DARK,
    )
    round_box(ax, 0.4, 0.4, 11.2, 1.9, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "VGG-16 / VGG-19: uniform template made depth easy to scale -- and heavy.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Stage pattern", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    y = 8.3
    stages = [
        ("conv 3x3 x2, 64", C_NEW),
        ("pool", C_TEAL),
        ("conv 3x3 x2, 128", C_NEW),
        ("pool", C_TEAL),
        ("conv 3x3 x3, 256", C_NEW),
        ("pool", C_TEAL),
        ("conv 3x3 x3, 512", C_NEW),
        ("pool", C_TEAL),
        ("FC ... Softmax", C_OLD),
    ]
    for label, c in stages:
        round_box(ax, 1.5, y - 0.55, 7.0, 0.75, c, C_DARK, lw=1.5)
        ax.text(5.0, y - 0.17, label, ha="center", va="center",
                fontsize=FS_TICK, fontweight="bold")
        y -= 0.9

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 5: Inception
# -------------------------------------------------------
def build_step_5():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 5: Inception (GoogLeNet) -- multi-scale in one block",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Idea", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.5, 11.2, 2.1, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.55,
        "Run several filter sizes in parallel, then concatenate.\n"
        "Let the net choose which scale matters.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.7, 11.2, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.9,
        r"branches: $1\times 1$, $3\times 3$, $5\times 5$, pool+$1\times 1$"
        + "\n\n"
        r"$1\times 1$ convs bottleneck channels (cheap depth mixing)."
        + "\n"
        "Output = concat(branches) along channel axis.",
        ha="center", va="center", fontsize=FS_TICK + 1, color=C_DARK,
    )
    round_box(ax, 0.4, 0.4, 11.2, 1.9, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "Formula: Y = [F1(X); F3(X); F5(X); Fp(X)]  (channel concat).",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Inception module", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 4.0, 0.6, 4.0, 1.0, C_MUTE, C_DARK, lw=1.8)
    ax.text(6.0, 1.1, "input X", ha="center", va="center",
            fontsize=FS_TICK, fontweight="bold")
    branches = [
        (1.0, "1x1", C_NEW),
        (3.8, "1x1→3x3", C_TEAL),
        (6.6, "1x1→5x5", C_OLD),
        (9.4, "pool→1x1", C_PURPLE),
    ]
    for x, label, c in branches:
        arrow(ax, 6.0, 1.7, x + 1.1, 3.3)
        round_box(ax, x, 3.4, 2.2, 2.2, c, C_DARK, lw=1.8)
        ax.text(x + 1.1, 4.5, label, ha="center", va="center",
                fontsize=FS_TICK, fontweight="bold")
        arrow(ax, x + 1.1, 5.7, 6.0, 7.0)
    round_box(ax, 3.5, 7.1, 5.0, 1.3, C_GOAL, C_OK, lw=2.0)
    ax.text(6.0, 7.75, "concat channels", ha="center", va="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 6: ResNet
# -------------------------------------------------------
def build_step_6():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 6: ResNet -- learn the residual, not the whole map",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Formula", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.5, 11.2, 2.1, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.55,
        r"$y = F(x,\{W_i\}) + x$"
        + "\n"
        r"(if dims differ: $y=F(x)+W_s x$)",
        ha="center", va="center", fontsize=FS_AX_TITLE - 3,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.7, 11.2, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.9,
        "F is typically Conv–BN–ReLU–Conv–BN.\n"
        "Identity skip gives gradients a highway.\n"
        "Makes 50 / 101 / 152-layer nets trainable.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.4, 0.4, 11.2, 1.9, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "Key insight: fit residual F(x)≈0 is easier than fit full H(x) from scratch.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Residual block", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 3.0, 1.0, 4.0, 1.1, C_MUTE, C_DARK, lw=1.8)
    ax.text(5.0, 1.55, "x", ha="center", va="center",
            fontsize=FS_ANNOT, fontweight="bold")
    # main path
    arrow(ax, 5.0, 2.2, 5.0, 3.2)
    round_box(ax, 2.8, 3.3, 4.4, 2.6, C_NEW, C_LOOP, lw=2.0)
    ax.text(5.0, 5.2, "F(x)", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(5.0, 4.2, "Conv–BN–ReLU\nConv–BN", ha="center", va="center",
            fontsize=FS_TICK + 1)
    arrow(ax, 5.0, 6.0, 5.0, 7.0)
    # skip
    ax.annotate(
        "", xy=(8.2, 7.3), xytext=(8.2, 1.55),
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2.5,
                        connectionstyle="arc3,rad=0.0"),
    )
    ax.plot([7.0, 8.2], [1.55, 1.55], color=C_ACCENT, lw=2.5)
    ax.plot([8.2, 7.0], [7.3, 7.3], color=C_ACCENT, lw=2.5)
    ax.text(8.7, 4.4, "skip\n+x", ha="left", fontsize=FS_TICK,
            fontweight="bold", color=C_ACCENT)
    round_box(ax, 2.8, 7.15, 4.4, 1.3, C_GOAL, C_OK, lw=2.0)
    ax.text(5.0, 7.8, r"$y=F(x)+x$", ha="center", va="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 7: DenseNet / EfficientNet note
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: After ResNet -- DenseNet and EfficientNet (brief)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 4.3, 5.5, 3.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 7.1, "DenseNet", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 5.6,
        r"$x_\ell = F_\ell([x_0,x_1,\ldots,x_{\ell-1}])$"
        + "\n\n"
        "Concatenate all past\nfeature maps (not add).\n"
        "Strong reuse, compact width.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 4.3, 5.5, 3.3, C_OLD, "#d35400", lw=2.0)
    ax.text(8.85, 7.1, "EfficientNet", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color="#d35400")
    ax.text(
        8.85, 5.6,
        "Compound scaling:\n"
        r"depth, width, resolution"
        + "\n"
        "grow together by a rule.\n\n"
        "MBConv + SE blocks\n(mobile inverted bottleneck).",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.45, 11.2, 3.4, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 2.15,
        "Both keep the CNN recipe (local conv + nonlinearity),\n"
        "but change connectivity (DenseNet) or scaling policy (EfficientNet).\n\n"
        "ResNet remains the default mental model for \"deep CNN\".",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 8: Comparison table
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: Side-by-side -- what each architecture contributed",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    header = ["Net", "Signature idea", "Watch for"]
    rows = [
        ["LeNet", "conv+pool prototype", "tiny inputs"],
        ["AlexNet", "ReLU, dropout, GPU scale", "large early kernels"],
        ["VGG", "stacked 3x3 only", "very heavy FC"],
        ["Inception", "parallel multi-scale", "1x1 bottlenecks"],
        ["ResNet", "y = F(x)+x skips", "default deep backbone"],
    ]
    # header
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
        "Progression: deeper  →  multi-scale  →  better gradient paths.",
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
        (5.9, "Conv formula", r"local weighted sum over a receptive field"),
        (4.5, "LeNet→VGG", "deeper stacks of the same block"),
        (3.1, "Inception", "parallel scales, then concat"),
        (1.7, "ResNet", r"$y=F(x)+x$ makes depth trainable"),
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
        "A CNN family = connectivity pattern around the same local operator.",
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
print("MAJOR CNNs")
print("=" * 60)
print("""
Conv:  (K*X)[c',i,j] = sum_{c,u,v} K[c',c,u,v] X[c,i+u,j+v] + b

LeNet:   conv/pool prototype
AlexNet: ReLU + dropout + ImageNet scale
VGG:     stacked 3x3 template
Inception: parallel 1x1 / 3x3 / 5x5 / pool, concat
ResNet:  y = F(x) + x
DenseNet: concatenate all past maps
EfficientNet: compound depth/width/resolution scaling
""")
