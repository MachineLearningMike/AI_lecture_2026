"""
1_00_Data.py

Chapter 1 title slide: Data

Step 2: models are only as good as the data they train on --
the informal claim, nearby theorems/slogans, and credited quotes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_ANNOT,
    FS_TICK,
    FIGURE_DPI,
    apply_rcparams,
    screen_figsize,
    run_slide_deck,
)
apply_rcparams(plt)


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
C_MUTE = "#f4f6f7"
C_LOOP = "#2874a6"
C_OK = "#d4edda"
C_OK_EDGE = "#1e8449"
C_OLD = "#fce4d6"
C_OLD_EDGE = "#d35400"


def build_step_1():
    fig, ax = new_figure()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5, 0.6681,
        'Data',
        ha="center", va="center",
        fontsize=FS_FIGURE_TITLE + 28,
        fontweight="bold",
        color="#1e8449",
    )
    return fig


def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        'Step 2: "No model is better than the dataset on which the model is trained."',
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    # Claim status
    round_box(ax, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 7.02,
        "Not an established, attributed quote -- but the idea is right.",
        ha="center", va="center", fontsize=FS_AX_TITLE - 3,
        fontweight="bold", color="#f9e79f",
    )

    # Two nearby principles
    round_box(ax, 0.4, 3.55, 5.4, 2.55, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        3.1, 5.7,
        "No Free Lunch (NFL)",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        3.1, 4.55,
        "Wolpert, 1996.\n"
        "No single model is best for every\n"
        "problem. Performance is tied to the\n"
        "specific data / task you face.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.2, 3.55, 5.4, 2.55, C_OLD, C_OLD_EDGE, lw=2.0)
    ax.text(
        8.9, 5.7,
        "Garbage In, Garbage Out",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OLD_EDGE,
    )
    ax.text(
        8.9, 4.55,
        "Classic computing slogan.\n"
        "Flawed input yields flawed output.\n"
        "A model's ceiling is set by the\n"
        "quality of its training data.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    # Credited alternatives
    round_box(ax, 0.4, 0.45, 11.2, 2.85, C_OK, C_OK_EDGE, lw=2.0)
    ax.text(
        6.0, 1.85,
        '"Essentially, all models are wrong, but some are useful."\n'
        "— George E. P. Box\n\n"
        '"We don\'t have better algorithms. We just have more data."\n'
        "— Peter Norvig",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )
    return fig


run_slide_deck([
    build_step_1,
    build_step_2,
])
