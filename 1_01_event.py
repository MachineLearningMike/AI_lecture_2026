"""
1_01_event.py

Chapter 1: Experiment, trial, sample space, and event.

Single step: definitions with the key clarification that a trial acts on
the whole population at once, and is not a draw of one unit.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_TICK,
    FIGURE_DPI,
    apply_rcparams,
    screen_figsize,
    show_maximized,
)
apply_rcparams(plt)


def new_figure(*args, **kwargs):
    # Keep figure kwargs off the Axes path -- some Matplotlib builds forward
    # unexpected keywords from subplots(**kwargs) into Axes.set().
    figsize = kwargs.pop("figsize", screen_figsize())
    dpi = kwargs.pop("dpi", FIGURE_DPI)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(*args, **kwargs) if args or kwargs else fig.add_subplot()
    return fig, ax


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
C_LOOP = "#2874a6"
C_OK = "#1e8449"
C_GOAL = "#d4edda"
C_OLD = "#fce4d6"
C_OLD_EDGE = "#d35400"


def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: Experiment, trial, sample space, event",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    # Top → bottom, non-overlapping. Trial is taller for the clarification.
    cards = [
        (
            5.80, 1.20, C_NEW, C_LOOP, C_DARK,
            "Experiment",
            "An abstract procedure that can be implemented repeatedly.",
        ),
        (
            3.15, 2.50, C_OLD, C_OLD_EDGE, C_OLD_EDGE,
            "Trial",
            "Each implementation of the experiment.\n"
            "Not a draw of one unit from a population.\n"
            "A single run of the entire procedure, acting on the whole\n"
            "population (the full system under study) at once.\n"
            "Produces exactly one outcome.",
        ),
        (
            1.80, 1.20, C_NEW, C_LOOP, C_DARK,
            "Sample space",
            "The set of all possible outcomes.",
        ),
        (
            0.45, 1.20, C_GOAL, C_OK, C_OK,
            "Event",
            "A subset of the sample space.",
        ),
    ]

    for y, h, face, edge, title_color, term, body in cards:
        round_box(ax, 0.4, y, 11.2, h, face, edge, lw=2.0)
        ax.text(
            2.0, y + h / 2, term,
            ha="center", va="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color=title_color,
        )
        ax.text(
            7.3, y + h / 2, body,
            ha="center", va="center",
            fontsize=FS_TICK + 1, color="#333333", linespacing=2.0,
        )
 
    return fig


fig = build_step_1()
show_maximized(fig)
