"""
7_000_LLM_and_VLM.py

Chapter 7 title slide: LLM & VLM
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from _config import (
    FS_FIGURE_TITLE,
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


def build_step_1():
    fig, ax = new_figure()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5, 0.6681,
        'LLM & VLM',
        ha="center", va="center",
        fontsize=FS_FIGURE_TITLE + 28,
        fontweight="bold",
        color="#1e8449",
    )
    return fig


run_slide_deck([build_step_1])
