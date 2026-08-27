"""
_config.py

Shared lecture figure font sizes.

Baseline matches 170_dropout.py Step 1 on a full-screen lecture figure.
Individual modules may still nudge sizes relatively (e.g. FS_AXIS - 2 on a
dense grid, or FS_AX_TITLE on a small GIF canvas) so text *looks* the same
to readers even when canvases scale differently.
"""

from __future__ import annotations

# Baseline (full-screen lecture figures, ~1-2 panels)
FS_FIGURE_TITLE = 24
FS_AX_TITLE = 18
FS_AXIS = 14
FS_TICK = 12
FS_ANNOT = 12
FS_LEGEND = 12

SCREEN_FRACTION = 0.95
FIGURE_DPI = 150


def apply_rcparams(plt_module) -> None:
    """Apply baseline sizes to matplotlib rcParams."""
    plt_module.rcParams.update({
        "figure.titlesize": FS_FIGURE_TITLE,
        "axes.titlesize": FS_AX_TITLE,
        "axes.labelsize": FS_AXIS,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "figure.dpi": FIGURE_DPI,
    })
