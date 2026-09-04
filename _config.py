"""
_config.py

Shared lecture figure font sizes.

Baseline matches 0710_dropout.py Step 1 on a full-screen lecture figure.
Individual modules may still nudge sizes relatively (e.g. FS_AXIS - 2 on a
dense grid, or FS_AX_TITLE on a small GIF canvas) so text *looks* the same
to readers even when canvases scale differently.
"""

from __future__ import annotations

from datetime import date

# Baseline (full-screen lecture figures, ~1-2 panels)
FS_FIGURE_TITLE = 24
FS_AX_TITLE = 18
FS_AXIS = 14
FS_TICK = 12
FS_ANNOT = 14
FS_LEGEND = 12

# Footer on every Step figure (shown via show_maximized -> add_step_footer)
FS_FOOTER = 9
FOOTER_COLOR = "0.45"
FOOTER_TEMPLATE = "Directed by Mike. Powered by Grok. Presented on {date}."
# Placement: "bottom" (horizontal, centered) or "left" (vertical along left edge)
FOOTER_PLACEMENT = "left"

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


def add_step_footer(
    fig,
    when: date | None = None,
    placement: str | None = None,
) -> None:
    """Stamp a Step figure with the shared lecture footer.

    placement:
      - \"bottom\": horizontal text centered under the figure (default)
      - \"left\":   vertical text along the left-most edge
    If placement is None, uses FOOTER_PLACEMENT from this module.
    """
    side = (placement or FOOTER_PLACEMENT).lower().strip()
    if side not in {"bottom", "left"}:
        raise ValueError(f"footer placement must be 'bottom' or 'left', got {placement!r}")

    day = (when or date.today()).isoformat()
    text = FOOTER_TEMPLATE.format(date=day)

    try:
        if side == "bottom":
            fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, 0.06))
        else:
            fig.subplots_adjust(left=max(fig.subplotpars.left, 0.08))
    except Exception:
        pass

    if side == "bottom":
        fig.text(
            0.5, 0.012, text,
            ha="center", va="bottom",
            fontsize=FS_FOOTER, color=FOOTER_COLOR,
        )
    else:
        # Vertical along the left-most side (reads upward).
        fig.text(
            0.012, 0.5, text,
            ha="center", va="center", rotation=90,
            fontsize=FS_FOOTER, color=FOOTER_COLOR,
        )
