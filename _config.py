"""
_config.py

Shared lecture figure fonts, footer, and slide-deck navigation.

Baseline matches 0710_dropout.py Step 1 on a full-screen lecture figure.
Individual modules may still nudge sizes relatively (e.g. FS_AXIS - 2 on a
dense grid, or FS_AX_TITLE on a small GIF canvas) so text *looks* the same
to readers even when canvases scale differently.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Sequence

# Baseline (full-screen lecture figures, ~1-2 panels)
FS_FIGURE_TITLE = 24
FS_AX_TITLE = 18
FS_AXIS = 14
FS_TICK = 14
FS_ANNOT = 14
FS_LEGEND = 14

# Footer on every Step figure (shown via show_maximized / prepare_slide)
FS_FOOTER = 9
FOOTER_COLOR = "0.45"
FOOTER_TEMPLATE = "DL/RL Basics. CCH. {date}."
# Placement: "bottom" (horizontal, centered) or "left" (vertical along left edge)
FOOTER_PLACEMENT = "left"

SCREEN_FRACTION = 0.95
FIGURE_DPI = 150

# Deck navigation strip (figure coordinates). Footer stays on the left,
# so bottom corners remain free for Prev / Next.
SLIDE_NAV_BOTTOM = 0.10
SLIDE_NAV_PREV = (0.02, 0.018, 0.13, 0.055)
SLIDE_NAV_NEXT = (0.85, 0.018, 0.13, 0.055)
SLIDE_NAV_FACE = "#f4f6f7"
SLIDE_NAV_HOVER_PREV = "#d6eaf8"
SLIDE_NAV_HOVER_NEXT = "#d4edda"
SLIDE_NAV_DISABLED = "#eeeeee"
SLIDE_NAV_EDGE = "#566573"


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


def screen_figsize(fraction: float = SCREEN_FRACTION, dpi: int = FIGURE_DPI):
    """Return (width_in, height_in) for a nearly full-screen lecture figure."""
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


def maximize_window(fig) -> None:
    """Size the figure to the lecture screen and request a maximized window."""
    fig.set_dpi(FIGURE_DPI)
    fig.set_size_inches(screen_figsize(), forward=True)
    fig.canvas.draw_idle()
    manager = getattr(fig.canvas, "manager", None)
    if manager is None:
        return
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


def prepare_slide(fig, *, after_footer: Callable | None = None) -> None:
    """Reserve bottom nav space, stamp footer, maximize."""
    try:
        fig.subplots_adjust(
            bottom=max(fig.subplotpars.bottom, SLIDE_NAV_BOTTOM),
            left=max(fig.subplotpars.left, 0.08),
        )
    except Exception:
        pass
    add_step_footer(fig)
    if after_footer is not None:
        after_footer(fig)
    maximize_window(fig)


def add_prev_next_buttons(fig, index: int, n_slides: int, on_prev, on_next):
    """Bottom-left Prev and bottom-right Next on every slide. Keep refs on fig."""
    from matplotlib.widgets import Button

    # Re-assert margin so tight_layout inside builders cannot steal the strip.
    try:
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, SLIDE_NAV_BOTTOM))
    except Exception:
        pass

    ax_prev = fig.add_axes(list(SLIDE_NAV_PREV))
    ax_next = fig.add_axes(list(SLIDE_NAV_NEXT))
    for _ax in (ax_prev, ax_next):
        _ax.set_navigate(False)
        for spine in _ax.spines.values():
            spine.set_color(SLIDE_NAV_EDGE)
            spine.set_linewidth(1.2)

    btn_prev = Button(
        ax_prev, "Prev",
        color=SLIDE_NAV_FACE, hovercolor=SLIDE_NAV_HOVER_PREV,
    )
    btn_next = Button(
        ax_next, "Next",
        color=SLIDE_NAV_FACE, hovercolor=SLIDE_NAV_HOVER_NEXT,
    )
    btn_prev.label.set_fontweight("bold")
    btn_next.label.set_fontweight("bold")

    if index <= 0:
        btn_prev.label.set_color("0.7")
        btn_prev.color = SLIDE_NAV_DISABLED
        btn_prev.hovercolor = SLIDE_NAV_DISABLED
    else:
        btn_prev.on_clicked(on_prev)
    if index >= n_slides - 1:
        btn_next.label.set_color("0.7")
        btn_next.color = SLIDE_NAV_DISABLED
        btn_next.hovercolor = SLIDE_NAV_DISABLED
    else:
        btn_next.on_clicked(on_next)

    # Optional index cue centered in the nav strip.
    fig._nav_index = fig.text(
        0.5, 0.042, f"{index + 1} / {n_slides}",
        ha="center", va="center",
        fontsize=FS_FOOTER + 1, color=FOOTER_COLOR, fontweight="bold",
    )

    fig._nav_buttons = (btn_prev, btn_next)
    try:
        fig.canvas.draw_idle()
    except Exception:
        pass
    return fig._nav_buttons


def run_slide_deck(
    builders: Sequence[Callable],
    *,
    start: int = 0,
    after_footer: Callable | None = None,
) -> None:
    """Present builders one at a time; Prev/Next rebuild the neighbor slide."""
    import matplotlib.pyplot as plt

    if not builders:
        return
    i = int(max(0, min(start, len(builders) - 1)))
    n = len(builders)
    while 0 <= i < n:
        action = {"go": None}
        fig = builders[i]()
        prepare_slide(fig, after_footer=after_footer)

        def _prev(event, i=i):
            if i > 0:
                action["go"] = i - 1
                plt.close(fig)

        def _next(event, i=i):
            if i < n - 1:
                action["go"] = i + 1
                plt.close(fig)

        add_prev_next_buttons(fig, i, n, _prev, _next)

        def _on_key(event, i=i):
            if event.key in ("left", "p"):
                _prev(event)
            elif event.key in ("right", "n", " "):
                _next(event)

        fig.canvas.mpl_connect("key_press_event", _on_key)
        plt.show()
        if action["go"] is None:
            break
        i = action["go"]


def show_maximized(fig, *, after_footer: Callable | None = None) -> None:
    """Single-figure path: prepare slide chrome, then block on plt.show()."""
    import matplotlib.pyplot as plt

    prepare_slide(fig, after_footer=after_footer)
    plt.show()
