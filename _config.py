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
FOOTER_TEMPLATE = "AI Agents. CCH. {date}."
# Placement: "bottom" (horizontal, centered) or "left" (vertical along left edge)
FOOTER_PLACEMENT = "left"

SCREEN_FRACTION = 0.95
FIGURE_DPI = 150

# Deck navigation (figure coordinates). Minimal ← / → buttons, pinned to the
# ultimate bottom-left / bottom-right corners (barely larger than the glyph).
SLIDE_NAV_BOTTOM = 0.10
SLIDE_NAV_PREV = (0.001, 0.001, 0.020, 0.024)
SLIDE_NAV_NEXT = (0.979, 0.001, 0.020, 0.024)
SLIDE_NAV_PREV_LABEL = "\u2190"  # ←
SLIDE_NAV_NEXT_LABEL = "\u2192"  # →
SLIDE_NAV_FACE = "#f4f6f7"
SLIDE_NAV_HOVER_PREV = "#d6eaf8"
SLIDE_NAV_HOVER_NEXT = "#d4edda"
SLIDE_NAV_DISABLED = "#eeeeee"
SLIDE_NAV_EDGE = "#566573"
SLIDE_NAV_LABEL_SIZE = 8


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
        # Keep clear of corner nav buttons (fig.text paints above axes).
        fig.text(
            0.5, 0.028, text,
            ha="center", va="bottom",
            fontsize=FS_FOOTER, color=FOOTER_COLOR,
        )
    else:
        # Vertical footer as its own axes so corner nav buttons (higher
        # zorder) stay visible and clickable in the true bottom-left.
        ax_foot = fig.add_axes([0.0, 0.04, 0.022, 0.92])
        ax_foot.set_axis_off()
        ax_foot.set_zorder(1)
        ax_foot.patch.set_visible(False)
        ax_foot.set_navigate(False)
        ax_foot.text(
            0.5, 0.5, text,
            ha="center", va="center", rotation=90,
            fontsize=FS_FOOTER, color=FOOTER_COLOR,
            transform=ax_foot.transAxes,
        )
        fig._step_footer_ax = ax_foot


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
    """Tiny bottom-left ← and bottom-right → on every slide. Keep refs on fig."""
    from matplotlib.widgets import Button

    # Re-assert margin so tight_layout inside builders cannot steal the strip.
    try:
        fig.subplots_adjust(bottom=max(fig.subplotpars.bottom, SLIDE_NAV_BOTTOM))
    except Exception:
        pass

    # Sliders/Buttons both call canvas.grab_mouse. If a Slider already owns
    # the grab (mid-drag) and a nav Button also tries to grab, matplotlib
    # raises RuntimeError. Releasing the prior grabber keeps both usable.
    if not getattr(fig.canvas, "_safe_mouse_grab_patched", False):
        _orig_grab = fig.canvas.grab_mouse

        def _grab_mouse_safe(ax, _orig=_orig_grab, _canvas=fig.canvas):
            grabber = _canvas.mouse_grabber
            if grabber is not None and grabber is not ax:
                try:
                    _canvas.release_mouse(grabber)
                except Exception:
                    pass
            return _orig(ax)

        fig.canvas.grab_mouse = _grab_mouse_safe
        fig.canvas._safe_mouse_grab_patched = True

    ax_prev = fig.add_axes(list(SLIDE_NAV_PREV))
    ax_next = fig.add_axes(list(SLIDE_NAV_NEXT))
    for _ax in (ax_prev, ax_next):
        _ax.set_navigate(False)
        _ax.set_zorder(1000)
        for spine in _ax.spines.values():
            spine.set_color(SLIDE_NAV_EDGE)
            spine.set_linewidth(0.8)

    btn_prev = Button(
        ax_prev, SLIDE_NAV_PREV_LABEL,
        color=SLIDE_NAV_FACE, hovercolor=SLIDE_NAV_HOVER_PREV,
    )
    btn_next = Button(
        ax_next, SLIDE_NAV_NEXT_LABEL,
        color=SLIDE_NAV_FACE, hovercolor=SLIDE_NAV_HOVER_NEXT,
    )
    for btn in (btn_prev, btn_next):
        btn.label.set_fontfamily("DejaVu Sans")
        btn.label.set_fontweight("bold")
        btn.label.set_fontsize(SLIDE_NAV_LABEL_SIZE)
        # Tighten padding so the box barely wraps the arrow.
        btn.label.set_horizontalalignment("center")
        btn.label.set_verticalalignment("center")

    def _wire_nav_button(btn, callback, enabled: bool):
        """Click without grab_mouse so Sliders and nav do not deadlock."""
        btn.disconnect_events()
        if not enabled:
            btn.label.set_color("0.7")
            btn.color = SLIDE_NAV_DISABLED
            btn.hovercolor = SLIDE_NAV_DISABLED
            btn.ax.set_facecolor(SLIDE_NAV_DISABLED)
            return

        def _on_release(event, _btn=btn, _cb=callback):
            if _btn.ignore(event):
                return
            if event.inaxes != _btn.ax or event.button != 1:
                return
            # Ensure a stuck Slider grab cannot block navigation.
            grabber = event.canvas.mouse_grabber
            if grabber is not None:
                try:
                    event.canvas.release_mouse(grabber)
                except Exception:
                    pass
            _cb(event)

        btn.connect_event("button_release_event", _on_release)

    _wire_nav_button(btn_prev, on_prev, enabled=(index > 0))
    _wire_nav_button(btn_next, on_next, enabled=(index < n_slides - 1))

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


def _retain_axes_widget(widget) -> None:
    """Pin a matplotlib widget on its figure so it is not garbage-collected."""
    try:
        fig = widget.ax.figure
    except Exception:
        return
    bag = getattr(fig, "_retained_widgets", None)
    if bag is None:
        bag = []
        fig._retained_widgets = bag
    bag.append(widget)


def _install_widget_retention() -> None:
    """Ensure Slider/Button/RadioButtons stay alive for the figure lifetime."""
    import matplotlib.widgets as mw

    if getattr(mw, "_lecture_widget_retention", False):
        return

    def _patch(cls):
        orig_init = cls.__init__

        def _init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            _retain_axes_widget(self)
            try:
                self.ax.set_navigate(False)
                if self.ax.get_zorder() < 20:
                    self.ax.set_zorder(20)
            except Exception:
                pass

        cls.__init__ = _init

    for _name in ("Slider", "Button", "RadioButtons"):
        if hasattr(mw, _name):
            _patch(getattr(mw, _name))
    mw._lecture_widget_retention = True


def _harden_interactive_axes(fig) -> None:
    """Bottom strip axes (sliders) should not fight pan/zoom or sit under plots."""
    try:
        axes = list(fig.axes)
    except Exception:
        return
    for ax in axes:
        try:
            pos = ax.get_position()
        except Exception:
            continue
        if pos.height <= 0.065 and pos.y0 <= 0.14:
            try:
                ax.set_navigate(False)
                if ax.get_zorder() < 20:
                    ax.set_zorder(20)
            except Exception:
                pass


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
    _install_widget_retention()
    i = int(max(0, min(start, len(builders) - 1)))
    n = len(builders)
    while 0 <= i < n:
        action = {"go": None}
        fig = builders[i]()
        _harden_interactive_axes(fig)
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

    _install_widget_retention()
    _harden_interactive_axes(fig)
    prepare_slide(fig, after_footer=after_footer)
    plt.show()
