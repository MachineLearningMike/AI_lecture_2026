"""
_chapter.py

Shared helpers for chapter-opener lecture slides.
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
    SCREEN_FRACTION,
    apply_rcparams,
    add_step_footer,
)

apply_rcparams(plt)

C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_GOAL = "#d4edda"
C_MUTE = "#f4f6f7"
C_ACCENT = "#c0392b"
C_WARN = "#f8d7da"
C_OLD = "#fce4d6"
C_FUN = "#8e44ad"
C_TODO = "#fdebd0"

TAG_FACE = {
    "Core": C_GOAL,
    "Lab": C_NEW,
    "Enrichment": C_OLD,
    "TODO": C_TODO,
}
TAG_EDGE = {
    "Core": C_OK,
    "Lab": "#2874a6",
    "Enrichment": "#e67e22",
    "TODO": "#d68910",
}


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
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


def run_chapter(meta: dict) -> None:
    """Render the standard 3-step chapter opener for one CHAPTERS entry."""
    code = meta["code"]
    title = meta["title"]
    blurb = meta["blurb"]
    attempt = meta["attempt"]
    rows = meta["rows"]

    # Step 1 -- why this chapter
    fig1, ax1 = new_figure()
    fig1.suptitle(
        f"{code}: {title}",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax1)
    round_box(ax1, 0.5, 5.4, 11.0, 2.1, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 6.45,
        f"Chapter opener -- {code}",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color="#f9e79f",
    )
    ax1.text(
        6.0, 5.75,
        title,
        ha="center", va="center", fontsize=FS_ANNOT + 2,
        fontweight="bold", color="white",
    )
    round_box(ax1, 0.5, 1.6, 11.0, 3.4, C_NEW, "#2874a6", lw=2.2)
    ax1.text(
        6.0, 3.3, blurb,
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold",
        color=C_DARK,
    )
    ax1.text(
        6.0, 0.8,
        "Teach in chapter order. File numbers are the map -- not museum labels.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                  edgecolor="0.55", alpha=0.95),
    )
    show_maximized(fig1)

    # Step 2 -- roadmap
    fig2, ax2 = new_figure()
    fig2.suptitle(
        f"{code} roadmap -- Core / Lab / Enrichment / TODO",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax2)

    n = max(len(rows), 1)
    row_h = min(0.55, 5.8 / n)
    y_top = 7.1
    for i, (tag, fname, note) in enumerate(rows):
        y = y_top - i * row_h
        face = TAG_FACE.get(tag, C_MUTE)
        edge = TAG_EDGE.get(tag, "0.45")
        round_box(ax2, 0.35, y - row_h * 0.42, 1.7, row_h * 0.78, face, edge, lw=1.4)
        ax2.text(1.2, y, tag, ha="center", va="center",
                 fontsize=FS_TICK, fontweight="bold", color=C_DARK)
        round_box(ax2, 2.2, y - row_h * 0.42, 9.45, row_h * 0.78, C_MUTE, "0.55", lw=1.0)
        label = fname if fname else "(to be written)"
        ax2.text(2.4, y, label, ha="left", va="center",
                 fontsize=FS_TICK, fontweight="bold", color=C_DARK)
        ax2.text(11.4, y, note, ha="right", va="center",
                 fontsize=FS_TICK - 1, color="0.25")
    show_maximized(fig2)

    # Step 3 -- attempt
    fig3, ax3 = new_figure()
    fig3.suptitle(
        f"{code}: your attempt",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax3)
    round_box(ax3, 1.0, 2.5, 10.0, 3.5, C_GOAL, C_OK, lw=2.4)
    ax3.text(
        6.0, 5.2, "Own it",
        ha="center", va="center", fontsize=FS_AX_TITLE,
        fontweight="bold", color=C_OK,
    )
    ax3.text(
        6.0, 3.8, attempt,
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold",
        color="#145a32",
    )
    ax3.text(
        6.0, 1.4,
        "See it run  ->  touch it  ->  name it  ->  own it",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        color=C_DARK,
    )
    show_maximized(fig3)
