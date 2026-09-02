"""
0210_you_x_ai_delivery.py

YOU x AI delivery -- ship a thin vertical slice using LLMs.

Sketch:
  1. What 'deliver' means in week 1
  2. Thin slice template
  3. Human strengths the model lacks
  4. Scorecard: done vs theater
  5. Bridge to the rest of the course

Companion: Ch03 LLM usage (delivery-first), before training-from-scratch.
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
C_FUN = "#8e44ad"
C_WARN = "#f8d7da"
C_OLD = "#fce4d6"


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



fig1, ax1 = new_figure()
fig1.suptitle("Step 1: Deliver means a usable slice -- not a finished thesis",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax1)
round_box(ax1, 0.4, 5.3, 11.2, 2.2, C_DARK, C_DARK, lw=0)
ax1.text(6.0, 6.4,
         "From the contract: begin delivery, advance toward understanding -- fast.",
         ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#f9e79f")
tri = [
    (0.5, "Input", "a real question\nor workflow pain"),
    (4.3, "Loop", "LLM + verify\n(+ RAG/tools if needed)"),
    (8.1, "Output", "artifact someone\ncan use tomorrow"),
]
for x, title, body in tri:
    round_box(ax1, x, 1.4, 3.5, 3.4, C_GOAL, C_OK, lw=2.2)
    ax1.text(x + 1.75, 4.1, title, ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_OK)
    ax1.text(x + 1.75, 2.7, body, ha="center", va="center", fontsize=FS_ANNOT)
show_maximized(fig1)

fig2, ax2 = new_figure()
fig2.suptitle("Step 2: Thin-slice template (copy this)",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax2)
rows = [
    "1. Name the user and the decision they need",
    "2. List allowed sources (files / URLs / none)",
    "3. Choose: prompt-only / RAG / tools",
    "4. Define verification checks before shipping",
    "5. Produce the artifact (memo, script, ticket reply, demo)",
    "6. Write 5 lines: what the model did vs what you verified",
]
y = 6.9
for r in rows:
    round_box(ax2, 0.5, y - 0.45, 11.0, 0.95, C_MUTE, "0.5", lw=1.2)
    ax2.text(0.8, y, r, ha="left", va="center", fontsize=FS_ANNOT + 1, fontweight="bold")
    y -= 1.05
show_maximized(fig2)

fig3, ax3 = new_figure()
fig3.suptitle("Step 3: What you bring that the model does not",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax3)
human = [
    (0.4, "Taste", "What good looks like\nfor this client"),
    (4.2, "Context", "Unexpressed constraints\nnot in the prompt"),
    (8.0, "Responsibility", "Who gets hurt\nif it is wrong"),
]
for x, title, body in human:
    round_box(ax3, x, 2.5, 3.6, 4.0, C_NEW, C_FUN, lw=2.2)
    ax3.text(x + 1.8, 5.7, title, ha="center", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_FUN)
    ax3.text(x + 1.8, 4.0, body, ha="center", va="center", fontsize=FS_ANNOT + 1)
ax3.text(6.0, 1.2, "Strategy: bet on irreverent human intuition -- then feed AI wisely.",
         ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK)
show_maximized(fig3)

fig4, ax4 = new_figure()
fig4.suptitle("Step 4: Done vs theater",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax4)
round_box(ax4, 0.4, 1.5, 5.4, 5.5, C_WARN, C_ACCENT, lw=2.2)
ax4.text(3.1, 6.3, "Theater", ha="center", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_ACCENT)
ax4.text(3.1, 4.0,
         "Chat screenshot\nNo checks\nNo owner\nCannot rerun\nCannot explain failure",
         ha="center", va="center", fontsize=FS_ANNOT + 1)
round_box(ax4, 6.2, 1.5, 5.4, 5.5, C_GOAL, C_OK, lw=2.2)
ax4.text(8.9, 6.3, "Done", ha="center", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
ax4.text(8.9, 4.0,
         "Reusable prompt/tooling\nVerification notes\nNamed owner\nRerun path\nKnown failure modes",
         ha="center", va="center", fontsize=FS_ANNOT + 1)
show_maximized(fig4)

fig5, ax5 = new_figure()
fig5.suptitle("Step 5: Bridge -- next we open the hood",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax5)
round_box(ax5, 0.5, 4.6, 11.0, 2.8, C_DARK, C_DARK, lw=0)
ax5.text(6.0, 6.0,
         "Ch03: use LLMs to deliver.\n"
         "Ch04+: train tiny models, then climb toward Transformer internals.",
         ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#f9e79f")
round_box(ax5, 0.5, 1.2, 11.0, 2.8, C_GOAL, C_OK, lw=2.2)
ax5.text(6.0, 2.6,
         "Attempt now: ship one thin YOU x AI slice this week.\n"
         "Bring it to class: artifact + verification notes.",
         ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#145a32")
show_maximized(fig5)
print("210 you x ai done")

