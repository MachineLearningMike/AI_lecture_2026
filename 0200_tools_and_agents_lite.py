"""
0200_tools_and_agents_lite.py

Tools and light agents -- let the model call functions, not invent side effects.

Sketch:
  1. Generation vs action
  2. Tool calling loop
  3. Agent = LLM + tools + stop policy (keep it thin)
  4. Safety rails
  5. Attempt

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
fig1.suptitle("Step 1: Text is not an action",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax1)
round_box(ax1, 0.5, 4.5, 5.3, 3.0, C_NEW, "#2874a6", lw=2.2)
ax1.text(3.15, 6.5, "Generate", ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#2874a6")
ax1.text(3.15, 5.4, "Words, plans, code drafts.\nNo world change yet.",
         ha="center", va="center", fontsize=FS_ANNOT)
round_box(ax1, 6.2, 4.5, 5.3, 3.0, C_OLD, "#d68910", lw=2.2)
ax1.text(8.85, 6.5, "Act (tools)", ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#d68910")
ax1.text(8.85, 5.4, "Search, SQL, calculator,\nemail, ticket API, code run.",
         ha="center", va="center", fontsize=FS_ANNOT)
ax1.text(6.0, 2.5,
         "Tool calling: the model proposes a function call;\n"
         "your runtime executes it and returns observations.",
         ha="center", fontsize=FS_ANNOT + 1, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", edgecolor="0.55"))
ax1.text(6.0, 0.9, "Orchestrate AI = you own the tools and the permissions.",
         ha="center", fontsize=FS_ANNOT, color=C_DARK)
show_maximized(fig1)

fig2, ax2 = new_figure()
fig2.suptitle("Step 2: Tool loop", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax2)
boxes = [
    (0.5, "User goal", C_MUTE),
    (3.2, "LLM plans\n/ calls tool", C_NEW),
    (5.9, "Runtime\nexecutes", C_OLD),
    (8.6, "Observation\nback to LLM", C_GOAL),
]
for i, (x, title, face) in enumerate(boxes):
    round_box(ax2, x, 3.5, 2.4, 2.8, face, "0.35", lw=1.8)
    ax2.text(x + 1.2, 4.9, title, ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold")
    if i < 3:
        ax2.annotate("", xy=(x + 2.55, 4.9), xytext=(x + 2.35, 4.9),
                     arrowprops=dict(arrowstyle="->", lw=2, color=C_DARK))
ax2.annotate("", xy=(1.7, 3.4), xytext=(9.8, 3.2),
             arrowprops=dict(arrowstyle="->", lw=2, color=C_FUN,
                             connectionstyle="arc3,rad=0.35"))
ax2.text(6.0, 2.3, "repeat until done / need user / hit limit",
         ha="center", fontsize=FS_ANNOT, color=C_FUN, fontweight="bold")
ax2.text(6.0, 1.0, "Keep the loop observable: log every call and every observation.",
         ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK)
show_maximized(fig2)

fig3, ax3 = new_figure()
fig3.suptitle("Step 3: Light agent -- not an autonomous employee",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax3)
round_box(ax3, 0.4, 4.6, 11.2, 2.8, C_DARK, C_DARK, lw=0)
ax3.text(6.0, 6.0,
         "Agent (lite) = LLM + a small tool menu + a stop policy",
         ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#f9e79f")
cols = [
    (0.4, "Allow", C_GOAL, C_OK, "read docs\ncalc\nsearch\ndraft"),
    (4.2, "Ask first", C_OLD, "#d68910", "send email\nwrite DB\nbuy / pay"),
    (8.0, "Never", C_WARN, C_ACCENT, "hidden spend\nunlogged shell\nsecret exfil"),
]
for x, title, face, edge, body in cols:
    round_box(ax3, x, 1.0, 3.6, 3.2, face, edge, lw=2.2)
    ax3.text(x + 1.8, 3.6, title, ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=edge)
    ax3.text(x + 1.8, 2.2, body, ha="center", va="center", fontsize=FS_ANNOT)
show_maximized(fig3)

fig4, ax4 = new_figure()
fig4.suptitle("Step 4: Rails that save you",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax4)
rails = [
    "Typed tool schemas (JSON) -- reject malformed calls",
    "Hard caps: max steps, max spend, max runtime",
    "Human approval for side effects",
    "Deterministic tools for math / DB -- do not let the LLM 'approximate' SQL results",
]
y = 6.5
for r in rails:
    round_box(ax4, 0.5, y - 0.55, 11.0, 1.15, C_MUTE, "0.5", lw=1.4)
    ax4.text(6.0, y, r, ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold")
    y -= 1.4
show_maximized(fig4)

fig5, ax5 = new_figure()
fig5.suptitle("Step 5: Your attempt", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax5)
round_box(ax5, 1.0, 2.2, 10.0, 4.0, C_GOAL, C_OK, lw=2.4)
ax5.text(6.0, 5.2, "Own it", ha="center", fontsize=FS_AX_TITLE, fontweight="bold", color=C_OK)
ax5.text(6.0, 3.6,
         "Design a 2-tool agent for one task (e.g. search notes + calculator).\n"
         "Write the allow / ask / never lists.\n"
         "Define the stop condition in one sentence.",
         ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold")
show_maximized(fig5)
print("200 tools done")

