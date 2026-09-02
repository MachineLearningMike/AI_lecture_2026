"""
0180_prompting_and_verification.py

Prompting and verification -- how to squeeze an LLM without trusting it blindly.

Sketch:
  1. Delivery-first: the model is a tool, not an oracle
  2. Prompt patterns that change behavior
  3. Verification loop: draft -> check -> revise / refuse
  4. Failure modes students must expect
  5. Attempt card

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



# FIGURE 1
fig1, ax1 = new_figure()
fig1.suptitle("Step 1: Delivery-first -- the LLM is a coworker, not a source of truth",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax1)
round_box(ax1, 0.4, 5.5, 11.2, 2.1, C_DARK, C_DARK, lw=0)
ax1.text(6.0, 6.55,
         "Mission reminder: harness and orchestrate AI.\n"
         "You stay accountable for the output that ships.",
         ha="center", va="center", fontsize=FS_ANNOT + 1,
         fontweight="bold", color="#f9e79f")
cards = [
    (0.4, 1.4, 3.6, 3.7, C_WARN, C_ACCENT, "Not this",
     "Paste a vague ask.\nAccept first answer.\nShip it."),
    (4.2, 1.4, 3.6, 3.7, C_GOAL, C_OK, "This",
     "State goal + constraints.\nDemand checks.\nYou verify."),
    (8.0, 1.4, 3.6, 3.7, C_NEW, "#2874a6", "Why",
     "LLMs remix expressed\nhistorical text.\nThey can be fluent\nand wrong."),
]
for x, y, w, h, face, edge, title, body in cards:
    round_box(ax1, x, y, w, h, face, edge, lw=2.2)
    ax1.text(x + w / 2, y + h - 0.55, title, ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=edge)
    ax1.text(x + w / 2, y + 1.5, body, ha="center", va="center", fontsize=FS_ANNOT)
show_maximized(fig1)

# FIGURE 2
fig2, ax2 = new_figure()
fig2.suptitle("Step 2: Prompt patterns that actually change behavior",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax2)
rows = [
    ("Role + goal", "You are a careful analyst. Goal: ..."),
    ("Constraints", "Only use the attached notes. If unknown, say so."),
    ("Format", "Return: findings / risks / next checks (bullets)."),
    ("Evidence ask", "Quote the span that supports each claim."),
    ("Self-check", "List what could be wrong before finalizing."),
]
y = 6.8
for title, ex in rows:
    round_box(ax2, 0.4, y - 0.55, 2.8, 1.0, C_NEW, "#2874a6", lw=1.6)
    ax2.text(1.8, y, title, ha="center", va="center", fontsize=FS_TICK, fontweight="bold")
    round_box(ax2, 3.4, y - 0.55, 8.2, 1.0, C_MUTE, "0.55", lw=1.2)
    ax2.text(3.6, y, ex, ha="left", va="center", fontsize=FS_ANNOT)
    y -= 1.2
show_maximized(fig2)

# FIGURE 3
fig3, ax3 = new_figure()
fig3.suptitle("Step 3: Verification loop -- draft, check, revise or refuse",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax3)
steps = [
    (0.5, "1. Draft", "Get a candidate\nanswer fast"),
    (3.3, "2. Check", "Facts, units,\ncode, citations"),
    (6.1, "3. Stress", "Ask for counter-\nexamples / edge cases"),
    (8.9, "4. Decide", "Revise, tool-call,\nor refuse to ship"),
]
for x, title, body in steps:
    round_box(ax3, x, 3.2, 2.5, 3.2, C_GOAL, C_OK, lw=2.0)
    ax3.text(x + 1.25, 5.7, title, ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_OK)
    ax3.text(x + 1.25, 4.3, body, ha="center", va="center", fontsize=FS_ANNOT)
for x in (2.9, 5.7, 8.5):
    ax3.annotate("", xy=(x + 0.3, 4.8), xytext=(x - 0.2, 4.8),
                 arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2))
ax3.text(6.0, 1.5,
         "Fluent text is cheap. Verified text is the product.",
         ha="center", fontsize=FS_ANNOT + 2, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", edgecolor="0.55"))
show_maximized(fig3)

# FIGURE 4
fig4, ax4 = new_figure()
fig4.suptitle("Step 4: Failure modes you must expect",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax4)
fails = [
    (0.4, 4.2, "Hallucinated facts", "Confident names, dates, APIs that do not exist."),
    (6.2, 4.2, "Silent constraint drop", "Ignores your 'only from notes' rule."),
    (0.4, 1.0, "Sycophancy", "Agrees with a wrong premise to please you."),
    (6.2, 1.0, "Overlong mush", "Looks thorough; hides the missing check."),
]
for x, y, title, body in fails:
    round_box(ax4, x, y, 5.4, 2.6, C_WARN, C_ACCENT, lw=2.0)
    ax4.text(x + 2.7, y + 1.85, title, ha="center", fontsize=FS_ANNOT + 1,
             fontweight="bold", color=C_ACCENT)
    ax4.text(x + 2.7, y + 0.95, body, ha="center", va="center", fontsize=FS_ANNOT)
show_maximized(fig4)

# FIGURE 5
fig5, ax5 = new_figure()
fig5.suptitle("Step 5: Your attempt", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax5)
round_box(ax5, 1.0, 2.2, 10.0, 4.0, C_GOAL, C_OK, lw=2.4)
ax5.text(6.0, 5.3, "Own it", ha="center", fontsize=FS_AX_TITLE, fontweight="bold", color=C_OK)
ax5.text(6.0, 3.7,
         "Take one real homework / work question.\n"
         "Write a constrained prompt.\n"
         "Force a verification pass.\n"
         "Ship only what you can defend.",
         ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold")
show_maximized(fig5)

print("180 prompting + verification done")

