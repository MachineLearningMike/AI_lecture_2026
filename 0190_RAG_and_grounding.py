"""
0190_RAG_and_grounding.py

RAG and grounding -- retrieve first, then generate.

Sketch:
  1. Why pure generation fails on private / fresh facts
  2. RAG loop: index -> retrieve -> condition -> answer
  3. What 'grounded' means (citations, refuse if missing)
  4. Failure modes of bad retrieval
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
fig1.suptitle("Step 1: Expressed corpus vs your fresh task context",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax1)
round_box(ax1, 0.4, 4.6, 5.4, 2.8, C_OLD, "#e67e22", lw=2.2)
ax1.text(3.1, 6.5, "Pretrain memory", ha="center", fontsize=FS_ANNOT + 1,
         fontweight="bold", color="#a04000")
ax1.text(3.1, 5.4, "Public / historical text\ncompressed into weights.\nMay be stale or absent.",
         ha="center", va="center", fontsize=FS_ANNOT)
round_box(ax1, 6.2, 4.6, 5.4, 2.8, C_GOAL, C_OK, lw=2.2)
ax1.text(8.9, 6.5, "Your documents", ha="center", fontsize=FS_ANNOT + 1,
         fontweight="bold", color=C_OK)
ax1.text(8.9, 5.4, "Policies, tickets, code,\nPDFs, tables -- the facts\nthis job actually needs.",
         ha="center", va="center", fontsize=FS_ANNOT)
ax1.text(6.0, 2.5,
         "RAG = give the model the relevant spans at ask-time,\n"
         "instead of hoping pretrain memorized them.",
         ha="center", fontsize=FS_ANNOT + 1, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", edgecolor="0.55"))
ax1.text(6.0, 0.9, "Connects to Vision: do not pay for guessing what files already say.",
         ha="center", fontsize=FS_ANNOT, color=C_DARK)
show_maximized(fig1)

fig2, ax2 = new_figure()
fig2.suptitle("Step 2: The RAG loop", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax2)
nodes = [
    (0.4, "1. Index", "Chunk + embed\nyour corpus"),
    (3.3, "2. Retrieve", "Top-k spans\nfor this query"),
    (6.2, "3. Condition", "Prompt = question\n+ retrieved text"),
    (9.1, "4. Answer", "Generate with\ncitations / refuse"),
]
for x, title, body in nodes:
    round_box(ax2, x, 3.0, 2.6, 3.4, C_NEW, "#2874a6", lw=2.0)
    ax2.text(x + 1.3, 5.7, title, ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#2874a6")
    ax2.text(x + 1.3, 4.2, body, ha="center", va="center", fontsize=FS_ANNOT)
for x in (2.9, 5.8, 8.7):
    ax2.annotate("", xy=(x + 0.3, 4.7), xytext=(x - 0.15, 4.7),
                 arrowprops=dict(arrowstyle="->", lw=2, color=C_DARK))
ax2.text(6.0, 1.4, "Retrieval is the binding to evidence. Generation is the writing layer.",
         ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK)
show_maximized(fig2)

fig3, ax3 = new_figure()
fig3.suptitle("Step 3: Grounded answer contract",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax3)
round_box(ax3, 0.5, 4.4, 11.0, 3.0, C_GOAL, C_OK, lw=2.2)
ax3.text(6.0, 6.4, "A grounded reply", ha="center", fontsize=FS_AX_TITLE - 2,
         fontweight="bold", color=C_OK)
ax3.text(6.0, 5.3,
         "1) Answer in plain language\n"
         "2) Point to retrieved spans (doc id / quote)\n"
         "3) If retrieval is empty or weak -- refuse or ask for more context",
         ha="center", va="center", fontsize=FS_ANNOT + 1)
round_box(ax3, 0.5, 1.0, 11.0, 2.8, C_WARN, C_ACCENT, lw=2.2)
ax3.text(6.0, 2.4,
         "Ungrounded fluency is the default failure.\n"
         "Your product rule must punish answers without evidence.",
         ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#641e16")
show_maximized(fig3)

fig4, ax4 = new_figure()
fig4.suptitle("Step 4: When RAG still fails",
              fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax4)
items = [
    ("Bad chunks", "Too big / too small / split mid-table"),
    ("Wrong top-k", "Similar text, not relevant text"),
    ("Prompt ignore", "Model invents beyond the spans"),
    ("Stale index", "Docs changed; embeddings did not"),
]
for i, (t, b) in enumerate(items):
    x = 0.5 + (i % 2) * 5.8
    y = 4.4 if i < 2 else 1.2
    round_box(ax4, x, y, 5.4, 2.6, C_MUTE, "0.45", lw=1.6)
    ax4.text(x + 2.7, y + 1.85, t, ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK)
    ax4.text(x + 2.7, y + 0.95, b, ha="center", va="center", fontsize=FS_ANNOT)
show_maximized(fig4)

fig5, ax5 = new_figure()
fig5.suptitle("Step 5: Your attempt", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax5)
round_box(ax5, 1.0, 2.2, 10.0, 4.0, C_GOAL, C_OK, lw=2.4)
ax5.text(6.0, 5.2, "Own it", ha="center", fontsize=FS_AX_TITLE, fontweight="bold", color=C_OK)
ax5.text(6.0, 3.6,
         "Pick 5 short documents from a real task.\n"
         "Ask a question that needs them.\n"
         "Require citations. Force a refuse path when empty.",
         ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold")
show_maximized(fig5)
print("190 RAG done")

