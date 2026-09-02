"""Write Ch03 LLM-usage lectures + regenerate chapter openers and README."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent

HEADER = '''\
"""
{name}

{doc}

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

'''


def write(name: str, doc: str, body: str) -> None:
    path = ROOT / name
    path.write_text(HEADER.format(name=name, doc=doc) + "\n" + body + "\n", encoding="utf-8")
    ast.parse(path.read_text(encoding="utf-8"))
    print("wrote", name)


# ---- 180 prompting ----
write(
    "0180_prompting_and_verification.py",
    """Prompting and verification -- how to squeeze an LLM without trusting it blindly.

Sketch:
  1. Delivery-first: the model is a tool, not an oracle
  2. Prompt patterns that change behavior
  3. Verification loop: draft -> check -> revise / refuse
  4. Failure modes students must expect
  5. Attempt card""",
    textwrap.dedent(r'''
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
'''),
)

# ---- 190 RAG ----
write(
    "0190_RAG_and_grounding.py",
    """RAG and grounding -- retrieve first, then generate.

Sketch:
  1. Why pure generation fails on private / fresh facts
  2. RAG loop: index -> retrieve -> condition -> answer
  3. What 'grounded' means (citations, refuse if missing)
  4. Failure modes of bad retrieval
  5. Attempt""",
    textwrap.dedent(r'''
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
'''),
)

# ---- 200 tools ----
write(
    "0200_tools_and_agents_lite.py",
    """Tools and light agents -- let the model call functions, not invent side effects.

Sketch:
  1. Generation vs action
  2. Tool calling loop
  3. Agent = LLM + tools + stop policy (keep it thin)
  4. Safety rails
  5. Attempt""",
    textwrap.dedent(r'''
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
'''),
)

# ---- 210 you x ai ----
write(
    "0210_you_x_ai_delivery.py",
    """YOU x AI delivery -- ship a thin vertical slice using LLMs.

Sketch:
  1. What 'deliver' means in week 1
  2. Thin slice template
  3. Human strengths the model lacks
  4. Scorecard: done vs theater
  5. Bridge to the rest of the course""",
    textwrap.dedent(r'''
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
'''),
)


def regenerate_openers_readme() -> None:
    import sys
    sys.path.insert(0, str(ROOT))
    import importlib
    import _curriculum
    importlib.reload(_curriculum)
    from _curriculum import CHAPTERS, CORE_PATH

    for meta in CHAPTERS:
        opener = meta["opener"]
        content = f'''"""
{opener}

Chapter opener: {meta["code"]} -- {meta["title"]}

Roadmap and student attempt for this chapter.
"""

from _chapter import run_chapter
from _curriculum import CHAPTERS

_META = next(c for c in CHAPTERS if c["opener"] == "{opener}")
run_chapter(_META)
'''
        (ROOT / opener).write_text(content, encoding="utf-8")
        print("opener", opener)

    lines = [
        "# AI Lecture 2026",
        "",
        "Deep-learning-focused AI course.",
        "",
        "## Gravity",
        "",
        "> Begin delivery. Advance toward understanding -- fast.",
        "",
        "**Ch03 LLM usage** comes right after the DL story, *before* training-from-scratch,",
        "so students keep the promise: squeeze models for work first.",
        "",
        "Consecutive lecture numbers differ by **10**.",
        "",
        "Map source: [`_curriculum.py`](_curriculum.py). Critique: [`comment.txt`](comment.txt).",
        "",
        "## Chapters",
        "",
    ]
    for meta in CHAPTERS:
        lines += [
            f"### {meta['code']}: {meta['title']}",
            "",
            f"- Opener: `{meta['opener']}`",
            f"- Attempt: {meta['attempt']}",
            "",
            "| Tag | File | Note |",
            "|-----|------|------|",
        ]
        for tag, fname, note in meta["rows"]:
            f = fname if fname else "*(TODO)*"
            lines.append(f"| {tag} | `{f}` | {note} |")
        lines.append("")
    lines += ["## Minimal core path", ""]
    for f in CORE_PATH:
        lines.append(f"- `{f}`")
    lines += [
        "",
        "## Notes",
        "",
        "- Ch03 = LLM **usage** (prompt / RAG / tools / ship).",
        "- Ch15 = LLM **internals** (BERT / GPT mechanisms).",
        "- Ch04 = first **training** delivery (tiny torch loop).",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("README updated")


if __name__ == "__main__":
    regenerate_openers_readme()
    print("ALL OK")
