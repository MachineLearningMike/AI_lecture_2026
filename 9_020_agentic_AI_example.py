"""
0236_agentic_AI_example.py

A concrete example: build an AI agent with an existing LLM -- no training.

Sketch:
  1. Start without training -- reuse a powerful LLM
  2. The Data Analyst Agent (user goal)
  3. Four ingredients: LLM, tools, instructions, loop
  4. Define ordinary Python tools
  5. Structured tool calls (JSON), not magic execution
  6. Give the LLM a job (instructions / policy)
  7. The agent loop (the important part)
  8. Mental run -- Turns 1-3 (find the drop)
  9. Mental run -- Turns 4-5 (find why)
  10. What just happened? Developer vs LLM roles
  11. Existing LLMs are the brain
  12. Next level: database + web tools
  13. Add memory
  14. Add verification (environment as critic)
  15. Connection to REINFORCE / trajectories
  16. Three-stage picture + teaching advice

Companion to 9_010_agentic_AI.py (concepts) then this concrete build example.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_AXIS,
    FS_TICK,
    FS_ANNOT,
    FS_LEGEND,
    FIGURE_DPI,
    apply_rcparams,
    screen_figsize,
    run_slide_deck,
)
apply_rcparams(plt)
import numpy as np

np.random.seed(236)


def new_figure(*args, **kwargs):
    kwargs.setdefault("figsize", screen_figsize())
    kwargs.setdefault("dpi", FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=2.3):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=15),
    )


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_WARN = "#f8d7da"
C_FUN = "#8e44ad"
C_OLD = "#fce4d6"
C_CODE = "#fbfcf8"
C_LOOP = "#2874a6"


# -------------------------------------------------------
# Step 1
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: Build an agent without training a model",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        "The easiest way to understand an AI agent\n"
        "is to build one without training anything.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.5, 2.7, 5.3, 2.4, C_WARN, C_ACCENT, lw=2.0)
    ax.text(3.15, 4.4, "Not this course path", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT)
    ax.text(
        3.15, 3.5,
        "train gigantic model\n"
        "        ↓\n"
        "create agent",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 6.2, 2.7, 5.3, 2.4, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 4.4, "This lecture's path", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 3.5,
        "existing LLM\n"
        "  + tools + instructions + loop\n"
        "        ↓\n"
        "your agent",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 0.5, 0.45, 11.0, 1.9, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.4,
        "We already have a powerful LLM.  We give it tools, a job, state, and a loop.\n"
        "That is already an agent -- model + tools + instructions.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: the example goal
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        'Step 2: Example -- a "Data Analyst Agent"',
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 1.0, 4.6, 10.0, 2.8, C_NEW, C_LOOP, lw=2.4)
    ax.text(6.0, 6.7, "User says:", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        6.0, 5.5,
        '"Look at my sales CSV, find why sales dropped\n'
        'in August, and give me a short report."',
        ha="center", va="center", fontsize=FS_AX_TITLE - 3,
        fontweight="bold", color=C_DARK,
    )

    round_box(ax, 0.5, 0.5, 11.0, 3.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 3.3,
        "We will not hard-code the investigation steps.\n"
        "We will only provide capability and a goal.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 1.7,
        "The agent must decide: which file? which groupby?\n"
        "which region? which product? when to stop and write?",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 3: four ingredients
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: Four ingredients (no training required)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    cards = [
        (0.5, 4.0, "LLM", "reasoning\n(the brain)"),
        (3.4, 4.0, "Tools", "Python, files,\nmaybe SQL"),
        (6.3, 4.0, "Instructions", "what its job is\n(the policy)"),
        (9.2, 4.0, "State + Loop", "discoveries +\nkeep going"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 2.6, 3.3, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.3, y + 2.5, title, ha="center",
                fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
        ax.text(x + 1.3, y + 1.2, body, ha="center", va="center",
                fontsize=FS_ANNOT)

    round_box(ax, 0.5, 0.5, 11.0, 3.0, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 2.0,
        "That's already an agent.\n\n"
        "OpenAI-style building blocks are the same idea:\n"
        "model + tools + instructions  (+ sessions, handoffs, guardrails later).",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 4: tools
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: Define some tools -- ordinary Python functions",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    tools = [
        (0.5, 4.55, "list_files()", "see what exists"),
        (3.4, 4.55, "read_csv(name)", "load a table"),
        (6.3, 4.55, "run_python(code)", "compute / groupby"),
        (9.2, 4.55, "write_report(text)", "emit the answer"),
    ]
    for x, y, title, body in tools:
        round_box(ax, x, y, 2.6, 2.8, C_MUTE, "0.45", lw=1.8)
        ax.text(x + 1.3, y + 2.0, title, ha="center",
                fontsize=FS_TICK, fontweight="bold", family="monospace", color=C_DARK)
        ax.text(x + 1.3, y + 0.9, body, ha="center", va="center",
                fontsize=FS_TICK + 1)

    round_box(ax, 0.5, 0.45, 11.0, 3.6, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.2,
        "Important: the LLM does not execute Python by magic.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 1.7,
        "It decides \"I need to run Python,\"\n"
        "and emits a structured tool call.\n"
        "Your program runs the function and returns the result.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 5: structured tool call
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: Tool calls are structured decisions",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 3.6, 11.0, 3.9, C_CODE, "0.55", lw=1.6)
    ax.text(
        6.0, 7.0,
        "Example tool call the LLM might generate",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 5.1,
        '{\n'
        '    "tool": "run_python",\n'
        '    "arguments": {\n'
        '        "code": "df.groupby(\'month\')[\'sales\'].sum()"\n'
        "    }\n"
        "}",
        ha="center", va="center", fontsize=FS_TICK + 1,
        family="monospace", color="#1c2833",
    )

    round_box(ax, 0.5, 0.4, 11.0, 2.8, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.8,
        "Your host program:\n"
        "1) parse the tool name and arguments\n"
        "2) execute the real Python function\n"
        "3) append the observation back into the conversation state",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 6: instructions
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Give the LLM a job (instructions = policy)",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.7, 1.0, 10.6, 6.4, C_CODE, "0.55", lw=1.6)
    ax.text(
        6.0, 6.8,
        "System / agent instructions (sketch)",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 3.7,
        "You are a data-analysis agent.\n\n"
        "Your goal is to investigate the user's question\n"
        "using the available files and Python tools.\n\n"
        "Do not guess.  Inspect the data.\n"
        "Perform calculations when necessary.\n"
        "When you have enough evidence, produce a concise report.\n\n"
        "You may use: list_files, read_csv, run_python, write_report.",
        ha="center", va="center", fontsize=FS_TICK + 1,
        family="monospace", color="#1c2833",
    )
    return fig


# -------------------------------------------------------
# Step 7: agent loop
# -------------------------------------------------------
def build_step_7():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 7: The agent loop -- this is the important part",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Code shape", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 0.6, 9.2, 10.8, C_CODE, "0.55", lw=1.5)
    ax.text(
        5.0, 6.2,
        "while not finished:\n"
        "    response = llm(\n"
        "        instructions=instructions,\n"
        "        conversation=state,\n"
        "    )\n"
        "    if response.is_final:\n"
        "        return response.text\n"
        "    if response.has_tool_call:\n"
        "        result = execute_tool(\n"
        "            response.tool_name,\n"
        "            response.arguments,\n"
        "        )\n"
        "        state.append(response)\n"
        "        state.append(result)",
        ha="center", va="center", fontsize=FS_TICK - 1,
        family="monospace", color="#1c2833",
    )

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("The loop is the agent", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    boxes = [
        (2.5, 9.4, "LLM", C_NEW, C_LOOP),
        (2.5, 6.6, "choose action", C_MUTE, "0.45"),
        (2.5, 3.8, "Tool", C_OLD, "#e67e22"),
        (2.5, 1.0, "result → LLM again", C_GOAL, C_OK),
    ]
    for x, y, lab, face, edge in boxes:
        round_box(ax, x, y, 5.0, 1.8, face, edge, lw=2.0)
        ax.text(x + 2.5, y + 0.9, lab, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold",
                color=edge if edge not in ("0.45",) else C_DARK)
    for y1, y2 in [(9.4, 8.4), (6.6, 5.6), (3.8, 2.8)]:
        arrow(ax, 5.0, y1, 5.0, y2, C_DARK, lw=2.0)
    ax.annotate(
        "", xy=(2.4, 10.3), xytext=(2.4, 1.9),
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.8,
                        connectionstyle="arc3,rad=0.55"),
    )

    fig.tight_layout(rect=[0, 0.12, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 8: mental run 1-3
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: Mental run -- find that August really dropped",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    turns = [
        (0.4, 5.0, "Turn 1", "list_files()",
         "sales.csv\ncustomers.csv\nproducts.csv"),
        (4.2, 5.0, "Turn 2", 'read_csv("sales.csv")',
         "date, region,\nproduct, revenue\n..."),
        (8.0, 5.0, "Turn 3", "groupby(month).sum()",
         "Jun 1.21M\nJul 1.18M\nAug 0.87M\nSep 1.15M"),
    ]
    for x, y, title, action, obs in turns:
        round_box(ax, x, y, 3.6, 2.55, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 2.15, title, ha="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 1.45, action, ha="center",
                fontsize=FS_TICK, family="monospace", color=C_LOOP)
        ax.text(x + 1.8, y + 0.65, obs, ha="center", va="center",
                fontsize=FS_TICK - 1, color="#333333")

    round_box(ax, 0.4, 0.45, 11.2, 4.1, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 3.5,
        "User: Why did sales drop in August?",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 2.0,
        "After Turn 3 the agent knows the drop is real.\n"
        "It has not yet explained why -- so the loop continues.\n\n"
        "No hard-coded \"first list files\" rule from the developer.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 9: mental run 4-5
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Mental run -- drill into why",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 4.55, 5.5, 3.0, C_OLD, "#e67e22", lw=2.0)
    ax.text(3.15, 7.0, "Turn 4 -- by region", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color="#a04000")
    ax.text(
        3.15, 5.7,
        "August revenue by region:\n\n"
        "North  -5%\nSouth  -7%\nWest  -42%\nEast   -4%",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 6.1, 4.55, 5.5, 3.0, C_NEW, C_LOOP, lw=2.0)
    ax.text(8.85, 7.0, "Turn 5 -- West products", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        8.85, 5.7,
        "West × August by product:\n\n"
        "Product A   -2%\nProduct B   -4%\nProduct C  -71%",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 0.4, 0.4, 11.2, 3.7, C_GOAL, C_OK, lw=2.2)
    ax.text(6.0, 3.4, "Eventual report (sketch)", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        6.0, 1.8,
        "Total August revenue fell ~26%. Most of the decline came from the West,\n"
        "where Product C revenue fell ~71%. The drop looks like fewer orders,\n"
        "not a price change.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )
    return fig


# -------------------------------------------------------
# Step 10: what happened
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: What just happened?",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 4.2, 5.3, 3.3, C_WARN, C_ACCENT, lw=2.2)
    ax.text(3.15, 6.9, "Developer did NOT write", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT)
    ax.text(
        3.15, 5.4,
        "1. group by month\n"
        "2. group by region\n"
        "3. group by product\n"
        "4. inspect orders\n"
        "5. check prices",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax, 6.2, 4.2, 5.3, 3.3, C_GOAL, C_OK, lw=2.2)
    ax.text(8.85, 6.9, "Developer only provided", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 5.4,
        "goal\n"
        "+\n"
        "tools\n"
        "+\n"
        "constraints / instructions",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )

    round_box(ax, 0.5, 0.45, 11.0, 3.3, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 2.1,
        "The LLM chose the investigation path.\n"
        "That is what makes it agentic.\n\n"
        "Capability + goal  →  emergent procedure.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Step 11: existing LLMs are the brain
# -------------------------------------------------------
def build_step_11():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 11: Existing LLMs are the brain",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    parts = [
        (0.5, 4.3, "GPT / Claude /\nGemini / ...", C_NEW, C_LOOP),
        (3.4, 4.3, "your tools", C_MUTE, "0.45"),
        (6.3, 4.3, "your\ninstructions", C_OLD, "#e67e22"),
        (9.2, 4.3, "agent loop", C_GOAL, C_OK),
    ]
    for x, y, lab, face, edge in parts:
        round_box(ax, x, y, 2.6, 2.9, face, edge, lw=2.0)
        ax.text(x + 1.3, y + 1.45, lab, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold",
                color=edge if edge not in ("0.45",) else C_DARK)
    for i in range(3):
        arrow(ax, 3.15 + i * 2.9, 5.75, 3.35 + i * 2.9, 5.75, C_DARK, lw=2.2)

    round_box(ax, 3.5, 2.4, 5.0, 1.4, C_DARK, C_DARK, lw=0)
    ax.text(6.0, 3.1, "= your agent", ha="center", va="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color="#f9e79f")

    round_box(ax, 0.5, 0.4, 11.0, 1.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.2,
        "That is why agent development became accessible:\n"
        "reuse a frontier model; own the tools, policy, and loop.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 12: database + web
# -------------------------------------------------------
def build_step_12():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 12: Next level -- database, Python, web",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 4.35, 6.2, 3.3, 1.3, C_DARK, C_DARK, lw=0)
    ax.text(6.0, 6.85, "LLM", ha="center", va="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color="white")

    tools = [
        (0.5, 3.6, "SQL tool", "Database"),
        (4.5, 3.6, "Python tool", "Data / ML"),
        (8.5, 3.6, "Web search", "Internet"),
    ]
    for x, y, top, bottom in tools:
        arrow(ax, 6.0, 6.2, x + 1.5, y + 2.0, "0.4", lw=1.8)
        round_box(ax, x, y + 1.0, 3.0, 1.0, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 1.5, y + 1.5, top, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        round_box(ax, x, y - 0.2, 3.0, 1.0, C_MUTE, "0.45", lw=1.5)
        ax.text(x + 1.5, y + 0.3, bottom, ha="center", va="center",
                fontsize=FS_TICK + 1)

    round_box(ax, 0.5, 0.35, 11.0, 2.9, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 2.3,
        '"Why did customer churn rise last quarter?"\n\n'
        "SQL → churn calc → segments → web context → stats → explanation\n\n"
        "That is a real agent, not just a chatbot.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#145a32",
    )
    return fig


# -------------------------------------------------------
# Step 13: memory
# -------------------------------------------------------
def build_step_13():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 13: Add memory -- persistence across days",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.4, 11.0, 2.1, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 6.45,
        "Yesterday: \"Product C inventory has been unstable since June.\"\n"
        "Today: the agent can remember that -- and start closer to the truth.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )

    round_box(ax, 0.5, 2.7, 11.0, 2.3, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 4.2,
        r"$S_t = \{\mathrm{conversation},\ \mathrm{observations},"
        r"\ \mathrm{previous\ actions},\ \mathrm{memory}\}$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 3.25,
        r"$a_t = \pi_{\mathrm{LLM}}(a \mid S_t, G)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_LOOP,
    )

    round_box(ax, 0.5, 0.4, 11.0, 1.9, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.35,
        "Memory turns a one-shot investigation into a persistent system.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 14: verification
# -------------------------------------------------------
def build_step_14():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 14: Add verification -- do not trust the LLM blindly",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    # Analysis path
    nodes = [
        (0.5, 5.7, "LLM proposes\nanalysis"),
        (3.4, 5.7, "Python\nexecutes"),
        (6.3, 5.7, "LLM\ninterprets"),
        (9.2, 5.7, "Verify /\nreport"),
    ]
    for i, (x, y, lab) in enumerate(nodes):
        round_box(ax, x, y, 2.6, 1.8, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 1.3, y + 0.9, lab, ha="center", va="center",
                fontsize=FS_TICK, fontweight="bold", color=C_DARK)
        if i < 3:
            arrow(ax, x + 2.65, y + 0.9, x + 2.85, y + 0.9, C_DARK, lw=2.0)

    # Coding agent path
    round_box(ax, 0.5, 2.55, 11.0, 2.7, C_OLD, "#e67e22", lw=2.0)
    ax.text(6.0, 4.6, "Coding-agent version of the same idea", ha="center",
            fontsize=FS_TICK + 1, fontweight="bold", color="#a04000")
    ax.text(
        6.0, 3.5,
        "write code → run tests → fail → see failure → modify → retest → ... → pass",
        ha="center", va="center", fontsize=FS_ANNOT, color="#6e2c00",
    )

    round_box(ax, 0.5, 0.4, 11.0, 1.8, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 1.3,
        r"Generate $\rightarrow$ Execute $\rightarrow$ Observe $\rightarrow$ Correct"
        "\n"
        "The external environment becomes the critic.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Step 15: REINFORCE connection
# -------------------------------------------------------
def build_step_15():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 15: Connection to REINFORCE / trajectories",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 6.5,
        r"Actions $a_1,\ldots,a_T$  then final reward  $R=1$  (task succeeded).",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )

    round_box(ax, 0.5, 3.0, 11.0, 2.1, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 4.05,
        r"$\nabla_\theta J ="
        r"\mathbb{E}\Big[R\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big]$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK,
    )

    cols = [
        (0.5, 0.4, "Agent loop", "gives the trajectory\nobserve→act→observe→..."),
        (4.25, 0.4, "Sparse reward", "often only success/\nfailure at the end"),
        (8.0, 0.4, "RL / prefs", "improve the policy\nover those trajectories"),
    ]
    for x, y, title, body in cols:
        round_box(ax, x, y, 3.5, 2.2, C_GOAL, C_OK, lw=1.8)
        ax.text(x + 1.75, y + 1.6, title, ha="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_OK)
        ax.text(x + 1.75, y + 0.75, body, ha="center", va="center",
                fontsize=FS_TICK)
    return fig


# -------------------------------------------------------
# Step 16: three-stage picture + teaching advice
# -------------------------------------------------------
def build_step_16():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 16: Today → next → ultimately  (+ how to teach it)",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    stages = [
        (0.4, 4.55, "TODAY",
         "Existing LLM\n+ tools\n+ agent loop\n→ useful agent", C_NEW, C_LOOP),
        (4.2, 4.55, "NEXT",
         "Agent\n+ feedback /\nverification\n→ better agent", C_OLD, "#e67e22"),
        (8.0, 4.55, "ULTIMATELY",
         "Trajectories\n+ reward\n+ RL / prefs\n→ learned policy", C_GOAL, C_OK),
    ]
    for x, y, title, body, face, edge in stages:
        round_box(ax, x, y, 3.6, 3.0, face, edge, lw=2.0)
        ax.text(x + 1.8, y + 2.45, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold",
                color=edge if edge != "#e67e22" else "#a04000")
        ax.text(x + 1.8, y + 1.15, body, ha="center", va="center",
                fontsize=FS_TICK)

    round_box(ax, 0.4, 0.35, 11.2, 3.7, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 2.9,
        "Don't start an agent course with multi-agent frameworks.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )
    ax.text(
        6.0, 1.5,
        "Start with one existing LLM and five Python functions.\n"
        "Let students see:\n"
        r"LLM $\rightarrow$ Tool $\rightarrow$ Observation $\rightarrow$ LLM $\rightarrow \cdots$"
        "\n"
        "Most modern agent jargon is built on that loop.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#d6eaf8",
    )
    return fig


# -------------------------------------------------------
# Deck
# -------------------------------------------------------
run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
    build_step_6,
    build_step_7,
    build_step_8,
    build_step_9,
    build_step_10,
    build_step_11,
    build_step_12,
    build_step_13,
    build_step_14,
    build_step_15,
    build_step_16,
])


print("\n" + "=" * 60)
print("AGENTIC AI EXAMPLE -- DATA ANALYST AGENT")
print("=" * 60)
print("""
Build without training:
  existing LLM + tools + instructions + loop = agent

Example goal:
  Investigate why August sales dropped; write a short report.

Core loop:
  LLM decides tool call -> host executes -> observation -> LLM again
  until a final report.

Developer supplies capability and goal.
LLM chooses the investigation path.

Then add: databases/web, memory, verification,
and eventually trajectory-level RL / preferences.

Teach with one LLM and five Python functions first.
""")
