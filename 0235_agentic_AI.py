"""
0235_agentic_AI.py

What is an AI agent, and what techniques make one?

Sketch:
  1. Definition -- more than an answer
  2. The agent equation
  3. LLM vs agent (open vs closed loop)
  4. Intuition: a debugging agent loop
  5. Eight layers of agent technique
  6. Tool use -- from text to action
  7. ReAct -- reason, act, observe
  8. Planning -- sequence vs conclusion
  9. Memory -- working, episodic, semantic, procedural
  10. Reflection / self-correction
  11. Search over actions
  12. Reinforcement learning for agents
  13. Multi-agent systems (with caution)
  14. The deeper definition
  15. A teaching hierarchy -- Generate to Adapt

Sits after 0232_Vision_Language_Model.py (eyes before action loops)
and before 0236_agentic_AI_example.py (concrete build-with-existing-LLM lab).
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

np.random.seed(235)


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
# Step 1: Definition
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: What is an AI agent?",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        "An AI agent does more than generate an answer.\n"
        "It pursues a goal by deciding, acting, observing, and adapting.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.5, 2.6, "Decide", "Choose the next\naction under a goal."),
        (3.4, 2.6, "Act", "Call a tool, edit code,\nbrowse, query, ..."),
        (6.3, 2.6, "Observe", "Read the result,\nerror, page, test."),
        (9.2, 2.6, "Adapt", "Update state and\nchoose again."),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 2.6, 2.5, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.3, y + 1.9, title, ha="center",
                fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
        ax.text(x + 1.3, y + 0.95, body, ha="center", va="center",
                fontsize=FS_TICK + 1)

    round_box(ax, 0.5, 0.4, 11.0, 1.8, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.3,
        "The key distinction is closed-loop behavior.\n"
        "Not one shot.  A loop that keeps going until the goal is met -- or abandoned.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 2: Agent equation
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: A useful abstraction",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.7, 5.2, 10.6, 2.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.3,
        r"Agent  $=$  Model $+$ State $+$ Goal $+$ Planning $+$ Tools $+$ Feedback",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color="#f9e79f",
    )

    parts = [
        (0.5, 2.6, "Model", "policy /\nreasoner\n(often an LLM)"),
        (2.5, 2.6, "State", "what it\nremembers\nnow"),
        (4.5, 2.6, "Goal", "what\nsuccess\nmeans"),
        (6.5, 2.6, "Planning", "how to\nsequence\nwork"),
        (8.5, 2.6, "Tools", "how it\naffects the\nworld"),
        (10.3, 2.6, "Feedback", "did the\naction\nwork?"),
    ]
    for x, y, title, body in parts:
        round_box(ax, x, y, 1.7, 2.2, C_MUTE, "0.45", lw=1.6)
        ax.text(x + 0.85, y + 1.7, title, ha="center",
                fontsize=FS_TICK, fontweight="bold", color=C_DARK)
        ax.text(x + 0.85, y + 0.75, body, ha="center", va="center",
                fontsize=FS_TICK - 1)

    round_box(ax, 0.5, 0.4, 11.0, 1.8, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 1.3,
        "Remove the loop (tools + feedback + state update)\n"
        "and you mostly have a chatbot again -- clever, but not agentic.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color=C_DARK,
    )
    return fig


# -------------------------------------------------------
# Step 3: LLM vs agent
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: LLM vs AI agent",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 4.5, 5.4, 3.0, C_OLD, "#e67e22", lw=2.2)
    ax.text(3.1, 6.95, "Conventional LLM", ha="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color="#a04000")
    ax.text(
        3.1, 5.7,
        r"$x \;\rightarrow\; \mathrm{LLM} \;\rightarrow\; y$"
        "\n\n"
        "One prompt.  One reply.\n"
        "Open loop.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )

    round_box(ax, 6.2, 4.5, 5.4, 3.0, C_GOAL, C_OK, lw=2.2)
    ax.text(8.9, 6.95, "Agent", ha="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    ax.text(
        8.9, 5.7,
        r"$s_t \rightarrow a_t \rightarrow o_{t+1} \rightarrow s_{t+1}$"
        "\n\n"
        "Decide, execute, observe,\n"
        "update -- then repeat.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )

    # Mini loop visualization
    xs = [1.5, 3.5, 5.5, 7.5, 9.5]
    labels = [r"$s_t$", r"$a_t$", r"$o_{t+1}$", r"$s_{t+1}$", r"$\cdots$"]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        round_box(ax, x - 0.55, 2.35, 1.3, 1.35, C_NEW, C_LOOP, lw=1.8)
        ax.text(x, 3.0, lab, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        if i < len(xs) - 1:
            arrow(ax, x + 0.8, 3.0, xs[i + 1] - 0.8, 3.0, C_DARK, lw=2.0)

    round_box(ax, 0.4, 0.35, 11.2, 1.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.15,
        "Same underlying model can sit in either pattern.\n"
        "Agency is the control loop around the model -- not the model alone.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 4: Debugging agent intuition
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        'Step 4: Intuition -- not "What is wrong?" but a loop that fixes it',
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.55, 11.2, 1.1, C_WARN, C_ACCENT, lw=1.8)
    ax.text(
        6.0, 7.1,
        'Instead of only asking "What is wrong with my Python program?" ...',
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#641e16",
    )

    steps = [
        (0.4, 4.55, "1", "Inspect\nrepository"),
        (2.7, 4.55, "2", "Read\nfiles"),
        (5.0, 4.55, "3", "Run\nprogram"),
        (7.3, 4.55, "4", "See\nerror"),
        (9.6, 4.55, "5", "Hypothesize"),
        (0.4, 2.35, "6", "Edit\ncode"),
        (2.7, 2.35, "7", "Run\ntests"),
        (5.0, 2.35, "8", "Inspect\nresult"),
        (7.3, 2.35, "9", "Repeat\nif needed"),
        (9.6, 2.35, "10", "Stop when\ntests pass"),
    ]
    for x, y, num, body in steps:
        round_box(ax, x, y, 2.0, 1.85, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 0.35, y + 1.45, num, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_LOOP)
        ax.text(x + 1.15, y + 0.85, body, ha="center", va="center",
                fontsize=FS_TICK)

    round_box(ax, 0.4, 0.35, 11.2, 1.65, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 1.15,
        "The loop is the important part.\n"
        "Generation is only one move inside a longer pursuit of a goal.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Step 5: Eight layers
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: Main techniques -- eight layers",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    rows = [
        ("1. Perception", "LLM/VLM, parsers, retrieval", "Understand the environment"),
        ("2. State / memory", "scratchpad, vector DB, episodes", "Remember what matters"),
        ("3. Reasoning", "CoT, decompose, verify", "Figure out what to do"),
        ("4. Planning", "ReAct, search, hierarchy", "Decide a sequence"),
        ("5. Action", "tools, APIs, code, browser", "Affect the environment"),
        ("6. Feedback", "tests, critics, rewards", "Know if it worked"),
        ("7. Multi-agent", "planner/executor, debate", "Divide complex work"),
        ("8. Learning", "RL, preferences, replay", "Improve over time"),
    ]
    y0 = 6.85
    round_box(ax, 0.35, y0, 3.3, 0.55, C_DARK, C_DARK, lw=0)
    round_box(ax, 3.75, y0, 4.0, 0.55, C_DARK, C_DARK, lw=0)
    round_box(ax, 7.85, y0, 3.8, 0.55, C_DARK, C_DARK, lw=0)
    ax.text(2.0, y0 + 0.27, "Layer", ha="center", va="center",
            fontsize=FS_TICK, fontweight="bold", color="white")
    ax.text(5.75, y0 + 0.27, "Techniques", ha="center", va="center",
            fontsize=FS_TICK, fontweight="bold", color="white")
    ax.text(9.75, y0 + 0.27, "Purpose", ha="center", va="center",
            fontsize=FS_TICK, fontweight="bold", color="white")

    for i, (layer, tech, purpose) in enumerate(rows):
        y = 6.15 - i * 0.72
        face = C_NEW if i % 2 == 0 else C_MUTE
        round_box(ax, 0.35, y, 3.3, 0.65, face, "0.55", lw=1.2)
        round_box(ax, 3.75, y, 4.0, 0.65, face, "0.55", lw=1.2)
        round_box(ax, 7.85, y, 3.8, 0.65, face, "0.55", lw=1.2)
        ax.text(2.0, y + 0.32, layer, ha="center", va="center",
                fontsize=FS_TICK - 1, fontweight="bold", color=C_DARK)
        ax.text(5.75, y + 0.32, tech, ha="center", va="center",
                fontsize=FS_TICK - 1)
        ax.text(9.75, y + 0.32, purpose, ha="center", va="center",
                fontsize=FS_TICK - 1)

    ax.text(
        6.0, 0.28,
        "The next slides zoom into the layers that change an LLM into an actor.",
        ha="center", fontsize=FS_TICK, color=C_DARK, fontweight="bold",
    )
    return fig


# -------------------------------------------------------
# Step 6: Tool use
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Tool use -- the most fundamental agent technique",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 5.55, 11.2, 2.0, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 6.85,
        r"$a_t = \pi_\theta(s_t)$"
        "     then     "
        r"$o_{t+1} = E(s_t, a_t)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 5.95,
        "The action may be a tool call, not text.  The environment answers.  Then decide again.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    tools = [
        "search()", "read_file()", "write_file()", "run_python()",
        "query_db()", "browser()", "send_email()", "book_flight()",
    ]
    for i, name in enumerate(tools):
        x = 0.45 + (i % 4) * 2.9
        y = 3.55 - (i // 4) * 1.35
        round_box(ax, x, y, 2.7, 1.15, C_MUTE, "0.45", lw=1.5)
        ax.text(x + 1.35, y + 0.55, name, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", family="monospace",
                color=C_DARK)

    round_box(ax, 0.4, 0.3, 11.2, 1.55, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.05,
        "Tool use turns an LLM from a text generator into an actor.\n"
        "Without tools, it can only talk about the world.  With tools, it can change it.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 7: ReAct + code example
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: ReAct -- Reason, Act, Observe",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    for i, (title, face, edge) in enumerate([
        ("Thought", C_NEW, C_LOOP),
        ("Action", C_OLD, "#e67e22"),
        ("Observation", C_GOAL, C_OK),
    ]):
        x = 1.2 + i * 3.5
        round_box(ax, x, 6.35, 3.0, 1.15, face, edge, lw=2.0)
        ax.text(x + 1.5, 6.9, title, ha="center", va="center",
                fontsize=FS_AX_TITLE - 2, fontweight="bold", color=edge if edge != "#e67e22" else "#a04000")
        if i < 2:
            arrow(ax, x + 3.05, 6.9, x + 3.4, 6.9, C_DARK, lw=2.0)
    arrow(ax, 10.7, 6.55, 10.7, 5.55, C_DARK, lw=1.8)
    ax.annotate(
        "", xy=(2.7, 6.35), xytext=(10.7, 5.4),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.6,
                        connectionstyle="arc3,rad=0.25"),
    )

    round_box(ax, 0.4, 0.35, 11.2, 5.7, C_CODE, "0.55", lw=1.5)
    ax.text(
        6.0, 5.55,
        "A miniature ReAct trace (exchange-rate toy)",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 2.85,
        "Thought: I need today's USD→JPY rate.\n"
        'Action:  get_exchange_rate("USD", "JPY")\n'
        "Observation: 147.2\n\n"
        "Thought: Now convert 1000 dollars.\n"
        "Action:  calculator(1000 * 147.2)\n"
        "Observation: 147200\n\n"
        "Answer: ¥147,200",
        ha="center", va="center", fontsize=FS_TICK + 1, family="monospace",
        color="#1c2833",
    )
    return fig


# -------------------------------------------------------
# Step 8: Planning
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: Planning -- sequence of actions, not one conclusion",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.4, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 7.0,
        r"$G \;\rightarrow\; \{g_1, g_2, \ldots, g_n\}$"
        "     Goal becomes an ordered agenda.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    plan = [
        "1 Find ERA5 data",
        "2 Inspect variables",
        "3 Preprocess",
        "4 Split data",
        "5 Implement model",
        "6 Train baseline",
        "7 Evaluate",
        "8 Diagnose errors",
        "9 Improve",
        "10 Retrain",
    ]
    for i, line in enumerate(plan):
        x = 0.4 + (i % 5) * 2.3
        y = 4.55 - (i // 5) * 1.55
        round_box(ax, x, y, 2.15, 1.35, C_NEW, C_LOOP, lw=1.5)
        ax.text(x + 1.075, y + 0.65, line, ha="center", va="center",
                fontsize=FS_TICK)

    round_box(ax, 0.4, 0.3, 5.4, 1.35, C_OLD, "#e67e22", lw=2.0)
    ax.text(
        3.1, 0.95,
        "Reasoning asks:\n\"What is the correct conclusion?\"",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#a04000",
    )
    round_box(ax, 6.2, 0.3, 5.4, 1.35, C_GOAL, C_OK, lw=2.0)
    ax.text(
        8.9, 0.95,
        "Planning asks:\n\"What sequence of actions reaches the goal?\"",
        ha="center", va="center", fontsize=FS_TICK + 1, color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 9: Memory
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Memory -- more than the context window",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    mems = [
        (0.4, 4.55, "Working", "What I am doing\nright now.",
         "Current plan step,\nopen files, hypothesis."),
        (3.35, 4.55, "Episodic", "What happened\nbefore.",
         '"Last time this API\nfailed auth."'),
        (6.3, 4.55, "Semantic", "Facts kept\nover time.",
         '"ERA5 has 13 levels\nused by this model."'),
        (9.25, 4.55, "Procedural", "How to do\nsomething.",
         '"Use bf16 -- FlashAttention\nneeds it."'),
    ]
    for x, y, title, tag, ex in mems:
        round_box(ax, x, y, 2.7, 3.0, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.35, y + 2.45, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.35, y + 1.75, tag, ha="center", va="center",
                fontsize=FS_TICK)
        ax.text(x + 1.35, y + 0.7, ex, ha="center", va="center",
                fontsize=FS_TICK - 1, style="italic", color="#333333")

    round_box(ax, 0.4, 2.35, 11.2, 1.8, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 3.25,
        r"$s_t = ($ current observation, working memory, long-term memory $)$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 2.65,
        "Memory is where agent architecture becomes richer than \"LLM + tools.\"",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.35, 11.2, 1.65, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.15,
        "Without memory, every step is amnesia with a clever voice.\n"
        "With memory, the agent can accumulate skill across a trajectory.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 10: Reflection
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: Reflection / self-correction",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    nodes = [
        (1.0, 5.6, "Generate\n$y_0 = f(x)$"),
        (4.0, 5.6, "Critic\n$c_0 = C(x,y_0)$"),
        (7.0, 5.6, "Revise\n$y_1 = f(x,c_0)$"),
        (10.0, 5.6, "Verify\n(tests / checks)"),
    ]
    for i, (x, y, lab) in enumerate(nodes):
        round_box(ax, x - 1.1, y - 0.85, 2.4, 1.9, C_NEW, C_LOOP, lw=2.0)
        ax.text(x, y, lab, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        if i < len(nodes) - 1:
            arrow(ax, x + 1.2, y, nodes[i + 1][0] - 1.2, y, C_DARK, lw=2.0)
    # feedback arrow
    ax.annotate(
        "", xy=(1.0, 4.55), xytext=(10.0, 4.55),
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2.0,
                        connectionstyle="arc3,rad=-0.35"),
    )
    ax.text(5.5, 3.55, "repeat if needed", ha="center",
            fontsize=FS_TICK, color=C_ACCENT, fontweight="bold")

    methods = [
        (0.4, 1.55, "Self-reflection"),
        (3.35, 1.55, "Verifier models"),
        (6.3, 1.55, "Critic–generator"),
        (9.25, 1.55, "Test-driven agents"),
    ]
    for x, y, title in methods:
        round_box(ax, x, y, 2.7, 1.35, C_MUTE, "0.45", lw=1.5)
        ax.text(x + 1.35, y + 0.65, title, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)

    round_box(ax, 0.4, 0.25, 11.2, 1.05, C_WARN, C_ACCENT, lw=1.8)
    ax.text(
        6.0, 0.75,
        "For code agents: running the program beats asking the LLM if the code is correct.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#641e16",
    )
    return fig


# -------------------------------------------------------
# Step 11: Search
# -------------------------------------------------------
def build_step_11():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 11: Search -- explore actions, then commit",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    ax.set_title("Branching action tree", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # root
    round_box(ax, 3.5, 7.6, 3.0, 1.2, C_DARK, C_DARK, lw=0)
    ax.text(5.0, 8.2, r"$s_0$", ha="center", va="center",
            fontsize=FS_ANNOT + 1, fontweight="bold", color="white")
    # children
    kids = [(1.0, 4.6, r"$a_1$", r"$s_1$", C_GOAL, C_OK),
            (6.0, 4.6, r"$a_2$", r"$s_2$", C_OLD, "#e67e22")]
    for x, y, a, s, face, edge in kids:
        arrow(ax, 5.0, 7.6, x + 1.5, y + 2.3, "0.35", lw=2.0)
        round_box(ax, x, y + 1.1, 3.0, 1.1, face, edge, lw=1.8)
        ax.text(x + 1.5, y + 1.65, a, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold")
        round_box(ax, x, y - 0.3, 3.0, 1.1, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 1.5, y + 0.25, s, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(5.0, 1.2, "Compare outcomes.  Keep the better branch.",
            ha="center", fontsize=FS_TICK + 1, color=C_DARK)

    ax = axes[1]
    ax.set_title("Classical ideas return", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ideas = [
        (0.5, 7.6, "State-space search"),
        (5.2, 7.6, "A* / heuristics"),
        (0.5, 5.7, "MCTS"),
        (5.2, 5.7, "Beam search"),
        (0.5, 3.8, "Planning"),
        (5.2, 3.8, "Constraints"),
    ]
    for x, y, title in ideas:
        round_box(ax, x, y, 4.3, 1.5, C_MUTE, "0.45", lw=1.6)
        ax.text(x + 2.15, y + 0.75, title, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    round_box(ax, 0.5, 0.6, 9.0, 2.6, C_GOAL, C_OK, lw=2.0)
    ax.text(
        5.0, 1.9,
        "Agent AI is not only an LLM problem.\n"
        "It mixes RL, planning, search, robotics,\n"
        "databases, software engineering, and LLMs.",
        ha="center", va="center", fontsize=FS_TICK + 1, color=C_OK,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    return fig


# -------------------------------------------------------
# Step 12: RL for agents
# -------------------------------------------------------
def build_step_12():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 12: Reinforcement learning -- the agent as a policy",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 5.7, 11.2, 1.8, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 6.85,
        r"$\pi_\theta(a_t \mid s_t)$"
        "      trajectory      "
        r"$\tau = (s_0,a_0,r_0,s_1,\ldots)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 6.05,
        r"Objective:  $J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}"
        r"\big[\sum_t \gamma^t r_t\big]$",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    round_box(ax, 0.4, 3.35, 11.2, 2.0, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 4.7,
        "Policy-gradient identity (trajectory return)",
        ha="center", va="center", fontsize=FS_TICK + 1,
        fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 3.9,
        r"$\nabla_\theta J ="
        r"\mathbb{E}\Big[R(\tau)\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big]$",
        ha="center", va="center", fontsize=FS_ANNOT, color="#1c2833",
    )

    round_box(ax, 0.4, 0.4, 5.4, 2.6, C_OLD, "#e67e22", lw=2.0)
    ax.text(
        3.1, 2.2,
        "Step feedback",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color="#a04000",
    )
    ax.text(
        3.1, 1.2,
        "\"This action was worth +0.3.\"\n"
        "Dense, informative, rare\nin messy real tasks.",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 6.2, 0.4, 5.4, 2.6, C_GOAL, C_OK, lw=2.0)
    ax.text(
        8.9, 2.2,
        "Trajectory feedback",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    ax.text(
        8.9, 1.2,
        "\"Task succeeded.\"\n"
        "Sparse -- and exactly why\nagentic RL is attractive.",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )
    return fig


# -------------------------------------------------------
# Step 13: Multi-agent
# -------------------------------------------------------
def build_step_13():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 13: Multi-agent systems -- power with a warning",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 4.35, 6.2, 3.3, 1.25, C_DARK, C_DARK, lw=0)
    ax.text(6.0, 6.8, "Manager", ha="center", va="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color="white")

    roles = [
        (0.5, 3.7, "Research"),
        (3.4, 3.7, "Coding"),
        (6.3, 3.7, "Testing"),
        (9.2, 3.7, "Review"),
    ]
    for x, y, title in roles:
        arrow(ax, 6.0, 6.2, x + 1.3, y + 1.6, "0.4", lw=1.8)
        round_box(ax, x, y, 2.6, 1.5, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.3, y + 0.75, title, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)

    round_box(ax, 0.5, 1.85, 11.0, 1.45, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 2.55,
        r"Planner $\rightarrow$ Researcher $\rightarrow$ Coder"
        r" $\rightarrow$ Tester $\rightarrow$ Reviewer",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )

    round_box(ax, 0.5, 0.3, 11.0, 1.25, C_WARN, C_ACCENT, lw=2.0)
    ax.text(
        6.0, 0.9,
        "Multi-agent ≠ automatically better.  Often one strong agent with tools,\n"
        "memory, verification, and planning is simpler and more reliable.",
        ha="center", va="center", fontsize=FS_TICK + 1,
        fontweight="bold", color="#641e16",
    )
    return fig


# -------------------------------------------------------
# Step 14: Deeper definition
# -------------------------------------------------------
def build_step_14():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 14: The deeper definition",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.7, 11.0, 1.8, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.6,
        "An AI agent is a goal-directed system that maintains state\n"
        "and repeatedly selects actions from observations of its environment.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.5, 3.35, 11.0, 2.0, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 4.7,
        r"$s_{t+1} = F(s_t, a_t, o_{t+1})$"
        "          "
        r"$a_t \sim \pi(a \mid s_t, G)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 3,
        fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 3.75,
        r"$G$ goal   ·   $s_t$ state   ·   $a_t$ action   ·   "
        r"$o_t$ observation   ·   $\pi$ policy   ·   $F$ update",
        ha="center", va="center", fontsize=FS_TICK, color="#333333",
    )

    round_box(ax, 0.5, 0.45, 11.0, 2.5, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 2.15,
        "The LLM is not necessarily the agent.",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color=C_OK,
    )
    ax.text(
        6.0, 1.15,
        "It can be merely the policy / reasoning component inside the agent.\n"
        "Agency lives in the loop: state, goal, action, observation, update.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 15: Hierarchy + essence + mini code
# -------------------------------------------------------
def build_step_15():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 15: Teaching hierarchy -- and a tiny agent skeleton",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Progression", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ladder = [
        (C_MUTE, "LLM"),
        (C_MUTE, "+ Context"),
        (C_NEW, "+ Tools"),
        (C_NEW, "+ Memory"),
        (C_GOAL, "→ Agent"),
        (C_GOAL, "+ Planning"),
        (C_OLD, "+ Feedback"),
        (C_OLD, "+ Verification"),
        (C_WARN, "+ Learning"),
        (C_DARK, "Autonomous agent"),
    ]
    for i, (face, lab) in enumerate(ladder):
        y = 10.7 - i * 1.05
        edge = "white" if face == C_DARK else "0.45"
        tc = "white" if face == C_DARK else C_DARK
        round_box(ax, 1.2, y, 7.6, 0.9, face, edge if face != C_DARK else C_DARK, lw=1.5)
        ax.text(5.0, y + 0.45, lab, ha="center", va="center",
                fontsize=FS_TICK + (1 if i >= 4 else 0),
                fontweight="bold", color=tc)

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Minimal code shape", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 1.0, 9.2, 10.2, C_CODE, "0.55", lw=1.5)
    ax.text(
        5.0, 6.2,
        "def agent_loop(goal, tools, max_steps=20):\n"
        "    state = {\"goal\": goal, \"mem\": []}\n"
        "    for t in range(max_steps):\n"
        "        thought = reason(state)\n"
        "        action  = decide(thought, tools)  # π\n"
        "        obs     = execute(action, tools)  # E\n"
        "        state   = update(state, action, obs)\n"
        "        if done(state):\n"
        "            return state[\"answer\"]\n"
        "    return state  # budget exhausted",
        ha="center", va="center", fontsize=FS_TICK - 1,
        family="monospace", color="#1c2833",
    )
    fig.tight_layout(rect=[0, 0.14, 1, 0.88])

    # Keep essence inside content band (above Prev/Next strip).
    fig.text(
        0.5, 0.125,
        r"Essence:   Generate  →  Act  →  Observe  →  Adapt",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color=C_DARK,
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
])


print("\n" + "=" * 60)
print("AGENTIC AI -- CLOSED-LOOP GOAL PURSUIT")
print("=" * 60)
print("""
Agent = Model + State + Goal + Planning + Tools + Feedback

LLM:   x -> y                         (open loop)
Agent: s -> a -> o -> s -> ...         (closed loop)

Layers: perception, memory, reasoning, planning,
        action, feedback, multi-agent, learning

Essence: Generate -> Act -> Observe -> Adapt

The LLM need not be the whole agent -- it can be the policy inside one.
""")
