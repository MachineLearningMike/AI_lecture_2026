"""
Why exploit LLMs now: local computers grew slowly; frontier LLMs and
their datacenters scaled many times over -- and will keep scaling.

Sketch:
  1. What does an SOTA LLM "know"?
  2. Opportunity -- LLMs and their upgrades (growth contrast)
  3. Intelligence -- local vs. datacenter (specialized local models vs frontier LLMs)
  4. Local vs. datacenter for special problems -- replace or combine?
  5. Broad, non-deterministic, flexible intelligence for business problems
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_AXIS,
    FS_TICK,
    FS_ANNOT,
    FS_LEGEND,
    FIGURE_DPI,
    SCREEN_FRACTION,
    apply_rcparams,
    add_step_footer,
    run_slide_deck,
)

apply_rcparams(plt)

C_LOCAL = "#85929e"
C_LLM = "#8e44ad"
C_OK = "#1e8449"
C_GOAL = "#d4edda"
C_WARN = "#f8d7da"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_DARK = "#1a5276"


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=2.2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=14),
    )


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


# -------------------------------------------------------

# -------------------------------------------------------
# FIGURE 1: What does an SOTA LLM "know"?
# -------------------------------------------------------
# "Know" is in quotes: next-token prediction over compressed training data,
# not human-like understanding or guaranteed truth.
def build_step_1():
    fig0, ax0 = new_figure()
    fig0.suptitle(
        "Step 1: What does an SOTA LLM 'know'?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    ax0.set_xlim(0, 12)
    ax0.set_ylim(0, 8)
    ax0.axis("off")

    round_box(ax0, 0.35, 6.55, 11.3, 1.2, C_DARK, C_DARK, lw=0)
    ax0.text(
        6.0, 7.15,
        "A state-of-the-art LLM does not 'know' like a person.\n"
        "It has compressed patterns from a huge public training corpus — fluent, useful, and fallible.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax0, 0.45, 4.55, 3.2, 1.6, C_MUTE, "0.45", lw=1.5)
    ax0.text(2.05, 5.55, "Training corpus", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax0.text(2.05, 5.0, "web, books, code,\narticles (public / licensed)",
             ha="center", va="center", fontsize=FS_TICK - 1)

    arrow(ax0, 3.75, 5.35, 4.15, 5.35, C_LLM, 2.2)
    round_box(ax0, 4.25, 4.55, 3.5, 1.6, "#f5eef8", C_LLM, lw=2.0)
    ax0.text(6.0, 5.55, "Compressed weights", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_LLM)
    ax0.text(6.0, 5.0, "statistical patterns,\nnot a searchable database",
             ha="center", va="center", fontsize=FS_TICK - 1)

    arrow(ax0, 7.85, 5.35, 8.25, 5.35, C_OK, 2.2)
    round_box(ax0, 8.35, 4.55, 3.2, 1.6, C_GOAL, C_OK, lw=2.0)
    ax0.text(9.95, 5.55, "Fluent answers", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax0.text(9.95, 5.0, "next-token prediction\nat generation time",
             ha="center", va="center", fontsize=FS_TICK - 1)

    round_box(ax0, 0.45, 1.55, 5.4, 2.7, C_GOAL, C_OK, lw=2.0)
    ax0.text(3.15, 3.85, "Often 'knows' (in the model sense)", ha="center",
             fontsize=FS_TICK + 1, fontweight="bold", color=C_OK)
    ax0.text(
        3.15, 2.55,
        "• language, style, and common concepts\n"
        "• widely published facts & procedures\n"
        "• code patterns and reasoning templates\n"
        "• how to follow instructions & formats",
        ha="center", va="center", fontsize=FS_TICK,
    )

    round_box(ax0, 6.15, 1.55, 5.4, 2.7, C_WARN, C_ACCENT, lw=2.0)
    ax0.text(8.85, 3.85, "Does not reliably 'know'", ha="center",
             fontsize=FS_TICK + 1, fontweight="bold", color=C_ACCENT)
    ax0.text(
        8.85, 2.55,
        "• your private files or live company data\n"
        "• events after the training cutoff (unless tools)\n"
        "• guaranteed truth — it can sound sure and be wrong\n"
        "• what only sensors / databases / specialists measure",
        ha="center", va="center", fontsize=FS_TICK,
    )

    ax0.text(
        6.0, 0.7,
        "Course link: this is 'expressed historical knowledge' compressed into weights.\n"
        "To use private or fresh facts, you must supply them (prompt, RAG, tools).",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f9e79f", edgecolor="#b7950b", alpha=0.96),
    )

    return fig0


# FIGURE 2: Opportunity -- LLMs and their upgrades
# -------------------------------------------------------
# Schematic (not a precise FLOPs chart): local PCs barely moved;
# frontier LLM + datacenter capability rose steeply and is projected to keep rising.
years = np.array([2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028])
# Relative capability index (illustrative)
local_cap = np.array([1.00, 1.05, 1.10, 1.14, 1.18, 1.22, 1.26, 1.30, 1.34])
# Observed-ish steep rise through 2025, then continued fast growth (dashed projection)
llm_cap = np.array([1.0, 1.8, 4.0, 10.0, 22.0, 40.0, 70.0, 110.0, 170.0])
split = 5  # index of 2025: history through 2025, projection after

def build_step_2():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 2: Opportunity — LLMs and their upgrades",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # Main growth chart
    ax1.plot(
        years[: split + 1], local_cap[: split + 1],
        "-o", color=C_LOCAL, lw=3.0, ms=8, label="Your local computer (CPU / GPU at hand)",
    )
    ax1.plot(
        years[split:], local_cap[split:],
        "--o", color=C_LOCAL, lw=2.4, ms=7, alpha=0.85,
    )
    ax1.plot(
        years[: split + 1], llm_cap[: split + 1],
        "-o", color=C_LLM, lw=3.2, ms=9,
        label="Frontier LLMs + hyperscale datacenters",
    )
    ax1.plot(
        years[split:], llm_cap[split:],
        "--o", color=C_LLM, lw=2.8, ms=8, alpha=0.9,
    )

    ax1.axvline(2025.0, color="0.55", ls=":", lw=1.4)
    ax1.text(
        2025.15, llm_cap[split] * 0.55, "now →",
        fontsize=FS_TICK, color="0.35", fontweight="bold",
    )

    ax1.set_xlim(2019.7, 2028.4)
    ax1.set_ylim(0, max(llm_cap) * 1.12)
    ax1.set_xlabel("Year", fontsize=FS_AXIS)
    ax1.set_ylabel("Relative capability (schematic index)", fontsize=FS_AXIS)
    ax1.tick_params(labelsize=FS_TICK)
    ax1.grid(True, alpha=0.28)
    ax1.legend(fontsize=FS_LEGEND, loc="upper left", framealpha=0.95)

    # Callout boxes on the chart (axes fraction via blended annotation)
    ax1.annotate(
        "Slow climb:\nnew laptop ≈ last laptop\nfor most everyday work",
        xy=(2023.0, 1.2), xytext=(2020.4, 55),
        fontsize=FS_ANNOT - 1, color="#2c3e50", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_LOCAL, lw=1.8),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eaecee", edgecolor=C_LOCAL, alpha=0.95),
    )
    ax1.annotate(
        "Steep climb:\nmodels, context, tools,\nand datacenter scale\nmultiplied in a few years",
        xy=(2024.2, 22), xytext=(2025.6, 95),
        fontsize=FS_ANNOT - 1, color="#4a235a", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_LLM, lw=1.8),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5eef8", edgecolor=C_LLM, alpha=0.95),
    )

    ax1.set_title(
        "Schematic contrast — not a precise FLOPs chart. The gap is the opportunity.",
        fontsize=FS_AX_TITLE - 3, color="0.35", pad=8,
    )

    # Bottom takeaway strip in figure coordinates
    fig1.text(
        0.5, 0.04,
        "Do not wait for your laptop to become a supercomputer.\n"
        "Exploit LLM intelligence that already lives in giant datacenters — and is still upgrading fast.",
        ha="center", va="bottom", fontsize=FS_ANNOT + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f9e79f", edgecolor="#b7950b", alpha=0.96),
    )

    fig1.tight_layout(rect=[0, 0.12, 1, 0.94])
    return fig1


# -------------------------------------------------------
# FIGURE 3: Intelligence -- local vs. datacenter
# -------------------------------------------------------
# Title meaning (polished):
#   Intelligence you can train and/or host locally (specialized models)
#   is far smaller than intelligence trained and hosted in datacenters
#   (frontier LLMs).
def build_step_3():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        "Step 3: Intelligence — local vs. datacenter",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # Schematic capability magnitudes (illustrative index, not FLOPs)
    # Local = specialized models you can train/host yourself
    local_models = [
        ("Image\nclassifier", 4.0),
        ("Speech /\nASR model", 5.0),
        ("Recommender\n/ ranker", 6.0),
        ("Small domain\nLM (local)", 8.0),
    ]
    # Datacenter = frontier LLM trained + served at hyperscale
    dc_height = 120.0

    x_local = np.arange(len(local_models))
    heights = [h for _, h in local_models]
    labels = [n for n, _ in local_models]

    bars_local = ax2.bar(
        x_local, heights, width=0.7, color=C_LOCAL, edgecolor="#5d6d7e",
        linewidth=1.4, label="Local intelligence (specialized AI you train/host)",
        zorder=3,
    )
    # Gap, then one giant datacenter bar
    x_dc = len(local_models) + 1.2
    bar_dc = ax2.bar(
        [x_dc], [dc_height], width=1.1, color=C_LLM, edgecolor="#5b2c6f",
        linewidth=1.6, label="Datacenter intelligence (frontier LLM, trained + hosted)",
        zorder=3,
    )

    # Value tags
    for rect, h in zip(bars_local, heights):
        ax2.text(
            rect.get_x() + rect.get_width() / 2, h + 1.5, f"{h:.0f}",
            ha="center", va="bottom", fontsize=FS_TICK, color="#2c3e50", fontweight="bold",
        )
    ax2.text(
        x_dc, dc_height + 2.5, f"{dc_height:.0f}+",
        ha="center", va="bottom", fontsize=FS_ANNOT, color="#4a235a", fontweight="bold",
    )

    ax2.set_xticks(list(x_local) + [x_dc])
    ax2.set_xticklabels(labels + ["Frontier LLM\n(datacenter)"], fontsize=FS_TICK - 1)
    ax2.set_ylabel("Relative general intelligence / breadth (schematic)", fontsize=FS_AXIS)
    ax2.set_ylim(0, dc_height * 1.18)
    ax2.grid(True, axis="y", alpha=0.28, zorder=0)
    ax2.legend(fontsize=FS_LEGEND - 1, loc="upper left", framealpha=0.95)

    # Callouts
    ax2.annotate(
        "Local = specialized AI\n"
        "trainable / hostable on\n"
        "your machine or small server.\n"
        "Narrow, useful — but limited.",
        xy=(1.5, 8), xytext=(0.2, 55),
        fontsize=FS_ANNOT - 1, color="#2c3e50", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_LOCAL, lw=1.8),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#eaecee", edgecolor=C_LOCAL, alpha=0.95),
    )
    ax2.annotate(
        "Datacenter = frontier LLM\n"
        "trained and hosted at scale.\n"
        "Broad intelligence you rent\n"
        "through an API — far larger.",
        xy=(x_dc, 90), xytext=(x_dc - 3.2, 70),
        fontsize=FS_ANNOT - 1, color="#4a235a", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_LLM, lw=1.8),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5eef8", edgecolor=C_LLM, alpha=0.95),
    )

    # Visual "much much less" marker
    ax2.annotate(
        "", xy=(x_dc - 0.7, 25), xytext=(len(local_models) - 0.3, 25),
        arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2.2),
    )
    ax2.text(
        (len(local_models) - 0.3 + x_dc - 0.7) / 2, 32,
        "orders of magnitude\nin breadth / generality",
        ha="center", va="bottom", fontsize=FS_TICK, fontweight="bold", color="#922b21",
    )

    ax2.set_title(
        "Local intelligence = specialized models. Datacenter intelligence = frontier LLMs.",
        fontsize=FS_AX_TITLE - 3, color="0.35", pad=8,
    )

    fig2.text(
        0.5, 0.04,
        "What you can train and host locally is useful — and still tiny next to datacenter LLMs.\n"
        "Exploit the large intelligence; keep local models for narrow, private, or real-time jobs.",
        ha="center", va="bottom", fontsize=FS_ANNOT + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f9e79f", edgecolor="#b7950b", alpha=0.96),
    )

    fig2.tight_layout(rect=[0, 0.12, 1, 0.94])

    return fig2


# -------------------------------------------------------
# FIGURE 4: Local vs. datacenter for special problems
# -------------------------------------------------------
# Question: replace specialized local models with datacenter LLMs,
# or combine them when solving concrete problems?
def build_step_4():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        "Step 4: Local vs. datacenter — replace or combine?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    # ~1 cm up/down on blank-slide axes (compensated so middle +2 cm stays visible).
    fig3.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.10)
    dy = 1.0
    ax3.set_xlim(0, 12)
    ax3.set_ylim(0.35 - dy - 0.25, 6.55 + dy + 1.2 + 0.25)
    ax3.axis("off")

    # Top banner moved up ~1 cm
    round_box(ax3, 0.35, 6.55 + dy, 11.3, 1.2, C_DARK, C_DARK, lw=0)
    ax3.text(
        6.0, 7.15 + dy,
        "For a special problem: must every local specialized model be replaced by a datacenter LLM?\n"
        "Usually no — combine them. Use each where it is strongest.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    # Middle panels: height +2 cm (grow 1 cm up and 1 cm down into freed space)
    mid_y = 2.55 - dy
    mid_h = 3.7 + 2 * dy

    # Left: Replace (weak default)
    round_box(ax3, 0.35, mid_y, 5.4, mid_h, C_WARN, C_ACCENT, lw=2.2)
    ax3.text(3.05, 5.85 + dy, "A. Replace", ha="center", fontsize=FS_AX_TITLE - 2,
             fontweight="bold", color=C_ACCENT)
    ax3.text(3.05, 5.25 + dy, "Throw away local models.\nRoute everything to a datacenter LLM.",
             ha="center", va="center", fontsize=FS_TICK + 1)

    round_box(ax3, 0.7, 3.55, 1.6, 0.95, C_MUTE, "0.55", lw=1.2)
    ax3.text(1.5, 4.02, "Problem", ha="center", va="center", fontsize=FS_TICK, fontweight="bold")
    arrow(ax3, 2.4, 4.02, 2.85, 4.02, C_ACCENT, 1.8)
    round_box(ax3, 2.9, 3.55, 2.4, 0.95, "#f5eef8", C_LLM, lw=1.6)
    ax3.text(4.1, 4.02, "Datacenter\nLLM only", ha="center", va="center",
             fontsize=FS_TICK, fontweight="bold", color=C_LLM)

    ax3.text(
        3.05, 2.95 - dy,
        "Risks: cost, latency, privacy,\nhallucination on narrow tasks,\nlost domain precision.",
        ha="center", va="center", fontsize=FS_TICK, color="#641e16",
    )

    # Right: Combine (recommended)
    round_box(ax3, 6.25, mid_y, 5.4, mid_h, C_GOAL, C_OK, lw=2.4)
    ax3.text(8.95, 5.85 + dy, "B. Combine  (recommended)", ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    ax3.text(8.95, 5.25 + dy, "Keep specialized local models.\nCall the datacenter LLM where breadth helps.",
             ha="center", va="center", fontsize=FS_TICK + 1)

    round_box(ax3, 6.5, 3.55, 1.35, 0.95, C_MUTE, "0.55", lw=1.2)
    ax3.text(7.175, 4.02, "Problem", ha="center", va="center", fontsize=FS_TICK - 1, fontweight="bold")
    arrow(ax3, 7.95, 4.02, 8.25, 4.02, C_OK, 1.8)
    round_box(ax3, 8.3, 3.95, 1.45, 0.7, "#eaecee", C_LOCAL, lw=1.4)
    ax3.text(9.025, 4.3, "Local\nspecialist", ha="center", va="center",
             fontsize=FS_TICK - 2, fontweight="bold", color="#2c3e50")
    round_box(ax3, 8.3, 3.15, 1.45, 0.7, "#f5eef8", C_LLM, lw=1.4)
    ax3.text(9.025, 3.5, "Datacenter\nLLM", ha="center", va="center",
             fontsize=FS_TICK - 2, fontweight="bold", color=C_LLM)
    arrow(ax3, 9.85, 4.02, 10.2, 4.02, C_OK, 1.8)
    round_box(ax3, 10.25, 3.55, 1.2, 0.95, C_GOAL, C_OK, lw=1.4)
    ax3.text(10.85, 4.02, "Better\nanswer", ha="center", va="center",
             fontsize=FS_TICK - 2, fontweight="bold", color=C_OK)

    ax3.text(
        8.95, 2.95 - dy,
        "Local: speed, privacy, precision.\nLLM: language, planning, glue.\nTogether: stronger systems.",
        ha="center", va="center", fontsize=FS_TICK, color="#145a32",
    )

    # Bottom banner moved down ~1 cm
    round_box(ax3, 0.35, 0.35 - dy, 11.3, 1.95, C_MUTE, "0.45", lw=1.5)
    ax3.text(6.0, 1.95 - dy, "When solving a special problem, ask:",
             ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax3.text(
        6.0, 1.15 - dy,
        "Need privacy / low latency / hard numeric precision?  → keep a local specialist.\n"
        "Need language, tools, planning, or messy unstructured input?  → call a datacenter LLM.\n"
        "Default for this course: combine — do not blindly replace.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    return fig3


# -------------------------------------------------------
# FIGURE 5: Broad / non-deterministic / flexible intelligence
# -------------------------------------------------------
# Business problems are largely natural-language problems.
# That is where datacenter LLMs are unusually strong.
def build_step_5():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        "Step 5: Broad, non-deterministic, flexible intelligence for business problems",
        fontsize=FS_FIGURE_TITLE - 3, fontweight="bold",
    )
    # ~1 cm up/down on blank-slide axes (compensated so middle +2 cm stays visible).
    fig4.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.10)
    dy = 1.0
    ax4.set_xlim(0, 12)
    ax4.set_ylim(0.35 - dy - 0.25, 6.55 + dy + 1.2 + 0.25)
    ax4.axis("off")

    # Top banner moved up ~1 cm
    round_box(ax4, 0.35, 6.55 + dy, 11.3, 1.2, C_DARK, C_DARK, lw=0)
    ax4.text(
        6.0, 7.15 + dy,
        "Most business work arrives as language: emails, tickets, contracts, chats, policies, reports.\n"
        "Messy, ambiguous, context-heavy — exactly where frontier LLMs excel.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    # Middle columns: height +2 cm (grow 1 cm up and 1 cm down into freed space)
    mid_y = 2.35 - dy
    mid_h = 3.9 + 2 * dy
    item_step = 0.45 + (2 * dy) / 5
    trait_step = 1.05 + (2 * dy) / 2

    # Left column: business NL inputs
    round_box(ax4, 0.35, mid_y, 3.4, mid_h, C_MUTE, "0.45", lw=1.6)
    ax4.text(2.05, 5.9 + dy, "Business inputs\n(natural language)", ha="center",
             fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
    nl_items = [
        "Emails & meetings",
        "Support tickets",
        "Contracts / RFPs",
        "Policies & SOPs",
        "Reports & slides",
        "Chat with customers",
    ]
    for i, t in enumerate(nl_items):
        y = 5.25 + dy - i * item_step
        round_box(ax4, 0.55, y - 0.18, 3.0, 0.4, "#eaf2f8", C_DARK, lw=1.0)
        ax4.text(2.05, y, t, ha="center", va="center", fontsize=FS_TICK - 1)

    # Center: LLM traits
    round_box(ax4, 4.1, mid_y, 3.8, mid_h, "#f5eef8", C_LLM, lw=2.2)
    ax4.text(6.0, 5.9 + dy, "Datacenter LLM", ha="center",
             fontsize=FS_TICK + 1, fontweight="bold", color=C_LLM)
    traits = [
        ("Broad", "One model covers many\ndepartments and tasks"),
        ("Flexible", "New job ≈ new prompt\n(+ tools), not a new model"),
        ("Non-deterministic", "Explores phrasings &\noptions — needs checks"),
    ]
    for i, (title, body) in enumerate(traits):
        y = 4.85 + dy - i * trait_step
        round_box(ax4, 4.35, y - 0.35, 3.3, 0.95, "white", C_LLM, lw=1.3)
        ax4.text(6.0, y + 0.25, title, ha="center", va="center",
                 fontsize=FS_TICK, fontweight="bold", color=C_LLM)
        ax4.text(6.0, y - 0.2, body, ha="center", va="center", fontsize=FS_TICK - 2)

    arrow(ax4, 3.85, 4.3, 4.15, 4.3, C_LLM, 2.0)

    # Right column: business outputs
    round_box(ax4, 8.25, mid_y, 3.4, mid_h, C_GOAL, C_OK, lw=2.0)
    ax4.text(9.95, 5.9 + dy, "Business outputs\nLLMs unlock", ha="center",
             fontsize=FS_TICK + 1, fontweight="bold", color=C_OK)
    out_items = [
        "Draft replies & memos",
        "Summarize & brief",
        "Extract fields → JSON",
        "Route / triage work",
        "Plan next steps",
        "Glue tools & people",
    ]
    for i, t in enumerate(out_items):
        y = 5.25 + dy - i * item_step
        round_box(ax4, 8.45, y - 0.18, 3.0, 0.4, "white", C_OK, lw=1.0)
        ax4.text(9.95, y, t, ha="center", va="center", fontsize=FS_TICK - 1)

    arrow(ax4, 7.95, 4.3, 8.25, 4.3, C_OK, 2.0)

    # Bottom banner moved down ~1 cm
    round_box(ax4, 0.35, 0.35 - dy, 11.3, 1.75, C_MUTE, "0.45", lw=1.5)
    ax4.text(
        6.0, 1.55 - dy,
        "Specialized local models shine on clean, repeated, numeric or sensor tasks.\n"
        "Business problems are often language problems: ambiguous goals, changing rules, human stakeholders.\n"
        "That is why broad, flexible (and carefully checked) LLM intelligence is a business lever — not a toy.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK,
    )

    return fig4


run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
])

print("\n" + "=" * 60)
print("OPPORTUNITY — LLMs AND THEIR UPGRADES")
print("=" * 60)
print("""
Step 1 -- what an SOTA LLM 'knows':
  Compressed patterns from a huge public training corpus.
  Fluent, useful, fallible -- not private facts or guaranteed truth.

Step 2 -- growth over time:
  Local machines rise slowly; frontier LLMs + datacenters rise steeply.

Step 3 -- intelligence you can own locally vs rent from a datacenter:
  Local = specialized AI models (train/host yourself): narrow, smaller.
  Datacenter = frontier LLMs (train/host at scale): broad, far larger.

Step 4 -- for special problems: replace or combine?
  Replace-everything-with-LLM is usually a weak default.
  Combine: local specialists where they win; datacenter LLM for
  language, planning, glue, and messy unstructured work.

Step 5 -- business problems are largely language problems:
  Emails, tickets, contracts, policies, meetings, reports.
  LLMs are broad, flexible, and non-deterministic -- strong on that terrain,
  if you verify outputs (verification ladder).

Strategy:
  Exploit datacenter LLM intelligence for business language work,
  keep local specialized models where they fit,
  then deepen how training and Transformers work.
""")
