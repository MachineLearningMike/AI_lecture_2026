"""
0232_Vision_Language_Model.py

What is a VLM (Vision-Language Model), and how did the idea evolve?

Sketch:
  1. Definition -- an LLM that can see
  2. Basic idea: image/video + text -> text
  3. Key trick: turn vision into tokens (patches)
  4. Classic modern architecture: encoder -> projector -> LLM
  5. Why the projector exists (dimension bridge)
  6. Generation 1 -- CLIP-style alignment
  7. Generation 2 -- vision encoder + LLM (BLIP-2, Flamingo, LLaVA)
  8. LLaVA as a clean mental model
  9. Same reasoning over heterogeneous tokens
  10. VLM ≠ classical computer vision
  11. Visual tokens are expensive (N^2 tradeoff)
  12. Toward agentic VLMs
  13. Conceptual progression: CV -> VLM -> multimodal agent
  14. Tiny code sketch of the forward path

Sits after 0230_exploit_deep_progression.py and before 0235_agentic_AI.py.
Teach as: LLM -> VLM -> multimodal LLM -> agent (one continuous evolution).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
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

np.random.seed(232)


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
        "Step 1: What is a VLM?",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        "A Vision-Language Model is an LLM that has learned\n"
        "to operate on visual information as well as language.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.5, 2.7, 5.3, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 4.4, "Useful warning", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 3.5,
        '"VLM" covers several different\n'
        "architectures -- not one recipe.\n"
        "The evolution is the story.",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 6.2, 2.7, 5.3, 2.4, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 4.4, "What it enables", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 3.5,
        "Captioning, VQA, OCR, diagrams,\n"
        "documents, localization, reasoning\n"
        "over screenshots and video frames.",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 0.5, 0.45, 11.0, 1.9, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.4,
        "Teach VLMs as the bridge between LLMs and multimodal agents --\n"
        "not as a separate island called \"computer vision.\"",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: The basic idea",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.7, 5.3, 1.8, C_OLD, "#e67e22", lw=2.0)
    ax.text(3.15, 7.0, "Ordinary LLM", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color="#a04000")
    ax.text(
        3.15, 6.25,
        r"text tokens $\rightarrow$ Transformer $\rightarrow$ text",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 6.2, 5.7, 5.3, 1.8, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 7.0, "VLM", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 6.25,
        r"image/video $+$ text $\rightarrow$ model $\rightarrow$ text",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    # Toy example cards
    round_box(ax, 0.5, 2.55, 3.5, 2.7, C_NEW, C_LOOP, lw=2.0)
    ax.text(2.25, 4.7, "Input image", ha="center",
            fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
    ax.text(2.25, 3.5, "[photo of a dog]\nrunning in grass",
            ha="center", va="center", fontsize=FS_ANNOT)

    round_box(ax, 4.25, 2.55, 3.5, 2.7, C_MUTE, "0.45", lw=1.6)
    ax.text(6.0, 4.7, "Prompt", ha="center",
            fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
    ax.text(6.0, 3.5, '"What is the dog\ndoing?"',
            ha="center", va="center", fontsize=FS_ANNOT)

    round_box(ax, 8.0, 2.55, 3.5, 2.7, C_GOAL, C_OK, lw=2.0)
    ax.text(9.75, 4.7, "Answer", ha="center",
            fontsize=FS_TICK + 1, fontweight="bold", color=C_OK)
    ax.text(9.75, 3.5, '"The dog is running\nthrough the grass."',
            ha="center", va="center", fontsize=FS_ANNOT)

    arrow(ax, 4.05, 3.9, 4.2, 3.9, C_DARK, lw=2.0)
    arrow(ax, 7.8, 3.9, 7.95, 3.9, C_DARK, lw=2.0)

    round_box(ax, 0.5, 0.4, 11.0, 1.8, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 1.3,
        "Same interface as chat -- but the context can now include pixels.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Step 3: patches -> visual tokens
# -------------------------------------------------------
def build_step_3():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 3: The key trick -- turn vision into tokens",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    ax.set_title("Image as patches (ViT-style)", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # Draw a 4x3 patch grid
    x0, y0, s = 1.5, 2.5, 1.7
    rng = np.random.default_rng(0)
    for r in range(3):
        for c in range(4):
            face = plt.cm.Greens(0.25 + 0.5 * rng.random())
            ax.add_patch(Rectangle((x0 + c * s, y0 + (2 - r) * s), s, s,
                                   facecolor=face, edgecolor="0.2", lw=1.5))
    ax.text(5.0, 8.6, "image", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.annotate("", xy=(5.0, 2.2), xytext=(5.0, 2.5),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2.2))

    ax = axes[1]
    ax.set_title(r"$I \rightarrow V=(v_1,\ldots,v_n)$", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    for i, lab in enumerate([r"$v_1$", r"$v_2$", r"$v_3$", r"$v_4$", "...", r"$v_n$"]):
        y = 8.2 - i * 1.15
        round_box(ax, 2.5, y, 5.0, 0.95, C_NEW, C_LOOP, lw=1.8)
        ax.text(5.0, y + 0.45, lab, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(5.0, 0.7, "visual feature vectors\n(not language yet)",
            ha="center", fontsize=FS_TICK + 1, color="#333333")

    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    return fig


# -------------------------------------------------------
# Step 4: classic architecture diagram
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: Classic modern VLM architecture",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 6.45, 11.0, 1.15, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 7.0,
        r"Image $\rightarrow$ Vision Encoder $\rightarrow$ Projector $\rightarrow$ LLM $\rightarrow$ Text",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    # Pipeline boxes
    round_box(ax, 0.5, 4.35, 2.4, 1.6, C_OLD, "#e67e22", lw=2.0)
    ax.text(1.7, 5.15, "Image", ha="center", fontsize=FS_ANNOT, fontweight="bold", color="#a04000")

    round_box(ax, 3.3, 4.35, 2.8, 1.6, C_NEW, C_LOOP, lw=2.0)
    ax.text(4.7, 5.15, "Vision\nEncoder", ha="center", va="center",
            fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)

    round_box(ax, 6.5, 4.35, 2.4, 1.6, C_MUTE, "0.45", lw=1.8)
    ax.text(7.7, 5.15, "Projector", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)

    round_box(ax, 9.3, 4.35, 2.2, 1.6, C_GOAL, C_OK, lw=2.0)
    ax.text(10.4, 5.15, "LLM", ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK)

    for x1, x2 in [(2.95, 3.25), (6.15, 6.45), (8.95, 9.25)]:
        arrow(ax, x1, 5.15, x2, 5.15, C_DARK, lw=2.0)

    round_box(ax, 0.5, 2.35, 5.3, 1.55, C_NEW, C_LOOP, lw=1.8)
    ax.text(3.15, 3.1, "Text prompt joins here", ha="center",
            fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
    ax.text(3.15, 2.65, "as ordinary language tokens", ha="center", fontsize=FS_TICK)

    round_box(ax, 6.2, 2.35, 5.3, 1.55, C_GOAL, C_OK, lw=1.8)
    ax.text(8.85, 3.1, "Answer comes out here", ha="center",
            fontsize=FS_TICK + 1, fontweight="bold", color=C_OK)
    ax.text(8.85, 2.65, "autoregressive text tokens", ha="center", fontsize=FS_TICK)

    arrow(ax, 3.15, 3.9, 3.15, 4.3, C_LOOP, lw=1.8)
    arrow(ax, 10.4, 4.35, 10.4, 3.95, C_OK, lw=1.8)

    round_box(ax, 0.5, 0.4, 11.0, 1.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.2,
        "This is the basic LLaVA-style mental model:\n"
        "pretrained vision + small connector + pretrained LLM + multimodal instruction tuning.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 5: projector
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: Why do we need the projector?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 5.3, 2.0, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 6.9, "Vision features", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(3.15, 6.1, r"$v_i \in \mathbb{R}^{1024}$",
            ha="center", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)

    round_box(ax, 6.2, 5.5, 5.3, 2.0, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 6.9, "LLM embeddings", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(8.85, 6.1, r"$e_i \in \mathbb{R}^{4096}$",
            ha="center", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)

    round_box(ax, 2.0, 3.15, 8.0, 1.9, C_MUTE, "0.45", lw=1.8)
    ax.text(
        6.0, 4.4,
        r"Need  $P:\mathbb{R}^{1024}\rightarrow\mathbb{R}^{4096}$"
        "   so   "
        r"$z_i = P(v_i)$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 3.6,
        r"Simplest: $z_i = W v_i + b$    |    richer: a small MLP",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.5, 0.45, 11.0, 2.3, C_OLD, "#e67e22", lw=2.0)
    ax.text(
        6.0, 1.9,
        "Architecturally, LLaVA can be surprisingly thin:\n"
        "frozen (or lightly tuned) vision encoder + connector + LLM.\n"
        "Much of the magic is alignment + instruction data, not a giant new backbone.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#6e2c00",
    )
    return fig


# -------------------------------------------------------
# Step 6: CLIP generation
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Generation 1 -- image/text alignment (CLIP)",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 6.35, 11.0, 1.25, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.95,
        r"Learn  $f_{\mathrm{image}}(I) \approx f_{\mathrm{text}}(T)$  for matching pairs",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.5, 3.55, 5.3, 2.4, C_GOAL, C_OK, lw=2.0)
    ax.text(3.15, 5.4, "High similarity", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        3.15, 4.4,
        "photo of a dog\n"
        "        ↕\n"
        '"a dog running"',
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 6.2, 3.55, 5.3, 2.4, C_WARN, C_ACCENT, lw=2.0)
    ax.text(8.85, 5.4, "Low similarity", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT)
    ax.text(
        8.85, 4.4,
        "photo of a dog\n"
        "        ↕\n"
        '"an airplane"',
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 0.5, 0.45, 11.0, 2.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 2.3,
        "CLIP is not primarily a question-answering model.\n"
        "It builds a shared semantic space -- a foundation later VLMs stand on.\n\n"
        "Alignment first.  Generative multimodal chat comes next.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 7: gen 2
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: Generation 2 -- vision encoder + LLM",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.55, 11.0, 2.0, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 6.85,
        "New goal (huge conceptual change)",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 6.1,
        r"$P(\mathrm{text}\mid\mathrm{image},\ \mathrm{text\ prompt})$"
        "     rather than     "
        r"$\mathrm{similarity}(I,T)$",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    cards = [
        (0.5, 2.55, "BLIP-2", "Q-Former bridge\nbetween frozen\nvision and LLM"),
        (4.25, 2.55, "Flamingo", "Interleaved\nimage/video + text\nconditioning"),
        (8.0, 2.55, "LLaVA", "Simple projector\n+ instruction\ntuning"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.5, 2.6, C_MUTE, "0.45", lw=1.8)
        ax.text(x + 1.75, y + 2.05, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.75, y + 0.95, body, ha="center", va="center",
                fontsize=FS_TICK + 1)

    round_box(ax, 0.5, 0.4, 11.0, 1.8, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.3,
        "Now the model can describe, answer, and reason in language\n"
        "conditioned on what it sees -- not only retrieve a matching caption.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 8: LLaVA mental model
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: LLaVA as a clean mental model",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 6.4, 11.0, 1.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 7.0,
        r"CLIP-like vision encoder  $+$  projection  $+$  LLM",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    # Token stream visualization
    toks = [
        "[VIS 1]", "[VIS 2]", "...", "[VIS N]",
        "What's", "happening", "in", "this", "image?",
    ]
    for i, t in enumerate(toks):
        x = 0.4 + i * 1.15
        face = C_NEW if t.startswith("[VIS") else C_GOAL
        edge = C_LOOP if t.startswith("[VIS") else C_OK
        round_box(ax, x, 4.35, 1.1, 1.5, face, edge, lw=1.5)
        ax.text(x + 0.55, 5.1, t, ha="center", va="center",
                fontsize=FS_TICK - 2, fontweight="bold", color=C_DARK)

    round_box(ax, 0.5, 2.35, 11.0, 1.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 3.15,
        "From the LLM's point of view, the image has become context tokens.\n"
        "Then it autoregressively generates:  \"The image shows ...\"",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    round_box(ax, 0.5, 0.4, 11.0, 1.6, C_OLD, "#e67e22", lw=2.0)
    ax.text(
        6.0, 1.2,
        "That is the powerful idea: vision enters the same computational stream\n"
        "as language -- not a separate API bolted on the side.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#6e2c00",
    )
    return fig


# -------------------------------------------------------
# Step 9: heterogeneous tokens
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Same reasoning machinery, many token types",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.7, 5.3, 1.9, C_OLD, "#e67e22", lw=2.0)
    ax.text(3.15, 7.05, "Old CV paradigm", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color="#a04000")
    ax.text(3.15, 6.3, "image → specialized model\n→ class / boxes",
            ha="center", va="center", fontsize=FS_TICK + 1)

    round_box(ax, 6.2, 5.7, 5.3, 1.9, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 7.05, "LLM / VLM paradigm", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(8.85, 6.3, "image → representation\n→ language model → reasoning",
            ha="center", va="center", fontsize=FS_TICK + 1)

    kinds = [
        (0.4, 3.35, "text"),
        (2.7, 3.35, "visual"),
        (5.0, 3.35, "audio"),
        (7.3, 3.35, "video"),
        (9.6, 3.35, "structured\n/ numerical"),
    ]
    for x, y, lab in kinds:
        round_box(ax, x, y, 2.1, 1.85, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 1.05, y + 0.9, lab, ha="center", va="center",
                fontsize=FS_TICK, fontweight="bold", color=C_DARK)

    round_box(ax, 0.5, 0.4, 11.0, 2.5, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 2.0,
        "The interesting part is not merely that the model can \"see.\"\n"
        "It is that one reasoning system can operate over heterogeneous representations.\n\n"
        "This is the same story as LLMs beginning to speak numerical / structured data.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 10: VLM vs CV
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: VLM ≠ classical computer vision",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 4.0, 5.3, 3.5, C_OLD, "#e67e22", lw=2.2)
    ax.text(3.15, 6.9, "Conventional vision", ha="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color="#a04000")
    ax.text(
        3.15, 5.4,
        r"$I \rightarrow \{\mathrm{cat},\mathrm{dog},\mathrm{car}\}$"
        "\n\n"
        "or bounding boxes /\n"
        "segmentation masks.\n\n"
        "Closed label set.\n"
        "Limited linguistic reasoning.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax, 6.2, 4.0, 5.3, 3.5, C_GOAL, C_OK, lw=2.2)
    ax.text(8.85, 6.9, "Vision-Language Model", ha="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 5.4,
        '"There are two cars.\n'
        "The red one is behind the blue one.\n"
        'The blue car appears to be turning left."\n\n'
        "Perception + language reasoning.\n"
        "Can also emit grounded boxes/masks\n"
        "when trained for that.",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 0.5, 0.45, 11.0, 3.1, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 2.0,
        "CV answers: what objects / where.\n"
        "A VLM can answer: what is going on, why it might matter, and what to say about it.\n\n"
        "When grounded, it can still return spatial structure -- but language is the interface.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 11: token cost
# -------------------------------------------------------
def build_step_11():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 11: Visual tokens are expensive -- detail $\leftrightarrow$ compute",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.7, 11.0, 1.85, C_WARN, C_ACCENT, lw=2.0)
    ax.text(
        6.0, 6.9,
        r"If context length is $N = N_{\mathrm{text}} + N_{\mathrm{vision}}$,"
        " attention scales roughly "
        r"$O(N^2)$.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#641e16",
    )
    ax.text(
        6.0, 6.15,
        "High-resolution vision can explode the token budget.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#641e16",
    )

    # Tradeoff triad
    triad = [
        (0.5, 2.7, "Visual detail", "more patches /\nfiner crops"),
        (4.25, 2.7, "# visual tokens", "hundreds →\nthousands"),
        (8.0, 2.7, "Compute", "quadratic\nattention cost"),
    ]
    for x, y, title, body in triad:
        round_box(ax, x, y, 3.5, 2.5, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.75, y + 1.9, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.75, y + 0.95, body, ha="center", va="center",
                fontsize=FS_TICK + 1)
    arrow(ax, 4.05, 3.95, 4.2, 3.95, C_DARK, lw=2.0)
    arrow(ax, 7.8, 3.95, 7.95, 3.95, C_DARK, lw=2.0)

    round_box(ax, 0.5, 0.35, 11.0, 2.0, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "Attacks on the tradeoff: MLP/linear projector, Q-Former, Perceiver/resampler,\n"
        "token pooling, adaptive resolution, cropping, hierarchical visual encoding.\n"
        "The connector is more important than it first looks.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 12: agentic VLM
# -------------------------------------------------------
def build_step_12():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 12: VLMs are moving toward agents",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    levels = [
        (0.4, 5.7, "Normal VLM", r"see $\rightarrow$ describe", C_MUTE, "0.45"),
        (4.2, 5.7, "Capable VLM", r"see $\rightarrow$ reason $\rightarrow$ answer", C_NEW, C_LOOP),
        (8.0, 5.7, "Agentic VLM", r"see $\rightarrow$ reason $\rightarrow$ act $\rightarrow$ observe $\rightarrow\cdots$",
         C_GOAL, C_OK),
    ]
    for x, y, title, body, face, edge in levels:
        round_box(ax, x, y, 3.6, 1.85, face, edge, lw=2.0)
        ax.text(x + 1.8, y + 1.4, title, ha="center",
                fontsize=FS_TICK + 1, fontweight="bold",
                color=edge if edge != "0.45" else C_DARK)
        ax.text(x + 1.8, y + 0.65, body, ha="center", va="center",
                fontsize=FS_TICK - 1)

    # Computer-use loop
    steps = [
        (0.5, 3.55, "1", "Screenshot"),
        (2.7, 3.55, "2", "VLM sees\nSettings"),
        (4.9, 3.55, "3", "click()\nSettings"),
        (7.1, 3.55, "4", "New\nscreenshot"),
        (9.3, 3.55, "5", "VLM sees\nNetwork"),
    ]
    for x, y, num, body in steps:
        round_box(ax, x, y, 2.0, 1.7, C_OLD, "#e67e22", lw=1.6)
        ax.text(x + 0.35, y + 1.3, num, ha="center",
                fontsize=FS_TICK, fontweight="bold", color="#a04000")
        ax.text(x + 1.15, y + 0.7, body, ha="center", va="center",
                fontsize=FS_TICK - 1)
    for i in range(4):
        arrow(ax, 2.55 + i * 2.2, 4.4, 2.65 + i * 2.2, 4.4, C_DARK, lw=1.8)

    round_box(ax, 0.5, 0.4, 11.0, 2.7, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 2.1,
        "Vision stops being only an input modality.\n"
        "It becomes perception inside a closed action loop --\n"
        "the same agent idea as 0235, now with eyes.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Step 13: progression ladder
# -------------------------------------------------------
def build_step_13():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 13: Conceptual progression",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    ladder = [
        (C_MUTE, "Computer Vision", "Recognize what's in an image"),
        (C_MUTE, "Vision + Language", "Align visual & linguistic spaces"),
        (C_NEW, "Vision-Language Model", "Understand + reason about vision"),
        (C_NEW, "Multimodal LLM", "Shared stream of heterogeneous tokens"),
        (C_GOAL, "See + reason + tools", "Connect perception to action APIs"),
        (C_DARK, "Multimodal Agent", r"See $\rightarrow$ think $\rightarrow$ act $\rightarrow$ observe $\rightarrow$ repeat"),
    ]
    for i, (face, title, body) in enumerate(ladder):
        y = 6.85 - i * 1.05
        edge = C_DARK if face == C_DARK else "0.45"
        tc = "white" if face == C_DARK else C_DARK
        bc = "#d6eaf8" if face == C_DARK else "#333333"
        round_box(ax, 0.7, y, 10.6, 0.95, face, edge, lw=1.6)
        ax.text(2.8, y + 0.48, title, ha="left", va="center",
                fontsize=FS_TICK, fontweight="bold", color=tc)
        ax.text(11.0, y + 0.48, body, ha="right", va="center",
                fontsize=FS_TICK - 1, color=bc)

    ax.text(
        6.0, 0.35,
        "A VLM is multimodal perception/reasoning.  Add tools + loop → vision-capable agent.",
        ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK,
    )
    return fig


# -------------------------------------------------------
# Step 14: tiny code sketch
# -------------------------------------------------------
def build_step_14():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 14: Tiny forward-path sketch (+ lecture takeaway)",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Pseudocode shape", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 0.8, 9.2, 10.5, C_CODE, "0.55", lw=1.5)
    ax.text(
        5.0, 6.2,
        "def vlm_answer(image, prompt, vision, proj, llm):\n"
        "    # 1) vision encoder: image -> patch features\n"
        "    V = vision.encode(image)          # (n, d_v)\n"
        "\n"
        "    # 2) projector into LLM embedding space\n"
        "    Z = proj(V)                       # (n, d_model)\n"
        "\n"
        "    # 3) fuse with text tokens\n"
        "    T = llm.embed_text(prompt)        # (m, d_model)\n"
        "    X = concat(Z, T)                  # (n+m, d_model)\n"
        "\n"
        "    # 4) same Transformer reasoning as an LLM\n"
        "    return llm.generate(X)",
        ha="center", va="center", fontsize=FS_TICK - 1,
        family="monospace", color="#1c2833",
    )

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("For this course", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    points = [
        (1.0, 9.5, "1", "Tokens are the universal currency."),
        (1.0, 7.7, "2", "Projector = dialect bridge."),
        (1.0, 5.9, "3", "CLIP aligns; LLaVA talks."),
        (1.0, 4.1, "4", "Detail costs quadratic compute."),
        (1.0, 2.3, "5", "Eyes + tools + loop = agent."),
    ]
    for x, y, num, body in points:
        round_box(ax, x, y, 8.0, 1.4, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 0.7, y + 0.7, num, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_LOOP)
        ax.text(x + 4.3, y + 0.7, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color=C_DARK)

    fig.tight_layout(rect=[0, 0.12, 1, 0.90])
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
])


print("\n" + "=" * 60)
print("VISION-LANGUAGE MODELS")
print("=" * 60)
print("""
VLM: multimodal model over image/video + text -> text.

Key trick: vision encoder turns pixels into visual tokens;
a projector maps them into the LLM's embedding space.

CLIP aligns.  LLaVA/Flamingo/BLIP-2 generate & reason.
Visual tokens are expensive (attention ~ N^2).
Connect tools + observe loop -> multimodal agent.

Course arc: LLM -> VLM -> multimodal LLM -> agent.
""")
