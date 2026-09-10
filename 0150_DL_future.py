"""
013_future_challenges_for_DL.py

Future challenges for deep learning -- a hierarchy built from first principles.

Correction to a naive 1..10 list:
  Generalization is NOT one rung.
  It is a vertical axis that cuts through every capability.

Capability ladder (what the system tries to do):
  Recognition -> Representation -> Prediction -> Generation
  -> Inference -> Simulation -> Planning -> Control -> Discovery

Generalization ladder (how far novelty goes):
  familiar -> new examples -> new distribution -> new regime
  -> structural novelty -> extrapolation / new rules

The frontier is the product:
  Capability  x  Generalization

Sketch:
  1. What are we ranking?
  2. Correction: generalization is vertical, not a step
  3. Two-dimensional picture
  4. Levels of generalization
  5. From first principles: start with x -> y
  6. Interactive: walk capability levels one by one
  7. Recognition in detail -- and its limit
  8. Where the hard frontier begins
  9. Expressed vs unexpressed knowledge
  10. Takeaway for the course

Companion to 012_historical_achievements_by_DL.py,
0110_lecture_strategy.py, 0380_train_validate_test_deploy.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.widgets import Button, Slider
from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_AXIS,
    FS_TICK,
    FS_ANNOT,
    FIGURE_DPI,
    SCREEN_FRACTION,
    apply_rcparams,
    add_step_footer,
    run_slide_deck,
)
apply_rcparams(plt)
import numpy as np

np.random.seed(13)


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


def show_maximized(fig, after_footer=None):
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
    # Footer may call subplots_adjust; apply inch tweaks after that.
    if after_footer is not None:
        after_footer(fig)
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


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=2.2):
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
C_BIN = "#d68910"
C_GRAY = "#85929e"
C_ACTIVE = "#f9e79f"

# Data-units per cm (measured for screen_figsize lecture axes).
CM_X = 0.533
CM_Y = 0.673
CM_Y_STEP6 = 0.899  # two-panel Step 6 axes, ylim 0..10


CAPABILITIES = [
    ("Recognition", "What is this?", r"$x \rightarrow y$"),
    ("Representation", "What structure is hidden?", r"$x \rightarrow z \rightarrow y$"),
    ("Prediction", "What happens next?", r"$x_t \rightarrow x_{t+1}$"),
    ("Generation", "Can I create a valid instance?", r"$z \rightarrow x$ / $P(x)$"),
    ("Inference", "What hidden cause explains this?", r"$x \rightarrow \hat z$"),
    ("Simulation", "Can I evolve the system?", r"$z_{t+1}=F(z_t)$"),
    ("Planning", "What sequence is best?", r"$\max_a \mathbb{E}[R]$"),
    ("Control", "How do I steer the world?", "observe-act-observe"),
    ("Discovery", "What don't we know yet?", "hypothesis + experiment"),
]

GEN_LEVELS = [
    "Familiar",
    "New examples",
    "New distribution",
    "New regime",
    "Structural novelty",
    "Extrapolation /\nnew rules",
]
# Generalization altitude: light blue -> deep blue (not near-black)
GEN_COLORS = [
    "#d6effa",  # lightest sky (familiar)
    "#7ec8f0",
    "#2e9cdb",
    "#1a6f9e",
    "#0d5a8f",
    "#154360",  # deep blue, still clearly blue
]
GEN_EDGE = [
    "#5dade2",
    "#2e9cdb",
    "#1a6f9e",
    "#0e4a6e",
    "#0a4068",
    "#0e2f44",
]
GEN_TEXT = [
    C_DARK,
    C_DARK,
    "white",
    "white",
    "white",
    "white",
]
# Empty / frontier tiles: light green (not on the blue scale)
EMPTY_FACE = "#a9dfbf"
EMPTY_EDGE = "#27ae60"


# -------------------------------------------------------
# FIGURE 1: What are we ranking?
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 1: Slow down -- what exactly are we ranking?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax1)

    round_box(ax1, 0.5, 5.3, 11.0, 2.2, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 6.4,
        "Not the historical order of papers.\n"
        "The kind of capability required.",
        ha="center", va="center", fontsize=FS_ANNOT + 2,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax1, 0.5, 1.0, 5.3, 3.8, C_WARN, C_ACCENT, lw=2.2)
    ax1.text(3.15, 4.1, "Weak hierarchy", ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_ACCENT)
    ax1.text(
        3.15, 2.5,
        "A flat list of fashionable tasks.\n"
        "Mixes difficulty dimensions.\n"
        "Puts 'generalization' as one rung.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )

    round_box(ax1, 6.2, 1.0, 5.3, 3.8, C_GOAL, C_OK, lw=2.2)
    ax1.text(8.85, 4.1, "Defensible hierarchy", ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    ax1.text(
        8.85, 2.5,
        "Capability ladder (what to do)\n"
        "x generalization (how far novelty goes).\n"
        "Built from $x\\rightarrow y$ upward.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )
    return fig1



# -------------------------------------------------------
# FIGURE 2: Correction -- generalization is vertical
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        "Step 2: Correction -- generalization is not Level 4",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax2)

    round_box(ax2, 0.5, 5.2, 11.0, 2.3, C_WARN, C_ACCENT, lw=2.2)
    ax2.text(
        6.0, 6.35,
        "A classifier that only memorizes is not really solving recognition.\n"
        "A forecast that only works on familiar weather is not really forecasting.\n"
        "Generalization is a requirement of every capability.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        color="#641e16",
    )

    # Fake bad ladder with generalization as a rung, crossed out
    bad = ["Recognition", "Representation", "Prediction", "GENERALIZATION", "Generation", "..."]
    for i, name in enumerate(bad):
        x = 0.7 + i * 1.85
        face = C_WARN if name == "GENERALIZATION" else C_MUTE
        edge = C_ACCENT if name == "GENERALIZATION" else "0.45"
        round_box(ax2, x, 2.2, 1.7, 2.2, face, edge, lw=2.0)
        ax2.text(x + 0.85, 3.3, name, ha="center", va="center",
                 fontsize=FS_TICK, fontweight="bold", color=edge)
        if name == "GENERALIZATION":
            ax2.plot([x + 0.15, x + 1.55], [2.5, 4.1], color=C_ACCENT, lw=3.0)
            ax2.plot([x + 0.15, x + 1.55], [4.1, 2.5], color=C_ACCENT, lw=3.0)

    ax2.text(
        6.0, 1.0,
        "Pull generalization out of the ladder. Make it the vertical axis.",
        ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK,
    )
    return fig2



# -------------------------------------------------------
# FIGURE 3: 2D picture
# -------------------------------------------------------
def build_step_3():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        r"Step 3: The real picture -- Capability $\times$ Generalization",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    caps_short = [c[0] for c in CAPABILITIES]
    # rows = gen levels bottom=familiar .. top=extrapolation; cols = capabilities
    # 1 = often workable today, 0 = frontier / fragile
    mat = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=float)

    n_gen, n_cap = mat.shape
    # Workable cells: row color = light blue -> deep blue. Empty: green.
    for i in range(n_gen):          # gen level index, 0=familiar
        for j in range(n_cap):      # capability index
            workable = mat[i, j] > 0.5
            if workable:
                face, ecol, alpha = GEN_COLORS[i], GEN_EDGE[i], 1.0
            else:
                face, ecol, alpha = EMPTY_FACE, EMPTY_EDGE, 1.0
            rect = Rectangle(
                (j - 0.5, i - 0.5), 1.0, 1.0,
                facecolor=face, edgecolor=ecol, linewidth=1.8,
                alpha=alpha,
            )
            ax3.add_patch(rect)

    # Red boundaries between generalization levels and capability columns
    BOUND_RED = "#c0392b"
    for i in range(n_gen + 1):
        ax3.axhline(i - 0.5, color=BOUND_RED, lw=2.6, zorder=5)
    for j in range(n_cap + 1):
        ax3.axvline(j - 0.5, color=BOUND_RED, lw=1.8, zorder=5)

    ax3.set_xlim(-0.5, n_cap - 0.5)
    ax3.set_ylim(-0.5, n_gen - 0.5)
    ax3.set_xticks(range(n_cap))
    ax3.set_xticklabels(caps_short, rotation=35, ha="right", fontsize=FS_TICK - 2)
    ax3.set_yticks(range(n_gen))
    # Distinguish the harder generalization altitudes on the axis itself.
    gen_labels_explained = [
        "Familiar",
        "New examples",
        "New distribution\n(same world, P shifts)",
        "New regime\n(off training support)",
        "Structural novelty\n(new combo of known parts)",
        "New rules\n(governing structure changes)",
    ]
    ax3.set_yticklabels(gen_labels_explained, fontsize=FS_TICK - 2)
    # Y labels stay dark (readable); row color carries the blue story
    for tick in ax3.get_yticklabels():
        tick.set_color(C_DARK)
        tick.set_fontweight("bold")
    ax3.set_xlabel("capability (harder →)", fontsize=FS_AXIS)
    ax3.set_ylabel("generalization (light blue → deep blue ↑)", fontsize=FS_AXIS)
    ax3.set_title(
        "Distribution ≠ regime ≠ structural novelty ≠ new rules  |  "
        "blue = workable altitude; green = frontier",
        fontsize=FS_AX_TITLE - 4,
    )
    ax3.set_aspect("auto")
    fig3.tight_layout(rect=[0, 0.06, 1, 0.90])
    fig3.text(
        0.5, 0.015,
        "Distribution: P_train ≠ P_test, still same support-ish world.   "
        "Regime: states outside training support.   "
        "Structural novelty: new combinations of familiar parts.   "
        "New rules: the dynamics / grammar themselves change.",
        ha="center", va="bottom", fontsize=FS_TICK - 2, color=C_DARK,
    )


    def _shift_step3_left(_fig):
        fw, _ = _fig.get_size_inches()
        p = ax3.get_position()
        ax3.set_position([p.x0 - 0.4 / fw, p.y0, p.width, p.height])


    return fig3



# -------------------------------------------------------
# FIGURE 4: Levels of generalization
# -------------------------------------------------------
def build_step_4():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        "Step 4: Generalization itself has levels",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax4)

    gen_detail = [
        ("Familiar /\nnew examples", "Same distribution.\nModern DL is strong here."),
        ("New\ndistribution", r"$P_{\mathrm{train}}\neq P_{\mathrm{test}}$." + "\nShift begins to hurt."),
        ("New regime", "States never seen\nin training support."),
        ("Structural\nnovelty", "New combinations of\nfamiliar parts."),
        ("Extrapolation /\nnew rules", "Beyond observed region;\nnew governing structure."),
    ]
    # Align 5 panels to light-blue -> deep-blue bands
    gen_faces = [GEN_COLORS[0], GEN_COLORS[2], GEN_COLORS[3], GEN_COLORS[4], GEN_COLORS[5]]
    gen_edges = [GEN_EDGE[0], GEN_EDGE[2], GEN_EDGE[3], GEN_EDGE[4], GEN_EDGE[5]]
    gen_tcols = [GEN_TEXT[0], GEN_TEXT[2], GEN_TEXT[3], GEN_TEXT[4], GEN_TEXT[5]]
    _s4_n = len(gen_detail)
    _s4_margin = 0.15
    _s4_gap = 0.06 + 0.5 * CM_X  # extend space between cards by 0.5cm
    _s4_w = min(2.15 + 1.0 * CM_X, (12.0 - 2 * _s4_margin - (_s4_n - 1) * _s4_gap) / _s4_n)
    _s4_xs = [_s4_margin + i * (_s4_w + _s4_gap) for i in range(_s4_n)]
    for i, (title, body) in enumerate(gen_detail):
        x = _s4_xs[i]
        face, edge, tcol = gen_faces[i], gen_edges[i], gen_tcols[i]
        round_box(ax4, x, 2.0, _s4_w, 4.8, face, edge, lw=2.6)
        # Vertical boundary separators between levels
        if i > 0:
            ax4.plot([x - _s4_gap / 2, x - _s4_gap / 2], [2.0, 6.8],
                     color="0.1", lw=2.5, zorder=5)
        ax4.text(x + _s4_w / 2, 5.85, title, ha="center",
                 fontsize=FS_TICK, fontweight="bold", color=tcol)
        ax4.text(x + _s4_w / 2, 3.6, body, ha="center", va="center",
                 fontsize=FS_TICK + 1, color=tcol)
        if i < 4:
            arrow(ax4, x + _s4_w + 0.01, 4.4, x + _s4_w + _s4_gap - 0.01, 4.4,
                  "0.2", lw=2.0)

    ax4.text(
        6.0, 0.8,
        "Saying \"AI can generalize\" is ambiguous until you name which level.",
        ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK,
    )
    return fig4



# -------------------------------------------------------
# FIGURE 5: First principles start
# -------------------------------------------------------
def build_step_5():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        r"Step 5: From first principles -- start with $x\rightarrow y$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax5)

    round_box(ax5, 3.5, 5.5, 5.0, 1.8, C_ACTIVE, C_BIN, lw=2.4)
    ax5.text(
        6.0, 6.4,
        r"Simplest learning problem:  $x \rightarrow y$",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold",
    )

    flow = [
        (0.5, "Recognition", r"$f(x)\rightarrow y$"),
        (2.7, "Representation", r"$x\rightarrow z\rightarrow y$"),
        (4.9, "Prediction", r"$x_t\rightarrow x_{t+1}$"),
        (7.1, "Generation", r"$z\rightarrow x$"),
        (9.3, "Inference+", r"$x\rightarrow\hat z\rightarrow\cdots$"),
    ]
    for i, (x0, title, form) in enumerate(flow):
        round_box(ax5, x0, 2.4, 2.0, 2.4, C_NEW, "#2874a6", lw=1.6)
        ax5.text(x0 + 1.0, 4.1, title, ha="center", fontsize=FS_TICK,
                 fontweight="bold", color="#2874a6")
        ax5.text(x0 + 1.0, 3.2, form, ha="center", fontsize=FS_TICK)
        if i < len(flow) - 1:
            arrow(ax5, x0 + 2.05, 3.5, x0 + 2.55, 3.5, C_DARK, lw=1.8)

    round_box(ax5, 0.5, 0.4, 11.0, 1.6, C_MUTE, "0.45", lw=1.6)
    ax5.text(
        6.0, 1.2,
        "Each step adds a demand: structure, time, latent causes, actions, or new knowledge.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )
    return fig5



# -------------------------------------------------------
# FIGURE 6: Interactive capability walk
# -------------------------------------------------------
def build_step_6():
    fig6, (ax6a, ax6b) = new_figure(1, 2)
    fig6.suptitle(
        "Step 6: Walk the capability ladder one by one",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig6.subplots_adjust(bottom=0.16, wspace=0.28)

    LEVEL_DETAIL = [
        ("Recognition",
         "Learn $f_\\theta(x)\\approx y$.\n"
         "Map observation → answer.\n\n"
         "Can be accurate without\n\"understanding.\"\n\n"
         "Assumes test ≈ train."),
        ("Representation",
         "Insert learned $z$:\n"
         "$x\\rightarrow z\\rightarrow y$.\n\n"
         "Discover a useful internal\ncoordinate system.\n\n"
         "Self-supervised learning lives here."),
        ("Prediction",
         "Target is a future:\n"
         "$x_t\\rightarrow x_{t+1}$.\n\n"
         "Forces structure about\nhow the world changes."),
        ("Generation",
         "Reverse direction:\n"
         "learn $P(x)$ or $P(x|c)$,\n"
         "then sample new instances.\n\n"
         "Convincing ≠ grounded."),
        ("Inference",
         "Recover hidden state:\n"
         "$x\\rightarrow\\hat z$.\n\n"
         "Inverse problems,\nworld-state estimation."),
        ("Simulation",
         "Evolve a state:\n"
         "$z_{t+1}=F(z_t)$ repeatedly.\n\n"
         "Coherence over many steps\nis much harder than one-step."),
        ("Planning",
         "Introduce actions $a_t$.\n"
         "Choose sequences by\ncounterfactual futures."),
        ("Control",
         "Close the loop:\n"
         "observe → estimate → act\n→ observe again.\n\n"
         "The world reacts."),
        ("Discovery",
         "Find unknown structure:\n"
         "hypothesis → experiment\n→ new observations.\n\n"
         "Beyond answers contained\nin the original data."),
    ]


    def draw_cap(k):
        ax6a.clear(); ax6b.clear()
        k = int(np.clip(k, 0, len(CAPABILITIES) - 1))
        _s6_dy = 1.0 * CM_Y_STEP6  # move both panels down 1cm
        ax6a.set_xlim(0, 10); ax6a.set_ylim(0, 10); ax6a.axis("off")
        ax6a.text(5.0, 9.55 - _s6_dy + 1.5 * CM_Y_STEP6, "capability",
                  ha="center", va="center",
                  fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK,
                  clip_on=False)
        for i, (name, q, _) in enumerate(CAPABILITIES):
            y = 9.0 - _s6_dy - i * 1.0
            face = C_ACTIVE if i == k else C_MUTE
            edge = C_ACCENT if i == k else "0.55"
            round_box(ax6a, 0.5, y, 9.0, 0.85, face, edge, lw=1.8 if i == k else 1.2)
            ax6a.text(5.0, y + 0.42, f"{i + 1}. {name}", ha="center", va="center",
                      fontsize=FS_TICK + (1 if i == k else 0),
                      fontweight="bold", color=C_DARK if i == k else "0.4")

        name, q, form = CAPABILITIES[k]
        detail = LEVEL_DETAIL[k][1]
        ax6b.set_xlim(0, 10); ax6b.set_ylim(0, 10); ax6b.axis("off")
        _s6_box_y = 0.8 - _s6_dy
        patch = round_box(ax6b, 0.4, _s6_box_y, 9.2, 8.4, C_NEW, "#2874a6", lw=2.2)
        patch.set_clip_on(False)
        ax6b.text(5.0, 8.4 - _s6_dy, name, ha="center", fontsize=FS_AX_TITLE - 2,
                  fontweight="bold", color=C_DARK, clip_on=False)
        ax6b.text(5.0, 7.4 - _s6_dy, q, ha="center", fontsize=FS_ANNOT, style="italic",
                  color="#2874a6", clip_on=False)
        ax6b.text(5.0, 6.5 - _s6_dy, form, ha="center", fontsize=FS_ANNOT + 1,
                  fontweight="bold", color=C_FUN, clip_on=False)
        ax6b.text(5.0, 3.6 - _s6_dy, detail, ha="center", va="center",
                  fontsize=FS_ANNOT, clip_on=False)


    ax_c = fig6.add_axes([0.30, 0.05, 0.40, 0.04])
    ax_p = fig6.add_axes([0.14, 0.045, 0.10, 0.045])
    ax_n = fig6.add_axes([0.76, 0.045, 0.10, 0.045])
    sl_c = Slider(ax_c, "level", 0, len(CAPABILITIES) - 1, valinit=0, valstep=1, color=C_FUN)
    btn_p = Button(ax_p, "Prev", color=C_MUTE, hovercolor=C_NEW)
    btn_n = Button(ax_n, "Next", color=C_MUTE, hovercolor=C_GOAL)


    def _set_c(v):
        v = int(np.clip(v, 0, len(CAPABILITIES) - 1))
        if int(sl_c.val) == v:
            draw_cap(v); fig6.canvas.draw_idle()
        else:
            sl_c.set_val(v)


    def _uc(_=None):
        draw_cap(sl_c.val); fig6.canvas.draw_idle()


    def _prev(_=None):
        _set_c(int(sl_c.val) - 1)


    def _next(_=None):
        _set_c(int(sl_c.val) + 1)


    sl_c.on_changed(_uc)
    btn_p.on_clicked(_prev)
    btn_n.on_clicked(_next)
    _uc()
    print("Step 6: walk capability levels.")
    return fig6



# -------------------------------------------------------
# FIGURE 7: Recognition in detail
# -------------------------------------------------------
def build_step_7():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        r"Step 7: Level 1 in detail -- Recognition is $x\rightarrow y$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax7)

    round_box(ax7, 0.5, 5.4, 11.0, 2.1, C_NEW, "#2874a6", lw=2.0)
    ax7.text(
        6.0, 6.45,
        r"Learn $f_\theta$ from examples $(x_i,y_i)$ by minimizing loss."
        "\n"
        "One-sentence definition: map an observation to an answer.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )

    round_box(ax7, 0.5, 2.7, 5.3, 2.3, C_GOAL, C_OK, lw=1.8)
    ax7.text(3.15, 4.3, "Can answer", ha="center",
             fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax7.text(3.15, 3.4, '"Is this a cat?"', ha="center",
             fontsize=FS_ANNOT + 1, fontweight="bold")

    round_box(ax7, 6.2, 2.7, 5.3, 2.3, C_WARN, C_ACCENT, lw=1.8)
    ax7.text(8.85, 4.3, "Need not answer", ha="center",
             fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT)
    ax7.text(8.85, 3.4, '"Why is this a cat?"', ha="center",
             fontsize=FS_ANNOT + 1, fontweight="bold")

    round_box(ax7, 0.5, 0.4, 11.0, 1.9, C_MUTE, "0.45", lw=1.6)
    ax7.text(
        6.0, 1.35,
        "High recognition accuracy ≠ human-like concept.\n"
        "That gap motivates Representation -- learn what makes the answer possible.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )
    return fig7



# -------------------------------------------------------
# FIGURE 8: Where the frontier begins
# -------------------------------------------------------
def build_step_8():
    fig8, ax8 = new_figure()
    fig8.suptitle(
        "Step 8: Where the hard frontier begins",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax8)

    # ladder on left
    for i, name in enumerate([c[0] for c in CAPABILITIES][::-1]):
        y = 0.5 + i * 0.8
        face = C_WARN if i >= 5 else (C_NEW if i >= 3 else C_GOAL)
        edge = C_ACCENT if i >= 5 else ("#2874a6" if i >= 3 else C_OK)
        round_box(ax8, 0.5, y, 4.0, 0.7, face, edge, lw=1.4)
        ax8.text(2.5, y + 0.35, name, ha="center", va="center",
                 fontsize=FS_TICK, fontweight="bold", color=edge)

    round_box(ax8, 5.2, 4.6, 6.3, 2.8, C_GOAL, C_OK, lw=2.0)
    ax8.text(8.35, 6.5, "Often strong today", ha="center",
             fontsize=FS_ANNOT + 1, fontweight="bold", color=C_OK)
    ax8.text(
        8.35, 5.5,
        "Recognition / representation /\n"
        "short-horizon prediction /\n"
        "generation in-distribution.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax8, 5.2, 0.7, 6.3, 3.4, C_WARN, C_ACCENT, lw=2.0)
    ax8.text(8.35, 3.4, "Frontier pressure", ha="center",
             fontsize=FS_ANNOT + 1, fontweight="bold", color=C_ACCENT)
    ax8.text(
        8.35, 2.0,
        "Long-horizon simulation,\n"
        "closed-loop control,\n"
        "extrapolation,\n"
        "true discovery.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )
    return fig8



# -------------------------------------------------------
# FIGURE 9: Expressed vs unexpressed
# -------------------------------------------------------
def build_step_9():
    fig9, ax9 = new_figure()
    fig9.suptitle(
        "Step 9: Connect to expressed vs unexpressed knowledge",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax9)

    round_box(ax9, 0.5, 4.4, 5.3, 3.1, C_NEW, "#2874a6", lw=2.2)
    ax9.text(3.15, 6.7, "Expressed knowledge", ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color="#2874a6")
    ax9.text(
        3.15, 5.4,
        "Patterns already present\nin historical data / text.\n"
        "DL is extraordinarily good\nat absorbing and remixing them.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )

    round_box(ax9, 6.2, 4.4, 5.3, 3.1, C_GOAL, C_OK, lw=2.2)
    ax9.text(8.85, 6.7, "Unexpressed knowledge", ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    ax9.text(
        8.85, 5.4,
        "Structure not explicitly\ncontained in the corpus.\n"
        "Needs interaction, experiment,\ncounterfactuals, new rules.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )

    round_box(ax9, 0.5, 0.5, 11.0, 3.4, C_MUTE, "0.45", lw=1.6)
    ax9.text(
        6.0, 2.2,
        "The frontier is not only \"higher tasks.\"\n"
        "It is: how much underlying structure can the model discover and exploit\n"
        "when its experience no longer contains the answer?",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )
    return fig9



# -------------------------------------------------------
# FIGURE 10: Takeaway
# -------------------------------------------------------
def build_step_10():
    fig10, axes10 = new_figure(1, 3)
    fig10.suptitle(
        "Step 10: Capability ladder  x  generalization axis",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    panels = [
        (axes10[0], "Capability",
         "recognize → represent →\npredict → generate →\ninfer → simulate →\nplan → control → discover",
         C_NEW, "#2874a6"),
        (axes10[1], "Generalization",
         "familiar → new examples →\nnew distribution →\nnew regime →\nextrapolation / new rules",
         C_GOAL, C_OK),
        (axes10[2], "Frontier",
         "Not just bigger models.\n"
         "Discover and use structure\nthat was not handed over\nin the training record.",
         "#f5eef8", C_FUN),
    ]
    for ax, title, body, face, edge in panels:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        round_box(ax, 0.4, 1.2, 9.2, 7.5, face, edge, lw=2.2)
        ax.text(5.0, 7.8, title, ha="center", fontsize=FS_AX_TITLE - 2,
                fontweight="bold", color=edge)
        ax.text(5.0, 4.3, body, ha="center", va="center", fontsize=FS_ANNOT)
    return fig10




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
])

print("\n" + "=" * 60)
print("FUTURE CHALLENGES FOR DEEP LEARNING")
print("=" * 60)
print("""
Hierarchy (capability):
  Recognition -> Representation -> Prediction -> Generation
  -> Inference -> Simulation -> Planning -> Control -> Discovery

Generalization is vertical, not a rung:
  familiar -> new examples -> new distribution -> new regime
  -> structural novelty -> extrapolation / new rules

True picture:
  Capability x Generalization

Frontier:
  discover and exploit structure when experience no longer contains the answer.
""")
