"""
006_continuous_end_to_end.py

The modern deep-learning breakthrough:

    Continuous, language-free representations
    learned by end-to-end models
    effectively eliminate human interventions
    that used to sit between raw input and final output.
"""

import matplotlib.pyplot as plt
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
)
apply_rcparams(plt)
import matplotlib.patches as patches
import numpy as np

np.random.seed(42)


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
    kwargs.setdefault('figsize', screen_figsize())
    kwargs.setdefault('dpi', FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


def show_maximized(fig):
    fig.set_dpi(FIGURE_DPI)
    fig.set_size_inches(screen_figsize(), forward=True)
    fig.canvas.draw_idle()
    manager = getattr(fig.canvas, 'manager', None)
    if manager is not None:
        try:
            manager.window.state('zoomed')
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
    ax.axis('off')


def round_box(ax, x, y, w, h, facecolor, edgecolor='0.2', lw=2.0):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.15',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(rect)
    return rect


def arrow(ax, x1, y1, x2, y2, color='0.25', lw=2.5):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        mutation_scale=18),
    )


# Palette
C_OLD = '#fce4d6'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_DARK = '#1a5276'


# -------------------------------------------------------
# FIGURE 1: The claim
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    'Step 1: The breakthrough claim',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax1)

round_box(ax1, 0.7, 4.6, 10.6, 2.6, C_DARK, C_DARK, lw=0)
ax1.text(
    6.0, 6.3,
    'Modern deep learning broke through by learning',
    ha='center', va='center', fontsize=FS_AX_TITLE + 2,
    fontweight='bold', color='white',
)
ax1.text(
    6.0, 5.35,
    'continuous, language-free representations\n'
    'inside end-to-end models.',
    ha='center', va='center', fontsize=FS_ANNOT + 3, color='#d6eaf8',
)

round_box(ax1, 0.7, 1.2, 10.6, 2.8, C_GOAL, C_OK, lw=2.5)
ax1.text(
    6.0, 3.2, 'Effect',
    ha='center', fontsize=FS_AX_TITLE, fontweight='bold', color=C_OK,
)
ax1.text(
    6.0, 2.1,
    'Human-designed intermediate steps get pushed out of the loop.\n'
    'The model maps raw input → useful output with far less hand-crafting.',
    ha='center', va='center', fontsize=FS_ANNOT + 2,
)

show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Before -- human pipeline
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    'Step 2: Before -- humans sit in the middle of the pipeline',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax2)

stages_old = [
    (0.4, 'Raw\ninput', C_MUTE),
    (2.5, 'Hand\nfeatures', C_OLD),
    (4.6, 'Rules /\nsymbols', C_OLD),
    (6.7, 'Tuned\nmodules', C_OLD),
    (8.8, 'Final\noutput', C_MUTE),
]
for x, title, color in stages_old:
    round_box(ax2, x, 4.2, 1.9, 2.4, color, '#e67e22' if color == C_OLD else '0.4')
    ax2.text(x + 0.95, 5.4, title, ha='center', va='center',
             fontsize=FS_ANNOT + 1, fontweight='bold')
for x in (2.3, 4.4, 6.5, 8.6):
    arrow(ax2, x, 5.4, x + 0.2, 5.4, '#e67e22', lw=2.2)

round_box(ax2, 0.7, 1.0, 10.6, 2.6, C_OLD, '#e67e22')
ax2.text(
    6.0, 2.9, 'Human interventions everywhere',
    ha='center', fontsize=FS_AX_TITLE - 2, fontweight='bold', color='#a04000',
)
ax2.text(
    6.0, 1.85,
    'Choose features. Invent symbols. Write rules. Glue modules.\n'
    'Each hand-off is discrete, language-heavy, and brittle.',
    ha='center', va='center', fontsize=FS_ANNOT + 2,
)

show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Continuous + language-free
# -------------------------------------------------------
fig3, ax3 = new_figure()
fig3.suptitle(
    'Step 3: Continuous, language-free representation',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax3)

round_box(ax3, 0.5, 1.2, 5.3, 5.8, C_OLD, '#e67e22', lw=2.5)
ax3.text(3.15, 6.5, 'Language / symbols', ha='center',
         fontsize=FS_AX_TITLE, fontweight='bold', color='#a04000')
ax3.text(
    3.15, 4.6,
    'Words, categories, hand labels\n'
    'Rules like “if … then …”\n'
    'Discrete slots humans invent\n\n'
    'Good for explaining to people.\n'
    'Bad as a forced middle language\n'
    'for every machine step.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

round_box(ax3, 6.2, 1.2, 5.3, 5.8, C_NEW, '#2874a6', lw=2.5)
ax3.text(8.85, 6.5, 'Continuous vectors', ha='center',
         fontsize=FS_AX_TITLE, fontweight='bold', color=C_DARK)
ax3.text(
    8.85, 4.6,
    'Arrays of real numbers\n'
    'Learned from data, not named by us\n'
    'Smooth: small changes matter\n\n'
    'No need for a human vocabulary\n'
    'between layers.\n'
    'Gradients can flow through.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: End-to-end
# -------------------------------------------------------
fig4, ax4 = new_figure()
fig4.suptitle(
    'Step 4: End-to-end -- one trainable map',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax4)

round_box(ax4, 0.8, 4.8, 2.4, 2.0, C_MUTE)
ax4.text(2.0, 5.8, 'Raw\ninput', ha='center', va='center',
         fontsize=FS_ANNOT + 2, fontweight='bold')

round_box(ax4, 4.2, 4.3, 3.6, 3.0, C_NEW, '#2874a6', lw=2.5)
ax4.text(6.0, 6.5, 'End-to-end model', ha='center',
         fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_DARK)
ax4.text(
    6.0, 5.4,
    'continuous\nlayers &\nrepresentations',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

round_box(ax4, 8.8, 4.8, 2.4, 2.0, C_MUTE)
ax4.text(10.0, 5.8, 'Final\noutput', ha='center', va='center',
         fontsize=FS_ANNOT + 2, fontweight='bold')

arrow(ax4, 3.25, 5.8, 4.15, 5.8, C_DARK, lw=2.8)
arrow(ax4, 7.85, 5.8, 8.75, 5.8, C_DARK, lw=2.8)

# Gradient feedback
ax4.annotate(
    '', xy=(3.4, 4.55), xytext=(8.6, 4.55),
    arrowprops=dict(arrowstyle='->', color=C_OK, lw=2.4,
                    connectionstyle='arc3,rad=-0.25', mutation_scale=16),
)
ax4.text(6.0, 3.85, 'loss → gradients → update everything',
         ha='center', fontsize=FS_ANNOT + 1, color=C_OK, fontweight='bold')

round_box(ax4, 0.7, 1.0, 10.6, 2.4, C_GOAL, C_OK)
ax4.text(
    6.0, 2.7,
    'One objective trains the whole path.\n'
    'No human-designed “middle language” that must be perfect before learning starts.',
    ha='center', va='center', fontsize=FS_ANNOT + 2,
)

show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: What gets eliminated
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    'Step 5: Human interventions that get pushed out',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax5)

items = [
    (0.5, 4.4, 'Hand-crafted\nfeatures'),
    (3.3, 4.4, 'Forced\nsymbolic steps'),
    (6.1, 4.4, 'Brittle\nmodule glue'),
    (8.9, 4.4, 'Manual\nstage tuning'),
    (2.0, 1.5, 'Human-chosen\nmiddle vocab'),
    (5.0, 1.5, 'Stop-and-fix\npipelines'),
    (8.0, 1.5, 'Expert rules\nas hard constraints'),
]
for x, y, text in items:
    round_box(ax5, x, y, 2.5, 2.0, '#f8d7da', C_ACCENT, lw=1.8)
    ax5.text(x + 1.25, y + 1.0, text, ha='center', va='center',
             fontsize=FS_ANNOT, fontweight='bold', color=C_ACCENT)

ax5.text(
    6.0, 0.55,
    'Not “humans disappear from AI.”  Humans move up: goals, data, evaluation, deployment.',
    ha='center', fontsize=FS_ANNOT + 1, style='italic',
)

show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: The compact story
# -------------------------------------------------------
fig6, ax6 = new_figure()
fig6.suptitle(
    'Step 6: Remember this sentence',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax6)

round_box(ax6, 0.6, 3.8, 10.8, 3.2, C_DARK, C_DARK, lw=0)
ax6.text(
    6.0, 6.2,
    'Continuous + language-free + end-to-end',
    ha='center', va='center', fontsize=FS_FIGURE_TITLE - 2,
    fontweight='bold', color='white',
)
ax6.text(
    6.0, 4.85,
    'Let the model learn the representation\n'
    'and the whole input→output map together.',
    ha='center', va='center', fontsize=FS_ANNOT + 3, color='#d6eaf8',
)

round_box(ax6, 0.6, 1.0, 10.8, 2.2, C_NEW, '#2874a6')
ax6.text(
    6.0, 2.1,
    'That is the practical breakthrough of modern deep learning:\n'
    'less human plumbing in the middle, more learning from data.',
    ha='center', va='center', fontsize=FS_ANNOT + 2,
)

show_maximized(fig6)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("CONTINUOUS END-TO-END DEEP LEARNING")
print("=" * 60)
print("""
Breakthrough:
  Continuous, language-free representations
  inside end-to-end models.

Before:
  Humans designed features, symbols, rules, and stage-by-stage glue.

After:
  Raw input -> learned continuous states -> output,
  trained by one loss with gradients through the whole path.

Effect:
  Many human interventions in the middle are eliminated
  (or moved upward to goals, data, and evaluation).
""")
