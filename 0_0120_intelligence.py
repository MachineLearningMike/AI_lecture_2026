"""
0120_intelligence.py

A short opening on what we mean by intelligence in this course.

Sketch (refined):
  1. What is intelligence? Must it be human? Must it come from consciousness?
  2. Working definition: Intelligence = making the right choice.
  3. That already assumes a goal / objective / aim.
  4. AI seeks right choices for a given goal.
  5. AI vs other sciences/tech: who does the choosing.
  6. More on intelligence in the sessions ahead.
  7. Consciousness: matter and consciousness; felt experience as the frontier; open questions.
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
    run_slide_deck,
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
C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_OLD = '#fce4d6'
C_WARN = '#f8d7da'


# -------------------------------------------------------
# FIGURE 1: Opening questions
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        'Step 1: What do we mean by intelligence?',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax1)

    round_box(ax1, 0.7, 5.4, 10.6, 1.8, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 6.3,
        'What is intelligence?',
        ha='center', va='center', fontsize=FS_FIGURE_TITLE,
        fontweight='bold', color='white',
    )

    questions = [
        (0.7, 1.2, 5.2, 3.6, 'Must it belong to humans?',
         'Animals navigate.\nMachines optimize.\nOrganizations decide.\n\n'
         'Human-looking is not\nthe same as intelligent.'),
        (6.1, 1.2, 5.2, 3.6, 'Must it come from consciousness?',
         'Felt experience is one thing.\n'
         'Good choosing is another.\n\n'
         'We can study right choices\n'
         'without settling the mind.'),
    ]
    for x, y, w, h, title, body in questions:
        round_box(ax1, x, y, w, h, C_NEW, '#2874a6', lw=2.2)
        ax1.text(x + w / 2, y + h - 0.45, title, ha='center', va='top',
                 fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_DARK)
        ax1.text(x + w / 2, y + 1.45, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 1)

    return fig1



# -------------------------------------------------------
# FIGURE 2: Working definition
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        'Step 2: A working definition for this course',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax2)

    round_box(ax2, 1.2, 4.6, 9.6, 2.6, C_DARK, C_DARK, lw=0)
    ax2.text(
        6.0, 5.9,
        'Intelligence  =  the capacity for right choice',
        ha='center', va='center', fontsize=FS_FIGURE_TITLE - 2,
        fontweight='bold', color='white',
    )

    trio = [
        (0.7, 1.1, 3.4, 2.9, 'Choice',
         'Among options,\npick one action\nor one belief.'),
        (4.3, 1.1, 3.4, 2.9, 'Right',
         'Better for the aim\nthan the alternatives\n(not mystical).'),
        (7.9, 1.1, 3.4, 2.9, 'Capacity',
         'A system that can\ndo this repeatedly,\nnot once by luck.'),
    ]
    for x, y, w, h, title, body in trio:
        round_box(ax2, x, y, w, h, C_GOAL, C_OK, lw=2.2)
        ax2.text(x + w / 2, y + h - 0.4, title, ha='center', va='top',
                 fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_OK)
        ax2.text(x + w / 2, y + 1.15, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 1)

    return fig2



# -------------------------------------------------------
# FIGURE 3: Assumes a goal
# -------------------------------------------------------
def build_step_3():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        'Step 3: "Right" already assumes a goal',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax3)

    round_box(ax3, 0.7, 5.2, 10.6, 2.0, C_NEW, '#2874a6', lw=2.5)
    ax3.text(
        6.0, 6.2,
        'No goal -- no yardstick for "right" or "wrong."',
        ha='center', va='center', fontsize=FS_AX_TITLE,
        fontweight='bold', color=C_DARK,
    )

    goals = [
        (0.7, 1.2, 3.4, 3.4, 'Goal',
         'What should get\nbetter?'),
        (4.3, 1.2, 3.4, 3.4, 'Objective',
         'How do we score\nsuccess?'),
        (7.9, 1.2, 3.4, 3.4, 'Aim',
         'Where is the\nprocess pointed?'),
    ]
    for x, y, w, h, title, body in goals:
        round_box(ax3, x, y, w, h, C_MUTE, '0.45', lw=1.8)
        ax3.text(x + w / 2, y + h - 0.5, title, ha='center', va='top',
                 fontsize=FS_AX_TITLE, fontweight='bold', color='#333333')
        ax3.text(x + w / 2, y + 1.4, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 2)

    return fig3



# -------------------------------------------------------
# FIGURE 4: What AI does
# -------------------------------------------------------
def build_step_4():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        'Step 4: What artificial intelligence seeks',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax4)

    # Flow: goal -> options -> right choice
    round_box(ax4, 0.6, 4.6, 3.2, 2.4, C_OLD, '#e67e22', lw=2.2)
    ax4.text(2.2, 6.3, 'Given a goal', ha='center', va='center',
             fontsize=FS_AX_TITLE - 2, fontweight='bold', color='#a04000')
    ax4.text(2.2, 5.35, 'loss, reward,\ntarget behavior…', ha='center',
             va='center', fontsize=FS_ANNOT + 1)

    arrow(ax4, 3.9, 5.8, 4.4, 5.8, C_DARK, lw=2.8)

    round_box(ax4, 4.5, 4.6, 3.0, 2.4, C_MUTE, '0.45', lw=1.8)
    ax4.text(6.0, 6.3, 'Options', ha='center', va='center',
             fontsize=FS_AX_TITLE - 2, fontweight='bold')
    ax4.text(6.0, 5.35, 'actions,\npredictions,\nplans…', ha='center',
             va='center', fontsize=FS_ANNOT + 1)

    arrow(ax4, 7.6, 5.8, 8.1, 5.8, C_DARK, lw=2.8)

    round_box(ax4, 8.2, 4.6, 3.2, 2.4, C_GOAL, C_OK, lw=2.5)
    ax4.text(9.8, 6.3, 'Right choice', ha='center', va='center',
             fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_OK)
    ax4.text(9.8, 5.35, 'the move that\nserves the goal', ha='center',
             va='center', fontsize=FS_ANNOT + 1)

    round_box(ax4, 0.6, 1.0, 10.8, 3.0, C_DARK, C_DARK, lw=0)
    ax4.text(
        6.0, 2.9,
        'AI seeks right choices for a given goal.',
        ha='center', va='center', fontsize=FS_FIGURE_TITLE - 4,
        fontweight='bold', color='white',
    )
    ax4.text(
        6.0, 1.75,
        'Not "become conscious."\n'
        'Not "imitate a person."\n'
        'Choose well under an objective we can state and improve.',
        ha='center', va='center', fontsize=FS_ANNOT + 2, color='#d6eaf8',
    )

    return fig4



# -------------------------------------------------------
# FIGURE 5: AI vs other sciences & technology
# -------------------------------------------------------
def build_step_5():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        'Step 5: Same aim -- different who does the choosing',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax5)

    round_box(ax5, 0.6, 5.6, 10.8, 1.7, C_MUTE, '0.45', lw=1.8)
    ax5.text(
        6.0, 6.45,
        'Science and technology also seek right choices for goals.',
        ha='center', va='center', fontsize=FS_AX_TITLE - 1,
        fontweight='bold', color='#333333',
    )

    # Left: other sciences/tech
    round_box(ax5, 0.5, 1.0, 5.3, 4.2, C_OLD, '#e67e22', lw=2.5)
    ax5.text(
        3.15, 4.7, 'Other sciences and technology',
        ha='center', va='center', fontsize=FS_AX_TITLE - 2,
        fontweight='bold', color='#a04000',
    )
    ax5.text(
        3.15, 2.7,
        'Humans invent theories, tools,\n'
        'and procedures.\n\n'
        'People then apply those tools\n'
        'to choose well (design, diagnose,\n'
        'control, prove, optimize by hand).\n\n'
        'The intelligence stays mostly\n'
        'in the human user of the method.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    # Right: AI
    round_box(ax5, 6.2, 1.0, 5.3, 4.2, C_NEW, '#2874a6', lw=2.5)
    ax5.text(
        8.85, 4.7, 'Artificial intelligence',
        ha='center', va='center', fontsize=FS_AX_TITLE - 2,
        fontweight='bold', color=C_DARK,
    )
    ax5.text(
        8.85, 2.7,
        'We still set the goal.\n\n'
        'But the system itself learns\n'
        'or searches how to choose --\n'
        'often from data, at scale,\n'
        'without a full hand-written recipe.\n\n'
        'The intelligence is built into\n'
        'the choice machinery.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    return fig5



# -------------------------------------------------------
# FIGURE 6: Preview of what comes next
# -------------------------------------------------------
def build_step_6():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        'Step 6: More on intelligence in the sessions ahead',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax6)

    round_box(ax6, 0.7, 5.0, 10.6, 2.2, C_NEW, '#2874a6', lw=2.5)
    ax6.text(
        6.0, 6.1,
        'This was only a sketch.',
        ha='center', va='center', fontsize=FS_AX_TITLE + 2,
        fontweight='bold', color=C_DARK,
    )

    ahead = [
        (0.7, 1.2, 3.4, 3.2, 'Goals and losses',
         'How "right" becomes\na number we can\noptimize.'),
        (4.3, 1.2, 3.4, 3.2, 'Learning to choose',
         'Gradients, search,\nand feedback that\nimprove decisions.'),
        (7.9, 1.2, 3.4, 3.2, 'Systems that act',
         'Models, training,\nand deployment as\nchoice machinery.'),
    ]
    for x, y, w, h, title, body in ahead:
        round_box(ax6, x, y, w, h, C_GOAL, C_OK, lw=2.2)
        ax6.text(x + w / 2, y + h - 0.45, title, ha='center', va='top',
                 fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_OK)
        ax6.text(x + w / 2, y + 1.25, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 1)

    return fig6



# -------------------------------------------------------
# FIGURE 7: Consciousness
# -------------------------------------------------------
def build_step_7():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        'Step 7: Consciousness',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax7)

    # Two ingredients
    round_box(ax7, 0.5, 6.35, 5.35, 1.15, C_OLD, '#e67e22', lw=2.0)
    ax7.text(3.175, 6.925, 'Matter', ha='center', va='center',
             fontsize=FS_AX_TITLE - 2, fontweight='bold', color='#a04000')

    round_box(ax7, 6.15, 6.35, 5.35, 1.15, C_NEW, '#2874a6', lw=2.0)
    ax7.text(8.825, 6.925, 'Consciousness', ha='center', va='center',
             fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_DARK)

    ax7.text(
        6.0, 5.95,
        'A convenient view: two interacting fundamental ingredients.',
        ha='center', va='center', fontsize=FS_ANNOT, color='#333333',
    )

    # What consciousness is / frontier
    round_box(ax7, 0.5, 3.85, 5.35, 1.85, C_MUTE, '0.45', lw=1.8)
    ax7.text(3.175, 5.3, 'Consciousness is \u2026', ha='center', va='center',
             fontsize=FS_AX_TITLE - 3, fontweight='bold', color='#333333')
    ax7.text(
        3.175, 4.45,
        'We leave what it is open.\n'
        'Not settled in this course.',
        ha='center', va='center', fontsize=FS_ANNOT,
    )

    round_box(ax7, 6.15, 3.85, 5.35, 1.85, C_GOAL, C_OK, lw=2.2)
    ax7.text(8.825, 5.3, 'Frontier in contact with others',
             ha='center', va='center',
             fontsize=FS_AX_TITLE - 4, fontweight='bold', color=C_OK)
    ax7.text(
        8.825, 4.45,
        'Felt experience:\n'
        'feeling / experiencing\n'
        'subjectively -- what it is like.',
        ha='center', va='center', fontsize=FS_ANNOT - 1,
    )

    # Stone / robot thought experiment
    round_box(ax7, 0.5, 2.05, 11.0, 1.55, C_WARN, C_ACCENT, lw=2.0)
    ax7.text(
        6.0, 3.2,
        'A stone or smart robot rolls downhill, hits a rock hard, and scatters.',
        ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
        color='#922b21',
    )
    ax7.text(
        6.0, 2.45,
        'We do not ask "Are you OK?" or feel sympathy --\n'
        'we believe it does not feel what it is like to hit.',
        ha='center', va='center', fontsize=FS_ANNOT, color='#641e16',
    )

    # Beliefs + open questions
    beliefs = [
        (0.5, 0.2, 3.5, 1.65, 'No feeling, no knowing?',
         'If it cannot feel,\ncan it understand and know?'),
        (4.25, 0.2, 3.5, 1.65, 'Feelings can reflect truth',
         'A person\'s feelings\nmay reflect the truth.'),
        (8.0, 0.2, 3.5, 1.65, 'Open questions',
         'Can machines have consciousness?\n'
         'Are we conscious machines\n'
         'simulated by intelligent giants?'),
    ]
    for x, y, w, h, title, body in beliefs:
        round_box(ax7, x, y, w, h, C_DARK, C_DARK, lw=0)
        ax7.text(x + w / 2, y + h - 0.28, title, ha='center', va='top',
                 fontsize=FS_ANNOT, fontweight='bold', color='#f9e79f')
        ax7.text(x + w / 2, y + 0.55, body, ha='center', va='center',
                 fontsize=FS_ANNOT - 2, color='#d6eaf8')

    return fig7




run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
    build_step_6,
    build_step_7,
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("INTELLIGENCE -- OPENING SKETCH")
print("=" * 60)
print("""
Questions (Step 1):
  What is intelligence?
  Must it belong to humans?
  Must it come from consciousness?

Working definition (Step 2):
  Intelligence = the capacity for right choice.

Assumption (Step 3):
  "Right" already requires a goal / objective / aim.

AI (Step 4):
  AI seeks right choices for a given goal --
  not consciousness, and not human imitation as the definition.

AI vs other science & tech (Step 5):
  Both seek right choices for goals.
  Other fields: humans apply methods to choose.
  AI: the system itself learns/searches how to choose.

Next (Step 6):
  Later sessions develop goals, learning, and systems that choose.

Consciousness (Step 7):
  Convenient dual view: matter and consciousness as two
  interacting fundamental ingredients.
  Consciousness itself is left open ("..."); its frontier in
  contact with others is felt experience -- feeling / experiencing
  subjectively (what it is like).
  We withhold sympathy from a stone or smart robot that scatters
  on impact because we believe it does not feel what it is like to hit.
  Common beliefs: without feeling, no understanding and knowing;
  a person's feelings can reflect the truth.
  Open: Can machines have consciousness?
  Are we conscious machines simulated by intelligent giants?
""")
