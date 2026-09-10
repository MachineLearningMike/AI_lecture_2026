"""
0110_lecture_strategy.py

Lecture strategy slides.

Vision / Mission / Strategy / Attitude:
    Vision -- They won't pay you for what AI can do cheaper.
    Mission -- Defeat AI - the awkward generalization of expressed
    historical common human knowledge.
    Strategy -- Bet on irreverent intuition. Master the un-expressible.
    Attitude -- Be resilient.

Baseline:
    Students do not yet have enough math or AI skill,
    but they must deliver AI as soon as possible.

Goal:
    Students gain confidence and fundamental intuition in AI,
    then begin to self-study AI, squeeze LLMs for AI work, and deliver AI.

Lecture gravity:
    Change the gravity of training.
    Begin with delivery and advance toward understanding,
    rather than begin with understanding and advance toward delivery.

Critical success factors:
    Student resilience, and insightful teaching from the lecturer.
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
C_WARN = '#f8d7da'
C_GOAL = '#d4edda'
C_OLD = '#fce4d6'
C_NEW = '#d6eaf8'
C_ACCENT = '#c0392b'
C_OK = '#1e8449'
C_MUTE = '#f4f6f7'
C_DARK = '#1a5276'
C_FUN = '#8e44ad'


# -------------------------------------------------------
# FIGURE -2: Opening quote
# -------------------------------------------------------
def build_step_1():
    fig_q, ax_q = new_figure()
    fig_q.suptitle(
        '',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax_q)

    round_box(ax_q, 1.5, 2.6, 9.0, 3.0, C_DARK, C_DARK, lw=0)
    ax_q.text(
        6.0, 4.35,
        '"AI is not magic."',
        ha='center', va='center',
        fontsize=FS_FIGURE_TITLE + 2, fontweight='bold', color='#f9e79f',
    )
    ax_q.text(
        6.0, 3.25,
        '--- Yoshua Bengio',
        ha='center', va='center',
        fontsize=FS_AX_TITLE - 2, color='#d6eaf8',
    )
    return fig_q


# -------------------------------------------------------
# FIGURE -2: Opening quote
# -------------------------------------------------------
def build_step_2():
    fig_q, ax_q = new_figure()
    fig_q.suptitle(
        '',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax_q)

    round_box(ax_q, 1.5, 2.6, 9.0, 3.0, C_DARK, C_DARK, lw=0)
    ax_q.text(
        6.0, 4.35,
        '"AI is underhyped."',
        ha='center', va='center',
        fontsize=FS_FIGURE_TITLE + 2, fontweight='bold', color='#f9e79f',
    )
    ax_q.text(
        6.0, 3.25,
        '--- Eric Schmidt',
        ha='center', va='center',
        fontsize=FS_AX_TITLE - 2, color='#d6eaf8',
    )
    return fig_q


# -------------------------------------------------------
# FIGURE 0: Vision / Mission / Strategy / Attitude
# -------------------------------------------------------
def build_step_3():
    fig0, ax0 = new_figure()
    fig0.suptitle(
        'Vision / Mission / Strategy / Attitude',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax0)

    round_box(ax0, 0.5, 6.15, 11.0, 1.75, C_DARK, C_DARK, lw=0)
    ax0.text(
        6.0, 7.55, 'VISION',
        ha='center', va='center', fontsize=FS_AX_TITLE - 2,
        fontweight='bold', color='#f9e79f',
    )
    ax0.text(
        6.1, 6.75,
        "You are losing jobs to AI - a fast-growing but awkwad generalization of\n"
        "symbolically expressed historical common acquired knowledge.",
        ha='center', va='center', fontsize=FS_ANNOT + 2,
        fontweight='bold', color='white',
    )

    round_box(ax0, 0.5, 4.15, 11.0, 1.7, C_WARN, C_ACCENT, lw=2.2)
    ax0.text(
        6.0, 5.4, 'MISSION',
        ha='center', va='center', fontsize=FS_AX_TITLE - 2,
        fontweight='bold', color=C_ACCENT,
    )
    ax0.text(
        6.0, 4.65,
        'Transcend (harness, incorporate, and orchestrate) AI\n',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
        fontweight='bold', color='#641e16',
    )

    round_box(ax0, 0.5, 2.05, 11.0, 1.75, C_GOAL, C_OK, lw=2.2)
    ax0.text(
        6.0, 3.45, 'STRATEGY',
        ha='center', va='center', fontsize=FS_AX_TITLE - 2,
        fontweight='bold', color=C_OK,
    )
    ax0.text(
        6.0, 2.65,
        'Bet on being more human, with irreverent, inexpressible intuition\n'
        'born from wild truth. Know AI. Build YOU x AI. Feed AI wisely.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
        fontweight='bold', color='#145a32',
    )

    round_box(ax0, 0.5, 0.30, 11.0, 1.50, '#f5eef8', C_FUN, lw=2.2)
    ax0.text(
        6.0, 1.45, 'ATTITUDE',
        ha='center', va='center', fontsize=FS_AX_TITLE - 2,
        fontweight='bold', color=C_FUN,
    )
    ax0.text(
        6.0, 0.6, 
        'Stay resilient. Stay human.\n',
        ha='center', va='center', fontsize=FS_ANNOT + 3,
        fontweight='bold', color='#4a235a',
    )

    return fig0



# -------------------------------------------------------
# FIGURE 1: Baseline -- the real starting point
# -------------------------------------------------------
def build_step_4():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        'Step 3: Baseline -- where we actually start',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax1)

    round_box(ax1, 0.6, 4.6, 10.8, 2.6, C_WARN, C_ACCENT, lw=2.5)
    ax1.text(
        6.0, 6.5, 'The situation',
        ha='center', va='center', fontsize=FS_AX_TITLE, fontweight='bold',
        color=C_ACCENT,
    )
    ax1.text(
        6.0, 5.55,
        'Students have little math skills.\n'
        'Students have surface-level AI skills.\n'
        'And still -- they want to deliver AI as soon as possible.',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
    )

    round_box(ax1, 0.6, 1.0, 5.0, 3.0, C_MUTE)
    ax1.text(3.1, 3.5, 'Little of', ha='center', fontsize=FS_AX_TITLE - 2,
             fontweight='bold')
    ax1.text(
        3.1, 2.3,
        r'$\bullet$  calculus and probability fluency' + '\n'
        r'$\bullet$  ML vocabulary and habits' + '\n'
        r'$\bullet$  practice delivering systems',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    round_box(ax1, 6.4, 1.0, 5.0, 3.0, C_MUTE)
    ax1.text(8.9, 3.5, 'Urgent demand', ha='center', fontsize=FS_AX_TITLE - 2,
             fontweight='bold')
    ax1.text(
        8.9, 2.3,
        r'$\bullet$  ship demos and prototypes' + '\n'
        r'$\bullet$  talk about AI as if they know it' + '\n'
        r'$\bullet$  learn while already delivering',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    return fig1



# -------------------------------------------------------
# FIGURE 2: Goal -- confidence, intuition, attempts
# -------------------------------------------------------
def build_step_5():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        'Step 4: Goal -- what success looks like',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax2)

    round_box(ax2, 0.6, 5.2, 10.8, 2.0, C_GOAL, C_OK, lw=2.5)
    ax2.text(
        6.0, 6.2,
        'Students leave with confidence and fundamental intuition in AI --\n'
        'then they self-study AI, squeeze LLMs for AI work, and deliver AI.',
        ha='center', va='center', fontsize=FS_ANNOT + 3, fontweight='bold',
        color='#145a32',
    )

    goals = [
        (0.6, 1.2, 3.4, 3.4, 'Confidence',
         'I can start.\nI can ask.\nI can try.'),
        (4.3, 1.2, 3.4, 3.4, 'Intuition',
         'I feel how AI\ncan transform\na business.'),
        (8.0, 1.2, 3.4, 3.4, 'Attempt',
         'I talk about AI.\nI squeeze LLMs.\nI deliver a first version.'),
    ]
    for x, y, w, h, title, body in goals:
        round_box(ax2, x, y, w, h, '#eafaf1', C_OK)
        ax2.text(x + w / 2, y + h - 0.55, title, ha='center', va='top',
                 fontsize=FS_AX_TITLE, fontweight='bold', color=C_OK)
        ax2.text(x + w / 2, y + 1.35, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 2)

    return fig2



# -------------------------------------------------------
# FIGURE 3: Two gravities of training
# -------------------------------------------------------
def build_step_6():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        'Step 5: Two possible gravities of training',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax3)

    # Old path
    round_box(ax3, 0.5, 1.3, 5.2, 5.8, C_OLD, '#e67e22', lw=2.5)
    ax3.text(3.1, 6.6, 'Traditional gravity', ha='center',
             fontsize=FS_AX_TITLE, fontweight='bold', color='#a04000')
    ax3.text(3.1, 5.95, 'Understand  →  then deliver', ha='center',
             fontsize=FS_ANNOT + 2, fontweight='bold')

    old_steps = [
        (5.35, '1. Theory first'),
        (4.45, '2. Math until comfortable'),
        (3.55, '3. Then all historical models'),
        (2.65, '4. Delivery only at the end'),
    ]
    for y, text in old_steps:
        round_box(ax3, 1.0, y - 0.35, 4.2, 0.7, 'white', '#e67e22', lw=1.5)
        ax3.text(3.1, y, text, ha='center', va='center', fontsize=FS_ANNOT + 1)
    arrow(ax3, 3.1, 5.0, 3.1, 4.7, '#e67e22')
    arrow(ax3, 3.1, 4.1, 3.1, 3.8, '#e67e22')
    arrow(ax3, 3.1, 3.2, 3.1, 2.9, '#e67e22')
    ax3.text(3.1, 1.65, 'Risk: late delivery,\nearly freeze',
             ha='center', fontsize=FS_ANNOT, color='#a04000', style='italic')

    # New path
    round_box(ax3, 6.3, 1.3, 5.2, 5.8, C_NEW, '#2874a6', lw=2.5)
    ax3.text(8.9, 6.6, 'Our gravity', ha='center',
             fontsize=FS_AX_TITLE, fontweight='bold', color='#1a5276')
    ax3.text(8.9, 5.95, 'Deliver  →  then deepen understanding', ha='center',
             fontsize=FS_ANNOT + 2, fontweight='bold')

    new_steps = [
        (5.35, '1. Delivery first (pretend, run, modify)'),
        (4.45, '2. Talk about Transformers and LLMs'),
        (3.55, '3. Pull in math as needed'),
        (2.65, '4. Understanding grows in use'),
    ]
    for y, text in new_steps:
        round_box(ax3, 6.8, y - 0.35, 4.2, 0.7, 'white', '#2874a6', lw=1.5)
        ax3.text(8.9, y, text, ha='center', va='center', fontsize=FS_ANNOT + 1)
    arrow(ax3, 8.9, 5.0, 8.9, 4.7, '#2874a6')
    arrow(ax3, 8.9, 4.1, 8.9, 3.8, '#2874a6')
    arrow(ax3, 8.9, 3.2, 8.9, 2.9, '#2874a6')
    ax3.text(8.9, 1.65, 'Gain: early attempts,\nliving intuition',
             ha='center', fontsize=FS_ANNOT, color='#1a5276', style='italic')

    return fig3



# -------------------------------------------------------
# FIGURE 4: Strategy slogan -- change the gravity
# -------------------------------------------------------
def build_step_7():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        'Step 6: Strategy -- change the gravity of training',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax4)

    # Big statement
    round_box(ax4, 0.7, 4.4, 10.6, 2.8, '#1a5276', '#1a5276', lw=0)
    ax4.text(
        6.0, 6.3, 'Begin delivery.',
        ha='center', va='center', fontsize=FS_FIGURE_TITLE + 4,
        fontweight='bold', color='white',
    )
    ax4.text(
        6.0, 5.25, 'Advance toward understanding, fast.',
        ha='center', va='center', fontsize=FS_FIGURE_TITLE,
        fontweight='bold', color='#d6eaf8',
    )

    # Contrast strip
    round_box(ax4, 0.7, 1.2, 5.0, 2.6, C_OLD, '#e67e22')
    ax4.text(3.2, 3.2, 'Not this', ha='center', fontsize=FS_AX_TITLE,
             fontweight='bold', color='#a04000')
    ax4.text(
        3.2, 2.15,
        'Begin understanding,\n'
        'advance toward delivery.',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
    )

    round_box(ax4, 6.3, 1.2, 5.0, 2.6, C_NEW, '#2874a6')
    ax4.text(8.8, 3.2, 'This', ha='center', fontsize=FS_AX_TITLE,
             fontweight='bold', color='#1a5276')
    ax4.text(
        8.8, 2.15,
        'Begin delivery,\n'
        'advance toward understanding, fast.',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
    )

    return fig4



# -------------------------------------------------------
# FIGURE 5: What the lecture does under this gravity
# -------------------------------------------------------
def build_step_8():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        'Step 7: How this lecture follows the strategy',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax5)

    # Timeline path
    stages = [
        (0.5, 'See it run', 'Watch a model\ntrain, or a figure move'),
        (3.4, 'Touch it', 'Change a knob.\nBreak it. Fix it.'),
        (6.3, 'Name it', 'Give the math\nand the vocabulary'),
        (9.2, 'Own it', 'Explain it.\nDeliver a small AI.'),
    ]
    ys = 4.6
    for i, (x, title, body) in enumerate(stages):
        round_box(ax5, x, ys, 2.5, 2.4, C_NEW if i < 2 else C_GOAL,
                  '#2874a6' if i < 2 else C_OK)
        ax5.text(x + 1.25, ys + 1.85, f'{i + 1}. {title}', ha='center',
                 fontsize=FS_ANNOT + 2, fontweight='bold')
        ax5.text(x + 1.25, ys + 0.85, body, ha='center', va='center',
                 fontsize=FS_ANNOT)
        if i < len(stages) - 1:
            arrow(ax5, x + 2.55, ys + 1.2, x + 2.85 + 0.4, ys + 1.2, '#2874a6', lw=2.2)

    round_box(ax5, 0.5, 0.8, 11.0, 3.0, C_MUTE)
    ax5.text(6.0, 3.3, 'Design rule for every module', ha='center',
             fontsize=FS_AX_TITLE - 2, fontweight='bold')
    ax5.text(
        6.0, 2.0,
        'Start from a runnable picture or demo -- then the symbol.\n'
        'Ask for an attempt (talk, tweak, or ship) before perfect mastery.\n'
        'Pull math in when it clarifies what students already saw working.',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
    )

    return fig5



# -------------------------------------------------------
# FIGURE 6: Contract with students
# -------------------------------------------------------
def build_step_9():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        'Step 8: The contract',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax6)

    round_box(ax6, 0.7, 4.3, 10.6, 2.9, '#145a32', '#145a32', lw=0)
    ax6.text(
        6.0, 6.4, 'You do not need to know everything first.',
        ha='center', va='center', fontsize=FS_FIGURE_TITLE - 2,
        fontweight='bold', color='white',
    )
    ax6.text(
        6.0, 5.3,
        'You need to start delivering, keep talking,\n'
        'and let understanding catch up.',
        ha='center', va='center', fontsize=FS_ANNOT + 3, color='#d5f5e3',
    )

    promises = [
        (0.7, 1.0, 3.4, 2.8, 'We provide',
         'Demos, pictures,\nvocabulary, and\njust-enough math.'),
        (4.3, 1.0, 3.4, 2.8, 'You attempt',
         'Run. Modify.\nExplain out loud.\nShip a small thing.'),
        (7.9, 1.0, 3.4, 2.8, 'Together',
         'Confidence grows.\nIntuition forms.\nDelivery becomes normal.'),
    ]
    for x, y, w, h, title, body in promises:
        round_box(ax6, x, y, w, h, '#eafaf1', C_OK)
        ax6.text(x + w / 2, y + h - 0.45, title, ha='center', va='top',
                 fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_OK)
        ax6.text(x + w / 2, y + 1.1, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 1)

    return fig6



# -------------------------------------------------------
# FIGURE 7: Critical success factors
# -------------------------------------------------------
def build_step_10():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        'Step 9: Critical success factors',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax7)

    round_box(ax7, 0.7, 5.5, 10.6, 1.7, '#1a5276', '#1a5276', lw=0)
    ax7.text(
        6.0, 6.35,
        'This gravity only works if both sides hold.',
        ha='center', va='center', fontsize=FS_AX_TITLE + 2,
        fontweight='bold', color='white',
    )

    # Student side
    round_box(ax7, 0.6, 1.0, 5.2, 4.0, C_NEW, '#2874a6', lw=2.5)
    ax7.text(
        3.2, 4.5, '1. Student resilience',
        ha='center', va='center', fontsize=FS_AX_TITLE,
        fontweight='bold', color='#1a5276',
    )
    ax7.text(
        3.2, 2.7,
        'Stay with confusion long enough\n'
        'to try again.\n\n'
        'Keep delivering while the math\n'
        'is still catching up.\n\n'
        'Treat early failures as material\n'
        'for the next attempt -- not as a stop.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    # Lecturer side
    round_box(ax7, 6.2, 1.0, 5.2, 4.0, C_GOAL, C_OK, lw=2.5)
    ax7.text(
        8.8, 4.5, '2. Insightful teaching',
        ha='center', va='center', fontsize=FS_AX_TITLE,
        fontweight='bold', color=C_OK,
    )
    ax7.text(
        8.8, 2.7,
        'Choose demos that reveal the idea\n'
        'before the formula.\n\n'
        'Name what students just saw,\n'
        'then add only the math that clarifies it.\n\n'
        'Protect confidence: challenge hard,\n'
        'but never leave them without a next step.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    # Pull the footnote box down by nearly its own height
    _box_h = 0.7
    _box_y = 0.15 - 0.65 * _box_h
    ax7.set_ylim(-0.55, 8)
    round_box(ax7, 0.6, _box_y, 10.8, _box_h, 'lightyellow', '0.55', lw=1.6)
    ax7.text(
        6.0, _box_y + _box_h / 2,
        'What we teach you today with ease took us years of painful self-study\n'
        'buried in endless AI literature.',
        ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
    )

    return fig7




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

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("LECTURE STRATEGY")
print("=" * 60)
print("""
Vision:
  They won't pay you for what AI can do cheaper.

Mission:
  Defeat AI - the awkward generalization of expressed historical common human knowledge.

Strategy:
  Bet on irreverent intuition. Master the un-expressible.

Attitude:
  Be resilient.

Baseline:
  Students lack enough math and AI skill,
  yet must deliver AI quickly.

Goal:
  Confidence + fundamental intuition in AI,
  then self-study, squeeze LLMs for AI work, and deliver.

Strategy (change the gravity):
  Begin delivery, advance toward understanding, fast.
  NOT: begin understanding, advance toward delivery.

Lecture habit:
  runnable demo  ->  language  ->  math as needed  ->  student attempt

Critical success factors:
  1. Student resilience -- keep attempting while understanding catches up.
  2. Insightful teaching -- reveal with demos, name clearly, add math that clarifies.
""")
