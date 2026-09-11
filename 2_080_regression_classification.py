"""
0310_regression_classification.py

Regression vs classification -- intuition, and which loss embraces which.

Sketch:
  1. Two kinds of questions we ask a model
  2. Regression -- predict a continuous amount
  3. Classification -- choose among discrete labels
  4. Which loss embraces which -- and why
  5. Side-by-side decision guide
  6. Continuous classification = regression?

Companion to 190_loss_functions.py (taxonomy + CE),
191_Mean_Squared_Error.py (MSE), 0400_loss_functions.py.
"""

from __future__ import annotations

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
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as patches
import numpy as np

np.random.seed(11)
RNG = np.random.default_rng(11)


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
    rect = FancyBboxPatch(
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


C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_OLD = '#fce4d6'
C_BIN = '#fdebd0'
C_MULTI = '#e8daef'


# -------------------------------------------------------
# FIGURE 1: Two kinds of questions
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        'Step 1: Two kinds of questions we ask a model',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax1)

    round_box(ax1, 0.4, 6.3, 11.2, 1.3, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 6.95,
        'Before choosing a loss, ask: what kind of answer do we want?',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
        fontweight='bold', color='white',
    )

    round_box(ax1, 0.5, 1.4, 5.3, 4.5, C_NEW, '#2874a6', lw=2.4)
    ax1.text(3.15, 5.4, 'Regression', ha='center',
             fontsize=FS_AX_TITLE - 1, fontweight='bold', color=C_DARK)
    ax1.text(
        3.15, 3.5,
        'Question:\n'
        '"How much?" / "How many?"\n\n'
        'Answer lives on a continuum.\n'
        'Examples: price, temperature,\n'
        'age, demand, risk score.\n\n'
        'Wrongness is about distance\n'
        'between numbers.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    round_box(ax1, 6.2, 1.4, 5.3, 4.5, C_GOAL, C_OK, lw=2.4)
    ax1.text(8.85, 5.4, 'Classification', ha='center',
             fontsize=FS_AX_TITLE - 1, fontweight='bold', color=C_OK)
    ax1.text(
        8.85, 3.5,
        'Question:\n'
        '"Which one?"\n\n'
        'Answer is a discrete label.\n'
        'Examples: spam/not spam,\n'
        'cat/dog/bird, disease class.\n\n'
        'Wrongness is about choosing\n'
        'the wrong category.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    ax1.text(
        6.0, 0.55,
        'Same features x can feed either task. The output type decides the game.',
        ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='0.55', alpha=0.95),
    )
    return fig1



# -------------------------------------------------------
# FIGURE 2: Regression intuition + picture
# -------------------------------------------------------
# Left panel +1 inch broader WITHOUT shrinking the right panel:
# keep right absolute width; grow whole figure by 1 inch.
_fw2, _fh2 = screen_figsize()
_usable2 = _fw2 * 0.92
_right_in2 = _usable2 * 0.5
_left_in2 = _usable2 * 0.5 + 1.0
_fw2_wide = (_left_in2 + _right_in2) / 0.92
fig2, (ax2a, ax2b) = plt.subplots(
    1, 2,
    figsize=(_fw2_wide, _fh2),
    dpi=FIGURE_DPI,
    gridspec_kw={'width_ratios': [_left_in2, _right_in2]},
)
fig2.suptitle(
    'Step 2: Regression -- predict a continuous amount',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax2a.set_xlim(0, 10)
ax2a.set_ylim(0, 10)
ax2a.axis('off')
ax2a.set_title('Intuition', fontsize=FS_AX_TITLE - 1)
round_box(ax2a, 0.4, 5.5, 9.2, 4.0, C_NEW, '#2874a6', lw=2.0)
ax2a.text(
    5.0, 7.5,
    'The model outputs a real number y_hat.\n'
    'The target y is also a real number.\n\n'
    'A miss of 0.1 is small; a miss of 10 is large.\n'
    'We need a loss that grades distance on a continuum.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
round_box(ax2a, 0.4, 0.5, 9.2, 4.6, C_GOAL, C_OK, lw=2.0)
ax2a.text(
    5.0, 2.8,
    'So regression embraces Mean Squared Error:\n\n'
    r'$\ell=(\hat y-y)^{2}$'
    '\n\n'
    'Why? Continuous output + continuous distribution story.\n'
    'Squared distance is a natural "how far off" score,\n'
    'smooth for gradient descent.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

# Right: scatter + fitted line
x = np.linspace(-2.2, 2.2, 24)
y = 1.1 * x + 0.35 + 1.6 * RNG.normal(size=x.size)
X = np.column_stack([x, np.ones_like(x)])
w, *_ = np.linalg.lstsq(X, y, rcond=None)
xg = np.linspace(-2.5, 2.5, 100)
ax2b.scatter(x, y, s=55, c='#2874a6', alpha=0.9, edgecolors='white',
             linewidths=0.5, zorder=3, label='data (continuous y)')
ax2b.plot(xg, w[0] * xg + w[1], color=C_ACCENT, lw=2.8, label='regression fit')
ax2b.set_xlabel('x', fontsize=FS_AXIS)
ax2b.set_ylabel('y (continuous)', fontsize=FS_AXIS)
ax2b.set_title('Picture: cloud of numbers, one smooth curve', fontsize=FS_AX_TITLE - 2)
ax2b.tick_params(labelsize=FS_TICK)
ax2b.legend(fontsize=FS_LEGEND, loc='upper left')
ax2b.grid(True, alpha=0.3)
ax2b.text(
    0.98, 0.02,
    'No labels like "class A/B".\n'
    'Only how close y_hat is to y.',
    transform=ax2b.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
fig2.tight_layout(rect=[0, 0, 1, 0.94])
# Keep the wider figsize (do not reset to default screen width).
fig2.set_dpi(FIGURE_DPI)
fig2.set_size_inches(_fw2_wide, _fh2, forward=True)
fig2.canvas.draw_idle()
_manager2 = getattr(fig2.canvas, 'manager', None)
if _manager2 is not None:
    try:
        _manager2.window.state('zoomed')
    except Exception:
        try:
            _manager2.window.showMaximized()
        except Exception:
            try:
                _manager2.frame.Maximize(True)
            except Exception:
                pass
add_step_footer(fig2)
plt.show()


# -------------------------------------------------------
# FIGURE 3: Classification intuition + picture
# -------------------------------------------------------
def build_step_2():
    fig3, (ax3a, ax3b) = new_figure(1, 2)
    fig3.suptitle(
        'Step 3: Classification -- choose among discrete labels',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    ax3a.set_xlim(0, 10)
    ax3a.set_ylim(0, 10)
    ax3a.axis('off')
    ax3a.set_title('Intuition', fontsize=FS_AX_TITLE - 1)

    round_box(ax3a, 0.35, 5.2, 9.3, 4.3, C_BIN, '#d68910', lw=2.0)
    ax3a.text(5.0, 8.7, 'Binary classification', ha='center',
              fontsize=FS_ANNOT + 2, fontweight='bold', color='#9a7d0a')
    ax3a.text(
        5.0, 6.9,
        'Two labels: 0 / 1  (spam / not spam).\n'
        'Model outputs a probability p in (0,1).\n'
        'Embraces Binary Cross-Entropy.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    round_box(ax3a, 0.35, 0.5, 9.3, 4.3, C_MULTI, '#8e44ad', lw=2.0)
    ax3a.text(5.0, 4.0, 'Multi-class classification', ha='center',
              fontsize=FS_ANNOT + 2, fontweight='bold', color='#6c3483')
    ax3a.text(
        5.0, 2.2,
        'Three or more labels; often one-hot target.\n'
        'Model outputs a probability vector (softmax).\n'
        'Embraces (categorical) Cross-Entropy.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    # Right: two clusters
    c0 = RNG.normal([-1.2, 0.0], 0.45, size=(40, 2))
    c1 = RNG.normal([1.2, 0.1], 0.45, size=(40, 2))
    ax3b.scatter(c0[:, 0], c0[:, 1], s=42, c='#2874a6', alpha=0.85,
                 edgecolors='white', linewidths=0.4, label='class 0')
    ax3b.scatter(c1[:, 0], c1[:, 1], s=42, c=C_OK, alpha=0.85,
                 edgecolors='white', linewidths=0.4, label='class 1')
    ax3b.axvline(0, color=C_ACCENT, lw=2.2, ls='--', label='decision boundary')
    ax3b.set_aspect('equal', adjustable='datalim')
    ax3b.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
    ax3b.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
    ax3b.set_title('Picture: regions of labels, not a number line', fontsize=FS_AX_TITLE - 2)
    ax3b.tick_params(labelsize=FS_TICK)
    ax3b.legend(fontsize=FS_LEGEND, loc='upper left')
    ax3b.grid(True, alpha=0.3)
    ax3b.text(
        0.98, 0.02,
        'Wrong class is wrong.\n'
        'Distance in label space\n'
        'is not "how many degrees".',
        transform=ax3b.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    return fig3



# -------------------------------------------------------
# FIGURE 4: Which embraces which -- and why
# -------------------------------------------------------
def build_step_3():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        'Step 4: Which loss embraces which -- and why',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax4)

    # Three flow cards
    flows = [
        (0.35, C_NEW, '#2874a6', 'Regression',
         'continuous y',
         'continuous\ndistribution',
         'Mean Squared Error',
         'Grade distance\nbetween numbers.\nLarge misses cost more.'),
        (4.15, C_BIN, '#d68910', 'Binary class.',
         'y in {0,1}',
         'Bernoulli',
         'Binary Cross-Entropy',
         'Grade probability p\nof the true bit.\nSmooth for GD.'),
        (7.95, C_MULTI, '#8e44ad', 'Multi-class',
         'one-hot / K labels',
         'categorical\ndistribution',
         'Categorical Cross-Entropy',
         'Grade probability\nof the true class\namong K options.'),
    ]
    for x0, face, edge, title, outp, dist, loss, why in flows:
        round_box(ax4, x0, 1.1, 3.55, 6.3, face, edge, lw=2.2)
        ax4.text(x0 + 1.775, 6.9, title, ha='center',
                 fontsize=FS_ANNOT + 2, fontweight='bold', color=edge)
        ax4.text(x0 + 1.775, 5.95, 'Output', ha='center',
                 fontsize=FS_ANNOT, fontweight='bold', color=C_DARK)
        ax4.text(x0 + 1.775, 5.45, outp, ha='center', va='center',
                 fontsize=FS_ANNOT)
        ax4.text(x0 + 1.775, 4.55, 'Probability story', ha='center',
                 fontsize=FS_ANNOT, fontweight='bold', color=C_DARK)
        ax4.text(x0 + 1.775, 3.95, dist, ha='center', va='center',
                 fontsize=FS_ANNOT)
        ax4.text(x0 + 1.775, 3.05, 'Embraces', ha='center',
                 fontsize=FS_ANNOT, fontweight='bold', color=C_DARK)
        ax4.text(x0 + 1.775, 2.55, loss, ha='center', va='center',
                 fontsize=FS_ANNOT + 2, fontweight='bold', color=edge)
        ax4.text(x0 + 1.775, 1.7, why, ha='center', va='center',
                 fontsize=FS_ANNOT)

    ax4.text(
        6.0, 0.45,
        'Rule: output type -> probability model -> loss.\n'
        'Do not put MSE on hard class IDs; do not put CE on raw continuous targets.',
        ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='0.55', alpha=0.95),
    )
    return fig4



# -------------------------------------------------------
# FIGURE 5: Decision guide table
# -------------------------------------------------------
def build_step_4():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        'Step 5: Side-by-side -- choose the embrace deliberately',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    ax5.set_xlim(0, 12)
    ax5.set_ylim(0, 8)
    ax5.axis('off')

    headers = ['If the target is...', 'You are doing...', 'Natural loss', 'Why']
    rows = [
        [
            'a continuous number',
            'Regression',
            'MSE\n(or MAE / Huber)',
            'Wrongness = distance\non a continuum',
        ],
        [
            'one of two labels',
            'Binary\nclassification',
            'Binary\nCross-Entropy\n(+sigmoid)',
            'Wrongness = bad probability\nfor a Bernoulli bit',
        ],
        [
            'one of K>2 labels\n(often one-hot)',
            'Multi-class\nclassification',
            'Categorical\nCross-Entropy\n(+ softmax)',
            'Wrongness = low probability\non the true class',
        ],
    ]

    col_x = [0.35, 3.35, 6.05, 8.55]
    col_w = [2.85, 2.55, 2.35, 3.1]
    header_y, header_h = 6.45, 0.85
    row_h = 1.75
    row0_y = 5.4

    for x, w, h in zip(col_x, col_w, headers):
        ax5.add_patch(patches.Rectangle(
            (x, header_y), w, header_h,
            facecolor='#d6eaf8', edgecolor='#7f8c8d', lw=1.5,
        ))
        ax5.text(x + w / 2, header_y + header_h / 2, h,
                 ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
                 color=C_DARK)

    for r, row in enumerate(rows):
        y = row0_y - r * row_h
        for c, (x, w, cell) in enumerate(zip(col_x, col_w, row)):
            ax5.add_patch(patches.Rectangle(
                (x, y), w, row_h - 0.12,
                facecolor='white', edgecolor='#7f8c8d', lw=1.5,
            ))
            ax5.text(
                x + w / 2, y + (row_h - 0.12) / 2, cell,
                ha='center', va='center', fontsize=FS_ANNOT,
                fontweight='bold' if c in (1, 2) else 'normal',
            )

    ax5.text(
        6.0, 0.45,
        'Classification still uses continuous scores inside (logits),\n'
        'but the loss judges them as probabilities of discrete labels -- not as regression numbers.',
        ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='0.55', alpha=0.95),
    )
    return fig5



# -------------------------------------------------------
# FIGURE 6: Continuous classification = regression?
# -------------------------------------------------------
def build_step_5():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        'Step 6: Continuous classification = regression?',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax6)

    round_box(ax6, 0.35, 6.45, 11.3, 1.25, C_DARK, C_DARK, lw=0)
    ax6.text(
        6.0, 7.05,
        'Thought experiment: classes cover the whole target space,\n'
        'and each class has only a single datapoint.',
        ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
        color='#f9e79f',
    )

    # Left: continuum of singleton "classes"
    round_box(ax6, 0.35, 3.55, 5.5, 2.65, C_MUTE, '0.45', lw=1.5)
    ax6.text(3.1, 5.85, 'Extreme "classification"', ha='center',
             fontsize=FS_TICK, fontweight='bold', color=C_DARK)
    # little class ticks along a line
    ax6.plot([0.85, 5.35], [4.85, 4.85], color='0.35', lw=2.0, solid_capstyle='round')
    for i, yi in enumerate(np.linspace(0.85, 5.35, 9)):
        ax6.plot(yi, 4.85, 'o', color=C_ACCENT if i in (2, 5, 7) else C_DARK, ms=7)
        if i in (2, 5, 7):
            ax6.text(yi, 5.2, rf'$c_{{{i}}}$', ha='center', fontsize=FS_TICK - 1,
                     fontweight='bold', color=C_ACCENT)
    ax6.text(
        3.1, 4.05,
        'Every real value is its own class.\n'
        'One observed point per class.\n'
        r'Choosing a class $=$ choosing a number $y$.',
        ha='center', va='center', fontsize=FS_TICK + 1, fontweight='bold',
    )

    # Right: regression
    round_box(ax6, 6.15, 3.55, 5.5, 2.65, C_GOAL, C_OK, lw=1.8)
    ax6.text(8.9, 5.85, 'That is regression', ha='center',
             fontsize=FS_TICK, fontweight='bold', color=C_OK)
    xs = np.linspace(6.55, 11.0, 40)
    ys = 4.55 + 0.55 * np.sin(1.3 * (xs - 6.55)) + 0.15 * (xs - 8.8)
    ax6.plot(xs, ys, color=C_OK, lw=2.5)
    ax6.scatter(
        [7.1, 8.2, 9.4, 10.4],
        [4.55 + 0.55 * np.sin(1.3 * (np.array([7.1, 8.2, 9.4, 10.4]) - 6.55))
         + 0.15 * (np.array([7.1, 8.2, 9.4, 10.4]) - 8.8)],
        s=45, color=C_DARK, zorder=3,
    )
    ax6.text(
        8.9, 4.0,
        r'Predict a continuous $y$ on a continuum.'
        '\n'
        'No discrete menu left to choose from.',
        ha='center', va='center', fontsize=FS_TICK + 1, fontweight='bold',
        color='#145a32',
    )

    arrow(ax6, 5.95, 4.85, 6.2, 4.85, C_DARK, 2.2)

    # Thought box
    round_box(ax6, 0.35, 0.35, 11.3, 2.95, '#fef9e7', '#b7950b', lw=1.8)
    ax6.text(6.0, 2.9, 'Thought', ha='center',
             fontsize=FS_TICK, fontweight='bold', color='#7d6608')
    ax6.text(
        6.0, 1.55,
        'Intuitively: yes — if labels tile the whole space and each label is a singleton,\n'
        '\"which class?\" collapses to \"which number?\", so continuous classification becomes regression.\n\n'
        'The catch is the geometry of wrongness. Plain categorical CE treats class $3.01$ as unrelated\n'
        'to class $3.00$; MSE knows they are almost the same. Equivalence is real only when the label set\n'
        'carries a continuum metric (or fine ordered bins). Otherwise you reinvented regression — without its ruler.',
        ha='center', va='center', fontsize=FS_TICK + 1, fontweight='bold',
    )
    return fig6




run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("035 REGRESSION vs CLASSIFICATION")
print("=" * 60)
print("""
Regression:  continuous target  ->  continuous distribution story  ->  MSE
Binary:      label in {0,1}     ->  Bernoulli                      ->  Binary CE
Multi-class: one of K labels    ->  categorical (+ softmax)        ->  Categorical Cross-Entropy

Why this embrace?
  The loss must match what "wrong" means for that output type.
  Distance between reals  !=  wrong category probability.

Continuous classification = regression?
  If classes cover the whole target space and each class has one datapoint,
  choosing a class is choosing a number -- classification collapses to regression.
  Caveat: CE on unordered labels does not know nearby numbers are "almost right";
  MSE does. Real equivalence needs a continuum metric on the labels (or fine bins).

See also: 190_loss_functions.py, 191_Mean_Squared_Error.py, 0400_loss_functions.py
""")
