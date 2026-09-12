"""
Why we study distributions -- from the lecture strategy.

Sketch:
  1. Begin with the client's business.
  2. Prefer their data over deep immersion in domain lore.
  3. Looking only at the data, everything is a distribution of points.
  4. Knowledge itself can be seen as a distribution
     (e.g. tree recognition = labels or likelihoods over objects).
  5. Next-word distribution after a given context.
  6. Next-sentence distribution after a given context.
  7. Everyday images as a distribution.
  8. Dog vs cat photo clusters (two modes).
  9. Therefore we study distribution.
 NumPy / Torch: .
 Normalize arc: .
 (Discriminative vs generative lives in .)
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


# -------------------------------------------------------
# FIGURE 1: Strategy -- understand the client's business
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 1: Begin with the client's business",
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax1)

    round_box(ax1, 0.7, 5.2, 10.6, 2.0, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 6.2,
        "Our lecture strategy begins with the client's business -- not with formulas.",
        ha='center', va='center', fontsize=FS_AX_TITLE - 1,
        fontweight='bold', color='white',
    )

    pair = [
        (0.7, 1.2, 5.2, 3.5, 'We must understand it',
         'Goals, decisions, risks,\n'
         'and what a right choice\n'
         'means in their world.'),
        (6.1, 1.2, 5.2, 3.5, 'We must not drown in it',
         'As AI engineers, we avoid\n'
         'going too deep into\n'
         'domain lore.'),
    ]
    for x, y, w, h, title, body in pair:
        round_box(ax1, x, y, w, h, C_NEW, '#2874a6', lw=2.2)
        ax1.text(x + w / 2, y + h - 0.45, title, ha='center', va='top',
                 fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_DARK)
        ax1.text(x + w / 2, y + 1.4, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 1)

    return fig1


# -------------------------------------------------------
# FIGURE 2: Use accumulated data, not endless domain study
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        'Step 2: Prefer their data over endless domain immersion',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax2)

    round_box(ax2, 0.5, 4.2, 5.2, 3.0, C_OLD, '#e67e22', lw=2.5)
    ax2.text(3.1, 6.6, 'Domain-knowledge path', ha='center',
             fontsize=FS_AX_TITLE - 2, fontweight='bold', color='#a04000')
    ax2.text(
        3.1, 5.3,
        'Read every manual.\n'
        'Learn every jargon term.\n'
        'Become a junior expert first.\n\n'
        'Slow -- and easy to get stuck.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    round_box(ax2, 6.3, 4.2, 5.2, 3.0, C_GOAL, C_OK, lw=2.5)
    ax2.text(8.9, 6.6, 'Data path (our preference)', ha='center',
             fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_OK)
    ax2.text(
        8.9, 5.3,
        'Use what they already recorded.\n'
        'Let patterns in the data\n'
        'reveal how the business works.\n\n'
        'Faster -- and closer to AI.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    round_box(ax2, 0.5, 0.8, 11.0, 2.8, C_MUTE, '0.45', lw=1.8)
    ax2.text(
        6.0, 2.2,
        'We still listen to the client.\n'
        'But we understand the business mainly through their data,\n'
        'not by becoming domain specialists ourselves.',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
    )

    return fig2


# -------------------------------------------------------
# FIGURE 3: Looking at data alone -- everything is a distribution
# -------------------------------------------------------
def build_step_3():
    fig3, axes3 = new_figure(1, 2)
    fig3.suptitle(
        'Step 3: Looking only at the data, everything is a distribution',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    # Left: scatter of "data points"
    ax = axes3[0]
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(180, 2)) @ np.array([[1.0, 0.35], [0.35, 0.7]])
    ax.scatter(pts[:, 0], pts[:, 1], s=28, c='#2874a6', alpha=0.75, edgecolors='none')
    ax.set_title('A cloud of recorded cases', fontsize=FS_AX_TITLE)
    ax.set_xlabel('feature 1', fontsize=FS_AXIS)
    ax.set_ylabel('feature 2', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.03, 0.97,
        'No jargon needed:\nonly points that occurred.',
        transform=ax.transAxes, va='top', ha='left', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    # Right: histogram / density idea
    ax = axes3[1]
    vals = pts[:, 0]
    ax.hist(vals, bins=18, density=True, color='#2874a6', alpha=0.55, edgecolor='white')
    xs = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 200)
    # simple smooth sketch
    mu, sig = vals.mean(), vals.std()
    dens = np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    ax.plot(xs, dens, color=C_ACCENT, lw=2.5)
    ax.set_title('The same cloud as a distribution', fontsize=FS_AX_TITLE)
    ax.set_xlabel('feature 1', fontsize=FS_AXIS)
    ax.set_ylabel('relative frequency', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.25)
    ax.text(
        0.97, 0.97,
        'Where points gather,\n'
        'probability is higher.',
        transform=ax.transAxes, va='top', ha='right', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    fig3.text(
        0.5, 0.02,
        'So before domain theory: study what a distribution is.',
        ha='center', fontsize=FS_ANNOT + 2, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
    )
    fig3.tight_layout(rect=[0, 0.06, 1, 0.94])
    return fig3


# -------------------------------------------------------
# FIGURE 4: Knowledge / capability as a distribution
# -------------------------------------------------------
def build_step_4():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        'Step 4: Knowledge -- and capability -- is also a distribution',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax4)

    round_box(ax4, 0.6, 5.4, 10.8, 1.9, C_DARK, C_DARK, lw=0)
    ax4.text(
        6.0, 6.35,
        'Example: the ability to recognize a tree',
        ha='center', va='center', fontsize=FS_AX_TITLE,
        fontweight='bold', color='white',
    )

    # Objects row
    objects = [
        (0.7, 'Oak photo', 'tree?', 0.95),
        (3.1, 'Pine photo', 'tree?', 0.90),
        (5.5, 'Bush photo', 'tree?', 0.35),
        (7.9, 'Lamp post', 'tree?', 0.05),
        (10.0, 'Cat photo', 'tree?', 0.02),
    ]
    for x, name, q, p in objects:
        round_box(ax4, x, 3.2, 1.9, 1.8, C_MUTE, '0.45', lw=1.5)
        ax4.text(x + 0.95, 4.55, name, ha='center', va='center',
                 fontsize=FS_ANNOT, fontweight='bold')
        ax4.text(x + 0.95, 3.7, f'{q}\n{p:.2f}', ha='center', va='center',
                 fontsize=FS_ANNOT)

    round_box(ax4, 0.6, 0.7, 10.8, 2.2, C_GOAL, C_OK, lw=2.2)
    ax4.text(
        6.0, 1.8,
        'That ability is a distribution of labels over objects:\n'
        'yes or no, or a likelihood that each object is a tree.\n'
        'Knowledge is not a magic sentence -- it is how labels spread across cases.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    return fig4


# -------------------------------------------------------
# FIGURE 5: Next-word distribution after a given context
# -------------------------------------------------------
def build_step_5():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        'Step 5: Another distribution -- the next word after a context',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    context = 'The cat sat on the'
    # Illustrative next-word probabilities (toy, not from a real model)
    next_words = [
        ('mat', 0.42),
        ('floor', 0.18),
        ('sofa', 0.14),
        ('table', 0.09),
        ('bed', 0.06),
        ('roof', 0.04),
        ('dog', 0.03),
        ('...', 0.04),
    ]
    labels5 = [w for w, _ in next_words]
    probs5 = np.array([p for _, p in next_words])
    # Draw highest bar first for visual ranking
    order5 = np.argsort(probs5)
    labels5 = [labels5[i] for i in order5]
    probs5 = probs5[order5]
    colors5 = ['#2874a6' if lab != '...' else '0.65' for lab in labels5]

    y_pos = np.arange(len(labels5))
    ax5.barh(y_pos, probs5, color=colors5, edgecolor='white', height=0.72)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(labels5, fontsize=FS_AXIS)
    ax5.set_xlabel('Probability of the next word', fontsize=FS_AXIS)
    ax5.tick_params(axis='x', labelsize=FS_TICK)
    ax5.set_xlim(0, max(probs5) * 1.25)
    ax5.grid(True, axis='x', alpha=0.25)
    for y, p in zip(y_pos, probs5):
        ax5.text(p + 0.01, y, f'{p:.2f}', va='center', ha='left',
                 fontsize=FS_ANNOT)

    # Context banner above the axes (~1 cm below prior placement)
    fig5.text(
        0.5, 0.85,
        f'Context:  "{context} ..."',
        ha='center', va='center', fontsize=FS_AX_TITLE,
        fontweight='bold', color=C_DARK,
        bbox=dict(boxstyle='round,pad=0.35', facecolor=C_NEW,
                  edgecolor='#2874a6', linewidth=1.8, alpha=0.97),
    )
    ax5.text(
        0.98, 0.05,
        'Given the context, the next word\n'
        'is not a single answer --\n'
        'it is a distribution over words.\n'
        'Language skill lives in that spread.',
        transform=ax5.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    fig5.tight_layout(rect=[0, 0, 1, 0.80])
    return fig5


# -------------------------------------------------------
# FIGURE 6: Next-sentence distribution after a given context
# -------------------------------------------------------
def build_step_6():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        'Step 6: Another distribution -- the next sentence after a context',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    context6 = '... (story B)...It was a rainy morning, so she'
    next_sentences = [
        ('stayed home and read.', 0.31),
        ('took an umbrella and left.', 0.24),
        ('called a taxi to work.', 0.16),
        ('watched the street from the window.', 0.12),
        ('went for a walk anyway.', 0.08),
        ('baked bread instead.', 0.05),
        ('...', 0.04),
    ]
    labels6 = [s for s, _ in next_sentences]
    probs6 = np.array([p for _, p in next_sentences])
    order6 = np.argsort(probs6)
    labels6 = [labels6[i] for i in order6]
    probs6 = probs6[order6]
    colors6 = ['#1e8449' if lab != '...' else '0.65' for lab in labels6]

    y_pos6 = np.arange(len(labels6))
    ax6.barh(y_pos6, probs6, color=colors6, edgecolor='white', height=0.72)
    ax6.set_yticks(y_pos6)
    ax6.set_yticklabels(labels6, fontsize=FS_ANNOT + 1)
    ax6.set_xlabel('Probability of the next sentence', fontsize=FS_AXIS)
    ax6.tick_params(axis='x', labelsize=FS_TICK)
    ax6.set_xlim(0, max(probs6) * 1.28)
    ax6.grid(True, axis='x', alpha=0.25)
    for y, p in zip(y_pos6, probs6):
        ax6.text(p + 0.008, y, f'{p:.2f}', va='center', ha='left',
                 fontsize=FS_ANNOT)

    # Context banner (~1 cm below prior placement)
    fig6.text(
        0.5, 0.85,
        f'Context:  "{context6} ..."',
        ha='center', va='center', fontsize=FS_AX_TITLE - 1,
        fontweight='bold', color=C_DARK,
        bbox=dict(boxstyle='round,pad=0.35', facecolor=C_GOAL,
                  edgecolor=C_OK, linewidth=1.8, alpha=0.97),
    )
    ax6.text(
        0.98, 0.05,
        'Same idea, larger units:\n'
        'after a context, the next sentence\n'
        'is again a distribution --\n'
        'not one destined line.',
        transform=ax6.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    fig6.tight_layout(rect=[0, 0, 1, 0.80])
    return fig6


# -------------------------------------------------------
# FIGURE 7: Everyday images as a distribution
# -------------------------------------------------------
def build_step_7():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        'Step 7: Everyday images are also a distribution',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    rng7 = np.random.default_rng(11)


    def sample_river(rng, polylines, n_per_seg=55, width_lo=0.04, width_hi=0.18):
        """Sample a river-like cloud: thin branching corridors with varying width."""
        clouds = []
        for poly in polylines:
            poly = np.asarray(poly, dtype=float)
            for a, b in zip(poly[:-1], poly[1:]):
                t = rng.random(n_per_seg)
                # Slightly denser near mid-segment bends
                along = a + (b - a) * t[:, None]
                tang = b - a
                tang_n = tang / (np.linalg.norm(tang) + 1e-9)
                normal = np.array([-tang_n[1], tang_n[0]])
                # Width breathes along the segment (narrow / wide / narrow)
                width = width_lo + (width_hi - width_lo) * (
                    0.35 + 0.65 * np.sin(np.pi * t) ** 2
                )
                offset = rng.normal(0.0, 1.0, size=n_per_seg) * width
                clouds.append(along + offset[:, None] * normal)
        return np.vstack(clouds)


    # Compact everyday modes (background of daily life)
    everyday = [
        ('Street', [-2.6, -1.8], [0.40, 0.32], '#5dade2'),
        ('Kitchen', [-2.4, 1.9], [0.38, 0.34], '#58d68d'),
        ('Faces', [2.5, 1.9], [0.36, 0.32], '#f5b041'),
        ('Nature', [2.6, -1.9], [0.40, 0.34], '#af7ac5'),
        ('Screens', [-2.5, 0.2], [0.32, 0.28], '#85929e'),
    ]
    for name, loc, scale, color in everyday:
        pts = rng7.normal(loc=loc, scale=scale, size=(70, 2))
        ax7.scatter(
            pts[:, 0], pts[:, 1], s=22, c=color, alpha=0.70,
            edgecolors='white', linewidths=0.25, zorder=2, label=name,
        )
        c = pts.mean(axis=0)
        ax7.annotate(
            name, xy=c, xytext=(c[0], c[1] + 0.75),
            fontsize=FS_ANNOT, fontweight='bold', color=color, ha='center',
            arrowprops=dict(arrowstyle='->', color=color, lw=1.1),
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                      edgecolor=color, alpha=0.95),
        )

    # Cat and Dog: complex river networks -- many branches, changing narrow width,
    # and the two rivers pass near / through each other.
    dog_river = [
        # main trunk
        [(-1.6, -0.9), (-0.7, -0.35), (0.15, 0.05), (0.9, 0.55), (1.7, 0.85)],
        # side branches
        [(-0.7, -0.35), (-1.1, 0.35), (-0.85, 0.95)],
        [(0.15, 0.05), (0.55, -0.55), (1.05, -0.95)],
        [(0.9, 0.55), (1.35, 1.25), (1.15, 1.75)],
        [(0.15, 0.05), (-0.25, 0.65), (0.2, 1.15)],
    ]
    cat_river = [
        # main trunk crosses near the dog river
        [(-1.5, 0.85), (-0.6, 0.45), (0.1, -0.05), (0.85, -0.45), (1.65, -0.75)],
        # side branches that weave close to dog branches
        [(-0.6, 0.45), (-1.05, -0.15), (-0.75, -0.85)],
        [(0.1, -0.05), (0.5, 0.55), (0.95, 1.05)],
        [(0.85, -0.45), (1.3, -1.15), (1.1, -1.65)],
        [(0.1, -0.05), (-0.35, -0.55), (0.15, -1.1)],
    ]

    dogs7 = sample_river(rng7, dog_river, n_per_seg=48, width_lo=0.035, width_hi=0.16)
    cats7 = sample_river(rng7, cat_river, n_per_seg=48, width_lo=0.035, width_hi=0.16)

    ax7.scatter(
        dogs7[:, 0], dogs7[:, 1], s=16, c='#1f77b4', alpha=0.78,
        edgecolors='none', zorder=4, label='Dog',
    )
    ax7.scatter(
        cats7[:, 0], cats7[:, 1], s=16, c='#2ca02c', alpha=0.78,
        edgecolors='none', zorder=4, label='Cat',
    )
    ax7.annotate(
        'Dog', xy=dogs7.mean(axis=0),
        xytext=(1.85, 1.55),
        fontsize=FS_ANNOT + 1, fontweight='bold', color='#1f77b4', ha='center',
        arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.4),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='#1f77b4', alpha=0.95),
    )
    ax7.annotate(
        'Cat', xy=cats7.mean(axis=0),
        xytext=(1.85, -1.55),
        fontsize=FS_ANNOT + 1, fontweight='bold', color='#2ca02c', ha='center',
        arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.4),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='#2ca02c', alpha=0.95),
    )

    ax7.set_xlabel('feature 1  (e.g. an embedding)', fontsize=FS_AXIS)
    ax7.set_ylabel('feature 2', fontsize=FS_AXIS)
    ax7.tick_params(labelsize=FS_TICK)
    ax7.grid(True, alpha=0.25)
    ax7.set_aspect('equal', adjustable='datalim')
    ax7.legend(loc='upper left', fontsize=FS_LEGEND, framealpha=0.92)
    ax7.text(
        0.02, 0.02,
        'Cat and Dog are not neat blobs.\n'
        'Each is a river of branches --\n'
        'narrow, widening, forking --\n'
        'and the two rivers pass\n'
        'near, even through, each other.',
        transform=ax7.transAxes, ha='left', va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    fig7.tight_layout(rect=[0, 0, 1, 0.94])
    return fig7


# -------------------------------------------------------
# FIGURE 8: Another example -- dog vs cat photo clusters
# -------------------------------------------------------
def build_step_8():
    fig8, ax8 = new_figure()
    fig8.suptitle(
        'Step 8: Zooming in -- dog photos and cat photos',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    rng8 = np.random.default_rng(3)
    # Two modes / clusters in a 2D feature plane (e.g. embedding coordinates)
    dogs = rng8.normal(loc=[-1.2, -0.3], scale=[0.55, 0.45], size=(120, 2))
    cats = rng8.normal(loc=[1.3, 0.4], scale=[0.50, 0.48], size=(120, 2))
    dog_c = dogs.mean(axis=0)
    cat_c = cats.mean(axis=0)

    ax8.scatter(
        dogs[:, 0], dogs[:, 1], s=36, c='#1f77b4', alpha=0.85,
        edgecolors='white', linewidths=0.4, zorder=3,
    )
    ax8.scatter(
        cats[:, 0], cats[:, 1], s=36, c='#2ca02c', alpha=0.85,
        edgecolors='white', linewidths=0.4, zorder=3,
    )
    # Place labels near each cluster (slightly offset from the mean)
    ax8.annotate(
        'Dog photos', xy=dog_c, xytext=(dog_c[0] - 0.15, dog_c[1] + 1.15),
        fontsize=FS_ANNOT + 2, fontweight='bold', color='#1f77b4', ha='center',
        arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.6),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#1f77b4', alpha=0.95),
    )
    ax8.annotate(
        'Cat photos', xy=cat_c, xytext=(cat_c[0] + 0.15, cat_c[1] + 1.15),
        fontsize=FS_ANNOT + 2, fontweight='bold', color='#2ca02c', ha='center',
        arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.6),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#2ca02c', alpha=0.95),
    )
    ax8.set_xlabel('feature 1  (e.g. an embedding or measurement)', fontsize=FS_AXIS)
    ax8.set_ylabel('feature 2', fontsize=FS_AXIS)
    ax8.tick_params(labelsize=FS_TICK)
    ax8.grid(True, alpha=0.25)
    ax8.set_aspect('equal', adjustable='datalim')
    ax8.text(
        0.98, 0.05,
        'Two modes (clusters)\n'
        'in one joint distribution.\n'
        'Still only points --\n'
        'now with two pile-ups.',
        transform=ax8.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    fig8.tight_layout(rect=[0, 0, 1, 0.94])
    return fig8


# -------------------------------------------------------
# FIGURE 9: Bridge to the next technical work
# -------------------------------------------------------
def build_step_9():
    fig14, ax14 = new_figure()
    fig14.suptitle(
        'Step 9: Therefore we study distribution, driven by the goal',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax14)

    flow = [
        (0.6, 'Client\nbusiness'),
        (3.5, 'Accumulated\ndata'),
        (6.4, 'Distribution\nof points'),
        (9.3, 'Models and\nright choices'),
    ]
    for i, (x, title) in enumerate(flow):
        color = C_NEW if i < 3 else C_GOAL
        edge = '#2874a6' if i < 3 else C_OK
        round_box(ax14, x, 4.2, 2.4, 2.4, color, edge, lw=2.2)
        ax14.text(x + 1.2, 5.4, title, ha='center', va='center',
                  fontsize=FS_ANNOT + 2, fontweight='bold')
        if i < len(flow) - 1:
            arrow(ax14, x + 2.45, 5.4, x + 2.85, 5.4, C_DARK, lw=2.4)

    # Middle panel extended; ~0.5 cm extra space between body lines.
    panel_y, panel_h = 0.35, 3.5
    round_box(ax14, 0.6, panel_y, 10.8, panel_h, C_MUTE, '0.45', lw=1.8)
    ax14.text(
        6.0, panel_y + panel_h / 2,
        'Distribution is the language of data without domain jargon.\n'
        'If we can read distributions, we can read businesses -- and choose well.',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
        linespacing=2.05,
    )

    return fig14


# -------------------------------------------------------

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
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("WHY DISTRIBUTION")
print("=" * 60)
print("""
Strategy:
  Begin with the client's business.
  As AI engineers, avoid going too deep into domain lore;
  use their accumulated data to understand how the business works.

Looking at data alone:
  Without domain vocabulary, everything is a distribution of points.
  So before domain theory, study what a distribution is.

Knowledge as distribution:
  Recognizing a tree is a distribution of labels -- or likelihoods --
  over objects across cases.

Next-word example (Step 5):
  Given a context such as "...(story A)...The cat sat on the ...",
  the next word is a distribution over candidates
  (mat, floor, sofa, ...) -- language skill lives in that spread.

Next-sentence example (Step 6):
  Same idea at a larger unit: after a context, the next sentence
  is again a distribution -- not one destined line.

Everyday images (Step 7):
  Photos from daily life form a distribution with many familiar modes
  (street, kitchen, faces, nature, screens, ...).
  Cat and Dog are not neat blobs: each is a river of branches --
  narrow, widening, forking -- and the two rivers pass near,
  even through, each other.

Zoom-in example (Step 8):
  Dog-photo and cat-photo points in 2D form two modes / clusters
  -- still a distribution, now with two pile-ups.
""")
