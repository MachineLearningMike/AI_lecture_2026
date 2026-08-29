"""
008_distribution.py

Why we study distributions -- from the lecture strategy.

Sketch:
  1. Begin with the client's business.
  2. Prefer their data over deep immersion in domain lore.
  3. Looking only at the data, everything is a distribution of points.
  4. Knowledge itself can be seen as a distribution
     (e.g. tree recognition = labels or likelihoods over objects).
  5. Another example: dog vs cat photo clusters (two modes).
  6. The same clusters, tiny in a corner -- then normalized;
     framing questions on scale, information, and boundaries.
  7. Therefore we study distribution.
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

show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Use accumulated data, not endless domain study
# -------------------------------------------------------
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

show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Looking at data alone -- everything is a distribution
# -------------------------------------------------------
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
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Knowledge / capability as a distribution
# -------------------------------------------------------
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

show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Another example -- dog vs cat photo clusters
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    'Step 5: Another distribution -- dog photos and cat photos',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

rng5 = np.random.default_rng(3)
# Two modes / clusters in a 2D feature plane (e.g. embedding coordinates)
dogs = rng5.normal(loc=[-1.2, -0.3], scale=[0.55, 0.45], size=(120, 2))
cats = rng5.normal(loc=[1.3, 0.4], scale=[0.50, 0.48], size=(120, 2))
dog_c = dogs.mean(axis=0)
cat_c = cats.mean(axis=0)

ax5.scatter(
    dogs[:, 0], dogs[:, 1], s=36, c='#1f77b4', alpha=0.85,
    edgecolors='white', linewidths=0.4, zorder=3,
)
ax5.scatter(
    cats[:, 0], cats[:, 1], s=36, c='#2ca02c', alpha=0.85,
    edgecolors='white', linewidths=0.4, zorder=3,
)
# Place labels near each cluster (slightly offset from the mean)
ax5.annotate(
    'Dog photos', xy=dog_c, xytext=(dog_c[0] - 0.15, dog_c[1] + 1.15),
    fontsize=FS_ANNOT + 2, fontweight='bold', color='#1f77b4', ha='center',
    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.6),
    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#1f77b4', alpha=0.95),
)
ax5.annotate(
    'Cat photos', xy=cat_c, xytext=(cat_c[0] + 0.15, cat_c[1] + 1.15),
    fontsize=FS_ANNOT + 2, fontweight='bold', color='#2ca02c', ha='center',
    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.6),
    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#2ca02c', alpha=0.95),
)
ax5.set_xlabel('feature 1  (e.g. an embedding or measurement)', fontsize=FS_AXIS)
ax5.set_ylabel('feature 2', fontsize=FS_AXIS)
ax5.tick_params(labelsize=FS_TICK)
ax5.grid(True, alpha=0.25)
ax5.set_aspect('equal', adjustable='datalim')
ax5.text(
    0.98, 0.05,
    'Two modes (clusters)\n'
    'in one joint distribution.\n'
    'Still only points --\n'
    'now with two pile-ups.',
    transform=ax5.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
fig5.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Corner view + normalized view -- what should we do?
# -------------------------------------------------------
fig6, (ax6, axq) = new_figure(1, 2, gridspec_kw={'width_ratios': [2.35, 1.0]})
fig6.suptitle(
    'Step 6: The same distribution -- from a corner, then normalized',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

# Shrink and translate the Step-5 clusters into the lower-left of a wide frame
scale6 = 0.22
shift6 = np.array([-3.2, -3.2])
dogs6 = dogs * scale6 + shift6
cats6 = cats * scale6 + shift6
dog_c6 = dogs6.mean(axis=0)
cat_c6 = cats6.mean(axis=0)

# Corner (shrunk) copy
ax6.scatter(
    dogs6[:, 0], dogs6[:, 1], s=18, c='#1f77b4', alpha=0.75,
    edgecolors='white', linewidths=0.25, zorder=3,
)
ax6.scatter(
    cats6[:, 0], cats6[:, 1], s=18, c='#2ca02c', alpha=0.75,
    edgecolors='white', linewidths=0.25, zorder=3,
)
# Normalized copy (same points as Step 5)
ax6.scatter(
    dogs[:, 0], dogs[:, 1], s=36, c='#1f77b4', alpha=0.85,
    edgecolors='white', linewidths=0.4, zorder=4,
)
ax6.scatter(
    cats[:, 0], cats[:, 1], s=36, c='#2ca02c', alpha=0.85,
    edgecolors='white', linewidths=0.4, zorder=4,
)

ax6.annotate(
    'Dog photos', xy=dog_c, xytext=(dog_c[0] - 0.15, dog_c[1] + 1.15),
    fontsize=FS_ANNOT + 1, fontweight='bold', color='#1f77b4', ha='center',
    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#1f77b4', alpha=0.95),
)
ax6.annotate(
    'Cat photos', xy=cat_c, xytext=(cat_c[0] + 0.15, cat_c[1] + 1.15),
    fontsize=FS_ANNOT + 1, fontweight='bold', color='#2ca02c', ha='center',
    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#2ca02c', alpha=0.95),
)
ax6.annotate(
    'As recorded\n(tiny, in a corner)',
    xy=((dog_c6 + cat_c6) / 2),
    xytext=(dog_c6[0] - 0.15, dog_c6[1] + 1.15),
    fontsize=FS_ANNOT - 1, color='0.25', ha='center',
    arrowprops=dict(arrowstyle='->', color='0.45', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.55', alpha=0.95),
)

# Bridge label: corner -> normalized
mid_norm = (dog_c + cat_c) / 2
ax6.annotate(
    'Normalize the distribution\nfor convenience',
    xy=mid_norm, xytext=(-0.4, -2.55),
    fontsize=FS_ANNOT + 1, fontweight='bold', color=C_DARK, ha='center',
    arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2.0,
                    connectionstyle='arc3,rad=0.15'),
    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
              edgecolor=C_DARK, linewidth=2.0, alpha=0.95),
)

ax6.set_xlim(-4.5, 4.5)
ax6.set_ylim(-4.5, 4.5)
ax6.set_xlabel('feature 1', fontsize=FS_AXIS)
ax6.set_ylabel('feature 2', fontsize=FS_AXIS)
ax6.tick_params(labelsize=FS_TICK)
ax6.grid(True, alpha=0.25)
ax6.set_aspect('equal', adjustable='box')
ax6.axhline(0, color='0.7', lw=0.8)
ax6.axvline(0, color='0.7', lw=0.8)

# Questions live outside the plot axes
axq.set_xlim(0, 1)
axq.set_ylim(0, 1)
axq.axis('off')
axq.text(
    0.98, 0.98,
    'Questions for the room:\n\n'
    '1. Would it help to bring these\n'
    '    clusters to the center and\n'
    '    enlarge them, so we can\n'
    '    study how they relate?\n\n'
    '2. If we do, do we lose the\n'
    '    information that tells cats\n'
    '    from dogs -- or do we merely\n'
    '    change the frame in which\n'
    '    we look?\n\n'
    '3. Beyond discriminating one\n'
    '    cluster from the other,\n'
    '    might we learn the boundary\n'
    '    of each cluster in its own\n'
    '    right?',
    transform=axq.transAxes, ha='right', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round,pad=0.55', facecolor='lightyellow',
              edgecolor='0.55', alpha=0.98),
    multialignment='right',
)
fig6.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Bridge to the next technical work
# -------------------------------------------------------
fig7, ax7 = new_figure()
fig7.suptitle(
    'Step 7: Therefore we study distribution',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax7)

flow = [
    (0.6, 'Client\nbusiness'),
    (3.5, 'Accumulated\ndata'),
    (6.4, 'Distribution\nof points'),
    (9.3, 'Models and\nright choices'),
]
for i, (x, title) in enumerate(flow):
    color = C_NEW if i < 3 else C_GOAL
    edge = '#2874a6' if i < 3 else C_OK
    round_box(ax7, x, 4.2, 2.4, 2.4, color, edge, lw=2.2)
    ax7.text(x + 1.2, 5.4, title, ha='center', va='center',
             fontsize=FS_ANNOT + 2, fontweight='bold')
    if i < len(flow) - 1:
        arrow(ax7, x + 2.45, 5.4, x + 2.85, 5.4, C_DARK, lw=2.4)

round_box(ax7, 0.6, 1.0, 10.8, 2.6, C_MUTE, '0.45', lw=1.8)
ax7.text(
    6.0, 2.3,
    'Distribution is the language of data without domain jargon.\n'
    'If we can read distributions, we can read businesses -- and choose well.',
    ha='center', va='center', fontsize=FS_ANNOT + 2,
)

show_maximized(fig7)


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

Another example (Step 5):
  Dog-photo and cat-photo points in 2D form two modes / clusters
  -- still a distribution, now with two pile-ups.

Framing questions (Step 6):
  Would it help to bring these clusters to the center and enlarge them,
  so we can study how they relate?
  If we do, do we lose the information that tells cats from dogs --
  or do we merely change the frame in which we look?
  Beyond discriminating one cluster from the other, might we learn
  the boundary of each cluster in its own right?
""")
