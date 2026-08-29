"""
009_discriminative_generative.py

Discriminative vs generative understanding of a distribution.

Sketch:
  1. Two readings: P(cat | cat-or-dog image) vs P(cat | image).
  2. Kitchen vs cat cross table, plus branching cat/dog rivers;
     boundary knowledge vs knowing the cat world (and creating from it).
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
# FIGURE 1: Two readings of a distribution
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    'Step 1: Discriminative vs generative understanding of a distribution',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)
blank_slide(ax1)

round_box(ax1, 0.6, 5.4, 10.8, 1.9, C_DARK, C_DARK, lw=0)
ax1.text(
    6.0, 6.35,
    'Same word "cat" -- two different questions',
    ha='center', va='center', fontsize=FS_AX_TITLE,
    fontweight='bold', color='white',
)

round_box(ax1, 0.5, 1.0, 5.3, 3.9, C_OLD, '#e67e22', lw=2.5)
ax1.text(
    3.15, 4.4, 'Discriminative',
    ha='center', va='center', fontsize=FS_AX_TITLE,
    fontweight='bold', color='#a04000',
)
ax1.text(
    3.15, 2.7,
    (
        r'$P(\mathrm{cat}\mid\mathrm{cat\ or\ dog\ image})$' + '\n\n'
        'Given that the image is a cat or a dog,\n'
        'which one is it?\n\n'
        'A forced choice between two labels.'
    ),
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

round_box(ax1, 6.2, 1.0, 5.3, 3.9, C_NEW, '#2874a6', lw=2.5)
ax1.text(
    8.85, 4.4, 'Generative',
    ha='center', va='center', fontsize=FS_AX_TITLE,
    fontweight='bold', color=C_DARK,
)
ax1.text(
    8.85, 2.7,
    (
        r'$P(\mathrm{cat}\mid\mathrm{image})$' + '\n\n'
        'Given any image,\n'
        'how likely is it a cat?\n\n'
        'Else is allowed -- kitchen, street, ...'
    ),
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Kitchen vs cat + complex cat/dog distribution
# -------------------------------------------------------
fig2, (ax_cases, ax_river) = new_figure(
    1, 2, gridspec_kw={'width_ratios': [6.4, 2.1]},
)
fig2.suptitle(
    'Step 2: A kitchen image exposes the difference; a cat image does not',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)

# Left: cross table + captions (wide columns; top-aligned without shrinking x)
x0 = 0.2
label_w, head_w, cell_w = 5.6 * 1.2, 5.2 * 1.2, 5.2 * 1.2
head_h, row_h = 1.3, 3.0
table_w = label_w + 2 * cell_w
table_h = head_h + 2 * row_h
gap_table_to_box = 0.85
gap_between_boxes = 0.70
caption_block_h = 2.6  # room for both caption boxes + gaps
y_top = table_h + gap_table_to_box + caption_block_h + 0.4
y0 = y_top - table_h  # table flush to the top of the data window

ax_cases.set_xlim(0, x0 + table_w + 0.25)
ax_cases.set_ylim(0, y_top + 0.15)
ax_cases.axis('off')

# Corner (empty)
round_box(ax_cases, x0, y0 + 2 * row_h, label_w, head_h, C_MUTE, '0.55', lw=1.4)

# Head row: Kitchen | Cat
for i, (name, face) in enumerate([
    ('Kitchen image', C_MUTE),
    ('Cat image', C_GOAL),
]):
    x = x0 + label_w + i * head_w
    round_box(ax_cases, x, y0 + 2 * row_h, head_w, head_h, face, '0.45', lw=1.6)
    ax_cases.text(
        x + head_w / 2, y0 + 2 * row_h + head_h / 2, name,
        ha='center', va='center', fontsize=FS_ANNOT + 2,
        fontweight='bold', color=C_DARK,
    )

# Row labels + answer cells
rows = [
    (
        r'$P(\mathrm{cat}\mid\mathrm{cat\ or\ dog\ image})$',
        '#fce4d6', '#e67e22', '#a04000',
        '"Cat 85%, so Dog 15%."',
        '"Cat 98%, so Dog 2%."',
    ),
    (
        r'$P(\mathrm{cat}\mid\mathrm{image})$',
        '#d6eaf8', '#2874a6', C_DARK,
        '"Cat 2%, so Else 98%"',
        '"Cat 98%, so Else 2%"',
    ),
]
for r, (row_label, face, edge, title_c, ans_k, ans_c) in enumerate(rows):
    y = y0 + (1 - r) * row_h
    # First column: probability query
    round_box(ax_cases, x0, y, label_w, row_h, face, edge, lw=1.6)
    ax_cases.text(
        x0 + label_w / 2, y + row_h / 2, row_label,
        ha='center', va='center', fontsize=FS_ANNOT,
        fontweight='bold', color=title_c,
    )
    # Cross cells: answers only
    for c, ans in enumerate([ans_k, ans_c]):
        x = x0 + label_w + c * cell_w
        round_box(ax_cases, x, y, cell_w, row_h, 'white', edge, lw=1.6)
        ax_cases.text(
            x + cell_w / 2, y + row_h / 2, ans,
            ha='center', va='center', fontsize=FS_ANNOT + 1,
            fontweight='bold', color=C_DARK,
        )

# Captions sit under the table, still in the left column (with clear gaps)
ax_cases.text(
    x0 + table_w / 2, y0 - gap_table_to_box,
    'P(cat|image) model knows every possible cat image;\n'
    'P(cat|cat or dog image) model knows only the boundary between cats and dogs.\n'
    'You can ask P(cat|image) model for a new cat image.',
    ha='center', va='top', fontsize=FS_ANNOT + 1, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
              edgecolor='0.55', alpha=0.97),
)
ax_cases.text(
    x0 + table_w / 2, y0 - gap_table_to_box - 1.55 - gap_between_boxes,
    r'So, $P(\mathrm{cat}\mid\mathrm{image})=P(\mathrm{cat}\mid\mathrm{cat\ or\ else\ image})$',
    ha='center', va='top', fontsize=FS_ANNOT + 2, fontweight='bold',
    color=C_DARK,
    bbox=dict(boxstyle='round,pad=0.4', facecolor=C_GOAL,
              edgecolor=C_OK, linewidth=1.6, alpha=0.97),
)

# Right: complex river-like cat / dog distribution
def sample_river(rng, polylines, n_per_seg=48, width_lo=0.035, width_hi=0.16):
    """Sample a river-like cloud: thin branching corridors with varying width."""
    clouds = []
    for poly in polylines:
        poly = np.asarray(poly, dtype=float)
        for a, b in zip(poly[:-1], poly[1:]):
            t = rng.random(n_per_seg)
            along = a + (b - a) * t[:, None]
            tang = b - a
            tang_n = tang / (np.linalg.norm(tang) + 1e-9)
            normal = np.array([-tang_n[1], tang_n[0]])
            width = width_lo + (width_hi - width_lo) * (
                0.35 + 0.65 * np.sin(np.pi * t) ** 2
            )
            offset = rng.normal(0.0, 1.0, size=n_per_seg) * width
            clouds.append(along + offset[:, None] * normal)
    return np.vstack(clouds)


rng2 = np.random.default_rng(11)
dog_river = [
    [(-1.6, -0.9), (-0.7, -0.35), (0.15, 0.05), (0.9, 0.55), (1.7, 0.85)],
    [(-0.7, -0.35), (-1.1, 0.35), (-0.85, 0.95)],
    [(0.15, 0.05), (0.55, -0.55), (1.05, -0.95)],
    [(0.9, 0.55), (1.35, 1.25), (1.15, 1.75)],
    [(0.15, 0.05), (-0.25, 0.65), (0.2, 1.15)],
]
cat_river = [
    [(-1.5, 0.85), (-0.6, 0.45), (0.1, -0.05), (0.85, -0.45), (1.65, -0.75)],
    [(-0.6, 0.45), (-1.05, -0.15), (-0.75, -0.85)],
    [(0.1, -0.05), (0.5, 0.55), (0.95, 1.05)],
    [(0.85, -0.45), (1.3, -1.15), (1.1, -1.65)],
    [(0.1, -0.05), (-0.35, -0.55), (0.15, -1.1)],
]
dogs2 = sample_river(rng2, dog_river)
cats2 = sample_river(rng2, cat_river)

ax_river.scatter(
    dogs2[:, 0], dogs2[:, 1], s=14, c='#1f77b4', alpha=0.78,
    edgecolors='none', zorder=3, label='Dog',
)
ax_river.scatter(
    cats2[:, 0], cats2[:, 1], s=14, c='#2ca02c', alpha=0.78,
    edgecolors='none', zorder=3, label='Cat',
)
ax_river.annotate(
    'Dog', xy=dogs2.mean(axis=0), xytext=(1.75, 1.55),
    fontsize=FS_ANNOT + 1, fontweight='bold', color='#1f77b4', ha='center',
    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.3),
    bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
              edgecolor='#1f77b4', alpha=0.95),
)
ax_river.annotate(
    'Cat', xy=cats2.mean(axis=0), xytext=(1.75, -1.55),
    fontsize=FS_ANNOT + 1, fontweight='bold', color='#2ca02c', ha='center',
    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.3),
    bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
              edgecolor='#2ca02c', alpha=0.95),
)

# Kitchen lies off both rivers -- "unknown / else" in the generative reading
kitchen_pt = np.array([0.15, -1.85])
ax_river.scatter(
    [kitchen_pt[0]], [kitchen_pt[1]], s=160, c='red', zorder=6,
    edgecolors='white', linewidths=1.6, label='Kitchen',
)
ax_river.annotate(
    'Kitchen', xy=kitchen_pt, xytext=(0.15, -2.45),
    fontsize=FS_ANNOT + 2, fontweight='bold', color='red', ha='center',
    arrowprops=dict(arrowstyle='->', color='red', lw=1.6),
    bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
              edgecolor='red', linewidth=1.5, alpha=0.95),
)

ax_river.set_title(
    'Cat and Dog in image space',
    fontsize=FS_AX_TITLE - 2,
)
ax_river.set_xlabel('feature 1', fontsize=FS_AXIS)
ax_river.set_ylabel('feature 2', fontsize=FS_AXIS)
ax_river.tick_params(labelsize=FS_TICK)
ax_river.grid(True, alpha=0.25)
ax_river.set_aspect('equal', adjustable='datalim')
ax_river.legend(loc='upper left', fontsize=FS_LEGEND, framealpha=0.92)

# Keep full width; nudge the layout slightly left without cropping
fig2.tight_layout(rect=[0.0, 0.02, 0.995, 0.94])
fig2.subplots_adjust(left=0.02, right=0.99, wspace=0.12)
show_maximized(fig2)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("DISCRIMINATIVE VS GENERATIVE")
print("=" * 60)
print("""
Step 1 -- two questions:
  Discriminative: P(cat | cat-or-dog image) -- forced choice between two labels.
  Generative:     P(cat | image) -- is it a cat at all? Else is allowed.

Step 2 -- the difference appears on a kitchen image:
  Discriminative may still say "Cat 85%, so Dog 15%."
  Generative says "Cat 2%, so Else 98%."
  On a cat image, both agree it is mostly a cat.

  P(cat|image) knows every possible cat image -- and can make a new one.
  P(cat|cat-or-dog image) knows only the boundary between cats and dogs.
""")
