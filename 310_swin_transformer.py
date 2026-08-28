"""
310_swin_transformer.py

Swin Transformer intuition (Shifted WINdow Transformer).

Core ideas vs plain ViT:
  1) LOCAL window attention instead of global token-token attention
     => compute ~ O(image size), not O((#patches)^2) globally
  2) SHIFTED windows across alternating layers
     => information crosses window boundaries
  3) HIERARCHICAL stages with patch merging
     => multi-scale feature maps (like a CNN backbone)

Companion to 190_transformer.py and 300_video_transformer.py.
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
RNG = np.random.default_rng(42)


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


def add_note(ax, text, x=0.98, y=0.46, ha='right', va='top', facecolor='white'):
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=FS_ANNOT,
        va=va, ha=ha,
        bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.95),
    )


# -------------------------------------------------------
# Toy image grid of tokens
# -------------------------------------------------------
H = W = 8          # token grid (after patch embed)
WINDOW = 4         # M x M window


def draw_token_grid(ax, shift=(0, 0), title='', highlight_query=None,
                    show_windows=True, window_color='crimson'):
    """Draw H x W tokens with optional shifted window partition."""
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlabel('token x', fontsize=FS_AXIS)
    ax.set_ylabel('token y', fontsize=FS_AXIS)
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(True, alpha=0.25)

    # token centers
    for y in range(H):
        for x in range(W):
            ax.scatter(x, y, s=40, c='#4c78a8', edgecolor='k', lw=0.5, zorder=3)

    if show_windows:
        # shifted partition: windows of size WINDOW starting at -shift
        sx, sy = shift
        # draw all windows that intersect the grid
        # effective origin of window lattice
        for wy in range(-WINDOW, H + WINDOW, WINDOW):
            for wx in range(-WINDOW, W + WINDOW, WINDOW):
                x0 = wx + sx
                y0 = wy + sy
                # clip visible rectangle
                x1 = max(x0, -0.45)
                y1 = max(y0, -0.45)
                x2 = min(x0 + WINDOW, W - 0.55)
                y2 = min(y0 + WINDOW, H - 0.55)
                if x2 > x1 and y2 > y1:
                    ax.add_patch(patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        fill=False, edgecolor=window_color, lw=2.5, zorder=4,
                    ))

    if highlight_query is not None:
        qx, qy = highlight_query
        ax.scatter([qx], [qy], s=160, c='crimson', edgecolor='k', zorder=5, label='query')
        # show allowed keys inside same window
        # find window containing query under this shift
        sx, sy = shift
        wx = ((qx - sx) // WINDOW) * WINDOW + sx
        wy = ((qy - sy) // WINDOW) * WINDOW + sy
        for y in range(H):
            for x in range(W):
                if wx <= x < wx + WINDOW and wy <= y < wy + WINDOW:
                    if (x, y) != (qx, qy):
                        ax.plot([qx, x], [qy, y], color='crimson', alpha=0.35, lw=1.0, zorder=2)


# -------------------------------------------------------
# FIGURE 1: ViT global vs Swin local windows
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    'Step 1: Swin replaces global attention with windowed attention',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes1[0]
ax.set_xlim(-0.5, W - 0.5)
ax.set_ylim(H - 0.5, -0.5)
ax.set_aspect('equal')
ax.set_title('ViT: every token attends globally', fontsize=FS_AX_TITLE)
ax.set_xlabel('token x', fontsize=FS_AXIS)
ax.set_ylabel('token y', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
for y in range(H):
    for x in range(W):
        ax.scatter(x, y, s=35, c='#4c78a8', edgecolor='k', lw=0.4, zorder=3)
qx, qy = 3, 3
ax.scatter([qx], [qy], s=160, c='crimson', edgecolor='k', zorder=5)
# sparse sample of global links for readability
for y in range(0, H, 2):
    for x in range(0, W, 2):
        ax.plot([qx, x], [qy, y], color='crimson', alpha=0.2, lw=0.8)
add_note(
    ax,
    f'Cost ~ (HW)^2\n'
    f'= {H*W}^2 = {(H*W)**2}\n'
    'links for one layer\n'
    '(grows fast with image).',
    y=0.02, va='bottom', facecolor='lightyellow',
)

ax = axes1[1]
draw_token_grid(
    ax, shift=(0, 0),
    title=f'Swin: attend only inside {WINDOW}x{WINDOW} windows',
    highlight_query=(3, 3),
    window_color='seagreen',
)
n_win = (H // WINDOW) * (W // WINDOW)
cost_swin = n_win * (WINDOW * WINDOW) ** 2
add_note(
    ax,
    f'Windows: {n_win}\n'
    f'Cost ~ #win * (M^2)^2\n'
    f'= {n_win}*{WINDOW**2}^2\n'
    f'= {cost_swin}\n'
    'linear in image area\n'
    'for fixed M.',
    y=0.02, va='bottom',
)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Shifted windows connect neighborhoods
# -------------------------------------------------------
fig2, axes2 = new_figure(1, 2)
fig2.suptitle(
    'Step 2: Alternating regular / shifted windows enable cross-window flow',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

draw_token_grid(
    axes2[0], shift=(0, 0),
    title='Layer l: regular windows',
    highlight_query=(3, 3),
    window_color='crimson',
)
draw_token_grid(
    axes2[1], shift=(-WINDOW // 2, -WINDOW // 2),
    title='Layer l+1: shifted windows',
    highlight_query=(3, 3),
    window_color='royalblue',
)

fig2.text(
    0.5, 0.02,
    'A token that was trapped in one window at layer l can meet new neighbors after the shift.\n'
    'Stacking W-MSA / SW-MSA propagates information across the whole image efficiently.',
    ha='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Why shifting matters -- connectivity cartoon
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    'Step 3: Without shifts, windows are isolated islands',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# Two adjacent tokens across a window border
ax = axes3[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('Same window layout every layer', fontsize=FS_AX_TITLE)
# two windows
ax.add_patch(patches.Rectangle((1, 1.5), 3.5, 3.0, fill=True, facecolor='#f8d7da',
                               edgecolor='crimson', lw=2.5))
ax.add_patch(patches.Rectangle((5.5, 1.5), 3.5, 3.0, fill=True, facecolor='#d1ecf1',
                               edgecolor='royalblue', lw=2.5))
ax.scatter([3.8], [3.0], s=200, c='crimson', edgecolor='k', zorder=3)
ax.scatter([6.2], [3.0], s=200, c='royalblue', edgecolor='k', zorder=3)
ax.text(3.8, 3.55, 'A', ha='center', fontsize=FS_ANNOT, fontweight='bold')
ax.text(6.2, 3.55, 'B', ha='center', fontsize=FS_ANNOT, fontweight='bold')
ax.text(0.5, 0.35,
        'A and B never attend directly if the border never moves.',
        transform=ax.transAxes, ha='left', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))

ax = axes3[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('After a window shift', fontsize=FS_AX_TITLE)
# shifted window covering both
ax.add_patch(patches.Rectangle((3.2, 1.5), 3.5, 3.0, fill=True, facecolor='#d4edda',
                               edgecolor='seagreen', lw=2.5))
ax.scatter([3.8], [3.0], s=200, c='crimson', edgecolor='k', zorder=3)
ax.scatter([6.2], [3.0], s=200, c='royalblue', edgecolor='k', zorder=3)
ax.plot([3.8, 6.2], [3.0, 3.0], color='seagreen', lw=3)
ax.text(3.8, 3.55, 'A', ha='center', fontsize=FS_ANNOT, fontweight='bold')
ax.text(6.2, 3.55, 'B', ha='center', fontsize=FS_ANNOT, fontweight='bold')
ax.text(5.0, 4.7, 'now in the SAME window', ha='center', color='seagreen',
        fontsize=FS_ANNOT, fontweight='bold')
ax.text(0.5, 0.35,
        'Shift creates new groups => A can attend to B next layer.',
        transform=ax.transAxes, ha='left', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 3b: Shift-window mechanism in detail (cyclic shift)
# -------------------------------------------------------
SHIFT = WINDOW // 2  # classic Swin: shift by M/2

fig3b, axes3b = new_figure(1, 3)
fig3b.suptitle(
    f'Step 3b: Shift-window (SW-MSA) -- cyclic shift by ({SHIFT},{SHIFT})',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)


def color_windows(ax, shift=(0, 0), title=''):
    """Color each window region; mark a cross-border pair A,B."""
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_title(title, fontsize=FS_AX_TITLE - 2)
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('y', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)

    sx, sy = shift
    palette = ['#f8d7da', '#d1ecf1', '#d4edda', '#fff3cd',
               '#e2d5f1', '#ffe0b2', '#cfe2ff', '#d9ead3']
    win_id = 0
    for wy in range(-WINDOW, H + WINDOW, WINDOW):
        for wx in range(-WINDOW, W + WINDOW, WINDOW):
            x0, y0 = wx + sx, wy + sy
            # visible intersection with image
            xs = range(max(0, x0), min(W, x0 + WINDOW))
            ys = range(max(0, y0), min(H, y0 + WINDOW))
            if not xs or not ys:
                continue
            color = palette[win_id % len(palette)]
            for y in ys:
                for x in xs:
                    ax.add_patch(patches.Rectangle(
                        (x - 0.5, y - 0.5), 1, 1,
                        facecolor=color, edgecolor='white', lw=0.8, zorder=1,
                    ))
            # window border
            x1, x2 = min(xs) - 0.5, max(xs) + 0.5
            y1, y2 = min(ys) - 0.5, max(ys) + 0.5
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor='k', lw=2.0, zorder=2,
            ))
            win_id += 1

    # tokens
    for y in range(H):
        for x in range(W):
            ax.scatter(x, y, s=28, c='0.2', zorder=3)


# Panel 1: before shift
color_windows(axes3b[0], shift=(0, 0), title='1) Regular partition (W-MSA)')
# mark A,B on a border: (3,3) and (4,3) with WINDOW=4 sit on boundary between windows
A = (3, 3)
B = (4, 3)
axes3b[0].scatter([A[0]], [A[1]], s=180, c='crimson', edgecolor='k', zorder=5)
axes3b[0].scatter([B[0]], [B[1]], s=180, c='navy', edgecolor='k', zorder=5)
axes3b[0].text(A[0], A[1] - 0.35, 'A', ha='center', color='crimson',
               fontsize=FS_ANNOT, fontweight='bold')
axes3b[0].text(B[0], B[1] - 0.35, 'B', ha='center', color='navy',
               fontsize=FS_ANNOT, fontweight='bold')
add_note(
    axes3b[0],
    'A and B are adjacent\n'
    'but in DIFFERENT\n'
    'windows => no direct\n'
    'attention this layer.',
    y=0.02, va='bottom', facecolor='lightyellow',
)

# Panel 2: after cyclic shift of the FEATURE MAP
# Visualize shifted coordinates: content moves by +SHIFT
ax = axes3b[1]
ax.set_xlim(-0.5, W - 0.5)
ax.set_ylim(H - 0.5, -0.5)
ax.set_aspect('equal')
ax.set_title(f'2) Cyclic shift features by ({SHIFT},{SHIFT})', fontsize=FS_AX_TITLE - 2)
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
# show arrows indicating cyclic roll
for y in range(0, H, 2):
    for x in range(0, W, 2):
        x2 = (x + SHIFT) % W
        y2 = (y + SHIFT) % H
        ax.annotate('', xy=(x2, y2), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='0.5', lw=1.0, alpha=0.7))
ax.scatter([A[0]], [A[1]], s=120, c='crimson', edgecolor='k', zorder=4, alpha=0.35)
ax.scatter([B[0]], [B[1]], s=120, c='navy', edgecolor='k', zorder=4, alpha=0.35)
A2 = ((A[0] + SHIFT) % W, (A[1] + SHIFT) % H)
B2 = ((B[0] + SHIFT) % W, (B[1] + SHIFT) % H)
ax.scatter([A2[0]], [A2[1]], s=180, c='crimson', edgecolor='k', zorder=5)
ax.scatter([B2[0]], [B2[1]], s=180, c='navy', edgecolor='k', zorder=5)
ax.text(A2[0], A2[1] - 0.35, "A'", ha='center', color='crimson',
        fontsize=FS_ANNOT, fontweight='bold')
ax.text(B2[0], B2[1] - 0.35, "B'", ha='center', color='navy',
        fontsize=FS_ANNOT, fontweight='bold')
add_note(
    ax,
    'Implementation trick:\n'
    'roll the map, then use\n'
    'the SAME efficient\n'
    'non-overlapping window\n'
    'kernel as W-MSA.',
    y=0.02, va='bottom',
)

# Panel 3: partition again on shifted map -- A' and B' share a window
color_windows(
    axes3b[2], shift=(0, 0),
    title='3) Re-partition into windows (SW-MSA)',
)
axes3b[2].scatter([A2[0]], [A2[1]], s=180, c='crimson', edgecolor='k', zorder=5)
axes3b[2].scatter([B2[0]], [B2[1]], s=180, c='navy', edgecolor='k', zorder=5)
axes3b[2].plot([A2[0], B2[0]], [A2[1], B2[1]], color='seagreen', lw=3, zorder=4)
axes3b[2].text(A2[0], A2[1] - 0.35, "A'", ha='center', color='crimson',
               fontsize=FS_ANNOT, fontweight='bold')
axes3b[2].text(B2[0], B2[1] - 0.35, "B'", ha='center', color='navy',
               fontsize=FS_ANNOT, fontweight='bold')
add_note(
    axes3b[2],
    "A' and B' can attend.\n"
    'Then reverse-roll back\n'
    'to the original layout.\n\n'
    'Edge windows may mix\n'
    'unrelated wrapped parts\n'
    '=> use an attn mask.',
    y=0.02, va='bottom', facecolor='lightyellow',
)

fig3b.text(
    0.5, 0.01,
    'SW-MSA pipeline:  cyclic shift  ->  window attention (+ mask)  ->  reverse shift',
    ha='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.95),
)
plt.tight_layout(rect=[0, 0.05, 1, 1])
show_maximized(fig3b)


# -------------------------------------------------------
# FIGURE 4: Hierarchical stages / patch merging
# -------------------------------------------------------
fig4, ax4 = new_figure()
fig4.suptitle(
    'Step 4: Hierarchical stages -- patch merging builds a feature pyramid',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax4.set_xlim(0, 12)
ax4.set_ylim(0, 8)
ax4.axis('off')

stages = [
    (0.5, 5.5, 2.8, 2.0, 'Stage 1\nH/4 x W/4\nC', '#dceaff'),
    (3.6, 5.0, 2.6, 2.5, 'Stage 2\nH/8 x W/8\n2C', '#f8d7da'),
    (6.5, 4.4, 2.4, 3.1, 'Stage 3\nH/16 x W/16\n4C', '#d4edda'),
    (9.2, 3.7, 2.2, 3.8, 'Stage 4\nH/32 x W/32\n8C', '#fff3cd'),
]
for x, y, w, h, text, color in stages:
    ax4.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor=color, edgecolor='k', lw=2,
    ))
    ax4.text(x + w / 2, y + h / 2, text, ha='center', va='center',
             fontsize=FS_AXIS - 2, fontweight='bold')

for x in [3.3, 6.2, 8.9]:
    ax4.annotate('', xy=(x + 0.2, 6.2), xytext=(x - 0.2, 6.2),
                 arrowprops=dict(arrowstyle='->', lw=2.5, color='0.2'))
    ax4.text(x, 6.7, 'patch\nmerge', ha='center', fontsize=FS_ANNOT - 1)

ax4.text(
    6.0, 1.5,
    'Patch merging: group 2x2 tokens, concat channels, linear project.\n'
    'Resolution decreases, channels increase -- like CNN strides / FPN backbones.\n'
    'This makes Swin a drop-in hierarchical backbone for detection & segmentation.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Complexity comparison
# -------------------------------------------------------
fig5, axes5 = new_figure(1, 2)
fig5.suptitle(
    'Step 5: Complexity -- Swin scales much more gently with image size',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# N = number of patches ~ image area
N = np.array([16, 64, 256, 1024, 4096])  # e.g. 4^2 ... 64^2
M = 7  # typical Swin window
vit_cost = N ** 2
swin_cost = N * (M ** 2)  # roughly: each token attends M^2 keys; or #windows*(M^2)^2 = N*(M^2)

ax = axes5[0]
ax.loglog(N, vit_cost, 'o-', color='crimson', lw=2.5, label=r'ViT global ~ $N^2$')
ax.loglog(N, swin_cost, 's-', color='seagreen', lw=2.5, label=rf'Swin window ~ $N M^2$ (M={M})')
ax.set_xlabel('number of patches N (~ image area)', fontsize=FS_AXIS)
ax.set_ylabel('attention relationships (scale)', fontsize=FS_AXIS)
ax.set_title('Attention cost vs image size', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, which='both', alpha=0.3)
add_note(
    ax,
    'Global ViT becomes\n'
    'expensive for high-res\n'
    'images / dense tasks.\n\n'
    'Swin keeps local windows\n'
    'with fixed M.',
    y=0.98, va='top', facecolor='lightyellow',
)

ax = axes5[1]
# ratio
ax.semilogx(N, vit_cost / swin_cost, 'D-', color='royalblue', lw=2.5)
ax.set_xlabel('number of patches N', fontsize=FS_AXIS)
ax.set_ylabel('ViT cost / Swin cost', fontsize=FS_AXIS)
ax.set_title('How much more expensive is global attention?', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, which='both', alpha=0.3)
add_note(
    ax,
    'Relative gap grows\n'
    'as images get larger.\n'
    'That is a big reason\n'
    'Swin became popular\n'
    'as a vision backbone.',
    y=0.98, va='top',
)

plt.tight_layout()
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Block recipe
# -------------------------------------------------------
fig6, ax6 = new_figure()
fig6.suptitle(
    'Step 6: One Swin stage = (W-MSA / SW-MSA blocks) then patch merge',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax6.set_xlim(0, 12)
ax6.set_ylim(0, 8)
ax6.axis('off')


def block(ax, x, y, w, h, text, color):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor=color, edgecolor='k', lw=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=FS_AXIS - 2, fontweight='bold')


block(ax6, 3.5, 6.6, 5.0, 0.8, 'tokens (H x W x C)', '#dceaff')
block(ax6, 3.5, 5.2, 5.0, 0.9, 'W-MSA block\n(regular windows)', '#f8d7da')
block(ax6, 3.5, 3.8, 5.0, 0.9, 'SW-MSA block\n(shifted windows)', '#d1ecf1')
block(ax6, 3.5, 2.4, 5.0, 0.8, '... more pairs ...', '#f0f0f0')
block(ax6, 3.5, 1.0, 5.0, 0.8, 'Patch Merging -> next stage', '#fff3cd')

for y0, y1 in [(6.6, 6.1), (5.2, 4.7), (3.8, 3.2), (2.4, 1.8)]:
    ax6.annotate('', xy=(6.0, y1), xytext=(6.0, y0),
                 arrowprops=dict(arrowstyle='->', lw=2, color='k'))

ax6.text(
    1.0, 4.5,
    'Each MSA block still has:\n'
    '  LN -> Attention -> residual\n'
    '  LN -> MLP -> residual\n\n'
    'Only the ATTENTION MASK /\n'
    'partition differs from ViT.',
    ha='left', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Takeaway cards
# -------------------------------------------------------
fig7, ax7 = new_figure()
fig7.suptitle(
    'Step 7: Swin in one sentence',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax7.axis('off')
ax7.text(
    0.5, 0.55,
    'Swin = Transformer attention + CNN-like hierarchy + efficiency tricks.\n\n'
    '1) Window attention: keep compute local / scalable.\n'
    '2) Shifted windows: restore cross-region communication.\n'
    '3) Patch merging stages: multi-scale features for dense prediction.\n\n'
    'Use when you want Transformer modeling power as a vision backbone\n'
    'without paying full global ViT attention cost at high resolution.',
    transform=ax7.transAxes, ha='center', va='center', fontsize=FS_AXIS - 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)
show_maximized(fig7)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("SWIN TRANSFORMER")
print("=" * 60)
print(f"""
Name: Shifted WINdow Transformer

Problems with plain ViT on high-res images:
  Global self-attention cost ~ N^2 with N = #patches ~ image area.

Swin answers:
  1) Restrict attention to non-overlapping M x M windows
     Cost roughly ~ N * M^2   (linear in image size for fixed M)
  2) Shift-window attention (SW-MSA):
       cyclic-shift features by (M/2, M/2)
       -> run the same window attention (+ mask at wrap edges)
       -> reverse-shift back
       so tokens meet new neighbors and information can diffuse
  3) Merge patches between stages to build a hierarchy
     (H/4, H/8, H/16, H/32) with increasing channels

Demo grid: {H}x{W} tokens, window M={WINDOW}
  Global links (ViT-style): {(H*W)**2}
  Windowed links (Swin-style): {cost_swin}

Where it is used:
  Classification backbones, COCO detection, ADE20K segmentation, etc.
  The hierarchical map is the key practical advantage over single-scale ViT.

See also:
  190_transformer.py -- attention mechanics
  300_video_transformer.py -- space-time variants of similar efficiency ideas
""")
