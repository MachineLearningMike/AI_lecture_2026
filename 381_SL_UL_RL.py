"""
381_SL_UL_RL.py

Supervised Learning (SL), Unsupervised Learning (UL), and
Reinforcement Learning (RL) -- three classical ways a model gets a signal.

Sketch:
  1. Three paradigms at a glance
  2. Supervised Learning -- learn from labeled pairs
  3. SL demo -- fit a boundary / curve from (x, y) examples
  4. Unsupervised Learning -- find structure without labels
  5. UL demo -- cluster unlabeled points
  6. Reinforcement Learning -- learn from reward over time
  7. RL demo -- agent, environment, actions, rewards
  8. Contrast table -- what is given, what is learned, when to use which

Companion to 080_model.py, 220_inductive_bias.py,
251_discriminative_generative.py.
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
from matplotlib.patches import FancyBboxPatch
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


# Palette
C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_OLD = '#fce4d6'
C_SL = '#d6eaf8'
C_UL = '#d5f5e3'
C_RL = '#fdebd0'


# -------------------------------------------------------
# FIGURE 1: Three paradigms at a glance
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    'Step 1: Three ways a learner gets its signal',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax1)

round_box(ax1, 0.5, 6.4, 11.0, 1.25, C_DARK, C_DARK, lw=0)
ax1.text(
    6.0, 7.0,
    'Learning = improving a model using feedback.\n'
    'The paradigms differ in WHAT feedback looks like.',
    ha='center', va='center', fontsize=FS_AX_TITLE - 2,
    fontweight='bold', color='white',
)

cards = [
    (0.5, 1.15, 3.4, 4.85, C_SL, '#2874a6', 'Supervised\nLearning (SL)',
     'Teacher gives the right answer.\n\n'
     'Data:  (x, y)\n'
     'Goal:  predict y from new x.\n\n'
     'Independent steps.\n'
     'Intra-step optimization.\n'
     'The simplest learning.'),
    (4.3, 1.15, 3.4, 4.85, C_UL, C_OK, 'Unsupervised\nLearning (UL)',
     'No labels. Find structure.\n\n'
     'Data:  x only\n'
     'Goal:  clusters, factors, density...\n\n'
     'Independent steps.\n'
     'Intra-step optimization.\n'
     'Non-supervised by definition.'),
    (8.1, 1.15, 3.4, 4.85, C_RL, '#d68910', 'Reinforcement\nLearning (RL)',
     'Act, then receive reward later.\n\n'
     'Data:  trajectory\n'
     'Goal:  a policy that maximizes return.\n\n'
     'Dependent steps.\n'
     'Inter-step optimization.'),
]
for x, y, w, h, face, edge, title, body in cards:
    round_box(ax1, x, y, w, h, face, edge, lw=2.4)
    ax1.text(x + w / 2, y + h - 0.75, title,
             ha='center', va='center', fontsize=FS_AX_TITLE - 2,
             fontweight='bold', color=C_DARK)
    ax1.text(x + w / 2, y + 1.85, body,
             ha='center', va='center', fontsize=FS_ANNOT)

ax1.text(
    6.0, 0.55,
    'Same math toolbox (gradients, probability). Different supervision contract.',
    ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
              edgecolor='0.55', alpha=0.95),
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Supervised Learning -- definition
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    'Step 2: Supervised Learning -- learn from labeled pairs',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax2)

round_box(ax2, 0.5, 5.7, 11.0, 1.9, C_DARK, C_DARK, lw=0)
ax2.text(
    6.0, 6.65,
    r'Dataset  $\mathcal{D}=\{(x_i, y_i)\}_{i=1}^{N}$'
    '\n'
    r'Learn  $f_\theta$  so that  $f_\theta(x)\approx y$  on new $x$.',
    ha='center', va='center', fontsize=FS_AX_TITLE - 1,
    fontweight='bold', color='white',
)

round_box(ax2, 0.5, 2.9, 5.3, 2.4, C_SL, '#2874a6', lw=2.2)
ax2.text(
    3.15, 4.7, 'Classification',
    ha='center', va='center', fontsize=FS_AX_TITLE - 2,
    fontweight='bold', color=C_DARK,
)
ax2.text(
    3.15, 3.7,
    'y is a category\n'
    '(spam / not spam, cat / dog).\n'
    'Loss often: cross-entropy.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

round_box(ax2, 6.2, 2.9, 5.3, 2.4, C_GOAL, C_OK, lw=2.2)
ax2.text(
    8.85, 4.7, 'Regression',
    ha='center', va='center', fontsize=FS_AX_TITLE - 2,
    fontweight='bold', color=C_OK,
)
ax2.text(
    8.85, 3.7,
    'y is a number\n'
    '(price, temperature).\n'
    'Loss often: MSE / MAE.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

round_box(ax2, 0.5, 0.4, 11.0, 2.1, C_MUTE, '0.45', lw=1.6)
ax2.text(
    6.0, 1.45,
    'The label is the supervision.\n'
    'Quality of labels often matters more than clever architecture.\n'
    'Inductive bias (model family, loss, regularizer) chooses how to generalize.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: SL tiny demo -- labeled fit
# -------------------------------------------------------
fig3, (ax3a, ax3b) = new_figure(1, 2)
fig3.suptitle(
    'Step 3: SL demo -- labels turn points into a prediction task',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

# Left: binary classification toy
n = 40
x0 = RNG.normal([-1.1, 0.0], 0.45, size=(n, 2))
x1 = RNG.normal([1.1, 0.2], 0.45, size=(n, 2))
ax3a.scatter(x0[:, 0], x0[:, 1], s=42, c='#2874a6', alpha=0.85,
             edgecolors='white', linewidths=0.5, label='y = 0')
ax3a.scatter(x1[:, 0], x1[:, 1], s=42, c='#1e8449', alpha=0.85,
             edgecolors='white', linewidths=0.5, label='y = 1')
# linear separator from class means
w = x1.mean(axis=0) - x0.mean(axis=0)
mid = 0.5 * (x0.mean(axis=0) + x1.mean(axis=0))
# line: w·(x - mid) = 0
t = np.linspace(-2.5, 2.5, 200)
# direction orthogonal to w
orth = np.array([-w[1], w[0]])
orth = orth / (np.linalg.norm(orth) + 1e-8)
line = mid + t[:, None] * orth
ax3a.plot(line[:, 0], line[:, 1], color=C_ACCENT, lw=2.4, label='learned boundary')
ax3a.set_title('Classification: predict the label', fontsize=FS_AX_TITLE - 2)
ax3a.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax3a.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax3a.tick_params(labelsize=FS_TICK)
ax3a.grid(True, alpha=0.25)
ax3a.set_aspect('equal', adjustable='datalim')
ax3a.legend(fontsize=FS_LEGEND, loc='upper left')
ax3a.text(
    0.98, 0.02,
    'Each point comes with y.\n'
    'Fit f(x) to match y.',
    transform=ax3a.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

# Right: regression toy
x_reg = np.linspace(-2.2, 2.2, 18)
y_true = 0.55 * x_reg ** 2 - 0.3 * x_reg
y_obs = y_true + 0.35 * RNG.normal(size=x_reg.size)
# quadratic least squares
Phi = np.column_stack([np.ones_like(x_reg), x_reg, x_reg ** 2])
coef, *_ = np.linalg.lstsq(Phi, y_obs, rcond=None)
xg = np.linspace(-2.5, 2.5, 200)
yg = coef[0] + coef[1] * xg + coef[2] * xg ** 2
ax3b.scatter(x_reg, y_obs, s=55, c='#2874a6', zorder=3,
             edgecolors='white', linewidths=0.6, label='labeled (x, y)')
ax3b.plot(xg, yg, color=C_ACCENT, lw=2.6, label=r'fitted $f_\theta(x)$')
ax3b.set_title('Regression: predict the number', fontsize=FS_AX_TITLE - 2)
ax3b.set_xlabel(r'$x$', fontsize=FS_AXIS)
ax3b.set_ylabel(r'$y$', fontsize=FS_AXIS)
ax3b.tick_params(labelsize=FS_TICK)
ax3b.grid(True, alpha=0.25)
ax3b.legend(fontsize=FS_LEGEND, loc='upper left')
ax3b.text(
    0.98, 0.02,
    'Supervision = the y values.\n'
    'Loss measures miss vs y.',
    transform=ax3b.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig3.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Unsupervised Learning -- definition
# -------------------------------------------------------
fig4, ax4 = new_figure()
fig4.suptitle(
    'Step 4: Unsupervised Learning -- structure without labels',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax4)

round_box(ax4, 0.5, 5.7, 11.0, 1.9, C_DARK, C_DARK, lw=0)
ax4.text(
    6.0, 6.65,
    r'Dataset  $\mathcal{D}=\{x_i\}_{i=1}^{N}$   --  no $y$.'
    '\n'
    'Learn a useful description of how the x live.',
    ha='center', va='center', fontsize=FS_AX_TITLE - 1,
    fontweight='bold', color='white',
)

ul_cards = [
    (0.5, 2.7, 'Clustering', 'Group similar x\n(k-means, GMM, ...).'),
    (4.3, 2.7, 'Dimensionality\nreduction', 'Find a lower-d map\n(PCA, autoencoders).'),
    (8.1, 2.7, 'Density /\ngenerative', 'Model p(x)\n(sample or score).'),
]
for x, y, title, body in ul_cards:
    round_box(ax4, x, y, 3.4, 2.6, C_UL, C_OK, lw=2.2)
    ax4.text(x + 1.7, y + 1.85, title, ha='center', va='center',
             fontsize=FS_ANNOT + 2, fontweight='bold', color=C_OK)
    ax4.text(x + 1.7, y + 0.85, body, ha='center', va='center',
             fontsize=FS_ANNOT + 1)

round_box(ax4, 0.5, 0.35, 11.0, 2.0, 'lightyellow', '0.55', lw=1.6)
ax4.text(
    6.0, 1.35,
    'There is no single "correct label" to match.\n'
    'Success is judged by usefulness: separation, compression, sampling quality,\n'
    'or how well the found structure helps a later supervised task.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: UL tiny demo -- clustering
# -------------------------------------------------------
fig5, (ax5a, ax5b) = new_figure(1, 2)
fig5.suptitle(
    'Step 5: UL demo -- same points, no labels; discover clusters',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

# Three blobs, shown unlabeled then colored by discovered cluster
c0 = RNG.normal([-1.6, -0.8], 0.35, size=(35, 2))
c1 = RNG.normal([1.5, -0.6], 0.38, size=(35, 2))
c2 = RNG.normal([0.1, 1.5], 0.32, size=(30, 2))
Xu = np.vstack([c0, c1, c2])

ax5a.scatter(Xu[:, 0], Xu[:, 1], s=42, c='0.45', alpha=0.85,
             edgecolors='white', linewidths=0.4)
ax5a.set_title('Unlabeled cloud of x', fontsize=FS_AX_TITLE - 2)
ax5a.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax5a.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax5a.tick_params(labelsize=FS_TICK)
ax5a.grid(True, alpha=0.25)
ax5a.set_aspect('equal', adjustable='datalim')
ax5a.text(
    0.98, 0.02,
    'No y is given.\n'
    'Only geometry / density.',
    transform=ax5a.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

# Tiny k-means (k=3)
K = 3
centers = Xu[RNG.choice(len(Xu), size=K, replace=False)].copy()
for _ in range(12):
    d2 = ((Xu[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1)
    for k in range(K):
        if np.any(labels == k):
            centers[k] = Xu[labels == k].mean(axis=0)

colors = ['#2874a6', '#1e8449', '#c0392b']
for k in range(K):
    pts = Xu[labels == k]
    ax5b.scatter(pts[:, 0], pts[:, 1], s=42, c=colors[k], alpha=0.85,
                 edgecolors='white', linewidths=0.4, label=f'cluster {k}')
ax5b.scatter(centers[:, 0], centers[:, 1], s=220, c='gold', marker='*',
             edgecolors='k', linewidths=0.8, zorder=5, label='centers')
ax5b.set_title('After clustering (e.g. k-means)', fontsize=FS_AX_TITLE - 2)
ax5b.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax5b.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax5b.tick_params(labelsize=FS_TICK)
ax5b.grid(True, alpha=0.25)
ax5b.set_aspect('equal', adjustable='datalim')
ax5b.legend(fontsize=FS_LEGEND, loc='upper left')
ax5b.text(
    0.98, 0.02,
    'Labels here are invented\n'
    'by the algorithm, not given.',
    transform=ax5b.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
)

fig5.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Reinforcement Learning -- definition
# -------------------------------------------------------
fig6, ax6 = new_figure()
fig6.suptitle(
    'Step 6: Reinforcement Learning -- learn from reward over time',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax6)

round_box(ax6, 0.5, 6.35, 11.0, 1.3, C_DARK, C_DARK, lw=0)
ax6.text(
    6.0, 7.0,
    'An agent acts in an environment, receives reward, and improves its policy.',
    ha='center', va='center', fontsize=FS_AX_TITLE - 2,
    fontweight='bold', color='white',
)

# Loop diagram
boxes = [
    (0.7, 3.5, 2.6, 1.8, C_RL, '#d68910', 'Agent\npolicy  π(a|s)'),
    (4.7, 3.5, 2.6, 1.8, C_NEW, '#2874a6', 'Environment\n(world / sim)'),
    (8.7, 3.5, 2.6, 1.8, C_GOAL, C_OK, 'Reward r\n(+ next state)'),
]
for x, y, w, h, face, edge, title in boxes:
    round_box(ax6, x, y, w, h, face, edge, lw=2.2)
    ax6.text(x + w / 2, y + h / 2, title, ha='center', va='center',
             fontsize=FS_ANNOT + 2, fontweight='bold')

arrow(ax6, 3.4, 4.4, 4.6, 4.4, C_DARK, lw=2.4)   # action
ax6.text(4.0, 4.85, 'action a', ha='center', fontsize=FS_ANNOT, color=C_DARK)
arrow(ax6, 7.4, 4.4, 8.6, 4.4, C_DARK, lw=2.4)   # to reward box
arrow(ax6, 10.0, 3.4, 10.0, 2.35, C_DARK, lw=2.2)
arrow(ax6, 10.0, 2.2, 2.0, 2.2, C_DARK, lw=2.2)
arrow(ax6, 2.0, 2.2, 2.0, 3.4, C_DARK, lw=2.2)
ax6.text(6.0, 2.55, 'feedback: r, s′   →   update π', ha='center',
         fontsize=FS_ANNOT + 1, fontweight='bold', color=C_DARK)

round_box(ax6, 0.5, 0.3, 11.0, 1.5, C_MUTE, '0.45', lw=1.6)
ax6.text(
    6.0, 1.05,
    'Unlike SL, nobody hands the correct action at each step.\n'
    r'Objective: maximize expected return  $R = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots$',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: RL tiny demo -- grid path with rewards
# -------------------------------------------------------
fig7, (ax7a, ax7b) = new_figure(1, 2)
fig7.suptitle(
    'Step 7: RL demo -- choose actions to collect return',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

# Small grid world -- rewards written on cells
rewards = {
    (0, 4): +1.0,   # goal
    (2, 2): -1.0,   # pit
    (4, 0): -0.2,   # mild trap
}
# A decent path avoiding the pit
path = [(4, 0), (3, 0), (2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
# A bad path that falls in the pit
bad = [(4, 0), (4, 1), (3, 1), (2, 1), (2, 2)]


def draw_grid(ax, path_cells, title, path_color):
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.tick_params(labelsize=FS_TICK)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=FS_AX_TITLE - 2)
    for r in range(5):
        for c in range(5):
            face = '#f8f9f9'
            if (r, c) in rewards:
                face = C_GOAL if rewards[(r, c)] > 0 else '#f5b7b1'
            ax.add_patch(patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=face, edgecolor='0.55', lw=1.2,
            ))
            if (r, c) in rewards:
                ax.text(c, r, f'{rewards[(r, c)]:+.1f}',
                        ha='center', va='center', fontsize=FS_ANNOT,
                        fontweight='bold')
    # path
    ys = [p[0] for p in path_cells]
    xs = [p[1] for p in path_cells]
    ax.plot(xs, ys, '-o', color=path_color, lw=2.6, ms=8,
            markerfacecolor='white', markeredgewidth=2.0)
    ax.plot(xs[0], ys[0], 's', color=C_DARK, ms=12, label='start')
    ax.plot(xs[-1], ys[-1], '*', color='#d4ac0d', ms=18, label='end')
    ret = sum(rewards.get(p, -0.05) for p in path_cells[1:])  # small step cost
    ax.text(
        0.02, 0.02,
        f'return ≈ {ret:.2f}\n(sum of rewards along path)',
        transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )


draw_grid(ax7a, path, 'Better policy: reach +1, avoid pit', C_OK)
draw_grid(ax7b, bad, 'Worse policy: falls into -1', C_ACCENT)

fig7.text(
    0.5, 0.02,
    'SL would need the correct action labeled at every state.\n'
    'RL only needs a scalar reward -- credit must be assigned across time.',
    ha='center', va='bottom', fontsize=FS_ANNOT + 1, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.35', facecolor=C_GOAL, edgecolor=C_OK, alpha=0.96),
)
fig7.tight_layout(rect=[0, 0.08, 1, 0.94])
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Contrast table
# -------------------------------------------------------
fig8, ax8 = new_figure()
fig8.suptitle(
    'Step 8: SL vs UL vs RL -- choose by the supervision you have',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax8.set_xlim(0, 1)
ax8.set_ylim(0, 1)
ax8.axis('off')

table = [
    ['', 'Supervised (SL)', 'Unsupervised (UL)', 'Reinforcement (RL)'],
    ['Given', '(x, y) pairs', 'x only', 'states, actions, rewards'],
    ['Signal', 'label / target y', 'structure in x', 'scalar reward over time'],
    ['Learns', 'predictor f(x)→y', 'clusters / latents / p(x)', 'policy π(a|s)'],
    ['Typical loss', 'CE, MSE, ...', 'recon, NLL, k-means, ...', 'return / TD / policy grad'],
    ['Hard part', 'label cost & noise', 'what "good structure" means', 'credit assignment, exploration'],
    ['Use when', 'you can label outcomes', 'you want representation', 'you can act & score episodes'],
]

col_x = [0.02, 0.18, 0.45, 0.72]
row_h = 0.10
y0 = 0.88
for r, row in enumerate(table):
    y = y0 - r * row_h
    if r == 0:
        faces = ['#e8eef8', C_SL, C_UL, C_RL]
    else:
        faces = ['#eef2f5', 'white', 'white', 'white']
    for c, (x, cell) in enumerate(zip(col_x, row)):
        w = (col_x[c + 1] - x - 0.01) if c < len(col_x) - 1 else (0.98 - x)
        ax8.add_patch(patches.FancyBboxPatch(
            (x, y - 0.035), w, row_h - 0.015,
            boxstyle='round,pad=0.004,rounding_size=0.01',
            transform=ax8.transAxes,
            facecolor=faces[c], edgecolor='0.65', lw=1.0,
        ))
        ax8.text(
            x + 0.012, y + 0.01, cell, transform=ax8.transAxes,
            fontsize=FS_ANNOT if r else FS_ANNOT + 1,
            fontweight='bold' if (r == 0 or c == 0) else 'normal',
            va='center', ha='left',
        )

ax8.text(
    0.5, 0.08,
    'Modern practice mixes them: pretrain UL / self-supervised, fine-tune SL,\n'
    'sometimes align with RL-style preference rewards (e.g. RLHF).',
    transform=ax8.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
              edgecolor='0.55', alpha=0.96),
)
show_maximized(fig8)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("SUPERVISED / UNSUPERVISED / REINFORCEMENT LEARNING")
print("=" * 60)
print("""
Step 1 -- three supervision contracts:
  SL: labeled pairs (x, y); independent steps; intra-step optimization;
      the simplest learning.
  UL: x only; invent structure; independent steps; intra-step optimization;
      non-supervised by definition.
  RL: act for return; dependent steps; inter-step optimization.

Step 2-3 -- Supervised Learning:
  Classification (discrete y) or regression (numeric y).
  Demo: labeled clouds => boundary; noisy (x, y) => fitted curve.
  Bottleneck is often labels, not model capacity.

Step 4-5 -- Unsupervised Learning:
  Clustering, compression, density / generative models.
  Demo: unlabeled points => algorithm-made cluster ids.
  "Correctness" is usefulness, not matching a given y.

Step 6-7 -- Reinforcement Learning:
  Agent <-> environment loop; reward may be delayed.
  Demo: grid paths with different returns; credit over time.
  Hard parts: exploration and assigning credit across steps.

Step 8 -- how to choose:
  Can you label outcomes? -> SL
  Need structure / representation from raw x? -> UL
  Can you act and score episodes? -> RL
  Real systems often stack UL pretraining + SL fine-tuning
  (+ optional RL-style preference / reward tuning).

See also: 080_model.py, 220_inductive_bias.py,
          251_discriminative_generative.py
""")
