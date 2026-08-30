"""
190_loss_functions.py

Different loss functions produce different optima AND different
convergence behavior under the same optimizer.

Demonstrated on a 1D linear fit  y ~ w x + b  with a few outliers:

    MSE   (squared error)   -- Gaussian MLE; sensitive to outliers; smooth bowl
    MAE   (absolute error)  -- Laplace MLE; robust; sharper valley / constant-scale grads
    Huber                   -- MSE near 0, MAE in the tails; a practical compromise

Taxonomy table (Regression / Binary / Multi-class), then
formulas + intuition + viz for MSE / MAE / Huber, then demos.

Then: classification error (0-1), sigmoid + binary CE (Shannon entropy),
multi-class CE / softmax, and why soft losses train instead.

Companion to 120_maximum_likelihood.py (MSE vs MAE as MLE),
080_model.py (training with MSE), and 320_softmax.py (multi-class scores).
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
        width_px = root.winfo_screenwidth()
        height_px = root.winfo_screenheight()
        root.destroy()
        return (width_px / dpi * fraction, height_px / dpi * fraction)
    except Exception:
        return (16 * fraction / 0.95, 9 * fraction / 0.95)


def new_figure(*subplot_args, **subplot_kwargs):
    subplot_kwargs.setdefault('figsize', screen_figsize())
    subplot_kwargs.setdefault('dpi', FIGURE_DPI)
    return plt.subplots(*subplot_args, **subplot_kwargs)


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


def add_legend_and_note(
    ax, note, *,
    legend_loc='lower right',
    legend_anchor=(0.98, 0.54),
    note_xy=(0.98, 0.46),
    note_ha='right',
    note_va='top',
    facecolor='white',
    show_legend=True,
):
    if show_legend:
        ax.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_anchor,
            fontsize=FS_LEGEND,
            framealpha=0.95,
            borderaxespad=0.0,
        )
    ax.text(
        note_xy[0], note_xy[1], note,
        transform=ax.transAxes,
        fontsize=FS_ANNOT,
        va=note_va, ha=note_ha,
        bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.95),
    )


def add_vcenter_legend_and_note(ax, note, facecolor='white'):
    """Default stacked legend+note on the mid-right (legacy helper)."""
    add_legend_and_note(ax, note, facecolor=facecolor)


# -------------------------------------------------------
# Loss definitions (per-residual and reductions)
# -------------------------------------------------------
HUBER_DELTA = 0.75  # tighter than 1: separates Huber from MSE on big outliers


def mse_residual(r):
    return 0.5 * r ** 2


def mae_residual(r):
    return np.abs(r)


def huber_residual(r, delta=HUBER_DELTA):
    abs_r = np.abs(r)
    return np.where(
        abs_r <= delta,
        0.5 * r ** 2,
        delta * (abs_r - 0.5 * delta),
    )


def mse_grad_r(r):
    return r


def mae_grad_r(r):
    return np.sign(r)


def huber_grad_r(r, delta=HUBER_DELTA):
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, r, delta * np.sign(r))


LOSSES = {
    'MSE': {
        'color': 'darkorange',
        'residual': mse_residual,
        'grad_r': mse_grad_r,
        'note': 'Gaussian MLE',
    },
    'MAE': {
        'color': 'purple',
        'residual': mae_residual,
        'grad_r': mae_grad_r,
        'note': 'Laplace MLE',
    },
    'Huber': {
        'color': 'seagreen',
        'residual': huber_residual,
        'grad_r': huber_grad_r,
        'note': f'MSE if |r|<{HUBER_DELTA}, else MAE-like (slope ±{HUBER_DELTA})',
    },
}


# -------------------------------------------------------
# Data: mostly linear, plus strong ONE-SIDED outliers
# Asymmetry makes MSE / MAE / Huber disagree more visibly.
# -------------------------------------------------------
N = 50
W_TRUE, B_TRUE = 1.5, -0.5

x = np.linspace(-2.0, 2.0, N)
y_clean = W_TRUE * x + B_TRUE + 0.20 * RNG.normal(size=N)
y = y_clean.copy()
# Cluster big positive outliers on the right: MSE tilts up; MAE resists
outlier_idx = np.array([N - 8, N - 6, N - 4, N - 2, N - 1])
y[outlier_idx] += np.array([8.0, 9.5, 11.0, 10.0, 12.0])
# One negative outlier on the far left for extra leverage disagreement
left_outlier = 1
outlier_idx = np.concatenate([np.array([left_outlier]), outlier_idx])
y[left_outlier] -= 7.0


def predict(theta, x_vals):
    w, b = theta
    return w * x_vals + b


def empirical_loss(theta, loss_name):
    r = predict(theta, x) - y
    return float(np.mean(LOSSES[loss_name]['residual'](r)))


def empirical_grad(theta, loss_name):
    """Gradient of mean loss w.r.t. (w, b)."""
    w, b = theta
    r = predict(theta, x) - y
    dL_dr = LOSSES[loss_name]['grad_r'](r) / len(x)  # mean
    # d r / d w = x,  d r / d b = 1
    gw = float(np.sum(dL_dr * x))
    gb = float(np.sum(dL_dr))
    return np.array([gw, gb], dtype=float)


def fit_gd(loss_name, theta0, lr, steps):
    theta = np.array(theta0, dtype=float)
    path = [theta.copy()]
    losses = [empirical_loss(theta, loss_name)]
    grad_norms = []
    for _ in range(steps):
        g = empirical_grad(theta, loss_name)
        grad_norms.append(float(np.linalg.norm(g)))
        theta = theta - lr * g
        path.append(theta.copy())
        losses.append(empirical_loss(theta, loss_name))
    return {
        'path': np.stack(path),
        'losses': np.array(losses),
        'grad_norms': np.array(grad_norms),
        'theta': theta,
    }


THETA0 = np.array([0.0, 2.5])  # shared init, away from the true line
SGD_STEPS = 200
# Per-loss learning rates so each can move meaningfully
# (MAE grads are ~O(1), MSE grads scale with residual size)
LR = {
    'MSE': 0.05,
    'MAE': 0.20,
    'Huber': 0.08,
}

results = {
    name: fit_gd(name, THETA0, LR[name], SGD_STEPS)
    for name in LOSSES
}


# -------------------------------------------------------
# -------------------------------------------------------
# FIGURE 1: Taxonomy table (English; same contents as the image)
# -------------------------------------------------------
fig0, ax0 = new_figure()
fig0.suptitle(
    'Step 1: Loss taxonomy -- regression / binary / multi-class',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax0.set_xlim(0, 12)
ax0.set_ylim(0, 8)
ax0.axis('off')

# Same English contents as the image
headers = ['Category', 'Task', 'Data', 'Loss Function']
rows = [
    [
        'Regression',
        'Predict continuous values',
        'Input: Continuous or Categorical\n'
        'Output: Continuous real values\n'
        '(Continuous Probability Distribution)',
        'MSE',
    ],
    [
        'Binary Classification',
        'Choose one of two',
        'Input: Continuous or Categorical\n'
        'Output: 0 or 1\n'
        '(Bernoulli Distribution)',
        'Binary Cross Entropy',
    ],
    [
        'Multi-class Classification',
        'Choose one of three or more',
        'Input: Continuous or Categorical\n'
        'Output: [0,0,0,1,0] One-hot vector\n'
        '(Categorical Distribution)',
        'Cross Entropy',
    ],
]

col_x = [0.40, 2.70, 5.20, 9.70]
col_w = [2.15, 2.35, 4.35, 1.90]
header_y, header_h = 6.35, 0.90
row_h = 1.95
row0_y = 5.25

for x, w, h in zip(col_x, col_w, headers):
    ax0.add_patch(patches.Rectangle(
        (x, header_y), w, header_h,
        facecolor='#d6eaf8', edgecolor='#7f8c8d', lw=1.5,
    ))
    ax0.text(
        x + w / 2, header_y + header_h / 2, h,
        ha='center', va='center', fontsize=FS_ANNOT + 2, fontweight='bold',
        color='#1a5276',
    )

for r, row in enumerate(rows):
    y = row0_y - r * row_h
    for c, (x, w, cell) in enumerate(zip(col_x, col_w, row)):
        ax0.add_patch(patches.Rectangle(
            (x, y), w, row_h - 0.10,
            facecolor='white', edgecolor='#7f8c8d', lw=1.5,
        ))
        ax0.text(
            x + w / 2, y + (row_h - 0.10) / 2, cell,
            ha='center', va='center',
            fontsize=FS_ANNOT + (1 if c in (0, 3) else 0),
            fontweight='bold' if c in (0, 3) else 'normal',
        )

show_maximized(fig0)


# FIGURE 2: Formulas + intuition + viz for MSE / MAE / Huber
# -------------------------------------------------------
fig1, axes1f = new_figure(2, 3, gridspec_kw={'height_ratios': [1.15, 1.0]})
fig1.suptitle(
    'Step 2: Formulas, intuition, and the shape of each residual loss',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

r_viz = np.linspace(-4.0, 4.0, 500)
cards_reg = [
    (
        'MSE',
        r'$\ell_{\mathrm{MSE}}(r)=\frac{1}{2} r^{2}$',
        r'$\partial\ell/\partial r = r$',
        'Intuition: punish large mistakes\n'
        'quadratically. Outliers dominate.\n'
        'Smooth bowl; Gaussian MLE.',
        'darkorange',
        mse_residual(r_viz),
    ),
    (
        'MAE',
        r'$\ell_{\mathrm{MAE}}(r)=|r|$',
        r'$\partial\ell/\partial r =\mathrm{sign}(r)$',
        'Intuition: every unit of error\n'
        'costs the same. More robust.\n'
        'Sharper valley; Laplace MLE.',
        'purple',
        mae_residual(r_viz),
    ),
    (
        'Huber',
        rf'$\ell_\delta(r)=\frac{{1}}{{2}}r^{2}$  if $|r|\leq{HUBER_DELTA:g}$'
        '\n'
        rf'$\ell_\delta(r)=\delta(|r|-\frac{{1}}{{2}}\delta)$  if $|r|>{HUBER_DELTA:g}$',
        r'$\partial\ell/\partial r =\mathrm{clip}(r,\pm\delta)$'
        f'\n'
        rf'($\delta={HUBER_DELTA:g}$)',
        'Intuition: MSE near zero,\n'
        'MAE in the tails. Stable grads\n'
        'without ignoring moderate residuals.',
        'seagreen',
        huber_residual(r_viz),
    ),
]

for j, (name, formula, grad, intuition, color, ys) in enumerate(cards_reg):
    ax_t = axes1f[0, j]
    ax_t.set_xlim(0, 1)
    ax_t.set_ylim(0, 1)
    ax_t.axis('off')
    ax_t.add_patch(patches.FancyBboxPatch(
        (0.04, 0.08), 0.92, 0.84,
        boxstyle='round,pad=0.02,rounding_size=0.04',
        transform=ax_t.transAxes,
        facecolor='white', edgecolor=color, lw=2.4,
    ))
    ax_t.text(0.5, 0.88, name, transform=ax_t.transAxes,
              ha='center', va='center', fontsize=FS_AX_TITLE - 1,
              fontweight='bold', color=color)
    ax_t.text(0.5, 0.68, formula, transform=ax_t.transAxes,
              ha='center', va='center', fontsize=FS_ANNOT + 1)
    ax_t.text(0.5, 0.48, grad, transform=ax_t.transAxes,
              ha='center', va='center', fontsize=FS_ANNOT)
    ax_t.text(0.5, 0.22, intuition, transform=ax_t.transAxes,
              ha='center', va='center', fontsize=FS_ANNOT,
              bbox=dict(boxstyle='round,pad=0.25', facecolor='lightyellow', alpha=0.95))

    ax_p = axes1f[1, j]
    ax_p.plot(r_viz, ys, color=color, lw=2.8)
    ax_p.axvline(0, color='k', alpha=0.25)
    ax_p.axhline(0, color='k', alpha=0.25)
    if name == 'Huber':
        ax_p.axvline(-HUBER_DELTA, color='0.5', ls='--', lw=1.0)
        ax_p.axvline(HUBER_DELTA, color='0.5', ls='--', lw=1.0)
    ax_p.set_title(rf'viz: $\ell(r)$ for {name}', fontsize=FS_AX_TITLE - 2)
    ax_p.set_xlabel(r'residual $r=\hat y-y$', fontsize=FS_AXIS)
    ax_p.set_ylabel(r'$\ell(r)$', fontsize=FS_AXIS)
    ax_p.tick_params(labelsize=FS_TICK)
    ax_p.grid(True, alpha=0.3)
    ax_p.set_xlim(-4, 4)

fig1.tight_layout(rect=[0, 0.03, 1, 0.94])
fig1.text(
    0.5, 0.01,
    r'Empirical training loss is the mean over examples:  '
    r'$L(\theta)=\frac{1}{N}\sum_i \ell\!\left(f_\theta(x_i)-y_i\right)$.',
    ha='center', va='bottom', fontsize=FS_ANNOT + 1, fontweight='bold',
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 3: Shapes of the losses / grads
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    'Step 3: Loss shape decides the gradient signal',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

r = np.linspace(-4, 4, 500)

ax = axes1[0]
for name, cfg in LOSSES.items():
    ax.plot(r, cfg['residual'](r), color=cfg['color'], linewidth=2.5, label=name)
ax.axvline(0, color='k', alpha=0.3)
ax.set_title(r'Penalty $\ell(r)$ on residual $r=\hat y-y$', fontsize=FS_AX_TITLE)
ax.set_xlabel('residual r', fontsize=FS_AXIS)
ax.set_ylabel(r'$\ell(r)$', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(loc='lower right', fontsize=FS_LEGEND, framealpha=0.95)
add_legend_and_note(
    ax,
    'MSE grows as r^2\n'
    '-> huge outliers dominate.\n\n'
    'MAE grows as |r|\n'
    '-> linear, more robust.\n\n'
    'Huber blends both.',
    show_legend=False,
    note_xy=(0.50, 0.98),
    note_ha='center',
    note_va='top',
)
ax.grid(True, alpha=0.3)

ax = axes1[1]
for name, cfg in LOSSES.items():
    ax.plot(r, cfg['grad_r'](r), color=cfg['color'], linewidth=2.5, label=name)
ax.axhline(0, color='k', alpha=0.3)
ax.axvline(0, color='k', alpha=0.3)
ax.set_title(r'Gradient $\partial\ell/\partial r$  (what GD actually uses)', fontsize=FS_AX_TITLE)
ax.set_xlabel('residual r', fontsize=FS_AXIS)
ax.set_ylabel(r'$\partial\ell/\partial r$', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(loc='lower right', fontsize=FS_LEGEND, framealpha=0.95)
add_legend_and_note(
    ax,
    'MSE: grad = r\n'
    '  (can explode)\n\n'
    'MAE: grad = sign(r)\n'
    '  (constant size)\n\n'
    'Huber: caps the grad\n'
    f'  at ±{HUBER_DELTA:g}',
    show_legend=False,
    note_xy=(0.02, 0.98),
    note_ha='left',
    note_va='top',
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 4: Fitted lines -- different losses, different answers
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    'Step 4: Same data / init / optimizer family -- different fits',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax2.scatter(x, y, s=35, color='0.55', alpha=0.85, label='data', zorder=2)
ax2.scatter(x[outlier_idx], y[outlier_idx], s=70, color='red',
            edgecolor='k', linewidth=0.6, label='outliers', zorder=3)

x_line = np.linspace(-2.2, 2.2, 200)
ax2.plot(x_line, W_TRUE * x_line + B_TRUE, 'k--', linewidth=2.0,
         label='true line', zorder=1)

# Ghost trajectories: intermediate predictions along GD (diversity of paths)
traj_steps = np.linspace(0, SGD_STEPS, 8, dtype=int)
for name, cfg in LOSSES.items():
    path = results[name]['path']
    for s in traj_steps[:-1]:
        th_s = path[s]
        ax2.plot(
            x_line, predict(th_s, x_line),
            color=cfg['color'], linewidth=1.0, alpha=0.18, zorder=3,
        )
    th = results[name]['theta']
    ax2.plot(
        x_line, predict(th, x_line),
        color=cfg['color'], linewidth=3.0,
        label=rf'{name}: $w={th[0]:.2f},\ b={th[1]:.2f}$',
        zorder=4,
    )

ax2.set_xlabel('x', fontsize=FS_AXIS)
ax2.set_ylabel('y', fontsize=FS_AXIS)
ax2.tick_params(axis='both', labelsize=FS_TICK)
ax2.set_title('GD trajectories + final fits (MSE / MAE / Huber)', fontsize=FS_AX_TITLE)
ax2.set_ylim(-8.5, 14.0)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
# Move annotation upward by 0.5 inches relative to the laid-out axes.
_pos2 = ax2.get_position()
_dy2 = 0.5 / max(fig2.get_figheight() * _pos2.height, 1e-6)
add_legend_and_note(
    ax2,
    'Faint lines = path during GD.\n'
    'Bold = final fit.\n\n'
    'One-sided outliers pull\n'
    'MSE up hard; MAE resists;\n'
    'Huber sits in between.\n\n'
    f'True: w={W_TRUE:g}, b={B_TRUE:g}',
    legend_loc='upper left',
    legend_anchor=(0.02, 0.98),
    note_xy=(0.02, 0.52 + _dy2),
    note_ha='left',
    note_va='top',
)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 5: Convergence curves differ
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    'Step 5: Convergence speed and stability depend on the loss',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes3[0]
for name, cfg in LOSSES.items():
    # Plot each curve in its OWN loss units (normalized to start at 1)
    L = results[name]['losses']
    ax.plot(L / L[0], color=cfg['color'], linewidth=2.5,
            label=rf'{name} (lr={LR[name]:g})')
ax.set_xlabel('SGD step', fontsize=FS_AXIS)
ax.set_ylabel(r'loss / loss$_0$  (own units)', fontsize=FS_AXIS)
ax.set_title('Normalized mini-batch loss', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    'Curves are not directly\n'
    'comparable in raw units.\n'
    'Normalization shows each\n'
    'optimizer-loss pair\n'
    'closing its own gap.',
)
ax.grid(True, alpha=0.3)

ax = axes3[1]
for name, cfg in LOSSES.items():
    ax.plot(results[name]['grad_norms'], color=cfg['color'],
            linewidth=2.5, label=name)
ax.set_xlabel('SGD step', fontsize=FS_AXIS)
ax.set_ylabel(r'$\|\nabla_\theta L\|$', fontsize=FS_AXIS)
ax.set_title('Gradient magnitude over training', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    'MSE starts with large\n'
    'grads (outlier-driven),\n'
    'then shrinks as r->0.\n\n'
    'MAE grads stay more\n'
    'evenly sized (sign).',
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 6: Parameter-space paths (geometry of each loss)
# -------------------------------------------------------
fig4, ax4 = new_figure()
fig4.suptitle(
    'Step 6: Paths in $(w,b)$ space -- each loss has its own landscape',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# Contours of MSE landscape as a reference backdrop
w_grid = np.linspace(-1.5, 3.0, 120)
b_grid = np.linspace(-2.5, 3.0, 120)
Wg, Bg = np.meshgrid(w_grid, b_grid)
# vectorized MSE contour
R = Wg[..., None] * x[None, None, :] + Bg[..., None] - y[None, None, :]
Mse_grid = 0.5 * np.mean(R ** 2, axis=-1)
cs = ax4.contour(Wg, Bg, Mse_grid, levels=20, colors='0.7', linewidths=0.8)
ax4.clabel(cs, inline=True, fontsize=7, fmt='%.1f')

ax4.scatter([THETA0[0]], [THETA0[1]], c='orange', s=70, zorder=5, label='start')
ax4.scatter([W_TRUE], [B_TRUE], c='k', s=90, marker='*', zorder=5, label='true $(w,b)$')

for name, cfg in LOSSES.items():
    path = results[name]['path']
    ax4.plot(path[:, 0], path[:, 1], '-', color=cfg['color'], linewidth=2.2, label=name)
    ax4.scatter(path[-1, 0], path[-1, 1], color=cfg['color'], s=55, zorder=4)

ax4.set_xlabel('w', fontsize=FS_AXIS)
ax4.set_ylabel('b', fontsize=FS_AXIS)
ax4.tick_params(axis='both', labelsize=FS_TICK)
ax4.set_title('Gray contours = MSE landscape (for reference)', fontsize=FS_AX_TITLE)
add_legend_and_note(
    ax4,
    'Same start, same GD.\n'
    'Different losses bend\n'
    'the path toward\n'
    'different endpoints.\n\n'
    'Loss choice = inductive\n'
    'bias + convergence style.',
    legend_loc='lower left',
    legend_anchor=(0.02, 0.30),
    note_xy=(0.02, 0.28),
    note_ha='left',
    note_va='top',
)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 7: Classification error -- formula, intuition, viz
# -------------------------------------------------------
# Make LEFT panel +2 inches broader WITHOUT shrinking the right panel:
# keep right's absolute width, grow the whole figure width by 2 inches.
_fw6, _fh6 = screen_figsize()
_usable6 = _fw6 * 0.92
_right_in6 = _usable6 * 1.0 / (1.15 + 1.0)          # unchanged vs original split
_left_in6 = _usable6 * 1.15 / (1.15 + 1.0) + 2.0     # original left + 2 in
_fw6_wide = (_left_in6 + _right_in6) / 0.92
fig5, axes5 = plt.subplots(
    1, 2,
    figsize=(_fw6_wide, _fh6),
    dpi=FIGURE_DPI,
    gridspec_kw={'width_ratios': [_left_in6, _right_in6]},
)
fig5.suptitle(
    'Step 7: Classification error -- formula, intuition, and the 0-1 shape',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax5a, ax5b = axes5

# Left: formulas + intuition cards (fill the wider left panel)
ax5a.set_xlim(0, 10)
ax5a.set_ylim(0, 10)
ax5a.axis('off')
ax5a.set_title('Formulas + meaning', fontsize=FS_AX_TITLE - 1)

_gap = 0.22
_x0, _w_full = 0.12, 9.76
_xc6 = _x0 + _w_full / 2.0
_h_bot, _h_mid, _h_top = 3.15, 3.35, 2.75
_bot_y = 0.12
_mid_y = _bot_y + _h_bot + _gap
_top_y = _mid_y + _h_mid + _gap
_w_half = (_w_full - _gap) / 2.0

ax5a.add_patch(patches.FancyBboxPatch(
    (_x0, _top_y), _w_full, _h_top,
    boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#1a5276', edgecolor='#1a5276', lw=0,
))
ax5a.text(
    _xc6, _top_y + _h_top * 0.68,
    r'Per example:   $\ell_{0\text{-}1}(\hat y,y)=\mathbf{1}[\hat y\neq y]$',
    ha='center', va='center', fontsize=FS_ANNOT + 2,
    fontweight='bold', color='white',
)
ax5a.text(
    _xc6, _top_y + _h_top * 0.32,
    r'Error rate:   $\mathrm{Err}=\frac{1}{N}\sum_{i=1}^{N}'
    r'\mathbf{1}[\hat y_i\neq y_i]$',
    ha='center', va='center', fontsize=FS_ANNOT + 2,
    fontweight='bold', color='white',
)

ax5a.add_patch(patches.FancyBboxPatch(
    (_x0, _mid_y), _w_half, _h_mid,
    boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#d6eaf8', edgecolor='#2874a6', lw=2.0,
))
ax5a.text(_x0 + _w_half / 2, _mid_y + _h_mid * 0.86, 'Binary',
          ha='center', fontsize=FS_ANNOT + 2, fontweight='bold', color='#2874a6')
ax5a.text(
    _x0 + _w_half / 2, _mid_y + _h_mid * 0.42,
    r'$\hat y=\mathbf{1}[s(x)>0]$'
    '\n\n'
    r'$\mathrm{Err}=\frac{\mathrm{FP}+\mathrm{FN}}{N}$'
    '\n\n'
    'Intuition: count points\n'
    'on the wrong side of\n'
    'the threshold.',
    ha='center', va='center', fontsize=FS_ANNOT,
)

ax5a.add_patch(patches.FancyBboxPatch(
    (_x0 + _w_half + _gap, _mid_y), _w_half, _h_mid,
    boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#d4edda', edgecolor='#1e8449', lw=2.0,
))
ax5a.text(_x0 + _w_half + _gap + _w_half / 2, _mid_y + _h_mid * 0.86, 'Multi-class',
          ha='center', fontsize=FS_ANNOT + 2, fontweight='bold', color='#1e8449')
ax5a.text(
    _x0 + _w_half + _gap + _w_half / 2, _mid_y + _h_mid * 0.42,
    r'$\hat y=\arg\max_k s_k(x)$'
    '\n\n'
    r'$\mathrm{Err}=\frac{1}{N}\sum_i'
    r'1[\hat y_i\neq y_i]$'
    '\n\n'
    'Intuition: same 0-1 cost;\n'
    'which wrong class you\n'
    'picked does not matter.',
    ha='center', va='center', fontsize=FS_ANNOT,
)

ax5a.add_patch(patches.FancyBboxPatch(
    (_x0, _bot_y), _w_full, _h_bot,
    boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#fce4d6', edgecolor='#e67e22', lw=2.0,
))
ax5a.text(
    _xc6, _bot_y + _h_bot / 2,
    'Intuition: 0-1 is the report card -- hard yes/no.\n'
    'It is flat almost everywhere and jumps at the decision boundary,\n'
    'so gradients vanish. We report Err; we train with soft surrogates\n'
    '(sigmoid/softmax + cross-entropy) that have useful slope.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

# Right: viz of 0-1 vs a soft surrogate in score space
ax5b.set_title(r'viz: hard 0-1 vs soft surrogate', fontsize=FS_AX_TITLE - 1)
s = np.linspace(-6, 6, 600)
# true label y=+1; predict sign(s); 0-1 is 1 when s<0
zero_one = (s < 0).astype(float)
# logistic / softplus-like surrogate for margin: log(1+e^{-s})  (binary CE with y=+1, logit=s)
soft = np.log1p(np.exp(-s))
ax5b.plot(s, zero_one, color='#c0392b', lw=3.0,
          label=r'0-1 hard jump (wrong if $s<0$, here $y=+1$)')
ax5b.plot(s, soft / soft.max(), color='#2874a6', lw=2.6,
          label=r'soft surrogate (scaled): $\log(1+e^{-s})$')
ax5b.axvline(0, color='0.45', ls='--', lw=1.2)
ax5b.set_xlabel(r'score / logit $s$', fontsize=FS_AXIS)
ax5b.set_ylabel('penalty (scaled)', fontsize=FS_AXIS)
ax5b.tick_params(labelsize=FS_TICK)
ax5b.grid(True, alpha=0.3)
ax5b.legend(
    fontsize=FS_LEGEND - 1,
    loc='center right',
    bbox_to_anchor=(0.98, 0.58),
    framealpha=0.95,
    borderaxespad=0.0,
)
ax5b.text(
    0.98, 0.40,
    'Hard loss jumps.\n'
    'Soft loss slopes --\n'
    'that is why GD can move.',
    transform=ax5b.transAxes, ha='right', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig5.tight_layout(rect=[0, 0, 1, 0.94])
# Keep the wider figsize (do not reset to default screen width).
fig5.set_dpi(FIGURE_DPI)
fig5.set_size_inches(_fw6_wide, _fh6, forward=True)
fig5.canvas.draw_idle()
_manager5 = getattr(fig5.canvas, 'manager', None)
if _manager5 is not None:
    try:
        _manager5.window.state('zoomed')
    except Exception:
        try:
            _manager5.window.showMaximized()
        except Exception:
            try:
                _manager5.frame.Maximize(True)
            except Exception:
                pass
add_step_footer(fig5)
plt.show()


# Shared classification toys for Steps 7-8
# -------------------------------------------------------
# Binary: two Gaussian blobs; a deliberately imperfect linear rule
n_bin = 80
Xb0 = RNG.normal(loc=[-1.15, 0.05], scale=[0.55, 0.50], size=(n_bin // 2, 2))
Xb1 = RNG.normal(loc=[1.10, -0.05], scale=[0.55, 0.50], size=(n_bin // 2, 2))
Xb = np.vstack([Xb0, Xb1])
yb = np.array([0] * (n_bin // 2) + [1] * (n_bin // 2))
# Slightly tilted / shifted separator (not the Bayes-optimal one)
w_bin = np.array([1.0, 0.35])
b_bin = -0.15
scores_bin = Xb @ w_bin + b_bin
yhat_bin = (scores_bin > 0).astype(int)
err_bin = float(np.mean(yhat_bin != yb))
# Confusion counts
tp = int(np.sum((yhat_bin == 1) & (yb == 1)))
tn = int(np.sum((yhat_bin == 0) & (yb == 0)))
fp = int(np.sum((yhat_bin == 1) & (yb == 0)))
fn = int(np.sum((yhat_bin == 0) & (yb == 1)))

# Multi-class: overlapping blobs; predict by nearest (slightly wrong) prototype
n_per = 36
means_mc = np.array([[-1.1, -0.6], [1.1, -0.5], [0.0, 1.0]])
Xm_parts = [
    RNG.normal(loc=means_mc[k], scale=0.70, size=(n_per, 2))
    for k in range(3)
]
Xm = np.vstack(Xm_parts)
ym = np.repeat(np.arange(3), n_per)
# Decision rule: nearest prototype, with prototypes shifted so errors appear
proto = means_mc.copy()
proto[0] += np.array([0.45, 0.35])
proto[1] += np.array([-0.25, 0.40])
proto[2] += np.array([0.10, -0.55])
d2 = ((Xm[:, None, :] - proto[None, :, :]) ** 2).sum(axis=2)
yhat_mc = d2.argmin(axis=1)
err_mc = float(np.mean(yhat_mc != ym))
# Confusion matrix 3x3
cm = np.zeros((3, 3), dtype=int)
for t, p in zip(ym, yhat_mc):
    cm[t, p] += 1


# -------------------------------------------------------
# FIGURE 8: Binary classification error
# -------------------------------------------------------
fig6, (ax6a, ax6b) = new_figure(1, 2)
fig6.suptitle(
    'Step 8: Binary classification error -- fraction of wrong hard labels',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

# Left: points + boundary; mark mistakes
correct = yhat_bin == yb
ax6a.scatter(
    Xb[correct & (yb == 0), 0], Xb[correct & (yb == 0), 1],
    s=42, c='#2874a6', alpha=0.85, edgecolors='white', linewidths=0.4,
    label='true 0, correct',
)
ax6a.scatter(
    Xb[correct & (yb == 1), 0], Xb[correct & (yb == 1), 1],
    s=42, c='#1e8449', alpha=0.85, edgecolors='white', linewidths=0.4,
    label='true 1, correct',
)
ax6a.scatter(
    Xb[~correct, 0], Xb[~correct, 1],
    s=70, facecolors='none', edgecolors='#c0392b', linewidths=2.0,
    label='mistake (0-1 = 1)', zorder=4,
)
# Decision boundary: w·x + b = 0
xs_line = np.linspace(-2.8, 2.8, 200)
ys_line = -(w_bin[0] * xs_line + b_bin) / (w_bin[1] + 1e-8)
ax6a.plot(xs_line, ys_line, color='#c0392b', lw=2.4, label='decision boundary')
ax6a.set_title('Binary labels + hard decisions', fontsize=FS_AX_TITLE - 2)
ax6a.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax6a.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax6a.tick_params(labelsize=FS_TICK)
ax6a.set_aspect('equal', adjustable='datalim')
ax6a.grid(True, alpha=0.25)
ax6a.legend(fontsize=FS_LEGEND - 1, loc='upper left')
ax6a.text(
    0.98, 0.02,
    r'Error $= \frac{1}{N}\sum_i 1[\hat y_i \neq y_i]$' + '\n'
    f'= {err_bin:.2f}  on this set',
    transform=ax6a.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

# Right: intuition via confusion counts
ax6b.set_xlim(0, 10)
ax6b.set_ylim(0, 10)
ax6b.axis('off')
ax6b.set_title('Binary error, read as confusion', fontsize=FS_AX_TITLE - 2)

# 2x2 confusion cells
cell = [
    (1.2, 5.2, tn, 'TN', '#d5f5e3'),
    (5.2, 5.2, fp, 'FP', '#f5b7b1'),
    (1.2, 1.5, fn, 'FN', '#f5b7b1'),
    (5.2, 1.5, tp, 'TP', '#d5f5e3'),
]
for x, y, val, name, face in cell:
    ax6b.add_patch(patches.FancyBboxPatch(
        (x, y), 3.2, 2.8,
        boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor=face, edgecolor='0.35', lw=1.8,
    ))
    ax6b.text(x + 1.6, y + 1.7, name, ha='center', va='center',
              fontsize=FS_ANNOT + 2, fontweight='bold')
    ax6b.text(x + 1.6, y + 0.85, str(val), ha='center', va='center',
              fontsize=FS_AX_TITLE - 2, fontweight='bold')

ax6b.text(2.8, 8.5, r'pred $0$', ha='center', fontsize=FS_ANNOT + 1,
          fontweight='bold')
ax6b.text(6.8, 8.5, r'pred $1$', ha='center', fontsize=FS_ANNOT + 1,
          fontweight='bold')
ax6b.text(0.35, 6.6, r'true $0$', ha='center', va='center', rotation=90,
          fontsize=FS_ANNOT + 1, fontweight='bold')
ax6b.text(0.35, 2.9, r'true $1$', ha='center', va='center', rotation=90,
          fontsize=FS_ANNOT + 1, fontweight='bold')

ax6b.text(
    5.0, 0.45,
    f'Error = (FP + FN) / N = ({fp} + {fn}) / {n_bin} = {err_bin:.2f}\n'
    'Intuition: count the red off-diagonal mass -- nothing else.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#eef5ff',
              edgecolor='#2874a6', alpha=0.95),
)

fig6.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 9: Multi-class classification error
# -------------------------------------------------------
fig7, (ax7a, ax7b) = new_figure(1, 2)
fig7.suptitle(
    'Step 9: Multi-class error -- still 0-1; every wrong class costs the same',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

colors_mc = ['#2874a6', '#1e8449', '#8e44ad']
for k in range(3):
    mask_ok = (ym == k) & (yhat_mc == k)
    mask_bad = (ym == k) & (yhat_mc != k)
    ax7a.scatter(
        Xm[mask_ok, 0], Xm[mask_ok, 1],
        s=42, c=colors_mc[k], alpha=0.85, edgecolors='white', linewidths=0.4,
        label=f'class {k} correct',
    )
    ax7a.scatter(
        Xm[mask_bad, 0], Xm[mask_bad, 1],
        s=70, facecolors='none', edgecolors='#c0392b', linewidths=2.0,
        zorder=4,
    )
ax7a.scatter(
    proto[:, 0], proto[:, 1], s=180, c='gold', marker='*',
    edgecolors='k', linewidths=0.7, zorder=5, label='prototypes',
)
# Mark mistakes once in legend
ax7a.scatter([], [], s=70, facecolors='none', edgecolors='#c0392b',
             linewidths=2.0, label='mistake')
ax7a.set_title('K = 3 labels; predict nearest prototype', fontsize=FS_AX_TITLE - 2)
ax7a.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax7a.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax7a.tick_params(labelsize=FS_TICK)
ax7a.set_aspect('equal', adjustable='datalim')
ax7a.grid(True, alpha=0.25)
ax7a.legend(fontsize=FS_LEGEND - 1, loc='upper left')
ax7a.text(
    0.98, 0.02,
    r'Error $= \frac{1}{N}\sum_i 1[\hat y_i \neq y_i]$' + '\n'
    f'= {err_mc:.2f}  (same formula as binary)',
    transform=ax7a.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

# Confusion matrix heatmap
ax7b.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max())
for i in range(3):
    for j in range(3):
        ax7b.text(j, i, str(cm[i, j]), ha='center', va='center',
                  fontsize=FS_AX_TITLE - 2,
                  color='white' if cm[i, j] > cm.max() * 0.55 else 'black',
                  fontweight='bold')
ax7b.set_xticks([0, 1, 2])
ax7b.set_yticks([0, 1, 2])
ax7b.set_xticklabels(['pred 0', 'pred 1', 'pred 2'], fontsize=FS_TICK)
ax7b.set_yticklabels(['true 0', 'true 1', 'true 2'], fontsize=FS_TICK)
ax7b.set_title('Confusion matrix (rows = truth)', fontsize=FS_AX_TITLE - 2)
ax7b.text(
    0.5, -0.18,
    'Off-diagonal = errors. Diagonal = correct.\n'
    'Intuition: with K classes, there are more ways to be wrong --\n'
    'but each wrong prediction still costs exactly 1 under 0-1.',
    transform=ax7b.transAxes, ha='center', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', edgecolor='#2874a6', alpha=0.95),
)

fig7.tight_layout(rect=[0, 0.08, 1, 0.94])
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 10: Sigmoid + binary cross-entropy + Shannon entropy
# -------------------------------------------------------
fig9, axes9 = new_figure(1, 2, gridspec_kw={'width_ratios': [1.2, 1.0]})
fig9.suptitle(
    'Step 10: Sigmoid and binary cross-entropy -- soft training for two classes',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)

ax9a, ax9b = axes9
# Widen left-panel boxes by +2 inches (data units via axes width estimate).
_left_frac9 = 1.2 / (1.2 + 1.0)
_axw9_in = max(fig9.get_figwidth() * _left_frac9 * 0.90, 1e-6)
_dx9 = 2.0 / _axw9_in * 10.0
_x9, _w9 = 0.25 - _dx9 / 2.0, 9.5 + _dx9
_xc9 = _x9 + _w9 / 2.0
ax9a.set_xlim(_x9 - 0.08, _x9 + _w9 + 0.08)
ax9a.set_ylim(0, 10)
ax9a.axis('off')
ax9a.set_title('Formulas + intuition', fontsize=FS_AX_TITLE - 1)

ax9a.add_patch(patches.FancyBboxPatch(
    (_x9, 7.15), _w9, 2.55,
    boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#1a5276', edgecolor='#1a5276', lw=0,
))
ax9a.text(
    _xc9, 8.85,
    r'Sigmoid:   $p=\sigma(s)=\frac{1}{1+e^{-s}}$'
    '   (score $s$ -> probability)',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    fontweight='bold', color='white',
)
ax9a.text(
    _xc9, 7.85,
    r'Binary CE:   $\ell_{\mathrm{BCE}}=-(y\log p+(1-y)\log(1-p))$'
    '\n'
    r'with $y\in\{0,1\}$ and $p=\sigma(s)$',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    fontweight='bold', color='white',
)

ax9a.add_patch(patches.FancyBboxPatch(
    (_x9, 3.85), _w9, 3.0,
    boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#eafaf1', edgecolor='#1e8449', lw=2.0,
))
ax9a.text(_xc9, 6.45, 'Entropy in communications (Shannon)',
          ha='center', fontsize=FS_ANNOT + 2, fontweight='bold', color='#1e8449')
ax9a.text(
    _xc9, 5.05,
    r'Source entropy:  $H(p)=-\sum_k p_k\log p_k$'
    '\n'
    'Average surprise of a random source -- how many bits you need\n'
    'on average if you encode optimally for that distribution.\n\n'
    r'Cross-entropy:  $H(p,q)=-\sum_k p_k\log q_k$'
    '\n'
    'Expected bits if the true distribution is $p$, but you encode with $q$.\n'
    r'Always $H(p,q)\geq H(p)$; the gap is $\mathrm{KL}(p,q)$.',
    ha='center', va='center', fontsize=FS_ANNOT,
)

ax9a.add_patch(patches.FancyBboxPatch(
    (_x9, 0.3), _w9, 3.25,
    boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#fdebd0', edgecolor='#d68910', lw=2.0,
))
ax9a.text(_xc9, 3.05, 'Intuition for BCE',
          ha='center', fontsize=FS_ANNOT + 2, fontweight='bold', color='#9a7d0a')
ax9a.text(
    _xc9, 1.55,
    'Think of the label as a one-bit message: y=1 or y=0.\n'
    'The model proposes a code via probability p = sigma(s).\n\n'
    'If y=1, BCE = -log p: punish a small p (model said "probably 0").\n'
    'If y=0, BCE = -log(1-p): punish a large p (model said "probably 1").\n\n'
    'Unlike 0-1 error, BCE has a smooth slope -- GD can move the score s.',
    ha='center', va='center', fontsize=FS_ANNOT,
)

# Right viz: sigmoid + BCE curves
s9 = np.linspace(-6, 6, 500)
p9 = 1.0 / (1.0 + np.exp(-s9))
# avoid log(0)
p_clip = np.clip(p9, 1e-6, 1 - 1e-6)
bce1 = -np.log(p_clip)          # y=1
bce0 = -np.log(1.0 - p_clip)    # y=0

ax9b.plot(s9, p9, color='#2874a6', lw=2.8, label=r'sigmoid $p=\sigma(s)$')
ax9b.plot(s9, bce1, color='#c0392b', lw=2.4, label=r'BCE if $y=1$: $-\log p$')
ax9b.plot(s9, bce0, color='#1e8449', lw=2.4, label=r'BCE if $y=0$: $-\log(1-p)$')
ax9b.axvline(0, color='0.5', ls='--', lw=1.0)
ax9b.set_title('viz: sigmoid and BCE vs score', fontsize=FS_AX_TITLE - 2)
ax9b.set_xlabel(r'score / logit $s$', fontsize=FS_AXIS)
ax9b.set_ylabel('value', fontsize=FS_AXIS)
ax9b.set_ylim(-0.05, 4.2)
ax9b.tick_params(labelsize=FS_TICK)
ax9b.grid(True, alpha=0.3)
ax9b.legend(fontsize=FS_LEGEND - 1, loc='upper right', framealpha=0.95)
ax9b.text(
    0.02, 0.98,
    'When the label is 1,\n'
    'large positive s -> small BCE.\n'
    'When the label is 0,\n'
    'large negative s -> small BCE.',
    transform=ax9b.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig9.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig9)


# -------------------------------------------------------
# FIGURE 11: Multi-class cross-entropy
# -------------------------------------------------------
fig10, axes10 = new_figure(1, 2, gridspec_kw={'width_ratios': [1.15, 1.0]})
fig10.suptitle(
    'Step 11: Multi-class cross-entropy -- soft training for K classes',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax10a, ax10b = axes10
# Widen left-panel boxes by +1 inch (data units via axes width estimate).
_left_frac10 = 1.15 / (1.15 + 1.0)
_axw10_in = max(fig10.get_figwidth() * _left_frac10 * 0.90, 1e-6)
_dx10 = 1.0 / _axw10_in * 10.0
_x10, _w10 = 0.25 - _dx10 / 2.0, 9.5 + _dx10
_xc10 = _x10 + _w10 / 2.0
ax10a.set_xlim(_x10 - 0.08, _x10 + _w10 + 0.08)
ax10a.set_ylim(0, 10)
ax10a.axis('off')
ax10a.set_title('Formulas + intuition', fontsize=FS_AX_TITLE - 1)

ax10a.add_patch(patches.FancyBboxPatch(
    (_x10, 6.9), _w10, 2.8,
    boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#1a5276', edgecolor='#1a5276', lw=0,
))
ax10a.text(
    _xc10, 8.85,
    r'Softmax:   $p_k=\frac{e^{s_k}}{\sum_j e^{s_j}}$',
    ha='center', va='center', fontsize=FS_ANNOT + 2,
    fontweight='bold', color='white',
)
ax10a.text(
    _xc10, 7.7,
    r'Multi-class CE:   $\ell_{\mathrm{CE}}=-\sum_k y_k\log p_k$'
    '\n'
    r'For a hard label $y_\star$:   $\ell_{\mathrm{CE}}=-\log p_{y_\star}$',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    fontweight='bold', color='white',
)

ax10a.add_patch(patches.FancyBboxPatch(
    (_x10, 3.55), _w10, 3.0,
    boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#d6eaf8', edgecolor='#2874a6', lw=2.0,
))
ax10a.text(_xc10, 6.1, 'Link to Shannon cross-entropy',
           ha='center', fontsize=FS_ANNOT + 2, fontweight='bold', color='#2874a6')
ax10a.text(
    _xc10, 4.7,
    'The one-hot label is a distribution p concentrated on the true class.\n'
    'The softmax output is the model distribution q.\n\n'
    r'CE loss is exactly $H(p,q)=-\sum_k p_k\log q_k$.'
    '\n'
    'Minimizing CE pushes q toward p -- put mass on the true class.',
    ha='center', va='center', fontsize=FS_ANNOT,
)

ax10a.add_patch(patches.FancyBboxPatch(
    (_x10, 0.3), _w10, 2.95,
    boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#fdebd0', edgecolor='#d68910', lw=2.0,
))
ax10a.text(_xc10, 2.75, 'Intuition',
           ha='center', fontsize=FS_ANNOT + 2, fontweight='bold', color='#9a7d0a')
ax10a.text(
    _xc10, 1.4,
    '0-1 only asks: was argmax correct?\n'
    'CE asks: how much probability did you put on the true class?\n\n'
    'Even a correct argmax with p_true=0.34 still pays -log(0.34).\n'
    'A confident correct answer with p_true=0.95 pays much less.\n'
    'That graded pressure is what makes softmax+CE trainable.',
    ha='center', va='center', fontsize=FS_ANNOT,
)

# Right viz: toy logits -> softmax -> CE
logits = np.array([1.2, 0.3, -0.8])
exp_l = np.exp(logits - logits.max())
probs = exp_l / exp_l.sum()
true_k = 0
ce_val = -np.log(probs[true_k])
ks = np.arange(3)

ax10b.bar(ks - 0.18, logits, width=0.36, color='#85929e', label='logits $s_k$')
ax10b.bar(ks + 0.18, probs, width=0.36, color='#2874a6', label='softmax $p_k$')
ax10b.axhline(0, color='0.5', lw=0.8)
ax10b.set_xticks(ks)
ax10b.set_xticklabels([r'$k=0$ (true)', r'$k=1$', r'$k=2$'], fontsize=FS_TICK)
ax10b.set_title('viz: logits -> softmax; CE uses $p_{true}$', fontsize=FS_AX_TITLE - 2)
ax10b.set_ylabel('value', fontsize=FS_AXIS)
ax10b.tick_params(labelsize=FS_TICK)
ax10b.grid(True, axis='y', alpha=0.3)
ax10b.legend(fontsize=FS_LEGEND - 1, loc='upper right', framealpha=0.95)
ax10b.text(
    0.02, 0.98,
    rf'true class $k=0$'
    '\n'
    rf'$p_0={probs[true_k]:.2f}$'
    '\n'
    rf'$\ell_{{\mathrm{{CE}}}}=-\log p_0={ce_val:.2f}$'
    '\n\n'
    'Raise $s_0$ (or lower others)\n'
    '-> raise $p_0$ -> lower CE.',
    transform=ax10b.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig10.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig10)


# -------------------------------------------------------
# FIGURE 12: Intuition bridge -- report card vs training signal
# -------------------------------------------------------
fig8, ax8 = new_figure()
fig8.suptitle(
    'Step 12: Intuition -- error is discrete; training needs a soft slope',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax8.set_xlim(0, 12)
ax8.set_ylim(0, 8)
ax8.axis('off')

ax8.add_patch(patches.FancyBboxPatch(
    (0.4, 4.6), 5.4, 2.9,
    boxstyle='round,pad=0.02,rounding_size=0.12',
    facecolor='#fce4d6', edgecolor='#e67e22', lw=2.2,
))
ax8.text(3.1, 7.0, 'Binary intuition', ha='center',
         fontsize=FS_ANNOT + 2, fontweight='bold', color='#e67e22')
ax8.text(
    3.1, 5.7,
    'One threshold (or sign of a score)\n'
    'splits the plane into two decisions.\n'
    'Error counts points on the wrong side.\n\n'
    'Training usually softens the hard step\n'
    'with a sigmoid + cross-entropy.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

ax8.add_patch(patches.FancyBboxPatch(
    (6.2, 4.6), 5.4, 2.9,
    boxstyle='round,pad=0.02,rounding_size=0.12',
    facecolor='#d6eaf8', edgecolor='#2874a6', lw=2.2,
))
ax8.text(8.9, 7.0, 'Multi-class intuition', ha='center',
         fontsize=FS_ANNOT + 2, fontweight='bold', color='#2874a6')
ax8.text(
    8.9, 5.7,
    'K scores compete; the predicted label\n'
    'is usually argmax (or nearest prototype).\n'
    'Error ignores which wrong class you chose.\n\n'
    'Training usually softens argmax\n'
    'with softmax + cross-entropy.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

ax8.add_patch(patches.FancyBboxPatch(
    (0.4, 0.5), 11.2, 3.6,
    boxstyle='round,pad=0.02,rounding_size=0.12',
    facecolor='#f4f6f7', edgecolor='0.45', lw=1.8,
))
ax8.text(
    6.0, 2.3,
    'Shared moral with MSE / MAE / Huber:\n'
    'the quantity you report and the quantity you optimize need not coincide.\n\n'
    'Classification error is the human-facing metric (how often are we wrong?).\n'
    'Soft classification losses are the machine-facing signal (how should parameters move?).\n\n'
    f'This demo: binary error = {err_bin:.2f},   multi-class error = {err_mc:.2f}.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig8)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("DIFFERENT LOSSES => DIFFERENT CONVERGENCE")
print("=" * 60)
print(f"""
Setup: fit y ~ w x + b on n={N} points with {len(outlier_idx)} outliers.
Shared init theta0 = (w,b) = ({THETA0[0]:g}, {THETA0[1]:g}), plain GD.

Final parameters after {SGD_STEPS} SGD steps:
""")
for name in LOSSES:
    th = results[name]['theta']
    print(f"  {name:6s}  w={th[0]:+.3f}  b={th[1]:+.3f}  "
          f"(lr={LR[name]:g}, final own-loss={results[name]['losses'][-1]:.3f})")

print(f"""
  True     w={W_TRUE:+.3f}  b={B_TRUE:+.3f}

Formulas (Step 2):
  MSE:   ell(r)=r^2/2 , grad=r
  MAE:   ell(r)=|r|   , grad=sign(r)
  Huber: quadratic near 0, linear in tails; grad=clip(r, +/- delta)

Why convergence differs:
  1) Shape of ell(r)
       MSE   ~ r^2     -> outliers dominate the objective
       MAE   ~ |r|     -> robust; optimum near median-like fit
       Huber ~ hybrid  -> MSE locally, MAE in the tails

  2) Gradient signal used by GD
       MSE   grad ~ r          (can be huge early / near outliers)
       MAE   grad ~ sign(r)    (bounded, less adaptive near 0)
       Huber grad ~ clip(r)    (stable compromise)

  3) Therefore the SAME learning-rate GD takes different routes
     and stops at different (w, b).

Classification error (Steps 7-9) and soft training losses (Steps 10-11):
  0-1 loss: 1[y_hat != y].  Error rate = mean 0-1 loss.
  Binary demo / multi-class demo: hard mistakes only (report card).

  Entropy in communications (Shannon):
    H(p) = -sum p log p          -- average surprise / coding length of a source
    H(p,q) = -sum p log q        -- bits if true is p but you encode with q

  Binary CE (Step 10):
    p = sigmoid(s)
    BCE = -( y log p + (1-y) log(1-p) )
    Soft slope for GD; graded punishment of wrong probabilities.

  Multi-class CE (Step 11):
    p = softmax(s)
    CE = -log p_true   (for a hard label)
    Same Shannon cross-entropy between one-hot label and model probs.

  Intuition: report card is hard 0-1; training signal is soft CE.

Takeaway:
  Choosing a loss is choosing both the destination (statistical bias)
  and the travel experience (gradient scale / convergence).
  Choosing a metric (0-1 error) is choosing what you report to humans --
  often different from what you optimize.
  See 120_maximum_likelihood.py for the MLE story behind MSE vs MAE;
  see 320_softmax.py for multi-class soft scores.
""")
