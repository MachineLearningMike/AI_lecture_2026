"""
100_loss_functions.py

Different loss functions produce different optima AND different
convergence behavior under the same optimizer.

Demonstrated on a 1D linear fit  y ~ w x + b  with a few outliers:

    MSE   (squared error)   -- Gaussian MLE; sensitive to outliers; smooth bowl
    MAE   (absolute error)  -- Laplace MLE; robust; sharper valley / constant-scale grads
    Huber                   -- MSE near 0, MAE in the tails; a practical compromise

Companion to 060_maximum_likelihood.py (MSE vs MAE as MLE) and
020_model.py (training with MSE).
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
)
apply_rcparams(plt)
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
    plt.show()


def add_vcenter_legend_and_note(ax, note, facecolor='white'):
    ax.legend(
        loc='lower right',
        bbox_to_anchor=(0.98, 0.54),
        fontsize=FS_LEGEND,
        framealpha=0.95,
        borderaxespad=0.0,
    )
    ax.text(
        0.98, 0.46, note,
        transform=ax.transAxes,
        fontsize=FS_ANNOT,
        va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.95),
    )


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
STEPS = 200
# Per-loss learning rates so each can move meaningfully
# (MAE grads are ~O(1), MSE grads scale with residual size)
LR = {
    'MSE': 0.05,
    'MAE': 0.20,
    'Huber': 0.08,
}

results = {
    name: fit_gd(name, THETA0, LR[name], STEPS)
    for name in LOSSES
}


# -------------------------------------------------------
# FIGURE 1: Shapes of the losses / grads
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    'Step 1: Loss shape decides the gradient signal',
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
add_vcenter_legend_and_note(
    ax,
    'MSE grows as r^2\n'
    '-> huge outliers dominate.\n\n'
    'MAE grows as |r|\n'
    '-> linear, more robust.\n\n'
    'Huber blends both.',
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
add_vcenter_legend_and_note(
    ax,
    'MSE: grad = r\n'
    '  (can explode)\n\n'
    'MAE: grad = sign(r)\n'
    '  (constant size)\n\n'
    'Huber: caps the grad\n'
    f'  at ±{HUBER_DELTA:g}',
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Fitted lines -- different losses, different answers
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    'Step 2: Same data / init / optimizer family -- different fits',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax2.scatter(x, y, s=35, color='0.55', alpha=0.85, label='data', zorder=2)
ax2.scatter(x[outlier_idx], y[outlier_idx], s=70, color='red',
            edgecolor='k', linewidth=0.6, label='outliers', zorder=3)

x_line = np.linspace(-2.2, 2.2, 200)
ax2.plot(x_line, W_TRUE * x_line + B_TRUE, 'k--', linewidth=2.0,
         label='true line', zorder=1)

# Ghost trajectories: intermediate predictions along GD (diversity of paths)
traj_steps = np.linspace(0, STEPS, 8, dtype=int)
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
add_vcenter_legend_and_note(
    ax2,
    'Faint lines = path during GD.\n'
    'Bold = final fit.\n\n'
    'One-sided outliers pull\n'
    'MSE up hard; MAE resists;\n'
    'Huber sits in between.\n\n'
    f'True: w={W_TRUE:g}, b={B_TRUE:g}',
)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Convergence curves differ
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    'Step 3: Convergence speed and stability depend on the loss',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes3[0]
for name, cfg in LOSSES.items():
    # Plot each curve in its OWN loss units (normalized to start at 1)
    L = results[name]['losses']
    ax.plot(L / L[0], color=cfg['color'], linewidth=2.5,
            label=rf'{name} (lr={LR[name]:g})')
ax.set_xlabel('gradient step', fontsize=FS_AXIS)
ax.set_ylabel(r'loss / loss$_0$  (own units)', fontsize=FS_AXIS)
ax.set_title('Normalized training loss', fontsize=FS_AX_TITLE)
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
ax.set_xlabel('gradient step', fontsize=FS_AXIS)
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
# FIGURE 4: Parameter-space paths (geometry of each loss)
# -------------------------------------------------------
fig4, ax4 = new_figure()
fig4.suptitle(
    'Step 4: Paths in $(w,b)$ space -- each loss has its own landscape',
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
add_vcenter_legend_and_note(
    ax4,
    'Same start, same GD.\n'
    'Different losses bend\n'
    'the path toward\n'
    'different endpoints.\n\n'
    'Loss choice = inductive\n'
    'bias + convergence style.',
)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("DIFFERENT LOSSES => DIFFERENT CONVERGENCE")
print("=" * 60)
print(f"""
Setup: fit y ~ w x + b on n={N} points with {len(outlier_idx)} outliers.
Shared init theta0 = (w,b) = ({THETA0[0]:g}, {THETA0[1]:g}), plain GD.

Final parameters after {STEPS} steps:
""")
for name in LOSSES:
    th = results[name]['theta']
    print(f"  {name:6s}  w={th[0]:+.3f}  b={th[1]:+.3f}  "
          f"(lr={LR[name]:g}, final own-loss={results[name]['losses'][-1]:.3f})")

print(f"""
  True     w={W_TRUE:+.3f}  b={B_TRUE:+.3f}

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

Takeaway:
  Choosing a loss is choosing both the destination (statistical bias)
  and the travel experience (gradient scale / convergence).
  See 060_maximum_likelihood.py for the MLE story behind MSE vs MAE.
""")
