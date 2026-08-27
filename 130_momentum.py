"""
130_momentum.py

Role of momentum in SGD optimizers -- especially under minibatch noise.

Vanilla minibatch SGD:
    g_t = gradient on a random minibatch   (noisy!)
    x <- x - lr * g_t

SGD with momentum:
    v <- beta * v + g_t
    x <- x - lr * v

Minibatches make g_t jitter every SGD step. Momentum keeps a running
average of those noisy pushes, so the "ball" can keep rolling in a
consistent direction instead of shaking in place.

Shows comparison animation (GIF export skipped if the file already exists):
    momentum_sgd.gif
"""

from pathlib import Path

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
import matplotlib.patches as patches
from matplotlib import animation
import numpy as np

np.random.seed(42)
RNG = np.random.default_rng(42)


ANIM_DPI = 120
FPS = 20

OUT_GIF = Path(__file__).resolve().parent / 'momentum_sgd.gif'


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
# Finite-sum ravine loss (each "example" has its own curvature)
#   L_i(x,y) = a_i x^2 + b_i y^2
#   L      = mean_i L_i
# Minibatch gradient uses a random subset -> noisy direction each SGD step.
# -------------------------------------------------------
A_MEAN, B_MEAN = 5.0, 0.2
N_DATA = 128
BATCH_SIZE = 8

a_i = np.clip(RNG.normal(A_MEAN, 2.5, size=N_DATA), 0.8, None)
b_i = np.clip(RNG.normal(B_MEAN, 0.25, size=N_DATA), 0.05, None)


def loss_full(xy):
    x, y = float(xy[0]), float(xy[1])
    return float(np.mean(a_i) * x ** 2 + np.mean(b_i) * y ** 2)


def loss_path(path):
    return np.array([loss_full(p) for p in path])


def full_grad(xy):
    x, y = float(xy[0]), float(xy[1])
    return np.array([2.0 * np.mean(a_i) * x, 2.0 * np.mean(b_i) * y], dtype=float)


def minibatch_grad(xy, idx):
    x, y = float(xy[0]), float(xy[1])
    return np.array([2.0 * np.mean(a_i[idx]) * x, 2.0 * np.mean(b_i[idx]) * y], dtype=float)


def make_batch_schedule(steps, batch_size=BATCH_SIZE, seed=0):
    """Shared minibatch indices so SGD and momentum see the same noise."""
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.choice(N_DATA, size=batch_size, replace=False) for _ in range(steps)],
        axis=0,
    )


def run_sgd(x0, lr, batches):
    path = [np.array(x0, dtype=float)]
    grads = []
    x = path[0].copy()
    for idx in batches:
        g = minibatch_grad(x, idx)
        grads.append(g.copy())
        x = x - lr * g
        path.append(x.copy())
    return np.stack(path), np.stack(grads)


def run_momentum(x0, lr, beta, batches):
    path = [np.array(x0, dtype=float)]
    vel = [np.zeros(2, dtype=float)]
    grads = []
    x = path[0].copy()
    v = vel[0].copy()
    for idx in batches:
        g = minibatch_grad(x, idx)
        grads.append(g.copy())
        v = beta * v + g
        x = x - lr * v
        path.append(x.copy())
        vel.append(v.copy())
    return np.stack(path), np.stack(vel), np.stack(grads)


X0 = np.array([1.0, 1.0])
SGD_STEPS = 80
LR = 0.05
BETA = 0.9

batches = make_batch_schedule(SGD_STEPS, BATCH_SIZE, seed=7)
path_sgd, grads_sgd = run_sgd(X0, LR, batches)
path_mom, vel_mom, grads_mom = run_momentum(X0, LR, BETA, batches)


# -------------------------------------------------------
# FIGURE 1: Contour + stochastic trajectories
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    'Step 1: Minibatch noise makes SGD jitter; momentum keeps rolling',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

gx = np.linspace(-1.3, 1.3, 200)
gy = np.linspace(-1.3, 1.3, 200)
GX, GY = np.meshgrid(gx, gy)
GZ = np.mean(a_i) * GX ** 2 + np.mean(b_i) * GY ** 2
cs = ax1.contour(GX, GY, GZ, levels=20, cmap='terrain', linewidths=1.0)
ax1.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

ax1.plot(path_sgd[:, 0], path_sgd[:, 1], '-', color='crimson',
         linewidth=1.6, alpha=0.9, label='Minibatch SGD')
ax1.plot(path_mom[:, 0], path_mom[:, 1], '-', color='royalblue',
         linewidth=1.8, alpha=0.9, label=rf'Momentum ($\beta={BETA}$)')
ax1.scatter(path_sgd[::4, 0], path_sgd[::4, 1], c='crimson', s=18, zorder=4)
ax1.scatter(path_mom[::4, 0], path_mom[::4, 1], c='royalblue', s=18, zorder=4)
ax1.scatter([0], [0], c='k', s=80, marker='*', zorder=5, label='minimum')
ax1.scatter([X0[0]], [X0[1]], c='orange', s=60, zorder=5, label='start')

ax1.set_xlabel('x (steep / noisy direction)', fontsize=FS_AXIS)
ax1.set_ylabel('y (shallow direction)', fontsize=FS_AXIS)
ax1.set_aspect('equal')
ax1.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax1,
    f'Each step: batch of {BATCH_SIZE}/{N_DATA}\n'
    'examples -> noisy gradient.\n\n'
    'SGD: ball shakes with noise.\n'
    'Momentum: velocity remembers\n'
    'the consistent downhill push.',
)
ax1.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Loss curves + noisy g vs smoothed v
# -------------------------------------------------------
fig2, axes2 = new_figure(1, 2)
fig2.suptitle(
    'Step 2: Same minibatches -- momentum filters the jitter',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes2[0]
ax.plot(loss_path(path_sgd), color='crimson', linewidth=2.5, label='SGD loss')
ax.plot(loss_path(path_mom), color='royalblue', linewidth=2.5, label='Momentum loss')
ax.set_xlabel('SGD step', fontsize=FS_AXIS)
ax.set_ylabel(r'$L(x,y)$ (full batch)', fontsize=FS_AXIS)
ax.set_title('Full-batch loss along the path', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    'Both see identical\n'
    'minibatch schedules.\n'
    'Momentum still descends\n'
    'more steadily.',
)
ax.grid(True, alpha=0.3)

ax = axes2[1]
ax.plot(grads_sgd[:, 0], color='crimson', alpha=0.45, linewidth=1.2,
        label=r'SGD noisy $g_x$')
ax.plot(vel_mom[1:, 0], color='royalblue', linewidth=2.2,
        label=r'Momentum $v_x$')
ax.plot(vel_mom[1:, 1], color='seagreen', linewidth=2.2,
        label=r'Momentum $v_y$')
ax.axhline(0, color='k', linewidth=0.8, alpha=0.4)
ax.set_xlabel('SGD step', fontsize=FS_AXIS)
ax.set_ylabel('component', fontsize=FS_AXIS)
ax.set_title('Noisy gradient vs momentum velocity', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    r'$v \leftarrow \beta v + g_{batch}$' + '\n\n'
    'g_x chatters every SGD step.\n'
    'v_x is a low-pass filter\n'
    'of that chatter.',
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Update rules
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    'Step 3: Minibatch SGD vs momentum gradient update',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes3[0]
ax.axis('off')
ax.set_title('Minibatch SGD', fontsize=FS_AX_TITLE, color='crimson')
ax.text(
    0.5, 0.55,
    r'$g_t = \nabla L_{\mathcal{B}_t}(x_t)$' + '\n'
    r'(random minibatch $\mathcal{B}_t$)' + '\n\n'
    r'$x_{t+1} = x_t - \eta\, g_t$' + '\n\n'
    'Noisy push every SGD step.\n'
    'The ball oscillates.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_AXIS,
    bbox=dict(boxstyle='round', facecolor='#ffe8e8', alpha=0.95),
)

ax = axes3[1]
ax.axis('off')
ax.set_title(rf'Momentum  ($\beta={BETA}$)', fontsize=FS_AX_TITLE, color='royalblue')
ax.text(
    0.5, 0.55,
    r'$g_t = \nabla L_{\mathcal{B}_t}(x_t)$' + '\n\n'
    r'$v_{t+1} = \beta\, v_t + g_t$' + '\n\n'
    r'$x_{t+1} = x_t - \eta\, v_{t+1}$' + '\n\n'
    'Velocity keeps inertia\n'
    'through the batch noise.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_AXIS,
    bbox=dict(boxstyle='round', facecolor='#e8efff', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# ANIMATION: two balls, stochastic pushes, momentum inertia
# -------------------------------------------------------
print(f'\nRendering animation to {OUT_GIF} ...')

fig_a, ax_a = plt.subplots(figsize=(9, 7), dpi=ANIM_DPI)
fig_a.suptitle(
    'Stochastic minibatches shake the SGD ball;\n'
    'momentum keeps the other ball rolling',
    fontsize=FS_AX_TITLE, fontweight='bold', y=0.98,
)

ax_a.contour(GX, GY, GZ, levels=20, cmap='terrain', linewidths=1.0)
ax_a.scatter([0], [0], c='k', s=90, marker='*', zorder=5, label='minimum')
ax_a.scatter([X0[0]], [X0[1]], c='orange', s=55, zorder=5, label='start')

line_s, = ax_a.plot([], [], '-', color='crimson', linewidth=1.8, alpha=0.85, label='SGD ball')
line_m, = ax_a.plot([], [], '-', color='royalblue', linewidth=2.0, alpha=0.9, label='Momentum ball')

BALL_R = 0.05
pt_s = patches.Circle(X0, BALL_R, facecolor='crimson', edgecolor='darkred',
                      linewidth=1.5, zorder=6)
pt_m = patches.Circle(X0, BALL_R, facecolor='royalblue', edgecolor='navy',
                      linewidth=1.5, zorder=6)
ax_a.add_patch(pt_s)
ax_a.add_patch(pt_m)

# Noisy minibatch force on SGD ball; velocity on momentum ball
arr_g = ax_a.annotate(
    '', xy=X0, xytext=X0,
    arrowprops=dict(arrowstyle='->', color='crimson', lw=2.0, alpha=0.9),
    zorder=7,
)
arr_v = ax_a.annotate(
    '', xy=X0, xytext=X0,
    arrowprops=dict(arrowstyle='->', color='navy', lw=2.2),
    zorder=7,
)

hud = ax_a.text(
    0.02, 0.98, '', transform=ax_a.transAxes, va='top', ha='left',
    fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
)
ax_a.text(
    0.98, 0.02,
    'red arrow: minibatch gradient (noisy)\n'
    'blue arrow: momentum velocity (smoothed)',
    transform=ax_a.transAxes, va='bottom', ha='right', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax_a.set_xlim(-1.3, 1.3)
ax_a.set_ylim(-1.3, 1.3)
ax_a.set_aspect('equal')
ax_a.set_xlabel('x', fontsize=FS_AXIS)
ax_a.set_ylabel('y', fontsize=FS_AXIS)
ax_a.legend(loc='upper right', fontsize=FS_LEGEND)
ax_a.grid(True, alpha=0.3)
fig_a.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.86)

n_anim = len(path_sgd)


def init_anim():
    line_s.set_data([], [])
    line_m.set_data([], [])
    return line_s, line_m, pt_s, pt_m, hud, arr_g, arr_v


def update_anim(i):
    line_s.set_data(path_sgd[: i + 1, 0], path_sgd[: i + 1, 1])
    line_m.set_data(path_mom[: i + 1, 0], path_mom[: i + 1, 1])
    pt_s.center = (path_sgd[i, 0], path_sgd[i, 1])
    pt_m.center = (path_mom[i, 0], path_mom[i, 1])

    # arrows use the gradient/velocity that produced the step into this state
    j = min(i, SGD_STEPS - 1)
    g = grads_sgd[j]
    v = vel_mom[j + 1] if i > 0 else vel_mom[0]

    g_scale = 0.04
    v_scale = 0.04
    arr_g.set_position((path_sgd[i, 0], path_sgd[i, 1]))
    arr_g.xy = (path_sgd[i, 0] + g_scale * g[0], path_sgd[i, 1] + g_scale * g[1])

    arr_v.set_position((path_mom[i, 0], path_mom[i, 1]))
    arr_v.xy = (path_mom[i, 0] + v_scale * v[0], path_mom[i, 1] + v_scale * v[1])

    hud.set_text(
        f'SGD step {i}/{n_anim - 1}   batch={BATCH_SIZE}/{N_DATA}\n'
        f'L_SGD = {loss_full(path_sgd[i]):.3f}\n'
        f'L_Mom = {loss_full(path_mom[i]):.3f}\n'
        f'|g_batch| = {np.linalg.norm(g):.2f}\n'
        f'|v_mom|   = {np.linalg.norm(v):.2f}'
    )
    return line_s, line_m, pt_s, pt_m, hud, arr_g, arr_v


anim = animation.FuncAnimation(
    fig_a, update_anim, init_func=init_anim,
    frames=n_anim, interval=1000 / FPS, blit=False,
)
if OUT_GIF.is_file() and OUT_GIF.stat().st_size > 0:
    print(f'Skipping GIF export (already exists): {OUT_GIF}')
else:
    try:
        anim.save(OUT_GIF, writer=animation.PillowWriter(fps=FPS))
        print(f'Saved: {OUT_GIF}')
    except Exception as exc:
        print(f'Skipping GIF export ({OUT_GIF.name}): {exc}')
        print('Install pillow to enable animation export: pip install pillow')
try:
    if plt.get_backend().lower() != 'agg':
        plt.show()
    else:
        plt.close(fig_a)
except Exception:
    plt.close(fig_a)



# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("MOMENTUM UNDER STOCHASTIC MINIBATCHES")
print("=" * 60)
print(f"""
Finite-sum ravine loss:
  L_i = a_i x^2 + b_i y^2 ,   L = mean_i L_i
  mean(a_i)~{np.mean(a_i):.2f} (steep), mean(b_i)~{np.mean(b_i):.2f} (shallow)

Each gradient update uses a random minibatch of {BATCH_SIZE}/{N_DATA} examples, so
the gradient jitters from SGD step to SGD step (stochastic oscillation).

Minibatch SGD:
  x <- x - lr * g_batch          # ball shakes with every batch

Momentum:
  v <- beta * v + g_batch
  x <- x - lr * v                # velocity keeps inertia through the noise
  (beta={BETA}, lr={LR})

Both optimizers in this demo share the SAME minibatch schedule.

Final full-batch losses after {SGD_STEPS} SGD steps:
  SGD      L = {loss_full(path_sgd[-1]):.6f}
  Momentum L = {loss_full(path_mom[-1]):.6f}
""")
