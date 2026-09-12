"""
Stochastic ascent: each step's direction is a NOISY version of the true
gradient. Even so, accumulated steps still reach the highest point with
high probability.

Intuition:
    true step  ~  grad f
    noisy step ~  grad f + noise
    E[noisy step] still points uphill, so the random walk has drift
    toward the summit. Variance averages out over many steps.

This is the geometric cousin of SGD (noisy gradients, reliable progress).

Also writes climbing animations (GIF export skipped if the file already exists; animation is still shown):
    stochastic_ascend_race.gif   -- true vs one noisy climber
    stochastic_ascend_swarm.gif   -- several noisy climbers at once
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
    add_step_footer,
    maybe_save_animation,
    run_slide_deck,
    should_export_gif,
)
apply_rcparams(plt)
import matplotlib.patches as patches
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

np.random.seed(42)
RNG = np.random.default_rng(42)


ANIM_DPI = 110
ANIM_FPS = 8          # slower playback
ANIM_HOLD_SEC = 2.5   # pause at the summit so convergence is obvious

OUT_RACE = Path(__file__).resolve().parent / 'stochastic_ascend_race.gif'
OUT_SWARM = Path(__file__).resolve().parent / 'stochastic_ascend_swarm.gif'


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
    print('Tip: drag 3D plots to rotate; scroll to zoom.')
    add_step_footer(fig)
    plt.show()


def add_note(ax, text, x=0.98, y=0.46, ha='right', va='top', facecolor='white'):
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=FS_ANNOT,
        va=va, ha=ha,
        bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.95),
    )


# -------------------------------------------------------
# Single smooth hill with a clear global max at the origin-ish
# (avoid secondary peaks that confuse the probability story)
# -------------------------------------------------------
def f(x, y):
    return np.exp(-0.45 * ((x - 0.2) ** 2 + (y - 0.3) ** 2))


def grad(xy):
    x, y = float(xy[0]), float(xy[1])
    val = f(x, y)
    return np.array([
        val * (-0.90 * (x - 0.2)),
        val * (-0.90 * (y - 0.3)),
    ], dtype=float)


X_MAX = np.array([0.2, 0.3])
F_MAX = f(X_MAX[0], X_MAX[1])

xs = np.linspace(-2.0, 2.0, 100)
ys = np.linspace(-2.0, 2.0, 100)
XX, YY = np.meshgrid(xs, ys)
ZZ = f(XX, YY)


def ascent_path(x0, lr=0.35, steps=80, noise_std=0.0, seed=0):
    """
    Stochastic gradient ASCENT with zero-mean gradient noise:
        g_hat = grad + noise
        x <- x + lr_t * g_hat

    Keep the TRUE gradient scale (do not renormalize after noise),
    otherwise large noise turns every step into a random unit vector
    and the uphill drift disappears.

    Mild lr decay helps the climber settle at the maximum instead of
    wandering when |grad| becomes tiny near the summit.
    """
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    angles = []  # angle between noisy and true grad (degrees)
    for t in range(steps):
        g = grad(x)
        # shrink noise as we approach flat summit so convergence is visible
        local_noise = noise_std * (0.25 + 0.75 * min(1.0, np.linalg.norm(g) / 0.35))
        g_hat = g + rng.normal(0.0, local_noise, size=2)
        if np.linalg.norm(g) > 1e-12 and np.linalg.norm(g_hat) > 1e-12:
            cosang = np.clip(
                np.dot(g, g_hat) / (np.linalg.norm(g) * np.linalg.norm(g_hat)),
                -1.0, 1.0,
            )
            angles.append(float(np.degrees(np.arccos(cosang))))
        else:
            angles.append(0.0)
        lr_t = lr * (0.35 + 0.65 * (1.0 - t / max(steps - 1, 1)))
        x = x + lr_t * g_hat
        x = np.clip(x, -2.0, 2.0)
        path.append(x.copy())
    return np.stack(path), np.array(angles)


def success(path, tol=0.45):
    """Reached a neighborhood of the summit (in xy) with high value."""
    final = path[-1]
    near = np.linalg.norm(final - X_MAX) < tol
    high = f(final[0], final[1]) > 0.70 * F_MAX
    # also count as success if any late iterate entered the basin
    late = path[len(path) // 2:]
    visited = np.any(np.linalg.norm(late - X_MAX[None, :], axis=1) < tol)
    return bool((near and high) or visited)


X0 = np.array([-1.6, -1.4])
STEPS = 160          # more steps => slower visual approach to the max
LR = 0.55            # milder steps so the climb is easier to follow


# -------------------------------------------------------
# FIGURE 1: True ascent vs one noisy trajectory
# -------------------------------------------------------
path_true, _ = ascent_path(X0, lr=LR, steps=STEPS, noise_std=0.0, seed=0)
NOISE = 0.35  # large angle errors, but E[step] still uphill
path_noisy, angs = ascent_path(X0, lr=LR, steps=STEPS, noise_std=NOISE, seed=1)

def build_step_1():
    fig1, axes1 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig1.suptitle(
        'Step 1: Each noisy step is crooked -- but still tends uphill',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    ax = axes1[0]
    cs = ax.contourf(XX, YY, ZZ, levels=24, cmap='terrain')
    ax.contour(XX, YY, ZZ, levels=24, colors='k', linewidths=0.25, alpha=0.3)
    ax.plot(path_true[:, 0], path_true[:, 1], '-', color='k', lw=2.5, label='true ascent')
    ax.plot(path_noisy[:, 0], path_noisy[:, 1], '-', color='crimson', lw=2.0, alpha=0.9,
            label='noisy ascent')
    ax.scatter([X0[0]], [X0[1]], c='orange', s=70, zorder=5, label='start')
    ax.scatter([X_MAX[0]], [X_MAX[1]], c='gold', edgecolor='k', s=120, marker='*',
               zorder=5, label='summit')
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('y', fontsize=FS_AXIS)
    ax.set_title('One run: noisy path still finds the peak', fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(loc='lower right', fontsize=FS_LEGEND)
    fig1.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)

    ax = axes1[1]
    ax.plot(angs, color='crimson', lw=1.8)
    ax.axhline(np.mean(angs), color='k', ls='--', lw=1.5,
               label=rf'mean deviation ${np.mean(angs):.0f}^\circ$')
    ax.set_xlabel('step', fontsize=FS_AXIS)
    ax.set_ylabel('angle between noisy & true grad (deg)', fontsize=FS_AXIS)
    ax.set_title('Per-step direction error', fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, alpha=0.3)
    add_note(
        ax,
        'Individual steps can be\n'
        'tens of degrees off.\n\n'
        'The bias of the noise\n'
        'is still uphill, so the\n'
        'path drifts to the top.',
        y=0.98, va='top', facecolor='lightyellow',
    )

    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    return fig1


# -------------------------------------------------------
# FIGURE 2: Many noisy runs -- high success probability
# -------------------------------------------------------
N_RUNS = 50
paths = []
ok = []
for i in range(N_RUNS):
    pth, _ = ascent_path(X0, lr=LR, steps=STEPS, noise_std=NOISE, seed=100 + i)
    paths.append(pth)
    ok.append(success(pth))
ok = np.array(ok)
success_rate = ok.mean()

def build_step_2():
    fig2, ax2 = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig2.suptitle(
        f'Step 2: {N_RUNS} noisy runs  |  success rate = {100 * success_rate:.0f}%',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    cs = ax2.contourf(XX, YY, ZZ, levels=24, cmap='terrain', alpha=0.8)
    ax2.contour(XX, YY, ZZ, levels=24, colors='k', linewidths=0.2, alpha=0.25)
    for i, pth in enumerate(paths):
        color = 'seagreen' if ok[i] else 'crimson'
        ax2.plot(pth[:, 0], pth[:, 1], '-', color=color, alpha=0.35, lw=1.2)
    ax2.plot([], [], color='seagreen', lw=2, label='reached summit')
    ax2.plot([], [], color='crimson', lw=2, label='failed / stuck')
    ax2.scatter([X0[0]], [X0[1]], c='orange', s=80, zorder=5, label='start')
    ax2.scatter([X_MAX[0]], [X_MAX[1]], c='gold', edgecolor='k', s=140,
                marker='*', zorder=5, label='summit')
    # success circle
    circ = patches.Circle(X_MAX, 0.45, fill=False, ec='k', ls='--', lw=1.5)
    ax2.add_patch(circ)
    ax2.set_xlim(float(XX.min()), float(XX.max()))
    ax2.set_ylim(float(YY.min()), float(YY.max()))
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_xlabel('x', fontsize=FS_AXIS)
    ax2.set_ylabel('y', fontsize=FS_AXIS)
    ax2.tick_params(axis='both', labelsize=FS_TICK)
    ax2.legend(loc='lower right', fontsize=FS_LEGEND)
    fig2.colorbar(cs, ax=ax2, fraction=0.046, pad=0.04)
    add_note(
        ax2,
        f'Noise std = {NOISE:g}\n'
        f'(vs |grad| scale ~ O(1))\n\n'
        'Most trajectories still\n'
        'enter the summit ball\n'
        'with high probability.',
        y=0.98, va='top', facecolor='lightyellow',
    )

    fig2.subplots_adjust(left=0.10, right=0.92, bottom=0.12, top=0.88)
    return fig2


# -------------------------------------------------------
# ANIMATION A: race -- true climber vs one noisy climber
# -------------------------------------------------------
def frame_schedule(n_steps, hold_sec=ANIM_HOLD_SEC, fps=ANIM_FPS):
    """Play each optimization step once, then hold on the final frame."""
    hold = max(1, int(round(hold_sec * fps)))
    return list(range(n_steps)) + [n_steps - 1] * hold


# Race GIF: opt-in export only (skipped during normal lecture launch).
if not should_export_gif():
    print('Skipping race GIF (set LECTURE_EXPORT_GIF=1 to enable)')
elif OUT_RACE.is_file() and OUT_RACE.stat().st_size > 0:
    print(f'Skipping race GIF figure (already exists): {OUT_RACE}')
else:
    print(f'\nRendering climbing animation -> {OUT_RACE.name} ...')

    fig_r, ax_r = plt.subplots(figsize=(8.5, 7.0), dpi=ANIM_DPI)
    fig_r.suptitle(
        'Live climb: true ascent vs noisy ascent\n'
        '(slow motion -- watch the point converge to the summit)',
        fontsize=FS_AX_TITLE, fontweight='bold', y=0.98,
    )
    ax_r.contourf(XX, YY, ZZ, levels=24, cmap='terrain', alpha=0.85)
    ax_r.contour(XX, YY, ZZ, levels=24, colors='k', linewidths=0.2, alpha=0.25)
    ax_r.scatter([X_MAX[0]], [X_MAX[1]], c='gold', edgecolor='k', s=140,
                 marker='*', zorder=5, label='summit')
    ax_r.scatter([X0[0]], [X0[1]], c='orange', s=70, zorder=5, label='start')

    trail_true, = ax_r.plot([], [], '-', color='k', lw=2.0, label='true trail')
    trail_noisy, = ax_r.plot([], [], '-', color='crimson', lw=1.8, alpha=0.9, label='noisy trail')
    pt_true = patches.Circle(X0, 0.07, facecolor='k', edgecolor='white', lw=1.2, zorder=6)
    pt_noisy = patches.Circle(X0, 0.07, facecolor='crimson', edgecolor='white', lw=1.2, zorder=6)
    ax_r.add_patch(pt_true)
    ax_r.add_patch(pt_noisy)

    # arrows: true grad (black) and noisy step direction (red) at noisy climber
    arr_true = ax_r.annotate(
        '', xy=X0, xytext=X0,
        arrowprops=dict(arrowstyle='->', color='0.15', lw=2.0), zorder=7,
    )
    arr_noisy = ax_r.annotate(
        '', xy=X0, xytext=X0,
        arrowprops=dict(arrowstyle='->', color='crimson', lw=2.2), zorder=7,
    )

    hud_r = ax_r.text(
        0.02, 0.98, '', transform=ax_r.transAxes, va='top', ha='left',
        fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
    )
    ax_r.text(
        0.98, 0.02,
        'black arrow: true grad at noisy point\n'
        'red arrow: noisy step actually taken',
        transform=ax_r.transAxes, va='bottom', ha='right', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    ax_r.set_xlim(-2.0, 2.0)
    ax_r.set_ylim(-2.0, 2.0)
    ax_r.set_aspect('equal')
    ax_r.set_xlabel('x', fontsize=FS_AXIS)
    ax_r.set_ylabel('y', fontsize=FS_AXIS)
    ax_r.legend(loc='upper right', fontsize=FS_LEGEND)
    ax_r.grid(True, alpha=0.25)
    fig_r.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.86)

    # Precompute noisy step vectors used along path_noisy
    noisy_steps = path_noisy[1:] - path_noisy[:-1]
    n_steps_race = len(path_noisy)
    race_frames = frame_schedule(n_steps_race)


    def init_race():
        trail_true.set_data([], [])
        trail_noisy.set_data([], [])
        return trail_true, trail_noisy, pt_true, pt_noisy, hud_r, arr_true, arr_noisy


    def update_race(i):
        trail_true.set_data(path_true[: i + 1, 0], path_true[: i + 1, 1])
        trail_noisy.set_data(path_noisy[: i + 1, 0], path_noisy[: i + 1, 1])
        pt_true.center = (path_true[i, 0], path_true[i, 1])
        pt_noisy.center = (path_noisy[i, 0], path_noisy[i, 1])

        g = grad(path_noisy[i])
        if np.linalg.norm(g) > 1e-12:
            gu = 0.35 * g / np.linalg.norm(g)
        else:
            gu = np.zeros(2)
        arr_true.set_position((path_noisy[i, 0], path_noisy[i, 1]))
        arr_true.xy = (path_noisy[i, 0] + gu[0], path_noisy[i, 1] + gu[1])

        if i < len(noisy_steps) and np.linalg.norm(noisy_steps[i]) > 1e-12:
            su = 0.35 * noisy_steps[i] / np.linalg.norm(noisy_steps[i])
        else:
            su = np.zeros(2)
        arr_noisy.set_position((path_noisy[i, 0], path_noisy[i, 1]))
        arr_noisy.xy = (path_noisy[i, 0] + su[0], path_noisy[i, 1] + su[1])

        ang = float(angs[i]) if i < len(angs) else float(angs[-1])
        dist = float(np.linalg.norm(path_noisy[i] - X_MAX))
        converged = dist < 0.45 and f(*path_noisy[i]) > 0.70 * F_MAX
        status = 'CONVERGED at summit' if converged else 'climbing...'
        hud_r.set_text(
            f'step {i}/{n_steps_race - 1}   [{status}]\n'
            f'f_true  = {f(*path_true[i]):.3f}\n'
            f'f_noisy = {f(*path_noisy[i]):.3f}\n'
            f'dist to max = {dist:.3f}\n'
            f'angle error = {ang:.0f} deg'
        )
        return trail_true, trail_noisy, pt_true, pt_noisy, hud_r, arr_true, arr_noisy


    anim_race = animation.FuncAnimation(
        fig_r, update_race, init_func=init_race,
        frames=race_frames, interval=1000 / ANIM_FPS, blit=False,
    )
    maybe_save_animation(anim_race, OUT_RACE, fps=ANIM_FPS)
    # Do not plt.show() here -- that would display before the slide deck.
    plt.close(fig_r)


# -------------------------------------------------------
# ANIMATION B: swarm -- several noisy climbers moving together
# -------------------------------------------------------
# Swarm GIF: opt-in export only (skipped during normal lecture launch).
if not should_export_gif():
    print('Skipping swarm GIF (set LECTURE_EXPORT_GIF=1 to enable)')
elif OUT_SWARM.is_file() and OUT_SWARM.stat().st_size > 0:
    print(f'Skipping swarm GIF figure (already exists): {OUT_SWARM}')
else:
    print(f'Rendering swarm animation -> {OUT_SWARM.name} ...')

    N_SWARM = 8
    swarm_paths = []
    for i in range(N_SWARM):
        pth, _ = ascent_path(X0, lr=LR, steps=STEPS, noise_std=NOISE, seed=200 + i)
        swarm_paths.append(pth)
    swarm_ok = np.array([success(p) for p in swarm_paths])

    fig_s, ax_s = plt.subplots(figsize=(8.5, 7.0), dpi=ANIM_DPI)
    fig_s.suptitle(
        'Live climb: several noisy climbers at once\n'
        'slow motion -- crooked steps, shared destination',
        fontsize=FS_AX_TITLE, fontweight='bold', y=0.98,
    )
    ax_s.contourf(XX, YY, ZZ, levels=24, cmap='terrain', alpha=0.85)
    ax_s.contour(XX, YY, ZZ, levels=24, colors='k', linewidths=0.2, alpha=0.25)
    ax_s.scatter([X_MAX[0]], [X_MAX[1]], c='gold', edgecolor='k', s=140,
                 marker='*', zorder=5, label='summit')
    ax_s.scatter([X0[0]], [X0[1]], c='orange', s=70, zorder=5, label='start')
    ax_s.add_patch(plt.Circle(X_MAX, 0.45, fill=False, ec='k', ls='--', lw=1.3))

    swarm_colors = ['seagreen' if swarm_ok[i] else 'crimson' for i in range(N_SWARM)]

    trails = []
    dots = []
    for i in range(N_SWARM):
        tr, = ax_s.plot([], [], '-', color=swarm_colors[i], lw=1.4, alpha=0.75)
        trails.append(tr)
        dot = patches.Circle(X0, 0.06, facecolor=swarm_colors[i], edgecolor='white',
                             lw=1.0, zorder=6)
        ax_s.add_patch(dot)
        dots.append(dot)

    ax_s.plot([], [], color='seagreen', lw=2, label='will succeed')
    ax_s.plot([], [], color='crimson', lw=2, label='will fail')
    hud_s = ax_s.text(
        0.02, 0.98, '', transform=ax_s.transAxes, va='top', ha='left',
        fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
    )

    ax_s.set_xlim(-2.0, 2.0)
    ax_s.set_ylim(-2.0, 2.0)
    ax_s.set_aspect('equal')
    ax_s.set_xlabel('x', fontsize=FS_AXIS)
    ax_s.set_ylabel('y', fontsize=FS_AXIS)
    ax_s.legend(loc='lower right', fontsize=FS_LEGEND)
    ax_s.grid(True, alpha=0.25)
    fig_s.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.86)

    n_steps_swarm = len(swarm_paths[0])
    swarm_frames = frame_schedule(n_steps_swarm)


    def init_swarm():
        for tr in trails:
            tr.set_data([], [])
        return trails + dots + [hud_s]


    def update_swarm(i):
        heights = []
        for k in range(N_SWARM):
            pth = swarm_paths[k]
            trails[k].set_data(pth[: i + 1, 0], pth[: i + 1, 1])
            dots[k].center = (pth[i, 0], pth[i, 1])
            heights.append(f(*pth[i]))
        n_near = sum(np.linalg.norm(swarm_paths[k][i] - X_MAX) < 0.45 for k in range(N_SWARM))
        status = 'MOSTLY CONVERGED' if n_near >= max(1, N_SWARM - 1) else 'climbing...'
        hud_s.set_text(
            f'step {i}/{n_steps_swarm - 1}   [{status}]\n'
            f'climbers near summit: {n_near}/{N_SWARM}\n'
            f'mean f = {np.mean(heights):.3f}   (f_max={F_MAX:.3f})'
        )
        return trails + dots + [hud_s]


    anim_swarm = animation.FuncAnimation(
        fig_s, update_swarm, init_func=init_swarm,
        frames=swarm_frames, interval=1000 / ANIM_FPS, blit=False,
    )
    maybe_save_animation(anim_swarm, OUT_SWARM, fps=ANIM_FPS)
    # Do not plt.show() here -- that would display before the slide deck.
    plt.close(fig_s)

plt.close('all')  # clear any GIF figures before the slide deck


# -------------------------------------------------------
# FIGURE 3: Success probability vs noise level
# -------------------------------------------------------
noise_grid = np.array([0.0, 0.25, 0.4, 0.55, 0.8, 1.1, 1.6, 2.2])
_noise_stats = None


def _noise_success_stats():
    """Lazy Monte Carlo for Step 3 -- avoid delaying earlier slides."""
    global _noise_stats
    if _noise_stats is not None:
        return _noise_stats
    n_per = 40
    rates = []
    final_dists = []
    for ns in noise_grid:
        hits = []
        dists = []
        for i in range(n_per):
            pth, _ = ascent_path(
                X0, lr=LR, steps=STEPS, noise_std=float(ns),
                seed=1000 + int(100 * ns) + i,
            )
            hits.append(success(pth))
            dists.append(np.linalg.norm(pth[-1] - X_MAX))
        rates.append(np.mean(hits))
        final_dists.append(np.mean(dists))
    _noise_stats = (rates, final_dists)
    return _noise_stats


def build_step_3():
    rates, final_dists = _noise_success_stats()
    fig3, axes3 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig3.suptitle(
        'Step 3: More noise => lower success, but moderate noise is still fine',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    ax = axes3[0]
    ax.plot(noise_grid, 100 * np.array(rates), 'o-', color='royalblue', lw=2.5)
    ax.set_xlabel('noise std on gradient', fontsize=FS_AXIS)
    ax.set_ylabel('success rate (%)', fontsize=FS_AXIS)
    ax.set_title('Probability of reaching the summit', fontsize=FS_AX_TITLE)
    ax.set_ylim(-5, 105)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    add_note(
        ax,
        'SGD-like lesson:\n'
        'noisy updates remain\n'
        'useful while the drift\n'
        '(true grad) dominates\n'
        'the jitter.',
        x=0.02, y=0.02, ha='left', va='bottom', facecolor='lightyellow',
    )

    ax = axes3[1]
    ax.plot(noise_grid, final_dists, 's-', color='crimson', lw=2.5)
    ax.set_xlabel('noise std on gradient', fontsize=FS_AXIS)
    ax.set_ylabel('mean final distance to summit', fontsize=FS_AXIS)
    ax.set_title('How close do we finish, on average?', fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)

    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    return fig3


# -------------------------------------------------------
# FIGURE 4: Objective value over time -- drift + jitter
# -------------------------------------------------------
def build_step_4():
    fig4, ax4 = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig4.suptitle(
        'Step 4: f(x_t) rises with jitter, but the trend is upward',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    # plot several noisy objective traces + the true-ascent baseline
    for i in range(12):
        pth, _ = ascent_path(X0, lr=LR, steps=STEPS, noise_std=NOISE, seed=50 + i)
        vals = np.array([f(p[0], p[1]) for p in pth])
        ax4.plot(vals, color='crimson', alpha=0.35, lw=1.3)
    ax4.plot([], [], color='crimson', alpha=0.8, lw=2, label='noisy runs')
    vals_true = np.array([f(p[0], p[1]) for p in path_true])
    ax4.plot(vals_true, color='k', lw=2.8, label='true ascent')
    ax4.axhline(F_MAX, color='gold', ls='--', lw=2, label='summit value')
    ax4.set_xlabel('step', fontsize=FS_AXIS)
    ax4.set_ylabel('f(x_t)', fontsize=FS_AXIS)
    ax4.set_title('Accumulation beats per-step error', fontsize=FS_AX_TITLE)
    ax4.tick_params(axis='both', labelsize=FS_TICK)
    ax4.legend(fontsize=FS_LEGEND, loc='lower right')
    ax4.grid(True, alpha=0.3)
    add_note(
        ax4,
        'Local dips happen when\n'
        'noise points downhill.\n\n'
        'Over many steps the\n'
        'positive drift wins\n'
        'with high probability.',
        y=0.98, va='top', facecolor='lightyellow',
    )

    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    return fig4


# -------------------------------------------------------
# FIGURE 5: 3D browsable noisy path
# -------------------------------------------------------
def build_step_5():
    fig5 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig5.suptitle(
        'Step 5: Browse a noisy climb on the surface',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    ax = fig5.add_subplot(111, projection='3d')
    ax.plot_surface(XX, YY, ZZ, cmap='terrain', linewidth=0, alpha=0.35)
    z_true = np.array([f(p[0], p[1]) for p in path_true])
    z_noisy = np.array([f(p[0], p[1]) for p in path_noisy])
    ax.plot(path_true[:, 0], path_true[:, 1], z_true, color='k', lw=2.5, label='true')
    ax.plot(path_noisy[:, 0], path_noisy[:, 1], z_noisy, color='crimson', lw=2.0, label='noisy')
    ax.scatter([X0[0]], [X0[1]], [f(*X0)], c='orange', s=50)
    ax.scatter([X_MAX[0]], [X_MAX[1]], [F_MAX], c='gold', edgecolor='k', s=80, marker='*')
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('y', fontsize=FS_AXIS)
    ax.set_zlabel('f', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_ANNOT)
    ax.legend(loc='upper left', fontsize=FS_LEGEND)
    ax.text2D(
        0.02, 0.02,
        'Crooked path, same destination.',
        transform=ax.transAxes, fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
    )
    return fig5


# -------------------------------------------------------
# FIGURE 6: Why it works -- bias/variance cartoon
# -------------------------------------------------------
def build_step_6():
    fig6, axes6 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig6.suptitle(
        'Step 6: Noise averages out; the expected step still points uphill',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    ax = axes6[0]
    # sample noisy gradients at a fixed point and show cloud + mean
    p_fixed = np.array([-0.8, -0.5])
    g_true = grad(p_fixed)
    noisy = g_true[None, :] + RNG.normal(0, NOISE, size=(80, 2))
    ax.scatter(noisy[:, 0], noisy[:, 1], s=25, c='crimson', alpha=0.45, label='noisy grads')
    ax.arrow(0, 0, g_true[0], g_true[1], head_width=0.08, color='k',
             length_includes_head=True, lw=2.5, label='true grad')
    mean_n = noisy.mean(axis=0)
    ax.arrow(0, 0, mean_n[0], mean_n[1], head_width=0.08, color='limegreen',
             length_includes_head=True, lw=2.5, label='mean noisy grad')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$g_x$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$g_y$', fontsize=FS_AXIS)
    ax.set_title('Noise cloud around the true gradient', fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc='best')
    ax.grid(True, alpha=0.3)
    add_note(
        ax,
        'E[g + noise] = g\n'
        'if noise is zero-mean.\n\n'
        'So the average update\n'
        'is still steepest-ascent.',
        y=0.98, va='top', facecolor='lightyellow',
    )

    ax = axes6[1]
    ax.axis('off')
    ax.set_title('Takeaway for SGD / stochastic climbing', fontsize=FS_AX_TITLE)
    ax.text(
        0.5, 0.50,
        '1. Per-step direction can be badly wrong.\n\n'
        '2. If noise is roughly zero-mean, the expected\n'
        '   step still follows grad f (uphill drift).\n\n'
        '3. Over many steps, the drift accumulates and\n'
        '   the variance grows slower than the signal\n'
        '   (law-of-large-numbers style cancellation).\n\n'
        '4. Therefore: with high probability, noisy ascent\n'
        '   still reaches a neighborhood of the summit --\n'
        '   exactly the SGD miracle in optimization form.',
        transform=ax.transAxes, ha='center', va='center',
        fontsize=FS_AXIS + 1,
        linespacing=1.4,
        bbox=dict(boxstyle='round,pad=1.2', facecolor='#f7f7ff',
                  edgecolor='#2874a6', linewidth=2.0, alpha=0.97),
    )

    fig6.tight_layout(rect=[0, 0, 1, 0.94])
    return fig6


# Ensure no leftover GIF/animation figures steal the first deck windows.
plt.close('all')

run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
    build_step_6,
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("STOCHASTIC ASCENT")
print("=" * 60)
print(f"""
Update:
  g_hat = grad f(x) + noise
  x <- x + lr * g_hat     (keep grad scale so E[step] stays uphill)

At start {X0}, summit ~ {X_MAX}, f_max ~ {F_MAX:.3f}

One noisy run (noise_std={NOISE}):
  mean angle error vs true grad: {np.mean(angs):.1f} degrees
  final distance to summit: {np.linalg.norm(path_noisy[-1] - X_MAX):.3f}
  success: {success(path_noisy)}

Monte Carlo ({N_RUNS} runs, noise_std={NOISE}):
  success rate: {100 * success_rate:.1f}%

Why it works:
  E[noisy grad] ~= true grad  (zero-mean noise)
  => there is a consistent uphill DRIFT
  => accumulated steps reach the top with high probability

SGD analogy:
  Minibatch gradients are noisy directional estimates of the true loss
  gradient. Training still works because progress is in expectation.

Animations written:
  {OUT_RACE}
  {OUT_SWARM}
""")
