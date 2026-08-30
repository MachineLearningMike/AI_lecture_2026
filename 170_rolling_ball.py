"""
170_rolling_ball.py

A ball rolling down a hill -- physics animation as intuition for
gradient descent: the landscape is a loss surface, gravity pulls
toward lower values, friction damps oscillations.

Shows the animation live. Also saves a GIF next to this script
(GIF export is skipped if the file already exists):
    rolling_ball.gif

Companion metaphor for training loops in 080_model.py.
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
)
apply_rcparams(plt)
import matplotlib.patches as patches
from matplotlib import animation
import numpy as np

np.random.seed(42)


FPS = 30
DURATION_S = 8.0
N_FRAMES = int(FPS * DURATION_S)

OUT_PATH = Path(__file__).resolve().parent / 'rolling_ball.gif'


# -------------------------------------------------------
# Hill geometry: height h(x) and derivatives
# Left shoulder -> valley near 0 -> smaller right bump
# -------------------------------------------------------
def h(x):
    x = np.asarray(x, dtype=float)
    return (
        0.35 * x ** 2 / (1.0 + 0.12 * x ** 2)
        + 1.10 * np.exp(-0.55 * (x + 2.4) ** 2)
        + 0.35 * np.exp(-0.80 * (x - 2.2) ** 2)
    )


def h_x(x):
    """dh/dx (analytic)."""
    x = np.asarray(x, dtype=float)
    # d/dx [a x^2 / (1 + b x^2)] = a * 2x / (1 + b x^2)^2
    a, b = 0.35, 0.12
    term1 = a * 2.0 * x / (1.0 + b * x ** 2) ** 2
    # d/dx [A exp(-c (x-m)^2)] = A exp(...) * (-2 c (x-m))
    term2 = 1.10 * np.exp(-0.55 * (x + 2.4) ** 2) * (-2.0 * 0.55 * (x + 2.4))
    term3 = 0.35 * np.exp(-0.80 * (x - 2.2) ** 2) * (-2.0 * 0.80 * (x - 2.2))
    return term1 + term2 + term3


def h_xx(x):
    """d2h/dx2 via central differences (smooth enough for the demo)."""
    eps = 1e-4
    return (h_x(x + eps) - h_x(x - eps)) / (2.0 * eps)


# -------------------------------------------------------
# Constrained dynamics on y = h(x)
#   (1 + h'^2) x'' + h' h'' (x')^2 = -g h' - gamma (1 + h'^2) x'
# -------------------------------------------------------
G = 3.2          # gravity (scaled for a snappy demo)
GAMMA = 0.55     # viscous friction along the track
BALL_R = 0.12    # drawn radius (world units)
X0 = -2.55       # start near the left peak
V0 = 0.0


def surface_point(x):
    """Contact point on the curve, offset by radius along the normal."""
    hx = h_x(x)
    # unit tangent ~ (1, hx) / norm, unit normal pointing "upward"
    norm = np.sqrt(1.0 + hx ** 2)
    # normal with positive y-ish component: (-hx, 1) / norm  or (hx, -1)?
    # For graph y=h(x), upward normal is (-h', 1) / sqrt(1+h'^2)
    nx, ny = -hx / norm, 1.0 / norm
    return x + BALL_R * nx, h(x) + BALL_R * ny


def step(x, v, dt):
    """One semi-implicit Euler step for x on the curve."""
    hx = float(h_x(x))
    hxx = float(h_xx(x))
    denom = 1.0 + hx ** 2
    accel = (-G * hx - hx * hxx * v ** 2) / denom - GAMMA * v
    v_new = v + accel * dt
    x_new = x + v_new * dt
    return x_new, v_new


def simulate(n_frames=N_FRAMES, fps=FPS):
    dt = 1.0 / fps
    xs = np.zeros(n_frames)
    vs = np.zeros(n_frames)
    x, v = float(X0), float(V0)
    for i in range(n_frames):
        xs[i] = x
        vs[i] = v
        x, v = step(x, v, dt)
    return xs, vs


# -------------------------------------------------------
# Build animation, optionally save GIF, then show
# -------------------------------------------------------
def render_rolling_ball_gif(out_path: Path = OUT_PATH) -> None:
    xs_sim, vs_sim = simulate()

    x_hill = np.linspace(-3.6, 3.6, 600)
    y_hill = h(x_hill)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=FIGURE_DPI)
    fig.suptitle(
        'Ball rolling down a hill\n(gravity + friction  ~  gradient descent)',
        fontsize=FS_FIGURE_TITLE, fontweight='bold', y=0.98,
    )

    ax.fill_between(x_hill, y_hill, -0.2, color='#c4a574', alpha=0.85, zorder=1)
    ax.plot(x_hill, y_hill, color='#5c4033', linewidth=2.5, zorder=2)

    trail_line, = ax.plot([], [], color='crimson', linewidth=2.0, alpha=0.7, zorder=3, label='path')

    bx0, by0 = surface_point(xs_sim[0])
    ball = patches.Circle((bx0, by0), BALL_R, facecolor='#e74c3c',
                          edgecolor='#6d0f0f', linewidth=1.5, zorder=5)
    ax.add_patch(ball)
    shine = patches.Circle((bx0 - 0.03, by0 + 0.04), BALL_R * 0.28,
                           facecolor='white', alpha=0.45, zorder=6)
    ax.add_patch(shine)

    arrow = ax.annotate(
        '', xy=(bx0, by0), xytext=(bx0, by0),
        arrowprops=dict(arrowstyle='->', color='navy', lw=2.0),
        zorder=7,
    )

    info = ax.text(
        0.02, 0.95, '', transform=ax.transAxes, fontsize=FS_ANNOT, va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
    )

    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-0.2, float(y_hill.max()) + 0.55)
    ax.set_aspect('equal')
    ax.set_xlabel('position x', fontsize=FS_AXIS)
    ax.set_ylabel('height h(x)', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK - 2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=FS_LEGEND)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.86)

    ax.text(
        0.98, 0.05,
        'Potential energy ~ loss\n'
        'Gravity ~ negative gradient\n'
        'Friction ~ damping / small step size',
        transform=ax.transAxes, fontsize=FS_ANNOT,
        va='bottom', ha='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    def energy(x, v):
        """Approximate PE + KE for display (unit mass)."""
        hx = float(h_x(x))
        speed_along = abs(v) * np.sqrt(1.0 + hx ** 2)
        pe = G * float(h(x))
        ke = 0.5 * speed_along ** 2
        return pe, ke

    def init():
        trail_line.set_data([], [])
        return ball, shine, trail_line, info, arrow

    def update(frame):
        x = xs_sim[frame]
        v = vs_sim[frame]
        bx, by = surface_point(x)
        ball.center = (bx, by)
        shine.center = (bx - 0.03, by + 0.04)

        trail_x = xs_sim[: frame + 1]
        trail_line.set_data(trail_x, h(trail_x))

        hx = float(h_x(x))
        tang_norm = np.sqrt(1.0 + hx ** 2)
        tx, ty = 1.0 / tang_norm, hx / tang_norm
        scale = 0.35 * np.clip(v, -2.5, 2.5)
        arrow.set_position((bx, by))
        arrow.xy = (bx + scale * tx, by + scale * ty)

        pe, ke = energy(x, v)
        info.set_text(
            f't = {frame / FPS:4.2f}s\n'
            f'x = {x:6.3f}\n'
            f'v = {v:6.3f}\n'
            f'PE = {pe:6.3f}\n'
            f'KE = {ke:6.3f}'
        )
        return ball, shine, trail_line, info, arrow

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=N_FRAMES, interval=1000 / FPS, blit=False,
    )

    if out_path.is_file() and out_path.stat().st_size > 0:
        print(f'\nSkipping GIF export (already exists): {out_path}')
    else:
        print(f'Saving animation to {out_path} ...')
        try:
            anim.save(out_path, writer=animation.PillowWriter(fps=FPS))
            print(f'Saved: {out_path}')
        except Exception as exc:
            print(f'Skipping GIF export ({out_path.name}): {exc}')
            print('Install pillow to enable animation export: pip install pillow')

    add_step_footer(fig)
    try:
        if plt.get_backend().lower() != 'agg':
            plt.show()
        else:
            plt.close(fig)
    except Exception:
        plt.close(fig)


render_rolling_ball_gif(OUT_PATH)


print("""
Intuition:
  - Height of the hill  ~  loss / potential energy
  - Steep slope         ~  large gradient  (fast acceleration)
  - Friction            ~  keeps the ball from oscillating forever
                          (like a small learning rate / momentum damping)

Gradient descent is the discrete cousin of this motion:
  x <- x - lr * dL/dx
""")
