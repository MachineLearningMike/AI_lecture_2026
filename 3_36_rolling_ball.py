"""
3_36_rolling_ball.py

A ball rolling down a hill -- physics animation as intuition for
optimization with momentum.

Dictionary (careful -- not slogans):
  height h(x)              ~  loss L(theta)
  gravity -> force -g h'   ~  force along -grad L  (downhill)
  friction / viscous drag  ~  velocity damping (momentum decay)
  learning rate            ~  discrete step size along -grad L
                              (NOT friction)

Shows the animation live. Also saves a GIF next to this script
(GIF export is skipped if the file already exists):
    rolling_ball.gif
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
    maybe_save_animation,
    run_slide_deck,
)
apply_rcparams(plt)
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from matplotlib import animation
import numpy as np


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


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))

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
# Step 1: animation (optionally also write GIF)
# -------------------------------------------------------
def build_step_1(out_path: Path = OUT_PATH):
    xs_sim, vs_sim = simulate()

    x_hill = np.linspace(-3.6, 3.6, 600)
    y_hill = h(x_hill)

    fig, ax = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle(
        'Step 1: Ball rolling down a hill',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
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
        'See Step 2 for the careful map\n'
        'physics  <->  optimization.\n'
        '(Slogans like "gravity = -grad"\n'
        'are wrong.)',
        transform=ax.transAxes, fontsize=FS_ANNOT - 1,
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

    maybe_save_animation(anim, out_path, fps=FPS)

    fig._ball_anim = anim  # keep alive for the deck
    return fig


# -------------------------------------------------------
# Step 2: careful physics <-> optimization map
# -------------------------------------------------------
def build_step_2():
    fig, ax = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle(
        r'Step 2: Gravity is not $-\nabla L$ -- mass + friction is momentum',
        fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
    )
    blank_slide(ax)
    # 1 cm in data units on this blank slide.
    dy = 0.664
    top_y = 6.35 + dy
    mid_y, mid_h = 3.55, 2.55 + dy
    bot_y, bot_h = 0.35 - dy, 2.95 + dy
    ax.set_ylim(bot_y - 0.2, top_y + 1.35 + 0.2)

    round_box(ax, 0.35, top_y, 11.3, 1.35, '#1a5276', '#1a5276', lw=0)
    ax.text(
        6.0, top_y + 1.35 / 2,
        'The rolling ball is intuition for momentum methods -- not a derivation of GD.',
        ha='center', va='center', fontsize=FS_ANNOT,
        fontweight='bold', color='#f9e79f',
    )

    round_box(ax, 0.35, mid_y, 5.5, mid_h, '#f8d7da', '#c0392b', lw=2.0)
    ax.text(
        3.1, mid_y + mid_h - 0.45, 'Why gravity ≠ -∇L',
        ha='center', fontsize=FS_ANNOT, fontweight='bold', color='#c0392b',
    )
    ax.text(
        3.1, mid_y + mid_h / 2 - 0.25,
        'Gravity pulls straight down in space.\n'
        'The ball is constrained to the hill.\n'
        'Only the tangential part of that force\n'
        'accelerates it along the track -- with\n'
        'extra geometric factors from the curve.\n\n'
        'Plain GD is a modeling choice:\n'
        r'$\theta\leftarrow\theta-\eta\nabla L(\theta)$'
        '\nin Euclidean parameter space.',
        ha='center', va='center', fontsize=FS_TICK - 2, color='#641e16',
        linespacing=1.25,
    )

    round_box(ax, 6.15, mid_y, 5.5, mid_h, '#d4edda', '#1e8449', lw=2.0)
    ax.text(
        8.9, mid_y + mid_h - 0.45, 'What still maps cleanly',
        ha='center', fontsize=FS_ANNOT, fontweight='bold', color='#1e8449',
    )
    ax.text(
        8.9, mid_y + mid_h / 2 - 0.25,
        'Height h(x)  ~  loss L  (a potential\n'
        'we want low).\n\n'
        'Steep slope  ~  large |∇L|\n'
        '(stronger local push).\n\n'
        'Learning rate η  ~  discrete jump size\n'
        'along -∇L  (not friction).',
        ha='center', va='center', fontsize=FS_TICK - 2, color='#145a32',
        linespacing=1.25,
    )

    round_box(ax, 0.35, bot_y, 11.3, bot_h, '#d6eaf8', '#2874a6', lw=2.0)
    ax.text(
        6.0, bot_y + bot_h - 0.45, 'Mass + friction  ↔  momentum methods',
        ha='center', fontsize=FS_ANNOT, fontweight='bold', color='#1a5276',
    )
    ax.text(
        6.0, bot_y + bot_h / 2 - 0.2,
        r"Physics:  $m\ddot{x}=-V'(x)-\gamma\dot{x}$"
        "\n"
        r"Discrete:  $v\leftarrow\beta v-\eta\nabla L$,   $\theta\leftarrow\theta+v$"
        '\n\n'
        'mass m  →  inertia / velocity buffer (momentum)\n'
        'friction γ  →  velocity damping (momentum decay β)\n'
        'Δt/m  →  scale of the gradient kick (~ step size)\n\n'
        'Plain GD ≈ no mass: each step is a fresh jump -η∇L from rest.\n'
        'Heavy-ball / momentum ≈ this rolling ball (mass + friction).',
        ha='center', va='center', fontsize=FS_TICK - 1, color='#1c2833',
        linespacing=1.3,
    )
    return fig


# -------------------------------------------------------
# Step 3: side-by-side similarity of the update equations
# -------------------------------------------------------
def build_step_3():
    fig, ax = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle(
        'Step 3: Same skeleton -- physics ODE vs momentum / GD updates',
        fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
    )
    blank_slide(ax)

    round_box(ax, 0.35, 6.45, 11.3, 1.2, '#1a5276', '#1a5276', lw=0)
    ax.text(
        6.0, 7.05,
        'Discretize the damped Newtonian ODE -- the algebra looks like heavy-ball GD.',
        ha='center', va='center', fontsize=FS_ANNOT - 1,
        fontweight='bold', color='#f9e79f',
    )

    # Left: physics
    round_box(ax, 0.35, 3.35, 5.5, 2.9, '#f8d7da', '#c0392b', lw=2.0)
    ax.text(
        3.1, 5.85, 'Physics (continuous)',
        ha='center', fontsize=FS_ANNOT - 1, fontweight='bold', color='#c0392b',
    )
    ax.text(
        3.1, 4.55,
        r"$m\,\ddot{x}=-V'(x)-\gamma\,\dot{x}$"
        "\n\n"
        r"With $v=\dot{x}$ and step $\Delta t$:"
        "\n"
        r"$v\leftarrow\left(1-\frac{\gamma\Delta t}{m}\right)v"
        r"-\frac{\Delta t}{m}V'(x)$"
        "\n"
        r"$x\leftarrow x+\Delta t\,v$",
        ha='center', va='center', fontsize=FS_TICK - 1, color='#641e16',
        linespacing=1.35,
    )

    # Right: optimizer
    round_box(ax, 6.15, 3.35, 5.5, 2.9, '#d4edda', '#1e8449', lw=2.0)
    ax.text(
        8.9, 5.85, 'Optimizer (discrete)',
        ha='center', fontsize=FS_ANNOT - 1, fontweight='bold', color='#1e8449',
    )
    ax.text(
        8.9, 4.55,
        'Heavy-ball / momentum:'
        "\n"
        r"$v\leftarrow\beta\,v-\eta\,\nabla L(\theta)$"
        "\n"
        r"$\theta\leftarrow\theta+v$"
        "\n\n"
        'Plain GD (no mass / no memory):'
        "\n"
        r"$\theta\leftarrow\theta-\eta\,\nabla L(\theta)$",
        ha='center', va='center', fontsize=FS_TICK - 1, color='#145a32',
        linespacing=1.35,
    )

    # Bottom: term-by-term match
    round_box(ax, 0.35, 0.35, 11.3, 2.75, '#d6eaf8', '#2874a6', lw=2.0)
    ax.text(
        6.0, 2.7, 'Term-by-term similarity',
        ha='center', fontsize=FS_ANNOT - 1, fontweight='bold', color='#1a5276',
    )
    ax.text(
        6.0, 1.35,
        r"$\beta\ \leftrightarrow\ 1-\gamma\Delta t/m$"
        "     (friction damps velocity = momentum decay)"
        "\n"
        r"$\eta\ \leftrightarrow\ \Delta t/m$"
        "     (gradient kick scale = step size)"
        "\n"
        r"$V'(x)\ \leftrightarrow\ \nabla L(\theta)$"
        "     (force from potential ≈ gradient of loss)"
        "\n\n"
        "Same update shape: keep some old velocity, add a downhill kick, move.\n"
        "Caveat from Step 2: on a real hill the force is only approximately -∇h;\n"
        "GD uses Euclidean -∇L by definition. The similarity is structural, not identity.",
        ha='center', va='center', fontsize=FS_TICK - 2, color='#1c2833',
        linespacing=1.3,
    )
    return fig


run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
])


print("""
Step 1: live rolling-ball animation (optional GIF export).
Step 2: careful map -- gravity is not -grad L; mass+friction is momentum.
Step 3: side-by-side ODE discretization vs heavy-ball / plain GD.
""")
