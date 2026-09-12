
"""
Why the fastest ascending direction is the GRADIENT vector
    grad f = (df/dx, df/dy, ...)
and NOT "pick the axis whose |partial| is largest".

Math in one line:
    directional derivative in unit direction u is  grad f · u
    = |grad f| cos(theta)

This is maximized when u is parallel to grad f (theta = 0),
with max value |grad f|.
Any axis-only move is usually a worse compromise.

Geometry:
    Contour lines are level sets f=const, so moving tangent to a contour
    does not change f. Therefore grad f is perpendicular to the contour:
    that normal direction is exactly the fastest ascent.
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
    run_slide_deck,
)
apply_rcparams(plt)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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
# Surface and analytic gradient
# -------------------------------------------------------
def f(x, y):
    return (
        1.15 * np.exp(-((x - 0.4) ** 2 + (y - 0.6) ** 2) / 0.9)
        + 0.55 * np.exp(-((x + 1.0) ** 2 + (y + 0.3) ** 2) / 1.1)
        + 0.20 * np.sin(1.8 * x) * np.cos(1.3 * y)
    )


def dfdx(x, y):
    return (
        1.15 * np.exp(-((x - 0.4) ** 2 + (y - 0.6) ** 2) / 0.9)
        * (-2.0 * (x - 0.4) / 0.9)
        + 0.55 * np.exp(-((x + 1.0) ** 2 + (y + 0.3) ** 2) / 1.1)
        * (-2.0 * (x + 1.0) / 1.1)
        + 0.20 * 1.8 * np.cos(1.8 * x) * np.cos(1.3 * y)
    )


def dfdy(x, y):
    return (
        1.15 * np.exp(-((x - 0.4) ** 2 + (y - 0.6) ** 2) / 0.9)
        * (-2.0 * (y - 0.6) / 0.9)
        + 0.55 * np.exp(-((x + 1.0) ** 2 + (y + 0.3) ** 2) / 1.1)
        * (-2.0 * (y + 0.3) / 1.1)
        + 0.20 * np.sin(1.8 * x) * (-1.3 * np.sin(1.3 * y))
    )


# Point where BOTH partials are sizable and comparable, so the greedy
# single-axis choice is clearly below the gradient (ratio ~0.73 here).
px, py = 1.0, 1.0
pz = float(f(px, py))
fx = float(dfdx(px, py))
fy = float(dfdy(px, py))
g_norm = float(np.hypot(fx, fy))

# Candidate unit directions
u_grad = np.array([fx, fy]) / g_norm
u_x = np.array([1.0, 0.0]) if fx >= 0 else np.array([-1.0, 0.0])
u_y = np.array([0.0, 1.0]) if fy >= 0 else np.array([0.0, -1.0])
# "greedy axis": pick the axis with larger |partial|
if abs(fx) >= abs(fy):
    u_greedy = u_x.copy()
    greedy_name = r'axis-x (larger $|\partial f/\partial x|$)'
else:
    u_greedy = u_y.copy()
    greedy_name = r'axis-y (larger $|\partial f/\partial y|$)'


def directional_derivative(u):
    return float(fx * u[0] + fy * u[1])


rise_grad = directional_derivative(u_grad)
rise_x = directional_derivative(u_x)
rise_y = directional_derivative(u_y)
rise_greedy = directional_derivative(u_greedy)

xs = np.linspace(-2.0, 2.0, 90)
ys = np.linspace(-2.0, 2.0, 90)
XX, YY = np.meshgrid(xs, ys)
ZZ = f(XX, YY)

# Shared finite-step demo (Steps 3-5)
step = 0.55
candidates = [
    ('gradient', u_grad, 'limegreen'),
    ('axis x', u_x, 'crimson'),
    ('axis y', u_y, 'royalblue'),
    ('greedy axis', u_greedy, 'darkorange'),
]
s_vals = np.linspace(0, step, 80)


# -------------------------------------------------------
# FIGURE 1: Setup -- both partials matter
# -------------------------------------------------------
def build_step_1():
    fig1, axes1 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig1.suptitle(
        'Step 1: At a typical point, BOTH partials are nonzero',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    ax = axes1[0]
    cs = ax.contourf(XX, YY, ZZ, levels=22, cmap='terrain')
    ax.contour(XX, YY, ZZ, levels=22, colors='k', linewidths=0.25, alpha=0.35)
    ax.scatter([px], [py], c='k', s=70, zorder=5)
    # Arrow LENGTH encodes the partial derivative size (same scale for all three).
    # Then green ∇f is literally the vector sum of the two partial arrows.
    vec_scale = 0.90 / g_norm
    ax.arrow(
        px, py, vec_scale * fx, 0.0,
        head_width=0.08, color='crimson', length_includes_head=True, lw=2.2,
        label=rf'$\partial f/\partial x={fx:+.2f}$  (length $\propto |\partial f/\partial x|$)',
    )
    ax.arrow(
        px, py, 0.0, vec_scale * fy,
        head_width=0.08, color='royalblue', length_includes_head=True, lw=2.2,
        label=rf'$\partial f/\partial y={fy:+.2f}$  (length $\propto |\partial f/\partial y|$)',
    )
    ax.arrow(
        px, py, vec_scale * fx, vec_scale * fy,
        head_width=0.09, color='limegreen', length_includes_head=True, lw=2.8,
        label=r'$\nabla f=(\partial f/\partial x,\partial f/\partial y)$',
    )
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('y', fontsize=FS_AXIS)
    ax.set_title(r'Partials as vectors -- length shows $|\partial f/\partial x|$, $|\partial f/\partial y|$',
                 fontsize=FS_AX_TITLE - 2)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(loc='best', fontsize=FS_LEGEND - 1)
    fig1.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)

    ax = axes1[1]
    ax.axis('off')
    ax.set_title('Question', fontsize=FS_AX_TITLE)
    ax.text(
        0.5, 0.55,
        f'At p = ({px:.2f}, {py:.2f}):\n\n'
        rf'  $\partial f/\partial x$ = {fx:+.3f}' + '\n'
        rf'  $\partial f/\partial y$ = {fy:+.3f}' + '\n\n'
        'Red/blue arrow lengths show those sizes.\n'
        'Green is their vector sum = gradient.\n\n'
        f'Greedy axis rule would pick:\n'
        f'  {greedy_name}\n\n'
        f'Rise rate: grad {rise_grad:.3f} vs greedy {rise_greedy:.3f}\n'
        'Greedy is clearly lower -- gradient wins.',
        transform=ax.transAxes, ha='center', va='center', fontsize=FS_AXIS - 1,
        bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
    )

    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    return fig1


# -------------------------------------------------------
# FIGURE 2: Directional derivative vs angle
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig2.suptitle(
        r'Step 2: Rise rate = $\nabla f\cdot u(\theta) = |\nabla f|\cos\theta$',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    angles = np.linspace(-np.pi, np.pi, 400)
    # unit directions parameterized by angle from +x axis
    dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    rises = dirs @ np.array([fx, fy])

    ax2.plot(np.degrees(angles), rises, color='0.2', lw=2.5, label=r'$\nabla f\cdot u(\theta)$')
    ax2.axhline(0, color='k', alpha=0.3)

    # mark special directions
    def mark(u, name, color):
        ang = np.degrees(np.arctan2(u[1], u[0]))
        val = directional_derivative(u)
        ax2.scatter([ang], [val], s=90, color=color, zorder=5, label=rf'{name}: {val:.3f}')
        ax2.axvline(ang, color=color, ls='--', alpha=0.45)

    mark(u_grad, 'gradient', 'limegreen')
    mark(u_x, 'axis x', 'crimson')
    mark(u_y, 'axis y', 'royalblue')
    mark(u_greedy, 'greedy axis', 'darkorange')

    ax2.set_xlabel(r'direction angle $\theta$ (degrees)', fontsize=FS_AXIS)
    ax2.set_ylabel('directional derivative (rise per unit step)', fontsize=FS_AXIS)
    ax2.set_title(
        r'$u(\theta)=(\cos\theta,\sin\theta)$ is the unit vector in direction $\theta$',
        fontsize=FS_AX_TITLE,
    )
    ax2.tick_params(axis='both', labelsize=FS_TICK)
    ax2.legend(fontsize=FS_LEGEND, loc='best')
    ax2.grid(True, alpha=0.3)
    add_note(
        ax2,
        r'$u(\theta)$ = unit vector'
        '\nin the direction of '
        r'$\theta$'
        '.\n\n'
        'Peak ONLY at the gradient.\n\n'
        f'grad rise = {rise_grad:.3f}\n'
        f'greedy rise = {rise_greedy:.3f}\n'
        f'(only {100 * rise_greedy / rise_grad:.0f}% of grad)\n\n'
        'Greedy axis sits clearly\n'
        'lower on this cosine bump.',
        x=0.02, y=0.02, ha='left', va='bottom', facecolor='lightyellow',
    )

    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    return fig2


# -------------------------------------------------------
# FIGURE 3: Same step length -- who climbs higher?
# -------------------------------------------------------
def build_step_3():
    fig3, axes3 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig3.suptitle(
        'Step 3: Same step length, different directions => different height gains',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    ax = axes3[0]
    cs = ax.contourf(XX, YY, ZZ, levels=22, cmap='terrain', alpha=0.85)
    ax.contour(XX, YY, ZZ, levels=22, colors='k', linewidths=0.25, alpha=0.3)
    ax.scatter([px], [py], c='k', s=70, zorder=6)
    gains = []
    for name, u, color in candidates:
        q = np.array([px, py]) + step * u
        z_end = float(f(q[0], q[1]))
        gain = z_end - pz
        gains.append((name, gain, color, q, z_end))
        ax.annotate(
            '', xy=q, xytext=(px, py),
            arrowprops=dict(arrowstyle='->', color=color, lw=2.5),
        )
        ax.scatter([q[0]], [q[1]], c=color, s=40, zorder=5)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('y', fontsize=FS_AXIS)
    ax.set_title(f'Horizontal steps of length {step:g}', fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            Line2D([0], [0], color='limegreen', lw=2.5, label='gradient'),
            Line2D([0], [0], color='crimson', lw=2.5, label='axis x'),
            Line2D([0], [0], color='royalblue', lw=2.5, label='axis y'),
            Line2D([0], [0], color='darkorange', lw=2.5, label='greedy axis'),
        ],
        loc='best', fontsize=FS_LEGEND - 1,
    )
    fig3.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)

    ax = axes3[1]
    names = [g[0] for g in gains]
    vals = [g[1] for g in gains]
    colors = [g[2] for g in gains]
    bars = ax.bar(names, vals, color=colors, edgecolor='k')
    ax.axhline(0, color='k', alpha=0.3)
    ax.set_ylabel(r'actual $\Delta f$ after the step', fontsize=FS_AXIS)
    ax.set_title('Height gained (finite step on the real surface)', fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.02),
                f'{v:.3f}', ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=FS_ANNOT)
    add_note(
        ax,
        'Linear prediction says\n'
        f'grad rises {rise_grad * step:.3f}\n'
        f'greedy rises {rise_greedy * step:.3f}\n'
        'per this step length.\n\n'
        'Finite-step reality\n'
        'matches the ranking.',
        x=0.02, y=0.02, ha='left', va='bottom', facecolor='lightyellow',
    )
    ax.grid(True, axis='y', alpha=0.3)

    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    return fig3


# -------------------------------------------------------
# FIGURE 4: Contour-perpendicular geometry + separable height curves
# -------------------------------------------------------
def build_step_4():
    fig4, axes4 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig4.suptitle(
        r'Step 4: Fastest ascent $\nabla f$ is perpendicular to the contour',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    # --- Left: zoomed contour with tangent + normal (gradient) ---
    ax = axes4[0]
    # zoom window around p
    zoom = 1.1
    ax.contourf(XX, YY, ZZ, levels=30, cmap='terrain', alpha=0.85)
    ax.contour(XX, YY, ZZ, levels=30, colors='k', linewidths=0.35, alpha=0.45)
    # highlight the level curve through p
    ax.contour(XX, YY, ZZ, levels=[pz], colors='k', linewidths=3.0)

    # unit tangent to contour = rotate grad by 90 degrees
    t_contour = np.array([-u_grad[1], u_grad[0]])
    seg = 0.70
    ax.plot(
        [px - seg * t_contour[0], px + seg * t_contour[0]],
        [py - seg * t_contour[1], py + seg * t_contour[1]],
        color='white', lw=4.5, solid_capstyle='round', zorder=4,
    )
    ax.plot(
        [px - seg * t_contour[0], px + seg * t_contour[0]],
        [py - seg * t_contour[1], py + seg * t_contour[1]],
        color='k', lw=2.5, solid_capstyle='round', zorder=5,
        label='contour tangent (level set)',
    )
    ax.annotate(
        '',
        xy=(px + 0.85 * u_grad[0], py + 0.85 * u_grad[1]),
        xytext=(px, py),
        arrowprops=dict(arrowstyle='->', color='limegreen', lw=3.0),
    )
    ax.scatter([px], [py], c='k', s=70, zorder=6)
    # small axis cues
    ax.annotate('', xy=(px + 0.55 * u_x[0], py + 0.55 * u_x[1]), xytext=(px, py),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.8))
    ax.annotate('', xy=(px + 0.55 * u_y[0], py + 0.55 * u_y[1]), xytext=(px, py),
                arrowprops=dict(arrowstyle='->', color='royalblue', lw=1.8))
    ax.text(px + 0.95 * u_grad[0], py + 0.95 * u_grad[1], r'$\nabla f$',
            color='limegreen', fontsize=FS_ANNOT + 2, fontweight='bold')
    ax.text(px + 0.75 * t_contour[0], py + 0.75 * t_contour[1] + 0.05,
            'along contour:\n' + r'$\nabla f\cdot t=0$',
            color='k', fontsize=FS_ANNOT,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlim(px - zoom, px + zoom)
    ax.set_ylim(py - zoom, py + zoom)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('y', fontsize=FS_AXIS)
    ax.set_title('Contour through p  +  normal = gradient', fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(loc='lower left', fontsize=FS_LEGEND)
    add_note(
        ax,
        'Moving along a contour\n'
        'keeps f constant.\n\n'
        'So the steepest climb\n'
        'must be orthogonal to\n'
        'that contour line.\n'
        'That normal is grad f.',
        x=0.98, y=0.98, ha='right', va='top', facecolor='lightyellow',
    )

    # --- Right: height vs path length (no overlapping 3D ribbons) ---
    ax = axes4[1]
    styles = {
        'gradient': ('-', 3.2),
        'axis x': ('--', 2.4),
        'axis y': ('-.', 2.4),
        'greedy axis': (':', 2.8),
    }
    for name, u, color in candidates:
        path = np.array([px, py])[None, :] + s_vals[:, None] * u[None, :]
        z_path = f(path[:, 0], path[:, 1])
        ls, lw = styles.get(name, ('-', 2.5))
        if name == 'greedy axis' and np.allclose(u, u_x):
            label = name + ' (= axis x here)'
            lw = max(lw, 3.4)  # thicker so it remains visible atop axis-x
        elif name == 'greedy axis' and np.allclose(u, u_y):
            label = name + ' (= axis y here)'
            lw = max(lw, 3.4)
        else:
            label = name
        ax.plot(s_vals, z_path, ls=ls, color=color, lw=lw, label=label)
        ax.scatter([s_vals[-1]], [z_path[-1]], color=color, s=55, zorder=5)

    ax.axhline(pz, color='0.4', ls='--', lw=1.0, label='start height')
    ax.set_xlabel('distance traveled in xy-plane', fontsize=FS_AXIS)
    ax.set_ylabel('height f along the path', fontsize=FS_AXIS)
    ax.set_title('Same path length -- separable height curves', fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc='best')
    ax.grid(True, alpha=0.3)
    # ~5 cm down from the top (≈0.45 of this panel's axes height).
    add_note(
        ax,
        'No overlapping 3D tubes.\n'
        'Green rises fastest for\n'
        'the same travel budget.\n\n'
        'Axis / greedy climbs\n'
        'are slower.',
        x=0.98, y=0.02, ha='right', va='bottom',
    )

    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    return fig4


# -------------------------------------------------------
# FIGURE 5: Cauchy-Schwarz / unit-circle picture
# -------------------------------------------------------
def build_step_5():
    fig5, axes5 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig5.suptitle(
        'Step 5: Geometry -- max of grad·u on the unit circle is at u ∥ grad',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    ax = axes5[0]
    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), color='0.5', lw=1.5, label='unit directions u')
    ax.arrow(0, 0, fx / g_norm, fy / g_norm,
             head_width=0.05, color='limegreen', length_includes_head=True, lw=2.5,
             label='unit grad')
    ax.arrow(0, 0, u_x[0], u_x[1],
             head_width=0.05, color='crimson', length_includes_head=True, lw=2,
             label='unit ±x')
    ax.arrow(0, 0, u_y[0], u_y[1],
             head_width=0.05, color='royalblue', length_includes_head=True, lw=2,
             label='unit ±y')
    # Small angle off the x-axis: early mix of both components
    theta_small = np.deg2rad(25)
    u_mix = np.array([np.cos(theta_small), np.sin(theta_small)])
    ax.arrow(0, 0, u_mix[0], u_mix[1],
             head_width=0.05, color='darkorange', length_includes_head=True, lw=2,
             label=r'small $\theta$ off axis')
    # level lines of grad·u = c  are lines perpendicular to grad
    for c in [0.0, 0.5 * g_norm, g_norm]:
        # fx u + fy v = c
        if abs(fy) > 1e-9:
            uu = np.linspace(-1.2, 1.2, 50)
            vv = (c - fx * uu) / fy
            ax.plot(uu, vv, ls=':', color='0.4', alpha=0.7)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$u_x$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$u_y$', fontsize=FS_AXIS)
    ax.set_title('Unit circle of move directions', fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc='upper left')
    ax.grid(True, alpha=0.3)

    ax = axes5[1]
    ax.axis('off')
    ax.set_title('Why not an axis -- or a blend of axes?', fontsize=FS_AX_TITLE)
    ax.text(
        0.5, 0.74,
        'Why not the largest partial alone?\n\n'
        'Axis moves use only ONE coordinate of grad:\n'
        f'  rise_x = |df/dx| = {abs(fx):.3f}\n'
        f'  rise_y = |df/dy| = {abs(fy):.3f}\n'
        'The gradient combines BOTH:\n'
        f'  |grad| = sqrt((df/dx)^2 + (df/dy)^2) = {g_norm:.3f}\n'
        'So |grad| is always >= every |partial|.',
        transform=ax.transAxes, ha='center', va='center',
        fontsize=FS_AXIS + 3,
        linespacing=1.4,
        bbox=dict(boxstyle='round,pad=1.15', facecolor='#e8f4fc',
                  edgecolor='#2874a6', linewidth=2.2, alpha=0.97),
    )
    ax.text(
        0.5, 0.26,
        'Why not an interpolation between the two axes?\n\n'
        'Insight: as the angle grows from zero,\n'
        'the x-component decreases only slowly at first,\n'
        'while the y-component rises quickly --\n'
        'so early on we benefit from both large x and fast growing y.',
        transform=ax.transAxes, ha='center', va='center',
        fontsize=FS_AXIS + 3,
        linespacing=1.4,
        bbox=dict(boxstyle='round,pad=1.15', facecolor='#fdebd0',
                  edgecolor='#d68910', linewidth=2.2, alpha=0.97),
    )

    fig5.tight_layout(rect=[0, 0, 1, 0.94])
    return fig5


run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("FASTEST ASCEND = GRADIENT, NOT BEST AXIS")
print("=" * 60)
print(f"""
At p = ({px:.3f}, {py:.3f}):
  df/dx = {fx:+.4f}
  df/dy = {fy:+.4f}
  |grad| = {g_norm:.4f}

Directional derivatives (rise per UNIT step):
  gradient direction : {rise_grad:.4f}   (= |grad|)
  axis x             : {rise_x:.4f}   (= |df/dx|)
  axis y             : {rise_y:.4f}   (= |df/dy|)
  greedy larger-axis : {rise_greedy:.4f}

Theorem:
  max_{{|u|=1}}  grad dot u  = |grad|
  achieved exactly at u* = grad / |grad|

Contour geometry:
  Along a level set / contour, f is constant, so the directional derivative
  in the contour-tangent direction is 0. Hence grad f is PERPENDICULAR to
  the contour line -- that normal is the unique fastest ascent direction.

Why greedy axis loses:
  |grad| = sqrt((df/dx)^2 + (df/dy)^2)
         >= max(|df/dx|, |df/dy|)
  with equality only if the other partial is zero.

ML link:
  Gradient descent / ascent uses the full grad vector for the same reason:
  it is the uniquely fastest local improvement direction in Euclidean space.
""")
