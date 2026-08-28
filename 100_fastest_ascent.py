
"""
100_fastest_ascend.py

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

Companion to 090_derivative.py.
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


# Point chosen so BOTH partials are nonzero and comparable-but-not-equal
px, py = -0.35, 0.25
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
    greedy_name = 'axis-x (larger |df/dx|)'
else:
    u_greedy = u_y.copy()
    greedy_name = 'axis-y (larger |df/dy|)'


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


# -------------------------------------------------------
# FIGURE 1: Setup -- both partials matter
# -------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig1.suptitle(
    'Step 1: At a typical point, BOTH partials are nonzero',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes1[0]
cs = ax.contourf(XX, YY, ZZ, levels=22, cmap='terrain')
ax.contour(XX, YY, ZZ, levels=22, colors='k', linewidths=0.25, alpha=0.35)
ax.scatter([px], [py], c='k', s=70, zorder=5)
# partial axis arrows (unit length in plane)
ax.arrow(px, py, 0.55 * u_x[0], 0.55 * u_x[1],
         head_width=0.08, color='crimson', length_includes_head=True, lw=2,
         label=rf'±x  (df/dx={fx:.2f})')
ax.arrow(px, py, 0.55 * u_y[0], 0.55 * u_y[1],
         head_width=0.08, color='royalblue', length_includes_head=True, lw=2,
         label=rf'±y  (df/dy={fy:.2f})')
ax.arrow(px, py, 0.75 * u_grad[0], 0.75 * u_grad[1],
         head_width=0.08, color='limegreen', length_includes_head=True, lw=2.5,
         label='gradient direction')
ax.set_aspect('equal')
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.set_title('Contour map at point p', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(loc='best', fontsize=FS_LEGEND)
fig1.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)

ax = axes1[1]
ax.axis('off')
ax.set_title('Question', fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.55,
    f'At p = ({px:.2f}, {py:.2f}):\n\n'
    f'  df/dx = {fx:+.3f}\n'
    f'  df/dy = {fy:+.3f}\n\n'
    f'Greedy axis rule would pick:\n'
    f'  {greedy_name}\n\n'
    'Is that the fastest climb?\n'
    'No -- the green gradient wins.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_AXIS - 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

fig1.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Directional derivative vs angle
# -------------------------------------------------------
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
    'The curve peaks ONLY at\n'
    'the gradient angle.\n\n'
    'Any single-axis choice\n'
    'is a lower point on\n'
    'this cosine bump.',
    y=0.98, va='top', facecolor='lightyellow',
)

fig2.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Same step length -- who climbs higher?
# -------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig3.suptitle(
    'Step 3: Same step length, different directions => different height gains',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

step = 0.55
candidates = [
    ('gradient', u_grad, 'limegreen'),
    ('axis x', u_x, 'crimson'),
    ('axis y', u_y, 'royalblue'),
    ('greedy axis', u_greedy, 'darkorange'),
]

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
    y=0.98, va='top', facecolor='lightyellow',
)
ax.grid(True, axis='y', alpha=0.3)

fig3.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Contour-perpendicular geometry + separable height curves
# -------------------------------------------------------
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
    y=0.98, va='top', facecolor='lightyellow',
)

# --- Right: height vs path length (no overlapping 3D ribbons) ---
ax = axes4[1]
s_vals = np.linspace(0, step, 80)
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
add_note(
    ax,
    'No overlapping 3D tubes.\n'
    'Green rises fastest for\n'
    'the same travel budget.\n\n'
    'Axis / greedy climbs\n'
    'are slower.',
    y=0.98, va='top',
)

fig4.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig4)

# Optional browsable 3D: ONE panel per direction (no overlap)
fig4b = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig4b.suptitle(
    'Step 4b: Same climbs on the 3D surface -- one path per panel',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
# drop duplicate greedy if it equals an axis, keep unique directions for panels
panel_cands = []
seen = []
for name, u, color in candidates:
    key = tuple(np.round(u, 6))
    if key in seen and name == 'greedy axis':
        continue
    seen.append(key)
    panel_cands.append((name, u, color))

z_floor4b = float(np.min(ZZ)) - 0.18
axes_b = []
for i, (name, u, color) in enumerate(panel_cands):
    ax = fig4b.add_subplot(1, len(panel_cands), i + 1, projection='3d')
    axes_b.append(ax)
    ax.plot_surface(XX, YY, ZZ, cmap='terrain', linewidth=0, alpha=0.22)
    path = np.array([px, py])[None, :] + s_vals[:, None] * u[None, :]
    z_path = f(path[:, 0], path[:, 1])
    # Climb on the surface
    ax.plot(path[:, 0], path[:, 1], z_path, color=color, lw=3.5)
    # Matching domain path on the floor (same style as 090 Step 3/4)
    ax.plot(
        path[:, 0], path[:, 1], np.full_like(z_path, z_floor4b),
        color=color, lw=2.4, alpha=0.95,
    )
    ax.scatter([px], [py], [pz], c='k', s=35, depthshade=False, zorder=5)
    ax.scatter([px], [py], [z_floor4b], c='k', s=32, depthshade=False, zorder=5)
    ax.scatter([path[-1, 0]], [path[-1, 1]], [z_path[-1]],
               c=color, s=50, depthshade=False, zorder=5)
    ax.scatter([path[-1, 0]], [path[-1, 1]], [z_floor4b],
               c=color, s=38, depthshade=False, zorder=5)
    # Drop lines from surface start / end down to the floor
    ax.plot([px, px], [py, py], [z_floor4b, pz], color='0.35', lw=1.0, ls=':')
    ax.plot(
        [path[-1, 0], path[-1, 0]],
        [path[-1, 1], path[-1, 1]],
        [z_floor4b, z_path[-1]],
        color='0.45', ls=':', lw=0.9, alpha=0.85,
    )
    # Vertical stem to emphasize final height (surface start -> end)
    ax.plot(
        [path[-1, 0], path[-1, 0]],
        [path[-1, 1], path[-1, 1]],
        [pz, z_path[-1]],
        color=color, ls='--', lw=1.5,
    )
    ax.set_title(f'{name}\n' + rf'$\Delta f={z_path[-1] - pz:.3f}$',
                 fontsize=FS_AX_TITLE - 4, color=color)
    ax.set_xlabel('x', fontsize=FS_AXIS - 2)
    ax.set_ylabel('y', fontsize=FS_AXIS - 2)
    ax.set_zlabel('f', fontsize=FS_AXIS - 2)
    ax.set_zlim(z_floor4b - 0.05, float(np.max(ZZ)) + 0.08)
    ax.view_init(elev=28, azim=-55)
    ax.tick_params(labelsize=FS_ANNOT - 2)

fig4b.text(
    0.5, 0.02,
    'Solid floor lines = domain paths in (x,y).  Surface curves = the same climbs in height.\n'
    'Dotted drops link endpoints to the floor; dashed stems compare endpoint heights.',
    ha='center', fontsize=FS_ANNOT,
)
show_maximized(fig4b)


# -------------------------------------------------------
# FIGURE 5: Cauchy-Schwarz / unit-circle picture
# -------------------------------------------------------
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
ax.legend(fontsize=FS_LEGEND, loc='lower left')
ax.grid(True, alpha=0.3)

ax = axes5[1]
ax.axis('off')
ax.set_title('Why not the largest partial alone?', fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.55,
    'Axis move uses only ONE coordinate of grad:\n'
    f'  rise_x = |df/dx| = {abs(fx):.3f}\n'
    f'  rise_y = |df/dy| = {abs(fy):.3f}\n\n'
    'Gradient combines BOTH coordinates:\n'
    f'  |grad| = sqrt((df/dx)^2 + (df/dy)^2)\n'
    f'         = {g_norm:.3f}\n\n'
    'Because squares add under the sqrt,\n'
    '|grad| is ALWAYS >= every |partial|.\n'
    'Equality only if the other partials are 0.\n\n'
    'So unless the surface depends on a single\n'
    'variable, pure-axis climbing wastes slope.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_AXIS - 2,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

fig5.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Numerical inequality table as bars for many random points
# -------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig6.suptitle(
    'Step 6: At random points, |grad| beats the best single-axis partial',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

rng = np.random.default_rng(1)
pts = rng.uniform(-1.5, 1.5, size=(40, 2))
best_axis = []
grad_mags = []
for qx, qy in pts:
    gx, gy = float(dfdx(qx, qy)), float(dfdy(qx, qy))
    best_axis.append(max(abs(gx), abs(gy)))
    grad_mags.append(np.hypot(gx, gy))
best_axis = np.array(best_axis)
grad_mags = np.array(grad_mags)
advantage = grad_mags - best_axis

order = np.argsort(advantage)
ax6.bar(np.arange(len(pts)), advantage[order], color='seagreen', edgecolor='k', alpha=0.85)
ax6.axhline(0, color='k', lw=1)
ax6.set_xlabel('random sample points (sorted)', fontsize=FS_AXIS)
ax6.set_ylabel(r'$|\nabla f| - \max(|\partial_x f|,|\partial_y f|)$', fontsize=FS_AXIS)
ax6.set_title('Advantage of full gradient over greedy axis', fontsize=FS_AX_TITLE)
ax6.tick_params(axis='both', labelsize=FS_TICK)
ax6.grid(True, axis='y', alpha=0.3)
add_note(
    ax6,
    'Advantage is >= 0 always.\n'
    'It is 0 only when one\n'
    'partial vanishes.\n\n'
    f'Mean advantage here:\n{advantage.mean():.3f}',
    y=0.98, va='top', facecolor='lightyellow',
)

fig6.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig6)


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
