"""
090_derivative.py

Derivatives and partial derivatives, with browsable 3D surfaces.

Rotate / zoom the 3D figures with the mouse (interactive matplotlib backend).

1D derivative:
    f'(x) = slope of the tangent line

Partial derivatives (2D input):
    df/dx = slope along the x-direction (y fixed)
    df/dy = slope along the y-direction (x fixed)

The gradient
    grad f = (df/dx, df/dy)
points uphill on the surface -- steepest ascent.

Coordinate freedom (important):
    Partials are NOT privileged to the x-axes. If z = A x is any invertible
    linear change of coordinates, the tangent vectors from (df/dx1, df/dx2)
    and from (df/dz1, df/dz2) span the SAME tangent plane in ambient space.
    The hyperplane of tangent vectors is coordinate-independent.

Companion to 080_exponential.py (log derivative identity) and
120_rolling_ball.py (motion follows -grad).
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)
from matplotlib.colors import LinearSegmentedColormap
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
    """Show figure maximized; 3D axes remain mouse-rotatable."""
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
    print('Tip: click-drag to rotate 3D views; scroll to zoom.')
    add_step_footer(fig)
    plt.show()


# -------------------------------------------------------
# Example functions
# -------------------------------------------------------
def f1d(x):
    return 0.15 * x ** 3 - 0.8 * x ** 2 + 0.2 * x + 1.5


def f1d_prime(x):
    return 0.45 * x ** 2 - 1.6 * x + 0.2


# Keep f (and its partials) modest so tangent planes stay compact in the figures.
F2D_AMP = 0.45


def f2d(x, y):
    """A smooth hill/valley surface (amplitude scaled for readable 3D demos)."""
    return F2D_AMP * (
        np.exp(-((x - 0.3) ** 2 + (y - 0.2) ** 2))
        - 0.7 * np.exp(-((x + 0.8) ** 2 + (y + 0.6) ** 2) / 0.8)
        + 0.15 * np.sin(2.2 * x) * np.cos(1.6 * y)
    )


def f2d_dx(x, y):
    return F2D_AMP * (
        np.exp(-((x - 0.3) ** 2 + (y - 0.2) ** 2)) * (-2.0 * (x - 0.3))
        - 0.7 * np.exp(-((x + 0.8) ** 2 + (y + 0.6) ** 2) / 0.8)
        * (-2.0 * (x + 0.8) / 0.8)
        + 0.15 * 2.2 * np.cos(2.2 * x) * np.cos(1.6 * y)
    )


def f2d_dy(x, y):
    return F2D_AMP * (
        np.exp(-((x - 0.3) ** 2 + (y - 0.2) ** 2)) * (-2.0 * (y - 0.2))
        - 0.7 * np.exp(-((x + 0.8) ** 2 + (y + 0.6) ** 2) / 0.8)
        * (-2.0 * (y + 0.6) / 0.8)
        + 0.15 * np.sin(2.2 * x) * (-1.6 * np.sin(1.6 * y))
    )


# Shared surface / line style (used from Step 1 onward).
# Deep blue -> green -> deep brown: rich on surfaces, washed-out on floors.
_SURF_COLORS = [
    (0.02, 0.12, 0.38),  # deep blue (low)
    (0.72, 0.92, 0.55),  # light green (mid)
    (0.42, 0.22, 0.06),  # deep brown (high)
]
_FLOOR_COLORS = [
    (0.35, 0.52, 0.82),  # blue
    (0.78, 0.94, 0.70),  # light green
    (0.72, 0.52, 0.28),  # brown
]
SURF_CMAP = LinearSegmentedColormap.from_list('lecture_surf', _SURF_COLORS, N=256)
FLOOR_CMAP = LinearSegmentedColormap.from_list('lecture_floor', _FLOOR_COLORS, N=256)
SURF_ALPHA = 0.62   # a bit thinner on the 3D graph
FLOOR_ALPHA = 0.62  # a bit thicker on domain-floor contours
C_DX = '#c0392b'     # red  -- x / df/dx
C_DY = '#1e8449'     # green -- y / df/dy
C_CLIMB = '#ff7f0e'  # orange -- +grad climb
C_DESC = '#9467bd'   # purple -- -grad descent
PARTIAL_LW = 1.7     # thin colored stroke
FLOOR_AXIS_LW = 2.0
COAT_EXTRA = 3.0      # white undercoat is lw + COAT_EXTRA (must read clearly in 3D)
# White-coat toggle: on for Step 1 and Step 10; off for Steps 2-9.
COAT_LINES = True


def _coated_line(ax, xs, ys, zs=None, *, color, lw=PARTIAL_LW, ls='-',
                 label=None, zorder=None):
    """Draw a colored stroke; optionally with a white undercoat (COAT_LINES)."""
    kw = dict(ls=ls, solid_capstyle='round')
    if zorder is not None:
        kw['zorder'] = zorder
    if zs is None:
        if COAT_LINES:
            ax.plot(xs, ys, color='white', lw=lw + COAT_EXTRA + 1.2, **kw)
            ax.plot(xs, ys, color='white', lw=lw + COAT_EXTRA, **kw)
        ax.plot(xs, ys, color=color, lw=lw, label=label, **kw)
    else:
        if COAT_LINES:
            ax.plot(xs, ys, zs, color='white', lw=lw + COAT_EXTRA + 1.2, **kw)
            ax.plot(xs, ys, zs, color='white', lw=lw + COAT_EXTRA, **kw)
        ax.plot(xs, ys, zs, color=color, lw=lw, label=label, **kw)


def _coated_arrow3d(ax, origin, direction, *, color, lw=PARTIAL_LW,
                    label=None, arrow_length_ratio=0.18):
    """Shaft + arrow head; white-coated when COAT_LINES is True."""
    o = np.asarray(origin, dtype=float)
    d = np.asarray(direction, dtype=float)
    tip = o + d
    _coated_line(
        ax,
        [o[0], tip[0]], [o[1], tip[1]], [o[2], tip[2]],
        color=color, lw=lw, label=label,
    )
    # Head only: short quiver near the tip
    head = float(arrow_length_ratio) * d
    if np.linalg.norm(head) < 1e-12:
        return
    start = tip - head
    if COAT_LINES:
        ax.quiver(
            start[0], start[1], start[2], head[0], head[1], head[2],
            color='white', linewidth=lw + COAT_EXTRA,
            arrow_length_ratio=0.95,
        )
    ax.quiver(
        start[0], start[1], start[2], head[0], head[1], head[2],
        color=color, linewidth=lw, arrow_length_ratio=0.85,
    )


def _contour_segments(XX, YY, ZZ, levels):
    """Extract contour polylines without drawing (for white-coated re-plot)."""
    fig_tmp, ax_tmp = plt.subplots()
    cs = ax_tmp.contour(XX, YY, ZZ, levels=levels)
    plt.close(fig_tmp)
    segs = []
    for level_segs in cs.allsegs:
        for seg in level_segs:
            if len(seg) >= 2:
                segs.append(seg)
    return segs


def _plot_floor_contour_polylines(ax3d, XX, YY, ZZ, z0, levels, color, lw,
                                  label=None):
    """Draw level-set polylines on a constant-z floor (always white-coated)."""
    labeled = False
    for seg in _contour_segments(XX, YY, ZZ, levels):
        lab = label if (label is not None and not labeled) else None
        if lab is not None:
            labeled = True
        _coated_line(
            ax3d, seg[:, 0], seg[:, 1], np.full(len(seg), z0),
            color=color, lw=lw, label=lab,
        )


def _plot_contour_polylines_2d(ax, XX, YY, ZZ, levels, color, lw, label=None):
    """Draw 2D level-set polylines (always white-coated)."""
    labeled = False
    for seg in _contour_segments(XX, YY, ZZ, levels):
        lab = label if (label is not None and not labeled) else None
        if lab is not None:
            labeled = True
        _coated_line(ax, seg[:, 0], seg[:, 1], color=color, lw=lw, label=lab)


def _draw_floor_xy_axes(ax3d, x0, y0, z0, half=1.1, lw=FLOOR_AXIS_LW, labels=True):
    """Red x-axis and green y-axis on the floor (white-coated)."""
    z = float(z0) + 0.03  # sit slightly above contour fill to avoid z-fighting
    _coated_line(
        ax3d, [x0 - half, x0 + half], [y0, y0], [z, z],
        color=C_DX, lw=lw, zorder=26,
        label=(r'floor $x$-axis' if labels else None),
    )
    _coated_line(
        ax3d, [x0, x0], [y0 - half, y0 + half], [z, z],
        color=C_DY, lw=lw, zorder=26,
        label=(r'floor $y$-axis' if labels else None),
    )
    return z


def _mirror_on_floor(ax3d, xs, ys, z_floor, *, color, lw=PARTIAL_LW, ls='-',
                     label=None):
    """White-coated domain projection of a surface/hyperplane line onto the floor."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.full(xs.shape, float(z_floor))
    _coated_line(ax3d, xs, ys, zs, color=color, lw=lw, ls=ls, label=label)


def _draw_partial_lines_on_tangent(
    ax3d, x0, y0, z0, dfdx, dfdy, half=0.55, lw=PARTIAL_LW, labels=True, ls='-',
    label_x=None, label_y=None, z_floor=None,
):
    """Red ∂f/∂x and green ∂f/∂y lines on the tangent plane (white-coated).

    If z_floor is set, also draw the matching domain lines on the floor.
    """
    t = np.linspace(-half, half, 50)
    # x-partial direction in ambient R^3: (1, 0, dfdx)
    xs_x, ys_x, zs_x = x0 + t, np.full_like(t, y0), z0 + dfdx * t
    # y-partial direction in ambient R^3: (0, 1, dfdy)
    xs_y, ys_y, zs_y = np.full_like(t, x0), y0 + t, z0 + dfdy * t
    if labels:
        if label_x is None:
            label_x = r'$\partial f/\partial x$ line'
        if label_y is None:
            label_y = r'$\partial f/\partial y$ line'
    else:
        label_x = label_y = None
    _coated_line(ax3d, xs_x, ys_x, zs_x, color=C_DX, lw=lw, ls=ls, label=label_x)
    _coated_line(ax3d, xs_y, ys_y, zs_y, color=C_DY, lw=lw, ls=ls, label=label_y)
    if z_floor is not None:
        _mirror_on_floor(ax3d, xs_x, ys_x, z_floor, color=C_DX, lw=lw, ls=ls)
        _mirror_on_floor(ax3d, xs_y, ys_y, z_floor, color=C_DY, lw=lw, ls=ls)


# -------------------------------------------------------
# FIGURE 1: 1D derivative = tangent slope
# -------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig1.suptitle(
    r"Step 1: Derivative $f'(x)$ is the slope of the tangent line",
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

x = np.linspace(-1.0, 5.5, 400)
y = f1d(x)
x0 = 3.2
y0 = f1d(x0)
s0 = f1d_prime(x0)

ax = axes1[0]
_coated_line(ax, x, y, color='royalblue', lw=1.8, label=r'$f(x)$')
xt = np.linspace(x0 - 1.2, x0 + 1.2, 50)
_coated_line(
    ax, xt, y0 + s0 * (xt - x0), color='crimson', lw=1.8,
    label=rf"tangent, slope $f'({x0:g})={s0:.2f}$",
)
ax.scatter([x0], [y0], s=80, c='k', zorder=5)
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('f(x)', fontsize=FS_AXIS)
ax.set_title('Curve + tangent at one point', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
ax.text(
    0.02, 0.98,
    'Steeper tangent => larger |f\'(x)|.\n'
    'Sign tells uphill (+) vs downhill (-).',
    transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes1[1]
_coated_line(ax, x, f1d_prime(x), color='crimson', lw=1.8, label=r"$f'(x)$")
_coated_line(ax, [x.min(), x.max()], [0, 0], color='k', lw=1.0)
_coated_line(ax, [x0, x0], [min(f1d_prime(x)), max(f1d_prime(x))],
             color='0.4', lw=1.0, ls='--')
ax.scatter([x0], [s0], s=80, c='k', zorder=5)
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel(r"$f'(x)$", fontsize=FS_AXIS)
ax.set_title('Derivative as a function of x', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)

fig1.tight_layout(rect=[0, 0, 1, 0.95])
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Browsable 3D surface of z = f(x,y)
# -------------------------------------------------------
COAT_LINES = False  # no white coats for Steps 2-10
fig2 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig2.suptitle(
    r'Step 2: Browse $z=f(x,y)$ in 3D  (drag to rotate)',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax = fig2.add_subplot(111, projection='3d')

xs = np.linspace(-2.0, 2.0, 80)
ys = np.linspace(-2.0, 2.0, 80)
XX, YY = np.meshgrid(xs, ys)
ZZ = f2d(XX, YY)

z_floor2 = float(np.min(ZZ)) - 0.28
surf = ax.plot_surface(
    XX, YY, ZZ, cmap=SURF_CMAP, linewidth=0, antialiased=True, alpha=SURF_ALPHA,
)
# Contour map on the domain floor under the graph (thinner tones)
ax.contourf(
    XX, YY, ZZ, zdir='z', offset=z_floor2,
    levels=20, cmap=FLOOR_CMAP, alpha=FLOOR_ALPHA,
)
_plot_floor_contour_polylines(
    ax, XX, YY, ZZ, z_floor2 + 0.02, levels=20, color='0.35', lw=0.9,
)
_draw_floor_xy_axes(ax, 0.0, 0.0, z_floor2, half=1.6)
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.set_zlabel('z = f(x,y)', fontsize=FS_AXIS)
ax.set_zlim(z_floor2 - 0.05, float(np.max(ZZ)) + 0.1)
ax.tick_params(labelsize=FS_ANNOT)
ax.legend(loc='upper left', fontsize=FS_LEGEND - 1, framealpha=0.92)
fig2.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label='f(x,y)')
ax.text2D(
    0.02, 0.95,
    'Surface = graph $z=f(x,y)$.\n'
    'Floor = contour map of $f$ on the domain $(x,y)$.\n'
    'Red = $x$-axis, green = $y$-axis on the floor.',
    transform=ax.transAxes, fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Partial derivatives as directional slopes
# -------------------------------------------------------
fig3 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig3.suptitle(
    r'Step 3: Partials $\partial f/\partial x$ and $\partial f/\partial y$ at a point',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax = fig3.add_subplot(111, projection='3d')

ax.plot_surface(XX, YY, ZZ, cmap=SURF_CMAP, linewidth=0, alpha=SURF_ALPHA)

# Point of interest -- modest height and modest partials so the tangent
# plane stays compact and the slice/tangent lines read cleanly.
px, py = 0.40, 0.12
pz = f2d(px, py)
fx = f2d_dx(px, py)
fy = f2d_dy(px, py)

ax.scatter([px], [py], [pz], s=80, c='k', depthshade=False)

# Curve along x (y fixed) and its tangent
t = np.linspace(-0.55, 0.55, 40)
x_line = px + t
y_line = np.full_like(t, py)
z_line = f2d(x_line, y_line)
_coated_line(ax, x_line, y_line, z_line, color=C_DX, label='slice y=const')
# Curve along y (x fixed)
y_line2 = py + t
x_line2 = np.full_like(t, px)
z_line2 = f2d(x_line2, y_line2)
_coated_line(ax, x_line2, y_line2, z_line2, color=C_DY, label='slice x=const')

# Contour map + domain x/y axes on the floor
z_floor3 = float(np.min(ZZ)) - 0.28
ax.contourf(
    XX, YY, ZZ, zdir='z', offset=z_floor3,
    levels=20, cmap=FLOOR_CMAP, alpha=FLOOR_ALPHA,
)
_plot_floor_contour_polylines(
    ax, XX, YY, ZZ, z_floor3 + 0.02, levels=20, color='0.35', lw=0.9,
)
_plot_floor_contour_polylines(
    ax, XX, YY, ZZ, z_floor3 + 0.02, levels=[pz], color='#111111', lw=2.0,
)
z_ax3 = _draw_floor_xy_axes(ax, px, py, z_floor3, half=0.75)
# Floor mirrors of the surface slices + tangent partial lines
_mirror_on_floor(ax, x_line, y_line, z_ax3, color=C_DX)
_mirror_on_floor(ax, x_line2, y_line2, z_ax3, color=C_DY)
_draw_partial_lines_on_tangent(
    ax, px, py, pz, fx, fy, half=0.55, ls='--',
    label_x=rf'$\partial f/\partial x={fx:.2f}$',
    label_y=rf'$\partial f/\partial y={fy:.2f}$',
    z_floor=z_ax3,
)
ax.scatter([px], [py], [z_ax3], s=55, c='k', depthshade=False, zorder=5)

ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.set_zlabel('f', fontsize=FS_AXIS)
ax.set_zlim(z_floor3 - 0.05, float(np.max(ZZ)) + 0.1)
ax.tick_params(labelsize=FS_ANNOT)
ax.legend(loc='upper left', fontsize=FS_LEGEND - 1)
ax.text2D(
    0.02, 0.02,
    'Red: move only x  -> slope = df/dx\n'
    'Green: move only y -> slope = df/dy\n'
    'Floor: red $x$-axis + green $y$-axis through $p$.',
    transform=ax.transAxes, fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Tangent plane + partial tangent lines + climb direction
# -------------------------------------------------------
fig4 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig4.suptitle(
    r'Step 4: Tangent plane with partial lines and climbing direction',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax = fig4.add_subplot(111, projection='3d')

ax.plot_surface(XX, YY, ZZ, cmap=SURF_CMAP, linewidth=0, alpha=SURF_ALPHA)
ax.scatter([px], [py], [pz], s=70, c='k', depthshade=False, label='point p')

# tangent plane patch around the point (kept compact with modest slopes)
u = np.linspace(-0.55, 0.55, 14)
v = np.linspace(-0.55, 0.55, 14)
UU, VV = np.meshgrid(u, v)
ZZ_plane = pz + fx * UU + fy * VV
ax.plot_surface(px + UU, py + VV, ZZ_plane, color='orange', alpha=0.45)

# Climbing / descent: same arrow length, opposite directions in the tangent plane
g_norm = np.hypot(fx, fy) + 1e-12
ux, uy = fx / g_norm, fy / g_norm
# 3D direction along the plane: (ux, uy, |grad|) has Euclidean length sqrt(1+|grad|^2)
climb_dir = np.array([ux, uy, g_norm], dtype=float)
climb_dir = climb_dir / (np.linalg.norm(climb_dir) + 1e-12)
arrow_len = 0.85

# Contour map + red/green x/y axes on the floor + climb/descent direction
z_floor4 = float(np.min(ZZ)) - 0.28
ax.contourf(
    XX, YY, ZZ, zdir='z', offset=z_floor4,
    levels=20, cmap=FLOOR_CMAP, alpha=FLOOR_ALPHA,
)
_plot_floor_contour_polylines(
    ax, XX, YY, ZZ, z_floor4 + 0.02, levels=20, color='0.35', lw=0.9,
)
_plot_floor_contour_polylines(
    ax, XX, YY, ZZ, z_floor4 + 0.02, levels=[pz], color='#111111', lw=2.0,
)
z_ax4 = _draw_floor_xy_axes(ax, px, py, z_floor4, half=0.75)
# Partial lines on the hyperplane + matching floor lines
_draw_partial_lines_on_tangent(
    ax, px, py, pz, fx, fy, half=0.55,
    label_x=rf'$\partial f/\partial x$ line (slope={fx:.2f})',
    label_y=rf'$\partial f/\partial y$ line (slope={fy:.2f})',
    z_floor=z_ax4,
)
_coated_arrow3d(
    ax, (px, py, pz), arrow_len * climb_dir,
    color=C_CLIMB, lw=PARTIAL_LW,
    label=rf'climb $\nabla f$  ($|\nabla f|={g_norm:.2f}$)',
)
_coated_arrow3d(
    ax, (px, py, pz), -arrow_len * climb_dir,
    color=C_DESC, lw=PARTIAL_LW,
    label=r'descent $-\nabla f$ (GD)',
)
# Floor mirrors of climb / descent (same colors, white-coated)
floor_seg = np.linspace(0.0, arrow_len, 40)
_mirror_on_floor(
    ax, px + floor_seg * ux, py + floor_seg * uy, z_ax4,
    color=C_CLIMB, label=r'floor: climb $\nabla f$',
)
_mirror_on_floor(
    ax, px - floor_seg * ux, py - floor_seg * uy, z_ax4,
    color=C_DESC, label=r'floor: descent $-\nabla f$',
)
ax.scatter([px], [py], [z_ax4], s=55, c='k', depthshade=False, zorder=5)

ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.set_zlabel('f', fontsize=FS_AXIS)
ax.set_zlim(z_floor4 - 0.05, float(np.max(ZZ)) + 0.1)
ax.tick_params(labelsize=FS_ANNOT)
ax.legend(loc='upper left', fontsize=FS_LEGEND)
ax.text2D(
    0.02, 0.02,
    'Floor: red $x$-axis + green $y$-axis + $\\pm\\nabla f$ direction.\n'
    'Red / green: partial lines in the plane (same colors as the floor axes).\n'
    'Orange / purple: climb $\\nabla f$ and descent $-\\nabla f$, equal length.',
    transform=ax.transAxes, fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Gradient field on the contour map + 3D uphill arrow
# -------------------------------------------------------
fig5 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig5.suptitle(
    r'Step 5: Gradient $\nabla f=(\partial_x f,\partial_y f)$ points uphill',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# Shared point on BOTH panels.
px5, py5 = 1.0, 0.5
pz5 = f2d(px5, py5)
fx5, fy5 = f2d_dx(px5, py5), f2d_dy(px5, py5)

ax0 = fig5.add_subplot(1, 2, 1)
cs = ax0.contourf(XX, YY, ZZ, levels=20, cmap=FLOOR_CMAP, alpha=0.75)
# 2D floor contours: NOT white-coated (filled map + plain contour lines only)
ax0.contour(XX, YY, ZZ, levels=20, colors='0.35', linewidths=0.6, alpha=0.45)
ax0.contour(
    XX, YY, ZZ, levels=[pz5], colors='#1a1a1a', linewidths=2.0, alpha=0.95,
)
# Background quiver field (subsampled) -- plain, not coated
step = 6
GX = f2d_dx(XX, YY)
GY = f2d_dy(XX, YY)
ax0.quiver(
    XX[::step, ::step], YY[::step, ::step],
    GX[::step, ::step], GY[::step, ::step],
    color='0.45', angles='xy', scale_units='xy', scale=8.0, alpha=0.45,
)
# Gradient vector AT p (horizontal): points to increasing f, normal to the contour
g5 = np.hypot(fx5, fy5) + 1e-12
grad_disp = 0.55  # display length in domain units
# White-coated climb arrow at p
_coated_line(
    ax0,
    [px5, px5 + grad_disp * fx5 / g5],
    [py5, py5 + grad_disp * fy5 / g5],
    color=C_CLIMB, lw=PARTIAL_LW, label=r'$\nabla f$ at $p$', zorder=6,
)
# Arrow head (white undercoat only when COAT_LINES is on)
if COAT_LINES:
    ax0.annotate(
        '',
        xy=(px5 + grad_disp * fx5 / g5, py5 + grad_disp * fy5 / g5),
        xytext=(px5, py5),
        arrowprops=dict(arrowstyle='->', color='white', lw=PARTIAL_LW + COAT_EXTRA,
                        mutation_scale=18),
        zorder=6,
    )
ax0.annotate(
    '',
    xy=(px5 + grad_disp * fx5 / g5, py5 + grad_disp * fy5 / g5),
    xytext=(px5, py5),
    arrowprops=dict(arrowstyle='->', color=C_CLIMB, lw=PARTIAL_LW, mutation_scale=14),
    zorder=7,
)
# Tangent to the contour at p: rotate grad by 90 deg -> (-fy, fx)
tx, ty = -fy5 / g5, fx5 / g5
tang_half = 0.70
_coated_line(
    ax0,
    [px5 - tang_half * tx, px5 + tang_half * tx],
    [py5 - tang_half * ty, py5 + tang_half * ty],
    color='navy', lw=PARTIAL_LW,
    label=r'contour tangent at $p$  ($\perp\nabla f$)', zorder=5,
)
# Red x-axis / green y-axis cues through p (same palette as 3D floors)
axis_half5 = 0.55
_coated_line(
    ax0, [px5 - axis_half5, px5 + axis_half5], [py5, py5],
    color=C_DX, lw=PARTIAL_LW, zorder=4, label=r'$x$-axis',
)
_coated_line(
    ax0, [px5, px5], [py5 - axis_half5, py5 + axis_half5],
    color=C_DY, lw=PARTIAL_LW, zorder=4, label=r'$y$-axis',
)
ax0.scatter([px5], [py5], c='k', s=70, zorder=7, label=r'$p=(1.0,\ 0.5)$')
ax0.set_aspect('equal')
ax0.set_xlabel('x', fontsize=FS_AXIS)
ax0.set_ylabel('y', fontsize=FS_AXIS)
ax0.set_title(r'$\nabla f \perp$ contour through $p$', fontsize=FS_AX_TITLE)
ax0.tick_params(axis='both', labelsize=FS_TICK)
ax0.legend(loc='upper right', fontsize=FS_LEGEND - 1, framealpha=0.92)
fig5.colorbar(cs, ax=ax0, fraction=0.046, pad=0.04)

ax1 = fig5.add_subplot(1, 2, 2, projection='3d')
z_floor5 = float(np.min(ZZ)) - 0.28
ax1.plot_surface(XX, YY, ZZ, cmap=SURF_CMAP, alpha=SURF_ALPHA, linewidth=0)
ax1.contourf(
    XX, YY, ZZ, zdir='z', offset=z_floor5,
    levels=20, cmap=FLOOR_CMAP, alpha=FLOOR_ALPHA,
)
_plot_floor_contour_polylines(
    ax1, XX, YY, ZZ, z_floor5 + 0.02, levels=20, color='0.35', lw=0.9,
)
_plot_floor_contour_polylines(
    ax1, XX, YY, ZZ, z_floor5 + 0.02, levels=[pz5], color='#111111', lw=2.0,
)
z_ax5 = _draw_floor_xy_axes(ax1, px5, py5, z_floor5, half=0.70)
# Compact tangent plane + same red/green partial lines (+ floor mirrors)
u5 = np.linspace(-0.55, 0.55, 12)
v5 = np.linspace(-0.55, 0.55, 12)
U5, V5 = np.meshgrid(u5, v5)
ax1.plot_surface(
    px5 + U5, py5 + V5, pz5 + fx5 * U5 + fy5 * V5,
    color='orange', alpha=0.40,
)
_draw_partial_lines_on_tangent(
    ax1, px5, py5, pz5, fx5, fy5, half=0.55, z_floor=z_ax5,
)
ax1.scatter([px5], [py5], [pz5], c='k', s=50)
ax1.scatter([px5], [py5], [z_ax5], c='k', s=40)
# Same point p as the left panel; view chosen so the uphill arrow reads clearly.
ax1.view_init(elev=22, azim=-35)
# uphill arrow in 3D (horizontal gradient direction, rising with slope)
scale = 0.55
gx, gy = fx5, fy5
gz = fx5 * gx + fy5 * gy  # directional rise for that horizontal move
# normalize horizontal part for display length
norm_h = np.hypot(gx, gy) + 1e-8
climb5 = np.array([
    scale * gx / norm_h, scale * gy / norm_h, scale * (gz / norm_h),
])
_coated_arrow3d(
    ax1, (px5, py5, pz5), climb5,
    color=C_CLIMB, lw=PARTIAL_LW, arrow_length_ratio=0.25,
)
# Contour-tangent on the surface (white-coated) + matching floor copy
tx5, ty5 = -fy5 / g5, fx5 / g5
_coated_line(
    ax1,
    [px5 - tang_half * tx5, px5 + tang_half * tx5],
    [py5 - tang_half * ty5, py5 + tang_half * ty5],
    [pz5, pz5],
    color='navy', lw=PARTIAL_LW,
)
_mirror_on_floor(
    ax1,
    [px5, px5 + climb5[0]], [py5, py5 + climb5[1]], z_ax5,
    color=C_CLIMB,
)
_mirror_on_floor(
    ax1,
    [px5 - tang_half * tx5, px5 + tang_half * tx5],
    [py5 - tang_half * ty5, py5 + tang_half * ty5],
    z_ax5, color='navy',
)
ax1.set_xlabel('x', fontsize=FS_AXIS)
ax1.set_ylabel('y', fontsize=FS_AXIS)
ax1.set_zlabel('f', fontsize=FS_AXIS)
ax1.set_zlim(z_floor5 - 0.05, float(np.max(ZZ)) + 0.1)
ax1.set_title('Uphill direction on the surface', fontsize=FS_AX_TITLE)
ax1.tick_params(labelsize=FS_ANNOT)
ax1.text2D(
    0.02, 0.02,
    r'Same $p=(1.0,\ 0.5)$ as the left panel.' + '\n'
    'Red / green / orange lines on the tangent plane\n'
    'have matching white-coated copies on the floor.',
    transform=ax1.transAxes, fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig5.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Contour map painted on the floor of the 3D graph
# -------------------------------------------------------
fig6 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig6.suptitle(
    r'Step 6: Contours live on the domain -- shown on the floor of the 3D graph',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax = fig6.add_subplot(111, projection='3d')
z_floor6 = float(np.min(ZZ)) - 0.28
z_lines6 = z_floor6 + 0.02  # slightly above the fill to avoid z-fighting

# Surface above (thick tones); floor contours stay thin
ax.plot_surface(
    XX, YY, ZZ, cmap=SURF_CMAP, linewidth=0, antialiased=True, alpha=SURF_ALPHA,
)
# Contour fill on the floor
ax.contourf(
    XX, YY, ZZ, zdir='z', offset=z_floor6,
    levels=20, cmap=FLOOR_CMAP, alpha=FLOOR_ALPHA,
)
# Contour polylines as real 3D plots (visible, unlike ax.contour in 3D)
_plot_floor_contour_polylines(
    ax, XX, YY, ZZ, z_lines6, levels=20, color='0.15', lw=1.2,
)
# Same lecture point as Step 5: p = (1.0, 0.5)
_plot_floor_contour_polylines(
    ax, XX, YY, ZZ, z_lines6, levels=[pz5], color='#000000', lw=PARTIAL_LW + 0.4,
    label=r'contour through $p$',
)
z_ax6 = _draw_floor_xy_axes(ax, px5, py5, z_floor6, half=0.90)

g6 = np.hypot(fx5, fy5) + 1e-12
grad_disp = 0.70
ux6, uy6 = fx5 / g6, fy5 / g6
tx, ty = -uy6, ux6
tang_half = 0.75
# Surface: contour tangent + climb (white-coated)
_coated_line(
    ax,
    [px5 - tang_half * tx, px5 + tang_half * tx],
    [py5 - tang_half * ty, py5 + tang_half * ty],
    [pz5, pz5],
    color='navy', lw=PARTIAL_LW,
    label=r'contour tangent on surface',
)
climb6 = np.array([grad_disp * ux6, grad_disp * uy6, grad_disp * g6])
_coated_arrow3d(
    ax, (px5, py5, pz5), climb6,
    color=C_CLIMB, lw=PARTIAL_LW,
    label=r'$\nabla f$ on surface', arrow_length_ratio=0.22,
)
# Matching white-coated lines on the 3D floor
_mirror_on_floor(
    ax,
    [px5 - tang_half * tx, px5 + tang_half * tx],
    [py5 - tang_half * ty, py5 + tang_half * ty],
    z_ax6, color='navy', label=r'contour tangent on floor',
)
_coated_arrow3d(
    ax, (px5, py5, z_ax6), (grad_disp * ux6, grad_disp * uy6, 0.0),
    color=C_CLIMB, lw=PARTIAL_LW,
    label=r'$\nabla f$ on the floor', arrow_length_ratio=0.22,
)
ax.scatter([px5], [py5], [pz5], s=90, c='k', depthshade=False, zorder=5,
           label=r'$p=(1.0,\ 0.5)$ on surface')
ax.scatter([px5], [py5], [z_ax6], s=80, c='k', depthshade=False, zorder=5,
           label=r'$p$ on floor')

ax.view_init(elev=32, azim=-45)
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.set_zlabel('f', fontsize=FS_AXIS)
ax.set_zlim(z_floor6 - 0.05, float(np.max(ZZ)) + 0.1)
ax.tick_params(labelsize=FS_ANNOT)
ax.legend(loc='upper left', fontsize=FS_LEGEND - 1, framealpha=0.92)
ax.text2D(
    0.02, 0.02,
    r'Point $p=(1.0,\ 0.5)$ (same as Step 5).' + '\n'
    'Red = $x$-axis, green = $y$-axis (through $p$).\n'
    r'$\nabla f$ on the floor is $\perp$ to the contour through $p$.',
    transform=ax.transAxes, fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
fig6.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Original f together with its partial-derivative fields
# -------------------------------------------------------
fig7_fields = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig7_fields.suptitle(
    r'Step 7: The surface $f$ and the fields $\partial f/\partial x_1$, $\partial f/\partial x_2$',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

Za = f2d_dx(XX, YY)
Zb = f2d_dy(XX, YY)

z_floor7 = float(np.min(ZZ)) - 0.28
ax0 = fig7_fields.add_subplot(1, 3, 1, projection='3d')
s0 = ax0.plot_surface(XX, YY, ZZ, cmap=SURF_CMAP, linewidth=0, alpha=SURF_ALPHA)
ax0.contourf(XX, YY, ZZ, zdir='z', offset=z_floor7, levels=16, cmap=FLOOR_CMAP, alpha=FLOOR_ALPHA)
z_ax7 = _draw_floor_xy_axes(ax0, px, py, z_floor7, half=0.7, labels=False)
ax0.scatter([px], [py], [pz], s=55, c='k', depthshade=False)
ax0.scatter([px], [py], [z_ax7], s=35, c='k', depthshade=False)
ax0.set_title(r'$f(x_1,x_2)$  (original)', fontsize=FS_AX_TITLE - 2)
ax0.set_xlabel(r'$x_1$', fontsize=FS_AXIS, color=C_DX)
ax0.set_ylabel(r'$x_2$', fontsize=FS_AXIS, color=C_DY)
ax0.set_zlabel(r'$f$', fontsize=FS_AXIS)
ax0.set_zlim(z_floor7 - 0.05, float(np.max(ZZ)) + 0.1)
ax0.tick_params(labelsize=FS_ANNOT)
fig7_fields.colorbar(s0, ax=ax0, shrink=0.45, pad=0.08)

z_floor7a = float(np.min(Za)) - 0.15 * (float(np.max(Za)) - float(np.min(Za)) + 1e-9)
ax1 = fig7_fields.add_subplot(1, 3, 2, projection='3d')
s1 = ax1.plot_surface(XX, YY, Za, cmap=SURF_CMAP, linewidth=0, alpha=SURF_ALPHA)
ax1.contourf(XX, YY, Za, zdir='z', offset=z_floor7a, levels=16, cmap=FLOOR_CMAP, alpha=FLOOR_ALPHA)
_draw_floor_xy_axes(ax1, px, py, z_floor7a, half=0.7, labels=False)
ax1.scatter([px], [py], [fx], s=55, c='k', depthshade=False)
ax1.set_title(r'$\partial f/\partial x_1$  (from $f$)', fontsize=FS_AX_TITLE - 2, color=C_DX)
ax1.set_xlabel(r'$x_1$', fontsize=FS_AXIS, color=C_DX)
ax1.set_ylabel(r'$x_2$', fontsize=FS_AXIS, color=C_DY)
ax1.set_zlabel(r'$\partial f/\partial x_1$', fontsize=FS_AXIS)
ax1.set_zlim(z_floor7a - 0.05, float(np.max(Za)) + 0.05)
ax1.tick_params(labelsize=FS_ANNOT)
fig7_fields.colorbar(s1, ax=ax1, shrink=0.45, pad=0.08)

z_floor7b = float(np.min(Zb)) - 0.15 * (float(np.max(Zb)) - float(np.min(Zb)) + 1e-9)
ax2 = fig7_fields.add_subplot(1, 3, 3, projection='3d')
s2 = ax2.plot_surface(XX, YY, Zb, cmap=SURF_CMAP, linewidth=0, alpha=SURF_ALPHA)
ax2.contourf(XX, YY, Zb, zdir='z', offset=z_floor7b, levels=16, cmap=FLOOR_CMAP, alpha=FLOOR_ALPHA)
_draw_floor_xy_axes(ax2, px, py, z_floor7b, half=0.7, labels=False)
ax2.scatter([px], [py], [fy], s=55, c='k', depthshade=False)
ax2.set_title(r'$\partial f/\partial x_2$  (from $f$)', fontsize=FS_AX_TITLE - 2, color=C_DY)
ax2.set_xlabel(r'$x_1$', fontsize=FS_AXIS, color=C_DX)
ax2.set_ylabel(r'$x_2$', fontsize=FS_AXIS, color=C_DY)
ax2.set_zlabel(r'$\partial f/\partial x_2$', fontsize=FS_AXIS)
ax2.set_zlim(z_floor7b - 0.05, float(np.max(Zb)) + 0.05)
ax2.tick_params(labelsize=FS_ANNOT)
fig7_fields.colorbar(s2, ax=ax2, shrink=0.45, pad=0.08)

fig7_fields.text(
    0.5, 0.02,
    'Left: the function whose slopes we take.  Middle/right: those slopes as surfaces.\n'
    'Floor: red $x_1$-axis, green $x_2$-axis through the same domain point.',
    ha='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
fig7_fields.tight_layout(rect=[0, 0.07, 1, 0.93])
show_maximized(fig7_fields)


# -------------------------------------------------------
# FIGURE 8: ONE tangent hyperplane, TWO bases {v1,v2} and {w1,w2}
#   Domain plane shows both (x1,x2) and (z1,z2) axes.
# -------------------------------------------------------
# Invertible linear change of domain coordinates:
#   z = A x,   x = B z,   B = A^{-1}
# Ambient graph R^3 with horizontal = (x1, x2):
#   v1 = (1, 0, df/dx1),          v2 = (0, 1, df/dx2)
#   w1 = (B e1, df·B e1)=(... , df/dz1)
#   w2 = (B e2, df·B e2)=(... , df/dz2)
# Identity: [w1 w2] = [v1 v2] B  =>  same span = same tangent plane.

A = np.array([
    [1.0, 0.6],
    [-0.4, 1.1],
], dtype=float)
B = np.linalg.inv(A)

# Same lecture point as Steps 5-6: p = (1.0, 0.5)
px8, py8, pz8 = px5, py5, pz5
fx8, fy8 = fx5, fy5

x_pt = np.array([px8, py8], dtype=float)
z_pt = A @ x_pt
grad_x = np.array([fx8, fy8], dtype=float)
grad_z = B.T @ grad_x
gz1, gz2 = float(grad_z[0]), float(grad_z[1])

v1 = np.array([1.0, 0.0, fx8])
v2 = np.array([0.0, 1.0, fy8])
w1 = np.array([B[0, 0], B[1, 0], gz1])
w2 = np.array([B[0, 1], B[1, 1], gz2])

Vmat = np.column_stack([v1, v2])
Wmat = np.column_stack([w1, w2])
recon_err = np.linalg.norm(Wmat - Vmat @ B)
n_plane = np.array([-fx8, -fy8, 1.0])
n_plane = n_plane / (np.linalg.norm(n_plane) + 1e-12)
orth_w = np.array([abs(n_plane @ w1), abs(n_plane @ w2)])

# Color language: red = x1, green = x2/y, blue = z-frame
C_PLANE = '#f5d76e'       # soft gold -- tangent hyperplane ONLY
C_X1 = C_DX               # red   -- x1 / v1
C_X2 = C_DY               # green -- x2 / v2
C_Z = '#2471a3'           # blue  -- z axes / w1,w2
C_FLOOR = '#d5d8dc'       # domain plane sheet
AXIS_LW = PARTIAL_LW
TANGENT_LW = PARTIAL_LW

# Domain-floor height (below the surface for visibility)
z_floor = float(ZZ.min()) - 0.35

# Axis directions in the domain (xy): x-frame and z-frame
ex1 = np.array([1.0, 0.0])
ex2 = np.array([0.0, 1.0])
ez1 = B[:, 0] / (np.linalg.norm(B[:, 0]) + 1e-12)  # increasing z1 in x-space
ez2 = B[:, 1] / (np.linalg.norm(B[:, 1]) + 1e-12)
axis_len = 1.15
arrow_scale = 0.85

u = np.linspace(-0.55, 0.55, 16)
vv = np.linspace(-0.55, 0.55, 16)
UU, VV = np.meshgrid(u, vv)
ZZ_plane = pz8 + fx8 * UU + fy8 * VV

fig8 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig8.suptitle(
    r'Step 8: One tangent hyperplane -- bases $\{v_1,v_2\}$ and $\{w_1,w_2\}$ '
    r'(and both coordinate frames on the domain)',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)
ax = fig8.add_subplot(111, projection='3d')

# Surface f + single tangent plane
ax.plot_surface(XX, YY, ZZ, cmap=SURF_CMAP, linewidth=0, alpha=SURF_ALPHA)
ax.plot_surface(
    px8 + UU, py8 + VV, ZZ_plane,
    color=C_PLANE, alpha=0.55, shade=True,
)
ax.scatter([px8], [py8], [pz8], s=80, c='k', depthshade=False,
           label=r'$p=(1.0,\ 0.5)$')

# Domain plane sheet + visible red/green x1/x2 axes + blue z-frame
floor_r = 1.6
xf = np.linspace(px8 - floor_r, px8 + floor_r, 2)
yf = np.linspace(py8 - floor_r, py8 + floor_r, 2)
XF, YF = np.meshgrid(xf, yf)
ax.plot_surface(XF, YF, np.full_like(XF, z_floor), color=C_FLOOR, alpha=0.22)
z_ax8 = _draw_floor_xy_axes(ax, px8, py8, z_floor, half=axis_len, labels=True)

def _domain_arrow(direction_xy, color, label=None, lw=AXIS_LW):
    d = direction_xy * axis_len
    _coated_arrow3d(
        ax, (px8, py8, z_ax8), (d[0], d[1], 0.0),
        color=color, lw=lw, label=label, arrow_length_ratio=0.12,
    )

# z-frame arrows only (x1/x2 already drawn as floor axes)
_domain_arrow(ez1, C_Z, r'domain $z$-axes')
_domain_arrow(ez2, C_Z, None)
ax.scatter([px8], [py8], [z_ax8], s=55, c='k', depthshade=False)

# Same red/green partial lines on the tangent hyperplane + floor mirrors
_draw_partial_lines_on_tangent(
    ax, px8, py8, pz8, fx8, fy8, half=0.55,
    label_x=r'$v_1$ / $\partial f/\partial x_1$ (red)',
    label_y=r'$v_2$ / $\partial f/\partial x_2$ (green)',
    z_floor=z_ax8,
)
# Tangent vectors ON the hyperplane (ambient R^3) -- white-coated shafts + heads
# Each has a matching white-coated domain arrow already on the floor (x or z).
for vec, color, lab in (
    (v1, C_X1, None),
    (v2, C_X2, None),
    (w1, C_Z, r'$w_1,w_2$ (z-partials)'),
    (w2, C_Z, None),
):
    _coated_arrow3d(
        ax, (px8, py8, pz8), arrow_scale * vec,
        color=color, lw=TANGENT_LW, label=lab, arrow_length_ratio=0.15,
    )
    # Horizontal projection of each tangent arrow onto the floor
    _mirror_on_floor(
        ax,
        [px8, px8 + arrow_scale * vec[0]],
        [py8, py8 + arrow_scale * vec[1]],
        z_ax8, color=color, lw=TANGENT_LW,
    )

ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax.set_zlabel(r'$f$', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_ANNOT)
ax.legend(loc='upper left', fontsize=FS_LEGEND - 2, framealpha=0.92)
ax.text2D(
    0.98, 0.02,
    r'Point $p=(1.0,\ 0.5)$ (same as Steps 5-6).' + '\n'
    'Gold sheet = THE tangent hyperplane (one object).\n'
    'Red = $x_1$ / $v_1$; green = $x_2$ / $v_2$.\n'
    'Blue = $z$ axes / $w_1,w_2$ (z-partials).',
    transform=ax.transAxes, fontsize=FS_ANNOT, ha='right', va='bottom',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.94),
)
fig8.text(
    0.5, 0.01,
    rf'$z=Ax$,  $[w_1\,w_2]=[v_1\,v_2]B$'
    rf'  |  recon err={recon_err:.2e}'
    rf'  |  $|n\cdot w|$={np.round(orth_w, 12).tolist()}',
    ha='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.96),
)
fig8.tight_layout(rect=[0, 0.04, 1, 0.94])
show_maximized(fig8)


# -------------------------------------------------------
# FIGURES 9-10: Asymmetric tilted Gaussian bump (x1 and x2);
#   Step 9 = red x-grid; Step 10 = red z-grid; each lifts to the tangent plane.
# -------------------------------------------------------
# f = exp(-(a x1^2 + b x2^2 + c x1 x2)) with a != b and c != 0.
# Lecture naming: B : x -> z  (z = B x),  g(z) = f(B^{-1} z).
# (Code symbols: A_r = B_lecture, B_r = B_lecture^{-1}.)

A_BUMP, B_BUMP, C_BUMP = 3.2, 0.35, 0.85  # a >> b (strong asymmetry), c != 0 (tilt)
# Keep bump (and g) modest so tangent planes / grids stay compact.
BUMP_AMP = 0.40


def bump(x1, x2, a=A_BUMP, b=B_BUMP, c=C_BUMP):
    q = a * x1 ** 2 + b * x2 ** 2 + c * x1 * x2
    return BUMP_AMP * np.exp(-q)


def bump_dx1(x1, x2, a=A_BUMP, b=B_BUMP, c=C_BUMP):
    return bump(x1, x2, a, b, c) * (-(2.0 * a * x1 + c * x2))


def bump_dx2(x1, x2, a=A_BUMP, b=B_BUMP, c=C_BUMP):
    return bump(x1, x2, a, b, c) * (-(2.0 * b * x2 + c * x1))


# Shear so the same bump looks differently oriented in z-coordinates
A_r = np.array([
    [1.0, 0.95],
    [-0.70, 1.0],
], dtype=float)
B_r = np.linalg.inv(A_r)

# Moderate flank: f (and g) not too high => smaller partials => compact tangent plane.
rx, ry = 0.50, 0.35
rz = float(bump(rx, ry))
r_fx = float(bump_dx1(rx, ry))
r_fy = float(bump_dx2(rx, ry))
r_grad_x = np.array([r_fx, r_fy])
r_grad_z = B_r.T @ r_grad_x
r_gz1, r_gz2 = float(r_grad_z[0]), float(r_grad_z[1])

rv1 = np.array([1.0, 0.0, r_fx])
rv2 = np.array([0.0, 1.0, r_fy])
rw1 = np.array([B_r[0, 0], B_r[1, 0], r_gz1])
rw2 = np.array([B_r[0, 1], B_r[1, 1], r_gz2])

C_PLANE_R = '#f9e79f'  # pale gold tangent plane (under the grid)
C_X_DOMAIN = C_DX          # red 3x3 x-grid on the domain floor
C_X_TANGENT = '#e74c3c'    # same x-grid lifted to the tangent plane
C_Z_DOMAIN = C_Z           # blue 3x3 z-grid on the domain floor
C_Z_TANGENT = '#5dade2'    # same z-grid lifted to the tangent plane
BUMP_SURF_ALPHA = 0.60     # match slightly thinner 3D surfaces
GRID_N = 3
GRID_HALF = 0.40  # compact 3x3 with modest slopes
GRID_LW = 1.4

# x-chart grid
xs_r = np.linspace(-1.7, 1.7, 100)
ys_r = np.linspace(-1.7, 1.7, 100)
RX, RY = np.meshgrid(xs_r, ys_r)
RF = bump(RX, RY)

# z-chart: sample z-box, map to x, evaluate bump; mask outside x-box
r_corners = np.array([[-1.7, -1.7], [-1.7, 1.7], [1.7, -1.7], [1.7, 1.7]])
rz_c = r_corners @ A_r.T
zs1 = np.linspace(rz_c[:, 0].min() + 0.05, rz_c[:, 0].max() - 0.05, 100)
zs2 = np.linspace(rz_c[:, 1].min() + 0.05, rz_c[:, 1].max() - 0.05, 100)
RZ1, RZ2 = np.meshgrid(zs1, zs2)
RX_from_z = B_r[0, 0] * RZ1 + B_r[0, 1] * RZ2
RY_from_z = B_r[1, 0] * RZ1 + B_r[1, 1] * RZ2
RG = bump(RX_from_z, RY_from_z)
RG = np.ma.array(RG, mask=(np.abs(RX_from_z) > 1.65) | (np.abs(RY_from_z) > 1.65))

plane_r = 0.55  # smaller tangent patch when slopes are gentle
pu = np.linspace(-plane_r, plane_r, 16)
pv = np.linspace(-plane_r, plane_r, 16)
PU, PV = np.meshgrid(pu, pv)
RP_x = rz + r_fx * PU + r_fy * PV
z_focus = A_r @ np.array([rx, ry])
RP_z = rz + r_gz1 * PU + r_gz2 * PV
Z_FLOOR = 0.0


def _make_3x3_nodes(origin_xy, n=GRID_N, half=GRID_HALF):
    """Return (n,n,2) array of domain nodes centered at origin_xy."""
    u0, v0 = float(origin_xy[0]), float(origin_xy[1])
    us = np.linspace(u0 - half, u0 + half, n)
    vs = np.linspace(v0 - half, v0 + half, n)
    UU, VV = np.meshgrid(us, vs, indexing='xy')
    return np.stack([UU, VV], axis=-1)  # (n, n, 2)


def _draw_grid_lines(ax, nodes_uv, heights, color, lw=GRID_LW, zorder=20):
    """Plot an n x n grid given node (u,v) positions and heights (always white-coated)."""
    n = nodes_uv.shape[0]
    for i in range(n):
        _coated_line(
            ax, nodes_uv[i, :, 0], nodes_uv[i, :, 1], heights[i, :],
            color=color, lw=lw, zorder=zorder,
        )
    for j in range(n):
        _coated_line(
            ax, nodes_uv[:, j, 0], nodes_uv[:, j, 1], heights[:, j],
            color=color, lw=lw, zorder=zorder,
        )


def _draw_x_and_z_grids(
    ax, panel,
    x_origin, z_origin, f_at_p, dfdx, dfdy, dgdz1, dgdz2,
    A_mat, B_mat, show_x=True, show_z=True, z_floor=Z_FLOOR,
):
    """Show x- and/or z- 3x3 grids on the domain floor and on the tangent plane.

    panel='x': horizontal axes are (x1,x2); tangent graph uses (dfdx, dfdy).
    panel='z': horizontal axes are (z1,z2); tangent graph uses (dgdz1, dgdz2).
    """
    x_nodes = _make_3x3_nodes(x_origin)
    z_nodes = _make_3x3_nodes(z_origin)

    if panel == 'x':
        x_uv = x_nodes
        z_uv = np.einsum('ij,...j->...i', B_mat, z_nodes)  # x = B z
        d_du, d_dv = dfdx, dfdy
        u0, v0 = x_origin
    else:
        z_uv = z_nodes
        x_uv = np.einsum('ij,...j->...i', A_mat, x_nodes)  # z = A x
        d_du, d_dv = dgdz1, dgdz2
        u0, v0 = z_origin

    def lift_uv(uv):
        return f_at_p + d_du * (uv[..., 0] - u0) + d_dv * (uv[..., 1] - v0)

    grids = []
    if show_x:
        grids.append((x_uv, C_X_DOMAIN, C_X_TANGENT))
    if show_z:
        grids.append((z_uv, C_Z_DOMAIN, C_Z_TANGENT))

    for uv, c_dom, c_tan in grids:
        _draw_grid_lines(ax, uv, np.full(uv.shape[:2], z_floor), c_dom, zorder=20)
        _draw_grid_lines(ax, uv, lift_uv(uv), c_tan, zorder=30)


VIEW_ELEV, VIEW_AZIM = 26, -50
FS_STEP8 = FS_ANNOT + 1  # a bit larger for Step-9/10 formula callouts

# Explicit formulas: f in x-coords, and g in z-coords via x = B^{-1} z
b00, b01 = float(B_r[0, 0]), float(B_r[0, 1])
b10, b11 = float(B_r[1, 0]), float(B_r[1, 1])
# Q(x)=a x1^2 + b x2^2 + c x1 x2, x=B^{-1}z  =>  Q = a_z z1^2 + b_z z2^2 + c_z z1 z2
a_z = A_BUMP * b00 ** 2 + B_BUMP * b10 ** 2 + C_BUMP * b00 * b10
b_z = A_BUMP * b01 ** 2 + B_BUMP * b11 ** 2 + C_BUMP * b01 * b11
c_z = (
    2.0 * A_BUMP * b00 * b01
    + 2.0 * B_BUMP * b10 * b11
    + C_BUMP * (b00 * b11 + b01 * b10)
)


def _quad_exp_formula(fn_name, name_args, a, b, c):
    """Build fn(u,v)=exp(-(a u^2 ± b v^2 ± c u v)) with explicit signs."""
    u, v = name_args

    def term(coef, mono, first=False):
        if first:
            return rf'{coef:g}\,{mono}'
        sign = '+' if coef >= 0 else '-'
        return rf'{sign}{abs(coef):g}\,{mono}'

    body = (
        term(a, rf'{u}^2', first=True)
        + term(b, rf'{v}^2')
        + term(c, rf'{u} {v}')
    )
    return rf'${fn_name}({u},{v})=\exp(-({body}))$'


formula_f_x = _quad_exp_formula('f', ('x_1', 'x_2'), A_BUMP, B_BUMP, C_BUMP)
# Lecture naming: B : x -> z  (z = B x).  Then g(z) = f(B^{-1} z).
# Code keeps A_r = B_lecture and B_r = B_lecture^{-1}.
formula_g_z = (
    _quad_exp_formula('g', ('z_1', 'z_2'), a_z, b_z, c_z)
    + r'   from $g(z)=f(B^{-1}z)$'
)
formula_A = (
    rf'$z = B x$ with '
    rf'$B = [[{A_r[0,0]:.2f},\ {A_r[0,1]:.2f}],\ '
    rf'[{A_r[1,0]:.2f},\ {A_r[1,1]:.2f}]]$'
    r'  ($B:x\mapsto z$),   '
    rf'$x = B^{{-1}} z$'
)
formula_g = r'$g(z)=f(B^{-1}z)$  with $B:x\mapsto z$  (same surface; coefficients above)'


def _show_bump_step(step_no, show_x, show_z, grid_note_x, grid_note_z, banner_tag):
    """Shared x/z bump figure; Step 9 shows red x-grid, Step 10 shows red z-grid."""
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    which = 'red $x$-grid' if show_x and not show_z else 'red $z$-grid'
    fig.suptitle(
        rf'Step {step_no}: Asymmetric bump -- {which} on floor $\to$ tangent plane',
        fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
    )
    axes = fig.subplots(1, 2, subplot_kw={'projection': '3d'})

    # Left: x-chart
    ax = axes[0]
    ax.plot_surface(
        RX, RY, RF, cmap=SURF_CMAP, linewidth=0, antialiased=True,
        alpha=BUMP_SURF_ALPHA, shade=True,
    )
    ax.plot_surface(
        rx + PU, ry + PV, RP_x,
        color=C_PLANE_R, alpha=0.35, linewidth=0, shade=True, zorder=10,
    )
    # Same red/green partial lines on the tangent plane + floor mirrors
    _draw_partial_lines_on_tangent(
        ax, rx, ry, rz, r_fx, r_fy, half=0.45, labels=False, z_floor=Z_FLOOR,
    )
    _draw_x_and_z_grids(
        ax, panel='x',
        x_origin=(rx, ry), z_origin=(z_focus[0], z_focus[1]),
        f_at_p=rz, dfdx=r_fx, dfdy=r_fy, dgdz1=r_gz1, dgdz2=r_gz2,
        A_mat=A_r, B_mat=B_r, show_x=show_x, show_z=show_z,
    )
    ax.scatter([rx], [ry], [rz], s=110, c='k', depthshade=False, zorder=40)
    ax.scatter([rx], [ry], [Z_FLOOR], s=40, c='k', depthshade=False, zorder=40)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
    ax.set_zlabel(r'$f$', fontsize=FS_AXIS)
    ax.set_title(r'$x$-chart: asymmetric tilted bump', fontsize=FS_AX_TITLE - 2, pad=-4)
    ax.tick_params(labelsize=FS_ANNOT)
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_zlim(float(np.min(RF)) - 0.05, float(np.max(RF)) + 0.15)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.text2D(
        0.02, 0.02,
        formula_f_x + '\n'
        rf'Depends on both $x_1$ and $x_2$ ($a\neq b$, cross term $c\neq 0$).' + '\n'
        + grid_note_x,
        transform=ax.transAxes, fontsize=FS_STEP8, va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
    )

    # Right: z-chart
    ax = axes[1]
    ax.plot_surface(
        RZ1, RZ2, RG, cmap=SURF_CMAP, linewidth=0, antialiased=True,
        alpha=BUMP_SURF_ALPHA, shade=True,
    )
    ax.plot_surface(
        z_focus[0] + PU, z_focus[1] + PV, RP_z,
        color=C_PLANE_R, alpha=0.35, linewidth=0, shade=True, zorder=10,
    )
    # Same red/green x-partial directions, plotted in (z1,z2,f) coordinates:
    # dx = e_i  =>  dz = A e_i = column i of A,  rise = df/dx_i.
    t_b = np.linspace(-0.45, 0.45, 50)
    for a_col, dfd, color in (
        (A_r[:, 0], r_fx, C_DX),
        (A_r[:, 1], r_fy, C_DY),
    ):
        xs = z_focus[0] + t_b * a_col[0]
        ys = z_focus[1] + t_b * a_col[1]
        zs = rz + t_b * dfd
        _coated_line(ax, xs, ys, zs, color=color, lw=PARTIAL_LW)
        _mirror_on_floor(ax, xs, ys, Z_FLOOR, color=color, lw=PARTIAL_LW)
    _draw_x_and_z_grids(
        ax, panel='z',
        x_origin=(rx, ry), z_origin=(z_focus[0], z_focus[1]),
        f_at_p=rz, dfdx=r_fx, dfdy=r_fy, dgdz1=r_gz1, dgdz2=r_gz2,
        A_mat=A_r, B_mat=B_r, show_x=show_x, show_z=show_z,
    )
    ax.scatter([z_focus[0]], [z_focus[1]], [rz], s=110, c='k', depthshade=False, zorder=40)
    ax.scatter([z_focus[0]], [z_focus[1]], [Z_FLOOR], s=40, c='k', depthshade=False, zorder=40)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlabel(r'$z_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$z_2$', fontsize=FS_AXIS)
    ax.set_zlabel(r'$g(z)=f(B^{-1}z)$', fontsize=FS_AXIS)
    ax.set_title(r'$z$-chart: SAME bump after $z=Bx$', fontsize=FS_AX_TITLE - 2, pad=-4)
    ax.tick_params(labelsize=FS_ANNOT)
    ax.set_xlim(float(zs1.min()), float(zs1.max()))
    ax.set_ylim(float(zs2.min()), float(zs2.max()))
    ax.set_zlim(float(np.nanmin(RG)), float(np.nanmax(RG)) + 0.15)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.text2D(
        0.02, 0.02,
        formula_g_z + '\n'
        r'$g(z)=f(B^{-1}z)$ with $B:x\mapsto z$  ($z=Bx$).' + '\n'
        + grid_note_z,
        transform=ax.transAxes, fontsize=FS_STEP8, va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    fig.text(
        0.5, 0.012,
        formula_f_x + '   |   ' + formula_g + '\n' + formula_A
        + '   |   ' + banner_tag,
        ha='center', va='bottom', fontsize=FS_STEP8 - 1,
        bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.96),
    )
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.10, top=0.92, wspace=0.08)
    show_maximized(fig)


# -------------------------------------------------------
# FIGURE 9: red x-grid only
# -------------------------------------------------------
_show_bump_step(
    step_no=9,
    show_x=True,
    show_z=False,
    grid_note_x=r'Red $3\times 3$: $x$-grid on the floor, lifted to the tangent plane.',
    grid_note_z=r'Red $3\times 3$: same $x$-grid via $z=Bx$, floor + tangent lift.',
    banner_tag=r'red $x$-grid only: domain floor $\to$ tangent plane',
)

# -------------------------------------------------------
# FIGURE 10: red z-grid only (no white coat; all overlay lines red)
# -------------------------------------------------------
C_Z_DOMAIN = C_DX          # red z-grid on the domain floor
C_Z_TANGENT = '#e74c3c'    # red z-grid lifted to the tangent plane
_C_DY_SAVE = C_DY
C_DY = C_DX                 # partial / mirror lines red too for this step
_show_bump_step(
    step_no=10,
    show_x=False,
    show_z=True,
    grid_note_x=r'Red $3\times 3$: $z$-grid via $x=B^{{-1}}z$, floor + tangent lift.',
    grid_note_z=r'Red $3\times 3$: $z$-grid on the floor, lifted to the tangent plane.',
    banner_tag=r'red $z$-grid only: domain floor $\to$ tangent plane',
)
C_DY = _C_DY_SAVE


# -------------------------------------------------------
# FIGURE 11: A wild landscape -- many local peaks and canyons
# -------------------------------------------------------
def rugged_f(x, y, n_features=140, seed=11):
    """Dense mix of peaks/canyons + multi-scale ripples (practical loss-like terrain)."""
    rng = np.random.default_rng(seed)
    z = np.zeros_like(np.asarray(x, dtype=float))
    # Broad undulation
    z = z + 0.22 * np.sin(2.4 * x) * np.cos(2.1 * y)
    z = z + 0.14 * np.sin(4.2 * x + 0.6) * np.sin(3.8 * y - 0.4)
    z = z + 0.10 * np.cos(5.5 * x) * np.cos(4.9 * y)
    # Many localized hills (+) and canyons (-)
    for _ in range(n_features):
        cx = rng.uniform(-1.85, 1.85)
        cy = rng.uniform(-1.85, 1.85)
        # Mix of strong and weak extrema; ~half inverted (canyons)
        amp = rng.uniform(0.18, 0.85) * rng.choice([-1.0, -1.0, 1.0, 1.0, 1.0])
        width = rng.uniform(0.12, 0.42)
        z = z + amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * width ** 2))
    # Fine-scale "practical" roughness at several frequencies
    for k in range(12):
        fx_ = rng.uniform(6.0, 16.0)
        fy_ = rng.uniform(6.0, 16.0)
        amp = rng.uniform(0.02, 0.07)
        phx = rng.uniform(0, 2 * np.pi)
        phy = rng.uniform(0, 2 * np.pi)
        z = z + amp * np.sin(fx_ * x + phx) * np.cos(fy_ * y + phy)
    return z


fig11 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig11.suptitle(
    r'Step 11: Practical reality -- a jungle of local peaks and canyons',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax = fig11.add_subplot(111, projection='3d')

xs11 = np.linspace(-2.0, 2.0, 220)
ys11 = np.linspace(-2.0, 2.0, 220)
XX11, YY11 = np.meshgrid(xs11, ys11)
ZZ11 = rugged_f(XX11, YY11)
z_floor11 = float(np.min(ZZ11)) - 0.40

ax.plot_surface(
    XX11, YY11, ZZ11, cmap=SURF_CMAP, linewidth=0, antialiased=True,
    alpha=0.92, shade=True,
)
ax.contourf(
    XX11, YY11, ZZ11, zdir='z', offset=z_floor11,
    levels=40, cmap=FLOOR_CMAP, alpha=FLOOR_ALPHA,
)
_plot_floor_contour_polylines(
    ax, XX11, YY11, ZZ11, z_floor11 + 0.02, levels=36, color='0.25', lw=0.55,
)
ax.view_init(elev=30, azim=-52)
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.set_zlabel('f', fontsize=FS_AXIS)
ax.set_zlim(z_floor11 - 0.05, float(np.max(ZZ11)) + 0.15)
ax.tick_params(labelsize=FS_ANNOT)
ax.text2D(
    0.02, 0.02,
    'Like a real training loss: peaks and canyons everywhere.\n'
    'A local gradient only sees its neighborhood --\n'
    'so steepest ascent / descent can stop far from the global best.\n'
    'This is the everyday reality of optimizing practical models.',
    transform=ax.transAxes, fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
fig11.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig11)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("DERIVATIVES & PARTIAL DERIVATIVES")
print("=" * 60)
print(f"""
1D:
  f'(x) = slope of the tangent to y=f(x).

2D surface z=f(x,y):
  df/dx = slope along x with y held fixed
  df/dy = slope along y with x held fixed
  Together they define the tangent plane and the gradient.

At demo point (x,y)=({px:g},{py:g}):
  f      = {pz:.4f}
  df/dx  = {fx:.4f}
  df/dy  = {fy:.4f}
  |grad| = {np.hypot(fx, fy):.4f}

Step 6:
  Contour map painted on the floor of the 3D graph -- domain (x,y) under z=f(x,y).
  Gradient on the floor is perpendicular to the contour through p.

Step 7:
  Shows f itself next to the partial fields df/dx1 and df/dx2
  (same domain point marked on all three).

Coordinate freedom (Step 8):
  Partials do NOT privilege the x-axes.
  For invertible z = A x  (here A = {A.tolist()}):
    g(z) = f(B z),  B = A^{{-1}}
    grad_z g = B^T grad_x f
      grad_x f = {np.round(grad_x, 4).tolist()}
      grad_z g = {np.round(grad_z, 4).tolist()}

  ONE tangent hyperplane in ambient (x1, x2, f)-space, TWO bases:
    v1 = (1, 0, df/dx1),  v2 = (0, 1, df/dx2)     [x-frame]
    w1 = (B e1, df/dz1),  w2 = (B e2, df/dz2)     [z-frame]
  Identity: [w1 w2] = [v1 v2] B
    reconstruction error = {recon_err:.2e}
    |n·w1|, |n·w2| (same plane normal) = {np.round(orth_w, 12).tolist()}

  Domain plane shows both (x1,x2) and (z1,z2) axes at p.
  Colors: red = x1; green = x2/y; blue = z-frame; gold = plane.

Steps 9-10 (asymmetric bump demo):
  f = exp(-(a x1^2 + b x2^2 + c x1 x2)) with a={A_BUMP:g}, b={B_BUMP:g}, c={C_BUMP:g}.
  Depends on BOTH x1 and x2, asymmetrically (a != b) with a tilt (c != 0).
  Under z = B x (B:x->z), g(z)=f(B^{{-1}}z): the linear warp is visible in 3D.
  Step 9: thin red 3x3 x-grid on the floor, lifted to the tangent plane.
  Step 10: thin red 3x3 z-grid on the floor, lifted to the tangent plane (no white coat).
  Domain axes throughout: red = x, green = y.

Step 11:
  A dense jungle of local peaks and canyons (practical-loss style terrain).
  Local gradients only see their neighborhood -- global optimization is hard.


Browsable figures:
  Drag with the mouse to rotate any 3D view; scroll to zoom.
  Use this to inspect hills, valleys, and slopes from multiple angles.

ML link:
  Training steps follow -grad of a loss L(w).
  Partials are the coordinates of that gradient -- and those coordinates
  can be rewritten in any linear basis without changing the tangent plane.
""")
