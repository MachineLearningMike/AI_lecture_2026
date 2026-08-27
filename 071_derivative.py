"""
071_derivative.py

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

Companion to 070_exponential.py (log derivative identity) and
080_rolling_ball.py (motion follows -grad).
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)
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
    plt.show()


# -------------------------------------------------------
# Example functions
# -------------------------------------------------------
def f1d(x):
    return 0.15 * x ** 3 - 0.8 * x ** 2 + 0.2 * x + 1.5


def f1d_prime(x):
    return 0.45 * x ** 2 - 1.6 * x + 0.2


def f2d(x, y):
    """A smooth hill/valley surface."""
    return (
        np.exp(-((x - 0.3) ** 2 + (y - 0.2) ** 2))
        - 0.7 * np.exp(-((x + 0.8) ** 2 + (y + 0.6) ** 2) / 0.8)
        + 0.15 * np.sin(2.2 * x) * np.cos(1.6 * y)
    )


def f2d_dx(x, y):
    return (
        np.exp(-((x - 0.3) ** 2 + (y - 0.2) ** 2)) * (-2.0 * (x - 0.3))
        - 0.7 * np.exp(-((x + 0.8) ** 2 + (y + 0.6) ** 2) / 0.8)
        * (-2.0 * (x + 0.8) / 0.8)
        + 0.15 * 2.2 * np.cos(2.2 * x) * np.cos(1.6 * y)
    )


def f2d_dy(x, y):
    return (
        np.exp(-((x - 0.3) ** 2 + (y - 0.2) ** 2)) * (-2.0 * (y - 0.2))
        - 0.7 * np.exp(-((x + 0.8) ** 2 + (y + 0.6) ** 2) / 0.8)
        * (-2.0 * (y + 0.6) / 0.8)
        + 0.15 * np.sin(2.2 * x) * (-1.6 * np.sin(1.6 * y))
    )


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
ax.plot(x, y, color='royalblue', lw=2.8, label=r'$f(x)$')
# tangent
xt = np.linspace(x0 - 1.2, x0 + 1.2, 50)
ax.plot(xt, y0 + s0 * (xt - x0), color='crimson', lw=2.5,
        label=rf"tangent, slope $f'({x0:g})={s0:.2f}$")
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
ax.plot(x, f1d_prime(x), color='crimson', lw=2.8, label=r"$f'(x)$")
ax.axhline(0, color='k', alpha=0.35)
ax.axvline(x0, color='0.4', ls='--', lw=1.2)
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

surf = ax.plot_surface(
    XX, YY, ZZ, cmap='terrain', linewidth=0, antialiased=True, alpha=0.9,
)
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.set_zlabel('z = f(x,y)', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_ANNOT)
fig2.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label='f(x,y)')
ax.text2D(
    0.02, 0.95,
    'This is the graph of a scalar field.\n'
    'Partials are slopes ON this surface.',
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

ax.plot_surface(XX, YY, ZZ, cmap='coolwarm', linewidth=0, alpha=0.35)

# Point of interest
px, py = 0.55, -0.35
pz = f2d(px, py)
fx = f2d_dx(px, py)
fy = f2d_dy(px, py)

ax.scatter([px], [py], [pz], s=80, c='k', depthshade=False)

# Curve along x (y fixed) and its tangent
t = np.linspace(-0.9, 0.9, 40)
x_line = px + t
y_line = np.full_like(t, py)
z_line = f2d(x_line, y_line)
ax.plot(x_line, y_line, z_line, color='crimson', lw=2.5, label='slice y=const')
ax.plot(px + t, py + 0 * t, pz + fx * t, color='crimson', ls='--', lw=2.2,
        label=rf'$\partial f/\partial x={fx:.2f}$')

# Curve along y (x fixed) and its tangent
y_line2 = py + t
x_line2 = np.full_like(t, px)
z_line2 = f2d(x_line2, y_line2)
ax.plot(x_line2, y_line2, z_line2, color='royalblue', lw=2.5, label='slice x=const')
ax.plot(px + 0 * t, py + t, pz + fy * t, color='royalblue', ls='--', lw=2.2,
        label=rf'$\partial f/\partial y={fy:.2f}$')

ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.set_zlabel('f', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_ANNOT)
ax.legend(loc='upper left', fontsize=FS_LEGEND)
ax.text2D(
    0.02, 0.02,
    'Red: move only x  -> slope = df/dx\n'
    'Blue: move only y -> slope = df/dy\n'
    'Rotate the view to see both hinges.',
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

ax.plot_surface(XX, YY, ZZ, cmap='viridis', linewidth=0, alpha=0.30)
ax.scatter([px], [py], [pz], s=70, c='k', depthshade=False, label='point p')

# tangent plane patch around the point
u = np.linspace(-0.85, 0.85, 14)
v = np.linspace(-0.85, 0.85, 14)
UU, VV = np.meshgrid(u, v)
ZZ_plane = pz + fx * UU + fy * VV
ax.plot_surface(px + UU, py + VV, ZZ_plane, color='orange', alpha=0.45)

# Partial-derivative tangent lines LYING IN the tangent plane
t_line = np.linspace(-0.85, 0.85, 50)
# along +x (y fixed): direction (1, 0, df/dx)
ax.plot(
    px + t_line, py + 0 * t_line, pz + fx * t_line,
    color='crimson', lw=3.0,
    label=rf'$\partial f/\partial x$ line (slope={fx:.2f})',
)
# along +y (x fixed): direction (0, 1, df/dy)
ax.plot(
    px + 0 * t_line, py + t_line, pz + fy * t_line,
    color='royalblue', lw=3.0,
    label=rf'$\partial f/\partial y$ line (slope={fy:.2f})',
)

# Climbing direction = steepest ascent in the tangent plane
# horizontal unit vector of grad, vertical rise = |grad|
g_norm = np.hypot(fx, fy) + 1e-12
ux, uy = fx / g_norm, fy / g_norm
# 3D unit-ish climb vector along plane: (ux, uy, |grad|)
climb_len = 1.1
ax.quiver(
    px, py, pz,
    climb_len * ux,
    climb_len * uy,
    climb_len * g_norm,
    color='limegreen', linewidth=3, arrow_length_ratio=0.18,
    label=rf'climb $\nabla f$  ($|\nabla f|={g_norm:.2f}$)',
)
# also draw the climb path segment in the plane for clarity
s = np.linspace(0.0, climb_len, 40)
ax.plot(
    px + s * ux, py + s * uy, pz + s * g_norm,
    color='limegreen', lw=2.0, ls='--', alpha=0.9,
)

# Descent / GD direction (opposite)
ax.quiver(
    px, py, pz,
    -0.7 * climb_len * ux,
    -0.7 * climb_len * uy,
    -0.7 * climb_len * g_norm,
    color='purple', linewidth=2.2, arrow_length_ratio=0.2,
    label=r'descent $-\nabla f$ (GD)',
)

ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.set_zlabel('f', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_ANNOT)
ax.legend(loc='upper left', fontsize=FS_LEGEND)
ax.text2D(
    0.02, 0.02,
    'Red / blue: partial tangent lines in the plane\n'
    '(move only x, or only y).\n'
    'Green: steepest climb on the plane (grad).\n'
    'Purple: training / GD goes the opposite way.',
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

ax0 = fig5.add_subplot(1, 2, 1)
cs = ax0.contourf(XX, YY, ZZ, levels=20, cmap='terrain', alpha=0.9)
ax0.contour(XX, YY, ZZ, levels=20, colors='k', linewidths=0.3, alpha=0.4)
# quiver subsample
step = 6
GX = f2d_dx(XX, YY)
GY = f2d_dy(XX, YY)
ax0.quiver(
    XX[::step, ::step], YY[::step, ::step],
    GX[::step, ::step], GY[::step, ::step],
    color='crimson', angles='xy', scale_units='xy', scale=8.0,
)
ax0.scatter([px], [py], c='k', s=60, zorder=5)
ax0.set_aspect('equal')
ax0.set_xlabel('x', fontsize=FS_AXIS)
ax0.set_ylabel('y', fontsize=FS_AXIS)
ax0.set_title('Contour map + gradient arrows', fontsize=FS_AX_TITLE)
ax0.tick_params(axis='both', labelsize=FS_TICK)
fig5.colorbar(cs, ax=ax0, fraction=0.046, pad=0.04)

ax1 = fig5.add_subplot(1, 2, 2, projection='3d')
ax1.plot_surface(XX, YY, ZZ, cmap='terrain', alpha=0.45, linewidth=0)
ax1.scatter([px], [py], [pz], c='k', s=50)
# uphill arrow in 3D (horizontal gradient direction, rising with slope)
scale = 0.55
gx, gy = fx, fy
gz = fx * gx + fy * gy  # directional rise for that horizontal move
# normalize horizontal part for display length
norm_h = np.hypot(gx, gy) + 1e-8
ax1.quiver(
    px, py, pz,
    scale * gx / norm_h, scale * gy / norm_h, scale * (gz / norm_h),
    color='crimson', linewidth=2, arrow_length_ratio=0.25,
)
ax1.set_xlabel('x', fontsize=FS_AXIS)
ax1.set_ylabel('y', fontsize=FS_AXIS)
ax1.set_zlabel('f', fontsize=FS_AXIS)
ax1.set_title('Uphill direction on the surface', fontsize=FS_AX_TITLE)
ax1.tick_params(labelsize=FS_ANNOT)
ax1.text2D(
    0.02, 0.02,
    'GD descends via -grad f\n'
    '(see 080_rolling_ball.py).',
    transform=ax1.transAxes, fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig5.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Original f together with its partial-derivative fields
# -------------------------------------------------------
fig6 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig6.suptitle(
    r'Step 6: The surface $f$ and the fields $\partial f/\partial x_1$, $\partial f/\partial x_2$',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

Za = f2d_dx(XX, YY)
Zb = f2d_dy(XX, YY)

ax0 = fig6.add_subplot(1, 3, 1, projection='3d')
s0 = ax0.plot_surface(XX, YY, ZZ, cmap='terrain', linewidth=0, alpha=0.92)
ax0.scatter([px], [py], [pz], s=55, c='k', depthshade=False)
ax0.set_title(r'$f(x_1,x_2)$  (original)', fontsize=FS_AX_TITLE - 2)
ax0.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax0.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax0.set_zlabel(r'$f$', fontsize=FS_AXIS)
ax0.tick_params(labelsize=FS_ANNOT)
fig6.colorbar(s0, ax=ax0, shrink=0.45, pad=0.08)

ax1 = fig6.add_subplot(1, 3, 2, projection='3d')
s1 = ax1.plot_surface(XX, YY, Za, cmap='RdBu_r', linewidth=0, alpha=0.92)
ax1.scatter([px], [py], [fx], s=55, c='k', depthshade=False)
ax1.set_title(r'$\partial f/\partial x_1$  (from $f$)', fontsize=FS_AX_TITLE - 2)
ax1.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax1.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax1.set_zlabel(r'$\partial f/\partial x_1$', fontsize=FS_AXIS)
ax1.tick_params(labelsize=FS_ANNOT)
fig6.colorbar(s1, ax=ax1, shrink=0.45, pad=0.08)

ax2 = fig6.add_subplot(1, 3, 3, projection='3d')
s2 = ax2.plot_surface(XX, YY, Zb, cmap='RdBu_r', linewidth=0, alpha=0.92)
ax2.scatter([px], [py], [fy], s=55, c='k', depthshade=False)
ax2.set_title(r'$\partial f/\partial x_2$  (from $f$)', fontsize=FS_AX_TITLE - 2)
ax2.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax2.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax2.set_zlabel(r'$\partial f/\partial x_2$', fontsize=FS_AXIS)
ax2.tick_params(labelsize=FS_ANNOT)
fig6.colorbar(s2, ax=ax2, shrink=0.45, pad=0.08)

fig6.text(
    0.5, 0.02,
    'Left: the function whose slopes we take.  Middle/right: those slopes as surfaces.\n'
    'Black dots mark the same domain point. Near-zero partial => locally flat in that axis.',
    ha='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
fig6.tight_layout(rect=[0, 0.07, 1, 0.93])
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: ONE tangent hyperplane, TWO bases {v1,v2} and {w1,w2}
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

x_pt = np.array([px, py], dtype=float)
z_pt = A @ x_pt
grad_x = np.array([fx, fy], dtype=float)
grad_z = B.T @ grad_x
gz1, gz2 = float(grad_z[0]), float(grad_z[1])

v1 = np.array([1.0, 0.0, fx])
v2 = np.array([0.0, 1.0, fy])
w1 = np.array([B[0, 0], B[1, 0], gz1])
w2 = np.array([B[0, 1], B[1, 1], gz2])

Vmat = np.column_stack([v1, v2])
Wmat = np.column_stack([w1, w2])
recon_err = np.linalg.norm(Wmat - Vmat @ B)
n_plane = np.array([-fx, -fy, 1.0])
n_plane = n_plane / (np.linalg.norm(n_plane) + 1e-12)
orth_w = np.array([abs(n_plane @ w1), abs(n_plane @ w2)])

# Color language (keep plane vs arrows distinct):
C_PLANE = '#f5d76e'       # soft gold -- tangent hyperplane ONLY
C_V1 = '#c0392b'          # crimson  -- x1 / v1
C_V2 = '#2471a3'          # blue     -- x2 / v2
C_W1 = '#1e8449'          # green    -- z1 / w1
C_W2 = '#6c3483'          # purple   -- z2 / w2
C_FLOOR = '#d5d8dc'       # domain plane sheet

# Domain-floor height (below the surface for visibility)
z_floor = float(ZZ.min()) - 0.35

# Axis directions in the domain (xy): x-frame and z-frame
ex1 = np.array([1.0, 0.0])
ex2 = np.array([0.0, 1.0])
ez1 = B[:, 0] / (np.linalg.norm(B[:, 0]) + 1e-12)  # increasing z1 in x-space
ez2 = B[:, 1] / (np.linalg.norm(B[:, 1]) + 1e-12)
axis_len = 1.15
arrow_scale = 0.85

u = np.linspace(-0.9, 0.9, 16)
vv = np.linspace(-0.9, 0.9, 16)
UU, VV = np.meshgrid(u, vv)
ZZ_plane = pz + fx * UU + fy * VV

fig7 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig7.suptitle(
    r'Step 7: One tangent hyperplane -- bases $\{v_1,v_2\}$ and $\{w_1,w_2\}$ '
    r'(and both coordinate frames on the domain)',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)
ax = fig7.add_subplot(111, projection='3d')

# Surface f (light) + single tangent plane
ax.plot_surface(XX, YY, ZZ, cmap='viridis', linewidth=0, alpha=0.22)
ax.plot_surface(
    px + UU, py + VV, ZZ_plane,
    color=C_PLANE, alpha=0.55, shade=True,
)
ax.scatter([px], [py], [pz], s=80, c='k', depthshade=False, label='point p')

# Domain plane sheet + BOTH coordinate frames drawn on it
floor_r = 1.6
xf = np.linspace(px - floor_r, px + floor_r, 2)
yf = np.linspace(py - floor_r, py + floor_r, 2)
XF, YF = np.meshgrid(xf, yf)
ax.plot_surface(XF, YF, np.full_like(XF, z_floor), color=C_FLOOR, alpha=0.35)

def _domain_arrow(direction_xy, color, label, lw=2.8):
    d = direction_xy * axis_len
    ax.quiver(
        px, py, z_floor,
        d[0], d[1], 0.0,
        color=color, linewidth=lw, arrow_length_ratio=0.12,
        label=label,
    )

_domain_arrow(ex1, C_V1, r'domain $x_1$-axis')
_domain_arrow(ex2, C_V2, r'domain $x_2$-axis')
_domain_arrow(ez1, C_W1, r'domain $z_1$-axis')
_domain_arrow(ez2, C_W2, r'domain $z_2$-axis')
ax.scatter([px], [py], [z_floor], s=40, c='k', depthshade=False)

# Drop line from p to domain (helps read the stacking)
ax.plot([px, px], [py, py], [z_floor, pz], color='0.35', lw=1.2, ls=':')

# Tangent vectors ON the hyperplane (ambient R^3)
ax.quiver(
    px, py, pz,
    arrow_scale * v1[0], arrow_scale * v1[1], arrow_scale * v1[2],
    color=C_V1, linewidth=3.0, arrow_length_ratio=0.15,
    label=r'$v_1=(1,0,\partial_{x_1}f)$',
)
ax.quiver(
    px, py, pz,
    arrow_scale * v2[0], arrow_scale * v2[1], arrow_scale * v2[2],
    color=C_V2, linewidth=3.0, arrow_length_ratio=0.15,
    label=r'$v_2=(0,1,\partial_{x_2}f)$',
)
ax.quiver(
    px, py, pz,
    arrow_scale * w1[0], arrow_scale * w1[1], arrow_scale * w1[2],
    color=C_W1, linewidth=3.0, arrow_length_ratio=0.15,
    label=r'$w_1$ from $\partial_{z_1}f$',
)
ax.quiver(
    px, py, pz,
    arrow_scale * w2[0], arrow_scale * w2[1], arrow_scale * w2[2],
    color=C_W2, linewidth=3.0, arrow_length_ratio=0.15,
    label=r'$w_2$ from $\partial_{z_2}f$',
)

ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax.set_zlabel(r'$f$', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_ANNOT)
ax.legend(loc='upper left', fontsize=FS_LEGEND - 2, framealpha=0.92)
ax.text2D(
    0.98, 0.02,
    'Gold sheet = THE tangent hyperplane (one object).\n'
    'Crimson/blue arrows = basis from x-partials.\n'
    'Green/purple arrows = basis from z-partials.\n'
    'Same colors on the floor = domain axes $(x)$ vs $(z)$.\n'
    'All four tangent arrows lie in the gold plane.',
    transform=ax.transAxes, fontsize=FS_ANNOT, ha='right', va='bottom',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.94),
)
fig7.text(
    0.5, 0.01,
    rf'$z=Ax$,  $[w_1\,w_2]=[v_1\,v_2]B$'
    rf'  |  recon err={recon_err:.2e}'
    rf'  |  $|n\cdot w|$={np.round(orth_w, 12).tolist()}',
    ha='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.96),
)
fig7.tight_layout(rect=[0, 0.04, 1, 0.94])
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Ridge (not a radial bell) -- X->Z warp tilts the ridge in 3D;
#            thick saturated rays; thin translucent surface
# -------------------------------------------------------
# Ridge along x2: f = exp(-x1^2 / (2 sig^2)).  Level sets are strips
# parallel to the x2-axis.  Under z = A x the ridge runs diagonally --
# the linear reparameterization is obvious in the 3D shape itself.

SIG_R = 0.45


def ridge(x1, x2, sig=SIG_R):
    return np.exp(-(x1 ** 2) / (2.0 * sig ** 2))


def ridge_dx1(x1, x2, sig=SIG_R):
    return ridge(x1, x2, sig) * (-x1 / (sig ** 2))


def ridge_dx2(x1, x2, sig=SIG_R):
    return np.zeros_like(np.asarray(x1, dtype=float))


# Strong shear: turns the axis-aligned ridge into a clear diagonal ramp-ridge
A_r = np.array([
    [1.0, 0.95],
    [-0.70, 1.0],
], dtype=float)
B_r = np.linalg.inv(A_r)

# Focus on the flank of the ridge (x1 != 0) so the tangent plane tilts
rx, ry = 0.55, 0.10
rz = float(ridge(rx, ry))
r_fx = float(ridge_dx1(rx, ry))
r_fy = float(ridge_dx2(rx, ry))  # 0
r_grad_x = np.array([r_fx, r_fy])
r_grad_z = B_r.T @ r_grad_x
r_gz1, r_gz2 = float(r_grad_z[0]), float(r_grad_z[1])

rv1 = np.array([1.0, 0.0, r_fx])
rv2 = np.array([0.0, 1.0, r_fy])  # (0,1,0): along the ridge
rw1 = np.array([B_r[0, 0], B_r[1, 0], r_gz1])
rw2 = np.array([B_r[0, 1], B_r[1, 1], r_gz2])

# Thick / saturated ray colors (surface stays pale)
C_RAY_V1 = '#e74c3c'   # vivid crimson
C_RAY_V2 = '#2980b9'   # vivid blue
C_RAY_W1 = '#27ae60'   # vivid green
C_RAY_W2 = '#9b59b6'   # vivid purple
C_RAY_FILL = '#d4ac0d'  # strong gold filler rays
C_PLANE_R = '#f9e79f'  # pale gold plane (behind thick rays)
SURF_ALPHA = 0.28      # thin / translucent 3D shape
RAY_LW = 4.2
RAY_FILL_LW = 2.0

# x-chart grid
xs_r = np.linspace(-1.7, 1.7, 100)
ys_r = np.linspace(-1.7, 1.7, 100)
RX, RY = np.meshgrid(xs_r, ys_r)
RF = ridge(RX, RY)

# z-chart: sample z-box, map to x, evaluate ridge; mask outside x-box
r_corners = np.array([[-1.7, -1.7], [-1.7, 1.7], [1.7, -1.7], [1.7, 1.7]])
rz_c = r_corners @ A_r.T
zs1 = np.linspace(rz_c[:, 0].min() + 0.05, rz_c[:, 0].max() - 0.05, 100)
zs2 = np.linspace(rz_c[:, 1].min() + 0.05, rz_c[:, 1].max() - 0.05, 100)
RZ1, RZ2 = np.meshgrid(zs1, zs2)
RX_from_z = B_r[0, 0] * RZ1 + B_r[0, 1] * RZ2
RY_from_z = B_r[1, 0] * RZ1 + B_r[1, 1] * RZ2
RG = ridge(RX_from_z, RY_from_z)
RG = np.ma.array(RG, mask=(np.abs(RX_from_z) > 1.65) | (np.abs(RY_from_z) > 1.65))

plane_r = 0.85
pu = np.linspace(-plane_r, plane_r, 16)
pv = np.linspace(-plane_r, plane_r, 16)
PU, PV = np.meshgrid(pu, pv)
RP_x = rz + r_fx * PU + r_fy * PV
z_focus = A_r @ np.array([rx, ry])
RP_z = rz + r_gz1 * PU + r_gz2 * PV


def _draw_plane_rays(ax, origin, basis_a, basis_b, colors, ray_len=0.90, n_extra=8):
    """Thick saturated rays in the tangent plane, all from the focus."""
    for vec, col in zip(basis_a + basis_b, colors):
        uvec = vec / (np.linalg.norm(vec) + 1e-12)
        for sgn in (+1.0, -1.0):
            q = origin + sgn * ray_len * uvec
            ax.plot(
                [origin[0], q[0]], [origin[1], q[1]], [origin[2], q[2]],
                color=col, lw=RAY_LW, solid_capstyle='round',
                solid_joinstyle='round', zorder=30,
            )
    rng_rays = np.random.default_rng(2)
    for _ in range(n_extra):
        a, b = rng_rays.normal(size=2)
        vec = a * basis_a[0] + b * basis_a[1]
        uvec = vec / (np.linalg.norm(vec) + 1e-12)
        q = origin + ray_len * uvec
        ax.plot(
            [origin[0], q[0]], [origin[1], q[1]], [origin[2], q[2]],
            color=C_RAY_FILL, lw=RAY_FILL_LW, alpha=0.95, zorder=25,
        )


fig8 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig8.suptitle(
    r'Step 8: Ridge along $x_2$ -- in $z=Ax$ the ridge tilts; thick rays on a thin surface',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)

VIEW_ELEV, VIEW_AZIM = 26, -50
FS_STEP8 = FS_ANNOT + 1  # a bit larger for Step-8 formula callouts

# Explicit formulas shown on the figure (matplotlib mathtext-safe)
formula_f = (
    rf'$f(x_1,x_2)=\exp(-x_1^2/(2\sigma^2))$  with $\sigma={SIG_R:g}$'
)
formula_A = (
    rf'$z = A x$ with '
    rf'$A = [[{A_r[0,0]:.2f},\ {A_r[0,1]:.2f}],\ '
    rf'[{A_r[1,0]:.2f},\ {A_r[1,1]:.2f}]]$,   '
    rf'$x = B z$,  $B = A^{{-1}}$'
)
formula_g = r'$g(z)=f(B z)$  (same surface in $z$-coordinates)'

# ---- Left: x-chart -- ridge parallel to x2 --------------------------------
ax = fig8.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(
    RX, RY, RF, cmap='Greys', linewidth=0, antialiased=True,
    alpha=SURF_ALPHA, shade=True,
)
ax.plot_surface(
    rx + PU, ry + PV, RP_x,
    color=C_PLANE_R, alpha=0.45, linewidth=0, shade=True, zorder=10,
)
origin_x = np.array([rx, ry, rz])
_draw_plane_rays(
    ax, origin_x,
    basis_a=[rv1, rv2],
    basis_b=[rw1, rw2],
    colors=[C_RAY_V1, C_RAY_V2, C_RAY_W1, C_RAY_W2],
)
ax.scatter([rx], [ry], [rz], s=110, c='k', depthshade=False, zorder=40)
ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax.set_zlabel(r'$f$', fontsize=FS_AXIS)
ax.set_title(r'$x$-chart: ridge $\parallel x_2$ (axis-aligned)', fontsize=FS_AX_TITLE - 2)
ax.tick_params(labelsize=FS_ANNOT)
ax.set_zlim(0.0, 1.15)
ax.text2D(
    0.02, 0.98,
    formula_f + '\n'
    'Crest runs along $x_2$ (independent of $x_2$).\n'
    'Thick rays from focus $p$; surface kept pale.',
    transform=ax.transAxes, fontsize=FS_STEP8, va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

# ---- Right: z-chart -- same ridge, diagonally sheared --------------------
ax = fig8.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(
    RZ1, RZ2, RG, cmap='Greys', linewidth=0, antialiased=True,
    alpha=SURF_ALPHA, shade=True,
)
ax.plot_surface(
    z_focus[0] + PU, z_focus[1] + PV, RP_z,
    color=C_PLANE_R, alpha=0.45, linewidth=0, shade=True, zorder=10,
)
wt1 = np.array([1.0, 0.0, r_gz1])
wt2 = np.array([0.0, 1.0, r_gz2])
vt1_in_z = np.array([A_r[0, 0], A_r[1, 0], r_fx])
vt2_in_z = np.array([A_r[0, 1], A_r[1, 1], r_fy])
origin_z = np.array([z_focus[0], z_focus[1], rz])
_draw_plane_rays(
    ax, origin_z,
    basis_a=[vt1_in_z, vt2_in_z],
    basis_b=[wt1, wt2],
    colors=[C_RAY_V1, C_RAY_V2, C_RAY_W1, C_RAY_W2],
)
ax.scatter([z_focus[0]], [z_focus[1]], [rz], s=110, c='k', depthshade=False, zorder=40)
ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
ax.set_xlabel(r'$z_1$', fontsize=FS_AXIS)
ax.set_ylabel(r'$z_2$', fontsize=FS_AXIS)
ax.set_zlabel(r'$g(z)=f(Bz)$', fontsize=FS_AXIS)
ax.set_title(r'$z$-chart: SAME ridge, now diagonal (linear warp)', fontsize=FS_AX_TITLE - 2)
ax.tick_params(labelsize=FS_ANNOT)
ax.set_zlim(0.0, 1.15)
ax.text2D(
    0.02, 0.98,
    formula_g + '\n'
    r'Reparameterization: $z=A x$  (see banner).' + '\n'
    'Crest tilts -- that tilt IS $x\\!\\to\\!z$.',
    transform=ax.transAxes, fontsize=FS_STEP8, va='top',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig8.text(
    0.5, 0.015,
    formula_f + '\n' + formula_A + '\n' + formula_g
    + r'   |   thick rays; thin surface; drag either panel to browse',
    ha='center', va='bottom', fontsize=FS_STEP8,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.96),
)
fig8.tight_layout(rect=[0, 0.10, 1, 0.93])
show_maximized(fig8)


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
  Shows f itself next to the partial fields df/dx1 and df/dx2
  (same domain point marked on all three).

Coordinate freedom (Step 7-8):
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
  Colors: crimson/blue = x-frame; green/purple = z-frame; gold = plane.

Step 8 (ridge demo):
  f = exp(-x1^2 / (2 sig^2)) -- a ridge along x2 (axis-aligned in x).
  Under z = A x the crest tilts diagonally: the linear warp is visible
  in the 3D shape itself (not only in level-set ellipses).
  Tangent plane on top; thick saturated rays from focus p; thin pale surface.


Browsable figures:
  Drag with the mouse to rotate any 3D view; scroll to zoom.
  Use this to inspect hills, valleys, and slopes from multiple angles.

ML link:
  Training steps follow -grad of a loss L(w).
  Partials are the coordinates of that gradient -- and those coordinates
  can be rewritten in any linear basis without changing the tangent plane.
""")
