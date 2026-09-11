"""
0810_dot_inner_scalar_product.py

Dot / inner / scalar product -- same operation on R^n, three names.

For vectors x, y in R^n:

    <x, y>  =  x1 y1 + ... + xn yn
            =  |x| |y| cos(theta)

where theta = angle(x, y).

Why three names (same number on standard R^n):
  * Dot product     -- notation x·y; common in calc / ML
  * Scalar product  -- stresses that the result is a scalar (vs cross product)
  * Inner product   -- abstract <x,y>; also for functions / Hilbert spaces

Intuition:
  * Project y onto the direction of x:  length of that shadow is |y| cos(theta)
  * Multiply by |x|: that is the scalar product
  * Same if you swap x and y (symmetric)

Cosine similarity (ML / NLP):
    cos(x, y) = <x, y> / (|x| |y|)   in [-1, 1]
  = pure angle agreement; lengths cancelled.

Level sets of a linear form:
    X(a, b) = { x | <x, a> = b }
are affine hyperplanes: dimension = n - 1 in R^n.
Changing b slides the hyperplane parallel to itself.
Shift formula (derived below):

    X(a, b1)  =  X(a, b2)  +  ((b1 - b2) / |a|^2) a

Companion to 0490_derivative.py (gradients / planes) and
0850_transformer.py (attention scores are scaled dots).
"""

from __future__ import annotations

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
from matplotlib.patches import Arc
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
    print('Tip: click-drag to rotate 3D views; scroll to zoom.')
    add_step_footer(fig)
    plt.show()


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


# ---------------------------------------------------------------------------
# Demo vectors
# ---------------------------------------------------------------------------
x = np.array([3.0, 1.2])
y = np.array([1.0, 2.4])
dot_xy = float(x @ y)
nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
cos_xy = dot_xy / (nx * ny)
theta = float(np.arccos(np.clip(cos_xy, -1.0, 1.0)))
# Projection of y onto x:  ((y·x)/|x|^2) x
proj_y_on_x = (dot_xy / (nx ** 2)) * x
# Signed shadow length along unit x
shadow = float(unit(x) @ y)  # = |y| cos(theta)


# -------------------------------------------------------
# FIGURE 1: Definition, then why three names, then geometry
# -------------------------------------------------------
# Middle column wider than the side axes; box widths fill most of that column.
def build_step_1():
    fig1, axes1 = plt.subplots(
        1, 3, figsize=screen_figsize(), dpi=FIGURE_DPI,
        gridspec_kw=dict(width_ratios=[1.15, 2.75, 1.15], wspace=0.22),
    )
    fig1.suptitle(
        r'Step 1: Definition  $\langle x,y\rangle = \sum_i x_i y_i = |x||y|\cos\theta$'
        '  -- and why three names',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    ax = axes1[0]
    ax.axhline(0, color='0.7', lw=1)
    ax.axvline(0, color='0.7', lw=1)
    ax.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1,
              color='#c0392b', width=0.012, label=r'$x$')
    ax.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1,
              color='#2471a3', width=0.012, label=r'$y$')
    # Annotation outside the axes (below), so the plot can be larger.
    ax.text(
        0.5, -0.36,
        'Mathematical definition\n'
        r'(standard $\mathbb{R}^n$):' + '\n'
        rf'$\langle x,y\rangle = \sum_i x_i y_i$' + '\n'
        rf'$= ({x[0]:.1f})({y[0]:.1f})$'
        rf'$+ ({x[1]:.1f})({y[1]:.1f})$'
        rf'$= {dot_xy:.2f}$' + '\n'
        r'Also $= |x||y|\cos\theta$.',
        transform=ax.transAxes, ha='center', va='top', fontsize=FS_ANNOT - 1,
        clip_on=False,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    ax.set_xlim(-0.35, 3.7)
    ax.set_ylim(-0.35, 3.0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$e_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$e_2$', fontsize=FS_AXIS)
    # y>1 places the title above the axes; pad was being ignored by tight_layout.
    ax.set_title('1) Coordinate definition', fontsize=FS_AX_TITLE, y=1.14)
    ax.legend(fontsize=FS_LEGEND, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)

    # Why three names for the same number on R^n
    ax = axes1[1]
    ax.set_xlim(0, 26)
    ax.set_ylim(-0.55, 10)
    ax.axis('off')
    # Place this title in data coords so tight_layout cannot "haunt" pad-based moves.
    # Sits in the gap above the top box (box top ≈ 9.15).
    ax.text(
        13.0, 9.55, '2) Same number -- three names',
        ha='center', va='center', fontsize=FS_AX_TITLE, fontweight='normal',
    )
    # Three name boxes with equal gaps (~0.15) between them.
    name_boxes = [
        (0.05, 6.25, 25.9, 2.9, '#fdebd0',
         'Dot product   $x \\cdot y$\n'
         '\n'
         'Meaning:  the "dot" notation;\n'
         '          multiply matching coordinates and add.\n'
         '\n'
         'Usage:    vector calc, physics, ML\n'
         '          ("dot-product attention", cosine via a dot).'),
        (0.05, 3.20, 25.9, 2.9, '#d6eaf8',
         'Scalar product\n'
         '\n'
         'Meaning:  the result is a SCALAR (one number), not a vector --\n'
         '          unlike the cross / vector product.\n'
         '\n'
         'Usage:    physics texts (esp. European); stresses output type.'),
        (0.05, 0.35, 25.9, 2.7, '#d5f5e3',
         'Inner product   $\\langle x, y \\rangle$\n'
         '\n'
         'Meaning:  abstract bilinear form with axioms\n'
         '          (linear, symmetric, positive-definite).\n'
         '\n'
         'Usage:    linear algebra / Hilbert spaces; same idea for functions,\n'
         r'          e.g. $\langle f,g\rangle=\int f g$.'),
    ]
    for x0, y0, w, h, color, text in name_boxes:
        ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor=color, edgecolor='k', lw=1.8))
        ax.text(x0 + w / 2, y0 + h / 2, text, ha='center', va='center',
                fontsize=FS_ANNOT - 1, fontweight='bold')
    # Special note under the lowest box (not one of the three aligned boxes).
    ax.text(
        13.0, -0.12,
        r'On $\mathbb{R}^n$ with the standard basis these three names denote THE SAME operation.',
        ha='center', va='top', fontsize=FS_ANNOT - 1, clip_on=False,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
    )

    ax = axes1[2]
    ax.axhline(0, color='0.7', lw=1)
    ax.axvline(0, color='0.7', lw=1)
    ax.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1,
              color='#c0392b', width=0.012, label=r'$x$')
    ax.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1,
              color='#2471a3', width=0.012, label=r'$y$')
    ax.plot([0, proj_y_on_x[0]], [0, proj_y_on_x[1]], color='#1e8449', lw=3.0,
            label=r'proj$_x y$')
    ax.plot([y[0], proj_y_on_x[0]], [y[1], proj_y_on_x[1]],
            color='0.4', lw=1.5, ls='--')
    arc = Arc((0, 0), 1.2, 1.2,
              theta1=np.degrees(np.arctan2(x[1], x[0])),
              theta2=np.degrees(np.arctan2(y[1], y[0])),
              color='black', lw=1.5)
    ax.add_patch(arc)
    ax.text(0.7, 0.55, r'$\theta$', fontsize=FS_AXIS)
    # Annotation outside the axes (below).
    ax.text(
        0.5, -0.36,
        'Geometry:\n'
        r'1) Drop a perpendicular from $y$ onto $x$' + '\n'
        r'2) Shadow length $= |y|\cos\theta$'
        rf' $= {shadow:.2f}$' + '\n'
        r'3) Multiply by $|x|$:  '
        rf'$\langle x,y\rangle = |x|\,|y|\cos\theta = {dot_xy:.2f}$',
        transform=ax.transAxes, ha='center', va='top', fontsize=FS_ANNOT - 1,
        clip_on=False,
        bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
    )
    ax.set_xlim(-0.35, 3.7)
    ax.set_ylim(-0.35, 3.0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$e_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$e_2$', fontsize=FS_AXIS)
    ax.set_title('3) Projection / angle reading', fontsize=FS_AX_TITLE, y=1.14)
    ax.legend(fontsize=FS_LEGEND, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FS_TICK)

    fig1.tight_layout(rect=[0, 0.18, 1, 0.94])
    return fig1



# -------------------------------------------------------
# FIGURE 2: Intuition board -- what the sign / magnitude say
# -------------------------------------------------------
def build_step_2():
    fig2, axes2 = plt.subplots(1, 3, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig2.suptitle(
        r'Step 2: Sign of $\langle x,y\rangle$ = acute / right / obtuse',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    cases = [
        (np.array([2.0, 0.3]), np.array([1.6, 1.0]), 'acute: cos > 0\ndot > 0\n"agree"'),
        (np.array([2.0, 0.0]), np.array([0.0, 1.8]), 'right: cos = 0\ndot = 0\northogonal'),
        (np.array([2.0, 0.4]), np.array([-1.5, 0.8]), 'obtuse: cos < 0\ndot < 0\n"oppose"'),
    ]
    colors = ['#1e8449', '#7f8c8d', '#c0392b']
    for ax, (u, v, note), c in zip(axes2, cases, colors):
        ax.axhline(0, color='0.75', lw=1)
        ax.axvline(0, color='0.75', lw=1)
        ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,
                  color='#c0392b', width=0.015)
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  color='#2471a3', width=0.015)
        d = float(u @ v)
        ax.set_title(rf'$\langle x,y\rangle={d:.2f}$', fontsize=FS_AX_TITLE - 2)
        ax.text(0.5, 0.05, note, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=FS_ANNOT,
                bbox=dict(boxstyle='round', facecolor=c, alpha=0.25))
        ax.set_xlim(-2.2, 2.8)
        ax.set_ylim(-1.2, 2.4)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FS_TICK)

    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    return fig2



# -------------------------------------------------------
# FIGURE 3: Cosine similarity
# -------------------------------------------------------
# Two pairs with same angle but different lengths -- cos identical, dot not
a1 = np.array([2.0, 1.0])
b1 = np.array([1.5, 2.0])
a2 = 2.5 * a1
b2 = 2.5 * b1
cos1 = float(a1 @ b1) / (np.linalg.norm(a1) * np.linalg.norm(b1))
cos2 = float(a2 @ b2) / (np.linalg.norm(a2) * np.linalg.norm(b2))
dot1 = float(a1 @ b1)
dot2 = float(a2 @ b2)

def build_step_3():
    fig3, axes3 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig3.suptitle(
        r'Step 3: Cosine similarity $=\langle x,y\rangle / (|x||y|)$ ignores length',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    for ax, (u, v, title) in zip(axes3, [
        (a1, b1, f'short pair\ndot={dot1:.2f},  cos={cos1:.3f}'),
        (a2, b2, f'scaled pair (same direction)\ndot={dot2:.2f},  cos={cos2:.3f}'),
    ]):
        ax.axhline(0, color='0.75', lw=1)
        ax.axvline(0, color='0.75', lw=1)
        ax.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1,
                  color='#c0392b', width=0.012, label='x')
        ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
                  color='#2471a3', width=0.012, label='y')
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.5, 7.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=FS_AX_TITLE - 4)
        ax.legend(fontsize=FS_LEGEND)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FS_TICK)

    axes3[0].text(
        0.02, 0.98,
        'Cosine similarity:\n'
        r'$\mathrm{cos}(x,y)=\frac{\langle x,y\rangle}{|x||y|}\in[-1,1]$' + '\n\n'
        'Used in embeddings / retrieval:\n'
        'care about DIRECTION agreement,\n'
        'not vector magnitude.',
        transform=axes3[0].transAxes, va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    axes3[1].text(
        0.02, 0.98,
        'Scaling both vectors changes\n'
        'the DOT but NOT the COSINE.\n\n'
        rf'cos short = {cos1:.3f}' + '\n'
        rf'cos scaled = {cos2:.3f}',
        transform=axes3[1].transAxes, va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
    )

    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    return fig3



# -------------------------------------------------------
# FIGURE 4: X(a,b) = {x | <x,a> = b} is a hyperplane (dim n-1)
# -------------------------------------------------------
# In R^2: a line. In R^3: a plane.
a2d = np.array([2.0, 1.0])
b_level = 2.0
# Line: 2 x1 + x2 = 2  =>  x2 = 2 - 2 x1
# Point on line closest to origin: (b/|a|^2) a
p0 = (b_level / (a2d @ a2d)) * a2d

def build_step_4():
    fig4 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig4.suptitle(
        r'Step 4: $X(a,b)=\{x\,|\,\langle x,a\rangle=b\}$ is a hyperplane (codimension 1)',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    # 2D panel
    ax = fig4.add_subplot(1, 2, 1)
    t = np.linspace(-1.5, 2.5, 50)
    # direction orthogonal to a
    orth = np.array([-a2d[1], a2d[0]])
    orth = unit(orth)
    line = p0[None, :] + t[:, None] * orth[None, :]
    ax.plot(line[:, 0], line[:, 1], color='#1e8449', lw=3.0,
            label=rf'$X(a,{b_level:g})$  (a line in $\mathbb{{R}}^2$)')
    ax.quiver(0, 0, a2d[0], a2d[1], angles='xy', scale_units='xy', scale=1,
              color='#c0392b', width=0.012, label=r'normal $a$')
    ax.scatter([p0[0]], [p0[1]], c='k', s=60, zorder=5)
    ax.plot([0, p0[0]], [0, p0[1]], color='0.4', ls='--', lw=1.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 3.0)
    ax.set_ylim(-1.5, 3.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc='upper right')
    ax.set_title(r'In $\mathbb{R}^2$: hyperplane = line (dim 1)', fontsize=FS_AX_TITLE - 2)
    ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(
        0.02, 0.02,
        r'Constraint $\langle x,a\rangle=b$ is ONE scalar' + '\n'
        r'equation => lose 1 degree of freedom.' + '\n'
        r'$\dim X(a,b) = n - 1$.',
        transform=ax.transAxes, va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    # 3D panel
    ax = fig4.add_subplot(1, 2, 2, projection='3d')
    a3 = np.array([1.0, 0.6, 0.8])
    b3 = 1.4
    # Plane: a·x = b3. Parameterize with two orthonormal vectors perp to a
    a_u = unit(a3)
    # build ONB for the plane
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a_u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = unit(np.cross(a_u, tmp))
    e2 = unit(np.cross(a_u, e1))
    center = (b3 / (a3 @ a3)) * a3
    uu = np.linspace(-1.6, 1.6, 20)
    vv = np.linspace(-1.6, 1.6, 20)
    UU, VV = np.meshgrid(uu, vv)
    P = center[None, None, :] + UU[:, :, None] * e1[None, None, :] + VV[:, :, None] * e2[None, None, :]
    ax.plot_surface(P[:, :, 0], P[:, :, 1], P[:, :, 2], color='#f5d76e', alpha=0.75)
    ax.quiver(0, 0, 0, a3[0], a3[1], a3[2], color='#c0392b', linewidth=2.5,
              arrow_length_ratio=0.12)
    ax.scatter([center[0]], [center[1]], [center[2]], c='k', s=50)
    ax.plot([0, center[0]], [0, center[1]], [0, center[2]], color='0.4', ls='--', lw=1.3)
    ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
    ax.set_zlabel(r'$x_3$', fontsize=FS_AXIS)
    ax.set_title(r'In $\mathbb{R}^3$: hyperplane = plane (dim 2)', fontsize=FS_AX_TITLE - 2)
    ax.tick_params(labelsize=FS_ANNOT)
    ax.view_init(elev=22, azim=-55)
    ax.text2D(
        0.02, 0.02,
        'Gold = X(a,b).\n'
        'Crimson = normal a.\n'
        'Drag to browse.',
        transform=ax.transAxes, fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.94),
    )

    fig4.tight_layout(rect=[0, 0, 1, 0.93])
    return fig4



# -------------------------------------------------------
# FIGURE 5: Parallel shift X(a,b2) -> X(a,b1)
# -------------------------------------------------------
# Derivation:
#   Take x in X(a, b2), so <x, a> = b2.
#   Set x' = x + t a.
#   Then <x', a> = b2 + t |a|^2.
#   Want = b1  =>  t = (b1 - b2) / |a|^2.
#   Hence shift vector  s = ((b1 - b2) / |a|^2) a
#   and  X(a, b1) = X(a, b2) + s.

a = np.array([2.0, 1.0])
# Level scalars for the parallel-shift demo (do not reuse cosine-demo names b1/b2).
b_lo, b_hi = 1.0, 3.0
s = ((b_hi - b_lo) / (a @ a)) * a
p2 = (b_lo / (a @ a)) * a   # foot of perpendicular on X(a,b_lo)
p1 = p2 + s                 # foot on X(a,b_hi)  (== (b_hi/|a|^2) a)

orth = unit(np.array([-a[1], a[0]]))
t = np.linspace(-2.0, 2.0, 60)
line2 = p2[None, :] + t[:, None] * orth[None, :]
line1 = p1[None, :] + t[:, None] * orth[None, :]

def build_step_5():
    fig5, axes5 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig5.suptitle(
        r'Step 5: $X(a,b_1)$ is $X(a,b_2)$ shifted by $\frac{b_1-b_2}{|a|^2}\,a$',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    ax = axes5[0]
    ax.plot(line2[:, 0], line2[:, 1], color='#2471a3', lw=3.0,
            label=rf'$X(a,{b_lo:g})$')
    ax.plot(line1[:, 0], line1[:, 1], color='#1e8449', lw=3.0,
            label=rf'$X(a,{b_hi:g})$')
    ax.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1,
              color='#c0392b', width=0.012, label=r'normal $a$')
    # shift arrow from a point on line2 to its image on line1
    src = p2 + 0.8 * orth
    dst = src + s
    ax.annotate(
        '', xy=dst, xytext=src,
        arrowprops=dict(arrowstyle='->', color='#6c3483', lw=3.0),
    )
    ax.text(
        0.5 * (src[0] + dst[0]) + 0.15,
        0.5 * (src[1] + dst[1]) + 0.15,
        r'$s=\frac{b_1-b_2}{|a|^2}a$',
        color='#6c3483', fontsize=FS_ANNOT, fontweight='bold',
    )
    ax.scatter([p2[0], p1[0]], [p2[1], p1[1]], c='k', s=50, zorder=5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 3.5)
    ax.set_ylim(-2.0, 3.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc='upper right')
    ax.set_title('Parallel slide along the normal', fontsize=FS_AX_TITLE - 2)
    ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.text(
        0.02, 0.02,
        'Same normal $a$ => planes/lines are parallel.\n'
        'Only the offset $b$ changes.\n'
        'Shift is ALWAYS parallel to $a$.',
        transform=ax.transAxes, va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    ax = axes5[1]
    ax.axis('off')
    ax.set_title('Derivation of the shift', fontsize=FS_AX_TITLE - 2)
    ax.text(
        0.5, 0.55,
        r'Let $x\in X(a,b_2)$, so $\langle x,a\rangle=b_2$.' + '\n\n'
        r'Try a pure normal move:  $\tilde{x} = x + t\,a$.' + '\n\n'
        r'$\langle \tilde{x},a\rangle = \langle x,a\rangle + t\,|a|^2'
        r' = b_2 + t\,|a|^2$.' + '\n\n'
        r'Want this equal to $b_1$:' + '\n'
        r'$\qquad t = \dfrac{b_1 - b_2}{|a|^2}$.' + '\n\n'
        r'Therefore the shift vector is' + '\n'
        r'$\qquad \mathbf{s} = \dfrac{b_1 - b_2}{|a|^2}\,a$' + '\n\n'
        r'and' + '\n'
        r'$\qquad X(a,b_1) = X(a,b_2) + \mathbf{s}$.' + '\n\n'
        r'Check: $\langle p_2+\mathbf{s},\,a\rangle'
        rf' = {float((p2+s)@a):.2f} = b_1$.',
        transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 2,
        bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
    )

    fig5.tight_layout(rect=[0, 0, 1, 0.94])
    return fig5



# -------------------------------------------------------
# FIGURE 6: 3D parallel planes X(a,b) for several b
# -------------------------------------------------------
def build_step_6():
    fig6 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig6.suptitle(
        r'Step 6: Browse parallel hyperplanes $X(a,b)$ as $b$ varies',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    ax = fig6.add_subplot(111, projection='3d')

    a3 = np.array([1.0, 0.4, 0.7])
    a_u = unit(a3)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a_u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = unit(np.cross(a_u, tmp))
    e2 = unit(np.cross(a_u, e1))
    bs = [-1.0, 0.5, 2.0]
    colors_b = ['#2471a3', '#f5d76e', '#1e8449']
    uu = np.linspace(-1.4, 1.4, 16)
    vv = np.linspace(-1.4, 1.4, 16)
    UU, VV = np.meshgrid(uu, vv)

    centers = []
    for b, col in zip(bs, colors_b):
        c = (b / (a3 @ a3)) * a3
        centers.append(c)
        P = c[None, None, :] + UU[:, :, None] * e1 + VV[:, :, None] * e2
        ax.plot_surface(P[:, :, 0], P[:, :, 1], P[:, :, 2], color=col, alpha=0.55)
        ax.text(c[0], c[1], c[2] + 0.15, rf'$b={b:g}$', fontsize=FS_ANNOT)

    # shift arrows between consecutive planes
    for i in range(len(bs) - 1):
        s_i = ((bs[i + 1] - bs[i]) / (a3 @ a3)) * a3
        ax.quiver(
            centers[i][0], centers[i][1], centers[i][2],
            s_i[0], s_i[1], s_i[2],
            color='#6c3483', linewidth=2.5, arrow_length_ratio=0.15,
        )

    ax.quiver(0, 0, 0, a3[0], a3[1], a3[2], color='#c0392b', linewidth=2.5,
              arrow_length_ratio=0.12)
    ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
    ax.set_zlabel(r'$x_3$', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_ANNOT)
    ax.view_init(elev=20, azim=-60)
    ax.text2D(
        0.02, 0.02,
        'Purple arrows = shifts\n'
        r'$s=((b_{k+1}-b_k)/|a|^2)\,a$' + '\n'
        'All planes share normal $a$\n'
        '(crimson). Drag to browse.',
        transform=ax.transAxes, fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.94),
    )
    return fig6




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
s_check = ((b_hi - b_lo) / (a @ a)) * a
print("\n" + "=" * 60)
print("DOT / INNER / SCALAR PRODUCT")
print("=" * 60)
print(f"""
Definition (R^n):
  <x, y> = sum_i x_i y_i
         = |x| |y| cos(theta)

Demo vectors:
  x = {x.tolist()}
  y = {y.tolist()}
  <x,y> = {dot_xy:.4f}
  |x|={nx:.4f}, |y|={ny:.4f}, theta={np.degrees(theta):.1f} deg
  cos(theta) = {cos_xy:.4f}
  |y| cos(theta) = shadow on unit-x = {shadow:.4f}
  |x| * shadow   = {nx * shadow:.4f}  (= dot)

Intuition:
  Dot = (length of x) * (signed length of y's shadow along x).
  Positive / zero / negative <=> acute / right / obtuse.

Cosine similarity:
  cos(x, y) = <x,y> / (|x||y|) in [-1, 1]
  Length-invariant measure of directional agreement.
  Demo: short pair cos={cos1:.4f}, scaled pair cos={cos2:.4f} (identical)
        but dots {dot1:.2f} vs {dot2:.2f} (changed).

Hyperplanes:
  X(a, b) = {{ x | <x, a> = b }}
  One linear constraint => dim = n - 1  (line in R^2, plane in R^3, ...).
  Normal direction = a; offset controlled by b.

Parallel shift formula:
  X(a, b1) = X(a, b2) + s
  with
      s = ((b1 - b2) / |a|^2) * a

  Demo: a={a.tolist()}, b2={b_lo}, b1={b_hi}
        s = {np.round(s_check, 6).tolist()}
        check <p2 + s, a> = {float((p2 + s_check) @ a):.6f}  (want {b_hi})

ML links:
  Attention scores ~ query·key  (scaled dots) -- see 150 / 151.
  Embedding retrieval ranks by cosine similarity.
  Linear classifiers: decision boundary is X(w, -bias) in feature space.
""")
