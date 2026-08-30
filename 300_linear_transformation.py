"""
300_linear_transformation.py

Linear spaces, R^n, and why a linear map T: R^n -> R^m is a matrix A.

Core dictionary:
  - R^n = n-tuples of real numbers, with vector add / scalar multiply
  - Linear map: T(u+v)=T(u)+T(v) and T(c u)=c T(u)
  - Matrix A: the unique array whose j-th COLUMN is T(e_j) written in the
    target basis. Then T(x) = A x for every x.

What a_ij means:
  T(e_j) = sum_i a_ij * f_i
  so a_ij = coordinate i of the image of source-basis vector e_j
            in the target basis {f_i}.

Why A x looks the way it does:
  x = sum_j x_j e_j
  T(x) = sum_j x_j T(e_j) = sum_j x_j (column j of A) = A x

Limits (Steps 7-8): cannot translate, bend lines, or apply local/gated rules alone;
  then why nonlinear layers.
2D grids (Steps 10-11): fixed paired-color map, then sliders on a_ij.
3D grids (Steps 12-13): same idea in 3D, browsable + sliders.
Customer takeaway (Step 14):
  Fixed linear preprocessing can be absorbed into a learned linear layer
  (W (P x) = (W P) x), so it does not reduce training power when the net
  already has learned linear maps -- which it always does.

Companion to 340_Wq_Wk_Wv.py (W_Q, W_K, W_V are linear maps on token space).
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
)
apply_rcparams(plt)
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)
import numpy as np

np.random.seed(42)
RNG = np.random.default_rng(42)


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
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        try:
            manager.window.state("zoomed")
        except Exception:
            try:
                manager.window.showMaximized()
            except Exception:
                try:
                    manager.frame.Maximize(True)
                except Exception:
                    pass
    # Tip only when a 3D axes is present
    if any(getattr(ax, "name", "") == "3d" for ax in fig.axes):
        print("Tip: click-drag to rotate 3D views; scroll to zoom.")
    add_step_footer(fig)
    plt.show()


def grid_lines_2d(lim=1.6, n=3, n_pts=40):
    """Axis-aligned grid lines in R^2. Each item is (points Nx2, family, index).

    family: 0 = vertical (x=const; vary y)
            1 = horizontal (y=const; vary x)
    """
    coords = np.linspace(-lim, lim, n)
    t = np.linspace(-lim, lim, n_pts)
    lines = []
    idx = 0
    for x in coords:  # vertical lines
        pts = np.stack([np.full_like(t, x), t], axis=1)
        lines.append((pts, 0, idx))
        idx += 1
    for y in coords:  # horizontal lines
        pts = np.stack([t, np.full_like(t, y)], axis=1)
        lines.append((pts, 1, idx))
        idx += 1
    return lines


def grid_lines_3d(lim=1.5, n=5, n_pts=25):
    """Axis-aligned grid lines in R^3. Each item is (points Nx3, family, index).

    family: 0 = parallel to x (vary x; y,z fixed)
            1 = parallel to y
            2 = parallel to z
    """
    coords = np.linspace(-lim, lim, n)
    t = np.linspace(-lim, lim, n_pts)
    lines = []
    idx = 0
    # lines // e1 : y=const, z=const
    for iy, y in enumerate(coords):
        for iz, z in enumerate(coords):
            pts = np.stack([t, np.full_like(t, y), np.full_like(t, z)], axis=1)
            lines.append((pts, 0, idx))
            idx += 1
    # lines // e2 : x=const, z=const
    for ix, x in enumerate(coords):
        for iz, z in enumerate(coords):
            pts = np.stack([np.full_like(t, x), t, np.full_like(t, z)], axis=1)
            lines.append((pts, 1, idx))
            idx += 1
    # lines // e3 : x=const, y=const
    for ix, x in enumerate(coords):
        for iy, y in enumerate(coords):
            pts = np.stack([np.full_like(t, x), np.full_like(t, y), t], axis=1)
            lines.append((pts, 2, idx))
            idx += 1
    return lines


def draw_basis(ax, origin, v1, v2, colors=("crimson", "navy"), labels=(r"$e_1$", r"$e_2$"), scale=1.0):
    o = np.asarray(origin, dtype=float)
    for v, c, lab in zip((v1, v2), colors, labels):
        vv = scale * np.asarray(v, dtype=float)
        ax.quiver(
            o[0], o[1], vv[0], vv[1],
            angles="xy", scale_units="xy", scale=1.0,
            color=c, width=0.012, zorder=5,
        )
        ax.text(o[0] + vv[0] * 1.08, o[1] + vv[1] * 1.08, lab,
                color=c, fontsize=FS_ANNOT, fontweight="bold")


def unit_grid(A, lim=2.2, n=9):
    """Image of an axis-aligned grid under 2x2 matrix A."""
    xs = np.linspace(-lim, lim, n)
    lines = []
    for x in xs:
        t = np.linspace(-lim, lim, 40)
        # vertical line x=const
        pts = np.stack([np.full_like(t, x), t], axis=1) @ A.T
        lines.append(pts)
        # horizontal line y=const
        pts = np.stack([t, np.full_like(t, x)], axis=1) @ A.T
        lines.append(pts)
    return lines


# -------------------------------------------------------
# FIGURE 1: What is R^n?  (show R^2 as the prototype)
# -------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig1.suptitle(
    r'Step 1: Linear space $\mathbb{R}^n$ -- lists of $n$ reals you can add and scale',
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

ax = axes1[0]
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect("equal")
ax.axhline(0, color="0.5", lw=0.8)
ax.axvline(0, color="0.5", lw=0.8)
ax.grid(True, alpha=0.3)
# A few points and their vector arrows from origin
pts = np.array([[2.0, 1.0], [-1.5, 2.0], [1.0, -2.2]])
labs = [r"$v=(2,1)$", r"$u=(-1.5,2)$", r"$w=(1,-2.2)$"]
cols = ["crimson", "navy", "seagreen"]
for p, lab, c in zip(pts, labs, cols):
    ax.quiver(0, 0, p[0], p[1], angles="xy", scale_units="xy", scale=1.0,
              color=c, width=0.01, zorder=3)
    ax.scatter([p[0]], [p[1]], s=50, color=c, zorder=4)
    ax.text(p[0] * 1.05 + 0.05, p[1] * 1.05, lab, color=c, fontsize=FS_ANNOT)
draw_basis(ax, (0, 0), (1, 0), (0, 1), labels=(r"$e_1=(1,0)$", r"$e_2=(0,1)$"))
ax.set_title(r"$\mathbb{R}^2$ with the standard basis", fontsize=FS_AX_TITLE)
ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)
ax.text(
    0.02, 0.02,
    r"Every $v\in\mathbb{R}^2$ is" + "\n"
    r"$v = x_1 e_1 + x_2 e_2$." + "\n"
    r"Same idea in $\mathbb{R}^n$:" + "\n"
    r"$v=\sum_{j=1}^n x_j e_j$.",
    transform=ax.transAxes, va="bottom", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)

ax = axes1[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Axioms you actually use", fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.55,
    "A real linear space (vector space) lets you:\n\n"
    "  • add two vectors:     u + v\n"
    "  • scale by a real:     c · v\n\n"
    r"$\mathbb{R}^n = \{(x_1,\ldots,x_n): x_j\in\mathbb{R}\}$" + "\n"
    "with componentwise + and · is the standard example.\n\n"
    "Linear algebra starts here because neural nets traffic in\n"
    "lists of activations -- and linear maps are the simplest\n"
    "structure-preserving transforms of those lists.",
    transform=ax.transAxes, ha="center", va="center", fontsize=FS_AXIS - 1,
    bbox=dict(boxstyle="round", facecolor="#eef5ff", alpha=0.96),
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: What is a linear transformation?
# -------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig2.suptitle(
    r"Step 2: Linear map $T:\mathbb{R}^n\to\mathbb{R}^m$ -- respects add and scale",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

# Left: geometric check in R^2 -- parallelogram law
A_demo = np.array([[1.2, 0.6], [0.2, 1.1]], dtype=float)
u = np.array([1.4, 0.3])
v = np.array([0.2, 1.3])
Tu, Tv, Tuv = A_demo @ u, A_demo @ v, A_demo @ (u + v)

ax = axes2[0]
ax.set_aspect("equal")
ax.axhline(0, color="0.5", lw=0.7)
ax.axvline(0, color="0.5", lw=0.7)
ax.grid(True, alpha=0.3)
for p, c, lab in [
    (u, "0.55", r"$u$"), (v, "0.55", r"$v$"), (u + v, "0.35", r"$u+v$"),
    (Tu, "crimson", r"$T(u)$"), (Tv, "navy", r"$T(v)$"),
    (Tuv, "seagreen", r"$T(u+v)=T(u)+T(v)$"),
]:
    ax.quiver(0, 0, p[0], p[1], angles="xy", scale_units="xy", scale=1.0,
              color=c, width=0.01, zorder=3)
    ax.text(p[0] * 1.05, p[1] * 1.05, lab, color=c, fontsize=FS_ANNOT)
# parallelogram at image
ax.plot([Tu[0], Tuv[0]], [Tu[1], Tuv[1]], "--", color="0.4", lw=1.2)
ax.plot([Tv[0], Tuv[0]], [Tv[1], Tuv[1]], "--", color="0.4", lw=1.2)
ax.set_title("Linearity: grids stay parallelograms", fontsize=FS_AX_TITLE)
ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)
ax.set_xlim(-0.5, 3.2)
ax.set_ylim(-0.5, 3.2)
ax.text(
    0.02, 0.98,
    r"$T(u+v)=T(u)+T(v)$" + "\n"
    r"$T(c u)=c\,T(u)$" + "\n\n"
    "No bending into curves.\n"
    "Origin stays fixed.",
    transform=ax.transAxes, va="top", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)

ax = axes2[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("What linearity forbids / allows", fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.40,
    "Allowed (linear):\n"
    "  rotation, scaling, shear, reflection,\n"
    "  projection onto a line/plane, squeezing rank...\n\n"
    "Forbidden (nonlinear):\n"
    "  translation  x |-> x + b     (unless you use affine = linear + bias)\n"
    "  ReLU, products, norms, attention softmax, ...\n\n"
    "Neural nets = many LINEAR maps interleaved with\n"
    "nonlinearities. Understanding A x is the linear half.",
    transform=ax.transAxes, ha="center", va="center", fontsize=FS_AXIS - 1,
    bbox=dict(boxstyle="round", facecolor="#f7f7ff", alpha=0.96),
)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Why a matrix? Columns are images of basis vectors
# -------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig3.suptitle(
    r"Step 3: $T$ is recorded by a matrix $A$ whose columns are $T(e_j)$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

A = np.array([[1.4, -0.5], [0.6, 1.1]], dtype=float)
e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
Te1, Te2 = A @ e1, A @ e2  # = columns of A

ax = axes3[0]
ax.set_aspect("equal")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.axhline(0, color="0.5", lw=0.7)
ax.axvline(0, color="0.5", lw=0.7)
ax.grid(True, alpha=0.3)
for line in unit_grid(np.eye(2), lim=2.0, n=9):
    ax.plot(line[:, 0], line[:, 1], color="0.8", lw=0.8, zorder=1)
draw_basis(ax, (0, 0), e1, e2, labels=(r"$e_1$", r"$e_2$"))
ax.set_title("Source $\\mathbb{R}^2$: choose the standard basis", fontsize=FS_AX_TITLE)
ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)
ax.text(
    0.02, 0.02,
    "To know T everywhere it is\n"
    "ENOUGH to know T(e_j):\n"
    r"$T(\sum x_j e_j)=\sum x_j T(e_j)$.",
    transform=ax.transAxes, va="bottom", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)

ax = axes3[1]
ax.set_aspect("equal")
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.axhline(0, color="0.5", lw=0.7)
ax.axvline(0, color="0.5", lw=0.7)
ax.grid(True, alpha=0.3)
for line in unit_grid(A, lim=2.0, n=9):
    ax.plot(line[:, 0], line[:, 1], color="0.75", lw=0.9, zorder=1)
draw_basis(ax, (0, 0), Te1, Te2, colors=("crimson", "navy"),
           labels=(r"$T(e_1)=$ col 1", r"$T(e_2)=$ col 2"))
# also show target standard basis lightly
draw_basis(ax, (0, 0), (1, 0), (0, 1), colors=("0.6", "0.6"),
           labels=(r"$f_1$", r"$f_2$"), scale=0.85)
ax.set_title("Target: store those images as columns of A", fontsize=FS_AX_TITLE)
ax.set_xlabel(r"$y_1$", fontsize=FS_AXIS)
ax.set_ylabel(r"$y_2$", fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)
ax.text(
    0.02, 0.02,
    r"$A=[\,T(e_1)\ |\ T(e_2)\,]$" + "\n"
    f"= [[{A[0, 0]:.1f}, {A[0, 1]:.1f}],\n"
    f"    [{A[1, 0]:.1f}, {A[1, 1]:.1f}]]\n\n"
    "Grey grid = image of the\n"
    "unit grid under T.",
    transform=ax.transAxes, va="bottom", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="#eef5ff", alpha=0.95),
)
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Meaning of a_ij from source and target bases
# -------------------------------------------------------
fig4, axes4 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig4.suptitle(
    r"Step 4: What each entry $a_{ij}$ means (source basis $e_j$, target basis $f_i$)",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

ax = axes4[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Dictionary for $a_{ij}$", fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.55,
    r"Write the image of source basis vector $e_j$ in the target basis:"
    "\n\n"
    r"    $T(e_j)=\sum_{i=1}^{m} a_{ij}\, f_i$"
    "\n\n"
    r"So $a_{ij}$ is:"
    "\n"
    "  • from the SOURCE side: how much $e_j$ contributes after T\n"
    "    to target direction $f_i$\n"
    "  • from the TARGET side: the $i$-th coordinate of the vector $T(e_j)$\n\n"
    r"Column $j$ = all target coordinates of $T(e_j)$."
    "\n"
    r"Row $i$ = how every source basis vector feeds coordinate $i$.",
    transform=ax.transAxes, ha="center", va="center", fontsize=FS_AXIS - 1,
    bbox=dict(boxstyle="round", facecolor="#eef5ff", alpha=0.96),
)

# Right: concrete callout for one entry of A
ax = axes4[1]
ax.set_aspect("equal")
ax.set_xlim(-0.4, 2.2)
ax.set_ylim(-0.4, 2.2)
ax.axhline(0, color="0.5", lw=0.7)
ax.axvline(0, color="0.5", lw=0.7)
ax.grid(True, alpha=0.3)
# Highlight T(e1) and its coordinates
ax.quiver(0, 0, Te1[0], Te1[1], angles="xy", scale_units="xy", scale=1.0,
          color="crimson", width=0.015, zorder=4)
ax.plot([Te1[0], Te1[0]], [0, Te1[1]], "--", color="navy", lw=1.5)
ax.plot([0, Te1[0]], [Te1[1], Te1[1]], "--", color="navy", lw=1.5)
ax.scatter([Te1[0]], [Te1[1]], s=60, c="crimson", zorder=5)
ax.text(Te1[0] + 0.08, Te1[1] + 0.08, r"$T(e_1)$", color="crimson",
        fontsize=FS_ANNOT, fontweight="bold")
ax.text(Te1[0] / 2, -0.18, rf"$a_{{11}}={A[0, 0]:.1f}$", color="navy",
        fontsize=FS_ANNOT, ha="center")
ax.text(-0.22, Te1[1] / 2, rf"$a_{{21}}={A[1, 0]:.1f}$", color="navy",
        fontsize=FS_ANNOT, va="center", rotation=90)
draw_basis(ax, (0, 0), (1, 0), (0, 1), colors=("0.5", "0.5"),
           labels=(r"$f_1$", r"$f_2$"), scale=0.9)
ax.set_title(r"Example: column 1 $= T(e_1)=(a_{11}, a_{21})$", fontsize=FS_AX_TITLE)
ax.set_xlabel(r"$y_1$", fontsize=FS_AXIS)
ax.set_ylabel(r"$y_2$", fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)
ax.text(
    0.98, 0.50,
    r"$T(e_1)=a_{11} f_1 + a_{21} f_2$" + "\n"
    "Each dashed length is one entry.",
    transform=ax.transAxes, va="center", ha="right", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Why A x has its formula
# -------------------------------------------------------
fig5, axes5 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig5.suptitle(
    r"Step 5: Why $Ax$ has that formula -- expand $x$ in the source basis",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

x_vec = np.array([1.5, 0.9])
Ax = A @ x_vec

ax = axes5[0]
ax.set_aspect("equal")
ax.set_xlim(-0.5, 2.8)
ax.set_ylim(-0.5, 2.8)
ax.axhline(0, color="0.5", lw=0.7)
ax.axvline(0, color="0.5", lw=0.7)
ax.grid(True, alpha=0.3)
# x = x1 e1 + x2 e2
ax.quiver(0, 0, x_vec[0], 0, angles="xy", scale_units="xy", scale=1.0,
          color="crimson", width=0.012, zorder=3)
ax.quiver(x_vec[0], 0, 0, x_vec[1], angles="xy", scale_units="xy", scale=1.0,
          color="navy", width=0.012, zorder=3)
ax.quiver(0, 0, x_vec[0], x_vec[1], angles="xy", scale_units="xy", scale=1.0,
          color="k", width=0.014, zorder=4)
ax.text(x_vec[0] / 2, -0.2, rf"$x_1 e_1$", color="crimson", fontsize=FS_ANNOT, ha="center")
ax.text(x_vec[0] + 0.1, x_vec[1] / 2, rf"$x_2 e_2$", color="navy", fontsize=FS_ANNOT)
ax.text(x_vec[0] + 0.08, x_vec[1] + 0.08, r"$x$", fontsize=FS_ANNOT, fontweight="bold")
ax.set_title(r"Source: $x=x_1 e_1 + x_2 e_2$", fontsize=FS_AX_TITLE)
ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)
ax.text(
    0.02, 0.98,
    rf"$x=({x_vec[0]:.1f},\ {x_vec[1]:.1f})$",
    transform=ax.transAxes, va="top", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
)

ax = axes5[1]
ax.set_aspect("equal")
ax.set_xlim(-0.5, 3.2)
ax.set_ylim(-0.5, 3.2)
ax.axhline(0, color="0.5", lw=0.7)
ax.axvline(0, color="0.5", lw=0.7)
ax.grid(True, alpha=0.3)
# Ax = x1 T(e1) + x2 T(e2)
p1 = x_vec[0] * Te1
p2 = x_vec[1] * Te2
ax.quiver(0, 0, p1[0], p1[1], angles="xy", scale_units="xy", scale=1.0,
          color="crimson", width=0.012, zorder=3)
ax.quiver(p1[0], p1[1], p2[0], p2[1], angles="xy", scale_units="xy", scale=1.0,
          color="navy", width=0.012, zorder=3)
ax.quiver(0, 0, Ax[0], Ax[1], angles="xy", scale_units="xy", scale=1.0,
          color="k", width=0.014, zorder=4)
ax.text(p1[0] * 0.5, p1[1] * 0.5 - 0.2, r"$x_1 T(e_1)$", color="crimson",
        fontsize=FS_ANNOT, ha="center")
ax.text(p1[0] + p2[0] * 0.5 + 0.1, p1[1] + p2[1] * 0.5, r"$x_2 T(e_2)$",
        color="navy", fontsize=FS_ANNOT)
ax.text(Ax[0] + 0.08, Ax[1] + 0.08, r"$Ax=T(x)$", fontsize=FS_ANNOT, fontweight="bold")
ax.set_title(r"Target: $T(x)=x_1 T(e_1)+x_2 T(e_2)=Ax$", fontsize=FS_AX_TITLE)
ax.set_xlabel(r"$y_1$", fontsize=FS_AXIS)
ax.set_ylabel(r"$y_2$", fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)
ax.text(
    0.02, 0.98,
    "Matrix-vector multiply is just\n"
    "\"scale each column by $x_j$\n"
    "and add the columns\".\n\n"
    r"$(Ax)_i=\sum_j a_{ij} x_j$",
    transform=ax.transAxes, va="top", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="#eef5ff", alpha=0.95),
)
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Diversity of linear maps (gallery)
# -------------------------------------------------------
fig6, axes6 = plt.subplots(2, 3, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig6.suptitle(
    r"Step 6: How diverse is a linear map?  (still just some matrix $A$)",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

gallery = [
    (np.array([[0.0, -1.0], [1.0, 0.0]]), "rotation 90°"),
    (np.array([[1.6, 0.0], [0.0, 0.55]]), "anisotropic scale"),
    (np.array([[1.0, 0.9], [0.0, 1.0]]), "shear"),
    (np.array([[1.0, 0.0], [0.0, -1.0]]), "reflect over x-axis"),
    (np.array([[0.8, 0.0], [0.0, 0.0]]), "project to x-axis (rank 1)"),
    (np.array([[0.7, 0.5], [0.35, 0.25]]), "squash (near rank 1)"),
]

theta = np.linspace(0, 2 * np.pi, 200)
circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
for ax, (M, title) in zip(axes6.ravel(), gallery):
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.grid(True, alpha=0.25)
    for line in unit_grid(np.eye(2), lim=1.6, n=7):
        ax.plot(line[:, 0], line[:, 1], color="0.85", lw=0.7, zorder=1)
    for line in unit_grid(M, lim=1.6, n=7):
        ax.plot(line[:, 0], line[:, 1], color="steelblue", lw=1.0, alpha=0.8, zorder=2)
    ax.plot(circle[:, 0], circle[:, 1], color="0.5", lw=1.2, ls="--", label="unit circle")
    img = circle @ M.T
    ax.plot(img[:, 0], img[:, 1], color="crimson", lw=2.2, label="image ellipse")
    draw_basis(ax, (0, 0), M[:, 0], M[:, 1],
               labels=(r"$T(e_1)$", r"$T(e_2)$"), scale=1.0)
    ax.set_title(title, fontsize=FS_AX_TITLE - 2)
    ax.tick_params(labelsize=FS_TICK - 1)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])

axes6[0, 0].legend(fontsize=FS_LEGEND - 2, loc="upper right")
fig6.text(
    0.5, 0.02,
    "All of these are linear. Diversity = which parallelogram the unit square becomes "
    "(= which pair of column vectors you pick).",
    ha="center", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)
fig6.tight_layout(rect=[0, 0.05, 1, 0.94])
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: What a linear map CANNOT do (only linear) -- three large demos
# -------------------------------------------------------
fig7c, axes7c = plt.subplots(1, 3, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig7c.suptitle(
    r"Step 7: What linearity forbids (visual demos)",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

# Left: Cannot translate / move the origin
ax = axes7c[0]
ax.set_aspect("equal")
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.axhline(0, color="0.5", lw=0.7)
ax.axvline(0, color="0.5", lw=0.7)
ax.grid(True, alpha=0.3)
ax.scatter([0], [0], s=90, c="k", zorder=4, label="origin")
A_lin = np.array([[1.1, 0.3], [0.2, 0.9]])
pts0 = np.array([[1.2, 0.4], [0.5, 1.3], [-1.0, 0.8]])
for p in pts0:
    q = A_lin @ p
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="->", color="#1a7f4b", lw=2.2))
    ax.scatter([p[0]], [p[1]], s=36, c="0.45", zorder=3)
    ax.scatter([q[0]], [q[1]], s=36, c="#1a7f4b", zorder=3)
b = np.array([0.9, 0.7])
ax.scatter([b[0]], [b[1]], s=80, c="crimson", marker="x", zorder=5)
ax.text(b[0] + 0.1, b[1] + 0.1, "want origin here?", color="crimson", fontsize=FS_ANNOT)
ax.set_title("Cannot move the origin", fontsize=FS_AX_TITLE)
ax.tick_params(labelsize=FS_TICK)
ax.text(
    0.02, 0.02,
    r"$T(0)=0$ always." + "\n"
    "Need bias / affine for shifts.",
    transform=ax.transAxes, va="bottom", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)

# Middle: Cannot bend a straight line into a curve
ax = axes7c[1]
ax.set_aspect("equal")
ax.set_xlim(-0.3, 3.2)
ax.set_ylim(-1.5, 2.2)
ax.axhline(0, color="0.5", lw=0.7)
ax.axvline(0, color="0.5", lw=0.7)
ax.grid(True, alpha=0.3)
t = np.linspace(0.0, 2.8, 40)
line = np.stack([t, 0.35 * t], axis=1)
A_shear = np.array([[1.0, 0.5], [0.0, 1.0]])
line_lin = line @ A_shear.T
line_curve = np.stack([t, 0.15 * t ** 2], axis=1)
ax.plot(line[:, 0], line[:, 1], color="0.35", lw=2.4, label="straight line")
ax.plot(line_lin[:, 0], line_lin[:, 1], color="#1a7f4b", lw=2.6, label="linear image (still straight)")
ax.plot(line_curve[:, 0], line_curve[:, 1], color="crimson", lw=2.6, ls="--", label="nonlinear wish (curve)")
ax.set_title("Cannot bend lines into curves", fontsize=FS_AX_TITLE)
ax.tick_params(labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc="upper left")
ax.text(
    0.98, 0.02,
    "Linear images of lines\n"
    "are lines (or points).\n"
    "Curves need nonlinearity.",
    transform=ax.transAxes, va="bottom", ha="right", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="#eef5ff", alpha=0.95),
)

# Right: Cannot apply different rules in different regions (thick flow)
ax = axes7c[2]
ax.set_aspect("equal")
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.2, 2.2)
ax.axhline(0, color="0.5", lw=0.7)
ax.axvline(0, color="0.5", lw=0.7)
ax.grid(True, alpha=0.3)
A_flow = np.array([[0.85, 0.35], [0.15, 1.15]], dtype=float)
xx = np.linspace(-1.8, 1.8, 7)
yy = np.linspace(-1.8, 1.8, 7)
for x0 in xx:
    for y0 in yy:
        p = np.array([x0, y0], dtype=float)
        q = A_flow @ p
        ax.annotate(
            "", xy=q, xytext=p,
            arrowprops=dict(
                arrowstyle="->",
                color="#0b3d91",  # thick deep blue
                lw=2.4,
                alpha=0.85,
                mutation_scale=12,
            ),
        )
        ax.scatter([p[0]], [p[1]], s=22, c="#0b3d91", zorder=3)
ax.text(
    0.02, 0.98,
    "One matrix = ONE rule\n"
    "everywhere.\n"
    "Cannot stretch only the\n"
    "right half, or gate only\n"
    "when x>0 (that is ReLU).",
    transform=ax.transAxes, va="top", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)
ax.set_title("Cannot use different local rules", fontsize=FS_AX_TITLE)
ax.tick_params(labelsize=FS_TICK)

fig7c.tight_layout(rect=[0, 0, 1, 0.93])
show_maximized(fig7c)


# -------------------------------------------------------
# FIGURE 8: Why we still need nonlinear layers (text step)
# -------------------------------------------------------
fig7d, ax7d = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig7d.suptitle(
    "Step 8: So why stack nonlinearities after linear maps?",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
ax7d.axis("off")
ax7d.text(
    0.5, 0.52,
    "A single linear map is powerful -- and also rigid.\n\n"
    "From Step 7, it cannot:\n"
    "  • translate (move the origin) by itself\n"
    "  • bend straight lines into curves\n"
    "  • treat different regions with different rules\n"
    "  • implement soft gating / thresholds / products\n\n"
    "That is why deep nets interleave linear maps (A x)\n"
    "with nonlinearities (ReLU, GELU, softmax, ...):\n"
    "linear pieces move / mix coordinates; nonlinear\n"
    "pieces create the bends and decisions.\n\n"
    "Attention softmax (250) and residual MLPs are exactly\n"
    "those missing ingredients after W_Q, W_K, W_V.",
    transform=ax7d.transAxes, ha="center", va="center", fontsize=FS_AXIS,
    bbox=dict(boxstyle="round", facecolor="#f7f7ff", alpha=0.96),
)
show_maximized(fig7d)


# -------------------------------------------------------
# FIGURE 9: R^n -> R^m  (change of dimension)
# -------------------------------------------------------
fig7, axes7 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig7.suptitle(
    r"Step 9: Maps $\mathbb{R}^n\to\mathbb{R}^m$ -- tall / wide / square matrices",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

ax = axes7[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Shape of A is m x n", fontsize=FS_AX_TITLE)


def tensor_box(ax, x, y, w, h, text, color):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor="k", lw=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=FS_ANNOT, fontweight="bold")


tensor_box(ax, 0.5, 3.0, 2.2, 2.2, "$x\\in\\mathbb{R}^n$\n(n x 1)", "#dceaff")
tensor_box(ax, 3.5, 2.4, 2.8, 3.4, "$A\\in\\mathbb{R}^{m\\times n}$\nm rows, n cols", "#fff3cd")
tensor_box(ax, 7.2, 2.6, 2.4, 3.0, "$Ax\\in\\mathbb{R}^m$\n(m x 1)", "#d4edda")
ax.annotate("", xy=(3.5, 4.1), xytext=(2.7, 4.1),
            arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(7.2, 4.1), xytext=(6.3, 4.1),
            arrowprops=dict(arrowstyle="->", lw=2))
ax.text(
    0.5, 0.2,
    "n columns  <->  n source basis images T(e_j)\n"
    "m rows     <->  m target coordinates per image",
    transform=ax.transAxes, fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)

ax = axes7[1]
# Visual: R^2 -> R^3 embedding and R^3 -> R^2 projection as sketches
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Dimension can change", fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.55,
    r"• Tall A (m>n): embed $\mathbb{R}^n$ into a higher space"
    "\n"
    "    (more coordinates; columns live in R^m).\n\n"
    r"• Wide A (m<n): compress / project $\mathbb{R}^n$ to $\mathbb{R}^m$"
    "\n"
    "    (must lose some directions if m<n).\n\n"
    r"• Square A (m=n): can be invertible (basis to basis)"
    "\n"
    "    or singular (collapse some direction).\n\n"
    "In Transformers, W_Q, W_K, W_V are exactly such matrices:\n"
    "shared linear maps R^{d_model} -> R^{d_k} (or d_v).",
    transform=ax.transAxes, ha="center", va="center", fontsize=FS_AXIS - 1,
    bbox=dict(boxstyle="round", facecolor="#f7f7ff", alpha=0.96),
)
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 9: 2D grid -- each source line shares color with its image
# -------------------------------------------------------
A2 = np.array([
    [1.25, 0.55],
    [-0.30, 0.95],
], dtype=float)

grid_lines_2 = grid_lines_2d(lim=1.6, n=3, n_pts=50)
line_colors_2 = plt.cm.hsv(np.linspace(0.0, 0.85, len(grid_lines_2)))

fig8b, axes8b = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig8b.suptitle(
    r"Step 10: 2D grid under $T(x)=A_2 x$ -- paired lines share a color",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

ax_src2, ax_img2 = axes8b
for (pts, _fam, _i), color in zip(grid_lines_2, line_colors_2):
    ax_src2.plot(pts[:, 0], pts[:, 1], color=color, lw=2.0, alpha=0.95)
    img = pts @ A2.T
    ax_img2.plot(img[:, 0], img[:, 1], color=color, lw=2.0, alpha=0.95)

# Basis images on the right
draw_basis(
    ax_img2, (0, 0), A2[:, 0], A2[:, 1],
    colors=("#111111", "#444444"),
    labels=(r"$T(e_1)$", r"$T(e_2)$"),
)

all_img2 = np.concatenate([pts @ A2.T for pts, _, _ in grid_lines_2], axis=0)
lo2 = float(min(-2.0, all_img2.min() - 0.3))
hi2 = float(max(2.0, all_img2.max() + 0.3))
for ax, title in (
    (ax_src2, r"Source grid in $\mathbb{R}^2$"),
    (ax_img2, r"Image grid $A_2 X$ (same colors = paired lines)"),
):
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.5", lw=0.7)
    ax.axvline(0, color="0.5", lw=0.7)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(lo2, hi2)
    ax.set_ylim(lo2, hi2)

fig8b.text(
    0.5, 0.02,
    "Columns of A2 are T(e1), T(e2).  "
    f"A2 = {np.array2string(A2, precision=2, suppress_small=True)}\n"
    "Each colored line on the left maps to the line of the SAME color on the right.",
    ha="center", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="#eef5ff", alpha=0.96),
)
fig8b.tight_layout(rect=[0, 0.07, 1, 0.93])
show_maximized(fig8b)


# -------------------------------------------------------
# FIGURE 10: Interactive 2D -- slide A_ij and watch the image grid change
# -------------------------------------------------------
A2_live = A2.copy()

fig9b = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig9b.suptitle(
    r"Step 11: Drag $a_{ij}$ sliders (2D) -- watch the target grid follow $T(x)=A x$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
fig9b.subplots_adjust(left=0.06, right=0.98, bottom=0.34, top=0.90, wspace=0.15)

ax2_src = fig9b.add_subplot(1, 2, 1)
ax2_img = fig9b.add_subplot(1, 2, 2)

img_artists_2d = []
for (pts, _fam, _i), color in zip(grid_lines_2, line_colors_2):
    ax2_src.plot(pts[:, 0], pts[:, 1], color=color, lw=2.0, alpha=0.95)
    img = pts @ A2_live.T
    (ln,) = ax2_img.plot(img[:, 0], img[:, 1], color=color, lw=2.0, alpha=0.95)
    img_artists_2d.append(ln)

basis_artists_2d = []


def _draw_basis_2d(A):
    global basis_artists_2d
    for art in basis_artists_2d:
        try:
            art.remove()
        except Exception:
            pass
    basis_artists_2d = []
    for j, (c, lab) in enumerate(zip(("crimson", "navy"), (r"$T(e_1)$", r"$T(e_2)$"))):
        col = A[:, j]
        qv = ax2_img.quiver(
            0, 0, col[0], col[1],
            angles="xy", scale_units="xy", scale=1.0,
            color=c, width=0.012, zorder=5,
        )
        txt = ax2_img.text(
            col[0] * 1.08, col[1] * 1.08, lab,
            color=c, fontsize=FS_ANNOT, fontweight="bold",
        )
        basis_artists_2d.extend([qv, txt])


_draw_basis_2d(A2_live)

for ax, title in (
    (ax2_src, r"Source grid (fixed)"),
    (ax2_img, r"Image grid $A X$ (updates live)"),
):
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.5", lw=0.7)
    ax.axvline(0, color="0.5", lw=0.7)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(lo2, hi2)
    ax.set_ylim(lo2, hi2)

matrix_hud_2d = fig9b.text(
    0.5, 0.30, "",
    ha="center", va="top", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="#eef5ff", alpha=0.96),
)

sliders_2d = []
labels_2d = [[rf"$a_{{{i+1}{j+1}}}$" for j in range(2)] for i in range(2)]
for i in range(2):
    for j in range(2):
        left = 0.18 + j * 0.40
        bottom = 0.10 - i * 0.055
        ax_sl = fig9b.add_axes([left, bottom, 0.28, 0.035])
        sl = Slider(
            ax_sl, labels_2d[i][j], -2.0, 2.0,
            valinit=float(A2_live[i, j]), valstep=0.05,
        )
        sliders_2d.append(sl)


def _A2_from_sliders():
    vals = [sl.val for sl in sliders_2d]
    return np.array(vals, dtype=float).reshape(2, 2)


def _update_live_2d(_=None):
    A = _A2_from_sliders()
    for (pts, _fam, _i), ln in zip(grid_lines_2, img_artists_2d):
        img = pts @ A.T
        ln.set_data(img[:, 0], img[:, 1])
    _draw_basis_2d(A)
    # Expand limits a bit if needed
    imgs = np.concatenate([pts @ A.T for pts, _, _ in grid_lines_2], axis=0)
    pad = 0.4
    lim0 = float(min(-2.0, imgs.min() - pad))
    lim1 = float(max(2.0, imgs.max() + pad))
    ax2_img.set_xlim(lim0, lim1)
    ax2_img.set_ylim(lim0, lim1)
    matrix_hud_2d.set_text(
        "Live matrix A (columns = T(e1), T(e2))\n"
        f"{np.array2string(A, precision=2, suppress_small=True)}"
    )
    fig9b.canvas.draw_idle()


for sl in sliders_2d:
    sl.on_changed(_update_live_2d)
_update_live_2d()

print("Step 11: move the 2D a_ij sliders to reshape the image grid.")
show_maximized(fig9b)


# -------------------------------------------------------
# FIGURE 11: 3D browsable grid -- each source line shares color with its image
# -------------------------------------------------------
# Specific invertible 3x3 map (rotation-ish + anisotropic scale + mild shear).
A3 = np.array([
    [1.15, 0.45, 0.10],
    [-0.20, 0.95, 0.35],
    [0.15, -0.25, 1.25],
], dtype=float)

grid_lines = grid_lines_3d(lim=1.4, n=3, n_pts=30)
# Distinct color per line; same color used for that line's image under A3.
line_colors = plt.cm.hsv(np.linspace(0.0, 0.85, len(grid_lines)))

fig8 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig8.suptitle(
    r"Step 12: Browse $T(x)=A_3 x$ in 3D -- paired grid lines share a color",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

ax_src = fig8.add_subplot(1, 2, 1, projection="3d")
ax_img = fig8.add_subplot(1, 2, 2, projection="3d")

for (pts, _fam, _i), color in zip(grid_lines, line_colors):
    ax_src.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=1.6, alpha=0.95)
    img = pts @ A3.T
    ax_img.plot(img[:, 0], img[:, 1], img[:, 2], color=color, lw=1.6, alpha=0.95)

# Draw images of the standard basis (columns of A3) on the right
o = np.zeros(3)
basis_cols = ["#111111", "#333333", "#555555"]
basis_labs = [r"$T(e_1)$", r"$T(e_2)$", r"$T(e_3)$"]
for j in range(3):
    col = A3[:, j]
    ax_img.quiver(
        o[0], o[1], o[2], col[0], col[1], col[2],
        color=basis_cols[j], linewidth=2.2, arrow_length_ratio=0.12,
    )
    ax_img.text(col[0] * 1.05, col[1] * 1.05, col[2] * 1.05, basis_labs[j],
                fontsize=FS_ANNOT, color=basis_cols[j])

for ax, title in (
    (ax_src, r"Source grid in $\mathbb{R}^3$ (domain $X$)"),
    (ax_img, r"Image grid $A_3 X$ (same colors = paired lines)"),
):
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$y$", fontsize=FS_AXIS)
    ax.set_zlabel(r"$z$", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.view_init(elev=22, azim=-58)

# Match axis limits roughly for a fair browse
all_img = np.concatenate([pts @ A3.T for pts, _, _ in grid_lines], axis=0)
lo = float(min(-1.6, all_img.min() - 0.2))
hi = float(max(1.6, all_img.max() + 0.2))
for ax in (ax_src, ax_img):
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_zlim(lo, hi)

fig8.text(
    0.5, 0.02,
    "Columns of A3 are T(e1), T(e2), T(e3).  "
    f"A3 = {np.array2string(A3, precision=2, suppress_small=True)}\n"
    "Each colored polyline on the left maps to the polyline of the SAME color on the right.",
    ha="center", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="#eef5ff", alpha=0.96),
)
fig8.tight_layout(rect=[0, 0.07, 1, 0.93])
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 9: Interactive -- slide A_ij and watch the image grid change
# -------------------------------------------------------
A3_live = A3.copy()
grid_lines_live = grid_lines  # same 3x3x3 axis-aligned grid as Step 8
line_colors_live = line_colors

fig9 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig9.suptitle(
    r"Step 13: Drag $a_{ij}$ sliders (3D) -- watch the target grid follow $T(x)=A x$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

# Compact bottom band: one-line matrix readout + 9 sliders (axes stay large)
fig9.subplots_adjust(left=0.05, right=0.98, bottom=0.26, top=0.90, wspace=0.12)
ax9_src = fig9.add_subplot(1, 2, 1, projection="3d")
ax9_img = fig9.add_subplot(1, 2, 2, projection="3d")

src_artists = []
img_artists = []
for (pts, _fam, _i), color in zip(grid_lines_live, line_colors_live):
    (ln_s,) = ax9_src.plot(
        pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=1.6, alpha=0.95,
    )
    img = pts @ A3_live.T
    (ln_i,) = ax9_img.plot(
        img[:, 0], img[:, 1], img[:, 2], color=color, lw=1.6, alpha=0.95,
    )
    src_artists.append(ln_s)
    img_artists.append(ln_i)

# Basis arrows on the image side (recreated on each update)
basis_artists = []


def _draw_basis_arrows(A):
    global basis_artists
    for art in basis_artists:
        try:
            art.remove()
        except Exception:
            pass
    basis_artists = []
    o = np.zeros(3)
    for j, (c, lab) in enumerate(zip(["#111111", "#333333", "#555555"],
                                       [r"$T(e_1)$", r"$T(e_2)$", r"$T(e_3)$"])):
        col = A[:, j]
        qv = ax9_img.quiver(
            o[0], o[1], o[2], col[0], col[1], col[2],
            color=c, linewidth=2.0, arrow_length_ratio=0.12,
        )
        txt = ax9_img.text(
            col[0] * 1.05, col[1] * 1.05, col[2] * 1.05, lab,
            fontsize=FS_ANNOT - 1, color=c,
        )
        basis_artists.extend([qv, txt])


_draw_basis_arrows(A3_live)

for ax, title in (
    (ax9_src, r"Source grid (fixed)"),
    (ax9_img, r"Image grid $A X$ (updates live)"),
):
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.set_xlabel(r"$x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$y$", fontsize=FS_AXIS)
    ax.set_zlabel(r"$z$", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.view_init(elev=22, azim=-58)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_zlim(lo, hi)

# Flat one-line matrix readout (short height so 3D axes stay large)
matrix_hud = fig9.text(
    0.5, 0.225,
    "",
    ha="center", va="center", fontsize=FS_ANNOT - 1,
    bbox=dict(boxstyle="round", facecolor="#eef5ff", alpha=0.96, pad=0.3),
)

# Build 3x3 slider grid (tight band under the flat HUD)
slider_axes = []
sliders = []
slider_labels = [[rf"$a_{{{i+1}{j+1}}}$" for j in range(3)] for i in range(3)]
for i in range(3):
    for j in range(3):
        left = 0.08 + j * 0.30
        bottom = 0.12 - i * 0.045
        ax_sl = fig9.add_axes([left, bottom, 0.22, 0.028])
        sl = Slider(
            ax_sl, slider_labels[i][j], -2.0, 2.0,
            valinit=float(A3_live[i, j]), valstep=0.05,
        )
        slider_axes.append(ax_sl)
        sliders.append(sl)


def _A_from_sliders():
    vals = [sl.val for sl in sliders]
    return np.array(vals, dtype=float).reshape(3, 3)


def _update_live(_=None):
    A = _A_from_sliders()
    for (pts, _fam, _i), ln in zip(grid_lines_live, img_artists):
        img = pts @ A.T
        ln.set_data_3d(img[:, 0], img[:, 1], img[:, 2])
    _draw_basis_arrows(A)
    # Single short line -- no tall multi-line box eating axis space
    flat = np.array2string(A, precision=2, separator=", ", suppress_small=True)
    flat = " ".join(flat.split())
    matrix_hud.set_text(f"A (cols = T(e1), T(e2), T(e3)):  {flat}")
    fig9.canvas.draw_idle()


for sl in sliders:
    sl.on_changed(_update_live)
_update_live()

print("Step 13: move the 3D a_ij sliders; drag the 3D views to rotate.")
show_maximized(fig9)


# -------------------------------------------------------
# FIGURE 13: Absorbing fixed linear preprocessing into a learned map
# -------------------------------------------------------
# Customer-facing point: linear maps are closed under composition, and rich
# enough that a fixed linear preprocess P can be absorbed into learned W.
# -------------------------------------------------------
P = np.array([[1.4, 0.0], [0.0, 0.55]], dtype=float)   # fixed preprocess (e.g. rescale)
W = np.array([[0.9, 0.4], [-0.3, 1.2]], dtype=float)  # stand-in for a learned layer
Wp = W @ P                                            # what training can learn instead

fig10, axes10 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig10.suptitle(
    r"Step 14: Fixed linear preprocessing is free -- a learned map can absorb it",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

# Left: two-stage pipeline vs one absorbed map -- same outputs on sample points
ax = axes10[0]
ax.set_aspect("equal")
ax.set_xlim(-2.8, 2.8)
ax.set_ylim(-2.8, 2.8)
ax.axhline(0, color="0.5", lw=0.7)
ax.axvline(0, color="0.5", lw=0.7)
ax.grid(True, alpha=0.3)
pts = RNG.normal(size=(14, 2)) * 0.9
for p in pts:
    y_two = W @ (P @ p)
    y_one = Wp @ p
    ax.scatter([p[0]], [p[1]], s=35, c="0.55", zorder=2)
    ax.scatter([y_two[0]], [y_two[1]], s=55, c="crimson", marker="o", zorder=3, alpha=0.85)
    ax.scatter([y_one[0]], [y_one[1]], s=70, facecolors="none", edgecolors="navy",
               linewidths=1.6, marker="o", zorder=4)
ax.scatter([], [], c="0.55", s=35, label=r"input $x$")
ax.scatter([], [], c="crimson", s=55, label=r"two-step $W(Px)$")
ax.scatter([], [], facecolors="none", edgecolors="navy", s=70, label=r"one-step $W'x$, $W'=WP$")
ax.set_title("Same outputs either way", fontsize=FS_AX_TITLE)
ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc="upper left")
ax.text(
    0.98, 0.02,
    "Red filled = preprocess then learn.\n"
    "Blue rings = one learned matrix\n"
    "that already includes preprocess.\n"
    "They coincide: $W(Px)=(WP)x$.",
    transform=ax.transAxes, va="bottom", ha="right", fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)

# Right: plain-language note for customers / lecture audience
ax = axes10[1]
ax.axis("off")
ax.set_title("What this means in practice", fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.52,
    "Because linear maps are so flexible -- and closed under composition --\n"
    "a FIXED linear change of the input can be absorbed into the FIRST\n"
    "learned linear layer without reducing what the network can represent.\n\n"
    "In formulas:\n"
    "    preprocess:   z = P x          (fixed; e.g. rescale / whiten / PCA)\n"
    "    learned map:  y = W z = W (P x)\n"
    "    equivalent:   y = W' x  with  W' = W P\n\n"
    "Training never needs to \"see\" P separately: it can simply learn W'.\n"
    "So if your network already contains learned linear maps (it always does:\n"
    "Dense / Conv / embedding / W_Q,W_K,W_V / ...), you do not need to worry\n"
    "that a fixed LINEAR preprocessing will lock the model out of a good fit.\n"
    "Any such P is just a change of coordinates the learned map can undo or\n"
    "re-absorb.\n\n"
    "Caveats (be honest with customers):\n"
    "  • Nonlinear preprocessing (clipping, log1p, tokenization quirks, ...)\n"
    "    is NOT automatically absorbed this way.\n"
    "  • Affine preprocess x |-> Px+b needs a bias term (still standard).\n"
    "  • Numerics / conditioning can still favor a sensible P in practice,\n"
    "    even when representation power is unchanged in exact arithmetic.",
    transform=ax.transAxes, ha="center", va="center", fontsize=FS_AXIS - 1,
    bbox=dict(boxstyle="round", facecolor="#f7f7ff", alpha=0.96),
)
show_maximized(fig10)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("LINEAR TRANSFORMATIONS AND MATRICES")
print("=" * 60)
print(rf"""
Linear space R^n:
  points are n-tuples (x_1,...,x_n); you can add them and scale by reals.
  Standard basis: e_j = one-hot at position j.
  Every vector: x = sum_j x_j e_j.

Linear map T: R^n -> R^m:
  T(u+v)=T(u)+T(v),  T(c u)=c T(u).
  Fixes the origin; sends parallelograms to parallelograms.

Why a matrix A:
  It is enough to know T(e_1),...,T(e_n).
  Stack those images as COLUMNS in the target basis {{f_i}}:
      T(e_j) = sum_i a_ij f_i
  That array A = (a_ij) is the matrix of T.
  Then for every x,
      T(x) = sum_j x_j T(e_j) = A x
  and the i-th output coordinate is
      (A x)_i = sum_j a_ij x_j
  i.e. "scale columns by x_j and add".

Meaning of a_ij:
  source view: contribution of basis direction e_j into target axis f_i after T
  target view: i-th coordinate of the vector T(e_j)

Diversity:
  rotations, scales, shears, reflections, projections, squeezes...
  all are just different choices of the column vectors T(e_j).

Cannot do (Step 7) -- only linear:
  move the origin (pure translation), bend lines into curves,
  use different rules in different regions, soft thresholds / products.
  That is why nets interleave linear maps with nonlinearities.

Absorbing fixed linear preprocessing (Step 14) -- tell customers this:
  Linear maps compose: W (P x) = (W P) x.
  So a fixed linear preprocess P can be folded into a learned matrix W'=WP.
  Representation power is unchanged: the network can still implement any
  linear map it could before. You need not fear that linear data
  preprocessing "breaks" a model that already has learned linear layers
  (it always does). Nonlinear preprocessing is a different story.

Bridge to 340_Wq_Wk_Wv.py:
  W_Q, W_K, W_V are shared matrices -- linear perspectives on token space --
  learned from the distribution of (X, Y), applied as Q=X W_Q, etc.
  A fixed linear change of X can be absorbed into those W_* the same way.

Demo matrix used above (2D):
  A = {A.tolist()}
  T(e1) = {Te1.tolist()}   T(e2) = {Te2.tolist()}
  x = {x_vec.tolist()}  =>  A x = {Ax.tolist()}

2D grid (Step 10) then interactive 2D sliders (Step 11).
3D browsable grid (Step 12):
  A3 = {A3.tolist()}
  Drag either 3D panel to rotate; matching colors track each grid line
  through the map T(x)=A3 x.  Step 13 adds live a_ij sliders in 3D.
""")
