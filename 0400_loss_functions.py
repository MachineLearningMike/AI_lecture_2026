"""
0400_loss_functions.py

What a loss function is, and Mean Squared Error as a browsable 3D shape.

Sketch:
  1. Loss functions -- what they are; why they matter for objective and performance
  2. Mean Squared Error -- formula + browsable 3D surface over one sample (y, y_hat)

Companion to 190_loss_functions.py, 0470_maximum_likelihood.py, 0280_model.py.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
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
    add_step_footer(fig)
    plt.show()


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')


def round_box(ax, x, y, w, h, facecolor, edgecolor='0.2', lw=2.0):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.15',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(rect)
    return rect


C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_OLD = '#fce4d6'


# -------------------------------------------------------
# FIGURE 1: What is a loss function?
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    'Step 1: Loss functions',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax1)

round_box(ax1, 0.4, 6.25, 11.2, 1.35, C_DARK, C_DARK, lw=0)
ax1.text(
    6.0, 6.9,
    'A loss function scores how wrong a prediction is.\n'
    'Training means: move parameters so this score goes down.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    fontweight='bold', color='white',
)

round_box(ax1, 0.4, 3.55, 5.5, 2.4, C_NEW, '#2874a6', lw=2.2)
ax1.text(3.15, 5.55, 'What it is', ha='center',
         fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_DARK)
ax1.text(
    3.15, 4.4,
    r'Given prediction $\hat y$ and target $y$,'
    '\n'
    r'loss $\ell(\hat y, y)$ is a number >= 0.'
    '\n\n'
    'Perfect prediction => loss near 0.\n'
    'Worse prediction => larger loss.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

round_box(ax1, 6.1, 3.55, 5.5, 2.4, C_GOAL, C_OK, lw=2.2)
ax1.text(8.85, 5.55, 'Training objective', ha='center',
         fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_OK)
ax1.text(
    8.85, 4.4,
    'The optimizer does not "want accuracy".\n'
    'It wants a smaller average loss.\n\n'
    'So the loss IS the training goal:\n'
    'what we minimize is what we get.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

round_box(ax1, 0.4, 0.35, 11.2, 2.9, C_OLD, '#e67e22', lw=2.2)
ax1.text(6.0, 2.75, 'Why it matters for training performance',
         ha='center', fontsize=FS_AX_TITLE - 2, fontweight='bold', color='#e67e22')
ax1.text(
    6.0, 1.4,
    'Shape of the loss changes how gradients look:\n'
    'smooth vs sharp, outlier-sensitive vs robust, flat vs steep.\n\n'
    'That changes convergence speed, stability, and which solution you reach.\n'
    'Same model + same data + different loss  =>  different training behavior.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: MSE formula + browsable 3D surface over (y, y_hat)
# -------------------------------------------------------
fig2 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig2.suptitle(
    'Step 2: Mean Squared Error',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# Left: formula + intuition cards
ax2a = fig2.add_subplot(1, 2, 1)
ax2a.set_xlim(0, 10)
ax2a.set_ylim(0, 10)
ax2a.axis('off')
ax2a.set_title('Formula (one sample)', fontsize=FS_AX_TITLE - 1)

round_box(ax2a, 0.4, 6.4, 9.2, 3.0, C_DARK, C_DARK, lw=0)
ax2a.text(
    5.0, 8.4,
    r'$\ell_{\mathrm{MSE}}(\hat y, y)=(\hat y - y)^{2}$',
    ha='center', va='center', fontsize=FS_AX_TITLE - 1,
    fontweight='bold', color='white',
)
ax2a.text(
    5.0, 7.15,
    r'(sometimes written $\frac{1}{2}(\hat y-y)^{2}$;'
    '\n'
    r'average over $N$ samples for the training loss)',
    ha='center', va='center', fontsize=FS_ANNOT, color='white',
)

round_box(ax2a, 0.4, 3.2, 9.2, 2.8, C_NEW, '#2874a6', lw=2.0)
ax2a.text(
    5.0, 4.6,
    'Intuition for one sample:\n\n'
    'Zero only on the diagonal  y_hat = y.\n'
    'Error of size 2 costs 4; error of size 3 costs 9.\n'
    'Large mistakes dominate.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

round_box(ax2a, 0.4, 0.4, 9.2, 2.5, C_GOAL, C_OK, lw=2.0)
ax2a.text(
    5.0, 1.65,
    'Right panel: surface z = (y_hat - y)^2\n'
    'over the plane of possible (y, y_hat).\n'
    'Drag to rotate -- browse the bowl.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

# Right: browsable 3D MSE surface
ax2b = fig2.add_subplot(1, 2, 2, projection='3d')
y_vals = np.linspace(-3.0, 3.0, 70)
yhat_vals = np.linspace(-3.0, 3.0, 70)
Y, YH = np.meshgrid(y_vals, yhat_vals)
Z = (YH - Y) ** 2

surf = ax2b.plot_surface(
    Y, YH, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.92,
)
# Diagonal ridge of zeros: y_hat = y
t = np.linspace(-3.0, 3.0, 60)
ax2b.plot(t, t, np.zeros_like(t), color='#c0392b', lw=3.0, label=r'$\hat y=y$ (loss 0)')
ax2b.set_xlabel(r'target $y$', fontsize=FS_AXIS)
ax2b.set_ylabel(r'prediction $\hat y$', fontsize=FS_AXIS)
ax2b.set_zlabel(r'$\ell=(\hat y-y)^2$', fontsize=FS_AXIS)
ax2b.set_title(r'Browsable 3D: $\ell(y,\hat y)=(\hat y-y)^2$', fontsize=FS_AX_TITLE - 2)
ax2b.tick_params(labelsize=FS_TICK)
ax2b.view_init(elev=28, azim=-55)
ax2b.legend(loc='upper left', fontsize=FS_LEGEND - 1, framealpha=0.92)
fig2.colorbar(surf, ax=ax2b, shrink=0.55, pad=0.08, label=r'$(\hat y-y)^2$')
ax2b.text2D(
    0.02, 0.02,
    'Mouse-drag to browse.\n'
    'Valley floor = perfect predictions.',
    transform=ax2b.transAxes, fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig2.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig2)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("185 LOSS FUNCTIONS")
print("=" * 60)
print("""
Step 1 -- Loss functions:
  A loss maps (prediction, target) -> a non-negative score of wrongness.
  Training objective: minimize average loss.
  Training performance: loss shape controls gradients, stability, and the solution.

Step 2 -- Mean Squared Error (one sample):
  ell(y_hat, y) = (y_hat - y)^2
  3D surface over (y, y_hat): parabolic bowl; zero on the diagonal y_hat = y.
  Browse by rotating the figure with the mouse.

See also: 190_loss_functions.py, 191_Mean_Squared_Error.py
""")
