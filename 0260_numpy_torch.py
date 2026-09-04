"""
041_numpy_torch.py

NumPy ndarray vs Torch Tensor -- after 0330_distribution.py.

Sketch:
  1. So what are NumPy and Torch?

See also: 0330_distribution.py, 0340_normalize_data.py, 0740_discriminative_generative.py.
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
import matplotlib.patches as patches
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
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.15',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(rect)
    return rect


def arrow(ax, x1, y1, x2, y2, color='0.25', lw=2.5):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        mutation_scale=18),
    )


# Palette
C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_OLD = '#fce4d6'



# FIGURE 1: So what are NumPy and Torch?
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    'Step 1: So what are NumPy and Torch?',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax1)

round_box(ax1, 0.5, 6.55, 11.0, 1.15, C_DARK, C_DARK, lw=0)
ax1.text(
    6.0, 7.1,
    'Two libraries for numbers in arrays -- same spirit, different engine.',
    ha='center', va='center', fontsize=FS_AX_TITLE - 2,
    fontweight='bold', color='white',
)

# NumPy
round_box(ax1, 0.5, 3.55, 5.3, 2.7, C_NEW, '#2874a6', lw=2.2)
ax1.text(
    3.15, 5.9, 'NumPy  --  ndarray',
    ha='center', va='center', fontsize=FS_AX_TITLE - 2,
    fontweight='bold', color=C_DARK,
)
ax1.text(
    3.15, 5.25,
    'CPU array. No autograd. Data prep & science.',
    ha='center', va='center', fontsize=FS_ANNOT,
)
ax1.text(
    3.15, 4.15,
    'import numpy as np\n\n'
    'a = np.array([[1., 2.],\n'
    '              [3., 4.]])',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    family='monospace',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
              edgecolor='#2874a6', alpha=0.98),
)

# Torch
round_box(ax1, 6.2, 3.55, 5.3, 2.7, C_GOAL, C_OK, lw=2.2)
ax1.text(
    8.85, 5.9, 'Torch  --  Tensor',
    ha='center', va='center', fontsize=FS_AX_TITLE - 2,
    fontweight='bold', color=C_OK,
)
ax1.text(
    8.85, 5.25,
    'CPU/GPU array + optional gradient tape.',
    ha='center', va='center', fontsize=FS_ANNOT,
)
ax1.text(
    8.85, 4.15,
    'import torch\n\n'
    't = torch.tensor([[1., 2.],\n'
    '                  [3., 4.]])',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    family='monospace',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
              edgecolor=C_OK, alpha=0.98),
)

# Difference strip
round_box(ax1, 0.5, 1.55, 11.0, 1.7, C_MUTE, '0.45', lw=1.6)
ax1.text(
    6.0, 2.85, 'ndarray vs Tensor -- the practical difference',
    ha='center', va='center', fontsize=FS_ANNOT + 2,
    fontweight='bold', color=C_DARK,
)
ax1.text(
    6.0, 2.05,
    'A Tensor is an array plus device + optional gradient tape.\n'
    'Bridge either way:  '
    r'$\mathrm{torch.from\_numpy}(a)$'
    '   /   '
    r'$t.detach().cpu().numpy()$',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

round_box(ax1, 0.5, 1.0, 11.0, 0.9, 'lightyellow', '0.55', lw=1.6)
ax1.text(
    6.0, 1.45,
    'Same mental model: operate on whole arrays, not one scalar at a time.\n'
    'NumPy explores and prepares data.  Torch learns from it.',
    ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
)

round_box(ax1, 0.5, 0.15, 11.0, 0.7, '#f8d7da', C_ACCENT, lw=1.8)
ax1.text(
    6.0, 0.5,
    'Study, independently, the indexing and operations of arrays and tensors.',
    ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
    color=C_ACCENT,
)

show_maximized(fig1)


# -------------------------------------------------------

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("NUMPY VS TORCH")
print("=" * 60)
print("""
NumPy vs Torch (Step 1):
  ndarray: CPU, no autograd.  Tensor: CPU/GPU + optional gradient tape.
  Bridge: torch.from_numpy(a) / t.detach().cpu().numpy().
  Same mental model: operate on whole arrays, not one scalar at a time.

Study, independently, the indexing and operations of arrays and tensors.

See also: 0330_distribution.py, 0340_normalize_data.py
""")
