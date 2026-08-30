"""
016_torch_training.py

Lecture slides that explain 015_torch_training.ipynb.

The notebook is a conceptual lean version of 070_torch_training.py:
  - one parameter w, no bias:  y_hat = w * x
  - MSE loss
  - full-batch SGD on 5 noisy points near y = 2x
  - no animation -- just the loop, stripped bare

Sketch:
  1. Notebook code block 1 -- setup (exact)
  2. Notebook code block 2 -- training loop (exact)
  3. Model: one weight, no bias
  4. Data: five noisy points on y ≈ 2x
  5. The four pieces -- model, loss, backward, step
  6. One SGD step as a cycle
  7. The training loop, line by line
  8. After training -- w approaches 2
  9. Mental checklist before richer models

Companion: 015_torch_training.ipynb, 070_torch_training.py, 080_model.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
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
from matplotlib.patches import FancyBboxPatch
import numpy as np

torch.manual_seed(523)


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


def arrow(ax, x1, y1, x2, y2, color='0.25', lw=2.5):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        mutation_scale=18),
    )


C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_OLD = '#fce4d6'


# -------------------------------------------------------
# Reproduce the notebook's tiny experiment (for live numbers)
# -------------------------------------------------------
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = 2.0 * x + 2.5 * torch.randn(5, 1)

model = nn.Linear(1, 1, bias=False)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.005)

w0 = model.weight.item()
SGD_STEPS = 100
w_path = [w0]
loss_path = []

for _ in range(SGD_STEPS):
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    w_path.append(model.weight.item())
    loss_path.append(loss.item())

w_final = model.weight.item()
mse_final = loss_fn(model(x), y).item()
w_path = np.array(w_path)
loss_path = np.array(loss_path)

x_np = x.numpy().ravel()
y_np = y.numpy().ravel()


# Exact code cells from 015_torch_training.ipynb (no edits).
NB_CODE_1 = (
    'import torch\n'
    'import torch.nn as nn\n'
    '\n'
    'torch.manual_seed(523)\n'
    '\n'
    'x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])\n'
    'y = 2.0 * x + 2.5 * torch.randn(5, 1)   # noisy observations of y = 2x\n'
    '\n'
    'model = nn.Linear(1, 1, bias=False)     # one weight w; y_hat = w * x\n'
    'loss_fn = nn.MSELoss()\n'
    'opt = torch.optim.SGD(model.parameters(), lr=0.005)\n'
)

NB_CODE_2 = (
    'SGD_STEPS = 100\n'
    '\n'
    'for sgd_step in range(SGD_STEPS):\n'
    '    y_hat = model(x)              # forward\n'
    '    loss = loss_fn(y_hat, y)      # scalar MSE\n'
    '    opt.zero_grad()\n'
    '    loss.backward()               # compute d(loss)/dw\n'
    '    opt.step()                    # w <- w - lr * grad\n'
    '\n'
    'w = model.weight.item()\n'
    'print(f"learned w = {w:.4f}   (true slope ~ 2)")\n'
    'print(f"final MSE = {loss_fn(model(x), y).item():.4f}")\n'
)


def show_code_slide(step_no, title, code):
    fig, ax = new_figure()
    fig.suptitle(title, fontsize=FS_FIGURE_TITLE - 2, fontweight='bold')
    blank_slide(ax)
    round_box(ax, 0.45, 0.45, 11.1, 7.1, '#fbfcfc', '#2874a6', lw=2.0)
    # Exact notebook source -- no add/remove; left-aligned monospace.
    ax.text(
        0.85, 4.0, code.rstrip('\n'),
        ha='left', va='center',
        fontsize=FS_ANNOT + 2,
        family='monospace',
        linespacing=1.45,
    )
    show_maximized(fig)


# -------------------------------------------------------
# FIGURE 1: Notebook code block 1 (exact)
# -------------------------------------------------------
show_code_slide(
    1,
    'Step 1: 015 code block 1 -- setup (exact)',
    NB_CODE_1,
)


# -------------------------------------------------------
# FIGURE 2: Notebook code block 2 (exact)
# -------------------------------------------------------
show_code_slide(
    2,
    'Step 2: 015 code block 2 -- training loop (exact)',
    NB_CODE_2,
)


# -------------------------------------------------------
# FIGURE 3: Model -- one weight
# -------------------------------------------------------
fig3m, (ax2a, ax2b) = new_figure(1, 2)
fig3m.suptitle(
    'Step 3: The model is deliberately tiny -- one free number w',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax2a.set_xlim(0, 10)
ax2a.set_ylim(0, 10)
ax2a.axis('off')
ax2a.set_title('Code meaning', fontsize=FS_AX_TITLE - 1)
round_box(ax2a, 0.4, 5.5, 9.2, 3.8, C_MUTE, '0.45', lw=1.6)
ax2a.text(
    5.0, 7.4,
    'model = nn.Linear(1, 1, bias=False)\n\n'
    '-> input dim 1, output dim 1\n'
    '-> no bias parameter\n'
    '-> only weight w\n\n'
    r'forward:   $\hat y = w\, x$',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    family='monospace',
)
round_box(ax2a, 0.4, 0.6, 9.2, 4.4, C_OLD, '#e67e22', lw=2.0)
ax2a.text(
    5.0, 2.8,
    'Intuition:\n'
    'All learning in this notebook is the motion of a single scalar w.\n'
    'If you understand how w moves, you understand the loop.\n\n'
    'Richer nets = many such numbers updated by the same pattern.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

# Right: geometry of y = w x family
xs = np.linspace(0, 5.5, 50)
for w_demo, col, lw in [(0.5, '0.7', 1.5), (1.0, '0.55', 1.8),
                          (2.0, C_OK, 3.0), (3.0, '0.55', 1.8)]:
    ax2b.plot(xs, w_demo * xs, color=col, lw=lw,
              label=rf'$w={w_demo:g}$' + ('  (truth)' if w_demo == 2 else ''))
ax2b.set_title(r'Family of models $\hat y = w x$', fontsize=FS_AX_TITLE - 1)
ax2b.set_xlabel('x', fontsize=FS_AXIS)
ax2b.set_ylabel('y', fontsize=FS_AXIS)
ax2b.tick_params(labelsize=FS_TICK)
ax2b.grid(True, alpha=0.3)
ax2b.legend(fontsize=FS_LEGEND, loc='upper left')
ax2b.text(
    0.98, 0.02,
    'Training chooses which\n'
    'line in this family\n'
    'best fits the data.',
    transform=ax2b.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
fig3m.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig3m)


# -------------------------------------------------------
# FIGURE 4: Data
# -------------------------------------------------------
fig3, (ax3a, ax3b) = new_figure(1, 2)
fig3.suptitle(
    'Step 4: Five noisy observations of y = 2x',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax3a.set_xlim(0, 10)
ax3a.set_ylim(0, 10)
ax3a.axis('off')
ax3a.set_title('Notebook data', fontsize=FS_AX_TITLE - 1)
round_box(ax3a, 0.4, 1.0, 9.2, 8.0, C_NEW, '#2874a6', lw=2.0)
ax3a.text(
    5.0, 7.4,
    'x = [[1], [2], [3], [4], [5]]\n\n'
    'y = 2*x + noise\n\n'
    'Tiny on purpose:\n'
    'every SGD step can use ALL points\n'
    '(full-batch GD on 5 rows).',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    family='monospace',
)
ax3a.text(
    5.0, 2.4,
    'Noise matters pedagogically:\n'
    'perfect interpolation is not the goal.\n'
    'We want w near 2 despite jitter.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

ax3b.scatter(x_np, y_np, s=90, c='#2874a6', zorder=3,
             edgecolors='white', linewidths=0.8, label='data')
ax3b.plot([0, 5.5], [0, 11], 'k--', lw=2.0, label='true y=2x')
ax3b.set_title('Cloud around the true line', fontsize=FS_AX_TITLE - 1)
ax3b.set_xlabel('x', fontsize=FS_AXIS)
ax3b.set_ylabel('y', fontsize=FS_AXIS)
ax3b.tick_params(labelsize=FS_TICK)
ax3b.grid(True, alpha=0.3)
ax3b.legend(fontsize=FS_LEGEND, loc='upper left')
fig3.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 5: Four pieces
# -------------------------------------------------------
fig4, ax4 = new_figure()
fig4.suptitle(
    'Step 5: Four pieces the notebook names explicitly',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax4)

pieces = [
    (0.4, 4.3, 'model', 'maps x -> w*x\n(the prediction machine)'),
    (3.3, 4.3, 'loss', 'how wrong y_hat is\n(MSE vs y)'),
    (6.2, 4.3, 'backward', 'fills w.grad\n= d(loss)/dw'),
    (9.1, 4.3, 'step', 'w <- w - lr * grad\n(the actual learning)'),
]
for x0, y0, title, body in pieces:
    round_box(ax4, x0, y0, 2.6, 2.8, C_GOAL, C_OK, lw=2.2)
    ax4.text(x0 + 1.3, y0 + 2.05, title, ha='center',
             fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_OK,
             family='monospace')
    ax4.text(x0 + 1.3, y0 + 1.0, body, ha='center', va='center',
             fontsize=FS_ANNOT)

for i in range(3):
    x1 = 0.4 + 2.6 + i * 2.9
    arrow(ax4, x1, 5.7, x1 + 0.25, 5.7, C_DARK, lw=2.2)

round_box(ax4, 0.4, 0.5, 11.2, 3.3, C_MUTE, '0.45', lw=1.8)
ax4.text(
    6.0, 2.15,
    'Notebook table (same idea):\n\n'
    'model  = prediction\n'
    'loss   = wrongness\n'
    'backward = compute gradient into .grad\n'
    'step   = apply SGD update\n\n'
    'If you mix these four up, debugging becomes superstition.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 6: One SGD step cycle
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    'Step 6: One SGD step is a closed cycle',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax5)

cycle = [
    (1.0, 5.2, 'predict\ny_hat = model(x)'),
    (4.5, 5.2, 'loss\nMSE(y_hat, y)'),
    (8.0, 5.2, 'zero_grad'),
    (8.0, 2.0, 'backward\nfill .grad'),
    (4.5, 2.0, 'step\nw -= lr*grad'),
    (1.0, 2.0, 'repeat'),
]
for i, (x0, y0, title) in enumerate(cycle):
    face = C_NEW if i < 3 else C_GOAL
    edge = '#2874a6' if i < 3 else C_OK
    round_box(ax5, x0, y0, 2.8, 1.8, face, edge, lw=2.0)
    ax5.text(x0 + 1.4, y0 + 0.9, title, ha='center', va='center',
             fontsize=FS_ANNOT + 1, fontweight='bold')

arrow(ax5, 3.9, 6.1, 4.4, 6.1, C_DARK, lw=2.2)
arrow(ax5, 7.4, 6.1, 7.9, 6.1, C_DARK, lw=2.2)
arrow(ax5, 9.4, 5.1, 9.4, 3.9, C_DARK, lw=2.2)
arrow(ax5, 7.9, 2.9, 7.4, 2.9, C_DARK, lw=2.2)
arrow(ax5, 4.4, 2.9, 3.9, 2.9, C_DARK, lw=2.2)
arrow(ax5, 2.4, 3.9, 2.4, 5.1, C_DARK, lw=2.2)

ax5.text(
    6.0, 0.55,
    'In 015: every step uses all 5 points (full batch).\n'
    'Same cycle scales to mini-batches later -- only the data slice changes.',
    ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='lightyellow',
              edgecolor='0.55', alpha=0.95),
)
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 7: Loop line by line
# -------------------------------------------------------
fig6, ax6 = new_figure()
fig6.suptitle(
    'Step 7: The notebook loop, line by line',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax6)

lines = [
    ('y_hat = model(x)', 'Forward pass: current w predicts.'),
    ('loss = loss_fn(y_hat, y)', 'Scalar MSE -- one number to reduce.'),
    ('opt.zero_grad()', 'Clear old .grad (grads accumulate by default).'),
    ('loss.backward()', 'Autograd writes d(loss)/dw into w.grad.'),
    ('opt.step()', 'SGD: w <- w - lr * w.grad.'),
]
for i, (code, meaning) in enumerate(lines):
    y0 = 6.5 - i * 1.15
    round_box(ax6, 0.4, y0, 5.5, 1.0, C_NEW, '#2874a6', lw=1.8)
    ax6.text(3.15, y0 + 0.5, code, ha='center', va='center',
             fontsize=FS_ANNOT + 1, family='monospace', fontweight='bold',
             color=C_DARK)
    round_box(ax6, 6.2, y0, 5.4, 1.0, C_MUTE, '0.45', lw=1.5)
    ax6.text(8.9, y0 + 0.5, meaning, ha='center', va='center',
             fontsize=FS_ANNOT)

ax6.text(
    6.0, 0.45,
    f'Notebook settings:  lr = 0.005,  SGD_STEPS = {SGD_STEPS},  seed = 523',
    ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
)
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 8: After training
# -------------------------------------------------------
fig7, (ax7a, ax7b) = new_figure(1, 2)
fig7.suptitle(
    'Step 8: After the loop -- w moves toward 2',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax7a.plot(w_path, color='#2874a6', lw=2.6)
ax7a.axhline(2.0, color=C_OK, ls='--', lw=2.0, label='true slope 2')
ax7a.set_title('Trajectory of w', fontsize=FS_AX_TITLE - 1)
ax7a.set_xlabel('SGD step', fontsize=FS_AXIS)
ax7a.set_ylabel('w', fontsize=FS_AXIS)
ax7a.tick_params(labelsize=FS_TICK)
ax7a.grid(True, alpha=0.3)
ax7a.legend(fontsize=FS_LEGEND, loc='best')
ax7a.text(
    0.98, 0.02,
    f'start w = {w0:.3f}\n'
    f'final w = {w_final:.3f}\n'
    f'final MSE = {mse_final:.3f}',
    transform=ax7a.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax7b.scatter(x_np, y_np, s=80, c='#2874a6', zorder=3,
             edgecolors='white', linewidths=0.7, label='data')
ax7b.plot([0, 5.5], [0, 11], 'k--', lw=1.8, label='true y=2x')
ax7b.plot([0, 5.5], [0, 5.5 * w_final], color=C_ACCENT, lw=2.8,
          label=rf'learned $w={w_final:.2f}$')
ax7b.set_title('Fit after training', fontsize=FS_AX_TITLE - 1)
ax7b.set_xlabel('x', fontsize=FS_AXIS)
ax7b.set_ylabel('y', fontsize=FS_AXIS)
ax7b.tick_params(labelsize=FS_TICK)
ax7b.grid(True, alpha=0.3)
ax7b.legend(fontsize=FS_LEGEND, loc='upper left')

fig7.tight_layout(rect=[0, 0.05, 1, 0.94])
fig7.text(
    0.5, 0.01,
    'Same seed and recipe as 015_torch_training.ipynb -- numbers match the notebook run.',
    ha='center', va='bottom', fontsize=FS_ANNOT + 1, fontweight='bold',
)
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 9: Checklist
# -------------------------------------------------------
fig8, ax8 = new_figure()
fig8.suptitle(
    'Step 9: Before you touch a bigger model -- this checklist',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax8)

checks = [
    'Can I write the predict -> loss -> zero_grad -> backward -> step cycle from memory?',
    'Do I know which tensor is the parameter, and where its .grad lives?',
    'Do I know why zero_grad is required (grads accumulate)?',
    'Do I know that step is the only place parameters change?',
    'Can I predict what happens if lr is 10x too large / too small?',
]
for i, line in enumerate(checks):
    y0 = 6.4 - i * 1.1
    round_box(ax8, 0.5, y0, 11.0, 0.95, C_GOAL if i % 2 == 0 else C_NEW,
              C_OK if i % 2 == 0 else '#2874a6', lw=1.8)
    ax8.text(1.0, y0 + 0.48, f'{i+1}.', ha='left', va='center',
             fontsize=FS_ANNOT + 2, fontweight='bold')
    ax8.text(1.7, y0 + 0.48, line, ha='left', va='center',
             fontsize=FS_ANNOT + 1)

ax8.text(
    6.0, 0.45,
    'Next: 070_torch_training.py (animated), then 080_model.py (bias / richer fits).',
    ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
)
show_maximized(fig8)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("016 -- EXPLAINING 015_torch_training.ipynb")
print("=" * 60)
print(f"""
Notebook essence:
  y_hat = w * x          (nn.Linear(..., bias=False))
  loss  = MSE(y_hat, y)
  loop  = predict -> loss -> zero_grad -> backward -> step

Data: 5 points, y = 2x + noise, seed 523, lr=0.005, steps={SGD_STEPS}

This run:
  start w = {w0:.4f}
  final w = {w_final:.4f}   (target near 2)
  final MSE = {mse_final:.4f}

Teaching point:
  Bigger models reuse this same loop.
  If the loop is unclear, scale will not save you.
""")
