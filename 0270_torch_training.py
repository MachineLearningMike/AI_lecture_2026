"""
0270_torch_training.py

Lecture slides that explain 0270_torch_training.ipynb.

The notebook is a conceptual lean version of 070_torch_training.py:
  - one parameter w, no bias:  y_hat = w * x
  - MSE loss
  - full-batch gradient descent on 30 noisy points near y = 2x
  - same fit/loss animation style as 070

Sketch:
  1. Notebook code block 1 -- setup
  2. Notebook code block 2 -- training loop
  3. Model: one weight, no bias
  4. Data: thirty noisy points on y ≈ 2x
  5. The four pieces -- model, loss, backward, step
  6. One full-batch training step as a cycle
  7. The training loop, line by line
  8. After training -- w approaches 2
  9. Animation -- prediction line fits one step at a time (as in 070)
  10. Mental checklist before richer models
  11. What's wrong with what you just learned? -- goal, field use, generalization

Companion: 0270_torch_training.ipynb, 070_torch_training.py, 0280_model.py,
0120_intelligence.py.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation
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

ANIM_FPS = 4
ANIM_HOLD_SEC = 2.0
OUT_GIF = Path(__file__).resolve().parent / '0270_torch_training.gif'


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
# Full-batch experiment (live numbers for later slides)
# N=30, mild noise so least-squares w sits near the true slope 2.
# -------------------------------------------------------
N = 30
x = torch.linspace(1.0, 5.0, N).unsqueeze(1)
y = 2.0 * x + 1.5 * torch.randn(N, 1)

model = nn.Linear(1, 1, bias=False)
with torch.no_grad():
    model.weight.fill_(-1.5)  # start far from 2 so the path is visible
loss_fn = nn.MSELoss()
# Full-batch gradient descent on all N rows each step (PyTorch optimizer API).
LR = 0.002
opt = torch.optim.SGD(model.parameters(), lr=LR)

w0 = model.weight.item()
STEPS = 150
xs_line = torch.linspace(0.5, 5.5, 50).unsqueeze(1)
w_path = [w0]
grad_path = []
loss_path = []
# Histories for the 070-style animation (include the initial state).
with torch.no_grad():
    loss_hist = [loss_fn(model(x), y).item()]
    pred_hist = [model(xs_line).squeeze(1).clone()]

for _ in range(STEPS):
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    opt.zero_grad()
    loss.backward()
    grad_path.append(model.weight.grad.item())
    opt.step()
    w_path.append(model.weight.item())
    loss_path.append(loss.item())
    with torch.no_grad():
        loss_hist.append(loss_fn(model(x), y).item())
        pred_hist.append(model(xs_line).squeeze(1).clone())

w_final = model.weight.item()
mse_final = loss_fn(model(x), y).item()
w_path = np.array(w_path)
grad_path = np.array(grad_path)
loss_path = np.array(loss_path)

x_np = x.numpy().ravel()
y_np = y.numpy().ravel()
xs_np = xs_line.numpy().ravel()
true_ys = (2.0 * xs_line).numpy().ravel()
pred_np = [p.numpy() for p in pred_hist]


# Code cells for Steps 1-2 (same recipe as the live run above).
NB_CODE_1 = (
    'import torch\n'
    'import torch.nn as nn\n'
    '\n'
    'torch.manual_seed(523)\n'
    '\n'
    'N = 30\n'
    'x = torch.linspace(1.0, 5.0, N).unsqueeze(1)\n'
    'y = 2.0 * x + 1.5 * torch.randn(N, 1)  # mild noise; truth y=2x\n'
    '\n'
    'model = nn.Linear(1, 1, bias=False)     # one weight w; y_hat = w * x\n'
    'with torch.no_grad():\n'
    '    model.weight.fill_(-0.5)            # start far from 2\n'
    'loss_fn = nn.MSELoss()\n'
    '# full-batch GD on all N rows (PyTorch API class name below)\n'
    'opt = torch.optim.SGD(model.parameters(), lr=0.005)\n'
)

NB_CODE_2 = (
    'STEPS = 150\n'
    '\n'
    'for step in range(STEPS):\n'
    '    y_hat = model(x)              # forward on ALL N points\n'
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
    ax.text(
        0.85, 4.0, code.rstrip('\n'),
        ha='left', va='center',
        fontsize=FS_ANNOT + 2,
        family='monospace',
        linespacing=1.45,
    )
    show_maximized(fig)


# -------------------------------------------------------
# FIGURE 1: Notebook code block 1
# -------------------------------------------------------
show_code_slide(
    1,
    'Step 1: 015 code block 1 -- setup',
    NB_CODE_1,
)


# -------------------------------------------------------
# FIGURE 2: Notebook code block 2
# -------------------------------------------------------
show_code_slide(
    2,
    'Step 2: 015 code block 2 -- training loop (full batch)',
    NB_CODE_2,
)


# -------------------------------------------------------
# FIGURE 3: Model -- one weight
# -------------------------------------------------------
# Left panel +2.5 inches broader WITHOUT shrinking the right panel:
# keep right absolute width; grow whole figure by 2.5 inches.
_fw3, _fh3 = screen_figsize()
_usable3 = _fw3 * 0.92
_right_in3 = _usable3 * 0.5
_left_in3 = _usable3 * 0.5 + 2.5
_fw3_wide = (_left_in3 + _right_in3) / 0.92
fig3m, (ax2a, ax2b) = plt.subplots(
    1, 2,
    figsize=(_fw3_wide, _fh3),
    dpi=FIGURE_DPI,
    gridspec_kw={'width_ratios': [_left_in3, _right_in3]},
)
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
# Keep the wider figsize (do not reset to default screen width).
fig3m.set_dpi(FIGURE_DPI)
fig3m.set_size_inches(_fw3_wide, _fh3, forward=True)
fig3m.canvas.draw_idle()
_manager3 = getattr(fig3m.canvas, 'manager', None)
if _manager3 is not None:
    try:
        _manager3.window.state('zoomed')
    except Exception:
        try:
            _manager3.window.showMaximized()
        except Exception:
            try:
                _manager3.frame.Maximize(True)
            except Exception:
                pass
add_step_footer(fig3m)
plt.show()


# -------------------------------------------------------
# FIGURE 4: Data
# -------------------------------------------------------
fig3, (ax3a, ax3b) = new_figure(1, 2)
fig3.suptitle(
    f'Step 4: {N} noisy observations of y = 2x',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax3a.set_xlim(0, 10)
ax3a.set_ylim(0, 10)
ax3a.axis('off')
ax3a.set_title('Training data', fontsize=FS_AX_TITLE - 1)
round_box(ax3a, 0.4, 1.0, 9.2, 8.0, C_NEW, '#2874a6', lw=2.0)
ax3a.text(
    5.0, 7.4,
    f'N = {N}\n'
    'x = linspace(1, 5, N)\n\n'
    'y = 2*x + mild noise\n\n'
    'Every training step uses ALL points\n'
    f'(full-batch GD on {N} rows).',
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
    'step   = apply the parameter update\n\n'
    'If you mix these four up, debugging becomes superstition.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 6: One full-batch training step cycle
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    'Step 6: One full-batch training step is a closed cycle',
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
    f'Here: every step uses all {N} points (full batch).\n'
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
    ('opt.step()', 'Update: w <- w - lr * w.grad.'),
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
    f'Settings:  lr = {LR:g},  STEPS = {STEPS},  seed = 523  (full batch)',
    ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
)
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 8: After training -- w path + gradient series
# -------------------------------------------------------
fig7, (ax7a, ax7b) = new_figure(1, 2)
fig7.suptitle(
    'Step 8: After the loop -- w converges near the true slope 2',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

steps_w = np.arange(len(w_path))
steps_g = np.arange(len(grad_path))  # one grad per update (before step)
line_w, = ax7a.plot(steps_w, w_path, color='#2874a6', lw=2.6, label='w')
line_true = ax7a.axhline(2.0, color=C_OK, ls='--', lw=2.0, label='true slope 2')
ax7a.set_title(r'Trajectory of $w$ and gradient $g=\partial\ell/\partial w$',
               fontsize=FS_AX_TITLE - 2)
ax7a.set_xlabel('training step', fontsize=FS_AXIS)
ax7a.set_ylabel('w', fontsize=FS_AXIS, color='#2874a6')
ax7a.tick_params(labelsize=FS_TICK, axis='y', labelcolor='#2874a6')
ax7a.grid(True, alpha=0.3)
ax7a.relim()
ax7a.autoscale_view()
ylo0, yhi0 = ax7a.get_ylim()
# Same numeric ticks for g as for w. If |g| overflows w's window, scale g down.
room = max(abs(ylo0), abs(yhi0), 1e-12)
g_peak = float(np.max(np.abs(grad_path))) if len(grad_path) else 0.0
if g_peak > room:
    g_scale = g_peak / room
    g_plot = grad_path / g_scale
    g_label = rf'gradient $g/{g_scale:.3g}$'
else:
    g_scale = 1.0
    g_plot = grad_path
    g_label = r'gradient $g$'

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
fig7.canvas.draw_idle()
# Lift the gradient curve by +1 physical inch (data units from axes height).
_pos7 = ax7a.get_position()
_dy_g = 1.0 / max(fig7.get_figheight() * _pos7.height, 1e-6) * (yhi0 - ylo0)
g_plot_lifted = g_plot + _dy_g
ylo, yhi = ylo0, yhi0 + _dy_g
ax7a.set_ylim(ylo, yhi)

ax7a_g = ax7a.twinx()
line_g, = ax7a_g.plot(steps_g, g_plot_lifted, color=C_ACCENT, lw=2.2,
                      alpha=0.9, label=g_label + ' (lifted)')
ax7a_g.axhline(_dy_g, color=C_ACCENT, ls=':', lw=1.2, alpha=0.7)  # g = 0 after lift
ax7a_g.set_ylim(ylo, yhi)
ax7a_g.set_yticks(ax7a.get_yticks())
ax7a_g.set_ylabel(g_label, fontsize=FS_AXIS, color=C_ACCENT)
ax7a_g.tick_params(labelsize=FS_TICK, axis='y', labelcolor=C_ACCENT)

lines = [line_w, line_true, line_g]
labels = [ln.get_label() for ln in lines]
ax7a.legend(lines, labels, fontsize=FS_LEGEND - 1, loc='upper left')

# Update rule for w: right edge, and 0.4 inch lower than the old 0.72 anchor.
_dy_ann = 0.4 / max(fig7.get_figheight() * _pos7.height, 1e-6)
ax7a.text(
    0.98, 0.72 - _dy_ann,
    r'$w \leftarrow w - \mathrm{lr}\cdot g$'
    '\n'
    rf'lr $= {LR:g}$'
    '\n'
    r'$g=\partial\ell/\partial w$',
    transform=ax7a.transAxes, ha='right', va='top', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#eafaf1', edgecolor=C_OK, alpha=0.95),
)
ax7a.text(
    0.98, 0.02,
    f'start w = {w0:.3f}\n'
    f'final w = {w_final:.3f}\n'
    f'final |g| = {abs(grad_path[-1]):.3f}\n'
    f'final MSE = {mse_final:.3f}'
    + (f'\n(g scaled ÷{g_scale:.3g})' if g_scale != 1.0 else '')
    + '\n(g curve lifted +1 in)',
    transform=ax7a.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
fig7.text(
    0.5, 0.01,
    'Mild noise + enough steps: w settles near 2; gradient series fades toward 0.',
    ha='center', va='bottom', fontsize=FS_ANNOT + 1, fontweight='bold',
)
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 9: Animation (same style as 070)
# -------------------------------------------------------
fig_anim, axes_anim = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig_anim.suptitle(
    'Step 9: Watch one weight learn  (y = w x, no bias)',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax_fit = axes_anim[0]
ax_fit.scatter(x_np, y_np, s=80, c='royalblue', label='data', zorder=3)
ax_fit.plot(xs_np, true_ys, 'k--', lw=1.8, alpha=0.65, label=r'true $y=2x$')
fit_line, = ax_fit.plot([], [], color='crimson', lw=2.8, label=r'predict $y=wx$')
hud = ax_fit.text(
    0.02, 0.98, '', transform=ax_fit.transAxes, va='top', ha='left',
    fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
ax_fit.set_xlabel('x', fontsize=FS_AXIS)
ax_fit.set_ylabel('y', fontsize=FS_AXIS)
ax_fit.set_title('Fit improves with each training step', fontsize=FS_AX_TITLE)
ax_fit.tick_params(axis='both', labelsize=FS_TICK)
ax_fit.legend(loc='lower right', fontsize=FS_LEGEND)
ax_fit.grid(True, alpha=0.3)
y_lo = min(float(y_np.min()), float(true_ys.min()), min(p.min() for p in pred_np)) - 1.0
y_hi = max(float(y_np.max()), float(true_ys.max()), max(p.max() for p in pred_np)) + 1.0
ax_fit.set_xlim(0.3, 5.7)
ax_fit.set_ylim(y_lo, y_hi)

ax_loss = axes_anim[1]
loss_line, = ax_loss.plot([], [], color='seagreen', lw=2.5)
loss_dot, = ax_loss.plot([], [], 'o', color='seagreen', ms=8)
ax_loss.set_xlim(0, len(loss_hist) - 1)
ax_loss.set_ylim(0.0, max(loss_hist) * 1.1 + 1e-6)
ax_loss.set_xlabel('training step', fontsize=FS_AXIS)
ax_loss.set_ylabel('MSE', fontsize=FS_AXIS)
ax_loss.set_title('Full-batch loss', fontsize=FS_AX_TITLE)
ax_loss.tick_params(axis='both', labelsize=FS_TICK)
ax_loss.grid(True, alpha=0.3)

fig_anim.tight_layout(rect=[0, 0, 1, 0.94])

n_frames = len(w_path)
hold = max(1, int(round(ANIM_HOLD_SEC * ANIM_FPS)))
frame_ids = list(range(n_frames)) + [n_frames - 1] * hold


def init_anim():
    fit_line.set_data([], [])
    loss_line.set_data([], [])
    loss_dot.set_data([], [])
    hud.set_text('')
    return fit_line, loss_line, loss_dot, hud


def update_anim(frame):
    fit_line.set_data(xs_np, pred_np[frame])
    steps = list(range(frame + 1))
    loss_line.set_data(steps, loss_hist[: frame + 1])
    loss_dot.set_data([frame], [loss_hist[frame]])
    hud.set_text(
        f'training step {frame}/{n_frames - 1}\n'
        f'w = {w_path[frame]:.3f}\n'
        f'MSE = {loss_hist[frame]:.3f}'
    )
    return fit_line, loss_line, loss_dot, hud


anim = animation.FuncAnimation(
    fig_anim, update_anim, init_func=init_anim,
    frames=frame_ids, interval=1000 / ANIM_FPS, blit=False,
)

if OUT_GIF.is_file() and OUT_GIF.stat().st_size > 0:
    print(f'\nSkipping GIF export (already exists): {OUT_GIF}')
else:
    print(f'Saving animation to {OUT_GIF} ...')
    try:
        anim.save(OUT_GIF, writer=animation.PillowWriter(fps=ANIM_FPS))
        print(f'Saved: {OUT_GIF}')
    except Exception as exc:
        print(f'Skipping GIF export ({OUT_GIF.name}): {exc}')
        print('Install pillow to enable animation export: pip install pillow')

show_maximized(fig_anim)


# -------------------------------------------------------
# FIGURE 10: Checklist
# -------------------------------------------------------
fig8, ax8 = new_figure()
fig8.suptitle(
    'Step 10: Before you touch a bigger model -- this checklist',
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
    'Next step asks what this loop is for. Then: 0280_model.py.',
    ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
)
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 11: What's wrong with what you just learned?
# -------------------------------------------------------
fig9, ax9 = new_figure()
fig9.suptitle(
    "Step 11: What's wrong with what you just learned?",
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax9)

round_box(ax9, 0.4, 6.35, 11.2, 1.35, '#1a5276', '#1a5276', lw=0)
ax9.text(
    6.0, 7.05,
    'We taught the machinery of training -- not yet its purpose.',
    ha='center', va='center', fontsize=FS_ANNOT + 2,
    fontweight='bold', color='#f9e79f',
)

# Gap: mechanism without objective
round_box(ax9, 0.4, 4.55, 5.4, 1.55, '#fadbd8', C_ACCENT, lw=2.0)
ax9.text(
    3.1, 5.55, 'What this file covered',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    fontweight='bold', color=C_ACCENT,
)
ax9.text(
    3.1, 4.95,
    'model, loss, backward, step\n'
    'how to drive the weight downhill',
    ha='center', va='center', fontsize=FS_ANNOT,
)

round_box(ax9, 6.2, 4.55, 5.4, 1.55, C_GOAL, C_OK, lw=2.0)
ax9.text(
    8.9, 5.55, 'What was missing',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    fontweight='bold', color=C_OK,
)
ax9.text(
    8.9, 4.95,
    'Why train at all?\n'
    'No goal => no honest assessment.',
    ha='center', va='center', fontsize=FS_ANNOT,
)

# Field use arrow story
round_box(ax9, 0.4, 2.55, 3.5, 1.7, C_NEW, '#2874a6', lw=1.8)
ax9.text(2.15, 3.7, 'Train', ha='center', fontsize=FS_ANNOT + 1,
         fontweight='bold', color=C_DARK)
ax9.text(2.15, 3.15, 'fit observed\ncases', ha='center', va='center',
         fontsize=FS_ANNOT)

arrow(ax9, 4.05, 3.4, 4.55, 3.4, C_DARK, lw=2.2)

round_box(ax9, 4.7, 2.55, 3.5, 1.7, C_NEW, '#2874a6', lw=1.8)
ax9.text(6.45, 3.7, 'Deploy', ha='center', fontsize=FS_ANNOT + 1,
         fontweight='bold', color=C_DARK)
ax9.text(6.45, 3.15, 'infer in the\nfield', ha='center', va='center',
         fontsize=FS_ANNOT)

arrow(ax9, 8.35, 3.4, 8.85, 3.4, C_DARK, lw=2.2)

round_box(ax9, 9.0, 2.55, 2.6, 1.7, C_GOAL, C_OK, lw=2.0)
ax9.text(10.3, 3.7, 'Succeed', ha='center', fontsize=FS_ANNOT + 1,
         fontweight='bold', color=C_OK)
ax9.text(10.3, 3.15, 'right choice\non new cases', ha='center', va='center',
         fontsize=FS_ANNOT)

# Generalization name + advice
round_box(ax9, 0.4, 0.35, 11.2, 1.95, C_MUTE, '0.45', lw=1.8)
ax9.text(
    6.0, 1.7,
    'That leap -- from observations to an unobserved case -- is generalization.\n'
    'We cannot forever escape it: learn from what we saw, then choose where we have not.',
    ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
    color=C_DARK,
)
ax9.text(
    6.0, 0.75,
    'Advice: training loss is the practice score; field success is the real exam.\n'
    'Hold out unseen cases. Optimize the loop -- but judge the model by what it has never trained on.',
    ha='center', va='center', fontsize=FS_ANNOT,
)
show_maximized(fig9)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("123 -- EXPLAINING 0270_torch_training.ipynb")
print("=" * 60)
print(f"""
Notebook essence:
  y_hat = w * x          (nn.Linear(..., bias=False))
  loss  = MSE(y_hat, y)
  loop  = predict -> loss -> zero_grad -> backward -> step

Data: {N} points, y = 2x + mild noise, seed 523, lr={LR:g}, steps={STEPS} (full batch)

This run:
  start w = {w0:.4f}
  final w = {w_final:.4f}   (true slope 2)
  final |grad| = {abs(grad_path[-1]):.4f}
  final MSE = {mse_final:.4f}

Teaching point:
  Bigger models reuse this same loop.
  If the loop is unclear, scale will not save you.

What's still missing (Step 11):
  The loop is mechanism. The goal is field inference on unobserved cases.
  That goal has a name: generalization.
  Without a goal, there is no assessment -- only a shrinking training loss.
""")
