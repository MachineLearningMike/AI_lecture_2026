"""
191_Mean_Squared_Error.py

Intuitive simulations for Mean Squared Error.

Sketch:
  1. Definition -- MSE as a formula
  2. Squaring stretches large residuals -- see it happen
  3. Drag the prediction: watch MSE light up
  4. One outlier can dominate the average
  5. Same points, different fits: who MSE prefers
  6. Takeaway: MSE is a soft, outlier-sensitive coach
  7. Torch -- from-scratch MSE vs nn.MSELoss

Companion to 0400_loss_functions.py and 190_loss_functions.py.
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
from matplotlib.widgets import Slider
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as patches
import numpy as np

np.random.seed(7)
RNG = np.random.default_rng(7)


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


# -------------------------------------------------------
# FIGURE 1: Definition formula
# -------------------------------------------------------
def build_step_1():
    fig0, ax0 = new_figure()
    fig0.suptitle(
        'Step 1: Mean Squared Error -- definition',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax0)

    # Top panel: more height (taken from middle row)
    round_box(ax0, 0.5, 4.65, 11.0, 2.95, C_DARK, C_DARK, lw=0)
    ax0.text(
        6.0, 7.05,
        'MSE is the standard loss for regression\n'
        '(predicting a continuous value).',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
        fontweight='bold', color='#f9e79f',
    )
    ax0.text(
        6.0, 5.95,
        r'One sample:   $\ell_{\mathrm{MSE}}(\hat y, y)=(\hat y - y)^{2}$',
        ha='center', va='center', fontsize=FS_AX_TITLE - 1,
        fontweight='bold', color='white',
    )
    ax0.text(
        6.0, 5.15,
        r'Training loss:   $L=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i - y_i)^{2}$',
        ha='center', va='center', fontsize=FS_AX_TITLE - 2,
        fontweight='bold', color='white',
    )

    # Middle row: a bit shorter
    round_box(ax0, 0.5, 1.85, 5.3, 2.55, C_NEW, '#2874a6', lw=2.0)
    ax0.text(
        3.15, 3.85,
        'Residual',
        ha='center', fontsize=FS_ANNOT + 2, fontweight='bold', color=C_DARK,
    )
    ax0.text(
        3.15, 2.95,
        r'$r=\hat y - y$'
        '\n'
        'signed prediction error',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    round_box(ax0, 6.2, 1.85, 5.3, 2.55, C_GOAL, C_OK, lw=2.0)
    ax0.text(
        8.85, 3.85,
        'Squared penalty',
        ha='center', fontsize=FS_ANNOT + 2, fontweight='bold', color=C_OK,
    )
    ax0.text(
        8.85, 2.95,
        r'$\ell=r^{2}=(\hat y-y)^{2}$'
        '\n'
        'always >= 0; zero iff perfect',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    # Bottom row: lowered; second sentence removed
    round_box(ax0, 0.5, 0.12, 11.0, 1.5, '#fdebd0', '#d68910', lw=2.0)
    ax0.text(
        6.0, 0.87,
        'Use MSE when the target y is continuous (regression).',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig0



# -------------------------------------------------------
# FIGURE 2: Squaring stretches errors
# -------------------------------------------------------
def build_step_2():
    fig1, (ax1a, ax1b) = new_figure(1, 2)
    fig1.suptitle(
        'Step 2: Squaring makes large mistakes loom large',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    residuals = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    abs_pen = np.abs(residuals)
    sq_pen = residuals ** 2

    x_pos = np.arange(len(residuals))
    w = 0.38
    ax1a.bar(x_pos - w / 2, abs_pen, width=w, color='#2874a6', label=r'$|r|$')
    ax1a.bar(x_pos + w / 2, sq_pen, width=w, color=C_ACCENT, label=r'$r^2$')
    ax1a.set_xticks(x_pos)
    ax1a.set_xticklabels([f'{r:g}' for r in residuals], fontsize=FS_TICK)
    ax1a.set_xlabel('residual r = y_hat - y', fontsize=FS_AXIS)
    ax1a.set_ylabel('penalty', fontsize=FS_AXIS)
    ax1a.set_title('Same residuals; two penalties', fontsize=FS_AX_TITLE - 1)
    ax1a.legend(fontsize=FS_LEGEND)
    ax1a.grid(True, axis='y', alpha=0.3)
    ax1a.text(
        0.02, 0.98,
        'At |r|=1: same cost.\n'
        'At |r|=3: square is 9,\n'
        'absolute is only 3.',
        transform=ax1a.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    r_line = np.linspace(-3.5, 3.5, 400)
    ax1b.plot(r_line, np.abs(r_line), color='#2874a6', lw=2.6, label=r'$|r|$')
    ax1b.plot(r_line, r_line ** 2, color=C_ACCENT, lw=2.6, label=r'$r^2$')
    ax1b.axvline(0, color='0.5', lw=0.8)
    ax1b.axhline(0, color='0.5', lw=0.8)
    ax1b.set_xlabel('residual r', fontsize=FS_AXIS)
    ax1b.set_ylabel('penalty', fontsize=FS_AXIS)
    ax1b.set_title('Continuous picture', fontsize=FS_AX_TITLE - 1)
    ax1b.legend(fontsize=FS_LEGEND, loc='upper center')
    ax1b.grid(True, alpha=0.3)
    ax1b.set_ylim(-0.2, 12.5)
    ax1b.text(
        0.50, 0.50,
        'MSE uses the red curve.\n'
        'Far-away mistakes get\n'
        'expensive fast.',
        transform=ax1b.transAxes, ha='center', va='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#fdebd0', edgecolor='#d68910', alpha=0.95),
    )

    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    return fig1



# -------------------------------------------------------
# FIGURE 3: Interactive -- drag prediction for one sample
# -------------------------------------------------------
def build_step_3():
    fig2, (ax2a, ax2b) = new_figure(1, 2)
    fig2.suptitle(
        'Step 3: Drag the prediction -- watch one-sample MSE',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    fig2.subplots_adjust(bottom=0.20, wspace=0.28)

    Y_TRUE = 1.5
    yhat0 = -1.0

    # Left: number line
    ax2a.set_xlim(-4, 4)
    ax2a.set_ylim(-1.2, 1.2)
    ax2a.axhline(0, color='0.6', lw=1.2)
    ax2a.scatter([Y_TRUE], [0], s=160, c=C_OK, zorder=5, label='target y')
    pred_pt, = ax2a.plot([yhat0], [0], 'o', ms=14, color=C_ACCENT, zorder=6, label='prediction y_hat')
    err_line, = ax2a.plot([Y_TRUE, yhat0], [0, 0], color='0.35', lw=3.0, zorder=4)
    ax2a.set_title('One sample on a number line', fontsize=FS_AX_TITLE - 1)
    ax2a.set_xlabel('value', fontsize=FS_AXIS)
    ax2a.set_yticks([])
    ax2a.legend(fontsize=FS_LEGEND, loc='upper left')
    ax2a.grid(True, axis='x', alpha=0.3)
    hud2a = ax2a.text(
        0.98, 0.05, '',
        transform=ax2a.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    # Right: residual -> squared map
    r_grid = np.linspace(-4, 4, 300)
    ax2b.plot(r_grid, r_grid ** 2, color='#2874a6', lw=2.5)
    marker2b, = ax2b.plot([yhat0 - Y_TRUE], [(yhat0 - Y_TRUE) ** 2], 'o',
                          ms=12, color=C_ACCENT)
    ax2b.axvline(0, color='0.5', lw=0.8)
    ax2b.set_xlim(-4, 4)
    ax2b.set_ylim(-0.5, 16)
    ax2b.set_xlabel(r'residual $r=\hat y-y$', fontsize=FS_AXIS)
    ax2b.set_ylabel(r'$r^2$', fontsize=FS_AXIS)
    ax2b.set_title('Where you sit on the MSE curve', fontsize=FS_AX_TITLE - 1)
    ax2b.grid(True, alpha=0.3)
    hud2b = ax2b.text(
        0.50, 0.98, '',
        transform=ax2b.transAxes, ha='center', va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
    )

    ax_sl = fig2.add_axes([0.20, 0.07, 0.60, 0.04])
    slider_yhat = Slider(ax_sl, r'$\hat y$', -3.5, 3.5, valinit=yhat0, valstep=0.05)


    def _update_one(_=None):
        yh = float(slider_yhat.val)
        r = yh - Y_TRUE
        mse = r ** 2
        pred_pt.set_data([yh], [0])
        err_line.set_data([Y_TRUE, yh], [0, 0])
        marker2b.set_data([r], [mse])
        hud2a.set_text(
            f'y = {Y_TRUE:.2f}\n'
            f'y_hat = {yh:.2f}\n'
            f'r = {r:.2f}'
        )
        hud2b.set_text(
            f'MSE = (y_hat - y)^2\n'
            f'    = ({yh:.2f} - {Y_TRUE:.2f})^2\n'
            f'    = {mse:.2f}'
        )
        fig2.canvas.draw_idle()


    slider_yhat.on_changed(_update_one)
    _update_one()
    print('Step 3: drag y_hat; watch residual and MSE move together.')
    return fig2



# -------------------------------------------------------
# FIGURE 4: Outlier dominates the average
# -------------------------------------------------------
def build_step_4():
    fig3, (ax3a, ax3b) = new_figure(1, 2)
    fig3.suptitle(
        'Step 4: One big residual can own the average MSE',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    # 8 well-behaved residuals + 1 huge one
    r_clean = RNG.normal(0.0, 0.4, size=8)
    r_out = np.array([4.5])
    r_all = np.concatenate([r_clean, r_out])
    sq_all = r_all ** 2
    mean_mse = float(sq_all.mean())
    mean_without = float((r_clean ** 2).mean())

    colors = ['#2874a6'] * 8 + [C_ACCENT]
    ax3a.bar(np.arange(1, 10), sq_all, color=colors, edgecolor='white', linewidth=0.6)
    ax3a.axhline(mean_mse, color='#e67e22', lw=2.4, ls='--',
                 label=f'mean MSE = {mean_mse:.2f}')
    ax3a.axhline(mean_without, color=C_OK, lw=2.0, ls=':',
                 label=f'mean without outlier = {mean_without:.2f}')
    ax3a.set_xlabel('sample index', fontsize=FS_AXIS)
    ax3a.set_ylabel(r'per-sample $r_i^2$', fontsize=FS_AXIS)
    ax3a.set_title('Contribution of each sample', fontsize=FS_AX_TITLE - 1)
    ax3a.legend(fontsize=FS_LEGEND - 1, loc='upper left')
    ax3a.grid(True, axis='y', alpha=0.3)
    ax3a.text(
        0.02, 0.50,
        'Red bar = outlier.\n'
        'Dashed orange = average\n'
        'with the outlier.\n'
        'Dotted green = average\n'
        'if we remove it.',
        transform=ax3a.transAxes, ha='left', va='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    # Pie of contribution
    ax3b.pie(
        [sq_all[:-1].sum(), sq_all[-1]],
        labels=['8 normal samples\n(sum of squares)', '1 outlier\n(sum of squares)'],
        colors=['#2874a6', C_ACCENT],
        autopct='%1.1f%%',
        textprops={'fontsize': FS_ANNOT},
        startangle=90,
    )
    ax3b.set_title('Who pays into the mean?', fontsize=FS_AX_TITLE - 1)

    fig3.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig3.text(
        0.5, 0.01,
        'Intuition: MSE average is pulled hard toward the worst mistakes.',
        ha='center', va='bottom', fontsize=FS_ANNOT + 1, fontweight='bold',
    )
    return fig3



# -------------------------------------------------------
# FIGURE 5: Which fit MSE likes
# -------------------------------------------------------
def build_step_5():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        'Step 5: Visible simulation -- MSE prefers chasing the outlier',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    x = np.linspace(-2.0, 2.0, 12)
    y = 0.8 * x + 0.15 * RNG.normal(size=x.size)
    # plant one huge outlier
    y[10] += 4.5

    # Candidate lines: "robust-ish" vs "MSE-ish" (fit by least squares)
    X = np.column_stack([x, np.ones_like(x)])
    w_mse, *_ = np.linalg.lstsq(X, y, rcond=None)
    # A flatter line closer to the bulk (ignore outlier approximately)
    mask = np.ones_like(y, dtype=bool)
    mask[10] = False
    w_bulk, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)

    xg = np.linspace(-2.3, 2.3, 100)
    y_mse = w_mse[0] * xg + w_mse[1]
    y_bulk = w_bulk[0] * xg + w_bulk[1]

    ax4.scatter(x[mask], y[mask], s=70, c='#2874a6', zorder=3, label='bulk points')
    ax4.scatter([x[10]], [y[10]], s=120, c=C_ACCENT, zorder=4, label='outlier')
    ax4.plot(xg, y_mse, color='#e67e22', lw=3.0,
             label=f'MSE fit  (mean sq err={np.mean((w_mse[0]*x + w_mse[1] - y)**2):.2f})')
    ax4.plot(xg, y_bulk, color=C_OK, lw=2.6, ls='--',
             label=f'bulk-only fit  (mean sq err={np.mean((w_bulk[0]*x + w_bulk[1] - y)**2):.2f})')
    ax4.set_xlabel('x', fontsize=FS_AXIS)
    ax4.set_ylabel('y', fontsize=FS_AXIS)
    ax4.set_title('Least-squares line vs a bulk-friendly line', fontsize=FS_AX_TITLE - 1)
    ax4.legend(fontsize=FS_LEGEND - 1, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.text(
        0.98, 0.02,
        'Orange minimizes MSE -- it leans toward the red outlier.\n'
        'Green ignores the outlier and fits the crowd,\n'
        'but its MSE score is worse. MSE rewards orange.',
        transform=ax4.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    return fig4



# -------------------------------------------------------
# FIGURE 6: Takeaway
# -------------------------------------------------------
def build_step_6():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        'Step 6: MSE in one breath',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax5)

    cards = [
        (0.4, 4.3, C_NEW, '#2874a6', 'Definition',
         r'$\ell=(\hat y-y)^2$'
         '\n'
         'average over samples\n'
         'for the training loss'),
        (4.2, 4.3, C_GOAL, C_OK, 'Geometry',
         'Parabolic bowl.\n'
         'Zero only when\n'
         'prediction = target.'),
        (8.0, 4.3, '#fdebd0', '#d68910', 'Personality',
         'Smooth gradients.\n'
         'Hates outliers.\n'
         'Chases big mistakes.'),
    ]

    for x0, y0, face, edge, title, body in cards:
        round_box(ax5, x0, y0, 3.5, 2.9, face, edge, lw=2.2)
        ax5.text(x0 + 1.75, y0 + 2.25, title, ha='center',
                 fontsize=FS_AX_TITLE - 2, fontweight='bold', color=edge)
        ax5.text(x0 + 1.75, y0 + 1.15, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 1)

    round_box(ax5, 0.4, 0.5, 11.2, 3.3, C_MUTE, '0.45', lw=1.8)
    ax5.text(
        6.0, 2.15,
        'Use MSE when squared deviations are the right notion of wrong,\n'
        'and when large errors should be taken very seriously.\n\n'
        'If one wild point should not own the fit, consider MAE / Huber\n'
        '(see 190_loss_functions.py).',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig5



# -------------------------------------------------------
# FIGURE 7: Torch -- from scratch vs API
# -------------------------------------------------------
import torch
import torch.nn as nn

torch.manual_seed(191)
y_t = torch.tensor([1.0, 2.0, 0.0, 4.0])
yhat_t = torch.tensor([1.5, 1.0, 0.5, 5.0])

mse_scratch = ((yhat_t - y_t) ** 2).mean()
mse_api = nn.MSELoss()(yhat_t, y_t)

CODE_MSE_SCRATCH = (
    'import torch\n'
    '\n'
    'y    = torch.tensor([1., 2., 0., 4.])\n'
    'yhat = torch.tensor([1.5, 1., 0.5, 5.])\n'
    '\n'
    '# from scratch\n'
    'loss = ((yhat - y) ** 2).mean()\n'
    'print(loss.item())'
)
CODE_MSE_API = (
    'import torch\n'
    'import torch.nn as nn\n'
    '\n'
    'y    = torch.tensor([1., 2., 0., 4.])\n'
    'yhat = torch.tensor([1.5, 1., 0.5, 5.])\n'
    '\n'
    '# off-the-shelf API\n'
    'loss = nn.MSELoss()(yhat, y)\n'
    'print(loss.item())'
)

def build_step_7():
    fig_t, (ax_ts, ax_ta) = new_figure(1, 2)
    fig_t.suptitle(
        'Step 7: Torch MSE -- from scratch vs nn.MSELoss',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    for ax, title, code, face, edge in [
        (ax_ts, 'From scratch', CODE_MSE_SCRATCH, C_NEW, '#2874a6'),
        (ax_ta, 'Off-the-shelf API', CODE_MSE_API, C_GOAL, C_OK),
    ]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=FS_AX_TITLE - 1, color=edge)
        round_box(ax, 0.35, 1.3, 9.3, 8.2, face, edge, lw=2.0)
        ax.text(0.7, 5.3, code, ha='left', va='center',
                fontsize=FS_ANNOT, family='monospace', linespacing=1.4)
    round_box(ax_ts, 0.35, 0.25, 9.3, 0.9, '#fdebd0', '#d68910', lw=1.6)
    ax_ts.text(
        5.0, 0.7,
        f'scratch = {mse_scratch.item():.6f}   |   API = {mse_api.item():.6f}   |   match = {torch.allclose(mse_scratch, mse_api)}',
        ha='center', va='center', fontsize=FS_ANNOT,
    )
    fig_t.tight_layout(rect=[0, 0.02, 1, 0.94])
    print(f'Step 7: Torch MSE scratch={mse_scratch.item():.6f} API={mse_api.item():.6f}')
    return fig_t




run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
    build_step_6,
    build_step_7,
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("191 MEAN SQUARED ERROR -- INTUITION")
print("=" * 60)
print("""
Step 1: Definition -- MSE for regression; ell=(y_hat-y)^2 ; L = mean over samples.
Step 2: Squaring stretches large residuals (3 costs 9, not 3).
Step 3: Drag y_hat -- residual and r^2 move together on the curve.
Step 4: One outlier can dominate the average MSE budget.
Step 5: Least squares leans toward outliers because that lowers MSE most.
Step 6: MSE = smooth, outlier-sensitive training coach.
Step 7: Torch -- ((yhat-y)**2).mean() matches nn.MSELoss().

See also: 0400_loss_functions.py (definition + 3D bowl), 190_loss_functions.py
""")
