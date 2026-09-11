"""
192_Mean_Absolute_Error.py

Intuitive simulations for Mean Absolute Error (MAE / L1 loss).

Sketch:
  1. Definition -- MAE as a formula (regression, robust cousin of MSE)
  2. Absolute penalty grows linearly -- no stretch of large residuals
  3. Drag the prediction: watch |r| light up
  4. Same outlier data: MAE average resists ownership
  5. Visible fit: MAE stays with the crowd; MSE chases the outlier
  6. Takeaway: MAE is a fair, outlier-tolerant coach
  7. Torch -- from-scratch MAE vs nn.L1Loss

Companion to 191_Mean_Squared_Error.py, 190_loss_functions.py,
0400_loss_functions.py, 0310_regression_classification.py.
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
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.optimize import minimize

np.random.seed(11)
RNG = np.random.default_rng(11)


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
C_MAE = '#6c3483'

# Shared outlier-ownership demo (Step 4 + summary print)
R_CLEAN4 = RNG.normal(0.0, 0.4, size=8)
R_OUT4 = np.array([4.5])
R_ALL4 = np.concatenate([R_CLEAN4, R_OUT4])
ABS_ALL4 = np.abs(R_ALL4)
SQ_ALL4 = R_ALL4 ** 2
share_mae = float(ABS_ALL4[-1] / ABS_ALL4.sum())
share_mse = float(SQ_ALL4[-1] / SQ_ALL4.sum())


# -------------------------------------------------------
# FIGURE 1: Definition
# -------------------------------------------------------
def build_step_1():
    fig0, ax0 = new_figure()
    fig0.suptitle(
        'Step 1: Mean Absolute Error -- definition',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax0)

    round_box(ax0, 0.5, 4.65, 11.0, 2.95, C_DARK, C_DARK, lw=0)
    ax0.text(
        6.0, 7.05,
        'MAE is also a regression loss --\n'
        'the robust cousin of MSE.',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
        fontweight='bold', color='#f9e79f',
    )
    ax0.text(
        6.0, 5.95,
        r'One sample:   $\ell_{\mathrm{MAE}}(\hat y, y)=|\hat y - y|$',
        ha='center', va='center', fontsize=FS_AX_TITLE - 1,
        fontweight='bold', color='white',
    )
    ax0.text(
        6.0, 5.15,
        r'Training loss:   $L=\frac{1}{N}\sum_{i=1}^{N}|\hat y_i - y_i|$',
        ha='center', va='center', fontsize=FS_AX_TITLE - 2,
        fontweight='bold', color='white',
    )

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
        'Absolute penalty',
        ha='center', fontsize=FS_ANNOT + 2, fontweight='bold', color=C_OK,
    )
    ax0.text(
        8.85, 2.95,
        r'$\ell=|r|=|\hat y-y|$'
        '\n'
        'every unit of error costs 1',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    round_box(ax0, 0.5, 0.12, 11.0, 1.5, '#f5eef8', C_MAE, lw=2.0)
    ax0.text(
        6.0, 0.87,
        'Use MAE for continuous y when you want robustness to wild points.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig0



# -------------------------------------------------------
# FIGURE 2: Linear vs quadratic growth
# -------------------------------------------------------
def build_step_2():
    fig1, (ax1a, ax1b) = new_figure(1, 2)
    fig1.suptitle(
        'Step 2: Absolute cost grows steadily -- no explosion of large r',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    residuals = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    abs_pen = np.abs(residuals)
    sq_pen = residuals ** 2
    x_pos = np.arange(len(residuals))
    bw = 0.38
    ax1a.bar(x_pos - bw / 2, abs_pen, width=bw, color=C_MAE, label=r'$|r|$  (MAE)')
    ax1a.bar(x_pos + bw / 2, sq_pen, width=bw, color=C_ACCENT, alpha=0.75, label=r'$r^2$  (MSE)')
    ax1a.set_xticks(x_pos)
    ax1a.set_xticklabels([f'{r:g}' for r in residuals], fontsize=FS_TICK)
    ax1a.set_xlabel('residual r = y_hat - y', fontsize=FS_AXIS)
    ax1a.set_ylabel('penalty', fontsize=FS_AXIS)
    ax1a.set_title('Same residuals; two personalities', fontsize=FS_AX_TITLE - 1)
    ax1a.legend(fontsize=FS_LEGEND)
    ax1a.grid(True, axis='y', alpha=0.3)
    ax1a.text(
        0.02, 0.98,
        'At |r|=3:\n'
        'MAE pays 3.\n'
        'MSE pays 9.',
        transform=ax1a.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    r_line = np.linspace(-3.5, 3.5, 400)
    ax1b.plot(r_line, np.abs(r_line), color=C_MAE, lw=3.0, label=r'$|r|$')
    ax1b.plot(r_line, r_line ** 2, color=C_ACCENT, lw=2.4, alpha=0.75, label=r'$r^2$')
    ax1b.axvline(0, color='0.5', lw=0.8)
    ax1b.axhline(0, color='0.5', lw=0.8)
    ax1b.set_xlabel('residual r', fontsize=FS_AXIS)
    ax1b.set_ylabel('penalty', fontsize=FS_AXIS)
    ax1b.set_title('V-shape vs bowl', fontsize=FS_AX_TITLE - 1)
    ax1b.legend(fontsize=FS_LEGEND, loc='upper center')
    ax1b.grid(True, alpha=0.3)
    ax1b.set_ylim(-0.2, 12.5)
    ax1b.text(
        0.50, 0.50,
        'MAE = purple V.\n'
        'Each extra unit of error\n'
        'costs exactly one more.',
        transform=ax1b.transAxes, ha='center', va='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#f5eef8', edgecolor=C_MAE, alpha=0.95),
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    return fig1



# -------------------------------------------------------
# FIGURE 3: Interactive slider
# -------------------------------------------------------
def build_step_3():
    fig2, (ax2a, ax2b) = new_figure(1, 2)
    fig2.suptitle(
        'Step 3: Drag the prediction -- watch one-sample MAE',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    fig2.subplots_adjust(bottom=0.20, wspace=0.28)

    Y_TRUE = 1.5
    yhat0 = -1.0

    ax2a.set_xlim(-4, 4)
    ax2a.set_ylim(-1.2, 1.2)
    ax2a.axhline(0, color='0.6', lw=1.2)
    ax2a.scatter([Y_TRUE], [0], s=160, c=C_OK, zorder=5, label='target y')
    pred_pt, = ax2a.plot([yhat0], [0], 'o', ms=14, color=C_MAE, zorder=6,
                         label='prediction y_hat')
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

    r_grid = np.linspace(-4, 4, 300)
    ax2b.plot(r_grid, np.abs(r_grid), color=C_MAE, lw=2.8)
    ax2b.plot(r_grid, r_grid ** 2, color=C_ACCENT, lw=1.8, alpha=0.45, ls='--',
              label=r'MSE $r^2$ (ghost)')
    marker2b, = ax2b.plot([yhat0 - Y_TRUE], [abs(yhat0 - Y_TRUE)], 'o',
                          ms=12, color=C_MAE)
    ax2b.axvline(0, color='0.5', lw=0.8)
    ax2b.set_xlim(-4, 4)
    ax2b.set_ylim(-0.3, 10)
    ax2b.set_xlabel(r'residual $r=\hat y-y$', fontsize=FS_AXIS)
    ax2b.set_ylabel(r'$|r|$', fontsize=FS_AXIS)
    ax2b.set_title('Where you sit on the MAE V', fontsize=FS_AX_TITLE - 1)
    ax2b.legend(fontsize=FS_LEGEND - 1, loc='upper right')
    ax2b.grid(True, alpha=0.3)
    hud2b = ax2b.text(
        0.50, 0.98, '',
        transform=ax2b.transAxes, ha='center', va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#f5eef8', edgecolor=C_MAE, alpha=0.95),
    )

    ax_sl = fig2.add_axes([0.20, 0.07, 0.60, 0.04])
    slider_yhat = Slider(ax_sl, r'$\hat y$', -3.5, 3.5, valinit=yhat0, valstep=0.05)


    def _update_one(_=None):
        yh = float(slider_yhat.val)
        r = yh - Y_TRUE
        mae = abs(r)
        pred_pt.set_data([yh], [0])
        err_line.set_data([Y_TRUE, yh], [0, 0])
        marker2b.set_data([r], [mae])
        hud2a.set_text(
            f'y = {Y_TRUE:.2f}\n'
            f'y_hat = {yh:.2f}\n'
            f'r = {r:.2f}'
        )
        hud2b.set_text(
            f'MAE = |y_hat - y|\n'
            f'    = |{yh:.2f} - {Y_TRUE:.2f}|\n'
            f'    = {mae:.2f}'
        )
        fig2.canvas.draw_idle()


    slider_yhat.on_changed(_update_one)
    _update_one()
    print('Step 3: drag y_hat; watch residual and MAE move together.')
    return fig2



# -------------------------------------------------------
# FIGURE 4: Outlier -- MAE vs MSE ownership
# -------------------------------------------------------
def build_step_4():
    fig3, axes3 = new_figure(1, 2)
    fig3.suptitle(
        'Step 4: Same outlier -- MAE shares the bill more fairly',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    abs_all = ABS_ALL4
    sq_all = SQ_ALL4
    mean_mae = float(abs_all.mean())
    mean_mse = float(sq_all.mean())

    ax3a, ax3b = axes3
    idx = np.arange(1, 10)
    ax3a.bar(idx - 0.18, abs_all, width=0.36, color=C_MAE, label=r'$|r_i|$')
    ax3a.bar(idx + 0.18, sq_all, width=0.36, color=C_ACCENT, alpha=0.75, label=r'$r_i^2$')
    ax3a.axhline(mean_mae, color=C_MAE, lw=2.0, ls='--',
                 label=f'mean MAE = {mean_mae:.2f}')
    ax3a.axhline(mean_mse, color=C_ACCENT, lw=2.0, ls=':',
                 label=f'mean MSE = {mean_mse:.2f}')
    ax3a.set_xlabel('sample index', fontsize=FS_AXIS)
    ax3a.set_ylabel('per-sample penalty', fontsize=FS_AXIS)
    ax3a.set_title('Contributions side by side', fontsize=FS_AX_TITLE - 1)
    ax3a.legend(fontsize=FS_LEGEND - 2, loc='upper left')
    ax3a.grid(True, axis='y', alpha=0.3)
    ax3a.text(
        0.02, 0.50,
        'Purple bars grow linearly.\n'
        'Red bars explode.\n'
        'Sample 9 is the outlier.',
        transform=ax3a.transAxes, ha='left', va='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    labels = ['8 normal\nsamples', '1 outlier']
    x_c = np.array([0, 1])
    bw = 0.35
    ax3b.bar(x_c - bw / 2,
             [1 - share_mae, share_mae],
             width=bw, color=['#aed6f1', C_MAE],
             label='MAE budget')
    ax3b.bar(x_c + bw / 2,
             [1 - share_mse, share_mse],
             width=bw, color=['#f5b7b1', C_ACCENT],
             label='MSE budget')
    ax3b.set_xticks(x_c)
    ax3b.set_xticklabels(labels, fontsize=FS_TICK)
    ax3b.set_ylim(0, 1.05)
    ax3b.set_ylabel('fraction of total penalty', fontsize=FS_AXIS)
    ax3b.set_title('Outlier share of the total penalty', fontsize=FS_AX_TITLE - 1)
    ax3b.legend(fontsize=FS_LEGEND - 1, loc='upper right')
    ax3b.grid(True, axis='y', alpha=0.3)
    ax3b.text(
        0.98, 0.55,
        f'Outlier owns\n'
        f'{100 * share_mae:.0f}% of MAE sum\n'
        f'vs {100 * share_mse:.0f}% of MSE sum',
        transform=ax3b.transAxes, ha='right', va='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#f5eef8', edgecolor=C_MAE, alpha=0.95),
    )

    fig3.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig3.text(
        0.5, 0.01,
        'Intuition: MAE still feels the outlier -- but does not hand it the whole wallet.',
        ha='center', va='bottom', fontsize=FS_ANNOT + 1, fontweight='bold',
    )
    return fig3



# -------------------------------------------------------
# FIGURE 5: Fit comparison MAE vs MSE
# -------------------------------------------------------
def build_step_5():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        'Step 5: Visible simulation -- MAE stays with the crowd',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    x = np.linspace(-2.0, 2.0, 12)
    y = 0.8 * x + 0.15 * RNG.normal(size=x.size)
    y[10] += 4.5  # outlier

    X = np.column_stack([x, np.ones_like(x)])
    w_mse, *_ = np.linalg.lstsq(X, y, rcond=None)


    def _mae_obj(wb):
        return float(np.mean(np.abs(wb[0] * x + wb[1] - y)))


    w_mae = minimize(_mae_obj, x0=np.array([0.0, 0.0]), method='Nelder-Mead').x

    xg = np.linspace(-2.3, 2.3, 100)
    mask = np.ones_like(y, dtype=bool)
    mask[10] = False

    ax4.scatter(x[mask], y[mask], s=70, c='#2874a6', zorder=3, label='bulk points')
    ax4.scatter([x[10]], [y[10]], s=120, c=C_ACCENT, zorder=4, label='outlier')
    ax4.plot(xg, w_mse[0] * xg + w_mse[1], color='#e67e22', lw=3.0,
             label=f'MSE fit   (MAE={_mae_obj(w_mse):.2f},  MSE={np.mean((w_mse[0]*x+w_mse[1]-y)**2):.2f})')
    ax4.plot(xg, w_mae[0] * xg + w_mae[1], color=C_MAE, lw=3.0,
             label=f'MAE fit   (MAE={_mae_obj(w_mae):.2f},  MSE={np.mean((w_mae[0]*x+w_mae[1]-y)**2):.2f})')
    ax4.set_xlabel('x', fontsize=FS_AXIS)
    ax4.set_ylabel('y', fontsize=FS_AXIS)
    ax4.set_title('Least squares vs least absolute deviations', fontsize=FS_AX_TITLE - 1)
    ax4.legend(fontsize=FS_LEGEND - 2, loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.text(
        0.98, 0.02,
        'Orange (MSE) leans toward the red outlier.\n'
        'Purple (MAE) stays with the blue crowd.\n'
        'Each loss elects the fit that looks best under its own score.',
        transform=ax4.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    return fig4



# -------------------------------------------------------
# FIGURE 6: Takeaway + median fact
# -------------------------------------------------------
def build_step_6():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        'Step 6: MAE in one breath',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax5)

    cards = [
        (0.4, 4.3, C_NEW, '#2874a6', 'Definition',
         r'$\ell=|\hat y-y|$'
         '\n'
         'average over samples\n'
         'for the training loss'),
        (4.2, 4.3, '#f5eef8', C_MAE, 'Geometry',
         'V-shape in residual.\n'
         'Linear sides.\n'
         'Kink at r = 0.'),
        (8.0, 4.3, C_GOAL, C_OK, 'Personality',
         'Fair unit pricing.\n'
         'Tolerates outliers.\n'
         'Median-minded.'),
    ]
    for x0, y0, face, edge, title, body in cards:
        round_box(ax5, x0, y0, 3.5, 2.9, face, edge, lw=2.2)
        ax5.text(x0 + 1.75, y0 + 2.25, title, ha='center',
                 fontsize=FS_AX_TITLE - 2, fontweight='bold', color=edge)
        ax5.text(x0 + 1.75, y0 + 1.15, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 1)

    round_box(ax5, 0.4, 0.45, 11.2, 3.4, C_MUTE, '0.45', lw=1.8)
    ax5.text(
        6.0, 2.15,
        'Bonus intuition: for predicting a single constant (no slope),\n'
        'MAE is minimized by the median of the targets -- not the mean.\n'
        'That is why MAE feels "majority-first".\n\n'
        'Use MAE when distance matters equally per unit, and wild points should not own the fit.\n'
        'Compare with MSE in 191_Mean_Squared_Error.py; see also Huber in 190_loss_functions.py.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig5



# -------------------------------------------------------
# FIGURE 7: Torch -- from scratch vs API
# -------------------------------------------------------
import torch
import torch.nn as nn

torch.manual_seed(192)
y_t = torch.tensor([1.0, 2.0, 0.0, 4.0])
yhat_t = torch.tensor([1.5, 1.0, 0.5, 5.0])

mae_scratch = (yhat_t - y_t).abs().mean()
mae_api = nn.L1Loss()(yhat_t, y_t)

CODE_MAE_SCRATCH = (
    'import torch\n'
    '\n'
    'y    = torch.tensor([1., 2., 0., 4.])\n'
    'yhat = torch.tensor([1.5, 1., 0.5, 5.])\n'
    '\n'
    '# from scratch\n'
    'loss = (yhat - y).abs().mean()\n'
    'print(loss.item())'
)
CODE_MAE_API = (
    'import torch\n'
    'import torch.nn as nn\n'
    '\n'
    'y    = torch.tensor([1., 2., 0., 4.])\n'
    'yhat = torch.tensor([1.5, 1., 0.5, 5.])\n'
    '\n'
    '# off-the-shelf API\n'
    'loss = nn.L1Loss()(yhat, y)\n'
    'print(loss.item())'
)

def build_step_7():
    fig_t, (ax_ts, ax_ta) = new_figure(1, 2)
    fig_t.suptitle(
        'Step 7: Torch MAE -- from scratch vs nn.L1Loss',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    for ax, title, code, face, edge in [
        (ax_ts, 'From scratch', CODE_MAE_SCRATCH, C_NEW, '#2874a6'),
        (ax_ta, 'Off-the-shelf API', CODE_MAE_API, C_GOAL, C_OK),
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
        f'scratch = {mae_scratch.item():.6f}   |   API = {mae_api.item():.6f}   |   match = {torch.allclose(mae_scratch, mae_api)}',
        ha='center', va='center', fontsize=FS_ANNOT,
    )
    fig_t.tight_layout(rect=[0, 0.02, 1, 0.94])
    print(f'Step 7: Torch MAE scratch={mae_scratch.item():.6f} API={mae_api.item():.6f}')
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
print("192 MEAN ABSOLUTE ERROR -- INTUITION")
print("=" * 60)
print(f"""
Step 1: Definition -- MAE for robust regression; ell=|y_hat-y|; L = mean |r|.
Step 2: Absolute penalty grows linearly (3 costs 3, not 9).
Step 3: Drag y_hat -- residual and |r| move together on the V.
Step 4: Outlier share of total penalty: MAE {100*share_mae:.0f}% vs MSE {100*share_mse:.0f}%.
Step 5: MAE fit stays with the bulk; MSE fit leans toward the outlier.
Step 6: MAE = fair unit pricing, kinked at 0, median-minded.
Step 7: Torch -- (yhat-y).abs().mean() matches nn.L1Loss().

See also: 191_Mean_Squared_Error.py, 190_loss_functions.py, 0310_regression_classification.py
""")
