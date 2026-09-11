"""
193_Huber_Loss.py

Huber loss -- the compromise between MSE and MAE, and why it exists.

Sketch:
  1. Definition -- quadratic near zero, linear in the tails
  2. Why Huber -- fix MSE's outlier panic without MAE's kink drama
  3. Browse the three curves: MSE, MAE, Huber
  4. Drag residual r=yhat-y: ball rides Huber; curves are losses vs r
  5. Same outlier data: who owns the average?
  6. Fit duel: Huber sits between MSE and MAE
  7. Takeaway -- when to choose Huber
  8. Torch -- from-scratch Huber vs nn.HuberLoss

Companion to 191_Mean_Squared_Error.py, 192_Mean_Absolute_Error.py,
190_loss_functions.py.
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

np.random.seed(13)
RNG = np.random.default_rng(13)

DELTA = 1.0


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
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.15',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


def huber(r, delta=DELTA):
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 0.5 * r ** 2, delta * (abs_r - 0.5 * delta))


C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_MAE = '#6c3483'
C_HUB = '#1e8449'

# Shared outlier-ownership demo (Step 5 + summary print)
R_CLEAN5 = RNG.normal(0.0, 0.35, size=8)
R_OUT5 = np.array([4.8])
R_ALL5 = np.concatenate([R_CLEAN5, R_OUT5])
PEN_MSE5 = 0.5 * R_ALL5 ** 2
PEN_MAE5 = np.abs(R_ALL5)
PEN_HUB5 = huber(R_ALL5, DELTA)
share = {
    'MSE': float(PEN_MSE5[-1] / PEN_MSE5.sum()),
    'MAE': float(PEN_MAE5[-1] / PEN_MAE5.sum()),
    'Huber': float(PEN_HUB5[-1] / PEN_HUB5.sum()),
}


# -------------------------------------------------------
# FIGURE 1: Definition
# -------------------------------------------------------
def build_step_1():
    fig0, ax0 = new_figure()
    fig0.suptitle(
        'Step 1: Huber loss -- definition',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax0)

    round_box(ax0, 0.5, 5.35, 11.0, 2.25, C_DARK, C_DARK, lw=0)
    ax0.text(
        6.0, 6.85,
        'Huber is a regression loss that switches personality with |r|.',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
        fontweight='bold', color='#f9e79f',
    )
    ax0.text(
        6.0, 5.95,
        rf'$\ell_\delta(r)=\frac{{1}}{{2}}r^{2}$   if $|r|\leq\delta$'
        '     |     '
        rf'$\ell_\delta(r)=\delta(|r|-\frac{{1}}{{2}}\delta)$   if $|r|>\delta$'
        '\n'
        rf'(here $\delta={DELTA:g}$)',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
        fontweight='bold', color='white',
    )

    round_box(ax0, 0.5, 2.85, 5.3, 2.2, C_NEW, '#2874a6', lw=2.0)
    ax0.text(3.15, 4.5, 'Near zero', ha='center',
             fontsize=FS_ANNOT + 2, fontweight='bold', color=C_DARK)
    ax0.text(3.15, 3.6,
             'Behaves like MSE:\n'
             'smooth bowl, sensitive\n'
             'to small residuals.',
             ha='center', va='center', fontsize=FS_ANNOT + 1)

    round_box(ax0, 6.2, 2.85, 5.3, 2.2, C_GOAL, C_OK, lw=2.0)
    ax0.text(8.85, 4.5, 'In the tails', ha='center',
             fontsize=FS_ANNOT + 2, fontweight='bold', color=C_OK)
    ax0.text(8.85, 3.6,
             'Behaves like MAE:\n'
             'linear growth,\n'
             'outliers stop exploding.',
             ha='center', va='center', fontsize=FS_ANNOT + 1)

    round_box(ax0, 0.5, 0.35, 11.0, 2.2, '#eafaf1', C_HUB, lw=2.0)
    ax0.text(
        6.0, 1.45,
        r'Training loss:  $L=\frac{1}{N}\sum_i \ell_\delta(\hat y_i-y_i)$'
        '\n'
        'delta is a knob: smaller delta -> more MAE-like; larger -> more MSE-like.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig0



# -------------------------------------------------------
# FIGURE 2: Why Huber
# -------------------------------------------------------
def build_step_2():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        'Step 2: Why Huber exists',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax1)

    cards = [
        (0.4, 4.2, C_ACCENT, '#c0392b', 'MSE problem',
         'Smooth and nice near zero,\n'
         'but one wild residual\n'
         'owns the average and\n'
         'yanks the fit.'),
        (4.2, 4.2, C_MAE, '#6c3483', 'MAE problem',
         'Robust to outliers,\n'
         'but has a sharp kink at 0.\n'
         'Gradients do not shrink\n'
         'as you approach the target.'),
        (8.0, 4.2, C_HUB, '#1e8449', 'Huber answer',
         'Keep MSE smoothness\n'
         'where data are honest.\n'
         'Switch to MAE linear\n'
         'when |r| gets suspicious.'),
    ]
    for x0, y0, edge, _, title, body in cards:
        face = '#fdebd0' if edge == C_ACCENT else ('#f5eef8' if edge == C_MAE else '#eafaf1')
        round_box(ax1, x0, y0, 3.5, 3.2, face, edge, lw=2.2)
        ax1.text(x0 + 1.75, y0 + 2.55, title, ha='center',
                 fontsize=FS_ANNOT + 2, fontweight='bold', color=edge)
        ax1.text(x0 + 1.75, y0 + 1.25, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 1)

    round_box(ax1, 0.4, 0.45, 11.2, 3.3, C_MUTE, '0.45', lw=1.8)
    ax1.text(
        6.0, 2.1,
        'Why Huber?\n\n'
        'Because real regression data are mostly well-behaved with occasional disasters.\n'
        'You want careful fine-tuning on the bulk, and capped revenge on the disasters.\n'
        'Huber is that policy, with one tunable threshold delta.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig1



# -------------------------------------------------------
# FIGURE 3: Three curves
# -------------------------------------------------------
def build_step_3():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        'Step 3: Three coaches on the same residual axis',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    r = np.linspace(-4, 4, 500)
    ax2.plot(r, 0.5 * r ** 2, color=C_ACCENT, lw=2.6, label=r'MSE $\frac{1}{2}r^2$')
    ax2.plot(r, np.abs(r), color=C_MAE, lw=2.6, label=r'MAE $|r|$')
    ax2.plot(r, huber(r, DELTA), color=C_HUB, lw=3.2, label=rf'Huber $\delta={DELTA:g}$')
    ax2.axvline(-DELTA, color='0.5', ls='--', lw=1.2)
    ax2.axvline(DELTA, color='0.5', ls='--', lw=1.2)
    ax2.axvline(0, color='0.6', lw=0.8)
    ax2.set_xlabel('residual r', fontsize=FS_AXIS)
    ax2.set_ylabel('penalty', fontsize=FS_AXIS)
    ax2.set_title('Huber hugs MSE inside +/- delta, then follows MAE slope', fontsize=FS_AX_TITLE - 2)
    ax2.legend(fontsize=FS_LEGEND, loc='upper center')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.2, 10)
    ax2.text(
        0.02, 0.98,
        'Dashed lines = +/- delta.\n'
        'Inside: quadratic.\n'
        'Outside: linear.',
        transform=ax2.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    return fig2



# -------------------------------------------------------
# FIGURE 4: Slider regime
# -------------------------------------------------------
def build_step_4():
    fig3, (ax3a, ax3b) = new_figure(1, 2)
    fig3.suptitle(
        r'Step 4: Drag residual $r=\hat y-y$ -- ride the Huber curve',
        fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
    )
    fig3.subplots_adjust(bottom=0.22, left=0.08, right=0.98, wspace=0.28)

    r0 = 0.4
    # Dense residual grid for smooth loss curves (do NOT reuse Torch demo `r`).
    r_grid = np.linspace(-4.0, 4.0, 500)
    # These are LOSS CURVES vs residual -- not training trajectories over time.
    ax3a.plot(r_grid, 0.5 * r_grid ** 2, color=C_ACCENT, lw=1.8, alpha=0.55,
              label=r'MSE: $\frac{1}{2}r^2$')
    ax3a.plot(r_grid, np.abs(r_grid), color=C_MAE, lw=1.8, alpha=0.55,
              label=r'MAE: $|r|$')
    ax3a.plot(r_grid, huber(r_grid, DELTA), color=C_HUB, lw=2.8,
              label=rf'Huber $\ell_\delta(r)$')
    ax3a.axvline(-DELTA, color='0.5', ls='--', lw=1.2)
    ax3a.axvline(DELTA, color='0.5', ls='--', lw=1.2)
    ax3a.text(
        DELTA + 0.08, 9.2, rf'$+\delta={DELTA:g}$',
        color='0.35', fontsize=FS_TICK, va='top',
    )
    ax3a.text(
        -DELTA - 0.08, 9.2, rf'$-\delta$',
        color='0.35', fontsize=FS_TICK, va='top', ha='right',
    )
    mk, = ax3a.plot([r0], [huber(r0)], 'o', ms=14, color=C_HUB, zorder=5,
                    label='ball = current $(r,\\ell_\\delta(r))$')
    ax3a.set_xlim(-4, 4)
    ax3a.set_ylim(-0.2, 10)
    ax3a.set_xlabel(r'residual $r=\hat y-y$  (prediction minus target)', fontsize=FS_AXIS)
    ax3a.set_ylabel(r'penalty $\ell(r)$  (how hard we punish this residual)', fontsize=FS_AXIS - 1)
    ax3a.set_title(
        'Three loss functions of the same residual r',
        fontsize=FS_AX_TITLE - 2,
    )
    ax3a.legend(fontsize=FS_LEGEND - 2, loc='upper center')
    ax3a.grid(True, alpha=0.3)
    # Middle-left callout.
    ax3a.text(
        0.02, 0.50,
        'Penalty = the loss number for one residual r.\n'
        'Bigger penalty => that error hurts training more.\n\n'
        'Curves plot penalty vs r (not vs epoch).\n'
        'The ball rides the green Huber curve\n'
        'as you drag r.',
        transform=ax3a.transAxes, ha='left', va='center', fontsize=FS_TICK,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    ax3b.set_xlim(0, 10)
    ax3b.set_ylim(0, 10)
    ax3b.axis('off')
    ax3b.set_title('What am I looking at?', fontsize=FS_AX_TITLE - 1)
    round_box(ax3b, 0.6, 1.2, 8.8, 7.4, C_GOAL, C_OK, lw=2.2)
    hud = ax3b.text(5.0, 5.0, '', ha='center', va='center', fontsize=FS_ANNOT)

    ax_sl = fig3.add_axes([0.20, 0.07, 0.60, 0.04])
    ax_sl.set_navigate(False)
    ax_sl.set_zorder(20)
    slider_r = Slider(ax_sl, r'residual $r$', -3.5, 3.5, valinit=r0, valstep=0.05)

    def _update(_=None):
        rr = float(slider_r.val)
        hh = float(huber(rr))
        mk.set_data([rr], [hh])
        if abs(rr) <= DELTA:
            regime = 'INSIDE +/- delta: quadratic (MSE-like)'
            tip = 'Small error -> Huber matches MSE bowl.'
        else:
            regime = 'OUTSIDE +/- delta: linear (MAE-like)'
            tip = 'Large error -> Huber grows like MAE, not r^2.'
        hud.set_text(
            f'r = prediction - target = {rr:.2f}\n'
            f'|r| = {abs(rr):.2f}    delta = {DELTA:g}\n\n'
            f'{regime}\n'
            f'{tip}\n\n'
            f'Ball sits on Huber at  ({rr:.2f}, {hh:.2f})\n\n'
            f'Compare penalties at this r:\n'
            f'  Huber     = {hh:.2f}\n'
            f'  MSE (1/2 r^2) = {0.5 * rr ** 2:.2f}\n'
            f'  MAE (|r|) = {abs(rr):.2f}'
        )
        fig3.canvas.draw_idle()

    slider_r.on_changed(_update)
    fig3._sliders = (slider_r,)
    _update()
    print('Step 4: drag residual r; ball rides the Huber curve.')
    return fig3



# -------------------------------------------------------
# FIGURE 5: Outlier ownership
# -------------------------------------------------------
def build_step_5():
    fig4, (ax4a, ax4b) = new_figure(1, 2)
    fig4.suptitle(
        'Step 5: Same outlier -- Huber caps the panic',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    pen_mse = PEN_MSE5
    pen_mae = PEN_MAE5
    pen_hub = PEN_HUB5

    idx = np.arange(1, 10)
    ax4a.bar(idx - 0.25, pen_mse, width=0.25, color=C_ACCENT, alpha=0.8, label='MSE/2')
    ax4a.bar(idx, pen_hub, width=0.25, color=C_HUB, label='Huber')
    ax4a.bar(idx + 0.25, pen_mae, width=0.25, color=C_MAE, alpha=0.85, label='MAE')
    ax4a.set_xlabel('sample index', fontsize=FS_AXIS)
    ax4a.set_ylabel('per-sample penalty', fontsize=FS_AXIS)
    ax4a.set_title('Sample 9 is the disaster', fontsize=FS_AX_TITLE - 1)
    ax4a.legend(fontsize=FS_LEGEND - 1, loc='upper left')
    ax4a.grid(True, axis='y', alpha=0.3)

    names = list(share.keys())
    vals = [share[n] for n in names]
    colors = [C_ACCENT, C_MAE, C_HUB]
    ax4b.bar(names, vals, color=colors, edgecolor='white')
    ax4b.set_ylim(0, 1.05)
    ax4b.set_ylabel('outlier fraction of total penalty', fontsize=FS_AXIS)
    ax4b.set_title('Who owns the wallet?', fontsize=FS_AX_TITLE - 1)
    ax4b.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(vals):
        ax4b.text(i, v + 0.03, f'{100 * v:.0f}%', ha='center', fontsize=FS_ANNOT, fontweight='bold')
    ax4b.text(
        0.98, 0.55,
        'Huber lands between\n'
        'MSE panic and MAE calm.',
        transform=ax4b.transAxes, ha='right', va='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    return fig4



# -------------------------------------------------------
# FIGURE 6: Fit duel
# -------------------------------------------------------
def build_step_6():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        'Step 6: Fit duel -- Huber sits between the two extremes',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    x = np.linspace(-2.0, 2.0, 14)
    y = 0.75 * x + 0.12 * RNG.normal(size=x.size)
    y[11] += 5.0
    X = np.column_stack([x, np.ones_like(x)])
    w_mse, *_ = np.linalg.lstsq(X, y, rcond=None)


    def mae_obj(wb):
        return float(np.mean(np.abs(wb[0] * x + wb[1] - y)))


    def hub_obj(wb):
        return float(np.mean(huber(wb[0] * x + wb[1] - y, DELTA)))


    w_mae = minimize(mae_obj, x0=np.zeros(2), method='Nelder-Mead').x
    w_hub = minimize(hub_obj, x0=np.zeros(2), method='Nelder-Mead').x
    xg = np.linspace(-2.3, 2.3, 120)
    mask = np.ones_like(y, dtype=bool)
    mask[11] = False

    ax5.scatter(x[mask], y[mask], s=65, c='#2874a6', zorder=3, label='bulk')
    ax5.scatter([x[11]], [y[11]], s=120, c=C_ACCENT, zorder=4, label='outlier')
    # Draw order / legend sequence: MSE, MAE, Huber (Huber last).
    ax5.plot(xg, w_mse[0] * xg + w_mse[1], color=C_ACCENT, lw=2.8, label='MSE fit', zorder=5)
    ax5.plot(xg, w_mae[0] * xg + w_mae[1], color=C_MAE, lw=2.6, ls='--', label='MAE fit', zorder=6)
    ax5.plot(xg, w_hub[0] * xg + w_hub[1], color=C_HUB, lw=3.0, label='Huber fit', zorder=7)
    ax5.set_xlabel('x', fontsize=FS_AXIS)
    ax5.set_ylabel('y', fontsize=FS_AXIS)
    ax5.legend(fontsize=FS_LEGEND - 1, loc='upper left')
    ax5.grid(True, alpha=0.3)
    ax5.text(
        0.98, 0.02,
        'Sequence: MSE, MAE, then Huber.\n'
        'MSE leans hardest toward the red point.\n'
        'MAE hugs the blue crowd.\n'
        'Huber sits between -- usually where you want to live.',
        transform=ax5.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    fig5.tight_layout(rect=[0, 0, 1, 0.94])
    return fig5



# -------------------------------------------------------
# FIGURE 7: Takeaway
# -------------------------------------------------------
def build_step_7():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        'Step 7: Huber in one breath',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax6)

    cards = [
        (0.4, 4.35, C_NEW, '#2874a6', 'Near 0', 'Like MSE:\nsmooth gradients\nfor fine tuning'),
        (4.2, 4.35, C_GOAL, C_OK, 'Far tails', 'Like MAE:\nlinear, capped\nrevenge on outliers'),
        (8.0, 4.35, '#fdebd0', '#d68910', 'Knob', rf'delta = {DELTA:g}'
         '\n'
         'your trust radius\n'
         'for "normal" errors'),
    ]
    for x0, y0, face, edge, title, body in cards:
        round_box(ax6, x0, y0, 3.5, 2.85, face, edge, lw=2.2)
        ax6.text(x0 + 1.75, y0 + 2.2, title, ha='center',
                 fontsize=FS_AX_TITLE - 2, fontweight='bold', color=edge)
        ax6.text(x0 + 1.75, y0 + 1.1, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 1)

    round_box(ax6, 0.4, 0.4, 11.2, 3.5, C_MUTE, '0.45', lw=1.8)
    ax6.text(
        6.0, 2.15,
        'Choose Huber when:\n'
        'you believe most residuals are trustworthy,\n'
        'but a few measurements can be catastrophic,\n'
        'and you still want a smooth local loss for optimization.\n\n'
        'See MSE in 191, MAE in 192, and the curve zoo in 190_loss_functions.py.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig6



# -------------------------------------------------------
# FIGURE 8: Torch -- from scratch vs API
# -------------------------------------------------------
import torch
import torch.nn as nn

torch.manual_seed(193)
y_t = torch.tensor([1.0, 2.0, 0.0, 4.0])
yhat_t = torch.tensor([1.5, 1.0, 0.5, 7.0])  # last residual is large
delta_t = float(DELTA)
r_t = yhat_t - y_t
abs_r = r_t.abs()
huber_scratch = torch.where(
    abs_r <= delta_t,
    0.5 * r_t ** 2,
    delta_t * (abs_r - 0.5 * delta_t),
).mean()
huber_api = nn.HuberLoss(delta=delta_t)(yhat_t, y_t)

CODE_HUBER_SCRATCH = (
    'import torch\n'
    '\n'
    f'delta = {delta_t:g}\n'
    'y    = torch.tensor([1., 2., 0., 4.])\n'
    'yhat = torch.tensor([1.5, 1., 0.5, 7.])\n'
    'r = yhat - y\n'
    'a = r.abs()\n'
    '\n'
    '# from scratch (same formula as the slides)\n'
    'loss = torch.where(\n'
    '    a <= delta,\n'
    '    0.5 * r**2,\n'
    '    delta * (a - 0.5 * delta),\n'
    ').mean()\n'
    'print(loss.item())'
)
CODE_HUBER_API = (
    'import torch\n'
    'import torch.nn as nn\n'
    '\n'
    f'delta = {delta_t:g}\n'
    'y    = torch.tensor([1., 2., 0., 4.])\n'
    'yhat = torch.tensor([1.5, 1., 0.5, 7.])\n'
    '\n'
    '# off-the-shelf API\n'
    'loss = nn.HuberLoss(delta=delta)(yhat, y)\n'
    'print(loss.item())'
)

def build_step_8():
    fig_t, (ax_ts, ax_ta) = new_figure(1, 2)
    fig_t.suptitle(
        'Step 8: Torch Huber -- from scratch vs nn.HuberLoss',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    for ax, title, code, face, edge in [
        (ax_ts, 'From scratch', CODE_HUBER_SCRATCH, C_NEW, '#2874a6'),
        (ax_ta, 'Off-the-shelf API', CODE_HUBER_API, C_GOAL, C_OK),
    ]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=FS_AX_TITLE - 1, color=edge)
        round_box(ax, 0.35, 1.3, 9.3, 8.2, face, edge, lw=2.0)
        ax.text(0.7, 5.3, code, ha='left', va='center',
                fontsize=FS_ANNOT - 1, family='monospace', linespacing=1.35)
    round_box(ax_ts, 0.35, 0.25, 9.3, 0.9, '#fdebd0', '#d68910', lw=1.6)
    ax_ts.text(
        5.0, 0.7,
        f'scratch = {huber_scratch.item():.6f}   |   API = {huber_api.item():.6f}   |   match = {torch.allclose(huber_scratch, huber_api)}',
        ha='center', va='center', fontsize=FS_ANNOT,
    )
    fig_t.tight_layout(rect=[0, 0.02, 1, 0.94])
    print(f'Step 8: Torch Huber scratch={huber_scratch.item():.6f} API={huber_api.item():.6f}')
    return fig_t




run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
    build_step_6,
    build_step_7,
    build_step_8,
])

print("\n" + "=" * 60)
print("193 HUBER LOSS")
print("=" * 60)
print(f"""
Why Huber: MSE is too angry at outliers; MAE is robust but kinked at 0.
Huber = MSE inside |r|<=delta, MAE-like outside.

Outlier ownership on the demo set:
  MSE {100*share['MSE']:.0f}% | MAE {100*share['MAE']:.0f}% | Huber {100*share['Huber']:.0f}%

Step 8: Torch -- scratch Huber matches nn.HuberLoss(delta={DELTA:g}).

See also: 191_Mean_Squared_Error.py, 192_Mean_Absolute_Error.py, 190_loss_functions.py
""")
