"""
193_Huber_Loss.py

Huber loss -- the compromise between MSE and MAE, and why it exists.

Sketch:
  1. Definition -- quadratic near zero, linear in the tails
  2. Why Huber -- fix MSE's outlier panic without MAE's kink drama
  3. Browse the three curves: MSE, MAE, Huber
  4. Drag residual: watch which regime you are in
  5. Same outlier data: who owns the average?
  6. Fit duel: Huber sits between MSE and MAE
  7. Takeaway -- when to choose Huber

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


# -------------------------------------------------------
# FIGURE 1: Definition
# -------------------------------------------------------
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
show_maximized(fig0)


# -------------------------------------------------------
# FIGURE 2: Why Huber
# -------------------------------------------------------
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
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 3: Three curves
# -------------------------------------------------------
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
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 4: Slider regime
# -------------------------------------------------------
fig3, (ax3a, ax3b) = new_figure(1, 2)
fig3.suptitle(
    'Step 4: Drag the residual -- which regime are you in?',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
fig3.subplots_adjust(bottom=0.20, wspace=0.28)

r0 = 0.4
ax3a.plot(r, 0.5 * r ** 2, color=C_ACCENT, lw=1.8, alpha=0.55, label='MSE')
ax3a.plot(r, np.abs(r), color=C_MAE, lw=1.8, alpha=0.55, label='MAE')
ax3a.plot(r, huber(r, DELTA), color=C_HUB, lw=2.8, label='Huber')
ax3a.axvline(-DELTA, color='0.5', ls='--', lw=1.0)
ax3a.axvline(DELTA, color='0.5', ls='--', lw=1.0)
mk, = ax3a.plot([r0], [huber(r0)], 'o', ms=12, color=C_HUB)
ax3a.set_xlim(-4, 4)
ax3a.set_ylim(-0.2, 10)
ax3a.set_xlabel('r', fontsize=FS_AXIS)
ax3a.set_ylabel('penalty', fontsize=FS_AXIS)
ax3a.set_title('Live point on Huber', fontsize=FS_AX_TITLE - 1)
ax3a.legend(fontsize=FS_LEGEND - 1, loc='upper center')
ax3a.grid(True, alpha=0.3)

ax3b.set_xlim(0, 10)
ax3b.set_ylim(0, 10)
ax3b.axis('off')
ax3b.set_title('Status card', fontsize=FS_AX_TITLE - 1)
status_box = round_box(ax3b, 0.6, 1.5, 8.8, 7.0, C_GOAL, C_OK, lw=2.2)
hud = ax3b.text(5.0, 5.0, '', ha='center', va='center', fontsize=FS_ANNOT + 1)

ax_sl = fig3.add_axes([0.20, 0.07, 0.60, 0.04])
slider_r = Slider(ax_sl, 'r', -3.5, 3.5, valinit=r0, valstep=0.05)


def _update(_=None):
    rr = float(slider_r.val)
    hh = float(huber(rr))
    mk.set_data([rr], [hh])
    regime = 'MSE-like (quadratic)' if abs(rr) <= DELTA else 'MAE-like (linear)'
    hud.set_text(
        f'r = {rr:.2f}\n'
        f'|r| = {abs(rr):.2f}\n'
        f'delta = {DELTA:g}\n\n'
        f'regime: {regime}\n\n'
        f'Huber = {hh:.2f}\n'
        f'MSE/2 = {0.5 * rr ** 2:.2f}\n'
        f'MAE   = {abs(rr):.2f}'
    )
    fig3.canvas.draw_idle()


slider_r.on_changed(_update)
_update()
print('Step 4: drag r across +/- delta; watch the regime switch.')
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 5: Outlier ownership
# -------------------------------------------------------
fig4, (ax4a, ax4b) = new_figure(1, 2)
fig4.suptitle(
    'Step 5: Same outlier -- Huber caps the panic',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

r_clean = RNG.normal(0.0, 0.35, size=8)
r_out = np.array([4.8])
r_all = np.concatenate([r_clean, r_out])
pen_mse = 0.5 * r_all ** 2
pen_mae = np.abs(r_all)
pen_hub = huber(r_all, DELTA)
share = {
    'MSE': pen_mse[-1] / pen_mse.sum(),
    'MAE': pen_mae[-1] / pen_mae.sum(),
    'Huber': pen_hub[-1] / pen_hub.sum(),
}

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
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 6: Fit duel
# -------------------------------------------------------
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
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 7: Takeaway
# -------------------------------------------------------
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
show_maximized(fig6)


print("\n" + "=" * 60)
print("193 HUBER LOSS")
print("=" * 60)
print(f"""
Why Huber: MSE is too angry at outliers; MAE is robust but kinked at 0.
Huber = MSE inside |r|<=delta, MAE-like outside.

Outlier ownership on the demo set:
  MSE {100*share['MSE']:.0f}% | MAE {100*share['MAE']:.0f}% | Huber {100*share['Huber']:.0f}%

See also: 191_Mean_Squared_Error.py, 192_Mean_Absolute_Error.py, 190_loss_functions.py
""")
