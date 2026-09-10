"""
194_Binary_Cross_Entropy.py

Binary Cross-Entropy (BCE) -- intuition and simulations for two-class learning.

Sketch:
  1. Definition -- BCE for binary classification (+ sigmoid)
  2. Browsable 3D BCE(y, p) over continuous y and p, plus p-slider
  3. Entropy H(p) = BCE(p, p) -- the diagonal slice; slider on p
  4. Probability, not distance -- the coin-flip story
  5. Drag the probability p: watch BCE for y=0 and y=1
  6. From logit s through sigmoid to BCE
  7. Correct class with low confidence still pays
  8. Takeaway -- BCE as a soft coach for a Bernoulli label
  9. Torch -- from-scratch BCE vs nn.BCELoss

Companion to 0310_regression_classification.py, 190_loss_functions.py,
0820_softmax.py.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

np.random.seed(17)
RNG = np.random.default_rng(17)


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


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def bce(p, y):
    """Hard-label BCE: y in {0,1}, p in (0,1). Argument order is (p, y)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bce_soft(y, p, eps=1e-3):
    """Continuous BCE over soft labels y in [0,1] and predictions p in (0,1)."""
    p = np.clip(p, eps, 1.0 - eps)
    y = np.clip(y, 0.0, 1.0)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_BIN = '#d68910'


# -------------------------------------------------------
# FIGURE 1: Definition
# -------------------------------------------------------
def build_step_1():
    fig0, ax0 = new_figure()
    fig0.suptitle(
        'Step 1: Binary Cross-Entropy -- definition',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax0)

    round_box(ax0, 0.5, 5.2, 11.0, 2.4, C_DARK, C_DARK, lw=0)
    ax0.text(
        6.0, 7.0,
        'BCE is the standard loss for binary classification\n'
        '(choose one of two labels).',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
        fontweight='bold', color='#f9e79f',
    )
    ax0.text(
        6.0, 5.85,
        r'$p=\sigma(s)=\frac{1}{1+e^{-s}}$'
        '      then      '
        r'$\ell_{\mathrm{BCE}}=-(y\log p+(1-y)\log(1-p))$',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
        fontweight='bold', color='white',
    )

    round_box(ax0, 0.5, 2.7, 5.3, 2.2, C_NEW, '#2874a6', lw=2.0)
    ax0.text(3.15, 4.4, 'Label', ha='center',
             fontsize=FS_ANNOT + 2, fontweight='bold', color=C_DARK)
    ax0.text(3.15, 3.5,
             r'$y\in\{0,1\}$'
             '\n'
             'a Bernoulli bit',
             ha='center', va='center', fontsize=FS_ANNOT + 1)

    round_box(ax0, 6.2, 2.7, 5.3, 2.2, C_GOAL, C_OK, lw=2.0)
    ax0.text(8.85, 4.4, 'Model output', ha='center',
             fontsize=FS_ANNOT + 2, fontweight='bold', color=C_OK)
    ax0.text(8.85, 3.5,
             'logit s -> sigmoid -> p\n'
             'probability of class 1',
             ha='center', va='center', fontsize=FS_ANNOT + 1)

    round_box(ax0, 0.5, 0.35, 11.0, 2.05, '#fdebd0', C_BIN, lw=2.0)
    ax0.text(
        6.0, 1.35,
        'Embraces binary classification because wrongness is about\n'
        'a bad probability for a true 0/1 event -- not about Euclidean distance.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig0



# -------------------------------------------------------
# FIGURE 2: Browsable 3D BCE(y, p) + p slider
# -------------------------------------------------------
# Continuous extension: y in (0,1) is a soft label; p = y_hat in (0,1).
# Hard labels y=0 and y=1 are the edges of that continuum.
#   BCE(y, p) = - ( y log p + (1-y) log(1-p) )
def build_step_2():
    fig_bce3d = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig_bce3d.suptitle(
        'Step 2: BCE(y, p) as a browsable surface -- continuous y and p',
        fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
    )
    fig_bce3d.subplots_adjust(bottom=0.18, wspace=0.25)

    ax3d = fig_bce3d.add_subplot(1, 2, 1, projection='3d')
    eps = 1e-3
    y_grid = np.linspace(eps, 1.0 - eps, 60)
    p_grid3 = np.linspace(eps, 1.0 - eps, 60)
    YY, PP = np.meshgrid(y_grid, p_grid3)
    ZZ = bce_soft(YY, PP, eps=eps)

    surf = ax3d.plot_surface(
        YY, PP, ZZ, cmap='viridis', linewidth=0, antialiased=True, alpha=0.92,
    )
    # Hard-label edges: classical binary cases y≈0 and y≈1
    ax3d.plot(
        np.full_like(p_grid3, eps), p_grid3, bce_soft(0.0, p_grid3, eps=eps),
        color=C_ACCENT, lw=2.8, label=r'$y\approx 0$ curve',
    )
    ax3d.plot(
        np.full_like(p_grid3, 1.0 - eps), p_grid3, bce_soft(1.0, p_grid3, eps=eps),
        color=C_OK, lw=2.8, label=r'$y\approx 1$ curve',
    )
    ax3d.set_xlabel(r'continuous label $y$', fontsize=FS_AXIS - 1)
    ax3d.set_ylabel(r'prediction $p=\hat y$', fontsize=FS_AXIS - 1)
    ax3d.set_zlabel(r'$\mathrm{BCE}(y,p)$', fontsize=FS_AXIS - 1)
    ax3d.set_title(
        r'$\mathrm{BCE}=-(y\log p+(1-y)\log(1-p))$',
        fontsize=FS_AX_TITLE - 3,
    )
    ax3d.tick_params(labelsize=FS_TICK)
    ax3d.view_init(elev=28, azim=-55)
    ax3d.legend(loc='upper left', fontsize=FS_LEGEND - 2, framealpha=0.92)
    fig_bce3d.colorbar(surf, ax=ax3d, shrink=0.55, pad=0.08, label='BCE')
    ax3d.text2D(
        0.02, -0.08,
        'Drag to browse.\n'
        'True hard labels are\n'
        'just y=0 and y=1\n'
        'on this continuum.',
        transform=ax3d.transAxes, fontsize=FS_ANNOT,
        va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    # Right: slider on p -- BCE response across continuous y
    ax_slice = fig_bce3d.add_subplot(1, 2, 2)
    p_slide0 = 0.35
    y_line = np.linspace(eps, 1.0 - eps, 300)
    curve, = ax_slice.plot(
        y_line, bce_soft(y_line, p_slide0, eps=eps),
        color='#2874a6', lw=2.8, label='BCE vs y at fixed p',
    )
    mark_y0, = ax_slice.plot(
        [0.0], [bce_soft(0.0, p_slide0, eps=eps)],
        'o', ms=10, color=C_ACCENT, label='hard y=0',
    )
    mark_y1, = ax_slice.plot(
        [1.0], [bce_soft(1.0, p_slide0, eps=eps)],
        'o', ms=10, color=C_OK, label='hard y=1',
    )
    ax_slice.set_xlim(-0.05, 1.05)
    ax_slice.set_ylim(-0.1, 4.5)
    ax_slice.set_xlabel(r'continuous label $y$', fontsize=FS_AXIS)
    ax_slice.set_ylabel('BCE', fontsize=FS_AXIS)
    ax_slice.set_title('Slice at fixed p -- BCE vs y', fontsize=FS_AX_TITLE - 2)
    ax_slice.legend(fontsize=FS_LEGEND - 1, loc='upper right')
    ax_slice.grid(True, alpha=0.3)
    hud_p = ax_slice.text(
        0.02, 0.98, '',
        transform=ax_slice.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#fdebd0', edgecolor=C_BIN, alpha=0.95),
    )

    # Yellow function line on the 3D surface: BCE(y, p) at the slider's p
    # (same slice drawn on the right as BCE vs y).
    p_line_3d, = ax3d.plot(
        y_grid,
        np.full_like(y_grid, p_slide0),
        bce_soft(y_grid, p_slide0, eps=eps),
        color='#f1c40f', lw=4.0, zorder=10,
        label=r'$\mathrm{BCE}(y,p)$ at slider $p$',
    )

    # Slider spans both panels -- one p drives the 3D line and the right curve
    ax_sl_p = fig_bce3d.add_axes([0.18, 0.06, 0.64, 0.04])
    slider_p3d = Slider(ax_sl_p, r'$p$', 0.05, 0.95, valinit=p_slide0, valstep=0.01)


    def _update_p3d(_=None):
        p = float(slider_p3d.val)
        z_right = bce_soft(y_line, p, eps=eps)
        z_3d = bce_soft(y_grid, p, eps=eps)
        # Right panel: BCE vs continuous y at this p
        curve.set_ydata(z_right)
        mark_y0.set_data([0.0], [bce_soft(0.0, p, eps=eps)])
        mark_y1.set_data([1.0], [bce_soft(1.0, p, eps=eps)])
        # Left panel: same function as a line on the 3D surface (fixed-p slice)
        p_line_3d.set_data_3d(y_grid, np.full_like(y_grid, p), z_3d)
        p_line_3d.set_label(rf'$\mathrm{{BCE}}(y,{p:.2f})$')
        ax3d.legend(loc='upper left', fontsize=FS_LEGEND - 2, framealpha=0.92)
        hud_p.set_text(
            f'p = y_hat = {p:.2f}\n\n'
            f'BCE(y=0, p) = {bce_soft(0.0, p, eps=eps):.2f}\n'
            f'BCE(y=1, p) = {bce_soft(1.0, p, eps=eps):.2f}\n\n'
            'One p drives both panels:\n'
            'yellow 3D line = right curve.'
        )
        fig_bce3d.canvas.draw_idle()


    slider_p3d.on_changed(_update_p3d)
    _update_p3d()
    print('Step 2: drag p -- yellow BCE(y,p) line on 3D and right curve move together.')
    return fig_bce3d



# -------------------------------------------------------
# FIGURE 3: Entropy H(p) = BCE(p, p) -- diagonal slice
# -------------------------------------------------------
# Binary entropy is BCE restricted to the diagonal y = p:
#   H(p) = BCE(p, p) = -(p log p + (1-p) log(1-p))
# That is one slice of the continuous surface from Step 2.
def build_step_3():
    fig_ent, (ax_ent, ax_ent_note) = new_figure(1, 2)
    fig_ent.suptitle(
        r'Step 3: Entropy $H(p)=\mathrm{BCE}(p,p)$ -- the diagonal slice',
        fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
    )
    fig_ent.subplots_adjust(bottom=0.20, wspace=0.28)

    eps_h = 1e-3
    p_curve = np.linspace(eps_h, 1.0 - eps_h, 400)
    h_curve = bce_soft(p_curve, p_curve, eps=eps_h)
    ax_ent.plot(p_curve, h_curve, color='#2874a6', lw=2.8, label=r'$H(p)=\mathrm{BCE}(p,p)$')
    p_ent0 = 0.35
    mk_ent, = ax_ent.plot(
        [p_ent0], [bce_soft(p_ent0, p_ent0, eps=eps_h)],
        'o', ms=12, color=C_BIN, zorder=5,
    )
    ax_ent.axvline(0.5, color='0.55', ls='--', lw=1.0, alpha=0.8)
    ax_ent.axhline(np.log(2), color='0.55', ls=':', lw=1.0, alpha=0.8)
    ax_ent.text(
        0.52, np.log(2) + 0.04,
        r'$\max=\log 2$ at $p=\frac{1}{2}$',
        fontsize=FS_ANNOT, color='0.35',
    )
    ax_ent.set_xlim(0, 1)
    ax_ent.set_ylim(-0.05, 0.85)
    ax_ent.set_xlabel(r'probability $p$', fontsize=FS_AXIS)
    ax_ent.set_ylabel(r'$H(p)$', fontsize=FS_AXIS)
    ax_ent.set_title(
        r'$H(p)=-(p\log p+(1-p)\log(1-p))$',
        fontsize=FS_AX_TITLE - 2,
    )
    ax_ent.legend(fontsize=FS_LEGEND - 1, loc='upper right')
    ax_ent.grid(True, alpha=0.3)

    ax_ent_note.set_xlim(0, 10)
    ax_ent_note.set_ylim(0, 10)
    ax_ent_note.axis('off')
    round_box(ax_ent_note, 0.4, 0.6, 9.2, 8.8, C_MUTE, '0.45', lw=1.8)
    hud_ent = ax_ent_note.text(
        5.0, 5.2, '', ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    ax_sl_ent = fig_ent.add_axes([0.15, 0.07, 0.35, 0.04])
    slider_ent = Slider(ax_sl_ent, r'$p$', 0.05, 0.95, valinit=p_ent0, valstep=0.01)


    def _update_ent(_=None):
        p = float(slider_ent.val)
        h = float(bce_soft(p, p, eps=eps_h))
        mk_ent.set_data([p], [h])
        hud_ent.set_text(
            f'p = {p:.2f}\n'
            f'H(p) = BCE(p, p) = {h:.3f}\n\n'
            'This is the diagonal slice\n'
            'y = p of continuous BCE(y, p)\n'
            'from Step 2.\n\n'
            'Hard labels y in {0,1} sit at\n'
            'the ends: H(0)=H(1)=0\n'
            '(no uncertainty).\n'
            'p=1/2 is maximum entropy.'
        )
        fig_ent.canvas.draw_idle()


    slider_ent.on_changed(_update_ent)
    _update_ent()
    print('Step 3: entropy H(p)=BCE(p,p) is the diagonal slice; drag p.')
    return fig_ent



# -------------------------------------------------------
# FIGURE 4: Coin-flip story
# -------------------------------------------------------
def build_step_4():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        'Step 4: Think of the label as a coin -- BCE scores your odds',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax1)

    round_box(ax1, 0.5, 5.5, 11.0, 2.0, C_NEW, '#2874a6', lw=2.0)
    ax1.text(
        6.0, 6.5,
        'If y = 1, the true event happened. BCE = -log(p).\n'
        'Saying p=0.9 is cheap; saying p=0.1 is expensive.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    round_box(ax1, 0.5, 3.15, 11.0, 2.0, C_GOAL, C_OK, lw=2.0)
    ax1.text(
        6.0, 4.15,
        'If y = 0, the true event did not happen. BCE = -log(1-p).\n'
        'Saying p=0.1 is cheap; saying p=0.9 is expensive.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )

    round_box(ax1, 0.5, 0.5, 11.0, 2.3, '#fdebd0', C_BIN, lw=2.0)
    ax1.text(
        6.0, 1.65,
        '0-1 error only asks: was the hard decision wrong?\n'
        'BCE asks: how sincerely did your probability support the truth?\n'
        'That sincerity has a smooth slope -- so GD can train.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig1



# -------------------------------------------------------
# FIGURE 5: Slider on p (hard labels)
# -------------------------------------------------------
def build_step_5():
    fig2, (ax2a, ax2b) = new_figure(1, 2)
    fig2.suptitle(
        'Step 5: Drag probability p -- BCE for y=0 and y=1',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    fig2.subplots_adjust(bottom=0.20, wspace=0.28)

    p_grid = np.linspace(0.02, 0.98, 300)
    ax2a.plot(p_grid, bce(p_grid, 1), color=C_OK, lw=2.8, label='BCE if y=1')
    ax2a.plot(p_grid, bce(p_grid, 0), color=C_ACCENT, lw=2.8, label='BCE if y=0')
    p0 = 0.35
    mk1, = ax2a.plot([p0], [bce(p0, 1)], 'o', ms=11, color=C_OK)
    mk0, = ax2a.plot([p0], [bce(p0, 0)], 'o', ms=11, color=C_ACCENT)
    ax2a.set_xlim(0, 1)
    ax2a.set_ylim(-0.1, 4.2)
    ax2a.set_xlabel(r'predicted $p=\sigma(s)$', fontsize=FS_AXIS)
    ax2a.set_ylabel('BCE', fontsize=FS_AXIS)
    ax2a.set_title('Two curves; the true label picks one', fontsize=FS_AX_TITLE - 2)
    ax2a.legend(fontsize=FS_LEGEND, loc='upper center')
    ax2a.grid(True, alpha=0.3)

    ax2b.set_xlim(0, 10)
    ax2b.set_ylim(0, 10)
    ax2b.axis('off')
    round_box(ax2b, 0.6, 1.4, 8.8, 7.2, C_MUTE, '0.45', lw=1.8)
    hud = ax2b.text(5.0, 5.0, '', ha='center', va='center', fontsize=FS_ANNOT + 1)

    ax_sl = fig2.add_axes([0.20, 0.07, 0.60, 0.04])
    slider_p = Slider(ax_sl, 'p', 0.02, 0.98, valinit=p0, valstep=0.01)


    def _update_p(_=None):
        p = float(slider_p.val)
        mk1.set_data([p], [bce(p, 1)])
        mk0.set_data([p], [bce(p, 0)])
        hud.set_text(
            f'p = {p:.2f}\n\n'
            f'If y=1:  BCE = -log(p) = {bce(p, 1):.2f}\n'
            f'If y=0:  BCE = -log(1-p) = {bce(p, 0):.2f}\n\n'
            'Confidently wrong -> huge loss.\n'
            'Confidently right -> near zero.'
        )
        fig2.canvas.draw_idle()


    slider_p.on_changed(_update_p)
    _update_p()
    print('Step 5: drag p; watch BCE for both labels.')
    return fig2



# -------------------------------------------------------
# FIGURE 6: logit -> sigmoid -> BCE
# -------------------------------------------------------
def build_step_6():
    fig3, (ax3a, ax3b) = new_figure(1, 2)
    fig3.suptitle(
        'Step 6: From logit s through sigmoid into BCE',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    s = np.linspace(-6, 6, 400)
    p = sigmoid(s)
    ax3a.plot(s, p, color='#2874a6', lw=2.8)
    ax3a.axhline(0.5, color='0.5', ls='--', lw=1.0)
    ax3a.axvline(0, color='0.5', ls='--', lw=1.0)
    ax3a.set_xlabel('logit s', fontsize=FS_AXIS)
    ax3a.set_ylabel(r'$p=\sigma(s)$', fontsize=FS_AXIS)
    ax3a.set_title('Sigmoid squashes scores to (0,1)', fontsize=FS_AX_TITLE - 2)
    ax3a.grid(True, alpha=0.3)
    ax3a.text(
        0.02, 0.98,
        'Hard decision is often\n'
        'sign(s) or p>0.5.\n'
        'BCE still uses soft p.',
        transform=ax3a.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    ax3b.plot(s, bce(p, 1), color=C_OK, lw=2.6, label='y=1')
    ax3b.plot(s, bce(p, 0), color=C_ACCENT, lw=2.6, label='y=0')
    ax3b.set_xlabel('logit s', fontsize=FS_AXIS)
    ax3b.set_ylabel('BCE', fontsize=FS_AXIS)
    ax3b.set_title('BCE vs logit (after sigmoid)', fontsize=FS_AX_TITLE - 2)
    ax3b.legend(fontsize=FS_LEGEND)
    ax3b.grid(True, alpha=0.3)
    ax3b.set_ylim(-0.1, 6)
    ax3b.text(
        0.98, 0.98,
        'Smooth everywhere.\n'
        'Unlike 0-1 jump.',
        transform=ax3b.transAxes, ha='right', va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#fdebd0', edgecolor=C_BIN, alpha=0.95),
    )
    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    return fig3



# -------------------------------------------------------
# FIGURE 7: Confidence tax
# -------------------------------------------------------
def build_step_7():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        'Step 7: Even a correct hard decision can pay BCE tax',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    # Examples: correct but uncertain vs correct and confident
    cases = [
        ('y=1, p=0.55\n(barely correct)', bce(0.55, 1), C_BIN),
        ('y=1, p=0.95\n(confident correct)', bce(0.95, 1), C_OK),
        ('y=1, p=0.20\n(wrong & confident)', bce(0.20, 1), C_ACCENT),
    ]
    names = [c[0] for c in cases]
    vals = [c[1] for c in cases]
    colors = [c[2] for c in cases]
    ax4.bar(np.arange(3), vals, color=colors, edgecolor='white', width=0.65)
    ax4.set_xticks(np.arange(3))
    ax4.set_xticklabels(names, fontsize=FS_TICK)
    ax4.set_ylabel('BCE', fontsize=FS_AXIS)
    ax4.set_title('Same true label y=1; different probabilities', fontsize=FS_AX_TITLE - 1)
    ax4.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(vals):
        ax4.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=FS_ANNOT, fontweight='bold')
    ax4.text(
        0.98, 0.55,
        '0-1 error treats the first\n'
        'two as equally "correct".\n'
        'BCE prefers the confident one.\n'
        'That preference trains better\n'
        'calibrated probabilities.',
        transform=ax4.transAxes, ha='right', va='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    return fig4



# -------------------------------------------------------
# FIGURE 8: Takeaway
# -------------------------------------------------------
def build_step_8():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        'Step 8: Binary Cross-Entropy in one breath',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax5)

    cards = [
        (0.4, 4.3, C_NEW, '#2874a6', 'Use when', 'Binary labels\ny in {0,1}\nregression? No.'),
        (4.2, 4.3, C_GOAL, C_OK, 'Pipeline', 'logit s\n-> sigmoid p\n-> BCE(p, y)'),
        (8.0, 4.3, '#fdebd0', C_BIN, 'Personality', 'Soft, strict on\nconfident mistakes,\nrewards calibrated p'),
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
        'BCE is Shannon cross-entropy between the true Bernoulli bit and your predicted p.\n'
        'That is why it embraces binary classification.\n\n'
        'Next: multi-class needs a full probability vector -- see Categorical Cross-Entropy.\n'
        'Compare with MSE/MAE/Huber for continuous targets in 191-193.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig5



# -------------------------------------------------------
# FIGURE 9: Torch -- from scratch vs API
# -------------------------------------------------------
import torch
import torch.nn as nn

torch.manual_seed(194)
y_t = torch.tensor([1.0, 0.0, 1.0, 0.0])
p_t = torch.tensor([0.9, 0.2, 0.6, 0.1])
eps = 1e-7
p_c = p_t.clamp(eps, 1.0 - eps)
bce_scratch = (-(y_t * p_c.log() + (1.0 - y_t) * (1.0 - p_c).log())).mean()
bce_api = nn.BCELoss()(p_t, y_t)

CODE_BCE_SCRATCH = (
    'import torch\n'
    '\n'
    'y = torch.tensor([1., 0., 1., 0.])\n'
    'p = torch.tensor([0.9, 0.2, 0.6, 0.1])  # probs\n'
    'eps = 1e-7\n'
    'p = p.clamp(eps, 1 - eps)\n'
    '\n'
    '# from scratch\n'
    'loss = (-(y * p.log()\n'
    '         + (1 - y) * (1 - p).log())).mean()\n'
    'print(loss.item())'
)
CODE_BCE_API = (
    'import torch\n'
    'import torch.nn as nn\n'
    '\n'
    'y = torch.tensor([1., 0., 1., 0.])\n'
    'p = torch.tensor([0.9, 0.2, 0.6, 0.1])  # probs\n'
    '\n'
    '# off-the-shelf API (probs in, not logits)\n'
    'loss = nn.BCELoss()(p, y)\n'
    'print(loss.item())\n'
    '\n'
    '# training tip: prefer\n'
    '# nn.BCEWithLogitsLoss()(logits, y)'
)

def build_step_9():
    fig_t, (ax_ts, ax_ta) = new_figure(1, 2)
    fig_t.suptitle(
        'Step 9: Torch BCE -- from scratch vs nn.BCELoss',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    for ax, title, code, face, edge in [
        (ax_ts, 'From scratch', CODE_BCE_SCRATCH, C_NEW, '#2874a6'),
        (ax_ta, 'Off-the-shelf API', CODE_BCE_API, C_GOAL, C_OK),
    ]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=FS_AX_TITLE - 1, color=edge)
        round_box(ax, 0.35, 1.3, 9.3, 8.2, face, edge, lw=2.0)
        ax.text(0.7, 5.3, code, ha='left', va='center',
                fontsize=FS_ANNOT - 1, family='monospace', linespacing=1.35)
    round_box(ax_ts, 0.35, 0.25, 9.3, 0.9, '#fdebd0', C_BIN, lw=1.6)
    ax_ts.text(
        5.0, 0.7,
        f'scratch = {bce_scratch.item():.6f}   |   API = {bce_api.item():.6f}   |   match = {torch.allclose(bce_scratch, bce_api)}',
        ha='center', va='center', fontsize=FS_ANNOT,
    )
    fig_t.tight_layout(rect=[0, 0.02, 1, 0.94])
    print(f'Step 9: Torch BCE scratch={bce_scratch.item():.6f} API={bce_api.item():.6f}')
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
    build_step_9,
])

print("\n" + "=" * 60)
print("194 BINARY CROSS-ENTROPY")
print("=" * 60)
print("""
BCE(y, p) = -(y log p + (1-y) log(1-p)),  p = sigmoid(s) = y_hat
Hard labels y in {0,1} are special cases of continuous y in [0,1].
Used for binary classification (Bernoulli labels).
Soft alternative to 0-1 error; punishes confident wrong answers heavily.

Steps:
  1. Definition (+ sigmoid)
  2. Browsable 3D BCE(y,p) + p-slider
  3. Entropy H(p)=BCE(p,p) -- diagonal slice
  4. Coin-flip probability story
  5. Drag p for hard y=0 / y=1
  6. Logit -> sigmoid -> BCE
  7. Confidence tax
  8. Takeaway
  9. Torch -- scratch BCE matches nn.BCELoss()

See also: 0310_regression_classification.py, 190_loss_functions.py,
          0450_Loss_Categorical_Cross_Entropy.py
""")
