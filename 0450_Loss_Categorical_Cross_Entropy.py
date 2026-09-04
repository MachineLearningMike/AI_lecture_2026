"""
195_Categorical_Cross_Entropy.py

Categorical Cross-Entropy -- intuition and simulations for K-class learning.

Sketch:
  1. Definition -- softmax + CE for multi-class classification
  2. One-hot truth vs probability vector
  3. Softmax turns competing logits into a simplex
  4. Softmax temperature tau -- how sharpness changes CE (slider)
  5. Drag the true-class probability: watch -log p_true
  6. Confusion of confidence: correct argmax can still be expensive
  7. Takeaway -- CE embraces multi-class because labels are categorical
  8. Torch -- from-scratch CE vs nn.CrossEntropyLoss

Companion to 194_Binary_Cross_Entropy.py, 0310_regression_classification.py,
190_loss_functions.py, 0820_softmax.py.
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

np.random.seed(19)
RNG = np.random.default_rng(19)


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


def softmax(logits, tau=1.0):
    """Numerically stable softmax with temperature tau > 0."""
    z = np.asarray(logits, dtype=float) / float(tau)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()


C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_MULTI = '#8e44ad'


# -------------------------------------------------------
# FIGURE 1: Definition
# -------------------------------------------------------
fig0, ax0 = new_figure()
fig0.suptitle(
    'Step 1: Categorical Cross-Entropy -- definition',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax0)

round_box(ax0, 0.5, 5.25, 11.0, 2.35, C_DARK, C_DARK, lw=0)
ax0.text(
    6.0, 7.0,
    'Categorical CE is the standard loss for multi-class classification\n'
    '(choose one of K >= 3 labels).',
    ha='center', va='center', fontsize=FS_ANNOT + 2,
    fontweight='bold', color='#f9e79f',
)
ax0.text(
    6.0, 5.85,
    r'$p_k=\mathrm{softmax}(s)_k=\frac{e^{s_k}}{\sum_j e^{s_j}}$'
    '      then      '
    r'$\ell_{\mathrm{CE}}=-\sum_k y_k\log p_k=-\log p_{y_\star}$',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    fontweight='bold', color='white',
)

round_box(ax0, 0.5, 2.7, 5.3, 2.2, C_NEW, '#2874a6', lw=2.0)
ax0.text(3.15, 4.4, 'Label', ha='center',
         fontsize=FS_ANNOT + 2, fontweight='bold', color=C_DARK)
ax0.text(3.15, 3.5,
         'one-hot vector\n'
         'e.g. [0,0,0,1,0]\n'
         'categorical distribution',
         ha='center', va='center', fontsize=FS_ANNOT + 1)

round_box(ax0, 6.2, 2.7, 5.3, 2.2, '#f5eef8', C_MULTI, lw=2.0)
ax0.text(8.85, 4.4, 'Model output', ha='center',
         fontsize=FS_ANNOT + 2, fontweight='bold', color=C_MULTI)
ax0.text(8.85, 3.5,
         'logits s -> softmax -> p\n'
         'probability vector on K classes',
         ha='center', va='center', fontsize=FS_ANNOT + 1)

round_box(ax0, 0.5, 0.35, 11.0, 2.05, C_GOAL, C_OK, lw=2.0)
ax0.text(
    6.0, 1.35,
    'For a hard label, only p_true enters the loss: CE = -log p_true.\n'
    'That is Shannon cross-entropy between one-hot truth and model p.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig0)


# -------------------------------------------------------
# FIGURE 2: One-hot vs p
# -------------------------------------------------------
fig1, (ax1a, ax1b) = new_figure(1, 2)
fig1.suptitle(
    'Step 2: Truth is a spike; the model offers a soft histogram',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

K = 5
true_k = 3
y_onehot = np.zeros(K)
y_onehot[true_k] = 1.0
logits = np.array([0.2, -0.4, 0.1, 1.4, 0.3])
p = softmax(logits)
ks = np.arange(K)

ax1a.bar(ks, y_onehot, color=C_OK, edgecolor='white')
ax1a.set_xticks(ks)
ax1a.set_xticklabels([f'k={k}' for k in ks], fontsize=FS_TICK)
ax1a.set_ylim(0, 1.15)
ax1a.set_ylabel('probability', fontsize=FS_AXIS)
ax1a.set_title('True label (one-hot)', fontsize=FS_AX_TITLE - 1)
ax1a.grid(True, axis='y', alpha=0.3)
ax1a.text(
    0.98, 0.95,
    f'true class y* = {true_k}',
    transform=ax1a.transAxes, ha='right', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax1b.bar(ks, p, color=C_MULTI, edgecolor='white')
ax1b.set_xticks(ks)
ax1b.set_xticklabels([f'k={k}' for k in ks], fontsize=FS_TICK)
ax1b.set_ylim(0, 1.15)
ax1b.set_ylabel('probability', fontsize=FS_AXIS)
ax1b.set_title('Model softmax p', fontsize=FS_AX_TITLE - 1)
ax1b.grid(True, axis='y', alpha=0.3)
ce = -np.log(p[true_k])
ax1b.text(
    0.98, 0.95,
    f'p_true = {p[true_k]:.2f}\n'
    f'CE = -log p_true = {ce:.2f}',
    transform=ax1b.transAxes, ha='right', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='#f5eef8', edgecolor=C_MULTI, alpha=0.95),
)
fig1.tight_layout(rect=[0, 0.04, 1, 0.94])
fig1.text(
    0.5, 0.01,
    'CE asks how much mass landed on the spike -- not whether argmax matched.',
    ha='center', va='bottom', fontsize=FS_ANNOT + 1, fontweight='bold',
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 3: Softmax competition
# -------------------------------------------------------
fig2, (ax2a, ax2b) = new_figure(1, 2)
fig2.suptitle(
    'Step 3: Softmax is a competition -- raise one, the others shrink',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
fig2.subplots_adjust(bottom=0.22, wspace=0.30)

base = np.array([0.4, 0.1, -0.2, 0.8, 0.0], dtype=float)
s_true0 = 0.8
ax2a.set_title('Logits vs softmax (true class highlighted)', fontsize=FS_AX_TITLE - 2)
logit_bars = ax2a.bar(ks - 0.18, base, width=0.36, color='#85929e', label='logits s')
prob_bars = ax2a.bar(ks + 0.18, softmax(base), width=0.36, color=C_MULTI, label='softmax p')
ax2a.set_xticks(ks)
ax2a.set_xlabel('class k', fontsize=FS_AXIS)
ax2a.legend(fontsize=FS_LEGEND - 1, loc='upper left')
ax2a.grid(True, axis='y', alpha=0.3)

ax2b.set_xlim(0, 10)
ax2b.set_ylim(0, 10)
ax2b.axis('off')
round_box(ax2b, 0.5, 1.3, 9.0, 7.3, C_MUTE, '0.45', lw=1.8)
hud2 = ax2b.text(5.0, 5.0, '', ha='center', va='center', fontsize=FS_ANNOT + 1)

ax_sl = fig2.add_axes([0.15, 0.07, 0.45, 0.04])
slider_true = Slider(ax_sl, r'$s_{true}$', -2.0, 4.0, valinit=s_true0, valstep=0.05)


def _update_soft(_=None):
    s = base.copy()
    s[true_k] = float(slider_true.val)
    p = softmax(s)
    for i, b in enumerate(logit_bars):
        b.set_height(s[i])
    for i, b in enumerate(prob_bars):
        b.set_height(p[i])
    ax2a.set_ylim(min(-2.5, s.min() - 0.3), max(4.5, s.max() + 0.3))
    ce = -np.log(max(p[true_k], 1e-6))
    hud2.set_text(
        f'true class k = {true_k}\n'
        f's_true = {s[true_k]:.2f}\n'
        f'p_true = {p[true_k]:.2f}\n'
        f'CE = -log p_true = {ce:.2f}\n\n'
        'Raising s_true steals probability\n'
        'from the other classes.'
    )
    fig2.canvas.draw_idle()


slider_true.on_changed(_update_soft)
_update_soft()
print('Step 3: drag s_true; watch softmax and CE.')
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 4: Softmax temperature vs CE
# -------------------------------------------------------
# Fixed logits; only tau moves. p = softmax(s / tau), CE = -log p_true.
# Low tau sharpens (near-argmax); high tau flattens (near-uniform).
fig_tau, (ax_tau_p, ax_tau_ce) = new_figure(1, 2)
fig_tau.suptitle(
    r'Step 4: Softmax temperature $\tau$ -- sharpness that changes CE',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)
fig_tau.subplots_adjust(bottom=0.22, wspace=0.30)

# True class already wins under these logits (same family as Step 2)
logits_tau = np.array([0.2, -0.4, 0.1, 1.4, 0.3], dtype=float)
tau0 = 1.0
colors_tau = ['#85929e'] * K
colors_tau[true_k] = C_OK

bars_tau = ax_tau_p.bar(ks, softmax(logits_tau, tau=tau0), color=colors_tau, edgecolor='white')
ax_tau_p.set_xticks(ks)
ax_tau_p.set_xticklabels([f'k={k}' for k in ks], fontsize=FS_TICK)
ax_tau_p.set_ylim(0, 1.05)
ax_tau_p.set_ylabel(r'$p_k=\mathrm{softmax}(s/\tau)_k$', fontsize=FS_AXIS)
ax_tau_p.set_title('Fixed logits; only τ reshapes p', fontsize=FS_AX_TITLE - 2)
ax_tau_p.grid(True, axis='y', alpha=0.3)
ax_tau_p.text(
    0.02, 0.98,
    f'true class k = {true_k} (green)\n'
    r'$p=\mathrm{softmax}(s/\tau)$',
    transform=ax_tau_p.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

# Right: CE as a function of tau, with live marker
tau_grid = np.linspace(0.1, 5.0, 200)
ce_grid = np.array([
    -np.log(max(softmax(logits_tau, tau=t)[true_k], 1e-12)) for t in tau_grid
])
ax_tau_ce.plot(tau_grid, ce_grid, color=C_MULTI, lw=2.8, label=r'CE$(\tau)=-\log p_{\mathrm{true}}$')
ax_tau_ce.axhline(np.log(K), color='0.5', ls=':', lw=1.2, label=rf'uniform CE$=\log K={np.log(K):.2f}$')
mk_tau, = ax_tau_ce.plot(
    [tau0],
    [-np.log(max(softmax(logits_tau, tau=tau0)[true_k], 1e-12))],
    'o', ms=12, color=C_ACCENT, zorder=5,
)
ax_tau_ce.set_xlim(0.05, 5.1)
ax_tau_ce.set_ylim(0.0, max(ce_grid.max(), np.log(K)) + 0.35)
ax_tau_ce.set_xlabel(r'temperature $\tau$', fontsize=FS_AXIS)
ax_tau_ce.set_ylabel('Categorical CE', fontsize=FS_AXIS)
ax_tau_ce.set_title('How τ moves the loss (true class already wins)', fontsize=FS_AX_TITLE - 3)
ax_tau_ce.legend(fontsize=FS_LEGEND - 2, loc='upper right')
ax_tau_ce.grid(True, alpha=0.3)
hud_tau = ax_tau_ce.text(
    0.02, 0.98, '',
    transform=ax_tau_ce.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='#f5eef8', edgecolor=C_MULTI, alpha=0.95),
)

ax_sl_tau = fig_tau.add_axes([0.20, 0.07, 0.60, 0.04])
slider_tau = Slider(ax_sl_tau, r'$\tau$', 0.1, 5.0, valinit=tau0, valstep=0.05)


def _update_tau_ce(_=None):
    tau = float(slider_tau.val)
    p = softmax(logits_tau, tau=tau)
    for i, b in enumerate(bars_tau):
        b.set_height(p[i])
    ce = -np.log(max(p[true_k], 1e-12))
    mk_tau.set_data([tau], [ce])
    if tau < 0.4:
        vibe = 'cold / peaky (near argmax)'
    elif tau > 3.0:
        vibe = 'hot / flat (near uniform)'
    else:
        vibe = 'intermediate'
    hud_tau.set_text(
        f'τ = {tau:.2f}  ({vibe})\n'
        f'p_true = {p[true_k]:.3f}\n'
        f'CE = -log p_true = {ce:.3f}\n\n'
        'Low τ: sharpens winners ->\n'
        '  CE falls when true wins.\n'
        'High τ: flattens toward 1/K ->\n'
        '  CE climbs toward log K.'
    )
    fig_tau.canvas.draw_idle()


slider_tau.on_changed(_update_tau_ce)
_update_tau_ce()
print('Step 4: drag temperature tau; watch softmax p and CE.')
show_maximized(fig_tau)


# -------------------------------------------------------
# FIGURE 5: -log p curve
# -------------------------------------------------------
fig3, (ax3a, ax3b) = new_figure(1, 2)
fig3.suptitle(
    'Step 5: Drag p_true -- Categorical CE is just -log p_true',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
fig3.subplots_adjust(bottom=0.20, wspace=0.28)

p_line = np.linspace(0.02, 0.98, 300)
ax3a.plot(p_line, -np.log(p_line), color=C_MULTI, lw=2.8)
p0 = 0.40
mk, = ax3a.plot([p0], [-np.log(p0)], 'o', ms=12, color=C_ACCENT)
ax3a.set_xlim(0, 1)
ax3a.set_ylim(-0.1, 4.0)
ax3a.set_xlabel(r'$p_{\mathrm{true}}$', fontsize=FS_AXIS)
ax3a.set_ylabel(r'$-\log p_{\mathrm{true}}$', fontsize=FS_AXIS)
ax3a.set_title('The whole loss, once the label is hard', fontsize=FS_AX_TITLE - 2)
ax3a.grid(True, alpha=0.3)
ax3a.text(
    0.02, 0.98,
    'Other class probabilities\n'
    'matter only through softmax\n'
    'competition that shapes p_true.',
    transform=ax3a.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax3b.set_xlim(0, 10)
ax3b.set_ylim(0, 10)
ax3b.axis('off')
round_box(ax3b, 0.6, 1.5, 8.8, 7.0, '#f5eef8', C_MULTI, lw=2.0)
hud3 = ax3b.text(5.0, 5.0, '', ha='center', va='center', fontsize=FS_ANNOT + 1)

ax_sl3 = fig3.add_axes([0.58, 0.07, 0.30, 0.04])
slider_pt = Slider(ax_sl3, r'$p_{true}$', 0.02, 0.98, valinit=p0, valstep=0.01)


def _update_pt(_=None):
    pt = float(slider_pt.val)
    ce = -np.log(pt)
    mk.set_data([pt], [ce])
    hud3.set_text(
        f'p_true = {pt:.2f}\n'
        f'CE = -log(p_true) = {ce:.2f}\n\n'
        'p=0.50 -> CE=0.69\n'
        'p=0.90 -> CE=0.11\n'
        'p=0.10 -> CE=2.30'
    )
    fig3.canvas.draw_idle()


slider_pt.on_changed(_update_pt)
_update_pt()
print('Step 5: drag p_true on -log curve.')
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 6: Correct argmax, different CE
# -------------------------------------------------------
fig4, ax4 = new_figure()
fig4.suptitle(
    'Step 6: Same correct argmax -- CE still ranks confidence',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

# Three softmax distributions, all with argmax at class 2
dists = [
    ('barely wins', np.array([0.30, 0.28, 0.42])),
    ('clear winner', np.array([0.10, 0.15, 0.75])),
    ('almost sure', np.array([0.03, 0.05, 0.92])),
]
true = 2
x = np.arange(len(dists))
ces = [-np.log(d[true]) for _, d in dists]
colors = ['#d68910', C_OK, C_DARK]
ax4.bar(x, ces, color=colors, edgecolor='white', width=0.6)
ax4.set_xticks(x)
ax4.set_xticklabels([name + f'\np_true={d[true]:.2f}' for name, d in dists],
                    fontsize=FS_TICK)
ax4.set_ylabel('Categorical CE = -log p_true', fontsize=FS_AXIS)
ax4.set_title('All three have argmax = true class', fontsize=FS_AX_TITLE - 1)
ax4.grid(True, axis='y', alpha=0.3)
for i, v in enumerate(ces):
    ax4.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=FS_ANNOT, fontweight='bold')
ax4.text(
    0.98, 0.55,
    '0-1 error scores all three as 0.\n'
    'CE prefers the confident one.\n'
    'That is the training pressure\n'
    'behind calibration.',
    transform=ax4.transAxes, ha='right', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
fig4.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 7: Takeaway
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    'Step 7: Categorical Cross-Entropy in one breath',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax5)

cards = [
    (0.4, 4.3, C_NEW, '#2874a6', 'Use when', 'K-class labels\none-hot / categorical\nnot raw continuous y'),
    (4.2, 4.3, '#f5eef8', C_MULTI, 'Pipeline', 'logits s\n-> softmax p\n-> -log p_true'),
    (8.0, 4.3, C_GOAL, C_OK, 'Personality', 'Judges probability\non the true class;\nsoft vs 0-1'),
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
    'Binary CE is the K=2 special case (often with sigmoid instead of softmax).\n'
    'Categorical CE is the general embrace for multi-class labels.\n\n'
    'Rule from 035: output type -> probability model -> loss.\n'
    'See 194_Binary_Cross_Entropy.py and 190_loss_functions.py.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 8: Torch -- from scratch vs API
# -------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(195)
logits_t = torch.tensor([
    [2.0, 0.5, -1.0],
    [0.1, 1.5, 0.2],
], dtype=torch.float32)
y_idx = torch.tensor([0, 1], dtype=torch.long)

log_p = F.log_softmax(logits_t, dim=-1)
ce_scratch = -log_p[torch.arange(len(y_idx)), y_idx].mean()
ce_api = nn.CrossEntropyLoss()(logits_t, y_idx)

CODE_CE_SCRATCH = (
    'import torch\n'
    'import torch.nn.functional as F\n'
    '\n'
    'logits = torch.tensor([\n'
    '    [2.0, 0.5, -1.0],\n'
    '    [0.1, 1.5,  0.2],\n'
    '])\n'
    'y = torch.tensor([0, 1])  # class indices\n'
    '\n'
    '# from scratch: softmax then -log p_true\n'
    'log_p = F.log_softmax(logits, dim=-1)\n'
    'loss = -log_p[torch.arange(len(y)), y].mean()\n'
    'print(loss.item())'
)
CODE_CE_API = (
    'import torch\n'
    'import torch.nn as nn\n'
    '\n'
    'logits = torch.tensor([\n'
    '    [2.0, 0.5, -1.0],\n'
    '    [0.1, 1.5,  0.2],\n'
    '])\n'
    'y = torch.tensor([0, 1])  # class indices\n'
    '\n'
    '# off-the-shelf API (logits in!)\n'
    'loss = nn.CrossEntropyLoss()(logits, y)\n'
    'print(loss.item())'
)

fig_t, (ax_ts, ax_ta) = new_figure(1, 2)
fig_t.suptitle(
    'Step 8: Torch CE -- from scratch vs nn.CrossEntropyLoss',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
for ax, title, code, face, edge in [
    (ax_ts, 'From scratch', CODE_CE_SCRATCH, C_NEW, '#2874a6'),
    (ax_ta, 'Off-the-shelf API', CODE_CE_API, '#f5eef8', C_MULTI),
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
    f'scratch = {ce_scratch.item():.6f}   |   API = {ce_api.item():.6f}   |   match = {torch.allclose(ce_scratch, ce_api)}',
    ha='center', va='center', fontsize=FS_ANNOT,
)
fig_t.tight_layout(rect=[0, 0.02, 1, 0.94])
print(f'Step 8: Torch CE scratch={ce_scratch.item():.6f} API={ce_api.item():.6f}')
show_maximized(fig_t)


print("\n" + "=" * 60)
print("195 CATEGORICAL CROSS-ENTROPY")
print("=" * 60)
print("""
CE = -sum y_k log p_k = -log p_true for hard labels
p = softmax(s / tau)   (tau = 1 is the usual training case)
Used for multi-class classification (categorical / one-hot targets).

Steps:
  1. Definition (softmax + CE)
  2. One-hot truth vs soft p
  3. Softmax competition (drag s_true)
  4. Temperature tau -- how sharpness changes CE
  5. Drag p_true on -log curve
  6. Correct argmax, different CE
  7. Takeaway
  8. Torch -- scratch CE matches nn.CrossEntropyLoss()

See also: 0440_Loss_Binary_Cross_Entropy.py, 0310_regression_classification.py,
          190_loss_functions.py, 0820_softmax.py
""")
