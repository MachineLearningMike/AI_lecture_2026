"""
Softmax: differentiable, nonlinear normalization of a vector.

For logits z = (z_1, ..., z_n) and temperature tau > 0:

    softmax(z)_i  =  exp(z_i / tau) / sum_j exp(z_j / tau)

Properties worth remembering:
  1) Output is a probability vector: >= 0 and sums to 1
  2) Nonlinear: large gaps get amplified (winner-friendly)
  3) Differentiable end-to-end (unlike hard argmax)
  4) Temperature tau controls sharpness:
        tau -> 0  : approaches one-hot / deterministic policy (argmax)
        tau = 1   : standard softmax
        tau -> inf: approaches uniform (everything washed out)

"Avoid diminishing components":
  A linear normalize z / sum(z) (or ReLU-then-normalize) can crush
  mid-tier logits relative to a dominant one in awkward ways, or fail
  on negative logits. Softmax always yields a valid distribution and
  keeps a smooth floor under losers -- they shrink but stay positive,
  so gradients can still flow.
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


def softmax(z, tau=1.0, axis=-1):
    """Numerically stable softmax with temperature tau."""
    z = np.asarray(z, dtype=float) / float(tau)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Demo logits
# ---------------------------------------------------------------------------
LABELS = ['A', 'B', 'C', 'D', 'E']
Z = np.array([2.2, 1.4, 0.3, -0.5, -1.2], dtype=float)  # mixed signs
P = softmax(Z, tau=1.0)


# -------------------------------------------------------
# FIGURE 1: Definition -- logits in, probabilities out
# -------------------------------------------------------
def build_step_1():
    fig1, axes1 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig1.suptitle(
        r'Step 1: Softmax maps logits $z$ to a probability vector $p$',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    ax = axes1[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(Z)))
    ax.bar(LABELS, Z, color=colors, edgecolor='k')
    ax.axhline(0, color='0.35', lw=1)
    ax.set_ylabel(r'logit $z_i$', fontsize=FS_AXIS)
    ax.set_title('Input: unbounded scores (can be negative)', fontsize=FS_AX_TITLE - 2)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(True, axis='y', alpha=0.3)
    ax.text(
        0.02, 0.98,
        'Logits are raw preferences.\n'
        'No requirement to be positive\n'
        'or to sum to anything.',
        transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    ax = axes1[1]
    ax.bar(LABELS, P, color=colors, edgecolor='k')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r'$p_i = \mathrm{softmax}(z)_i$', fontsize=FS_AXIS)
    ax.set_title(r'Output: $p_i\geq 0$, $\sum_i p_i = 1$', fontsize=FS_AX_TITLE - 2)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(True, axis='y', alpha=0.3)
    for i, p in enumerate(P):
        ax.text(i, p + 0.02, f'{p:.2f}', ha='center', fontsize=FS_ANNOT - 1)
    ax.text(
        0.98, 0.98,
        r'$softmax(z)_i = e^{z_i} / \sum_j e^{z_j}$' + '\n\n'
        f'sum(p) = {P.sum():.6f}',
        transform=ax.transAxes, va='top', ha='right', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
    )

    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    return fig1


# -------------------------------------------------------
# FIGURE 2: Nonlinear normalization (vs linear normalize)
# -------------------------------------------------------
# Compare on positive-shifted logits so linear normalize is defined.
Z_pos = Z - Z.min() + 0.2  # all positive
p_soft = softmax(Z_pos, tau=1.0)
p_lin = Z_pos / Z_pos.sum()

def build_step_2():
    fig2, axes2 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig2.suptitle(
        'Step 2: Softmax is a NONLINEAR normalization -- winners get amplified',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    ax = axes2[0]
    x = np.arange(len(Z_pos))
    w = 0.35
    ax.bar(x - w / 2, p_lin, width=w, color='0.65', edgecolor='k', label=r'linear: $z/\|z\|_1$')
    ax.bar(x + w / 2, p_soft, width=w, color='#2471a3', edgecolor='k', label='softmax')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel('normalized weight', fontsize=FS_AXIS)
    ax.set_title('Same positive scores; two normalizers', fontsize=FS_AX_TITLE - 2)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, axis='y', alpha=0.3)
    ax.text(
        0.02, 0.98,
        'Linear share is proportional to z.\n'
        'Softmax stretches gaps via exp:\n'
        'the leader pulls farther ahead.',
        transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    ax = axes2[1]
    # Fix second logit=1, vary first
    z2 = 1.0
    z1 = np.linspace(-1, 5, 200)
    p_soft_top = np.exp(z1) / (np.exp(z1) + np.exp(z2))
    p_lin_top = np.maximum(z1, 0) / (np.maximum(z1, 0) + np.maximum(z2, 0) + 1e-12)
    ax.plot(z1, p_soft_top, color='#2471a3', lw=3.0, label='softmax p1')
    ax.plot(z1, p_lin_top, color='0.5', lw=2.5, ls='--', label='ReLU-linear p1')
    ax.axvline(z2, color='crimson', ls=':', lw=1.5, label='z2 = 1')
    ax.set_xlabel(r'$z_1$ (vary)', fontsize=FS_AXIS)
    ax.set_ylabel(r'$p_1$', fontsize=FS_AXIS)
    ax.set_title(r'Two options: vary $z_1$, fix $z_2=1$', fontsize=FS_AX_TITLE - 2)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.text(
        0.02, 0.98,
        'Softmax: smooth S-curve,\n'
        'works for negative logits.\n'
        'Linear-after-ReLU: breaks /\n'
        'flattens when z1 < 0.',
        transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
    )

    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    return fig2


# -------------------------------------------------------
# FIGURE 3: Temperature
# -------------------------------------------------------
taus = [0.2, 0.5, 1.0, 2.0, 5.0]
def build_step_3():
    fig3, axes3 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig3.suptitle(
        r'Step 3: Temperature $\tau$ -- sharpness knob on $\mathrm{softmax}(z/\tau)$',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    ax = axes3[0]
    x = np.arange(len(Z))
    width = 0.15
    for i, tau in enumerate(taus):
        p = softmax(Z, tau=tau)
        ax.bar(x + (i - 2) * width, p, width=width, edgecolor='k',
               label=rf'$\tau={tau:g}$',
               color=plt.cm.coolwarm(i / (len(taus) - 1)))
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel(r'$p_i$', fontsize=FS_AXIS)
    ax.set_ylim(0, 1.05)
    ax.set_title('Same logits; sweep temperature', fontsize=FS_AX_TITLE - 2)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND - 1, ncol=2, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes3[1]
    tau_grid = np.logspace(-1.2, 1.2, 80)  # ~0.06 to ~16
    # track mass on argmax and entropy
    imax = int(np.argmax(Z))
    p_max = []
    entropy = []
    for tau in tau_grid:
        p = softmax(Z, tau=tau)
        p_max.append(p[imax])
        entropy.append(-np.sum(p * np.log(p + 1e-12)))
    ax.semilogx(tau_grid, p_max, color='#c0392b', lw=3.0, label=r'$p_{\mathrm{argmax}}$')
    ax.set_xlabel(r'temperature $\tau$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$p$ on the best logit', fontsize=FS_AXIS)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax2 = ax.twinx()
    ax2.semilogx(tau_grid, entropy, color='#2471a3', lw=2.5, ls='--', label='entropy')
    ax2.set_ylabel('entropy (nats)', fontsize=FS_AXIS, color='#2471a3')
    ax2.tick_params(axis='y', labelsize=FS_TICK, labelcolor='#2471a3')
    ax.set_title(r'$\tau\to 0$: peaky;  $\tau\to\infty$: flat', fontsize=FS_AX_TITLE - 2)
    # combined legend
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=FS_LEGEND, loc='center right')
    ax.text(
        0.02, 0.98,
        r'Small $\tau$: almost argmax' + '\n'
        r'(deterministic policy).' + '\n'
        r'Large $\tau$: near-uniform' + '\n'
        '(exploratory / soft).',
        transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    return fig3


# -------------------------------------------------------
# FIGURE 4: Avoid diminishing / dead components; keep gradients
# -------------------------------------------------------
# Hardmax vs softmax: hardmax zeros losers completely
hard = np.zeros_like(Z)
hard[np.argmax(Z)] = 1.0
# Also a "crushing" linear scheme on large-margin logits
Z_gap = np.array([8.0, 1.0, 0.5, 0.2, 0.0])
p_gap = softmax(Z_gap, tau=1.0)
# relative floor: min(p)/max(p)
floor_soft = p_gap.min() / p_gap.max()

def build_step_4():
    fig4, axes4 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig4.suptitle(
        'Step 4: Softmax avoids hard zeros -- losers shrink but stay alive',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    ax = axes4[0]
    x = np.arange(len(Z))
    w = 0.35
    ax.bar(x - w / 2, hard, width=w, color='#c0392b', edgecolor='k', label='hard argmax')
    ax.bar(x + w / 2, P, width=w, color='#1e8449', edgecolor='k', label='softmax')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel('weight', fontsize=FS_AXIS)
    ax.set_title('Hardmax kills alternatives; softmax keeps a floor', fontsize=FS_AX_TITLE - 4)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, axis='y', alpha=0.3)
    ax.text(
        0.02, 0.98,
        'Hardmax: gradient w.r.t. loser\n'
        'logits is often blocked (0 mass).\n'
        'Softmax: every component > 0,\n'
        'so small changes still matter.',
        transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    ax = axes4[1]
    # Softmax CE Jacobian: d(-log p_t)/dz = p - one_hot(t)
    # Hard argmax has (sub)gradient 0 almost everywhere -- do not fake a CE grad.
    true = 0  # class A
    g_soft = P.copy()
    g_soft[true] -= 1.0
    g_hard = np.zeros_like(hard)
    ax.bar(x - w / 2, g_hard, width=w, color='#c0392b', edgecolor='k', label='hardmax (sub)grad ~ 0')
    ax.bar(x + w / 2, g_soft, width=w, color='#1e8449', edgecolor='k', label='softmax CE grad')
    ax.axhline(0, color='0.35', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel(r'$\partial(-\log p_A)/\partial z_i$', fontsize=FS_AXIS)
    ax.set_title('Softmax CE grads touch every logit; hardmax blocks them', fontsize=FS_AX_TITLE - 4)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.text(
        0.02, 0.02,
        'Differentiable normalization:\n'
        r'$\partial p_i/\partial z_j = p_i(1_{i=j}-p_j)$' + '\n'
        'Hard argmax is flat a.e.,\n'
        'so gradients do not flow.',
        transform=ax.transAxes, va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
    )

    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    return fig4


# -------------------------------------------------------
# FIGURE 5: Simulate a deterministic policy (low temperature)
# -------------------------------------------------------
def build_step_5():
    fig5, axes5 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig5.suptitle(
        r'Step 5: Low temperature $\approx$ deterministic policy (soft argmax)',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    # Policy over actions from logits
    actions = ['left', 'forward', 'right', 'wait']
    logits_pi = np.array([0.5, 2.8, 1.1, -0.4], dtype=float)
    tau_soft, tau_det = 1.5, 0.15
    pi_soft = softmax(logits_pi, tau=tau_soft)
    pi_det = softmax(logits_pi, tau=tau_det)
    pi_hard = np.zeros_like(logits_pi)
    pi_hard[np.argmax(logits_pi)] = 1.0

    ax = axes5[0]
    x = np.arange(len(actions))
    w = 0.25
    ax.bar(x - w, pi_soft, width=w, color='#2471a3', edgecolor='k',
           label=rf'softmax $\tau={tau_soft:g}$')
    ax.bar(x, pi_det, width=w, color='#1e8449', edgecolor='k',
           label=rf'softmax $\tau={tau_det:g}$')
    ax.bar(x + w, pi_hard, width=w, color='#c0392b', edgecolor='k',
           label='true argmax')
    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    ax.set_ylabel('policy mass', fontsize=FS_AXIS)
    ax.set_ylim(0, 1.1)
    ax.set_title('Stochastic vs near-deterministic vs hard', fontsize=FS_AX_TITLE - 2)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND - 1)
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes5[1]
    ax.axis('off')
    ax.set_title('Why this matters', fontsize=FS_AX_TITLE)
    ax.text(
        0.5, 0.55,
        'A deterministic policy picks ONE action.\n'
        'Hard argmax does that -- but is not (usefully)\n'
        'differentiable through the choice.\n\n'
        'Softmax with small temperature:\n'
        '  • almost one-hot (simulates deterministic behavior)\n'
        '  • still smooth: gradients flow to all logits\n'
        '  • used in attention, classifiers, RL (soft policies)\n\n'
        'Raise temperature to explore;\n'
        'lower it to exploit / commit.\n\n'
        rf'Demo: argmax action = "{actions[int(np.argmax(logits_pi))]}"' + '\n'
        rf'  soft  p = {np.round(pi_soft, 3).tolist()}' + '\n'
        rf'  sharp p = {np.round(pi_det, 3).tolist()}',
        transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
        bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
    )

    fig5.tight_layout(rect=[0, 0, 1, 0.94])
    return fig5


# -------------------------------------------------------
# FIGURE 6: 2D simplex path as temperature changes (3-class)
# -------------------------------------------------------
def build_step_6():
    fig6, ax6 = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig6.suptitle(
        r'Step 6: On the probability simplex -- cooling drives $p$ to a vertex',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    # 3-class logits; barycentric plot in equilateral triangle
    z3 = np.array([1.8, 1.0, 0.2], dtype=float)
    labs3 = ['c0', 'c1', 'c2']
    # triangle vertices
    V0 = np.array([0.0, 0.0])
    V1 = np.array([1.0, 0.0])
    V2 = np.array([0.5, np.sqrt(3) / 2])
    verts = np.stack([V0, V1, V2], axis=0)

    ax6.plot([V0[0], V1[0], V2[0], V0[0]],
             [V0[1], V1[1], V2[1], V0[1]], color='k', lw=2)
    for v, lab in zip(verts, labs3):
        ax6.scatter([v[0]], [v[1]], s=80, c='k', zorder=5)
        ax6.text(v[0], v[1] - 0.06, lab, ha='center', fontsize=FS_ANNOT, fontweight='bold')

    taus_path = np.logspace(1.0, -1.5, 40)  # hot -> cold
    pts = []
    for tau in taus_path:
        p = softmax(z3, tau=tau)
        pts.append(p[0] * V0 + p[1] * V1 + p[2] * V2)
    pts = np.stack(pts, axis=0)
    ax6.plot(pts[:, 0], pts[:, 1], color='#2471a3', lw=2.5, label='path as τ decreases')
    ax6.scatter(pts[0, 0], pts[0, 1], c='#2471a3', s=70, zorder=6, label='high τ (soft)')
    ax6.scatter(pts[-1, 0], pts[-1, 1], c='#c0392b', s=90, zorder=6, label='low τ (~vertex)')
    # mark true argmax vertex
    ax6.scatter([V0[0]], [V0[1]], c='#c0392b', s=160, marker='*', zorder=7,
                label='argmax vertex')

    ax6.set_aspect('equal')
    ax6.axis('off')
    ax6.legend(fontsize=FS_LEGEND, loc='upper right')
    ax6.text(
        0.02, 0.50,
        'Interior point = stochastic policy.\n'
        'Cooling (↓τ) slides p toward a corner:\n'
        'deterministic choice, still via softmax.',
        transform=ax6.transAxes, va='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    return fig6


# -------------------------------------------------------
# FIGURE 7: Interactive -- fixed logits, live temperature slider
# -------------------------------------------------------
# Specific logits held fixed; only tau moves. Watch the distribution reshape.
Z_LIVE = np.array([2.5, 1.2, 0.4, -0.3, -1.5], dtype=float)
LABELS_LIVE = ['A', 'B', 'C', 'D', 'E']
COLORS_LIVE = plt.cm.viridis(np.linspace(0.15, 0.85, len(Z_LIVE)))
TAU0 = 1.0

def build_step_7():
    fig7 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig7.suptitle(
        r'Step 7: Interactive -- fixed logits $z$, drag temperature $\tau$',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    fig7.subplots_adjust(left=0.07, right=0.98, bottom=0.22, top=0.88, wspace=0.28)

    ax7a = fig7.add_subplot(1, 2, 1)
    ax7b = fig7.add_subplot(1, 2, 2)

    # Left: fixed logits (never change)
    bars_z = ax7a.bar(LABELS_LIVE, Z_LIVE, color=COLORS_LIVE, edgecolor='k')
    ax7a.axhline(0, color='0.35', lw=1)
    ax7a.set_ylabel(r'logit $z_i$ (fixed)', fontsize=FS_AXIS)
    ax7a.set_title('Specific logits (held fixed)', fontsize=FS_AX_TITLE)
    ax7a.tick_params(axis='both', labelsize=FS_TICK)
    ax7a.grid(True, axis='y', alpha=0.3)
    for i, zi in enumerate(Z_LIVE):
        ax7a.text(i, zi + (0.12 if zi >= 0 else -0.25), f'{zi:.1f}',
                  ha='center', va='bottom' if zi >= 0 else 'top', fontsize=FS_ANNOT - 1)
    ax7a.text(
        0.02, 0.98,
        r'$z = [' + ', '.join(f'{zi:g}' for zi in Z_LIVE) + r']$' + '\n'
        'Only $\\tau$ changes on the right.',
        transform=ax7a.transAxes, va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    # Right: live softmax bars
    p0 = softmax(Z_LIVE, tau=TAU0)
    bars_p = ax7b.bar(LABELS_LIVE, p0, color=COLORS_LIVE, edgecolor='k')
    ax7b.set_ylim(0, 1.05)
    ax7b.set_ylabel(r'$p_i=\mathrm{softmax}(z/\tau)_i$', fontsize=FS_AXIS)
    ax7b.set_title('Distribution under current $\\tau$', fontsize=FS_AX_TITLE)
    ax7b.tick_params(axis='both', labelsize=FS_TICK)
    ax7b.grid(True, axis='y', alpha=0.3)
    value_texts = [
        ax7b.text(i, p0[i] + 0.02, f'{p0[i]:.2f}', ha='center', va='bottom',
                  fontsize=FS_ANNOT - 1)
        for i in range(len(Z_LIVE))
    ]
    hud7 = ax7b.text(
        0.98, 0.98, '',
        transform=ax7b.transAxes, va='top', ha='right', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
    )

    ax_tau = fig7.add_axes([0.20, 0.08, 0.60, 0.05])
    # log-spaced feel via slider on log10(tau)
    slider_tau = Slider(
        ax_tau, r'$\tau$', 0.05, 8.0,
        valinit=TAU0, valstep=0.05,
    )


    def _update_tau(_=None):
        tau = float(slider_tau.val)
        p = softmax(Z_LIVE, tau=tau)
        for bar, pi, txt in zip(bars_p, p, value_texts):
            bar.set_height(pi)
            txt.set_y(pi + 0.02)
            txt.set_text(f'{pi:.2f}')
        ent = float(-np.sum(p * np.log(p + 1e-12)))
        hud7.set_text(
            rf'$\tau = {tau:.2f}$' + '\n'
            f'argmax mass = {p.max():.3f}\n'
            f'entropy = {ent:.3f} nats\n'
            + ('peaky / near-argmax' if tau < 0.4 else
               'soft / near-uniform' if tau > 3.0 else
               'intermediate')
        )
        fig7.canvas.draw_idle()


    slider_tau.on_changed(_update_tau)
    _update_tau()

    print("Step 7: drag the temperature slider; logits stay fixed.")
    return fig7


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
# Temperature extremes for the demo vector
p_cold = softmax(Z, tau=0.15)
p_hot = softmax(Z, tau=8.0)
print("\n" + "=" * 60)
print("SOFTMAX")
print("=" * 60)
print(f"""
Definition (temperature tau > 0):
  softmax(z)_i = exp(z_i / tau) / sum_j exp(z_j / tau)

Demo logits z = {Z.tolist()}
  tau=1:    p = {np.round(P, 4).tolist()}   sum={P.sum():.6f}
  tau=0.15: p = {np.round(p_cold, 4).tolist()}   (near one-hot / deterministic)
  tau=8:    p = {np.round(p_hot, 4).tolist()}   (near uniform)

Interactive Step 7 logits:
  z = {Z_LIVE.tolist()}
  Drag tau to see the same z go from peaky to nearly uniform.

What it is:
  Differentiable NONLINEAR normalization of a vector.
  Always returns a probability distribution (even if logits are negative).

Compared to linear normalize / hard argmax:
  * Amplifies gaps (exp) rather than sharing mass proportionally
  * Never forces exact zeros -- avoids fully diminished / dead components
  * Gradients of cross-entropy touch every logit:
        dp_i/dz_j = p_i (1{{i=j}} - p_j)

Temperature:
  small tau -> sharp / exploitative / almost argmax (deterministic policy)
  large tau -> flat / exploratory
  Attention and classifiers usually use tau=1 (or absorb scale in 1/sqrt(d)).

Simulate deterministic policy:
  Use softmax with small tau (or large logit margins) to approximate argmax
  while keeping a smooth computational graph for training.

  -- attention weights are a softmax over key scores
  -- softmax(QK^T/sqrt(d))
""")
