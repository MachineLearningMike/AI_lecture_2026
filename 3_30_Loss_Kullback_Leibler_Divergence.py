"""
Kullback-Leibler divergence -- the gap between cross-entropy and entropy.

Sketch:
  1. Definition -- KL(p || q) as extra coding cost
  2. Shannon identity -- CE(p, q) = H(p) + KL(p || q)
  3. Hard labels -- H(y)=0, so CE = KL (why training CE is KL to one-hot)
  4. Asymmetry -- KL(p || q) is not KL(q || p); slider on q
  5. Drag the model toward a fixed target: watch KL fall to zero
  6. Soft target -- H(p) is constant; CE and KL move together
  7. Takeaway -- KL is the geometric gap; CE is what we usually optimize
  8. Torch -- from-scratch KL vs nn.KLDivLoss
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

np.random.seed(23)
RNG = np.random.default_rng(23)


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


def entropy(p, eps=1e-12):
    """Shannon entropy H(p) = -sum p log p (nats). 0 log 0 := 0."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    out = 0.0
    for pk in p:
        if pk > eps:
            out -= float(pk) * np.log(float(pk))
    return out


def cross_entropy(p, q, eps=1e-12):
    """Cross-entropy H(p, q) = -sum p log q."""
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    return float(-np.sum(p * np.log(q)))


def kl(p, q, eps=1e-12):
    """KL(p || q) = sum p log(p/q).  Terms with p_k=0 contribute 0."""
    p = np.asarray(p, dtype=float)
    q = np.clip(np.asarray(q, dtype=float), eps, 1.0)
    out = 0.0
    for pk, qk in zip(p, q):
        if pk > eps:
            out += float(pk) * np.log(float(pk) / float(qk))
    return out


def bernoulli(theta):
    """Two-class probability vector [1-theta, theta] for class-1 mass theta."""
    t = float(np.clip(theta, 0.0, 1.0))
    return np.array([1.0 - t, t], dtype=float)


C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_KL = '#16a085'
C_BIN = '#d68910'


# -------------------------------------------------------
# FIGURE 1: Definition
# -------------------------------------------------------
def build_step_1():
    fig0, ax0 = new_figure()
    fig0.suptitle(
        'Step 1: Kullback-Leibler divergence -- definition',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax0)

    round_box(ax0, 0.5, 5.25, 11.0, 2.35, C_DARK, C_DARK, lw=0)
    ax0.text(
        6.0, 7.0,
        'KL asks: how different is q from p, when p is the truth?',
        ha='center', va='center', fontsize=FS_ANNOT + 2,
        fontweight='bold', color='#f9e79f',
    )
    ax0.text(
        6.0, 5.85,
        r'$\mathrm{KL}(p\|q)=\sum_k p_k\log\frac{p_k}{q_k}'
        r'=-\,H(p)+H(p,q)$',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
        fontweight='bold', color='white',
    )

    round_box(ax0, 0.5, 2.7, 5.3, 2.2, C_NEW, '#2874a6', lw=2.0)
    ax0.text(3.15, 4.4, 'Truth p', ha='center',
             fontsize=FS_ANNOT + 2, fontweight='bold', color=C_DARK)
    ax0.text(3.15, 3.5,
             'the distribution you\n'
             'actually sample from\n'
             '(label / data belief)',
             ha='center', va='center', fontsize=FS_ANNOT + 1)

    round_box(ax0, 6.2, 2.7, 5.3, 2.2, '#e8f8f5', C_KL, lw=2.0)
    ax0.text(8.85, 4.4, 'Model / code q', ha='center',
             fontsize=FS_ANNOT + 2, fontweight='bold', color=C_KL)
    ax0.text(8.85, 3.5,
             'the distribution you\n'
             'use to predict / encode\n'
             '(softmax probs, etc.)',
             ha='center', va='center', fontsize=FS_ANNOT + 1)

    round_box(ax0, 0.5, 0.35, 11.0, 2.05, '#fdebd0', C_BIN, lw=2.0)
    ax0.text(
        6.0, 1.35,
        'Coding story: H(p) is the ideal average message length for p.\n'
        'If you encode with the wrong code q, you pay H(p,q) instead.\n'
        'The waste is KL(p || q) -- always >= 0, and 0 iff p = q.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig0


# -------------------------------------------------------
# FIGURE 2: CE = H + KL
# -------------------------------------------------------
def build_step_2():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        r'Step 2: The Shannon identity -- $\mathrm{CE}=H+\mathrm{KL}$',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax1)

    # Three stacked meaning cards
    cards2 = [
        (0.5, 5.4, C_NEW, '#2874a6',
         r'$H(p)=-\sum p_k\log p_k$',
         'Entropy of the truth.\nDepends only on p.\nNot a training knob once p is fixed.'),
        (0.5, 2.95, '#e8f8f5', C_KL,
         r'$\mathrm{KL}(p\|q)=\sum p_k\log(p_k/q_k)$',
         'Extra cost of using q instead of p.\nThis is the gap we care about\nwhen we say "match the label".'),
        (0.5, 0.5, C_GOAL, C_OK,
         r'$H(p,q)=-\sum p_k\log q_k$',
         'Cross-entropy = entropy + KL.\nSame q-dependence as KL when p is fixed.\nThat is why CE training minimizes KL.'),
    ]
    for x0, y0, face, edge, title, body in cards2:
        round_box(ax1, x0, y0, 7.2, 2.2, face, edge, lw=2.0)
        ax1.text(x0 + 3.6, y0 + 1.65, title, ha='center',
                 fontsize=FS_ANNOT + 1, fontweight='bold', color=edge)
        ax1.text(x0 + 3.6, y0 + 0.75, body, ha='center', va='center',
                 fontsize=FS_ANNOT)

    round_box(ax1, 8.0, 0.5, 3.5, 7.1, C_MUTE, '0.45', lw=1.8)
    ax1.text(
        9.75, 4.2,
        'Identity\n\n'
        r'$\mathrm{CE}(p,q)$' '\n'
        r'$= H(p)$' '\n'
        r'$+ \mathrm{KL}(p\|q)$'
        '\n\n'
        'Fix p, vary q:\n'
        'argmin CE = argmin KL.\n'
        'H(p) is just a constant.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig1


# -------------------------------------------------------
# FIGURE 3: Hard labels => CE = KL
# -------------------------------------------------------
def build_step_3():
    fig2, (ax2a, ax2b) = new_figure(1, 2)
    fig2.suptitle(
        'Step 3: Hard one-hot labels -- H(y)=0, so CE = KL',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    K3 = 5
    true_k = 2
    y_hard = np.zeros(K3)
    y_hard[true_k] = 1.0
    # A soft model that puts most mass on the true class
    q_demo = np.array([0.08, 0.10, 0.62, 0.12, 0.08])
    ks3 = np.arange(K3)

    ax2a.bar(ks3 - 0.18, y_hard, width=0.36, color=C_OK, edgecolor='white', label='one-hot y')
    ax2a.bar(ks3 + 0.18, q_demo, width=0.36, color=C_KL, edgecolor='white', label='model q')
    ax2a.set_xticks(ks3)
    ax2a.set_xticklabels([f'k={k}' for k in ks3], fontsize=FS_TICK)
    ax2a.set_ylim(0, 1.15)
    ax2a.set_ylabel('probability', fontsize=FS_AXIS)
    ax2a.set_title('Hard label vs model', fontsize=FS_AX_TITLE - 1)
    ax2a.legend(fontsize=FS_LEGEND - 1, loc='upper right')
    ax2a.grid(True, axis='y', alpha=0.3)

    h_y = entropy(y_hard)
    ce_yq = cross_entropy(y_hard, q_demo)
    kl_yq = kl(y_hard, q_demo)
    ax2b.set_xlim(0, 10)
    ax2b.set_ylim(0, 10)
    ax2b.axis('off')
    round_box(ax2b, 0.5, 1.0, 9.0, 8.0, C_MUTE, '0.45', lw=1.8)
    ax2b.text(
        5.0, 5.2,
        f'H(y)  = {h_y:.3f}   (spike has no uncertainty)\n'
        f'CE(y, q) = -log q_true = {ce_yq:.3f}\n'
        f'KL(y || q) = {kl_yq:.3f}\n\n'
        'With a hard label, CE and KL are identical.\n'
        'Training with categorical CE is exactly\n'
        'minimizing KL(y || q) to the one-hot spike.\n\n'
        'Soft labels (knowledge distillation, mixup):\n'
        'H(y) > 0, so CE = H(y) + KL -- same argmin in q.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    return fig2


# -------------------------------------------------------
# FIGURE 4: Asymmetry -- KL(p||q) vs KL(q||p)
# -------------------------------------------------------
def build_step_4():
    fig3, (ax3a, ax3b) = new_figure(1, 2)
    fig3.suptitle(
        r'Step 4: KL is a directed gap -- $\mathrm{KL}(p\|q)\neq\mathrm{KL}(q\|p)$',
        fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
    )
    fig3.subplots_adjust(bottom=0.22, wspace=0.28)

    # Fixed Bernoulli truth p; slider moves model mass q1
    p_fixed = 0.20
    p_vec = bernoulli(p_fixed)
    q_grid = np.linspace(0.02, 0.98, 300)
    kl_pq = np.array([kl(p_vec, bernoulli(q)) for q in q_grid])
    kl_qp = np.array([kl(bernoulli(q), p_vec) for q in q_grid])

    ax3a.plot(q_grid, kl_pq, color=C_KL, lw=2.8, label=r'$\mathrm{KL}(p\|q)$')
    ax3a.plot(q_grid, kl_qp, color=C_ACCENT, lw=2.8, ls='--',
              label=r'$\mathrm{KL}(q\|p)$')
    q0 = 0.70
    mk_pq, = ax3a.plot([q0], [kl(p_vec, bernoulli(q0))], 'o', ms=11, color=C_KL)
    mk_qp, = ax3a.plot([q0], [kl(bernoulli(q0), p_vec)], 'o', ms=11, color=C_ACCENT)
    ax3a.axvline(p_fixed, color='0.5', ls=':', lw=1.2)
    ax3a.set_xlim(0, 1)
    ax3a.set_ylim(-0.05, 3.2)
    ax3a.text(p_fixed + 0.02, 2.8,
              rf'$p={p_fixed:.2f}$', fontsize=FS_ANNOT, color='0.35')
    ax3a.set_xlabel(r'model mass $q$ on class 1', fontsize=FS_AXIS)
    ax3a.set_ylabel('KL (nats)', fontsize=FS_AXIS)
    ax3a.set_title(rf'Bernoulli truth $p={p_fixed:.2f}$; move $q$', fontsize=FS_AX_TITLE - 2)
    ax3a.legend(fontsize=FS_LEGEND - 1, loc='upper center')
    ax3a.grid(True, alpha=0.3)

    ax3b.set_xlim(0, 10)
    ax3b.set_ylim(0, 10)
    ax3b.axis('off')
    round_box(ax3b, 0.5, 1.2, 9.0, 7.5, '#e8f8f5', C_KL, lw=2.0)
    hud_asym = ax3b.text(5.0, 5.2, '', ha='center', va='center', fontsize=FS_ANNOT + 1)

    ax_sl_q = fig3.add_axes([0.15, 0.07, 0.40, 0.04])
    slider_q = Slider(ax_sl_q, r'$q$', 0.02, 0.98, valinit=q0, valstep=0.01)


    def _update_asym(_=None):
        q = float(slider_q.val)
        v_pq = kl(p_vec, bernoulli(q))
        v_qp = kl(bernoulli(q), p_vec)
        mk_pq.set_data([q], [v_pq])
        mk_qp.set_data([q], [v_qp])
        hud_asym.set_text(
            f'Fixed truth p = {p_fixed:.2f}\n'
            f'Model q = {q:.2f}\n\n'
            f'KL(p || q) = {v_pq:.3f}\n'
            f'KL(q || p) = {v_qp:.3f}\n\n'
            'Same two numbers, two directions.\n'
            'Training usually uses KL(label || model),\n'
            'not the reverse.'
        )
        fig3.canvas.draw_idle()


    slider_q.on_changed(_update_asym)
    _update_asym()
    print('Step 4: drag q; watch KL(p||q) and KL(q||p) disagree.')
    return fig3


# -------------------------------------------------------
# FIGURE 5: Drag model toward fixed categorical target
# -------------------------------------------------------
def build_step_5():
    fig4, (ax4a, ax4b) = new_figure(1, 2)
    fig4.suptitle(
        'Step 5: Drag the model toward a fixed target -- KL falls to 0',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    fig4.subplots_adjust(bottom=0.22, wspace=0.28)

    K5 = 4
    ks5 = np.arange(K5)
    p_tgt = np.array([0.10, 0.55, 0.25, 0.10])  # fixed soft target
    # Model starts from a temperature-controlled softmax of fixed logits
    logits5 = np.array([0.0, 0.8, 0.2, -0.4], dtype=float)


    def softmax(z, tau=1.0):
        z = np.asarray(z, dtype=float) / float(tau)
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()


    tau0 = 1.5
    q0_cat = softmax(logits5, tau=tau0)
    bars_p = ax4a.bar(ks5 - 0.18, p_tgt, width=0.36, color=C_OK, edgecolor='white', label='target p')
    bars_q = ax4a.bar(ks5 + 0.18, q0_cat, width=0.36, color=C_KL, edgecolor='white', label='model q')
    ax4a.set_xticks(ks5)
    ax4a.set_xticklabels([f'k={k}' for k in ks5], fontsize=FS_TICK)
    ax4a.set_ylim(0, 0.85)
    ax4a.set_ylabel('probability', fontsize=FS_AXIS)
    ax4a.set_title(r'$q=\mathrm{softmax}(s/\tau)$ -- cool toward a peak', fontsize=FS_AX_TITLE - 3)
    ax4a.legend(fontsize=FS_LEGEND - 1, loc='upper right')
    ax4a.grid(True, axis='y', alpha=0.3)

    # Right: KL vs tau curve + live marker
    tau_grid = np.linspace(0.2, 4.0, 160)
    kl_vs_tau = np.array([kl(p_tgt, softmax(logits5, tau=t)) for t in tau_grid])
    ax4b.plot(tau_grid, kl_vs_tau, color=C_KL, lw=2.8)
    mk_kl_tau, = ax4b.plot(
        [tau0], [kl(p_tgt, q0_cat)], 'o', ms=12, color=C_ACCENT, zorder=5,
    )
    ax4b.set_xlim(0.15, 4.1)
    ax4b.set_ylim(0.0, max(kl_vs_tau.max() * 1.15, 0.4))
    ax4b.set_xlabel(r'temperature $\tau$', fontsize=FS_AXIS)
    ax4b.set_ylabel(r'$\mathrm{KL}(p\|q)$', fontsize=FS_AXIS)
    ax4b.set_title('Lower τ sharpens q (not always closer to p)', fontsize=FS_AX_TITLE - 3)
    ax4b.grid(True, alpha=0.3)
    hud5 = ax4b.text(
        0.98, 0.98, '',
        transform=ax4b.transAxes, ha='right', va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#e8f8f5', edgecolor=C_KL, alpha=0.95),
    )

    ax_sl_tau = fig4.add_axes([0.20, 0.07, 0.60, 0.04])
    slider_tau = Slider(ax_sl_tau, r'$\tau$', 0.2, 4.0, valinit=tau0, valstep=0.05)


    def _update_match(_=None):
        tau = float(slider_tau.val)
        q = softmax(logits5, tau=tau)
        for i, b in enumerate(bars_q):
            b.set_height(q[i])
        v = kl(p_tgt, q)
        mk_kl_tau.set_data([tau], [v])
        hud5.set_text(
            f'τ = {tau:.2f}\n'
            f'KL(p || q) = {v:.3f}\n'
            f'CE(p, q) = {cross_entropy(p_tgt, q):.3f}\n'
            f'H(p) = {entropy(p_tgt):.3f}\n\n'
            'CE - KL = H(p) (constant).'
        )
        fig4.canvas.draw_idle()


    slider_tau.on_changed(_update_match)
    _update_match()
    print('Step 5: drag tau; watch model q and KL(p||q).')
    return fig4


# -------------------------------------------------------
# FIGURE 6: Soft target -- CE and KL share the same slope in q
# -------------------------------------------------------
def build_step_6():
    fig5, (ax5a, ax5b) = new_figure(1, 2)
    fig5.suptitle(
        'Step 6: Soft target -- H(p) is constant; CE and KL move together',
        fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
    )
    fig5.subplots_adjust(bottom=0.22, wspace=0.28)

    # Binary soft label p; slider on model q
    p_soft = 0.65
    p_soft_vec = bernoulli(p_soft)
    q_line = np.linspace(0.02, 0.98, 300)
    ce_line = np.array([cross_entropy(p_soft_vec, bernoulli(q)) for q in q_line])
    kl_line = np.array([kl(p_soft_vec, bernoulli(q)) for q in q_line])
    h_const = entropy(p_soft_vec)

    ax5a.plot(q_line, ce_line, color='#2874a6', lw=2.8, label=r'$\mathrm{CE}(p,q)$')
    ax5a.plot(q_line, kl_line, color=C_KL, lw=2.8, label=r'$\mathrm{KL}(p\|q)$')
    ax5a.axhline(h_const, color=C_BIN, ls='--', lw=1.6, label=rf'$H(p)={h_const:.2f}$')
    q_soft0 = 0.30
    mk_ce, = ax5a.plot([q_soft0], [cross_entropy(p_soft_vec, bernoulli(q_soft0))],
                       'o', ms=11, color='#2874a6')
    mk_kl, = ax5a.plot([q_soft0], [kl(p_soft_vec, bernoulli(q_soft0))],
                       'o', ms=11, color=C_KL)
    ax5a.axvline(p_soft, color='0.5', ls=':', lw=1.1)
    ax5a.set_xlim(0, 1)
    ax5a.set_ylim(-0.05, max(ce_line.max(), kl_line.max()) + 0.35)
    ax5a.set_xlabel(r'model $q$ (class-1 mass)', fontsize=FS_AXIS)
    ax5a.set_ylabel('nats', fontsize=FS_AXIS)
    ax5a.set_title(rf'Soft truth $p={p_soft:.2f}$; move model $q$', fontsize=FS_AX_TITLE - 2)
    ax5a.legend(fontsize=FS_LEGEND - 1, loc='upper right')
    ax5a.grid(True, alpha=0.3)

    ax5b.set_xlim(0, 10)
    ax5b.set_ylim(0, 10)
    ax5b.axis('off')
    round_box(ax5b, 0.5, 1.2, 9.0, 7.5, C_MUTE, '0.45', lw=1.8)
    hud6 = ax5b.text(5.0, 5.2, '', ha='center', va='center', fontsize=FS_ANNOT + 1)

    ax_sl_soft = fig5.add_axes([0.15, 0.07, 0.40, 0.04])
    slider_soft = Slider(ax_sl_soft, r'$q$', 0.02, 0.98, valinit=q_soft0, valstep=0.01)


    def _update_soft(_=None):
        q = float(slider_soft.val)
        ce_v = cross_entropy(p_soft_vec, bernoulli(q))
        kl_v = kl(p_soft_vec, bernoulli(q))
        mk_ce.set_data([q], [ce_v])
        mk_kl.set_data([q], [kl_v])
        hud6.set_text(
            f'p = {p_soft:.2f} (fixed soft label)\n'
            f'q = {q:.2f}\n\n'
            f'H(p)      = {h_const:.3f}\n'
            f'KL(p || q) = {kl_v:.3f}\n'
            f'CE(p, q)   = {ce_v:.3f}\n'
            f'CE - KL    = {ce_v - kl_v:.3f}\n\n'
            'The vertical gap is exactly H(p).\n'
            'Minima sit at the same q (= p).'
        )
        fig5.canvas.draw_idle()


    slider_soft.on_changed(_update_soft)
    _update_soft()
    print('Step 6: drag q; CE and KL stay H(p) apart.')
    return fig5


# -------------------------------------------------------
# FIGURE 7: Takeaway
# -------------------------------------------------------
def build_step_7():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        'Step 7: Kullback-Leibler divergence in one breath',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax6)

    cards7 = [
        (0.4, 4.3, C_NEW, '#2874a6', 'Measures', 'Extra nats to code\ntruth p using q\n>= 0, =0 iff p=q'),
        (4.2, 4.3, '#e8f8f5', C_KL, 'Identity', 'CE(p,q)\n= H(p)+KL(p||q)\nsame argmin in q'),
        (8.0, 4.3, C_GOAL, C_OK, 'In training', 'Hard label: CE=KL\nSoft label: CE=H+KL\nDirected, not a metric'),
    ]
    for x0, y0, face, edge, title, body in cards7:
        round_box(ax6, x0, y0, 3.5, 2.9, face, edge, lw=2.2)
        ax6.text(x0 + 1.75, y0 + 2.25, title, ha='center',
                 fontsize=FS_AX_TITLE - 2, fontweight='bold', color=edge)
        ax6.text(x0 + 1.75, y0 + 1.15, body, ha='center', va='center',
                 fontsize=FS_ANNOT + 1)

    round_box(ax6, 0.4, 0.45, 11.2, 3.4, C_MUTE, '0.45', lw=1.8)
    ax6.text(
        6.0, 2.15,
        'KL is the information-geometry reason CE works.\n'
        'Binary / categorical CE are the practical losses; KL is the gap they minimize.\n\n'
        'BCE / categorical CE / Shannon entropy are the everyday cousins.\n'
        'Rule: output type -> probability model -> loss.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
    )
    return fig6


# -------------------------------------------------------
# FIGURE 8: Torch -- from scratch vs API
# -------------------------------------------------------
import torch
import torch.nn as nn

torch.manual_seed(196)
# Batch of categorical distributions (rows sum to 1)
p_t = torch.tensor([
    [0.10, 0.55, 0.25, 0.10],
    [0.40, 0.20, 0.20, 0.20],
], dtype=torch.float32)
q_t = torch.tensor([
    [0.20, 0.40, 0.30, 0.10],
    [0.25, 0.25, 0.25, 0.25],
], dtype=torch.float32)

# KL(p || q) = sum p * (log p - log q), averaged over batch
kl_scratch = (p_t * (p_t.clamp_min(1e-12).log() - q_t.clamp_min(1e-12).log())).sum(dim=-1).mean()
# F.kl_div / nn.KLDivLoss expect log(q) first, target p second
kl_api = nn.KLDivLoss(reduction='batchmean')(q_t.clamp_min(1e-12).log(), p_t)

CODE_KL_SCRATCH = (
    'import torch\n'
    '\n'
    'p = torch.tensor([\n'
    '    [0.10, 0.55, 0.25, 0.10],\n'
    '    [0.40, 0.20, 0.20, 0.20],\n'
    '])\n'
    'q = torch.tensor([\n'
    '    [0.20, 0.40, 0.30, 0.10],\n'
    '    [0.25, 0.25, 0.25, 0.25],\n'
    '])\n'
    '\n'
    '# from scratch: KL(p || q)\n'
    'loss = (p * (p.clamp_min(1e-12).log()\n'
    '            - q.clamp_min(1e-12).log())\n'
    '        ).sum(dim=-1).mean()\n'
    'print(loss.item())'
)
CODE_KL_API = (
    'import torch\n'
    'import torch.nn as nn\n'
    '\n'
    'p = torch.tensor([...])  # target\n'
    'q = torch.tensor([...])  # model probs\n'
    '\n'
    '# off-the-shelf API\n'
    '# WARNING: first arg is log(q), second is p\n'
    'loss = nn.KLDivLoss(reduction="batchmean")(\n'
    '    q.clamp_min(1e-12).log(), p\n'
    ')\n'
    'print(loss.item())'
)

def build_step_8():
    fig_t, (ax_ts, ax_ta) = new_figure(1, 2)
    fig_t.suptitle(
        r'Step 8: Torch KL -- from scratch vs nn.KLDivLoss',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    for ax, title, code, face, edge in [
        (ax_ts, 'From scratch', CODE_KL_SCRATCH, C_NEW, '#2874a6'),
        (ax_ta, 'Off-the-shelf API', CODE_KL_API, '#e8f8f5', C_KL),
    ]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title(title, fontsize=FS_AX_TITLE - 1, color=edge)
        round_box(ax, 0.35, 1.3, 9.3, 8.2, face, edge, lw=2.0)
        ax.text(0.7, 5.3, code, ha='left', va='center',
                fontsize=FS_ANNOT - 1, family='monospace', linespacing=1.3)
    round_box(ax_ts, 0.35, 0.25, 9.3, 0.9, '#fdebd0', C_BIN, lw=1.6)
    ax_ts.text(
        5.0, 0.7,
        f'scratch = {kl_scratch.item():.6f}   |   API = {kl_api.item():.6f}   |   match = {torch.allclose(kl_scratch, kl_api)}',
        ha='center', va='center', fontsize=FS_ANNOT,
    )
    fig_t.tight_layout(rect=[0, 0.02, 1, 0.94])
    print(f'Step 8: Torch KL scratch={kl_scratch.item():.6f} API={kl_api.item():.6f}')
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
print("196 KULLBACK-LEIBLER DIVERGENCE")
print("=" * 60)
print("""
KL(p || q) = sum_k p_k log(p_k / q_k) = CE(p, q) - H(p)
Always >= 0; zero iff p = q. Directed: KL(p||q) != KL(q||p).

Hard one-hot y: H(y)=0, so CE(y,q) = KL(y||q).
Soft target p:  CE and KL differ by the constant H(p); same argmin in q.

Steps:
  1. Definition (extra coding cost)
  2. CE = H + KL
  3. Hard labels => CE = KL
  4. Asymmetry (slider on q)
  5. Drag model toward target (temperature)
  6. Soft target: CE and KL move together
  7. Takeaway
  8. Torch -- scratch KL matches nn.KLDivLoss(batchmean)

,
""")
