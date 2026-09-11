"""
0870_Wq_Wk_Wv.py

Learnable W_Q, W_K, W_V are linear "perspective" maps on tokens X:

    Q = X W_Q ,   K = X W_K ,   V = X W_V

Crucial: the same W_Q, W_K, W_V are SHARED for every sequence the net sees
in training and inference. They are not fit to one particular (X, Y).
They absorb structure of the DATA DISTRIBUTION over (X, Y).

Key idea (one sentence):
    Which perspective on Q, matched against which perspective on K, creates
    attention weights for interpolating which perspective on V -- and that
    write, ADDED into the residual stream with every other layer's write,
    is what training judges by the loss on the whole distribution.

Easy logic:
  1) Each X is content.  Shared W_* only change the VIEW applied to any X.
  2) Attention weights = similarity of the Q-view to the K-view.
  3) Those weights mix the V-view (soft interpolation / combination).
  4) The mix is ADDED into the residual stream (chain of adds).
  5) Many perspective pairs can be useful knowledge (different layers / maps).
     "Different" is not "failed learning" -- it is another route the
     distribution made valuable for the loss.

Companion to 0860_QKV_attention.py, 0820_softmax.py, 0890_chain_of_adds.py.
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
import matplotlib.patches as patches
import numpy as np

np.random.seed(42)
RNG = np.random.default_rng(42)


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
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        try:
            manager.window.state("zoomed")
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


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def rot(theta_deg):
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]], dtype=float)


def scale_mat(sx, sy):
    return np.array([[sx, 0.0], [0.0, sy]], dtype=float)


# ---------------------------------------------------------------------------
# Toy sequence: token embeddings in 2D (the shared "content" X)
# ---------------------------------------------------------------------------
LABELS = ["the", "cat", "sat", "mat"]
# Hand-placed so "cat"~animal, "sat"~verb, "mat"~object sit in distinct regions
X = np.array([
    [0.15, 0.10],   # the  (function word, near origin)
    [1.80, 0.35],   # cat  (subject / animal)
    [0.40, 1.70],   # sat  (verb / action)
    [1.55, 1.45],   # mat  (object / place)
], dtype=float)
N, D = X.shape
COLORS = plt.cm.tab10(np.linspace(0, 0.75, N))

# Two SHARED linear perspective triples (two useful learned views).
# Full-rank maps: no information deleted -- only a change of coordinates.
# Both are legitimate knowledge: they attend differently because the
# distribution rewards more than one useful view of (X, Y).
W_Q_A = rot(35) @ scale_mat(1.3, 0.7)         # perspective A: "who/subject?" ask
W_K_A = rot(-20) @ scale_mat(0.85, 1.4)       # perspective A: subject-ish ads
W_V_A = rot(10) @ scale_mat(1.1, 1.1)         # perspective A: subject-ish payload

W_Q_B = rot(-50) @ scale_mat(0.6, 1.5)        # perspective B: "what happened?" ask
W_K_B = rot(70) @ scale_mat(1.6, 0.5)         # perspective B: action/place ads
W_V_B = rot(-15) @ scale_mat(0.9, 1.2)        # perspective B: action/place payload

Q_A, K_A, V_A = X @ W_Q_A, X @ W_K_A, X @ W_V_A
Q_B, K_B, V_B = X @ W_Q_B, X @ W_K_B, X @ W_V_B

# Aliases used in early figures (show one shared triple first)
W_Q, W_K, W_V = W_Q_A, W_K_A, W_V_A
Q, K, V = Q_A, K_A, V_A


def attention(Q, K, V, q_index=2):
    """Single-query attention from token q_index over all keys/values."""
    d = K.shape[1]
    scores = (Q[q_index] @ K.T) / np.sqrt(d)
    alpha = softmax(scores)
    out = alpha @ V
    return scores, alpha, out


# Same shared W_* applied to this X; query token = "sat"
Q_IDX = 2  # sat
scores_A, alpha_A, out_A = attention(Q_A, K_A, V_A, Q_IDX)
scores_B, alpha_B, out_B = attention(Q_B, K_B, V_B, Q_IDX)
dist_A = float(np.linalg.norm(out_A - V_A[1]))  # vs V_A("cat")
dist_B = float(np.linalg.norm(out_B - V_B[2]))  # vs V_B("sat")


# -------------------------------------------------------
# FIGURE 1: Same X; W turns it into a different perspective
# -------------------------------------------------------
def build_step_1():
    fig1, axes1 = plt.subplots(1, 3, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig1.suptitle(
        r'Step 1: Same tokens $X$ -- $W_Q,W_K,W_V$ only change the linear perspective',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    panels = [
        (X, np.eye(2), 'content $X$\n(shared raw tokens)', 'X'),
        (Q, W_Q, r'Q-view  $X W_Q$', 'Q'),
        (K, W_K, r'K-view  $X W_K$', 'K'),
    ]
    for ax, (pts, W, title, tag) in zip(axes1, panels):
        # Draw original as ghosts when showing a view
        if tag != 'X':
            ax.scatter(X[:, 0], X[:, 1], s=40, c='0.75', zorder=1)
            for i, lab in enumerate(LABELS):
                ax.text(X[i, 0] + 0.05, X[i, 1] + 0.05, lab, color='0.6', fontsize=FS_ANNOT - 2)
            # arrows from X to mapped point
            for i in range(N):
                ax.annotate(
                    '', xy=pts[i], xytext=X[i],
                    arrowprops=dict(arrowstyle='->', color='0.7', lw=1.2),
                )
        for i, (p, lab, c) in enumerate(zip(pts, LABELS, COLORS)):
            ax.scatter([p[0]], [p[1]], s=160, color=c, edgecolor='k', zorder=3)
            ax.text(p[0] + 0.08, p[1] + 0.08, lab, fontsize=FS_ANNOT, fontweight='bold')
        # Show basis image of e1, e2 under W
        o = np.array([0.0, 0.0])
        e1, e2 = W[:, 0], W[:, 1]
        ax.quiver(o[0], o[1], e1[0], e1[1], angles='xy', scale_units='xy', scale=1.0,
                  color='crimson', width=0.008, zorder=4)
        ax.quiver(o[0], o[1], e2[0], e2[1], angles='xy', scale_units='xy', scale=1.0,
                  color='navy', width=0.008, zorder=4)
        ax.set_title(title, fontsize=FS_AX_TITLE)
        ax.set_xlabel('dim 0', fontsize=FS_AXIS)
        ax.set_ylabel('dim 1', fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='0.5', lw=0.6)
        ax.axvline(0, color='0.5', lw=0.6)

    axes1[0].text(
        0.02, 0.02,
        'Red/blue arrows = images of\n'
        'the standard basis under W.\n'
        'A full-rank W does not delete\n'
        'information -- it re-views it.\n\n'
        'The SAME W is reused for\n'
        'every sequence in train/test:\n'
        'it encodes distributional\n'
        'knowledge, not one X.',
        transform=axes1[0].transAxes, va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    return fig1



# -------------------------------------------------------
# FIGURE 1b: Shared W applied to MANY sequences from a distribution
# -------------------------------------------------------
# Sample several short sequences by jittering / resampling roles from the
# same token vocabulary -- the SHARED W_Q maps all of them.
def build_step_2():
    fig1b, axes1b = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig1b.suptitle(
        r'Step 1b: One shared $W_Q$ views many $X$ from the data distribution',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    # Build a tiny "dataset" of sequences: same roles, different draws
    dataset_X = []
    for s in range(12):
        noise = 0.18 * RNG.normal(size=X.shape)
        # occasional swap of object/place emphasis
        Xs = X + noise
        if s % 3 == 0:
            Xs[3] = Xs[3] + np.array([0.35, -0.25])
        dataset_X.append(Xs)

    ax = axes1b[0]
    for s, Xs in enumerate(dataset_X):
        ax.scatter(Xs[:, 0], Xs[:, 1], s=28, c=COLORS, alpha=0.35, edgecolors='none')
    for i, lab in enumerate(LABELS):
        ax.scatter([X[i, 0]], [X[i, 1]], s=120, color=COLORS[i], edgecolor='k', zorder=4)
        ax.text(X[i, 0] + 0.06, X[i, 1] + 0.06, lab, fontsize=FS_ANNOT, fontweight='bold')
    ax.set_title('Many sequences X ~ data distribution', fontsize=FS_AX_TITLE)
    ax.set_xlabel('content dim 0', fontsize=FS_AXIS)
    ax.set_ylabel('content dim 1', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.02,
        'Training sees clouds of X,\n'
        'paired with targets Y\n'
        '(labels / next tokens / ...).\n'
        'W_* must work for all of them.',
        transform=ax.transAxes, va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    ax = axes1b[1]
    for s, Xs in enumerate(dataset_X):
        Qs = Xs @ W_Q_A   # SAME shared W_Q for every sequence
        ax.scatter(Qs[:, 0], Qs[:, 1], s=28, c=COLORS, alpha=0.35, edgecolors='none')
    Q0 = X @ W_Q_A
    for i, lab in enumerate(LABELS):
        ax.scatter([Q0[i, 0]], [Q0[i, 1]], s=120, color=COLORS[i], edgecolor='k', zorder=4)
        ax.text(Q0[i, 0] + 0.06, Q0[i, 1] + 0.06, lab, fontsize=FS_ANNOT, fontweight='bold')
    ax.set_title(r'Same shared $W_Q$ on every $X$  $\Rightarrow$ one learned view', fontsize=FS_AX_TITLE)
    ax.set_xlabel('Q-view dim 0', fontsize=FS_AXIS)
    ax.set_ylabel('Q-view dim 1', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98, 0.02,
        'W_Q is not customized\n'
        'per sentence. It is a\n'
        'single linear camera\n'
        'fit to P(X, Y).',
        transform=ax.transAxes, va='bottom', ha='right', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
    )

    fig1b.tight_layout(rect=[0, 0, 1, 0.93])
    return fig1b



# -------------------------------------------------------
# FIGURE 2: Q-perspective over K-perspective => attention weights
# -------------------------------------------------------
def build_step_3():
    fig2, axes2 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig2.suptitle(
        r'Step 2: Which Q-perspective over which K-perspective creates the weights?',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    ax = axes2[0]
    # Plot K-view points; query is Q-view of "sat"
    for i, (p, lab, c) in enumerate(zip(K, LABELS, COLORS)):
        ax.scatter([p[0]], [p[1]], s=160, color=c, edgecolor='k', zorder=3)
        ax.text(p[0] + 0.08, p[1] + 0.08, lab, fontsize=FS_ANNOT, fontweight='bold')
    ax.scatter([Q[Q_IDX, 0]], [Q[Q_IDX, 1]], s=280, c='crimson', marker='*',
               edgecolor='k', zorder=5, label=r'query = Q-view of "sat"')
    for i, (p, c, a) in enumerate(zip(K, COLORS, alpha_A)):
        ax.annotate(
            '', xy=p, xytext=Q[Q_IDX],
            arrowprops=dict(
                arrowstyle='->', color=c,
                lw=1.0 + 6.0 * float(a),
                alpha=0.35 + 0.65 * float(a),
            ),
        )
    ax.set_title('Perspective A: Q-view asks; K-view answers', fontsize=FS_AX_TITLE)
    ax.set_xlabel('K-view dim 0', fontsize=FS_AXIS)
    ax.set_ylabel('K-view dim 1', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc='best')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.02,
        'Thicker arrow = larger\n'
        r'score $(q W_Q)\cdot(k W_K)/\sqrt{d}$.' + '\n'
        'Perspective A learned a subject-ish\n'
        'matching geometry (sat->cat).',
        transform=ax.transAxes, va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    ax = axes2[1]
    x = np.arange(N)
    w = 0.35
    ax.bar(x - w / 2, alpha_A, width=w, color='seagreen', edgecolor='k',
           label='perspective A $(W_Q^A,W_K^A)$')
    ax.bar(x + w / 2, alpha_B, width=w, color='royalblue', edgecolor='k',
           label='perspective B $(W_Q^B,W_K^B)$')
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r'attention weight $\alpha_i$', fontsize=FS_AXIS)
    ax.set_title('Two shared perspective triples: different views, both learned knowledge', fontsize=FS_AX_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, axis='y', alpha=0.3)
    ax.text(
        0.98, 0.98,
        'B is not "failed A".\n'
        'It is another route the\n'
        'distribution made useful\n'
        '(here: more action/place).\n'
        'A network may keep several\n'
        'such shared perspectives at once.',
        transform=ax.transAxes, va='top', ha='right', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
    )

    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    return fig2



# -------------------------------------------------------
# FIGURE 3: Those weights interpolate the V-perspective
# -------------------------------------------------------
def build_step_4():
    fig3, axes3 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig3.suptitle(
        r'Step 3: Weights interpolate / combine the V-perspective  ($\sum_i \alpha_i v_i$)',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    ax = axes3[0]
    for i, (p, lab, c, a) in enumerate(zip(V_A, LABELS, COLORS, alpha_A)):
        ax.scatter([p[0]], [p[1]], s=70 + 500 * a, color=c, edgecolor='k',
                   alpha=0.35 + 0.65 * a, zorder=3)
        ax.text(p[0] + 0.06, p[1] + 0.06, lab, fontsize=FS_ANNOT, fontweight='bold')
        ax.annotate(
            '', xy=out_A, xytext=p,
            arrowprops=dict(arrowstyle='->', color=c, lw=1.0 + 4.0 * float(a),
                            alpha=0.25 + 0.6 * float(a)),
        )
    ax.scatter([out_A[0]], [out_A[1]], s=220, c='k', marker='*',
               zorder=5, label='output mix (perspective A)')
    ax.set_title('V-view A: weights interpolate this payload geometry', fontsize=FS_AX_TITLE)
    ax.set_xlabel('V-view dim 0', fontsize=FS_AXIS)
    ax.set_ylabel('V-view dim 1', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc='best')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.02,
        'W_V chooses WHICH payload\n'
        'coordinates get mixed.\n'
        'alpha (from W_Q,W_K) chooses\n'
        'HOW MUCH of each token.',
        transform=ax.transAxes, va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    ax = axes3[1]
    # Two learned mixes: different knowledge routes on the SAME shared X
    ref_A = V_A[1]   # cat in V_A coords (subject-ish target)
    ref_B = V_B[2]   # sat in V_B coords (action-ish reference for B)
    ax.scatter([ref_A[0]], [ref_A[1]], s=160, c=COLORS[1], edgecolor='k',
               marker='o', label='V_A("cat") reference', zorder=4)
    ax.scatter([ref_B[0]], [ref_B[1]], s=160, c=COLORS[2], edgecolor='k',
               marker='s', label='V_B("sat") reference', zorder=4)
    ax.scatter([out_A[0]], [out_A[1]], s=220, c='k', marker='*',
               label='mix A (subject-ish view)', zorder=5)
    ax.scatter([out_B[0]], [out_B[1]], s=200, c='0.35', marker='*',
               label='mix B (action/place view)', zorder=5)
    ax.plot([out_A[0], ref_A[0]], [out_A[1], ref_A[1]], '--', color='seagreen', lw=1.8)
    ax.plot([out_B[0], ref_B[0]], [out_B[1], ref_B[1]], '--', color='royalblue', lw=1.8)
    ax.set_title('Different perspectives write different useful mixes -- both are knowledge', fontsize=FS_AX_TITLE)
    ax.set_xlabel('dim 0', fontsize=FS_AXIS)
    ax.set_ylabel('dim 1', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND - 1, loc='best')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    dist_A = float(np.linalg.norm(out_A - ref_A))
    dist_B = float(np.linalg.norm(out_B - ref_B))
    ax.text(
        0.98, 0.02,
        f'|mixA - cat_A| = {dist_A:.2f}\n'
        f'|mixB - sat_B| = {dist_B:.2f}\n\n'
        'Not "right vs wrong":\n'
        'two shared perspectives\n'
        'serving different parts\n'
        'of the (X,Y) distribution.',
        transform=ax.transAxes, va='bottom', ha='right', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
    )

    fig3.tight_layout(rect=[0, 0, 1, 0.93])
    return fig3



# -------------------------------------------------------
# FIGURE 4: One sentence diagram -- perspectives => weights => mix V
# -------------------------------------------------------
def build_step_5():
    fig4, ax4 = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig4.suptitle(
        'Step 4: The key sentence, as a flowchart',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    ax4.set_xlim(0, 12)
    ax4.set_ylim(0, 8)
    ax4.axis('off')


    def box(ax, x, y, w, h, text, color):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
            facecolor=color, edgecolor='k', lw=2,
        ))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=FS_ANNOT, fontweight='bold')


    box(ax4, 0.4, 5.5, 2.4, 1.6, 'tokens X\n(shared content)', '#dceaff')
    box(ax4, 3.3, 6.4, 2.2, 1.2, r'$W_Q$ view', '#f8d7da')
    box(ax4, 3.3, 4.8, 2.2, 1.2, r'$W_K$ view', '#fff3cd')
    box(ax4, 3.3, 3.2, 2.2, 1.2, r'$W_V$ view', '#d4edda')
    box(ax4, 6.2, 5.2, 2.6, 1.8, 'match Q-view\nover K-view\n=> weights α', '#ffe0cc')
    box(ax4, 9.2, 5.2, 2.4, 1.8, 'interpolate\nV-view with α\n=> write Δ', '#e5d4ff')
    box(ax4, 6.2, 1.4, 5.4, 2.0,
        'ADD Δ into residual stream\n'
        '(join the sequence of adds from each layer)\n'
        'Loss asks: did these perspectives help?',
        '#f7f7ff')

    # arrows
    for (x0, y0, x1, y1) in [
        (2.8, 6.3, 3.3, 7.0),
        (2.8, 6.1, 3.3, 5.4),
        (2.8, 5.9, 3.3, 3.8),
        (5.5, 7.0, 6.2, 6.4),
        (5.5, 5.4, 6.2, 6.0),
        (8.8, 6.1, 9.2, 6.1),
        (10.4, 5.2, 10.4, 3.4),
    ]:
        ax4.annotate('', xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle='->', lw=2.0, color='k'))

    ax4.text(
        0.4, 0.35,
        'The SAME W_Q, W_K, W_V apply to every sequence in train and inference.\n'
        'Training shapes those shared maps from the DISTRIBUTION of (X, Y),\n'
        'so each new X is viewed with knowledge accumulated across the dataset.\n'
        'Many perspective pairs (layers / alternate maps) can all be useful knowledge.',
        fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    return fig4



# -------------------------------------------------------
# FIGURE 5: Joining the chain of adds -- each layer writes a perspective-mix
# -------------------------------------------------------
def build_step_6():
    fig5, axes5 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig5.suptitle(
        'Step 5: Each layer ADDS its perspective-mix -- loss judges the whole sum',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    # Toy residual unrolling in 2D: start at embed, add layer writes
    L = 4
    x0 = X.mean(axis=0)  # toy "stream state"
    # Pretend each layer uses a slightly different perspective mix as its write
    layer_Ws = [
        (W_Q_A, W_K_A, W_V_A),                 # subject-ish shared perspective
        (rot(15) @ W_Q_A, rot(-10) @ W_K_A, rot(5) @ W_V_A),
        (W_Q_B, W_K_B, W_V_B),                 # action/place shared perspective (also knowledge)
        (rot(-25) @ W_Q_A, rot(30) @ W_K_A, rot(-5) @ W_V_A),
    ]
    deltas = []
    alphas_layers = []
    for Wq, Wk, Wv in layer_Ws:
        _, a, out = attention(X @ Wq, X @ Wk, X @ Wv, Q_IDX)
        # residual write = projected mix (toy: just the out vector as delta)
        deltas.append(0.55 * out)
        alphas_layers.append(a)
    deltas = np.stack(deltas)
    stream = [x0.copy()]
    for d in deltas:
        stream.append(stream[-1] + d)
    stream = np.stack(stream)

    ax = axes5[0]
    ax.plot(stream[:, 0], stream[:, 1], 'o-', color='royalblue', lw=2.5, ms=8, label='residual stream')
    for i in range(L):
        ax.annotate(
            '', xy=stream[i + 1], xytext=stream[i],
            arrowprops=dict(arrowstyle='->', color='crimson', lw=2.0),
        )
        mid = 0.5 * (stream[i] + stream[i + 1])
        ax.text(mid[0] + 0.03, mid[1] + 0.03, rf'$\Delta_{i}$', color='crimson', fontsize=FS_ANNOT)
    ax.scatter([stream[0, 0]], [stream[0, 1]], s=120, c='orange', zorder=5, label=r'$x_0$ embed')
    ax.scatter([stream[-1, 0]], [stream[-1, 1]], s=160, c='k', marker='*', zorder=5, label=r'$x_L$ final')
    ax.set_title(r'$x_L = x_0 + \sum_\ell \Delta_\ell(W_Q^{(\ell)},W_K^{(\ell)},W_V^{(\ell)})$',
                 fontsize=FS_AX_TITLE)
    ax.set_xlabel('residual dim 0', fontsize=FS_AXIS)
    ax.set_ylabel('residual dim 1', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc='best')
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.02,
        'Each Δ is an attention mix\n'
        'from THAT layer\'s perspectives.\n'
        'Nothing is overwritten --\n'
        'writes JOIN by addition.',
        transform=ax.transAxes, va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

    ax = axes5[1]
    im = ax.imshow(np.stack(alphas_layers, axis=0), aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(N))
    ax.set_xticklabels(LABELS)
    ax.set_yticks(range(L))
    ax.set_yticklabels([f'layer {i}' for i in range(L)])
    ax.set_xlabel('key / value token', fontsize=FS_AXIS)
    ax.set_ylabel('layer', fontsize=FS_AXIS)
    ax.set_title('Per-layer attention weights (from each layer\'s Q/K views)', fontsize=FS_AX_TITLE)
    ax.tick_params(labelsize=FS_TICK)
    fig5.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=FS_TICK)
    ax.text(
        0.98, 0.02,
        'Different layers keep their own\n'
        'shared W_Q, W_K, W_V -- each\n'
        'a distributional perspective.\n'
        'Different patterns are knowledge.\n'
        'Loss scores the SUM of writes.',
        transform=ax.transAxes, va='bottom', ha='right', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
    )

    fig5.tight_layout(rect=[0, 0, 1, 0.93])
    return fig5



# -------------------------------------------------------
# FIGURE 6: Loss picks the perspectives
# -------------------------------------------------------
def build_step_7():
    fig6, ax6 = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig6.suptitle(
        'Step 6: Easy moral -- loss selects perspectives whose residual writes help',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 8)
    ax6.axis('off')

    moral = (
        '1. Each X is content.  Shared W_Q, W_K, W_V are linear cameras reused on ALL X.\n\n'
        '2. Those cameras are learned from the DISTRIBUTION of (X, Y), not one example.\n\n'
        '3. Weights = Q-camera question matched to K-camera ads; mix = weights on V-camera.\n\n'
        '4. Each layer ADDS its write into the residual stream (chain of adds).\n\n'
        '5. Many perspective pairs can help: different is not "less learning" --\n'
        '      it is more knowledge about the distribution, joined by addition.\n\n'
        '6. Loss keeps the shared cameras whose summed writes improve predictions\n'
        '      across the training distribution (and hopefully at inference).\n\n'
        '7. Fixed LINEAR preprocessing of X is free to absorb into W_* (see 242 Step 14):\n'
        '      W (P X) = (W P) X -- training can learn W\'=WP with same power.'
    )
    ax6.text(
        0.5, 0.55, moral, transform=ax6.transAxes, ha='center', va='center',
        fontsize=FS_AXIS - 1, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
    )
    ax6.text(
        0.5, 0.08,
        r'$x \leftarrow x + \mathrm{Attention}(X W_Q,\ X W_K,\ X W_V)$'
        '   with the SAME W_* for every sequence',
        transform=ax6.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    return fig6




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
print("LEARNABLE W_Q, W_K, W_V = LEARNABLE PERSPECTIVES")
print("=" * 60)
print(f"""
Shared maps (one set of W_* reused for every sequence X):
  Q = X W_Q     # how each token ASKS
  K = X W_K     # how each token ADVERTISES itself for matching
  V = X W_V     # what payload each token OFFERS to be mixed

  W_Q, W_K, W_V are learned from the training DISTRIBUTION over (X, Y).
  At inference the same W_* view new X with that distributional knowledge.

Attention write from one query token:
  alpha = softmax( (q W_Q)(K W_K)^T / sqrt(d) )
  delta = alpha @ (X W_V)     # interpolate V-perspective with those weights

Residual join (per layer, simplified):
  x <- x + delta   (+ FFN write, ...)
  Final x_L = x_0 + sum_layer delta_layer

Toy demo (query token = '{LABELS[Q_IDX]}'), two shared perspective triples:
  perspective A  alpha ~ {np.round(alpha_A, 3).tolist()}
  perspective B  alpha ~ {np.round(alpha_B, 3).tolist()}
  |mixA - V_A(cat)| = {dist_A:.3f}
  |mixB - V_B(sat)| = {dist_B:.3f}
  B is not "failed A" -- another useful view of the same distribution.

The key, in one breath:
  Which Q-perspective over which K-perspective creates weights for
  interpolating which V-perspective -- those writes JOIN by addition;
  many such shared perspectives can be knowledge; loss keeps what helps
  on the distribution, not on a single (X, Y).

Preprocessing note (for customers):
  Because learned linear maps are so flexible -- and compose -- a fixed
  linear change of the input (rescale, whiten, fixed projection, ...) can
  be absorbed into W_Q / W_K / W_V without cutting representation power:
      Q = (X P) W_Q = X (P W_Q)
  so training can just learn an equivalent matrix. You need not fear that
  such linear data prep will ruin accuracy for a net that already has
  learned linear layers (it always does). Nonlinear prep is different.
  Details / picture: 0800_linear_transformation.py Step 14.

See also:
  0860_QKV_attention.py     -- Q/K/V roles without emphasizing W
  0800_linear_transformation.py -- why matrices, and absorbing preprocess
  0820_softmax.py           -- weights from scores
  0890_chain_of_adds.py     -- residual stream as running sum
""")
