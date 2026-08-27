"""
156_QKV_attention.py

What Q, K, V mean in Transformer attention.

Metaphor (a lookup database):
  * Each item stores a VALUE (the payload you might want).
  * Each item also has a KEY -- a representation / handle of that value
    (what the item "advertises" so others can find it).
  * A QUERY is a question: "what am I looking for right now?"

Procedure:
  1) Score how similar the query is to each key  (usually a scaled dot).
  2) Softmax those scores -> attention weights (nonnegative, sum to 1).
  3) Return a mixture of values, weighted by those attentions.
     You draw as much of each value as the similarity allows.

In a Transformer, tokens play all three roles after linear maps:
  Q = X W_Q ,   K = X W_K ,   V = X W_V

What training learns:
  Which projections (perspectives) of the same underlying tokens make
  this retrieve-and-mix step reduce the loss best.

Companion to 150_transformer.py, 151_transformer_code.py, 155_dot_*.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

np.random.seed(42)
RNG = np.random.default_rng(42)

FS_FIGURE_TITLE = 28
FS_AX_TITLE = 22
FS_AXIS = 16
FS_ANNOT = 15
FS_LEGEND = 14

SCREEN_FRACTION = 0.95
FIGURE_DPI = 120

plt.rcParams.update({
    'figure.titlesize': FS_FIGURE_TITLE,
    'axes.titlesize': FS_AX_TITLE,
    'axes.labelsize': FS_AXIS,
    'xtick.labelsize': FS_AXIS,
    'ytick.labelsize': FS_AXIS,
    'legend.fontsize': FS_LEGEND,
    'figure.dpi': FIGURE_DPI,
})


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
    plt.show()


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# Toy "database": (key, value) pairs + one question
# ---------------------------------------------------------------------------
# Keys live in a 2D "advertising" space; values in a 2D "content" space.
# Keys are *about* their values -- similar topics sit near each other as keys,
# while values carry the actual payload vectors.

ITEMS = [
    #          key (advert)         value (payload)        label
    (np.array([2.2, 0.3]), np.array([1.0, 0.1]), 'cat'),
    (np.array([2.0, 0.5]), np.array([0.9, 0.2]), 'dog'),
    (np.array([-0.2, 2.1]), np.array([0.1, 1.0]), 'run'),
    (np.array([0.1, 1.9]), np.array([0.2, 0.9]), 'jump'),
    (np.array([-1.8, -0.4]), np.array([-1.0, 0.0]), 'red'),
]
KEYS = np.stack([k for k, _, _ in ITEMS], axis=0)    # (N, 2)
VALS = np.stack([v for _, v, _ in ITEMS], axis=0)    # (N, 2)
LABELS = [lab for _, _, lab in ITEMS]
N = len(LABELS)

# Question looking for an animal subject ("who?")
QUERY = np.array([2.1, 0.2])
D_K = KEYS.shape[1]

# Attention
scores = (KEYS @ QUERY) / np.sqrt(D_K)               # (N,)
attn = softmax(scores)                               # (N,)
retrieved = attn @ VALS                              # (2,)


# -------------------------------------------------------
# FIGURE 1: Database of (key, value) -- key represents its value
# -------------------------------------------------------
fig1, axes1 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig1.suptitle(
    'Step 1: A database of (key, value) pairs -- the key is a handle for the value',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes1[0]
colors = plt.cm.tab10(np.linspace(0, 0.8, N))
for i, (k, lab, c) in enumerate(zip(KEYS, LABELS, colors)):
    ax.scatter([k[0]], [k[1]], s=220, color=c, edgecolor='k', zorder=3)
    ax.text(k[0] + 0.08, k[1] + 0.08, lab, fontsize=FS_ANNOT, fontweight='bold')
ax.set_title('KEY space -- what each item advertises', fontsize=FS_AX_TITLE - 2)
ax.set_xlabel('key dim 0', fontsize=FS_AXIS)
ax.set_ylabel('key dim 1', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_AXIS)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='datalim')
ax.text(
    0.02, 0.98,
    'Nearby keys = similar "topics".\n'
    'Animals cluster; actions cluster.\n'
    'The key is NOT the payload --\n'
    'it is a representation used for LOOKUP.',
    transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes1[1]
for i, (v, lab, c) in enumerate(zip(VALS, LABELS, colors)):
    ax.scatter([v[0]], [v[1]], s=220, color=c, edgecolor='k', zorder=3)
    ax.text(v[0] + 0.05, v[1] + 0.05, lab, fontsize=FS_ANNOT, fontweight='bold')
ax.set_title('VALUE space -- the payload you may retrieve', fontsize=FS_AX_TITLE - 2)
ax.set_xlabel('value dim 0', fontsize=FS_AXIS)
ax.set_ylabel('value dim 1', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_AXIS)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='datalim')
ax.text(
    0.02, 0.98,
    'Values carry content to mix.\n'
    'Attention never compares values\n'
    'to the question -- only KEYS.\n'
    'Values are what get blended.',
    transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
)

fig1.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Query matches keys by similarity
# -------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig2.suptitle(
    r'Step 2: Query asks; similarity to each key = score $q\cdot k / \sqrt{d}$',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes2[0]
ax.scatter([QUERY[0]], [QUERY[1]], s=280, c='crimson', marker='*',
           edgecolor='k', zorder=5, label='query q')
for i, (k, lab, c) in enumerate(zip(KEYS, LABELS, colors)):
    ax.scatter([k[0]], [k[1]], s=180, color=c, edgecolor='k', zorder=3)
    ax.annotate(
        '', xy=k, xytext=QUERY,
        arrowprops=dict(
            arrowstyle='->', color=c,
            lw=1.0 + 5.0 * float(attn[i]),
            alpha=0.35 + 0.65 * float(attn[i]),
        ),
    )
    ax.text(k[0] + 0.08, k[1] + 0.08, lab, fontsize=FS_ANNOT)
ax.set_title('Question in key-space', fontsize=FS_AX_TITLE)
ax.set_xlabel('key dim 0', fontsize=FS_AXIS)
ax.set_ylabel('key dim 1', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_AXIS)
ax.legend(fontsize=FS_LEGEND, loc='lower left')
ax.grid(True, alpha=0.3)
ax.text(
    0.98, 0.98,
    'Thicker arrow => larger raw\n'
    'dot-product with the query.\n'
    'Animals align with this q;\n'
    'actions / colors do not.',
    transform=ax.transAxes, va='top', ha='right', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes2[1]
bars = ax.bar(LABELS, scores, color=colors, edgecolor='k')
ax.axhline(0, color='0.4', lw=1)
ax.set_ylabel(r'score $q\cdot k_i / \sqrt{d}$', fontsize=FS_AXIS)
ax.set_title('Raw similarity scores', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_AXIS)
ax.grid(True, axis='y', alpha=0.3)
for b, s in zip(bars, scores):
    ax.text(b.get_x() + b.get_width() / 2, s + 0.05, f'{s:.2f}',
            ha='center', va='bottom', fontsize=FS_ANNOT - 1)
ax.text(
    0.02, 0.98,
    'Same geometry as 155:\n'
    'dot product = length x\n'
    'shadow agreement.\n'
    'Scaling by 1/sqrt(d) keeps\n'
    'softmax well-behaved.',
    transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
)

fig2.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Softmax -> attention; mix values
# -------------------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig3.suptitle(
    r'Step 3: Softmax turns scores into attention; output = $\sum_i \alpha_i v_i$',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes3[0]
bars = ax.bar(LABELS, attn, color=colors, edgecolor='k')
ax.set_ylim(0, 1.05)
ax.set_ylabel(r'attention $\alpha_i$', fontsize=FS_AXIS)
ax.set_title(r'$\alpha = \mathrm{softmax}(\mathrm{scores})$', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_AXIS)
ax.grid(True, axis='y', alpha=0.3)
for b, a in zip(bars, attn):
    ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f'{a:.2f}',
            ha='center', va='bottom', fontsize=FS_ANNOT - 1)
ax.text(
    0.98, 0.98,
    'Nonnegative, sum to 1.\n'
    'How MUCH of each value\n'
    'you are allowed to draw.\n'
    f'Sum check: {attn.sum():.4f}',
    transform=ax.transAxes, va='top', ha='right', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes3[1]
for i, (v, lab, c, a) in enumerate(zip(VALS, LABELS, colors, attn)):
    ax.scatter([v[0]], [v[1]], s=80 + 400 * a, color=c, edgecolor='k',
               alpha=0.35 + 0.65 * a, zorder=3)
    ax.text(v[0] + 0.05, v[1] + 0.05, lab, fontsize=FS_ANNOT,
            alpha=0.4 + 0.6 * a)
# retrieved vector
ax.scatter([retrieved[0]], [retrieved[1]], s=320, c='crimson', marker='*',
           edgecolor='k', zorder=5, label='retrieved mix')
ax.annotate(
    'weighted sum of values',
    xy=retrieved, xytext=(retrieved[0] - 0.9, retrieved[1] + 0.55),
    fontsize=FS_ANNOT, color='crimson',
    arrowprops=dict(arrowstyle='->', color='crimson', lw=2),
)
ax.set_title('Draw value in proportion to attention', fontsize=FS_AX_TITLE)
ax.set_xlabel('value dim 0', fontsize=FS_AXIS)
ax.set_ylabel('value dim 1', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_AXIS)
ax.legend(fontsize=FS_LEGEND, loc='lower left')
ax.grid(True, alpha=0.3)
ax.text(
    0.02, 0.98,
    'Marker size ~ attention.\n'
    'Output leans toward cat/dog\n'
    'payloads -- what the question\n'
    'was allowed to take.',
    transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
)

fig3.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Pipeline diagram Q -> scores -> softmax -> mix V
# -------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig4.suptitle(
    'Step 4: End-to-end attention as soft database lookup',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax4.set_xlim(0, 14)
ax4.set_ylim(0, 8)
ax4.axis('off')


def box(ax, x, y, w, h, text, color):
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
        facecolor=color, edgecolor='k', lw=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=FS_ANNOT, fontweight='bold')


box(ax4, 0.4, 5.5, 2.6, 1.6, 'QUERY q\n"what I need"', '#f8d7da')
box(ax4, 3.6, 5.5, 2.6, 1.6, 'KEYS k_i\n"handles"', '#d1ecf1')
box(ax4, 6.8, 5.5, 2.6, 1.6, 'scores\nq·k / sqrt(d)', '#fff3cd')
box(ax4, 10.0, 5.5, 3.4, 1.6, 'softmax -> α_i\nattention', '#d4edda')

box(ax4, 3.6, 2.2, 2.6, 1.6, 'VALUES v_i\n"payloads"', '#e2d5f1')
box(ax4, 7.5, 2.2, 5.0, 1.6, 'output = sum α_i v_i\nsoft retrieval', '#ffe0b2')

ax4.annotate('', xy=(3.6, 6.3), xytext=(3.0, 6.3),
             arrowprops=dict(arrowstyle='->', lw=2.5))
ax4.annotate('', xy=(6.8, 6.3), xytext=(6.2, 6.3),
             arrowprops=dict(arrowstyle='->', lw=2.5))
ax4.annotate('', xy=(10.0, 6.3), xytext=(9.4, 6.3),
             arrowprops=dict(arrowstyle='->', lw=2.5))
ax4.annotate('', xy=(4.9, 3.8), xytext=(4.9, 5.5),
             arrowprops=dict(arrowstyle='->', lw=2.0, color='0.4'))
ax4.annotate('', xy=(7.5, 3.0), xytext=(11.0, 5.5),
             arrowprops=dict(arrowstyle='->', lw=2.5, color='seagreen',
                             connectionstyle='arc3,rad=-0.25'))
ax4.annotate('', xy=(7.5, 3.0), xytext=(4.9, 2.2),
             arrowprops=dict(arrowstyle='->', lw=2.5, color='purple'))

ax4.text(
    7.0, 0.7,
    'Key insight: matching happens in KEY space; mixing happens in VALUE space.\n'
    'The key is a representation of its accompanying value -- an address card for the payload.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: In Transformers, Q,K,V are learned projections of tokens
# -------------------------------------------------------
# Same token embeddings X; two different projection "perspectives"
TOKENS = ['the', 'cat', 'sat', 'mat']
T = len(TOKENS)
# 2D toy embeddings
X = np.array([
    [0.2, 0.1],
    [1.5, 0.2],
    [0.3, 1.4],
    [0.4, 1.2],
], dtype=float)

# Perspective A: emphasizes "subject/object noun" vs "verb" in key/query
Wq_a = np.array([[1.0, 0.1], [0.0, 0.2]])
Wk_a = np.array([[1.0, 0.0], [0.1, 0.3]])
Wv_a = np.eye(2)

# Perspective B: muddled / different axes -- worse for "who sat?"
Wq_b = np.array([[0.2, 0.9], [0.8, 0.1]])
Wk_b = np.array([[0.1, 1.0], [0.9, 0.2]])
Wv_b = np.array([[0.0, 1.0], [1.0, 0.0]])


def attn_from_proj(X, Wq, Wk, Wv, q_index=2):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    d = Q.shape[-1]
    scores = (Q[q_index] @ K.T) / np.sqrt(d)
    alpha = softmax(scores)
    out = alpha @ V
    return Q, K, V, scores, alpha, out


Qa, Ka, Va, sa, aa, out_a = attn_from_proj(X, Wq_a, Wk_a, Wv_a, q_index=2)
Qb, Kb, Vb, sb, ab, out_b = attn_from_proj(X, Wq_b, Wk_b, Wv_b, q_index=2)

fig5, axes5 = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig5.suptitle(
    r'Step 5: Learned $W_Q,W_K,W_V$ = choosing a perspective that helps the loss',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes5[0]
xidx = np.arange(T)
w = 0.35
ax.bar(xidx - w / 2, aa, width=w, color='#1e8449', edgecolor='k',
       label='perspective A (useful)')
ax.bar(xidx + w / 2, ab, width=w, color='#c0392b', edgecolor='k', alpha=0.85,
       label='perspective B (muddled)')
ax.set_xticks(xidx)
ax.set_xticklabels(TOKENS)
ax.set_ylabel('attention from query "sat"', fontsize=FS_AXIS)
ax.set_title('Same tokens X; different projections -> different α', fontsize=FS_AX_TITLE - 4)
ax.tick_params(axis='both', labelsize=FS_AXIS)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, axis='y', alpha=0.3)
ax.text(
    0.02, 0.98,
    'Query token = "sat".\n'
    'Useful perspective attends to "cat".\n'
    'Muddled perspective spreads mass.\n'
    'Training prefers the projection\n'
    'that reduces task loss.',
    transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes5[1]
ax.axis('off')
ax.set_title('What is learned?', fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.55,
    'Tokens start as raw embeddings X.\n\n'
    r'$Q = X W_Q,\quad K = X W_K,\quad V = X W_V$' + '\n\n'
    'W_Q / W_K / W_V are linear maps -- a CHANGE OF PERSPECTIVE:\n'
    '  • W_K: how should I advertise this token as a key?\n'
    '  • W_Q: how should I phrase my question?\n'
    '  • W_V: what payload should I pass if selected?\n\n'
    'Gradient descent tunes those maps so that\n'
    '  "ask in this way / match on that handle / pass this content"\n'
    'yields a better objective (lower loss).\n\n'
    'Multi-head = several perspectives in parallel.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

fig5.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Roles cheat-sheet
# -------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig6.suptitle(
    'Step 6: Q, K, V roles -- one picture',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax6.set_xlim(0, 12)
ax6.set_ylim(0, 8)
ax6.axis('off')

roles = [
    (0.5, 4.5, 3.3, 2.8, 'QUERY (Q)',
     'The question.\n"What information\ndo I need now?"', '#f8d7da'),
    (4.35, 4.5, 3.3, 2.8, 'KEY (K)',
     'Address card for a value.\n"Here is how to find\nmy payload."', '#d1ecf1'),
    (8.2, 4.5, 3.3, 2.8, 'VALUE (V)',
     'The payload.\n"If you select me,\nhere is what I give."', '#d4edda'),
]
for x, y, w, h, title, body, color in roles:
    ax6.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
        facecolor=color, edgecolor='k', lw=2,
    ))
    ax6.text(x + w / 2, y + h - 0.35, title, ha='center', va='top',
             fontsize=FS_ANNOT + 2, fontweight='bold')
    ax6.text(x + w / 2, y + 0.9, body, ha='center', va='center',
             fontsize=FS_ANNOT)

ax6.text(
    6.0, 2.2,
    r'Attention$(Q,K,V)=\mathrm{softmax}\!\left(QK^\top/\sqrt{d}\right)V$',
    ha='center', fontsize=FS_AXIS,
    bbox=dict(boxstyle='round', facecolor='#fff3cd', alpha=0.95),
)
ax6.text(
    6.0, 0.7,
    'Similarity is between questions and keys; values are what get mixed.\n'
    'Learning = finding projections where that soft lookup helps the task.',
    ha='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
show_maximized(fig6)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("Q, K, V IN TRANSFORMER ATTENTION")
print("=" * 60)
print(f"""
Database metaphor:
  KEY   = representation / handle of a VALUE (how the item advertises itself)
  VALUE = payload you may retrieve
  QUERY = question ("what do I need?")

Algorithm:
  scores_i = <q, k_i> / sqrt(d)
  alpha    = softmax(scores)          # attention weights
  output   = sum_i alpha_i * v_i      # draw value ~ similarity

Toy lookup demo:
  query q = {QUERY.tolist()}
  items   = {LABELS}
  scores  = {np.round(scores, 3).tolist()}
  alpha   = {np.round(attn, 3).tolist()}
  retrieved value = {np.round(retrieved, 3).tolist()}

In a Transformer layer:
  Q = X W_Q,  K = X W_K,  V = X W_V
  Training learns W_Q, W_K, W_V -- which PERSPECTIVE on the tokens
  makes this soft database lookup reduce the loss.
  Multi-head = several such perspectives at once.

See also:
  150_transformer.py -- attention heatmaps / block diagram
  151_transformer_code.py -- shapes through QKV
  155_dot_inner_scalar_product.py -- why dots measure agreement
""")
