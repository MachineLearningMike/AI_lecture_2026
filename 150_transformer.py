"""
150_transformer.py

Transformer intuition -- especially self-attention.

Core idea:
    Each token builds a query ("what am I looking for?"),
    keys ("what do I contain?"), and values ("what do I offer?").
    Attention weights = softmax(Q K^T / sqrt(d)).
    Output = weighted sum of values.

That lets every position directly mix information from every other
position in one step (global receptive field), without recurrence.

Companion to 120_normalization.py (LayerNorm in Transformers),
130_dropout.py, and 140_neural_non_linearity.py (FFN non-linearity).

Also sketches recent improvements (efficiency, long context, MoE,
alignment) and how Transformer usage diversified far beyond MT.
"""

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
    plt.show()


def add_note(ax, text, x=0.98, y=0.46, ha='right', va='top', facecolor='white'):
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=FS_ANNOT,
        va=va, ha=ha,
        bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.95),
    )


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def self_attention(X, Wq, Wk, Wv, causal=False):
    """
    X: (T, d_model)
    returns: out (T, d_v), weights (T, T), Q, K, V
    """
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    if causal:
        T = scores.shape[0]
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = scores.copy()
        scores[mask] = -1e9
    weights = softmax(scores, axis=-1)
    out = weights @ V
    return out, weights, Q, K, V, scores


# -------------------------------------------------------
# Toy sentence embeddings (hand-crafted for clear attention)
# tokens:  "the cat sat on the mat"
# Make "sat" attend to "cat", second "the" to nearby nouns, etc.
# -------------------------------------------------------
TOKENS = ["the", "cat", "sat", "on", "the", "mat"]
T = len(TOKENS)
D = 8  # model dim

# Base random embeddings, then nudge so related words align in a subspace
X = RNG.normal(0, 0.3, size=(T, D))
# Semantic knobs in dims 0..2
# subject-ish
X[1, 0:3] = np.array([1.5, 0.2, -0.1])   # cat
X[2, 0:3] = np.array([1.2, 0.9, 0.0])    # sat  (verb looking for subject)
X[5, 0:3] = np.array([0.2, -0.2, 1.4])   # mat  (object-ish)
X[3, 0:3] = np.array([0.1, 0.1, 0.8])    # on   (prep looking for object)
# article slots
X[0, 0:3] = np.array([0.4, 0.0, 0.2])    # the (1)
X[4, 0:3] = np.array([0.3, -0.1, 0.9])   # the (2) near object

# Simple projection matrices (identity-ish + small noise) so Q/K preserve cues
Wq = np.eye(D) + 0.1 * RNG.normal(size=(D, D))
Wk = np.eye(D) + 0.1 * RNG.normal(size=(D, D))
Wv = np.eye(D) + 0.05 * RNG.normal(size=(D, D))

out, attn, Q, K, V, scores = self_attention(X, Wq, Wk, Wv, causal=False)
out_c, attn_c, *_ = self_attention(X, Wq, Wk, Wv, causal=True)


# -------------------------------------------------------
# FIGURE 1: Big picture -- tokens talk to tokens
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    'Step 1: Self-attention = every token can look at every token',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')

# draw tokens as boxes
box_x = np.linspace(1.0, 9.0, T)
box_y = 4.5
for i, (tok, bx) in enumerate(zip(TOKENS, box_x)):
    rect = patches.FancyBboxPatch(
        (bx - 0.55, box_y - 0.35), 1.1, 0.7,
        boxstyle='round,pad=0.02,rounding_size=0.15',
        facecolor='#dceaff', edgecolor='royalblue', linewidth=2,
    )
    ax1.add_patch(rect)
    ax1.text(bx, box_y, tok, ha='center', va='center', fontsize=FS_AXIS, fontweight='bold')

# highlight attention from "sat" (index 2) using real weights
src = 2
for j in range(T):
    w = float(attn[src, j])
    if w < 0.05:
        continue
    ax1.annotate(
        '',
        xy=(box_x[j], box_y - 0.4),
        xytext=(box_x[src], box_y - 0.4),
        arrowprops=dict(
            arrowstyle='->', color='crimson',
            lw=1.0 + 6.0 * w, alpha=0.35 + 0.65 * w,
            connectionstyle='arc3,rad=0.15',
        ),
    )

ax1.text(
    5, 2.6,
    'Example: token \"sat\" queries others.\n'
    'Arrow thickness ~ attention weight.\n'
    'It can jump directly to \"cat\" (subject) in ONE layer.',
    ha='center', va='center', fontsize=FS_AXIS - 2,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax1.text(
    5, 1.1,
    'RNN: information crawls step by step along the sequence\n'
    'Transformer attention: any pair interacts immediately (path length 1)',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Q, K, V roles
# -------------------------------------------------------
fig2, axes2 = new_figure(1, 3)
fig2.suptitle(
    'Step 2: Queries ask, Keys answer, Values deliver content',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# Show first 3 dims of Q/K as bar signatures per token
dims_show = 3
panels = [
    (Q[:, :dims_show], 'Queries Q = X W_Q', 'What am I looking for?', '#f8d7da'),
    (K[:, :dims_show], 'Keys K = X W_K', 'What do I contain?', '#d1ecf1'),
    (V[:, :dims_show], 'Values V = X W_V', 'What do I pass along?', '#d4edda'),
]

x_idx = np.arange(T)
width = 0.25
for ax, (mat, title, subtitle, face) in zip(axes2, panels):
    for d in range(dims_show):
        ax.bar(x_idx + (d - 1) * width, mat[:, d], width=width,
               label=f'dim {d}', alpha=0.9)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(TOKENS, rotation=45, ha='right')
    ax.set_title(title, fontsize=FS_AX_TITLE - 2)
    ax.set_ylabel('value', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK - 2)
    ax.grid(True, axis='y', alpha=0.3)
    ax.text(
        0.5, 0.98, subtitle,
        transform=ax.transAxes, ha='center', va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor=face, alpha=0.95),
    )
    if ax is axes2[0]:
        ax.legend(fontsize=FS_LEGEND - 1, loc='lower left')

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Score matrix -> softmax attention map
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    r'Step 3: Attention weights = softmax$(QK^\top / \sqrt{d_k})$',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes3[0]
im = ax.imshow(scores, cmap='magma')
ax.set_xticks(range(T))
ax.set_yticks(range(T))
ax.set_xticklabels(TOKENS, rotation=45, ha='right')
ax.set_yticklabels(TOKENS)
ax.set_xlabel('key token', fontsize=FS_AXIS)
ax.set_ylabel('query token', fontsize=FS_AXIS)
ax.set_title('Raw scores  QKᵀ/√d', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK - 2)
fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
add_note(
    ax,
    'Higher score => better\n'
    'match between this\n'
    'query and that key.',
    y=0.02, va='bottom', facecolor='lightyellow',
)

ax = axes3[1]
im = ax.imshow(attn, cmap='viridis', vmin=0, vmax=1)
ax.set_xticks(range(T))
ax.set_yticks(range(T))
ax.set_xticklabels(TOKENS, rotation=45, ha='right')
ax.set_yticklabels(TOKENS)
ax.set_xlabel('key / value token', fontsize=FS_AXIS)
ax.set_ylabel('query token', fontsize=FS_AXIS)
ax.set_title('Softmax attention weights', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK - 2)
for i in range(T):
    for j in range(T):
        ax.text(j, i, f'{attn[i, j]:.2f}', ha='center', va='center',
                color='white' if attn[i, j] > 0.35 else 'black', fontsize=9)
fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
add_note(
    ax,
    'Each ROW sums to 1.\n'
    'Output token i =\n'
    'sum_j  A_ij * V_j',
    y=0.02, va='bottom',
)

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Causal mask (decoder) vs full attention (encoder)
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    'Step 4: Encoder is bidirectional; decoder is causally masked',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes4[0]
ax.imshow(attn, cmap='viridis', vmin=0, vmax=1)
ax.set_title('Encoder-style (bidirectional)', fontsize=FS_AX_TITLE)
ax.set_xticks(range(T))
ax.set_yticks(range(T))
ax.set_xticklabels(TOKENS, rotation=45, ha='right')
ax.set_yticklabels(TOKENS)
ax.tick_params(axis='both', labelsize=FS_TICK - 2)
ax.set_xlabel('key', fontsize=FS_AXIS)
ax.set_ylabel('query', fontsize=FS_AXIS)
add_note(
    ax,
    'Every token sees the\n'
    'FULL input context.\n'
    'Job: build a rich\n'
    'representation of\n'
    'the source.',
    x=0.02, y=0.98, ha='left', va='top', facecolor='lightyellow',
)

ax = axes4[1]
ax.imshow(attn_c, cmap='viridis', vmin=0, vmax=1)
# outline forbidden upper triangle
for i in range(T):
    for j in range(i + 1, T):
        ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='red', lw=1.5))
ax.set_title('Decoder-style (causal mask)', fontsize=FS_AX_TITLE)
ax.set_xticks(range(T))
ax.set_yticks(range(T))
ax.set_xticklabels(TOKENS, rotation=45, ha='right')
ax.set_yticklabels(TOKENS)
ax.tick_params(axis='both', labelsize=FS_TICK - 2)
ax.set_xlabel('key', fontsize=FS_AXIS)
ax.set_ylabel('query', fontsize=FS_AXIS)
add_note(
    ax,
    'Cannot peek at future\n'
    'output tokens (red).\n'
    'Job: GENERATE the\n'
    'next token / target\n'
    'sequence step by step.',
    x=0.02, y=0.98, ha='left', va='top', facecolor='lightyellow',
)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 4b: Division of labor -- encode then generate
# -------------------------------------------------------
fig4b, ax4b = new_figure()
fig4b.suptitle(
    'Step 4b: Encoder represents the input; decoder acts on that representation',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax4b.set_xlim(0, 12)
ax4b.set_ylim(0, 8)
ax4b.axis('off')


def _box(ax, x, y, w, h, title, body, color):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
        facecolor=color, edgecolor='k', linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.35, title, ha='center', va='top',
            fontsize=FS_ANNOT + 2, fontweight='bold')
    ax.text(x + w / 2, y + h / 2 - 0.2, body, ha='center', va='center',
            fontsize=FS_ANNOT)


_box(
    ax4b, 0.5, 2.5, 4.2, 4.5,
    'ENCODER',
    'Reads the full input\n'
    '(bidirectional attention).\n\n'
    'Learns an optimal\n'
    'contextual representation\n'
    'of the source sequence:\n'
    'who relates to whom,\n'
    'what matters in context.',
    '#d1ecf1',
)
_box(
    ax4b, 7.3, 2.5, 4.2, 4.5,
    'DECODER',
    'Conditioned on that\n'
    'encoder representation\n'
    '(cross-attention).\n\n'
    'Learns to PRODUCE the\n'
    'output: next tokens /\n'
    'translation / answer /\n'
    'whatever the task needs,\n'
    'one step at a time.',
    '#f8d7da',
)

ax4b.annotate(
    '', xy=(7.2, 4.7), xytext=(4.9, 4.7),
    arrowprops=dict(arrowstyle='->', lw=3.5, color='seagreen'),
)
ax4b.text(
    6.05, 5.3,
    'cross-attention\n(use the representation)',
    ha='center', va='bottom', fontsize=FS_ANNOT, color='seagreen',
    fontweight='bold',
)

ax4b.text(
    6.0, 1.1,
    'Classic seq2seq Transformer:\n'
    'Encoder = understand / compress the input into useful features.\n'
    'Decoder = generate the target sequence from those features\n'
    '(translation, summarization, captioning, ...).\n\n'
    'Decoder-only models (GPT) fold "understanding + generation" into one stack,\n'
    'still with causal masking so they can predict the next token.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig4b)


# -------------------------------------------------------
# FIGURE 5: Block diagram of a Transformer layer
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    'Step 5: One Transformer block = Attention + FFN + residuals + LayerNorm',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax5.set_xlim(0, 12)
ax5.set_ylim(0, 8)
ax5.axis('off')


def block(ax, x, y, w, h, text, color):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor=color, edgecolor='k', linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=FS_ANNOT + 1, fontweight='bold')


# main stack -- original Vaswani post-norm layout
block(ax5, 4.0, 6.6, 4.0, 0.8, 'Input tokens X', '#dceaff')
block(ax5, 4.0, 5.2, 4.0, 0.9, 'Multi-Head\nSelf-Attention', '#f8d7da')
block(ax5, 4.0, 4.0, 4.0, 0.7, 'Add & LayerNorm', '#fff3cd')
block(ax5, 4.0, 2.7, 4.0, 0.9, 'Feed-Forward\nMLP (non-linear)', '#d4edda')
block(ax5, 4.0, 1.5, 4.0, 0.7, 'Add & LayerNorm', '#fff3cd')
block(ax5, 4.0, 0.3, 4.0, 0.8, 'Block output', '#dceaff')

# arrows
for y0, y1 in [(6.6, 6.1), (5.2, 4.7), (4.0, 3.6), (2.7, 2.2), (1.5, 1.1)]:
    ax5.annotate('', xy=(6.0, y1), xytext=(6.0, y0),
                 arrowprops=dict(arrowstyle='->', lw=2, color='k'))

# residual side notes
ax5.annotate('residual', xy=(8.1, 4.35), xytext=(9.3, 5.4),
             fontsize=FS_ANNOT, color='crimson',
             arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5))
ax5.annotate('residual', xy=(8.1, 1.85), xytext=(9.3, 2.9),
             fontsize=FS_ANNOT, color='crimson',
             arrowprops=dict(arrowstyle='->', color='crimson', lw=1.5))

ax5.text(
    1.2, 4.0,
    'Attention:\n'
    '  route information\n'
    '  between tokens\n\n'
    'FFN:\n'
    '  think per token\n'
    '  (non-linear MLP)\n\n'
    'Diagram: original\n'
    '  post-norm (Vaswani).\n'
    '151_transformer_code.py\n'
    '  uses Pre-LN for stability\n'
    '  (LN before Attn/FFN).\n'
    'See also 120_normalization.py',
    ha='left', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Why /sqrt(d) scaling matters
# -------------------------------------------------------
fig6, axes6 = new_figure(1, 2)
fig6.suptitle(
    r'Step 6: Scale by $1/\sqrt{d_k}$ keeps softmax out of saturation',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

d_list = [1, 8, 64, 256]
ax = axes6[0]
for d in d_list:
    # random Q,K with unit-ish coords; scores without scaling
    q = RNG.normal(size=(200, d))
    k = RNG.normal(size=(200, d))
    raw = np.sum(q * k, axis=1)  # diagonal-ish sample of scores
    ax.hist(raw, bins=30, density=True, alpha=0.45, label=rf'$d={d}$ (no scale)')
ax.set_title('Unscaled $q\\cdot k$ spreads with $d$', fontsize=FS_AX_TITLE)
ax.set_xlabel('score', fontsize=FS_AXIS)
ax.set_ylabel('density', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)

ax = axes6[1]
for d in d_list:
    q = RNG.normal(size=(200, d))
    k = RNG.normal(size=(200, d))
    scaled = np.sum(q * k, axis=1) / np.sqrt(d)
    ax.hist(scaled, bins=30, density=True, alpha=0.45, label=rf'$d={d}$ (scaled)')
ax.set_title(r'Scaled $q\cdot k / \sqrt{d}$ stays stable', fontsize=FS_AX_TITLE)
ax.set_xlabel('score', fontsize=FS_AXIS)
ax.set_ylabel('density', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'If scores are huge,\n'
    'softmax becomes a\n'
    'near one-hot => tiny\n'
    'gradients. Scaling\n'
    'prevents that.',
    y=0.98, va='top', facecolor='lightyellow',
)

plt.tight_layout()
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Wide adoption across deep learning
# -------------------------------------------------------
fig7, axes7 = new_figure(1, 2)
fig7.suptitle(
    'Step 7: Why Transformers became the default stack in deep learning',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes7[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Where they spread', fontsize=FS_AX_TITLE)

domains = [
    (0.5, 7.2, 4.2, 2.2, 'NLP\nGPT / BERT / T5 / ...', '#dceaff'),
    (5.3, 7.2, 4.2, 2.2, 'Vision\nViT / DETR / ...', '#f8d7da'),
    (0.5, 4.4, 4.2, 2.2, 'Speech & audio\nWhisper-style / ...', '#d4edda'),
    (5.3, 4.4, 4.2, 2.2, 'Multimodal\nCLIP / Flamingo / ...', '#fff3cd'),
    (0.5, 1.6, 4.2, 2.2, 'Video\nTimeSformer / ViViT / ...', '#e2d5f1'),
    (5.3, 1.6, 4.2, 2.2, 'Agents / tools / code\nLLM ecosystems', '#ffe0b2'),
]
for x, y, w, h, text, color in domains:
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
        facecolor=color, edgecolor='k', lw=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=FS_ANNOT + 1, fontweight='bold')

ax = axes7[1]
ax.axis('off')
ax.set_title('Why adoption exploded', fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.55,
    '1) Parallelism\n'
    '   No RNN time-step loop => trains fast on GPUs/TPUs.\n\n'
    '2) Long-range dependency\n'
    '   Path length 1 between any tokens (via attention).\n\n'
    '3) One architecture, many modalities\n'
    '   Anything tokenizable (text, patches, audio frames,\n'
    '   video tubelets) can share the same recipe.\n\n'
    '4) Scaling laws + pretraining\n'
    '   Bigger models + more data kept paying off,\n'
    '   creating a shared foundation-model ecosystem.\n\n'
    '5) Transfer & tooling\n'
    '   Checkpoints, libraries, and fine-tuning workflows\n'
    '   made Transformers the path of least resistance.\n\n'
    'Caveat: attention is O(T^2), so long context still needs\n'
    'engineering (efficient attention, sparse/linear variants).',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Recent improvements + diversified usage
# -------------------------------------------------------
fig8, axes8 = new_figure(1, 2)
fig8.suptitle(
    'Step 8: Recent improvements -- and how Transformer usage diversified',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes8[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis('off')
ax.set_title('Recent improvements (same core, better engineering)', fontsize=FS_AX_TITLE - 2)

improvements = [
    (0.4, 8.7, 9.2, 1.8, 'Long context & efficient attention',
     'FlashAttention, KV cache, sliding / sparse / linear attention\n'
     'RoPE / ALiBi for length transfer  (see 160_positional_encoding.py)',
     '#dceaff'),
    (0.4, 6.5, 9.2, 1.8, 'Scale without one giant dense FFN',
     'Mixture-of-Experts (MoE): route tokens to a few experts\n'
     'More capacity at similar active FLOPs',
     '#f8d7da'),
    (0.4, 4.3, 9.2, 1.8, 'Training stability & recipes',
     'Pre-LN / RMSNorm, better init, mixed precision, data curation\n'
     'Instruction tuning + preference / RLHF-style alignment',
     '#d4edda'),
    (0.4, 2.1, 9.2, 1.8, 'Systems around the stack',
     'RAG / tools / agents, speculative decoding, quantization\n'
     'Distillation and small on-device cousins of big models',
     '#fff3cd'),
]
for x, y, w, h, title, body, color in improvements:
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
        facecolor=color, edgecolor='k', lw=2,
    ))
    ax.text(x + 0.25, y + h - 0.35, title, ha='left', va='top',
            fontsize=FS_ANNOT + 1, fontweight='bold')
    ax.text(x + 0.25, y + 0.35, body, ha='left', va='bottom',
            fontsize=FS_ANNOT)

ax.text(
    5.0, 0.7,
    'Core recipe unchanged: tokenize -> attend -> FFN (+ residual/norm).\n'
    'Most "progress" is making that recipe longer, cheaper, stabler, usable.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

ax = axes8[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis('off')
ax.set_title('Diversification of usage', fontsize=FS_AX_TITLE - 2)

usages = [
    (0.4, 8.7, 4.4, 1.8, 'Language & code\nchat, copilots, agents', '#dceaff'),
    (5.2, 8.7, 4.4, 1.8, 'Vision & video\nViT, DETR, ViViT, Swin', '#f8d7da'),
    (0.4, 6.5, 4.4, 1.8, 'Speech & audio\nASR, TTS, music', '#d4edda'),
    (5.2, 6.5, 4.4, 1.8, 'Multimodal\nimage-text, any-to-any', '#fff3cd'),
    (0.4, 4.3, 4.4, 1.8, 'Science\nproteins, molecules, math', '#e2d5f1'),
    (5.2, 4.3, 4.4, 1.8, 'Time series &\nrecommendation / tabular', '#ffe0b2'),
    (0.4, 2.1, 4.4, 1.8, 'Robotics & control\npolicies as sequences', '#cce5ff'),
    (5.2, 2.1, 4.4, 1.8, 'Retrieval & search\nembedding + rerank stacks', '#f5d0e6'),
]
for x, y, w, h, text, color in usages:
    ax.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
        facecolor=color, edgecolor='k', lw=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=FS_ANNOT, fontweight='bold')

ax.text(
    5.0, 0.7,
    'Anything that can be turned into a sequence of tokens / patches / frames\n'
    'can reuse the Transformer inductive bias -- often with small domain tweaks\n'
    '(see 190_*, 200_video_transformer.py, 210_swin_transformer.py).',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig8)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
row_sums = attn.sum(axis=1)
print("\n" + "=" * 60)
print("TRANSFORMER INTUITION")
print("=" * 60)
print(f"""
Self-attention in one line:
    Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V

Roles:
  Q  query   -- what this token is looking for
  K  key     -- what each token advertises
  V  value   -- payload mixed into the output

Demo sentence: {" ".join(TOKENS)}
  Attention rows sum to 1: {np.round(row_sums, 3)}
  \"sat\" attention row:
    {list(zip(TOKENS, np.round(attn[2], 3)))}

Why Transformers scaled:
  1) Global interactions in one layer (no long RNN chain)
  2) Highly parallel on GPUs/TPUs (no time-step loop)
  3) Residual + LayerNorm enable deep stacks
  4) Multi-head attention = several specialized routing patterns

Encoder vs Decoder (division of labor):
  Encoder (bidirectional):
    Learn an optimal contextual REPRESENTATION of the input --
    every token can see the whole source and refine what it means.

  Decoder (causal + often cross-attention):
    Based on that representation, learn to DO THE TASK OUTPUT --
    typically GENERATE the target sequence token by token
    (translate, summarize, answer, caption, ...).
    Causal masking prevents cheating by looking at future outputs.

  Decoder-only (GPT-style): one stack does both understanding and
  next-token generation under a causal mask.

Wide adoption in deep learning:
  Started in machine translation (2017), then became the default for
  language models, and spread to vision (ViT), speech, multimodal,
  video, and tool-using LLM systems. The same "token + attend + FFN"
  recipe transfers across modalities, scales well with data/compute,
  and benefits from a huge shared ecosystem of pretrained models.

Recent improvements (engineering on the same core):
  1) Long context / efficiency: FlashAttention, KV cache, sparse or
     sliding-window attention; RoPE/ALiBi for length transfer
  2) Capacity scaling: Mixture-of-Experts (MoE) -- more parameters,
     similar active compute
  3) Stable recipes: Pre-LN / RMSNorm, better data, instruction tuning,
     preference / RLHF-style alignment
  4) Serving & systems: RAG, tools/agents, speculative decoding,
     quantization, distillation

Diversification of usage:
  Beyond MT/NLP -- vision, video, speech, multimodal, code copilots,
  science (proteins/molecules), time series & tabular, robotics,
  retrieval/search. Domain tweaks (patching, windows, masks) sit on
  top of the same attend+FFN inductive bias
  (see 190_Transformer_inductive_bias.py, 200_video_transformer.py,
   210_swin_transformer.py).

What the FFN adds:
  Attention routes information across tokens.
  The feed-forward MLP applies non-linear processing per token
  (see 140_neural_non_linearity.py).
""")
