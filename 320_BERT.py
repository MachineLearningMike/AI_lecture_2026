"""
320_BERT.py

BERT (Bidirectional Encoder Representations from Transformers)
and its training techniques -- plus the post-BERT survey.

BERT (Devlin et al., 2018) showed that an encoder-only Transformer
pretrained with:
  1) Masked Language Modeling (MLM)
  2) Next Sentence Prediction (NSP)
then fine-tuned on downstream tasks, beat heavily engineered NLP
pipelines.

Later work kept the encoder idea but revised the recipe:
  - RoBERTa: drop NSP, dynamic masking, more data / longer training
  - ALBERT: parameter sharing + sentence-order objective
  - ELECTRA: replaced-token detection (more compute-efficient)

Companion to 190_transformer.py (encoder vs decoder) and
260_positional_encoding.py (learned absolute PE in original BERT).
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
    return e / e.sum(axis=axis, keepdims=True)


def box(ax, x, y, w, h, text, color, fontsize=None):
    fontsize = FS_ANNOT if fontsize is None else fontsize
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor=color, edgecolor='k', lw=2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold')


# -------------------------------------------------------
# Toy sentence for MLM / attention demos
# -------------------------------------------------------
TOKENS = ['[CLS]', 'the', 'cat', 'sat', 'on', 'the', 'mat', '[SEP]']
T = len(TOKENS)
# Mask position index for "cat"
MASK_IDX = 2
MASKED = TOKENS.copy()
MASKED[MASK_IDX] = '[MASK]'

# Bidirectional vs causal attention toy maps
attn_bi = softmax(RNG.normal(0, 0.8, size=(T, T)), axis=-1)
attn_causal = attn_bi.copy()
for i in range(T):
    attn_causal[i, i + 1:] = 0.0
    row = attn_causal[i].sum()
    if row > 0:
        attn_causal[i] /= row


# -------------------------------------------------------
# FIGURE 1: What BERT is
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    'Step 1: BERT = bidirectional encoder Transformer + pretrain/fine-tune',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes1[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Where BERT sits in the Transformer family', fontsize=FS_AX_TITLE - 2)

box(ax, 0.5, 5.5, 4.0, 2.0, 'Encoder-only\nBERT / RoBERTa', '#d1ecf1')
box(ax, 5.5, 5.5, 4.0, 2.0, 'Decoder-only\nGPT family', '#f8d7da')
box(ax, 2.5, 2.8, 5.0, 2.0, 'Encoder-Decoder\nT5 / original MT Transformer', '#d4edda')

ax.text(
    5.0, 1.2,
    'BERT: every token may attend to EVERY other token\n'
    '(left and right context) -- great for understanding,\n'
    'not for open-ended left-to-right generation.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes1[1]
ax.axis('off')
ax.set_title('The BERT pitch (2018)', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    '1) Use a deep bidirectional Transformer encoder.\n\n'
    '2) Pretrain on huge unlabeled text with self-supervised\n'
    '   tasks (MLM + NSP) -- no hand labels needed.\n\n'
    '3) Fine-tune ONE model on many labeled NLP tasks\n'
    '   (classification, QA, NER, ...) with small heads.\n\n'
    'Result: transfer learning became the default in NLP.\n'
    'Same idea later powered GPT-style generative LMs,\n'
    'but BERT itself is an UNDERSTANDING model.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Input representation
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    'Step 2: BERT input = TokenEmbed + SegmentEmbed + PositionEmbed',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 9)
ax2.axis('off')

# Example: sentence A + sentence B packing
sent_a = ['[CLS]', 'the', 'cat', 'sat', '[SEP]']
sent_b = ['on', 'the', 'mat', '[SEP]']
seq = sent_a + sent_b
segs = [0] * len(sent_a) + [1] * len(sent_b)
n = len(seq)
xs = np.linspace(1.0, 13.0, n)

# token row
for i, (x, tok) in enumerate(zip(xs, seq)):
    color = '#dceaff' if segs[i] == 0 else '#ffe0b2'
    if tok in ('[CLS]', '[SEP]'):
        color = '#e2d5f1'
    box(ax2, x - 0.55, 6.8, 1.1, 0.9, tok, color, fontsize=FS_ANNOT - 1)

ax2.text(0.3, 7.25, 'tokens', ha='left', va='center', fontsize=FS_ANNOT, fontweight='bold')

# segment row
for i, (x, s) in enumerate(zip(xs, segs)):
    color = '#d1ecf1' if s == 0 else '#fff3cd'
    box(ax2, x - 0.55, 5.2, 1.1, 0.9, f'E_A' if s == 0 else 'E_B',
        color, fontsize=FS_ANNOT - 1)
ax2.text(0.3, 5.65, 'segment', ha='left', va='center', fontsize=FS_ANNOT, fontweight='bold')

# position row
for i, x in enumerate(xs):
    box(ax2, x - 0.55, 3.6, 1.1, 0.9, f'P{i}', '#d4edda', fontsize=FS_ANNOT - 1)
ax2.text(0.3, 4.05, 'position', ha='left', va='center', fontsize=FS_ANNOT, fontweight='bold')

# sum arrow
ax2.annotate('', xy=(7.0, 2.6), xytext=(7.0, 3.5),
             arrowprops=dict(arrowstyle='->', lw=2.5))
box(ax2, 4.5, 1.2, 5.0, 1.2, 'sum -> Transformer encoder', '#f8d7da')

ax2.text(
    7.0, 0.45,
    '[CLS] pools a sequence-level vector for classification.\n'
    '[SEP] marks sentence boundaries. Segment ids tell A from B.\n'
    'Original BERT used LEARNED absolute position embeddings (not sinusoidal).',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Bidirectional vs causal
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    'Step 3: Bidirectional context is the point of BERT (vs GPT causal)',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes3[0]
im = ax.imshow(attn_bi, cmap='viridis', vmin=0, vmax=attn_bi.max())
ax.set_xticks(range(T))
ax.set_yticks(range(T))
ax.set_xticklabels(TOKENS, rotation=45, ha='right')
ax.set_yticklabels(TOKENS)
ax.set_xlabel('key', fontsize=FS_AXIS)
ax.set_ylabel('query', fontsize=FS_AXIS)
ax.set_title('BERT encoder: full bidirectional attention', fontsize=FS_AX_TITLE - 2)
ax.tick_params(axis='both', labelsize=FS_TICK - 4)
fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
add_note(
    ax,
    '"sat" may look at\n'
    'BOTH "cat" (left)\n'
    'and "mat" (right).\n\n'
    'Deep bidirectional\n'
    'context fusion.',
    x=0.02, y=0.98, ha='left', va='top', facecolor='lightyellow',
)

ax = axes3[1]
im = ax.imshow(attn_causal, cmap='viridis', vmin=0, vmax=attn_bi.max())
for i in range(T):
    for j in range(i + 1, T):
        ax.add_patch(patches.Rectangle(
            (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', lw=1.2,
        ))
ax.set_xticks(range(T))
ax.set_yticks(range(T))
ax.set_xticklabels(TOKENS, rotation=45, ha='right')
ax.set_yticklabels(TOKENS)
ax.set_xlabel('key', fontsize=FS_AXIS)
ax.set_ylabel('query', fontsize=FS_AXIS)
ax.set_title('GPT-style: causal (no future)', fontsize=FS_AX_TITLE - 2)
ax.tick_params(axis='both', labelsize=FS_TICK - 4)
fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
add_note(
    ax,
    'Future tokens blocked.\n'
    'Great for generation;\n'
    'weaker for tasks that\n'
    'need full-sentence\n'
    'understanding in one go.',
    x=0.02, y=0.98, ha='left', va='top', facecolor='lightyellow',
)

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Masked Language Modeling (MLM)
# -------------------------------------------------------
# BERT 80/10/10 masking recipe among the 15% selected tokens
fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    'Step 4: Pretraining task 1 -- Masked Language Modeling (MLM)',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes4[0]
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Cloze-style prediction with [MASK]', fontsize=FS_AX_TITLE - 2)

# original vs masked sequence
xs = np.linspace(1.2, 10.8, T)
for i, (x, tok) in enumerate(zip(xs, TOKENS)):
    box(ax, x - 0.55, 6.2, 1.1, 0.85, tok, '#dceaff', fontsize=FS_ANNOT - 2)
for i, (x, tok) in enumerate(zip(xs, MASKED)):
    color = '#f8d7da' if tok == '[MASK]' else '#d4edda'
    box(ax, x - 0.55, 4.4, 1.1, 0.85, tok, color, fontsize=FS_ANNOT - 2)

ax.annotate('', xy=(xs[MASK_IDX], 5.3), xytext=(xs[MASK_IDX], 6.2),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='crimson'))
ax.text(xs[MASK_IDX] + 0.7, 5.7, 'mask ~15%', color='crimson',
        fontsize=FS_ANNOT, fontweight='bold')

box(ax, 3.5, 2.2, 5.0, 1.4, 'predict original token\nP("cat" | context)', '#fff3cd')
ax.annotate('', xy=(6.0, 3.6), xytext=(xs[MASK_IDX], 4.4),
            arrowprops=dict(arrowstyle='->', lw=2.2, color='seagreen'))

ax.text(
    6.0, 0.7,
    'Loss only on masked positions (not the whole sentence).\n'
    'Forces the encoder to use LEFT + RIGHT context to fill gaps.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

ax = axes4[1]
# 80/10/10 pie among selected tokens
sizes = [80, 10, 10]
labels = ['80% -> [MASK]', '10% random token', '10% keep original']
colors = ['#f8d7da', '#fff3cd', '#d4edda']
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%',
       textprops={'fontsize': FS_ANNOT}, startangle=90)
ax.set_title('Among the ~15% chosen tokens', fontsize=FS_AX_TITLE - 2)
add_note(
    ax,
    'Why not always [MASK]?\n'
    'Fine-tuning has no\n'
    '[MASK] token. Mixing\n'
    'random/keep reduces\n'
    'train/test mismatch.\n\n'
    'Still: most signal\n'
    'comes from [MASK].',
    x=0.5, y=-0.05, ha='center', va='top', facecolor='lightyellow',
)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Next Sentence Prediction (NSP)
# -------------------------------------------------------
fig5, axes5 = new_figure(1, 2)
fig5.suptitle(
    'Step 5: Pretraining task 2 -- Next Sentence Prediction (NSP)',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes5[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Is sentence B a true next sentence after A?', fontsize=FS_AX_TITLE - 2)

box(ax, 0.5, 5.5, 4.0, 1.8, 'Sentence A\n"the cat sat"', '#dceaff')
box(ax, 5.5, 5.5, 4.0, 1.8, 'Sentence B\n"on the mat"', '#d4edda')
box(ax, 2.5, 3.2, 5.0, 1.4, '[CLS] -> IsNext / NotNext', '#fff3cd')
ax.annotate('', xy=(5.0, 4.6), xytext=(2.5, 5.5),
            arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(5.0, 4.6), xytext=(7.5, 5.5),
            arrowprops=dict(arrowstyle='->', lw=2))

ax.text(
    5.0, 1.5,
    'Training mix (~50/50):\n'
    '  IsNext  -- B really follows A in the corpus\n'
    '  NotNext -- B is a random sentence\n\n'
    'Goal: teach inter-sentence / discourse relations\n'
    '(useful for QA, NLI, sentence-pair tasks).',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes5[1]
# toy NSP decision margin
ax.bar(['IsNext', 'NotNext'], [0.82, 0.18], color=['seagreen', 'crimson'],
       edgecolor='k')
ax.set_ylim(0, 1.05)
ax.set_ylabel('P from [CLS] head (toy)', fontsize=FS_AXIS)
ax.set_title('Binary classifier on [CLS]', fontsize=FS_AX_TITLE - 2)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, axis='y', alpha=0.3)
add_note(
    ax,
    'Later work (RoBERTa)\n'
    'found NSP was a WEAK\n'
    'signal vs more MLM\n'
    'compute -- see Step 8.',
    y=0.98, va='top', facecolor='lightyellow',
)

plt.tight_layout()
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Pretrain then fine-tune
# -------------------------------------------------------
fig6, ax6 = new_figure()
fig6.suptitle(
    'Step 6: Recipe -- expensive pretrain once, cheap fine-tune many times',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax6.set_xlim(0, 14)
ax6.set_ylim(0, 9)
ax6.axis('off')

box(ax6, 0.5, 5.5, 4.0, 2.5,
    'PRETRAIN\n(unlabeled text)\nMLM + NSP', '#d1ecf1', fontsize=FS_ANNOT + 1)
box(ax6, 5.0, 5.5, 4.0, 2.5,
    'SHARED\nencoder\nweights', '#fff3cd', fontsize=FS_ANNOT + 1)
box(ax6, 9.5, 6.8, 4.0, 1.4, 'Fine-tune: sentiment', '#d4edda')
box(ax6, 9.5, 5.0, 4.0, 1.4, 'Fine-tune: QA / NER', '#f8d7da')
box(ax6, 9.5, 3.2, 4.0, 1.4, 'Fine-tune: NLI / ...', '#ffe0b2')

ax6.annotate('', xy=(5.0, 6.7), xytext=(4.5, 6.7),
             arrowprops=dict(arrowstyle='->', lw=3, color='seagreen'))
ax6.annotate('', xy=(9.5, 7.5), xytext=(9.0, 6.9),
             arrowprops=dict(arrowstyle='->', lw=2))
ax6.annotate('', xy=(9.5, 5.7), xytext=(9.0, 6.7),
             arrowprops=dict(arrowstyle='->', lw=2))
ax6.annotate('', xy=(9.5, 3.9), xytext=(9.0, 6.5),
             arrowprops=dict(arrowstyle='->', lw=2))

ax6.text(
    7.0, 1.3,
    'Fine-tuning typically:\n'
    '  • keep the pretrained encoder (sometimes freeze early layers)\n'
    '  • add a small task head on [CLS] (classification) or on tokens (NER/QA span)\n'
    '  • train on labeled data with a much smaller learning rate / few epochs\n\n'
    'This is the transfer-learning template modern NLP still uses.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Why MLM enables deep bidirectionality
# -------------------------------------------------------
fig7, axes7 = new_figure(1, 2)
fig7.suptitle(
    'Step 7: Why not just train a bidirectional LM the naive way?',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes7[0]
ax.axis('off')
ax.set_title('Naive bidirectional LM would cheat', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    'If every position predicts its OWN token while seeing\n'
    'the full unmasked sentence, the model can trivially\n'
    'copy the token from the input embedding -- no need\n'
    'to learn context.\n\n'
    'Left-to-right LM (GPT) avoids this by causal masking,\n'
    'but then never sees future context.\n\n'
    'MLM solves the dilemma: HIDE some tokens, predict them\n'
    'from the remaining bidirectional context. The answer is\n'
    'not visible in the input, so the model must reason.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.96),
)

ax = axes7[1]
# schematic: information must flow from context into [MASK]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Information must flow into [MASK]', fontsize=FS_AX_TITLE - 2)

ctx = ['the', '[MASK]', 'sat', 'mat']
cols = ['#dceaff', '#f8d7da', '#dceaff', '#dceaff']
xpos = [1.5, 4.0, 6.5, 9.0]
for x, tok, c in zip(xpos, ctx, cols):
    box(ax, x - 0.7, 5.5, 1.4, 1.0, tok, c)
# arrows into mask
for x in [1.5, 6.5, 9.0]:
    ax.annotate('', xy=(4.0, 5.5), xytext=(x, 5.5),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=2,
                                connectionstyle='arc3,rad=0.25'))
box(ax, 2.5, 2.5, 5.0, 1.6, 'softmax over vocab\nat masked position', '#fff3cd')
ax.annotate('', xy=(5.0, 4.1), xytext=(4.0, 5.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='seagreen'))
ax.text(
    5.0, 1.0,
    'Deep stack + MLM => rich contextual embeddings\n'
    'usable for many understanding tasks.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: RoBERTa -- refining the BERT recipe
# -------------------------------------------------------
fig8, axes8 = new_figure(1, 2)
fig8.suptitle(
    'Step 8: RoBERTa -- same architecture, stronger training recipe',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes8[0]
ax.axis('off')
ax.set_title('What RoBERTa changed', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    '1) DROP NSP\n'
    '   Sentence-pair packing + NSP did not help much;\n'
    '   full-sentences + more MLM worked better.\n\n'
    '2) DYNAMIC masking\n'
    '   Remask each sequence every time it is seen\n'
    '   (BERT masked once offline / statically).\n\n'
    '3) Bigger data, larger batches, longer training\n'
    '   Optimize the ORIGINAL BERT design harder.\n\n'
    '4) Remove BERT engineering quirks\n'
    '   e.g. train with longer sequences more consistently.\n\n'
    'Takeaway: a large fraction of "BERT gains" were from\n'
    'the RECIPE, not a magical new architecture.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.96),
)

ax = axes8[1]
# static vs dynamic masking schematic
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Static vs dynamic masking', fontsize=FS_AX_TITLE - 2)

box(ax, 0.5, 5.2, 4.0, 2.0, 'BERT static\nsame masks every epoch', '#f8d7da')
box(ax, 5.5, 5.2, 4.0, 2.0, 'RoBERTa dynamic\nnew masks each time', '#d4edda')

# show three mask variants of same sentence
variants = [
    (1.5, 'the [MASK] sat'),
    (4.0, 'the cat [MASK]'),
    (6.5, '[MASK] cat sat'),
]
for y, text in zip([3.6, 2.4, 1.2], ['epoch 1', 'epoch 2', 'epoch 3']):
    ax.text(0.4, y + 0.35, text, fontsize=FS_ANNOT, va='center')
for i, (x, text) in enumerate(variants):
    y = [3.6, 2.4, 1.2][i]
    box(ax, 2.2, y, 6.5, 0.7, text, '#dceaff', fontsize=FS_ANNOT)

ax.text(
    5.0, 0.35,
    'Dynamic masking = more diverse cloze problems per document.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 9: ALBERT -- parameter efficiency
# -------------------------------------------------------
fig9, axes9 = new_figure(1, 2)
fig9.suptitle(
    'Step 9: ALBERT -- share parameters; replace NSP with sentence order',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes9[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Cross-layer parameter sharing', fontsize=FS_AX_TITLE - 2)

# BERT: distinct layers
for i, y in enumerate([6.2, 4.8, 3.4, 2.0]):
    box(ax, 0.8, y, 3.2, 1.0, f'BERT L{i}\nunique W', '#dceaff', fontsize=FS_ANNOT - 1)
# ALBERT: shared
for i, y in enumerate([6.2, 4.8, 3.4, 2.0]):
    box(ax, 5.8, y, 3.2, 1.0, 'ALBERT\nshared W', '#d4edda', fontsize=FS_ANNOT - 1)
ax.text(2.4, 7.4, 'BERT', ha='center', fontsize=FS_ANNOT + 1, fontweight='bold')
ax.text(7.4, 7.4, 'ALBERT', ha='center', fontsize=FS_ANNOT + 1, fontweight='bold')
ax.text(
    5.0, 0.7,
    'Also: factorized embedding (V x d_emb then project to d_model)\n'
    '=> fewer parameters for huge vocabularies.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

ax = axes9[1]
ax.axis('off')
ax.set_title('SOP instead of NSP', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    'NSP was too easy: topic mismatch often gives away\n'
    'NotNext without needing fine discourse reasoning.\n\n'
    'ALBERT Sentence Order Prediction (SOP):\n'
    '  positive -- (A, B) in correct order\n'
    '  negative -- (B, A) swapped (same sentences)\n\n'
    'Harder: must model coherence / order, not just topic.\n\n'
    'ALBERT = parameter-efficient BERT-like encoder\n'
    'with a cleaner inter-sentence objective.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig9)


# -------------------------------------------------------
# FIGURE 10: ELECTRA -- replaced token detection
# -------------------------------------------------------
fig10, axes10 = new_figure(1, 2)
fig10.suptitle(
    'Step 10: ELECTRA -- learn from EVERY token, not only the 15% masks',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes10[0]
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Generator fills masks; discriminator detects fakes', fontsize=FS_AX_TITLE - 2)

box(ax, 0.4, 5.8, 3.5, 1.5, 'Input with [MASK]', '#dceaff')
box(ax, 4.3, 5.8, 3.5, 1.5, 'Small generator\n(MLM)', '#fff3cd')
box(ax, 8.2, 5.8, 3.5, 1.5, 'Plausible fakes\nin the sequence', '#f8d7da')
ax.annotate('', xy=(4.3, 6.5), xytext=(3.9, 6.5),
            arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(8.2, 6.5), xytext=(7.8, 6.5),
            arrowprops=dict(arrowstyle='->', lw=2))

box(ax, 3.5, 3.2, 5.0, 1.6, 'Discriminator:\nreal vs replaced?', '#d4edda')
ax.annotate('', xy=(6.0, 4.8), xytext=(10.0, 5.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='crimson'))

# token-level labels
toks = ['the', 'dog*', 'sat', 'on', 'the', 'mat']
labs = ['real', 'FAKE', 'real', 'real', 'real', 'real']
xs = np.linspace(1.0, 11.0, len(toks))
for x, tok, lab in zip(xs, toks, labs):
    color = '#f8d7da' if lab == 'FAKE' else '#d4edda'
    box(ax, x - 0.7, 1.1, 1.4, 1.2, f'{tok}\n{lab}', color, fontsize=FS_ANNOT - 2)

ax = axes10[1]
ax.axis('off')
ax.set_title('Why this is more sample-efficient', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    'MLM loss touches ~15% of tokens per sequence.\n'
    'ELECTRA\'s discriminator classifies ALL tokens\n'
    '(replaced or not) => denser learning signal.\n\n'
    'Generator is typically a SMALL MLM model.\n'
    'After pretraining, keep the discriminator encoder\n'
    'and fine-tune it like BERT.\n\n'
    'Same transfer recipe; smarter pretraining objective\n'
    'for the compute budget.\n\n'
    'Theme across RoBERTa / ALBERT / ELECTRA:\n'
    '  improve the SELF-SUPERVISED TASK and recipe,\n'
    '  not only stack more raw Transformer layers.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig10)


# -------------------------------------------------------
# FIGURE 11: Family snapshot
# -------------------------------------------------------
fig11, ax11 = new_figure()
fig11.suptitle(
    'Step 11: Encoder-LM family -- what each technique optimized',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax11.set_xlim(0, 12)
ax11.set_ylim(0, 9)
ax11.axis('off')

family = [
    (0.4, 6.2, 5.4, 2.2, 'BERT',
     'MLM + NSP; learned PE; fine-tune template', '#dceaff'),
    (6.2, 6.2, 5.4, 2.2, 'RoBERTa',
     'MLM only; dynamic masks; scale the recipe', '#d1ecf1'),
    (0.4, 3.5, 5.4, 2.2, 'ALBERT',
     'Shared weights; SOP; factorized embeddings', '#d4edda'),
    (6.2, 3.5, 5.4, 2.2, 'ELECTRA',
     'Replaced-token detection; dense supervision', '#fff3cd'),
]
for x, y, w, h, title, body, color in family:
    ax11.add_patch(patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
        facecolor=color, edgecolor='k', lw=2,
    ))
    ax11.text(x + w / 2, y + h - 0.45, title, ha='center', va='top',
              fontsize=FS_ANNOT + 2, fontweight='bold')
    ax11.text(x + w / 2, y + 0.55, body, ha='center', va='bottom',
              fontsize=FS_ANNOT)

ax11.text(
    6.0, 1.5,
    'All of these are still "BERT-like": bidirectional encoders for understanding.\n'
    'Generative LLMs (GPT family) took a different fork -- causal LM + scale --\n'
    'but borrowed BERT\'s pretrain/fine-tune (and later instruct/align) culture.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig11)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
# Tiny MLM sanity: cross-entropy on a one-hot toy prediction
vocab = ['cat', 'dog', 'mat', 'sat', 'the']
true_id = vocab.index('cat')
logits = np.array([2.5, 0.2, -0.5, 0.1, 0.0])  # prefers "cat"
probs = softmax(logits)
mlm_nll = -np.log(probs[true_id] + 1e-12)

print("\n" + "=" * 60)
print("BERT AND ITS TRAINING TECHNIQUES")
print("=" * 60)
print(f"""
What BERT is:
  Bidirectional Encoder Representations from Transformers
  -- encoder-only stack, full attention both directions,
  -- pretrained with self-supervision, then fine-tuned.

Input packing:
  [CLS] tokens_A [SEP] tokens_B [SEP]
  Embedding = Token + Segment(A/B) + Position (learned absolute)

Core pretraining techniques:
  1) Masked Language Modeling (MLM)
     - Select ~15% of tokens
     - Among them: 80% [MASK], 10% random, 10% keep
     - Predict original ids ONLY at those positions
     - Enables deep bidirectionality without trivial copy-cheating
     Toy MLM NLL for correct token \"cat\": {mlm_nll:.3f}
       (probs={np.round(probs, 3)})

  2) Next Sentence Prediction (NSP)
     - [CLS] classifies IsNext vs NotNext (~50/50)
     - Aimed at discourse / sentence-pair reasoning
     - Later shown to be a relatively weak signal

Transfer recipe:
  Pretrain once on unlabeled text -> fine-tune many labeled tasks
  Heads: [CLS] classifier, span start/end (QA), per-token (NER)

Post-BERT diversification of the RECIPE (not just depth/width):
  RoBERTa  -- drop NSP; dynamic masking; more data/batch/steps
  ALBERT   -- cross-layer parameter sharing; factorized embeddings;
              SOP (order) instead of NSP
  ELECTRA  -- generator proposes replacements; discriminator detects
              fakes on ALL tokens (denser signal than 15% MLM)

Big picture:
  BERT made unsupervised pretraining + fine-tuning the NLP default.
  Follow-ups mostly improved the self-supervised OBJECTIVE and
  training recipe. Generative LLMs later scaled a causal fork, but
  still live in the culture BERT popularized.

See also:
  190_transformer.py -- encoder vs decoder attention
  260_positional_encoding.py -- PE alternatives (BERT used learned PE)
  170_dropout.py / 160_normalization.py -- regularizers inside the stack
""")
