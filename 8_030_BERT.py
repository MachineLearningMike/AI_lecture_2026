"""
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

Interactive playgrounds (Steps 12-14):
  - MLM: drag mask position + mask rate; watch prediction bars
  - NSP: drag same-pair <-> random-pair; watch IsNext scores
  - Attention: morph causal <-> bidirectional heatmaps
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
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
def build_step_1():
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
    return fig1


# -------------------------------------------------------
# FIGURE 2: Input representation
# -------------------------------------------------------
def build_step_2():
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
    return fig2


# -------------------------------------------------------
# FIGURE 3: Bidirectional vs causal
# -------------------------------------------------------
def build_step_3():
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
    return fig3


# -------------------------------------------------------
# FIGURE 4: Masked Language Modeling (MLM)
# -------------------------------------------------------
# BERT 80/10/10 masking recipe among the 15% selected tokens
def build_step_4():
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
    return fig4


# -------------------------------------------------------
# FIGURE 5: Next Sentence Prediction (NSP)
# -------------------------------------------------------
def build_step_5():
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
    return fig5


# -------------------------------------------------------
# FIGURE 6: Pretrain then fine-tune
# -------------------------------------------------------
def build_step_6():
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
    return fig6


# -------------------------------------------------------
# FIGURE 7: Why MLM enables deep bidirectionality
# -------------------------------------------------------
def build_step_7():
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
    return fig7


# -------------------------------------------------------
# FIGURE 8: RoBERTa -- refining the BERT recipe
# -------------------------------------------------------
def build_step_8():
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
    return fig8


# -------------------------------------------------------
# FIGURE 9: ALBERT -- parameter efficiency
# -------------------------------------------------------
def build_step_9():
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
    return fig9


# -------------------------------------------------------
# FIGURE 10: ELECTRA -- replaced token detection
# -------------------------------------------------------
def build_step_10():
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
    return fig10


# -------------------------------------------------------
# FIGURE 11: Family snapshot
# -------------------------------------------------------
def build_step_11():
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
    return fig11


# -------------------------------------------------------
# FIGURE 12: Interactive MLM playground
# -------------------------------------------------------
VOCAB_LIVE = ['cat', 'dog', 'mat', 'sat', 'the', 'on']
# Content positions only ([CLS]/[SEP] stay fixed)
CONTENT_IDX = [i for i, tok in enumerate(TOKENS) if tok not in ('[CLS]', '[SEP]')]
# Soft affinities: each vocab word "likes" certain neighbors
_AFFINITY = {
    'cat': {'the': 1.4, 'sat': 1.8, 'on': 0.6, 'mat': 0.9},
    'dog': {'the': 1.2, 'sat': 1.1, 'on': 0.5, 'mat': 0.4},
    'mat': {'the': 1.0, 'on': 1.6, 'sat': 0.7, 'cat': 0.8},
    'sat': {'cat': 1.7, 'the': 0.8, 'on': 1.2, 'mat': 0.9},
    'the': {'cat': 0.9, 'mat': 0.8, 'sat': 0.5, 'on': 0.4},
    'on': {'the': 1.1, 'mat': 1.5, 'sat': 0.9, 'cat': 0.5},
}


def toy_mlm_probs(mask_idx, mask_rate):
    """Toy bidirectional cloze: context votes; extra masks flatten the peak."""
    true = TOKENS[mask_idx]
    # secondary masks from mask_rate (exclude primary + specials)
    n_extra = int(round(mask_rate * max(len(CONTENT_IDX) - 1, 1)))
    others = [i for i in CONTENT_IDX if i != mask_idx]
    # deterministic order from mask_idx for stable demos
    order = sorted(others, key=lambda i: (mask_idx * 17 + i * 31) % 997)
    extra = set(order[:n_extra])
    visible = [
        TOKENS[i] for i in range(T)
        if i != mask_idx and i not in extra and TOKENS[i] not in ('[CLS]', '[SEP]')
    ]
    logits = np.zeros(len(VOCAB_LIVE))
    for k, w in enumerate(VOCAB_LIVE):
        score = 0.15
        for ctx in visible:
            score += _AFFINITY.get(w, {}).get(ctx, 0.05)
        # slight prior for the true token when context is rich
        if w == true:
            score += 0.55
        logits[k] = score
    # Higher mask rate / fewer visible neighbors => flatter (confused) distribution
    context_strength = max(len(visible), 0)
    temperature = 0.55 + 0.55 * mask_rate + 0.35 * (1.0 - context_strength / 5.0)
    return softmax(logits / temperature), sorted(extra)


def build_step_12():
    fig12, (ax12a, ax12b) = new_figure(1, 2)
    fig12.suptitle(
        'Step 12 (interactive): MLM playground -- drag mask position & rate',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    fig12.subplots_adjust(bottom=0.22)


    def _draw_mlm_seq(ax, mask_idx, extra):
        ax.clear()
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 6)
        ax.axis('off')
        ax.set_title('Cloze input (pink = masked)', fontsize=FS_AX_TITLE - 2)
        xs = np.linspace(1.0, 11.0, T)
        for i, (x, tok) in enumerate(zip(xs, TOKENS)):
            show = tok
            if i == mask_idx or i in extra:
                show = '[MASK]'
                color = '#f8d7da'
            elif tok in ('[CLS]', '[SEP]'):
                color = '#e2d5f1'
            else:
                color = '#d4edda'
            box(ax, x - 0.55, 3.4, 1.1, 1.0, show, color, fontsize=FS_ANNOT - 2)
            ax.text(x, 2.9, tok if i == mask_idx else '', ha='center', va='top',
                    fontsize=FS_ANNOT - 2, color='crimson', fontweight='bold')
        ax.text(
            6.0, 1.2,
            'Green = visible context  |  Pink = hidden\n'
            'More masks => weaker bidirectional clues => flatter bars.',
            ha='center', va='center', fontsize=FS_ANNOT,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
        )
        return xs


    bars12 = None
    txts12 = None
    hud12 = None


    def _update_mlm(_=None):
        global bars12, txts12, hud12
        mask_idx = CONTENT_IDX[int(round(sl_pos.val))]
        mask_rate = float(sl_rate.val)
        probs, extra = toy_mlm_probs(mask_idx, mask_rate)
        _draw_mlm_seq(ax12a, mask_idx, extra)

        ax12b.clear()
        colors = ['#2ecc71' if w == TOKENS[mask_idx] else '#5dade2' for w in VOCAB_LIVE]
        bars12 = ax12b.bar(VOCAB_LIVE, probs, color=colors, edgecolor='k')
        txts12 = [
            ax12b.text(i, probs[i] + 0.02, f'{probs[i]:.2f}',
                       ha='center', va='bottom', fontsize=FS_ANNOT - 2)
            for i in range(len(VOCAB_LIVE))
        ]
        ax12b.set_ylim(0, 1.05)
        ax12b.set_ylabel('toy P(token | context)', fontsize=FS_AXIS)
        ax12b.set_title(f'Predict masked \"{TOKENS[mask_idx]}\"', fontsize=FS_AX_TITLE - 2)
        ax12b.tick_params(axis='both', labelsize=FS_TICK - 2)
        ax12b.grid(True, axis='y', alpha=0.3)
        nll = -np.log(probs[VOCAB_LIVE.index(TOKENS[mask_idx])] + 1e-12)
        hud12 = ax12b.text(
            0.98, 0.98,
            f'mask idx = {mask_idx}\n'
            f'extra masks = {len(extra)}\n'
            f'rate ~ {100 * mask_rate:.0f}%\n'
            f'NLL(true) = {nll:.2f}',
            transform=ax12b.transAxes, ha='right', va='top', fontsize=FS_ANNOT,
            bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
        )
        fig12.canvas.draw_idle()


    ax_pos = fig12.add_axes([0.15, 0.10, 0.32, 0.04])
    ax_rate = fig12.add_axes([0.58, 0.10, 0.32, 0.04])
    sl_pos = Slider(
        ax_pos, 'mask pos', 0, len(CONTENT_IDX) - 1,
        valinit=CONTENT_IDX.index(MASK_IDX) if MASK_IDX in CONTENT_IDX else 0,
        valstep=1, color='#8e44ad',
    )
    sl_rate = Slider(
        ax_rate, 'mask rate', 0.05, 0.55,
        valinit=0.15, valstep=0.05, color='#c0392b',
    )
    sl_pos.on_changed(_update_mlm)
    sl_rate.on_changed(_update_mlm)
    _update_mlm()
    print('Step 12: drag mask position + mask rate; watch prediction bars.')
    return fig12


# -------------------------------------------------------
# FIGURE 13: Interactive NSP playground
# -------------------------------------------------------
SENT_A = 'the cat sat'
TRUE_B = 'on the mat'
RANDOM_BS = [
    'stocks rose today',
    'bake two cups flour',
    'quantum spins align',
    'traffic jam ahead',
]


def nsp_probs(continuity):
    """continuity=1 => true next; 0 => random topic mismatch."""
    # Logit margin shrinks as continuity drops
    margin = 2.8 * continuity - 0.9  # high => IsNext, low => NotNext
    logits = np.array([margin, -margin])
    return softmax(logits)


def build_step_13():
    fig13, (ax13a, ax13b) = new_figure(1, 2)
    fig13.suptitle(
        'Step 13 (interactive): NSP -- same-pair vs random-pair vibe',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    fig13.subplots_adjust(bottom=0.24)


    def _draw_nsp_pair(ax, continuity):
        ax.clear()
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis('off')
        ax.set_title('Sentence pair packing', fontsize=FS_AX_TITLE - 2)
        # Pick a random-ish B when continuity is low
        if continuity >= 0.55:
            sent_b = TRUE_B
            tag = 'IsNext candidate'
            col_b = '#d4edda'
        else:
            sent_b = RANDOM_BS[int((1.0 - continuity) * (len(RANDOM_BS) - 1) + 1e-9)]
            tag = 'NotNext (topic jump)'
            col_b = '#f8d7da'
        box(ax, 0.5, 5.4, 9.0, 1.4, f'A: \"{SENT_A}\"', '#dceaff')
        box(ax, 0.5, 3.4, 9.0, 1.4, f'B: \"{sent_b}\"', col_b)
        box(ax, 2.0, 1.3, 6.0, 1.2, f'[CLS] head -> {tag}', '#fff3cd')
        return sent_b


    def _update_nsp(_=None):
        continuity = float(sl_cont.val)
        _draw_nsp_pair(ax13a, continuity)
        p = nsp_probs(continuity)
        ax13b.clear()
        cols = ['seagreen', 'crimson']
        bars = ax13b.bar(['IsNext', 'NotNext'], p, color=cols, edgecolor='k')
        for rect, pi in zip(bars, p):
            ax13b.text(rect.get_x() + rect.get_width() / 2, pi + 0.03,
                       f'{pi:.2f}', ha='center', va='bottom', fontsize=FS_ANNOT)
        ax13b.set_ylim(0, 1.08)
        ax13b.set_ylabel('P from [CLS] (toy)', fontsize=FS_AXIS)
        ax13b.set_title('NSP decision margin', fontsize=FS_AX_TITLE - 2)
        ax13b.tick_params(axis='both', labelsize=FS_TICK)
        ax13b.grid(True, axis='y', alpha=0.3)
        vibe = ('strong same-pair' if continuity > 0.75 else
                'ambiguous' if continuity > 0.4 else
                'random-pair mismatch')
        ax13b.text(
            0.5, 0.92, f'continuity = {continuity:.2f}\n{vibe}',
            transform=ax13b.transAxes, ha='center', va='top', fontsize=FS_ANNOT,
            bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
        )
        fig13.canvas.draw_idle()


    ax_cont = fig13.add_axes([0.22, 0.12, 0.45, 0.045])
    sl_cont = Slider(
        ax_cont, 'continuity', 0.0, 1.0,
        valinit=0.85, valstep=0.05, color='#1e8449',
    )
    ax_btn_same = fig13.add_axes([0.74, 0.132, 0.09, 0.028])
    ax_btn_rand = fig13.add_axes([0.855, 0.132, 0.08, 0.028])
    btn_same = Button(ax_btn_same, 'Same pair', color='#d4edda', hovercolor='#a9dfbf')
    btn_rand = Button(ax_btn_rand, 'Random', color='#f8d7da', hovercolor='#f5b7b1')
    btn_same.label.set_fontsize(7)
    btn_rand.label.set_fontsize(7)


    def _snap_same(_):
        sl_cont.set_val(1.0)


    def _snap_rand(_):
        sl_cont.set_val(0.0)


    btn_same.on_clicked(_snap_same)
    btn_rand.on_clicked(_snap_rand)
    sl_cont.on_changed(_update_nsp)
    _update_nsp()
    print('Step 13: drag continuity (or click Same/Random); watch IsNext bars.')
    return fig13


# -------------------------------------------------------
# FIGURE 14: Interactive bidirectional <-> causal attention
# -------------------------------------------------------
def mixed_attention(alpha):
    """alpha=0: causal; alpha=1: full bidirectional (row-normalized)."""
    m = attn_bi.copy()
    for i in range(T):
        if i + 1 < T:
            m[i, i + 1:] *= alpha
        row = m[i].sum()
        if row > 0:
            m[i] /= row
    return m


def build_step_14():
    fig14, ax14 = new_figure()
    fig14.suptitle(
        'Step 14 (interactive): morph causal GPT mask <-> BERT bidirectional',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    fig14.subplots_adjust(bottom=0.20)
    im14 = ax14.imshow(mixed_attention(1.0), cmap='viridis', vmin=0, vmax=attn_bi.max())
    ax14.set_xticks(range(T))
    ax14.set_yticks(range(T))
    ax14.set_xticklabels(TOKENS, rotation=45, ha='right')
    ax14.set_yticklabels(TOKENS)
    ax14.set_xlabel('key', fontsize=FS_AXIS)
    ax14.set_ylabel('query', fontsize=FS_AXIS)
    ax14.tick_params(axis='both', labelsize=FS_TICK - 4)
    cbar14 = fig14.colorbar(im14, ax=ax14, fraction=0.046, pad=0.04)
    future_patches = []
    hud14 = ax14.text(
        0.02, 0.98, '',
        transform=ax14.transAxes, ha='left', va='top', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )


    def _update_attn(_=None):
        alpha = float(sl_bi.val)
        m = mixed_attention(alpha)
        im14.set_data(m)
        # Red outlines on still-blocked / partially suppressed future cells
        for p in future_patches:
            p.remove()
        future_patches.clear()
        if alpha < 0.999:
            for i in range(T):
                for j in range(i + 1, T):
                    lw = 1.6 if alpha < 0.05 else 1.0
                    ec = 'red' if alpha < 0.35 else 'orange'
                    future_patches.append(
                        ax14.add_patch(patches.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor=ec, lw=lw, alpha=0.85,
                        ))
                    )
        mode = ('BERT full bidirectional' if alpha > 0.9 else
                'GPT causal (no future)' if alpha < 0.1 else
                'hybrid / partial future')
        hud14.set_text(
            f'bidirectionality α = {alpha:.2f}\n'
            f'{mode}\n\n'
            'Red/orange boxes mark\n'
            'future keys (j > i).'
        )
        ax14.set_title(
            f'Attention heatmap (α={alpha:.2f})',
            fontsize=FS_AX_TITLE - 2,
        )
        fig14.canvas.draw_idle()


    ax_bi = fig14.add_axes([0.20, 0.08, 0.42, 0.045])
    sl_bi = Slider(
        ax_bi, r'$\alpha$ bi', 0.0, 1.0,
        valinit=1.0, valstep=0.05, color='#2980b9',
    )
    ax_btn_bert = fig14.add_axes([0.70, 0.086, 0.08, 0.028])
    ax_btn_gpt = fig14.add_axes([0.805, 0.086, 0.08, 0.028])
    btn_bert = Button(ax_btn_bert, 'BERT', color='#d1ecf1', hovercolor='#abd8e3')
    btn_gpt = Button(ax_btn_gpt, 'GPT', color='#f8d7da', hovercolor='#f5b7b1')
    btn_bert.label.set_fontsize(7)
    btn_gpt.label.set_fontsize(7)


    def _snap_bert(_):
        sl_bi.set_val(1.0)


    def _snap_gpt(_):
        sl_bi.set_val(0.0)


    btn_bert.on_clicked(_snap_bert)
    btn_gpt.on_clicked(_snap_gpt)
    sl_bi.on_changed(_update_attn)
    _update_attn()
    print('Step 14: drag alpha (or click BERT/GPT); watch future keys unlock.')
    return fig14


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
    build_step_10,
    build_step_11,
    build_step_12,
    build_step_13,
    build_step_14,
])

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

Interactive playgrounds in this file:
  Step 12 -- MLM: mask position + mask rate -> live prediction bars
  Step 13 -- NSP: continuity slider / Same vs Random buttons
  Step 14 -- Attention: alpha morphs causal GPT <-> bidirectional BERT

Big picture:
  BERT made unsupervised pretraining + fine-tuning the NLP default.
  Follow-ups mostly improved the self-supervised OBJECTIVE and
  training recipe. Generative LLMs later scaled a causal fork, but
  still live in the culture BERT popularized.

See also:
  0850_transformer.py -- encoder vs decoder attention
  0880_positional_encoding.py -- PE alternatives (BERT used learned PE)
  0710_dropout.py / 0720_normalize_activations.py -- regularizers inside the stack
""")
