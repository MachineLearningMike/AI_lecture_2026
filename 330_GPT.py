"""
330_GPT.py

LLMs and GPT -- decoder-only Transformers that predict the next token.

Pipeline sketched here:
  1) Pretrain: causal language modeling on huge text (next-token prediction)
  2) Fine-tune: supervised instruction / chat data (SFT)
  3) Align: preference methods (RLHF, DPO, ...) so answers are helpful/harmless
  4) Use: zero-shot / one-shot / few-shot prompting, chain-of-thought, tools

GPT = Generative Pre-trained Transformer (Radford et al.), scaled into
today's chat LLMs. Companion to 190_transformer.py (causal mask) and
320_BERT.py (encoder / MLM fork of the same family).
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
    add_step_footer,
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
# Toy next-token demo
# -------------------------------------------------------
CONTEXT = ['The', 'cat', 'sat', 'on', 'the']
VOCAB = ['mat', 'hat', 'floor', 'cat', 'the', 'dog']
# Hand-crafted logits favoring "mat"
logits = np.array([3.2, 0.4, 1.1, -0.5, 0.2, -1.0], dtype=float)
probs = softmax(logits)
true_id = VOCAB.index('mat')
nll = -np.log(probs[true_id] + 1e-12)

TOKENS = ['BOS'] + CONTEXT + ['mat']
T = len(TOKENS)
# Causal attention toy map
scores = RNG.normal(0, 0.6, size=(T, T))
for i in range(T):
    scores[i, i + 1:] = -1e9
attn_causal = softmax(scores, axis=-1)


# -------------------------------------------------------
# FIGURE 1: What an LLM / GPT is
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    'Step 1: GPT / LLM = decoder-only Transformer that predicts the next token',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)

ax = axes1[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('One architecture, three eras', fontsize=FS_AX_TITLE - 2)

box(ax, 0.5, 5.8, 9.0, 1.5, 'GPT-1 / GPT-2: generative pretrain + fine-tune / zero-shot', '#dceaff')
box(ax, 0.5, 3.9, 9.0, 1.5, 'GPT-3: in-context learning emerges at scale (few-shot)', '#fff3cd')
box(ax, 0.5, 2.0, 9.0, 1.5, 'ChatGPT era: SFT + alignment -> helpful assistants', '#d4edda')

ax.text(
    5.0, 0.7,
    'LLM = Large Language Model: same Transformer idea,\n'
    'pushed to enough parameters + data that new behaviors appear.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes1[1]
ax.axis('off')
ax.set_title('Vs BERT (320_BERT.py)', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    'BERT (encoder):\n'
    '  bidirectional MLM -- UNDERSTAND / classify / span extract\n\n'
    'GPT (decoder):\n'
    '  causal next-token LM -- GENERATE left-to-right text\n\n'
    'Both are Transformers. Different mask + objective\n'
    '=> different strengths.\n\n'
    'Modern "chat models" are GPT-style stacks plus\n'
    'instruction fine-tuning and alignment on top.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Causal LM objective
# -------------------------------------------------------
fig2, axes2 = new_figure(1, 2)
fig2.suptitle(
    'Step 2: Pretraining = maximize P(token_t | token_1..t-1) over the corpus',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)

ax = axes2[0]
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Teacher-forced next-token prediction', fontsize=FS_AX_TITLE - 2)

xs = np.linspace(1.0, 11.0, len(TOKENS))
for i, (x, tok) in enumerate(zip(xs, TOKENS)):
    color = '#f8d7da' if i == len(TOKENS) - 1 else '#dceaff'
    box(ax, x - 0.65, 5.8, 1.3, 1.0, tok, color, fontsize=FS_ANNOT - 2)

# arrows: each position predicts the next
for i in range(len(TOKENS) - 1):
    ax.annotate(
        '', xy=(xs[i + 1], 5.8), xytext=(xs[i], 5.8),
        arrowprops=dict(arrowstyle='->', color='crimson', lw=1.8,
                        connectionstyle='arc3,rad=-0.35'),
    )
ax.text(
    6.0, 4.3,
    'At each step: read the PAST only, predict the NEXT token.\n'
    'Loss = sum_t cross-entropy( predicted_t , true_t )',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
box(ax, 2.5, 1.5, 7.0, 1.8,
    'Self-supervised: labels = the text itself\n(no human annotation needed)',
    '#d4edda')

ax = axes2[1]
ax.bar(VOCAB, probs, color='royalblue', edgecolor='k')
ax.set_ylabel(r'$P($next$)$', fontsize=FS_AXIS)
ax.set_title('Softmax over vocab given context', fontsize=FS_AX_TITLE - 2)
ax.tick_params(axis='both', labelsize=FS_TICK - 4)
ax.set_ylim(0, 1.05)
for i, (v, p) in enumerate(zip(VOCAB, probs)):
    ax.text(i, p + 0.03, f'{p:.2f}', ha='center', fontsize=FS_ANNOT - 2)
ax.grid(True, axis='y', alpha=0.3)
add_note(
    ax,
    f'Context: "{" ".join(CONTEXT)}"\n'
    f'Target: "mat"\n'
    f'NLL = {nll:.3f}\n\n'
    'Pretrain minimizes\n'
    'this NLL everywhere.',
    y=0.98, va='top', facecolor='lightyellow',
)

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Causal mask
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    'Step 3: Causal attention -- each token may only see the past',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes3[0]
im = ax.imshow(attn_causal, cmap='viridis', vmin=0, vmax=attn_causal.max())
for i in range(T):
    for j in range(i + 1, T):
        ax.add_patch(patches.Rectangle(
            (j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='red', lw=1.3,
        ))
ax.set_xticks(range(T))
ax.set_yticks(range(T))
ax.set_xticklabels(TOKENS, rotation=45, ha='right')
ax.set_yticklabels(TOKENS)
ax.set_xlabel('key (past allowed)', fontsize=FS_AXIS)
ax.set_ylabel('query', fontsize=FS_AXIS)
ax.set_title('Lower-triangular attention', fontsize=FS_AX_TITLE - 2)
ax.tick_params(axis='both', labelsize=FS_TICK - 4)
fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
add_note(
    ax,
    'Red cells = future\n'
    '(blocked).\n\n'
    'Enables generation:\n'
    'sample token t, append,\n'
    'repeat.',
    x=0.02, y=0.98, ha='left', va='top', facecolor='lightyellow',
)

ax = axes3[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Autoregressive generation loop', fontsize=FS_AX_TITLE - 2)
steps = [
    (1.0, 6.2, 'context'),
    (4.0, 6.2, 'forward\npass'),
    (7.0, 6.2, 'sample\nnext tok'),
]
for x, y, text in steps:
    box(ax, x, y, 2.4, 1.4, text, '#dceaff')
ax.annotate('', xy=(4.0, 6.9), xytext=(3.4, 6.9),
            arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(7.0, 6.9), xytext=(6.4, 6.9),
            arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(2.2, 6.2), xytext=(8.2, 5.5),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='crimson',
                            connectionstyle='arc3,rad=-0.4'))
ax.text(5.0, 4.3, 'append & repeat', color='crimson',
        ha='center', fontsize=FS_ANNOT, fontweight='bold')
ax.text(
    5.0, 1.8,
    'Training uses teacher forcing (true past tokens).\n'
    'Inference uses its own samples (error can accumulate).\n'
    'KV cache makes long generation practical.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Scaling -- why "large"
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    'Step 4: Scaling -- more params + data + compute keep paying off (roughly)',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)

ax = axes4[0]
# Toy power-law-ish loss vs compute
C = np.array([1e2, 1e3, 1e4, 1e5, 1e6, 1e7])
loss = 2.8 * C ** (-0.08) + 1.35  # schematic
ax.semilogx(C, loss, 'o-', color='royalblue', lw=2.5, ms=8)
ax.set_xlabel('pretrain compute (toy units)', fontsize=FS_AXIS)
ax.set_ylabel('held-out NLL (toy)', fontsize=FS_AXIS)
ax.set_title('Schematic scaling curve', fontsize=FS_AX_TITLE - 2)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3, which='both')
add_note(
    ax,
    'Scaling laws:\n'
    'loss falls smoothly\n'
    'as you scale model,\n'
    'data, and FLOPs\n'
    'together.\n\n'
    'Not magic -- but\n'
    'predictable enough\n'
    'to plan training.',
    y=0.98, va='top', facecolor='lightyellow',
)

ax = axes4[1]
ax.axis('off')
ax.set_title('What emerges with scale', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    'Small models: fluent local text, weak tasks.\n\n'
    'Larger models often gain:\n'
    '  • broader world / code / math knowledge\n'
    '  • in-context learning (few-shot)\n'
    '  • tool-use / instruction following (after SFT)\n'
    '  • chain-of-thought when prompted\n\n'
    'Caveats: data quality, contamination, compute cost,\n'
    'and diminishing returns still matter. Scale is a\n'
    'lever, not a guarantee of truthfulness or safety.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Pretrain -> SFT -> Align pipeline
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    'Step 5: Modern LLM pipeline -- Pretrain, then SFT, then Align',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax5.set_xlim(0, 14)
ax5.set_ylim(0, 9)
ax5.axis('off')

box(ax5, 0.4, 5.5, 4.0, 2.8,
    '1. PRETRAIN\ncausal LM\non web-scale text\n\nraw next-token skill',
    '#dceaff', fontsize=FS_ANNOT)
box(ax5, 5.0, 5.5, 4.0, 2.8,
    '2. SFT\ninstruction / chat\ndemos (human or\nsynthetic)\n\nfollow the format',
    '#fff3cd', fontsize=FS_ANNOT)
box(ax5, 9.6, 5.5, 4.0, 2.8,
    '3. ALIGN\npreferences / RLHF\n/ DPO / ...\n\nbe helpful & safe',
    '#d4edda', fontsize=FS_ANNOT)

ax5.annotate('', xy=(5.0, 6.9), xytext=(4.4, 6.9),
             arrowprops=dict(arrowstyle='->', lw=3, color='seagreen'))
ax5.annotate('', xy=(9.6, 6.9), xytext=(9.0, 6.9),
             arrowprops=dict(arrowstyle='->', lw=3, color='seagreen'))

ax5.text(
    7.0, 2.5,
    'Pretrain teaches "how language / knowledge continue".\n'
    'SFT teaches "how to answer like an assistant given a prompt".\n'
    'Alignment steers "which answers humans prefer / allow".\n\n'
    'Base GPT (pretrained only) can complete text but may ignore instructions.\n'
    'Chat models are mostly the later stages on top of a strong base.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)


plt.tight_layout()
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Supervised fine-tuning (SFT)
# -------------------------------------------------------
fig6, axes6 = new_figure(1, 2)
fig6.suptitle(
    'Step 6: Supervised fine-tuning (SFT) -- learn to follow instructions',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes6[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Chat-style training example', fontsize=FS_AX_TITLE - 2)

box(ax, 0.5, 5.8, 9.0, 1.6,
    'User: Explain gravity to a child.', '#dceaff')
box(ax, 0.5, 3.6, 9.0, 1.8,
    'Assistant: Gravity is the pull that keeps us\n'
    'on the ground and the Moon around Earth...',
    '#d4edda')
ax.annotate('', xy=(5.0, 5.4), xytext=(5.0, 5.8),
            arrowprops=dict(arrowstyle='->', lw=2))
ax.text(
    5.0, 1.5,
    'Still next-token loss -- but only (or mainly) on the\n'
    'ASSISTANT tokens. The model learns the chat template\n'
    'and how good answers look.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes6[1]
ax.axis('off')
ax.set_title('What SFT buys you', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    '• Instruction following ("do the task asked")\n'
    '• Format: markdown, JSON, bullet answers, ...\n'
    '• Multi-turn chat roles (system / user / assistant)\n'
    '• Safety refusals that appear in the demos\n\n'
    'Limits of SFT alone:\n'
    '  - only as good as the demo distribution\n'
    '  - does not rank TWO ok answers by preference\n'
    '  - can still be sycophantic / verbose / unsafe\n'
    '    outside the supervised set\n\n'
    '=> preference alignment is the next stage.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Zero / one / few-shot
# -------------------------------------------------------
fig7, axes7 = new_figure(1, 3)
fig7.suptitle(
    'Step 7: In-context learning -- zero-shot, one-shot, few-shot',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

panels = [
    ('Zero-shot',
     'No examples.\nJust the instruction.\n\n'
     'Translate to French:\n"cheese" ->',
     '#dceaff',
     'Relies on pretrain +\nSFT to know the task.'),
    ('One-shot',
     'One example in the prompt.\n\n'
     'sea otter => loutre de mer\n'
     'cheese =>',
     '#fff3cd',
     'The example shows\nthe FORMAT and mapping.'),
    ('Few-shot\n(multi-shot)',
     'Several examples.\n\n'
     'a => 1\nb => 2\nc => 3\nd =>',
     '#d4edda',
     'More shots can teach\nnovel patterns in-context\nwithout weight updates.'),
]
for ax, (title, body, color, note) in zip(axes7, panels):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title(title, fontsize=FS_AX_TITLE - 2)
    box(ax, 0.6, 3.8, 8.8, 5.2, body, color, fontsize=FS_ANNOT)
    ax.text(
        5.0, 1.5, note, ha='center', va='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
    )

fig7.text(
    0.5, 0.02,
    'Few-shot is NOT fine-tuning: weights stay frozen; examples live only in the context window.\n'
    'GPT-3 popularized that this works surprisingly well once models are large enough.',
    ha='center', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
fig7.tight_layout(rect=[0, 0.08, 1, 1])
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Chain-of-thought
# -------------------------------------------------------
fig8, axes8 = new_figure(1, 2)
fig8.suptitle(
    'Step 8: Chain-of-thought (CoT) -- reason in tokens before the answer',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)

ax = axes8[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis('off')
ax.set_title('Direct answer vs scratchpad reasoning', fontsize=FS_AX_TITLE - 2)

box(ax, 0.4, 6.5, 9.2, 1.8,
    'Q: A shop had 15 apples. Sold 1/3, then bought 10.\nHow many now?',
    '#dceaff', fontsize=FS_ANNOT - 1)
box(ax, 0.4, 4.2, 4.2, 1.8, 'Direct\n"20"  (fragile)', '#f8d7da')
box(ax, 5.4, 3.5, 4.2, 2.5,
    'CoT\n15/3=5 sold\n15-5=10\n10+10=20\nAnswer: 20',
    '#d4edda', fontsize=FS_ANNOT - 1)

ax.text(
    5.0, 1.3,
    'Extra generated tokens = more serial compute\n'
    'on hard problems (math, logic, multi-step).',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes8[1]
ax.axis('off')
ax.set_title('How to elicit CoT', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    '1) Prompt: "Let\'s think step by step." (zero-shot CoT)\n\n'
    '2) Few-shot CoT: show worked examples with reasoning\n'
    '   traces, then ask the new question.\n\n'
    '3) Self-consistency: sample several CoT paths,\n'
    '   majority-vote the final answer.\n\n'
    '4) Tool-augmented: reason, call a calculator / code,\n'
    '   continue (agentic loop).\n\n'
    'CoT helps most when the task needs multi-step\n'
    'computation -- not when the answer is a single lookup.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 9: Alignment -- RLHF sketch
# -------------------------------------------------------
fig9, axes9 = new_figure(1, 2)
fig9.suptitle(
    'Step 9: Alignment -- steer the model toward preferred behavior',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes9[0]
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis('off')
ax.set_title('RLHF loop (schematic)', fontsize=FS_AX_TITLE - 2)

box(ax, 0.4, 6.8, 3.5, 1.5, 'Prompt', '#dceaff')
box(ax, 4.3, 6.8, 3.5, 1.5, 'Policy LLM\nsamples answers', '#fff3cd')
box(ax, 8.2, 6.8, 3.5, 1.5, 'Humans rank\nA > B > C', '#f8d7da')
ax.annotate('', xy=(4.3, 7.5), xytext=(3.9, 7.5),
            arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(8.2, 7.5), xytext=(7.8, 7.5),
            arrowprops=dict(arrowstyle='->', lw=2))

box(ax, 2.0, 4.2, 8.0, 1.6, 'Train reward model R(prompt, answer)', '#e2d5f1')
ax.annotate('', xy=(6.0, 5.8), xytext=(10.0, 6.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='crimson'))

box(ax, 2.0, 1.8, 8.0, 1.6,
    'RL (e.g. PPO): update policy to raise R\n'
    'with KL penalty vs the SFT reference',
    '#d4edda')
ax.annotate('', xy=(6.0, 3.4), xytext=(6.0, 4.2),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='seagreen'))

ax = axes9[1]
ax.axis('off')
ax.set_title('What "aligned" usually targets', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    'Common goals (HHH-style):\n'
    '  • Helpful -- solve the user\'s task\n'
    '  • Honest -- avoid fabrications when possible\n'
    '  • Harmless -- refuse dangerous / disallowed asks\n\n'
    'Also: style, brevity, citation habits, tool use, ...\n\n'
    'Alignment is preference shaping on top of a base\n'
    'model that already "knows" a lot from pretraining.\n'
    'It does not magically add new world knowledge\n'
    '(and can trade off capability if overdone).',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig9)


# -------------------------------------------------------
# FIGURE 10: DPO and preference alternatives
# -------------------------------------------------------
fig10, axes10 = new_figure(1, 2)
fig10.suptitle(
    'Step 10: Preferences without a full RL loop -- DPO and cousins',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)

ax = axes10[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Preference pair', fontsize=FS_AX_TITLE - 2)

box(ax, 0.5, 5.8, 9.0, 1.5, 'Prompt: "Summarize this article"', '#dceaff')
box(ax, 0.5, 3.8, 4.2, 1.6, 'y_w  WINNER\n(preferred)', '#d4edda')
box(ax, 5.3, 3.8, 4.2, 1.6, 'y_l  LOSER\n(rejected)', '#f8d7da')
ax.text(
    5.0, 1.6,
    'DPO: directly increase likelihood of y_w\n'
    'relative to y_l vs a reference (SFT) model.\n'
    'Same preference data; simpler than PPO-RLHF.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes10[1]
ax.axis('off')
ax.set_title('Alignment menu (conceptual)', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    '• SFT -- imitate good demos\n'
    '• RLHF (PPO + reward model) -- optimize preference reward\n'
    '• DPO / IPO / KTO -- preference losses without RL sampling\n'
    '• RLAIF / constitutional -- AI feedback / written principles\n'
    '• Rejection sampling / best-of-N -- filter at inference\n\n'
    'Shared idea: a base model proposes; a preference signal\n'
    'chooses; training (or decoding) shifts mass toward winners.\n\n'
    'Still open: over-refusal, sycophancy, reward hacking,\n'
    'and evaluating "aligned" beyond chat benchmarks.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig10)


# -------------------------------------------------------
# FIGURE 11: Prompting stack at inference
# -------------------------------------------------------
fig11, ax11 = new_figure()
fig11.suptitle(
    'Step 11: At inference -- prompts, roles, tools, and decoding',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax11.set_xlim(0, 12)
ax11.set_ylim(0, 9)
ax11.axis('off')

layers = [
    (0.5, 7.0, 11.0, 1.4, 'System prompt -- persona, policies, tool rules', '#e2d5f1'),
    (0.5, 5.3, 11.0, 1.4, 'User / few-shot / CoT instructions -- the actual task', '#dceaff'),
    (0.5, 3.6, 11.0, 1.4, 'Model + optional tools (search, code, retrieval / RAG)', '#fff3cd'),
    (0.5, 1.9, 11.0, 1.4, 'Decoding: temperature, top-p, max tokens, stop strings', '#d4edda'),
]
for x, y, w, h, text, color in layers:
    box(ax11, x, y, w, h, text, color, fontsize=FS_ANNOT)

ax11.text(
    6.0, 0.6,
    'Pretrain + SFT + alignment set the PRIOR over answers.\n'
    'Prompting / tools / decoding steer ONE trajectory at use time.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig11)


# -------------------------------------------------------
# FIGURE 12: Family map
# -------------------------------------------------------
fig12, axes12 = new_figure(1, 2)
fig12.suptitle(
    'Step 12: Putting it together -- from next-token to assistant',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes12[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Capability ladder', fontsize=FS_AX_TITLE - 2)

ladder = [
    (0.5, 8.0, 'Base LM: complete text', '#dceaff'),
    (0.5, 6.4, 'SFT: follow instructions / chat', '#fff3cd'),
    (0.5, 4.8, 'Aligned: preferred / safer answers', '#d4edda'),
    (0.5, 3.2, 'Prompted: zero/few-shot + CoT', '#f8d7da'),
    (0.5, 1.6, 'Agentic: tools, memory, multi-step', '#ffe0b2'),
]
for x, y, text, color in ladder:
    box(ax, x, y, 9.0, 1.2, text, color)

ax = axes12[1]
ax.axis('off')
ax.set_title('Takeaway checklist', fontsize=FS_AX_TITLE - 2)
ax.text(
    0.5, 0.55,
    '1. GPT-style LLMs are causal Transformers.\n\n'
    '2. Pretrain = next-token prediction at scale.\n\n'
    '3. SFT teaches assistant format & instructions.\n\n'
    '4. Alignment shapes preferences (RLHF / DPO / ...).\n\n'
    '5. Zero / one / few-shot = in-context learning.\n\n'
    '6. Chain-of-thought spends tokens to think.\n\n'
    '7. Prompting and tools steer a frozen (or lightly\n'
    '   adapted) model at inference time.\n\n'
    'Same stack as 190_transformer.py -- pushed into\n'
    'a full product recipe for generative AI.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig12)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("LLMs AND GPT")
print("=" * 60)
print(f"""
What GPT / an LLM is:
  Decoder-only Transformer with a causal mask.
  Trained to model P(x_t | x_<t). Generation = sample, append, repeat.

Toy next-token demo:
  Context: "{" ".join(CONTEXT)}"
  Vocab probs: {list(zip(VOCAB, [float(x) for x in np.round(probs, 3)]))}
  NLL of "mat": {nll:.3f}

Pretraining:
  Self-supervised causal LM on large corpora.
  Teaches fluency, knowledge patterns, and (at scale) in-context skills.
  Scaling laws: more params + data + compute -> lower loss (schematically).

Post-training recipe (chat LLMs):
  1) SFT  -- supervised instruction / chat demos (loss on assistant tokens)
  2) Align -- preference optimization so answers match human (or AI) prefs
       RLHF: reward model + RL (e.g. PPO) with KL to reference
       DPO:  closed-form preference loss on (win, lose) pairs
       Also: RLAIF, constitutional principles, best-of-N, ...

Using the model (weights often frozen):
  Zero-shot  -- instruction only
  One-shot   -- one example in the prompt
  Few-shot   -- several examples (in-context learning; not gradient updates)
  Chain-of-thought -- generate intermediate reasoning tokens, then answer
  Tools / RAG / agents -- interleave reasoning with external actions

Vs BERT:
  BERT: bidirectional encoder + MLM (+ NSP) -> understanding / discriminative tasks
  GPT:  causal decoder + next-token LM (+ SFT/align) -> generation / assistants

See also:
  190_transformer.py -- causal vs bidirectional attention
  260_positional_encoding.py -- PE / RoPE used in modern GPTs
  320_BERT.py -- the encoder / fine-tune fork of the same family
""")
