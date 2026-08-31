"""
420_GPT.py

LLMs and GPT -- decoder-only Transformers that predict the next token.

Pipeline sketched here:
  1) Pretrain: causal language modeling on huge text (next-token prediction)
  2) Fine-tune: supervised instruction / chat data (SFT)
  3) Align: preference methods (RLHF, DPO, ...) so answers are helpful/harmless
  4) Use: zero-shot / one-shot / few-shot prompting, chain-of-thought, tools

GPT = Generative Pre-trained Transformer (Radford et al.), scaled into
today's chat LLMs. Companion to 260_transformer.py (causal mask) and
410_BERT.py (encoder / MLM fork of the same family).
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
from matplotlib.widgets import Button, Slider
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


def softmax(z, axis=-1, tau=1.0):
    z = np.asarray(z, dtype=float) / max(float(tau), 1e-8)
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
ax.set_title('Vs BERT (410_BERT.py)', fontsize=FS_AX_TITLE - 2)
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
# FIGURE 2: Causal LM objective + interactive temperature
# -------------------------------------------------------
fig2, axes2 = new_figure(1, 2)
fig2.suptitle(
    'Step 2: Pretraining = next-token prediction  (drag temperature)',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)
fig2.subplots_adjust(bottom=0.18, wspace=0.28)

ax = axes2[0]
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Teacher-forced causal LM', fontsize=FS_AX_TITLE - 2)

xs = np.linspace(1.0, 11.0, len(TOKENS))
for i, (x, tok) in enumerate(zip(xs, TOKENS)):
    color = '#f8d7da' if i == len(TOKENS) - 1 else '#dceaff'
    box(ax, x - 0.65, 5.8, 1.3, 1.0, tok, color, fontsize=FS_ANNOT - 2)

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
    'Self-supervised: labels = the text itself\n'
    '(no human annotation needed)',
    '#d4edda')

ax2b = axes2[1]
TAU0 = 1.0
p2 = softmax(logits, tau=TAU0)
colors2 = ['seagreen' if v == 'mat' else 'royalblue' for v in VOCAB]
bars2 = ax2b.bar(VOCAB, p2, color=colors2, edgecolor='k')
ax2b.set_ylabel(r'$P($next$)$', fontsize=FS_AXIS)
ax2b.set_title(
    f'Context: "{" ".join(CONTEXT)}"  -> next?',
    fontsize=FS_AX_TITLE - 2,
)
ax2b.tick_params(axis='both', labelsize=FS_TICK - 4)
ax2b.set_ylim(0, 1.05)
ax2b.grid(True, axis='y', alpha=0.3)
txt2 = [
    ax2b.text(i, p2[i] + 0.03, f'{p2[i]:.2f}', ha='center', fontsize=FS_ANNOT - 2)
    for i in range(len(VOCAB))
]
hud2 = ax2b.text(
    0.98, 0.98, '',
    transform=ax2b.transAxes, ha='right', va='top', fontsize=FS_ANNOT - 1,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax_tau2 = fig2.add_axes([0.58, 0.06, 0.35, 0.04])
sl_tau2 = Slider(ax_tau2, r'decode $\tau$', 0.15, 3.0, valinit=TAU0, valstep=0.05,
                 color='royalblue')


def _update_tau2(_=None):
    tau = float(sl_tau2.val)
    p = softmax(logits, tau=tau)
    for bar, pi, txt in zip(bars2, p, txt2):
        bar.set_height(pi)
        txt.set_y(pi + 0.03)
        txt.set_text(f'{pi:.2f}')
    nll_t = -np.log(p[true_id] + 1e-12)
    vibe = ('greedy / peaky' if tau < 0.5 else
            'nucleus-ish mid' if tau < 1.5 else
            'wild / flat')
    hud2.set_text(
        f'$\\tau$ = {tau:.2f}  ({vibe})\n'
        f'P(mat) = {p[true_id]:.3f}\n'
        f'NLL(mat) = {nll_t:.3f}\n\n'
        'Pretrain fits $\\tau$=1 logits.\n'
        'Temperature is a decode knob\n'
        'on the learned distribution.'
    )
    fig2.canvas.draw_idle()


sl_tau2.on_changed(_update_tau2)
_update_tau2()
print('Step 2: drag temperature -- pretrained next-token distribution reshapes.')
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
    'Interactive deep-dives: Step 2 (tau), Step 6 (SFT vibe),\n'
    'Step 9 (RLHF stages), Step 10 (HHH knobs + DPO).',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)


plt.tight_layout()
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Supervised fine-tuning (SFT) -- before / after vibe
# -------------------------------------------------------
fig6, axes6 = new_figure(1, 2)
fig6.suptitle(
    'Step 6: SFT -- drag the slider: base completer -> instruction follower',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)
fig6.subplots_adjust(bottom=0.20, wspace=0.28)

SFT_PROMPT = 'User: Explain gravity to a child.'
SFT_REPLIES = [
    # stage 0: raw pretrained continuation vibe
    'gravity to a child. Explain gravity to a\n'
    'child. The child asked again. Chapter 3...',
    # stage 1: half-aware
    'Gravity... is a force. Force equals mass\n'
    'times... (rambling textbook paste)',
    # stage 2: mostly on-format
    'Sure! Gravity is the invisible pull that\n'
    'keeps your feet on the ground.',
    # stage 3: polished SFT assistant
    'Gravity is the pull that keeps us on the\n'
    'ground and the Moon around Earth -- like\n'
    'an invisible hug from the planet!',
]
# meters: follow / format / helpful  (0..1) per stage
SFT_METERS = np.array([
    [0.05, 0.10, 0.15],
    [0.35, 0.30, 0.40],
    [0.75, 0.80, 0.70],
    [0.95, 0.95, 0.92],
], dtype=float)

ax6a = axes6[0]
ax6a.set_xlim(0, 10)
ax6a.set_ylim(0, 9)
ax6a.axis('off')
ax6a.set_title('Same prompt, changing model vibe', fontsize=FS_AX_TITLE - 2)
box(ax6a, 0.5, 7.2, 9.0, 1.2, SFT_PROMPT, '#dceaff', fontsize=FS_ANNOT - 1)
reply_box = patches.FancyBboxPatch(
    (0.5, 3.6), 9.0, 3.0, boxstyle='round,pad=0.02,rounding_size=0.1',
    facecolor='#f8d7da', edgecolor='k', lw=2,
)
ax6a.add_patch(reply_box)
reply_txt = ax6a.text(
    5.0, 5.1, SFT_REPLIES[0], ha='center', va='center',
    fontsize=FS_ANNOT - 1, fontweight='bold',
)
stage_lbl = ax6a.text(
    5.0, 3.1, '', ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
ax6a.text(
    5.0, 1.3,
    'Still next-token loss -- mainly on ASSISTANT tokens.\n'
    'SFT teaches chat template + "what a good answer looks like".',
    ha='center', va='center', fontsize=FS_ANNOT - 1,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

ax6b = axes6[1]
meter_names = ['Follows\ninstruction', 'Chat\nformat', 'Helpful\ntone']
meter_colors = ['#5b8def', '#f0ad4e', '#5cb85c']
bars6 = ax6b.bar(meter_names, SFT_METERS[0], color=meter_colors, edgecolor='k', width=0.65)
ax6b.set_ylim(0, 1.15)
ax6b.set_ylabel('toy score', fontsize=FS_AXIS)
ax6b.set_title('Instruction-following meters', fontsize=FS_AX_TITLE - 2)
ax6b.tick_params(axis='both', labelsize=FS_TICK - 2)
ax6b.grid(True, axis='y', alpha=0.3)
txt6 = [
    ax6b.text(i, SFT_METERS[0, i] + 0.04, f'{SFT_METERS[0, i]:.2f}',
              ha='center', fontsize=FS_ANNOT - 1)
    for i in range(3)
]
hud6 = ax6b.text(
    0.98, 0.98, '',
    transform=ax6b.transAxes, ha='right', va='top', fontsize=FS_ANNOT - 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.95),
)

ax_sft = fig6.add_axes([0.20, 0.06, 0.60, 0.045])
sl_sft = Slider(ax_sft, 'SFT progress', 0, 3, valinit=0, valstep=1, color='#f0ad4e')


def _update_sft(_=None):
    k = int(sl_sft.val)
    reply_txt.set_text(SFT_REPLIES[k])
    reply_box.set_facecolor(['#f8d7da', '#fff3cd', '#d4edda', '#c3e6cb'][k])
    labels = [
        '0  BASE LM -- continues text, ignores the ask',
        '1  early SFT -- knows topic, weak format',
        '2  mid SFT -- on-template, shorter & clearer',
        '3  SFT done -- answers like an assistant',
    ]
    stage_lbl.set_text(labels[k])
    for bar, val, txt in zip(bars6, SFT_METERS[k], txt6):
        bar.set_height(val)
        txt.set_y(val + 0.04)
        txt.set_text(f'{val:.2f}')
    hud6.set_text(
        'SFT alone:\n'
        '  + follow / format\n'
        '  - no ranking of two\n'
        '    OK answers\n'
        '  - still preference-\n'
        '    blind outside demos\n\n'
        'Next: RLHF / DPO'
    )
    fig6.canvas.draw_idle()


sl_sft.on_changed(_update_sft)
_update_sft()
print('Step 6: drag SFT progress -- watch base completer become an assistant.')
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
# FIGURE 9: RLHF pipeline -- SFT -> reward model -> PPO
# -------------------------------------------------------
fig9, axes9 = new_figure(1, 2)
fig9.suptitle(
    'Step 9: RLHF pipeline -- SFT  ->  Reward Model  ->  PPO  (step through)',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)
fig9.subplots_adjust(bottom=0.22, wspace=0.30)

RLHF_STAGES = [
    {
        'title': '1) Start from SFT policy',
        'blurb': 'Supervised chat model is the\ninitial policy AND the KL reference.',
        'highlight': 0,
        'reward': 0.25,
        'kl': 0.0,
        'pref': 0.35,
    },
    {
        'title': '2) Sample answers, humans rank',
        'blurb': 'For each prompt, sample A/B/C...\nLabelers mark preferences A > B > C.',
        'highlight': 1,
        'reward': 0.25,
        'kl': 0.0,
        'pref': 0.55,
    },
    {
        'title': '3) Train reward model R',
        'blurb': 'Fit R(prompt, answer) so preferred\nreplies score higher (Bradley-Terry).',
        'highlight': 2,
        'reward': 0.55,
        'kl': 0.05,
        'pref': 0.70,
    },
    {
        'title': '4) PPO policy update',
        'blurb': 'Maximize R minus beta * KL(policy\n|| SFT ref). Steer without collapsing.',
        'highlight': 3,
        'reward': 0.88,
        'kl': 0.35,
        'pref': 0.90,
    },
]

ax9a = axes9[0]
ax9a.set_xlim(0, 12)
ax9a.set_ylim(0, 10)
ax9a.axis('off')
ax9a.set_title('Pipeline (active stage highlighted)', fontsize=FS_AX_TITLE - 2)

stage_boxes = []
stage_specs = [
    (0.4, 7.2, 2.6, 2.0, 'SFT\npolicy', '#dceaff'),
    (3.4, 7.2, 2.6, 2.0, 'Sample +\nrank A>B', '#fff3cd'),
    (6.4, 7.2, 2.6, 2.0, 'Reward\nmodel R', '#e2d5f1'),
    (9.4, 7.2, 2.2, 2.0, 'PPO\nupdate', '#d4edda'),
]
for x, y, w, h, text, color in stage_specs:
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor=color, edgecolor='k', lw=2,
    )
    ax9a.add_patch(rect)
    ax9a.text(x + w / 2, y + h / 2, text, ha='center', va='center',
              fontsize=FS_ANNOT - 1, fontweight='bold')
    stage_boxes.append(rect)

for x0, x1 in [(3.0, 3.4), (6.0, 6.4), (9.0, 9.4)]:
    ax9a.annotate('', xy=(x1, 8.2), xytext=(x0, 8.2),
                  arrowprops=dict(arrowstyle='->', lw=2.5, color='seagreen'))

# feedback arrow PPO -> policy
ax9a.annotate(
    '', xy=(1.7, 7.2), xytext=(10.5, 7.2),
    arrowprops=dict(arrowstyle='->', lw=2, color='crimson',
                    connectionstyle='arc3,rad=0.55'),
)
ax9a.text(6.0, 5.85, 'updated weights', color='crimson',
          ha='center', fontsize=FS_ANNOT - 1, fontweight='bold')

blurb9 = ax9a.text(
    6.0, 3.6, '', ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
ax9a.text(
    6.0, 1.3,
    'Objective sketch:  E[R]  -  β · KL(π || π_SFT)\n'
    'Reward pulls toward preferences; KL keeps fluency / knowledge.',
    ha='center', va='center', fontsize=FS_ANNOT - 1,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

ax9b = axes9[1]
metric_names = ['Reward R', 'KL to SFT', 'Win-rate\nvs SFT']
bars9 = ax9b.bar(metric_names, [0.25, 0.0, 0.35],
                 color=['#9b59b6', '#e74c3c', '#27ae60'], edgecolor='k', width=0.65)
ax9b.set_ylim(0, 1.15)
ax9b.set_ylabel('toy units', fontsize=FS_AXIS)
ax9b.set_title('As RLHF proceeds...', fontsize=FS_AX_TITLE - 2)
ax9b.tick_params(axis='both', labelsize=FS_TICK - 2)
ax9b.grid(True, axis='y', alpha=0.3)
txt9 = [
    ax9b.text(i, 0.29, '', ha='center', fontsize=FS_ANNOT - 1)
    for i in range(3)
]
hud9 = ax9b.text(
    0.98, 0.98, '',
    transform=ax9b.transAxes, ha='right', va='top', fontsize=FS_ANNOT - 1,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
)

ax_rlhf = fig9.add_axes([0.15, 0.08, 0.45, 0.045])
sl_rlhf = Slider(ax_rlhf, 'RLHF stage', 0, 3, valinit=0, valstep=1, color='#9b59b6')
ax_btn_prev = fig9.add_axes([0.64, 0.075, 0.12, 0.05])
ax_btn_next = fig9.add_axes([0.78, 0.075, 0.12, 0.05])
btn_prev = Button(ax_btn_prev, 'Prev', color='#dee2e6', hovercolor='#adb5bd')
btn_next = Button(ax_btn_next, 'Next', color='#dee2e6', hovercolor='#adb5bd')


def _update_rlhf(_=None):
    k = int(sl_rlhf.val)
    st = RLHF_STAGES[k]
    for i, rect in enumerate(stage_boxes):
        rect.set_linewidth(4.5 if i == st['highlight'] else 2.0)
        rect.set_edgecolor('crimson' if i == st['highlight'] else 'k')
    blurb9.set_text(f"{st['title']}\n{st['blurb']}")
    vals = [st['reward'], st['kl'], st['pref']]
    for bar, val, txt in zip(bars9, vals, txt9):
        bar.set_height(val)
        txt.set_y(val + 0.04)
        txt.set_text(f'{val:.2f}')
    hud9.set_text(
        'Classic InstructGPT-style\n'
        'RLHF loop.\n\n'
        'Too little β (KL): reward\n'
        'hacking / style collapse.\n'
        'Too much β: barely moves\n'
        'off the SFT model.'
    )
    fig9.canvas.draw_idle()


def _rlhf_prev(_):
    sl_rlhf.set_val(max(0, int(sl_rlhf.val) - 1))


def _rlhf_next(_):
    sl_rlhf.set_val(min(3, int(sl_rlhf.val) + 1))


sl_rlhf.on_changed(_update_rlhf)
btn_prev.on_clicked(_rlhf_prev)
btn_next.on_clicked(_rlhf_next)
_update_rlhf()
print('Step 9: step through RLHF -- SFT, rank, reward model, PPO.')
show_maximized(fig9)


# -------------------------------------------------------
# FIGURE 10: Alignment HHH tradeoffs + DPO
# -------------------------------------------------------
fig10, axes10 = new_figure(1, 2)
fig10.suptitle(
    'Step 10: Alignment tradeoffs (HHH) + DPO as a simpler preference method',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)
fig10.subplots_adjust(bottom=0.28, wspace=0.30)

ax10a = axes10[0]
hhh_names = ['Helpful', 'Harmless', 'Honest']
hhh_colors = ['#5cb85c', '#f0ad4e', '#5b8def']
bars_hhh = ax10a.bar(
    hhh_names, [0.7, 0.7, 0.7], width=0.65,
    color=hhh_colors, edgecolor='k',
)
ax10a.set_ylim(0, 1.15)
ax10a.set_ylabel('emphasis', fontsize=FS_AXIS)
ax10a.set_title('Helpful / Harmless / Honest knobs', fontsize=FS_AX_TITLE - 2)
ax10a.tick_params(axis='both', labelsize=FS_TICK - 2)
ax10a.grid(True, axis='y', alpha=0.3)
txt_hhh = [
    ax10a.text(i, 0.74, '0.70', ha='center', fontsize=FS_ANNOT - 1)
    for i in range(3)
]
behavior10 = fig10.text(
    0.28, 0.20, '',
    ha='center', va='top', fontsize=FS_ANNOT - 1,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax10b = axes10[1]
ax10b.set_xlim(0, 10)
ax10b.set_ylim(0, 10)
ax10b.axis('off')
ax10b.set_title('DPO: preferences without PPO', fontsize=FS_AX_TITLE - 2)
box(ax10b, 0.5, 7.8, 9.0, 1.4, 'Prompt: "Summarize this article"', '#dceaff',
    fontsize=FS_ANNOT - 1)
box(ax10b, 0.5, 5.5, 4.2, 1.8, 'y_w  WINNER\n(preferred)', '#d4edda',
    fontsize=FS_ANNOT - 1)
box(ax10b, 5.3, 5.5, 4.2, 1.8, 'y_l  LOSER\n(rejected)', '#f8d7da',
    fontsize=FS_ANNOT - 1)
ax10b.text(
    5.0, 3.3,
    'DPO raises log π(y_w) relative to π(y_l)\n'
    'vs a frozen SFT reference -- same prefs,\n'
    'no reward model, no RL sampling loop.',
    ha='center', va='center', fontsize=FS_ANNOT - 1,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
ax10b.text(
    5.0, 1.2,
    'Menu: SFT · RLHF/PPO · DPO/IPO/KTO\n'
    'RLAIF · constitutional · best-of-N\n'
    'Open issues: over-refusal, sycophancy, reward hacking',
    ha='center', va='center', fontsize=FS_ANNOT - 1,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.95),
)

ax_h = fig10.add_axes([0.08, 0.14, 0.35, 0.035])
ax_ha = fig10.add_axes([0.08, 0.09, 0.35, 0.035])
ax_ho = fig10.add_axes([0.08, 0.04, 0.35, 0.035])
sl_help = Slider(ax_h, 'Helpful', 0.05, 1.0, valinit=0.70, valstep=0.05, color='#5cb85c')
sl_harm = Slider(ax_ha, 'Harmless', 0.05, 1.0, valinit=0.70, valstep=0.05, color='#f0ad4e')
sl_hon = Slider(ax_ho, 'Honest', 0.05, 1.0, valinit=0.70, valstep=0.05, color='#5b8def')


def _hhh_behavior(h, s, o):
    # toy conflict vignettes from the three knobs
    if s >= 0.85 and h >= 0.75:
        return ('High Harmless + Helpful: may OVER-REFUSE\n'
                'borderline asks, or hedge every answer.')
    if h >= 0.85 and o <= 0.35:
        return ('High Helpful, low Honest: SYCOPHANCY risk --\n'
                'eager answers that invent details.')
    if o >= 0.85 and h <= 0.40:
        return ('High Honest, low Helpful: safe hedges &\n'
                '"I don\'t know" -- under-helps the user.')
    if s >= 0.85 and o >= 0.75 and h <= 0.45:
        return ('Harmless+Honest dominate: cautious refusals,\n'
                'citations, narrow scope -- less can-do vibe.')
    if abs(h - s) < 0.1 and abs(h - o) < 0.1:
        return ('Balanced HHH: the usual chat-assistant target.\n'
                'Real systems still argue over the weights.')
    return ('Mixed tradeoff: alignment shapes style & refusals;\n'
            'it does not magically add new world knowledge.')


def _update_hhh(_=None):
    vals = [float(sl_help.val), float(sl_harm.val), float(sl_hon.val)]
    for bar, val, txt, x in zip(bars_hhh, vals, txt_hhh, (1.5, 5.0, 8.5)):
        bar.set_height(val)
        txt.set_y(val + 0.04)
        txt.set_text(f'{val:.2f}')
    behavior10.set_text(_hhh_behavior(*vals))
    fig10.canvas.draw_idle()


sl_help.on_changed(_update_hhh)
sl_harm.on_changed(_update_hhh)
sl_hon.on_changed(_update_hhh)
_update_hhh()
print('Step 10: drag HHH knobs -- see alignment tradeoffs; DPO on the right.')
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
    'Same stack as 260_transformer.py -- pushed into\n'
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
  260_transformer.py -- causal vs bidirectional attention
  350_positional_encoding.py -- PE / RoPE used in modern GPTs
  410_BERT.py -- the encoder / fine-tune fork of the same family
""")
