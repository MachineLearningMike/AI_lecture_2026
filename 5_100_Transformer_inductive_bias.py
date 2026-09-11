"""
0920_Transformer_inductive_bias.py

What inductive biases does a Transformer bring?

Inductive bias = preferences baked into the architecture that guide
generalization when data alone underdetermine the answer
(see 0690_inductive_bias.py).

Transformer biases (high level):
  1) Token-in-parallel processing with SHARED weights across positions
  2) Content-based routing (attention), not fixed local kernels
  3) Global pairwise interactions in one layer
  4) Soft mixing (convex combination of values)
  5) Order sensitivity only through positional encoding / masks
  6) Autoregressive bias when causally masked (decoder / GPT)
  7) Residual additive accumulation of layer writes
  8) Weaker locality / translation biases than CNNs (unless added)

Companion to 0850_transformer.py, 0880_positional_encoding.py,
0890_chain_of_adds.py, 0900_model_dimension.py.
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


# Equivariance demo shared by Step 2 and the summary printout
_EQ_E = np.array([
    [1.0, 0.1],
    [0.2, 1.0],
    [-0.8, 0.3],
], dtype=float)
_EQ_SCORES = _EQ_E @ _EQ_E.T / np.sqrt(_EQ_E.shape[1])
_EQ_Y = softmax(_EQ_SCORES, axis=-1) @ _EQ_E
_EQ_PERM = np.array([2, 0, 1])
_EQ_Y_P = softmax(
    _EQ_E[_EQ_PERM] @ _EQ_E[_EQ_PERM].T / np.sqrt(_EQ_E.shape[1]), axis=-1
) @ _EQ_E[_EQ_PERM]
_EQ_Y_P_UNPERM = np.empty_like(_EQ_Y_P)
_EQ_Y_P_UNPERM[_EQ_PERM] = _EQ_Y_P
err = np.max(np.abs(_EQ_Y - _EQ_Y_P_UNPERM))


# -------------------------------------------------------
# FIGURE 1: Bias catalog overview
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        'Step 1: Transformer inductive biases at a glance',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 8)
    ax1.axis('off')

    cards = [
        (0.4, 4.6, 'Shared weights\nacross positions', '#dceaff'),
        (4.2, 4.6, 'Content-based\nrouting (QK)', '#f8d7da'),
        (8.0, 4.6, 'Global pairwise\nmixing in 1 layer', '#d4edda'),
        (0.4, 1.2, 'Soft convex\ncombination of V', '#fff3cd'),
        (4.2, 1.2, 'Order via PE / mask\n(not via convolution)', '#e2d5f1'),
        (8.0, 1.2, 'Residual stream\nadditive writes', '#ffe0b2'),
    ]
    for x, y, text, color in cards:
        ax1.add_patch(patches.FancyBboxPatch(
            (x, y), 3.5, 2.4, boxstyle='round,pad=0.02,rounding_size=0.12',
            facecolor=color, edgecolor='k', lw=2,
        ))
        ax1.text(x + 1.75, y + 1.2, text, ha='center', va='center',
                 fontsize=FS_ANNOT + 1, fontweight='bold')

    plt.tight_layout()
    return fig1



# -------------------------------------------------------
# FIGURE 2: Shared weights / permutation without PE
# -------------------------------------------------------
def build_step_2():
    fig2, axes2 = new_figure(1, 2)
    fig2.suptitle(
        'Step 2: Without positions, attention is permutation-equivariant',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    tokens = np.array(['A', 'B', 'C'])
    # simple embeddings
    E = np.array([
        [1.0, 0.1],
        [0.2, 1.0],
        [-0.8, 0.3],
    ], dtype=float)


    def attn_pool(X):
        # self-attention with W=I for illustration
        scores = X @ X.T / np.sqrt(X.shape[1])
        A = softmax(scores, axis=-1)
        return A @ X, A


    Y, A = attn_pool(E)
    perm = np.array([2, 0, 1])
    Y_p, A_p = attn_pool(E[perm])
    # unpermute outputs to compare
    Y_p_unperm = np.empty_like(Y_p)
    Y_p_unperm[perm] = Y_p

    ax = axes2[0]
    ax.imshow(A, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(3), tokens)
    ax.set_yticks(range(3), tokens)
    ax.set_title('Attention on order [A,B,C]', fontsize=FS_AX_TITLE)
    ax.set_xlabel('key', fontsize=FS_AXIS)
    ax.set_ylabel('query', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)

    ax = axes2[1]
    ax.imshow(A_p, cmap='viridis', vmin=0, vmax=1)
    ax.set_xticks(range(3), tokens[perm])
    ax.set_yticks(range(3), tokens[perm])
    ax.set_title('Attention on shuffled [C,A,B]', fontsize=FS_AX_TITLE)
    ax.set_xlabel('key', fontsize=FS_AXIS)
    ax.set_ylabel('query', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_note(
        ax,
        'Same pairwise content\n'
        'relations, just relocated.\n\n'
        'Bias: unless PE/mask\n'
        'is added, ORDER is not\n'
        'built into attention.',
        y=0.02, va='bottom', facecolor='lightyellow',
    )

    fig2.text(
        0.5, 0.02,
        f'Equivariance check: max |Y - unpermute(Y_shuffled)| = {err:.2e}',
        ha='center', fontsize=FS_ANNOT,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig2



# -------------------------------------------------------
# FIGURE 3: Content routing vs CNN locality bias
# -------------------------------------------------------
def build_step_3():
    fig3, axes3 = new_figure(1, 2)
    fig3.suptitle(
        'Step 3: CNN bias = local fixed filters; Transformer bias = learned routing',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    # Synthetic 1D signal with distant dependency: label depends on matching ends
    T = 16
    x = np.linspace(0, 1, T)
    signal = np.sin(2 * np.pi * x)
    signal[0] = 2.0
    signal[-1] = 2.0  # two spikes far apart that "belong together"

    ax = axes3[0]
    ax.stem(np.arange(T), signal, linefmt='C0-', markerfmt='C0o', basefmt='k-')
    # CNN receptive field boxes
    for c in [3, 7, 11]:
        ax.axvspan(c - 1.5, c + 1.5, color='crimson', alpha=0.15)
    ax.set_title('CNN-like: local windows', fontsize=FS_AX_TITLE)
    ax.set_xlabel('position', fontsize=FS_AXIS)
    ax.set_ylabel('activation', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_note(
        ax,
        'Local kernels need\n'
        'DEPTH (or dilation)\n'
        'to link distant spikes.',
        y=0.98, va='top', facecolor='lightyellow',
    )
    ax.grid(True, alpha=0.3)

    ax = axes3[1]
    # Content-match illustration: attention from position 0 follows signal peaks
    scores = signal * signal[0]
    A = softmax(scores)
    ax.stem(np.arange(T), signal, linefmt='C7-', markerfmt='C7o', basefmt='k-')
    ax.plot(np.arange(T), A * 3, 's-', color='seagreen', lw=2.2, label='attn from pos 0')
    ax.set_title('Transformer-like: jump to matching content', fontsize=FS_AX_TITLE)
    ax.set_xlabel('position', fontsize=FS_AXIS)
    ax.set_ylabel('value / attention', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, alpha=0.3)
    add_note(
        ax,
        'Bias: relate tokens by\n'
        'SIMILARITY of queries\n'
        'and keys, anywhere in\n'
        'the sequence.',
        y=0.98, va='top',
    )

    plt.tight_layout()
    return fig3



# -------------------------------------------------------
# FIGURE 4: Soft mixing bias
# -------------------------------------------------------
def build_step_4():
    fig4, axes4 = new_figure(1, 2)
    fig4.suptitle(
        'Step 4: Softmax bias -- outputs are soft mixtures, not hard pointers',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    values = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.5],
    ])
    labels = ['V1', 'V2', 'V3']
    # two different attention distributions
    A_sharp = np.array([0.85, 0.10, 0.05])
    A_diffuse = np.array([0.34, 0.33, 0.33])
    out_sharp = A_sharp @ values
    out_diffuse = A_diffuse @ values

    ax = axes4[0]
    ax.bar(labels, A_sharp, color='crimson', edgecolor='k')
    ax.set_ylim(0, 1)
    ax.set_title('Peaked attention', fontsize=FS_AX_TITLE)
    ax.set_ylabel('weight', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.text(
        0.5, 0.9,
        f'output ≈ {np.round(out_sharp, 2)}',
        transform=ax.transAxes, ha='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
    )
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes4[1]
    ax.bar(labels, A_diffuse, color='royalblue', edgecolor='k')
    ax.set_ylim(0, 1)
    ax.set_title('Diffuse attention', fontsize=FS_AX_TITLE)
    ax.set_ylabel('weight', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.text(
        0.5, 0.9,
        f'output ≈ {np.round(out_diffuse, 2)}',
        transform=ax.transAxes, ha='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
    )
    ax.grid(True, axis='y', alpha=0.3)
    add_note(
        ax,
        'Architectural prior:\n'
        'each update is a\n'
        'CONVEX MIX of values.\n\n'
        'Hard selection must be\n'
        'approximated by sharp\n'
        'softmax (or variants).',
        y=0.45, va='center', facecolor='lightyellow',
    )

    plt.tight_layout()
    return fig4



# -------------------------------------------------------
# FIGURE 5: Causal decoder bias
# -------------------------------------------------------
def build_step_5():
    fig5, axes5 = new_figure(1, 2)
    fig5.suptitle(
        'Step 5: Causal masking adds an autoregressive inductive bias',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    T = 6
    full = np.ones((T, T))
    causal = np.tril(np.ones((T, T)))

    ax = axes5[0]
    ax.imshow(full, cmap='Greens', vmin=0, vmax=1)
    ax.set_title('Encoder / bidirectional\n(no generation constraint)', fontsize=FS_AX_TITLE)
    ax.set_xlabel('key position', fontsize=FS_AXIS)
    ax.set_ylabel('query position', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_note(
        ax,
        'Bias toward holistic\n'
        'representation of a\n'
        'whole input.',
        x=0.02, y=0.98, ha='left', va='top', facecolor='lightyellow',
    )

    ax = axes5[1]
    ax.imshow(causal, cmap='Blues', vmin=0, vmax=1)
    for i in range(T):
        for j in range(i + 1, T):
            ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, ec='red', lw=1.5))
    ax.set_title('Decoder / causal\n(next-token generation bias)', fontsize=FS_AX_TITLE)
    ax.set_xlabel('key position', fontsize=FS_AXIS)
    ax.set_ylabel('query position', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_note(
        ax,
        'Bias: predict the future\n'
        'from the past only.\n'
        'Enables left-to-right\n'
        'generation (GPT-style).',
        x=0.02, y=0.98, ha='left', va='top', facecolor='lightyellow',
    )

    plt.tight_layout()
    return fig5



# -------------------------------------------------------
# FIGURE 6: Residual / depth / width biases
# -------------------------------------------------------
def build_step_6():
    fig6, axes6 = new_figure(1, 3)
    fig6.suptitle(
        'Step 6: Other built-in preferences -- residual depth, PE, width',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    ax = axes6[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Residual bias', fontsize=FS_AX_TITLE)
    ax.text(
        0.5, 0.5,
        'Layers prefer to ADD\n'
        'small updates to a\n'
        'persistent stream\n'
        'rather than rewrite it.\n\n'
        '(see 0890_chain_of_adds.py)',
        transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
        bbox=dict(boxstyle='round', facecolor='#ffe0b2', alpha=0.95),
    )

    ax = axes6[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Positional bias', fontsize=FS_AX_TITLE)
    ax.text(
        0.5, 0.5,
        'Order/distance structure\n'
        'enters only through PE\n'
        '(sinusoidal, RoPE, ...)\n'
        'or relative biases.\n\n'
        '(see 0880_positional_encoding.py)',
        transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
        bbox=dict(boxstyle='round', facecolor='#e2d5f1', alpha=0.95),
    )

    ax = axes6[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Width / capacity bias', fontsize=FS_AX_TITLE)
    ax.text(
        0.5, 0.5,
        'd_model sets how wide\n'
        'the shared maps are.\n'
        'Larger d => more capacity,\n'
        'more overfit risk.\n\n'
        '(see 0900_model_dimension.py)',
        transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
        bbox=dict(boxstyle='round', facecolor='#d4edda', alpha=0.95),
    )

    plt.tight_layout()
    return fig6



# -------------------------------------------------------
# FIGURE 7: Comparison table vs CNN / RNN
# -------------------------------------------------------
def build_step_7():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        'Step 7: Transformer biases compared with CNN / RNN priors',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    ax7.axis('off')

    table = [
        ['Prior', 'CNN', 'RNN', 'Transformer'],
        ['Locality', 'strong (kernels)', 'weak / sequential', 'weak unless sparse/rel-PE'],
        ['Long-range links', 'needs depth', 'path length ~T', 'path length 1 (attn)'],
        ['Position handling', 'equivariance/grid', 'time order baked in', 'via PE / causal mask'],
        ['Parameter sharing', 'over space', 'over time', 'over positions (same W)'],
        ['Interaction type', 'fixed local filter', 'state recursion', 'content-based routing'],
    ]

    y0 = 0.84
    for r, row in enumerate(table):
        y = y0 - r * 0.11
        bg = '#e8eef8' if r == 0 else ('#f8f8f8' if r % 2 == 0 else 'white')
        ax7.add_patch(patches.Rectangle(
            (0.04, y - 0.04), 0.92, 0.095, transform=ax7.transAxes,
            facecolor=bg, edgecolor='0.7', lw=1,
        ))
        xs = [0.06, 0.28, 0.50, 0.72]
        for x, cell in zip(xs, row):
            ax7.text(
                x, y + 0.01, cell, transform=ax7.transAxes,
                fontsize=FS_AXIS if r else FS_AXIS + 2,
                fontweight='bold' if r == 0 else 'normal',
                va='center', ha='left',
            )

    ax7.text(
        0.5, 0.08,
        'No free lunch: Transformers trade CNN locality for flexible global routing.\n'
        'That is powerful for language, and often needs PE / data / regularization\n'
        'to recover structures CNNs assumed for free.',
        transform=ax7.transAxes, ha='center', va='center', fontsize=FS_AXIS,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )

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
print("\n" + "=" * 60)
print("TRANSFORMER INDUCTIVE BIASES")
print("=" * 60)
print(f"""
1) Shared weights across positions
   Same W is reused at every token (capacity set by d, not by T).

2) Content-based routing
   Attention links tokens by Q-K similarity, not by a fixed neighborhood.

3) Global pairwise mixing
   Any token can influence any other in a single layer.

4) Soft mixing
   Outputs are softmax-weighted averages of values (soft ensemble / blur).

5) Order is injected, not intrinsic
   Without PE, attention is permutation-equivariant.
   Sinusoidal/RoPE/ALiBi/causal masks supply order/distance structure.

6) Causal decoder bias
   Masking forbids future tokens => autoregressive generation prior.

7) Residual additive updates
   Depth accumulates writes into a residual stream rather than overwriting.

8) Compared with CNNs
   Weaker locality / translation priors; stronger set-routing priors.
   Useful when relations are semantic/long-range rather than local.

Equivariance demo error (should be ~0): {err:.2e}
""")
