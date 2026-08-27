"""
270_chain_of_adds.py

How Transformer's chained residual additions form a cumulative mix.

In a Transformer block (simplified):

    x_{l+1} = x_l + F_l(x_l)

where F_l is attention and/or the feed-forward sublayer (LayerNorm details
omitted for clarity). Unrolling the recurrence:

    x_L = x_0 + F_0(x_0) + F_1(x_1) + ... + F_{L-1}(x_{L-1})

So the final residual stream is a RUNNING SUM of layer updates.
Early token embeddings are never erased -- later layers ADD refinements
into the same stream. That is the "chain of adds".

Beyond the fixed add x <- x + F(x), we can also LEARN how much each
layer contributes (LayerScale, ReZero, gated residuals):

    x_{l+1} = x_l + alpha_l * F_l(x_l)

The learned alphas are a simple readout of which layers write strongly
into the residual -- and help deep nets start near identity.

Companion to 190_transformer.py and 260_positional_encoding.py.
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


# -------------------------------------------------------
# Toy residual stream in 2D: each layer adds a delta vector
# -------------------------------------------------------
L = 6
# Start embedding
x0 = np.array([0.8, 0.2], dtype=float)

# Hand-designed layer updates (as if attn/FFN wrote these deltas)
deltas = np.array([
    [0.35, 0.10],   # layer 0: lexical / local
    [0.10, 0.45],   # layer 1: syntax-ish
    [-0.05, 0.35],  # layer 2
    [0.40, -0.05],  # layer 3: semantic pull
    [0.15, 0.25],   # layer 4
    [0.05, 0.15],   # layer 5: small refinement
], dtype=float)

# Residual stream states
stream = [x0.copy()]
for d in deltas:
    stream.append(stream[-1] + d)
stream = np.stack(stream, axis=0)  # (L+1, 2)

# Also a NON-residual "overwrite" baseline for contrast
overwrite = [x0.copy()]
for d in deltas:
    # pretend layer replaces representation with a transform of previous
    overwrite.append(0.3 * overwrite[-1] + d)
overwrite = np.stack(overwrite, axis=0)


# -------------------------------------------------------
# FIGURE 1: Residual vs overwrite
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    r'Step 1: Residual update $x\!\leftarrow\!x+F(x)$ keeps a running memory',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes1[0]
ax.plot(stream[:, 0], stream[:, 1], 'o-', color='royalblue', lw=2.5, ms=8, label='residual stream')
for i in range(L):
    ax.annotate(
        '', xy=stream[i + 1], xytext=stream[i],
        arrowprops=dict(arrowstyle='->', color='crimson', lw=2.0),
    )
    mid = 0.5 * (stream[i] + stream[i + 1])
    ax.text(mid[0] + 0.03, mid[1] + 0.03, rf'$\Delta_{i}$', color='crimson', fontsize=FS_ANNOT)
ax.scatter(stream[0, 0], stream[0, 1], s=120, c='orange', zorder=5, label=r'$x_0$ (embed+PE)')
ax.scatter(stream[-1, 0], stream[-1, 1], s=140, c='k', marker='*', zorder=5, label=r'$x_L$ final')
ax.set_title('Residual chain of adds', fontsize=FS_AX_TITLE)
ax.set_xlabel('residual dim 0', fontsize=FS_AXIS)
ax.set_ylabel('residual dim 1', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc='best')
ax.grid(True, alpha=0.3)
add_note(
    ax,
    r'$x_L = x_0 + \sum_l \Delta_l$' + '\n\n'
    'Every layer contributes\n'
    'an additive edit to the\n'
    'SAME stream.',
    y=0.50, va='center', facecolor='lightyellow',
)

ax = axes1[1]
ax.plot(overwrite[:, 0], overwrite[:, 1], 'o-', color='0.45', lw=2.5, ms=8, label='overwrite path')
ax.scatter(overwrite[0, 0], overwrite[0, 1], s=120, c='orange', zorder=5)
ax.scatter(overwrite[-1, 0], overwrite[-1, 1], s=140, c='k', marker='*', zorder=5)
ax.set_title('If layers overwrote instead', fontsize=FS_AX_TITLE)
ax.set_xlabel('dim 0', fontsize=FS_AXIS)
ax.set_ylabel('dim 1', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Early information is\n'
    'partially erased each\n'
    'layer.\n\n'
    'Residual nets avoid\n'
    'that forced forgetting.',
    y=0.98, va='top',
)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Unrolled equation as stacked blocks
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    'Step 2: Unrolling the recurrence -- final state is an explicit sum',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 8)
ax2.axis('off')

# draw summands as bars
colors = plt.cm.viridis(np.linspace(0.15, 0.9, L + 1))
labels = [r'$x_0$'] + [rf'$\Delta_{i}=F_{i}(x_{i})$' for i in range(L)]
# magnitudes for bar widths ~ L2 of contribution
mags = np.concatenate([[np.linalg.norm(x0)], np.linalg.norm(deltas, axis=1)])
mags = mags / mags.sum()

x_cursor = 1.0
y = 4.0
h = 1.6
pieces = []
for mag, lab, c in zip(mags, labels, colors):
    w = 1.0 + 4.5 * mag
    rect = patches.FancyBboxPatch(
        (x_cursor, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.08',
        facecolor=c, edgecolor='k', lw=1.5,
    )
    ax2.add_patch(rect)
    ax2.text(x_cursor + w / 2, y + h / 2, lab, ha='center', va='center',
             fontsize=FS_ANNOT, color='white', fontweight='bold')
    pieces.append(x_cursor + w)
    x_cursor += w + 0.25
    if mag != mags[-1]:
        ax2.text(x_cursor - 0.15, y + h / 2, '+', ha='center', va='center',
                 fontsize=FS_AXIS, fontweight='bold')

ax2.text(
    6, 2.2,
    r'$x_L = x_0 + \Delta_0 + \Delta_1 + \cdots + \Delta_{L-1}$',
    ha='center', fontsize=FS_AXIS,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
)
ax2.text(
    6, 0.9,
    'Attention/FFN blocks do not replace x; they WRITE UPDATES into x.\n'
    'This is why people call x the residual stream / communication channel.',
    ha='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Multi-token residual streams mixing via attention deltas
# -------------------------------------------------------
# Toy: 4 tokens, 2D residual each. Attention deltas move info across tokens.
tokens = ['the', 'cat', 'sat', 'mat']
n_tok = len(tokens)
# initial embeddings
X = np.array([
    [0.6, 0.1],
    [1.0, 0.2],
    [0.2, 0.9],
    [0.3, 1.1],
], dtype=float)

# Layer deltas: simulate "sat" receiving subject info from "cat", etc.
layer_deltas = [
    # layer 0: mostly local FFN-ish
    np.array([[0.05, 0.00], [0.08, 0.02], [0.02, 0.08], [0.02, 0.05]]),
    # layer 1: attention copies cat -> sat
    np.array([[0.00, 0.00], [-0.05, 0.00], [0.35, -0.05], [0.00, 0.00]]),
    # layer 2: attention copies mat -> sat (object)
    np.array([[0.00, 0.00], [0.00, 0.00], [0.05, 0.30], [-0.05, -0.05]]),
    # layer 3: small cleanup
    np.array([[0.02, 0.01], [0.02, 0.01], [0.04, 0.04], [0.02, 0.02]]),
]

states = [X.copy()]
for d in layer_deltas:
    states.append(states[-1] + d)
states = np.stack(states, axis=0)  # (L+1, T, 2)

fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    'Step 3: Each token has a residual stream;\n'
    'attention ADDS cross-token messages',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
)

ax = axes3[0]
colors = ['steelblue', 'crimson', 'seagreen', 'darkorange']
for t, (tok, c) in enumerate(zip(tokens, colors)):
    traj = states[:, t, :]
    ax.plot(traj[:, 0], traj[:, 1], 'o-', color=c, lw=2.2, ms=6, label=tok)
    ax.scatter(traj[0, 0], traj[0, 1], s=50, color=c, edgecolor='k', zorder=4)
    ax.scatter(traj[-1, 0], traj[-1, 1], s=90, color=c, marker='*', zorder=5)
# show the big attention delta into "sat" from layer 1
ax.annotate(
    'attn write:\ncat -> sat',
    xy=states[2, 2], xytext=(0.55, 0.55),
    fontsize=FS_ANNOT, color='crimson',
    arrowprops=dict(arrowstyle='->', color='crimson', lw=1.8),
)
ax.set_title('Token trajectories in residual space', fontsize=FS_AX_TITLE)
ax.set_xlabel('dim 0', fontsize=FS_AXIS)
ax.set_ylabel('dim 1', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc='upper left')
ax.grid(True, alpha=0.3)

ax = axes3[1]
# stacked area: contribution norms per layer into token "sat"
sat = 2
contrib = np.concatenate([
    [np.linalg.norm(X[sat])],
    [np.linalg.norm(d[sat]) for d in layer_deltas],
])
labels = ['x0'] + [f'D{i}' for i in range(len(layer_deltas))]
ax.bar(labels, contrib, color=plt.cm.plasma(np.linspace(0.2, 0.85, len(contrib))),
       edgecolor='k')
ax.set_title('Additive contributions into token \"sat\"', fontsize=FS_AX_TITLE)
ax.set_ylabel(r'$\|\mathrm{contribution}\|_2$', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, axis='y', alpha=0.3)
add_note(
    ax,
    'Final sat vector =\n'
    'embed + local FFNs\n'
    '+ attention writes\n'
    'from other tokens.\n\n'
    'All via addition.',
    y=0.98, va='top', facecolor='lightyellow',
)

plt.tight_layout(rect=[0, 0, 1, 0.90])
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Cumulative mix curves (how much of final is from each layer)
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    'Step 4: Cumulative mix -- later reads see ALL previous writes',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# For the 2D single-stream toy, project each prefix sum's alignment with final
final = stream[-1]
prefix_align = []
for i in range(len(stream)):
    # energy of prefix relative to final
    prefix_align.append(np.dot(stream[i], final) / (np.linalg.norm(final) ** 2))
prefix_align = np.array(prefix_align)

ax = axes4[0]
ax.plot(np.arange(len(stream)), prefix_align, 'o-', color='royalblue', lw=2.5)
ax.set_xlabel('after layer k (0 = embedding only)', fontsize=FS_AXIS)
ax.set_ylabel('fraction of final (projection)', fontsize=FS_AXIS)
ax.set_title('How the residual fills in over depth', fontsize=FS_AX_TITLE)
ax.set_ylim(0, 1.05)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Depth is not a pipe that\n'
    'replaces features.\n'
    'It is an integrator that\n'
    'accumulates features.',
    y=0.50, va='center', facecolor='lightyellow',
)

ax = axes4[1]
# Layer contribution share for single stream
shares = np.concatenate([[np.linalg.norm(x0)], np.linalg.norm(deltas, axis=1)])
shares = shares / shares.sum()
ax.pie(
    shares,
    labels=['embed'] + [f'L{i}' for i in range(L)],
    autopct='%1.0f%%',
    colors=plt.cm.viridis(np.linspace(0.15, 0.9, L + 1)),
    textprops={'fontsize': FS_ANNOT},
)
ax.set_title('Share of total additive mass by source', fontsize=FS_AX_TITLE)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Block diagram emphasizing Add paths
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    'Step 5: Inside one block -- two additive writes into the stream',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax5.set_xlim(0, 12)
ax5.set_ylim(0, 8)
ax5.axis('off')


def box(ax, x, y, w, h, text, color):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor=color, edgecolor='k', lw=2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=FS_ANNOT, fontweight='bold')


box(ax5, 4.5, 6.7, 3.0, 0.7, 'x_l  (residual stream)', '#dceaff')
box(ax5, 0.8, 4.8, 3.0, 1.0, 'Self-Attention\n(+ LN)', '#f8d7da')
box(ax5, 4.5, 4.9, 3.0, 0.7, 'Add', '#fff3cd')
box(ax5, 8.2, 4.8, 3.0, 1.0, 'FFN / MLP\n(+ LN)', '#d4edda')
box(ax5, 4.5, 3.0, 3.0, 0.7, 'Add', '#fff3cd')
box(ax5, 4.5, 1.2, 3.0, 0.7, 'x_{l+1}', '#dceaff')

# arrows
ax5.annotate('', xy=(2.3, 5.8), xytext=(5.5, 6.7),
             arrowprops=dict(arrowstyle='->', lw=2))
ax5.annotate('', xy=(9.7, 5.8), xytext=(6.5, 6.7),
             arrowprops=dict(arrowstyle='->', lw=2))
ax5.annotate('', xy=(4.5, 5.25), xytext=(3.8, 5.25),
             arrowprops=dict(arrowstyle='->', lw=2, color='crimson'))
ax5.annotate('', xy=(7.5, 5.25), xytext=(8.2, 5.25),
             arrowprops=dict(arrowstyle='->', lw=2, color='crimson'))
# residual bypasses
ax5.annotate('', xy=(6.0, 5.6), xytext=(6.0, 6.7),
             arrowprops=dict(arrowstyle='->', lw=2, color='royalblue'))
ax5.annotate('', xy=(6.0, 3.7), xytext=(6.0, 4.9),
             arrowprops=dict(arrowstyle='->', lw=2, color='royalblue'))
ax5.annotate('', xy=(6.0, 1.9), xytext=(6.0, 3.0),
             arrowprops=dict(arrowstyle='->', lw=2))

ax5.text(6.2, 6.1, 'bypass', color='royalblue', fontsize=FS_ANNOT)
ax5.text(1.0, 3.6, 'Attention writes a delta\n(message mix from tokens)',
         fontsize=FS_ANNOT,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
ax5.text(8.0, 3.6, 'FFN writes a delta\n(per-token non-linear think)',
         fontsize=FS_ANNOT,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
ax5.text(
    6.0, 0.45,
    r'$x_{l+1} = x_l + Attn(x_l) + FFN(x_l+\cdots)$   (schematic)',
    ha='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Gradient highway intuition
# -------------------------------------------------------
fig6, axes6 = new_figure(1, 2)
fig6.suptitle(
    'Step 6: Chain of adds is also a gradient highway',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

depths = np.arange(1, 13)
# toy: without residual, gradient shrinks exponentially; with residual, stays O(1)
grad_plain = 0.7 ** depths
grad_res = np.ones_like(depths, dtype=float) * 0.9  # roughly preserved via identity path

ax = axes6[0]
ax.semilogy(depths, grad_plain, 'o-', color='crimson', lw=2.5, label='no residual (vanish risk)')
ax.semilogy(depths, grad_res, 's-', color='royalblue', lw=2.5, label='with residual bypass')
ax.set_xlabel('depth from output', fontsize=FS_AXIS)
ax.set_ylabel('relative gradient magnitude (toy)', fontsize=FS_AXIS)
ax.set_title('Identity path keeps early layers trainable', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3, which='both')
add_note(
    ax,
    r'$\partial x_{l+1}/\partial x_l$' + '\n'
    r'$= I + \partial F_l/\partial x_l$' + '\n\n'
    'The I term is the\n'
    'additive highway.',
    y=0.98 - 1.0 / 3.0, va='top', facecolor='lightyellow',
)

ax = axes6[1]
ax.axis('off')
ax.set_title('Takeaway checklist', fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.55,
    '1. Each layer ADDS an update; it does not overwrite x.\n\n'
    '2. Final residual = embedding + sum of all writes\n'
    '   (attention messages + FFN refinements).\n\n'
    '3. Attention writes move information BETWEEN tokens\n'
    '   by adding into their streams.\n\n'
    '4. The same additive structure also stabilizes deep\n'
    '   training via an identity gradient path.\n\n'
    '5. Beyond fixed add: we can LEARN how much each\n'
    '   layer contributes (LayerScale, gates, alphas).\n\n'
    'This is the Transformer "chain of adds".',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_AXIS - 2,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Beyond simple add -- learn layer contributions
# -------------------------------------------------------
# Same deltas as Step 1, but with learned per-layer scales alpha_l
# (LayerScale / ReZero / gated residual style).
alphas = np.array([0.15, 0.55, 0.35, 1.20, 0.80, 0.25], dtype=float)
scaled_deltas = alphas[:, None] * deltas
stream_scaled = [x0.copy()]
for d in scaled_deltas:
    stream_scaled.append(stream_scaled[-1] + d)
stream_scaled = np.stack(stream_scaled, axis=0)

fig7, axes7 = new_figure(1, 2)
fig7.suptitle(
    r'Step 7: Beyond simple add -- learn how much each layer contributes',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes7[0]
# Equal-weight (simple add) vs learned alpha
layer_idx = np.arange(L)
w = 0.38
ax.bar(layer_idx - w / 2, np.ones(L), width=w, color='0.75', edgecolor='k',
       label=r'simple add: $\alpha_l = 1$')
ax.bar(layer_idx + w / 2, alphas, width=w, color='royalblue', edgecolor='k',
       label=r'learned: $\alpha_l$')
ax.axhline(1.0, color='0.4', ls='--', lw=1.2)
ax.set_xticks(layer_idx)
ax.set_xticklabels([f'L{i}' for i in range(L)])
ax.set_xlabel('layer', fontsize=FS_AXIS)
ax.set_ylabel(r'contribution scale $\alpha_l$', fontsize=FS_AXIS)
ax.set_title(r'Learnable residual weights $x\!\leftarrow\!x+\alpha_l F_l(x)$',
             fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc='upper left')
ax.grid(True, axis='y', alpha=0.3)
add_note(
    ax,
    'LayerScale / ReZero /\n'
    'gated residuals:\n'
    'train alpha (or a gate)\n'
    'per layer or channel.\n\n'
    'Init small -> deep nets\n'
    'start near identity.',
    y=0.98, va='top', facecolor='lightyellow',
)

ax = axes7[1]
ax.plot(stream[:, 0], stream[:, 1], 'o--', color='0.55', lw=2.0, ms=7,
        label=r'simple: $x+\Delta$')
ax.plot(stream_scaled[:, 0], stream_scaled[:, 1], 'o-', color='royalblue',
        lw=2.5, ms=8, label=r'learned: $x+\alpha\odot\Delta$')
for i in range(L):
    mid = 0.5 * (stream_scaled[i] + stream_scaled[i + 1])
    ax.text(mid[0] + 0.02, mid[1] + 0.02, rf'$\alpha_{i}$',
            color='crimson', fontsize=FS_ANNOT)
ax.scatter(stream_scaled[0, 0], stream_scaled[0, 1], s=120, c='orange',
           zorder=5, label=r'$x_0$')
ax.scatter(stream_scaled[-1, 0], stream_scaled[-1, 1], s=140, c='k',
           marker='*', zorder=5, label=r'$x_L$ (scaled)')
ax.set_title('Same deltas, different mix -> different final', fontsize=FS_AX_TITLE)
ax.set_xlabel('residual dim 0', fontsize=FS_AXIS)
ax.set_ylabel('residual dim 1', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc='best')
ax.grid(True, alpha=0.3)
add_note(
    ax,
    r'$x_L = x_0 + \sum_l \alpha_l\,\Delta_l$' + '\n\n'
    'Alphas reveal which\n'
    'layers write strongly\n'
    'into the stream -- and\n'
    'can be read out after\n'
    'training.',
    y=0.02, va='bottom', facecolor='lightyellow',
)

plt.tight_layout()
show_maximized(fig7)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
recon = x0 + deltas.sum(axis=0)
recon_err = np.linalg.norm(recon - stream[-1])
recon_scaled = x0 + scaled_deltas.sum(axis=0)
recon_scaled_err = np.linalg.norm(recon_scaled - stream_scaled[-1])

print("\n" + "=" * 60)
print("TRANSFORMER CHAIN OF ADDS")
print("=" * 60)
print(f"""
Layer rule (schematic):
  x_{{l+1}} = x_l + F_l(x_l)

Unrolled:
  x_L = x_0 + sum_{{l=0..L-1}} F_l(x_l)

Demo reconstruction error (x0 + sum deltas vs final stream): {recon_err:.2e}
Learned-scale recon error (x0 + sum alpha*deltas):          {recon_scaled_err:.2e}

Interpretation:
  - x is a residual stream / bus
  - Attention and FFN modules are writers onto that bus
  - Depth accumulates a MIX of many layer contributions
  - Information from early layers remains available unless
    later writes actively cancel it

Beyond the simple add:
  - Default is alpha_l = 1 for every layer (fixed mix)
  - We CAN learn contribution scales:
      x <- x + alpha_l * F_l(x)     (LayerScale / ReZero)
      or a soft gate g_l in (0,1)
  - After training, the alphas (or gates) tell us which
    layers write strongly into the residual -- a simple
    readout of "what contributes how"
  - Small init of alpha also helps deep nets start near
    the identity map (stable early training)

Why this matters:
  1) Representational: features compose by accumulation
  2) Algorithmic: attention can implement message passing by adds
  3) Optimization: identity skip keeps gradients healthy
  4) Interpretability / control: learnable scales expose and
     reshape each layer's residual contribution

See also:
  190_transformer.py -- what attention writes
  180_neural_non_linearity.py -- what FFN writes
  160_normalization.py -- LayerNorm around the writes
""")
