"""
160_positional_encoding.py

Sinusoidal positional encoding (Vaswani et al., 2017) -- intuition.

Problem:
    Self-attention is permutation-invariant: without positions, a Transformer
    sees only a bag of tokens.

Sinusoidal solution:
    For position pos and dimension i (0-based among d_model dims):

        PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

Properties worth remembering:
    1) Deterministic -- no learned parameters
    2) Each position gets a unique geometric fingerprint
    3) Relative offsets are linear in the PE space (phase-shift / rotation)
    4) Locally linear / smooth in position (easy to learn)
    5) Extrapolates to sequence lengths longer than those seen in training

RoPE (rotary PE) applies the SAME rotation idea directly to Q/K instead
of adding a PE vector. Same geometric family as sinusoidal PE; RoPE is
the usual choice in modern decoder LLMs because relative phase lives in
the attention logit and pairs well with length-transfer recipes.

Companion to 150_transformer.py and 200_video_transformer.py.
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


def sinusoidal_pe(n_positions, d_model, base=10000.0):
    """Return PE matrix of shape (n_positions, d_model)."""
    pos = np.arange(n_positions)[:, None]                 # (T, 1)
    i = np.arange(d_model)[None, :]                       # (1, d)
    angles = pos / (base ** (2 * (i // 2) / d_model))
    pe = np.zeros((n_positions, d_model), dtype=float)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


def wavelengths(d_model, base=10000.0):
    """Geometric wavelength for each sin/cos pair."""
    i = np.arange(0, d_model, 2)
    return 2 * np.pi * (base ** (i / d_model))


N_POS = 100
D_MODEL = 64
BASE = 10000.0

PE = sinusoidal_pe(N_POS, D_MODEL, base=BASE)
WAVES = wavelengths(D_MODEL, base=BASE)


# -------------------------------------------------------
# FIGURE 1: Why position is needed
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    'Step 1: Attention alone does not know token order',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

tokens_a = ['dog', 'bites', 'man']
tokens_b = ['man', 'bites', 'dog']
# bag-of-vectors view: same multiset of embeddings => identical attention pool
embed = {
    'dog': np.array([0.9, 0.1]),
    'bites': np.array([0.2, 0.8]),
    'man': np.array([-0.7, 0.3]),
}

ax = axes1[0]
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-0.2, 1.2)
for i, tok in enumerate(tokens_a):
    x, y = embed[tok]
    ax.scatter([x], [y], s=180, zorder=3)
    ax.text(x, y + 0.08, f'{i}:{tok}', ha='center', fontsize=FS_ANNOT, fontweight='bold')
ax.set_title('Sentence A: dog bites man', fontsize=FS_AX_TITLE)
ax.set_xlabel('embed dim 0', fontsize=FS_AXIS)
ax.set_ylabel('embed dim 1', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Content vectors only.\n'
    'Order is invisible to\n'
    'plain attention.',
    y=0.98, va='top', facecolor='lightyellow',
)

ax = axes1[1]
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-0.2, 1.2)
for i, tok in enumerate(tokens_b):
    x, y = embed[tok]
    ax.scatter([x], [y], s=180, zorder=3)
    ax.text(x, y + 0.08, f'{i}:{tok}', ha='center', fontsize=FS_ANNOT, fontweight='bold')
ax.set_title('Sentence B: man bites dog', fontsize=FS_AX_TITLE)
ax.set_xlabel('embed dim 0', fontsize=FS_AXIS)
ax.set_ylabel('embed dim 1', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Same three points!\n'
    'Meaning flipped, but\n'
    'bag-of-tokens view\n'
    'cannot tell A from B.\n\n'
    '=> inject positions.',
    y=0.98, va='top',
)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: The PE heatmap -- sin/cos across dims
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    r'Step 2: $PE_{(pos,2i)}=\sin(pos/\theta_i),\; PE_{(pos,2i+1)}=\cos(pos/\theta_i)$',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

im = ax2.imshow(PE, aspect='auto', cmap='twilight', interpolation='nearest')
ax2.set_xlabel('embedding dimension', fontsize=FS_AXIS)
ax2.set_ylabel('token position', fontsize=FS_AXIS)
ax2.set_title(f'Sinusoidal PE matrix  (T={N_POS}, d={D_MODEL})', fontsize=FS_AX_TITLE)
ax2.tick_params(axis='both', labelsize=FS_TICK)
cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
cbar.set_label('PE value', fontsize=FS_ANNOT)
cbar.ax.tick_params(labelsize=FS_ANNOT)
add_note(
    ax2,
    'Left dims: high frequency\n'
    '(rapid oscillation).\n\n'
    'Right dims: low frequency\n'
    '(slow waves along pos).\n\n'
    'Together: unique code\n'
    'for every position.',
    y=0.98, va='top', facecolor='lightyellow',
)

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: A few dimensions as waves over position
# -------------------------------------------------------
fig3, axes3 = new_figure(2, 2)
fig3.suptitle(
    'Step 3: Each dimension pair is a sinusoid with its own wavelength',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

pos = np.arange(N_POS)
dim_pairs = [0, 2, 10, 30]  # even indices -> sin channel of that pair
for ax, dim in zip(axes3.flat, dim_pairs):
    ax.plot(pos, PE[:, dim], color='crimson', lw=2.2, label=f'sin dim {dim}')
    ax.plot(pos, PE[:, dim + 1], color='royalblue', lw=2.2, label=f'cos dim {dim + 1}')
    wave = WAVES[dim // 2]
    ax.set_title(rf'dims ({dim},{dim + 1})  |  $\lambda \approx {wave:.1f}$',
                 fontsize=FS_AX_TITLE - 2)
    ax.set_xlabel('position', fontsize=FS_AXIS)
    ax.set_ylabel('PE', fontsize=FS_AXIS)
    ax.set_ylim(-1.2, 1.2)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc='lower right')

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Geometric progression of wavelengths
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    r'Step 4: Wavelengths grow geometrically via $10000^{2i/d}$',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes4[0]
pair_idx = np.arange(len(WAVES))
ax.semilogy(pair_idx, WAVES, 'o-', color='seagreen', lw=2.5)
ax.set_xlabel('pair index i', fontsize=FS_AXIS)
ax.set_ylabel('wavelength (in token steps)', fontsize=FS_AXIS)
ax.set_title('From ~2π to ~2π·10000 across dimensions', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3, which='both')
add_note(
    ax,
    'Multi-scale clock:\n'
    'fine dims resolve nearby\n'
    'positions; coarse dims\n'
    'encode long-range place.',
    y=0.98, va='top', facecolor='lightyellow',
)

ax = axes4[1]
# show angle frequency omega_i = 1 / 10000^{2i/d}
i = np.arange(0, D_MODEL, 2)
omega = 1.0 / (BASE ** (i / D_MODEL))
ax.semilogy(i, omega, 's-', color='darkorange', lw=2.5)
ax.set_xlabel('dimension index (even)', fontsize=FS_AXIS)
ax.set_ylabel(r'angular frequency $\omega_i$', fontsize=FS_AXIS)
ax.set_title(r'$\mathrm{PE}=\sin(\omega_i\cdot pos),\ \cos(\omega_i\cdot pos)$',
             fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3, which='both')
add_note(
    ax,
    'Same idea as a mixed-\n'
    'radix positional numeral:\n'
    'many frequencies combine\n'
    'into one address.',
    y=0.98, va='top',
)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Relative position / phase-shift property
# -------------------------------------------------------
fig5, axes5 = new_figure(1, 2)
fig5.suptitle(
    'Step 5: Relative offsets become linear phase shifts in PE space',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# For a fixed frequency, PE(pos+k) is a rotation of PE(pos) in the (sin,cos) plane
ax = axes5[0]
dim = 2  # one frequency pair
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), color='0.7', lw=1.0, label='unit circle')
# sample a few positions
sample_pos = [0, 5, 10, 20, 35]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sample_pos)))
for p, c in zip(sample_pos, colors):
    # PE uses sin on even, cos on odd -- plot that pair
    y_s, x_c = PE[p, dim], PE[p, dim + 1]
    ax.scatter([x_c], [y_s], s=70, color=c, zorder=3, label=f'pos={p}')
    ax.plot([0, x_c], [0, y_s], color=c, lw=1.5)
ax.set_aspect('equal')
ax.set_xlabel(f'PE[{dim + 1}] (cos)', fontsize=FS_AXIS)
ax.set_ylabel(f'PE[{dim}] (sin)', fontsize=FS_AXIS)
ax.set_title('One frequency lives on a circle', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND - 1, loc='lower left')
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Moving by Δpos rotates\n'
    'the (sin,cos) pair by\n'
    'ω·Δpos.\n\n'
    'Relative distance <=>\n'
    'predictable rotation.',
    y=0.98, va='top', facecolor='lightyellow',
)

ax = axes5[1]
# Soft evidence: cosine similarity of PE vectors vs position distance
ref = 20
sims = (PE @ PE[ref]) / (np.linalg.norm(PE, axis=1) * np.linalg.norm(PE[ref]) + 1e-8)
ax.plot(pos, sims, color='royalblue', lw=2.5)
ax.axvline(ref, color='crimson', ls='--', lw=1.5, label=f'reference pos={ref}')
ax.set_xlabel('position', fontsize=FS_AXIS)
ax.set_ylabel('cosine similarity to PE[ref]', fontsize=FS_AXIS)
ax.set_title('Nearby positions have similar PE vectors', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Attention can learn to\n'
    'use these similarities\n'
    'as soft relative-distance\n'
    'cues.',
    y=0.98, va='top',
)

plt.tight_layout()
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 5b: Local linearity -- why it helps learning + extrapolation
# -------------------------------------------------------
fig5b, axes5b = new_figure(1, 3)
fig5b.suptitle(
    'Step 5b: PE is locally linear\n'
    'that speeds learning and aids extrapolation',
    fontsize=FS_FIGURE_TITLE - 4, fontweight='bold', y=0.98,
)

# --- Panel 1: local linear approx of a PE coordinate ---
ax = axes5b[0]
dim_lin = 6
pos_c = 40
window = np.arange(pos_c - 8, pos_c + 9)
pe_vals = PE[window, dim_lin]
# finite-difference slope at center
slope = (PE[pos_c + 1, dim_lin] - PE[pos_c - 1, dim_lin]) / 2.0
linear_approx = PE[pos_c, dim_lin] + slope * (window - pos_c)
ax.plot(window, pe_vals, 'o-', color='royalblue', lw=2.2, label='true PE dim')
ax.plot(window, linear_approx, '--', color='crimson', lw=2.2, label='local linear approx')
ax.axvline(pos_c, color='0.4', ls=':', lw=1.2)
ax.set_xlabel('position', fontsize=FS_AXIS)
ax.set_ylabel(f'PE[:, {dim_lin}]', fontsize=FS_AXIS)
ax.set_title(
    'Local linear approx\n'
    r'$PE(p+\delta)\approx PE(p)+\delta PE\'(p)$',
    fontsize=FS_AX_TITLE - 6,
)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc='best')
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Sin/cos are smooth.\n'
    'For small steps, the\n'
    'encoding changes almost\n'
    'like a straight line.\n'
    'Networks love that.',
    y=0.98, va='top', facecolor='lightyellow',
)

# --- Panel 2: relative offset = ONE linear map, reusable everywhere ---
ax = axes5b[1]
# Show ||PE(p+k) - R_k PE(p)|| ~ 0 for many p, fixed k (using one freq pair is exact;
# for full vector, relative structure still supports linear relative readout)
k_rel = 5
# Build block-diagonal rotation that maps each (sin,cos) pair by omega_i * k
def rotate_pe_rows(pe_rows, k):
    out = np.zeros_like(pe_rows)
    for i in range(0, pe_rows.shape[1], 2):
        omega_i = 1.0 / (BASE ** (i / pe_rows.shape[1]))
        ang = omega_i * k
        s, c = pe_rows[:, i], pe_rows[:, i + 1]
        out[:, i] = s * np.cos(ang) + c * np.sin(ang)
        out[:, i + 1] = c * np.cos(ang) - s * np.sin(ang)
    return out


ps = np.arange(0, N_POS - k_rel)
pe_p = PE[ps]
pe_pk = PE[ps + k_rel]
pe_pred = rotate_pe_rows(pe_p, k_rel)
err = np.linalg.norm(pe_pk - pe_pred, axis=1)
ax.plot(ps, err, color='seagreen', lw=2.3)
ax.set_ylim(-0.05, max(1e-6, err.max() * 1.2) + 1e-9)
ax.set_xlabel('absolute position p', fontsize=FS_AXIS)
ax.set_ylabel(r'$\|PE(p+k)-R_k PE(p)\|$', fontsize=FS_AXIS)
ax.set_title(
    rf'Same linear map $R_{{k={k_rel}}}$' + '\n'
    'works for every p',
    fontsize=FS_AX_TITLE - 6,
)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Learning "attend +k"\n'
    'is learning ONE linear\n'
    'relation on PE, not a\n'
    'separate rule per\n'
    'absolute index.\n\n'
    '=> fewer patterns to fit\n'
    '=> faster learning.',
    y=0.98, va='top',
)

# --- Panel 3: extrapolation beyond training length ---
ax = axes5b[2]
train_T = 50
extra_T = 120
PE_long = sinusoidal_pe(extra_T, D_MODEL, base=BASE)
# Compare structured PE continuation vs "no structure" baseline:
# freeze PE[:train_T] and pretend learned PE would be undefined / random after
rng_ex = np.random.default_rng(0)
learned_like = np.zeros((extra_T, D_MODEL))
learned_like[:train_T] = PE_long[:train_T]
learned_like[train_T:] = rng_ex.normal(0, 0.7, size=(extra_T - train_T, D_MODEL))

# similarity of position t to position 10, as a probe of usable structure
ref = 10
sim_sin = (PE_long @ PE_long[ref]) / (
    np.linalg.norm(PE_long, axis=1) * np.linalg.norm(PE_long[ref]) + 1e-8
)
sim_rand = (learned_like @ learned_like[ref]) / (
    np.linalg.norm(learned_like, axis=1) * np.linalg.norm(learned_like[ref]) + 1e-8
)
ax.plot(np.arange(extra_T), sim_sin, color='royalblue', lw=2.5, label='sinusoidal PE')
ax.plot(np.arange(extra_T), sim_rand, color='crimson', lw=2.0, alpha=0.85,
        label='unstructured after train_T')
ax.axvline(train_T, color='0.3', ls='--', lw=1.5, label=f'train length T={train_T}')
ax.set_xlabel('position', fontsize=FS_AXIS)
ax.set_ylabel(f'cos-sim to PE[{ref}]', fontsize=FS_AXIS)
ax.set_title('Extrapolation past\nthe training context', fontsize=FS_AX_TITLE - 6)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND - 1, loc='best')
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Sinusoids are defined\n'
    'for ANY length.\n'
    'Relative rotations keep\n'
    'working past train_T.\n\n'
    'Arbitrary learned PE\n'
    'vectors usually do not\n'
    'extrapolate.',
    y=0.98, va='top', facecolor='lightyellow',
)

fig5b.tight_layout(rect=[0, 0, 1, 0.92])
show_maximized(fig5b)


# -------------------------------------------------------
# FIGURE 6: Adding PE to token embeddings
# -------------------------------------------------------
fig6, axes6 = new_figure(1, 3)
fig6.suptitle(
    'Step 6: Transformers add PE to token embeddings:  X = Embed(tokens) + PE',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

T_show = 32
D_show = 32
# fake token embeddings + PE
tok_embed = np.random.default_rng(0).normal(0, 0.4, size=(T_show, D_show))
pe_show = sinusoidal_pe(T_show, D_show)
mixed = tok_embed + pe_show

panels = [
    (tok_embed, 'Token embeddings\n(content only)', 'crimson'),
    (pe_show, 'Sinusoidal PE\n(position only)', 'seagreen'),
    (mixed, 'Sum embeddings\n(content + position)', 'royalblue'),
]
vmax = np.percentile(np.abs(mixed), 98)
for ax, (mat, title, _) in zip(axes6, panels):
    im = ax.imshow(mat, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=FS_AX_TITLE - 2)
    ax.set_xlabel('dim', fontsize=FS_AXIS)
    ax.set_ylabel('position', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK - 2)

axes6[2].text(
    0.02, 0.02,
    'Attention / FFN then process this sum.\n'
    'Order sensitivity comes from the PE part.',
    transform=axes6[2].transAxes, ha='left', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: PE enables parallel processing (vs RNN sequential)
# -------------------------------------------------------
fig7, axes7 = new_figure(1, 2)
fig7.suptitle(
    'Step 7: PE lets Transformers encode order WITHOUT sequential RNN steps',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

# Left: RNN must run t=1,2,3,... in order
ax = axes7[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('RNN: order comes from time', fontsize=FS_AX_TITLE)

# timeline of hidden states
ys = [6.5, 5.0, 3.5, 2.0]
labs = [r'$h_1$', r'$h_2$', r'$h_3$', r'$h_4$']
toks = [r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$']
for i, (y, h, tok) in enumerate(zip(ys, labs, toks)):
    ax.add_patch(patches.Circle((3.0, y), 0.45, facecolor='#f8d7da', edgecolor='k', lw=2))
    ax.text(3.0, y, h, ha='center', va='center', fontsize=FS_ANNOT, fontweight='bold')
    ax.add_patch(patches.FancyBboxPatch(
        (0.7, y - 0.35), 1.2, 0.7, boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor='#dceaff', edgecolor='k', lw=1.5,
    ))
    ax.text(1.3, y, tok, ha='center', va='center', fontsize=FS_ANNOT)
    ax.annotate('', xy=(2.55, y), xytext=(1.95, y),
                arrowprops=dict(arrowstyle='->', lw=1.8, color='0.3'))
    if i < len(ys) - 1:
        ax.annotate('', xy=(3.0, ys[i + 1] + 0.45), xytext=(3.0, y - 0.45),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='crimson'))

ax.text(5.2, 4.5,
        'Must wait for $h_{t-1}$\nbefore computing $h_t$.\n\n'
        'Order is baked into\nthe sequential scan.\n\n'
        'Hard to parallelize\nacross time.',
        ha='left', va='center', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

# Right: Transformer all positions at once + PE
ax = axes7[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Transformer: order comes from PE', fontsize=FS_AX_TITLE)

xs = [1.5, 3.5, 5.5, 7.5]
for i, (x, tok) in enumerate(zip(xs, toks)):
    ax.add_patch(patches.FancyBboxPatch(
        (x - 0.55, 5.8), 1.1, 0.9, boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor='#dceaff', edgecolor='k', lw=1.5,
    ))
    ax.text(x, 6.25, tok, ha='center', va='center', fontsize=FS_ANNOT, fontweight='bold')
    # PE badge
    ax.add_patch(patches.FancyBboxPatch(
        (x - 0.55, 4.5), 1.1, 0.8, boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor='#d4edda', edgecolor='seagreen', lw=2,
    ))
    ax.text(x, 4.9, rf'PE({i+1})', ha='center', va='center',
            fontsize=FS_ANNOT - 1, color='seagreen', fontweight='bold')
    ax.annotate('', xy=(x, 5.8), xytext=(x, 5.3),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='seagreen'))

# big parallel attention block
ax.add_patch(patches.FancyBboxPatch(
    (1.2, 1.8), 7.0, 1.8, boxstyle='round,pad=0.02,rounding_size=0.12',
    facecolor='#fff3cd', edgecolor='k', lw=2,
))
ax.text(4.7, 2.7,
        'Self-Attention + FFN\nALL positions in ONE parallel step',
        ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold')

for x in xs:
    ax.annotate('', xy=(x, 3.6), xytext=(x, 4.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='0.3'))

ax.text(
    0.5, 0.06,
    'PE injects "I am position t" into each token UP FRONT.\n'
    'Then every token can be processed together -- no left-to-right wait.',
    transform=ax.transAxes, ha='center', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Beauty of sinusoidal PE -- and is it unavoidable?
# -------------------------------------------------------
fig8, axes8 = new_figure(1, 2)
fig8.suptitle(
    'Step 8: The beauty of sinusoidal PE -- and why it is NOT unavoidable',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes8[0]
ax.axis('off')
ax.set_title('Why sinusoidal PE is beautiful', fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.52,
    '1) Parameter-free geometry\n'
    '   Position is a closed-form code, not a lookup table.\n\n'
    '2) Multi-scale clock\n'
    '   Geometric wavelengths give coarse + fine address bits\n'
    '   in one vector -- like mixed-radix place value.\n\n'
    '3) Relative position = linear algebra\n'
    '   PE(p+k) = R_k PE(p)  (block rotations).\n'
    '   "Attend +k" can be one reusable linear rule.\n\n'
    '4) Local smoothness\n'
    '   Small moves in text => small moves in PE space.\n'
    '   Easy to interpolate; stable for optimizers.\n\n'
    '5) Extrapolation for free\n'
    '   The same formula works past any training length.\n\n'
    'Beauty = strong inductive bias with almost no knobs.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 2,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.96),
)

ax = axes8[1]
ax.axis('off')
ax.set_title('Is sinusoidal PE unavoidable?', fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.55,
    'No. Attention only needs SOME position signal.\n'
    'Sinusoidal PE is one elegant choice, not the only one.\n\n'
    'Common alternatives:\n'
    '  • Learned absolute PE (original GPT/BERT style)\n'
    '      flexible, but often weak at long extrapolation\n'
    '  • Relative attention bias (T5)\n'
    '      put distance into attention logits directly\n'
    '  • RoPE (LLaMA / many modern LLMs)\n'
    '      rotate Q/K by position -- same geometry family\n'
    '      (see Step 9: tradeoffs vs sinusoidal)\n'
    '  • ALiBi\n'
    '      simple distance penalty; strong length transfer\n'
    '  • No PE + causal mask only\n'
    '      sometimes enough for pure left-to-right LMs\n\n'
    'So: PE (or an equivalent bias) is usually necessary\n'
    'for order-sensitive tasks, but the SINUSOIDAL form\n'
    'is optional -- a beautiful prior, not a law of nature.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 9: RoPE intuition -- same family as sinusoidal, different tradeoffs
# -------------------------------------------------------
def rope_rotate_2d(xy, pos, omega):
    """Rotate 2D vectors by angle = pos * omega (RoPE on one pair)."""
    ang = pos * omega
    c, s = np.cos(ang), np.sin(ang)
    x, y = xy[..., 0], xy[..., 1]
    return np.stack([x * c - y * s, x * s + y * c], axis=-1)


fig9, axes9 = new_figure(1, 2)
fig9.suptitle(
    'Step 9: RoPE intuition -- same rotations as sinusoidal, different packaging',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

# Left: rotate Q and K by absolute position; dot product depends on relative
ax = axes9[0]
ax.set_aspect('equal')
theta = np.linspace(0, 2 * np.pi, 300)
ax.plot(np.cos(theta), np.sin(theta), color='0.75', lw=1.0)

# Fixed content vectors (as if one 2D head slice of Q/K before rotation)
q_content = np.array([0.85, 0.20])
k_content = np.array([0.70, 0.45])
omega_rope = 0.12
m_pos, n_pos = 2, 7  # query at m, key at n
q_rot = rope_rotate_2d(q_content, m_pos, omega_rope)
k_rot = rope_rotate_2d(k_content, n_pos, omega_rope)

ax.annotate('', xy=q_content, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='0.45', lw=1.8))
ax.annotate('', xy=k_content, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='0.45', lw=1.8))
ax.annotate('', xy=q_rot, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='crimson', lw=2.4))
ax.annotate('', xy=k_rot, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='royalblue', lw=2.4))
ax.scatter(*q_content, s=40, c='0.45', zorder=3)
ax.scatter(*k_content, s=40, c='0.45', zorder=3)
ax.scatter(*q_rot, s=70, c='crimson', zorder=4)
ax.scatter(*k_rot, s=70, c='royalblue', zorder=4)
ax.text(q_content[0] + 0.05, q_content[1] - 0.12, r'$q$ content',
        color='0.35', fontsize=FS_ANNOT - 1)
ax.text(k_content[0] + 0.05, k_content[1] + 0.08, r'$k$ content',
        color='0.35', fontsize=FS_ANNOT - 1)
ax.text(q_rot[0] + 0.05, q_rot[1] - 0.05, rf'$R_m q$ (m={m_pos})',
        color='crimson', fontsize=FS_ANNOT, fontweight='bold')
ax.text(k_rot[0] + 0.05, k_rot[1] + 0.05, rf'$R_n k$ (n={n_pos})',
        color='royalblue', fontsize=FS_ANNOT, fontweight='bold')

# Arc hint for relative angle
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.15, 1.25)
ax.set_xlabel('dim pair 0', fontsize=FS_AXIS)
ax.set_ylabel('dim pair 1', fontsize=FS_AXIS)
ax.set_title('RoPE: rotate Q/K by absolute position', fontsize=FS_AX_TITLE - 2)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    r'Attention score uses' + '\n'
    r'$(R_m q)^\top (R_n k)$' + '\n'
    r'$= q^\top R_{n-m} k$' + '\n\n'
    'Only the RELATIVE\n'
    'offset (n-m) remains.\n'
    'Absolute m,n cancel.',
    y=0.98, va='top', facecolor='lightyellow',
)

# Right: same relative structure as sinusoidal; packaging tradeoffs
ax = axes9[1]
# Demonstrate: score vs relative distance is identical for any absolute anchor
rel = np.arange(-20, 21)
# Fix content; score as function of (n-m) only
scores = []
for d in rel:
    # q at 0, k at d  (equivalent to any m, n=m+d by RoPE identity)
    qr = rope_rotate_2d(q_content, 0, omega_rope)
    kr = rope_rotate_2d(k_content, d, omega_rope)
    scores.append(float(qr @ kr))
scores = np.array(scores)

# Sanity: same curve if we shift both by +10
scores_shift = []
for d in rel:
    qr = rope_rotate_2d(q_content, 10, omega_rope)
    kr = rope_rotate_2d(k_content, 10 + d, omega_rope)
    scores_shift.append(float(qr @ kr))
scores_shift = np.array(scores_shift)

ax.plot(rel, scores, color='royalblue', lw=2.6, label=r'query at $m=0$')
ax.plot(rel, scores_shift, '--', color='crimson', lw=2.2, label=r'query at $m=10$ (same)')
ax.axvline(0, color='0.4', ls=':', lw=1.2)
ax.set_xlabel(r'relative offset $n - m$', fontsize=FS_AXIS)
ax.set_ylabel(r'attention logit $q^\top R_{n-m} k$', fontsize=FS_AXIS)
ax.set_title('Relative-only scores (absolute cancels)', fontsize=FS_AX_TITLE - 2)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc='best')
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Shared geometry:\n'
    '  PE(p+k)=R_k PE(p)\n'
    '  relative = rotation.\n\n'
    'RoPE puts that\n'
    'rotation on Q/K so\n'
    'relative phase is in\n'
    'the attention logit.\n\n'
    'Tradeoff: sinusoidal\n'
    'add-PE is simple to\n'
    'teach; RoPE is the\n'
    'usual modern default.',
    y=0.98, va='top', facecolor='lightyellow',
)

fig9.text(
    0.5, 0.02,
    'Sinusoidal: add PE to x, then learn relative from PE geometry.   |   '
    'RoPE: bake the same rotations into Q/K so relative is automatic.\n'
    'Same inductive-bias family; RoPE is preferred in many decoder LLMs for relative logits / length recipes -- not a wholly new principle.',
    ha='center', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.96),
)

fig9.tight_layout(rect=[0, 0.08, 1, 1])
show_maximized(fig9)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
# Sanity: uniqueness of rows
uniq = np.unique(np.round(PE, 10), axis=0).shape[0]
# Relative rotation check for one frequency
dim = 4
omega = 1.0 / (BASE ** (dim / D_MODEL))
delta = 7
# PE even=sin, odd=cos. Rotation by omega*delta:
# [sin(p+d)] = sin p cos d + cos p sin d
# [cos(p+d)] = cos p cos d - sin p sin d
p0 = 15
s0, c0 = PE[p0, dim], PE[p0, dim + 1]
s1, c1 = PE[p0 + delta, dim], PE[p0 + delta, dim + 1]
s_pred = s0 * np.cos(omega * delta) + c0 * np.sin(omega * delta)
c_pred = c0 * np.cos(omega * delta) - s0 * np.sin(omega * delta)
rot_err = max(abs(s_pred - s1), abs(c_pred - c1))

# RoPE relative identity check
rope_rel_err = float(np.max(np.abs(scores - scores_shift)))

print("\n" + "=" * 60)
print("SINUSOIDAL POSITIONAL ENCODING")
print("=" * 60)
print(f"""
Formula (d_model={D_MODEL}, base={BASE:g}):
  PE(pos, 2i)   = sin(pos / {BASE:g}^{{2i/d}})
  PE(pos, 2i+1) = cos(pos / {BASE:g}^{{2i/d}})

Why it exists:
  Self-attention is a set operation over tokens. PE injects ORDER.

Parallelism vs RNN:
  RNNs get order "for free" by computing h_t from h_(t-1) -- but that
  forces sequential time steps (cannot fully parallelize across t).
  Transformers compute all positions together; PE supplies "I am at
  position t" up front so order does not require a left-to-right scan.

What the waves do:
  - Low-frequency dims: coarse absolute location
  - High-frequency dims: fine local distinctions
  - Wavelengths span ~[{WAVES.min():.2f}, {WAVES.max():.1f}] token steps

Nice mathematical property:
  PE(pos+delta) at one frequency is a rotation of PE(pos).
  Demo rotation error for delta={delta} at dim pair ({dim},{dim+1}): {rot_err:.2e}

Local linearity:
  PE is smooth in pos, so PE(pos+delta) ~= PE(pos) + delta * dPE/dpos
  for small delta. That smooth / locally linear geometry is easy for
  linear layers and attention to exploit.

Why this helps learning speed:
  1) Nearby positions share similar PE => interpolate instead of memorize
  2) "Attend to relative offset +k" is ONE linear map R_k on PE,
     reused at every absolute position => fewer patterns to fit

Why this helps extrapolation:
  1) The sin/cos formula is defined for ANY length, even beyond train_T
  2) The same relative map R_k still holds for unseen absolute positions
  Learned lookup-table PE usually breaks outside the trained range.

The beauty:
  Parameter-free multi-scale codes + exact relative linear structure
  + smoothness + extrapolation -- a strong inductive bias with almost
  no trainable knobs.

Is sinusoidal PE unavoidable?
  No. Some form of position/order signal is usually needed (otherwise
  attention is permutation-equivariant). But the signal can be:
    learned absolute PE, relative bias (T5), RoPE, ALiBi, or even
    mostly causal masking in decoder-only LMs.
  Sinusoidal PE is optional elegance -- especially great for teaching
  and as a baseline -- not a unique mathematical necessity.

RoPE (rotary positional embedding) -- intuition:
  Instead of ADDING a PE vector to embeddings, RoPE ROTATES each
  2D slice of Q and K by an angle proportional to absolute position:
      q_m = R_m q ,   k_n = R_n k
  Then the attention score depends only on the relative offset:
      (R_m q)^T (R_n k) = q^T R_(n-m) k
  Demo: scores for query at m=0 vs m=10 overlay (max |diff|={rope_rel_err:.2e}).

RoPE vs sinusoidal -- tradeoffs:
  Sinusoidal PE already uses the same multi-frequency rotation geometry
  (PE(p+k) = R_k PE(p)). RoPE applies those rotations to Q/K so relative
  structure is automatic in the attention logit.
  Same inductive-bias family. Sinusoidal add-PE is parameter-free and
  great for teaching; RoPE is the usual choice in current GPT-style
  models and pairs well with modern length-transfer recipes.

Uniqueness check:
  Distinct PE rows among first {N_POS} positions: {uniq}/{N_POS}

Usage in a Transformer:
  X = TokenEmbed(tokens) + PE[:T]
""")
