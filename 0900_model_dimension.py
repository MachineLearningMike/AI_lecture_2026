"""
0900_model_dimension.py

Model dimension (d_model / width) vs sequence length (T).

Key facts for Transformers / sequence models:
  1) Parameter count is set by WIDTH (and depth, vocab), NOT by T.
       Attention projections: O(d^2)
       FFN:                   O(d * d_ff) ~ O(d^2)
       Token embedding:       O(V * d)
  2) Sequence length T changes COMPUTE / memory of attention (O(T^2 d)),
     but does not add new weight matrices.
  3) Therefore d_model is a primary knob for CAPACITY:
       too small -> underfit
       too large -> overfit (especially with limited data)

Companion to 0850_transformer.py and 0690_inductive_bias.py.
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


# -------------------------------------------------------
# Parameter accounting for a GPT-style Transformer
#
# Param formula (sinusoidal / no learned PE; Pre-LN blocks):
#   embedding + per-block terms; ignores final ln_f (~2d) and other smalls.
#
#   Let d = d_model,  L = n_layers,  V = vocab_size,
#       d_ff = expansion width (default 4d).
#
#   Token embedding:          V * d
#   Per layer:
#     Attention Q,K,V,O:      4 * d * d          (+ 4*d biases)
#     FFN W1,W2:              2 * d * d_ff       (+ d_ff + d biases)
#     LayerNorm x2:           4 * d              (gamma,beta each)
#   Output LM head:           0 if tied to embed, else V * d
#
#   With biases and d_ff = 4d, tied head:
#     P_layer(d) = 12 d^2 + 13 d
#     P(d)       = V d + L (12 d^2 + 13 d)
#
#   Sequence length T does NOT appear.
# -------------------------------------------------------
DEFAULT_VOCAB = 32000
DEFAULT_LAYERS = 6


def transformer_params(d_model, n_layers=DEFAULT_LAYERS, n_heads=8, d_ff=None,
                       vocab=DEFAULT_VOCAB, tie_embeddings=True, include_bias=True):
    """
    Exact count under the assumptions in the module docstring formula.
    n_heads does not change the count when Q/K/V/O are full d x d matrices.
    """
    d = int(d_model)
    if d_ff is None:
        d_ff = 4 * d

    embed = vocab * d

    attn_w = 4 * d * d
    attn_b = 4 * d if include_bias else 0
    attn = attn_w + attn_b

    ffn_w = 2 * d * d_ff
    ffn_b = (d_ff + d) if include_bias else 0
    ffn = ffn_w + ffn_b

    ln = 4 * d  # two LayerNorms: gamma+beta each

    per_layer = attn + ffn + ln
    layers = n_layers * per_layer
    head = 0 if tie_embeddings else vocab * d
    total = embed + layers + head

    return {
        'embed': embed,
        'attn_all_layers': n_layers * attn,
        'ffn_all_layers': n_layers * ffn,
        'ln_all_layers': n_layers * ln,
        'head': head,
        'total': total,
        'per_layer': per_layer,
        'd_ff': d_ff,
        'include_bias': include_bias,
    }


def params_closed_form(d, L=DEFAULT_LAYERS, V=DEFAULT_VOCAB, d_ff_ratio=4,
                       tie_embeddings=True, include_bias=True):
    """Closed form P(d) matching transformer_params (d_ff = ratio * d)."""
    d = int(d)
    d_ff = d_ff_ratio * d
    embed = V * d
    if include_bias:
        per_layer = 4 * d * d + 2 * d * d_ff + d_ff + 9 * d  # 4d attn bias + d_ff + d + 4d LN
    else:
        per_layer = 4 * d * d + 2 * d * d_ff + 4 * d
    head = 0 if tie_embeddings else V * d
    return embed + L * per_layer + head


def attention_flops(T, d_model, n_layers=1):
    """Dominant attention matmul cost scale: n_layers * T^2 * d (ignore constants)."""
    return n_layers * (T ** 2) * d_model


# -------------------------------------------------------
# FIGURE 1: Params grow with d, flat in T
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    'Step 1: Parameters track model dimension -- not sequence length',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

d_grid = np.array([64, 128, 256, 384, 512, 768, 1024])
params_vs_d = np.array([
    transformer_params(int(d), n_layers=DEFAULT_LAYERS)['total'] for d in d_grid
])
# sanity: closed form must match the detailed counter
assert all(
    transformer_params(int(d), n_layers=DEFAULT_LAYERS)['total']
    == params_closed_form(int(d), L=DEFAULT_LAYERS)
    for d in d_grid
)

ax = axes1[0]
ax.plot(d_grid, params_vs_d / 1e6, 'o-', color='royalblue', lw=2.5)
# reference quadratic curve through first point
c = (params_vs_d[0] / 1e6) / (d_grid[0] ** 2)
ax.plot(d_grid, c * d_grid ** 2, '--', color='0.4', lw=1.8,
        label=r'$\propto d^2$ guide')
ax.set_xlabel(r'model dimension $d_{\mathrm{model}}$', fontsize=FS_AXIS)
ax.set_ylabel('parameters (millions)', fontsize=FS_AXIS)
ax.set_title(rf'Params vs width  (L={DEFAULT_LAYERS}, fixed vocab)', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Weights are matrices\n'
    'like R^{d x d}.\n'
    'Doubling d ~ 4x params\n'
    'in the big blocks.',
    y=0.98, va='top', facecolor='lightyellow',
)

# params vs T: flat line
T_grid = np.array([16, 64, 256, 1024, 4096, 16384])
d_fixed = 512
params_fixed = transformer_params(d_fixed, n_layers=DEFAULT_LAYERS)['total'] / 1e6
ax = axes1[1]
ax.plot(T_grid, np.full_like(T_grid, params_fixed, dtype=float),
        's-', color='seagreen', lw=2.5, label='parameter count')
flops = np.array([
    attention_flops(int(T), d_fixed, n_layers=DEFAULT_LAYERS) for T in T_grid
], dtype=float)
ax2 = ax.twinx()
ax2.plot(T_grid, flops / flops[0], 'o--', color='crimson', lw=2.0, label='attn compute (rel.)')
ax.set_xscale('log', base=2)
ax.set_xlabel(r'sequence length $T$', fontsize=FS_AXIS)
ax.set_ylabel('parameters (millions)', fontsize=FS_AXIS, color='seagreen')
ax2.set_ylabel('relative attention compute', fontsize=FS_AXIS, color='crimson')
ax.set_title(rf'Fix $d={d_fixed}$: params flat, compute grows', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax2.tick_params(axis='y', labelsize=FS_TICK)
# combined legend
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=FS_LEGEND, loc='upper left')
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Longer context needs\n'
    'more FLOPs / memory,\n'
    'NOT more weights.\n\n'
    'Capacity lives in d.',
    y=0.50, va='center',
)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Parameter formula for P(d)
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    r'Step 2: Parameter formula in terms of $d=d_{\mathrm{model}}$ '
    r'(embed + blocks; ignores final LN)',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
ax2.axis('off')
ax2.text(
    0.5, 0.78,
    r'Assumptions: $L$ layers, vocab $V$, $d_{ff}=4d$, tied LM head,'
    '\n'
    r'sinusoidal PE (0 params), full $d\times d$ Q/K/V/O, biases included.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    transform=ax2.transAxes,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
)
ax2.text(
    0.5, 0.52,
    r'$P(d)= Vd + L\,(12d^{2}+13d)$'
    '\n\n'
    'Expanded per layer (with '
    r'$d_{ff}=4d$'
    '):\n'
    r'  attn Q/K/V/O :  $4d^{2}+4d$'
    '\n'
    r'  FFN          :  $8d^{2}+5d$'
    '\n'
    r'  2 x LayerNorm:  $4d$'
    '\n'
    r'  layer total  :  $12d^{2}+13d$',
    ha='center', va='center', fontsize=FS_AXIS - 1,
    transform=ax2.transAxes,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.98),
)
ax2.text(
    0.5, 0.14,
    'No T anywhere: sequence length does not change the weight count.\n'
    r'Dominant term: $12 L d^{2}$  (doubling $d$ ~ $4\times$ the big blocks).'
    '\n'
    r'If embeddings are NOT tied, add another $+Vd$ for the LM head.'
    '\n'
    r'If biases are omitted: $P(d)=Vd + L(12d^{2}+4d)$.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    transform=ax2.transAxes,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Where the parameters sit
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    'Step 3: Anatomy of the count -- embeddings + per-layer $d\\times d$ maps',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

d_show = 512
n_layers = DEFAULT_LAYERS
parts = transformer_params(d_show, n_layers=n_layers)
labels = ['embed', 'attn', 'ffn', 'LayerNorm', 'untied head']
values = [
    parts['embed'],
    parts['attn_all_layers'],
    parts['ffn_all_layers'],
    parts['ln_all_layers'],
    parts['head'],
]
# head is 0 when tied
colors = ['#4c78a8', '#f58518', '#54a24b', '#bda10c', '#e45756']

ax = axes3[0]
ax.pie(
    [v for v in values if v > 0],
    labels=[lab for lab, v in zip(labels, values) if v > 0],
    autopct='%1.1f%%',
    colors=[c for c, v in zip(colors, values) if v > 0],
    textprops={'fontsize': FS_ANNOT},
)
ax.set_title(rf'$d={d_show}$, L={n_layers}, tied LM head', fontsize=FS_AX_TITLE)

ax = axes3[1]
# show scaling of components vs d
comp_attn = []
comp_ffn = []
comp_embed = []
for d in d_grid:
    p = transformer_params(int(d), n_layers=n_layers)
    comp_attn.append(p['attn_all_layers'] / 1e6)
    comp_ffn.append(p['ffn_all_layers'] / 1e6)
    comp_embed.append(p['embed'] / 1e6)
ax.plot(d_grid, comp_embed, 'o-', label=r'embed $= V d$', lw=2.3)
ax.plot(d_grid, comp_attn, 's-', label=r'attn $= L(4d^2+4d)$', lw=2.3)
ax.plot(d_grid, comp_ffn, '^-', label=r'ffn $= L(8d^2+5d)$', lw=2.3)
ax.set_xlabel(r'$d_{\mathrm{model}}$', fontsize=FS_AXIS)
ax.set_ylabel('parameters (millions)', fontsize=FS_AXIS)
ax.set_title('Component growth vs d', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'None of these curves\n'
    'depend on T.\n\n'
    'T only appears in\n'
    'activations / attention\n'
    'scores, not in W.',
    y=0.98, va='top', facecolor='lightyellow',
)

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 3: Toy underfitting / overfitting vs width
# -------------------------------------------------------
# Fit polynomials of increasing "width" (degree+1 params) as a stand-in for d_model
# Same idea: more parameters => more capacity.

def true_fn(x):
    return np.sin(2.0 * x) + 0.3 * np.cos(3.5 * x)


n_train = 18
x_train = np.linspace(-2.2, 2.2, n_train)
y_train = true_fn(x_train) + 0.45 * RNG.normal(size=n_train)
x_grid = np.linspace(-2.8, 2.8, 400)
y_true = true_fn(x_grid)
# held-out test points on same domain
x_test = np.linspace(-2.2, 2.2, 80)
y_test = true_fn(x_test) + 0.45 * RNG.normal(size=80)


def fit_poly(x, y, degree):
    coef = np.polyfit(x, y, degree)
    return coef


def pred_poly(coef, x):
    return np.polyval(coef, x)


degrees = [1, 3, 6, 12]  # param counts = degree+1
fig4, axes4 = new_figure(1, 4)
fig4.suptitle(
    'Step 4: Capacity knob -- too narrow underfits, too wide overfits',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

train_mses = []
test_mses = []
for ax, deg in zip(axes4, degrees):
    coef = fit_poly(x_train, y_train, deg)
    y_hat = pred_poly(coef, x_grid)
    tr = float(np.mean((pred_poly(coef, x_train) - y_train) ** 2))
    te = float(np.mean((pred_poly(coef, x_test) - y_test) ** 2))
    train_mses.append(tr)
    test_mses.append(te)
    ax.plot(x_grid, y_true, 'k--', lw=1.6, alpha=0.7, label='truth')
    ax.scatter(x_train, y_train, s=28, c='red', zorder=3)
    ax.plot(x_grid, y_hat, color='royalblue', lw=2.3)
    n_params = deg + 1
    ax.set_title(f'dim~{n_params} params\ntrain {tr:.2f} | test {te:.2f}',
                 fontsize=FS_AX_TITLE - 4)
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK - 2)
    ax.grid(True, alpha=0.3)
    if ax is axes4[0]:
        ax.set_ylabel('y', fontsize=FS_AXIS)
        ax.legend(fontsize=FS_LEGEND - 1, loc='lower left')

axes4[0].text(
    0.02, 0.98, 'underfit', transform=axes4[0].transAxes,
    va='top', fontsize=FS_ANNOT, color='crimson', fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
)
axes4[-1].text(
    0.02, 0.98, 'overfit', transform=axes4[-1].transAxes,
    va='top', fontsize=FS_ANNOT, color='crimson', fontweight='bold',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Train/test curves vs dimension
# -------------------------------------------------------
fig5, axes5 = new_figure(1, 2)
fig5.suptitle(
    'Step 5: Model dimension controls the bias--variance tradeoff',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

deg_sweep = np.arange(1, 15)
tr_curve, te_curve, nparams = [], [], []
for deg in deg_sweep:
    coef = fit_poly(x_train, y_train, int(deg))
    tr_curve.append(float(np.mean((pred_poly(coef, x_train) - y_train) ** 2)))
    te_curve.append(float(np.mean((pred_poly(coef, x_test) - y_test) ** 2)))
    nparams.append(int(deg) + 1)

ax = axes5[0]
ax.plot(nparams, tr_curve, 'o-', color='crimson', lw=2.5, label='train MSE')
ax.plot(nparams, te_curve, 's-', color='royalblue', lw=2.5, label='test MSE')
ax.set_xlabel('number of parameters (~ model dimension)', fontsize=FS_AXIS)
ax.set_ylabel('MSE', fontsize=FS_AXIS)
ax.set_title('Classic U-shape on test error', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Small d: high bias\n'
    '(underfit).\n\n'
    'Large d: low train\n'
    'error, high variance\n'
    '(overfit).',
    y=0.98, va='top', facecolor='lightyellow',
)

ax = axes5[1]
# Transformer analogy: params vs d, with a "data budget" line
data_budget = 5e6  # toy: tokens * something -- just a horizontal reference
ax.plot(d_grid, params_vs_d, 'o-', color='seagreen', lw=2.5, label='Transformer params')
ax.axhline(data_budget, color='crimson', ls='--', lw=2.0,
           label=f'toy data budget = {data_budget/1e6:.0f}M')
ax.set_xlabel(r'$d_{\mathrm{model}}$', fontsize=FS_AXIS)
ax.set_ylabel('parameter count', fontsize=FS_AXIS)
ax.set_title('Capacity vs available data', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'If params >> data,\n'
    'overfitting risk rises\n'
    'unless regularized\n'
    '(dropout, LN, weight\n'
    'decay, more data).',
    y=0.98, va='top',
)

plt.tight_layout()
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Schematic -- T tokens share the SAME weights
# -------------------------------------------------------
fig6, ax6 = new_figure()
fig6.suptitle(
    'Step 6: One shared $W$ is reused at every position -- that is why T is irrelevant',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax6.set_xlim(0, 12)
ax6.set_ylim(0, 7)
ax6.axis('off')

# draw tokens
for i, x in enumerate(np.linspace(1.2, 10.5, 6)):
    ax6.add_patch(plt.Circle((x, 5.3), 0.35, facecolor='#dceaff', edgecolor='k', lw=1.5))
    ax6.text(x, 5.3, fr'$t_{i+1}$', ha='center', va='center', fontsize=FS_ANNOT)

# shared weight box
ax6.add_patch(plt.Rectangle((4.0, 2.4), 4.0, 1.4, facecolor='#ffe0b2', edgecolor='k', lw=2))
ax6.text(6.0, 3.1, r'shared $W\in\mathbb{R}^{d\times d}$',
         ha='center', va='center', fontsize=FS_AXIS - 1, fontweight='bold')

# arrows from all tokens to W
for x in np.linspace(1.2, 10.5, 6):
    ax6.annotate('', xy=(6.0, 3.8), xytext=(x, 4.9),
                 arrowprops=dict(arrowstyle='->', color='0.35', lw=1.2))

ax6.text(
    6.0, 1.2,
    'Whether the sequence has 16 or 16,000 tokens,\n'
    'they all reuse the same projection matrices.\n'
    'Model dimension sets the SIZE of those matrices -- hence capacity.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Summary table visual
# -------------------------------------------------------
fig7, ax7 = new_figure()
fig7.suptitle(
    'Step 7: What scales with what?',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax7.axis('off')

table = [
    ['Quantity', 'Scales with d?', 'Scales with T?', 'Controls'],
    ['Parameter count', 'YES (~ d^2, V d)', 'NO', 'capacity / over-underfit'],
    ['Attention compute', 'YES (linear in d)', 'YES (~ T^2)', 'runtime / memory'],
    ['Activation memory', 'YES', 'YES (~ T d)', 'batch limits'],
    ['Inductive bias of width', 'YES', 'NO', 'expressivity'],
]

col_x = [0.08, 0.32, 0.58, 0.78]
y0 = 0.82
for r, row in enumerate(table):
    y = y0 - r * 0.13
    weight = 'bold' if r == 0 else 'normal'
    bg = '#e8eef8' if r == 0 else ('#f7f7f7' if r % 2 else 'white')
    ax7.add_patch(plt.Rectangle((0.05, y - 0.05), 0.90, 0.11,
                                transform=ax7.transAxes, facecolor=bg,
                                edgecolor='0.7', lw=1))
    for x, cell in zip(col_x, row):
        ax7.text(x, y, cell, transform=ax7.transAxes, fontsize=FS_ANNOT + 1,
                 fontweight=weight, va='center', ha='left')

ax7.text(
    0.5, 0.12,
    'Practical slogan:\n'
    '  T stretches the CONTEXT window and the bill for compute.\n'
    '  d stretches the BRAIN width and the risk of over/underfitting.',
    transform=ax7.transAxes, ha='center', va='center', fontsize=FS_AXIS - 2,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

show_maximized(fig7)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
L = DEFAULT_LAYERS
V = DEFAULT_VOCAB
p64 = params_closed_form(64, L=L, V=V)
p512 = params_closed_form(512, L=L, V=V)
p1024 = params_closed_form(1024, L=L, V=V)

print("\n" + "=" * 60)
print("COMPLETE PARAMETER FORMULA")
print("=" * 60)
print(f"""
GPT-style Transformer (tied LM head, sinusoidal PE, biases on):

  d      = d_model
  L      = n_layers          (here L={L})
  V      = vocab_size        (here V={V})
  d_ff   = 4 d

  Token embed:     V * d
  Per layer:
    Attn Q,K,V,O:  4 d^2 + 4 d
    FFN:           2 d d_ff + d_ff + d  = 8 d^2 + 5 d
    LayerNorm x2:  4 d
  --------------------------------
  Per layer total: 12 d^2 + 13 d

  COMPLETE FORMULA:
    P(d) = V*d + L*(12*d^2 + 13*d)

  Untied LM head: add another + V*d
  No biases:      P(d) = V*d + L*(12*d^2 + 4*d)

Checks:
  d=64   -> {p64/1e6:.2f}M
  d=512  -> {p512/1e6:.2f}M
  d=1024 -> {p1024/1e6:.2f}M

Note: T (sequence length) does not appear. Longer context changes
compute (~T^2 d), not the number of trainable weights.
""")
