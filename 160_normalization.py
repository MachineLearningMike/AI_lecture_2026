"""
160_normalization.py

Intuition for Batch Normalization vs Layer Normalization.

Both re-center and re-scale activations to roughly zero-mean / unit-variance,
THEN allow the network to re-learn a scale/shift (gamma, beta):

    x_hat = (x - mean) / sqrt(var + eps)     # fixed standardize
    y     = gamma * x_hat + beta             # learnable affine

Default gamma=1, beta=0 keeps the standardized behavior.
Learned gamma, beta can recover the old scale/shift -- or invent a new one.

The difference is WHICH AXIS the mean/var are computed over:

    BatchNorm  -- for each FEATURE, pool over the BATCH  (and spatial dims)
    LayerNorm  -- for each SAMPLE (token), pool over the FEATURES

Companion to 030_model.py (deeper nets need stable activation scales) and
130_momentum.py / 140_loss_functions.py (optimization geometry).
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


EPS = 1e-5


def screen_figsize(fraction=SCREEN_FRACTION, dpi=FIGURE_DPI):
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width_px = root.winfo_screenwidth()
        height_px = root.winfo_screenheight()
        root.destroy()
        return (width_px / dpi * fraction, height_px / dpi * fraction)
    except Exception:
        return (16 * fraction / 0.95, 9 * fraction / 0.95)


def new_figure(*subplot_args, **subplot_kwargs):
    subplot_kwargs.setdefault('figsize', screen_figsize())
    subplot_kwargs.setdefault('dpi', FIGURE_DPI)
    return plt.subplots(*subplot_args, **subplot_kwargs)


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


def add_vcenter_legend_and_note(ax, note, facecolor='white'):
    ax.legend(
        loc='lower right',
        bbox_to_anchor=(0.98, 0.54),
        fontsize=FS_LEGEND,
        framealpha=0.95,
        borderaxespad=0.0,
    )
    ax.text(
        0.98, 0.46, note,
        transform=ax.transAxes,
        fontsize=FS_ANNOT,
        va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.95),
    )


# -------------------------------------------------------
# Toy activation matrix: rows = batch samples, cols = features
# Deliberately messy scales across features AND samples.
# -------------------------------------------------------
B, F = 8, 6  # batch size, features

# Feature scales differ a lot (feature 0 tiny, feature 5 huge)
feature_scale = np.array([0.3, 1.0, 2.5, 0.6, 4.0, 8.0])
feature_shift = np.array([-2.0, 0.0, 3.0, 1.0, -1.5, 5.0])

# Sample-level gain (some examples are globally brighter / larger activations)
sample_gain = np.linspace(0.4, 2.2, B)

X = RNG.normal(size=(B, F)) * feature_scale[None, :] + feature_shift[None, :]
X *= sample_gain[:, None]


def batch_norm(x, gamma=None, beta=None, eps=EPS):
    """BN over batch axis for each feature. x: (batch, features)."""
    mean = x.mean(axis=0, keepdims=True)          # (1, F)
    var = x.var(axis=0, keepdims=True)            # (1, F)
    x_hat = (x - mean) / np.sqrt(var + eps)
    if gamma is None:
        gamma = np.ones((1, x.shape[1]))
    if beta is None:
        beta = np.zeros((1, x.shape[1]))
    return gamma * x_hat + beta, mean.ravel(), var.ravel()


def layer_norm(x, gamma=None, beta=None, eps=EPS):
    """LN over feature axis for each sample. x: (batch, features)."""
    mean = x.mean(axis=1, keepdims=True)          # (B, 1)
    var = x.var(axis=1, keepdims=True)            # (B, 1)
    x_hat = (x - mean) / np.sqrt(var + eps)
    if gamma is None:
        gamma = np.ones((1, x.shape[1]))
    if beta is None:
        beta = np.zeros((1, x.shape[1]))
    return gamma * x_hat + beta, mean.ravel(), var.ravel()


X_bn, bn_mean, bn_var = batch_norm(X)
X_ln, ln_mean, ln_var = layer_norm(X)


# -------------------------------------------------------
# FIGURE 1: Why normalize? -- wild activation scales
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    'Step 1: Raw activations are poorly scaled -- hard for the next layer',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes1[0]
im = ax.imshow(X, aspect='auto', cmap='coolwarm')
ax.set_xlabel('feature j', fontsize=FS_AXIS)
ax.set_ylabel('batch sample i', fontsize=FS_AXIS)
ax.set_title('Activation matrix X[i, j]', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=FS_ANNOT)
ax.text(
    0.02, 0.98,
    'Columns: different feature scales\n'
    'Rows: some samples globally larger',
    transform=ax.transAxes, va='top', ha='left', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes1[1]
for j in range(F):
    ax.scatter(np.full(B, j), X[:, j], s=40, alpha=0.85, label=f'f{j}' if j < 3 else None)
ax.axhline(0, color='k', alpha=0.3)
ax.set_xlabel('feature j', fontsize=FS_AXIS)
ax.set_ylabel('activation value', fontsize=FS_AXIS)
ax.set_title('Per-feature spread is all over the place', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    'Next layer sees a mix of\n'
    'tiny and huge numbers.\n'
    'Gradients / learning rates\n'
    'become awkward.\n\n'
    'Normalization rebalances\n'
    'these scales.',
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: What axis do we average over?
# -------------------------------------------------------
fig2, axes2 = new_figure(1, 2)
fig2.suptitle(
    'Step 2: BatchNorm pools over batch; LayerNorm pools over features',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)


def draw_matrix_with_groups(ax, title, mode):
    """Schematic: highlight BN columns vs LN rows."""
    ax.set_xlim(-0.5, F - 0.5)
    ax.set_ylim(B - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('feature j', fontsize=FS_AXIS)
    ax.set_ylabel('sample i', fontsize=FS_AXIS)
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.set_xticks(range(F))
    ax.set_yticks(range(B))

    # cell faces from data (normalized display)
    vmax = np.percentile(np.abs(X), 95)
    for i in range(B):
        for j in range(F):
            val = X[i, j] / (vmax + 1e-8)
            color = plt.cm.coolwarm((val + 1) / 2)
            rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor=color, edgecolor='white', linewidth=1.0)
            ax.add_patch(rect)

    if mode == 'bn':
        # emphasize one feature column
        j = 4
        ax.add_patch(patches.Rectangle(
            (j - 0.5, -0.5), 1, B, fill=False, edgecolor='crimson', linewidth=3.5,
        ))
        ax.text(
            0.5, -0.12,
            r'BN: mean/var over $i$ (batch) for fixed $j$',
            transform=ax.transAxes, ha='center', va='top', fontsize=FS_ANNOT,
            color='crimson', fontweight='bold',
        )
    else:
        i = 2
        ax.add_patch(patches.Rectangle(
            (-0.5, i - 0.5), F, 1, fill=False, edgecolor='royalblue', linewidth=3.5,
        ))
        ax.text(
            0.5, -0.12,
            r'LN: mean/var over $j$ (features) for fixed $i$',
            transform=ax.transAxes, ha='center', va='top', fontsize=FS_ANNOT,
            color='royalblue', fontweight='bold',
        )


draw_matrix_with_groups(
    axes2[0],
    'BatchNorm -- normalize each COLUMN',
    'bn',
)
draw_matrix_with_groups(
    axes2[1],
    'LayerNorm -- normalize each ROW',
    'ln',
)

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: After BN vs after LN
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 3)
fig3.suptitle(
    'Step 3: Same X, two normalizations -- different invariants',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

panels = [
    (X, 'Raw X', 'Raw'),
    (X_bn, 'After BatchNorm', 'BN'),
    (X_ln, 'After LayerNorm', 'LN'),
]
vmax = max(np.percentile(np.abs(m), 98) for m, _, _ in panels)
for ax, (mat, title, tag) in zip(axes3, panels):
    im = ax.imshow(mat, aspect='auto', cmap='coolwarm', vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.set_xlabel('feature j', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    if tag == 'Raw':
        ax.set_ylabel('sample i', fontsize=FS_AXIS)
    # stats box
    if tag == 'BN':
        note = (
            'Each column ~ N(0,1)\n'
            f'col means≈0\n'
            f'col stds≈1\n\n'
            'Depends on the\n'
            'other batch items'
        )
    elif tag == 'LN':
        note = (
            'Each row ~ N(0,1)\n'
            f'row means≈0\n'
            f'row stds≈1\n\n'
            'Independent of\n'
            'batch mates'
        )
    else:
        note = (
            f'feature stds:\n'
            + ', '.join(f'{s:.1f}' for s in X.std(axis=0))
        )
    ax.text(
        0.02, 0.98, note,
        transform=ax.transAxes, va='top', ha='left', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
    )

fig3.colorbar(im, ax=axes3.ravel().tolist(), fraction=0.02, pad=0.02)
plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Distributions + when to use which
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    'Step 4: BN couples samples; LN does not -- pick based on the setting',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes4[0]
# Show one feature's values across the batch, before/after BN
feat = 5
ax.plot(np.arange(B), X[:, feat], 'o-', color='0.5', linewidth=2, label='raw feature 5')
ax.plot(np.arange(B), X_bn[:, feat], 's-', color='crimson', linewidth=2, label='after BN')
ax.plot(np.arange(B), X_ln[:, feat], '^-', color='royalblue', linewidth=2, label='after LN')
ax.axhline(0, color='k', alpha=0.3)
ax.set_xlabel('batch index i', fontsize=FS_AXIS)
ax.set_ylabel('activation', fontsize=FS_AXIS)
ax.set_title('Tracking one feature across the batch', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    'BN forces THIS feature\n'
    'to have mean 0 / var 1\n'
    'across the batch.\n\n'
    'LN changes it only as\n'
    'part of each sample\'s\n'
    'own feature vector.',
)
ax.grid(True, alpha=0.3)

ax = axes4[1]
ax.axis('off')
ax.set_title('Practical intuition', fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.55,
    'BatchNorm\n'
    '  • Stats over the batch (per channel/feature)\n'
    '  • Great for CNNs / large batches\n'
    '  • Train vs eval differ (running averages)\n'
    '  • Noisy / odd if batch size is tiny\n\n'
    'LayerNorm\n'
    '  • Stats over features (per sample / token)\n'
    '  • Great for Transformers / RNNs / NLP\n'
    '  • Same behavior train and eval\n'
    '  • Works with batch size 1\n\n'
    'Both keep activations in a friendly numeric range,\n'
    'then gamma/beta let the net re-scale if needed.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_AXIS - 2,
    bbox=dict(boxstyle='round', facecolor='#f7f7ff', alpha=0.96),
)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: gamma/beta -- keep standardized behavior XOR learn new affine
# -------------------------------------------------------
# For one feature under BN:  x_hat = (x-mu)/sigma,  y = gamma x_hat + beta.
#   (gamma, beta) = (1, 0)     -> keep pure normalized behavior
#   (gamma, beta) = (sigma, mu) -> undo BN: y ~= x  (original scale/shift)
#   other (gamma, beta)         -> any new scale/shift on the unit axis
# -------------------------------------------------------
fig5, axes5 = new_figure(1, 2)
fig5.suptitle(
    r'Step 5: $\gamma,\beta$ keep the option of "just normalize" AND open new scale/shift',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

feat_demo = 5
x_feat = X[:, feat_demo]
mu_f = float(x_feat.mean())
sig_f = float(x_feat.std() + 1e-12)
x_hat_f = (x_feat - mu_f) / sig_f

# Three affine readouts of the same x_hat
y_keep = 1.0 * x_hat_f + 0.0                    # default init: standardized
y_orig = sig_f * x_hat_f + mu_f                  # recover original (~ identity)
gamma_new, beta_new = 2.2, -1.5
y_new = gamma_new * x_hat_f + beta_new           # a new learned scale/shift

ax = axes5[0]
ax.axis('off')
ax.set_title('Formula: standardize, then optional affine', fontsize=FS_AX_TITLE)
ax.text(
    0.5, 0.55,
    r'$\hat x = \dfrac{x-\mu}{\sqrt{\sigma^2+\varepsilon}}$'
    '\n\n'
    r'$y = \gamma\,\hat x + \beta$'
    '\n\n'
    r'$\bullet\ \gamma=1,\ \beta=0$'
    '   keeps standardized behavior'
    '\n'
    r'$\bullet\ \gamma=\sigma,\ \beta=\mu$'
    '   recovers original $x$ (approx.)'
    '\n'
    r'$\bullet\ \gamma,\beta$ free'
    '   learn ANY new scale & shift'
    '\n\n'
    'So BN/LN do not permanently erase scale/shift capacity:\n'
    'they fix a clean coordinate, then let the net re-paint affine.',
    transform=ax.transAxes, ha='center', va='center', fontsize=FS_AXIS - 1,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.96),
)

ax = axes5[1]
idx = np.arange(B)
ax.plot(idx, x_feat, 'o-', color='0.45', lw=2.0, label=r'raw $x$')
ax.plot(idx, y_keep, 's-', color='seagreen', lw=2.2, label=r'$\gamma=1,\beta=0$ (keep norm)')
ax.plot(idx, y_orig, '^-', color='royalblue', lw=2.2, label=r'$\gamma=\sigma,\beta=\mu$ (≈ original)')
ax.plot(idx, y_new, 'D-', color='crimson', lw=2.2,
        label=rf'$\gamma={gamma_new:g},\beta={beta_new:g}$ (new)')
ax.axhline(0, color='k', alpha=0.3)
ax.set_xlabel('batch index', fontsize=FS_AXIS)
ax.set_ylabel('activation (one BN feature)', fontsize=FS_AXIS)
ax.set_title('Same $\\hat x$; different $(\\gamma,\\beta)$ readouts', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc='best')
ax.grid(True, alpha=0.3)
ax.text(
    0.02, 0.02,
    rf'feature {feat_demo}:  $\mu={mu_f:.2f}$,  $\sigma={sig_f:.2f}$' + '\n'
    r'green $\approx$ unit scale; blue tracks raw; red is a new affine.',
    transform=ax.transAxes, va='bottom', ha='left', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Tiny training cartoon -- loss curvature / scales
# -------------------------------------------------------
fig6, axes6 = new_figure(1, 2)
fig6.suptitle(
    'Step 6: Normalization makes the loss landscape less awkwardly stretched',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# 2D quadratic with very different axis scales = "before BN"
# After normalizing features, axes become comparable.

def bowl(w1, w2, s1, s2):
    return 0.5 * ((s1 * w1) ** 2 + (s2 * w2) ** 2)


w = np.linspace(-3, 3, 200)
W1, W2 = np.meshgrid(w, w)

ax = axes6[0]
Z = bowl(W1, W2, s1=4.0, s2=0.4)
cs = ax.contour(W1, W2, Z, levels=15, cmap='Reds')
ax.plot(0, 0, 'k*', markersize=12)
lr = 0.08
x = np.array([2.5, 2.0], dtype=float)
traj = [x.copy()]
for _ in range(25):
    g = np.array([4.0 ** 2 * x[0], 0.4 ** 2 * x[1]])
    x = x - lr * g
    traj.append(x.copy())
traj = np.stack(traj)
ax.plot(traj[:, 0], traj[:, 1], 'o-', color='crimson', markersize=3, linewidth=1.5)
ax.set_title('Unnormalized features\n(elongated / zigzaggy GD)', fontsize=FS_AX_TITLE)
ax.set_xlabel(r'$w_1$', fontsize=FS_AXIS)
ax.set_ylabel(r'$w_2$', fontsize=FS_AXIS)
ax.set_aspect('equal')
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3)

ax = axes6[1]
Z2 = bowl(W1, W2, s1=1.0, s2=1.0)
ax.contour(W1, W2, Z2, levels=15, cmap='Blues')
ax.plot(0, 0, 'k*', markersize=12)
x = np.array([2.5, 2.0], dtype=float)
traj = [x.copy()]
for _ in range(25):
    g = np.array([1.0 ** 2 * x[0], 1.0 ** 2 * x[1]])
    x = x - lr * g
    traj.append(x.copy())
traj = np.stack(traj)
ax.plot(traj[:, 0], traj[:, 1], 'o-', color='royalblue', markersize=3, linewidth=1.5)
ax.set_title('Normalized features\n(rounder bowl, smoother path)', fontsize=FS_AX_TITLE)
ax.set_xlabel(r'$w_1$', fontsize=FS_AXIS)
ax.set_ylabel(r'$w_2$', fontsize=FS_AXIS)
ax.set_aspect('equal')
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3)
ax.text(
    0.02, 0.98,
    'Same GD, same lr.\n'
    'Normalization changes\n'
    'the geometry the\n'
    'optimizer walks on.',
    transform=ax.transAxes, va='top', ha='left', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig6)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("BATCH NORM vs LAYER NORM")
print("=" * 60)
print(f"""
Shared formula (two stages):
  x_hat = (x - mean) / sqrt(var + eps)     # standardize
  y     = gamma * x_hat + beta             # learnable scale/shift

  gamma=1, beta=0  -> keep standardized behavior
  gamma=sigma, beta=mu -> recover original x (approx.) for that axis
  free gamma, beta -> learn a NEW scale and shift on the unit axis

  So normalization does not remove affine capacity: it reparameterizes it.

Difference = the reduction axes for mean/var:

  BatchNorm (on X with shape [batch, features]):
      mean/var over axis=0 (the batch), separately per feature
      -> each feature column is standardized across examples

  LayerNorm:
      mean/var over axis=1 (the features), separately per sample
      -> each example's feature vector is standardized on its own

Demo matrix shape: batch={B}, features={F}
  BN column stds after norm: {np.std(X_bn, axis=0).round(3)}
  LN row    stds after norm: {np.std(X_ln, axis=1).round(3)}

Why it helps:
  1) Keeps activation magnitudes in a friendly range
  2) Makes loss landscapes less awkwardly stretched (Step 6)
  3) gamma/beta keep "original-like" behavior available AND allow new affines (Step 5)

Rule of thumb:
  CNNs / large vision batches  -> BatchNorm is classic
  Transformers / tokens / NLP  -> LayerNorm is classic
  Tiny batches / batch size 1  -> prefer LayerNorm
""")
