"""
0710_dropout.py

Intuition for Dropout regularization.

Training:
    Each unit is kept with probability p and zeroed with probability 1-p.
    Surviving activations are typically scaled by 1/p (inverted dropout)
    so the expected activation matches evaluation.

Evaluation:
    Use the full network (no dropping). With inverted dropout, no
    rescaling is needed at test time.

Why it helps:
    - Stops co-adaptation ("conspiracies" between units)
    - Trains an implicit ensemble of subnetworks
    - Acts like a strong inductive bias toward redundancy

Companion to 0690_inductive_bias.py and 0720_normalize_activations.py.
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
from matplotlib.colors import ListedColormap
import numpy as np

np.random.seed(42)
RNG = np.random.default_rng(42)


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
    add_step_footer(fig)
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


def dropout(x, p=0.5, training=True, rng=RNG):
    """Inverted dropout. p = keep probability."""
    if (not training) or p >= 1.0:
        return x, np.ones_like(x, dtype=bool)
    mask = rng.random(x.shape) < p
    # scale by 1/p so E[y] ~= E[x] during training
    y = np.where(mask, x / p, 0.0)
    return y, mask


# -------------------------------------------------------
# FIGURE 1: What dropout does to a hidden layer
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    r'Step 1: During training, randomly drop units (keep prob $p$)',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

n_units = 12
h = RNG.normal(size=n_units)
p_keep = 0.5
h_drop, mask = dropout(h, p=p_keep, training=True, rng=np.random.default_rng(0))

ax = axes1[0]
colors = ['seagreen' if m else 'lightgray' for m in mask]
ax.bar(np.arange(n_units), h, color='steelblue', edgecolor='k', linewidth=0.5)
ax.axhline(0, color='k', alpha=0.3)
ax.set_title('Before dropout', fontsize=FS_AX_TITLE)
ax.set_xlabel('hidden unit', fontsize=FS_AXIS)
ax.set_ylabel('activation', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, axis='y', alpha=0.3)
ax.text(
    0.98, 0.02,
    'Full layer is active.\n'
    'Units can co-adapt\n'
    'to each other.',
    transform=ax.transAxes, va='bottom', ha='right', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes1[1]
ax.bar(np.arange(n_units), h_drop, color=colors, edgecolor='k', linewidth=0.5)
ax.axhline(0, color='k', alpha=0.3)
ax.set_title(rf'After dropout (p={p_keep:g}), inverted scaling', fontsize=FS_AX_TITLE)
ax.set_xlabel('hidden unit', fontsize=FS_AXIS)
ax.set_ylabel('activation', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, axis='y', alpha=0.3)
kept = int(mask.sum())
ax.text(
    0.98, 0.02,
    f'Kept {kept}/{n_units} units.\n'
    f'Gray = dropped (0).\n'
    f'Green kept values are\n'
    f'scaled by 1/p = {1/p_keep:g}.',
    transform=ax.transAxes, va='bottom', ha='right', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Different minibatches = different subnetworks
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    'Step 2: Each step trains a random subnetwork -- an implicit ensemble',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

n_sgd_steps = 10
masks = np.stack(
    [np.random.default_rng(s).random(n_units) < p_keep for s in range(n_sgd_steps)],
    axis=0,
)
# show as a binary grid: rows=steps, cols=units
cmap = ListedColormap(['#d0d0d0', '#2e8b57'])
im = ax2.imshow(masks.astype(float), aspect='auto', cmap=cmap, vmin=0, vmax=1)
ax2.set_xlabel('hidden unit', fontsize=FS_AXIS)
ax2.set_ylabel('training step (minibatch)', fontsize=FS_AXIS)
ax2.set_title('Green = kept, gray = dropped', fontsize=FS_AX_TITLE)
ax2.set_yticks(range(n_sgd_steps))
ax2.set_xticks(range(n_units))
ax2.tick_params(axis='both', labelsize=FS_TICK)
ax2.text(
    0.98, 0.50,
    'No two steps see the\n'
    'exact same active set.\n\n'
    'At test time we average\n'
    'their behavior by using\n'
    'the FULL network\n'
    '(no drops).',
    transform=ax2.transAxes, va='center', ha='right', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Toy fit -- dropout reduces overfitting
# -------------------------------------------------------
def true_fn(x):
    return np.sin(2.0 * x)


n_train = 14
x_train = np.linspace(-2.5, 2.5, n_train)
y_train = true_fn(x_train) + 0.70 * RNG.normal(size=n_train)
x_grid = np.linspace(-3.0, 3.0, 300)
y_true = true_fn(x_grid)


def tanh(z):
    return np.tanh(z)


def mlp_forward(x, W1, b1, W2, b2, p=1.0, training=False, rng=None):
    """1-hidden-layer MLP: x -> tanh -> (dropout) -> linear."""
    rng = rng or RNG
    if x.ndim == 1:
        x = x[:, None]
    h = tanh(x @ W1 + b1)              # (N, H)
    h, mask = dropout(h, p=p, training=training, rng=rng)
    y = (h @ W2 + b2).ravel()
    return y, h, mask


def train_mlp(p_keep, steps=6000, lr=0.04, seed=0, l2=0.0):
    """Over-parameterized fit; dropout is the main regularizer."""
    rng = np.random.default_rng(seed)
    H = 100
    W1 = rng.normal(0, 0.35, size=(1, H))
    b1 = np.zeros(H)
    W2 = rng.normal(0, 0.35, size=(H, 1))
    b2 = np.zeros(1)

    losses = []
    for t in range(steps):
        xb = x_train[:, None]
        yb = y_train

        z = xb @ W1 + b1
        h_pre = tanh(z)
        h, mask = dropout(h_pre, p=p_keep, training=True, rng=rng)
        y_hat = (h @ W2 + b2).ravel()
        err = y_hat - yb
        loss = 0.5 * np.mean(err ** 2) + 0.5 * l2 * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
        losses.append(loss)

        dY = (err / n_train)[:, None]
        dW2 = h.T @ dY + l2 * W2
        db2 = dY.sum(axis=0)
        dH = dY @ W2.T
        dH = np.where(mask, dH / max(p_keep, 1e-8), 0.0)
        dZ = dH * (1.0 - np.clip(h_pre, -0.999, 0.999) ** 2)
        dW1 = xb.T @ dZ + l2 * W1
        db1 = dZ.sum(axis=0)

        # gradient clipping for stability
        for g in (dW1, db1, dW2, db2):
            np.clip(g, -5.0, 5.0, out=g)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return (W1, b1, W2, b2), np.array(losses)


params_nodrop, loss_nodrop = train_mlp(p_keep=1.0, seed=1)
params_drop, loss_drop = train_mlp(p_keep=0.5, seed=1)

y_nodrop, _, _ = mlp_forward(x_grid, *params_nodrop, p=1.0, training=False)
y_drop, _, _ = mlp_forward(x_grid, *params_drop, p=1.0, training=False)

fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    'Step 3: Same MLP -- dropout hurts train fit a bit, helps generalization',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes3[0]
ax.plot(x_grid, y_true, 'k--', linewidth=2.0, alpha=0.7, label='truth')
ax.scatter(x_train, y_train, s=45, c='red', zorder=5, edgecolor='k',
           linewidth=0.4, label='train data')
ax.plot(x_grid, y_nodrop, color='crimson', linewidth=2.6, label='no dropout')
ax.plot(x_grid, y_drop, color='royalblue', linewidth=2.6, label=r'dropout $p=0.5$')
ax.set_title('Learned functions', fontsize=FS_AX_TITLE)
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(loc='lower right', fontsize=FS_LEGEND, framealpha=0.95)
ax.text(
    0.98, 0.98,
    'Without dropout the net\n'
    'chases noise wiggles.\n\n'
    'With dropout it stays\n'
    'closer to the smooth\n'
    'truth (stronger bias).',
    transform=ax.transAxes, ha='right', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)
ax.grid(True, alpha=0.3)

ax = axes3[1]
ax.plot(loss_nodrop, color='crimson', linewidth=2.2, label='no dropout')
ax.plot(loss_drop, color='royalblue', linewidth=2.2, label=r'dropout $p=0.5$')
ax.set_title('Mini-batch loss', fontsize=FS_AX_TITLE)
ax.set_xlabel('SGD step', fontsize=FS_AXIS)
ax.set_ylabel('MSE', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    'Dropout usually raises\n'
    'mini-batch loss (harder fit)\n'
    'but can lower true risk.\n\n'
    'Regularization often\n'
    'looks like "worse"\n'
    'training numbers.',
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Co-adaptation cartoon + inverted dropout expectation
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    'Step 4: Break co-adaptation; keep expectations consistent at test time',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes4[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title('Co-adaptation without dropout', fontsize=FS_AX_TITLE)

# simple nodes
positions_in = [(1.5, 4.5), (1.5, 3.0), (1.5, 1.5)]
positions_h = [(5.0, 5.0), (5.0, 3.5), (5.0, 2.0), (5.0, 0.8)]
positions_out = [(8.5, 3.0)]

for (x, y) in positions_in:
    ax.add_patch(patches.Circle((x, y), 0.28, facecolor='steelblue', edgecolor='k'))
for (x, y) in positions_h:
    ax.add_patch(patches.Circle((x, y), 0.28, facecolor='darkorange', edgecolor='k'))
ax.add_patch(patches.Circle(positions_out[0], 0.32, facecolor='seagreen', edgecolor='k'))

# emphasize a clique of co-adapted hidden units
for i, hpos in enumerate(positions_h):
    for ip in positions_in:
        ax.plot([ip[0], hpos[0]], [ip[1], hpos[1]], color='0.75', lw=0.8, zorder=0)
    ax.plot([hpos[0], positions_out[0][0]], [hpos[1], positions_out[0][1]],
            color='crimson' if i < 2 else '0.75',
            lw=2.5 if i < 2 else 0.8, zorder=0)

ax.text(1.5, 5.3, 'inputs', ha='center', fontsize=FS_ANNOT)
ax.text(5.0, 5.7, 'hidden', ha='center', fontsize=FS_ANNOT)
ax.text(8.5, 3.7, 'out', ha='center', fontsize=FS_ANNOT)
ax.text(
    0.72, 0.08,
    'Two hidden units "conspire" to explain the label together.\n'
    'If either is unreliable alone, the model is brittle.\n'
    'Dropout forces each kept unit to be useful without its partners.',
    transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

ax = axes4[1]
# Expectation cartoon: train scaling
ps = np.array([0.3, 0.5, 0.7, 0.9, 1.0])
# Suppose unit activation a=2. Without scaling, E[train]=p*a; with inverted, E=a
a = 2.0
e_naive = ps * a
e_inverted = np.ones_like(ps) * a
ax.plot(ps, e_naive, 'o-', color='crimson', linewidth=2.5, label='naive drop (no /p)')
ax.plot(ps, e_inverted, 's-', color='royalblue', linewidth=2.5, label='inverted dropout (x/p)')
ax.axhline(a, color='k', linestyle='--', alpha=0.5, label='test-time activation a')
ax.set_xlabel(r'keep probability $p$', fontsize=FS_AXIS)
ax.set_ylabel('expected activation', fontsize=FS_AXIS)
ax.set_title('Why scale by 1/p in training', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(
    loc='upper left', bbox_to_anchor=(0.0, 0.88),
    fontsize=FS_LEGEND, framealpha=0.95, borderaxespad=0.0,
)
ax.text(
    0.98, 0.46,
    'Inverted dropout:\n'
    '  train: keep x/p, else 0\n'
    '  test:  use x as-is\n\n'
    'Then E[train] ≈ test,\n'
    'so no test rescaling.',
    transform=ax.transAxes, ha='right', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
mse_nodrop = float(np.mean((y_nodrop - y_true) ** 2))
mse_drop = float(np.mean((y_drop - y_true) ** 2))
mse_nodrop_train = float(np.mean((mlp_forward(x_train, *params_nodrop)[0] - y_train) ** 2))
mse_drop_train = float(np.mean((mlp_forward(x_train, *params_drop)[0] - y_train) ** 2))

print("\n" + "=" * 60)
print("DROPOUT")
print("=" * 60)
print(f"""
Mechanism:
  For each unit, keep with probability p, else set to 0.
  Inverted dropout scales kept units by 1/p during training.
  At evaluation: use the full network (no drops).

Intuition:
  1) Random masks => train many subnetworks (ensemble)
  2) Breaks co-adaptation between units
  3) Inductive bias toward redundant / robust features

Toy MLP on noisy sine data:
  no dropout:   train MSE={mse_nodrop_train:.3f}   true-curve MSE={mse_nodrop:.3f}
  dropout p=0.5: train MSE={mse_drop_train:.3f}   true-curve MSE={mse_drop:.3f}

Typical keep probabilities:
  hidden layers: p ~ 0.5 to 0.8
  input layer:   p closer to 1 (drop less)

Related ideas:
  DropConnect, Stochastic Depth, and Mixup are cousins -- all inject
  noise that shapes which solutions SGD prefers (see also 0690_inductive_bias.py).
""")
