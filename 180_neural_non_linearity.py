"""
180_neural_non_linearity.py

How neural nets create non-linearity and non-linear manifolds.

Key facts:
  1) Stacking linear layers WITHOUT activations is still just ONE linear map.
  2) Pointwise non-linearities (ReLU, GELU, SiLU, tanh, ...) break that collapse.
  3) One activated neuron only reshapes HEIGHT along parallel contour lines.
  4) Mixing several neurons (and stacking layers) bends those contours into
     curved level sets -- a non-linear manifold in input space.

Companion to 030_model.py (simple vs complex net) and 150_inductive_bias.py.
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
from matplotlib.colors import ListedColormap
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
# Tiny MLP utilities
# -------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def leaky_relu(z, alpha=0.1):
    return np.where(z > 0.0, z, alpha * z)


def tanh(z):
    return np.tanh(z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def gelu(z):
    # Common tanh approximation used in many DL libraries
    return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * z ** 3)))


def silu(z):
    # Swish / SiLU: z * sigmoid(z) -- common in modern conv / MLP blocks
    return z * sigmoid(z)


def linear_forward(X, weights):
    """X: (N, d). weights: list of (W, b) with NO activations."""
    H = X
    for W, b in weights:
        H = H @ W + b
    return H


def mlp_forward(X, weights, activation=relu):
    """All hidden layers use `activation`; last layer is linear."""
    H = X
    for i, (W, b) in enumerate(weights):
        H = H @ W + b
        if i < len(weights) - 1:
            H = activation(H)
    return H


def make_grid(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), n=25):
    xs = np.linspace(*xlim, n)
    ys = np.linspace(*ylim, n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    return XX, YY, pts


def plot_warped_grid(ax, XX, YY, warped, title, color='royalblue'):
    """warped: (n*n, 2) corresponding to XX/YY ravel order."""
    n = XX.shape[0]
    WX = warped[:, 0].reshape(n, n)
    WY = warped[:, 1].reshape(n, n)
    for i in range(n):
        ax.plot(WX[i, :], WY[i, :], color=color, linewidth=0.9, alpha=0.85)
        ax.plot(WX[:, i], WY[:, i], color=color, linewidth=0.9, alpha=0.85)
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(True, alpha=0.25)


# Fixed random-ish layers for demos (hand-tuned for clear visuals)
W1 = np.array([[1.2, -0.8],
               [0.6, 1.1]], dtype=float)
b1 = np.array([0.0, -0.3], dtype=float)
W2 = np.array([[0.9, 0.7],
               [-1.1, 0.8]], dtype=float)
b2 = np.array([0.2, 0.0], dtype=float)
W3 = np.array([[1.0, -0.5],
               [0.4, 1.2]], dtype=float)
b3 = np.array([-0.1, 0.2], dtype=float)

weights_linear = [(W1, b1), (W2, b2), (W3, b3)]


# -------------------------------------------------------
# FIGURE 1: Linear layers collapse; activations prevent that
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 3)
fig1.suptitle(
    'Step 1: Without activations, depth is an illusion',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

XX, YY, pts = make_grid(n=18)

# Original grid
ax = axes1[0]
for i in range(XX.shape[0]):
    ax.plot(XX[i, :], YY[i, :], color='0.4', lw=0.9)
    ax.plot(XX[:, i], YY[:, i], color='0.4', lw=0.9)
ax.set_title('Input space', fontsize=FS_AX_TITLE)
ax.set_aspect('equal')
ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.25)
ax.text(
    0.02, 0.98,
    'Start with a regular\n'
    '2D grid of points.',
    transform=ax.transAxes, va='top', ha='left', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
)

# Three linear layers, no activation == one affine map
ax = axes1[1]
warped_lin = linear_forward(pts, weights_linear)
plot_warped_grid(ax, XX, YY, warped_lin, '3 linear layers\n(no activation)', color='crimson')
ax.set_xlabel(r'$z_1$', fontsize=FS_AXIS)
ax.set_ylabel(r'$z_2$', fontsize=FS_AXIS)
# Equivalent single map
W_eq = W1 @ W2 @ W3
# biases accumulate; for visualization the grid warp already shows affinity
ax.text(
    0.02, 0.98,
    'Still a single affine\n'
    'transform:\n'
    r'$x \mapsto xW+b$' + '\n'
    'Straight lines stay\n'
    'straight.',
    transform=ax.transAxes, va='top', ha='left', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

# With ReLU between layers
ax = axes1[2]
warped_relu = mlp_forward(pts, weights_linear, activation=relu)
plot_warped_grid(ax, XX, YY, warped_relu, 'Same layers + ReLU\n(non-linear warp)', color='royalblue')
ax.set_xlabel(r'$z_1$', fontsize=FS_AXIS)
ax.set_ylabel(r'$z_2$', fontsize=FS_AXIS)
ax.text(
    0.02, 0.98,
    'ReLU folds / clips\n'
    'half-spaces.\n'
    'Lines become\n'
    'piecewise-linear\n'
    'polylines.',
    transform=ax.transAxes, va='top', ha='left', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Typical Deep Learning activations (1D profiles)
# -------------------------------------------------------
fig2, axes2 = new_figure(2, 3)
fig2.suptitle(
    'Step 2: Typical activations in Deep Learning',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

z = np.linspace(-3.5, 3.5, 500)
acts = [
    ('ReLU', relu(z), 'royalblue',
     'max(0, z)\nCNNs / MLPs default.\nHard hinge; sparse.'),
    ('Leaky ReLU', leaky_relu(z), 'dodgerblue',
     'alpha*z if z<0\nKeeps a small slope\nso negatives are not dead.'),
    ('GELU', gelu(z), 'darkorange',
     'Smooth ReLU-like\nTransformers / BERT.\nSoft probabilistic gate.'),
    ('SiLU / Swish', silu(z), 'crimson',
     'z * sigmoid(z)\nModern conv / MLP.\nSmooth, non-monotone tip.'),
    ('tanh', tanh(z), 'seagreen',
     'Saturates at +/-1\nClassic RNNs / gated\nunits; zero-centered.'),
    ('sigmoid', sigmoid(z), 'purple',
     'Squashes to (0,1)\nGates & old nets.\nRare as hidden act now.'),
]
for ax, (name, yv, color, note) in zip(axes2.ravel(), acts):
    ax.plot(z, yv, color=color, linewidth=3.0)
    ax.axhline(0, color='k', alpha=0.25)
    ax.axvline(0, color='k', alpha=0.25)
    ax.set_title(name, fontsize=FS_AX_TITLE)
    ax.set_xlabel('pre-activation z', fontsize=FS_AXIS)
    ax.set_ylabel(r'$\sigma(z)$', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.98, note,
        transform=ax.transAxes, ha='left', va='top', fontsize=FS_ANNOT - 1,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
    )

fig2.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: One neuron -- surfaces and contour lines
# -------------------------------------------------------
# A monotone activation of ONE linear form keeps PARALLEL contour lines.
# Non-linearity shows up as uneven spacing / clipping of the HEIGHT, not as bent contours.
fig3, axes3 = new_figure(2, 3)
fig3.suptitle(
    'Step 3: One neuron -- surfaces and contour lines (still parallel)',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

xs = np.linspace(-2.5, 2.5, 160)
ys = np.linspace(-2.5, 2.5, 160)
XXS, YYS = np.meshgrid(xs, ys)
# Fixed linear pre-activation: a tilted plane
Zlin = 1.15 * XXS + 0.85 * YYS - 0.15

surface_specs = [
    ('Linear score  z = w·x+b', Zlin, 'The raw plane.\nContour lines =\nparallel hyperplanes.'),
    ('ReLU(z)', relu(Zlin), 'Hinges flat below 0.\nContours still parallel;\nnegative half is dead.'),
    ('GELU(z)', gelu(Zlin), 'Soft hinge.\nSame parallel family;\nsmoother height ramp.'),
    ('SiLU(z)', silu(Zlin), 'Slight undershoot\nfor z<0.\nContours still straight.'),
    ('tanh(z)', tanh(Zlin), 'Saturates both sides.\nContours crowd near\nthe decision band.'),
    ('sigmoid(z)', sigmoid(Zlin), 'Soft step to (0,1).\nLevel sets = same\nparallel lines.'),
]

for ax, (title, field, note) in zip(axes3.ravel(), surface_specs):
    vmax = float(np.max(np.abs(field))) + 1e-6
    im = ax.contourf(XXS, YYS, field, levels=18, cmap='coolwarm',
                     vmin=-vmax, vmax=vmax, alpha=0.95)
    cs = ax.contour(XXS, YYS, field, levels=10, colors='k', linewidths=0.9, alpha=0.75)
    ax.clabel(cs, inline=True, fontsize=FS_TICK - 3, fmt='%.1f')
    ax.set_title(title, fontsize=FS_AX_TITLE - 2)
    ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=FS_TICK - 2)
    ax.text(
        0.02, 0.02, note,
        transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_ANNOT - 1,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
    )

axes3[0, 0].text(
    0.02, 0.98,
    'Key: for ONE unit, activation\n'
    'warps HEIGHT along the plane.\n'
    'Contour lines do NOT bend yet.',
    transform=axes3[0, 0].transAxes, ha='left', va='top', fontsize=FS_ANNOT - 1,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig3.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Many neurons -- curved contour manifolds
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 3)
fig4.suptitle(
    'Step 4: Many neurons bend the contours into a non-linear manifold',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# Several units with DIFFERENT orientations; mix them into a scalar field
rng_act = np.random.default_rng(5)
n_units = 5
W_u = rng_act.normal(0, 1.0, size=(n_units, 2))
W_u /= np.linalg.norm(W_u, axis=1, keepdims=True) + 1e-8
b_u = rng_act.uniform(-0.8, 0.8, size=n_units)
a_u = rng_act.normal(0, 1.0, size=n_units)  # output mixing weights

pre = XXS[..., None] * W_u[:, 0] + YYS[..., None] * W_u[:, 1] + b_u  # (H,W,U)


def mix_field(act_fn):
    return np.tensordot(act_fn(pre), a_u, axes=([2], [0]))


mix_specs = [
    ('Sum of ReLUs', mix_field(relu),
     'Piecewise-linear\nlandscape.\nLevel sets kink\nat hinge boundaries.'),
    ('Sum of GELUs', mix_field(gelu),
     'Same recipe,\nsmoother hills.\nContours curve\nwithout sharp folds.'),
    ('Sum of tanhs', mix_field(tanh),
     'Bounded bumps.\nSaturating units\nmake soft ridges\nand closed bowls.'),
]

for ax, (title, field, note) in zip(axes4, mix_specs):
    vmax = float(np.percentile(np.abs(field), 99)) + 1e-6
    ax.contourf(XXS, YYS, field, levels=20, cmap='coolwarm',
                vmin=-vmax, vmax=vmax, alpha=0.95)
    cs = ax.contour(XXS, YYS, field, levels=12, colors='k', linewidths=1.0, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=FS_TICK - 3, fmt='%.1f')
    # Draw each unit's zero line (the hinge / decision of that neuron)
    for u in range(n_units):
        # w0 x + w1 y + b = 0  =>  y = -(w0 x + b)/w1
        if abs(W_u[u, 1]) > 1e-6:
            xline = np.linspace(-2.5, 2.5, 50)
            yline = -(W_u[u, 0] * xline + b_u[u]) / W_u[u, 1]
            mask = (yline >= -2.5) & (yline <= 2.5)
            ax.plot(xline[mask], yline[mask], color='0.25', ls='--', lw=1.0, alpha=0.55)
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.text(
        0.02, 0.02, note,
        transform=ax.transAxes, ha='left', va='bottom', fontsize=FS_ANNOT - 1,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
    )

axes4[0].plot([], [], color='0.25', ls='--', lw=1.0, label='each unit zero-line')
axes4[0].legend(loc='upper right', fontsize=FS_LEGEND)
axes4[2].text(
    0.98, 0.98,
    'Intuition:\n'
    'each activation paints\n'
    'a soft half-space.\n'
    'Their SUM makes curved\n'
    'contour manifolds.\n'
    'Depth then warps those\n'
    'again (next step).',
    transform=axes4[2].transAxes, ha='right', va='top', fontsize=FS_ANNOT - 1,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig4.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Layer-by-layer manifold warping (ReLU net)
# -------------------------------------------------------
fig5, axes5 = new_figure(1, 4)
fig5.suptitle(
    'Step 5: Depth + ReLU progressively folds the input manifold',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# Color points by angle to show how neighborhoods move
angles = np.arctan2(pts[:, 1], pts[:, 0])
cmap = plt.cm.hsv

H = pts.copy()
stage_data = [('Input', H.copy())]
H = relu(H @ W1 + b1)
stage_data.append(('After Linear+ReLU 1', H.copy()))
H = relu(H @ W2 + b2)
stage_data.append(('After Linear+ReLU 2', H.copy()))
H = H @ W3 + b3
stage_data.append(('After final Linear', H.copy()))

for ax, (title, Hstage) in zip(axes5, stage_data):
    ax.scatter(Hstage[:, 0], Hstage[:, 1], c=angles, cmap=cmap, s=10, alpha=0.9)
    # also draw coarse warped grid for structure
    n = XX.shape[0]
    WX = Hstage[:, 0].reshape(n, n)
    WY = Hstage[:, 1].reshape(n, n)
    for i in range(0, n, 2):
        ax.plot(WX[i, :], WY[i, :], color='k', lw=0.4, alpha=0.35)
        ax.plot(WX[:, i], WY[:, i], color='k', lw=0.4, alpha=0.35)
    ax.set_title(title, fontsize=FS_AX_TITLE - 4)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=FS_TICK - 2)
    ax.grid(True, alpha=0.25)

axes5[0].text(
    0.02, 0.02,
    'Color = input angle.\n'
    'Watch neighborhoods\n'
    'stretch, fold, stack.',
    transform=axes5[0].transAxes, ha='left', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig5.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Decision boundary -- linear vs non-linear classifier
# -------------------------------------------------------
# Concentric rings: NO straight line can separate them -- ReLU can.
n_each = 220
ang = RNG.uniform(0, 2 * np.pi, n_each)
r_in = RNG.normal(0.55, 0.07, n_each)
r_out = RNG.normal(1.55, 0.08, n_each)
ring0 = np.stack([r_in * np.cos(ang), r_in * np.sin(ang)], axis=1)
ring1 = np.stack([r_out * np.cos(ang), r_out * np.sin(ang)], axis=1)
X_cls = np.vstack([ring0, ring1])
y_cls = np.array([0] * n_each + [1] * n_each)


def train_classifier(use_relu, steps=4000, lr=0.12, H=48, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.6, size=(2, H))
    b1 = np.zeros(H)
    W2 = rng.normal(0, 0.6, size=(H, 1))
    b2 = np.zeros(1)

    def forward(X):
        z1 = X @ W1 + b1
        h = relu(z1) if use_relu else z1
        logits = (h @ W2 + b2).ravel()
        return logits, h, z1

    for t in range(steps):
        # Mild lr decay helps the ReLU net settle a clean ring boundary.
        step_lr = lr * (0.4 + 0.6 * (1.0 - t / steps))
        logits, h, z1 = forward(X_cls)
        p = 1 / (1 + np.exp(-np.clip(logits, -20, 20)))
        dlogits = (p - y_cls) / len(y_cls)
        dW2 = h.T @ dlogits[:, None]
        db2 = dlogits.sum()
        dh = dlogits[:, None] * W2.T
        if use_relu:
            dz1 = dh * (z1 > 0)
        else:
            dz1 = dh
        dW1 = X_cls.T @ dz1
        db1 = dz1.sum(axis=0)
        W1 -= step_lr * dW1
        b1 -= step_lr * db1
        W2 -= step_lr * dW2
        b2 -= step_lr * db2

    return (W1, b1, W2, b2), forward


fig6, axes6 = new_figure(1, 2)
fig6.suptitle(
    'Step 6: Concentric rings -- a linear boundary MUST fail; ReLU can close a loop',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

xx = np.linspace(-2.2, 2.2, 260)
yy = np.linspace(-2.2, 2.2, 260)
XXC, YYC = np.meshgrid(xx, yy)
grid = np.stack([XXC.ravel(), YYC.ravel()], axis=1)
cmap_bg = ListedColormap(['#f5c6c6', '#c6d4f5'])

configs = [
    (False, 'Linear net (no activation)\nboundary = straight cut -- hopeless here', axes6[0]),
    (True, 'MLP with ReLU\nboundary can form a closed ring', axes6[1]),
]

accs = []
for use_relu, title, ax in configs:
    params, forward = train_classifier(use_relu=use_relu, seed=2)
    logits_g, _, _ = forward(grid)
    logits_data, _, _ = forward(X_cls)
    acc = float(np.mean((logits_data > 0).astype(int) == y_cls))
    accs.append(acc)
    Z = (logits_g > 0).astype(float).reshape(XXC.shape)
    ax.contourf(XXC, YYC, Z, levels=[-0.1, 0.5, 1.1], cmap=cmap_bg, alpha=0.9)
    ax.contour(XXC, YYC, logits_g.reshape(XXC.shape), levels=[0.0],
               colors='k', linewidths=2.4)
    ax.scatter(X_cls[y_cls == 0, 0], X_cls[y_cls == 0, 1],
               s=16, c='crimson', edgecolor='k', linewidth=0.15, label='inner ring')
    ax.scatter(X_cls[y_cls == 1, 0], X_cls[y_cls == 1, 1],
               s=16, c='royalblue', edgecolor='k', linewidth=0.15, label='outer ring')
    ax.set_title(title + f'\ntrain accuracy = {100 * acc:.0f}%', fontsize=FS_AX_TITLE - 2)
    ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=FS_LEGEND)

axes6[0].text(
    0.98, 0.02,
    'Any hyperplane cuts\n'
    'BOTH rings.\n'
    f'Accuracy stuck ~{100 * accs[0]:.0f}%.',
    transform=axes6[0].transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
axes6[1].text(
    0.98, 0.02,
    'Black curve = decision\n'
    'manifold in input space.\n'
    'ReLU pieces assemble\n'
    f'into a loop (~{100 * accs[1]:.0f}%).',
    transform=axes6[1].transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

fig6.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: 1D curve approximation by piecewise linear ReLU net
# -------------------------------------------------------
fig7, ax7 = new_figure()
fig7.suptitle(
    'Step 7: ReLU nets build non-linear curves from linear pieces',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

x = np.linspace(-3, 3, 500)


def target_curve(x):
    return np.sin(1.7 * x) + 0.3 * np.cos(3.1 * x)


# Fit a wide shallow ReLU net in 1D by least squares on random ReLU features
H = 25
rng = np.random.default_rng(3)
centers = rng.uniform(-3, 3, size=H)
scales = rng.uniform(0.6, 2.0, size=H)
# features phi_j(x) = ReLU(s_j (x - c_j))
Phi = np.stack([relu(scales[j] * (x - centers[j])) for j in range(H)], axis=1)
Phi = np.concatenate([Phi, np.ones((len(x), 1))], axis=1)  # bias
y = target_curve(x)
coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
y_hat = Phi @ coef

# Also show a few individual ReLU bumps (scaled)
ax7.plot(x, y, 'k--', lw=2.5, label='target non-linear curve')
ax7.plot(x, y_hat, color='royalblue', lw=2.8, label=f'ReLU expansion (H={H})')
for j in range(0, H, 5):
    ax7.plot(x, coef[j] * relu(scales[j] * (x - centers[j])),
             color='0.6', lw=1.0, alpha=0.8)
ax7.plot([], [], color='0.6', lw=1.0, label='individual ReLU pieces')
ax7.set_xlabel('x', fontsize=FS_AXIS)
ax7.set_ylabel('y', fontsize=FS_AXIS)
ax7.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax7,
    'Each ReLU is a hinged\n'
    'linear piece.\n\n'
    'A weighted sum of many\n'
    'hinges approximates\n'
    'smooth non-linear\n'
    'manifolds / curves.',
)
ax7.grid(True, alpha=0.3)

fig7.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig7)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
# Quick numeric check: composition of linear maps
W_prod = W1 @ W2 @ W3
err = np.max(np.abs(linear_forward(pts, weights_linear) - (pts @ W_prod + (
    (b1 @ W2 + b2) @ W3 + b3
))))

print("\n" + "=" * 60)
print("NEURAL NON-LINEARITY & MANIFOLDS")
print("=" * 60)
print(f"""
1) Linear depth collapses:
      Linear o Linear o Linear  =  one Affine map
   Max |difference| between 3-layer linear net and single product map
   on the demo grid: {err:.2e}

2) Typical DL activations (ReLU, Leaky ReLU, GELU, SiLU, tanh, sigmoid)
   are pointwise maps on the pre-activation z = w·x+b.
   They stop the linear collapse when placed BETWEEN layers.

3) Contour / surface intuition:
   - ONE activated neuron: contour lines stay PARALLEL (same as the
     linear score). Non-linearity only reshapes the HEIGHT profile
     (hard hinge, soft hinge, saturation, ...).
   - MANY neurons with different orientations: their weighted sum
     bends level sets into CURVED contour manifolds.
   - DEPTH then warps those manifolds again (fold, stretch, stack).

4) Geometry of ReLU nets:
      Piecewise-linear: space splits into polyhedral regions; inside
      each region the map is affine. GELU/SiLU are smooth cousins.

5) Why this matters:
      Real data often lie on non-linear manifolds (circles, moons,
      speech, images). Non-linear nets can straighten / separate them;
      purely linear models cannot.

See also:
  030_model.py  -- linear neuron fails on a cubic
  150_inductive_bias.py -- which non-linear family you choose is a bias
""")
