"""
140_neural_non_linearity.py

How neural nets create non-linearity and non-linear manifolds.

Key facts:
  1) Stacking linear layers WITHOUT activations is still just ONE linear map.
  2) Pointwise non-linearities (ReLU, tanh, ...) break that collapse.
  3) The result is a piecewise warping of space: linear regions glued into a
     flexible non-linear manifold / decision boundary.

Companion to 020_model.py (simple vs complex net) and 110_inductive_bias.py.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

np.random.seed(42)
RNG = np.random.default_rng(42)

FS_FIGURE_TITLE = 28
FS_AX_TITLE = 22
FS_AXIS = 18
FS_ANNOT = 12
FS_LEGEND = 12

SCREEN_FRACTION = 0.95
FIGURE_DPI = 150

plt.rcParams.update({
    'figure.titlesize': FS_FIGURE_TITLE,
    'axes.titlesize': FS_AX_TITLE,
    'axes.labelsize': FS_AXIS,
    'xtick.labelsize': FS_AXIS,
    'ytick.labelsize': FS_AXIS,
    'legend.fontsize': FS_LEGEND,
    'figure.dpi': FIGURE_DPI,
})


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


def tanh(z):
    return np.tanh(z)


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
    ax.tick_params(axis='both', labelsize=FS_AXIS)
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
ax.tick_params(axis='both', labelsize=FS_AXIS)
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
# FIGURE 2: Activation functions -- local non-linear gadgets
# -------------------------------------------------------
fig2, axes2 = new_figure(1, 3)
fig2.suptitle(
    'Step 2: Activations are the source of non-linearity',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

z = np.linspace(-3, 3, 400)
acts = [
    ('ReLU', relu(z), 'royalblue', 'max(0,z)\npiecewise linear'),
    ('tanh', tanh(z), 'darkorange', 'saturates at ±1\nsmooth bends'),
    ('sigmoid', 1 / (1 + np.exp(-z)), 'seagreen', 'squashes to (0,1)\nsmooth gates'),
]
for ax, (name, y, color, note) in zip(axes2, acts):
    ax.plot(z, y, color=color, linewidth=3.0)
    ax.axhline(0, color='k', alpha=0.25)
    ax.axvline(0, color='k', alpha=0.25)
    ax.set_title(name, fontsize=FS_AX_TITLE)
    ax.set_xlabel('z', fontsize=FS_AXIS)
    ax.set_ylabel('activation', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_AXIS)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98, 0.05, note,
        transform=ax.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
    )

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Layer-by-layer manifold warping (ReLU net)
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 4)
fig3.suptitle(
    'Step 3: Depth + ReLU progressively folds the input manifold',
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

for ax, (title, Hstage) in zip(axes3, stage_data):
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
    ax.tick_params(axis='both', labelsize=FS_AXIS - 2)
    ax.grid(True, alpha=0.25)

axes3[0].text(
    0.02, 0.02,
    'Color = input angle.\n'
    'Watch neighborhoods\n'
    'stretch, fold, stack.',
    transform=axes3[0].transAxes, ha='left', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Decision boundary -- linear vs non-linear classifier
# -------------------------------------------------------
# Two interlocking moons-like blobs via simple construction
n_each = 120
t = RNG.uniform(0, np.pi, n_each)
r = RNG.normal(1.0, 0.12, n_each)
moon1 = np.stack([r * np.cos(t), r * np.sin(t)], axis=1) + np.array([0.0, 0.15])
moon2 = np.stack([1.0 - r * np.cos(t), 0.5 - r * np.sin(t)], axis=1) + np.array([-0.2, -0.2])
X_cls = np.vstack([moon1, moon2])
y_cls = np.array([0] * n_each + [1] * n_each)


def train_classifier(use_relu, steps=1500, lr=0.15, H=20, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.5, size=(2, H))
    b1 = np.zeros(H)
    W2 = rng.normal(0, 0.5, size=(H, 1))
    b2 = np.zeros(1)

    def forward(X):
        z1 = X @ W1 + b1
        h = relu(z1) if use_relu else z1
        logits = (h @ W2 + b2).ravel()
        return logits, h, z1

    for _ in range(steps):
        logits, h, z1 = forward(X_cls)
        # sigmoid BCE grad
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
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    return (W1, b1, W2, b2), forward


fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    'Step 4: Non-linear features make non-linear decision manifolds',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

xx = np.linspace(-1.5, 2.2, 200)
yy = np.linspace(-1.2, 1.6, 200)
XXC, YYC = np.meshgrid(xx, yy)
grid = np.stack([XXC.ravel(), YYC.ravel()], axis=1)
cmap_bg = ListedColormap(['#f5c6c6', '#c6d4f5'])

configs = [
    (False, 'Linear net (no activation)\nboundary must be a hyperplane', axes4[0]),
    (True, 'MLP with ReLU\nboundary can curve / fold', axes4[1]),
]

for use_relu, title, ax in configs:
    params, forward = train_classifier(use_relu=use_relu, seed=1)
    logits_g, _, _ = forward(grid)
    Z = (logits_g > 0).astype(float).reshape(XXC.shape)
    ax.contourf(XXC, YYC, Z, levels=[-0.1, 0.5, 1.1], cmap=cmap_bg, alpha=0.9)
    ax.contour(XXC, YYC, logits_g.reshape(XXC.shape), levels=[0.0],
               colors='k', linewidths=2.0)
    ax.scatter(X_cls[y_cls == 0, 0], X_cls[y_cls == 0, 1],
               s=18, c='crimson', edgecolor='k', linewidth=0.2, label='class 0')
    ax.scatter(X_cls[y_cls == 1, 0], X_cls[y_cls == 1, 1],
               s=18, c='royalblue', edgecolor='k', linewidth=0.2, label='class 1')
    ax.set_title(title, fontsize=FS_AX_TITLE - 2)
    ax.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', labelsize=FS_AXIS)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='lower left', fontsize=FS_LEGEND)

axes4[1].text(
    0.98, 0.98,
    'Black curve = decision\n'
    'manifold in input space.\n'
    'ReLU lets it bend.',
    transform=axes4[1].transAxes, ha='right', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: 1D curve approximation by piecewise linear ReLU net
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    'Step 5: ReLU nets build non-linear curves from linear pieces',
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
ax5.plot(x, y, 'k--', lw=2.5, label='target non-linear curve')
ax5.plot(x, y_hat, color='royalblue', lw=2.8, label=f'ReLU expansion (H={H})')
for j in range(0, H, 5):
    ax5.plot(x, coef[j] * relu(scales[j] * (x - centers[j])),
             color='0.6', lw=1.0, alpha=0.8)
ax5.plot([], [], color='0.6', lw=1.0, label='individual ReLU pieces')
ax5.set_xlabel('x', fontsize=FS_AXIS)
ax5.set_ylabel('y', fontsize=FS_AXIS)
ax5.tick_params(axis='both', labelsize=FS_AXIS)
add_vcenter_legend_and_note(
    ax5,
    'Each ReLU is a hinged\n'
    'linear piece.\n\n'
    'A weighted sum of many\n'
    'hinges approximates\n'
    'smooth non-linear\n'
    'manifolds / curves.',
)
ax5.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig5)


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

2) Activations stop the collapse:
      ReLU / tanh / sigmoid applied between layers make the overall
      map NON-linear. Straight grids become folded manifolds.

3) Geometry:
      - ReLU nets are piecewise-linear: space splits into polyhedral
        regions; inside each region the map is affine.
      - Stacking layers creates richer folds => curved decision
        boundaries and flexible data manifolds.

4) Why this matters:
      Real data often lie on non-linear manifolds (circles, moons,
      speech, images). Non-linear nets can straighten / separate them;
      purely linear models cannot.

See also:
  020_model.py  -- linear neuron fails on a cubic
  110_inductive_bias.py -- which non-linear family you choose is a bias
""")
