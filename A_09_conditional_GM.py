import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Grid ---
lim = 5.0
n = 120
x = np.linspace(-lim, lim, n)
y = np.linspace(-lim, lim, n)
X, Y = np.meshgrid(x, y)

# --- Mixture components: (weight, mux, muy, sx, sy, rho) ---
COMPONENTS = [
    (0.45, -1.5, -1.0, 1.0, 0.6,  0.6),   # tilted lobe, lower-left
    (0.30,  1.5,  1.5, 0.8, 1.4, -0.5),   # tall narrow lobe, upper-right
    (0.25,  0.5, -2.0, 1.6, 0.5,  0.0),   # flat wide lobe, bottom
]

def gaussian2d(x, y, mux, muy, sx, sy, rho):
    zx, zy = (x - mux) / sx, (y - muy) / sy
    norm = 1.0 / (2 * np.pi * sx * sy * np.sqrt(1 - rho**2))
    expo = -(zx**2 - 2 * rho * zx * zy + zy**2) / (2 * (1 - rho**2))
    return norm * np.exp(expo)

def joint_pdf(x, y, w_scale=1.0, spread=1.0):
    w0, w1, w2 = 0.45, 0.30, 0.25 * w_scale
    total = w0 + w1 + w2
    weights = [w0 / total, w1 / total, w2 / total]
    out = np.zeros_like(x, dtype=float)
    for w, (_, mux, muy, sx, sy, rho) in zip(weights, COMPONENTS):
        out += w * gaussian2d(x, y, mux, muy, sx * spread, sy * spread, rho)
    return out

# --- Figure & axes ---
fig = plt.figure(figsize=(11, 8))

# 3D plot takes almost all the vertical space
ax = fig.add_axes([0.04, 0.20, 0.92, 0.78], projection='3d')

# Sliders packed tightly at the bottom
slider_h = 0.022       # slider track height
slider_w = 0.72        # slider track width
slider_x = 0.16        # left edge of sliders
gap      = 0.036       # vertical spacing between sliders

ax_w   = fig.add_axes([slider_x, 0.145,             slider_w, slider_h])
ax_spr = fig.add_axes([slider_x, 0.145 - 1 * gap,   slider_w, slider_h])
ax_x0  = fig.add_axes([slider_x, 0.145 - 2 * gap,   slider_w, slider_h])
ax_y0  = fig.add_axes([slider_x, 0.145 - 3 * gap,   slider_w, slider_h])

s_w   = Slider(ax_w,   'w₃ scale', 0.0,  3.0, valinit=1.0, valstep=0.01)
s_spr = Slider(ax_spr, 'spread',   0.4,  2.0, valinit=1.0, valstep=0.01)
s_x0  = Slider(ax_x0,  'x₀',      -4.0,  4.0, valinit=0.0, valstep=0.05)
s_y0  = Slider(ax_y0,  'y₀',      -4.0,  4.0, valinit=0.0, valstep=0.05)

# --- Draw ---
def draw(w_scale, spread, x0, y0):
    ax.clear()

    Z = joint_pdf(X, Y, w_scale=w_scale, spread=spread)

    # Dense wireframe
    ax.plot_wireframe(
        X, Y, Z,
        rstride=1, cstride=1,
        color='steelblue', linewidth=0.25, alpha=0.35,
    )

    # Joint density along the two slice lines
    joint_slice_y = joint_pdf(np.full_like(y, x0), y, w_scale, spread)
    joint_slice_x = joint_pdf(x, np.full_like(x, y0), w_scale, spread)

    dy = y[1] - y[0]
    dx = x[1] - x[0]
    cond_y = joint_slice_y / (joint_slice_y.sum() * dy)
    cond_x = joint_slice_x / (joint_slice_x.sum() * dx)

    ax.plot(np.full_like(y, x0), y, joint_slice_y,
            color='red', lw=1.5, ls='--', label='joint f(x₀, y)')
    ax.plot(np.full_like(y, x0), y, cond_y * 0.9,
            color='red', lw=3, label='P(Y | X = x₀)')
    ax.plot(x, np.full_like(x, y0), joint_slice_x,
            color='orange', lw=1.5, ls='--', label='joint f(x, y₀)')
    ax.plot(x, np.full_like(x, y0), cond_x * 0.9,
            color='orange', lw=3, label='P(X | Y = y₀)')

    mean_y_given_x = (y * cond_y).sum() * dy
    var_y_given_x  = ((y - mean_y_given_x)**2 * cond_y).sum() * dy
    mean_x_given_y = (x * cond_x).sum() * dx
    var_x_given_y  = ((x - mean_x_given_y)**2 * cond_x).sum() * dx

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('density')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_zlim(0, Z.max() * 1.05)
    ax.set_title(
        f'Gaussian mixture — dense wireframe\n'
        f'x₀={x0:.2f}, y₀={y0:.2f} | '
        f'Y|X: mean={mean_y_given_x:.2f}, var={var_y_given_x:.2f} | '
        f'X|Y: mean={mean_x_given_y:.2f}, var={var_x_given_y:.2f}',
        fontsize=10
    )
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=28, azim=-60)

def update(_):
    draw(s_w.val, s_spr.val, s_x0.val, s_y0.val)
    fig.canvas.draw_idle()

for s in (s_w, s_spr, s_x0, s_y0):
    s.on_changed(update)

draw(s_w.val, s_spr.val, s_x0.val, s_y0.val)
plt.show()