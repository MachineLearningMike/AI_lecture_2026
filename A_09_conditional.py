import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Grid ---
lim = 4.0
n = 90
x = np.linspace(-lim, lim, n)
y = np.linspace(-lim, lim, n)
X, Y = np.meshgrid(x, y)

# --- Slice planes ---
x0, y0 = 1.0, 1.0

def joint_pdf(x, y, sx, sy, rho):
    zx, zy = x / sx, y / sy
    norm = 1.0 / (2 * np.pi * sx * sy * np.sqrt(1 - rho**2))
    expo = -(zx**2 - 2 * rho * zx * zy + zy**2) / (2 * (1 - rho**2))
    return norm * np.exp(expo)

def marginal_x(x, sx):
    return (1.0 / (np.sqrt(2 * np.pi) * sx)) * np.exp(-x**2 / (2 * sx**2))

# --- Figure & axes ---
fig = plt.figure(figsize=(11, 8))
ax = fig.add_axes([0.05, 0.28, 0.9, 0.68], projection='3d')

# Slider axes: [left, bottom, width, height]
ax_rho  = fig.add_axes([0.15, 0.16, 0.7, 0.03])
ax_sig  = fig.add_axes([0.15, 0.10, 0.7, 0.03])
ax_x0   = fig.add_axes([0.15, 0.04, 0.7, 0.03])

s_rho = Slider(ax_rho, 'ρ',   -0.95, 0.95, valinit=0.0,  valstep=0.01)
s_sig = Slider(ax_sig, 'σ',    0.3,  2.5,  valinit=1.0,  valstep=0.01)
s_x0  = Slider(ax_x0,  'x₀',  -3.0,  3.0,  valinit=1.0,  valstep=0.01)

# --- Draw function ---
def draw(rho, sigma, x0):
    ax.clear()

    # Joint surface
    Z = joint_pdf(X, Y, sigma, sigma, rho)
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.75,
                    linewidth=0, antialiased=True, rstride=2, cstride=2)

    # Conditional P(Y | X = x0)
    joint_slice_y = joint_pdf(x0, y, sigma, sigma, rho)
    mx0 = marginal_x(x0, sigma)
    cond_y = joint_slice_y / mx0

    ax.plot(np.full_like(y, x0), y, joint_slice_y,
            color='red', lw=1.5, ls='--', label=f'joint slice at X={x0:.2f}')
    ax.plot(np.full_like(y, x0), y, cond_y * 0.9,
            color='red', lw=3, label='P(Y | X=x₀)')

    # Conditional P(X | Y = y0)
    joint_slice_x = joint_pdf(x, y0, sigma, sigma, rho)
    my0 = marginal_x(y0, sigma)
    cond_x = joint_slice_x / my0

    ax.plot(x, np.full_like(x, y0), joint_slice_x,
            color='orange', lw=1.5, ls='--', label=f'joint slice at Y={y0:.2f}')
    ax.plot(x, np.full_like(x, y0), cond_x * 0.9,
            color='orange', lw=3, label='P(X | Y=y₀)')

    # Conditional stats (analytic)
    mean_cond = rho * (sigma / sigma) * x0       # = rho * x0 for sx=sy
    var_cond  = sigma**2 * (1 - rho**2)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('density')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_zlim(0, 0.5)
    ax.set_title(
        f'P(X,Y)=N(0, Σ),  ρ={rho:.2f},  σ={sigma:.2f},  x₀={x0:.2f}\n'
        f'P(Y|X=x₀):  mean={mean_cond:.2f},  var={var_cond:.2f}',
        fontsize=11
    )
    ax.legend(loc='upper left', fontsize=8)

def update(_):
    draw(s_rho.val, s_sig.val, s_x0.val)
    fig.canvas.draw_idle()

s_rho.on_changed(update)
s_sig.on_changed(update)
s_x0.on_changed(update)

draw(s_rho.val, s_sig.val, s_x0.val)
plt.show()