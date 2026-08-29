"""
050_sum_of_random.py

Intuition for the mean of TWO identical random variables.

If X1 and X2 are i.i.d. with mean mu and variance sigma^2, then

    M = (X1 + X2) / 2

satisfies

    E[M]   = mu                 (same center)
    Var(M) = sigma^2 / 2        (half the variance)
    Std(M) = sigma / sqrt(2)    (narrower by ~1.41x)

This is the n=2 special case of the Standard Error sigma/sqrt(n),
and the first step toward the Central Limit Theorem (see 040_central_limit_theorem.py).
"""

import torch
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
from scipy.stats import norm

torch.manual_seed(42)
np.random.seed(42)

# Shared style (match 040_central_limit_theorem.py)


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
    """Legend and note on the right flank, split above/below the vertical center.

    Both sit near mid-height so they stay clear of the central peak, and the
    split keeps the legend from covering the text box.
    """
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
# Setup: two i.i.d. Gaussians
# -------------------------------------------------------
MU, SIGMA = 0.0, 1.0
N_PAIRS = 20000

X1 = MU + SIGMA * torch.randn(N_PAIRS)
X2 = MU + SIGMA * torch.randn(N_PAIRS)  # identical distribution, independent draw
M = (X1 + X2) / 2.0                     # mean of the two

sigma_m_theory = SIGMA / np.sqrt(2)


# -------------------------------------------------------
# FIGURE 1: One draw vs the average of two
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    r'Step 1: $X$ vs $M=(X_1+X_2)/2$  --  same mean, half the variance',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

x_grid = np.linspace(MU - 4 * SIGMA, MU + 4 * SIGMA, 500)

# Left: single variable
ax = axes1[0]
ax.hist(X1.numpy(), bins=60, density=True, alpha=0.7, color='steelblue',
        edgecolor='black', linewidth=0.3, label=r'samples of $X_1$')
ax.plot(x_grid, norm.pdf(x_grid, MU, SIGMA), 'r-', linewidth=2.5,
        label=rf'$N(\mu={MU:g},\ \sigma^2={SIGMA**2:g})$')
ax.axvline(MU, color='k', linestyle='--', alpha=0.6)
ax.set_title(r'One random variable $X$', fontsize=FS_AX_TITLE)
ax.set_xlabel('value', fontsize=FS_AXIS)
ax.set_ylabel('Density', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    f'empirical std = {X1.std(unbiased=True).item():.3f}\n'
    f'theory sigma  = {SIGMA:.3f}',
)
ax.grid(True, alpha=0.3)
ax.set_xlim(x_grid[0], x_grid[-1])

# Right: average of two
ax = axes1[1]
ax.hist(M.numpy(), bins=60, density=True, alpha=0.7, color='darkorange',
        edgecolor='black', linewidth=0.3, label=r'samples of $M$')
ax.plot(x_grid, norm.pdf(x_grid, MU, sigma_m_theory), 'r-', linewidth=2.5,
        label=rf'$N(\mu,\ \sigma^2/2)$  (SE$\approx{sigma_m_theory:.3f}$)')
# Ghost of the original PDF for comparison
ax.plot(x_grid, norm.pdf(x_grid, MU, SIGMA), 'b--', linewidth=2, alpha=0.5,
        label=r'original $N(\mu,\sigma^2)$')
ax.axvline(MU, color='k', linestyle='--', alpha=0.6)
ax.set_title(r'Mean of two: $M=(X_1+X_2)/2$', fontsize=FS_AX_TITLE)
ax.set_xlabel('value', fontsize=FS_AXIS)
ax.set_ylabel('Density', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    f'empirical std = {M.std(unbiased=True).item():.3f}\n'
    f'theory sigma/sqrt(2) = {sigma_m_theory:.3f}',
)
ax.grid(True, alpha=0.3)
ax.set_xlim(x_grid[0], x_grid[-1])

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Geometric intuition in the (X1, X2) plane
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    r'Step 2: In the $(X_1,X_2)$ plane, the average is the projection onto $X_1=X_2$',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# Subsample for a readable scatter
idx = torch.randperm(N_PAIRS)[:3000]
m_plot = M[idx].numpy()
m_lim = float(np.percentile(np.abs(m_plot), 98))  # stretch colors for contrast
ax2.scatter(
    X1[idx].numpy(), X2[idx].numpy(),
    c=m_plot, cmap='seismic', s=14, alpha=0.75, linewidths=0,
    vmin=-m_lim, vmax=m_lim,
)
cbar = fig2.colorbar(ax2.collections[0], ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label(r'$M=(X_1+X_2)/2$', fontsize=FS_AXIS)
cbar.ax.tick_params(labelsize=FS_TICK)

# Diagonal X1 = X2 (points with X1=X2 have M = X1)
lim = MU + 3.5 * SIGMA
ax2.plot([-lim, lim], [-lim, lim], 'k-', linewidth=2, label=r'$X_1 = X_2$ (then $M=X_1$)')
# Contours of constant average: X1 + X2 = 2m  =>  X2 = 2m - X1
for m_level, ls in [(-1, ':'), (0, '--'), (1, ':')]:
    ax2.plot(
        [-lim, lim], [2 * m_level + lim, 2 * m_level - lim],
        linestyle=ls, color='green', linewidth=2,
        label=rf'$M = {m_level:g}$' if m_level != 0 else r'$M = 0$',
    )

ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.set_aspect('equal')
ax2.set_xlabel(r'$X_1$', fontsize=FS_AXIS)
ax2.set_ylabel(r'$X_2$', fontsize=FS_AXIS)
ax2.tick_params(axis='both', labelsize=FS_TICK)
ax2.set_title(
    r'Noise perpendicular to the diagonal cancels in the average',
    fontsize=FS_AX_TITLE,
)
ax2.legend(
    loc='lower right',
    bbox_to_anchor=(0.98, 0.54),
    fontsize=FS_LEGEND,
    framealpha=0.95,
    borderaxespad=0.0,
)
ax2.text(
    0.02, 0.50,
    'Along the diagonal, the signal remains.\n'
    'Across the diagonal, noise cancels in M.',
    transform=ax2.transAxes,
    fontsize=FS_ANNOT,
    va='center', ha='left',
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Algebra — why variance halves
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 3)
fig3.suptitle(
    r'Step 3: Why $\mathrm{Var}(M)=\sigma^2/2$ -- and why the sum gets wider',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

S = X1 + X2  # sum of two

panels = [
    # data, mean, theory_std, title, color, var_note
    (X1, MU, SIGMA, r'$X_1\sim N(\mu,\sigma^2)$', 'steelblue',
     r'$\mathrm{Var}=\sigma^2$'),
    (M, MU, sigma_m_theory, r'$M=(X_1+X_2)/2$', 'darkorange',
     r'$\mathrm{Var}=\sigma^2/2$'),
    (S, 2 * MU, SIGMA * np.sqrt(2), r'$S=X_1+X_2$', 'seagreen',
     r'$\mathrm{Var}=2\sigma^2$'),
]

for ax, (data, mean_plot, theory_std, title, color, var_note) in zip(axes3, panels):
    ax.hist(data.numpy(), bins=60, density=True, alpha=0.7, color=color,
            edgecolor='black', linewidth=0.3)
    xs = np.linspace(mean_plot - 4 * theory_std, mean_plot + 4 * theory_std, 400)
    ax.plot(xs, norm.pdf(xs, mean_plot, theory_std), 'r-', linewidth=2.5,
            label='theory PDF')
    ax.axvline(mean_plot, color='k', linestyle='--', alpha=0.6)
    emp_var = data.var(unbiased=True).item()
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.set_xlabel('value', fontsize=FS_AXIS)
    ax.set_ylabel('Density', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(ax, var_note + f'\nemp. var = {emp_var:.3f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Same story for a NON-Gaussian pair (Uniform)
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    'Step 4: Works for non-Gaussian twins too (Uniform) -- mean preserved, spread shrinks',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# Uniform[-3, 3]: mu=0, var=3
U1 = torch.rand(N_PAIRS) * 6.0 - 3.0
U2 = torch.rand(N_PAIRS) * 6.0 - 3.0
UM = (U1 + U2) / 2.0
u_mu, u_var = 0.0, 3.0
u_std = np.sqrt(u_var)
u_std_m = u_std / np.sqrt(2)

ax = axes4[0]
ax.hist(U1.numpy(), bins=50, density=True, alpha=0.75, color='steelblue',
        edgecolor='black', linewidth=0.3, label=r'$U_1$ samples')
ax.axhline(1.0 / 6.0, color='r', linewidth=2.5, label=r'Uniform PDF $=1/6$')
ax.axvline(u_mu, color='k', linestyle='--', alpha=0.6)
ax.set_title(r'One Uniform[-3, 3]', fontsize=FS_AX_TITLE)
ax.set_xlabel('value', fontsize=FS_AXIS)
ax.set_ylabel('Density', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    f'std = {U1.std(unbiased=True).item():.3f}\n'
    f'theory = {u_std:.3f}',
)
ax.grid(True, alpha=0.3)
ax.set_xlim(-3.5, 3.5)

ax = axes4[1]
ax.hist(UM.numpy(), bins=50, density=True, alpha=0.75, color='darkorange',
        edgecolor='black', linewidth=0.3, label=r'$M=(U_1+U_2)/2$')
# Irwin-Hall / triangular PDF for average of two Uniform[-3,3]
# Average of two Uniform[a,b] is a triangular (piecewise linear) distribution.
# Overlay Normal(mu, sigma^2/2) as the n=2 CLT approximation.
xs = np.linspace(-3.5, 3.5, 400)
ax.plot(xs, norm.pdf(xs, u_mu, u_std_m), 'r-', linewidth=2.5,
        label=rf'CLT approx $N(0,\sigma^2/2)$')
ax.axhline(1.0 / 6.0, color='b', linestyle='--', linewidth=2, alpha=0.5,
           label='original Uniform height')
ax.axvline(u_mu, color='k', linestyle='--', alpha=0.6)
ax.set_title(r'Mean of two Uniforms', fontsize=FS_AX_TITLE)
ax.set_xlabel('value', fontsize=FS_AXIS)
ax.set_ylabel('Density', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    f'std = {UM.std(unbiased=True).item():.3f}\n'
    f'theory sigma/sqrt(2) = {u_std_m:.3f}\n'
    '(shape already more peaked)',
)
ax.grid(True, alpha=0.3)
ax.set_xlim(-3.5, 3.5)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("MEAN OF TWO IDENTICAL RANDOM VARIABLES")
print("=" * 60)
print(f"""
Setup: X1, X2 iid with mean mu and variance sigma^2.

Define the average:
    M = (X1 + X2) / 2

Then ALWAYS (no Gaussian assumption needed):
    E[M]   = mu
    Var(M) = sigma^2 / 2
    Std(M) = sigma / sqrt(2)   ~= 0.707 * sigma

Intuition (Figure 2):
    - Noise shared along X1=X2 survives in M (the signal direction)
    - Noise opposite across the diagonal cancels in the average

Contrast with the SUM S = X1 + X2:
    Var(S) = 2 * sigma^2   (wider, not narrower)

So: averaging shrinks uncertainty; summing grows it.

This is why n=2 already helps, and why in general
    Std(sample mean) = sigma / sqrt(n)
(see 010_gaussian.py and 040_central_limit_theorem.py).

Demo numbers (Gaussian, sigma={SIGMA:g}):
    std(X1) = {X1.std(unbiased=True).item():.4f}   (theory {SIGMA:.4f})
    std(M)  = {M.std(unbiased=True).item():.4f}   (theory {sigma_m_theory:.4f})
    std(S)  = {S.std(unbiased=True).item():.4f}   (theory {SIGMA * np.sqrt(2):.4f})
""")
