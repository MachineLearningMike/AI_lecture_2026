"""
120_maximum_likelihood.py

Intuition for Maximum Likelihood Estimation (MLE), and why for a
Gaussian noise model:

    maximizing the likelihood  <==>  minimizing MSE  (squared error)

NOT mean absolute error (MAE).  MAE is the MLE for Laplace noise instead.

Companion to 060_gaussian.py (Gaussian PDF) and 080_model.py (MSE training).
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
from scipy.stats import norm, laplace

torch.manual_seed(42)
np.random.seed(42)

# Shared style (match 04 / 05)


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


# -------------------------------------------------------
# Data: n iid draws from N(mu_true, sigma^2)
# Goal: recover mu by maximum likelihood
# -------------------------------------------------------
MU_TRUE = 2.0
SIGMA = 1.0
N = 40

data = MU_TRUE + SIGMA * torch.randn(N)
data_np = data.numpy()

mu_grid = np.linspace(MU_TRUE - 3.0, MU_TRUE + 3.0, 400)


def gaussian_loglik(mu, x, sigma=SIGMA):
    """log L(mu) for iid N(mu, sigma^2), up to nothing -- full log-likelihood."""
    mu = np.asarray(mu, dtype=float)
    # broadcast: (n_mu, n_data)
    resid = x[None, :] - mu[:, None]
    return (
        -0.5 * len(x) * np.log(2 * np.pi * sigma ** 2)
        - (resid ** 2).sum(axis=1) / (2 * sigma ** 2)
    )


def mse_objective(mu, x):
    """Mean squared error as a function of candidate mean mu."""
    mu = np.asarray(mu, dtype=float)
    resid = x[None, :] - mu[:, None]
    return (resid ** 2).mean(axis=1)


def mae_objective(mu, x):
    """Mean absolute error as a function of candidate mean mu."""
    mu = np.asarray(mu, dtype=float)
    resid = x[None, :] - mu[:, None]
    return np.abs(resid).mean(axis=1)


loglik = gaussian_loglik(mu_grid, data_np)
mse = mse_objective(mu_grid, data_np)
mae = mae_objective(mu_grid, data_np)

mu_mle = mu_grid[np.argmax(loglik)]          # = sample mean (for Gaussian)
mu_mse = mu_grid[np.argmin(mse)]             # same as MLE
mu_mae = mu_grid[np.argmin(mae)]             # = sample median
mu_mean = float(data_np.mean())
mu_median = float(np.median(data_np))


# -------------------------------------------------------
# FIGURE 1: What is likelihood?
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    r'Step 1: Likelihood = how well a candidate $\mu$ explains the data',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# Left: data + three candidate Gaussians
ax = axes1[0]
ax.scatter(data_np, np.zeros_like(data_np), s=40, color='k', zorder=5,
           label='observed data', clip_on=False)
xs = np.linspace(MU_TRUE - 4, MU_TRUE + 4, 400)
for mu_c, color, ls in [
    (0.0, 'steelblue', '--'),
    (MU_TRUE, 'seagreen', '-'),
    (4.0, 'darkorange', '--'),
]:
    ax.plot(xs, norm.pdf(xs, mu_c, SIGMA), color=color, linestyle=ls, linewidth=2.5,
            label=rf'candidate $N(\mu={mu_c:g},1)$')
ax.set_title('Same data, different candidate means', fontsize=FS_AX_TITLE)
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('Density', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.set_ylim(bottom=-0.02)
add_vcenter_legend_and_note(
    ax,
    r'Likelihood of a $\mu$:' + '\n'
    r'$L(\mu)=\prod_i\,f(x_i\mid\mu)$' + '\n'
    'Taller density at each\n'
    'point means higher L',
)
ax.grid(True, alpha=0.3)

# Right: L(mu) and log L(mu)
ax = axes1[1]
# Use likelihood relative to max for readable scale
lik = np.exp(loglik - loglik.max())
ax.plot(mu_grid, lik, color='seagreen', linewidth=2.5, label=r'$L(\mu)/L_{\max}$')
ax.axvline(mu_mle, color='red', linestyle='--', linewidth=2, label=rf'MLE $\hat\mu={mu_mle:.2f}$')
ax.axvline(MU_TRUE, color='k', linestyle=':', linewidth=1.5, alpha=0.7,
           label=rf'true $\mu={MU_TRUE:g}$')
ax.set_title(r'Likelihood as a function of $\mu$', fontsize=FS_AX_TITLE)
ax.set_xlabel(r'candidate $\mu$', fontsize=FS_AXIS)
ax.set_ylabel('relative likelihood', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    f'n = {N} Gaussian samples\n'
    f'sample mean = {mu_mean:.2f}\n'
    'MLE peaks at the\n'
    'sample mean',
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Gaussian log-likelihood <-> MSE
# -------------------------------------------------------
fig2, axes2 = new_figure(1, 2)
fig2.suptitle(
    r'Step 2: For Gaussians, maximizing $\log L$ is exactly minimizing MSE',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes2[0]
ax.plot(mu_grid, loglik, color='seagreen', linewidth=2.5, label=r'$\log L(\mu)$')
ax.axvline(mu_mle, color='red', linestyle='--', linewidth=2, label='argmax')
ax.set_title(r'Maximize $\log L(\mu)$', fontsize=FS_AX_TITLE)
ax.set_xlabel(r'candidate $\mu$', fontsize=FS_AXIS)
ax.set_ylabel(r'$\log L(\mu)$', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    r'$\log L(\mu)=$' + '\n'
    r'const $-\frac{1}{2\sigma^2}\sum(x_i-\mu)^2$' + '\n\n'
    'Only the squared term\n'
    r'depends on $\mu$',
)
ax.grid(True, alpha=0.3)

ax = axes2[1]
ax.plot(mu_grid, mse, color='darkorange', linewidth=2.5, label=r'MSE$(\mu)$')
ax.axvline(mu_mse, color='red', linestyle='--', linewidth=2, label='argmin')
ax.set_title(r'Minimize MSE$(\mu)=\frac{1}{n}\sum(x_i-\mu)^2$', fontsize=FS_AX_TITLE)
ax.set_xlabel(r'candidate $\mu$', fontsize=FS_AXIS)
ax.set_ylabel('MSE', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    f'argmin MSE = {mu_mse:.2f}\n'
    f'argmax log L = {mu_mle:.2f}\n'
    'Same location.\n\n'
    'So training with MSE\n'
    'is Gaussian MLE.',
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Why NOT MAE for Gaussians?
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    r'Step 3: MAE minimizes a different objective -- the Laplace MLE',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes3[0]
mse_n = (mse - mse.min()) / (mse.max() - mse.min() + 1e-12)
mae_n = (mae - mae.min()) / (mae.max() - mae.min() + 1e-12)
ax.plot(mu_grid, mse_n, color='darkorange', linewidth=2.5, label='MSE (scaled)')
ax.plot(mu_grid, mae_n, color='purple', linewidth=2.5, label='MAE (scaled)')
ax.axvline(mu_mse, color='darkorange', linestyle='--', linewidth=2,
           label=rf'MSE min $\rightarrow$ mean={mu_mean:.2f}')
ax.axvline(mu_mae, color='purple', linestyle='--', linewidth=2,
           label=rf'MAE min $\rightarrow$ median={mu_median:.2f}')
ax.set_title('MSE vs MAE objectives (normalized shapes)', fontsize=FS_AX_TITLE)
ax.set_xlabel(r'candidate $\mu$', fontsize=FS_AXIS)
ax.set_ylabel('scaled objective', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    'Gaussian MLE -> mean\n'
    'Laplace MLE -> median\n\n'
    'MAE is not the\n'
    'Gaussian likelihood',
)
ax.grid(True, alpha=0.3)

# Right: which noise model matches which loss
ax = axes3[1]
z = np.linspace(-4, 4, 400)
ax.plot(z, norm.pdf(z, 0, 1), color='darkorange', linewidth=2.5,
        label=r'Gaussian $\propto e^{-z^2/2}$  -> MSE')
ax.plot(z, laplace.pdf(z, 0, 1 / np.sqrt(2)), color='purple', linewidth=2.5,
        label=r'Laplace $\propto e^{-\sqrt{2}|z|}$  -> MAE')
# Match variance of Laplace for fair visual: scale b = 1/sqrt(2) gives var=1
ax.set_title('Noise PDF decides the loss', fontsize=FS_AX_TITLE)
ax.set_xlabel(r'residual $z = x - \mu$', fontsize=FS_AXIS)
ax.set_ylabel('Density', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    'Neg. log-density:\n'
    'Gaussian -> z^2  (MSE)\n'
    'Laplace  -> |z|  (MAE)\n\n'
    'Pick the loss that\n'
    'matches the noise.',
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Outlier stress test -- MSE vs MAE estimates
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    r'Step 4: One outlier -- MSE (mean) moves more than MAE (median)',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

data_out = data_np.copy()
data_out[0] = MU_TRUE + 8.0  # plant a large outlier

mse_out = mse_objective(mu_grid, data_out)
mae_out = mae_objective(mu_grid, data_out)
mu_mse_out = mu_grid[np.argmin(mse_out)]
mu_mae_out = mu_grid[np.argmin(mae_out)]

ax = axes4[0]
ax.scatter(data_out[1:], np.zeros(N - 1), s=40, color='k', zorder=5, label='inliers')
ax.scatter([data_out[0]], [0.0], s=80, color='red', zorder=6, label='outlier')
ax.axvline(mu_mse_out, color='darkorange', linestyle='--', linewidth=2,
           label=rf'MSE fit (mean)={mu_mse_out:.2f}')
ax.axvline(mu_mae_out, color='purple', linestyle='--', linewidth=2,
           label=rf'MAE fit (median)={mu_mae_out:.2f}')
ax.axvline(MU_TRUE, color='k', linestyle=':', linewidth=1.5, alpha=0.7,
           label=rf'true $\mu={MU_TRUE:g}$')
ax.set_title('Fits after adding one outlier', fontsize=FS_AX_TITLE)
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('')
ax.set_yticks([])
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    f'clean mean  = {mu_mean:.2f}\n'
    f'outlier mean= {data_out.mean():.2f}\n'
    f'outlier med.= {np.median(data_out):.2f}\n\n'
    'MSE chases outliers;\n'
    'MAE is more robust.',
)
ax.grid(True, alpha=0.3)
ax.set_xlim(MU_TRUE - 3.5, MU_TRUE + 9.0)

ax = axes4[1]
mse_out_n = (mse_out - mse_out.min()) / (mse_out.max() - mse_out.min() + 1e-12)
mae_out_n = (mae_out - mae_out.min()) / (mae_out.max() - mae_out.min() + 1e-12)
ax.plot(mu_grid, mse_out_n, color='darkorange', linewidth=2.5, label='MSE (scaled)')
ax.plot(mu_grid, mae_out_n, color='purple', linewidth=2.5, label='MAE (scaled)')
ax.axvline(mu_mse_out, color='darkorange', linestyle='--', linewidth=2)
ax.axvline(mu_mae_out, color='purple', linestyle='--', linewidth=2)
ax.set_title('Objectives with the outlier present', fontsize=FS_AX_TITLE)
ax.set_xlabel(r'candidate $\mu$', fontsize=FS_AXIS)
ax.set_ylabel('scaled objective', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_vcenter_legend_and_note(
    ax,
    'Still:\n'
    'MSE argmin = mean\n'
    'MAE argmin = median\n\n'
    'Gaussian MLE still\n'
    'means MSE -- even\n'
    'when that is brittle.',
)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("MAXIMUM LIKELIHOOD AND WHY MSE (NOT MAE) FOR GAUSSIANS")
print("=" * 60)
print(f"""
Setup: data x_1..x_n assumed iid ~ N(mu, sigma^2).  sigma known.

Likelihood:
    L(mu) = product_i  N(x_i | mu, sigma^2)

Log-likelihood (easier, same maximizer):
    log L(mu) = const - (1/(2 sigma^2)) * sum_i (x_i - mu)^2

Therefore:
    argmax_mu log L(mu)  =  argmin_mu sum_i (x_i - mu)^2
                         =  argmin_mu MSE(mu)
                         =  sample mean

MAE is a DIFFERENT probabilistic story:
    Laplace noise  =>  log L(mu) = const - (1/b) sum_i |x_i - mu|
                   =>  argmax log L = argmin MAE = sample median

Demo numbers (n={N}, true mu={MU_TRUE:g}, sigma={SIGMA:g}):
    sample mean   (Gaussian MLE / MSE) = {mu_mean:.4f}
    sample median (Laplace MLE / MAE)  = {mu_median:.4f}
    grid argmax logL                   = {mu_mle:.4f}
    grid argmin MSE                    = {mu_mse:.4f}
    grid argmin MAE                    = {mu_mae:.4f}

Takeaway for ML training (see 080_model.py):
    Using MSE loss is not an arbitrary choice -- it IS maximum
    likelihood under Gaussian residual noise.
""")
