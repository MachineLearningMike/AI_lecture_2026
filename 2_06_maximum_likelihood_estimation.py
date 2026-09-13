"""
2_06_maximum_likelihood_estimation.py

Maximum Likelihood Estimation (MLE): choose parameters that make the
observed data most likely.

Sketch:
  1-4. Gaussian story -- max log L <=> MSE; Laplace <=> MAE
  5. What MLE means
  6. Coin-toss example (7 heads, 3 tails)
  7. Likelihood vs probability viewpoints
  8. Log-likelihood, NLL, and cross-entropy
  9. MLE inside Naive Bayes
  10. MLE vs MAP (add a prior)
"""

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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
    run_slide_deck,
)
apply_rcparams(plt)
import numpy as np
from scipy.stats import norm, laplace

torch.manual_seed(42)
np.random.seed(42)

# Shared style


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


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')


def round_box(ax, x, y, w, h, facecolor, edgecolor='0.2', lw=2.0):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.14',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_WARN = '#f8d7da'
C_OLD = '#fce4d6'
C_LOOP = '#2874a6'
C_CODE = '#fbfcf8'


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
def build_step_1():
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
    return fig1


# -------------------------------------------------------
# FIGURE 2: Gaussian log-likelihood <-> MSE
# -------------------------------------------------------
def build_step_2():
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
    return fig2


# -------------------------------------------------------
# FIGURE 3: Why NOT MAE for Gaussians?
# -------------------------------------------------------
def build_step_3():
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
    return fig3


# -------------------------------------------------------
# FIGURE 4: Outlier stress test -- MSE vs MAE estimates
# -------------------------------------------------------
def build_step_4():
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
    return fig4


# -------------------------------------------------------
# Step 5: What MLE means
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        'Step 5: What is MLE?',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        'Maximum Likelihood Estimation:\n'
        'choose the model parameters that make the observed data most likely.',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
        fontweight='bold', color='#f9e79f',
    )

    round_box(ax, 0.5, 2.7, 11.0, 2.4, C_NEW, C_LOOP, lw=2.2)
    ax.text(
        6.0, 4.3,
        r'$\hat\theta=\arg\max_\theta L(\theta)$'
        '     with     '
        r'$L(\theta)=P(D\mid\theta)$',
        ha='center', va='center', fontsize=FS_AX_TITLE - 3,
        fontweight='bold', color=C_DARK,
    )
    ax.text(
        6.0, 3.3,
        r'Data fixed.  Parameter varies.  Pick the most plausible $\theta$.',
        ha='center', va='center', fontsize=FS_ANNOT, color='#333333',
    )

    round_box(ax, 0.5, 0.45, 11.0, 1.9, C_MUTE, '0.45', lw=1.6)
    ax.text(
        6.0, 1.4,
        'MLE = the method (or the resulting parameter value:\n'
        'the maximum likelihood estimate).',
        ha='center', va='center', fontsize=FS_ANNOT, color='#333333',
    )
    return fig


# -------------------------------------------------------
# Step 6: Coin example
# -------------------------------------------------------
def build_step_6():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r'Step 6: Coin example -- 7 heads, 3 tails',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Setup', fontsize=FS_AX_TITLE - 2, fontweight='bold')
    round_box(ax, 0.5, 6.8, 9.0, 2.5, C_NEW, C_LOOP, lw=2.0)
    ax.text(5.0, 8.5, r'Unknown  $\theta=P(\mathrm{heads})$',
            ha='center', fontsize=FS_ANNOT, fontweight='bold', color=C_DARK)
    ax.text(5.0, 7.5, 'H H T H T H H H T H\n→ 7 heads, 3 tails',
            ha='center', va='center', fontsize=FS_TICK + 1)

    round_box(ax, 0.5, 3.4, 9.0, 2.8, C_MUTE, '0.45', lw=1.6)
    ax.text(5.0, 5.4, 'Likelihood', ha='center',
            fontsize=FS_TICK + 1, fontweight='bold', color=C_DARK)
    ax.text(5.0, 4.3,
            r'$L(\theta)=\theta^7(1-\theta)^3$',
            ha='center', va='center', fontsize=FS_ANNOT + 1, color='#1c2833')

    round_box(ax, 0.5, 0.6, 9.0, 2.3, C_GOAL, C_OK, lw=2.0)
    ax.text(5.0, 1.75,
            r'$\hat\theta=\arg\max_\theta L(\theta)=0.7$',
            ha='center', va='center', fontsize=FS_ANNOT,
            fontweight='bold', color=C_OK)

    ax = axes[1]
    th = np.linspace(0.01, 0.99, 400)
    L = th ** 7 * (1 - th) ** 3
    ax.plot(th, L, color='seagreen', lw=2.8)
    ax.axvline(0.7, color='red', ls='--', lw=2.0, label=r'MLE $\hat\theta=0.7$')
    ax.set_title(r'$L(\theta)=\theta^7(1-\theta)^3$', fontsize=FS_AX_TITLE - 2)
    ax.set_xlabel(r'$\theta=P(\mathrm{heads})$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$L(\theta)$', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98, 0.08,
        'Best estimate of the\ncoin: 70% heads.',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=FS_ANNOT,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
    )

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 7: Likelihood vs probability
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        'Step 7: Why "likelihood" rather than "probability"?',
        fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.7, 11.0, 1.8, C_MUTE, '0.45', lw=1.6)
    ax.text(
        6.0, 6.6,
        r'Same expression:  $P(\mathrm{data}\mid\theta)$',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
        fontweight='bold', color=C_DARK,
    )

    round_box(ax, 0.5, 2.9, 5.3, 2.4, C_OLD, '#e67e22', lw=2.2)
    ax.text(3.15, 4.7, 'Probability viewpoint', ha='center',
            fontsize=FS_ANNOT, fontweight='bold', color='#a04000')
    ax.text(
        3.15, 3.7,
        r'Fix $\theta$.' + '\n'
        'Ask how probable the data are.\n\n'
        r'Vary the data; hold $\theta$.',
        ha='center', va='center', fontsize=FS_TICK + 1,
    )

    round_box(ax, 6.2, 2.9, 5.3, 2.4, C_GOAL, C_OK, lw=2.2)
    ax.text(8.85, 4.7, 'Likelihood viewpoint', ha='center',
            fontsize=FS_ANNOT, fontweight='bold', color=C_OK)
    ax.text(
        8.85, 3.7,
        'Fix the data.\n'
        r'Ask which $\theta$ makes them' + '\n'
        'most plausible.\n\n'
        r'Vary $\theta$; hold the data.',
        ha='center', va='center', fontsize=FS_TICK + 1,
    )

    round_box(ax, 0.5, 0.4, 11.0, 2.1, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 1.45,
        r'For MLE:  $L(\theta)=P(D\mid\theta)$  with $D$ fixed,'
        '\n'
        r'then  $\hat\theta=\arg\max_\theta P(D\mid\theta)$.',
        ha='center', va='center', fontsize=FS_ANNOT,
        fontweight='bold', color='#f9e79f',
    )
    return fig


# -------------------------------------------------------
# Step 8: log-likelihood / NLL / cross-entropy
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        'Step 8: Why maximize log-likelihood? (and the NLL link)',
        fontsize=FS_FIGURE_TITLE - 4, fontweight='bold',
    )
    blank_slide(ax)

    round_box(ax, 0.4, 5.55, 11.2, 2.0, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 6.85,
        'Independent observations → a product',
        ha='center', fontsize=FS_ANNOT, fontweight='bold', color=C_DARK,
    )
    ax.text(
        6.0, 6.1,
        r'$P(D\mid\theta)=\prod_{i=1}^{N} P(x_i\mid\theta)$'
        '     → tiny numbers, awkward derivatives',
        ha='center', va='center', fontsize=FS_TICK + 1, color='#333333',
    )

    round_box(ax, 0.4, 3.15, 11.2, 2.05, C_MUTE, '0.45', lw=1.6)
    ax.text(
        6.0, 4.5,
        r'$\log L(\theta)=\sum_i \log P(x_i\mid\theta)$'
        '      (log is monotonic → same maximizer)',
        ha='center', va='center', fontsize=FS_ANNOT, fontweight='bold', color=C_DARK,
    )
    ax.text(
        6.0, 3.6,
        r'$\arg\max_\theta L(\theta)=\arg\max_\theta \log L(\theta)$',
        ha='center', va='center', fontsize=FS_TICK + 1, color='#333333',
    )

    round_box(ax, 0.4, 0.35, 11.2, 2.45, C_GOAL, C_OK, lw=2.2)
    ax.text(
        6.0, 2.15,
        'Modern ML often minimizes negative log-likelihood',
        ha='center', fontsize=FS_ANNOT, fontweight='bold', color=C_OK,
    )
    ax.text(
        6.0, 1.15,
        r'$\mathcal{L}(\theta)=-\sum_i\log P_\theta(x_i)$'
        '\n'
        'Cross-entropy for a classifier is NLL of '
        r'$P_\theta(y\mid x)$.',
        ha='center', va='center', fontsize=FS_TICK + 1, color='#145a32',
    )
    return fig


# -------------------------------------------------------
# Step 9: MLE in Naive Bayes
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        'Step 9: MLE inside Naive Bayes',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.55, 11.0, 2.0, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 6.55,
        'Naive Bayes needs '
        r'$P(x_j\mid C)$ and $P(C)$.'
        '\n'
        'Estimate them from training counts with MLE.',
        ha='center', va='center', fontsize=FS_ANNOT, color=C_DARK,
    )

    round_box(ax, 0.5, 2.7, 11.0, 2.5, C_MUTE, '0.45', lw=1.6)
    ax.text(
        6.0, 4.4,
        'Example: 80 of 100 spam emails contain "free"',
        ha='center', fontsize=FS_ANNOT, fontweight='bold', color=C_DARK,
    )
    ax.text(
        6.0, 3.4,
        r'$\hat P(\mathrm{free}\mid\mathrm{spam})=\frac{80}{100}=0.8$',
        ha='center', va='center', fontsize=FS_AX_TITLE - 3,
        fontweight='bold', color='#1c2833',
    )

    round_box(ax, 0.5, 0.4, 11.0, 1.95, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.35,
        'Estimate probabilities with MLE from the training data,\n'
        "then use Bayes' theorem to classify new points.",
        ha='center', va='center', fontsize=FS_ANNOT, fontweight='bold', color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 10: MLE vs MAP
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        'Step 10: MLE vs MAP -- add a prior preference',
        fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
    )
    blank_slide(ax)

    round_box(ax, 0.5, 4.55, 5.3, 3.0, C_NEW, C_LOOP, lw=2.2)
    ax.text(3.15, 6.95, 'MLE', ha='center',
            fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_DARK)
    ax.text(
        3.15, 5.7,
        'Maximize likelihood only\n\n'
        r'$\hat\theta_{\mathrm{MLE}}='
        r'\arg\max_\theta P(D\mid\theta)$',
        ha='center', va='center', fontsize=FS_ANNOT,
    )

    round_box(ax, 6.2, 4.55, 5.3, 3.0, C_GOAL, C_OK, lw=2.2)
    ax.text(8.85, 6.95, 'MAP', ha='center',
            fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_OK)
    ax.text(
        8.85, 5.7,
        'Maximize posterior\n\n'
        r'$\hat\theta_{\mathrm{MAP}}='
        r'\arg\max_\theta P(D\mid\theta)P(\theta)$',
        ha='center', va='center', fontsize=FS_ANNOT,
    )

    round_box(ax, 0.5, 2.35, 11.0, 1.8, C_MUTE, '0.45', lw=1.6)
    ax.text(
        6.0, 3.25,
        r'$P(\theta\mid D)\propto P(D\mid\theta)P(\theta)$',
        ha='center', va='center', fontsize=FS_ANNOT + 1,
        fontweight='bold', color=C_DARK,
    )

    round_box(ax, 0.5, 0.4, 11.0, 1.6, C_OLD, '#e67e22', lw=2.0)
    ax.text(
        6.0, 1.2,
        'MAP = MLE + a prior preference for certain parameter values.\n'
        'Training a probabilistic model is often: make the training data likely.',
        ha='center', va='center', fontsize=FS_ANNOT, color='#6e2c00',
    )
    return fig


run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
    build_step_6,
    build_step_7,
    build_step_8,
    build_step_9,
    build_step_10,
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("MAXIMUM LIKELIHOOD ESTIMATION")
print("=" * 60)
print(f"""
Core idea:
  hat theta = argmax_theta P(D | theta)
  Data fixed; parameters vary.

Coin demo: 7H 3T  =>  L(theta)=theta^7 (1-theta)^3  =>  hat theta=0.7

Log-likelihood:
  same maximizer as L; NLL = -sum log P_theta(x_i)
  Cross-entropy training is conditional NLL.

Gaussian residual noise  =>  MLE is MSE (Steps 1-4).
Laplace residual noise   =>  MLE is MAE.

Demo numbers (n={N}, true mu={MU_TRUE:g}, sigma={SIGMA:g}):
  sample mean   (Gaussian MLE / MSE) = {mu_mean:.4f}
  sample median (Laplace MLE / MAE)  = {mu_median:.4f}

Naive Bayes uses MLE for P(x_j|C) and P(C) from counts.
MAP adds a prior: argmax P(D|theta) P(theta).
""")
