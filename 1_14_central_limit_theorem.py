"""
Central Limit Theorem (CLT):

  Formal statement first, then an intuitive demo:

  No matter how non-Gaussian the data are, the distribution of the
  SAMPLE MEAN becomes approximately Gaussian as the sample size grows.
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
    run_slide_deck,
)
apply_rcparams(plt)
import numpy as np
from scipy.stats import norm

torch.manual_seed(42)
np.random.seed(42)

NUM_EXPERIMENTS = 4000  # how many sample-means to collect per panel
SAMPLE_SIZES = [1, 2, 5, 10, 30, 50]


# -------------------------------------------------------
# Parent distributions (intentionally NOT Gaussian)
# Each sampler(shape) returns a tensor of that shape.
# -------------------------------------------------------
def sample_uniform(shape) -> torch.Tensor:
    # Uniform on [-3, 3]: mean 0, variance = 6^2 / 12 = 3
    return torch.rand(shape) * 6.0 - 3.0


def sample_exponential(shape) -> torch.Tensor:
    # Exp(rate=1): mean 1, variance 1  (strongly right-skewed)
    return torch.empty(shape).exponential_(lambd=1.0)


def sample_bimodal(shape) -> torch.Tensor:
    # Equal mixture of N(-3, 1) and N(+3, 1): mean 0
    # Variance = E[X^2] = 0.5*((-3)^2+1) + 0.5*(3^2+1) = 10
    data = torch.randn(shape)
    # Independently assign each draw to the left or right component
    mask = torch.rand(shape) < 0.5
    data = torch.where(mask, data - 3.0, data + 3.0)
    return data


DISTRIBUTIONS = [
    {
        'name': 'Uniform[-3, 3]',
        'sampler': sample_uniform,
        'color': 'steelblue',
        'mu': 0.0,
        'var': 3.0,
        'note': 'flat and bounded',
        'pdf_text': r'$f(x)=\dfrac{1}{6}\ \mathrm{for}\ x \in [-3,3]$',
    },
    {
        'name': 'Exponential(1)',
        'sampler': sample_exponential,
        'color': 'darkorange',
        'mu': 1.0,
        'var': 1.0,
        'note': 'right-skewed',
        'pdf_text': r'$f(x)=e^{-x}\ \mathrm{for}\ x \geq 0$',
    },
    {
        'name': 'Bimodal mixture',
        'sampler': sample_bimodal,
        'color': 'seagreen',
        'mu': 0.0,
        'var': 10.0,
        'note': 'two peaks',
        'pdf_text': r'$f(x)=\frac{1}{2}\phi(x;-3,1)+\frac{1}{2}\phi(x;3,1)$',
    },
]


def sample_means(sampler, n: int, num_experiments: int = NUM_EXPERIMENTS) -> torch.Tensor:
    """Draw (num_experiments x n) and average across the sample axis."""
    return sampler((num_experiments, n)).mean(dim=1)


# Shared font sizes -- identical roles use identical sizes across ALL figures

# Same screen-fit rate and DPI for every figure


def screen_figsize(fraction=SCREEN_FRACTION, dpi=FIGURE_DPI):
    """Figure size in inches at the shared screen-fit rate."""
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
    """Create a figure using the same screen fraction and DPI as every other plot."""
    subplot_kwargs.setdefault('figsize', screen_figsize())
    subplot_kwargs.setdefault('dpi', FIGURE_DPI)
    return plt.subplots(*subplot_args, **subplot_kwargs)


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_LOOP = "#2874a6"


# -------------------------------------------------------
# Step 1: Formal statement
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: Central Limit Theorem -- formal statement",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.25, 11.2, 1.45, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.95,
        r"Setup: i.i.d. $X_1,X_2,\ldots$ with mean $\mu$ and finite variance $\sigma^2 > 0$"
        + "\n"
        + r"Sample mean:  $\bar X_n = (X_1 + \cdots + X_n)/n$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.25, 11.2, 2.75, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 5.55,
        "CLT (standardized form)",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 4.25,
        r"$\dfrac{\sqrt{n}\,(\bar{X}_n - \mu)}{\sigma}"
        r"\;\overset{d}{\longrightarrow}\; N(0,1)$"
        + "\n\n"
        r"Equivalently, for large $n$:   "
        r"$\bar X_n \approx \mathrm{Normal}(\mu,\, \sigma^2 / n)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.4, 11.2, 2.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.7,
        "The parent distribution need not be Gaussian.\n"
        r"CLT says the distribution of the sample mean becomes approximately Gaussian."
        + "\n"
        r"LLN: $\bar X_n$ goes to $\mu$.   "
        r"CLT: the leftover fluctuation has a predictable bell shape.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# FIGURE 1: Look at the parent distributions themselves
# -------------------------------------------------------
def build_step_2():
    fig1, axes1 = new_figure(1, 3)
    fig1.suptitle(
        'Step 2: The parent distributions are not Gaussian',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    for ax, dist in zip(axes1, DISTRIBUTIONS):
        samples = dist['sampler'](5000)
        ax.hist(
            samples.numpy(), bins=50, density=True, alpha=0.75,
            color=dist['color'], edgecolor='black', linewidth=0.4,
        )
        ax.axvline(dist['mu'], color='red', linestyle='--', linewidth=1.5,
                   label=f"mean = {dist['mu']:g}")
        ax.set_title(f"{dist['name']}\n({dist['note']})", fontsize=FS_AX_TITLE)
        ax.set_xlabel('x', fontsize=FS_AXIS)
        ax.set_ylabel('Density', fontsize=FS_AXIS)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        ax.text(
            0.5, 0.5, dist['pdf_text'],
            transform=ax.transAxes, fontsize=FS_ANNOT,
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        )
        ax.legend(loc='upper left', fontsize=FS_LEGEND)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig1


# -------------------------------------------------------
# FIGURE 2: CLT in action -- one high-res figure per distribution
# Each figure: 2x3 grid (6 sample sizes)
# -------------------------------------------------------
STEP2_EXPERIMENTS = 8000  # denser histograms at high resolution


def _build_clt_grid(dist, step_num: int):
    fig2, axes2 = new_figure(2, 3)
    fig2.suptitle(
        f"Step {step_num}: CLT for {dist['name']} -- the distribution of sample means",
        fontsize=FS_FIGURE_TITLE, fontweight='bold', y=0.98,
    )

    # Shared x-limits across the 6 panels so narrowing is obvious
    raw = dist['sampler'](5000)
    pad = 0.5
    x_lo = raw.min().item() - pad
    x_hi = raw.max().item() + pad

    for ax, n in zip(axes2.flat, SAMPLE_SIZES):
        means = sample_means(dist['sampler'], n, num_experiments=STEP2_EXPERIMENTS)

        ax.hist(
            means.numpy(), bins=50, density=True, alpha=0.75,
            color=dist['color'], edgecolor='black', linewidth=0.4,
        )

        se = np.sqrt(dist['var'] / n)
        xs = np.linspace(x_lo, x_hi, 600)
        ax.plot(
            xs, norm.pdf(xs, dist['mu'], se),
            'r-', linewidth=2.5, label='CLT Normal',
        )

        ax.set_xlim(x_lo, x_hi)
        ax.set_title(f'n = {n}', fontsize=FS_AX_TITLE, fontweight='bold', pad=4)
        ax.set_xlabel('Sample mean', fontsize=FS_AXIS)
        ax.set_ylabel('Density', fontsize=FS_AXIS)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        ax.text(
            0.96, 0.05,
            f"mean={means.mean().item():.2f}\n"
            f"std={means.std(unbiased=True).item():.2f}\n"
            f"SE={se:.2f}",
            transform=ax.transAxes, fontsize=FS_ANNOT,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
        )
        if n > 1:
            ax.legend(loc='upper right', fontsize=FS_LEGEND)
        ax.grid(True, alpha=0.3)

    # Extra gap under the figure title; leave bottom strip for Prev/Next
    fig2.subplots_adjust(
        left=0.06, right=0.99, bottom=0.10, top=0.88,
        hspace=0.42, wspace=0.18,
    )
    return fig2


def build_step_3():
    return _build_clt_grid(DISTRIBUTIONS[0], 3)


def build_step_4():
    return _build_clt_grid(DISTRIBUTIONS[1], 4)


def build_step_5():
    return _build_clt_grid(DISTRIBUTIONS[2], 5)


# -------------------------------------------------------
# FIGURE 3: Same idea, zoomed -- standardize to N(0,1)
# Shows all parents collapse onto the same bell curve
# -------------------------------------------------------
def build_step_6():
    fig3, axes3 = new_figure(1, 3)
    fig3.suptitle(
        'Step 6: After standardizing, every parent looks like N(0,1)',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    n_focus = 30
    z_grid = np.linspace(-4, 4, 400)

    for ax, dist in zip(axes3, DISTRIBUTIONS):
        means = sample_means(dist['sampler'], n_focus)
        se = np.sqrt(dist['var'] / n_focus)
        z = (means.numpy() - dist['mu']) / se

        ax.hist(
            z, bins=40, density=True, alpha=0.75,
            color=dist['color'], edgecolor='black', linewidth=0.4,
            label=f'standardized means (n={n_focus})',
        )
        ax.plot(z_grid, norm.pdf(z_grid, 0, 1), 'r-', linewidth=2, label='N(0,1)')
        ax.set_title(dist['name'], fontsize=FS_AX_TITLE)
        ax.set_xlabel('Standardized sample mean', fontsize=FS_AXIS)
        ax.set_ylabel('Density', fontsize=FS_AXIS)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        ax.set_xlim(-4, 4)
        ax.legend(loc='center', fontsize=FS_LEGEND)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig3


# -------------------------------------------------------
# FIGURE 4: Standard Error shrinks like 1/sqrt(n)
# -------------------------------------------------------
def build_step_7():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        r'Step 7: Precision improves as $1/\sqrt{n}$  (standard error)',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    n_values = np.array([1, 2, 5, 10, 20, 30, 50, 100])

    for dist in DISTRIBUTIONS:
        empirical_std = []
        for n in n_values:
            means = sample_means(dist['sampler'], int(n), num_experiments=3000)
            empirical_std.append(means.std(unbiased=True).item())
        theory = np.sqrt(dist['var'] / n_values)
        ax4.plot(n_values, empirical_std, 'o-', color=dist['color'],
                 linewidth=2, label=f"{dist['name']} (simulated)")
        ax4.plot(n_values, theory, '--', color=dist['color'], alpha=0.7,
                 linewidth=1.5)

    ax4.set_xlabel('Sample size n', fontsize=FS_AXIS)
    ax4.set_ylabel('Std. dev. of the sample mean (SE)', fontsize=FS_AXIS)
    ax4.tick_params(axis='both', labelsize=FS_TICK)
    ax4.legend(fontsize=FS_LEGEND)
    ax4.grid(True, alpha=0.3)
    ax4.set_title(
        r'Dashed: theory $\sigma/\sqrt{n}$   |   Dots: simulation',
        fontsize=FS_AX_TITLE,
    )

    plt.tight_layout()
    return fig4


run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
    build_step_6,
    build_step_7,
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("CENTRAL LIMIT THEOREM -- INTUITIVE TAKEAWAYS")
print("=" * 60)
print("""
1. Parent distributions can look NOTHING like a bell curve
   (flat Uniform, skewed Exponential, two-peaked Bimodal).

2. Still: the SAMPLE MEAN of n draws becomes approximately
   Normal(mu, sigma/sqrt(n)) as n grows.
   Watch Steps 3-5: by n=30 the red CLT curve fits well.

3. Standardization (Step 6) makes the point sharp:
   (mean - mu) / (sigma/sqrt(n))  -->  N(0,1)
   for EVERY parent distribution.

4. Precision (Step 7) improves like 1/sqrt(n):
   - 4x more data  =>  SE halves
   - 100x more data => SE shrinks 10x
   More data makes the mean more trustworthy.

5. Why this matters for ML / science:
   - Polls and A/B tests rely on means being roughly Gaussian
   - Confidence intervals and p-values lean on the CLT
   - Even messy real-world data often yields well-behaved averages

Note: Figure 3 used a Gaussian source, so means were
EXACTLY normal for every n (that was Standard Error, not CLT).
This script is the real CLT demo -- non-Gaussian parents.
""")
