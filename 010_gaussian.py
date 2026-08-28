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

# Reproducibility for every figure
torch.manual_seed(42)
np.random.seed(42)


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


def gaussian_pdf(x: torch.Tensor, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    """Gaussian / normal PDF for N(μ, σ²) evaluated at x (PyTorch)."""
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def plot_pdf_curves(ax, means, stds, colors, title):
    """Overlay Gaussian PDFs for several (μ, σ) pairs."""
    x = torch.linspace(-5, 5, 1000)
    for m, s, c in zip(means, stds, colors):
        y = gaussian_pdf(x, m, s)
        ax.plot(x.numpy(), y.numpy(), color=c, linewidth=2, label=f'μ={m}, σ={s}')
    ax.set_title(title, fontsize=FS_AX_TITLE)
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('Probability Density', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, alpha=0.3)


# -------------------------------------------------------
# FIGURE 1: GAUSSIAN DISTRIBUTION OVERVIEW
# -------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig.suptitle(
    'Understanding the Gaussian (Normal) Distribution',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

# Plot 1: PDF - Different Standard Deviations
plot_pdf_curves(
    axes[0, 0],
    means=[0, 0, 0],
    stds=[0.5, 1.0, 2.0],
    colors=['blue', 'red', 'green'],
    title='Effect of Standard Deviation (σ)',
)

# Plot 2: PDF - Different Means
plot_pdf_curves(
    axes[0, 1],
    means=[-2, 0, 2],
    stds=[1.0, 1.0, 1.0],
    colors=['orange', 'purple', 'brown'],
    title='Effect of Mean (μ) - Shifting the Center',
)

# Plot 3: 68-95-99.7 Rule
ax3 = axes[0, 2]
mu, sigma = 0.0, 1.0
x_rule = torch.linspace(-4, 4, 1000)
y_rule = gaussian_pdf(x_rule, mu, sigma)

ax3.plot(x_rule.numpy(), y_rule.numpy(), 'k-', linewidth=2)

colors_shade = ['lightblue', 'lightgreen', 'lightyellow']
ranges = [(-1, 1), (-2, 2), (-3, 3)]
labels_shade = ['68.27% (±1σ)', '95.45% (±2σ)', '99.73% (±3σ)']

# Draw widest band first so narrower bands stay visible on top
for (low, high), color, label in zip(reversed(ranges), reversed(colors_shade), reversed(labels_shade)):
    mask = (x_rule >= low) & (x_rule <= high)
    ax3.fill_between(
        x_rule.numpy()[mask], 0, y_rule.numpy()[mask],
        alpha=0.35, color=color, label=label,
    )

for k, ls in [(1, ':'), (2, '--'), (3, '-.')]:
    ax3.axvline(+k * sigma, color='gray', linestyle=ls, alpha=0.6, linewidth=1)
    ax3.axvline(-k * sigma, color='gray', linestyle=ls, alpha=0.6, linewidth=1)
ax3.axvline(mu, color='red', linestyle='--', alpha=0.7, label=f'μ={mu:g}')

ax3.set_title('The 68-95-99.7 Rule', fontsize=FS_AX_TITLE)
ax3.set_xlabel('x', fontsize=FS_AXIS)
ax3.set_ylabel('Probability Density', fontsize=FS_AXIS)
ax3.tick_params(axis='both', labelsize=FS_TICK)
ax3.legend(fontsize=FS_LEGEND)
ax3.grid(True, alpha=0.3)

# Plot 4: CDF
ax4 = axes[1, 0]
x = torch.linspace(-5, 5, 1000)
for m, s, c in zip([0, 0, 0], [0.5, 1.0, 2.0], ['blue', 'red', 'green']):
    ax4.plot(x.numpy(), norm.cdf(x.numpy(), loc=m, scale=s),
             color=c, linewidth=2, label=f'μ={m}, σ={s}')
ax4.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax4.set_title('CDF: Probability of X ≤ x', fontsize=FS_AX_TITLE)
ax4.set_xlabel('x', fontsize=FS_AXIS)
ax4.set_ylabel('Cumulative Probability', fontsize=FS_AXIS)
ax4.tick_params(axis='both', labelsize=FS_TICK)
ax4.legend(fontsize=FS_LEGEND)
ax4.grid(True, alpha=0.3)

# Plot 5: QQ Plot (Check for Normality)
# Use matching plotting positions so each sample quantile lines up with
# the corresponding theoretical normal quantile.
ax5 = axes[1, 1]
n_qq = 500
normal_data = np.random.normal(0, 1, n_qq)
uniform_data = np.random.uniform(-2, 2, n_qq)
exponential_data = np.random.exponential(1, n_qq)

# (i + 0.5) / n plotting positions → theoretical N(0,1) quantiles
plotting_positions = (np.arange(n_qq) + 0.5) / n_qq
theoretical_quantiles = norm.ppf(plotting_positions)

datasets = [
    (normal_data, 'blue', 'Normal (linear)'),
    (uniform_data, 'red', 'Uniform (S-shaped)'),
    (exponential_data, 'green', 'Exponential (curved)'),
]
for data, color, label in datasets:
    ax5.scatter(
        theoretical_quantiles, np.sort(data),
        alpha=0.45, s=12, color=color, label=label,
    )

ref = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
ax5.plot(ref, ref, 'k--', alpha=0.6, label='y = x (perfect normal)')
ax5.set_title('QQ Plot: Visual Normality Test', fontsize=FS_AX_TITLE)
ax5.set_xlabel('Theoretical Quantiles (N(0,1))', fontsize=FS_AXIS)
ax5.set_ylabel('Sample Quantiles', fontsize=FS_AXIS)
ax5.tick_params(axis='both', labelsize=FS_TICK)
ax5.legend(fontsize=FS_LEGEND)
ax5.grid(True, alpha=0.3)

# Plot 6: Histogram of Samples vs true PDF
ax6 = axes[1, 2]
samples = torch.randn(1000)
ax6.hist(
    samples.numpy(), bins=40, density=True, alpha=0.7,
    color='steelblue', edgecolor='black', linewidth=0.5,
)
x_pdf = torch.linspace(-4, 4, 1000)
ax6.plot(x_pdf.numpy(), gaussian_pdf(x_pdf).numpy(), 'r-', linewidth=2, label='True PDF')
ax6.set_title('Histogram of Samples (n=1000)', fontsize=FS_AX_TITLE)
ax6.set_xlabel('x', fontsize=FS_AXIS)
ax6.set_ylabel('Density', fontsize=FS_AXIS)
ax6.tick_params(axis='both', labelsize=FS_TICK)
ax6.legend(fontsize=FS_LEGEND)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig)

# -------------------------------------------------------
# FIGURE 2: HISTOGRAMS SHOWING SAMPLING (Law of Large Numbers)
# -------------------------------------------------------
fig2, axes2 = plt.subplots(2, 3, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig2.suptitle(
    'Histograms: Sampling from Gaussian Distribution',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

sample_sizes_hist = [10, 50, 100, 500, 1000, 5000]
x_pdf = torch.linspace(-4, 4, 1000)
y_pdf = gaussian_pdf(x_pdf)

for ax, n in zip(axes2.flat, sample_sizes_hist):
    samples = torch.randn(n)

    # Fewer bins for tiny n so the histogram is readable
    bins = max(5, min(40, n // 3))
    ax.hist(
        samples.numpy(), bins=bins, density=True, alpha=0.7,
        color='steelblue', edgecolor='black', linewidth=0.5,
    )
    ax.plot(x_pdf.numpy(), y_pdf.numpy(), 'r-', linewidth=2, label='True PDF')

    mean_val = samples.mean().item()
    std_val = samples.std(unbiased=True).item()

    ax.set_title(f'n = {n}', fontsize=FS_AX_TITLE)
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('Density', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.set_xlim(-4, 4)
    ax.text(
        0.05, 0.95, f'μ̂ = {mean_val:.3f}\nσ̂ = {std_val:.3f}',
        transform=ax.transAxes, fontsize=FS_ANNOT, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
    )
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig2)

# -------------------------------------------------------
# FIGURE 3: DISTRIBUTION OF SAMPLE MEANS (same x-scale)
# For Gaussian data, sample means are exactly N(0, (1/√n)²) = N(0, 1/n).
# The narrowing shows the Standard Error shrinking — not CLT "kicking in".
# (See 040_central_limit_theorem.py for CLT with non-Gaussian sources.)
# -------------------------------------------------------
fig3, axes3 = plt.subplots(2, 3, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig3.suptitle(
    'Distribution of Sample Means (Gaussian source, same x-scale)',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

sample_sizes_means = [2, 5, 10, 30, 50, 100]
num_experiments = 5000
x_min, x_max = -3.0, 3.0
x_means = torch.linspace(x_min, x_max, 1000)

for ax, n in zip(axes3.flat, sample_sizes_means):
    # Vectorized: one (num_experiments × n) draw, then mean over the sample axis
    sample_means = torch.randn(num_experiments, n).mean(dim=1)

    ax.hist(
        sample_means.numpy(), bins=40, density=True, alpha=0.7,
        color='coral', edgecolor='black', linewidth=0.5,
    )

    theoretical_std = 1.0 / np.sqrt(n)  # Standard Error: σ/√n with σ=1
    y_means = gaussian_pdf(x_means, mu=0.0, sigma=theoretical_std)
    ax.plot(
        x_means.numpy(), y_means.numpy(), 'b-', linewidth=2,
        label=f'N(0, (1/√n)²)  SE={theoretical_std:.3f}',
    )

    mean_of_means = sample_means.mean().item()
    std_of_means = sample_means.std(unbiased=True).item()

    ax.set_xlim(x_min, x_max)
    ax.set_title(f'n = {n}', fontsize=FS_AX_TITLE)
    ax.set_xlabel('Sample Mean', fontsize=FS_AXIS)
    ax.set_ylabel('Density', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.text(
        0.05, 0.95, f'μ̂ = {mean_of_means:.3f}\nσ̂ = {std_of_means:.3f}',
        transform=ax.transAxes, fontsize=FS_ANNOT, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
    )
    ax.legend(loc='upper right', fontsize=FS_LEGEND)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig3)

# -------------------------------------------------------
# PRINT KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("KEY INSIGHTS ABOUT GAUSSIAN DISTRIBUTION")
print("=" * 60)
print("""
1. The Bell Curve:
   - Most values cluster around the mean (mu)
   - Values farther from mu become exponentially less likely

2. Standard Deviation (sigma) controls spread:
   - Small sigma -> narrow, tall curve (values tightly clustered)
   - Large sigma -> wide, flat curve (values spread out)

3. The 68-95-99.7 Rule (Empirical Rule):
   - 68.27% of data falls within +/-1 sigma of mu
   - 95.45% of data falls within +/-2 sigma of mu
   - 99.73% of data falls within +/-3 sigma of mu

4. Sampling Behavior (Histograms -- Law of Large Numbers):
   - Small n (e.g., n=10): Histogram looks noisy, doesn't match PDF well
   - Large n (e.g., n=5000): Histogram closely matches the true PDF

5. Distribution of Sample Means (Figure 3, SAME X-SCALE):
   - Source is already Gaussian, so sample means are exactly N(0, 1/n)
     (variance 1/n; SE = 1/sqrt(n)) for every n -- this plot shows the
     Standard Error shrinking, not CLT.
   - n=2:   SE ~ 0.707  (wide)
   - n=10:  SE ~ 0.316
   - n=30:  SE ~ 0.183
   - n=100: SE ~ 0.100  (very precise)
   - As n grows the distribution stays centered at 0 but gets taller and narrower.

6. QQ Plot:
   - Points on a straight line -> data looks Gaussian
   - S-shaped curve -> light tails / bounded support (uniform)
   - Strong curve -> skewness (exponential)
""")

print("\n" + "=" * 60)
print("CENTRAL LIMIT THEOREM - KEY TAKEAWAY")
print("=" * 60)
print("""
The CLT says: the distribution of SAMPLE MEANS approaches a Gaussian
even when the underlying data are NOT Gaussian (given large enough n).

Figure 3 above uses a Gaussian source, so the means are already exactly
normal for every n -- look at 040_central_limit_theorem.py for the real CLT
demo with Uniform / Exponential / Bimodal sources.

Why this matters in practice:
- Polls work (sample mean approximates population mean)
- A/B testing works
- Confidence intervals and hypothesis tests rely on this
""")

print("\n" + "=" * 60)
print("VISUAL OBSERVATION FROM THE SAME X-SCALE PLOTS")
print("=" * 60)
print("""
Look at Figure 3 carefully:

n=2:   Wide and short  (imprecise estimate of the mean)
n=10:  Still wide, starting to get taller
n=30:  Noticeably narrower
n=50:  Much narrower, taller peak
n=100: Very narrow, tall peak (precise!)

As n increases, the distribution of the sample mean becomes:
- NARROWER  -- Standard Error sigma/sqrt(n) shrinks
- TALLER    -- more concentrated around the true mean
- Still Gaussian here (because the source was Gaussian)

This is why statisticians love large sample sizes!
""")
