"""
0640_exponential.py

Symmetry between exp and log, and the calculus identity

    d/dx  log(f(x))  =  f'(x) / f(x)

Why it matters in ML / probability:
    - log turns products into sums (likelihood -> log-likelihood)
    - the score / gradient of a log-density is (density)' / density
    - softmax / cross-entropy / Gaussian MLE all lean on this

Companion to 0660_product_of_random.py (log turns products into sums)
and 0470_maximum_likelihood.py (we maximize log L, not L).
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
    run_slide_deck,
)
apply_rcparams(plt)
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Shared style (match 04 / 05 / 06)


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
# FIGURE 1: exp and log are mirrors (inverses)
# -------------------------------------------------------
def build_step_1():
    fig1, axes1 = new_figure(1, 2)
    fig1.suptitle(
        r'Step 1: $\exp$ and $\log$ are inverse mirrors -- each undoes the other',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    x_pos = np.linspace(0.05, 4.0, 400)
    x_all = np.linspace(-2.0, 2.0, 400)

    ax = axes1[0]
    ax.plot(x_all, np.exp(x_all), color='seagreen', linewidth=2.5, label=r'$y=\exp(x)$')
    ax.plot(x_pos, np.log(x_pos), color='darkorange', linewidth=2.5, label=r'$y=\log(x)$')
    ax.plot([-2, 4], [-2, 4], 'k--', linewidth=1.5, alpha=0.6, label=r'$y=x$ (mirror line)')
    # Mark a few inverse pairs
    pairs = [(-1.0, np.exp(-1.0)), (0.0, 1.0), (1.0, np.exp(1.0))]
    for a, b in pairs:
        ax.scatter([a, b], [b, a], s=50, zorder=5,
                   color='steelblue', edgecolor='k', linewidth=0.5)
        ax.plot([a, b], [b, a], color='steelblue', linewidth=1.0, alpha=0.5)
    ax.set_xlim(-2.2, 4.2)
    ax.set_ylim(-2.2, 4.2)
    ax.set_aspect('equal')
    ax.set_title(r'Reflection across $y=x$', fontsize=FS_AX_TITLE)
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('y', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        r'If $y=\exp(x)$' + '\n'
        r'then $x=\log(y)$' + '\n\n'
        'Blue dots: inverse pairs\n'
        r'$(\,a,\,e^{a}\,)$ and $(\,e^{a},\,a\,)$',
    )
    ax.grid(True, alpha=0.3)

    ax = axes1[1]
    t = np.linspace(-1.5, 1.5, 300)
    ax.plot(t, np.exp(np.log(np.exp(t))), 'seagreen', linewidth=2.5,
            label=r'$\exp(\log(\exp(t)))=\exp(t)$')
    # compose the other way on positive domain
    u = np.linspace(0.2, 5.0, 300)
    ax.plot(u, np.log(np.exp(np.log(u))), 'darkorange', linewidth=2.5,
            label=r'$\log(\exp(\log(u)))=\log(u)$')
    ax.plot(t, np.exp(t), 'seagreen', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.plot(u, np.log(u), 'darkorange', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_title('Composition recovers the original', fontsize=FS_AX_TITLE)
    ax.set_xlabel('input', fontsize=FS_AXIS)
    ax.set_ylabel('output', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        r'$\exp(\log(x))=x$  $(x>0)$' + '\n'
        r'$\log(\exp(x))=x$  (all $x$)' + '\n\n'
        'They are exact inverses\n'
        '(on the valid domain)',
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig1



# -------------------------------------------------------
# FIGURE 2: Products <-> sums under log / exp
# -------------------------------------------------------
def build_step_2():
    fig2, axes2 = new_figure(1, 2)
    fig2.suptitle(
        r'Step 2: $\log$ turns products into sums; $\exp$ turns sums into products',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    a, b = 2.0, 3.0
    ax = axes2[0]
    # Visual bars for a, b, a*b vs log a, log b, log a + log b
    categories = [r'$a$', r'$b$', r'$a\cdot b$']
    values = [a, b, a * b]
    colors = ['steelblue', 'steelblue', 'darkorange']
    ax.bar(categories, values, color=colors, edgecolor='k', linewidth=0.5)
    ax.set_title(rf'Product space: ${a:g}\times{b:g}={a*b:g}$', fontsize=FS_AX_TITLE)
    ax.set_ylabel('value', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.text(
        0.02, 0.50,
        'Multiplying grows\n'
        'fast and is awkward\n'
        'for gradients and MLE',
        transform=ax.transAxes, fontsize=FS_ANNOT, va='center', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes2[1]
    categories = [r'$\log a$', r'$\log b$', r'$\log a+\log b$']
    values = [np.log(a), np.log(b), np.log(a) + np.log(b)]
    colors = ['seagreen', 'seagreen', 'darkorange']
    ax.bar(categories, values, color=colors, edgecolor='k', linewidth=0.5)
    ax.axhline(np.log(a * b), color='k', linestyle='--', linewidth=1.5,
               label=rf'$\log(ab)=\log({a*b:g})={np.log(a*b):.3f}$')
    ax.set_title(r'Log space: $\log(ab)=\log a+\log b$', fontsize=FS_AX_TITLE)
    ax.set_ylabel('log value', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        'Addition is easy.\n'
        'That is why we use\n'
        'log-likelihood and\n'
        'cross-entropy.',
    )
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig2



# -------------------------------------------------------
# FIGURE 3: Core identity  (log f)' = f' / f
# -------------------------------------------------------
def build_step_3():
    fig3, axes3 = new_figure(1, 2)
    fig3.suptitle(
        r"Step 3: $\frac{d}{dx}\log f(x)=\frac{f'(x)}{f(x)}$  (chain rule)",
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    # Example f(x) = exp(-0.5 x^2) * (positive bump) -- unnormalized Gaussian-like
    # Use f(x) = 1 + 0.5*sin(x) on a positive range, or f(x) = exp(-x^2/2)
    x = np.linspace(-3.0, 3.0, 500)
    f = np.exp(-0.5 * x ** 2)                 # > 0 everywhere
    f_prime = -x * np.exp(-0.5 * x ** 2)      # analytic derivative
    log_f = np.log(f)
    # numerical derivative of log f
    dlog_num = np.gradient(log_f, x)
    # analytic: f'/f
    dlog_analytic = f_prime / f

    ax = axes3[0]
    ax.plot(x, f, color='steelblue', linewidth=2.5, label=r'$f(x)=e^{-x^2/2}$')
    ax.plot(x, f_prime, color='darkorange', linewidth=2.5, label=r"$f'(x)=-x\,e^{-x^2/2}$")
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.4)
    ax.set_title(r"A positive function $f$ and its slope $f'$", fontsize=FS_AX_TITLE)
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('value', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        'f must stay > 0\n'
        'for log(f) to exist.\n\n'
        'Densities, likelihoods,\n'
        'softplus outputs: OK',
    )
    ax.grid(True, alpha=0.3)

    ax = axes3[1]
    ax.plot(x, dlog_analytic, color='seagreen', linewidth=2.5,
            label=r"$f'(x)/f(x)=-x$")
    ax.plot(x, dlog_num, color='k', linestyle='--', linewidth=2.0, alpha=0.7,
            label=r'numerical $\frac{d}{dx}\log f(x)$')
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.4)
    ax.set_title(r'Both sides of the identity match', fontsize=FS_AX_TITLE)
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel(r"$(log f)'$", fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        r"$\frac{d}{dx}\log f = \frac{f'}{f}$" + '\n\n'
        'Relative rate of change:\n'
        'how fast f grows as a\n'
        'fraction of its height',
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig3



# -------------------------------------------------------
# FIGURE 4: Intuition -- relative slope, and MLE link
# -------------------------------------------------------
def build_step_4():
    fig4, axes4 = new_figure(1, 2)
    fig4.suptitle(
        r"Step 4: Intuition -- $\frac{f'}{f}$ is relative change; why we prefer $\log L$",
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    # Left: geometric picture of relative slope at a point
    ax = axes4[0]
    x0 = 1.0
    f0 = np.exp(-0.5 * x0 ** 2)
    fp0 = -x0 * np.exp(-0.5 * x0 ** 2)
    ax.plot(x, f, color='steelblue', linewidth=2.5, label=r'$f(x)$')
    ax.scatter([x0], [f0], s=80, color='red', zorder=5, label=rf'point $x={x0:g}$')
    # tangent
    tang_x = np.array([x0 - 0.8, x0 + 0.8])
    tang_y = f0 + fp0 * (tang_x - x0)
    ax.plot(tang_x, tang_y, color='darkorange', linewidth=2.5, label=r"tangent (slope $f'$)")
    # vertical height marker
    ax.vlines(x0, 0, f0, colors='gray', linestyles=':', linewidth=1.5)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(r"Absolute slope $f'$ vs height $f$", fontsize=FS_AX_TITLE)
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('f(x)', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        rf"At x={x0:g}:" + '\n'
        rf"f' = {fp0:.3f}" + '\n'
        rf"f  = {f0:.3f}" + '\n'
        rf"f'/f = {fp0/f0:.3f}" + '\n\n'
        'Same as (log f)\'',
    )
    ax.grid(True, alpha=0.3)

    # Right: likelihood vs log-likelihood landscape (toy 1D)
    # L(mu) = prod N(x_i | mu, 1) for a few points -- show L is tiny / peaked,
    # log L is nicer to optimize
    ax = axes4[1]
    samples = np.array([-0.5, 0.2, 1.0, 0.7])
    mu_grid = np.linspace(-2, 3, 400)
    # log-likelihood for N(mu,1)
    logL = np.sum(-0.5 * np.log(2 * np.pi) - 0.5 * (samples[None, :] - mu_grid[:, None]) ** 2, axis=1)
    L = np.exp(logL - logL.max())  # relative likelihood
    logL_shift = logL - logL.max()

    ax.plot(mu_grid, L, color='darkorange', linewidth=2.5, label=r'relative $L(\mu)$')
    ax.plot(mu_grid, (logL_shift - logL_shift.min()) / (logL_shift.max() - logL_shift.min() + 1e-12),
            color='seagreen', linewidth=2.5, label=r'scaled $\log L(\mu)$')
    ax.axvline(mu_grid[np.argmax(logL)], color='red', linestyle='--', linewidth=2,
               label='same maximizer')
    ax.set_title(r'Max $L$ and max $\log L$ agree -- $\log L$ is more stable', fontsize=FS_AX_TITLE)
    ax.set_xlabel(r'candidate $\mu$', fontsize=FS_AXIS)
    ax.set_ylabel('scaled objective', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        r'$\nabla \log L = \frac{\nabla L}{L}$' + '\n\n'
        'The gradient of log L is\n'
        'the relative gradient of L.\n'
        'Same critical points,\n'
        'better numerics.',
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig4




run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
# Quick numerical check of the identity at a few points
check_x = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
check_f = np.exp(-0.5 * check_x ** 2)
check_fp = -check_x * check_f
check_ratio = check_fp / check_f
check_dlog = np.gradient(np.log(np.exp(-0.5 * np.linspace(-3, 3, 500))), np.linspace(-3, 3, 500))
# interpolate numerical dlog at check_x
x_fine = np.linspace(-3, 3, 500)
dlog_fine = np.gradient(np.log(np.exp(-0.5 * x_fine ** 2)), x_fine)
dlog_at = np.interp(check_x, x_fine, dlog_fine)

print("\n" + "=" * 60)
print("EXP / LOG SYMMETRY AND  (log f)' = f' / f")
print("=" * 60)
print("""
1. exp and log are inverses (mirrors across y = x):
       exp(log(x)) = x   for x > 0
       log(exp(x)) = x   for all x

2. Algebraic symmetry that ML exploits constantly:
       log(a*b) = log a + log b
       exp(u+v) = exp(u) * exp(v)
   Products of probabilities become sums of log-probs.

3. Calculus identity (chain rule):
       d/dx log(f(x)) = f'(x) / f(x)     whenever f(x) > 0

   Reading: the derivative of log f is the RELATIVE rate of
   change of f -- slope divided by height.

4. Why maximize log L instead of L in MLE (see 0470_maximum_likelihood.py):
       d/d theta log L = (1/L) * dL/d theta
   Critical points coincide, but log L is numerically stable
   and turns the product-likelihood into a sum.
""")

print("Identity check for f(x)=exp(-x^2/2):")
hdr_x, hdr_ratio, hdr_num = 'x', "f'/f", "(log f)'_num"
print(f"  {hdr_x:>6}  {hdr_ratio:>10}  {hdr_num:>14}")
for xi, ri, di in zip(check_x, check_ratio, dlog_at):
    print(f"  {xi:6.1f}  {ri:10.4f}  {di:14.4f}")
