"""
1_26_product_of_randomity.py

Intuition for the PRODUCT of two random variables.

If X and Y are independent, then

    P = X * Y

behaves very differently from a sum:

    E[P]   = E[X] E[Y]
    Var(P) = Var(X)Var(Y) + Var(X)(E[Y])^2 + Var(Y)(E[X])^2

For i.i.d. factors with mean mu and variance sigma^2 this collapses to

    Var(XY) = sigma^2 (sigma^2 + 2 mu^2) = sigma^4 + 2 mu^2 sigma^2

Unlike sums of Gaussians (which stay Gaussian), a product of Gaussians
is peaked near 0 with HEAVY tails. For positive variables, taking logs
turns products into sums -- the bridge to log-normal thinking.
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
from scipy.stats import norm
from scipy.special import kn  # modified Bessel K_0 for product-normal PDF

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


def product_normal_pdf(z):
    """PDF of Z = X*Y for X,Y iid N(0,1): f(z) = K_0(|z|) / pi."""
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    mask = z != 0
    out[mask] = kn(0, np.abs(z[mask])) / np.pi
    # K_0(0) = +inf; density has a logarithmic singularity at 0
    out[~mask] = np.nan
    return out


def var_of_product(ex, ey, vx, vy):
    """Var(XY) for independent X,Y."""
    return vx * vy + vx * (ey ** 2) + vy * (ex ** 2)


def var_of_product_iid(mu, sigma2):
    """Var(XY) for i.i.d. independent X,Y with mean mu and variance sigma2."""
    return sigma2 * (sigma2 + 2.0 * (mu ** 2))


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
# Setup: two i.i.d. standard normals -- product vs one factor
# -------------------------------------------------------
N_PAIRS = 30000
X = torch.randn(N_PAIRS)
Y = torch.randn(N_PAIRS)
P = X * Y
S = X + Y  # contrast: sum stays Gaussian

# Theory for N(0,1) * N(0,1): E[P]=0, Var(P)=1
p_mean_theory = 0.0
p_var_theory = var_of_product(0.0, 0.0, 1.0, 1.0)  # = 1
assert abs(p_var_theory - var_of_product_iid(0.0, 1.0)) < 1e-12


# -------------------------------------------------------
# FIGURE 1: One Gaussian vs product of two Gaussians
# -------------------------------------------------------
def build_step_1():
    fig1, axes1 = new_figure(1, 2)
    fig1.suptitle(
        r'Step 1: $X\sim N(0,1)$ vs $P=X\cdot Y$  --  product is NOT Gaussian',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    # Left: single normal
    ax = axes1[0]
    xg = np.linspace(-4.5, 4.5, 500)
    ax.hist(X.numpy(), bins=60, density=True, alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=0.3, label=r'samples of $X$')
    ax.plot(xg, norm.pdf(xg, 0, 1), 'r-', linewidth=2.5, label=r'$N(0,1)$ PDF')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title(r'One factor $X\sim N(0,1)$', fontsize=FS_AX_TITLE)
    ax.set_xlabel('value', fontsize=FS_AXIS)
    ax.set_ylabel('Density', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        f'emp. mean = {X.mean().item():.3f}\n'
        f'emp. var  = {X.var(unbiased=True).item():.3f}',
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4.5, 4.5)

    # Right: product -- sharp peak, heavy tails
    ax = axes1[1]
    # Clip display range; tails go farther than a normal with same variance
    p_np = P.numpy()
    ax.hist(p_np, bins=80, density=True, alpha=0.7, color='darkorange',
            edgecolor='black', linewidth=0.3, range=(-6, 6),
            label=r'samples of $P=X\cdot Y$')
    zg = np.linspace(-6, 6, 800)
    zg_nz = zg[zg != 0]
    ax.plot(zg_nz, product_normal_pdf(zg_nz), 'r-', linewidth=2.5,
            label=r'product-normal PDF $K_0(|z|)/\pi$')
    # Same-variance Gaussian for contrast (looks nothing like the product)
    ax.plot(zg, norm.pdf(zg, 0, 1), 'b--', linewidth=2, alpha=0.6,
            label=r'$N(0,1)$ (same variance)')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title(r'Product $P=X\cdot Y$', fontsize=FS_AX_TITLE)
    ax.set_xlabel('value', fontsize=FS_AXIS)
    ax.set_ylabel('Density', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        f'emp. mean = {P.mean().item():.3f}\n'
        f'emp. var  = {P.var(unbiased=True).item():.3f}\n'
        f'theory var = {p_var_theory:.3f}\n'
        'tall spike + heavy tails',
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.35)

    plt.tight_layout()
    return fig1


# -------------------------------------------------------
# FIGURE 2: Variance of an iid product -- formula
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 2: Variance of an i.i.d. product -- the formula",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.15, 11.2, 1.5, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.9,
        r"Let $X,Y$ be i.i.d. with $E[X]=\mu$ and $\mathrm{Var}(X)=\sigma^2$, and independent.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.55, 11.2, 2.35, C_GOAL, C_OK, lw=2.2)
    ax.text(
        6.0, 5.35, "iid product variance",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    ax.text(
        6.0, 4.35,
        r"$\mathrm{Var}(XY)=\sigma^2(\sigma^2+2\mu^2)=\sigma^4+2\mu^2\sigma^2$",
        ha="center", va="center", fontsize=FS_ANNOT + 3, color="#145a32",
    )

    round_box(ax, 0.4, 0.35, 5.5, 2.95, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        3.15, 2.85, "From the independent formula",
        ha="center", fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK,
    )
    ax.text(
        3.15, 1.55,
        r"$\mathrm{Var}(XY)=\mathrm{Var}(X)\mathrm{Var}(Y)$"
        + "\n"
        r"$+\mathrm{Var}(X)(E[Y])^2$"
        + "\n"
        r"$+\mathrm{Var}(Y)(E[X])^2$"
        + "\n\n"
        r"Set $E[X]=E[Y]=\mu$,"
        + "\n"
        r"$\mathrm{Var}(X)=\mathrm{Var}(Y)=\sigma^2$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
        linespacing=1.25,
    )

    round_box(ax, 6.1, 0.35, 5.5, 2.95, C_MUTE, "0.45", lw=1.8)
    ax.text(
        8.85, 2.85, "Checks",
        ha="center", fontsize=FS_ANNOT - 1, fontweight="bold", color=C_DARK,
    )
    ax.text(
        8.85, 1.55,
        r"$N(0,1)$ iid: $\mu=0,\sigma^2=1$"
        + "\n"
        r"$\Rightarrow\ \mathrm{Var}(XY)=1$"
        + "\n\n"
        r"Also: $\mathrm{Var}(XY)=E[X^2]^2-(E[X])^4$"
        + "\n"
        r"since $E[X^2]=\sigma^2+\mu^2$.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
        linespacing=1.25,
    )
    return fig


# -------------------------------------------------------
# FIGURE 3: Geometry -- constant-product hyperbolas
# -------------------------------------------------------
def build_step_3():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        r'Step 3: In the $(X,Y)$ plane, constant product $=c$ is a hyperbola $xy=c$',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    idx = torch.randperm(N_PAIRS)[:4000]
    p_plot = P[idx].numpy()
    p_lim = float(np.percentile(np.abs(p_plot), 98))  # stretch colors for contrast
    ax2.scatter(
        X[idx].numpy(), Y[idx].numpy(),
        c=p_plot, cmap='seismic', s=12, alpha=0.75, linewidths=0,
        vmin=-p_lim, vmax=p_lim,
    )
    cbar = fig2.colorbar(ax2.collections[0], ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label(r'$P = X\cdot Y$', fontsize=FS_AXIS)
    cbar.ax.tick_params(labelsize=FS_TICK)

    lim = 3.5
    t = np.linspace(0.15, lim, 300)
    for c, ls in [(1.0, '-'), (2.0, '--'), (-1.0, '-'), (-2.0, '--')]:
        # xy = c  =>  y = c/x  (both branches)
        ax2.plot(t, c / t, color='green', linestyle=ls, linewidth=2,
                 label=rf'$xy={c:g}$')
        ax2.plot(-t, c / (-t), color='green', linestyle=ls, linewidth=2)

    ax2.axhline(0, color='k', linewidth=0.8, alpha=0.4)
    ax2.axvline(0, color='k', linewidth=0.8, alpha=0.4)
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_aspect('equal')
    ax2.set_xlabel(r'$X$', fontsize=FS_AXIS)
    ax2.set_ylabel(r'$Y$', fontsize=FS_AXIS)
    ax2.tick_params(axis='both', labelsize=FS_TICK)
    ax2.set_title(
        r'Large $|P|$ only when BOTH $|X|$ and $|Y|$ are large (rare)',
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
        'Near an axis: one factor near 0, so the product is near 0 (common).\n'
        'Far in the corners: both factors large, so |product| is large (rare, heavy tails).',
        transform=ax2.transAxes,
        fontsize=FS_ANNOT,
        va='center', ha='left',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
    )
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig2


# -------------------------------------------------------
# FIGURE 3: Product vs sum -- same inputs, different story
# -------------------------------------------------------
def build_step_4():
    fig3, axes3 = new_figure(1, 3)
    fig3.suptitle(
        r'Step 4: Same $X,Y\sim N(0,1)$ -- sum stays Gaussian, product does not',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    panels = [
        (X, 0.0, 1.0, r'$X\sim N(0,1)$', 'steelblue', True),
        (S, 0.0, np.sqrt(2), r'$S=X+Y\sim N(0,2)$', 'seagreen', True),
        (P, 0.0, 1.0, r'$P=X\cdot Y$ (not Gaussian)', 'darkorange', False),
    ]

    for ax, (data, mean_t, std_t, title, color, is_gaussian) in zip(axes3, panels):
        lo, hi = (-5, 5) if is_gaussian else (-6, 6)
        ax.hist(data.numpy(), bins=70, density=True, alpha=0.7, color=color,
                edgecolor='black', linewidth=0.3, range=(lo, hi))
        xs = np.linspace(lo, hi, 500)
        if is_gaussian:
            ax.plot(xs, norm.pdf(xs, mean_t, std_t), 'r-', linewidth=2.5,
                    label='Gaussian PDF')
        else:
            xs_nz = xs[xs != 0]
            ax.plot(xs_nz, product_normal_pdf(xs_nz), 'r-', linewidth=2.5,
                    label=r'product PDF $K_0/\pi$')
            ax.plot(xs, norm.pdf(xs, 0, 1), 'b--', linewidth=2, alpha=0.55,
                    label=r'$N(0,1)$ contrast')
        ax.axvline(mean_t, color='k', linestyle='--', alpha=0.5)
        ax.set_title(title, fontsize=FS_AX_TITLE)
        ax.set_xlabel('value', fontsize=FS_AXIS)
        ax.set_ylabel('Density', fontsize=FS_AXIS)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        add_vcenter_legend_and_note(
            ax,
            f'emp. var = {data.var(unbiased=True).item():.3f}\n'
            f'theory   = {std_t ** 2:.3f}',
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lo, hi)

    plt.tight_layout()
    return fig3


# -------------------------------------------------------
# FIGURE 4: Positive variables -- product <-> sum in log space
# -------------------------------------------------------
def build_step_5():
    fig4, axes4 = new_figure(1, 2)
    fig4.suptitle(
        r'Step 5: For positive RVs, $\log(XY)=\log X+\log Y$  (product becomes a sum)',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    # Lognormal: if log A, log B ~ N(0, LOG_SIGMA^2), then A,B > 0 and AB is lognormal
    LOG_SIGMA = 1.0  # sigma of log A, log B (matches sigma=1 theme)
    logA = LOG_SIGMA * torch.randn(N_PAIRS)
    logB = LOG_SIGMA * torch.randn(N_PAIRS)
    A = torch.exp(logA)
    B = torch.exp(logB)
    AB = A * B
    logAB = logA + logB  # exactly log(A*B)

    ax = axes4[0]
    # Show A (skewed) and AB (even more skewed / heavier right tail)
    amax = np.quantile(AB.numpy(), 0.99)
    ax.hist(A.numpy(), bins=60, density=True, alpha=0.55, color='steelblue',
            range=(0, amax), label=r'$A$ (lognormal)', edgecolor='black', linewidth=0.2)
    ax.hist(AB.numpy(), bins=60, density=True, alpha=0.55, color='darkorange',
            range=(0, amax), label=r'$A\cdot B$ (also lognormal)',
            edgecolor='black', linewidth=0.2)
    ax.set_title(r'Positive factors: product stretches the right tail', fontsize=FS_AX_TITLE)
    ax.set_xlabel('value', fontsize=FS_AXIS)
    ax.set_ylabel('Density', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        f'E[A]E[B]={A.mean().item() * B.mean().item():.3f}\n'
        f'E[AB]    ={AB.mean().item():.3f}\n'
        f'(independent => match)',
    )
    ax.grid(True, alpha=0.3)

    ax = axes4[1]
    # In log space the product is just a sum of Gaussians
    ax.hist(logA.numpy(), bins=50, density=True, alpha=0.55, color='steelblue',
            label=r'$\log A\sim N(0,\sigma^2)$', edgecolor='black', linewidth=0.2)
    ax.hist(logAB.numpy(), bins=50, density=True, alpha=0.55, color='darkorange',
            label=r'$\log(AB)=\log A+\log B$', edgecolor='black', linewidth=0.2)
    xs = np.linspace(-4.5, 4.5, 400)
    ax.plot(xs, norm.pdf(xs, 0, LOG_SIGMA), 'b-', linewidth=2,
            label=rf'$N(0,{LOG_SIGMA:g}^2)$')
    ax.plot(xs, norm.pdf(xs, 0, LOG_SIGMA * np.sqrt(2)), 'r-', linewidth=2.5,
            label=rf'$N(0,2\cdot{LOG_SIGMA:g}^2)$ for $\log(AB)$')
    ax.set_title(r'Log space: product rule becomes the sum rule', fontsize=FS_AX_TITLE)
    ax.set_xlabel('log value', fontsize=FS_AXIS)
    ax.set_ylabel('Density', fontsize=FS_AXIS)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        'Taking the log turns\n'
        'multiplication into addition.\n'
        'That is why log-normals\n'
        'are closed under products.',
        facecolor='lightyellow',
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig4


run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("PRODUCT OF TWO RANDOM VARIABLES")
print("=" * 60)
print(f"""
Setup: independent X, Y.  Define P = X * Y.

Means multiply:
    E[P] = E[X] E[Y]

Variances do NOT simply multiply. For independent factors:
    Var(P) = Var(X)Var(Y) + Var(X)(E[Y])^2 + Var(Y)(E[X])^2

iid specialization (E[X]=mu, Var(X)=sigma^2):
    Var(XY) = sigma^2 (sigma^2 + 2 mu^2) = sigma^4 + 2 mu^2 sigma^2
    equivalently Var(XY) = (E[X^2])^2 - (E[X])^4

Special case X,Y iid N(0,1):
    E[P] = 0,   Var(P) = 1
    but the SHAPE is not Gaussian -- tall spike at 0, heavy tails
    (product-normal PDF proportional to K_0(|z|)).

Geometry (Step 3):
    Level sets of xy = c are hyperbolas.
    |P| large only in the corners (both |X| and |Y| large) -- rare.
    Near an axis, one factor is ~0, so P piles up near 0 -- common.

Contrast with sums (Step 4 / ):
    X+Y of Gaussians stays Gaussian (variance adds).
    X*Y of Gaussians does NOT stay Gaussian.

Positive variables (Step 5):
    log(XY) = log X + log Y
    Products become sums in log space => log-normal family is closed
    under multiplication.

Demo numbers (N(0,1) factors):
    mean(P) = {P.mean().item():.4f}   (theory {p_mean_theory:.4f})
    var(P)  = {P.var(unbiased=True).item():.4f}   (theory {p_var_theory:.4f})
    var(S)  = {S.var(unbiased=True).item():.4f}   (theory 2.0000)
""")
