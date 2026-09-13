"""
Typical generative models -- listed and shown as interactive toys.

Generative models learn P(x) or P(x|y) (and often P(y)), so they can
score unusual inputs and sample new ones -- not only draw a label boundary.

Sketch:
  1. Generative idea -- density over x, plus samples
  2. Family gallery -- typical generative models as mini pictures
  3. Class-conditional Gaussians (GDA / LDA vibe)
  4. Gaussian mixture -- drag mix weight
  5. Naive Bayes -- independent feature likelihoods
  6. Autoregressive -- next-step / next-token bars
  7. VAE toy -- drag latent z, decode a blob
  8. GAN toy -- generator samples vs discriminator heat
  9. Diffusion toy -- drag noise level (forward / reverse vibe)
  10. Same world, many generative skins
  11. Takeaway collage
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
from matplotlib.widgets import Slider
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

np.random.seed(142)
RNG = np.random.default_rng(142)


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
    kwargs.setdefault("figsize", screen_figsize())
    kwargs.setdefault("dpi", FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


def show_maximized(fig):
    fig.set_dpi(FIGURE_DPI)
    fig.set_size_inches(screen_figsize(), forward=True)
    fig.canvas.draw_idle()
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        try:
            manager.window.state("zoomed")
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


C_DARK = "#1a5276"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_FUN = "#8e44ad"
C_BIN = "#d68910"
C_GRAY = "#85929e"
CMAP2 = ListedColormap(["#d6eaf8", "#d5f5e3"])


def gaussian_pdf_2d(X, mean, cov):
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    diff = X - mean
    expo = -0.5 * np.einsum("...i,ij,...j->...", diff, inv, diff)
    return np.exp(expo) / (2 * np.pi * np.sqrt(det) + 1e-12)


def paint2(ax, X, y=None, s=42, c0=C_GRAY, c1=C_OK):
    if y is None:
        ax.scatter(X[:, 0], X[:, 1], s=s, c=C_DARK, alpha=0.75,
                   edgecolors="white", linewidths=0.4, zorder=3)
        return
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=s, c=c0,
               edgecolors="white", linewidths=0.4, zorder=3, label="y=0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=s, c=c1,
               edgecolors="white", linewidths=0.4, zorder=3, label="y=1")


def mesh(xlim, ylim, n=220):
    xx = np.linspace(*xlim, n)
    yy = np.linspace(*ylim, n)
    XX, YY = np.meshgrid(xx, yy)
    return XX, YY, np.column_stack([XX.ravel(), YY.ravel()])


def cov_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-9))
    ell = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ell)
    return ell


def make_two_gaussians(n=70):
    m0, m1 = np.array([-1.2, 0.15]), np.array([1.25, -0.1])
    c0 = np.array([[0.35, 0.08], [0.08, 0.28]])
    c1 = np.array([[0.30, -0.06], [-0.06, 0.40]])
    a = RNG.multivariate_normal(m0, c0, size=n)
    b = RNG.multivariate_normal(m1, c1, size=n)
    return np.vstack([a, b]), np.array([0] * n + [1] * n), (m0, c0), (m1, c1)


def make_gmm_data(n=90):
    weights = np.array([0.45, 0.35, 0.20])
    means = [np.array([-1.3, -0.4]), np.array([1.2, 0.2]), np.array([0.1, 1.3])]
    covs = [
        np.array([[0.28, 0.05], [0.05, 0.22]]),
        np.array([[0.25, -0.08], [-0.08, 0.32]]),
        np.array([[0.20, 0.02], [0.02, 0.20]]),
    ]
    comps = RNG.choice(3, size=n, p=weights)
    xs = np.zeros((n, 2))
    for k in range(3):
        idx = comps == k
        if np.any(idx):
            xs[idx] = RNG.multivariate_normal(means[k], covs[k], size=int(idx.sum()))
    return xs, weights, means, covs


# Shared demo state for gallery / collage steps
X1, y1, (m0, c0), (m1, c1) = make_two_gaussians(60)
XX1, YY1, flat1 = mesh((-3.2, 3.2), (-2.6, 2.6))
dens = (0.5 * gaussian_pdf_2d(flat1, m0, c0) +
        0.5 * gaussian_pdf_2d(flat1, m1, c1)).reshape(XX1.shape)

Xgmm, w_gmm, means_gmm, covs_gmm = make_gmm_data(100)
XX2, YY2, flat2 = mesh((-3, 3), (-2.5, 2.5), n=160)
dens_gmm = np.zeros(len(flat2))
for wk, mk, ck in zip(w_gmm, means_gmm, covs_gmm):
    dens_gmm += wk * gaussian_pdf_2d(flat2, mk, ck)

tok = np.array([0.38, 0.24, 0.16, 0.10, 0.07, 0.05])
z_cloud = RNG.normal(0, 1, size=(200, 2))


def vae_decode(z):
    z = np.asarray(z, dtype=float)
    return np.array([1.1 * z[0] + 0.2 * np.sin(2 * z[1]),
                     0.9 * z[1] + 0.25 * z[0] ** 2 - 0.2])


_th_real = RNG.uniform(0, 2 * np.pi, 120)
real = (np.column_stack([np.cos(_th_real), np.sin(_th_real)]) * 1.1
        + RNG.normal(0, 0.06, (120, 2)))

clean9 = RNG.multivariate_normal([0.0, 0.0], [[0.55, 0.1], [0.1, 0.40]], size=100)
clean9 = np.vstack([
    clean9,
    RNG.multivariate_normal([1.5, 1.0], [[0.18, 0], [0, 0.18]], size=40),
])


# -------------------------------------------------------
# FIGURE 1: Generative idea
# -------------------------------------------------------
def build_step_1():
    fig1, axes1 = new_figure(1, 2)
    fig1.suptitle(
        r"Step 1: Generative -- model $P(x)$ (or $P(x\mid y)$), then sample",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    axes1[0].contourf(XX1, YY1, dens, levels=14, cmap="viridis", alpha=0.95)
    paint2(axes1[0], X1, y1, s=28)
    cov_ellipse(axes1[0], m0, c0, n_std=2, fill=False, edgecolor="white", lw=2.0)
    cov_ellipse(axes1[0], m1, c1, n_std=2, fill=False, edgecolor="white", lw=2.0)
    axes1[0].set_title(r"density $P(x)$", fontsize=FS_AX_TITLE - 2)
    axes1[0].set_xticks([]); axes1[0].set_yticks([])
    axes1[0].set_aspect("equal")

    # samples from the model
    samp0 = RNG.multivariate_normal(m0, c0, size=40)
    samp1 = RNG.multivariate_normal(m1, c1, size=40)
    axes1[1].contourf(XX1, YY1, dens, levels=14, cmap="viridis", alpha=0.35)
    axes1[1].scatter(samp0[:, 0], samp0[:, 1], s=36, c=C_GRAY, edgecolors="white", label="sample y=0")
    axes1[1].scatter(samp1[:, 0], samp1[:, 1], s=36, c=C_OK, edgecolors="white", label="sample y=1")
    axes1[1].set_title("draw new x ~ model", fontsize=FS_AX_TITLE - 2)
    axes1[1].set_xticks([]); axes1[1].set_yticks([])
    axes1[1].set_aspect("equal")
    axes1[1].legend(fontsize=FS_LEGEND - 1, loc="upper left")
    for ax in axes1:
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.6, 2.6)
        ax.grid(True, alpha=0.2)
    fig1.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig1


# -------------------------------------------------------
# FIGURE 2: Family gallery
# -------------------------------------------------------
def build_step_2():
    fig2, axes2 = new_figure(2, 3)
    fig2.suptitle("Step 2: Typical generative models (visual roll call)", fontsize=FS_FIGURE_TITLE, fontweight="bold")

    # GDA / class-conditional
    axes2[0, 0].contourf(XX1, YY1, dens, levels=12, cmap="viridis")
    paint2(axes2[0, 0], X1, y1, s=14)
    axes2[0, 0].set_title("GDA / Gauss. class-cond.", fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_DARK)

    # GMM
    axes2[0, 1].contourf(XX2, YY2, dens_gmm.reshape(XX2.shape), levels=12, cmap="magma")
    paint2(axes2[0, 1], Xgmm, s=12)
    axes2[0, 1].set_title("GMM", fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_FUN)

    # Naive Bayes 1D likelihoods cartoon
    bins = np.linspace(-3, 3, 40)
    axes2[0, 2].hist(X1[y1 == 0, 0], bins=bins, density=True, alpha=0.55, color=C_GRAY, label="x1 | y=0")
    axes2[0, 2].hist(X1[y1 == 1, 0], bins=bins, density=True, alpha=0.55, color=C_OK, label="x1 | y=1")
    xs = np.linspace(-3, 3, 200)
    axes2[0, 2].plot(xs, np.exp(-(xs - m0[0]) ** 2 / (2 * c0[0, 0])) / np.sqrt(2 * np.pi * c0[0, 0]),
                     color=C_DARK, lw=2)
    axes2[0, 2].plot(xs, np.exp(-(xs - m1[0]) ** 2 / (2 * c1[0, 0])) / np.sqrt(2 * np.pi * c1[0, 0]),
                     color=C_ACCENT, lw=2)
    axes2[0, 2].set_title("Naive Bayes", fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_BIN)
    axes2[0, 2].set_yticks([])

    # Autoregressive next-token
    axes2[1, 0].bar(np.arange(len(tok)), tok, color=[C_FUN if i == 0 else C_GRAY for i in range(len(tok))])
    axes2[1, 0].set_xticks(np.arange(len(tok)))
    axes2[1, 0].set_xticklabels(["the", "a", "cat", "dog", "red", "…"], fontsize=FS_TICK - 1, rotation=30)
    axes2[1, 0].set_title("autoregressive", fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_FUN)
    axes2[1, 0].set_ylim(0, 0.5); axes2[1, 0].set_yticks([])

    # VAE latent ring -> decode
    z = RNG.normal(0, 1, size=(80, 2))
    # decode: stretch + slight swirl
    dec = np.column_stack([1.2 * z[:, 0] + 0.15 * z[:, 1], 0.9 * z[:, 1] - 0.2 * z[:, 0] ** 2])
    axes2[1, 1].scatter(z[:, 0], z[:, 1], s=10, c=C_GRAY, alpha=0.5, label="z")
    axes2[1, 1].scatter(dec[:, 0], dec[:, 1], s=14, c=C_OK, alpha=0.75, label="decode")
    axes2[1, 1].set_title("VAE", fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_OK)
    axes2[1, 1].legend(fontsize=FS_LEGEND - 2, loc="upper left")

    # Diffusion: clean -> noisy ladder
    clean = RNG.multivariate_normal([0, 0], [[0.35, 0], [0, 0.35]], size=60)
    for i, sig in enumerate([0.0, 0.45, 0.9]):
        noisy = clean + RNG.normal(0, sig, size=clean.shape)
        axes2[1, 2].scatter(noisy[:, 0] + i * 2.2, noisy[:, 1], s=10, alpha=0.7,
                            c=[C_OK, C_BIN, C_ACCENT][i])
    axes2[1, 2].set_title("diffusion", fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_ACCENT)
    axes2[1, 2].set_xticks([]); axes2[1, 2].set_yticks([])

    for ax in axes2.ravel():
        if ax is not axes2[0, 2] and ax is not axes2[1, 0]:
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
    fig2.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig2


# -------------------------------------------------------
# FIGURE 3: Interactive class-conditional Gaussians
# -------------------------------------------------------
def build_step_3():
    fig3, (ax3, ax3b) = new_figure(1, 2)
    fig3.suptitle("Step 3: Class-conditional Gaussians -- drag prior P(y=1)", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig3.subplots_adjust(bottom=0.18, wspace=0.25)

    X3, y3, (m0_3, c0_3), (m1_3, c1_3) = make_two_gaussians(65)
    XX3, YY3, flat3 = mesh((-3.2, 3.2), (-2.6, 2.6))
    prior0 = 0.5


    def draw_gda(ax_left, ax_right, prior1):
        ax_left.clear(); ax_right.clear()
        p0 = (1 - prior1) * gaussian_pdf_2d(flat3, m0_3, c0_3)
        p1 = prior1 * gaussian_pdf_2d(flat3, m1_3, c1_3)
        dens = (p0 + p1).reshape(XX3.shape)
        post1 = (p1 / (p0 + p1 + 1e-12)).reshape(XX3.shape)
        ax_left.contourf(XX3, YY3, dens, levels=14, cmap="viridis", alpha=0.95)
        paint2(ax_left, X3, y3, s=32)
        cov_ellipse(ax_left, m0_3, c0_3, fill=False, edgecolor="white", lw=1.8)
        cov_ellipse(ax_left, m1_3, c1_3, fill=False, edgecolor="white", lw=1.8)
        ax_left.set_title(r"$P(x)$ mixture", fontsize=FS_AX_TITLE - 2)
        ax_right.contourf(XX3, YY3, post1, levels=14, cmap="RdYlGn", alpha=0.95)
        ax_right.contour(XX3, YY3, post1, levels=[0.5], colors=[C_DARK], linewidths=2.4)
        paint2(ax_right, X3, y3, s=32)
        ax_right.set_title(r"$P(y=1\mid x)$ via Bayes", fontsize=FS_AX_TITLE - 2)
        for ax in (ax_left, ax_right):
            ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.6, 2.6)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=FS_LEGEND - 1, loc="upper left")


    draw_gda(ax3, ax3b, prior0)
    sl_prior = Slider(fig3.add_axes([0.25, 0.07, 0.50, 0.04]), r"$P(y=1)$", 0.05, 0.95,
                      valinit=prior0, valstep=0.05, color=C_FUN)


    def _u3(_=None):
        draw_gda(ax3, ax3b, float(sl_prior.val))
        fig3.canvas.draw_idle()


    sl_prior.on_changed(_u3)
    print("Step 3: GDA interactive.")
    return fig3


# -------------------------------------------------------
# FIGURE 4: Interactive GMM
# -------------------------------------------------------
def build_step_4():
    fig4, (ax4, ax4b) = new_figure(1, 2)
    fig4.suptitle("Step 4: Gaussian mixture -- drag component-0 weight", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig4.subplots_adjust(bottom=0.18, wspace=0.25)

    means4 = [np.array([-1.3, -0.35]), np.array([1.15, 0.15]), np.array([0.05, 1.25])]
    covs4 = [
        np.array([[0.30, 0.05], [0.05, 0.24]]),
        np.array([[0.26, -0.07], [-0.07, 0.34]]),
        np.array([[0.22, 0.02], [0.02, 0.22]]),
    ]
    XX4, YY4, flat4 = mesh((-3, 3), (-2.5, 2.6))
    w0_init = 0.45


    def gmm_density(flat, w0):
        # w0 for comp0; split remainder between 1 and 2 as 0.55/0.45 of leftover
        rem = 1.0 - w0
        weights = np.array([w0, 0.55 * rem, 0.45 * rem])
        dens = np.zeros(len(flat))
        for wk, mk, ck in zip(weights, means4, covs4):
            dens += wk * gaussian_pdf_2d(flat, mk, ck)
        return dens, weights


    def draw_gmm(ax_d, ax_s, w0):
        ax_d.clear(); ax_s.clear()
        dens, weights = gmm_density(flat4, w0)
        Z = dens.reshape(XX4.shape)
        ax_d.contourf(XX4, YY4, Z, levels=14, cmap="magma", alpha=0.95)
        for mk, ck, col in zip(means4, covs4, [C_GRAY, C_OK, C_ACCENT]):
            cov_ellipse(ax_d, mk, ck, fill=False, edgecolor=col, lw=2.0)
            ax_d.scatter([mk[0]], [mk[1]], s=80, c=col, zorder=4)
        ax_d.set_title(r"$P(x)=\sum_k \pi_k\,\mathcal{N}$", fontsize=FS_AX_TITLE - 2)
        # sample
        comps = RNG.choice(3, size=120, p=weights)
        samp = np.zeros((120, 2))
        for k in range(3):
            idx = comps == k
            if np.any(idx):
                samp[idx] = RNG.multivariate_normal(means4[k], covs4[k], size=int(idx.sum()))
        ax_s.contourf(XX4, YY4, Z, levels=14, cmap="magma", alpha=0.25)
        cols = np.array([C_GRAY, C_OK, C_ACCENT])[comps]
        ax_s.scatter(samp[:, 0], samp[:, 1], s=28, c=cols, edgecolors="white", linewidths=0.3)
        ax_s.set_title("samples", fontsize=FS_AX_TITLE - 2)
        for ax in (ax_d, ax_s):
            ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.6)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(True, alpha=0.2)
        # mix bars on density axis bottom via twin text
        ax_d.text(0.02, 0.05, f"π=({weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f})",
                  transform=ax_d.transAxes, fontsize=FS_TICK, color="white",
                  bbox=dict(boxstyle="round,pad=0.25", fc="0.2", alpha=0.55))


    draw_gmm(ax4, ax4b, w0_init)
    sl_w0 = Slider(fig4.add_axes([0.25, 0.07, 0.50, 0.04]), r"$\pi_0$", 0.05, 0.85,
                   valinit=w0_init, valstep=0.05, color=C_FUN)


    def _u4(_=None):
        draw_gmm(ax4, ax4b, float(sl_w0.val))
        fig4.canvas.draw_idle()


    sl_w0.on_changed(_u4)
    print("Step 4: GMM interactive.")
    return fig4


# -------------------------------------------------------
# FIGURE 5: Interactive Naive Bayes
# -------------------------------------------------------
def build_step_5():
    fig5, axes5 = new_figure(1, 3)
    fig5.suptitle("Step 5: Naive Bayes -- independent feature likelihoods", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig5.subplots_adjust(bottom=0.18, wspace=0.28)

    # 2 binary-ish features from continuous projections
    X5, y5, (m0_5, c0_5), (m1_5, c1_5) = make_two_gaussians(80)
    # feature likelihoods along x and y axes separately
    sig_scale0 = 1.0


    def nb_curves(sig_scale):
        xs = np.linspace(-3.2, 3.2, 260)
        # P(x1|y) using diagonal variance * scale
        s0x = max(c0_5[0, 0] * sig_scale, 1e-3)
        s1x = max(c1_5[0, 0] * sig_scale, 1e-3)
        s0y = max(c0_5[1, 1] * sig_scale, 1e-3)
        s1y = max(c1_5[1, 1] * sig_scale, 1e-3)
        p0x = np.exp(-(xs - m0_5[0]) ** 2 / (2 * s0x)) / np.sqrt(2 * np.pi * s0x)
        p1x = np.exp(-(xs - m1_5[0]) ** 2 / (2 * s1x)) / np.sqrt(2 * np.pi * s1x)
        p0y = np.exp(-(xs - m0_5[1]) ** 2 / (2 * s0y)) / np.sqrt(2 * np.pi * s0y)
        p1y = np.exp(-(xs - m1_5[1]) ** 2 / (2 * s1y)) / np.sqrt(2 * np.pi * s1y)
        return xs, p0x, p1x, p0y, p1y, s0x, s1x, s0y, s1y


    def draw_nb(sig_scale):
        for ax in axes5:
            ax.clear()
        xs, p0x, p1x, p0y, p1y, s0x, s1x, s0y, s1y = nb_curves(sig_scale)
        axes5[0].plot(xs, p0x, color=C_GRAY, lw=2.4, label="y=0")
        axes5[0].plot(xs, p1x, color=C_OK, lw=2.4, label="y=1")
        axes5[0].hist(X5[y5 == 0, 0], bins=24, density=True, alpha=0.25, color=C_GRAY)
        axes5[0].hist(X5[y5 == 1, 0], bins=24, density=True, alpha=0.25, color=C_OK)
        axes5[0].set_title(r"$P(x_1\mid y)$", fontsize=FS_AX_TITLE - 2)
        axes5[0].legend(fontsize=FS_LEGEND - 1)

        axes5[1].plot(xs, p0y, color=C_GRAY, lw=2.4, label="y=0")
        axes5[1].plot(xs, p1y, color=C_OK, lw=2.4, label="y=1")
        axes5[1].hist(X5[y5 == 0, 1], bins=24, density=True, alpha=0.25, color=C_GRAY)
        axes5[1].hist(X5[y5 == 1, 1], bins=24, density=True, alpha=0.25, color=C_OK)
        axes5[1].set_title(r"$P(x_2\mid y)$", fontsize=FS_AX_TITLE - 2)
        axes5[1].legend(fontsize=FS_LEGEND - 1)

        # posterior heat under independence assumption
        XX, YY, flat = mesh((-3.2, 3.2), (-2.6, 2.6), n=180)
        # P(x|y)=P(x1|y)P(x2|y)
        def feat_pdf(val, mu, var):
            return np.exp(-(val - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

        px_y0 = feat_pdf(flat[:, 0], m0_5[0], s0x) * feat_pdf(flat[:, 1], m0_5[1], s0y)
        px_y1 = feat_pdf(flat[:, 0], m1_5[0], s1x) * feat_pdf(flat[:, 1], m1_5[1], s1y)
        post = px_y1 / (px_y0 + px_y1 + 1e-12)
        axes5[2].contourf(XX, YY, post.reshape(XX.shape), levels=14, cmap="RdYlGn")
        axes5[2].contour(XX, YY, post.reshape(XX.shape), levels=[0.5], colors=[C_DARK], linewidths=2.2)
        paint2(axes5[2], X5, y5, s=22)
        axes5[2].set_title(r"$P(y=1\mid x)$ (NB)", fontsize=FS_AX_TITLE - 2)
        axes5[2].set_aspect("equal")
        axes5[2].set_xticks([]); axes5[2].set_yticks([])
        for ax in axes5[:2]:
            ax.set_xlim(-3.2, 3.2)
            ax.grid(True, alpha=0.25)
            ax.set_yticks([])
        axes5[2].grid(True, alpha=0.2)


    draw_nb(sig_scale0)
    sl_nb = Slider(fig5.add_axes([0.25, 0.07, 0.50, 0.04]), r"variance scale", 0.4, 2.5,
                   valinit=sig_scale0, valstep=0.05, color=C_BIN)


    def _u5(_=None):
        draw_nb(float(sl_nb.val))
        fig5.canvas.draw_idle()


    sl_nb.on_changed(_u5)
    print("Step 5: Naive Bayes interactive.")
    return fig5


# -------------------------------------------------------
# FIGURE 6: Interactive autoregressive
# -------------------------------------------------------
def build_step_6():
    fig6, (ax6a, ax6b) = new_figure(1, 2)
    fig6.suptitle("Step 6: Autoregressive -- drag temperature on next-token probs", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig6.subplots_adjust(bottom=0.18, wspace=0.28)

    logits6 = np.array([2.2, 1.4, 0.9, 0.4, 0.1, -0.3])
    labels6 = ["the", "a", "cat", "dog", "red", "…"]
    tau6 = 1.0


    def softmax_tau(logits, tau):
        z = logits / max(tau, 1e-3)
        z = z - z.max()
        p = np.exp(z)
        return p / p.sum()


    def draw_ar(tau):
        ax6a.clear(); ax6b.clear()
        p = softmax_tau(logits6, tau)
        cols = [C_FUN if i == int(np.argmax(p)) else C_GRAY for i in range(len(p))]
        ax6a.bar(np.arange(len(p)), p, color=cols, edgecolor="white")
        ax6a.set_xticks(np.arange(len(p)))
        ax6a.set_xticklabels(labels6, fontsize=FS_TICK)
        ax6a.set_ylim(0, 1.0)
        ax6a.set_ylabel("probability", fontsize=FS_AXIS)
        ax6a.set_title(r"$P(x_t\mid x_{<t})$", fontsize=FS_AX_TITLE - 2)
        ax6a.grid(True, alpha=0.25, axis="y")

        # sample sequences of length 8 greedily / randomly under tau
        # toy: reuse same logits each step with small noise for visual chain
        seq_probs = []
        rng_local = np.random.default_rng(0)
        for _ in range(8):
            noisy = logits6 + rng_local.normal(0, 0.15, size=len(logits6))
            seq_probs.append(softmax_tau(noisy, tau))
        M = np.vstack(seq_probs)
        ax6b.imshow(M.T, aspect="auto", cmap="magma", vmin=0, vmax=M.max())
        ax6b.set_yticks(np.arange(len(labels6)))
        ax6b.set_yticklabels(labels6, fontsize=FS_TICK)
        ax6b.set_xlabel("time step", fontsize=FS_AXIS)
        ax6b.set_title("rolling next-token heat", fontsize=FS_AX_TITLE - 2)
        # mark sampled tokens
        picks = M.argmax(axis=1)
        ax6b.scatter(np.arange(8), picks, s=60, c="white", marker="o", zorder=3)


    draw_ar(tau6)
    sl_tau6 = Slider(fig6.add_axes([0.25, 0.07, 0.50, 0.04]), r"temperature $\tau$", 0.2, 2.5,
                     valinit=tau6, valstep=0.05, color=C_FUN)


    def _u6(_=None):
        draw_ar(float(sl_tau6.val))
        fig6.canvas.draw_idle()


    sl_tau6.on_changed(_u6)
    print("Step 6: autoregressive interactive.")
    return fig6


# -------------------------------------------------------
# FIGURE 7: Interactive VAE toy
# -------------------------------------------------------
def build_step_7():
    fig7, (ax7a, ax7b) = new_figure(1, 2)
    fig7.suptitle("Step 7: VAE toy -- drag latent z, watch decode", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig7.subplots_adjust(bottom=0.20, wspace=0.28)

    z0x, z0y = 0.2, -0.3


    def draw_vae(zx, zy):
        ax7a.clear(); ax7b.clear()
        ax7a.scatter(z_cloud[:, 0], z_cloud[:, 1], s=12, c=C_GRAY, alpha=0.45)
        ax7a.scatter([zx], [zy], s=160, c=C_FUN, edgecolors="k", zorder=4)
        # unit circle prior vibe
        th = np.linspace(0, 2 * np.pi, 120)
        ax7a.plot(np.cos(th), np.sin(th), color=C_DARK, lw=1.5, alpha=0.7)
        ax7a.set_title(r"latent $z\sim\mathcal{N}(0,I)$", fontsize=FS_AX_TITLE - 2)
        ax7a.set_aspect("equal")
        ax7a.set_xlim(-3, 3); ax7a.set_ylim(-3, 3)
        ax7a.grid(True, alpha=0.25)

        mean = vae_decode([zx, zy])
        cloud = mean + RNG.normal(0, 0.12, size=(80, 2))
        # background of many decoded prior samples
        dec_bg = np.array([vae_decode(z) for z in z_cloud[::2]])
        ax7b.scatter(dec_bg[:, 0], dec_bg[:, 1], s=10, c=C_GRAY, alpha=0.35)
        ax7b.scatter(cloud[:, 0], cloud[:, 1], s=28, c=C_OK, edgecolors="white", linewidths=0.3)
        ax7b.scatter([mean[0]], [mean[1]], s=140, c=C_ACCENT, marker="*", zorder=4)
        ax7b.set_title(r"decode $x=g_\theta(z)$", fontsize=FS_AX_TITLE - 2)
        ax7b.set_aspect("equal")
        ax7b.set_xlim(-2.5, 2.5); ax7b.set_ylim(-2.0, 2.8)
        ax7b.grid(True, alpha=0.25)


    draw_vae(z0x, z0y)
    sl_zx = Slider(fig7.add_axes([0.12, 0.08, 0.35, 0.035]), r"$z_1$", -2.5, 2.5,
                   valinit=z0x, valstep=0.05, color=C_FUN)
    sl_zy = Slider(fig7.add_axes([0.55, 0.08, 0.30, 0.035]), r"$z_2$", -2.5, 2.5,
                   valinit=z0y, valstep=0.05, color=C_BIN)


    def _u7(_=None):
        draw_vae(float(sl_zx.val), float(sl_zy.val))
        fig7.canvas.draw_idle()


    sl_zx.on_changed(_u7); sl_zy.on_changed(_u7)
    print("Step 7: VAE interactive.")
    return fig7


# -------------------------------------------------------
# FIGURE 8: Interactive GAN toy
# -------------------------------------------------------
def build_step_8():
    fig8, (ax8a, ax8b) = new_figure(1, 2)
    fig8.suptitle("Step 8: GAN toy -- drag generator shift vs discriminator", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig8.subplots_adjust(bottom=0.18, wspace=0.28)

    shift0 = 0.0


    def disc_score(X, real_mean_r=1.1):
        r = np.sqrt((X ** 2).sum(axis=1))
        # high score near the real ring radius
        return np.exp(-((r - real_mean_r) ** 2) / (2 * 0.18 ** 2))


    def draw_gan(shift):
        ax8a.clear(); ax8b.clear()
        # generator: noise -> ring with angular bias / shift
        z = RNG.normal(0, 1, size=(120, 2))
        ang = np.arctan2(z[:, 1], z[:, 0]) + shift
        rad = 1.1 + 0.05 * z[:, 0]
        fake = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]) + RNG.normal(0, 0.05, (120, 2))

        XX, YY, flat = mesh((-2.2, 2.2), (-2.2, 2.2), n=180)
        D = disc_score(flat).reshape(XX.shape)
        ax8a.contourf(XX, YY, D, levels=14, cmap="YlOrRd", alpha=0.9)
        ax8a.scatter(real[:, 0], real[:, 1], s=18, c=C_OK, label="real", zorder=3)
        ax8a.scatter(fake[:, 0], fake[:, 1], s=18, c=C_ACCENT, label="fake", zorder=3)
        ax8a.set_title(r"discriminator $D(x)$ heat", fontsize=FS_AX_TITLE - 2)
        ax8a.legend(fontsize=FS_LEGEND - 1, loc="upper left")
        ax8a.set_aspect("equal")
        ax8a.set_xlim(-2.2, 2.2); ax8a.set_ylim(-2.2, 2.2)

        ax8b.scatter(real[:, 0], real[:, 1], s=22, c=C_OK, alpha=0.7, label="real")
        ax8b.scatter(fake[:, 0], fake[:, 1], s=22, c=C_ACCENT, alpha=0.7, label="G(z)")
        ax8b.set_title("generator samples", fontsize=FS_AX_TITLE - 2)
        ax8b.legend(fontsize=FS_LEGEND - 1, loc="upper left")
        ax8b.set_aspect("equal")
        ax8b.set_xlim(-2.2, 2.2); ax8b.set_ylim(-2.2, 2.2)
        for ax in (ax8a, ax8b):
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(True, alpha=0.2)


    draw_gan(shift0)
    sl_shift = Slider(fig8.add_axes([0.25, 0.07, 0.50, 0.04]), r"G angle shift", -np.pi, np.pi,
                      valinit=shift0, valstep=0.1, color=C_ACCENT)


    def _u8(_=None):
        draw_gan(float(sl_shift.val))
        fig8.canvas.draw_idle()


    sl_shift.on_changed(_u8)
    print("Step 8: GAN interactive.")
    return fig8


# -------------------------------------------------------
# FIGURE 9: Interactive diffusion toy
# -------------------------------------------------------
def build_step_9():
    fig9, (ax9a, ax9b) = new_figure(1, 2)
    fig9.suptitle("Step 9: Diffusion toy -- drag noise level t", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig9.subplots_adjust(bottom=0.18, wspace=0.28)

    t0 = 0.35


    def draw_diff(t):
        ax9a.clear(); ax9b.clear()
        # forward: x_t = sqrt(1-b)*x0 + sqrt(b)*eps  with b~t
        b = np.clip(t, 0.0, 1.0)
        eps = RNG.normal(0, 1, size=clean9.shape)
        xt = np.sqrt(1 - b) * clean9 + np.sqrt(b) * eps
        ax9a.scatter(clean9[:, 0], clean9[:, 1], s=18, c=C_OK, alpha=0.35, label=r"$x_0$")
        ax9a.scatter(xt[:, 0], xt[:, 1], s=22, c=C_ACCENT, alpha=0.8, label=r"$x_t$")
        ax9a.set_title("forward noise", fontsize=FS_AX_TITLE - 2)
        ax9a.legend(fontsize=FS_LEGEND - 1, loc="upper left")

        # reverse vibe: denoise estimate toward clean (simple posterior mean toy)
        xhat = np.sqrt(1 - b) * xt   # crude
        # better: shrink toward empirical clean centroid mixture -- use xt*(1-b)+mean*b
        xhat = (1 - b) * xt + b * 0  # pull to 0 as t->1, keep structure as t->0
        # show a one-step denoise toward clean9 nearest (visual only)
        # interpolate displayed points between xt and clean for intuition
        x_den = (1 - b) * clean9 + b * xt
        ax9b.scatter(xt[:, 0], xt[:, 1], s=14, c=C_GRAY, alpha=0.35, label=r"$x_t$")
        ax9b.scatter(x_den[:, 0], x_den[:, 1], s=22, c=C_FUN, alpha=0.85, label="denoised vibe")
        ax9b.set_title("reverse / denoise vibe", fontsize=FS_AX_TITLE - 2)
        ax9b.legend(fontsize=FS_LEGEND - 1, loc="upper left")
        for ax in (ax9a, ax9b):
            ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.8)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(True, alpha=0.2)


    draw_diff(t0)
    sl_t = Slider(fig9.add_axes([0.25, 0.07, 0.50, 0.04]), r"noise level $t$", 0.0, 1.0,
                  valinit=t0, valstep=0.05, color=C_FUN)


    def _u9(_=None):
        draw_diff(float(sl_t.val))
        fig9.canvas.draw_idle()


    sl_t.on_changed(_u9)
    print("Step 9: diffusion interactive.")
    return fig9


# -------------------------------------------------------
# FIGURE 10: Same data, many generative skins
# -------------------------------------------------------
def build_step_10():
    fig10, axes10 = new_figure(2, 3)
    fig10.suptitle("Step 10: Same points -- many generative skins", fontsize=FS_FIGURE_TITLE, fontweight="bold")

    X10, y10, (m0_10, c0_10), (m1_10, c1_10) = make_two_gaussians(55)
    XX10, YY10, flat10 = mesh((-3.2, 3.2), (-2.6, 2.6), n=160)

    dens_mix = (0.5 * gaussian_pdf_2d(flat10, m0_10, c0_10) +
                0.5 * gaussian_pdf_2d(flat10, m1_10, c1_10)).reshape(XX10.shape)
    post = (0.5 * gaussian_pdf_2d(flat10, m1_10, c1_10) /
            (0.5 * gaussian_pdf_2d(flat10, m0_10, c0_10) +
             0.5 * gaussian_pdf_2d(flat10, m1_10, c1_10) + 1e-12)).reshape(XX10.shape)

    axes10[0, 0].contourf(XX10, YY10, dens_mix, levels=12, cmap="viridis")
    paint2(axes10[0, 0], X10, y10, s=14)
    axes10[0, 0].set_title("GDA density", fontsize=FS_AX_TITLE - 3, fontweight="bold")

    axes10[0, 1].contourf(XX10, YY10, dens_gmm.reshape(XX2.shape), levels=12, cmap="magma")
    paint2(axes10[0, 1], Xgmm, s=10)
    axes10[0, 1].set_title("GMM", fontsize=FS_AX_TITLE - 3, fontweight="bold")

    axes10[0, 2].contourf(XX10, YY10, post, levels=12, cmap="RdYlGn")
    paint2(axes10[0, 2], X10, y10, s=14)
    axes10[0, 2].set_title("Bayes posterior", fontsize=FS_AX_TITLE - 3, fontweight="bold")

    # VAE decode cloud
    z = RNG.normal(0, 1, size=(100, 2))
    dec = np.array([vae_decode(zi) for zi in z])
    axes10[1, 0].scatter(dec[:, 0], dec[:, 1], s=12, c=C_OK, alpha=0.75)
    axes10[1, 0].set_title("VAE samples", fontsize=FS_AX_TITLE - 3, fontweight="bold")

    # GAN ring
    th = np.linspace(0, 2 * np.pi, 100)
    axes10[1, 1].scatter(1.1 * np.cos(th), 1.1 * np.sin(th), s=10, c=C_OK, alpha=0.4)
    fake = np.column_stack([1.1 * np.cos(th), 1.1 * np.sin(th)]) + RNG.normal(0, 0.08, (100, 2))
    axes10[1, 1].scatter(fake[:, 0], fake[:, 1], s=12, c=C_ACCENT, alpha=0.7)
    axes10[1, 1].set_title("GAN samples", fontsize=FS_AX_TITLE - 3, fontweight="bold")

    # diffusion ladder
    for i, sig in enumerate([0.0, 0.5, 1.0]):
        pts = X10 + RNG.normal(0, sig, size=X10.shape)
        axes10[1, 2].scatter(pts[:, 0] + i * 0.0, pts[:, 1], s=8, alpha=0.45,
                             c=[C_OK, C_BIN, C_ACCENT][i])
    axes10[1, 2].set_title("diffusion noise", fontsize=FS_AX_TITLE - 3, fontweight="bold")

    for ax in axes10.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
    fig10.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig10


# -------------------------------------------------------
# FIGURE 11: Takeaway collage
# -------------------------------------------------------
def build_step_11():
    fig11, axes11 = new_figure(1, 4)
    fig11.suptitle(
        r"Step 11: Generative family -- density, latents, adversaries, noise",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    axes11[0].contourf(XX1, YY1, dens, levels=10, cmap="viridis")
    paint2(axes11[0], X1, y1, s=10)
    axes11[0].set_title("density\nGDA / GMM / NB", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)

    axes11[1].bar(np.arange(len(tok)), tok, color=[C_FUN if i == 0 else C_GRAY for i in range(len(tok))])
    axes11[1].set_title("factorize\nautoregressive", fontsize=FS_ANNOT, fontweight="bold", color=C_FUN)
    axes11[1].set_xticks([]); axes11[1].set_yticks([])

    axes11[2].scatter(z_cloud[::3, 0], z_cloud[::3, 1], s=8, c=C_GRAY, alpha=0.4)
    axes11[2].scatter([0.2], [-0.3], s=90, c=C_FUN)
    mean = vae_decode([0.2, -0.3])
    axes11[2].annotate("", xy=mean, xytext=(0.2, -0.3),
                       arrowprops=dict(arrowstyle="->", color=C_OK, lw=2))
    axes11[2].scatter([mean[0]], [mean[1]], s=90, c=C_OK)
    axes11[2].set_title("latent\nVAE", fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    axes11[2].set_xticks([]); axes11[2].set_yticks([])

    axes11[3].scatter(real[:, 0], real[:, 1], s=8, c=C_OK, alpha=0.5)
    b = 0.55
    xt = np.sqrt(1 - b) * clean9[:80] + np.sqrt(b) * RNG.normal(0, 1, size=(80, 2))
    axes11[3].scatter(xt[:, 0], xt[:, 1], s=8, c=C_ACCENT, alpha=0.6)
    axes11[3].set_title("noise games\nGAN / diffusion", fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT)
    axes11[3].set_xticks([]); axes11[3].set_yticks([])

    for ax in axes11:
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
    axes11[1].set_aspect("auto")
    fig11.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig11


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
    build_step_11,
])

print("\n" + "=" * 60)
print("142 GENERATIVE MODELS -- VISUAL / INTERACTIVE")
print("=" * 60)
print("""
Typical generative models (shown):
  GDA / class-conditional Gaussians, GMM, Naive Bayes,
  autoregressive, VAE, GAN, diffusion

Idea: model P(x) or P(x|y) -- score and sample, not only label.

,
""")
