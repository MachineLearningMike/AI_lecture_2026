"""
Variational AutoEncoder (VAE) -- how it is trained, with the loss formulas.

Sketch:
  1. What a VAE is -- encoder, latent, decoder
  2. Deterministic autoencoder vs variational
  3. Generative model: prior p(z) and likelihood p_theta(x|z)
  4. Why the true posterior is hard -- introduce q_phi(z|x)
  5. ELBO: the training objective
  6. Loss = reconstruction + KL (practical form)
  7. Closed-form KL for diagonal Gaussians
  8. Reparameterization trick -- backprop through sampling
  9. One training step, end to end
  10. What each term pushes (recon vs prior)
  11. After training: encode, reconstruct, sample
  12. Visual intuition -- encode clouds in latent space
  13. Visual intuition -- prior samples decode to data
  14. Takeaway

generative-model bridge before Ch16 variants.
"""

from __future__ import annotations

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
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle
import numpy as np

np.random.seed(61)
RNG = np.random.default_rng(61)


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


def blank(ax, xlim=(0, 12), ylim=(0, 8)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(rect)
    return rect


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=2.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=18),
    )


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_OLD = "#fce4d6"
C_FUN = "#6c3483"
C_WARN = "#f9e79f"


# -------------------------------------------------------
# FIGURE 1: What a VAE is
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 1: A VAE maps data through a latent distribution",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax1)

    round_box(ax1, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 7.1,
        r"Encoder $q_\phi(z\mid x)$  $\rightarrow$  sample $z$  $\rightarrow$  "
        r"Decoder $p_\theta(x\mid z)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    boxes = [
        (0.5, 3.6, 2.8, 2.2, C_NEW, C_DARK, r"$x$" "\n" "data", "input"),
        (3.7, 3.6, 2.8, 2.2, "#f5eef8", C_FUN, r"$q_\phi(z\mid x)$" "\n" r"$(\mu,\sigma)$", "encoder"),
        (6.9, 3.6, 2.0, 2.2, C_WARN, "#b7950b", r"$z$", "latent"),
        (9.3, 3.6, 2.3, 2.2, C_GOAL, C_OK, r"$p_\theta(x\mid z)$" "\n" r"$\hat x$", "decoder"),
    ]
    for x, y, w, h, face, edge, title, _sub in boxes:
        round_box(ax1, x, y, w, h, face, edge, lw=1.8)
        ax1.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                 fontsize=FS_ANNOT, fontweight="bold")

    for x1_, x2_ in ((3.3, 3.7), (6.5, 6.9), (8.9, 9.3)):
        arrow(ax1, x1_, 4.7, x2_, 4.7, C_DARK, 2.0)

    ax1.text(
        6.0, 2.2,
        "VAE = AutoEncoder whose bottleneck is a probability distribution,\n"
        "not a single code vector.  Training teaches both reconstruction and a usable prior.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )
    ax1.text(
        6.0, 0.7,
        r"Kingma & Welling (2013): Auto-Encoding Variational Bayes.",
        ha="center", fontsize=FS_TICK, style="italic", color="0.35",
    )
    return fig1


# -------------------------------------------------------
# FIGURE 2: AE vs VAE
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        "Step 2: Deterministic autoencoder vs variational autoencoder",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax2)

    round_box(ax2, 0.4, 4.2, 5.5, 3.4, C_MUTE, "0.45", lw=1.6)
    ax2.text(3.15, 7.2, "Autoencoder", ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
    ax2.text(
        3.15, 5.5,
        r"$x \rightarrow z = f_\phi(x) \rightarrow \hat x = g_\theta(z)$"
        "\n\n"
        "One point code $z$.\n"
        "Great compressor; weak generator\n"
        "(latent holes; hard to sample).",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax2, 6.1, 4.2, 5.5, 3.4, "#f5eef8", C_FUN, lw=1.8)
    ax2.text(8.85, 7.2, "VAE", ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_FUN)
    ax2.text(
        8.85, 5.5,
        r"$x \rightarrow q_\phi(z\mid x) \rightarrow z\sim q \rightarrow \hat x$"
        "\n\n"
        "A cloud of codes for each $x$.\n"
        "Prior $p(z)$ organizes the latent space\n"
        "so new $z$ still decode sensibly.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax2, 0.4, 0.5, 11.2, 3.3, C_GOAL, C_OK, lw=1.8)
    ax2.text(
        6.0, 2.15,
        "Training difference in one sentence:\n"
        "AE minimizes reconstruction only.\n"
        "VAE minimizes reconstruction plus a KL that pulls each posterior toward the prior.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#145a32",
    )
    return fig2


# -------------------------------------------------------
# FIGURE 3: Generative model story
# -------------------------------------------------------
def build_step_3():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        r"Step 3: Generative story -- $p(z)$ then $p_\theta(x\mid z)$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax3)

    round_box(ax3, 0.4, 6.45, 11.2, 1.25, C_DARK, C_DARK, lw=0)
    ax3.text(
        6.0, 7.05,
        "A VAE is a latent-variable generative model.  Generation = sample latent, then decode.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    # prior panel
    round_box(ax3, 0.45, 2.6, 5.4, 3.5, C_MUTE, "0.45", lw=1.5)
    ax3.text(3.15, 5.75, r"Prior $p(z)=\mathcal{N}(0,I)$", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    th = np.linspace(0, 2 * np.pi, 120)
    ax3.plot(3.15 + 1.1 * np.cos(th), 4.15 + 1.1 * np.sin(th), color=C_DARK, lw=1.6, alpha=0.7)
    zs = RNG.normal(0, 0.55, size=(60, 2))
    ax3.scatter(3.15 + zs[:, 0], 4.15 + zs[:, 1], s=18, c=C_FUN, alpha=0.7, edgecolors="none")
    ax3.text(3.15, 2.9, "easy to sample", ha="center", fontsize=FS_TICK, style="italic")

    # likelihood panel
    round_box(ax3, 6.15, 2.6, 5.4, 3.5, C_GOAL, C_OK, lw=1.6)
    ax3.text(8.85, 5.75, r"Likelihood $p_\theta(x\mid z)$", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax3.text(
        8.85, 4.2,
        "Decoder network:\n"
        r"$z \mapsto$ parameters of $x$"
        "\n\n"
        r"Gaussian $x$: predict $\hat\mu(z)$ (often MSE)"
        "\n"
        r"Bernoulli $x$: predict probabilities (BCE)",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax3.text(
        6.0, 1.2,
        r"Joint: $p_\theta(x,z)=p(z)\,p_\theta(x\mid z)$."
        "   Marginal:  "
        r"$p_\theta(x)=\int p_\theta(x\mid z)\,p(z)\,dz$  (usually intractable).",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    return fig3


# -------------------------------------------------------
# FIGURE 4: Intractable posterior
# -------------------------------------------------------
def build_step_4():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        r"Step 4: True posterior $p_\theta(z\mid x)$ is hard -- learn $q_\phi(z\mid x)$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax4)

    round_box(ax4, 0.4, 5.5, 11.2, 2.1, C_MUTE, "0.45", lw=1.5)
    ax4.text(
        6.0, 6.55,
        r"Bayes:  $p_\theta(z\mid x)=\dfrac{p_\theta(x\mid z)\,p(z)}{p_\theta(x)}$"
        "\n\n"
        r"Denominator $p_\theta(x)=\int p_\theta(x\mid z)\,p(z)\,dz$  has no closed form for deep decoders.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    round_box(ax4, 0.45, 2.55, 5.4, 2.6, C_OLD, "#d35400", lw=1.6)
    ax4.text(3.15, 4.7, "Problem", ha="center", fontsize=FS_TICK, fontweight="bold", color="#6e2c00")
    ax4.text(
        3.15, 3.5,
        "Cannot evaluate / sample\n"
        r"exact $p_\theta(z\mid x)$ easily."
        "\n"
        "Cannot maximize $\\log p_\\theta(x)$ directly.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax4, 6.15, 2.55, 5.4, 2.6, C_GOAL, C_OK, lw=1.8)
    ax4.text(8.85, 4.7, "VAE move", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax4.text(
        8.85, 3.5,
        "Introduce an encoder family\n"
        r"$q_\phi(z\mid x)$  (usually diagonal Gaussian)"
        "\n"
        "and optimize a tractable lower bound.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax4.text(
        6.0, 1.1,
        r"Encoder outputs $(\mu_\phi(x),\sigma_\phi(x))$;  "
        r"$q_\phi(z\mid x)=\mathcal{N}\!\left(\mu_\phi(x),\mathrm{diag}(\sigma_\phi^2(x))\right)$.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f5eef8", edgecolor=C_FUN, alpha=0.95),
    )
    return fig4


# -------------------------------------------------------
# FIGURE 5: ELBO
# -------------------------------------------------------
def build_step_5():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        "Step 5: ELBO -- the Evidence Lower BOund we maximize",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax5)

    round_box(ax5, 0.4, 6.4, 11.2, 1.3, C_DARK, C_DARK, lw=0)
    ax5.text(
        6.0, 7.05,
        r"For any $q_\phi$:  $\log p_\theta(x) = \mathrm{ELBO}(\theta,\phi;x)"
        r" + D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right)$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax5, 0.5, 3.55, 11.0, 2.55, C_MUTE, "0.45", lw=1.6)
    ax5.text(
        6.0, 4.8,
        r"$\mathrm{ELBO}(\theta,\phi;x)"
        r"= \mathbb{E}_{q_\phi(z\mid x)}\!\left[\log p_\theta(x\mid z)\right]"
        r" - D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )

    round_box(ax5, 0.45, 0.45, 5.4, 2.8, C_GOAL, C_OK, lw=1.8)
    ax5.text(3.15, 2.8, "Useful facts", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax5.text(
        3.15, 1.5,
        r"$D_{\mathrm{KL}}(q\|p_{\mathrm{post}})\geq 0$"
        "\n"
        r"$\Rightarrow$ ELBO $\leq \log p_\theta(x)$"
        "\n\n"
        "Tighten ELBO $\\approx$ better fit\n"
        "+ better approximate posterior.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax5, 6.15, 0.45, 5.4, 2.8, "#f5eef8", C_FUN, lw=1.8)
    ax5.text(8.85, 2.8, "Training stance", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_FUN)
    ax5.text(
        8.85, 1.5,
        r"Maximize ELBO over $\theta,\phi$"
        "\n"
        r"(or minimize $-\mathrm{ELBO}$)."
        "\n\n"
        "No need to touch the true posterior\n"
        "inside the gradient step.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig5


# -------------------------------------------------------
# FIGURE 6: Practical loss
# -------------------------------------------------------
def build_step_6():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        "Step 6: Practical loss = reconstruction + KL",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax6)

    round_box(ax6, 0.4, 6.45, 11.2, 1.25, C_DARK, C_DARK, lw=0)
    ax6.text(
        6.0, 7.05,
        r"$L(\theta,\phi;x)=-\mathrm{ELBO}"
        r"= L_{\mathrm{recon}}+L_{\mathrm{KL}}$"
        "   (minimize $L$)",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax6, 0.4, 3.35, 5.5, 2.85, C_NEW, C_DARK, lw=1.8)
    ax6.text(3.15, 5.85, r"Reconstruction $L_{\mathrm{recon}}$", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax6.text(
        3.15, 4.4,
        r"$L_{\mathrm{recon}}"
        r"= -\mathbb{E}_{q_\phi(z\mid x)}\!\left[\log p_\theta(x\mid z)\right]$"
        "\n\n"
        r"Gaussian decoder $\approx$ MSE:  $\|x-\hat x\|^2$"
        "\n"
        r"Bernoulli decoder $\approx$ BCE$(x,\hat x)$",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax6, 6.1, 3.35, 5.5, 2.85, "#f5eef8", C_FUN, lw=1.8)
    ax6.text(8.85, 5.85, r"Regularizer $L_{\mathrm{KL}}$", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_FUN)
    ax6.text(
        8.85, 4.4,
        r"$L_{\mathrm{KL}}"
        r"= D_{\mathrm{KL}}\!\left(q_\phi(z\mid x)\,\|\,p(z)\right)$"
        "\n\n"
        r"Pulls $q_\phi(z\mid x)$ toward $\mathcal{N}(0,I)$."
        "\n"
        "Keeps latents sampleable from the prior.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax6.text(
        6.0, 1.7,
        r"Monte Carlo estimate (one sample is common):"
        "\n"
        r"$z\sim q_\phi(z\mid x)$,   "
        r"$L \approx -\log p_\theta(x\mid z)"
        r" + D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))$",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    ax6.text(
        6.0, 0.45,
        r"Optional $\beta$-VAE:  $L=L_{\mathrm{recon}}+\beta L_{\mathrm{KL}}$  "
        r"($\beta>1$ stronger disentangling pressure).",
        ha="center", fontsize=FS_TICK, style="italic", color="0.35",
    )
    return fig6


# -------------------------------------------------------
# FIGURE 7: Closed-form KL
# -------------------------------------------------------
def build_step_7():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        r"Step 7: Closed-form KL when $q$ and $p$ are Gaussians",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax7)

    round_box(ax7, 0.4, 6.4, 11.2, 1.3, C_DARK, C_DARK, lw=0)
    ax7.text(
        6.0, 7.05,
        r"$q_\phi(z\mid x)=\mathcal{N}(\mu,\mathrm{diag}(\sigma^2))$,   "
        r"$p(z)=\mathcal{N}(0,I)$"
        "  $\Rightarrow$ analytic KL, no sampling needed for this term.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax7, 0.5, 3.2, 11.0, 2.9, C_MUTE, "0.45", lw=1.6)
    ax7.text(
        6.0, 4.65,
        r"$D_{\mathrm{KL}}(q\|p)"
        r"= \dfrac{1}{2}\sum_{j=1}^{d}"
        r"\left("
        r"\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2"
        r"\right)$",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold",
    )

    round_box(ax7, 0.45, 0.45, 5.4, 2.45, C_GOAL, C_OK, lw=1.6)
    ax7.text(
        3.15, 1.65,
        r"If $\mu=0$ and $\sigma=1$:"
        "\n"
        "KL $=0$  (posterior $=$ prior)."
        "\n\n"
        "Large $|\mu|$ or $\\sigma\\to 0$:\n"
        "KL grows — encoder pays rent.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#145a32",
    )

    round_box(ax7, 6.15, 0.45, 5.4, 2.45, C_OLD, "#d35400", lw=1.6)
    ax7.text(
        8.85, 1.65,
        "Implementation tip:\n"
        r"predict $\log\sigma^2$ for stability,"
        "\n"
        r"then $\sigma^2=\exp(\log\sigma^2)$."
        "\n\n"
        "KL uses $\\mu$ and $\\log\\sigma^2$ directly.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig7


# -------------------------------------------------------
# FIGURE 8: Reparameterization
# -------------------------------------------------------
def build_step_8():
    fig8, ax8 = new_figure()
    fig8.suptitle(
        "Step 8: Reparameterization -- gradients through the sample",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax8)

    round_box(ax8, 0.4, 6.45, 11.2, 1.25, C_DARK, C_DARK, lw=0)
    ax8.text(
        6.0, 7.05,
        r"Naive $z\sim q_\phi$ blocks gradients into $\phi$.  "
        r"Reparameterize: randomness outside the network.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    round_box(ax8, 0.45, 3.55, 11.1, 2.6, C_MUTE, "0.45", lw=1.5)
    ax8.text(
        6.0, 4.85,
        r"$z = \mu_\phi(x) + \sigma_\phi(x)\circ\varepsilon$"
        r",\qquad \varepsilon\sim\mathcal{N}(0,I)$"
        "\n\n"
        r"$\mu_\phi,\sigma_\phi$ are deterministic network outputs;  "
        r"$\varepsilon$ is noise that does not need a gradient.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )

    # flow boxes
    flow = [
        (0.5, 0.55, 2.4, 2.5, C_NEW, r"$x$", "data"),
        (3.2, 0.55, 2.6, 2.5, "#f5eef8", r"$\mu,\sigma$", "encoder"),
        (6.1, 0.55, 2.5, 2.5, C_WARN, r"$\varepsilon$", "noise"),
        (9.0, 0.55, 2.5, 2.5, C_GOAL, r"$z$", r"$\mu+\sigma\circ\varepsilon$"),
    ]
    for x, y, w, h, face, top, bot in flow:
        round_box(ax8, x, y, w, h, face, "0.4", lw=1.5)
        ax8.text(x + w / 2, y + 1.55, top, ha="center", va="center",
                 fontsize=FS_ANNOT + 1, fontweight="bold")
        ax8.text(x + w / 2, y + 0.7, bot, ha="center", va="center", fontsize=FS_TICK)
    for x1_, x2_ in ((2.9, 3.2), (5.8, 6.1), (8.6, 9.0)):
        arrow(ax8, x1_, 1.8, x2_, 1.8, C_DARK, 2.0)

    return fig8


# -------------------------------------------------------
# FIGURE 9: One training step
# -------------------------------------------------------
def build_step_9():
    fig9, ax9 = new_figure()
    fig9.suptitle(
        "Step 9: One training step (minibatch)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax9)

    steps = [
        (7.0, "1. Take minibatch $x$", C_NEW),
        (6.05, r"2. Encoder $\rightarrow (\mu,\log\sigma^2)$", "#f5eef8"),
        (5.1, r"3. Sample $\varepsilon$; form $z=\mu+\sigma\circ\varepsilon$", C_WARN),
        (4.15, r"4. Decoder $\rightarrow \hat x$ (or likelihood params)", C_GOAL),
        (3.2, r"5. $L_{\mathrm{recon}}=-\log p_\theta(x\mid z)$  (MSE / BCE)", C_OLD),
        (2.25, r"6. $L_{\mathrm{KL}}=\frac{1}{2}\sum_j(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2)$", "#f5eef8"),
        (1.3, r"7. $L=L_{\mathrm{recon}}+L_{\mathrm{KL}}$;  Adam step on $\theta,\phi$", C_MUTE),
    ]
    for y, text, face in steps:
        round_box(ax9, 0.5, y - 0.4, 11.0, 0.85, face, "0.45", lw=1.3)
        ax9.text(6.0, y, text, ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

    ax9.text(
        6.0, 0.35,
        "Same pattern every step: encode distribution → reparameterize → decode → recon + KL → backprop.",
        ha="center", fontsize=FS_TICK, style="italic", color="0.35",
    )
    return fig9


# -------------------------------------------------------
# FIGURE 10: What each term pushes
# -------------------------------------------------------
def build_step_10():
    fig10, (ax10a, ax10b) = new_figure(1, 2)
    fig10.suptitle(
        "Step 10: What reconstruction and KL each push",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # left: recon wants informative z
    ax10a.set_xlim(-3, 3)
    ax10a.set_ylim(-3, 3)
    ax10a.set_aspect("equal")
    ax10a.set_title(r"$L_{\mathrm{recon}}$ wants informative $z$", fontsize=FS_AX_TITLE - 2)
    ax10a.grid(True, alpha=0.25)
    # separated class blobs in latent (what recon prefers)
    c1 = RNG.normal([-1.4, 0.2], 0.25, size=(40, 2))
    c2 = RNG.normal([1.3, -0.3], 0.25, size=(40, 2))
    ax10a.scatter(c1[:, 0], c1[:, 1], s=28, c=C_DARK, alpha=0.75, label="digits A")
    ax10a.scatter(c2[:, 0], c2[:, 1], s=28, c=C_ACCENT, alpha=0.75, label="digits B")
    ax10a.legend(fontsize=FS_LEGEND, loc="upper right")
    ax10a.text(
        0.5, -0.08,
        "Spread codes apart so decode\ncan tell inputs apart.",
        transform=ax10a.transAxes, ha="center", va="top", fontsize=FS_TICK,
    )

    # right: KL wants unit Gaussian
    ax10b.set_xlim(-3, 3)
    ax10b.set_ylim(-3, 3)
    ax10b.set_aspect("equal")
    ax10b.set_title(r"$L_{\mathrm{KL}}$ wants $q\approx\mathcal{N}(0,I)$", fontsize=FS_AX_TITLE - 2)
    ax10b.grid(True, alpha=0.25)
    prior = RNG.normal(0, 1, size=(120, 2))
    ax10b.scatter(prior[:, 0], prior[:, 1], s=16, c=C_FUN, alpha=0.45)
    circ = Circle((0, 0), 1.0, fill=False, edgecolor=C_DARK, lw=2.0)
    ax10b.add_patch(circ)
    # collapsed tiny posteriors near origin as "too much KL" cartoon
    for m in [(-0.2, 0.1), (0.15, -0.15), (0.05, 0.25)]:
        ax10b.add_patch(Ellipse(m, 0.35, 0.25, fill=True, facecolor=C_WARN,
                                edgecolor="#b7950b", alpha=0.85, lw=1.2))
    ax10b.text(
        0.5, -0.08,
        "Too much KL → posterior collapse:\n"
        r"$q(z\mid x)\approx p(z)$, weak $z$.",
        transform=ax10b.transAxes, ha="center", va="top", fontsize=FS_TICK,
    )

    fig10.tight_layout(rect=[0, 0.06, 1, 0.92])
    return fig10


# -------------------------------------------------------
# FIGURE 11: After training
# -------------------------------------------------------
def build_step_11():
    fig11, ax11 = new_figure()
    fig11.suptitle(
        "Step 11: After training -- three uses of the same model",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax11)

    cards = [
        (0.4, 3.3, 3.6, 4.2, C_NEW, "Reconstruct",
         r"$x \rightarrow z \rightarrow \hat x$"
         "\n"
         r"via $q_\phi$, then $p_\theta$"
         "\n\n"
         "Denoise / compress.\n"
         "Check recon loss."),
        (4.2, 3.3, 3.6, 4.2, "#f5eef8", "Encode",
         r"$x \rightarrow (\mu,\sigma)$ or $z$"
         "\n\n"
         "Latent features for\n"
         "clustering / transfer."),
        (8.0, 3.3, 3.6, 4.2, C_GOAL, "Generate",
         r"$z\sim p(z)=\mathcal{N}(0,I)$"
         "\n"
         r"$\hat x \sim p_\theta(x\mid z)$"
         "\n\n"
         "No encoder needed.\n"
         "This is the generative use."),
    ]
    for x, y, w, h, face, title, body in cards:
        round_box(ax11, x, y, w, h, face, "0.4", lw=1.8)
        ax11.text(x + w / 2, y + h - 0.55, title, ha="center",
                  fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_DARK)
        ax11.text(x + w / 2, y + h / 2 - 0.35, body, ha="center", va="center",
                  fontsize=FS_TICK + 1, fontweight="bold")

    ax11.text(
        6.0, 1.6,
        "Generation quality hinges on the KL having actually organized the latent space.\n"
        "If KL was ignored, sampling from $p(z)$ lands in holes the decoder never saw.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    ax11.text(
        6.0, 0.45,
        "Interactive latent drag toy: 0760_generative_toys.py  |  KL intuition: 0460_Loss_Kullback_Leibler_Divergence.py",
        ha="center", fontsize=FS_TICK, style="italic", color="0.35",
    )
    return fig11


# -------------------------------------------------------
# FIGURE 12: Visual intuition -- encode
# -------------------------------------------------------
def build_step_12():
    fig12, (ax12a, ax12b) = new_figure(1, 2)
    fig12.suptitle(
        r"Step 12: Visual intuition -- each $x$ becomes a latent cloud $q_\phi(z\mid x)$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # toy 2D data: two classes
    xa = RNG.normal([-1.4, 0.1], 0.28, size=(50, 2))
    xb = RNG.normal([1.3, -0.15], 0.28, size=(50, 2))
    ax12a.scatter(xa[:, 0], xa[:, 1], s=28, c=C_DARK, alpha=0.8, label="class A")
    ax12a.scatter(xb[:, 0], xb[:, 1], s=28, c=C_ACCENT, alpha=0.8, label="class B")
    ax12a.set_title(r"data space $x$", fontsize=FS_AX_TITLE - 2)
    ax12a.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    ax12a.set_aspect("equal")
    ax12a.set_xlim(-2.8, 2.8)
    ax12a.set_ylim(-2.2, 2.2)
    ax12a.grid(True, alpha=0.25)
    ax12a.set_xticks([])
    ax12a.set_yticks([])

    # latent: prior circle + posterior ellipses
    th = np.linspace(0, 2 * np.pi, 120)
    ax12b.plot(np.cos(th), np.sin(th), color="0.4", lw=1.5, alpha=0.7, label=r"$p(z)$ unit")
    # posterior means
    mus = [(-0.9, 0.35), (-0.55, -0.4), (0.85, 0.25), (0.6, -0.55)]
    colors = [C_DARK, C_DARK, C_ACCENT, C_ACCENT]
    for (mx, my), col in zip(mus, colors):
        zs = RNG.normal([mx, my], [0.22, 0.18], size=(35, 2))
        ax12b.scatter(zs[:, 0], zs[:, 1], s=16, c=col, alpha=0.55)
        ax12b.add_patch(Ellipse((mx, my), 0.9, 0.7, fill=False, edgecolor=col, lw=2.0))
    ax12b.scatter([m[0] for m in mus], [m[1] for m in mus], s=60, c=colors,
                  edgecolors="k", zorder=4, label=r"$\mu(x)$")
    ax12b.set_title(r"latent $q_\phi(z\mid x)$ clouds", fontsize=FS_AX_TITLE - 2)
    ax12b.legend(fontsize=FS_LEGEND - 2, loc="upper right")
    ax12b.set_aspect("equal")
    ax12b.set_xlim(-2.5, 2.5)
    ax12b.set_ylim(-2.2, 2.2)
    ax12b.grid(True, alpha=0.25)
    ax12b.set_xticks([])
    ax12b.set_yticks([])

    fig12.text(
        0.5, 0.02,
        r"Recon wants clouds separable; KL pulls them into the prior ball so holes do not open.",
        ha="center", fontsize=FS_TICK, fontweight="bold",
    )
    fig12.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig12


# -------------------------------------------------------
# FIGURE 13: Visual intuition -- generate
# -------------------------------------------------------
def build_step_13():
    fig13, (ax13a, ax13b) = new_figure(1, 2)
    fig13.suptitle(
        r"Step 13: Visual intuition -- sample $z\sim p(z)$, decode to new $x$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    th = np.linspace(0, 2 * np.pi, 120)
    xa = RNG.normal([-1.4, 0.1], 0.28, size=(50, 2))
    xb = RNG.normal([1.3, -0.15], 0.28, size=(50, 2))
    z_prior = RNG.normal(0, 1, size=(120, 2))
    ax13a.scatter(z_prior[:, 0], z_prior[:, 1], s=16, c=C_FUN, alpha=0.55)
    ax13a.plot(np.cos(th), np.sin(th), color=C_DARK, lw=1.8, alpha=0.8)
    ax13a.set_title(r"draw $z\sim\mathcal{N}(0,I)$", fontsize=FS_AX_TITLE - 2)
    ax13a.set_aspect("equal")
    ax13a.set_xlim(-3, 3)
    ax13a.set_ylim(-3, 3)
    ax13a.grid(True, alpha=0.25)
    ax13a.set_xticks([])
    ax13a.set_yticks([])


    def _toy_decode(z):
        z = np.asarray(z, dtype=float)
        # map latent half-planes to two blobs (learned decoder cartoon)
        gate = 1 / (1 + np.exp(-3.0 * z[0]))
        m_a = np.array([-1.4, 0.1])
        m_b = np.array([1.3, -0.15])
        return (1 - gate) * m_a + gate * m_b + 0.15 * z[::-1]


    dec = np.array([_toy_decode(z) for z in z_prior])
    ax13b.scatter(xa[:, 0], xa[:, 1], s=18, c=C_DARK, alpha=0.25)
    ax13b.scatter(xb[:, 0], xb[:, 1], s=18, c=C_ACCENT, alpha=0.25)
    ax13b.scatter(dec[:, 0], dec[:, 1], s=20, c=C_OK, alpha=0.75, label="decoded prior")
    ax13b.set_title(r"decode $p_\theta(x\mid z)$", fontsize=FS_AX_TITLE - 2)
    ax13b.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    ax13b.set_aspect("equal")
    ax13b.set_xlim(-2.8, 2.8)
    ax13b.set_ylim(-2.2, 2.2)
    ax13b.grid(True, alpha=0.25)
    ax13b.set_xticks([])
    ax13b.set_yticks([])

    fig13.text(
        0.5, 0.02,
        "If KL did its job, random prior draws land where the decoder was trained — new samples appear.",
        ha="center", fontsize=FS_TICK, fontweight="bold",
    )
    fig13.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig13


# -------------------------------------------------------
# FIGURE 14: Takeaway
# -------------------------------------------------------
def build_step_14():
    fig14, ax14 = new_figure()
    fig14.suptitle(
        "Step 14: Takeaway -- how a VAE is trained",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax14)

    round_box(ax14, 0.4, 5.7, 11.2, 1.9, C_DARK, C_DARK, lw=0)
    ax14.text(
        6.0, 6.65,
        r"Maximize ELBO / minimize  "
        r"$L = -\mathbb{E}_{q_\phi}\!\left[\log p_\theta(x\mid z)\right]"
        r" + D_{\mathrm{KL}}(q_\phi(z\mid x)\|p(z))$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    bullets = [
        r"1. Encoder defines $q_\phi(z\mid x)=\mathcal{N}(\mu,\mathrm{diag}\sigma^2)$.",
        r"2. Reparameterize $z=\mu+\sigma\circ\varepsilon$ so gradients reach $\phi$.",
        r"3. Decoder scores reconstruction via $\log p_\theta(x\mid z)$ (MSE or BCE).",
        r"4. Add analytic KL to $\mathcal{N}(0,I)$; optionally weight it by $\beta$.",
        r"5. At test time, sample $z\sim\mathcal{N}(0,I)$ and decode to generate.",
    ]
    y = 4.7
    for line in bullets:
        round_box(ax14, 0.5, y - 0.35, 11.0, 0.75, C_MUTE, "0.5", lw=1.1)
        ax14.text(6.0, y, line, ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")
        y -= 0.9

    ax14.text(
        6.0, 0.45,
        "Binding lens: reconstruction binds $z$ to $x$; KL binds the latent cloud to a shared prior.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
    )
    return fig14


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
    build_step_12,
    build_step_13,
    build_step_14,
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("0961 VARIATIONAL AUTOENCODER -- TRAINING")
print("=" * 60)
print("""
Model:
  prior      p(z) = N(0, I)
  encoder    q_phi(z|x) = N(mu_phi(x), diag(sigma_phi^2(x)))
  decoder    p_theta(x|z)

Objective (ELBO):
  ELBO = E_{q_phi(z|x)}[ log p_theta(x|z) ] - KL( q_phi(z|x) || p(z) )
  log p_theta(x) = ELBO + KL( q_phi(z|x) || p_theta(z|x) )  >= ELBO

Training loss (minimize):
  L = L_recon + L_KL
  L_recon = - E_q [ log p_theta(x|z) ]   ~ MSE or BCE
  L_KL    = KL(q_phi(z|x) || N(0,I))
          = (1/2) sum_j ( mu_j^2 + sigma_j^2 - 1 - log sigma_j^2 )

Reparameterization:
  z = mu + sigma * eps,   eps ~ N(0,I)

Generate:
  z ~ N(0,I);  x_hat ~ p_theta(x|z)

, ,
          0740_discriminative_generative.py
""")
