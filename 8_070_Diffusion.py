"""
Denoising diffusion models -- how they are trained, with the loss formulas.

Sketch:
  1. What diffusion is -- destroy then learn to reverse
  2. Forward process formulas
  3. Reverse process / denoising network
  4. Why predict noise epsilon
  5. Training loss (Ho et al. simple objective)
  6. Visual intuition -- forward noise ladder
  7. Visual intuition -- reverse denoising
  8. Sampling after training
  9. Score-matching glance + variants
  10. Takeaway
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
from matplotlib.patches import FancyBboxPatch
import numpy as np

np.random.seed(63)
RNG = np.random.default_rng(63)


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


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_OLD = "#fce4d6"
C_FUN = "#6c3483"
C_WARN = "#f9e79f"
C_GRAY = "#b0b0b0"


# Toy clean data: two blobs
CLEAN = np.vstack([
    RNG.multivariate_normal([-1.1, 0.2], [[0.18, 0.04], [0.04, 0.22]], 50),
    RNG.multivariate_normal([1.2, -0.1], [[0.20, -0.05], [-0.05, 0.16]], 50),
])


def q_sample(x0, t, eps=None):
    """x_t = sqrt(abar)*x0 + sqrt(1-abar)*eps with abar=(1-t)^2 toy schedule."""
    t = np.clip(t, 0.0, 1.0)
    abar = (1.0 - t) ** 2
    if eps is None:
        eps = RNG.normal(0, 1, size=x0.shape)
    return np.sqrt(abar) * x0 + np.sqrt(1.0 - abar) * eps, eps, abar


# -------------------------------------------------------
# FIGURE 1: What diffusion is
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 1: Diffusion -- add noise, then learn to remove it",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax1)

    round_box(ax1, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 7.1,
        r"Forward: data $\rightarrow$ noise.   Reverse: noise $\rightarrow$ data (learned).",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    flow = [
        (0.5, C_GOAL, r"$x_0$" "\n" "data"),
        (3.0, C_OLD, r"$x_t$" "\n" "noisy"),
        (5.5, C_MUTE, r"$x_T$" "\n" r"$\approx$ noise"),
        (8.0, "#f5eef8", r"$\epsilon_\theta$" "\n" "denoise net"),
        (10.0, C_NEW, r"$\hat x_0$" "\n" "sample"),
    ]
    for x, face, lab in flow:
        round_box(ax1, x, 3.7, 1.9, 2.2, face, "0.4", lw=1.6)
        ax1.text(x + 0.95, 4.8, lab, ha="center", va="center",
                 fontsize=FS_TICK + 1, fontweight="bold")

    ax1.annotate("", xy=(2.9, 4.8), xytext=(2.4, 4.8),
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2.2))
    ax1.annotate("", xy=(5.4, 4.8), xytext=(4.9, 4.8),
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2.2))
    ax1.annotate("", xy=(7.9, 4.8), xytext=(7.4, 4.8),
                 arrowprops=dict(arrowstyle="->", color=C_FUN, lw=2.2))
    ax1.annotate("", xy=(9.9, 4.8), xytext=(9.4, 4.8),
                 arrowprops=dict(arrowstyle="->", color=C_OK, lw=2.2))

    ax1.text(3.9, 3.2, "forward (fixed)", ha="center", fontsize=FS_TICK,
             color=C_ACCENT, fontweight="bold")
    ax1.text(9.0, 3.2, "reverse (learned)", ha="center", fontsize=FS_TICK,
             color=C_FUN, fontweight="bold")

    ax1.text(
        6.0, 1.5,
        "No adversary.  No encoder posterior.\n"
        "A single network learns the reverse denoising dynamics.\n"
        "Ho et al. (2020) DDPM; Song et al. score-based SDEs.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )
    return fig1


# -------------------------------------------------------
# FIGURE 2: Forward process
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        "Step 2: Forward process -- a fixed Markov noise chain",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax2)

    round_box(ax2, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax2.text(
        6.0, 7.0,
        r"$q(x_t\mid x_{t-1})"
        r"= \mathcal{N}\!\left("
        r"\sqrt{1-\beta_t}\,x_{t-1},\;"
        r"\beta_t I"
        r"\right)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax2, 0.5, 3.35, 11.0, 2.7, C_MUTE, "0.45", lw=1.6)
    ax2.text(
        6.0, 4.7,
        r"Closed form (nice for training):"
        "\n"
        r"$q(x_t\mid x_0)"
        r"= \mathcal{N}\!\left("
        r"\sqrt{\bar\alpha_t}\,x_0,\;"
        r"(1-\bar\alpha_t)I"
        r"\right)$"
        "\n\n"
        r"$\alpha_t=1-\beta_t$,   "
        r"$\bar\alpha_t=\prod_{s=1}^t \alpha_s$"
        "\n"
        r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$"
        r",\quad \epsilon\sim\mathcal{N}(0,I)$",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax2.text(
        6.0, 1.5,
        r"As $t$ grows, $x_t$ forgets $x_0$ and approaches isotropic Gaussian noise."
        "\n"
        "Forward process is not learned — schedule $\\beta_t$ is chosen.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    return fig2


# -------------------------------------------------------
# FIGURE 3: Reverse process
# -------------------------------------------------------
def build_step_3():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        "Step 3: Reverse process -- the learned generative model",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax3)

    round_box(ax3, 0.4, 6.4, 11.2, 1.3, C_DARK, C_DARK, lw=0)
    ax3.text(
        6.0, 7.05,
        r"$p_\theta(x_{t-1}\mid x_t)"
        r"= \mathcal{N}\!\left("
        r"\mu_\theta(x_t,t),\;"
        r"\Sigma_\theta(x_t,t)"
        r"\right)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax3, 0.45, 3.3, 5.5, 2.8, C_GOAL, C_OK, lw=1.8)
    ax3.text(3.2, 5.7, "Generate", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax3.text(
        3.2, 4.3,
        r"Start $x_T\sim\mathcal{N}(0,I)$."
        "\n"
        r"Step $t=T,\ldots,1$:"
        "\n"
        r"sample $x_{t-1}\sim p_\theta(\cdot\mid x_t)$."
        "\n\n"
        r"End at $x_0$ (image / point).",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax3, 6.15, 3.3, 5.4, 2.8, "#f5eef8", C_FUN, lw=1.8)
    ax3.text(8.85, 5.7, "What the net learns", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_FUN)
    ax3.text(
        8.85, 4.3,
        r"Often parameterize $\mu_\theta$"
        "\n"
        "via a noise prediction"
        "\n"
        r"$\epsilon_\theta(x_t,t)$."
        "\n\n"
        "Same U-Net / DiT backbone\n"
        "seen in modern image models.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax3.text(
        6.0, 1.5,
        r"True reverse $q(x_{t-1}\mid x_t,x_0)$ is Gaussian and tractable given $x_0$."
        "\n"
        r"Training matches $p_\theta$ to that reverse (or an equivalent noise target).",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )
    return fig3


# -------------------------------------------------------
# FIGURE 4: Why epsilon
# -------------------------------------------------------
def build_step_4():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        r"Step 4: Training target -- predict the noise $\epsilon$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax4)

    round_box(ax4, 0.4, 5.5, 11.2, 2.15, C_MUTE, "0.45", lw=1.6)
    ax4.text(
        6.0, 6.55,
        r"From $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$,"
        "\n"
        r"knowing $\epsilon$ recovers $x_0$ (and builds $\mu_\theta$)."
        "\n\n"
        r"So train a network $\epsilon_\theta(x_t,t)$ to estimate $\epsilon$.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    round_box(ax4, 0.45, 2.55, 5.4, 2.6, C_NEW, C_DARK, lw=1.6)
    ax4.text(3.15, 4.7, "Inputs", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax4.text(
        3.15, 3.5,
        r"noisy $x_t$"
        "\n"
        r"timestep $t$ (embedding)"
        "\n"
        "optional condition $c$"
        "\n"
        "(class / text / image)",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax4, 6.15, 2.55, 5.4, 2.6, C_OLD, C_ACCENT, lw=1.6)
    ax4.text(8.85, 4.7, "Output", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
    ax4.text(
        8.85, 3.5,
        r"$\hat\epsilon=\epsilon_\theta(x_t,t)$"
        "\n\n"
        "same shape as $x$"
        "\n"
        "(pixel-wise / token-wise)",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax4.text(
        6.0, 1.0,
        "Other parameterizations exist ($x_0$-pred, $v$-pred) — same training spirit.",
        ha="center", fontsize=FS_TICK, style="italic", color="0.35",
    )
    return fig4


# -------------------------------------------------------
# FIGURE 5: Loss formula
# -------------------------------------------------------
def build_step_5():
    fig5, ax5 = new_figure()
    fig5.suptitle(
        "Step 5: Training loss -- the simple DDPM objective",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax5)

    round_box(ax5, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax5.text(
        6.0, 7.0,
        r"$L_{\mathrm{simple}}"
        r"= \mathbb{E}_{t,x_0,\epsilon}"
        r"\left["
        r"\|\epsilon - \epsilon_\theta(x_t,t)\|^2"
        r"\right]$",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#f9e79f",
    )

    round_box(ax5, 0.5, 3.2, 11.0, 2.85, C_MUTE, "0.45", lw=1.6)
    ax5.text(
        6.0, 4.6,
        "One SGD step:"
        "\n"
        r"1) sample $x_0$ from data,  $t$ uniform in $1..T$,  $\epsilon\sim\mathcal{N}(0,I)$"
        "\n"
        r"2) form $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$"
        "\n"
        r"3) descend $\|\epsilon-\epsilon_\theta(x_t,t)\|^2$"
        "\n\n"
        r"This is a weighted variational bound on $-\log p_\theta(x_0)$ (simplified).",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax5, 0.45, 0.45, 5.4, 2.45, C_GOAL, C_OK, lw=1.6)
    ax5.text(
        3.15, 1.65,
        "Reconstruction spirit:\n"
        "MSE in noise space,\n"
        "not adversarial CE.\n\n"
        "Stable supervised target.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#145a32",
    )

    round_box(ax5, 6.15, 0.45, 5.4, 2.45, "#f5eef8", C_FUN, lw=1.6)
    ax5.text(
        8.85, 1.65,
        r"Optional reweighting:"
        "\n"
        r"$L=\mathbb{E}[w(t)\|\epsilon-\epsilon_\theta\|^2]$"
        "\n\n"
        "SNR / sigmoid schedules\n"
        "in modern trainers.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig5


# -------------------------------------------------------
# FIGURE 6: Visual forward ladder
# -------------------------------------------------------
def build_step_6():
    fig6, axes6 = new_figure(1, 4)
    fig6.suptitle(
        r"Step 6: Visual intuition -- forward noise ladder $x_0\rightarrow x_T$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ts = [0.0, 0.25, 0.55, 0.95]
    labels = [r"$t=0$ data", r"$t$ small", r"$t$ mid", r"$t\approx 1$ noise"]
    for ax, t, lab in zip(axes6, ts, labels):
        xt, _, _ = q_sample(CLEAN, t)
        ax.scatter(xt[:, 0], xt[:, 1], s=14, c=C_ACCENT if t > 0.1 else C_OK, alpha=0.8)
        ax.set_title(lab, fontsize=FS_AX_TITLE - 3, fontweight="bold")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    fig6.text(
        0.5, 0.02,
        "Structure dissolves into Gaussian dust.  The reverse net must learn the opposite movie.",
        ha="center", fontsize=FS_TICK, fontweight="bold",
    )
    fig6.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig6


# -------------------------------------------------------
# FIGURE 7: Visual reverse
# -------------------------------------------------------
def build_step_7():
    fig7, axes7 = new_figure(1, 4)
    fig7.suptitle(
        r"Step 7: Visual intuition -- reverse denoising $x_T\rightarrow x_0$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # Toy reverse: interpolate from pure noise toward clean (oracle vibe for intuition)
    eps_fixed = RNG.normal(0, 1, size=CLEAN.shape)
    ts_rev = [0.95, 0.55, 0.25, 0.0]
    labels_r = [r"$x_T$", "denoise", "more structure", r"$x_0$"]
    for ax, t, lab in zip(axes7, ts_rev, labels_r):
        # display convex mix for intuition (not the real sampler)
        xt = (1 - t) * CLEAN + t * eps_fixed
        ax.scatter(xt[:, 0], xt[:, 1], s=14,
                   c=C_FUN if t > 0.05 else C_OK, alpha=0.85)
        ax.set_title(lab, fontsize=FS_AX_TITLE - 3, fontweight="bold")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.6, 2.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)

    fig7.text(
        0.5, 0.02,
        r"Training teaches $\epsilon_\theta$ so each reverse step removes the right speck of noise.",
        ha="center", fontsize=FS_TICK, fontweight="bold",
    )
    fig7.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig7


# -------------------------------------------------------
# FIGURE 8: Sampling algorithm
# -------------------------------------------------------
def build_step_8():
    fig8, ax8 = new_figure()
    fig8.suptitle(
        "Step 8: Sampling after training (DDPM step sketch)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax8)

    steps = [
        (6.85, r"1. Draw $x_T\sim\mathcal{N}(0,I)$", C_MUTE),
        (5.9, r"2. For $t=T,\ldots,1$: predict $\hat\epsilon=\epsilon_\theta(x_t,t)$", "#f5eef8"),
        (4.95, r"3. Form mean $\mu_\theta$ from $\hat\epsilon$ (and $\bar\alpha_t$)", C_NEW),
        (4.0, r"4. Sample $x_{t-1}\sim\mathcal{N}(\mu_\theta,\sigma_t^2 I)$  (no noise if $t=1$)", C_OLD),
        (3.05, r"5. Return $x_0$ as the generated sample", C_GOAL),
    ]
    for y, text, face in steps:
        round_box(ax8, 0.5, y - 0.4, 11.0, 0.85, face, "0.45", lw=1.3)
        ax8.text(6.0, y, text, ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

    round_box(ax8, 0.5, 0.45, 11.0, 2.1, C_WARN, "#b7950b", lw=1.6)
    ax8.text(
        6.0, 1.45,
        "Faster samplers (DDIM, distilled 1–4 step, consistency models) change the\n"
        "discretization — not the core trained objective $L_{\\mathrm{simple}}$.\n"
        r"Classifier-free guidance: mix conditional / unconditional $\epsilon_\theta$.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )
    return fig8


# -------------------------------------------------------
# FIGURE 9: Score glance
# -------------------------------------------------------
def build_step_9():
    fig9, ax9 = new_figure()
    fig9.suptitle(
        "Step 9: Broader view -- score matching and family variants",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax9)

    round_box(ax9, 0.4, 4.2, 5.5, 3.4, C_MUTE, "0.45", lw=1.6)
    ax9.text(3.15, 7.2, "Score connection", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax9.text(
        3.15, 5.5,
        r"Score $\nabla_{x}\log q_t(x)$"
        "\n"
        "points to denser regions."
        "\n\n"
        r"$\epsilon$-prediction $\Leftrightarrow$"
        "\n"
        "score prediction up to scaling."
        "\n\n"
        "SDE / ODE samplers unify\n"
        "the reverse dynamics.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax9, 6.1, 4.2, 5.5, 3.4, C_GOAL, C_OK, lw=1.8)
    ax9.text(8.85, 7.2, "Family map", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax9.text(
        8.85, 5.5,
        "DDPM / DDIM\n"
        "score-based SDE\n"
        "latent diffusion (LDM)\n"
        "flow matching / rectified flow\n"
        "consistency models\n\n"
        "Shared idea: learn a field\n"
        "that transports noise → data.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#145a32",
    )

    ax9.text(
        6.0, 2.0,
        "Vs VAE: no encoder KL game.  Vs GAN: supervised noise target, usually stabler.\n"
        "Cost: many network evaluations at sample time (unless distilled).",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    ax9.text(
        6.0, 0.55,
        "Interactive noise-level toy: 0760_generative_toys.py Step 9.",
        ha="center", fontsize=FS_TICK, style="italic", color="0.35",
    )
    return fig9


# -------------------------------------------------------
# FIGURE 10: Takeaway
# -------------------------------------------------------
def build_step_10():
    fig10, ax10 = new_figure()
    fig10.suptitle(
        "Step 10: Takeaway -- how diffusion is trained",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax10)

    round_box(ax10, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax10.text(
        6.0, 7.0,
        r"$L_{\mathrm{simple}}"
        r"= \mathbb{E}_{t,x_0,\epsilon}"
        r"\|\epsilon-\epsilon_\theta("
        r"\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,\,t"
        r")\|^2$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    bullets = [
        r"1. Forward: $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$ (fixed schedule).",
        r"2. Network $\epsilon_\theta(x_t,t)$ predicts that $\epsilon$.",
        r"3. Loss is MSE in noise space (simple DDPM objective).",
        r"4. Sample by iterating the learned reverse chain from $x_T\sim\mathcal{N}(0,I)$.",
        r"5. Guidance / latent / flow variants change the wrapper, not the core idea.",
    ]
    y = 5.35
    for line in bullets:
        round_box(ax10, 0.5, y - 0.35, 11.0, 0.75, C_MUTE, "0.5", lw=1.1)
        ax10.text(6.0, y, line, ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")
        y -= 0.9

    ax10.text(
        6.0, 0.45,
        "Binding lens: each $t$ binds noisy observations to the clean data manifold via denoising.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
    )
    return fig10


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

print("\n" + "=" * 60)
print("0963 DIFFUSION -- TRAINING")
print("=" * 60)
print("""
Forward:
  q(x_t | x_{t-1}) = N(sqrt(1-beta_t) x_{t-1}, beta_t I)
  x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps

Train (simple objective):
  L = E_{t,x0,eps} || eps - eps_theta(x_t, t) ||^2

Sample:
  x_T ~ N(0,I); iterate p_theta(x_{t-1}|x_t) down to x_0

, , 
""")
