"""
0962_GAN.py

Generative Adversarial Network (GAN) -- how it is trained, with the loss formulas.

Sketch:
  1. What a GAN is -- generator vs discriminator
  2. The game: minimax value function
  3. Discriminator loss (with formula)
  4. Generator loss -- saturating vs non-saturating
  5. Visual intuition -- real/fake clouds + D heat
  6. Alternating training step
  7. Visual intuition -- training dynamics cartoon
  8. Failure mode: mode collapse (picture)
  9. Practical notes / WGAN glance
  10. Takeaway

Companion to 0961_VAE.py, 0760_generative_toys.py,
0740_discriminative_generative.py.
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

np.random.seed(62)
RNG = np.random.default_rng(62)


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
C_GRAY = "#b0b0b0"


def ring_samples(n=80, r=1.15, noise=0.06):
    th = RNG.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(th), r * np.sin(th)]) + RNG.normal(0, noise, (n, 2))


def disc_heat(X, r0=1.15, bw=0.22):
    rad = np.sqrt((X ** 2).sum(axis=1))
    return np.exp(-((rad - r0) ** 2) / (2 * bw ** 2))


# -------------------------------------------------------
# FIGURE 1: What a GAN is
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 1: A GAN is a two-player game that learns to generate",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax1)

    round_box(ax1, 0.4, 6.5, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 7.1,
        r"Generator $G_\theta(z)$ invents fakes.  Discriminator $D_\psi(x)$ judges real vs fake.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax1, 0.5, 3.5, 3.4, 2.6, C_NEW, C_DARK, lw=1.8)
    ax1.text(2.2, 5.6, r"Noise $z\sim p(z)$", ha="center", fontsize=FS_TICK, fontweight="bold")
    ax1.text(2.2, 4.5, r"$G_\theta$" "\n" "generator", ha="center",
             fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)

    round_box(ax1, 4.3, 3.5, 3.4, 2.6, C_OLD, C_ACCENT, lw=1.8)
    ax1.text(6.0, 5.6, r"Fake $G(z)$", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_ACCENT)
    ax1.text(6.0, 4.5, "synthetic data\ntrying to look real", ha="center",
             fontsize=FS_TICK + 1, fontweight="bold")

    round_box(ax1, 8.1, 3.5, 3.4, 2.6, C_GOAL, C_OK, lw=1.8)
    ax1.text(9.8, 5.6, r"Judge $D_\psi$", ha="center", fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax1.text(9.8, 4.5, r"$D(x)\in(0,1)$" "\n" "P(real)", ha="center",
             fontsize=FS_ANNOT, fontweight="bold")

    arrow(ax1, 3.9, 4.8, 4.3, 4.8, C_DARK, 2.0)
    arrow(ax1, 7.7, 4.8, 8.1, 4.8, C_DARK, 2.0)

    ax1.text(
        6.0, 2.0,
        "No explicit density $p(x)$.  Generation is learned by fooling a critic.\n"
        "Goodfellow et al. (2014): Generative Adversarial Nets.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_MUTE, edgecolor="0.5", alpha=0.95),
    )
    ax1.text(
        6.0, 0.55,
        "Contrast VAE (0961): likelihood + KL.  GAN: adversarial classification signal.",
        ha="center", fontsize=FS_TICK, style="italic", color="0.35",
    )
    return fig1



# -------------------------------------------------------
# FIGURE 2: Minimax value
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        "Step 2: The training game -- a minimax value function",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax2)

    round_box(ax2, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax2.text(
        6.0, 7.0,
        r"$\min_G \max_D \; V(D,G)$",
        ha="center", va="center", fontsize=FS_ANNOT + 3, fontweight="bold", color="#f9e79f",
    )

    round_box(ax2, 0.5, 2.9, 11.0, 3.1, C_MUTE, "0.45", lw=1.6)
    ax2.text(
        6.0, 4.45,
        r"$V(D,G)"
        r"= \mathbb{E}_{x\sim p_{\mathrm{data}}}\!\left[\log D(x)\right]"
        r"+ \mathbb{E}_{z\sim p_z}\!\left[\log(1-D(G(z)))\right]$"
        "\n\n"
        r"$D$ maximizes $V$ (better detective).   "
        r"$G$ minimizes $V$ (better forger).",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    round_box(ax2, 0.45, 0.45, 5.4, 2.15, C_GOAL, C_OK, lw=1.6)
    ax2.text(
        3.15, 1.5,
        r"Optimum (ideal $G$):"
        "\n"
        r"$p_g = p_{\mathrm{data}}$"
        "\n"
        r"$D^*(x)=\frac{1}{2}$ everywhere.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#145a32",
    )

    round_box(ax2, 6.15, 0.45, 5.4, 2.15, "#f5eef8", C_FUN, lw=1.6)
    ax2.text(
        8.85, 1.5,
        "Intuition:\n"
        "When fakes are perfect,\n"
        "the detective cannot tell\n"
        "real from fake — coin flip.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig2



# -------------------------------------------------------
# FIGURE 3: Discriminator loss
# -------------------------------------------------------
def build_step_3():
    fig3, ax3 = new_figure()
    fig3.suptitle(
        "Step 3: Discriminator loss -- binary classification",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax3)

    round_box(ax3, 0.4, 6.4, 11.2, 1.3, C_DARK, C_DARK, lw=0)
    ax3.text(
        6.0, 7.05,
        r"Fix $G$.  Train $D$ like logistic regression on real=1, fake=0.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#f9e79f",
    )

    round_box(ax3, 0.5, 3.4, 11.0, 2.7, C_MUTE, "0.45", lw=1.6)
    ax3.text(
        6.0, 4.75,
        r"$L_D"
        r"= -\mathbb{E}_{x\sim p_{\mathrm{data}}}[\log D(x)]"
        r" - \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$"
        "\n\n"
        r"Equivalent: maximize $V(D,G)$ w.r.t. $D$  "
        r"(or minimize $-V$).",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )

    round_box(ax3, 0.45, 0.5, 5.4, 2.55, C_GOAL, C_OK, lw=1.6)
    ax3.text(
        3.15, 1.75,
        "Wants $D(\\mathrm{real})\\to 1$\n"
        "Wants $D(G(z))\\to 0$\n\n"
        "Signal: cross-entropy.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#145a32",
    )

    round_box(ax3, 6.15, 0.5, 5.4, 2.55, C_OLD, C_ACCENT, lw=1.6)
    ax3.text(
        8.85, 1.75,
        "If $D$ is perfect too early,\n"
        r"$\log(1-D(G(z)))$ saturates"
        "\n"
        "and $G$ gets vanishing gradients.\n"
        "→ do not over-train $D$.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig3



# -------------------------------------------------------
# FIGURE 4: Generator loss
# -------------------------------------------------------
def build_step_4():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        "Step 4: Generator loss -- saturating vs non-saturating",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax4)

    round_box(ax4, 0.4, 4.15, 5.5, 3.4, C_MUTE, "0.45", lw=1.6)
    ax4.text(3.15, 7.15, "Original (saturating)", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax4.text(
        3.15, 5.4,
        r"$L_G^{\mathrm{sat}}"
        r"= \mathbb{E}_z[\log(1-D(G(z)))]$"
        "\n\n"
        "Minimize this (part of min $V$).\n"
        "When $D(G)\\approx 0$, gradient dies.\n"
        "Hard early in training.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax4, 6.1, 4.15, 5.5, 3.4, C_GOAL, C_OK, lw=1.8)
    ax4.text(8.85, 7.15, "Non-saturating (common)", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_OK)
    ax4.text(
        8.85, 5.4,
        r"$L_G"
        r"= -\mathbb{E}_z[\log D(G(z))]$"
        "\n\n"
        "Maximize $\\log D(G)$ instead.\n"
        "Stronger gradients when fakes\n"
        "are obviously fake.",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold", color="#145a32",
    )

    round_box(ax4, 0.4, 0.45, 11.2, 3.35, "#f5eef8", C_FUN, lw=1.6)
    ax4.text(6.0, 3.3, "Same equilibrium idea", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_FUN)
    ax4.text(
        6.0, 1.7,
        r"$G$ never sees real $x$ in its loss — only $D$'s opinion of $G(z)$."
        "\n"
        "That is the adversarial trick: the learning signal is relative, not MSE-to-data."
        "\n\n"
        r"Heuristic training:  step $D$ on $L_D$, then step $G$ on $L_G$ (often $k_D{:}k_G$ ratio).",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )
    return fig4



# -------------------------------------------------------
# FIGURE 5: Visual intuition -- D heat
# -------------------------------------------------------
def build_step_5():
    fig5, (ax5a, ax5b) = new_figure(1, 2)
    fig5.suptitle(
        "Step 5: Visual intuition -- discriminator field over real vs fake",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    real = ring_samples(100)
    # early generator: blob not yet on the ring
    fake_early = RNG.normal(0, 0.55, size=(100, 2))
    # later generator: near ring with gap
    fake_late = ring_samples(100, r=1.15, noise=0.1)
    fake_late = fake_late[fake_late[:, 0] > -0.2]  # mode-ish
    if len(fake_late) < 60:
        fake_late = ring_samples(80)

    xx, yy = np.meshgrid(np.linspace(-2.4, 2.4, 160), np.linspace(-2.4, 2.4, 160))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    heat = disc_heat(grid).reshape(xx.shape)

    ax5a.contourf(xx, yy, heat, levels=14, cmap="YlOrRd", alpha=0.9)
    ax5a.scatter(real[:, 0], real[:, 1], s=16, c=C_OK, label="real", zorder=3)
    ax5a.scatter(fake_early[:, 0], fake_early[:, 1], s=16, c=C_ACCENT, label="fake (early G)", zorder=3)
    ax5a.set_title("Early: D easily separates", fontsize=FS_AX_TITLE - 2)
    ax5a.legend(fontsize=FS_LEGEND - 1, loc="upper left")
    ax5a.set_aspect("equal")
    ax5a.set_xlim(-2.4, 2.4)
    ax5a.set_ylim(-2.4, 2.4)
    ax5a.grid(True, alpha=0.2)
    ax5a.set_xticks([])
    ax5a.set_yticks([])

    ax5b.contourf(xx, yy, heat, levels=14, cmap="YlOrRd", alpha=0.9)
    ax5b.scatter(real[:, 0], real[:, 1], s=16, c=C_OK, label="real", zorder=3)
    ax5b.scatter(fake_late[:, 0], fake_late[:, 1], s=16, c=C_ACCENT, label="fake (later G)", zorder=3)
    ax5b.set_title("Later: fakes invade high-D region", fontsize=FS_AX_TITLE - 2)
    ax5b.legend(fontsize=FS_LEGEND - 1, loc="upper left")
    ax5b.set_aspect("equal")
    ax5b.set_xlim(-2.4, 2.4)
    ax5b.set_ylim(-2.4, 2.4)
    ax5b.grid(True, alpha=0.2)
    ax5b.set_xticks([])
    ax5b.set_yticks([])

    fig5.text(
        0.5, 0.02,
        r"Heat $\approx D(x)$: bright near the data manifold (ring).  $G$ climbs the heat; $D$ reshapes it.",
        ha="center", fontsize=FS_TICK, fontweight="bold",
    )
    fig5.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig5



# -------------------------------------------------------
# FIGURE 6: Alternating step
# -------------------------------------------------------
def build_step_6():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        "Step 6: One training iteration (alternating updates)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax6)

    steps = [
        (6.9, r"1. Sample real minibatch $x\sim p_{\mathrm{data}}$", C_GOAL),
        (5.95, r"2. Sample $z$; build fakes $\tilde x=G_\theta(z)$", C_OLD),
        (5.0, r"3. Update $D$ by descending $L_D$  (or ascending $V$)", C_NEW),
        (4.05, r"4. Sample fresh $z$; update $G$ by descending $L_G=-\mathbb{E}\log D(G(z))$", "#f5eef8"),
        (3.1, r"5. Repeat.  Optionally $k$ discriminator steps per generator step.", C_MUTE),
    ]
    for y, text, face in steps:
        round_box(ax6, 0.5, y - 0.4, 11.0, 0.85, face, "0.45", lw=1.3)
        ax6.text(6.0, y, text, ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")

    round_box(ax6, 0.5, 0.45, 11.0, 2.15, C_WARN, "#b7950b", lw=1.6)
    ax6.text(
        6.0, 1.5,
        "No reconstruction term.  No KL to a prior inside $G$.\n"
        "The only teacher for $G$ is $D$'s score on fakes.\n"
        r"Equilibrium when $p_g\approx p_{\mathrm{data}}$ and $D\approx\frac{1}{2}$.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
    )
    return fig6



# -------------------------------------------------------
# FIGURE 7: Visual dynamics cartoon
# -------------------------------------------------------
def build_step_7():
    fig7, axes7 = new_figure(1, 3)
    fig7.suptitle(
        "Step 7: Visual intuition -- how the two clouds co-evolve",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    panels = [
        ("Start", ring_samples(70), RNG.normal([-0.2, 0.1], 0.7, (70, 2)), "G far from data"),
        ("Mid", ring_samples(70), ring_samples(70, r=0.7, noise=0.2), "G moves toward manifold"),
        ("Late", ring_samples(70), ring_samples(70, r=1.12, noise=0.08), r"$p_g$ hugs $p_{\mathrm{data}}$"),
    ]
    for ax, (title, real_p, fake_p, note) in zip(axes7, panels):
        ax.scatter(real_p[:, 0], real_p[:, 1], s=18, c=C_OK, alpha=0.75, label="real")
        ax.scatter(fake_p[:, 0], fake_p[:, 1], s=18, c=C_ACCENT, alpha=0.75, label="fake")
        ax.set_title(title, fontsize=FS_AX_TITLE - 2, fontweight="bold")
        ax.text(0.5, -0.08, note, transform=ax.transAxes, ha="center", va="top", fontsize=FS_TICK)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=FS_LEGEND - 2, loc="upper right")

    fig7.tight_layout(rect=[0, 0.06, 1, 0.92])
    return fig7



# -------------------------------------------------------
# FIGURE 8: Mode collapse
# -------------------------------------------------------
def build_step_8():
    fig8, (ax8a, ax8b) = new_figure(1, 2)
    fig8.suptitle(
        "Step 8: Failure mode -- mode collapse (visual)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # multi-mode real
    real_m = np.vstack([
        RNG.normal([-1.3, 0.0], 0.22, (40, 2)),
        RNG.normal([1.3, 0.2], 0.22, (40, 2)),
        RNG.normal([0.0, 1.3], 0.22, (40, 2)),
    ])
    fake_collapse = RNG.normal([1.25, 0.15], 0.18, (120, 2))

    ax8a.scatter(real_m[:, 0], real_m[:, 1], s=22, c=C_OK, alpha=0.8)
    ax8a.set_title(r"Real $p_{\mathrm{data}}$: three modes", fontsize=FS_AX_TITLE - 2)
    ax8a.set_xlim(-2.5, 2.5)
    ax8a.set_ylim(-2.0, 2.5)
    ax8a.set_aspect("equal")
    ax8a.grid(True, alpha=0.25)
    ax8a.set_xticks([])
    ax8a.set_yticks([])

    ax8b.scatter(real_m[:, 0], real_m[:, 1], s=14, c=C_GRAY, alpha=0.35, label="real")
    ax8b.scatter(fake_collapse[:, 0], fake_collapse[:, 1], s=22, c=C_ACCENT, alpha=0.85, label="G")
    ax8b.set_title("Collapsed $G$: one mode only", fontsize=FS_AX_TITLE - 2)
    ax8b.legend(fontsize=FS_LEGEND - 1, loc="upper left")
    ax8b.set_xlim(-2.5, 2.5)
    ax8b.set_ylim(-2.0, 2.5)
    ax8b.set_aspect("equal")
    ax8b.grid(True, alpha=0.25)
    ax8b.set_xticks([])
    ax8b.set_yticks([])

    fig8.text(
        0.5, 0.02,
        r"Low $L_G$ can mean \"always emit the one fake $D$ currently likes\" — not full coverage of $p_{\mathrm{data}}$.",
        ha="center", fontsize=FS_TICK, fontweight="bold",
    )
    fig8.tight_layout(rect=[0, 0.05, 1, 0.92])
    return fig8



# -------------------------------------------------------
# FIGURE 9: Practical / WGAN glance
# -------------------------------------------------------
def build_step_9():
    fig9, ax9 = new_figure()
    fig9.suptitle(
        "Step 9: Practical training notes (and a WGAN glance)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax9)

    round_box(ax9, 0.4, 4.2, 5.5, 3.4, C_MUTE, "0.45", lw=1.6)
    ax9.text(3.15, 7.2, "Vanilla GAN practice", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    ax9.text(
        3.15, 5.5,
        "• non-saturating $L_G$\n"
        "• label smoothing / noise\n"
        "• careful $D$/$G$ step ratio\n"
        "• spectral norm, TTUR, ...\n"
        "• watch mode collapse",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    round_box(ax9, 6.1, 4.2, 5.5, 3.4, "#f5eef8", C_FUN, lw=1.8)
    ax9.text(8.85, 7.2, "WGAN idea (sketch)", ha="center",
             fontsize=FS_TICK, fontweight="bold", color=C_FUN)
    ax9.text(
        8.85, 5.5,
        r"Replace JS-style CE game"
        "\n"
        "with Earth-Mover / Wasserstein:"
        "\n\n"
        r"$L_D = \mathbb{E}[D(G(z))]-\mathbb{E}[D(x)]$"
        "\n"
        r"$L_G = -\mathbb{E}[D(G(z))]$"
        "\n\n"
        "with Lipschitz $D$ (clip / penalty).",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
    )

    ax9.text(
        6.0, 2.0,
        "Same spirit: a critic teaches the generator.\n"
        "Different geometry of \"how wrong is this fake distribution.\"",
        ha="center", fontsize=FS_ANNOT, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_WARN, edgecolor="#b7950b", alpha=0.95),
    )
    ax9.text(
        6.0, 0.55,
        "Interactive G-vs-D toy: 0760_generative_toys.py Step 8.",
        ha="center", fontsize=FS_TICK, style="italic", color="0.35",
    )
    return fig9



# -------------------------------------------------------
# FIGURE 10: Takeaway
# -------------------------------------------------------
def build_step_10():
    fig10, ax10 = new_figure()
    fig10.suptitle(
        "Step 10: Takeaway -- how a GAN is trained",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank(ax10)

    round_box(ax10, 0.4, 6.35, 11.2, 1.35, C_DARK, C_DARK, lw=0)
    ax10.text(
        6.0, 7.0,
        r"$\min_G\max_D V(D,G)"
        r"= \mathbb{E}_{x}[\log D(x)]"
        r"+ \mathbb{E}_{z}[\log(1-D(G(z)))]$",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )

    bullets = [
        r"1. $D$ loss:  $L_D=-\mathbb{E}\log D(x)-\mathbb{E}\log(1-D(G(z)))$.",
        r"2. $G$ loss (common):  $L_G=-\mathbb{E}\log D(G(z))$  (non-saturating).",
        r"3. Alternate SGD/Adam steps on $D$ then $G$.",
        r"4. No pixelwise recon — signal is \"does $D$ believe this?\"",
        r"5. Watch vanishing gradients and mode collapse; consider WGAN-style critics.",
    ]
    y = 5.35
    for line in bullets:
        round_box(ax10, 0.5, y - 0.35, 11.0, 0.75, C_MUTE, "0.5", lw=1.1)
        ax10.text(6.0, y, line, ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold")
        y -= 0.9

    ax10.text(
        6.0, 0.45,
        "Binding lens: $D$ binds \"realness\"; $G$ binds noise $z$ to the data manifold through $D$.",
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
print("0962 GAN -- TRAINING")
print("=" * 60)
print("""
Minimax:
  min_G max_D  V(D,G)
  V = E_x[log D(x)] + E_z[log(1 - D(G(z)))]

Discriminator (minimize):
  L_D = -E log D(x) - E log(1 - D(G(z)))

Generator (common non-saturating, minimize):
  L_G = -E log D(G(z))

Train by alternating updates.  Equilibrium: p_g = p_data, D=1/2.

See also: 0961_VAE.py, 0760_generative_toys.py, 0963_Diffusion.py
""")
