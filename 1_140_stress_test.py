"""
Stress-testing a popular misuse of the CLT / averaging intuition.

Setup (i.i.d. draws with mean mu):
  Xbar_k = (X_1 + ... + X_k) / k
  P_k = P( |Xbar_k - mu| > 0.01 )

For large enough k0, concentration (LLN/CLT) says P_{k0} is small:
the average is usually near mu.

Fallacious leap (seen in streak / "BC game" thinking):
  Suppose X_1 = ... = X_{k0-1} = 0 (a long losing streak).
  Then Xbar_{k0} = X_{k0} / k0.
  "To keep the average near mu, X_{k0} must be about mu*k0 -- huge --
   so after many losses, the next outcome is due to be a win."

This lecture shows why that leap is wrong for i.i.d. Bernoulli trials.

Sketch:
  1. What concentration actually says (unconditionally)
  2. The catch-up story people tell after a losing streak
  3. The binary wall -- one flip cannot restore the mean
  4. Independence -- the next draw does not remember the streak
  5. Interactive: streak length vs true P(win next) vs fallacy belief
  6. Conditional view -- after many zeros, you are MORE often still far from mu
  7. Where CLT still applies (and where it does not)
  8. Takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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

np.random.seed(116)
RNG = np.random.default_rng(116)


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


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_WARN = "#f8d7da"
C_FUN = "#8e44ad"
C_BIN = "#d68910"
C_GRAY = "#85929e"

# Fair-ish coin / BC-style binary outcome
P = 0.5
MU = P
EPS = 0.01


# -------------------------------------------------------
# FIGURE 1: What concentration says (unconditionally)
# -------------------------------------------------------
def build_step_1():
    fig1, (ax1a, ax1b) = new_figure(1, 2)
    fig1.suptitle(
        fr"Step 1: Concentration -- for large $k$, $\bar X_k$ is usually near "
        fr"$\mu={MU:g}$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # Bernoulli(p=MU): CLT curve is smooth and matches the "settles quickly" intuition.
    # (Pure MC on a log scale looks noisy/slow near tiny probabilities.)
    from scipy.stats import norm
    ks = np.unique(np.geomspace(10, 2500, 40).astype(int))
    sigma = float(np.sqrt(P * (1.0 - P)))
    # P(|Xbar-mu| > EPS) ≈ 2 (1 - Φ(EPS * sqrt(k) / σ))
    z = EPS * np.sqrt(ks) / sigma
    p_far = 2.0 * (1.0 - norm.cdf(z))
    # light MC check at a few k (for honesty, not the main curve)
    ks_mc = np.array([20, 50, 100, 200, 500, 1000])
    p_mc = []
    n_mc = 12000
    for k in ks_mc:
        means = RNG.binomial(int(k), P, size=n_mc) / k
        p_mc.append(np.mean(np.abs(means - MU) > EPS))

    rare = 0.001
    ax1a.plot(ks, p_far, color=C_DARK, lw=2.6, label="CLT approximation")
    ax1a.scatter(ks_mc, p_mc, s=36, c=C_FUN, zorder=3, label="Monte Carlo check")
    ax1a.axhline(
        rare, color=C_ACCENT, ls="--", lw=1.8,
        label=fr"rare-event level {rare:g}  (not $\mu$)",
    )
    ax1a.set_xlabel(r"sample size $k$", fontsize=FS_AXIS)
    ax1a.set_ylabel(rf"$P(|\bar X_k-\mu|>{EPS})$", fontsize=FS_AXIS)
    ax1a.set_title(
        fr"how often is the average farther than {EPS} from $\mu={MU:g}$?",
        fontsize=FS_AX_TITLE - 3,
    )
    ax1a.set_yscale("log")
    ax1a.set_xlim(ks.min(), ks.max())
    ax1a.legend(fontsize=FS_LEGEND - 2, loc="upper right")
    ax1a.grid(True, alpha=0.3, which="both")
    ax1a.text(
        0.02, 0.02,
        fr"Setup: i.i.d. Bernoulli($p$), so $\mu=p={MU:g}$."
        + "\n"
        r"Red line = a small probability threshold,"
        + "\n"
        r"not the mean $\mu$.",
        transform=ax1a.transAxes, ha="left", va="bottom", fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )

    # histogram at a large k
    k_big = 1000
    means_big = RNG.binomial(k_big, P, size=5000) / k_big
    ax1b.hist(means_big, bins=40, density=True, color=C_NEW, edgecolor=C_DARK, alpha=0.9)
    ax1b.axvline(MU, color=C_ACCENT, ls="--", lw=2.2, label=fr"true mean $\mu={MU:g}$")
    ax1b.axvspan(MU - EPS, MU + EPS, color=C_OK, alpha=0.15, label=fr"within {EPS} of $\mu$")
    ax1b.set_xlabel(fr"$\bar X_{{{k_big}}}$", fontsize=FS_AXIS)
    ax1b.set_title("many experiments at large k", fontsize=FS_AX_TITLE - 2)
    ax1b.legend(fontsize=FS_LEGEND - 1)
    ax1b.grid(True, alpha=0.3)
    fig1.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig1


# -------------------------------------------------------
# FIGURE 2: The catch-up fallacy
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        "Step 2: The catch-up story after a long losing streak",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax2)

    round_box(ax2, 0.5, 5.3, 11.0, 2.2, C_DARK, C_DARK, lw=0)
    ax2.text(
        6.0, 6.4,
        "Misreading concentration as a debt the next outcome must repay.",
        ha="center", va="center", fontsize=FS_ANNOT + 2,
        fontweight="bold", color="#f9e79f",
    )

    steps = [
        (0.4, 1.2, "1", "Long streak", r"$X_1=\cdots=X_{k_0-1}=0$"),
        (3.3, 1.2, "2", "Then average is", r"$\bar X_{k_0}=X_{k_0}/k_0$"),
        (6.2, 1.2, "3", "Fallacy", r"need $X_{k_0}\approx \mu k_0$" + "\nto rescue the mean"),
        (9.1, 1.2, "4", "Conclusion", "so a win is\n\"due\" now"),
    ]
    for x0, y0, num, title, body in steps:
        round_box(ax2, x0, y0, 2.7, 3.6, C_WARN, C_ACCENT, lw=1.8)
        ax2.text(x0 + 1.35, y0 + 3.05, num, ha="center",
                 fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_ACCENT)
        ax2.text(x0 + 1.35, y0 + 2.35, title, ha="center",
                 fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax2.text(x0 + 1.35, y0 + 1.2, body, ha="center", va="center",
                 fontsize=FS_TICK + 1)
    return fig2


# -------------------------------------------------------
# FIGURE 3: Binary wall
# -------------------------------------------------------
def build_step_3():
    fig3, (ax3a, ax3b) = new_figure(1, 2)
    fig3.suptitle(
        r"Step 3: Binary wall -- one $0/1$ draw cannot restore the mean",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    k0 = 200
    # After k0-1 zeros, possible averages at k0:
    avg_if_0 = 0.0
    avg_if_1 = 1.0 / k0
    needed = MU  # to be near mu
    needed_x = MU * k0

    ax3a.bar(["if lose again\n(X=0)", "if win now\n(X=1)", "needed for\nmean≈μ"],
             [avg_if_0, avg_if_1, needed],
             color=[C_GRAY, C_FUN, C_ACCENT], edgecolor="white", width=0.6)
    ax3a.set_ylabel(fr"$\bar X_{{{k0}}}$", fontsize=FS_AXIS)
    ax3a.set_title(fr"after {k0-1} zeros in a row", fontsize=FS_AX_TITLE - 2)
    ax3a.set_ylim(0, 0.6)
    ax3a.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate([avg_if_0, avg_if_1, needed]):
        ax3a.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=FS_ANNOT, fontweight="bold")

    ax3b.set_xlim(0, 10)
    ax3b.set_ylim(0, 10)
    ax3b.axis("off")
    round_box(ax3b, 0.5, 1.2, 9.0, 7.5, C_MUTE, "0.45", lw=1.8)
    ax3b.text(
        5.0, 5.2,
        fr"Fallacy asks for $X_{{k_0}}\approx\mu k_0={needed_x:.0f}$."
        "\n\n"
        "But a binary outcome\n"
        "is only 0 or 1.\n"
        fr"Best possible average after the streak:\n$1/{k0}={avg_if_1:.4f}$ -- still tiny."
        "\n\n"
        "One flip cannot pay back a long streak.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        color=C_DARK,
    )
    return fig3


# -------------------------------------------------------
# FIGURE 4: Independence
# -------------------------------------------------------
def build_step_4():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        "Step 4: Independence -- the next draw does not remember the streak",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax4)

    round_box(ax4, 0.5, 4.6, 11.0, 2.8, C_GOAL, C_OK, lw=2.2)
    ax4.text(
        6.0, 6.0,
        r"For i.i.d. Bernoulli($p$):"
        "\n"
        r"$P(X_k=1\mid X_1=\cdots=X_{k-1}=0)=p$"
        "\n"
        "same as the unconditional chance.",
        ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold",
        color="#145a32",
    )

    trio = [
        (0.5, 0.7, C_WARN, C_ACCENT, "Fallacy feels", "\"due for a win\""),
        (4.25, 0.7, C_NEW, "#2874a6", "Math says", "memoryless next draw"),
        (8.0, 0.7, C_GOAL, C_OK, "CLT/LLN say", "about averages over\nmany futures --\nnot this conditional debt"),
    ]
    for x0, y0, face, edge, title, body in trio:
        round_box(ax4, x0, y0, 3.5, 3.4, face, edge, lw=2.0)
        ax4.text(x0 + 1.75, y0 + 2.55, title, ha="center",
                 fontsize=FS_ANNOT + 1, fontweight="bold", color=edge)
        ax4.text(x0 + 1.75, y0 + 1.3, body, ha="center", va="center",
                 fontsize=FS_ANNOT)
    return fig4


# -------------------------------------------------------
# FIGURE 5: Interactive streak vs true odds
# -------------------------------------------------------
def build_step_5():
    fig5, (ax5a, ax5b) = new_figure(1, 2)
    fig5.suptitle(
        "Step 5: Drag streak length -- true P(win next) does not rise",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig5.subplots_adjust(bottom=0.22, left=0.08, right=0.98, wspace=0.28)
    ax5a.set_navigate(False)
    ax5b.set_navigate(False)

    def fallacy_belief(streak, k0=200, p=P):
        """Toy 'due' belief that grows with streak (not a real probability)."""
        # saturating curve people feel: more losses => more 'sure' of a win
        return float(1 - np.exp(-streak / 25.0)) * 0.85 + 0.5 * 0.15

    def draw_streak(streak):
        ax5a.clear(); ax5b.clear()
        streak = int(np.clip(streak, 1, 150))
        true_p = P
        felt_p = fallacy_belief(streak)

        # path of mean if continue with zeros then optional next
        k = np.arange(0, streak + 1)
        mean_path = np.zeros_like(k, dtype=float)  # all zeros so far
        ax5a.plot(k, mean_path, color=C_GRAY, lw=2.2, label="mean during losing streak")
        ax5a.axhline(MU, color=C_ACCENT, ls="--", lw=2.0, label=fr"$\mu={MU:g}$")
        # after one more win vs lose
        ax5a.scatter([streak + 1], [1.0 / (streak + 1)], s=80, c=C_OK, zorder=3,
                     label="mean if next=1")
        ax5a.scatter([streak + 1], [0.0], s=80, c=C_ACCENT, zorder=3,
                     label="mean if next=0")
        ax5a.set_xlabel("k", fontsize=FS_AXIS)
        ax5a.set_ylabel(r"$\bar X_k$", fontsize=FS_AXIS)
        ax5a.set_title(f"after {streak} losses, even a win barely moves the mean",
                       fontsize=FS_AX_TITLE - 3)
        ax5a.legend(fontsize=FS_LEGEND - 2, loc="upper right")
        ax5a.set_ylim(-0.05, 0.7)
        ax5a.grid(True, alpha=0.3)

        ax5b.bar(["true P(win next)", "fallacy 'due' feeling"],
                 [true_p, felt_p],
                 color=[C_OK, C_ACCENT], edgecolor="white", width=0.55)
        ax5b.set_ylim(0, 1.05)
        ax5b.set_ylabel("probability-like score", fontsize=FS_AXIS)
        ax5b.set_title("independence vs catch-up intuition", fontsize=FS_AX_TITLE - 2)
        ax5b.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate([true_p, felt_p]):
            ax5b.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=FS_ANNOT, fontweight="bold")

    # Keep clear of corner nav buttons; store ref so the widget stays alive.
    ax_st = fig5.add_axes([0.22, 0.08, 0.56, 0.04])
    ax_st.set_navigate(False)
    ax_st.set_zorder(20)
    sl_st = Slider(ax_st, "losing streak", 1, 150, valinit=40, valstep=1, color=C_FUN)

    def _u5(_=None):
        # Release a stuck mouse grab at the slider end-stops so Next stays usable.
        grabber = fig5.canvas.mouse_grabber
        if grabber is not None:
            try:
                fig5.canvas.release_mouse(grabber)
            except Exception:
                pass
        draw_streak(sl_st.val)
        fig5.canvas.draw_idle()

    sl_st.on_changed(_u5)
    fig5._sliders = (sl_st,)
    _u5()
    print("Step 5: drag streak length.")
    return fig5


# -------------------------------------------------------
# FIGURE 6: Conditional -- after many zeros you are often still far
# -------------------------------------------------------
def build_step_6():
    fig6, (ax6a, ax6b) = new_figure(1, 2)
    fig6.suptitle(
        "Step 6: Conditioning on a long zero streak -- you are usually STILL far from μ",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    streak0 = 50
    k_end = streak0 + 1
    # Simulate many futures AFTER a fixed prefix of zeros
    n_sims = 5000
    # Prefix contributes 0; remaining draws are free i.i.d.
    # At k_end, mean = (sum of one new draw + later...) wait for fairness at fixed k0:
    k0 = 200
    prefix = streak0
    remain = k0 - prefix
    # Conditional on first `prefix` zeros: mean at k0 = S_remain / k0
    s_rem = RNG.binomial(remain, P, size=n_sims)
    means_cond = s_rem / k0
    means_uncond = RNG.binomial(k0, P, size=n_sims) / k0

    ax6a.hist(means_uncond, bins=40, density=True, alpha=0.55, color=C_GRAY,
              label="unconditional")
    ax6a.hist(means_cond, bins=40, density=True, alpha=0.65, color=C_FUN,
              label=fr"after {prefix} leading zeros")
    ax6a.axvline(MU, color=C_ACCENT, ls="--", lw=2.2)
    ax6a.set_xlabel(fr"$\bar X_{{{k0}}}$", fontsize=FS_AXIS)
    ax6a.set_title("distribution of the average at fixed k0", fontsize=FS_AX_TITLE - 2)
    ax6a.legend(fontsize=FS_LEGEND - 1)
    ax6a.grid(True, alpha=0.3)

    # Probability still far from mu
    p_far_uncond = np.mean(np.abs(means_uncond - MU) > EPS)
    p_far_cond = np.mean(np.abs(means_cond - MU) > EPS)
    ax6b.bar(["unconditional", f"after {prefix}\nleading zeros"],
             [p_far_uncond, p_far_cond],
             color=[C_GRAY, C_ACCENT], edgecolor="white", width=0.55)
    ax6b.set_ylabel(rf"$P(|\bar X-\mu|>{EPS})$", fontsize=FS_AXIS)
    ax6b.set_title("conditioning on the streak makes FAR more likely",
                   fontsize=FS_AX_TITLE - 3)
    ax6b.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate([p_far_uncond, p_far_cond]):
        ax6b.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=FS_ANNOT, fontweight="bold")
    fig6.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig6


# -------------------------------------------------------
# FIGURE 7: Where CLT still applies
# -------------------------------------------------------
def build_step_7():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        "Step 7: CLT / LLN are about many futures -- not a debt after one path",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax7)

    round_box(ax7, 0.5, 4.4, 5.3, 3.0, C_GOAL, C_OK, lw=2.2)
    ax7.text(3.15, 6.6, "Still true", ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    ax7.text(
        3.15, 5.4,
        "Across many i.i.d. experiments,\n"
        "large-k averages concentrate near μ.\n"
        "CLT describes fluctuation shape.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )

    round_box(ax7, 6.2, 4.4, 5.3, 3.0, C_WARN, C_ACCENT, lw=2.2)
    ax7.text(8.85, 6.6, "Not licensed", ha="center",
             fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_ACCENT)
    ax7.text(
        8.85, 5.4,
        "After an atypical streak,\n"
        "the next binary outcome\n"
        "is not forced to \"correct\" it.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )

    round_box(ax7, 0.5, 0.5, 11.0, 3.4, C_MUTE, "0.45", lw=1.6)
    ax7.text(
        6.0, 2.2,
        "BC-game / streak thinking swaps an unconditional concentration fact\n"
        "for a conditional fairness fantasy.\n\n"
        "Independence survives the streak. The average's \"debt\" does not.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )
    return fig7


# -------------------------------------------------------
# FIGURE 8: Takeaway
# -------------------------------------------------------
def build_step_8():
    fig8, axes8 = new_figure(1, 3)
    fig8.suptitle(
        "Step 8: Stress-test the slogan -- \"due for a win\" fails",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    panels = [
        (axes8[0], "Concentration", "Large-k averages\nare usually near μ\n(unconditionally).", C_NEW, "#2874a6"),
        (axes8[1], "Catch-up fallacy", "One 0/1 draw cannot\nrepay a long streak.\nNext odds unchanged.", C_WARN, C_ACCENT),
        (axes8[2], "Discipline", "Do not bet on\nCLT as a debt collector.", C_GOAL, C_OK),
    ]
    for ax, title, body, face, edge in panels:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        round_box(ax, 0.5, 1.5, 9.0, 7.0, face, edge, lw=2.2)
        ax.text(5.0, 7.0, title, ha="center", fontsize=FS_AX_TITLE - 2,
                fontweight="bold", color=edge)
        ax.text(5.0, 4.2, body, ha="center", va="center", fontsize=FS_ANNOT + 1)
    return fig8


run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
    build_step_5,
    build_step_6,
    build_step_7,
    build_step_8,
])

print("\n" + "=" * 60)
print("STRESS TEST -- CLT CATCH-UP FALLACY")
print("=" * 60)
print(rf"""
True: for large k, P(|Xbar_k - mu| > {EPS}) becomes small (unconditionally).

False leap:
  after many zeros, the next X must be huge / a win is "due"
  so that Xbar returns to mu.

Why false for Bernoulli:
  1) X_next is only 0 or 1 -- cannot equal mu*k
  2) independence: P(win | long losing streak) = p
  3) conditioning on the streak makes being FAR from mu MORE likely at fixed k0

CLT/LLN describe typical averages across futures -- not a debt after one path.
""")
