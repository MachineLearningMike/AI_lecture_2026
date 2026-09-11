"""
0630_Bayesian_theorem.py

Bayes' theorem -- update a belief when new evidence arrives.

  P(H | E) = P(E | H) P(H) / P(E)

Prior belief times how well the hypothesis predicts the evidence,
renormalized by how common the evidence is overall.

Sketch:
  1. The question Bayes answers
  2. The four pieces: prior, likelihood, evidence, posterior
  3. Frequency picture -- count your way to the same formula
  4. Classic test: rare disease + imperfect test
  5. Interactive: drag the prior
  6. Interactive: drag sensitivity / false-positive rate
  7. Continuous Bayes: prior curve x likelihood -> posterior
  8. Interactive continuous update
  9. Takeaway

Companion intuition for probability updates; pairs with 0590_gaussian.py,
0610_central_limit_theorem.py, 0470_maximum_likelihood.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
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

np.random.seed(95)
RNG = np.random.default_rng(95)


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


def bayes_binary(prior, sens, fpr):
    """Posterior P(H|E+) for binary hypothesis and positive test."""
    prior = float(np.clip(prior, 1e-6, 1 - 1e-6))
    sens = float(np.clip(sens, 1e-6, 1 - 1e-6))
    fpr = float(np.clip(fpr, 1e-6, 1 - 1e-6))
    p_e = sens * prior + fpr * (1 - prior)
    post = sens * prior / p_e
    return post, p_e


# -------------------------------------------------------
# FIGURE 1: The question
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 1: Bayes answers -- how should a belief move when evidence arrives?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax1)

    round_box(ax1, 0.5, 5.3, 11.0, 2.2, C_DARK, C_DARK, lw=0)
    ax1.text(
        6.0, 6.4,
        "Start with a prior belief about a hypothesis H.\n"
        "Observe evidence E.  What is a coherent posterior belief?",
        ha="center", va="center", fontsize=FS_ANNOT + 2,
        fontweight="bold", color="#f9e79f",
    )

    trio = [
        (0.5, 1.2, C_NEW, "#2874a6", "Before", r"$P(H)$" + "\nprior"),
        (4.25, 1.2, "#f5eef8", C_FUN, "Evidence", r"$E$" + "\ndata / test / signal"),
        (8.0, 1.2, C_GOAL, C_OK, "After", r"$P(H\mid E)$" + "\nposterior"),
    ]
    for x0, y0, face, edge, title, body in trio:
        round_box(ax1, x0, y0, 3.5, 3.5, face, edge, lw=2.2)
        ax1.text(x0 + 1.75, y0 + 2.6, title, ha="center",
                 fontsize=FS_AX_TITLE - 2, fontweight="bold", color=edge)
        ax1.text(x0 + 1.75, y0 + 1.3, body, ha="center", va="center",
                 fontsize=FS_ANNOT + 1)
    return fig1



# -------------------------------------------------------
# FIGURE 2: Four pieces of the formula
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        r"Step 2: $P(H\mid E)=\dfrac{P(E\mid H)\,P(H)}{P(E)}$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax2)

    pieces = [
        (0.4, 4.4, C_NEW, "#2874a6", "Prior", r"$P(H)$", "belief before E"),
        (3.3, 4.4, "#f5eef8", C_FUN, "Likelihood", r"$P(E\mid H)$", "how H predicts E"),
        (6.2, 4.4, C_WARN, C_ACCENT, "Evidence", r"$P(E)$", "how common E is"),
        (9.1, 4.4, C_GOAL, C_OK, "Posterior", r"$P(H\mid E)$", "belief after E"),
    ]
    for x0, y0, face, edge, title, formula, body in pieces:
        round_box(ax2, x0, y0, 2.7, 3.1, face, edge, lw=2.0)
        ax2.text(x0 + 1.35, y0 + 2.45, title, ha="center",
                 fontsize=FS_ANNOT + 1, fontweight="bold", color=edge)
        ax2.text(x0 + 1.35, y0 + 1.55, formula, ha="center",
                 fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
        ax2.text(x0 + 1.35, y0 + 0.7, body, ha="center", va="center",
                 fontsize=FS_TICK + 1)

    round_box(ax2, 0.4, 0.5, 11.2, 3.4, C_MUTE, "0.45", lw=1.6)
    ax2.text(
        6.0, 2.2,
        "Multiply prior by likelihood, then divide by the marginal chance of the evidence.\n"
        "Rare hypotheses stay rare unless the evidence is unusually hard to explain otherwise.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )
    return fig2



# -------------------------------------------------------
# FIGURE 3: Frequency / count picture
# -------------------------------------------------------
def build_step_3():
    fig3, axes3 = new_figure(1, 2)
    fig3.suptitle(
        "Step 3: Count picture -- Bayes is bookkeeping on a population",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    # 1000 people, prior 1%, sens 99%, fpr 5%  (classic-ish)
    N = 1000
    prior0, sens0, fpr0 = 0.01, 0.99, 0.05
    n_h = int(round(prior0 * N))
    n_not = N - n_h
    tp = int(round(sens0 * n_h))
    fn = n_h - tp
    fp = int(round(fpr0 * n_not))
    tn = n_not - fp

    axes3[0].axis("off")
    axes3[0].set_xlim(0, 10)
    axes3[0].set_ylim(0, 10)
    axes3[0].set_title(f"Population N = {N}", fontsize=FS_AX_TITLE - 2)
    round_box(axes3[0], 0.5, 6.2, 4.2, 3.0, C_WARN, C_ACCENT, lw=1.8)
    axes3[0].text(2.6, 8.3, f"Has H\n{n_h}", ha="center", va="center",
                  fontsize=FS_ANNOT + 1, fontweight="bold", color=C_ACCENT)
    axes3[0].text(2.6, 6.9, f"TP {tp}   FN {fn}", ha="center", fontsize=FS_TICK)
    round_box(axes3[0], 5.3, 6.2, 4.2, 3.0, C_NEW, "#2874a6", lw=1.8)
    axes3[0].text(7.4, 8.3, f"No H\n{n_not}", ha="center", va="center",
                  fontsize=FS_ANNOT + 1, fontweight="bold", color="#2874a6")
    axes3[0].text(7.4, 6.9, f"FP {fp}   TN {tn}", ha="center", fontsize=FS_TICK)
    round_box(axes3[0], 1.5, 1.2, 7.0, 4.2, C_GOAL, C_OK, lw=2.0)
    axes3[0].text(
        5.0, 3.3,
        f"Among test-positive people:\n"
        f"TP + FP = {tp + fp}\n\n"
        f"True H | positive ≈ {tp}/{tp + fp}\n"
        f"= {tp / (tp + fp):.2f}",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
    )

    # Bar: prior vs posterior
    post0, _ = bayes_binary(prior0, sens0, fpr0)
    axes3[1].bar(["prior\nP(H)", "posterior\nP(H | +)"], [prior0, post0],
                 color=[C_GRAY, C_OK], edgecolor="white", width=0.55)
    axes3[1].set_ylim(0, 1.05)
    axes3[1].set_ylabel("probability", fontsize=FS_AXIS)
    axes3[1].set_title("same numbers, belief update", fontsize=FS_AX_TITLE - 2)
    axes3[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate([prior0, post0]):
        axes3[1].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=FS_ANNOT, fontweight="bold")
    fig3.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig3



# -------------------------------------------------------
# FIGURE 4: Why the posterior can stay modest
# -------------------------------------------------------
def build_step_4():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        "Step 4: A positive test is not a diagnosis when H is rare",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax4)

    # Same classic numbers as Step 3 (local -- not shared across builders)
    N = 1000
    prior0, sens0, fpr0 = 0.01, 0.99, 0.05
    n_h = int(round(prior0 * N))
    n_not = N - n_h
    tp = int(round(sens0 * n_h))
    fp = int(round(fpr0 * n_not))
    post0, _ = bayes_binary(prior0, sens0, fpr0)
    share_tp = tp / (tp + fp)

    # Stack of positive tests: mostly FP if base rate tiny
    round_box(ax4, 1.5, 1.5, 3.5, 5.5, C_WARN, C_ACCENT, lw=2.0)
    ax4.add_patch(Rectangle((1.7, 1.7), 3.1, 5.1 * share_tp,
                            facecolor=C_OK, edgecolor="white", lw=1.5))
    ax4.add_patch(Rectangle((1.7, 1.7 + 5.1 * share_tp), 3.1,
                            5.1 * (1.0 - share_tp),
                            facecolor=C_ACCENT, edgecolor="white", lw=1.5))
    ax4.text(3.25, 6.5, "all who\ntested +", ha="center", fontsize=FS_ANNOT,
             fontweight="bold", color=C_DARK)
    ax4.text(3.25, 2.3, "mostly\nfalse +", ha="center", fontsize=FS_ANNOT,
             fontweight="bold", color="white")
    ax4.text(3.25, 4.5, "true +", ha="center", fontsize=FS_ANNOT,
             fontweight="bold", color="white")

    round_box(ax4, 6.0, 1.5, 5.5, 5.5, C_MUTE, "0.45", lw=1.8)
    ax4.text(
        8.75, 4.25,
        f"Prior P(H) = {prior0:.0%}\n"
        f"Sensitivity P(+|H) = {sens0:.0%}\n"
        f"False-positive P(+|no H) = {fpr0:.0%}\n\n"
        f"Posterior P(H|+) ≈ {post0:.0%}\n\n"
        "The test is good.\n"
        "The base rate is tiny.\n"
        "Bayes multiplies both facts.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
    )
    return fig4



# -------------------------------------------------------
# FIGURE 5: Interactive prior
# -------------------------------------------------------
def build_step_5():
    fig5, (ax5a, ax5b) = new_figure(1, 2)
    fig5.suptitle(
        "Step 5: Drag the prior -- watch the posterior move",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig5.subplots_adjust(bottom=0.22, left=0.08, right=0.98, wspace=0.28)
    ax5a.set_navigate(False)
    ax5b.set_navigate(False)

    sens_fix, fpr_fix = 0.95, 0.05
    prior_init = 0.02

    def draw_prior_toy(prior):
        ax5a.clear(); ax5b.clear()
        post, p_e = bayes_binary(prior, sens_fix, fpr_fix)
        # mosaic population of 200 for visual
        n = 200
        n_h = max(1, int(round(prior * n)))
        n_not = n - n_h
        tp = int(round(sens_fix * n_h))
        fp = int(round(fpr_fix * n_not))
        # simple stacked bars of + tests
        ax5a.bar(["Has H", "No H"], [n_h, n_not], color=[C_ACCENT, C_GRAY],
                 edgecolor="white", width=0.55)
        ax5a.set_ylabel("people (toy N=200)", fontsize=FS_AXIS)
        ax5a.set_title(f"prior P(H) = {prior:.3f}", fontsize=FS_AX_TITLE - 2)
        ax5a.grid(True, alpha=0.3, axis="y")

        ax5b.barh(["P(+)", "TP among +", "posterior"], [p_e, tp / max(tp + fp, 1), post],
                  color=[C_BIN, C_FUN, C_OK], edgecolor="white", height=0.55)
        ax5b.set_xlim(0, 1.05)
        ax5b.set_xlabel("probability / fraction", fontsize=FS_AXIS)
        ax5b.set_title(
            f"sens={sens_fix:.2f}, fpr={fpr_fix:.2f}   |   P(H|+)={post:.3f}",
            fontsize=FS_AX_TITLE - 3,
        )
        ax5b.grid(True, alpha=0.3, axis="x")

    ax_pr = fig5.add_axes([0.25, 0.08, 0.50, 0.04])
    ax_pr.set_navigate(False)
    ax_pr.set_zorder(20)
    sl_pr = Slider(ax_pr, r"prior $P(H)$", 0.001, 0.50, valinit=prior_init,
                   valstep=0.001, color=C_FUN)

    def _u5(_=None):
        draw_prior_toy(float(sl_pr.val))
        fig5.canvas.draw_idle()

    sl_pr.on_changed(_u5)
    fig5._sliders = (sl_pr,)  # keep alive; otherwise the slider goes dead
    _u5()
    print("Step 5: drag prior.")
    return fig5



# -------------------------------------------------------
# FIGURE 6: Interactive sens / fpr
# -------------------------------------------------------
def build_step_6():
    fig6, (ax6a, ax6b) = new_figure(1, 2)
    fig6.suptitle(
        "Step 6: Drag sensitivity and false-positive rate",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig6.subplots_adjust(bottom=0.24, left=0.08, right=0.98, wspace=0.28)
    ax6a.set_navigate(False)
    ax6b.set_navigate(False)

    prior_fix = 0.02

    def draw_test_toy(sens, fpr):
        ax6a.clear(); ax6b.clear()
        post, p_e = bayes_binary(prior_fix, sens, fpr)
        priors = np.linspace(0.001, 0.4, 200)
        posts = np.array([bayes_binary(p, sens, fpr)[0] for p in priors])
        ax6a.plot(priors, posts, color=C_OK, lw=2.6, label=r"$P(H\mid +)$")
        ax6a.plot(priors, priors, color=C_GRAY, ls="--", lw=1.5, label="no update")
        ax6a.scatter([prior_fix], [post], s=80, c=C_ACCENT, zorder=3)
        ax6a.set_xlabel(r"prior $P(H)$", fontsize=FS_AXIS)
        ax6a.set_ylabel(r"posterior $P(H\mid +)$", fontsize=FS_AXIS)
        ax6a.set_title("update curve for this test", fontsize=FS_AX_TITLE - 2)
        ax6a.legend(fontsize=FS_LEGEND - 1, loc="lower right")
        ax6a.grid(True, alpha=0.3)
        ax6a.set_xlim(0, 0.4); ax6a.set_ylim(0, 1.02)

        ax6b.bar(["prior", "posterior"], [prior_fix, post],
                 color=[C_GRAY, C_OK], edgecolor="white", width=0.55)
        ax6b.set_ylim(0, 1.05)
        ax6b.set_title(
            f"sens={sens:.2f}  fpr={fpr:.2f}  P(+)= {p_e:.3f}",
            fontsize=FS_AX_TITLE - 3,
        )
        ax6b.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate([prior_fix, post]):
            ax6b.text(i, v + 0.03, f"{v:.3f}", ha="center", fontsize=FS_ANNOT, fontweight="bold")

    # Keep clear of corner nav buttons.
    ax_s = fig6.add_axes([0.12, 0.09, 0.32, 0.04])
    # Right slider moved ~3 cm right (≈0.109 figure fraction).
    ax_f = fig6.add_axes([0.629, 0.09, 0.32, 0.04])
    for _ax in (ax_s, ax_f):
        _ax.set_navigate(False)
        _ax.set_zorder(20)
    sl_s = Slider(ax_s, "sensitivity", 0.50, 0.999, valinit=0.95, valstep=0.01, color=C_OK)
    sl_f = Slider(ax_f, "false + rate", 0.001, 0.30, valinit=0.05, valstep=0.01, color=C_ACCENT)

    def _u6(_=None):
        draw_test_toy(float(sl_s.val), float(sl_f.val))
        fig6.canvas.draw_idle()

    sl_s.on_changed(_u6)
    sl_f.on_changed(_u6)
    fig6._sliders = (sl_s, sl_f)  # keep alive; otherwise the sliders go dead
    _u6()
    print("Step 6: drag sensitivity / FPR.")
    return fig6



# -------------------------------------------------------
# FIGURE 7: Continuous Bayes picture
# -------------------------------------------------------
def build_step_7():
    fig7, axes7 = new_figure(1, 3)
    fig7.suptitle(
        r"Step 7: Continuous form -- posterior $\propto$ likelihood $\times$ prior",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    theta = np.linspace(-4, 4, 400)
    prior_pdf = np.exp(-0.5 * ((theta - 0.0) / 1.2) ** 2)
    prior_pdf /= np.trapezoid(prior_pdf, theta)
    # evidence: observation x=1.6 under N(theta, 1)
    x_obs = 1.6
    like = np.exp(-0.5 * (x_obs - theta) ** 2)
    like /= np.trapezoid(like, theta)  # shape only for display
    unnorm = like * prior_pdf
    post_pdf = unnorm / np.trapezoid(unnorm, theta)

    axes7[0].fill_between(theta, prior_pdf, color=C_GRAY, alpha=0.45)
    axes7[0].plot(theta, prior_pdf, color=C_DARK, lw=2.2)
    axes7[0].set_title("prior", fontsize=FS_AX_TITLE - 2)
    axes7[1].fill_between(theta, like, color=C_FUN, alpha=0.35)
    axes7[1].plot(theta, like, color=C_FUN, lw=2.2)
    axes7[1].axvline(x_obs, color=C_ACCENT, ls="--", lw=1.6)
    axes7[1].set_title("likelihood shape", fontsize=FS_AX_TITLE - 2)
    axes7[2].fill_between(theta, post_pdf, color=C_OK, alpha=0.35)
    axes7[2].plot(theta, post_pdf, color=C_OK, lw=2.2)
    axes7[2].plot(theta, prior_pdf, color=C_GRAY, ls="--", lw=1.4, alpha=0.8)
    axes7[2].set_title("posterior", fontsize=FS_AX_TITLE - 2)
    for ax in axes7:
        ax.set_xlim(-4, 4)
        ax.set_yticks([])
        ax.grid(True, alpha=0.25)
        ax.set_xlabel(r"$\theta$", fontsize=FS_AXIS)
    fig7.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig7



# -------------------------------------------------------
# FIGURE 8: Interactive continuous update
# -------------------------------------------------------
def build_step_8():
    fig8, (ax8a, ax8b) = new_figure(1, 2)
    fig8.suptitle(
        "Step 8: Drag observation and prior width -- live posterior",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig8.subplots_adjust(bottom=0.22, left=0.08, right=0.98, wspace=0.28)
    ax8a.set_navigate(False)
    ax8b.set_navigate(False)

    theta8 = np.linspace(-4, 4, 500)

    def draw_cont(x_obs, prior_sd):
        ax8a.clear(); ax8b.clear()
        prior = np.exp(-0.5 * (theta8 / max(prior_sd, 0.05)) ** 2)
        prior /= np.trapezoid(prior, theta8)
        like = np.exp(-0.5 * (x_obs - theta8) ** 2)  # obs noise sd=1
        post_u = prior * like
        post = post_u / np.trapezoid(post_u, theta8)

        ax8a.fill_between(theta8, prior, color=C_GRAY, alpha=0.35, label="prior")
        ax8a.plot(theta8, prior, color=C_DARK, lw=2.0)
        ax8a.fill_between(theta8, like / np.trapezoid(like, theta8), color=C_FUN,
                          alpha=0.25, label="likelihood (normed)")
        ax8a.axvline(x_obs, color=C_ACCENT, ls="--", lw=1.6, label="observation")
        ax8a.set_title("prior and likelihood", fontsize=FS_AX_TITLE - 2)
        ax8a.legend(fontsize=FS_LEGEND - 1, loc="upper left")
        ax8a.set_xlim(-4, 4); ax8a.set_yticks([]); ax8a.grid(True, alpha=0.25)

        ax8b.fill_between(theta8, post, color=C_OK, alpha=0.35)
        ax8b.plot(theta8, post, color=C_OK, lw=2.4, label="posterior")
        ax8b.plot(theta8, prior, color=C_GRAY, ls="--", lw=1.5, label="prior")
        ax8b.axvline(theta8[np.argmax(post)], color=C_BIN, lw=1.6, label="MAP")
        ax8b.set_title(
            f"posterior mean ≈ {np.trapezoid(theta8 * post, theta8):.2f}",
            fontsize=FS_AX_TITLE - 2,
        )
        ax8b.legend(fontsize=FS_LEGEND - 1, loc="upper left")
        ax8b.set_xlim(-4, 4); ax8b.set_yticks([]); ax8b.grid(True, alpha=0.25)

    ax_x = fig8.add_axes([0.12, 0.08, 0.32, 0.04])
    ax_w = fig8.add_axes([0.52, 0.08, 0.32, 0.04])
    for _ax in (ax_x, ax_w):
        _ax.set_navigate(False)
        _ax.set_zorder(20)
    sl_x = Slider(ax_x, r"obs $x$", -3.0, 3.0, valinit=1.4, valstep=0.1, color=C_ACCENT)
    sl_w = Slider(ax_w, r"prior $\sigma$", 0.3, 2.5, valinit=1.2, valstep=0.1, color=C_FUN)

    def _u8(_=None):
        draw_cont(float(sl_x.val), float(sl_w.val))
        fig8.canvas.draw_idle()

    sl_x.on_changed(_u8)
    sl_w.on_changed(_u8)
    fig8._sliders = (sl_x, sl_w)
    _u8()
    print("Step 8: drag observation / prior width.")
    return fig8



# -------------------------------------------------------
# FIGURE 9: Takeaway
# -------------------------------------------------------
def build_step_9():
    fig9, axes9 = new_figure(1, 3)
    fig9.suptitle(
        "Step 9: Prior  x  Likelihood  /  Evidence  =  Posterior",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    panels = [
        (axes9[0], "Respect the base rate", "Rare H stays rare\nunless E is sharp.", C_WARN, C_ACCENT),
        (axes9[1], "Evidence renormalizes", r"$P(E)$ stops you" + "\nfrom overclaiming.", C_NEW, "#2874a6"),
        (axes9[2], "Beliefs must update", "Same formula for\ntests, filters, models.", C_GOAL, C_OK),
    ]
    for ax, title, body, face, edge in panels:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        round_box(ax, 0.5, 1.5, 9.0, 7.0, face, edge, lw=2.2)
        ax.text(5.0, 6.8, title, ha="center", fontsize=FS_AX_TITLE - 3,
                fontweight="bold", color=edge)
        ax.text(5.0, 4.2, body, ha="center", va="center", fontsize=FS_ANNOT + 1)
    return fig9




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
])

print("\n" + "=" * 60)
print("BAYES' THEOREM")
print("=" * 60)
print(r"""
P(H|E) = P(E|H) P(H) / P(E)

Prior:      belief before evidence
Likelihood: how the hypothesis predicts the evidence
Evidence:   marginal chance of the evidence
Posterior:  belief after evidence

Classic lesson:
  A sensitive test + tiny base rate can still yield a modest posterior.

Continuous form:
  posterior(theta) proportional to likelihood(data|theta) * prior(theta)
""")
