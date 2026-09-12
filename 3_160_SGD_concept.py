"""
Stochastic Gradient Descent (SGD) -- why minibatches, what convergence
we can expect, and how learning rate + batch size control stability.

Setup (finite-sum / empirical risk) -- REQUIRED for standard SGD:
    L(w) = (1/N) sum_i L_i(w)

Full-batch GD:
    w <- w - lr * grad L(w)          # exact but costs O(N) every SGD step

Minibatch SGD:
    pick a random batch B of size b
    w <- w - lr * grad L_B(w)        # noisy, costs O(b) every SGD step

Intuition this lecture makes visible:
  1) Why introduce minibatches (cost vs progress)
  2) SGD needs a finite-sum / average-of-examples loss
  3) How much "guarantee" we get (noise averages out; lr must be small enough)
  4) What hurts stability (too-large lr, too-small batches, sharp curvature)
  5) Why batch size matters (noise scale ~ 1/sqrt(b); same data budget, different paths)
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
import numpy as np

np.random.seed(42)
RNG = np.random.default_rng(42)


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


def add_note(ax, text, x=0.98, y=0.98, ha="right", va="top", facecolor="lightyellow"):
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=FS_ANNOT,
        ha=ha, va=va,
        bbox=dict(boxstyle="round", facecolor=facecolor, alpha=0.95),
    )


# -------------------------------------------------------
# Toy finite-sum loss in 2D (ravine): each example has its own curvature
#   L_i(x, y) = a_i x^2 + b_i y^2
#   L        = mean_i L_i
# Optimum at (0, 0). Minibatch grads are noisy estimates of the full grad.
# -------------------------------------------------------
N_DATA = 256
A_MEAN, B_MEAN = 4.0, 0.25
a_i = np.clip(RNG.normal(A_MEAN, 1.8, size=N_DATA), 0.5, None)
b_i = np.clip(RNG.normal(B_MEAN, 0.15, size=N_DATA), 0.05, None)
A_BAR, B_BAR = float(a_i.mean()), float(b_i.mean())


def loss_full(xy):
    x, y = float(xy[0]), float(xy[1])
    return A_BAR * x ** 2 + B_BAR * y ** 2


def grad_full(xy):
    x, y = float(xy[0]), float(xy[1])
    return np.array([2.0 * A_BAR * x, 2.0 * B_BAR * y], dtype=float)


def grad_batch(xy, idx):
    x, y = float(xy[0]), float(xy[1])
    return np.array(
        [2.0 * float(a_i[idx].mean()) * x, 2.0 * float(b_i[idx].mean()) * y],
        dtype=float,
    )


def run_gd(x0, lr, steps):
    w = np.array(x0, dtype=float)
    path = [w.copy()]
    losses = [loss_full(w)]
    for _ in range(steps):
        w = w - lr * grad_full(w)
        path.append(w.copy())
        losses.append(loss_full(w))
    return np.stack(path), np.array(losses)


def run_sgd(x0, lr, steps, batch_size, seed=0):
    rng = np.random.default_rng(seed)
    w = np.array(x0, dtype=float)
    path = [w.copy()]
    losses = [loss_full(w)]
    for _ in range(steps):
        idx = rng.choice(N_DATA, size=batch_size, replace=False)
        w = w - lr * grad_batch(w, idx)
        path.append(w.copy())
        losses.append(loss_full(w))
    return np.stack(path), np.array(losses)


X0 = np.array([1.15, 1.15], dtype=float)

# Contour grid for path plots
gx = np.linspace(-1.4, 1.4, 160)
gy = np.linspace(-1.4, 1.4, 160)
GX, GY = np.meshgrid(gx, gy)
GZ = A_BAR * GX ** 2 + B_BAR * GY ** 2


# -------------------------------------------------------
# FIGURE 1: Why introduce minibatches?
# -------------------------------------------------------
def build_step_1():
    fig1, axes1 = new_figure(1, 2)
    fig1.suptitle(
        "Step 1: Why minibatches?  Cheap noisy SGD steps vs one expensive exact SGD step",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )

    # Left: cost accounting cartoon
    ax = axes1[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Compute budget intuition", fontsize=FS_AX_TITLE)

    # One vertical column of 3 textboxes (shared left edge and width).
    boxes = [
        (0.25, 5.55, 9.50, 2.05, "#f8d7da",
         "Full-batch GD\n"
         "1 SGD step = touch all N examples\n"
         "exact direction, slow cadence"),
        (0.25, 3.30, 9.50, 2.05, "#d4edda",
         "Minibatch SGD\n"
         "1 SGD step = touch b << N examples\n"
         "noisy direction, fast cadence"),
        (0.25, 0.55, 9.50, 2.55, "#fff3cd",
         "Same wall-clock / data-touch budget:\n"
         "many cheap SGD steps often make MORE progress\n"
         "than a few exact full-batch steps --\n"
         "as long as the noise is not catastrophic."),
    ]
    for x, y, w, h, color, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="k", lw=2, zorder=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", zorder=2)

    # Right: progress per unit data-touch
    ax = axes1[1]
    # Compare: after seeing K example-gradients (K = steps * batch)
    budget = N_DATA * 8  # touch the dataset ~8 times
    configs = [
        ("full GD\n(b=N)", N_DATA, "crimson"),
        ("SGD b=16", 16, "royalblue"),
        ("SGD b=4", 4, "seagreen"),
    ]
    for label, b, color in configs:
        steps = budget // b
        if b == N_DATA:
            path, losses = run_gd(X0, lr=0.08, steps=steps)
        else:
            path, losses = run_sgd(X0, lr=0.08, steps=steps, batch_size=b, seed=1)
        # x-axis: examples seen
        xs = np.arange(len(losses)) * b
        ax.plot(xs, losses, color=color, lw=2.5, label=f"{label}  ({steps} steps)")

    ax.set_xlabel("examples touched (compute proxy)", fontsize=FS_AXIS)
    ax.set_ylabel(r"full-batch loss $L(w)$", fontsize=FS_AXIS)
    ax.set_title("Same data budget, different batch sizes", fontsize=FS_AX_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_yscale("log")
    ax.legend(fontsize=FS_LEGEND, loc="lower left")
    ax.grid(True, alpha=0.3, which="both")
    add_note(
        ax,
        "Minibatches are not just a hack:\n"
        "they buy MORE SGD steps\n"
        "per unit of data/compute.",
        x=0.98, y=0.55, va="top",
    )

    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    return fig1


# -------------------------------------------------------
# FIGURE 2: When SGD applies -- finite-sum / separable losses
# -------------------------------------------------------
# Yes: standard SGD is justified when full-batch loss is an AVERAGE (or sum)
# of per-example losses. Then grad L = average of per-example grads, so a
# random mini-batch is an unbiased estimate of grad L.
# The average is also a FAIR frequency weighting: common cases appear more often.
def build_step_2():
    fig2, axes2 = new_figure(1, 2)
    fig2.suptitle(
        r"Step 2: SGD needs $L$ to be an average of per-example losses",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )

    ax = axes2[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Finite-sum is fair -- and SGD-ready", fontsize=FS_AX_TITLE)
    notes_ok = [
        (8.2, "Empirical risk / finite-sum loss\n"
         r"$L(w)=\frac{1}{N}\sum_{i=1}^{N} \ell_i(w)$" + "\n"
         r"each $\ell_i$ uses ONE example $(x_i,y_i)$"),
        (5.7, "Why this is FAIR\n"
         "Repeated / similar cases appear more often in the sum,\n"
         "so they weigh more in $L$ -- matching how often they occur.\n"
         "A random mini-batch inherits the same frequencies."),
        (3.2, "Why SGD works here\n"
         r"$\nabla L=\frac{1}{N}\sum_i\nabla\ell_i$"
         "  =>  batch avg is an unbiased estimate of "
         r"$\nabla L$"),
        (1.0, r"SGD step:  $w\leftarrow w-\eta\cdot\frac{1}{|B|}\sum_{i\in B}\nabla\ell_i(w)$"),
    ]
    for y, text in notes_ok:
        ax.text(5.0, y, text, ha="center", va="center",
                fontsize=FS_ANNOT - 1, fontweight="bold")

    ax = axes2[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Counterexample: L is NOT a sum of per-example losses", fontsize=FS_AX_TITLE - 2)
    notes_bad = [
        (8.2, "Joint / coupled objective (toy)\n"
         r"$L(w)=\left(\sum_{i=1}^{N} f_w(x_i)\right)^2$" + "\n\n"
         "Cross terms $f_i f_j$ couple EVERY pair of examples.\n"
         r"$L$ cannot be written as $\sum_i \ell_i(w)$."),
        (5.5, "Full gradient needs the GLOBAL sum\n"
         r"$\nabla L = 2\left(\sum_k f_w(x_k)\right)"
         r"\sum_i \nabla f_w(x_i)$" + "\n\n"
         "A mini-batch replaces both sums by subset sums\n"
         "-- that is NOT an unbiased estimate of "
         r"$\nabla L$."),
        (2.85, "So classic mini-batch SGD does not apply:\n"
         "sampling a few $i$'s does not target this $L$.\n"
         "(Same issue for many global ranking / assignment losses.)"),
        (0.85, r"Still OK: mean vs sum, $\lambda\|w\|^2$, streaming "
         r"$L=\mathbb{E}_z[\ell(w;z)]$"),
    ]
    for y, text in notes_bad:
        ax.text(5.0, y, text, ha="center", va="center",
                fontsize=FS_ANNOT - 1, fontweight="bold")

    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    return fig2


# -------------------------------------------------------
# FIGURE 3: Gradient noise vs batch size
# -------------------------------------------------------
def build_step_3():
    fig3, axes3 = new_figure(1, 2)
    fig3.suptitle(
        r"Step 3: Minibatch gradient noise shrinks like $1/\sqrt{b}$",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )

    w_probe = np.array([0.9, 0.9], dtype=float)
    g_true = grad_full(w_probe)
    batch_sizes = [1, 4, 16, 64, N_DATA]
    n_samples = 200

    ax = axes3[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(batch_sizes)))
    # Larger batch size => thicker / more opaque markers.
    b_max = float(max(batch_sizes))
    for b, c in zip(batch_sizes, colors):
        gs = []
        for _ in range(n_samples):
            idx = RNG.choice(N_DATA, size=b, replace=False)
            gs.append(grad_batch(w_probe, idx))
        gs = np.stack(gs)
        weight = np.sqrt(b / b_max)  # 0..1, grows with b
        ms = 10 + 36 * weight
        alpha = 0.20 + 0.70 * weight
        ax.scatter(
            gs[:, 0], gs[:, 1], s=ms, alpha=alpha, color=c,
            linewidths=0.6 + 1.4 * weight, edgecolors=c, label=f"b={b}",
        )
    ax.scatter([g_true[0]], [g_true[1]], s=160, c="crimson", marker="*",
               zorder=5, edgecolor="k", label="true full grad")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel(r"$g_x$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$g_y$", fontsize=FS_AXIS)
    ax.set_title("Cloud of minibatch gradients at one $w$", fontsize=FS_AX_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    add_note(
        ax,
        "Small b: wild directions.\n"
        "Large b: cloud collapses\n"
        "onto the true gradient.",
        x=0.02, y=0.98, ha="left",
    )

    ax = axes3[1]
    stds = []
    for b in batch_sizes:
        gs = []
        for _ in range(n_samples):
            idx = RNG.choice(N_DATA, size=b, replace=False)
            gs.append(grad_batch(w_probe, idx))
        gs = np.stack(gs)
        stds.append(float(np.linalg.norm(gs.std(axis=0))))
    ax.loglog(batch_sizes, stds, "o-", color="royalblue", lw=2.5, ms=9, label="empirical grad noise")
    # reference slope 1/sqrt(b)
    ref = stds[0] * np.sqrt(batch_sizes[0] / np.array(batch_sizes, dtype=float))
    ax.loglog(batch_sizes, ref, "--", color="crimson", lw=2.0, label=r"$\propto 1/\sqrt{b}$")
    ax.set_xlabel("batch size b", fontsize=FS_AXIS)
    ax.set_ylabel(r"stddev of minibatch grad (proxy)", fontsize=FS_AXIS)
    ax.set_title("Noise scale vs batch size", fontsize=FS_AX_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, alpha=0.3, which="both")
    add_note(
        ax,
        "Rule of thumb:\n"
        "4x larger batch\n"
        "~ 2x less gradient noise.\n\n"
        "(Independent samples.)",
        x=0.02, y=0.50, ha="left", va="center",
    )

    fig3.tight_layout(rect=[0, 0, 1, 0.94])
    return fig3


# -------------------------------------------------------
# FIGURE 4: Paths -- full GD vs SGD for different batch sizes
# -------------------------------------------------------
def build_step_4():
    fig4, axes4 = new_figure(1, 2)
    fig4.suptitle(
        "Step 4: Same learning rate -- smaller batches jitter more",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )

    LR_PATH = 0.12
    STEPS_PATH = 80
    path_full, loss_full_path = run_gd(X0, LR_PATH, STEPS_PATH)
    path_b4, loss_b4 = run_sgd(X0, LR_PATH, STEPS_PATH, batch_size=4, seed=2)
    path_b32, loss_b32 = run_sgd(X0, LR_PATH, STEPS_PATH, batch_size=32, seed=2)

    ax = axes4[0]
    ax.contour(GX, GY, GZ, levels=18, cmap="Greys", linewidths=0.8)
    ax.plot(path_full[:, 0], path_full[:, 1], "-", color="k", lw=2.6, label="full GD")
    ax.plot(path_b32[:, 0], path_b32[:, 1], "-", color="royalblue", lw=2.2, label="SGD b=32")
    ax.plot(path_b4[:, 0], path_b4[:, 1], "-", color="crimson", lw=1.8, alpha=0.9, label="SGD b=4")
    ax.scatter([X0[0]], [X0[1]], c="orange", s=70, zorder=5, label="start")
    ax.scatter([0], [0], c="k", s=110, marker="*", zorder=5, label="optimum")
    ax.set_xlabel("x", fontsize=FS_AXIS)
    ax.set_ylabel("y", fontsize=FS_AXIS)
    ax.set_title("Optimization paths on the ravine", fontsize=FS_AX_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc="center left")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    add_note(
        ax,
        "Noise is not always bad:\n"
        "it can help escape sharp\n"
        "bad valleys in deep nets.\n"
        "Here the bowl is convex --\n"
        "noise mostly just jitters.",
        x=0.98, y=0.50, ha="right", va="center",
    )

    ax = axes4[1]
    ax.plot(loss_full_path, color="k", lw=2.6, label="full GD")
    ax.plot(loss_b32, color="royalblue", lw=2.2, label="SGD b=32")
    ax.plot(loss_b4, color="crimson", lw=2.0, label="SGD b=4")
    ax.set_xlabel("SGD step", fontsize=FS_AXIS)
    ax.set_ylabel(r"full-batch loss $L(w)$", fontsize=FS_AXIS)
    ax.set_title("Loss along the path", fontsize=FS_AX_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_yscale("log")
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, alpha=0.3, which="both")
    add_note(
        ax,
        "With fixed lr:\n"
        "too-small batches keep\n"
        "rattling near the optimum\n"
        "(floor set by noise).",
        x=0.98, y=0.55, va="top",
    )

    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    return fig4


# -------------------------------------------------------
# FIGURE 5: Convergence "guarantee" intuition -- lr must be small enough
# -------------------------------------------------------
def build_step_5():
    fig5, axes5 = new_figure(1, 2)
    fig5.suptitle(
        "Step 5: Convergence intuition -- SGD works when the SGD step is not too bold",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )

    # Left: loss curves for several learning rates (SGD b=8)
    ax = axes5[0]
    lrs = [0.03, 0.10, 0.25, 0.55]
    colors_lr = ["seagreen", "royalblue", "darkorange", "crimson"]
    for lr, c in zip(lrs, colors_lr):
        _, losses = run_sgd(X0, lr=lr, steps=120, batch_size=8, seed=3)
        ax.plot(losses, color=c, lw=2.3, label=rf"lr={lr:g}")
    ax.set_xlabel("SGD step", fontsize=FS_AXIS)
    ax.set_ylabel(r"$L(w)$", fontsize=FS_AXIS)
    ax.set_title("SGD (b=8): learning rate scan", fontsize=FS_AX_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.set_yscale("log")
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, alpha=0.3, which="both")
    add_note(
        ax,
        "Small lr: steady descent.\n"
        "Large lr: bounce / diverge.\n\n"
        "Noise makes the safe lr\n"
        "region SMALLER than for\n"
        "full-batch GD.",
        x=0.02, y=0.50, ha="left", va="center",
    )

    # Right: expected decrease cartoon / empirical success vs lr
    ax = axes5[1]
    lr_grid = np.geomspace(0.02, 0.8, 14)
    success = []
    final_med = []
    for lr in lr_grid:
        finals = []
        for s in range(12):
            _, losses = run_sgd(X0, lr=float(lr), steps=100, batch_size=8, seed=10 + s)
            finals.append(losses[-1])
        finals = np.array(finals)
        final_med.append(float(np.median(finals)))
        success.append(float(np.mean(finals < 0.05)))

    ax.plot(lr_grid, success, "o-", color="seagreen", lw=2.5, label="P(final L < 0.05)")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("learning rate", fontsize=FS_AXIS)
    ax.set_ylabel("success rate over random seeds", fontsize=FS_AXIS)
    ax.set_title("Empirical reliability vs lr", fontsize=FS_AX_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=FS_LEGEND, loc="lower left")

    ax2 = ax.twinx()
    ax2.plot(lr_grid, final_med, "s--", color="crimson", lw=2.0, label="median final L")
    ax2.set_yscale("log")
    ax2.set_ylabel("median final loss", fontsize=FS_AXIS, color="crimson")
    ax2.tick_params(axis="y", labelsize=FS_TICK, colors="crimson")

    add_note(
        ax,
        "What theory roughly says:\n"
        "for smooth convex L, small\n"
        "enough lr => E[L] trends down.\n"
        "Too large lr voids that.\n\n"
        "SGD does NOT promise\n"
        "monotone decrease every SGD step.",
        x=0.02, y=0.50, ha="left", va="center", facecolor="white",
    )

    fig5.tight_layout(rect=[0, 0, 1, 0.94])
    return fig5


# -------------------------------------------------------
# FIGURE 6: Stability depends on lr, batch size, and curvature
# -------------------------------------------------------
def build_step_6():
    fig6, axes6 = new_figure(1, 2)
    fig6.suptitle(
        "Step 6: Stability -- learning rate, batch size, and sharpness interact",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )

    # Left: heatmap of final loss over (lr, batch)
    ax = axes6[0]
    lr_h = np.array([0.05, 0.1, 0.2, 0.35, 0.5])
    b_h = np.array([1, 4, 16, 64, N_DATA])
    heat = np.zeros((len(b_h), len(lr_h)))
    for i, b in enumerate(b_h):
        for j, lr in enumerate(lr_h):
            finals = []
            for s in range(8):
                if b >= N_DATA:
                    _, losses = run_gd(X0, lr=float(lr), steps=60)
                else:
                    _, losses = run_sgd(X0, lr=float(lr), steps=60, batch_size=int(b), seed=20 + s)
                finals.append(losses[-1])
            heat[i, j] = float(np.median(finals))

    im = ax.imshow(np.log10(heat + 1e-12), origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks(range(len(lr_h)))
    ax.set_xticklabels([f"{lr:g}" for lr in lr_h])
    ax.set_yticks(range(len(b_h)))
    ax.set_yticklabels([str(b) for b in b_h])
    ax.set_xlabel("learning rate", fontsize=FS_AXIS)
    ax.set_ylabel("batch size", fontsize=FS_AXIS)
    ax.set_title(r"median final $\log_{10} L$  (darker = worse)", fontsize=FS_AX_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    cbar = fig6.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FS_TICK)
    cbar.set_label(r"$\log_{10}$ final loss", fontsize=FS_ANNOT)
    add_note(
        ax,
        "Unsafe corner:\n"
        "large lr + tiny batch.\n\n"
        "Safer corner:\n"
        "moderate lr, larger batch\n"
        "(or smaller lr).",
        x=0.02, y=0.98, ha="left",
    )

    # Right: curvature / sharpness notes (no boxes)
    ax = axes6[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("What else affects stability?", fontsize=FS_AX_TITLE)
    rows = [
        (6.2, "Sharp curvature (large Hessian)\n"
         "Same lr overshoots more -- like a stiff spring."),
        (4.0, "Gradient noise (small batch)\n"
         "Pushes you sideways each SGD step; large lr amplifies that."),
        (1.7, "Practical knobs\n"
         "decrease lr, increase batch, add momentum (0530_momentum.py),\n"
         "or normalize/scale features (0720_normalize_activations.py)."),
    ]
    for y, text in rows:
        ax.text(5.0, y, text, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold")

    fig6.tight_layout(rect=[0, 0, 1, 0.94])
    return fig6


# -------------------------------------------------------
# FIGURE 7: Why batch size is important -- bias/variance of the gradient update
# -------------------------------------------------------
def build_step_7():
    fig7, axes7 = new_figure(1, 2)
    fig7.suptitle(
        "Step 7: Batch size sets the noise floor and the useful SGD step size",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )

    # Left: late-mini-batch loss floor vs batch size (fixed lr)
    ax = axes7[0]
    bs = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    floors = []
    for b in bs:
        finals = []
        for s in range(10):
            _, losses = run_sgd(X0, lr=0.15, steps=150, batch_size=int(b), seed=40 + s)
            finals.append(np.mean(losses[-20:]))  # average late loss
        floors.append(float(np.median(finals)))
    ax.plot(bs, floors, "o-", color="crimson", lw=2.5, ms=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size b", fontsize=FS_AXIS)
    ax.set_ylabel("late-mini-batch loss floor", fontsize=FS_AXIS)
    ax.set_title(r"Fixed lr=0.15: larger $b$ => lower noise floor", fontsize=FS_AX_TITLE)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, alpha=0.3, which="both")
    add_note(
        ax,
        "With constant lr, SGD\n"
        "does not sit still at 0:\n"
        "it wanders in a ball whose\n"
        "radius grows with noise\n"
        "(shrinks as b grows).",
        x=0.98, y=0.98,
    )

    # Right: steps-to-target vs batch size under equal-example budget framing
    ax = axes7[1]
    target = 0.02
    # For each batch size, how many SGD steps until median loss < target (cap steps)
    steps_needed = []
    for b in bs:
        need = []
        for s in range(10):
            _, losses = run_sgd(X0, lr=0.12, steps=200, batch_size=int(b), seed=60 + s)
            hit = np.where(losses < target)[0]
            need.append(int(hit[0]) if len(hit) else 200)
        steps_needed.append(float(np.median(need)))
    examples_needed = np.array(steps_needed) * bs

    ax.plot(bs, steps_needed, "o-", color="royalblue", lw=2.5, label="steps to target")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size b", fontsize=FS_AXIS)
    ax.set_ylabel("SGD steps until L < 0.02", fontsize=FS_AXIS, color="royalblue")
    ax.tick_params(axis="y", labelsize=FS_TICK, colors="royalblue")
    ax.set_title("Speed: steps vs examples touched", fontsize=FS_AX_TITLE)
    ax.grid(True, alpha=0.3, which="both")

    ax2 = ax.twinx()
    ax2.plot(bs, examples_needed, "s--", color="darkorange", lw=2.3, label="examples touched")
    ax2.set_ylabel("examples touched (= steps x b)", fontsize=FS_AXIS, color="darkorange")
    ax2.tick_params(axis="y", labelsize=FS_TICK, colors="darkorange")

    add_note(
        ax,
        "Tiny batches: more SGD steps\n"
        "but each is cheap -- and\n"
        "sometimes MORE total data\n"
        "touches because of noise.\n\n"
        "Huge batches: fewer SGD steps,\n"
        "each expensive; closer to GD.",
        x=0.02, y=0.12, ha="left", va="bottom", facecolor="white",
    )

    fig7.tight_layout(rect=[0, 0, 1, 0.94])
    return fig7


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
print("STOCHASTIC GRADIENT DESCENT")
print("=" * 60)
print(f"""
Finite-sum objective on N={N_DATA} ravine examples:
  L(w) = mean_i (a_i x^2 + b_i y^2),   optimum at 0.

0) When SGD applies (important assumption)
   Standard SGD needs L(w) = (1/N) sum_i ell_i(w)  [or E_z ell(w;z)].
   That average is FAIR: similar cases that occur more often weigh more,
   and random mini-batches inherit the same frequencies.
   Then grad L is the average of per-example grads, so a mini-batch
   is an unbiased estimate of grad L.
   Counterexample: L(w) = (sum_i f_w(x_i))^2 couples all examples via
   cross terms -- a subset gradient does NOT estimate grad L, so classic
   mini-batch SGD does not apply.

1) Why minibatches?
   Full-batch GD is exact but spends O(N) before every gradient update.
   SGD spends O(b), so you can take many more SGD steps per unit compute.
   Often those extra noisy SGD steps win on a fixed data-touch budget.

2) What "convergence" means here
   With a small enough learning rate, the EXPECTED loss trends downward
   for smooth objectives -- but individual SGD steps can go up.
   SGD typically converges to a NOISE BALL around the optimum when lr
   is held constant; decaying lr (or larger batches) shrinks that ball.

3) Stability knobs
   - learning rate too large  -> bounce / diverge (worse under noise)
   - batch size too small     -> huge gradient variance
   - sharp curvature          -> same lr overshoots more
   Momentum (0530_momentum.py) helps average out noisy pushes.

4) Why batch size matters
   Gradient noise scales ~ 1/sqrt(b).
   Larger b: smoother path, lower late-training floor (at fixed lr),
   but fewer SGD steps per unit compute and less "useful noise".
   In practice: pick b for hardware efficiency, then tune lr
   (and often scale lr when you scale b).

Demo tip: re-run with different seeds -- the qualitative picture stays,
the jitter changes.
""")
