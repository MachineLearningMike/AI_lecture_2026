"""
Adam / AdamW and learning-rate schedules -- what students actually meet
after plain SGD + momentum.

Sketch:
  1. Why adaptive steps: steep vs shallow coordinates
  2. Adam -- moments of the gradient
  3. AdamW -- decouple weight decay from the adaptive step
  4. Learning-rate schedules (constant, step, cosine, warmup)
  5. Same optimizer, different schedules -- loss paths
  6. PyTorch-facing recipe + when to use what
  7. Takeaway for Ch08
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import RadioButtons, Slider
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

np.random.seed(7)
RNG = np.random.default_rng(7)


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
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_GOAL = "#d4edda"
C_MUTE = "#f4f6f7"
C_ACCENT = "#c0392b"
C_MOM = "#2874a6"
C_ADAM = "#8e44ad"
C_ADAMW = "#d68910"


# -------------------------------------------------------
# Ill-conditioned ravine: needs different step sizes per axis
#   L(x,y) = 0.5 * (a x^2 + b y^2)
# -------------------------------------------------------
A_CURV, B_CURV = 40.0, 0.5


def loss_xy(xy):
    x, y = float(xy[0]), float(xy[1])
    return 0.5 * (A_CURV * x * x + B_CURV * y * y)


def grad_xy(xy):
    x, y = float(xy[0]), float(xy[1])
    return np.array([A_CURV * x, B_CURV * y], dtype=float)


def run_momentum(x0, lr, beta, steps):
    x = np.array(x0, dtype=float)
    v = np.zeros(2)
    path = [x.copy()]
    for _ in range(steps):
        g = grad_xy(x)
        v = beta * v + g
        x = x - lr * v
        path.append(x.copy())
    return np.stack(path)


def run_adam(x0, lr, steps, beta1=0.9, beta2=0.999, eps=1e-8):
    x = np.array(x0, dtype=float)
    m = np.zeros(2)
    v = np.zeros(2)
    path = [x.copy()]
    for t in range(1, steps + 1):
        g = grad_xy(x)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        mhat = m / (1.0 - beta1 ** t)
        vhat = v / (1.0 - beta2 ** t)
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        path.append(x.copy())
    return np.stack(path)


def run_adamw(x0, lr, steps, wd=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
    """AdamW: adaptive step on grad, then decoupled decay toward 0."""
    x = np.array(x0, dtype=float)
    m = np.zeros(2)
    v = np.zeros(2)
    path = [x.copy()]
    for t in range(1, steps + 1):
        g = grad_xy(x)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        mhat = m / (1.0 - beta1 ** t)
        vhat = v / (1.0 - beta2 ** t)
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        x = x - lr * wd * x  # decoupled weight decay
        path.append(x.copy())
    return np.stack(path)


X0 = np.array([1.0, 1.0])
STEPS = 60


# -------------------------------------------------------
# FIGURE 1: Why adaptive steps
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        "Step 1: Steep vs shallow coordinates -- one LR is a bad compromise",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    gx = np.linspace(-1.2, 1.2, 220)
    gy = np.linspace(-1.2, 1.2, 220)
    GX, GY = np.meshgrid(gx, gy)
    GZ = 0.5 * (A_CURV * GX ** 2 + B_CURV * GY ** 2)
    cs = ax1.contour(GX, GY, GZ, levels=18, cmap="terrain", linewidths=1.0)
    ax1.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

    path_mom = run_momentum(X0, lr=0.03, beta=0.9, steps=STEPS)
    path_adam = run_adam(X0, lr=0.05, steps=STEPS)
    ax1.plot(path_mom[:, 0], path_mom[:, 1], "-", color=C_MOM, lw=2.0,
             label="Momentum SGD (one lr)")
    ax1.plot(path_adam[:, 0], path_adam[:, 1], "-", color=C_ADAM, lw=2.2,
             label="Adam (adaptive per axis)")
    ax1.scatter([0], [0], c="k", s=90, marker="*", zorder=5, label="minimum")
    ax1.scatter([X0[0]], [X0[1]], c="orange", s=70, zorder=5, label="start")
    ax1.set_xlabel(r"$x$ (steep: curv $\approx 40$)", fontsize=FS_AXIS)
    ax1.set_ylabel(r"$y$ (shallow: curv $\approx 0.5$)", fontsize=FS_AXIS)
    ax1.set_aspect("equal")
    ax1.tick_params(labelsize=FS_TICK)
    ax1.legend(fontsize=FS_LEGEND, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.text(
        0.02, 0.02,
        "Momentum uses the same lr on every coordinate.\n"
        "Adam scales each axis by recent gradient energy.",
        transform=ax1.transAxes, ha="left", va="bottom", fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.94])
    return fig1


# -------------------------------------------------------
# FIGURE 2: Adam equations
# -------------------------------------------------------
def build_step_2():
    fig2, ax2 = new_figure()
    fig2.suptitle(
        "Step 2: Adam -- first and second moments of the gradient",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax2)

    round_box(ax2, 0.4, 5.5, 11.2, 2.1, C_DARK, C_DARK, lw=0)
    ax2.text(
        6.0, 6.55,
        "Adam ≈ momentum on the gradient  +  adaptive denominator per parameter",
        ha="center", va="center", fontsize=FS_ANNOT + 2,
        fontweight="bold", color="#f9e79f",
    )

    boxes = [
        (0.4, 2.7, 3.6, 2.5, C_NEW, C_MOM,
         r"$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$",
         "1st moment\n(smoothed direction)"),
        (4.2, 2.7, 3.6, 2.5, "#f5eef8", C_ADAM,
         r"$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$",
         "2nd moment\n(energy / scale)"),
        (8.0, 2.7, 3.6, 2.5, C_GOAL, C_OK,
         r"$x \leftarrow x - \eta\,\hat m / (\sqrt{\hat v}+\epsilon)$",
         "Adaptive step\n(bias-corrected)"),
    ]
    for x, y, w, h, face, edge, formula, body in boxes:
        round_box(ax2, x, y, w, h, face, edge, lw=2.0)
        ax2.text(x + w / 2, y + 1.7, formula, ha="center", va="center",
                 fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        ax2.text(x + w / 2, y + 0.7, body, ha="center", va="center",
                 fontsize=FS_ANNOT)

    ax2.text(
        6.0, 1.4,
        r"Defaults students will see:  $\beta_1=0.9$,  $\beta_2=0.999$,  $\epsilon=10^{-8}$."
        "\n"
        r"Bias correction:  $\hat m = m/(1-\beta_1^t)$,  $\hat v = v/(1-\beta_2^t)$.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
                  edgecolor="0.55", alpha=0.95),
    )
    ax2.text(
        6.0, 0.45,
        "Intuition: big recent |g| → larger v → smaller effective step on that coordinate.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    return fig2


# -------------------------------------------------------
# FIGURE 3: Adam vs AdamW
# -------------------------------------------------------
def build_step_3():
    fig3, (ax3a, ax3b) = new_figure(1, 2)
    fig3.suptitle(
        "Step 3: AdamW decouples weight decay from the adaptive update",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax3a.set_xlim(0, 10)
    ax3a.set_ylim(0, 10)
    ax3a.axis("off")
    ax3a.set_title("Adam + L2 in the loss (coupled)", fontsize=FS_AX_TITLE - 2,
                   color=C_ADAM, fontweight="bold")
    round_box(ax3a, 0.4, 1.0, 9.2, 8.0, "#f5eef8", C_ADAM, lw=2.2)
    ax3a.text(
        5.0, 7.2,
        "People often add  (wd/2)||x||^2  to the loss.\n"
        "Then the gradient already contains  wd · x.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )
    ax3a.text(
        5.0, 5.0,
        r"$g \leftarrow \nabla L + \lambda x$"
        "\n\n"
        r"$x \leftarrow x - \eta\,\hat m(g)/(\sqrt{\hat v}+\epsilon)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        color=C_DARK,
    )
    ax3a.text(
        5.0, 2.2,
        "Problem: decay is filtered by the\n"
        "adaptive denominator. Effective\n"
        "regularization depends on v -- messy.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_ACCENT,
    )

    ax3b.set_xlim(0, 10)
    ax3b.set_ylim(0, 10)
    ax3b.axis("off")
    ax3b.set_title("AdamW (decoupled decay)", fontsize=FS_AX_TITLE - 2,
                   color=C_ADAMW, fontweight="bold")
    round_box(ax3b, 0.4, 1.0, 9.2, 8.0, "#fef5e7", C_ADAMW, lw=2.2)
    ax3b.text(
        5.0, 7.2,
        "Adaptive step uses the true loss grad only.\n"
        "Then shrink weights toward 0 separately.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )
    ax3b.text(
        5.0, 5.0,
        r"$g \leftarrow \nabla L$"
        "\n\n"
        r"$x \leftarrow x - \eta\,\hat m/(\sqrt{\hat v}+\epsilon)$"
        "\n"
        r"$x \leftarrow x - \eta\,\lambda x$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, fontweight="bold",
        color=C_DARK,
    )
    ax3b.text(
        5.0, 2.2,
        "Decay strength is stable and\n"
        "interpretable. This is the default\n"
        "students will see in modern recipes.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_OK,
    )
    fig3.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig3


# -------------------------------------------------------
# FIGURE 4: Schedule shapes
# -------------------------------------------------------
def lr_constant(t, T, peak=1.0):
    return np.full_like(t, peak, dtype=float)


def lr_step(t, T, peak=1.0, drops=(0.4, 0.7), gamma=0.2):
    lr = np.full_like(t, peak, dtype=float)
    for d in drops:
        lr = np.where(t >= d * T, lr * gamma, lr)
    return lr


def lr_cosine(t, T, peak=1.0, floor=0.05):
    return floor + 0.5 * (peak - floor) * (1.0 + np.cos(np.pi * t / T))


def lr_warmup_cosine(t, T, peak=1.0, floor=0.05, warm=0.1):
    tw = max(int(warm * T), 1)
    lr = np.empty_like(t, dtype=float)
    for i, ti in enumerate(t):
        if ti < tw:
            lr[i] = peak * (ti + 1) / tw
        else:
            u = (ti - tw) / max(T - tw, 1)
            lr[i] = floor + 0.5 * (peak - floor) * (1.0 + np.cos(np.pi * u))
    return lr


T = 100
t = np.arange(T)
def build_step_4():
    fig4, ax4 = new_figure()
    fig4.suptitle(
        "Step 4: Learning-rate schedules -- the same optimizer, different fuel curves",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    ax4.plot(t, lr_constant(t, T), lw=2.6, label="constant", color="0.35")
    ax4.plot(t, lr_step(t, T), lw=2.6, label="step decay", color=C_ACCENT)
    ax4.plot(t, lr_cosine(t, T), lw=2.6, label="cosine", color=C_MOM)
    ax4.plot(t, lr_warmup_cosine(t, T), lw=2.8, label="warmup + cosine", color=C_ADAM)
    ax4.set_xlabel("training step / epoch index", fontsize=FS_AXIS)
    ax4.set_ylabel(r"learning rate $\eta_t$ (relative)", fontsize=FS_AXIS)
    ax4.tick_params(labelsize=FS_TICK)
    ax4.legend(fontsize=FS_LEGEND, loc="upper right")
    ax4.grid(True, alpha=0.3)
    ax4.text(
        0.02, 0.02,
        "Warmup: start small so early Adam moments are not wild.\n"
        "Cosine: smooth landing -- common in Transformer / LLM recipes.",
        transform=ax4.transAxes, ha="left", va="bottom", fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )
    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    return fig4


# -------------------------------------------------------
# FIGURE 5: Interactive -- schedule choice on a 1D bowl
# -------------------------------------------------------
# Minimize f(w)=0.5*(w-2)^2 with gradient g=w-2, using Adam-like scalar steps
# but with scheduled lr so students see schedule effects cleanly.


def adam_1d(w0, lrs, beta1=0.9, beta2=0.999, eps=1e-8):
    w = float(w0)
    m = 0.0
    v = 0.0
    path = [w]
    for t_i, lr in enumerate(lrs, start=1):
        g = w - 2.0
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        mhat = m / (1.0 - beta1 ** t_i)
        vhat = v / (1.0 - beta2 ** t_i)
        w = w - float(lr) * mhat / (np.sqrt(vhat) + eps)
        path.append(w)
    return np.array(path)


def build_step_5():
    fig5, (ax5a, ax5b) = new_figure(1, 2)
    fig5.suptitle(
        "Step 5: Interactive -- pick a schedule; watch Adam on a 1D bowl",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    fig5.subplots_adjust(bottom=0.22, wspace=0.28)

    SCHEDULES = {
        "constant": lambda T, peak: lr_constant(np.arange(T), T, peak),
        "step": lambda T, peak: lr_step(np.arange(T), T, peak),
        "cosine": lambda T, peak: lr_cosine(np.arange(T), T, peak),
        "warmup+cosine": lambda T, peak: lr_warmup_cosine(np.arange(T), T, peak),
    }
    sched_names = list(SCHEDULES.keys())
    state = {"name": "warmup+cosine", "peak": 0.35, "T": 40}


    def _draw_sched(_=None):
        ax5a.clear()
        ax5b.clear()
        name = state["name"]
        peak = state["peak"]
        Tloc = int(state["T"])
        lrs = SCHEDULES[name](Tloc, peak)
        path = adam_1d(w0=-1.5, lrs=lrs)
        losses = 0.5 * (path - 2.0) ** 2

        ax5a.plot(np.arange(Tloc), lrs, color=C_ADAM, lw=2.6)
        ax5a.set_title(f"schedule: {name}", fontsize=FS_AX_TITLE - 2)
        ax5a.set_xlabel("step", fontsize=FS_AXIS)
        ax5a.set_ylabel(r"$\eta_t$", fontsize=FS_AXIS)
        ax5a.tick_params(labelsize=FS_TICK)
        ax5a.grid(True, alpha=0.3)

        ax5b.plot(losses, color=C_OK, lw=2.6, label="loss")
        ax5b.plot(path, color=C_MOM, lw=1.8, alpha=0.85, label="w")
        ax5b.axhline(2.0, color="0.4", ls="--", lw=1.2, label="target w=2")
        ax5b.set_title("Adam trajectory under that schedule", fontsize=FS_AX_TITLE - 2)
        ax5b.set_xlabel("step", fontsize=FS_AXIS)
        ax5b.set_ylabel("value", fontsize=FS_AXIS)
        ax5b.tick_params(labelsize=FS_TICK)
        ax5b.legend(fontsize=FS_LEGEND - 1, loc="upper right")
        ax5b.grid(True, alpha=0.3)
        fig5.canvas.draw_idle()


    ax_radio = fig5.add_axes([0.12, 0.025, 0.22, 0.095])
    radio = RadioButtons(ax_radio, sched_names, active=sched_names.index("warmup+cosine"))
    for _lab in radio.labels:
        _lab.set_fontsize(8)
    ax_peak = fig5.add_axes([0.50, 0.08, 0.18, 0.04])
    ax_T = fig5.add_axes([0.75, 0.08, 0.18, 0.04])
    sl_peak = Slider(ax_peak, "peak lr", 0.05, 0.8, valinit=0.35, valstep=0.01, color=C_ADAM)
    sl_T = Slider(ax_T, "steps", 10, 80, valinit=40, valstep=1, color=C_MOM)


    def _on_radio(label):
        state["name"] = label
        _draw_sched()


    def _on_peak(val):
        state["peak"] = float(val)
        _draw_sched()


    def _on_T(val):
        state["T"] = int(val)
        _draw_sched()


    radio.on_clicked(_on_radio)
    sl_peak.on_changed(_on_peak)
    sl_T.on_changed(_on_T)
    _draw_sched()
    print("Step 5: toggle schedule / peak lr / steps.")
    return fig5


# -------------------------------------------------------
# FIGURE 6: Practical recipe
# -------------------------------------------------------
def build_step_6():
    fig6, ax6 = new_figure()
    fig6.suptitle(
        "Step 6: What you will type -- and when",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax6)

    round_box(ax6, 0.4, 5.2, 11.2, 2.4, C_DARK, C_DARK, lw=0)
    ax6.text(
        6.0, 6.4,
        "torch.optim.AdamW(params, lr=..., weight_decay=...)\n"
        "+ a schedule (LambdaLR / CosineAnnealingLR / custom warmup)",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.4, 1.5, 3.6, 3.3, C_NEW, C_MOM, "Momentum SGD",
         "Baselines, simple vision.\n"
         "Need careful lr + tuning.\n"
         "Good teaching default."),
        (4.2, 1.5, 3.6, 3.3, "#f5eef8", C_ADAM, "Adam",
         "Fast early progress.\n"
         "Prefer AdamW when you\n"
         "want real weight decay."),
        (8.0, 1.5, 3.6, 3.3, "#fef5e7", C_ADAMW, "AdamW + schedule",
         "Transformers / LLMs.\n"
         "Warmup + cosine is the\n"
         "boring modern default."),
    ]
    for x, y, w, h, face, edge, title, body in cards:
        round_box(ax6, x, y, w, h, face, edge, lw=2.2)
        ax6.text(x + w / 2, y + h - 0.55, title, ha="center",
                 fontsize=FS_ANNOT + 1, fontweight="bold", color=edge)
        ax6.text(x + w / 2, y + 1.35, body, ha="center", va="center",
                 fontsize=FS_ANNOT)

    ax6.text(
        6.0, 0.7,
        "Schedule is not decoration: too-large early lr breaks Adam; no decay can stall late training.",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    return fig6


# -------------------------------------------------------
# FIGURE 7: Takeaway
# -------------------------------------------------------
def build_step_7():
    fig7, ax7 = new_figure()
    fig7.suptitle(
        "Step 7: Ch08 takeaway -- after SGD + momentum",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax7)

    points = [
        (0.5, 5.5, "1. Adaptive", "Adam rescales each parameter by recent gradient energy."),
        (0.5, 3.9, "2. Decouple", "AdamW applies weight decay outside the adaptive fraction."),
        (0.5, 2.3, "3. Schedule", "Warmup + cosine (or step) is part of the optimizer story."),
        (0.5, 0.7, "4. Practice", "For Transformers/LLMs, start from AdamW + warmup cosine, then ablate."),
    ]
    for x, y, title, body in points:
        round_box(ax7, x, y, 11.0, 1.35, C_GOAL if "Practice" not in title else C_NEW,
                  C_OK if "Practice" not in title else C_MOM, lw=1.8)
        ax7.text(1.0, y + 0.7, title, ha="left", va="center",
                 fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK)
        ax7.text(4.2, y + 0.7, body, ha="left", va="center", fontsize=FS_ANNOT)
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

print("\n" + "=" * 60)
print("ADAM / ADAMW + LR SCHEDULES")
print("=" * 60)
print("""
Adam: momentum-like m  +  energy v  ->  per-parameter step sizes.
AdamW: keep adaptive step on grad; decay weights separately (decoupled).
Schedules: constant / step / cosine / warmup+cosine.
Modern default many students will meet: AdamW + warmup + cosine.
""")
