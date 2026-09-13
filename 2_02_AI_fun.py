"""
The fun of learning AI -- visual playgrounds, weird data, interactive toys.

Sketch:
  1. Four toy worlds at a glance
  2. Same loop as pictures: data -> guess -> score -> update
  3. Task gallery as mini-plots
  4. Drag a line: cookies vs stars
  5. Drag a line: interlocking moons (linear stalls)
  6. Drag a circle: ring task
  7. XOR playground -- twist the cut
  8. RL fun -- gridworld treasure hunt (Q-learning)
  9. RL fun -- multi-armed bandit
  10. Watch supervised training: boundary + loss
  11. Visual takeaway collage
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
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

np.random.seed(41)
RNG = np.random.default_rng(41)


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
CMAP_BG = ListedColormap(["#eaf2f8", "#d5f5e3"])


def make_blobs(n=50):
    a = RNG.normal([-1.1, 0.15], 0.5, size=(n, 2))
    b = RNG.normal([1.15, -0.1], 0.5, size=(n, 2))
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


def make_moons(n=70):
    t = np.linspace(0, np.pi, n)
    a = np.column_stack([np.cos(t), np.sin(t)]) + RNG.normal(0, 0.07, (n, 2))
    b = np.column_stack([1.0 - np.cos(t), 0.55 - np.sin(t)]) + RNG.normal(0, 0.07, (n, 2))
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


def make_rings(n=80):
    th = RNG.uniform(0, 2 * np.pi, n)
    r0 = 0.55 + RNG.normal(0, 0.06, n)
    r1 = 1.35 + RNG.normal(0, 0.07, n)
    a = np.column_stack([r0 * np.cos(th), r0 * np.sin(th)])
    b = np.column_stack([r1 * np.cos(th), r1 * np.sin(th)])
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


def make_spiral(n=90):
    t = np.linspace(0.4, 3.4 * np.pi, n)
    r = 0.12 * t
    a = np.column_stack([r * np.cos(t), r * np.sin(t)]) + RNG.normal(0, 0.04, (n, 2))
    b = np.column_stack([r * np.cos(t + np.pi), r * np.sin(t + np.pi)]) + RNG.normal(0, 0.04, (n, 2))
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


def paint_points(ax, X, y, s=42):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=s, c=C_GRAY,
               edgecolors="white", linewidths=0.6, zorder=3, label="A")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=s, c=C_OK,
               edgecolors="white", linewidths=0.6, zorder=3, label="B")


def accuracy_bar(ax, acc, title="accuracy"):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.barh([0.45], [acc], height=0.28, left=0, color=C_OK if acc >= 0.85 else C_ACCENT,
            zorder=2)
    ax.barh([0.45], [1.0], height=0.28, left=0, color="#d5d8dc", zorder=1)
    ax.text(0.5, 0.82, title, ha="center", va="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
    ax.text(0.5, 0.18, f"{100 * acc:.0f}%", ha="center", va="center",
            fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
            color=C_OK if acc >= 0.85 else C_ACCENT)


def linear_pred(X, w1, b):
    return ((w1 * X[:, 0] + X[:, 1] + b) >= 0).astype(int)


def ring_pred(X, radius):
    r = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    return (r >= radius).astype(int)


# -------------------------------------------------------
# FIGURE 1: Four toy worlds
# -------------------------------------------------------
def build_step_1():
    fig1, axes1 = new_figure(2, 2)
    fig1.suptitle("Step 1: Fun starts with weird little worlds", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    worlds = [
        (make_blobs(), "blobs"),
        (make_moons(), "moons"),
        (make_rings(), "rings"),
        (make_spiral(), "spirals"),
    ]
    for ax, ((Xw, yw), name) in zip(axes1.ravel(), worlds):
        paint_points(ax, Xw, yw, s=36)
        ax.set_title(name, fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
    fig1.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig1


# -------------------------------------------------------
# FIGURE 2: Loop as four visual panels
# -------------------------------------------------------
def build_step_2():
    fig2, axes2 = new_figure(1, 4)
    fig2.suptitle("Step 2: data  ->  guess  ->  score  ->  update", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    Xb, yb = make_blobs(40)

    # data
    paint_points(axes2[0], Xb, yb, s=50)
    axes2[0].set_title("data", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    axes2[0].set_xticks([]); axes2[0].set_yticks([])
    axes2[0].set_aspect("equal")

    # guess: shaded half-planes + line
    xx = np.linspace(-3, 3, 200)
    yy = np.linspace(-2.5, 2.5, 200)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (0.1 * XX + YY) >= 0
    axes2[1].contourf(XX, YY, ZZ.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP_BG, alpha=0.9)
    paint_points(axes2[1], Xb, yb, s=40)
    axes2[1].plot(xx, -0.1 * xx, color=C_ACCENT, lw=2.6)
    axes2[1].set_title("guess", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    axes2[1].set_xticks([]); axes2[1].set_yticks([])
    axes2[1].set_xlim(-3, 3); axes2[1].set_ylim(-2.5, 2.5)
    axes2[1].set_aspect("equal")

    # score: confusion-style 2x2 mosaic by count
    pred = linear_pred(Xb, 0.1, 0.0)
    tp = int(np.sum((yb == 1) & (pred == 1)))
    tn = int(np.sum((yb == 0) & (pred == 0)))
    fp = int(np.sum((yb == 0) & (pred == 1)))
    fn = int(np.sum((yb == 1) & (pred == 0)))
    mos = np.array([[tn, fp], [fn, tp]], dtype=float)
    im = axes2[2].imshow(mos, cmap="YlGn", vmin=0, vmax=max(mos.max(), 1))
    for (i, j), v in np.ndenumerate(mos):
        axes2[2].text(j, i, f"{int(v)}", ha="center", va="center",
                      fontsize=FS_ANNOT + 2, fontweight="bold", color=C_DARK)
    axes2[2].set_xticks([0, 1]); axes2[2].set_yticks([0, 1])
    axes2[2].set_xticklabels(["pred 0", "pred 1"], fontsize=FS_TICK)
    axes2[2].set_yticklabels(["true 0", "true 1"], fontsize=FS_TICK)
    axes2[2].set_title("score", fontsize=FS_AX_TITLE - 2, fontweight="bold")

    # update: loss landscape cartoon + arrow downhill
    w_grid = np.linspace(-2, 2, 80)
    loss_curve = (w_grid - 0.7) ** 2 + 0.15
    axes2[3].plot(w_grid, loss_curve, color=C_DARK, lw=2.4)
    axes2[3].scatter([-0.6], [(-0.6 - 0.7) ** 2 + 0.15], s=90, c=C_ACCENT, zorder=3)
    axes2[3].annotate(
        "", xy=(0.2, (0.2 - 0.7) ** 2 + 0.15),
        xytext=(-0.6, (-0.6 - 0.7) ** 2 + 0.15),
        arrowprops=dict(arrowstyle="->", color=C_FUN, lw=2.4, mutation_scale=16),
    )
    axes2[3].set_title("update", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    axes2[3].set_xticks([]); axes2[3].set_yticks([])
    axes2[3].set_xlabel("parameter", fontsize=FS_TICK)
    axes2[3].set_ylabel("loss", fontsize=FS_TICK)

    for ax in axes2:
        ax.grid(True, alpha=0.2)
    fig2.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig2


# -------------------------------------------------------
# FIGURE 3: Task gallery as mini-plots
# -------------------------------------------------------
def build_step_3():
    fig3, axes3 = new_figure(2, 3)
    fig3.suptitle("Step 3: Interesting training tasks (as pictures)", fontsize=FS_FIGURE_TITLE, fontweight="bold")

    # vision: fake doodle digit heatmap
    dig = np.zeros((14, 14))
    dig[2:12, 3:5] = 1
    dig[2:4, 3:11] = 1
    dig[6:8, 3:10] = 1
    dig[10:12, 3:11] = 1
    axes3[0, 0].imshow(dig, cmap="magma")
    axes3[0, 0].set_title("vision", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
    axes3[0, 0].set_xticks([]); axes3[0, 0].set_yticks([])

    # language: next-token bar probs
    tok = np.array([0.42, 0.22, 0.14, 0.10, 0.07, 0.05])
    axes3[0, 1].bar(np.arange(len(tok)), tok, color=[C_FUN if i == 0 else C_GRAY for i in range(len(tok))])
    axes3[0, 1].set_title("language", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_FUN)
    axes3[0, 1].set_xticks(np.arange(len(tok)))
    axes3[0, 1].set_xticklabels(["cat", "dog", "the", "a", "red", "…"], fontsize=FS_TICK - 1, rotation=30)
    axes3[0, 1].set_ylim(0, 0.55)
    axes3[0, 1].set_yticks([])

    # games / RL: reward over episodes
    ep = np.arange(1, 41)
    rew = 8 * (1 - np.exp(-ep / 12)) + RNG.normal(0, 0.35, len(ep))
    axes3[0, 2].plot(ep, rew, color=C_OK, lw=2.2)
    axes3[0, 2].fill_between(ep, rew, alpha=0.2, color=C_OK)
    axes3[0, 2].set_title("games / RL", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    axes3[0, 2].set_xlabel("episode", fontsize=FS_TICK)
    axes3[0, 2].set_ylabel("reward", fontsize=FS_TICK)

    # generation: sampled noisy strokes
    for k in range(6):
        ang = np.linspace(0, 2 * np.pi, 80)
        rr = 0.6 + 0.25 * np.sin(3 * ang + k) + RNG.normal(0, 0.03, 80)
        axes3[1, 0].plot(rr * np.cos(ang) + 0.15 * k, rr * np.sin(ang), lw=1.8, alpha=0.75)
    axes3[1, 0].set_title("generation", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_BIN)
    axes3[1, 0].set_aspect("equal")
    axes3[1, 0].set_xticks([]); axes3[1, 0].set_yticks([])

    # science: noisy signal + fit
    xs = np.linspace(0, 4 * np.pi, 80)
    ys = np.sin(xs) + 0.15 * xs + RNG.normal(0, 0.25, len(xs))
    axes3[1, 1].scatter(xs, ys, s=18, c=C_GRAY, alpha=0.8)
    axes3[1, 1].plot(xs, np.sin(xs) + 0.15 * xs, color=C_DARK, lw=2.4)
    axes3[1, 1].set_title("science", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
    axes3[1, 1].set_xticks([]); axes3[1, 1].set_yticks([])

    # robots: path to a goal
    path_t = np.linspace(0, 1, 60)
    path = np.column_stack([path_t * 3, 0.6 * np.sin(2.5 * np.pi * path_t)])
    axes3[1, 2].plot(path[:, 0], path[:, 1], color=C_ACCENT, lw=2.6)
    axes3[1, 2].scatter([0], [0], s=120, c=C_DARK, zorder=3)
    axes3[1, 2].scatter([3], [0], s=180, marker="*", c=C_BIN, zorder=3)
    axes3[1, 2].set_title("robots", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_ACCENT)
    axes3[1, 2].set_xticks([]); axes3[1, 2].set_yticks([])
    axes3[1, 2].set_aspect("equal")

    for ax in axes3.ravel():
        ax.grid(True, alpha=0.2)
    fig3.tight_layout(rect=[0, 0.02, 1, 0.94])
    return fig3


# -------------------------------------------------------
# FIGURE 4: Interactive blobs -- drag a line
# -------------------------------------------------------
def build_step_4():
    fig4, (ax4, ax4m) = new_figure(1, 2, gridspec_kw={"width_ratios": [3.2, 1.0]})
    fig4.suptitle("Step 4: Drag the line -- cookies vs stars", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig4.subplots_adjust(bottom=0.18, wspace=0.25)

    X4, y4 = make_blobs(55)
    xx4 = np.linspace(-3.2, 3.2, 220)
    yy4 = np.linspace(-2.6, 2.6, 220)
    XX4, YY4 = np.meshgrid(xx4, yy4)
    w1_0, b_0 = -0.15, 0.05


    def _draw_halfplane(ax, XX, YY, w1, b, X, y):
        ax.clear()
        Z = (w1 * XX + YY + b) >= 0
        ax.contourf(XX, YY, Z.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP_BG, alpha=0.95)
        paint_points(ax, X, y, s=50)
        xs = XX[0]
        ax.plot(xs, -w1 * xs - b, color=C_ACCENT, lw=3.0)
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(YY[:, 0].min(), YY[:, 0].max())
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
        ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=FS_LEGEND - 1, loc="upper left")


    _draw_halfplane(ax4, XX4, YY4, w1_0, b_0, X4, y4)
    accuracy_bar(ax4m, float(np.mean(linear_pred(X4, w1_0, b_0) == y4)))

    ax_w4 = fig4.add_axes([0.12, 0.07, 0.35, 0.04])
    ax_b4 = fig4.add_axes([0.55, 0.07, 0.30, 0.04])
    sl_w4 = Slider(ax_w4, r"$w_1$", -2.5, 2.5, valinit=w1_0, valstep=0.05, color=C_FUN)
    sl_b4 = Slider(ax_b4, r"$b$", -2.0, 2.0, valinit=b_0, valstep=0.05, color=C_BIN)


    def _update4(_=None):
        w1 = float(sl_w4.val)
        b = float(sl_b4.val)
        _draw_halfplane(ax4, XX4, YY4, w1, b, X4, y4)
        accuracy_bar(ax4m, float(np.mean(linear_pred(X4, w1, b) == y4)))
        fig4.canvas.draw_idle()


    sl_w4.on_changed(_update4)
    sl_b4.on_changed(_update4)
    print("Step 4: drag w1 / b on blobs.")
    return fig4


# -------------------------------------------------------
# FIGURE 5: Interactive moons -- linear stalls
# -------------------------------------------------------
def build_step_5():
    fig5, (ax5, ax5m) = new_figure(1, 2, gridspec_kw={"width_ratios": [3.2, 1.0]})
    fig5.suptitle("Step 5: Same knobs -- moons fight back", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig5.subplots_adjust(bottom=0.18, wspace=0.25)

    X5, y5 = make_moons(75)
    xx5 = np.linspace(-1.8, 2.8, 220)
    yy5 = np.linspace(-1.2, 1.9, 220)
    XX5, YY5 = np.meshgrid(xx5, yy5)
    w1_5, b_5 = 0.55, -0.35


    def _draw_halfplane_loc(ax, XX, YY, w1, b, X, y, legend_loc="upper right", s=42):
        ax.clear()
        Z = (w1 * XX + YY + b) >= 0
        ax.contourf(XX, YY, Z.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP_BG, alpha=0.95)
        paint_points(ax, X, y, s=s)
        xs = XX[0]
        ax.plot(xs, -w1 * xs - b, color=C_ACCENT, lw=3.0)
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(YY[:, 0].min(), YY[:, 0].max())
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
        ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=FS_LEGEND - 1, loc=legend_loc)


    _draw_halfplane_loc(ax5, XX5, YY5, w1_5, b_5, X5, y5)
    accuracy_bar(ax5m, float(np.mean(linear_pred(X5, w1_5, b_5) == y5)))

    ax_w5 = fig5.add_axes([0.12, 0.07, 0.35, 0.04])
    ax_b5 = fig5.add_axes([0.55, 0.07, 0.30, 0.04])
    sl_w5 = Slider(ax_w5, r"$w_1$", -2.5, 2.5, valinit=w1_5, valstep=0.05, color=C_FUN)
    sl_b5 = Slider(ax_b5, r"$b$", -2.0, 2.0, valinit=b_5, valstep=0.05, color=C_BIN)


    def _update5(_=None):
        w1 = float(sl_w5.val)
        b = float(sl_b5.val)
        _draw_halfplane_loc(ax5, XX5, YY5, w1, b, X5, y5)
        accuracy_bar(ax5m, float(np.mean(linear_pred(X5, w1, b) == y5)))
        fig5.canvas.draw_idle()


    sl_w5.on_changed(_update5)
    sl_b5.on_changed(_update5)
    print("Step 5: moons -- straight cuts stall.")
    return fig5


# -------------------------------------------------------
# FIGURE 6: Interactive rings -- drag a radius
# -------------------------------------------------------
def build_step_6():
    fig6, (ax6, ax6m) = new_figure(1, 2, gridspec_kw={"width_ratios": [3.2, 1.0]})
    fig6.suptitle("Step 6: Drag the radius -- ring world", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig6.subplots_adjust(bottom=0.18, wspace=0.25)

    X6, y6 = make_rings(90)
    xx6 = np.linspace(-2.0, 2.0, 220)
    yy6 = np.linspace(-2.0, 2.0, 220)
    XX6, YY6 = np.meshgrid(xx6, yy6)
    r0 = 0.95


    def _draw_ring(ax, XX, YY, radius, X, y):
        ax.clear()
        R = np.sqrt(XX ** 2 + YY ** 2)
        ax.contourf(XX, YY, (R >= radius).astype(float), levels=[-0.1, 0.5, 1.1],
                    cmap=CMAP_BG, alpha=0.95)
        paint_points(ax, X, y, s=36)
        ax.add_patch(Circle((0, 0), radius, fill=False, color=C_ACCENT, lw=3.0))
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
        ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")


    _draw_ring(ax6, XX6, YY6, r0, X6, y6)
    accuracy_bar(ax6m, float(np.mean(ring_pred(X6, r0) == y6)))

    ax_r6 = fig6.add_axes([0.25, 0.07, 0.45, 0.04])
    sl_r6 = Slider(ax_r6, r"radius", 0.2, 1.8, valinit=r0, valstep=0.02, color=C_FUN)


    def _update6(_=None):
        r = float(sl_r6.val)
        _draw_ring(ax6, XX6, YY6, r, X6, y6)
        accuracy_bar(ax6m, float(np.mean(ring_pred(X6, r) == y6)))
        fig6.canvas.draw_idle()


    sl_r6.on_changed(_update6)
    print("Step 6: drag ring radius.")
    return fig6


# -------------------------------------------------------
# FIGURE 7: XOR playground -- twist a nonlinear cut
# -------------------------------------------------------
def make_xor(n=60):
    a = RNG.normal([-1.0, -1.0], 0.35, size=(n // 2, 2))
    b = RNG.normal([1.0, 1.0], 0.35, size=(n // 2, 2))
    c = RNG.normal([-1.0, 1.0], 0.35, size=(n // 2, 2))
    d = RNG.normal([1.0, -1.0], 0.35, size=(n // 2, 2))
    X = np.vstack([a, b, c, d])
    y = np.array([0] * n + [1] * n)
    return X, y


def build_step_7():
    fig7, (ax7, ax7m) = new_figure(1, 2, gridspec_kw={"width_ratios": [3.2, 1.0]})
    fig7.suptitle("Step 7: XOR playground -- twist the cut", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig7.subplots_adjust(bottom=0.18, wspace=0.25)

    X7x, y7x = make_xor(64)
    xx7x = np.linspace(-2.4, 2.4, 220)
    yy7x = np.linspace(-2.4, 2.4, 220)
    XX7x, YY7x = np.meshgrid(xx7x, yy7x)
    ang0, thr0 = 0.7, 0.0


    def xor_score(X, ang, thr):
        # rotated product feature: sign(x' * y' - thr)
        ca, sa = np.cos(ang), np.sin(ang)
        xr = ca * X[:, 0] + sa * X[:, 1]
        yr = -sa * X[:, 0] + ca * X[:, 1]
        return (xr * yr >= thr).astype(int)


    def _draw_xor(ax, XX, YY, ang, thr, X, y):
        ax.clear()
        flat = np.column_stack([XX.ravel(), YY.ravel()])
        Z = xor_score(flat, ang, thr).reshape(XX.shape)
        ax.contourf(XX, YY, Z.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP_BG, alpha=0.95)
        paint_points(ax, X, y, s=40)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
        ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=FS_LEGEND - 1, loc="upper left")


    _draw_xor(ax7, XX7x, YY7x, ang0, thr0, X7x, y7x)
    accuracy_bar(ax7m, float(np.mean(xor_score(X7x, ang0, thr0) == y7x)))

    ax_ang = fig7.add_axes([0.12, 0.07, 0.35, 0.04])
    ax_thr = fig7.add_axes([0.55, 0.07, 0.30, 0.04])
    sl_ang = Slider(ax_ang, r"twist", 0.0, np.pi, valinit=ang0, valstep=0.05, color=C_FUN)
    sl_thr = Slider(ax_thr, r"thr", -1.0, 1.0, valinit=thr0, valstep=0.05, color=C_BIN)


    def _update7(_=None):
        ang = float(sl_ang.val)
        thr = float(sl_thr.val)
        _draw_xor(ax7, XX7x, YY7x, ang, thr, X7x, y7x)
        accuracy_bar(ax7m, float(np.mean(xor_score(X7x, ang, thr) == y7x)))
        fig7.canvas.draw_idle()


    sl_ang.on_changed(_update7)
    sl_thr.on_changed(_update7)
    print("Step 7: XOR twist playground.")
    return fig7


# -------------------------------------------------------
# FIGURE 8: RL fun -- gridworld treasure hunt
# -------------------------------------------------------
# Tiny grid: start S, goal G, one pit X. Actions: U D L R.
GW = 5
GH = 5
START = (0, 0)
GOAL = (4, 4)
PIT = (2, 2)
ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]  # N S W E as (dx, dy) on (x,y)
ACT_LABEL = ["N", "S", "W", "E"]


def gw_step(state, action):
    x, y = state
    dx, dy = ACTIONS[action]
    nx, ny = x + dx, y + dy
    if not (0 <= nx < GW and 0 <= ny < GH):
        nx, ny = x, y
        reward = -0.2
    else:
        reward = -0.05
    nxt = (nx, ny)
    done = False
    if nxt == PIT:
        return nxt, -1.0, True
    if nxt == GOAL:
        return nxt, 1.0, True
    return nxt, reward, done


def run_q_learning(n_episodes=120, alpha=0.35, gamma=0.92, epsilon=0.25, seed=41):
    rng = np.random.default_rng(seed)
    Q = np.zeros((GW, GH, 4))
    returns = []
    snapshots = []  # (episode, Q copy, path of last episode)
    for ep in range(n_episodes):
        s = START
        path = [s]
        G = 0.0
        for _t in range(60):
            if rng.random() < epsilon:
                a = int(rng.integers(0, 4))
            else:
                a = int(np.argmax(Q[s[0], s[1]]))
            s2, r, done = gw_step(s, a)
            G += r
            best = np.max(Q[s2[0], s2[1]])
            Q[s[0], s[1], a] += alpha * (r + gamma * best * (0.0 if done else 1.0) - Q[s[0], s[1], a])
            s = s2
            path.append(s)
            if done:
                break
        returns.append(G)
        if ep % 4 == 0 or ep == n_episodes - 1:
            snapshots.append((ep, Q.copy(), path, float(np.mean(returns[-10:]))))
    return Q, returns, snapshots


def draw_gridworld(ax, Q, path=None, title="gridworld"):
    ax.clear()
    V = np.max(Q, axis=2).T  # image row = y
    ax.imshow(V, origin="lower", cmap="YlGn", vmin=-0.3, vmax=1.0, alpha=0.95)
    # walls / special cells
    ax.scatter([START[0]], [START[1]], s=220, c=C_DARK, marker="o", zorder=4)
    ax.scatter([GOAL[0]], [GOAL[1]], s=320, c=C_BIN, marker="*", zorder=4)
    ax.scatter([PIT[0]], [PIT[1]], s=260, c=C_ACCENT, marker="X", zorder=4)
    # policy arrows
    for x in range(GW):
        for y in range(GH):
            if (x, y) in (GOAL, PIT):
                continue
            a = int(np.argmax(Q[x, y]))
            dx, dy = ACTIONS[a]
            ax.arrow(x, y, 0.28 * dx, 0.28 * dy, head_width=0.12, head_length=0.1,
                     fc=C_FUN, ec=C_FUN, length_includes_head=True, alpha=0.85)
    if path is not None and len(path) > 1:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, color=C_ACCENT, lw=2.2, alpha=0.8, zorder=3)
        ax.scatter([px[-1]], [py[-1]], s=120, c=C_ACCENT, zorder=5)
    ax.set_xticks(range(GW))
    ax.set_yticks(range(GH))
    ax.set_xlim(-0.5, GW - 0.5)
    ax.set_ylim(-0.5, GH - 0.5)
    ax.set_title(title, fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.grid(True, color="white", lw=1.2)
    ax.set_aspect("equal")


def build_step_8():
    fig8, (ax8a, ax8b) = new_figure(1, 2)
    fig8.suptitle("Step 8: RL fun -- treasure hunt (Q-learning)", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig8.subplots_adjust(bottom=0.18, wspace=0.25)

    eps0 = 0.25
    Q8, rets8, snaps8 = run_q_learning(epsilon=eps0)
    draw_gridworld(ax8a, snaps8[0][1], snaps8[0][2], title="early policy")
    ret_line, = ax8b.plot([], [], color=C_OK, lw=2.4)
    ax8b.set_xlim(0, len(rets8))
    ax8b.set_ylim(min(min(rets8) - 0.2, -1.2), max(max(rets8) + 0.2, 1.2))
    ax8b.set_xlabel("episode", fontsize=FS_AXIS)
    ax8b.set_ylabel("return", fontsize=FS_AXIS)
    ax8b.set_title("learning curve", fontsize=FS_AX_TITLE - 2)
    ax8b.grid(True, alpha=0.3)
    ax8b.axhline(0.0, color="0.5", lw=1.0, ls="--")

    ax_eps = fig8.add_axes([0.25, 0.07, 0.50, 0.04])
    sl_eps = Slider(ax_eps, r"explore $\epsilon$", 0.0, 0.8, valinit=eps0, valstep=0.05, color=C_FUN)

    _rl_state = {"snaps": snaps8, "rets": rets8, "ani": None}


    def _rebuild_rl(eps):
        Q, rets, snaps = run_q_learning(epsilon=float(eps), seed=41)
        _rl_state["snaps"] = snaps
        _rl_state["rets"] = rets
        ax8b.set_ylim(min(min(rets) - 0.2, -1.2), max(max(rets) + 0.2, 1.2))
        return snaps, rets


    def _anim8(frame):
        snaps = _rl_state["snaps"]
        rets = _rl_state["rets"]
        i = int(frame) % len(snaps)
        ep, Q, path, _avg = snaps[i]
        draw_gridworld(ax8a, Q, path, title=f"episode {ep}")
        ret_line.set_data(np.arange(ep + 1), rets[: ep + 1])
        return []


    def _update_eps(_=None):
        snaps, rets = _rebuild_rl(sl_eps.val)
        draw_gridworld(ax8a, snaps[0][1], snaps[0][2], title="early policy")
        ret_line.set_data([], [])
        fig8.canvas.draw_idle()


    sl_eps.on_changed(_update_eps)
    ani8 = FuncAnimation(fig8, _anim8, frames=len(snaps8), interval=220, blit=False, repeat=True)
    _rl_state["ani"] = ani8
    print("Step 8: RL gridworld -- drag epsilon.")
    return fig8


# -------------------------------------------------------
# FIGURE 9: RL fun -- multi-armed bandit
# -------------------------------------------------------
def build_step_9():
    fig9, (ax9a, ax9b) = new_figure(1, 2)
    fig9.suptitle("Step 9: RL fun -- multi-armed bandit", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    fig9.subplots_adjust(bottom=0.20, wspace=0.28)

    N_ARMS = 5
    TRUE_MEAN = np.array([0.15, 0.45, 0.30, 0.70, 0.40])
    best_arm = int(np.argmax(TRUE_MEAN))


    def run_bandit(epsilon=0.15, steps=120, seed=7):
        rng = np.random.default_rng(seed)
        q = np.zeros(N_ARMS)
        n = np.zeros(N_ARMS)
        rewards = []
        regrets = []
        q_hist = []
        for t in range(1, steps + 1):
            if rng.random() < epsilon:
                a = int(rng.integers(0, N_ARMS))
            else:
                a = int(np.argmax(q))
            r = float(rng.normal(TRUE_MEAN[a], 0.12))
            n[a] += 1
            q[a] += (r - q[a]) / n[a]
            rewards.append(r)
            regrets.append(TRUE_MEAN[best_arm] - TRUE_MEAN[a])
            q_hist.append(q.copy())
        return np.array(rewards), np.array(regrets), np.array(q_hist)


    eps_b0 = 0.15
    rew_b, reg_b, qhist_b = run_bandit(epsilon=eps_b0)

    bars9 = ax9a.bar(np.arange(N_ARMS), qhist_b[0], color=C_GRAY, edgecolor="white", width=0.65)
    truth9 = ax9a.scatter(np.arange(N_ARMS), TRUE_MEAN, s=80, c=C_BIN, zorder=3, label="true mean")
    ax9a.set_ylim(-0.1, 1.0)
    ax9a.set_xticks(np.arange(N_ARMS))
    ax9a.set_xticklabels([f"arm {i}" for i in range(N_ARMS)], fontsize=FS_TICK)
    ax9a.set_ylabel("estimated Q", fontsize=FS_AXIS)
    ax9a.set_title("arm values", fontsize=FS_AX_TITLE - 2)
    ax9a.legend(fontsize=FS_LEGEND - 1, loc="upper left")
    ax9a.grid(True, alpha=0.25, axis="y")

    cum_reg = np.cumsum(reg_b)
    line_reg, = ax9b.plot([], [], color=C_ACCENT, lw=2.5)
    line_avg, = ax9b.plot([], [], color=C_OK, lw=2.2)
    ax9b.set_xlim(0, len(rew_b))
    ax9b.set_ylim(0, max(cum_reg.max() * 1.1, 1.0))
    ax9b.set_xlabel("pull", fontsize=FS_AXIS)
    ax9b.set_title("regret & avg reward", fontsize=FS_AX_TITLE - 2)
    ax9b.grid(True, alpha=0.3)
    ax9b.legend([line_reg, line_avg], ["cum. regret", "avg reward"], fontsize=FS_LEGEND - 1, loc="upper left")

    ax_eps_b = fig9.add_axes([0.25, 0.07, 0.50, 0.04])
    sl_eps_b = Slider(ax_eps_b, r"explore $\epsilon$", 0.0, 0.8, valinit=eps_b0, valstep=0.05, color=C_FUN)

    _bandit = {"rew": rew_b, "reg": reg_b, "qhist": qhist_b, "ani": None}


    def _rebuild_bandit(eps):
        rew, reg, qhist = run_bandit(epsilon=float(eps), seed=7)
        _bandit["rew"] = rew
        _bandit["reg"] = reg
        _bandit["qhist"] = qhist
        ax9b.set_ylim(0, max(np.cumsum(reg).max() * 1.1, 1.0))
        return rew, reg, qhist


    def _anim9(frame):
        i = int(frame)
        qhist = _bandit["qhist"]
        rew = _bandit["rew"]
        reg = _bandit["reg"]
        i = min(i, len(qhist) - 1)
        for bar, val in zip(bars9, qhist[i]):
            bar.set_height(val)
            bar.set_color(C_OK if val == qhist[i].max() else C_GRAY)
        line_reg.set_data(np.arange(1, i + 2), np.cumsum(reg[: i + 1]))
        line_avg.set_data(np.arange(1, i + 2), np.cumsum(rew[: i + 1]) / np.arange(1, i + 2))
        return list(bars9) + [line_reg, line_avg]


    def _update_bandit(_=None):
        _rebuild_bandit(sl_eps_b.val)
        for bar, val in zip(bars9, _bandit["qhist"][0]):
            bar.set_height(val)
            bar.set_color(C_GRAY)
        line_reg.set_data([], [])
        line_avg.set_data([], [])
        fig9.canvas.draw_idle()


    sl_eps_b.on_changed(_update_bandit)
    ani9 = FuncAnimation(fig9, _anim9, frames=len(qhist_b), interval=40, blit=False, repeat=True)
    _bandit["ani"] = ani9
    print("Step 9: bandit -- drag epsilon.")
    return fig9


# -------------------------------------------------------
# FIGURE 10: Watch supervised training -- boundary + loss
# -------------------------------------------------------
def build_step_10():
    fig10, (ax10a, ax10b) = new_figure(1, 2)
    fig10.suptitle("Step 10: Watch supervised training -- boundary learns", fontsize=FS_FIGURE_TITLE, fontweight="bold")

    X10, y10 = make_blobs(60)
    w1, b = 2.0, -1.5
    lr = 0.35
    hist_w, hist_b, hist_loss, hist_acc = [], [], [], []
    for _ in range(45):
        logits = w1 * X10[:, 0] + X10[:, 1] + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20, 20)))
        loss = float(-np.mean(y10 * np.log(probs + 1e-9) + (1 - y10) * np.log(1 - probs + 1e-9)))
        acc = float(np.mean((probs >= 0.5).astype(int) == y10))
        hist_w.append(w1); hist_b.append(b); hist_loss.append(loss); hist_acc.append(acc)
        grad = probs - y10
        w1 -= lr * np.mean(grad * X10[:, 0])
        b -= lr * np.mean(grad)

    xx10 = np.linspace(-3.2, 3.2, 200)
    paint_points(ax10a, X10, y10, s=42)
    line10, = ax10a.plot(xx10, -hist_w[0] * xx10 - hist_b[0], color=C_ACCENT, lw=3.0)
    ax10a.set_xlim(-3.2, 3.2); ax10a.set_ylim(-2.6, 2.6)
    ax10a.set_aspect("equal")
    ax10a.set_title("decision boundary", fontsize=FS_AX_TITLE - 2)
    ax10a.grid(True, alpha=0.25)

    loss_line, = ax10b.plot([], [], color=C_DARK, lw=2.6)
    acc_line, = ax10b.plot([], [], color=C_OK, lw=2.2, alpha=0.85)
    ax10b.set_xlim(0, len(hist_loss) - 1)
    ax10b.set_ylim(0, max(max(hist_loss) * 1.15, 1.0))
    ax10b.set_xlabel("step", fontsize=FS_AXIS)
    ax10b.set_ylabel("loss / accuracy", fontsize=FS_AXIS)
    ax10b.set_title("curves", fontsize=FS_AX_TITLE - 2)
    ax10b.grid(True, alpha=0.3)
    ax10b.legend([loss_line, acc_line], ["loss", "accuracy"], fontsize=FS_LEGEND - 1, loc="upper right")


    def _anim10(frame):
        i = int(frame)
        w1i, bi = hist_w[i], hist_b[i]
        line10.set_ydata(-w1i * xx10 - bi)
        loss_line.set_data(np.arange(i + 1), hist_loss[: i + 1])
        acc_line.set_data(np.arange(i + 1), hist_acc[: i + 1])
        return line10, loss_line, acc_line


    ani10 = FuncAnimation(fig10, _anim10, frames=len(hist_loss), interval=80, blit=False, repeat=True)
    print("Step 10: supervised training animation.")
    return fig10


# -------------------------------------------------------
# FIGURE 11: Visual takeaway collage
# -------------------------------------------------------
def build_step_11():
    fig11, axes11 = new_figure(1, 4)
    fig11.suptitle("Step 11: Play  /  score  /  explore  /  demo", fontsize=FS_FIGURE_TITLE, fontweight="bold")

    Xs, ys = make_spiral(70)
    axes11[0].scatter(Xs[ys == 0, 0], Xs[ys == 0, 1], s=22, c=C_FUN, alpha=0.85)
    axes11[0].scatter(Xs[ys == 1, 0], Xs[ys == 1, 1], s=22, c=C_BIN, alpha=0.85)
    axes11[0].set_title("play", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_FUN)
    axes11[0].set_xticks([]); axes11[0].set_yticks([])
    axes11[0].set_aspect("equal")

    labs = ["acc", "F1"]
    vals = [0.92, 0.41]
    axes11[1].bar(labs, vals, color=[C_GRAY, C_ACCENT], width=0.55)
    axes11[1].set_ylim(0, 1.05)
    axes11[1].set_title("score", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_ACCENT)
    for i, v in enumerate(vals):
        axes11[1].text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=FS_ANNOT, fontweight="bold")

    # mini RL path souvenir
    Q8, _rets8, snaps8 = run_q_learning(epsilon=0.25)
    draw_gridworld(axes11[2], Q8, snaps8[-1][2], title="explore")

    Xf, yf = make_blobs(40)
    paint_points(axes11[3], Xf, yf, s=32)
    xf = np.linspace(-3, 3, 50)
    axes11[3].plot(xf, 0.05 * xf, color=C_ACCENT, lw=2.4)
    axes11[3].set_title("demo", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    axes11[3].set_xticks([]); axes11[3].set_yticks([])
    axes11[3].set_aspect("equal")

    for ax in axes11:
        ax.grid(True, alpha=0.2)
    fig11.tight_layout(rect=[0, 0.02, 1, 0.92])
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
print("041 AI FUN -- VISUAL PLAYGROUND + RL")
print("=" * 60)
print("""
Toys: weird worlds, drag-line / ring / XOR,
      RL gridworld treasure hunt, multi-armed bandit,
      supervised training animation.

, ,
""")
