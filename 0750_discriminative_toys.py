"""
0750_discriminative_toys.py

Typical discriminative models -- listed and shown as interactive toys.

Discriminative models learn a direct map from x to y (or P(y|x)),
not a full model of P(x).

Sketch:
  1. Discriminative idea -- label regions over x
  2. Family gallery -- typical models as mini boundaries
  3. Logistic regression -- drag the linear cut + probability
  4. Softmax (multiclass) -- three-way territories
  5. Linear SVM -- margin slab between classes
  6. k-NN -- drag k, watch the patchwork
  7. Tiny neural net -- nonlinear boundary from hidden units
  8. Decision stump forest vibe -- axis cuts stack up
  9. Side-by-side: same data, many discriminative skins
  10. Takeaway collage

Companion to 0310_regression_classification.py, 0740_discriminative_generative.py,
0820_softmax.py, 0700_neural_non_linearity.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
)
apply_rcparams(plt)
import numpy as np

np.random.seed(141)
RNG = np.random.default_rng(141)


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
CMAP3 = ListedColormap(["#d6eaf8", "#d5f5e3", "#fadbd8"])


def make_blobs2(n=55):
    a = RNG.normal([-1.15, 0.1], 0.48, size=(n, 2))
    b = RNG.normal([1.2, -0.05], 0.48, size=(n, 2))
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


def make_blobs3(n=40):
    centers = [(-1.3, -0.7), (1.3, -0.5), (0.1, 1.25)]
    xs, ys = [], []
    for i, c in enumerate(centers):
        xs.append(RNG.normal(c, 0.42, size=(n, 2)))
        ys.append(np.full(n, i))
    return np.vstack(xs), np.concatenate(ys)


def make_moons(n=70):
    t = np.linspace(0, np.pi, n)
    a = np.column_stack([np.cos(t), np.sin(t)]) + RNG.normal(0, 0.07, (n, 2))
    b = np.column_stack([1.0 - np.cos(t), 0.55 - np.sin(t)]) + RNG.normal(0, 0.07, (n, 2))
    return np.vstack([a, b]), np.array([0] * n + [1] * n)


def paint2(ax, X, y, s=42):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=s, c=C_GRAY,
               edgecolors="white", linewidths=0.5, zorder=3, label="y=0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=s, c=C_OK,
               edgecolors="white", linewidths=0.5, zorder=3, label="y=1")


def paint3(ax, X, y, s=40):
    colors = [C_GRAY, C_OK, C_ACCENT]
    for k, col in enumerate(colors):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=s, c=col,
                   edgecolors="white", linewidths=0.5, zorder=3, label=f"y={k}")


def mesh(xlim, ylim, n=220):
    xx = np.linspace(*xlim, n)
    yy = np.linspace(*ylim, n)
    XX, YY = np.meshgrid(xx, yy)
    return XX, YY, np.column_stack([XX.ravel(), YY.ravel()])


def accuracy_meter(ax, acc, title="accuracy"):
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.barh([0.45], [1.0], height=0.28, color="#d5d8dc", zorder=1)
    ax.barh([0.45], [acc], height=0.28,
            color=C_OK if acc >= 0.85 else C_ACCENT, zorder=2)
    ax.text(0.5, 0.82, title, ha="center", fontsize=FS_AX_TITLE - 2,
            fontweight="bold", color=C_DARK)
    ax.text(0.5, 0.18, f"{100 * acc:.0f}%", ha="center",
            fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
            color=C_OK if acc >= 0.85 else C_ACCENT)


# -------------------------------------------------------
# FIGURE 1: Discriminative idea
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    r"Step 1: Discriminative -- learn $P(y\mid x)$ / a decision rule over $x$",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
X1, y1 = make_blobs2(50)
XX1, YY1, flat1 = mesh((-3, 3), (-2.5, 2.5))
# logistic-ish probs
logits = 1.2 * flat1[:, 0] + 0.2 * flat1[:, 1]
probs = 1 / (1 + np.exp(-logits))
axes1[0].contourf(XX1, YY1, probs.reshape(XX1.shape), levels=12, cmap="RdYlGn", alpha=0.9)
paint2(axes1[0], X1, y1, s=36)
axes1[0].contour(XX1, YY1, probs.reshape(XX1.shape), levels=[0.5], colors=[C_DARK], linewidths=2.4)
axes1[0].set_title(r"$P(y=1\mid x)$ heat", fontsize=FS_AX_TITLE - 2)
axes1[0].set_xticks([]); axes1[0].set_yticks([])
axes1[0].set_aspect("equal")

axes1[1].contourf(XX1, YY1, (probs.reshape(XX1.shape) >= 0.5).astype(float),
                  levels=[-0.1, 0.5, 1.1], cmap=CMAP2, alpha=0.95)
paint2(axes1[1], X1, y1, s=36)
axes1[1].set_title("hard decision regions", fontsize=FS_AX_TITLE - 2)
axes1[1].set_xticks([]); axes1[1].set_yticks([])
axes1[1].set_aspect("equal")
for ax in axes1:
    ax.grid(True, alpha=0.2)
fig1.tight_layout(rect=[0, 0.02, 1, 0.92])
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Family gallery of typical discriminative models
# -------------------------------------------------------
fig2, axes2 = new_figure(2, 3)
fig2.suptitle("Step 2: Typical discriminative models (visual roll call)", fontsize=FS_FIGURE_TITLE, fontweight="bold")

Xg, yg = make_blobs2(40)
Xm, ym = make_moons(50)
XX, YY, flat = mesh((-3, 3), (-2.5, 2.5), n=180)
XXm, YYm, flatm = mesh((-1.8, 2.8), (-1.2, 1.9), n=180)

# logistic
Z = (1.1 * flat[:, 0] + 0.15 * flat[:, 1] >= 0).reshape(XX.shape)
axes2[0, 0].contourf(XX, YY, Z.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP2)
paint2(axes2[0, 0], Xg, yg, s=22)
axes2[0, 0].set_title("logistic", fontsize=FS_AX_TITLE - 2, color=C_DARK, fontweight="bold")

# softmax 3-class
X3g, y3g = make_blobs3(28)
XX3, YY3, flat3 = mesh((-2.5, 2.5), (-2.0, 2.2), n=160)
# distances to class prototypes -> softmax argmax
protos = np.array([[-1.3, -0.7], [1.3, -0.5], [0.1, 1.25]])
d2 = ((flat3[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2)
Z3 = np.argmin(d2, axis=1).reshape(XX3.shape)
axes2[0, 1].contourf(XX3, YY3, Z3, levels=[-0.5, 0.5, 1.5, 2.5], cmap=CMAP3, alpha=0.9)
paint3(axes2[0, 1], X3g, y3g, s=18)
axes2[0, 1].set_title("softmax", fontsize=FS_AX_TITLE - 2, color=C_FUN, fontweight="bold")

# SVM margin look
Zsvm = (1.0 * flat[:, 0]).reshape(XX.shape)
axes2[0, 2].contourf(XX, YY, (Zsvm >= 0).astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP2)
axes2[0, 2].contour(XX, YY, Zsvm, levels=[-1, 0, 1], colors=[C_ACCENT, C_DARK, C_ACCENT],
                    linewidths=[1.6, 2.4, 1.6], linestyles=["--", "-", "--"])
paint2(axes2[0, 2], Xg, yg, s=22)
axes2[0, 2].set_title("linear SVM", fontsize=FS_AX_TITLE - 2, color=C_ACCENT, fontweight="bold")

# k-NN patchwork on moons
def knn_predict(Xtrain, ytrain, Xquery, k=5):
    # chunked for speed
    pred = np.empty(len(Xquery), dtype=int)
    bs = 800
    for i in range(0, len(Xquery), bs):
        q = Xquery[i:i + bs]
        d2 = ((q[:, None, :] - Xtrain[None, :, :]) ** 2).sum(axis=2)
        nn = np.argpartition(d2, kth=min(k, d2.shape[1] - 1), axis=1)[:, :k]
        votes = ytrain[nn]
        # majority
        pred[i:i + bs] = (votes.mean(axis=1) >= 0.5).astype(int)
    return pred


Zk = knn_predict(Xm, ym, flatm, k=7).reshape(XXm.shape)
axes2[1, 0].contourf(XXm, YYm, Zk.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP2)
paint2(axes2[1, 0], Xm, ym, s=16)
axes2[1, 0].set_title("k-NN", fontsize=FS_AX_TITLE - 2, color=C_BIN, fontweight="bold")

# neural-ish: RBF / quadratic feature
phi = np.column_stack([flatm[:, 0], flatm[:, 1], flatm[:, 0] ** 2, flatm[:, 1] ** 2, flatm[:, 0] * flatm[:, 1]])
# hand weights that roughly separate moons
w_nn = np.array([-0.2, 1.6, -1.1, -0.3, 0.9])
b_nn = -0.35
Znn = (phi @ w_nn + b_nn >= 0).reshape(XXm.shape)
axes2[1, 1].contourf(XXm, YYm, Znn.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP2)
paint2(axes2[1, 1], Xm, ym, s=16)
axes2[1, 1].set_title("neural net", fontsize=FS_AX_TITLE - 2, color=C_OK, fontweight="bold")

# tree stump stack: x1>0 then x2>0.2 etc -- piecewise
Zt = np.zeros(len(flat), dtype=int)
Zt[(flat[:, 0] > 0) & (flat[:, 1] > -0.2)] = 1
Zt[(flat[:, 0] <= 0) & (flat[:, 1] > 0.8)] = 1
axes2[1, 2].contourf(XX, YY, Zt.reshape(XX.shape).astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP2)
paint2(axes2[1, 2], Xg, yg, s=22)
axes2[1, 2].axvline(0, color=C_FUN, lw=1.5, ls="--")
axes2[1, 2].axhline(-0.2, color=C_FUN, lw=1.2, ls=":", xmin=0.5)
axes2[1, 2].set_title("tree / stumps", fontsize=FS_AX_TITLE - 2, color=C_FUN, fontweight="bold")

for ax in axes2.ravel():
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
fig2.tight_layout(rect=[0, 0.02, 1, 0.94])
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Interactive logistic regression
# -------------------------------------------------------
fig3, (ax3, ax3m) = new_figure(1, 2, gridspec_kw={"width_ratios": [3.2, 1.0]})
fig3.suptitle("Step 3: Logistic regression -- drag the linear logit", fontsize=FS_FIGURE_TITLE, fontweight="bold")
fig3.subplots_adjust(bottom=0.18, wspace=0.25)

X3, y3 = make_blobs2(60)
XX3b, YY3b, flat3b = mesh((-3.2, 3.2), (-2.6, 2.6))
w1_0, b_0 = 1.0, 0.0


def logistic_prob(X, w1, b):
    return 1 / (1 + np.exp(-(w1 * X[:, 0] + X[:, 1] + b)))


def draw_logistic(ax, w1, b):
    ax.clear()
    p = logistic_prob(flat3b, w1, b).reshape(XX3b.shape)
    ax.contourf(XX3b, YY3b, p, levels=14, cmap="RdYlGn", alpha=0.92)
    ax.contour(XX3b, YY3b, p, levels=[0.5], colors=[C_DARK], linewidths=2.6)
    paint2(ax, X3, y3, s=44)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper left")


def _acc_log(w1, b):
    return float(np.mean((logistic_prob(X3, w1, b) >= 0.5).astype(int) == y3))


draw_logistic(ax3, w1_0, b_0)
accuracy_meter(ax3m, _acc_log(w1_0, b_0))

sl_w3 = Slider(fig3.add_axes([0.12, 0.07, 0.35, 0.04]), r"$w_1$", -3, 3, valinit=w1_0, valstep=0.05, color=C_FUN)
sl_b3 = Slider(fig3.add_axes([0.55, 0.07, 0.30, 0.04]), r"$b$", -2, 2, valinit=b_0, valstep=0.05, color=C_BIN)


def _u3(_=None):
    w1, b = float(sl_w3.val), float(sl_b3.val)
    draw_logistic(ax3, w1, b)
    accuracy_meter(ax3m, _acc_log(w1, b))
    fig3.canvas.draw_idle()


sl_w3.on_changed(_u3); sl_b3.on_changed(_u3)
print("Step 3: logistic interactive.")
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Interactive softmax territories
# -------------------------------------------------------
fig4, (ax4, ax4m) = new_figure(1, 2, gridspec_kw={"width_ratios": [3.2, 1.0]})
fig4.suptitle("Step 4: Softmax -- drag temperature over 3-class territories", fontsize=FS_FIGURE_TITLE, fontweight="bold")
fig4.subplots_adjust(bottom=0.18, wspace=0.25)

X4, y4 = make_blobs3(45)
XX4, YY4, flat4 = mesh((-2.6, 2.6), (-2.2, 2.4))
protos4 = np.array([[-1.3, -0.7], [1.3, -0.5], [0.1, 1.25]], dtype=float)
tau0 = 0.55


def softmax_labels(X, protos, tau):
    # logits = -||x-mu||^2 / tau
    d2 = ((X[:, None, :] - protos[None, :, :]) ** 2).sum(axis=2)
    logits = -d2 / max(tau, 1e-3)
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    P /= P.sum(axis=1, keepdims=True)
    return P.argmax(axis=1), P


def draw_softmax(ax, tau):
    ax.clear()
    lab, P = softmax_labels(flat4, protos4, tau)
    Z = lab.reshape(XX4.shape)
    ax.contourf(XX4, YY4, Z, levels=[-0.5, 0.5, 1.5, 2.5], cmap=CMAP3, alpha=0.92)
    # max-prob contours for softness
    pmax = P.max(axis=1).reshape(XX4.shape)
    ax.contour(XX4, YY4, pmax, levels=[0.45, 0.7], colors=["0.35", "0.15"], linewidths=1.2, alpha=0.7)
    paint3(ax, X4, y4, s=40)
    ax.scatter(protos4[:, 0], protos4[:, 1], s=160, c=[C_GRAY, C_OK, C_ACCENT],
               marker="P", edgecolors="k", linewidths=0.8, zorder=5)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.2, 2.4)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper left")
    pred, _ = softmax_labels(X4, protos4, tau)
    return float(np.mean(pred == y4))


accuracy_meter(ax4m, draw_softmax(ax4, tau0))
sl_tau = Slider(fig4.add_axes([0.25, 0.07, 0.50, 0.04]), r"temperature $\tau$", 0.15, 2.5,
                valinit=tau0, valstep=0.05, color=C_FUN)


def _u4(_=None):
    accuracy_meter(ax4m, draw_softmax(ax4, float(sl_tau.val)))
    fig4.canvas.draw_idle()


sl_tau.on_changed(_u4)
print("Step 4: softmax interactive.")
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Interactive linear SVM margin
# -------------------------------------------------------
fig5, (ax5, ax5m) = new_figure(1, 2, gridspec_kw={"width_ratios": [3.2, 1.0]})
fig5.suptitle("Step 5: Linear SVM -- drag orientation and margin width", fontsize=FS_FIGURE_TITLE, fontweight="bold")
fig5.subplots_adjust(bottom=0.20, wspace=0.25)

X5, y5 = make_blobs2(55)
# labels as +/- 1 for SVM geometry
y5pm = np.where(y5 == 0, -1, 1)
XX5, YY5, flat5 = mesh((-3.2, 3.2), (-2.6, 2.6))
ang0, mar0 = 0.0, 1.0


def svm_score(X, ang, margin):
    # unit normal from angle; decision w·x = 0, margins at +/- margin
    w = np.array([np.cos(ang), np.sin(ang)])
    return X @ w, w


def draw_svm(ax, ang, margin):
    ax.clear()
    s, w = svm_score(flat5, ang, margin)
    Z = (s >= 0).astype(float).reshape(XX5.shape)
    ax.contourf(XX5, YY5, Z, levels=[-0.1, 0.5, 1.1], cmap=CMAP2, alpha=0.95)
    S = s.reshape(XX5.shape)
    ax.contour(XX5, YY5, S, levels=[-margin, 0, margin],
               colors=[C_ACCENT, C_DARK, C_ACCENT],
               linewidths=[1.8, 2.8, 1.8],
               linestyles=["--", "-", "--"])
    paint2(ax, X5, y5, s=44)
    # support-ish points: close to margin
    s_pts = X5 @ w
    dist = np.abs(s_pts)
    support = dist <= margin * 1.05
    if np.any(support):
        ax.scatter(X5[support, 0], X5[support, 1], s=160, facecolors="none",
                   edgecolors=C_FUN, linewidths=2.0, zorder=4)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper left")
    pred = (X5 @ w >= 0).astype(int)
    return float(np.mean(pred == y5))


accuracy_meter(ax5m, draw_svm(ax5, ang0, mar0), title="accuracy")
sl_ang5 = Slider(fig5.add_axes([0.12, 0.08, 0.35, 0.035]), r"angle", -np.pi / 2, np.pi / 2,
                 valinit=ang0, valstep=0.05, color=C_FUN)
sl_mar5 = Slider(fig5.add_axes([0.55, 0.08, 0.30, 0.035]), r"margin", 0.2, 2.0,
                 valinit=mar0, valstep=0.05, color=C_BIN)


def _u5(_=None):
    accuracy_meter(ax5m, draw_svm(ax5, float(sl_ang5.val), float(sl_mar5.val)))
    fig5.canvas.draw_idle()


sl_ang5.on_changed(_u5); sl_mar5.on_changed(_u5)
print("Step 5: SVM interactive.")
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Interactive k-NN
# -------------------------------------------------------
fig6, (ax6, ax6m) = new_figure(1, 2, gridspec_kw={"width_ratios": [3.2, 1.0]})
fig6.suptitle("Step 6: k-NN -- drag k on interlocking moons", fontsize=FS_FIGURE_TITLE, fontweight="bold")
fig6.subplots_adjust(bottom=0.18, wspace=0.25)

X6, y6 = make_moons(80)
XX6, YY6, flat6 = mesh((-1.8, 2.8), (-1.2, 1.9), n=160)
k0 = 5


def draw_knn(ax, k):
    ax.clear()
    Z = knn_predict(X6, y6, flat6, k=int(k)).reshape(XX6.shape)
    ax.contourf(XX6, YY6, Z.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP2, alpha=0.95)
    paint2(ax, X6, y6, s=36)
    ax.set_xlim(-1.8, 2.8); ax.set_ylim(-1.2, 1.9)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    pred = knn_predict(X6, y6, X6, k=int(k))
    return float(np.mean(pred == y6))


accuracy_meter(ax6m, draw_knn(ax6, k0))
sl_k = Slider(fig6.add_axes([0.25, 0.07, 0.50, 0.04]), r"$k$", 1, 25, valinit=k0, valstep=1, color=C_FUN)


def _u6(_=None):
    accuracy_meter(ax6m, draw_knn(ax6, int(sl_k.val)))
    fig6.canvas.draw_idle()


sl_k.on_changed(_u6)
print("Step 6: k-NN interactive.")
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Interactive tiny neural net (RBF features)
# -------------------------------------------------------
fig7, (ax7, ax7m) = new_figure(1, 2, gridspec_kw={"width_ratios": [3.2, 1.0]})
fig7.suptitle("Step 7: Tiny neural net -- drag hidden RBF scale", fontsize=FS_FIGURE_TITLE, fontweight="bold")
fig7.subplots_adjust(bottom=0.20, wspace=0.25)

X7, y7 = make_moons(75)
XX7, YY7, flat7 = mesh((-1.8, 2.8), (-1.2, 1.9), n=180)
# two hidden RBF centers roughly on each moon
c0 = np.array([0.0, 0.9])
c1 = np.array([1.0, -0.3])
scale0, mix0 = 0.55, 1.0


def nn_rbf_predict(X, scale, mix):
    h0 = np.exp(-((X - c0) ** 2).sum(axis=1) / (2 * scale ** 2))
    h1 = np.exp(-((X - c1) ** 2).sum(axis=1) / (2 * scale ** 2))
    # output logit: favor class1 near c1, class0 near c0
    logits = mix * (h1 - h0) + 0.15 * X[:, 0]
    return (logits >= 0).astype(int), logits


def draw_nn(ax, scale, mix):
    ax.clear()
    pred, logits = nn_rbf_predict(flat7, scale, mix)
    Z = pred.reshape(XX7.shape)
    ax.contourf(XX7, YY7, Z.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP2, alpha=0.95)
    ax.contour(XX7, YY7, logits.reshape(XX7.shape), levels=[0], colors=[C_DARK], linewidths=2.4)
    paint2(ax, X7, y7, s=36)
    ax.scatter([c0[0], c1[0]], [c0[1], c1[1]], s=180, c=[C_GRAY, C_OK],
               marker="P", edgecolors="k", zorder=5)
    ax.set_xlim(-1.8, 2.8); ax.set_ylim(-1.2, 1.9)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    pX, _ = nn_rbf_predict(X7, scale, mix)
    return float(np.mean(pX == y7))


accuracy_meter(ax7m, draw_nn(ax7, scale0, mix0))
sl_sc = Slider(fig7.add_axes([0.12, 0.08, 0.35, 0.035]), r"RBF scale", 0.2, 1.6,
               valinit=scale0, valstep=0.05, color=C_FUN)
sl_mix = Slider(fig7.add_axes([0.55, 0.08, 0.30, 0.035]), r"mix", 0.2, 2.5,
                valinit=mix0, valstep=0.05, color=C_BIN)


def _u7(_=None):
    accuracy_meter(ax7m, draw_nn(ax7, float(sl_sc.val), float(sl_mix.val)))
    fig7.canvas.draw_idle()


sl_sc.on_changed(_u7); sl_mix.on_changed(_u7)
print("Step 7: neural net interactive.")
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Interactive decision stumps (axis-aligned tree vibe)
# -------------------------------------------------------
fig8, (ax8, ax8m) = new_figure(1, 2, gridspec_kw={"width_ratios": [3.2, 1.0]})
fig8.suptitle("Step 8: Decision stumps -- drag two axis cuts", fontsize=FS_FIGURE_TITLE, fontweight="bold")
fig8.subplots_adjust(bottom=0.20, wspace=0.25)

X8, y8 = make_blobs2(60)
XX8, YY8, flat8 = mesh((-3.2, 3.2), (-2.6, 2.6))
cut_x0, cut_y0 = 0.0, 0.0


def stump_predict(X, cut_x, cut_y):
    # simple rule: class1 if x>cut_x OR (x<=cut_x and y>cut_y) -- stump stack
    return ((X[:, 0] > cut_x) | ((X[:, 0] <= cut_x) & (X[:, 1] > cut_y))).astype(int)


def draw_stumps(ax, cut_x, cut_y):
    ax.clear()
    Z = stump_predict(flat8, cut_x, cut_y).reshape(XX8.shape)
    ax.contourf(XX8, YY8, Z.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP2, alpha=0.95)
    paint2(ax, X8, y8, s=44)
    ax.axvline(cut_x, color=C_FUN, lw=2.2)
    ax.axhline(cut_y, color=C_BIN, lw=2.0, ls="--")
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper left")
    return float(np.mean(stump_predict(X8, cut_x, cut_y) == y8))


accuracy_meter(ax8m, draw_stumps(ax8, cut_x0, cut_y0))
sl_cx = Slider(fig8.add_axes([0.12, 0.08, 0.35, 0.035]), r"cut $x$", -2, 2,
               valinit=cut_x0, valstep=0.05, color=C_FUN)
sl_cy = Slider(fig8.add_axes([0.55, 0.08, 0.30, 0.035]), r"cut $y$", -2, 2,
               valinit=cut_y0, valstep=0.05, color=C_BIN)


def _u8(_=None):
    accuracy_meter(ax8m, draw_stumps(ax8, float(sl_cx.val), float(sl_cy.val)))
    fig8.canvas.draw_idle()


sl_cx.on_changed(_u8); sl_cy.on_changed(_u8)
print("Step 8: stumps interactive.")
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 9: Same moons, many discriminative skins
# -------------------------------------------------------
fig9, axes9 = new_figure(2, 3)
fig9.suptitle("Step 9: Same moons -- many discriminative skins", fontsize=FS_FIGURE_TITLE, fontweight="bold")

X9, y9 = make_moons(70)
XX9, YY9, flat9 = mesh((-1.8, 2.8), (-1.2, 1.9), n=170)

# linear (fails)
Zl = (0.4 * flat9[:, 0] + flat9[:, 1] - 0.3 >= 0).reshape(XX9.shape)
# logistic heat at 0.5 same as linear-ish
# knn
Zk9 = knn_predict(X9, y9, flat9, k=9).reshape(XX9.shape)
# rbf net
Zn9, _ = nn_rbf_predict(flat9, 0.5, 1.2)
Zn9 = Zn9.reshape(XX9.shape)
# stump-ish on moons (poor)
Zs9 = stump_predict(flat9, 0.5, 0.2).reshape(XX9.shape)
# svm-like angle
s9, _ = svm_score(flat9, 1.2, 1.0)
Zv9 = (s9 >= 0).reshape(XX9.shape)
# softmax 2-proto
protos2 = np.array([[0.0, 1.0], [1.0, -0.2]])
lab9, _ = softmax_labels(flat9, protos2, 0.4)
Zm9 = lab9.reshape(XX9.shape)

panels = [
    (axes9[0, 0], Zl, "linear / logistic"),
    (axes9[0, 1], Zv9, "SVM direction"),
    (axes9[0, 2], Zm9, "prototype softmax"),
    (axes9[1, 0], Zk9, "k-NN"),
    (axes9[1, 1], Zn9, "neural / RBF"),
    (axes9[1, 2], Zs9, "axis stumps"),
]
for ax, Z, title in panels:
    ax.contourf(XX9, YY9, Z.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP2, alpha=0.95)
    paint2(ax, X9, y9, s=18)
    ax.set_title(title, fontsize=FS_AX_TITLE - 3, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
fig9.tight_layout(rect=[0, 0.02, 1, 0.94])
show_maximized(fig9)


# -------------------------------------------------------
# FIGURE 10: Takeaway collage
# -------------------------------------------------------
fig10, axes10 = new_figure(1, 4)
fig10.suptitle(
    r"Step 10: Discriminative family -- direct $y$ from $x$",
    fontsize=FS_FIGURE_TITLE, fontweight="bold",
)

# list as compact visual icons
icons = [
    (axes10[0], "linear\nlogistic / SVM", CMAP2, lambda: (flat[:, 0] >= 0).reshape(XX.shape), XX, YY, Xg, yg),
    (axes10[1], "multiclass\nsoftmax", CMAP3, None, XX3, YY3, X3g, y3g),
    (axes10[2], "local\nk-NN", CMAP2, lambda: knn_predict(Xm, ym, flatm, 5).reshape(XXm.shape), XXm, YYm, Xm, ym),
    (axes10[3], "nonlinear\nnet / tree", CMAP2, lambda: Zn9, XX9, YY9, X9, y9),
]
# panel 0
axes10[0].contourf(XX, YY, (flat[:, 0] >= 0).reshape(XX.shape).astype(float),
                   levels=[-0.1, 0.5, 1.1], cmap=CMAP2)
paint2(axes10[0], Xg, yg, s=14)
axes10[0].set_title("linear\nlogistic / SVM", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)

# panel 1
axes10[1].contourf(XX3, YY3, Z3, levels=[-0.5, 0.5, 1.5, 2.5], cmap=CMAP3)
paint3(axes10[1], X3g, y3g, s=12)
axes10[1].set_title("multiclass\nsoftmax", fontsize=FS_ANNOT, fontweight="bold", color=C_FUN)

# panel 2
axes10[2].contourf(XXm, YYm, knn_predict(Xm, ym, flatm, 5).reshape(XXm.shape).astype(float),
                   levels=[-0.1, 0.5, 1.1], cmap=CMAP2)
paint2(axes10[2], Xm, ym, s=12)
axes10[2].set_title("local\nk-NN", fontsize=FS_ANNOT, fontweight="bold", color=C_BIN)

# panel 3
axes10[3].contourf(XX9, YY9, Zn9.astype(float), levels=[-0.1, 0.5, 1.1], cmap=CMAP2)
paint2(axes10[3], X9, y9, s=12)
axes10[3].set_title("nonlinear\nnet / tree", fontsize=FS_ANNOT, fontweight="bold", color=C_OK)

for ax in axes10:
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15)
fig10.tight_layout(rect=[0, 0.02, 1, 0.90])
show_maximized(fig10)


print("\n" + "=" * 60)
print("141 DISCRIMINATIVE MODELS -- VISUAL / INTERACTIVE")
print("=" * 60)
print("""
Typical discriminative models (shown):
  logistic regression, softmax, linear SVM,
  k-NN, neural net (RBF), decision stumps / trees

Idea: model P(y|x) or a decision rule -- not full P(x).

See also: 0310_regression_classification.py,
          0740_discriminative_generative.py,
          0820_softmax.py, 0700_neural_non_linearity.py
""")
