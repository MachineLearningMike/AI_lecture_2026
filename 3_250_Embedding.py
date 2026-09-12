"""
What an embedding is -- from discrete IDs to dense vectors you can compute with.

Intuition:
  Tokens / categories / users are symbols.
  An embedding table turns each symbol into a point in R^d.
  Nearby points share behavior; a model learns the geometry.

Sketch:
  1. Discrete IDs need a numeric face
  2. Embedding = learned lookup table (rows are vectors)
  3. One-hot -> matmul W  ==  embedding lookup
  4. Geometry: neighbors, clusters, analogies (preview)
  5. Interactive playground -- pick a token, see its vector & neighbors
  6. Where embeddings show up (tokens, positions, items, ...)
  7. Training moves the rows (tiny seeker)
  8. Takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.widgets import Button, Slider
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

np.random.seed(210)
RNG = np.random.default_rng(210)


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
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_FUN = "#8e44ad"
C_BIN = "#d68910"
C_GRAY = "#85929e"
C_ACTIVE = "#f9e79f"

TOKENS = ["cat", "dog", "king", "queen", "apple", "run", "PAD", "MASK"]
# Rows of embedding table E in R^{V x d} with d=2 for pictures (plus a hidden denser view)
E2 = np.array([
    [-1.6, 1.3],   # cat
    [-1.0, 1.4],   # dog
    [1.5, 1.7],    # king
    [2.2, 1.7],    # queen
    [0.1, -1.5],   # apple
    [1.9, -1.0],   # run
    [0.0, 0.0],    # PAD
    [-0.3, 0.2],   # MASK
], dtype=float)
COLORS = [C_OK, C_OK, C_BIN, C_BIN, C_FUN, C_ACCENT, C_GRAY, C_GRAY]


def cosine(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# -------------------------------------------------------
# FIGURE 1: Discrete IDs need numbers
# -------------------------------------------------------
def build_step_1():
    fig1, axes1 = new_figure(1, 2)
    fig1.suptitle("Step 1: Models eat numbers -- tokens start as discrete IDs", fontsize=FS_FIGURE_TITLE, fontweight="bold")

    axes1[0].barh(np.arange(len(TOKENS))[::-1], np.arange(len(TOKENS))[::-1] * 0 + 1,
                  color=C_GRAY, edgecolor="white")
    axes1[0].set_yticks(np.arange(len(TOKENS))[::-1])
    axes1[0].set_yticklabels([f"{i}: {t}" for i, t in enumerate(TOKENS)][::-1], fontsize=FS_TICK)
    axes1[0].set_xticks([])
    axes1[0].set_title("ID table", fontsize=FS_AX_TITLE - 2)
    axes1[0].set_xlim(0, 1.3)

    # neural net wants a vector
    axes1[1].set_xlim(0, 10)
    axes1[1].set_ylim(0, 10)
    axes1[1].axis("off")
    axes1[1].set_title("ID alone is not a feature vector", fontsize=FS_AX_TITLE - 2)
    round_box(axes1[1], 0.8, 6.5, 3.2, 2.0, C_MUTE, C_DARK, lw=1.8)
    axes1[1].text(2.4, 7.5, "token id = 0\n(\"cat\")", ha="center", va="center",
                  fontsize=FS_ANNOT, fontweight="bold")
    axes1[1].annotate("", xy=(6.0, 7.5), xytext=(4.2, 7.5),
                      arrowprops=dict(arrowstyle="->", lw=2.4, color=C_ACCENT))
    round_box(axes1[1], 6.0, 6.5, 3.4, 2.0, C_ACTIVE, C_BIN, lw=1.8)
    axes1[1].text(7.7, 7.5, "need a vector\nin R^d", ha="center", va="center",
                  fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    round_box(axes1[1], 1.5, 1.5, 7.0, 3.2, C_NEW, "#2874a6", lw=1.8)
    axes1[1].text(5.0, 3.1,
                  "Embedding = the learned map\nID  ->  dense vector",
                  ha="center", va="center", fontsize=FS_AX_TITLE - 2,
                  fontweight="bold", color=C_DARK)
    fig1.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig1


# -------------------------------------------------------
# FIGURE 2: Lookup table
# -------------------------------------------------------
def build_step_2():
    fig2, (ax2a, ax2b) = new_figure(1, 2)
    fig2.suptitle(r"Step 2: Embedding table $E$ -- each row is a vector", fontsize=FS_FIGURE_TITLE, fontweight="bold")

    im = ax2a.imshow(E2, cmap="coolwarm", aspect="auto", vmin=-2.5, vmax=2.5)
    ax2a.set_yticks(range(len(TOKENS)))
    ax2a.set_yticklabels(TOKENS, fontsize=FS_TICK)
    ax2a.set_xticks([0, 1])
    ax2a.set_xticklabels([r"$d_1$", r"$d_2$"], fontsize=FS_TICK)
    ax2a.set_title(r"matrix $E \in \mathbb{R}^{V \times d}$", fontsize=FS_AX_TITLE - 2)
    fig2.colorbar(im, ax=ax2a, fraction=0.046, pad=0.04)

    # highlight lookup for cat
    ax2b.set_xlim(0, 10)
    ax2b.set_ylim(0, 10)
    ax2b.axis("off")
    ax2b.set_title(r"lookup: $e_{\mathrm{cat}} = E[0]$", fontsize=FS_AX_TITLE - 2)
    round_box(ax2b, 0.6, 6.8, 4.0, 2.2, C_GOAL, C_OK, lw=2.0)
    ax2b.text(2.6, 7.9, "id = 0", ha="center", fontsize=FS_ANNOT + 1, fontweight="bold")
    ax2b.annotate("", xy=(6.2, 7.9), xytext=(4.8, 7.9),
                  arrowprops=dict(arrowstyle="->", lw=2.5, color=C_FUN))
    round_box(ax2b, 6.2, 6.8, 3.2, 2.2, C_ACTIVE, C_BIN, lw=2.0)
    ax2b.text(7.8, 7.9, f"[{E2[0, 0]:.1f}, {E2[0, 1]:.1f}]", ha="center",
              fontsize=FS_ANNOT, fontweight="bold", family="monospace")
    # scatter of all rows
    for i, (xy, c) in enumerate(zip(E2, COLORS)):
        ax2b.scatter([3.0 + xy[0] * 0.7], [3.2 + xy[1] * 0.7], s=90, c=c,
                     edgecolors="white", zorder=3)
        ax2b.text(3.1 + xy[0] * 0.7, 3.3 + xy[1] * 0.7, TOKENS[i], fontsize=FS_TICK - 1)
    ax2b.set_xlim(0, 10)
    round_box(ax2b, 0.6, 0.6, 8.8, 1.5, C_MUTE, "0.5", lw=1.4)
    ax2b.text(5.0, 1.35, "Same table is reused every time the token appears.",
              ha="center", va="center", fontsize=FS_ANNOT)
    return fig2


# -------------------------------------------------------
# FIGURE 3: One-hot matmul == embedding lookup
# -------------------------------------------------------
def build_step_3():
    fig3, axes3 = new_figure(1, 3)
    fig3.suptitle(r"Step 3: One-hot $\times$ $E$  is exactly the row lookup", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")

    oh = np.zeros(len(TOKENS))
    oh[0] = 1.0
    axes3[0].imshow(oh[:, None], cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes3[0].set_yticks(range(len(TOKENS)))
    axes3[0].set_yticklabels(TOKENS, fontsize=FS_TICK - 1)
    axes3[0].set_xticks([0])
    axes3[0].set_xticklabels(["1-hot"], fontsize=FS_TICK)
    axes3[0].set_title("select row 0", fontsize=FS_AX_TITLE - 3)

    axes3[1].imshow(E2, cmap="coolwarm", aspect="auto", vmin=-2.5, vmax=2.5)
    axes3[1].set_yticks(range(len(TOKENS)))
    axes3[1].set_yticklabels(TOKENS, fontsize=FS_TICK - 1)
    axes3[1].set_xticks([0, 1])
    axes3[1].set_xticklabels([r"$d_1$", r"$d_2$"], fontsize=FS_TICK)
    axes3[1].set_title("times E", fontsize=FS_AX_TITLE - 3)

    out = oh @ E2
    axes3[2].bar([0, 1], out, color=[C_OK, C_FUN], edgecolor="white")
    axes3[2].set_xticks([0, 1])
    axes3[2].set_xticklabels([r"$d_1$", r"$d_2$"], fontsize=FS_TICK)
    axes3[2].set_title("= embedding vector", fontsize=FS_AX_TITLE - 3)
    axes3[2].set_ylim(min(0, out.min() - 0.3), out.max() + 0.4)
    for i, v in enumerate(out):
        axes3[2].text(i, v + 0.08 * np.sign(v + 1e-9), f"{v:.2f}", ha="center", fontsize=FS_ANNOT)
    axes3[2].grid(True, alpha=0.3, axis="y")
    fig3.tight_layout(rect=[0, 0.02, 1, 0.90])
    return fig3


# -------------------------------------------------------
# FIGURE 4: Geometry preview
# -------------------------------------------------------
def build_step_4():
    fig4, (ax4a, ax4b) = new_figure(1, 2)
    fig4.suptitle("Step 4: Embeddings live in a geometry -- neighbors mean something", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")

    for i, (xy, c, t) in enumerate(zip(E2, COLORS, TOKENS)):
        ax4a.scatter([xy[0]], [xy[1]], s=140, c=c, edgecolors="white", zorder=3)
        ax4a.text(xy[0] + 0.08, xy[1] + 0.08, t, fontsize=FS_TICK, fontweight="bold")
    ax4a.annotate("", xy=E2[1], xytext=E2[0],
                  arrowprops=dict(arrowstyle="<->", color=C_OK, lw=2.2))
    ax4a.set_title("similar tokens sit near each other", fontsize=FS_AX_TITLE - 2)
    ax4a.axhline(0, color="0.85"); ax4a.axvline(0, color="0.85")
    ax4a.set_aspect("equal")
    ax4a.grid(True, alpha=0.25)
    ax4a.set_xlim(-2.5, 3.0); ax4a.set_ylim(-2.2, 2.4)

    # analogy
    ax4b.scatter(E2[:, 0], E2[:, 1], s=40, c=COLORS, alpha=0.35)
    for idx in (2, 3, 0, 1):
        ax4b.scatter([E2[idx, 0]], [E2[idx, 1]], s=150, c=COLORS[idx], edgecolors="k", zorder=3)
        ax4b.text(E2[idx, 0] + 0.08, E2[idx, 1] + 0.08, TOKENS[idx], fontsize=FS_TICK, fontweight="bold")
    ax4b.annotate("", xy=E2[2], xytext=E2[0] * 0 + np.array([1.5, 0.7]),  # fake man-ish
                  arrowprops=dict(arrowstyle="->", color=C_BIN, lw=2))
    # use king-queen horizontal as relation
    ax4b.annotate("", xy=E2[2], xytext=E2[2] - np.array([0.0, 1.0]),
                  arrowprops=dict(arrowstyle="->", color=C_BIN, lw=2.0))
    ax4b.annotate("", xy=E2[3], xytext=E2[3] - np.array([0.0, 1.0]),
                  arrowprops=dict(arrowstyle="->", color=C_FUN, lw=2.0))
    ax4b.set_title("relations can look like arrows", fontsize=FS_AX_TITLE - 2)
    ax4b.set_aspect("equal")
    ax4b.grid(True, alpha=0.25)
    ax4b.set_xlim(-2.5, 3.0); ax4b.set_ylim(-2.2, 2.4)
    fig4.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig4


# -------------------------------------------------------
# FIGURE 5: Interactive playground
# -------------------------------------------------------
def build_step_5():
    fig5, (ax5a, ax5b) = new_figure(1, 2, gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig5.suptitle("Step 5: Playground -- pick a token, inspect its embedding", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    fig5.subplots_adjust(bottom=0.20, wspace=0.28)


    def draw_play(idx, dim_scale=1.0):
        ax5a.clear(); ax5b.clear()
        E = E2 * np.array([1.0, dim_scale])
        idx = int(idx)
        for i, (xy, c, t) in enumerate(zip(E, COLORS, TOKENS)):
            s = 200 if i == idx else 110
            ec = "k" if i == idx else "white"
            ax5a.scatter([xy[0]], [xy[1]], s=s, c=c, edgecolors=ec, linewidths=1.2, zorder=3)
            ax5a.text(xy[0] + 0.08, xy[1] + 0.08, t, fontsize=FS_TICK,
                      fontweight="bold" if i == idx else "normal")
        ax5a.add_patch(Circle(E[idx], 0.55, fill=False, ls="--", color=C_BIN, lw=1.5))
        ax5a.axhline(0, color="0.85"); ax5a.axvline(0, color="0.85")
        ax5a.set_xlim(-2.8, 3.2); ax5a.set_ylim(-2.4, 2.6)
        ax5a.set_aspect("equal")
        ax5a.grid(True, alpha=0.25)
        ax5a.set_title(f"token = {TOKENS[idx]}  (id {idx})", fontsize=FS_AX_TITLE - 2)

        # vector components + neighbors
        ax5b.axis("off")
        ax5b.set_xlim(0, 1); ax5b.set_ylim(0, 1)
        ax5b.set_title("vector + nearest", fontsize=FS_AX_TITLE - 2, fontweight="bold")
        round_box(ax5b, 0.08, 0.70, 0.84, 0.22, C_MUTE, C_DARK, lw=1.4)
        ax5b.text(0.5, 0.81, f"E[{idx}] = [{E[idx, 0]:.2f}, {E[idx, 1]:.2f}]",
                  ha="center", va="center", fontsize=FS_ANNOT, family="monospace",
                  transform=ax5b.transAxes)
        sims = sorted(
            ((cosine(E[idx], E[j]), j) for j in range(len(TOKENS)) if j != idx),
            reverse=True,
        )
        for i, (s, j) in enumerate(sims[:5]):
            y = 0.58 - i * 0.11
            ax5b.barh([y], [max(0.05, (s + 1) / 2)], height=0.07, left=0.08,
                      color=COLORS[j], align="center")
            ax5b.text(0.10, y, TOKENS[j], va="center", fontsize=FS_ANNOT - 1,
                      color="white", fontweight="bold", transform=ax5b.transAxes)
            ax5b.text(0.90, y, f"{s:.2f}", va="center", ha="right", fontsize=FS_TICK,
                      transform=ax5b.transAxes)


    ax_tok = fig5.add_axes([0.12, 0.07, 0.40, 0.04])
    ax_dim = fig5.add_axes([0.60, 0.07, 0.28, 0.04])
    sl_tok = Slider(ax_tok, "token", 0, len(TOKENS) - 1, valinit=0, valstep=1, color=C_FUN)
    sl_dim = Slider(ax_dim, "stretch d2", 0.4, 1.8, valinit=1.0, valstep=0.05, color=C_BIN)


    def _u5(_=None):
        draw_play(sl_tok.val, float(sl_dim.val))
        fig5.canvas.draw_idle()


    sl_tok.on_changed(_u5)
    sl_dim.on_changed(_u5)
    _u5()
    print("Step 5: embedding playground.")
    return fig5


# -------------------------------------------------------
# FIGURE 6: Where embeddings appear
# -------------------------------------------------------
def build_step_6():
    fig6, axes6 = new_figure(2, 3)
    fig6.suptitle("Step 6: Embeddings show up everywhere discrete meets deep learning", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")

    panels = [
        (axes6[0, 0], "token / word", "WordPiece, BPE ids\n-> E_tok[id]", C_NEW, "#2874a6"),
        (axes6[0, 1], "position", "absolute / relative\nPE vectors", C_GOAL, C_OK),
        (axes6[0, 2], "segment / type", "sentence A vs B\n(BERT)", "#f5eef8", C_FUN),
        (axes6[1, 0], "user / item", "recommenders\nuser & movie rows", "#fdebd0", C_BIN),
        (axes6[1, 1], "graph node", "node2vec / GNN\ninput rows", "#fadbd8", C_ACCENT),
        (axes6[1, 2], "codebook", "VQ / audio units\ndiscrete latents", C_MUTE, C_DARK),
    ]
    for ax, title, body, face, edge in panels:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
        round_box(ax, 0.6, 1.5, 8.8, 7.0, face, edge, lw=2.0)
        ax.text(5.0, 6.8, title, ha="center", fontsize=FS_AX_TITLE - 2,
                fontweight="bold", color=edge)
        ax.text(5.0, 4.2, body, ha="center", va="center", fontsize=FS_ANNOT + 1)
    return fig6


# -------------------------------------------------------
# FIGURE 7: Training moves rows (seeker)
# -------------------------------------------------------
def build_step_7():
    fig7, (ax7a, ax7b) = new_figure(1, 2)
    fig7.suptitle("Step 7: Learning = moving rows of E so the task gets easier", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    fig7.subplots_adjust(bottom=0.20, wspace=0.28)

    E_start = E2 + RNG.normal(0, 1.0, size=E2.shape)
    E_end = E2.copy()


    def lerp_E(t):
        return (1 - t) * E_start + t * E_end


    def task_score(E):
        # toy: want cat near dog, king near queen
        return 0.5 * (cosine(E[0], E[1]) + cosine(E[2], E[3]))


    def draw_train(t):
        E = lerp_E(t)
        ax7a.clear(); ax7b.clear()
        for i, (xy, c, tok) in enumerate(zip(E, COLORS, TOKENS)):
            ax7a.scatter([xy[0]], [xy[1]], s=120, c=c, edgecolors="white", zorder=3)
            ax7a.text(xy[0] + 0.08, xy[1] + 0.08, tok, fontsize=FS_TICK - 1)
        ax7a.annotate("", xy=E[1], xytext=E[0],
                      arrowprops=dict(arrowstyle="<->", color=C_OK, lw=1.8))
        ax7a.annotate("", xy=E[3], xytext=E[2],
                      arrowprops=dict(arrowstyle="<->", color=C_BIN, lw=1.8))
        ax7a.set_xlim(-3, 3.5); ax7a.set_ylim(-3, 3)
        ax7a.set_aspect("equal"); ax7a.grid(True, alpha=0.25)
        ax7a.set_title(f"rows of E  (t={t:.2f})", fontsize=FS_AX_TITLE - 2)

        ts = np.linspace(0, 1, 25)
        scores = [task_score(lerp_E(tt)) for tt in ts]
        ax7b.plot(ts, scores, color=C_FUN, lw=2.4)
        ax7b.scatter([t], [task_score(E)], s=80, c=C_ACCENT, zorder=3)
        ax7b.set_xlim(0, 1); ax7b.set_ylim(0, 1.05)
        ax7b.set_xlabel("training progress", fontsize=FS_AXIS)
        ax7b.set_ylabel("neighbor cosine (toy)", fontsize=FS_AXIS)
        ax7b.set_title("geometry improves with learning", fontsize=FS_AX_TITLE - 2)
        ax7b.grid(True, alpha=0.3)


    ax_tr = fig7.add_axes([0.22, 0.06, 0.40, 0.04])
    ax_p = fig7.add_axes([0.155, 0.058, 0.035, 0.028])
    ax_n = fig7.add_axes([0.650, 0.058, 0.035, 0.028])
    sl_tr = Slider(ax_tr, "train", 0.0, 1.0, valinit=0.0, valstep=0.05, color=C_FUN)
    btn_p = Button(ax_p, "\u2190", color=C_MUTE, hovercolor=C_NEW)
    btn_n = Button(ax_n, "\u2192", color=C_MUTE, hovercolor=C_GOAL)
    btn_p.label.set_fontsize(8)
    btn_n.label.set_fontsize(8)


    def _set_tr(t):
        t = float(np.clip(t, 0.0, 1.0))
        if abs(float(sl_tr.val) - t) < 1e-12:
            draw_train(t); fig7.canvas.draw_idle()
        else:
            sl_tr.set_val(t)


    def _u7(_=None):
        draw_train(float(sl_tr.val)); fig7.canvas.draw_idle()


    def _prev(_=None):
        _set_tr(float(sl_tr.val) - 0.1)


    def _next(_=None):
        _set_tr(float(sl_tr.val) + 0.1)


    def _key7(event):
        if event.key in ("right", " ", "n", "enter"):
            _next()
        elif event.key in ("left", "p", "backspace"):
            _prev()


    sl_tr.on_changed(_u7)
    btn_p.on_clicked(_prev)
    btn_n.on_clicked(_next)
    fig7.canvas.mpl_connect("key_press_event", _key7)
    _u7()
    print("Step 7: training seeker.")
    return fig7


# -------------------------------------------------------
# FIGURE 8: Takeaway
# -------------------------------------------------------
def build_step_8():
    fig8, axes8 = new_figure(1, 3)
    fig8.suptitle("Step 8: Embedding = learned ID-to-vector dictionary", fontsize=FS_FIGURE_TITLE, fontweight="bold")

    axes8[0].imshow(E2, cmap="coolwarm", aspect="auto", vmin=-2.5, vmax=2.5)
    axes8[0].set_yticks(range(len(TOKENS)))
    axes8[0].set_yticklabels(TOKENS, fontsize=FS_TICK - 1)
    axes8[0].set_title("table of vectors", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)

    axes8[1].scatter(E2[:, 0], E2[:, 1], s=90, c=COLORS, edgecolors="white")
    for xy, t in zip(E2, TOKENS):
        axes8[1].text(xy[0] + 0.06, xy[1] + 0.06, t, fontsize=FS_TICK - 1)
    axes8[1].set_aspect("equal")
    axes8[1].set_title("geometry", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    axes8[1].grid(True, alpha=0.25)

    axes8[2].set_xlim(0, 10); axes8[2].set_ylim(0, 10); axes8[2].axis("off")
    round_box(axes8[2], 0.6, 1.5, 8.8, 7.0, C_GOAL, C_OK, lw=2.0)
    axes8[2].text(5.0, 6.5, "then compute", ha="center", fontsize=FS_AX_TITLE - 2,
                  fontweight="bold", color=C_OK)
    axes8[2].text(5.0, 4.2,
                  "attention, MLP,\nsimilarity, loss,\ngeneration...",
                  ha="center", va="center", fontsize=FS_ANNOT + 1)
    fig8.tight_layout(rect=[0, 0.02, 1, 0.90])
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
print("210 EMBEDDING -- ID TO DENSE VECTOR")
print("=" * 60)
print("""
Embedding = learned lookup table E[id] -> vector in R^d.
One-hot @ E is the same lookup.
Geometry of rows carries meaning; training moves the rows.

, ,
          0880_positional_encoding.py, 0950_BERT.py
""")
