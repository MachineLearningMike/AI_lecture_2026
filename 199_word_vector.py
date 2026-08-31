"""
199_word_vector.py

Word2Vec, explained intuitively.

Big idea:
  A word is known by the company it keeps.
  Train vectors so words that share contexts sit near each other,
  and so a word can predict its neighbors (skip-gram) or be predicted
  from them (CBOW).

Sketch:
  1. Words as IDs -- discrete labels, no geometry
  2. Context window -- company of a word
  3. Similar company -> similar vectors
  4. Skip-gram vs CBOW (two training games)
  5. Score a neighbor with a dot product + softmax
  6. Interactive 2D embedding playground
  7. Analogy arrows -- king - man + woman ~ queen
  8. Training moves the points
  9. Takeaway collage

Companion to 198_back_propagation.py, 320_softmax.py, 330_QKV_attention.py,
260_transformer.py.
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
)
apply_rcparams(plt)
import numpy as np

np.random.seed(199)
RNG = np.random.default_rng(199)


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


# Tiny vocabulary for demos
WORDS = [
    "king", "queen", "man", "woman",
    "cat", "dog", "fish",
    "apple", "orange",
    "walk", "run",
]
# Hand-placed 2D embeddings with readable clusters + analogy geometry
EMB0 = {
    "king": np.array([1.6, 1.8]),
    "queen": np.array([2.4, 1.8]),
    "man": np.array([1.6, 0.7]),
    "woman": np.array([2.4, 0.7]),
    "cat": np.array([-1.7, 1.4]),
    "dog": np.array([-1.0, 1.5]),
    "fish": np.array([-1.4, 0.5]),
    "apple": np.array([-0.2, -1.5]),
    "orange": np.array([0.5, -1.4]),
    "walk": np.array([1.8, -1.2]),
    "run": np.array([2.4, -0.9]),
}
CLUSTER_COLOR = {
    "king": C_BIN, "queen": C_BIN, "man": C_BIN, "woman": C_BIN,
    "cat": C_OK, "dog": C_OK, "fish": C_OK,
    "apple": C_FUN, "orange": C_FUN,
    "walk": C_ACCENT, "run": C_ACCENT,
}


def cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def paint_embed(ax, emb, highlight=None, title=None):
    ax.clear()
    for w, xy in emb.items():
        face = C_ACTIVE if (highlight and w in highlight) else CLUSTER_COLOR[w]
        ax.scatter([xy[0]], [xy[1]], s=160 if (highlight and w in highlight) else 110,
                   c=face, edgecolors="white", linewidths=0.8, zorder=3)
        ax.text(xy[0] + 0.08, xy[1] + 0.08, w, fontsize=FS_TICK,
                color=C_DARK, fontweight="bold" if (highlight and w in highlight) else "normal")
    ax.axhline(0, color="0.8", lw=1)
    ax.axvline(0, color="0.8", lw=1)
    ax.set_xlim(-2.6, 3.2)
    ax.set_ylim(-2.2, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title, fontsize=FS_AX_TITLE - 2, fontweight="bold")


# -------------------------------------------------------
# FIGURE 1: Words as IDs
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle("Step 1: Raw words are IDs -- labels, not meanings", fontsize=FS_FIGURE_TITLE, fontweight="bold")

ids = np.arange(len(WORDS))
axes1[0].barh(ids[::-1], np.ones(len(WORDS)), color=C_GRAY, edgecolor="white")
axes1[0].set_yticks(ids[::-1])
axes1[0].set_yticklabels(WORDS[::-1], fontsize=FS_TICK)
axes1[0].set_xlim(0, 1.2)
axes1[0].set_xticks([])
axes1[0].set_title("vocabulary list", fontsize=FS_AX_TITLE - 2)

# one-hot matrix cartoon
oh = np.eye(min(8, len(WORDS)))
axes1[1].imshow(oh, cmap="Blues", aspect="equal")
axes1[1].set_xticks(range(oh.shape[1]))
axes1[1].set_yticks(range(oh.shape[0]))
axes1[1].set_xticklabels([f"d{i}" for i in range(oh.shape[1])], fontsize=FS_TICK - 1)
axes1[1].set_yticklabels(WORDS[:oh.shape[0]], fontsize=FS_TICK - 1)
axes1[1].set_title("one-hot rows (orthogonal)", fontsize=FS_AX_TITLE - 2)
for i in range(oh.shape[0]):
    for j in range(oh.shape[1]):
        axes1[1].text(j, i, f"{int(oh[i, j])}", ha="center", va="center",
                      color="white" if oh[i, j] > 0.5 else C_DARK, fontsize=FS_TICK)
fig1.tight_layout(rect=[0, 0.02, 1, 0.92])
axes1[0].text(
    0.5, -0.12,
    r"king $\perp$ queen in one-hot space -- distance ignores meaning",
    transform=axes1[0].transAxes, ha="center", fontsize=FS_ANNOT, color=C_ACCENT,
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Context window
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle('Step 2: "You shall know a word by the company it keeps"', fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
blank_slide(ax2)

sentence = ["the", "cat", "sat", "on", "the", "mat", "near", "dog"]
center = 1  # cat
win = 2
x0 = 0.7
for i, tok in enumerate(sentence):
    x = x0 + i * 1.4
    if i == center:
        face, edge = C_ACTIVE, C_BIN
    elif abs(i - center) <= win:
        face, edge = C_GOAL, C_OK
    else:
        face, edge = C_MUTE, "0.55"
    round_box(ax2, x, 4.6, 1.25, 1.5, face, edge, lw=2.0)
    ax2.text(x + 0.625, 5.35, tok, ha="center", va="center",
             fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)

# bracket over context
ax2.annotate(
    "", xy=(x0 + (center - win) * 1.4, 4.4),
    xytext=(x0 + (center + win) * 1.4 + 1.25, 4.4),
    arrowprops=dict(arrowstyle="-", color=C_OK, lw=3.0),
)
ax2.text(6.0, 3.7, "context window around cat", ha="center",
         fontsize=FS_AX_TITLE - 2, color=C_OK, fontweight="bold")

# co-occur chips
pairs = [("cat", "sat"), ("cat", "the"), ("cat", "on"), ("dog", "near"), ("dog", "the")]
for i, (a, b) in enumerate(pairs):
    x = 0.8 + (i % 5) * 2.2
    y = 1.6 if i < 5 else 0.4
    round_box(ax2, x, y, 2.0, 1.1, C_NEW, "#2874a6", lw=1.5)
    ax2.text(x + 1.0, y + 0.55, f"{a} ~ {b}", ha="center", va="center",
             fontsize=FS_ANNOT, family="monospace", color=C_DARK)

ax2.text(6.0, 0.55, "Word2Vec trains on many such (word, neighbor) pairs from a corpus.",
         ha="center", fontsize=FS_ANNOT + 1)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Similar company -> similar vectors
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle("Step 3: Shared contexts pull words together in vector space", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")

# co-occurrence sketch counts
ctx_words = ["sat", "mat", "meow", "bark", "walk", "run"]
rows = ["cat", "dog", "king"]
# fake PMI-ish matrix
M = np.array([
    [3, 2, 4, 0, 1, 0],
    [1, 1, 0, 4, 2, 1],
    [0, 0, 0, 0, 1, 1],
], dtype=float)
im = axes3[0].imshow(M, cmap="YlGn", aspect="auto")
axes3[0].set_xticks(range(len(ctx_words)))
axes3[0].set_xticklabels(ctx_words, fontsize=FS_TICK, rotation=30)
axes3[0].set_yticks(range(len(rows)))
axes3[0].set_yticklabels(rows, fontsize=FS_TICK)
axes3[0].set_title("who shows up near whom", fontsize=FS_AX_TITLE - 2)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        axes3[0].text(j, i, f"{int(M[i, j])}", ha="center", va="center",
                      fontsize=FS_TICK, color=C_DARK)

paint_embed(axes3[1], EMB0, highlight={"cat", "dog", "king"},
            title="embedding plane (intuition)")
# pull arrow cat-dog
axes3[1].annotate(
    "", xy=EMB0["dog"], xytext=EMB0["cat"],
    arrowprops=dict(arrowstyle="<->", color=C_OK, lw=2.2),
)
axes3[1].text(-1.5, 1.95, "similar company", color=C_OK, fontsize=FS_TICK, fontweight="bold")
fig3.tight_layout(rect=[0, 0.02, 1, 0.92])
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Skip-gram vs CBOW
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 2)
fig4.suptitle("Step 4: Two training games -- skip-gram vs CBOW", fontsize=FS_FIGURE_TITLE, fontweight="bold")

# Skip-gram: center -> context
ax = axes4[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Skip-gram: center predicts neighbors", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_FUN)
round_box(ax, 3.5, 4.2, 3.0, 1.6, C_ACTIVE, C_BIN, lw=2.2)
ax.text(5.0, 5.0, "cat", ha="center", va="center", fontsize=FS_AX_TITLE - 2, fontweight="bold")
for xy, lab in [((1.0, 7.5), "the"), ((5.0, 8.0), "sat"), ((8.2, 7.3), "on"), ((1.2, 2.0), "mat")]:
    round_box(ax, xy[0], xy[1], 1.8, 1.1, C_GOAL, C_OK, lw=1.6)
    ax.text(xy[0] + 0.9, xy[1] + 0.55, lab, ha="center", va="center", fontsize=FS_ANNOT)
    ax.annotate("", xy=(xy[0] + 0.9, xy[1] + (0.0 if xy[1] > 5 else 1.1)),
                xytext=(5.0, 5.8 if xy[1] > 5 else 4.2),
                arrowprops=dict(arrowstyle="->", color=C_FUN, lw=2.0))
ax.text(5.0, 0.6, "one word  ->  many context guesses", ha="center", fontsize=FS_ANNOT, color=C_FUN)

# CBOW: context -> center
ax = axes4[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("CBOW: neighbors predict center", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
round_box(ax, 3.5, 4.2, 3.0, 1.6, C_ACTIVE, C_BIN, lw=2.2)
ax.text(5.0, 5.0, "cat", ha="center", va="center", fontsize=FS_AX_TITLE - 2, fontweight="bold")
for xy, lab in [((1.0, 7.5), "the"), ((5.0, 8.0), "sat"), ((8.2, 7.3), "on"), ((1.2, 2.0), "mat")]:
    round_box(ax, xy[0], xy[1], 1.8, 1.1, C_NEW, "#2874a6", lw=1.6)
    ax.text(xy[0] + 0.9, xy[1] + 0.55, lab, ha="center", va="center", fontsize=FS_ANNOT)
    ax.annotate("", xy=(5.0, 5.8 if xy[1] > 5 else 4.2),
                xytext=(xy[0] + 0.9, xy[1] + (0.0 if xy[1] > 5 else 1.1)),
                arrowprops=dict(arrowstyle="->", color=C_OK, lw=2.0))
ax.text(5.0, 0.6, "many words  ->  one center guess", ha="center", fontsize=FS_ANNOT, color=C_OK)
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Dot product scores neighbors
# -------------------------------------------------------
fig5, (ax5a, ax5b) = new_figure(1, 2)
fig5.suptitle(r"Step 5: Neighbor score $\approx$ dot product -- then softmax", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")

center = "cat"
scores = {w: float(np.dot(EMB0[center], EMB0[w])) for w in WORDS if w != center}
order = sorted(scores, key=scores.get, reverse=True)
ypos = np.arange(len(order))
ax5a.barh(ypos[::-1], [scores[w] for w in order],
          color=[C_OK if i < 3 else C_GRAY for i in range(len(order))])
ax5a.set_yticks(ypos[::-1])
ax5a.set_yticklabels(order, fontsize=FS_TICK)
ax5a.set_xlabel(r"$\mathbf{v}_{cat}\cdot\mathbf{v}_{w}$", fontsize=FS_AXIS)
ax5a.set_title("raw scores from cat", fontsize=FS_AX_TITLE - 2)
ax5a.grid(True, alpha=0.3, axis="x")

logits = np.array([scores[w] for w in order], dtype=float)
logits = logits - logits.max()
prob = np.exp(logits) / np.exp(logits).sum()
ax5b.bar(np.arange(len(order)), prob, color=[C_FUN if i < 3 else C_GRAY for i in range(len(order))])
ax5b.set_xticks(np.arange(len(order)))
ax5b.set_xticklabels(order, fontsize=FS_TICK - 1, rotation=45, ha="right")
ax5b.set_ylabel("softmax probability", fontsize=FS_AXIS)
ax5b.set_title("skip-gram style P(context | cat)", fontsize=FS_AX_TITLE - 2)
ax5b.grid(True, alpha=0.3, axis="y")
fig5.tight_layout(rect=[0, 0.05, 1, 0.92])
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Interactive embedding playground
# -------------------------------------------------------
fig6, (ax6, ax6b) = new_figure(1, 2, gridspec_kw={"width_ratios": [1.5, 1.0]})
fig6.suptitle("Step 6: Embedding playground -- pick a word, watch neighbors", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
fig6.subplots_adjust(bottom=0.22, wspace=0.28)

word_idx0 = WORDS.index("king")
emb_live = {w: EMB0[w].copy() for w in WORDS}


def draw_playground(idx):
    w = WORDS[int(idx)]
    paint_embed(ax6, emb_live, highlight={w}, title=f"query: {w}")
    # nearest by cosine
    sims = sorted(
        ((cosine(emb_live[w], emb_live[o]), o) for o in WORDS if o != w),
        reverse=True,
    )
    ax6b.clear()
    ax6b.axis("off")
    ax6b.set_xlim(0, 1)
    ax6b.set_ylim(0, 1)
    ax6b.set_title("nearest by cosine", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    for i, (s, o) in enumerate(sims[:6]):
        y = 0.88 - i * 0.13
        ax6b.barh([y], [max(s, 0.01)], height=0.08, left=0.05,
                  color=CLUSTER_COLOR[o], align="center")
        ax6b.text(0.08, y, f"{o}", va="center", ha="left", fontsize=FS_ANNOT,
                  color="white", fontweight="bold")
        ax6b.text(0.92, y, f"{s:.2f}", va="center", ha="right", fontsize=FS_ANNOT,
                  color=C_DARK)
    # ring around query
    xy = emb_live[w]
    ax6.add_patch(Circle(xy, 0.55, fill=False, ls="--", color=C_BIN, lw=1.5))


# temperature-like slider: blend toward random (noise) to show structure mattering
noise_amt0 = 0.0
noise_base = {w: RNG.normal(0, 1, size=2) for w in WORDS}


def apply_noise(amount):
    for w in WORDS:
        emb_live[w] = (1 - amount) * EMB0[w] + amount * noise_base[w] * 1.5


ax_word = fig6.add_axes([0.12, 0.08, 0.40, 0.04])
ax_noise = fig6.add_axes([0.60, 0.08, 0.28, 0.04])
sl_word = Slider(ax_word, "word", 0, len(WORDS) - 1, valinit=word_idx0, valstep=1, color=C_FUN)
sl_noise = Slider(ax_noise, "scramble", 0.0, 1.0, valinit=noise_amt0, valstep=0.05, color=C_BIN)


def _u6(_=None):
    apply_noise(float(sl_noise.val))
    draw_playground(int(sl_word.val))
    # update word slider label-ish via title already
    fig6.canvas.draw_idle()


sl_word.on_changed(_u6)
sl_noise.on_changed(_u6)
_u6()
print("Step 6: drag word index / scramble.")
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Analogy arrows
# -------------------------------------------------------
fig7, (ax7a, ax7b) = new_figure(1, 2)
fig7.suptitle("Step 7: Relations as arrows -- king - man + woman ~ queen", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")

paint_embed(ax7a, EMB0, highlight={"king", "queen", "man", "woman"}, title="gender arrow transfers")
for a, b, col in [("man", "king", C_BIN), ("woman", "queen", C_FUN)]:
    ax7a.annotate(
        "", xy=EMB0[b], xytext=EMB0[a],
        arrowprops=dict(arrowstyle="->", color=col, lw=2.4, mutation_scale=14),
    )
ax7a.text(1.05, 1.25, "royal", color=C_BIN, fontsize=FS_TICK, fontweight="bold")

# vector arithmetic panel
ax7b.set_xlim(0, 10)
ax7b.set_ylim(0, 10)
ax7b.axis("off")
ax7b.set_title("vector arithmetic", fontsize=FS_AX_TITLE - 2, fontweight="bold")
expr = EMB0["king"] - EMB0["man"] + EMB0["woman"]
# nearest to expr
sims = sorted(((cosine(expr, EMB0[w]), w) for w in WORDS), reverse=True)
round_box(ax7b, 0.8, 6.5, 8.4, 2.4, C_MUTE, C_DARK, lw=1.6)
ax7b.text(5.0, 7.7, "king - man + woman", ha="center", fontsize=FS_AX_TITLE - 2,
          fontweight="bold", family="monospace", color=C_DARK)
ax7b.text(5.0, 6.95, f"= ({expr[0]:.2f}, {expr[1]:.2f})", ha="center",
          fontsize=FS_ANNOT, family="monospace")

y = 5.2
for i, (s, w) in enumerate(sims[:5]):
    face = C_GOAL if i == 0 else C_NEW
    round_box(ax7b, 1.5, y - i * 1.0, 7.0, 0.85, face, C_OK if i == 0 else "#2874a6", lw=1.4)
    ax7b.text(5.0, y - i * 1.0 + 0.42, f"{i + 1}. {w}   cos={s:.2f}",
              ha="center", va="center", fontsize=FS_ANNOT, family="monospace")
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Training moves the points (interactive seeker)
# -------------------------------------------------------
fig8, (ax8a, ax8b) = new_figure(1, 2)
fig8.suptitle("Step 8: Training moves vectors so true neighbors score higher", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
fig8.subplots_adjust(bottom=0.20, wspace=0.28)

# Simulate a tiny skip-gram GD on 2D for ("cat","sat"-like) using dog as positive for cat
# Start from scrambled animal cluster, end near EMB0
start = {w: EMB0[w] + RNG.normal(0, 1.1, size=2) for w in WORDS}
end = {w: EMB0[w].copy() for w in WORDS}
N_EPOCH = 12


def lerp_emb(t):
    return {w: (1 - t) * start[w] + t * end[w] for w in WORDS}


def neighbor_prob(emb, center="cat", positive="dog"):
    logits = np.array([np.dot(emb[center], emb[w]) for w in WORDS if w != center])
    names = [w for w in WORDS if w != center]
    logits = logits - logits.max()
    p = np.exp(logits) / np.exp(logits).sum()
    return names, p, float(p[names.index(positive)])


def draw_train(t):
    emb = lerp_emb(t)
    paint_embed(ax8a, emb, highlight={"cat", "dog"}, title=f"epoch vibe t={t:.2f}")
    names, p, p_pos = neighbor_prob(emb)
    ax8b.clear()
    order = np.argsort(-p)
    ax8b.barh(np.arange(len(names)), p[order],
              color=[C_OK if names[i] == "dog" else C_GRAY for i in order])
    ax8b.set_yticks(np.arange(len(names)))
    ax8b.set_yticklabels([names[i] for i in order], fontsize=FS_TICK)
    ax8b.set_xlim(0, max(0.4, p.max() * 1.15))
    ax8b.set_xlabel("P(context | cat)", fontsize=FS_AXIS)
    ax8b.set_title(f"P(dog | cat) = {p_pos:.2f}", fontsize=FS_AX_TITLE - 2, color=C_OK)
    ax8b.grid(True, alpha=0.3, axis="x")


ax_t = fig8.add_axes([0.22, 0.06, 0.40, 0.04])
ax_prev = fig8.add_axes([0.10, 0.055, 0.08, 0.045])
ax_next = fig8.add_axes([0.66, 0.055, 0.08, 0.045])
sl_t = Slider(ax_t, "train", 0.0, 1.0, valinit=0.0, valstep=0.05, color=C_FUN)
btn_prev = Button(ax_prev, "Prev", color=C_MUTE, hovercolor=C_NEW)
btn_next = Button(ax_next, "Next", color=C_MUTE, hovercolor=C_GOAL)


def _set_t(t):
    t = float(np.clip(t, 0.0, 1.0))
    if abs(float(sl_t.val) - t) < 1e-12:
        draw_train(t)
        fig8.canvas.draw_idle()
    else:
        sl_t.set_val(t)


def _u8(_=None):
    draw_train(float(sl_t.val))
    fig8.canvas.draw_idle()


def _prev(_=None):
    _set_t(float(sl_t.val) - 0.1)


def _next(_=None):
    _set_t(float(sl_t.val) + 0.1)


def _on_key8(event):
    if event.key in ("right", " ", "enter", "n"):
        _next()
    elif event.key in ("left", "p", "backspace"):
        _prev()


sl_t.on_changed(_u8)
btn_prev.on_clicked(_prev)
btn_next.on_clicked(_next)
fig8.canvas.mpl_connect("key_press_event", _on_key8)
_u8()
print("Step 8: train seeker -- Next/Prev or slider.")
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 9: Takeaway collage
# -------------------------------------------------------
fig9, axes9 = new_figure(2, 2)
fig9.suptitle("Step 9: Word2Vec in one glance", fontsize=FS_FIGURE_TITLE, fontweight="bold")

# company
axes9[0, 0].set_xlim(0, 10)
axes9[0, 0].set_ylim(0, 10)
axes9[0, 0].axis("off")
axes9[0, 0].set_title("1. use company", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
for i, tok in enumerate(["the", "cat", "sat", "on"]):
    face = C_ACTIVE if tok == "cat" else C_GOAL
    round_box(axes9[0, 0], 1.0 + i * 2.0, 4.0, 1.8, 1.4, face, C_DARK, lw=1.5)
    axes9[0, 0].text(1.9 + i * 2.0, 4.7, tok, ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold")

# geometry
paint_embed(axes9[0, 1], EMB0, title="2. geometry of meaning")

# predict
names, p, _ = neighbor_prob(EMB0, "cat", "dog")
order = np.argsort(-p)[:5]
axes9[1, 0].bar([names[i] for i in order], p[order], color=C_FUN)
axes9[1, 0].set_title("3. predict neighbors", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_FUN)
axes9[1, 0].tick_params(axis="x", labelrotation=30)
axes9[1, 0].set_ylim(0, max(p) * 1.25)

# analogy
paint_embed(axes9[1, 1], EMB0, highlight={"king", "queen", "man", "woman"}, title="4. relations as arrows")
axes9[1, 1].annotate("", xy=EMB0["king"], xytext=EMB0["man"],
                     arrowprops=dict(arrowstyle="->", color=C_BIN, lw=2))
axes9[1, 1].annotate("", xy=EMB0["queen"], xytext=EMB0["woman"],
                     arrowprops=dict(arrowstyle="->", color=C_FUN, lw=2))
fig9.tight_layout(rect=[0, 0.02, 1, 0.94])
show_maximized(fig9)


print("\n" + "=" * 60)
print("199 WORD VECTOR -- WORD2VEC INTUITION")
print("=" * 60)
print("""
Word2Vec intuition:
  1) A word is known by its company (context window).
  2) Shared contexts -> nearby vectors.
  3) Skip-gram: center predicts neighbors.
     CBOW: neighbors predict center.
  4) Dot product + softmax = neighbor probabilities.
  5) Analogies emerge as vector arrows.

See also: 198_back_propagation.py, 320_softmax.py,
          330_QKV_attention.py, 260_transformer.py
""")
