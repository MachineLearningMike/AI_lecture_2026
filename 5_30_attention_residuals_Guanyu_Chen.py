"""
5_30_attention_residuals.py

Attention Residuals (AttnRes): replacing the fixed residual sum with softmax
attention over depth.

Paper
  Kimi Team (first author: Guangyu Chen), "Attention Residuals",
  arXiv:2603.15031 [cs.CL], v1 submitted 16 March 2026.
  https://arxiv.org/abs/2603.15031   code: https://github.com/MoonshotAI/Attention-Residuals
  (arXiv lists "Kimi Team" as the first author entry; Guangyu Chen is the first
  named individual author. The paper's Appendix A orders authors by contribution.)

Follows 5_16_chain_of_adds.py (the residual stream as a chain of additions).

Slides:
  0. Title and citation                       13. Two-phase inference + online softmax
  1. The idea in one picture                  13.1 Code: the two-phase merge is exact
  2. Standard residuals, unrolled             14. Memory I/O per layer (Table 1)
  3. PreNorm dilution (live toy)              15. The 48B Kimi Linear experiment
  4. Highway and the three limitations        16. Scaling laws (Fig. 4, Table 2)
  5. The duality of time and depth            17. Training dynamics (Fig. 5 + toy)
  6. Full Attention Residuals                 18. Downstream benchmarks (Table 3)
  6.1 Code: Full AttnRes in numpy             19. Ablations and block size (Tab. 4, Fig. 6)
  7. Depth-attention weights (live toy)       20. AttnRes prefers deeper networks (Fig. 7)
  8. Why Full AttnRes is hard at scale        21. What the learned weights look like (Fig. 8)
  9. Block Attention Residuals                22. Residuals as structured matrices (Table 5)
  9.1 Code: Block AttnRes in numpy            23. Takeaways and open questions
 10. Depth-mixing matrix M (live, part 1)
 10.1 Depth-mixing matrix M (live, part 2)
 11. Prior residuals = depth-wise linear attention
 12. Training at scale: pipeline caching

"live" slides are computed in this file with a small numpy network
(random, untrained weights). They illustrate the mechanics; every number about
trained models comes from the paper and is labelled with its table or figure.

Run:
  python 5_30_attention_residuals.py          # slide deck
  python 5_30_attention_residuals.py --demo   # numeric checks in the terminal
"""

from __future__ import annotations

import argparse
import inspect
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _config import (
    C_ACCENT, C_CODE, C_DARK, C_GOAL, C_MUTE, C_NEW, C_OK, C_OLD, C_WARN,
    FIGURE_DPI, FS_FIGURE_TITLE, apply_rcparams, run_slide_deck, screen_figsize,
)

apply_rcparams(plt)

CITE = ("Kimi Team (G. Chen et al.), \"Attention Residuals\", "
        "arXiv:2603.15031, submitted 16 Mar 2026")

FS_TITLE = FS_FIGURE_TITLE - 2
FS_HEAD = 15
FS_BODY = 13
FS_SMALL = 11
FS_NOTE = 10

C_BASE = "#7f8c8d"     # baseline / standard residual
C_FULL = "#c0392b"     # Full AttnRes
C_BLOCK = "#2874a6"    # Block AttnRes
C_MHC = "#8e44ad"      # mHC

# ════════════════════════════════════════════════════════════════════════════
# Part A. Numbers reported in the paper
# ════════════════════════════════════════════════════════════════════════════

# Table 2: five model sizes, validation loss of each residual scheme.
SCALING = [  # (activated params, tokens, L_b, val loss: baseline, block, full, mHC-lite)
    ("194M", "38.7B", 12, 1.931, 1.909, 1.899, 1.906),
    ("241M", "45.4B", 13, 1.895, 1.875, 1.874, 1.869),
    ("296M", "62.1B", 14, 1.829, 1.809, 1.804, 1.807),
    ("436M", "87.9B", 16, 1.766, 1.746, 1.737, 1.747),
    ("528M", "119.0B", 17, 1.719, 1.693, 1.692, 1.694),
]

# Fig. 4: fitted power laws L = A * C^(-alpha), C in PFLOP/s-days.
POWER_LAW = {"Baseline": (1.891, 0.057), "Block AttnRes": (1.870, 0.058),
             "Full AttnRes": (1.865, 0.057)}

# Table 3: 48B / 3B-activated Kimi Linear, same 1.4T-token recipe.
BENCH = [  # (area, benchmark, baseline, AttnRes)
    ("General", "MMLU", 73.5, 74.6), ("General", "MMLU-Pro", 52.2, 52.2),
    ("General", "GPQA-Diamond", 36.9, 44.4), ("General", "BBH", 76.3, 78.0),
    ("General", "ARC-Challenge", 64.6, 65.7), ("General", "HellaSwag", 83.2, 83.4),
    ("General", "TriviaQA", 69.9, 71.8),
    ("Math & Code", "GSM8K", 81.7, 82.4), ("Math & Code", "MGSM", 64.9, 66.1),
    ("Math & Code", "Math", 53.5, 57.1), ("Math & Code", "CMath", 84.7, 85.1),
    ("Math & Code", "HumanEval", 59.1, 62.2), ("Math & Code", "MBPP", 72.0, 73.9),
    ("Chinese", "CMMLU", 82.0, 82.9), ("Chinese", "C-Eval", 79.6, 82.5),
]

# Table 4: ablations on the 16-block (436M) model; validation loss.
ABLATION = [
    ("Baseline (PreNorm)", 1.766, "base"),
    ("DenseFormer (static scalars)", 1.767, "prior"),
    ("mHC (m streams)", 1.747, "prior"),
    ("Full AttnRes", 1.737, "full"),
    ("  w/ input-dependent query", 1.731, "full"),
    ("  w/ input-independent mixing", 1.749, "full"),
    ("  w/ sigmoid instead of softmax", 1.741, "full"),
    ("  w/o RMSNorm on keys", 1.743, "full"),
    ("Sliding window (W = 1 + 8)", 1.764, "prior"),
    ("Block AttnRes (S = 4)", 1.746, "block"),
    ("  w/ multihead (H = 16)", 1.752, "block"),
    ("  w/o RMSNorm on keys", 1.750, "block"),
]

# Fig. 6: block size S vs validation loss (S = 1 is Full AttnRes).
BLOCK_SWEEP = [(1, 1.737), (2, 1.746), (4, 1.746), (8, 1.748), (16, 1.753), (32, 1.757)]

# Fig. 7: fixed compute (~6.5e19 FLOPs) and ~2.3e8 active params.
SWEEP_D = [15, 30, 45, 60, 75]           # d_model / L_b
SWEEP_H = [0.7, 0.6, 0.5, 0.4, 0.3]      # H / L_b (rows, top to bottom)
SWEEP_BASE = [[2.017, 1.909, 1.875, 1.851, 1.858],
              [1.990, 1.902, 1.862, 1.852, 1.862],
              [1.973, 1.883, 1.859, 1.849, 1.854],
              [1.952, 1.868, 1.850, 1.849, 1.857],
              [1.926, 1.857, 1.851, 1.847, 1.858]]
SWEEP_ATTN = [[1.954, 1.890, 1.843, 1.828, 1.824],
              [1.931, 1.863, 1.830, 1.817, 1.818],
              [1.917, 1.841, 1.819, 1.812, 1.817],
              [1.893, 1.823, 1.815, 1.813, 1.813],
              [1.877, 1.816, 1.802, 1.806, 1.820]]


def power_law(name, c):
    a, alpha = POWER_LAW[name]
    return a * np.asarray(c, dtype=float) ** (-alpha)


def compute_multiplier(c=5.6):
    """Baseline compute needed to match Block AttnRes's loss at compute c."""
    target = power_law("Block AttnRes", c)
    a, alpha = POWER_LAW["Baseline"]
    return (a / target) ** (1 / alpha) / c


# ════════════════════════════════════════════════════════════════════════════
# Part B. Cost formulas (Section 4, Table 1, Appendix B)
# ════════════════════════════════════════════════════════════════════════════

def pipeline_comm(P, V, Np=1):
    """Eq. 7 (naive) and Eq. 8 (cross-stage caching), in units of d per token."""
    C = P * V
    naive = C * (C - 1) / 2 * Np
    cached = P * (P - 1) / 2 * Np + (V - 1) * P * P * Np
    return naive, cached


def io_per_layer(L=128, N=8, m=4):
    """Table 1: residual-mechanism memory I/O per token per layer, in units of d."""
    S = L // N
    return {
        "Standard residual": 3.0,
        f"mHC (m={m})": 8 * m + 2,        # + 2m^2 + 4m scalars, negligible next to d
        "Full AttnRes (two-phase)": float(S + N),
        "Block AttnRes (two-phase)": N / S + 5,
    }


def prefill_memory_gb(T=128 * 1024, N=8, d=7168, bytes_per=2, tp=1, chunk=None):
    """Block representations stored during prefill: N x T x d elements.

    The paper quotes 15 GB (128K tokens, 8 blocks), ~1.9 GB per device after
    sequence sharding, and < 0.3 GB with 16K chunked prefill. d = 7168 in bf16 with
    8 tensor-parallel devices reproduces all three numbers (our reconstruction;
    the paper does not state d or P here).
    """
    t = chunk if chunk else T
    return N * t * d * bytes_per / tp / 1e9


# ════════════════════════════════════════════════════════════════════════════
# Part C. A small numpy network with three residual schemes
# ════════════════════════════════════════════════════════════════════════════

D_TOY, L_TOY, T_TOY = 64, 32, 24      # hidden size, layers, tokens


def rms_norm(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def softmax(z, axis=0):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def make_layers(L=L_TOY, d=D_TOY, seed=0):
    """Random 2-layer MLPs: f(h) = W2 relu(W1 RMSNorm(h)) (the PreNorm layer)."""
    rng = np.random.default_rng(seed)
    return [(rng.normal(0, 1 / np.sqrt(d), (d, 2 * d)),
             rng.normal(0, 1 / np.sqrt(2 * d), (2 * d, d))) for _ in range(L)]


def layer_fn(layer, h):
    w1, w2 = layer
    return np.maximum(rms_norm(h) @ w1, 0.0) @ w2


def embed(T=T_TOY, d=D_TOY, seed=1):
    return np.random.default_rng(seed).normal(0, 1, (T, d))


def standard_residual(h1, layers):
    """h_l = h_{l-1} + f_{l-1}(h_{l-1})."""
    h, inputs, outputs = h1, [], []
    for f in layers:
        inputs.append(h)
        out = layer_fn(f, h)
        outputs.append(out)
        h = h + out
    return inputs, outputs


def depth_attention(w, values):
    """Eq. 2-4: alpha_{i->l} = softmax_i( w_l . RMSNorm(v_i) ), h_l = sum alpha v_i.

    w: [d] pseudo-query of layer l;  values: [n, T, d] sources.  Returns h, alpha [n, T].
    """
    logits = np.einsum("d,ntd->nt", w, rms_norm(values))
    alpha = softmax(logits, axis=0)
    return np.einsum("nt,ntd->td", alpha, values), alpha


def full_attn_res(h1, layers, queries):
    """Full AttnRes: layer l attends over [h_1, f_1(h_1), ..., f_{l-1}(h_{l-1})]."""
    values, inputs, alphas = [h1], [], []
    for f, w in zip(layers, queries):
        h, alpha = depth_attention(w, np.stack(values))
        inputs.append(h)
        alphas.append(alpha.mean(axis=1))          # average over tokens
        values.append(layer_fn(f, h))
    return inputs, values[1:], alphas


def block_attn_res(h1, layers, queries, S=4):
    """Block AttnRes: sum outputs inside a block, attend over block summaries.

    Sources for layer i of block n:  [b_0, ..., b_{n-1}]            if i = 1
                                     [b_0, ..., b_{n-1}, b_n^{i-1}] if i >= 2
    with b_0 = h_1 (token embedding) and b_n^{i-1} the partial sum so far.
    """
    blocks, partial, inputs, alphas = [h1], None, [], []
    for l, (f, w) in enumerate(zip(layers, queries)):
        srcs = blocks if partial is None else blocks + [partial]
        h, alpha = depth_attention(w, np.stack(srcs))
        inputs.append(h)
        alphas.append(alpha.mean(axis=1))
        out = layer_fn(f, h)
        partial = out if partial is None else partial + out
        if (l + 1) % S == 0:                       # block boundary
            blocks.append(partial)
            partial = None
    return inputs, alphas, blocks


def zero_queries(L=L_TOY, d=D_TOY):
    """The paper initialises every pseudo-query to zero -> uniform weights."""
    return [np.zeros(d) for _ in range(L)]


def random_queries(L=L_TOY, d=D_TOY, scale=0.6, seed=3):
    """Untrained, random pseudo-queries: only to show that weights can be uneven."""
    rng = np.random.default_rng(seed)
    return [rng.normal(0, scale, d) for _ in range(L)]


def norm(x):
    return float(np.mean(np.linalg.norm(x, axis=-1)))


def toy_norms(S=4):
    h1, layers = embed(), make_layers()
    std_in, std_out = standard_residual(h1, layers)
    full_in, _, _ = full_attn_res(h1, layers, zero_queries())
    blk_in, _, _ = block_attn_res(h1, layers, zero_queries(), S=S)
    share = [norm(o) / norm(i + o) for i, o in zip(std_in, std_out)]
    return ([norm(h) for h in std_in], [norm(h) for h in full_in],
            [norm(h) for h in blk_in], share)


# ── depth-mixing matrices M (Section 6.2): row l = weights on v_0..v_{l-1} ──

def mixing_standard(L):
    return np.tril(np.ones((L, L)))


def mixing_highway(L, seed=5):
    """h_l = (1-g_l) h_{l-1} + g_l f_{l-1}: unroll the coefficients exactly."""
    g = np.random.default_rng(seed).uniform(0.2, 0.8, L)
    M, c = np.zeros((L, L)), np.zeros(L)
    c[0] = 1.0
    M[0] = c
    for l in range(1, L):
        c = (1 - g[l]) * c
        c[l] += g[l]
        M[l] = c
    return M


def mixing_full(L, seed=7):
    h1, layers = embed(seed=seed), make_layers(L, seed=seed)
    _, _, alphas = full_attn_res(h1, layers, random_queries(L, scale=1.5, seed=seed))
    M = np.zeros((L, L))
    for l, a in enumerate(alphas):
        M[l, :len(a)] = a
    return M


def mixing_block(L, S=2, seed=7):
    """Expand block-level weights back to the individual source layers."""
    h1, layers = embed(seed=seed), make_layers(L, seed=seed)
    _, alphas, _ = block_attn_res(h1, layers, random_queries(L, scale=1.5, seed=seed), S=S)
    M = np.zeros((L, L))
    for l, a in enumerate(alphas):
        n_done = l // S                        # completed blocks before layer l
        M[l, 0] = a[0]                         # embedding b_0 = source 0
        for n in range(1, n_done + 1):         # block n holds sources (n-1)S+1 .. nS
            M[l, (n - 1) * S + 1:n * S + 1] = a[n]
        if len(a) > n_done + 1:                # partial sum of the current block
            M[l, n_done * S + 1:l + 1] = a[-1]
    return M


# ── two-phase inference (Algorithm 1) ──

def attn_with_stats(Q, K, V):
    """Softmax attention that also returns the max m and the sum s.

    Q: [S, d];  K, V: [n, T, d].  o: [S, T, d] unnormalised;  m, s: [S, T].
    """
    logits = np.einsum("sd,ntd->snt", Q, rms_norm(K))
    m = logits.max(axis=1)
    p = np.exp(logits - m[:, None, :])
    return np.einsum("snt,ntd->std", p, V), m, p.sum(axis=1)


def two_phase_block(queries, blocks, layers):
    """Algorithm 1: batch all S inter-block queries once, then walk the block."""
    Q, KV = np.stack(queries), np.stack(blocks)
    o1, m1, s1 = attn_with_stats(Q, KV, KV)        # Phase 1: one read
    partial, outs = None, []
    for i, (w, f) in enumerate(zip(queries, layers)):  # Phase 2
        if partial is None:
            h = o1[i] / s1[i][:, None]
        else:
            o2, m2, s2 = attn_with_stats(w[None], partial[None], partial[None])
            o2, m2, s2 = o2[0], m2[0], s2[0]
            m = np.maximum(m1[i], m2)              # online softmax
            a, b = np.exp(m1[i] - m), np.exp(m2 - m)
            num = a[:, None] * o1[i] + b[:, None] * o2
            h = num / (a * s1[i] + b * s2)[:, None]
        outs.append(h)
        y = layer_fn(f, h)
        partial = y if partial is None else partial + y
    return outs


def check_two_phase(S=4, n_prev=3, seed=11):
    """Two-phase result vs direct attention over [b_0..b_{n-1}, partial]."""
    rng = np.random.default_rng(seed)
    blocks = [rng.normal(0, 1 + k, (T_TOY, D_TOY)) for k in range(n_prev)]
    layers, queries = make_layers(S, seed=seed), random_queries(S, scale=1.0, seed=seed)
    fast = two_phase_block(queries, blocks, layers)
    partial, worst = None, 0.0
    for i, (w, f) in enumerate(zip(queries, layers)):
        srcs = blocks if partial is None else blocks + [partial]
        h, _ = depth_attention(w, np.stack(srcs))
        worst = max(worst, float(np.abs(h - fast[i]).max()))
        y = layer_fn(f, h)
        partial = y if partial is None else partial + y
    return worst


# ════════════════════════════════════════════════════════════════════════════
# Part D. Slide helpers
# ════════════════════════════════════════════════════════════════════════════

def new_slide(title):
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle(title, fontsize=FS_TITLE, fontweight="bold", y=0.965)
    fig.text(0.5, 0.055, CITE, ha="center", va="center", fontsize=FS_NOTE,
             color="0.4", style="italic")
    return fig


def canvas(fig, rect=(0.03, 0.09, 0.94, 0.82)):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    return ax


def rbox(ax, x, y, w, h, face, edge="0.35", lw=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=face, edgecolor=edge, linewidth=lw))


def wrap(text, width):
    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return "\n".join(out)


def card(ax, x, y, w, h, head, body, face=C_NEW, edge=C_DARK, width=None, fs=FS_BODY,
         math=False):
    """Box with a bold heading and a top-aligned body (wrapped to the box width)."""
    rbox(ax, x, y, w, h, face, edge)
    ax.text(x + 0.2, y + h - 0.22, head, ha="left", va="top", fontsize=FS_HEAD,
            fontweight="bold", color=edge)
    if width is None:
        width = int(w * 8.2 * FS_BODY / fs)
    ax.text(x + 0.2, y + h - 0.8, body if math else wrap(body, width), ha="left", va="top",
            fontsize=fs, linespacing=1.35)


def banner(ax, text, y=6.9, h=0.9):
    rbox(ax, 0.2, y, 11.6, h, C_DARK, C_DARK, 0)
    ax.text(6.0, y + h / 2, text, ha="center", va="center", fontsize=FS_HEAD,
            fontweight="bold", color="#f9e79f")


def note(ax, text, y=0.3, face="lightyellow"):
    ax.text(6.0, y, text, ha="center", va="center", fontsize=FS_SMALL, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=face, edgecolor="0.55"))


def code_slide(title, funcs, caption):
    fig = new_slide(title)
    src = "\n\n".join(inspect.getsource(f).rstrip() for f in funcs)
    lines = src.splitlines()
    ax = fig.add_axes((0.04, 0.12, 0.92, 0.79))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.01",
                                facecolor=C_CODE, edgecolor="#111", transform=ax.transAxes))
    h_pt = 0.92 * 0.79 * fig.get_figheight() * 72
    w_pt = 0.95 * 0.92 * fig.get_figwidth() * 72
    fs = max(6.5, min(12.0, h_pt / (1.3 * 1.25 * len(lines)),
                      w_pt / (0.6 * max(map(len, lines)))))
    ax.text(0.015, 0.975, src, ha="left", va="top", family="monospace", fontsize=fs,
            color="#d4d4d4", linespacing=1.3, transform=ax.transAxes, parse_math=False)
    fig.text(0.5, 0.093, caption, ha="center", va="center", fontsize=FS_SMALL,
             fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                          edgecolor="0.55"))
    return fig


def arrow(ax, x0, y0, x1, y1, color="0.3", lw=1.8, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FS_HEAD, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=FS_BODY)
    ax.set_ylabel(ylabel, fontsize=FS_BODY)
    ax.tick_params(labelsize=FS_SMALL)
    ax.grid(alpha=0.3)


# ════════════════════════════════════════════════════════════════════════════
# Part E. Slides
# ════════════════════════════════════════════════════════════════════════════

def preface():
    fig = new_slide("Attention Residuals (AttnRes)")
    ax = canvas(fig)
    rbox(ax, 0.6, 4.3, 10.8, 3.2, C_NEW, C_DARK, 2)
    ax.text(6, 6.85, "\"Attention Residuals\"  (Technical Report)", ha="center",
            fontsize=FS_HEAD + 5, fontweight="bold", color=C_DARK)
    ax.text(6, 6.05, "Kimi Team, Moonshot AI. First author: Guangyu Chen (37 named authors)",
            ha="center", fontsize=FS_HEAD)
    ax.text(6, 5.35, "arXiv:2603.15031 [cs.CL]   ·   v1 submitted 16 March 2026",
            ha="center", fontsize=FS_HEAD, fontweight="bold", color=C_ACCENT)
    ax.text(6, 4.7, "github.com/MoonshotAI/Attention-Residuals", ha="center",
            fontsize=FS_BODY, family="monospace", color="0.3")
    card(ax, 0.6, 0.9, 5.25, 3.0, "The question",
         "Every Transformer adds each layer's output into one running sum with weight 1. "
         "Sequences got rid of fixed recurrence with attention. Can depth do the same?",
         face=C_MUTE, edge=C_DARK)
    card(ax, 6.15, 0.9, 5.25, 3.0, "The answer in one line",
         "Let each layer take a softmax-weighted mix of all earlier layer outputs, using "
         "one learned query vector per layer. Group layers into ~8 blocks to make it "
         "cheap at 48B scale.", face=C_GOAL, edge=C_OK)
    return fig


def step_1():
    fig = new_slide("Step 1: The idea in one picture")
    ax = canvas(fig)
    titles = ["(a) Standard residual", "(b) Full AttnRes", "(c) Block AttnRes"]
    for k, t in enumerate(titles):
        x0 = 0.25 + k * 3.9
        rbox(ax, x0, 1.2, 3.6, 6.2, C_MUTE, "0.5", 1.2)
        ax.text(x0 + 1.85, 7.05, t, ha="center", fontsize=FS_HEAD, fontweight="bold")
        ys = [1.7, 2.8, 3.9, 5.0, 6.1]
        names = ["embed", "layer 1", "layer 2", "layer 3", "layer 4"]
        for y, n in zip(ys, names):
            face = C_OLD if n == "embed" else "white"
            if k == 2 and n in ("layer 1", "layer 2"):
                face = "#d6eaf8"
            if k == 2 and n in ("layer 3", "layer 4"):
                face = "#d5f5e3"
            rbox(ax, x0 + 0.3, y - 0.3, 1.5, 0.6, face, "0.3", 1.2)
            ax.text(x0 + 1.05, y, n, ha="center", va="center", fontsize=FS_SMALL)
        if k == 0:
            for y0, y1 in zip(ys[:-1], ys[1:]):
                arrow(ax, x0 + 1.05, y0 + 0.3, x0 + 1.05, y1 - 0.3)
            ax.text(x0 + 2.0, 3.9, "h + f(h)\nweight 1,\nalways", fontsize=FS_SMALL,
                    va="center", color=C_BASE, fontweight="bold")
        elif k == 1:
            for i, y in enumerate(ys[:-1]):
                arrow(ax, x0 + 1.8, y, x0 + 1.8, ys[-1] - 0.05, color=C_FULL,
                      rad=-0.35 - 0.05 * i, lw=1.3)
            ax.text(x0 + 2.45, 3.4, r"$\alpha_{i\to 4}$" + "\nsoftmax\nweights",
                    fontsize=FS_SMALL, color=C_FULL, fontweight="bold")
        else:
            ax.text(x0 + 2.0, 2.35, "block 1\n= sum", fontsize=FS_SMALL, color=C_BLOCK,
                    fontweight="bold")
            arrow(ax, x0 + 1.8, ys[0], x0 + 1.8, ys[-1] - 0.05, color=C_FULL, rad=-0.45)
            arrow(ax, x0 + 1.8, ys[2], x0 + 1.8, ys[-1] - 0.05, color=C_FULL, rad=-0.35)
            arrow(ax, x0 + 1.8, ys[3], x0 + 1.8, ys[-1] - 0.05, color=C_FULL, rad=-0.3)
            ax.text(x0 + 2.55, 4.6, "attend\nover\nblocks", fontsize=FS_SMALL,
                    color=C_FULL, fontweight="bold")
    note(ax, "Paper Fig. 1. AttnRes changes only HOW layer inputs are formed; the layers "
             "themselves (attention, MoE) are untouched.", y=0.55)
    return fig


def step_2():
    fig = new_slide("Step 2: Standard residuals, unrolled")
    ax = canvas(fig)
    card(ax, 0.3, 4.1, 5.6, 3.5, "The update (He et al., 2015)",
         r"$h_l = h_{l-1} + f_{l-1}(h_{l-1})$" "\n\n"
         "Unroll it:\n"
         r"$h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)$" "\n\n"
         "Every earlier output enters with weight exactly 1.", math=True)
    card(ax, 6.1, 4.1, 5.6, 3.5, "Why it trains (the gradient highway)",
         "\n" r"$\dfrac{\partial \mathcal{L}}{\partial h_l} = \dfrac{\partial \mathcal{L}}"
         r"{\partial h_L}\prod_{j=l}^{L-1}\left(I + \dfrac{\partial f_j}{\partial h_j}\right)$"
         "\n\nThe product always contains the identity I,\nso gradients reach layer l at "
         "any depth.", math=True, face=C_GOAL, edge=C_OK)
    card(ax, 0.3, 0.9, 11.4, 2.85, "The second role nobody tuned",
         "Residuals also decide how information is aggregated over depth. Sequence mixing "
         "(attention) and expert routing (MoE) both learned input-dependent weights years "
         "ago, but depth-wise aggregation is still a fixed sum with unit weights: no layer "
         "can emphasise or suppress any particular earlier layer.",
         face=C_OLD, edge=C_ACCENT)
    return fig


def step_3():
    fig = new_slide("Step 3: PreNorm dilution (live toy, 32 random layers)")
    std, full, blk, share = toy_norms()
    layers = np.arange(1, L_TOY + 1)
    ax1 = fig.add_axes((0.07, 0.25, 0.4, 0.6))
    ax1.plot(layers, std, "o-", color=C_BASE, lw=2, label="Standard residual (sum)")
    ax1.plot(layers, full, "s-", color=C_FULL, lw=2, label="Full AttnRes (zero-init query)")
    ax1.plot(layers, blk, "^-", color=C_BLOCK, lw=2, label="Block AttnRes, S = 4")
    style_axes(ax1, "Size of the input to each layer", "layer l", r"mean $\|h_l\|$")
    ax1.legend(fontsize=FS_SMALL)
    ax2 = fig.add_axes((0.56, 0.25, 0.4, 0.6))
    ax2.plot(layers, share, "o-", color=C_BASE, lw=2, label="standard residual (toy)")
    style_axes(ax2, "Share of the newest output",
               "layer l", r"$\|f_l\| \;/\; \|h_l + f_l\|$")
    ax2.legend(fontsize=FS_SMALL)
    fig.text(0.5, 0.115,
             "PreNorm normalises each layer's input but adds its output to an unnormalised, "
             "ever-growing sum,\nso each new layer's share shrinks. AttnRes inputs are "
             "softmax-weighted (convex) mixes, so their size stays bounded.",
             ha="center", va="center", fontsize=FS_SMALL,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="0.55"))
    return fig


def step_4():
    fig = new_slide("Step 4: Gated residuals and the three limitations")
    ax = canvas(fig)
    card(ax, 0.3, 4.6, 5.6, 3.0, "Highway networks (Srivastava et al.)",
         r"$h_l = (1-g_l)\odot h_{l-1} + g_l \odot f_{l-1}(h_{l-1})$" "\n\n"
         "General form:  " r"$h_l = \alpha_l h_{l-1} + \beta_l f_{l-1}(h_{l-1})$" "\n"
         r"residual: $\alpha=\beta=1$;   highway: $\alpha=1-g,\ \beta=g$", math=True)
    card(ax, 6.1, 4.6, 5.6, 3.0, "The shared constraint",
         "Fixed or gated, layer l sees only h_{l-1}: one compressed state that mixes all "
         "earlier outputs together. It cannot look at an individual earlier layer.",
         face=C_OLD, edge=C_ACCENT)
    lim = [("1. No selective access",
            "Attention and MLP layers get the same aggregated state, though they may want "
            "different mixtures."),
           ("2. Irreversible loss",
            "What the sum blurs away cannot be recovered selectively by deeper layers."),
           ("3. Output growth",
            "Later layers learn larger and larger outputs to be heard over the sum; this "
            "can destabilise training.")]
    for k, (h, b) in enumerate(lim):
        card(ax, 0.3 + k * 3.85, 1.0, 3.6, 3.2, h, b, face=C_WARN, edge=C_ACCENT)
    note(ax, "Empirically, many deep layers of LLMs can be pruned with little loss "
             "(Gromov et al.): a symptom of dilution.", y=0.45)
    return fig


def step_5():
    fig = new_slide("Step 5: The duality of time and depth")
    ax = canvas(fig)
    rows = [("", "Over TIME (sequence)", "Over DEPTH (layers)"),
            ("Recurrence", "RNN: s_t = s_{t-1} + k_t v_t\none state compresses the past",
             "Residual: h_l = h_{l-1} + f_{l-1}\none state compresses all layers"),
            ("Gating", "Gated linear attention, GLA", "Highway networks"),
            ("Delta rule", "DeltaNet", "DDL (deep delta learning)"),
            ("Attention", "Transformer: softmax over all\nearlier positions",
             "AttnRes: softmax over all\nearlier layers")]
    ys = [6.9, 5.6, 4.45, 3.5, 2.2]
    hs = [0.7, 1.15, 0.8, 0.8, 1.15]
    for (a, b, c), y, h in zip(rows, ys, hs):
        head = a == ""
        last = a == "Attention"
        for x, w, txt in [(0.3, 2.4, a), (2.9, 4.3, b), (7.4, 4.3, c)]:
            face = C_DARK if head else (C_GOAL if last else C_MUTE)
            rbox(ax, x, y - h / 2, w, h, face, "0.5", 1.0)
            ax.text(x + w / 2, y, txt, ha="center", va="center",
                    fontsize=FS_BODY, fontweight="bold" if (head or x < 1) else None,
                    color="#f9e79f" if head else "black")
    note(ax, "Depth is small (L < 1000) next to sequence length (millions):\n"
             "plain O(L^2) softmax attention over depth is affordable.", y=0.85)
    return fig


def step_6():
    fig = new_slide("Step 6: Full Attention Residuals")
    ax = canvas(fig)
    card(ax, 0.3, 3.6, 6.4, 4.0, "Definition (Eq. 1-4)", "", math=True)
    for y, txt in [(6.45, r"$h_l = \sum_{i=0}^{l-1} \alpha_{i\to l}\, v_i,\qquad "
                          r"\sum_{i} \alpha_{i\to l} = 1$"),
                   (5.25, r"$\alpha_{i\to l} = \dfrac{\exp(w_l^{\top}\,\mathrm{RMSNorm}(k_i))}"
                          r"{\sum_{j}\exp(w_l^{\top}\,\mathrm{RMSNorm}(k_j))}$"),
                   (4.1, r"$k_i = v_i = h_1$ (embedding) for $i=0$,   $f_i(h_i)$ for "
                         r"$i \geq 1$")]:
        ax.text(0.55, y, txt, va="center", fontsize=FS_BODY + 1)
    card(ax, 6.9, 3.6, 4.8, 4.0, "Design choices",
         "• Query w_l: ONE learned d-vector per layer, not computed from the input.\n"
         "• Keys = values = earlier layer outputs.\n"
         "• RMSNorm on keys: big outputs cannot win the softmax by magnitude alone.\n"
         "• Weights still depend on the input, through the keys.",
         face=C_GOAL, edge=C_OK, width=46, fs=FS_SMALL + 1)
    card(ax, 0.3, 0.9, 5.6, 2.35, "Zero initialisation is required",
         "All w_l start at 0, so every alpha is uniform: AttnRes begins as an equal-weight "
         "average of earlier layers. The paper found this prevents training volatility.",
         face=C_OLD, edge=C_ACCENT)
    card(ax, 6.1, 0.9, 5.6, 2.35, "Cost per token",
         "O(L^2 d) arithmetic and O(L d) memory. Extra parameters: one RMSNorm and one "
         "d-vector per layer, a negligible fraction of the model.", face=C_MUTE, edge=C_DARK)
    return fig


def step_6_1():
    return code_slide("Step 6.1: Code - Full AttnRes in numpy",
                      [rms_norm, depth_attention, full_attn_res],
                      "depth_attention is the whole mechanism: a dot product with w_l, "
                      "a softmax over sources, a weighted sum.")


def step_7():
    fig = new_slide("Step 7: Depth-attention weights (live toy, untrained)")
    h1, layers = embed(), make_layers()
    mats = []
    for q in (zero_queries(), random_queries(scale=1.5)):
        _, _, alphas = full_attn_res(h1, layers, q)
        M = np.full((L_TOY, L_TOY), np.nan)
        for l, a in enumerate(alphas):
            M[l, :len(a)] = a
        mats.append(M)
    for k, (M, t) in enumerate(zip(mats, ["Zero-init queries (start of training)",
                                         "Random queries (illustration)"])):
        ax = fig.add_axes((0.06 + 0.49 * k, 0.24, 0.36, 0.6))
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=np.nanmax(M) if k else 1.0,
                       aspect="auto")
        style_axes(ax, t, "source i (0 = embedding)", "layer l")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label(r"$\alpha_{i\to l}$")
    fig.text(0.5, 0.105,
             "Left: every row is uniform (1/l): AttnRes starts as an average, whose size "
             "stays bounded, unlike the residual sum.\nRight: the same layers with non-zero "
             "queries. Each layer can put its weight anywhere in its past; training decides where.",
             ha="center", va="center", fontsize=FS_SMALL,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="0.55"))
    return fig


def step_8():
    fig = new_slide("Step 8: Why Full AttnRes is hard at scale")
    ax = canvas(fig)
    card(ax, 0.3, 4.4, 5.6, 3.2, "Small scale: free",
         "In vanilla training every layer output is already kept for backpropagation. "
         "Full AttnRes reads those same tensors, so it adds no memory.",
         face=C_GOAL, edge=C_OK)
    card(ax, 6.1, 4.4, 5.6, 3.2, "Large scale: not free",
         "With activation recomputation, outputs that would be freed must stay alive for "
         "every later layer. With pipeline parallelism, each must be SENT across stage "
         "boundaries. Both costs grow as O(L d).", face=C_WARN, edge=C_ACCENT)
    card(ax, 0.3, 0.9, 11.4, 3.1, "The key trick that makes everything later possible",
         "The query w_l is a parameter, not a function of h_l. So the attention scores of "
         "a whole group of layers can be computed together, before those layers run. "
         "Batching cuts local memory reads from O(L d) to O((S+N) d) per layer. It cannot fix "
         "cross-stage communication, which is the real bottleneck today. That motivates "
         "Block AttnRes: send N block summaries instead of L layer outputs.",
         face=C_NEW, edge=C_DARK)
    return fig


def step_9():
    fig = new_slide("Step 9: Block Attention Residuals")
    ax = canvas(fig)
    card(ax, 0.3, 3.9, 5.8, 3.7, "Inside a block: plain sum (Eq. 5)",
         r"$b_n = \sum_{j \in B_n} f_j(h_j)$,   $b_n^{i}$ = partial sum of the first i layers"
         "\n\n" r"Split $L$ layers into $N$ blocks of $S = L/N$ layers." "\n"
         r"$b_0 = h_1$: the embedding is always a source.", math=True)
    card(ax, 6.3, 3.9, 5.4, 3.7, "Across blocks: attention (Eq. 6)",
         "Layer i of block n attends over\n"
         r"$[b_0,\ldots,b_{n-1}]$" "                if i = 1\n"
         r"$[b_0,\ldots,b_{n-1},\ b_n^{\,i-1}]$" "     if i >= 2\n\n"
         "Same query, RMSNorm keys and softmax as Full.", math=True, face=C_GOAL, edge=C_OK)
    rows = [("N = L", "Full AttnRes (every layer its own block)"),
            ("N = 1", "standard residual, with the embedding kept separate"),
            ("N ≈ 8", "paper's choice: keeps most of the gain")]
    for k, (a, b) in enumerate(rows):
        y = 2.95 - k * 0.8
        rbox(ax, 0.3, y - 0.32, 1.6, 0.64, C_DARK, C_DARK, 0)
        ax.text(1.1, y, a, ha="center", va="center", fontsize=FS_BODY, color="#f9e79f",
                fontweight="bold")
        ax.text(2.1, y, b, va="center", fontsize=FS_BODY)
    card(ax, 8.3, 0.9, 3.4, 2.6, "Cost", "Memory O(L d) -> O(N d)\nCompute O(L^2) -> O(N^2)\n"
         "Bounded KV cache over depth", face=C_MUTE, edge=C_DARK, width=30)
    return fig


def step_9_1():
    return code_slide("Step 9.1: Code - Block AttnRes (mirrors paper Fig. 2)",
                      [block_attn_res],
                      "The paper's PyTorch version does the same per Transformer layer, "
                      "twice: once before attention, once before the MLP.")


def _matrix_panel(fig, rect, M, title, fmt=True):
    ax = fig.add_axes(rect)
    L = M.shape[0]
    show = np.where(np.tril(np.ones_like(M)) > 0, M, np.nan)
    im = ax.imshow(show, cmap="YlOrRd", vmin=0, vmax=np.nanmax(show), aspect="equal")
    ax.set_title(title, fontsize=FS_HEAD, fontweight="bold")
    ax.set_xticks(range(L), [f"v{i}" for i in range(L)], fontsize=FS_SMALL - 1)
    ax.set_yticks(range(L), [f"h{l + 1}" for l in range(L)], fontsize=FS_SMALL - 1)
    if fmt:
        for l in range(L):
            for i in range(l + 1):
                ax.text(i, l, f"{M[l, i]:.2f}", ha="center", va="center", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    return ax


def step_10():
    fig = new_slide("Step 10: Depth-mixing matrix M, part 1 (live, L = 8)")
    _matrix_panel(fig, (0.05, 0.22, 0.4, 0.62), mixing_standard(8),
                  "Standard residual: all ones")
    _matrix_panel(fig, (0.54, 0.22, 0.4, 0.62), mixing_highway(8),
                  "Highway (random scalar gates)")
    fig.text(0.5, 0.105,
             r"Input to layer l:  $h_l = \sum_i M_{il}\, v_i$  (row = layer, column = source). "
             "Residual: rank-1 structure, weights never change.\nHighway: weights are cumulative "
             "products of gates, sum to 1 (stick-breaking), still 1-semiseparable: only h_{l-1} is visible.",
             ha="center", va="center", fontsize=FS_SMALL,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="0.55"))
    return fig


def step_10_1():
    fig = new_slide("Step 10.1: Depth-mixing matrix M, part 2 (live, L = 8)")
    _matrix_panel(fig, (0.05, 0.22, 0.4, 0.62), mixing_full(8), "Full AttnRes: dense (random queries)")
    _matrix_panel(fig, (0.54, 0.22, 0.4, 0.62), mixing_block(8, S=2),
                  "Block AttnRes, S = 2: shared per block")
    fig.text(0.5, 0.105,
             "Full AttnRes: every entry is its own input-dependent softmax weight (rank up to L).\n"
             "Block AttnRes: sources in a finished block share one weight (equal columns in a row), "
             "so the rank lies between N and N + S.",
             ha="center", va="center", fontsize=FS_SMALL,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="0.55"))
    return fig


def step_11():
    fig = new_slide("Step 11: Prior residuals are depth-wise LINEAR attention")
    ax = canvas(fig)
    card(ax, 0.3, 4.3, 5.6, 3.3, "(m)HC unrolled (Eq. 10)",
         r"$M_{i\to l} = \beta_i^{\top} A^{\times}_{i+1\to l}\, \alpha_l$" "\n\n"
         r"$\alpha_l$ acts as a query of layer l," "\n" r"$\beta_i$ as a key of layer i," "\n"
         r"$A^{\times}$ (product of m x m mixers) as a" "\n" "depth-relative position operator.",
         math=True)
    card(ax, 6.1, 4.3, 5.6, 3.3, "Linear vs softmax",
         "If the score kernel factorises, phi(q, k) = phi(q)^T phi(k), attention over "
         "depth collapses back into a recurrence with a matrix state. That is exactly "
         "(m)HC, DDL and MRLA. AttnRes keeps the exp kernel: softmax attention.",
         face=C_GOAL, edge=C_OK)
    card(ax, 0.3, 0.9, 11.4, 3.0, "The historical parallel the paper draws",
         "Over sequences, the field moved RNN -> linear attention -> softmax attention. Over "
         "depth, residual -> Highway / mHC (gated, multi-state recurrences) were still "
         "linear-attention-like. AttnRes completes the same linear-to-softmax transition for "
         "depth. The m streams of mHC correspond to 'state expansion' along depth.",
         face=C_NEW, edge=C_DARK)
    return fig


def step_12():
    fig = new_slide("Step 12: Training at scale - cross-stage caching")
    ax = fig.add_axes((0.07, 0.22, 0.4, 0.62))
    P = 8
    Vs = np.arange(1, 9)
    naive = [pipeline_comm(P, v)[0] for v in Vs]
    cached = [pipeline_comm(P, v)[1] for v in Vs]
    ax.plot(Vs, naive, "o-", color=C_ACCENT, lw=2, label="naive: resend all blocks (Eq. 7)")
    ax.plot(Vs, cached, "s-", color=C_OK, lw=2, label="cached: send only new blocks (Eq. 8)")
    style_axes(ax, f"Total block traffic, P = {P} stages", "virtual stages per rank V",
               r"communication per token  [$N_p\,d$]")
    ax.legend(fontsize=FS_SMALL)
    c = canvas(fig, (0.52, 0.12, 0.46, 0.78))
    n, ca = pipeline_comm(4, 2)
    card(c, 0.2, 3.7, 11.6, 4.1, "What is sent between pipeline stages",
         "Standard residual: one hidden state, whatever the depth. Block AttnRes: every "
         "block summary so far, because later layers attend to all of them. With C = P V "
         "chunks, chunk j holds j N_p blocks, so naive traffic is C(C-1)/2 N_p d. Each rank "
         "keeps blocks it already received in earlier virtual stages, so it only needs the "
         "new ones.", width=48, fs=FS_SMALL + 1)
    card(c, 0.2, 0.0, 11.6, 3.5, "Paper's example (Fig. 3)",
         f"P = 4, V = 2: naive {n:.0f} vs cached {ca:.0f} block sends; in the second virtual "
         "stage caching removes 6 redundant sends. Peak per-transition cost drops from O(C) "
         "to O(P), a V x improvement, so it overlaps with compute. Measured overhead: "
         "< 4% with pipeline parallelism, negligible without.", face=C_GOAL, edge=C_OK,
         width=48, fs=FS_SMALL + 1)
    return fig


def step_13():
    fig = new_slide("Step 13: Inference - two-phase computation (Algorithm 1)")
    ax = canvas(fig)
    card(ax, 0.3, 4.2, 5.6, 3.4, "Phase 1: parallel, inter-block",
         "All S queries of block n are known in advance (they are parameters). Stack them "
         "into Q [S, d] and attend once to the cached blocks b_0..b_{n-1}. Return outputs "
         "plus softmax statistics (max, sum). Blocks are read once per block, not S times.",
         face=C_GOAL, edge=C_OK)
    card(ax, 6.1, 4.2, 5.6, 3.4, "Phase 2: sequential, intra-block",
         "Walk through the block layer by layer. Attend to the evolving partial sum b_n^{i-1} "
         "and merge with Phase 1 by online softmax (as in FlashAttention). Elementwise, so it "
         "fuses with nearby kernels.", face=C_NEW, edge=C_DARK)
    card(ax, 0.3, 0.9, 11.4, 2.9, "Online-softmax merge (exact, not an approximation)",
         r"$m = \max(m^{(1)}, m^{(2)}),\qquad h_l = \dfrac{e^{m^{(1)}-m}\, o^{(1)} + "
         r"e^{m^{(2)}-m}\, o^{(2)}}{e^{m^{(1)}-m}\, \ell^{(1)} + e^{m^{(2)}-m}\, \ell^{(2)}}$"
         "\n\n" r"$o$ = unnormalised weighted values, $\ell$ = sum of exponentials, "
         r"$m$ = max logit of each part.", math=True, face=C_MUTE, edge=C_DARK)
    return fig


def step_13_1():
    err = check_two_phase()
    return code_slide("Step 13.1: Code - the two-phase schedule",
                      [attn_with_stats, two_phase_block],
                      f"check_two_phase(): max difference from direct attention = {err:.1e} "
                      "(round-off). Same result, fewer reads.")


def step_14():
    fig = new_slide("Step 14: Memory I/O per layer (Table 1) and prefill")
    io = io_per_layer()
    ax = fig.add_axes((0.22, 0.2, 0.27, 0.62))
    names = [n.replace(" (two-phase)", "") for n in io]
    colors = [C_BASE, C_MHC, C_FULL, C_BLOCK]
    ax.barh(names[::-1], list(io.values())[::-1], color=colors[::-1])
    for k, v in enumerate(list(io.values())[::-1]):
        ax.text(v + 0.5, k, f"{v:g} d", va="center", fontsize=FS_BODY, fontweight="bold")
    style_axes(ax, "I/O per token per layer",
               "multiples of d", "")
    ax.set_xlim(0, 40)
    ax.tick_params(axis="y", labelsize=FS_SMALL)
    c = canvas(fig, (0.53, 0.12, 0.45, 0.78))
    card(c, 0.2, 4.0, 11.6, 3.8, "Formulas (Appendix B)",
         "Block (N/S+5) d  ·  Full (S+N) d  ·  std 3d\n"
         "mHC (8m+2) d + 2m^2 + 4m   (L=128, N=8, m=4)\n"
         "Block AttnRes matches mHC's loss (Table 2) at about 1/6 of its I/O. "
         "End-to-end inference latency overhead: < 2%.", width=46, fs=FS_SMALL + 1)
    g = [prefill_memory_gb(), prefill_memory_gb(tp=8), prefill_memory_gb(tp=8, chunk=16384)]
    card(c, 0.2, 0.0, 11.6, 3.7, "Prefill: 128K tokens, 8 blocks",
         f"Store N x T x d elements: {g[0]:.1f} GB. Shard along the sequence over the "
         f"tensor-parallel devices: {g[1]:.1f} GB each. Add 16K chunked prefill: {g[2]:.2f} GB "
         "each. (Paper: 15 GB, ~1.9 GB, < 0.3 GB. We assume d = 7168, bf16, 8 devices.)",
         face=C_GOAL, edge=C_OK, width=46, fs=FS_SMALL + 1)
    return fig


def step_15():
    fig = new_slide("Step 15: Main experiment - Kimi Linear 48B")
    ax = canvas(fig)
    card(ax, 0.3, 3.3, 5.6, 4.3, "Architecture (only residuals change)",
         "• Kimi Linear MoE, Moonlight / DeepSeek-V3 style\n"
         "• 48B total, 3B activated parameters\n"
         "• 27 Transformer blocks = 54 layers\n"
         "• KDA : MLA attention layers = 3 : 1, each followed by MoE\n"
         "• 8 of 256 routed experts + 1 shared\n"
         "• Block AttnRes, 6 layers per block: 9 blocks + embedding = 10 sources", width=47)
    card(ax, 6.1, 3.3, 5.6, 4.3, "Training recipe (both models)",
         "• 1T tokens WSD pre-training, then ~400B high-quality mid-training\n"
         "• Muon optimizer, 8M-token global batch\n"
         "• 4096-token context, then extended to 32K\n"
         "• MLA uses no positional encoding (NoPE), so no YaRN needed", face=C_GOAL,
         edge=C_OK, width=47)
    card(ax, 0.3, 0.7, 11.4, 2.3, "Fair-comparison choices",
         "Scaling-law runs use hyperparameters tuned for the BASELINE, then reused for AttnRes, "
         "which the paper notes makes the comparison conservative. The only new parameters are "
         "one RMSNorm and one zero-initialised query vector per layer.", face=C_MUTE,
         edge=C_DARK)
    return fig


def step_16():
    fig = new_slide("Step 16: Scaling laws (Fig. 4, Table 2)")
    ax = fig.add_axes((0.09, 0.24, 0.38, 0.6))
    c = np.geomspace(0.4, 6.0, 100)
    for name, col in [("Baseline", C_BASE), ("Block AttnRes", C_BLOCK), ("Full AttnRes", C_FULL)]:
        a, al = POWER_LAW[name]
        ax.plot(c, power_law(name, c), color=col, lw=2.2, label=f"{name}: {a} C^-{al}")
    mult = compute_multiplier()
    yb, yk = float(power_law("Baseline", 5.6)), float(power_law("Block AttnRes", 5.6))
    ax.plot([5.6, 5.6], [yk, yb], "k-", lw=1)
    ax.annotate(f"at 5.6 PFLOP/s-days\n{yb:.3f} vs {yk:.3f}\n= {mult:.2f}x compute",
                (5.6, (yb + yk) / 2), xytext=(1.4, 1.715), fontsize=FS_SMALL,
                arrowprops=dict(arrowstyle="->"))
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5], ["0.5", "1", "2", "5"])
    ax.minorticks_off()
    style_axes(ax, "Fitted power laws", "compute C  [PFLOP/s-days]", "validation loss")
    ax.legend(fontsize=FS_SMALL - 1, loc="upper right")
    ax2 = fig.add_axes((0.57, 0.24, 0.4, 0.6))
    x = np.arange(len(SCALING))
    for k, (lab, col) in enumerate([("Baseline", C_BASE), ("Block AttnRes", C_BLOCK),
                                    ("Full AttnRes", C_FULL), ("mHC-lite", C_MHC)]):
        gap = [row[3] - row[3 + k] for row in SCALING]
        if k:
            ax2.bar(x + (k - 2) * 0.27, gap, 0.27, color=col, label=lab)
    ax2.set_xticks(x, [f"{r[0]}\n{r[1]} tok" for r in SCALING], fontsize=FS_SMALL - 1)
    style_axes(ax2, "Loss reduction vs baseline, per size", "",
               "baseline loss - method loss")
    ax2.legend(fontsize=FS_SMALL)
    fig.text(0.5, 0.11, "Same slope, lower curve: the gain does not fade with scale.\n"
             "The Full-Block gap shrinks to 0.001 at the largest size; Block matches mHC at far lower I/O.",
             ha="center", va="center", fontsize=FS_SMALL,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="0.55"))
    return fig


def step_17():
    fig = new_slide("Step 17: Training dynamics (paper Fig. 5) and a toy view")
    c = canvas(fig, (0.03, 0.12, 0.47, 0.78))
    items = [("Validation loss",
              "Lower throughout the 1T-token run; the gap widens during learning-rate decay.",
              C_GOAL, C_OK),
             ("Output magnitude per block",
              "Baseline: grows monotonically with depth (PreNorm dilution). Block AttnRes: "
              "bounded, periodic pattern; block boundaries reset the accumulation.",
              C_NEW, C_DARK),
             ("Gradient magnitude per block",
              "Baseline: earliest layers get disproportionately large gradients. AttnRes: "
              "softmax sources compete for weight, so gradients spread much more evenly.",
              C_OLD, C_ACCENT)]
    for k, (h, b, f, e) in enumerate(items):
        card(c, 0.2, 5.4 - k * 2.55, 11.6, 2.35, h, b, face=f, edge=e, width=50)
    h1, layers = embed(), make_layers()
    std_in, std_out = standard_residual(h1, layers)
    blk_in, _, _ = block_attn_res(h1, layers, zero_queries(), S=4)
    blk_out = [layer_fn(f, h) for f, h in zip(layers, blk_in)]
    ax = fig.add_axes((0.57, 0.22, 0.4, 0.62))
    ls = np.arange(1, L_TOY + 1)
    ax.plot(ls, [norm(h) for h in std_in], "o-", color=C_BASE, lw=2, label=r"standard: $\|h_l\|$")
    ax.plot(ls, [norm(h) for h in blk_in], "^-", color=C_BLOCK, lw=2,
            label=r"Block AttnRes S=4: $\|h_l\|$")
    for b in range(4, L_TOY, 4):
        ax.axvline(b + 0.5, color=C_BLOCK, alpha=0.2)
    style_axes(ax, "Toy (random layers): input size", "layer l", "mean norm")
    ax.legend(fontsize=FS_SMALL)
    fig.text(0.77, 0.105, "Toy, untrained: it shows the bounded, periodic shape,\n"
             "not the paper's measured values.", ha="center", va="center", fontsize=FS_NOTE,
             style="italic")
    return fig


def step_18():
    fig = new_slide("Step 18: Downstream benchmarks, 48B model (Table 3)")
    ax = fig.add_axes((0.17, 0.17, 0.5, 0.68))
    names = [b[1] for b in BENCH][::-1]
    gain = [b[3] - b[2] for b in BENCH][::-1]
    area_col = {"General": C_BLOCK, "Math & Code": C_FULL, "Chinese": C_OK}
    cols = [area_col[b[0]] for b in BENCH][::-1]
    ax.barh(names, gain, color=cols)
    for k, (g, b) in enumerate(zip(gain, BENCH[::-1])):
        ax.text(g + 0.1, k, f"+{g:.1f}  ({b[2]} -> {b[3]})", va="center", fontsize=FS_SMALL)
    style_axes(ax, "AttnRes minus baseline (points)", "gain", "")
    ax.set_xlim(0, 10.5)
    ax.tick_params(axis="y", labelsize=FS_SMALL)
    for a, col in area_col.items():
        ax.bar(0, 0, color=col, label=a)
    ax.legend(fontsize=FS_SMALL, loc="lower right")
    c = canvas(fig, (0.72, 0.12, 0.26, 0.78))
    card(c, 0.1, 4.2, 11.8, 3.6, "Pattern",
         "Biggest gains on multi-step reasoning: GPQA-Diamond +7.5, Math +3.6, HumanEval +3.1.",
         face=C_GOAL, edge=C_OK, width=26)
    card(c, 0.1, 0.6, 11.8, 3.3, "Interpretation",
         "Later layers can fetch and build on earlier representations: helps "
         "compositional tasks. No benchmark got worse (MMLU-Pro tied).",
         face=C_MUTE, edge=C_DARK, width=26)
    return fig


def step_19():
    fig = new_slide("Step 19: Ablations (Table 4) and block size (Fig. 6)")
    ax = fig.add_axes((0.25, 0.17, 0.33, 0.68))
    col = {"base": C_BASE, "prior": C_MHC, "full": C_FULL, "block": C_BLOCK}
    labels = [a[0] for a in ABLATION][::-1]
    vals = [a[1] for a in ABLATION][::-1]
    ax.barh(range(len(vals)), vals, color=[col[a[2]] for a in ABLATION][::-1])
    ax.set_yticks(range(len(vals)), labels)
    for k, v in enumerate(vals):
        ax.text(v + 0.0005, k, f"{v:.3f}", va="center", fontsize=FS_SMALL)
    ax.axvline(1.766, color=C_BASE, ls="--")
    ax.set_xlim(1.725, 1.775)
    style_axes(ax, "436M model: validation loss", "loss (lower is better)", "")
    ax.tick_params(axis="y", labelsize=FS_SMALL)
    ax2 = fig.add_axes((0.68, 0.41, 0.29, 0.44))
    s, v = zip(*BLOCK_SWEEP)
    ax2.plot(s, v, "o-", color=C_BLOCK, lw=2)
    ax2.axhline(1.766, color=C_BASE, ls="--", label="baseline 1.766")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(s, [f"{x}" for x in s])
    style_axes(ax2, "Block size S (S=1 is Full)", "layers per block S", "loss")
    ax2.legend(fontsize=FS_SMALL)
    fig.text(0.825, 0.19, "Static weights (DenseFormer) give nothing;\n"
             "sigmoid < softmax; one weight per layer beats\nper-head weights; RMSNorm on keys matters;\n"
             "a sliding window of recent layers barely helps:\nreaching DISTANT layers is what counts.",
             ha="center", va="center", fontsize=FS_SMALL,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="0.55"))
    return fig


def step_20():
    fig = new_slide("Step 20: AttnRes prefers deeper, narrower networks (Fig. 7)")
    vmin = min(map(min, SWEEP_ATTN))
    vmax = max(map(max, SWEEP_BASE))
    for k, (M, t) in enumerate([(SWEEP_BASE, "Baseline"), (SWEEP_ATTN, "Attention Residuals")]):
        ax = fig.add_axes((0.07 + 0.45 * k, 0.24, 0.37, 0.58))
        A = np.array(M)
        im = ax.imshow(A, cmap="viridis_r", vmin=vmin, vmax=vmax, aspect="auto")
        r, cc = np.unravel_index(A.argmin(), A.shape)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=FS_SMALL,
                        color="white" if A[i, j] > 1.88 else "black",
                        fontweight="bold" if (i, j) == (r, cc) else None)
        ax.plot(cc, r, "*", ms=26, mfc="none", mec=C_ACCENT, mew=2.5)
        ax.set_xticks(range(5), SWEEP_D)
        ax.set_yticks(range(5), SWEEP_H)
        style_axes(ax, f"{t}: best {A.min():.3f} at {SWEEP_D[cc]}", "d_model / L_b",
                   "heads / L_b")
        ax.grid(False)
    fig.colorbar(im, ax=fig.axes, fraction=0.02, pad=0.01)
    fig.text(0.5, 0.11, "Fixed compute (~6.5e19 FLOPs) and active params (~2.3e8). "
             "AttnRes wins all 25 cells (by 0.019-0.063).\nIts optimum moves from "
             "d/L_b = 60 to 45: deeper and narrower, so it uses depth better. "
             "(A diagnostic, not a recipe: deeper is slower.)",
             ha="center", va="center", fontsize=FS_SMALL,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="0.55"))
    return fig


def step_21():
    fig = new_slide("Step 21: What the learned depth weights look like (Fig. 8)")
    ax = canvas(fig)
    obs = [("Locality is preserved",
            "Each layer puts most weight on its immediate predecessor: the diagonal "
            "dominates, so the usual residual path survives. Off-diagonal spots act as "
            "learned skip connections (e.g. layer 4 reaching early sources, layers 15-16 "
            "reaching back).", C_NEW, C_DARK),
           ("Layers specialise",
            "The embedding keeps non-trivial weight throughout, especially before attention "
            "layers. Pre-MLP inputs stay sharply local; pre-attention inputs look wider. "
            "Consistent with attention routing information and MLPs working locally.",
            C_GOAL, C_OK),
           ("Blocks keep the structure",
            "With N = 8 blocks the same three features appear, with sharper, more decisive "
            "weights. The paper suggests block compression acts as implicit regularisation.",
            C_OLD, C_ACCENT)]
    for k, (h, b, f, e) in enumerate(obs):
        card(ax, 0.3 + k * 3.85, 2.0, 3.6, 5.6, h, b, face=f, edge=e)
    note(ax, "Some sources attract weight regardless of input: depth-wise 'attention sinks', "
             "like the sinks seen in sequence attention.", y=1.1)
    return fig


def step_22():
    fig = new_slide("Step 22: Where AttnRes sits among residual variants (Table 5)")
    ax = canvas(fig)
    rows = [("Group", "Methods", "Weights", "What layer l can see"),
            ("Single state", "Residual, ReZero, LayerScale, Highway, DeepNorm, KEEL",
             "fixed / static / gated", "only h_{l-1}"),
            ("Multi-state", "SiameseNorm (2 streams), HC / mHC (m streams), DDL",
             "fixed or dynamic", "m streams"),
            ("Cross-layer, static", "DenseNet, ELMo, DenseFormer, ANCRe", "static after training",
             "all earlier outputs"),
            ("Cross-layer, dynamic", "MRLA (sigmoid, separable), MUDDFormer",
             "input-dependent", "all earlier outputs"),
            ("AttnRes", "Full / Block: softmax with one query per layer",
             "input-dependent", "all outputs / block sums")]
    xs, ws = [0.2, 2.6, 7.35, 9.45], [2.3, 4.65, 2.0, 2.35]
    for r, row in enumerate(rows):
        y = 7.2 - r * 1.08
        for x, w, txt in zip(xs, ws, row):
            face = C_DARK if r == 0 else (C_GOAL if r == len(rows) - 1 else C_MUTE)
            rbox(ax, x, y - 0.46, w, 0.92, face, "0.5", 1.0)
            ax.text(x + 0.1, y, wrap(txt, int(w * 8.5)), va="center", fontsize=FS_SMALL + 1,
                    color="#f9e79f" if r == 0 else "black",
                    fontweight="bold" if (r == 0 or x < 1) else None)
    note(ax, "AttnRes is orthogonal to normalisation and gating: it changes which\n"
             "sources a layer can reach and how they are weighted.", y=0.45)
    return fig


def step_23():
    fig = new_slide("Step 23: Takeaways and open questions")
    ax = canvas(fig)
    take = ("1. Residuals aggregate over depth with fixed unit weights; AttnRes makes that "
            "aggregation a learned softmax.\n"
            "2. Cost is one d-vector + one RMSNorm per layer; zero-init starts it as an average.\n"
            "3. Block AttnRes (~8 blocks) keeps most of the gain; caching and two-phase "
            "inference keep overhead < 4% (training) and < 2% (inference).\n"
            f"4. Same scaling slope, lower loss: ~{compute_multiplier():.2f}x compute "
            "equivalent; gains on every benchmark, biggest on reasoning.")
    card(ax, 0.3, 3.5, 11.4, 4.1, "What to remember", take, face=C_GOAL, edge=C_OK, width=95)
    card(ax, 0.3, 0.8, 5.6, 2.35, "Left open by the paper",
         "Input-dependent queries help more (1.731 vs 1.737) but cost a d x d projection and "
         "sequential decoding. Linear-cost depth attention is future work.",
         face=C_MUTE, edge=C_DARK)
    card(ax, 6.1, 0.8, 5.6, 2.35, "Read with care",
         "One lab, one architecture family (Kimi Linear); single runs, no error bars. "
         "Full AttnRes waits on faster interconnects.", face=C_WARN, edge=C_ACCENT)
    return fig


BUILDERS = [preface, step_1, step_2, step_3, step_4, step_5, step_6, step_6_1, step_7, step_8,
            step_9, step_9_1, step_10, step_10_1, step_11, step_12, step_13, step_13_1, step_14,
            step_15, step_16, step_17, step_18, step_19, step_20, step_21, step_22, step_23]


# ════════════════════════════════════════════════════════════════════════════
# Part F. Terminal demo
# ════════════════════════════════════════════════════════════════════════════

def print_demo():
    print(CITE)
    print("\n[toy] mean input norm at layers 1, 8, 16, 32 (random 32-layer PreNorm MLP):")
    std, full, blk, _ = toy_norms()
    for name, v in [("standard", std), ("full AttnRes", full), ("block AttnRes", blk)]:
        print(f"  {name:14s}", "  ".join(f"{v[i]:6.2f}" for i in (0, 7, 15, 31)))
    print(f"\n[check] two-phase vs direct block attention, max |diff| = {check_two_phase():.2e}")
    print("\n[Table 1] residual I/O per token per layer (L=128, N=8, S=16, m=4):")
    for k, v in io_per_layer().items():
        print(f"  {k:28s} {v:5.1f} d")
    n, c = pipeline_comm(4, 2)
    print(f"\n[Eq. 7/8] P=4, V=2: naive {n:.0f}, cached {c:.0f} block transfers")
    print("[prefill] 128K tokens, 8 blocks: "
          f"{prefill_memory_gb():.1f} GB -> {prefill_memory_gb(tp=8):.2f} GB/device -> "
          f"{prefill_memory_gb(tp=8, chunk=16384):.2f} GB/device (16K chunks)")
    print(f"\n[Fig. 4] baseline compute to match Block AttnRes at 5.6 PFLOP/s-days: "
          f"{compute_multiplier():.3f}x")
    gains = sorted(((b[3] - b[2], b[1]) for b in BENCH), reverse=True)
    print("[Table 3] largest gains:", ", ".join(f"{n} +{g:.1f}" for g, n in gains[:4]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention Residuals (arXiv:2603.15031)")
    parser.add_argument("--demo", action="store_true", help="print numeric checks")
    args = parser.parse_args()
    if args.demo:
        print_demo()
    else:
        run_slide_deck(BUILDERS)
