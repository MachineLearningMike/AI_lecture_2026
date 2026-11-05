"""
Proving and Improving the Platonic Representation Hypothesis with JEPAs

Paper read: https://arxiv.org/pdf/2610.39302  (arXiv v1, 29 Oct 2026)
Authors: Randall Balestriero (Brown, AMI Labs), Yann LeCun (NYU, AMI Labs)

The Platonic Representation Hypothesis (PRH, Huh et al. 2024) says that models
trained on different modalities drift toward one shared representation of the
world as they scale. This paper turns the observation into a theorem for JEPAs
and then uses the theorem as a training recipe.

Lecture arc (one slide per step):
  0. Preface: title, authors, publish date
  1. The question and the paper's claims
  2. Formal setup: a world, its renderings, positive pairs, a JEPA loss
  3. Theorem 1 (Platonic Representation Theorem): live numerical check
  4. Theorem 2: kernel mismatch bounds misalignment (re-creates Fig. 2)
  5. Proposition 2: k pairs identify the rotation, moments never do
  6. Good views vs bad views: nuisance costs alignment (Thm 3 / Prop 3)
  7. Proposition 4: next-token prediction is a bad-view JEPA
  8. Evidence on identical data (Flickr, MSR-VTT, ablations)
  9. Scale: 42x fewer samples, a 109M JEPA beats 72B language models
 10. A scaling law of alignment
 11. Platonic representations as a tool: captioning with one rotation
 12. Limitations and takeaways

Steps 3-6 run small numpy simulations of the theory. Steps 8-11 plot the
numbers reported in the paper (tables and figures cited on each slide).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

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
    run_slide_deck,
)

apply_rcparams(plt)

np.random.seed(14)
RNG = np.random.default_rng(14)

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

# One colour per objective, used on every plot
C_GOODV = C_OK        # good-view JEPA
C_BADV = "#e67e22"    # bad-view JEPA
C_NTP = C_ACCENT      # next-token prediction
C_LLM = "#7f8c8d"     # pretrained language models / references


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


def card(ax, x, y, w, h, title, body, face, edge, title_color=None):
    round_box(ax, x, y, w, h, face, edge, lw=1.8)
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top",
            fontsize=FS_ANNOT + 1, fontweight="bold", color=title_color or edge)
    ax.text(x + w / 2, y + (h - 0.6) / 2, body, ha="center", va="center",
            fontsize=FS_TICK - 1, color="0.15", linespacing=1.45)


def source_note(fig, text):
    fig.text(0.5, 0.115, text, ha="center", va="center", fontsize=FS_TICK - 3,
             color="0.4", style="italic")


# -------------------------------------------------------
# Small spectral toolkit (Section 2 of the paper)
# -------------------------------------------------------
def sym_sinkhorn(W, p, iters=500):
    """Scale a symmetric positive W to Pi = diag(d) W diag(d) with row sums p."""
    d = np.sqrt(p / W.sum(axis=1))
    for _ in range(iters):
        d = np.sqrt(d * p / (W @ d))
    return d[:, None] * W * d[None, :]


def normalized_adjacency(Pi):
    """K = D^{-1/2} Pi D^{-1/2}, D = diag(p): the graph that positives draw."""
    p = Pi.sum(axis=1)
    s = 1.0 / np.sqrt(p)
    return s[:, None] * Pi * s[None, :]


def top_eigs(K):
    vals, vecs = np.linalg.eigh(K)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def spectral_embedding(Pi, k):
    """Proposition 1: phi(z) = (v_1(z), ..., v_k(z)) / sqrt(p(z)), skipping v_0."""
    p = Pi.sum(axis=1)
    mu, V = top_eigs(normalized_adjacency(Pi))
    return V[:, 1:k + 1] / np.sqrt(p)[:, None], mu


def procrustes(A, B):
    """Orthogonal Q minimizing ||A - B Q||_F (rows paired)."""
    U, _, Vt = np.linalg.svd(B.T @ A)
    return U @ Vt


def random_orthogonal(k, rng=RNG):
    Q, R = np.linalg.qr(rng.normal(size=(k, k)))
    return Q * np.sign(np.diag(R))


# -------------------------------------------------------
# Preface: the paper
# -------------------------------------------------------
def build_preface():
    fig, ax = new_figure()
    blank(ax)

    ax.text(6.0, 7.3, "Paper reading", ha="center", va="center", fontsize=FS_ANNOT + 2,
            fontweight="bold", color="0.45")

    round_box(ax, 0.8, 4.3, 10.4, 2.5, C_NEW, C_DARK, lw=2.2)
    ax.text(6.0, 5.55,
            "Proving and Improving the Platonic\nRepresentation Hypothesis with JEPAs",
            ha="center", va="center", fontsize=FS_FIGURE_TITLE + 8, fontweight="bold",
            color=C_DARK, linespacing=1.3)

    ax.text(6.0, 3.45, "Randall Balestriero   &   Yann LeCun", ha="center", va="center",
            fontsize=FS_FIGURE_TITLE, fontweight="bold")
    ax.text(6.0, 2.8, "Brown University  ·  New York University  ·  AMI Labs", ha="center",
            va="center", fontsize=FS_ANNOT + 2, color="0.3")

    ax.text(6.0, 1.85, "Published 29 October 2026  (arXiv v1)", ha="center", va="center",
            fontsize=FS_ANNOT + 4, fontweight="bold", color=C_ACCENT)
    ax.text(6.0, 1.2, "arXiv:2610.39302  [cs.AI, cs.CV, stat.ML]   ·   https://arxiv.org/abs/2610.39302",
            ha="center", va="center", fontsize=FS_TICK, color="0.4")
    return fig


# -------------------------------------------------------
# Step 1: the question and the claims
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle("Step 1: Do separately trained models converge to one picture of the world?",
                 fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank(ax)

    round_box(ax, 0.4, 5.5, 3.6, 2.0, C_MUTE, "0.45", lw=1.6)
    ax.text(2.2, 7.15, "The paper", ha="center", fontsize=FS_ANNOT + 1, fontweight="bold")
    ax.text(2.2, 6.35,
            "Proving and Improving the Platonic\nRepresentation Hypothesis with JEPAs\n"
            "Balestriero & LeCun, arXiv:2610.39302",
            ha="center", va="center", fontsize=FS_TICK - 2, linespacing=1.4)

    round_box(ax, 4.3, 5.5, 3.6, 2.0, C_OLD, C_ACCENT, lw=1.6)
    ax.text(6.1, 7.15, "Before: the PRH", ha="center", fontsize=FS_ANNOT + 1,
            fontweight="bold", color=C_ACCENT)
    ax.text(6.1, 6.3,
            "Huh et al. (2024) observed that vision\nand language models grow more similar\n"
            "as they scale. An observation, no proof,\nso no guarantee and no recipe.",
            ha="center", va="center", fontsize=FS_TICK - 2, linespacing=1.4)

    round_box(ax, 8.2, 5.5, 3.4, 2.0, C_GOAL, C_OK, lw=1.6)
    ax.text(9.9, 7.15, "After: a theorem", ha="center", fontsize=FS_ANNOT + 1,
            fontweight="bold", color="#145a32")
    ax.text(9.9, 6.3,
            "When do models trained on\ndifferent modalities PROVABLY\n"
            "learn the same representation?\nAnswer: when they share positives.",
            ha="center", va="center", fontsize=FS_TICK - 2, linespacing=1.4)

    claims = [
        ("Theory", "JEPAs trained separately on any number of modalities learn the same embedding up to one rotation\n"
                   "when their positive pairs are two observations of one scene (good views); bounded misalignment otherwise."),
        ("Diagnosis", "Next-token prediction is a JEPA with bad views (a prefix and its continuation in one document),\n"
                      "which explains why language models align with vision only partly, and mostly through scale."),
        ("Evidence", "Good views need 42x fewer training samples than next-token prediction on the same corpus;\n"
                     "a 109M-parameter text JEPA aligns with DINOv2 better than every LLM evaluated, up to 72B."),
        ("Tool", "One rotation joins frozen image encoders to a text-only decoder: an image captioner (CIDEr .838 on COCO).\n"
                 "The rotation can even be found with zero pairs (64% top-5 pair-free zero-shot on ImageNet-100)."),
    ]
    colors = [C_DARK, C_ACCENT, C_OK, C_FUN]
    for i, ((tag, text), col) in enumerate(zip(claims, colors)):
        y = 4.35 - i * 1.1
        round_box(ax, 0.4, y, 1.9, 0.85, col, col, lw=0)
        ax.text(1.35, y + 0.42, tag, ha="center", va="center", fontsize=FS_ANNOT,
                fontweight="bold", color="white")
        ax.text(2.55, y + 0.42, text, ha="left", va="center", fontsize=FS_TICK - 2,
                linespacing=1.35)

    ax.text(6.0, 0.05, "Bottom line of the paper: positive views, not scale alone, are the lever for platonic representations.",
            ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    return fig


# -------------------------------------------------------
# Step 2: formal setup
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle("Step 2: The setup: one world, many renderings, positive pairs",
                 fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank(ax)

    # world -> renderings -> encoders -> R^k
    round_box(ax, 0.3, 3.4, 2.2, 1.6, C_WARN, "#b7950b", lw=1.8)
    ax.text(1.4, 4.55, "world state z", ha="center", fontsize=FS_ANNOT, fontweight="bold")
    ax.text(1.4, 3.9, "finite set Z,\np(z) > 0", ha="center", va="center", fontsize=FS_TICK - 2)

    for y, name, x_txt, col, face in [
        (5.6, "image", "x = g_img(z)\npixels of a photo", C_DARK, C_NEW),
        (1.8, "text", "x = g_txt(z)\ntokens of a caption", C_ACCENT, C_OLD),
    ]:
        round_box(ax, 3.3, y, 2.3, 1.3, face, col, lw=1.6)
        ax.text(4.45, y + 0.65, x_txt, ha="center", va="center", fontsize=FS_TICK - 2)
        round_box(ax, 6.4, y, 2.0, 1.3, face, col, lw=1.6)
        ax.text(7.4, y + 0.65, f"{name} JEPA\nencoder f", ha="center", va="center",
                fontsize=FS_TICK - 1, fontweight="bold", color=col)
        arrow(ax, 2.5, 4.2, 3.3, y + 0.65, color=col)
        arrow(ax, 5.6, y + 0.65, 6.4, y + 0.65, color=col)
        arrow(ax, 8.4, y + 0.65, 9.2, 4.2, color=col)
    ax.text(4.45, 4.2, "injective renderings g_m\n(never shown to each other)", ha="center",
            va="center", fontsize=FS_TICK - 3, color="0.35", style="italic")

    round_box(ax, 9.2, 3.4, 2.5, 1.6, C_GOAL, C_OK, lw=1.8)
    ax.text(10.45, 4.55, r"embedding in $\mathbb{R}^k$", ha="center", fontsize=FS_ANNOT, fontweight="bold",
            color="#145a32")
    ax.text(10.45, 3.9, "same up to\none rotation?", ha="center", va="center", fontsize=FS_TICK - 2)

    # equations
    round_box(ax, 0.3, 0.1, 5.6, 1.45, C_MUTE, "0.45", lw=1.4)
    ax.text(3.1, 1.2, "JEPA loss of modality m  (Eq. 1)", ha="center", fontsize=FS_TICK - 1, fontweight="bold")
    ax.text(3.1, 0.55,
            r"$L_m(f)=\mathbb{E}_{(z,z')\sim\pi_m}\,\ell(f(g_m(z)),f(g_m(z')))+R(f)$",
            ha="center", fontsize=FS_ANNOT)

    round_box(ax, 6.1, 0.1, 5.6, 1.45, C_MUTE, "0.45", lw=1.4)
    ax.text(8.9, 1.2, "Lemma 1: a rendering only changes coordinates", ha="center",
            fontsize=FS_TICK - 1, fontweight="bold")
    ax.text(8.9, 0.55, r"$L_m(f)=L(f\circ g_m)$,   one problem on world states $Z$",
            ha="center", fontsize=FS_ANNOT)

    ax.text(6.0, 7.55,
            r"$\pi_m$ = which pairs of world states modality m declares positives (views of each other).   "
            r"$\ell$ = prediction term, $R$ = anti-collapse;  both rotation invariant.",
            ha="center", fontsize=FS_TICK - 2, color="0.25")
    ax.text(6.0, 7.15,
            "Fits this form: spectral contrastive, whitened MSE, VICReg / Barlow Twins (whitened limit), LeJEPA's SIGReg (in expectation).",
            ha="center", fontsize=FS_TICK - 2, color="0.25")
    return fig


# -------------------------------------------------------
# Step 3: Theorem 1 numerically
# -------------------------------------------------------
def build_step_3():
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle("Step 3: Theorem 1 (Platonic Representation Theorem), checked numerically",
                 fontsize=FS_FIGURE_TITLE, fontweight="bold")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.42], hspace=0.35, wspace=0.28,
                          top=0.88, bottom=0.14)

    # A finite world: 3 clusters of 20 states in a 2-D latent space
    centers = np.array([[0.0, 2.0], [-1.8, -1.0], [1.8, -1.0]])
    z = np.vstack([c + 0.45 * RNG.normal(size=(20, 2)) for c in centers])
    label = np.repeat(np.arange(3), 20)
    p = RNG.uniform(0.5, 1.5, size=len(z))
    p /= p.sum()
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    Pi = sym_sinkhorn(np.exp(-d2 / (2 * 0.8 ** 2)), p)

    k = 2
    phi, mu = spectral_embedding(Pi, k)
    # Each separately trained JEPA lands on SOME member of {Q phi : Q in O(k)}
    Q_img, Q_txt = random_orthogonal(k), random_orthogonal(k)
    e_img, e_txt = phi @ Q_img.T, phi @ Q_txt.T
    Q_hat = procrustes(e_img, e_txt)
    err = np.sqrt((p[:, None] * (e_img - e_txt @ Q_hat) ** 2).sum())

    cols = np.array(["tab:blue", "tab:orange", "tab:green"])[label]
    ax0 = fig.add_subplot(gs[0, 0])
    w = Pi / Pi.max()
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            if w[i, j] > 0.15:
                ax0.plot(*z[[i, j]].T, color="0.6", lw=0.6 * w[i, j], zorder=1)
    ax0.scatter(*z.T, c=cols, s=2500 * p, edgecolor="k", lw=0.4, zorder=2)
    ax0.set_title(r"world $Z$ + positive pairs $\pi$", fontsize=FS_AX_TITLE - 2)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.scatter(*e_img.T, c=cols, s=40, edgecolor="k", lw=0.3)
    ax1.set_title("image JEPA embedding", fontsize=FS_AX_TITLE - 2)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(*e_txt.T, c=cols, s=40, marker="s", edgecolor="k", lw=0.3)
    ax2.set_title("text JEPA embedding", fontsize=FS_AX_TITLE - 2)
    ax3 = fig.add_subplot(gs[0, 3])
    aligned = e_txt @ Q_hat
    ax3.scatter(*e_img.T, c=cols, s=90, alpha=0.35, label="image")
    ax3.scatter(*aligned.T, c=cols, s=18, marker="s", edgecolor="k", lw=0.3, label="text, rotated")
    ax3.set_title("after ONE rotation", fontsize=FS_AX_TITLE - 2)
    ax3.legend(fontsize=FS_LEGEND - 4, loc="lower right")
    ax3.text(0.5, 0.94, f"RMS error = {err:.1e}", transform=ax3.transAxes, ha="center",
             fontsize=FS_TICK - 1, fontweight="bold", color=C_OK)
    for a in (ax0, ax1, ax2, ax3):
        a.set_xticks([])
        a.set_yticks([])
        a.set_aspect("equal", adjustable="datalim")

    axt = fig.add_subplot(gs[1, :])
    blank(axt, ylim=(0, 3))
    round_box(axt, 0.1, 0.15, 7.0, 2.7, C_GOAL, C_OK, lw=1.6)
    axt.text(3.6, 2.45, "Theorem 1 + Proposition 1", ha="center", fontsize=FS_ANNOT, fontweight="bold",
             color="#145a32")
    axt.text(3.6, 1.25,
             r"If $M\geq2$ modalities share one positive-pair law $\pi$ and minimizers are unique up to rotation, then"
             "\n"
             r"every minimizer satisfies  $f_m\circ g_m = Q_m\,h$,  $Q_m\in O(k)$:  one world embedding $h$, one rotation each."
             "\n"
             r"Whitened MSE: $h$ = spectral embedding $\phi(z)=(v_1(z),\dots,v_k(z))/\sqrt{p(z)}$ of $K=D^{-1/2}\Pi D^{-1/2}$,"
             "\n"
             r"unique up to rotation when $\mu_k>\mu_{k+1}$ (Laplacian eigenmaps, population form).",
             ha="center", va="center", fontsize=FS_TICK - 2, linespacing=1.55)
    round_box(axt, 7.35, 0.15, 4.55, 2.7, C_MUTE, "0.45", lw=1.4)
    axt.text(9.62, 1.5,
             "What this demo does: 60 world states,\n"
             "Gaussian positives, eigendecomposition of K.\n"
             f"Spectral gap here: mu_2 - mu_3 = {mu[k] - mu[k + 1]:.3f} > 0.\n"
             "Each modality gets an arbitrary rotation\n"
             "(what separate training would pick);\n"
             "orthogonal Procrustes undoes it exactly.",
             ha="center", va="center", fontsize=FS_TICK - 3, linespacing=1.45)
    return fig


# -------------------------------------------------------
# Step 4: Theorem 2 (re-creates Figure 2)
# -------------------------------------------------------
def theorem2_experiment(n_worlds=300, rng=RNG):
    errs, bounds, clustered = [], [], []
    for w in range(n_worlds):
        n = int(rng.integers(8, 40))
        k = int(rng.integers(1, 6))
        p = rng.dirichlet(np.ones(n) * 2.0)
        is_clu = w % 2 == 1
        if is_clu:
            # clustered world: near-block kernel, so eigengaps are small
            g = rng.integers(0, k + 1, size=n)
            W = 0.02 + (g[:, None] == g[None, :]) + 0.05 * rng.random((n, n))
        else:
            W = rng.random((n, n)) + 0.05
        W = (W + W.T) / 2
        Pi1 = sym_sinkhorn(W, p)
        U = rng.random((n, n)) + 0.05
        Pi_u = sym_sinkhorn((U + U.T) / 2, p)
        eps = 10 ** rng.uniform(-4, 0)
        Pi2 = (1 - eps) * Pi1 + eps * Pi_u          # same marginal p, partly different positives

        K1, K2 = normalized_adjacency(Pi1), normalized_adjacency(Pi2)
        mu1, V1 = top_eigs(K1)
        mu2, V2 = top_eigs(K2)
        gap = mu2[k] - mu2[k + 1]
        if gap < 1e-9:
            continue
        A, B = V1[:, 1:k + 1], V2[:, 1:k + 1]          # rows = sqrt(p) * phi
        Q = procrustes(A, B)
        err = np.linalg.norm(A - B @ Q)                 # = (E_p ||phi1 - Q phi2||^2)^{1/2}
        dK = K1 - K2
        bound = 2 ** 1.5 / gap * min(np.sqrt(k) * np.linalg.norm(dK, 2), np.linalg.norm(dK, "fro"))
        errs.append(err)
        bounds.append(bound)
        clustered.append(is_clu)
    return np.array(errs), np.array(bounds), np.array(clustered)


def build_step_4():
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle("Step 4: Theorem 2: kernel mismatch bounds misalignment",
                 fontsize=FS_FIGURE_TITLE, fontweight="bold")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.12, top=0.88, bottom=0.16)

    errs, bounds, clu = theorem2_experiment()
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(bounds[~clu], errs[~clu], s=22, alpha=0.75, color=C_DARK, label="random world")
    ax.scatter(bounds[clu], errs[clu], s=22, alpha=0.75, color=C_BADV, label="clustered world (small gaps)")
    lo, hi = 1e-5, max(bounds.max(), 1.0) * 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="error = bound")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("bound of Theorem 2 (log)", fontsize=FS_AXIS)
    ax.set_ylabel("error after best rotation (log)", fontsize=FS_AXIS)
    ax.legend(fontsize=FS_LEGEND - 3, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    ratio = (errs / bounds).max()
    ax.set_title(f"{len(errs)} random worlds: all below the bound (max ratio {ratio:.2f})",
                 fontsize=FS_AX_TITLE - 3)

    axt = fig.add_subplot(gs[0, 1])
    blank(axt, ylim=(0, 8))
    round_box(axt, 0.3, 4.3, 11.4, 3.5, C_GOAL, C_OK, lw=1.6)
    axt.text(6.0, 7.35, "Theorem 2", ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color="#145a32")
    axt.text(6.0, 5.75,
             r"$\min_{Q\in O(k)}\left(\mathbb{E}_p\,\|\phi_1(z)-Q\phi_2(z)\|^2\right)^{1/2}"
             r"\;\leq\;\frac{2^{3/2}}{\delta}\,\min\left(\sqrt{k}\,\|K_1-K_2\|_2,\;\|K_1-K_2\|_F\right)$",
             ha="center", va="center", fontsize=FS_ANNOT - 2)
    axt.text(6.0, 4.75, r"$\delta=\mu_k-\mu_{k+1}$ = eigengap of $K_2$  (proof: Davis-Kahan variant, Yu et al. 2015)",
             ha="center", fontsize=FS_TICK - 4, color="0.25")

    axt.text(0.4, 3.75, "Only two quantities matter:", ha="left", fontsize=FS_ANNOT, fontweight="bold")
    axt.text(0.6, 2.2,
             "1.  How different the graphs drawn by the two modalities'\n"
             "     positives are   (numerator: kernel mismatch)\n"
             "2.  How separated the top of the spectrum is\n"
             "     (denominator: eigengap)\n\n"
             "Corollary (block version): directions separated by large\n"
             "gaps are protected; tail directions need not align at all.",
             ha="left", va="center", fontsize=FS_TICK - 2, linespacing=1.4)
    axt.text(6.0, 0.15,
             r"Demo: $\Pi_2=(1-\varepsilon)\Pi_1+\varepsilon\,\Pi_{\mathrm{unrelated}}$, $\varepsilon\in[10^{-4},1]$, "
             "8-39 states, k = 1..5\n(paper's Fig. 2 reports error at most 18% of the bound)",
             ha="center", fontsize=FS_TICK - 3, color="0.35", style="italic")
    return fig


# -------------------------------------------------------
# Step 5: Proposition 2, what pairs are for
# -------------------------------------------------------
def build_step_5():
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle("Step 5: Proposition 2: moments cannot find the rotation; k pairs can",
                 fontsize=FS_FIGURE_TITLE, fontweight="bold")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.3, top=0.86, bottom=0.2)

    # (i) whitened embedding and its rotation have identical first two moments
    h = RNG.normal(size=(400, 2))
    h = (h - h.mean(0)) @ np.linalg.inv(np.linalg.cholesky(np.cov(h.T))).T
    Q = random_orthogonal(2)
    hq = h @ Q.T
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(*h.T, s=8, alpha=0.5, color=C_DARK, label="modality 1: h")
    ax.scatter(*hq.T, s=8, alpha=0.5, color=C_ACCENT, label="modality 2: Qh")
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(2 * np.cos(t), 2 * np.sin(t), "k--", lw=1.2, label="2-sigma ellipse (both)")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=FS_LEGEND - 4, loc="upper right")
    ax.set_title("(i) mean 0, cov I for BOTH\nunpaired moments are blind to Q", fontsize=FS_AX_TITLE - 3)

    # (ii) Procrustes from n pairs; noise = residual misalignment of Theorem 2
    k = 16
    world = RNG.normal(size=(5000, k))
    Q_true = random_orthogonal(k)
    ns = np.unique(np.logspace(np.log10(2), np.log10(2000), 30).astype(int))
    ax = fig.add_subplot(gs[0, 1])
    for noise, col in [(0.0, C_OK), (0.5, C_DARK), (1.5, C_BADV)]:
        e1 = world
        e2 = world @ Q_true.T + noise * RNG.normal(size=world.shape)
        errs = []
        for n in ns:
            Qh = procrustes(e2[:n], e1[:n])  # e2 ~ e1 Q_hat
            errs.append(np.linalg.norm(Qh - Q_true.T) / np.sqrt(k))
        ax.plot(ns, errs, "o-", ms=3, color=col, label=f"residual noise {noise}")
    ax.axvline(k, color="0.3", ls=":", lw=1.5)
    ax.text(k * 1.1, 1.25, f"n = k = {k}", fontsize=FS_TICK - 3)
    ax.set_xscale("log")
    ax.set_xlabel("number of paired examples n", fontsize=FS_AXIS - 1)
    ax.set_ylabel("rotation error ||Q_hat - Q|| / sqrt(k)", fontsize=FS_AXIS - 2)
    ax.legend(fontsize=FS_LEGEND - 4)
    ax.grid(alpha=0.25)
    ax.set_title("(ii) orthogonal Procrustes from n pairs", fontsize=FS_AX_TITLE - 3)

    axt = fig.add_subplot(gs[0, 2])
    blank(axt)
    card(axt, 0.2, 4.3, 11.6, 3.6, "Statement",
         "(i) every rotation of a whitened embedding is whitened,\n"
         "so no first/second-moment statistic of unpaired data\n"
         "identifies the rotation.\n"
         "(ii) if n >= k paired states span R^k, the rotation is the\n"
         "unique orthogonal Procrustes minimizer.",
         C_NEW, C_DARK)
    card(axt, 0.2, 0.2, 11.6, 3.8, "In practice",
         "Pairs enter ONLY through one k x k orthogonal matrix.\n"
         "With mismatched positives, more pairs average out the\n"
         "residual of Theorem 2 (curves above).\n"
         "Different widths (DINOv2 768 vs text JEPA 384): a linear\n"
         "map with orthonormal rows, 0.22M parameters.\n"
         "Higher-order geometry CAN find Q with zero pairs (Step 11).",
         C_MUTE, "0.45", title_color="0.2")
    return fig


# -------------------------------------------------------
# Step 6: good vs bad views, nuisance (Thm 3, Prop 3)
# -------------------------------------------------------
def nuisance_experiment(taus, n_states=12, n_nuis=4, k=4, rng=RNG):
    """Product kernel K_state (x) K_nuis on observations (z, xi) (Prop. 3 setting).

    Good views draw independent nuisances: K_nuis = 11^T / n (keeps 0 nuisance).
    Bad views corrupt one observation: K_nuis keeps correlation tau of its nuisance.
    Returns the scene-explained fraction rho(f) = tr(C_f^{1/2}) / k.
    """
    zlat = np.sort(rng.uniform(0, 1, size=n_states))
    Pi = sym_sinkhorn(np.exp(-(zlat[:, None] - zlat[None, :]) ** 2 / (2 * 0.12 ** 2)),
                      np.ones(n_states) / n_states)
    Ks = normalized_adjacency(Pi)
    lam = top_eigs(Ks)[0]
    J = np.ones((n_nuis, n_nuis)) / n_nuis

    def rho_of(Kn):
        K = np.kron(Ks, Kn)
        vals, vecs = top_eigs(K)
        f = vecs[:, 1:k + 1] * np.sqrt(len(K))                 # whitened under uniform p
        cond = f.reshape(n_states, n_nuis, k).mean(axis=1)    # E[f | z]
        C = np.cov(cond.T, bias=True)
        ev = np.clip(np.linalg.eigvalsh(C), 0, None)
        return np.sqrt(ev).sum() / k

    good = np.array([rho_of(J) for _ in taus])
    bad = np.array([rho_of((1 - t) * J + t * np.eye(n_nuis)) for t in taus])
    return good, bad, lam[k]


def build_step_6():
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle("Step 6: Good views ignore nuisance, bad views capture it (Thm 3, Prop 3)",
                 fontsize=FS_FIGURE_TITLE, fontweight="bold")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.32, top=0.86, bottom=0.2)

    axd = fig.add_subplot(gs[0, 0])
    blank(axd, ylim=(0, 10))
    card(axd, 0.1, 6.7, 11.8, 3.1, "Good view",
         'two observations of ONE scene\n"a dog runs on the beach"\n"a puppy plays by the sea"\nshare only the scene',
         C_GOAL, C_OK, title_color="#145a32")
    card(axd, 0.1, 3.4, 11.8, 3.1, "Bad view",
         'two corruptions of ONE observation\n"a dog [M] on the beach"\n"a dog runs [M] the beach"\nshare scene AND wording (nuisance)',
         C_OLD, C_BADV)
    axd.text(6.0, 2.85, "Theorem 3 (nuisance costs alignment)", ha="center",
             fontsize=FS_TICK, fontweight="bold")
    axd.text(6.0, 1.95,
             r"$\min_Q\mathbb{E}\|Qf_1(x_1)-f_2(x_2)\|^2\;\geq\;2k\,(1-\min_m\rho(f_m))$",
             ha="center", fontsize=FS_TICK)
    axd.text(6.0, 0.75,
             r"$\rho(f)=\mathrm{tr}\,C_f^{1/2}/k$: scene-explained fraction"
             "\n"
             r"($C_f$ = Cov $\mathbb{E}[f(x)\,|\,z]$;  $\rho=1$ iff f depends on z alone)",
             ha="center", fontsize=FS_TICK - 3, color="0.3")

    taus = np.linspace(0.0, 0.98, 50)
    good, bad, lam_k = nuisance_experiment(taus)
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(taus, good, color=C_GOODV, lw=3, label="good views")
    ax.plot(taus, bad, color=C_BADV, lw=3, label="bad views")
    ax.axvline(lam_k, color="0.3", ls=":", lw=1.5)
    ax.text(lam_k + 0.02, 0.55, r"$\tau=\lambda_k$", fontsize=FS_TICK - 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(r"nuisance kept by a corruption, $\tau$", fontsize=FS_AXIS - 1)
    ax.set_ylabel(r"scene-explained fraction $\rho$", fontsize=FS_AXIS - 1)
    ax.legend(fontsize=FS_LEGEND - 3, loc="lower left")
    ax.grid(alpha=0.25)
    ax.set_title("Simulation of Prop. 3 (k = 4)\nbad views lose directions\n" r"once $\tau>\lambda_k$",
                 fontsize=FS_AX_TITLE - 4)

    ax = fig.add_subplot(gs[0, 2])
    names = ["good-view\nJEPA", "bad-view\nJEPA", "next-token"]
    rho = [0.804, 0.655, 0.608]
    lb = [12.6, 22.1, 25.1]
    meas = [33.5, 39.4, 40.3]
    cols = [C_GOODV, C_BADV, C_NTP]
    x = np.arange(3)
    ax.bar(x - 0.2, lb, 0.38, color=cols, alpha=0.45, label="Thm 3 lower bound")
    ax.bar(x + 0.2, meas, 0.38, color=cols, label="measured error vs DINOv2")
    for i in range(3):
        ax.text(x[i], meas[i] + 1.5, f"rho={rho[i]:.3f}", ha="center", fontsize=FS_TICK - 3, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=FS_TICK - 2)
    ax.set_ylim(0, 50)
    ax.set_ylabel("best-rotation error", fontsize=FS_AXIS - 1)
    ax.legend(fontsize=FS_LEGEND - 4, loc="upper left")
    ax.set_title("Measured on Flickr (k = 32)\n5 captions per image\n= 5 nuisances",
                 fontsize=FS_AX_TITLE - 4)
    source_note(fig, "Right panel: numbers reported in Section 3.1 of the paper (Table 6). "
                     "Middle panel: product-kernel simulation, 12 states x 4 nuisances.")
    return fig


# -------------------------------------------------------
# Step 7: next-token prediction is a bad-view JEPA (Prop 4)
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle("Step 7: Proposition 4: next-token prediction is a JEPA with bad views",
                 fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank(ax)

    round_box(ax, 0.5, 5.6, 11.0, 1.9, C_MUTE, "0.45", lw=1.6)
    ax.text(6.0, 6.95,
            r"$-\log\frac{e^{\langle h(c),w_y\rangle}}{\sum_v e^{\langle h(c),w_v\rangle}}"
            r"\;=\;-\langle h(c),w_y\rangle\;+\;\log\sum_v e^{\langle h(c),w_v\rangle}$",
            ha="center", va="center", fontsize=FS_ANNOT + 6)
    ax.text(5.95, 6.1, "prediction term", ha="center", fontsize=FS_TICK - 1, fontweight="bold", color=C_NTP)
    ax.text(8.4, 6.1, "anti-collapse A(h(c))", ha="center", fontsize=FS_TICK - 1, fontweight="bold",
            color=C_DARK)
    ax.text(6.0, 5.72, "context encoder h, target encoder y -> w_y, inner-product prediction  (Eq. 6)",
            ha="center", fontsize=FS_TICK - 2, color="0.3")

    card(ax, 0.5, 2.5, 3.5, 2.8, "(i) The positives",
         "a prefix and its continuation\ninside ONE document\n\n= corruptions of a single\nobservation: bad views",
         C_OLD, C_NTP)
    card(ax, 4.25, 2.5, 3.5, 2.8, "(ii) The anti-collapse",
         "log-partition A does not depend\non the target y; as t -> 0 the loss\n"
         "becomes a margin, like the\nuniformity term of contrastive\nlearning",
         C_NEW, C_DARK)
    card(ax, 8.0, 2.5, 3.5, 2.8, "(iii) The identifiability",
         "minimizers: <h(c), w_v> =\nlog p(v|c) + b(c)\n\ninvariant under GL(k):\n"
         "an invertible map, NOT a rotation",
         C_WARN, "#b7950b", title_color="#7d6608")

    round_box(ax, 0.5, 0.25, 11.0, 1.95, C_GOAL, C_OK, lw=1.6)
    ax.text(6.0, 1.25,
            "Reading (heuristic, then tested): a prefix and its continuation share the document's nuisance (author, wording, style),\n"
            "so next-token prediction spends directions on it. Contexts with the same next token become implicit positives: very weak views,\n"
            "chained only by very dense data, plus spill across scenes through frequent tokens. Why LLMs align with vision mostly through scale.",
            ha="center", va="center", fontsize=FS_TICK - 2, linespacing=1.5)
    return fig


# -------------------------------------------------------
# Step 8: evidence on identical data
# -------------------------------------------------------
def build_step_8():
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle("Step 8: Same data, same encoder, only the positives change: good views win",
                 fontsize=FS_FIGURE_TITLE, fontweight="bold")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.9, 1.0], wspace=0.6, top=0.86, bottom=0.22, right=0.97)

    # Table 1: image JEPA vs text encoders, Flickr30k, all trained from scratch
    ax = fig.add_subplot(gs[0, 0])
    metrics = ["mutual 10-NN", "R@1", "R@5"]
    rows = [
        ("good-view JEPA", [0.062, 0.044, 0.133], C_GOODV),
        ("bad-view JEPA", [0.047, 0.027, 0.095], C_BADV),
        ("next-token", [0.042, 0.021, 0.077], C_NTP),
        ("next-token, caption pairs", [0.043, 0.023, 0.083], "#f1948a"),
    ]
    x = np.arange(3)
    w = 0.2
    for i, (name, vals, col) in enumerate(rows):
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=name)
        if i == 0:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + w / 2, v + 0.003, f"{v:.3f}", ha="center", fontsize=FS_TICK - 4,
                        fontweight="bold")
    for xi, c in zip(x, [0.010, 0.001, 0.005]):
        ax.plot([xi - 0.45, xi + 0.45], [c, c], "k--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=FS_TICK - 1)
    ax.legend(fontsize=FS_LEGEND - 4, loc="upper left")
    ax.set_title("Flickr30k, image JEPA vs text encoder\n(Table 1; dashed = chance)", fontsize=FS_AX_TITLE - 4)

    # MSR-VTT: good views help on both sides
    ax = fig.add_subplot(gs[0, 1])
    vals = np.array([[0.067, 0.043], [0.059, 0.039]])  # rows: text good/bad, cols: image good/bad
    im = ax.imshow(vals, cmap="Greens", vmin=0.03, vmax=0.07)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=FS_ANNOT + 2,
                    fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["good\n(frames)", "bad\n(crops)"], fontsize=FS_TICK - 2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["good\n(captions)", "bad\n(spans)"], fontsize=FS_TICK - 2)
    ax.set_xlabel("image-side views", fontsize=FS_AXIS - 2)
    ax.set_ylabel("text-side views", fontsize=FS_AXIS - 2)
    ax.set_title("MSR-VTT video, mutual 10-NN\ngood views help on BOTH sides", fontsize=FS_AX_TITLE - 4)
    _ = im

    # Figure 7: trimming is the knob that matters
    ax = fig.add_subplot(gs[0, 2])
    names = ["good,\ndefault", "good, no\ntrimming", "bad-view", "next-token"]
    r1 = [0.246, 0.185, 0.13, 0.12]
    cols = [C_GOODV, "#82e0aa", C_BADV, C_NTP]
    y = np.arange(4)[::-1]
    ax.barh(y, r1, color=cols)
    for yi, v in zip(y, r1):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=FS_TICK - 2, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=FS_TICK - 3)
    ax.set_xlim(0, 0.31)
    ax.set_xlabel("R@1 with frozen DINOv2 (one rotation)", fontsize=FS_AXIS - 2)
    ax.set_title("Ablations (Fig. 7):\ntrimming matters most",
                 fontsize=FS_AX_TITLE - 4)

    fig.text(0.5, 0.1,
             "Key control: next-token prediction trained on two captions of the SAME image barely moves (R@1 .021 -> .023).\n"
             "The gain comes from the objective, not from the pairing information.",
             ha="center", fontsize=FS_TICK - 2, fontweight="bold", color=C_DARK)
    return fig


# -------------------------------------------------------
# Step 9: scale
# -------------------------------------------------------
def build_step_9():
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle("Step 9: Next-token prediction is an inefficient route to platonic representations",
                 fontsize=FS_FIGURE_TITLE, fontweight="bold")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.55, top=0.86, bottom=0.2, right=0.95)

    ax = fig.add_subplot(gs[0, 0])
    M = [21.6e6, 900e6]
    ax.plot(M, [0.109, 0.184], "o-", color=C_GOODV, lw=3, ms=10, label="good-view JEPA, 109M")
    ax.plot([21.6e6], [0.108], "o", color=C_GOODV, mfc="white", mew=2, ms=10, label="good-view JEPA, 33M")
    ax.plot([21.6e6], [0.071], "s", color=C_NTP, mfc="white", mew=2, ms=10, label="next-token, 33M")
    ax.plot([900e6], [0.111], "s", color=C_NTP, ms=11, label="next-token, 109M")
    ax.plot([21.6e6], [0.053], "^", color=C_BADV, mfc="white", mew=2, ms=10, label="bad-view JEPA, 33M")
    ax.plot([900e6], [0.088], "^", color=C_BADV, ms=11, label="bad-view JEPA, 109M")
    ax.axhline(0.154, color=C_LLM, ls="--", lw=1.5)
    ax.text(14e6, 0.157, "Qwen2.5-72B (.154), Llama-3-70B (.151): 15-18T tokens", fontsize=FS_TICK - 4,
            color="0.3")
    ax.annotate("", xy=(24e6, 0.111), xytext=(850e6, 0.111),
                arrowprops=dict(arrowstyle="->", color="k", lw=2))
    ax.text(140e6, 0.115, "42x fewer samples", ha="center", fontsize=FS_TICK - 1, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(12e6, 2e9)
    ax.set_ylim(0.04, 0.2)
    ax.set_xlabel("training samples seen (log)", fontsize=FS_AXIS - 1)
    ax.set_ylabel("mutual 10-NN with DINOv2 on WIT-1024", fontsize=FS_AXIS - 1)
    ax.legend(fontsize=FS_LEGEND - 5, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    ax.grid(alpha=0.25, which="both")
    ax.set_title("Same corpus (texts of web images, Recap-DataComp)\nFig. 1 right, reported points",
                 fontsize=FS_AX_TITLE - 4)

    ax = fig.add_subplot(gs[0, 1])
    rows = [
        ("good-view 109M", 0.184, C_GOODV),
        ("Qwen2.5-72B", 0.154, C_LLM),
        ("Llama-3-70B", 0.151, C_LLM),
        ("CLIP text", 0.124, "#5d6d7e"),
        ("all-mpnet", 0.118, "#5d6d7e"),
        ("next-token 109M", 0.111, C_NTP),
        ("bad-view 109M", 0.088, C_BADV),
    ]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows])
    for yi, (_, v, _) in zip(y, rows):
        ax.text(v + 0.002, yi, f"{v:.3f}", va="center", fontsize=FS_TICK - 2, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS_TICK - 3)
    ax.set_xlim(0, 0.21)
    ax.set_xlabel("WIT-1024 mutual 10-NN with DINOv2 ViT-B", fontsize=FS_AXIS - 2)
    ax.set_title("Table 3: the PRH paper's own benchmark", fontsize=FS_AX_TITLE - 4)

    fig.text(0.5, 0.09,
             "The 109M encoder read 116B text tokens and never saw a pixel; it beats models 660x larger trained on 100x more. "
             "Caveat: most of its texts were written by an image captioner (LLaVA, CLIP ViT-L vision).",
             ha="center", fontsize=FS_TICK - 3, color="0.25")
    return fig


# -------------------------------------------------------
# Step 10: a scaling law of alignment
# -------------------------------------------------------
def build_step_10():
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle("Step 10: A scaling law for the PRH", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], wspace=0.15, top=0.86, bottom=0.18)

    # Illustrative constants chosen to give the shape of Fig. 11, NOT the paper's fit
    a, d, b, c, alpha, beta, N = 0.17, 0.025, 0.03, 0.146, 0.30, 0.19, 10.2
    Ms = np.logspace(np.log10(5), np.log10(80), 100)  # millions of samples
    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.get_cmap("viridis")
    for i, q in enumerate([0.0, 0.01, 0.1, 0.3, 1.0]):
        r1 = a + d * np.log(Ms) + b * np.log1p(q * Ms ** alpha * N ** beta / c)
        ax.plot(Ms, r1, lw=3, color="k" if q == 0 else cmap(0.15 + 0.2 * i),
                label=f"q = {q:g}" + ("  (bad views only)" if q == 0 else ""))
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 40, 80])
    ax.set_xticklabels(["5M", "10M", "20M", "40M", "80M"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("samples seen M (log)", fontsize=FS_AXIS - 1)
    ax.set_ylabel("Flickr R@1 with DINOv2", fontsize=FS_AXIS - 1)
    ax.legend(fontsize=FS_LEGEND - 3, title="fraction of good views", title_fontsize=FS_LEGEND - 3)
    ax.grid(alpha=0.25, which="both")
    ax.set_title("Shape of Eq. 7 (illustrative constants, not the fitted ones)", fontsize=FS_AX_TITLE - 4)

    axt = fig.add_subplot(gs[0, 1])
    blank(axt)
    round_box(axt, 0.2, 5.6, 11.6, 2.2, C_GOAL, C_OK, lw=1.6)
    axt.text(6.0, 7.3, "Eq. 7: bad-view floor + good-view term", ha="center", fontsize=FS_ANNOT,
             fontweight="bold", color="#145a32")
    axt.text(6.0, 6.3, r"$R@1 = a + d\,\log M + b\,\log\left(1 + q\,M^{\alpha}N^{\beta}/c\right)$",
             ha="center", fontsize=FS_ANNOT + 4)
    axt.text(6.0, 5.3, "q = fraction of items with good views,  M = samples seen,  N = unique images",
             ha="center", fontsize=FS_TICK - 2, color="0.3")
    facts = [
        "Fitted on a 45-cell grid up to 80M samples: RMSE .007",
        "85 runs at random (q, M) predicted BEFORE training: RMSE .0093,",
        "    while two seeds of the same run differ by .0096",
        "Iso-alignment lines: q ~ M^-0.30, more samples substitute for",
        "    good views, but only sub-linearly",
        "Unique images enter weakly (beta = .19); without good views",
        "    WIT stays at .054-.057 from 0.41M to 50.9M images",
        "Breaks further out: at 320M samples it over-predicts good views",
        "    (no term for repeated data); exponents poorly identified",
    ]
    axt.text(0.5, 2.6, "\n".join(facts), ha="left", va="center", fontsize=FS_TICK - 2, linespacing=1.5)
    axt.text(6.0, 0.0, "An empirical summary of one recipe (BERT-S), not a consequence of Theorem 4.",
             ha="center", fontsize=FS_TICK - 2, style="italic", color=C_ACCENT)
    return fig


# -------------------------------------------------------
# Step 11: platonic representations as a tool
# -------------------------------------------------------
def build_step_11():
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle("Step 11: One rotation turns separate encoders into a captioner",
                 fontsize=FS_FIGURE_TITLE, fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[0.55, 1], hspace=0.45, wspace=0.3, top=0.88, bottom=0.14)

    axp = fig.add_subplot(gs[0, :])
    blank(axp, ylim=(0, 3))
    steps = [
        ("image", C_MUTE, "0.45"),
        ("frozen DINOv2 /\nDINOv3 encoder", C_NEW, C_DARK),
        ("ONE rotation Q\n(only part fit on pairs)", C_WARN, "#b7950b"),
        ("caption decoder,\ntrained on TEXT only", C_OLD, C_ACCENT),
        ('"A surfer riding a\nwave on a surfboard."', C_GOAL, C_OK),
    ]
    for i, (txt, face, edge) in enumerate(steps):
        x = 0.1 + i * 2.4
        round_box(axp, x, 0.8, 2.0, 1.6, face, edge, lw=1.8)
        axp.text(x + 1.0, 1.6, txt, ha="center", va="center", fontsize=FS_TICK - 2, fontweight="bold")
        if i < len(steps) - 1:
            arrow(axp, x + 2.02, 1.6, x + 2.38, 1.6)
    axp.text(6.0, 0.25,
             "Decoder learns to rebuild each caption from its noisy good-view JEPA embedding. "
             "COCO CIDEr .838 (Flickr .519), MSR-VTT video .246 from mean of frames.",
             ha="center", fontsize=FS_TICK - 3, color="0.25")

    # Noise matched to the alignment error, not tuned
    ax = fig.add_subplot(gs[1, 0])
    sig = np.linspace(0, 8, 300)
    ax.plot(sig, 1 / np.sqrt(1 + sig ** 2), color=C_DARK, lw=3,
            label=r"cos(e, e + noise) $=(1+\sigma^2)^{-1/2}$")
    c = 0.30
    s_star = np.sqrt(c ** -2 - 1)
    ax.axhline(c, color=C_ACCENT, ls="--", lw=1.5, label="image-caption cosine after rotation, c = .3")
    ax.axvline(s_star, color="0.3", ls=":", lw=1.5)
    ax.text(s_star + 0.15, 0.7, rf"$\sigma=(c^{{-2}}-1)^{{1/2}}\approx{s_star:.1f}$", fontsize=FS_TICK)
    ax.set_xlabel(r"decoder training noise $\sigma$ (x embedding norm)", fontsize=FS_AXIS - 2)
    ax.set_ylabel("expected cosine", fontsize=FS_AXIS - 2)
    ax.legend(fontsize=FS_LEGEND - 4)
    ax.grid(alpha=0.25)
    ax.set_title("Noise is predicted, not tuned (transfer peaks at sigma = 3)", fontsize=FS_AX_TITLE - 4)

    # Pair-free alignment, Table 4
    ax = fig.add_subplot(gs[1, 1])
    rows = [
        ("good-view JEPA 109M", [67.2, 64.4], C_GOODV),
        ("good-view JEPA 33M", [60.9, 57.9], "#82e0aa"),
        ("Qwen3-8B (rerun)", [61.7, 47.6], C_LLM),
        ("prior best, any setting", [62, 56], "#aab7b8"),
        ("CLIP text (400M pairs)", [67.0, 68.7], "#5d6d7e"),
    ]
    x = np.arange(2)
    w = 0.16
    for i, (name, vals, col) in enumerate(rows):
        ax.bar(x + (i - 2) * w, vals, w, color=col, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["CIFAR-100 top-5", "ImageNet-100 top-5"], fontsize=FS_TICK - 2)
    ax.set_ylim(0, 85)
    ax.set_ylabel("zero-shot accuracy (%)", fontsize=FS_AXIS - 2)
    ax.legend(fontsize=FS_LEGEND - 5, ncol=2, loc="upper left")
    ax.set_title("ZERO pairs: rotation found from geometry alone\n(Schnaus et al. benchmark, Table 4)",
                 fontsize=FS_AX_TITLE - 4)
    return fig


# -------------------------------------------------------
# Step 12: limitations and takeaways
# -------------------------------------------------------
def build_step_12():
    fig, ax = new_figure()
    fig.suptitle("Step 12: What to take away, and what is not proven",
                 fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank(ax)

    round_box(ax, 0.3, 1.4, 5.6, 6.1, C_GOAL, C_OK, lw=1.8)
    ax.text(3.1, 7.05, "Takeaways", ha="center", fontsize=FS_ANNOT + 2, fontweight="bold", color="#145a32")
    ax.text(0.6, 4.25,
            "1. The PRH is a statement about POSITIVE VIEWS:\n"
            "    shared positives -> same embedding up to rotation.\n\n"
            "2. Good views (two observations of one scene) learn\n"
            "    the world; bad views (corruptions of one\n"
            "    observation) also learn the modality's nuisance.\n\n"
            "3. Next-token prediction is a bad-view JEPA: it gets\n"
            "    there only through scale.\n\n"
            "4. Recipe: views from different observations of one\n"
            "    state, trimmed/cropped, lightly masked, with\n"
            "    gradient clipping.",
            ha="left", va="center", fontsize=FS_TICK - 1, linespacing=1.35)

    round_box(ax, 6.1, 1.4, 5.6, 6.1, C_OLD, C_ACCENT, lw=1.8)
    ax.text(8.9, 7.05, "Known limitations", ha="center", fontsize=FS_ANNOT + 2, fontweight="bold",
            color=C_ACCENT)
    ax.text(6.4, 4.25,
            "- Population optima over a FINITE world with\n"
            "  injective renderings; real modalities are lossy.\n\n"
            "- Uniqueness up to rotation is proved for 2nd-order\n"
            "  anti-collapse, assumed for LeJEPA's SIGReg.\n\n"
            "- No experiment meets Theorem 1's hypotheses exactly;\n"
            "  Prop. 4's link to Prop. 3 is heuristic.\n\n"
            "- Good views need grouping: human captions per image,\n"
            "  or LLaVA recaptions (CLIP vision inside). Without a\n"
            "  captioner: WIT .060 instead of .107.",
            ha="left", va="center", fontsize=FS_TICK - 1, linespacing=1.35)

    round_box(ax, 0.3, 0.1, 11.4, 1.05, C_DARK, C_DARK, lw=0)
    ax.text(6.0, 0.62,
            "\"Positive views, and not scale alone, are the lever to build platonic representations\"",
            ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_WARN)
    return fig


# -------------------------------------------------------
# Main deck
# -------------------------------------------------------
if __name__ == "__main__":
    builders = [
        build_preface,
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
        build_step_12,
    ]
    run_slide_deck(builders)
