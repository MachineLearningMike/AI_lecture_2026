"""
test.py

A concise note on:
  (1) the likelihood-threshold / posterior-majority equivalence and its
      finite and continuous extensions;
  (2) a teaching view of Frequentist vs Bayesian inference
      (fixed world + random data  vs  evolving belief about possible worlds).

Sketch:
  1. Title
  2. Core equivalence
  3. Proof chain
  4. Meaning: posterior below one half
  5. Many hypotheses
  6. Against a particular rival
  7. Continuous hypothesis regions
  8. World vs belief
  9. Two questions
  10. Physical randomness vs epistemic uncertainty
  11. Literature caution
  12. Takeaway
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_ANNOT,
    FS_TICK,
    FIGURE_DPI,
    apply_rcparams,
    screen_figsize,
    run_slide_deck,
)
apply_rcparams(plt)


def new_figure(*args, **kwargs):
    figsize = kwargs.pop("figsize", screen_figsize())
    dpi = kwargs.pop("dpi", FIGURE_DPI)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    if args or kwargs:
        return fig, fig.add_subplot(*args, **kwargs)
    return fig, fig.add_subplot()


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


def section_banner(ax, text, y=6.45, h=1.15):
    round_box(ax, 0.35, y, 11.3, h, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, y + h / 2, text,
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_LOOP = "#2874a6"
C_OK = "#1e8449"
C_GOAL = "#d4edda"
C_MUTE = "#f4f6f7"
C_OLD = "#fce4d6"
C_OLD_EDGE = "#d35400"
C_ACCENT = "#c0392b"
C_WARN = "#f8d7da"
C_FUN = "#8e44ad"
C_FUN_FACE = "#f5eef8"


# -------------------------------------------------------
# Step 1: Title
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: A threshold that means posterior majority against H",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 4.6, 11.0, 2.6, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 5.9,
        "Same experimental scaffold.\n"
        "A simple inequality on "
        r"$P(D\mid H)$"
        " that is exactly equivalent to\n"
        r"$H$ being less probable than $\bar H$ after seeing $D$.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.5, 1.0, 5.3, 3.1, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 3.5, "Math half", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 2.2,
        "Binary case, then n rivals,\n"
        "then continuous regions.\n"
        "All the same algebra.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.2, 1.0, 5.3, 3.1, C_GOAL, C_OK, lw=2.0)
    ax.text(8.85, 3.5, "Teaching half", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 2.2,
        "Frequentist: fixed world,\nrandom data across repeats.\n"
        "Bayesian: updating belief\nabout possible worlds.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: Core equivalence
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: The claim -- an equivalence, not a one-way hint",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        r"Assume $P(H)>0$ and $P(D)>0$. Let $\bar H$ be the complement of $H$.",
    )

    round_box(ax, 0.5, 2.6, 11.0, 3.4, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 4.3,
        r"$P(D\mid H)<\dfrac{P(D)}{2P(H)}"
        r"\;\Longleftrightarrow\;"
        r"P(H\mid D)<P(\bar H\mid D)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 1, color="#f9e79f",
    )

    round_box(ax, 0.5, 0.5, 11.0, 1.7, C_MUTE, "0.55", lw=1.5)
    ax.text(
        6.0, 1.35,
        "Left side looks like a bound on how surprising D is under H.\n"
        "Right side is a posterior majority decision against H.\n"
        "They are the same statement, rewritten.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 3: Proof chain
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: Core chain -- four rewrites",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(ax, "Every step is reversible. That is why we get both directions.")

    rows = [
        (4.55, C_NEW, C_LOOP,
         r"$P(D\mid H)<\dfrac{P(D)}{2P(H)}$",
         r"$\times\,P(H)>0$"),
        (3.35, C_MUTE, "0.55",
         r"$P(D\cap H)<\dfrac{P(D)}{2}$",
         r"split $D$ by $\{H,\bar H\}$"),
        (2.15, C_OLD, C_OLD_EDGE,
         r"$P(D\cap H)<P(D\cap\bar H)$",
         r"$\div\,P(D)>0$"),
        (0.95, C_GOAL, C_OK,
         r"$P(H\mid D)<P(\bar H\mid D)$",
         r"$H$ loses to $\bar H$"),
    ]
    for i, (y, face, edge, left, right) in enumerate(rows):
        round_box(ax, 0.4, y, 7.5, 1.0, face, edge, lw=1.7)
        ax.text(4.15, y + 0.5, left, ha="center", va="center",
                fontsize=FS_ANNOT, color="#333333")
        round_box(ax, 8.1, y, 3.5, 1.0, C_MUTE, "0.65", lw=1.2)
        ax.text(9.85, y + 0.5, right, ha="center", va="center",
                fontsize=FS_TICK, color="0.35")
        if i < len(rows) - 1:
            ax.text(4.15, y - 0.05, "\u21d5", ha="center", va="top",
                    fontsize=FS_ANNOT + 2, color=C_LOOP, fontweight="bold")
    return fig


# -------------------------------------------------------
# Step 4: Meaning -- half
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 4: The number 2 means posterior probability below $1/2$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        r"$P(\bar H\mid D)=1-P(H\mid D)$, so losing to the complement is $P(H\mid D)<1/2$.",
    )

    round_box(ax, 1.2, 2.4, 9.6, 3.5, C_GOAL, C_OK, lw=2.2)
    ax.text(
        6.0, 4.15,
        r"$P(D\mid H)<\dfrac{P(D)}{2P(H)}"
        r"\;\Longleftrightarrow\;"
        r"P(H\mid D)<\dfrac{1}{2}$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2, color="#145a32",
    )

    round_box(ax, 0.4, 0.45, 11.2, 1.55, C_MUTE, "0.55", lw=1.5)
    ax.text(
        6.0, 1.225,
        "You are not inventing a mysterious constant.\n"
        "The 2 is exactly the threshold for a posterior majority against H.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 5: Many hypotheses
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: Many mutually exclusive, exhaustive hypotheses",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        r"$H_1,\ldots,H_n$ partition $\Omega$. For one $H_i$, set $\bar H_i=\bigcup_{j\ne i}H_j$.",
    )

    round_box(ax, 0.4, 3.2, 11.2, 2.85, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 4.6,
        r"$P(D\mid H_i)<\dfrac{P(D)}{2P(H_i)}"
        r"\;\Longleftrightarrow\;"
        r"P(H_i\mid D)<\sum_{j\ne i}P(H_j\mid D)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.45, 11.2, 2.4, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.65,
        "Same binary theorem, applied to "
        r"$(H_i,\bar H_i)$" + ".\n"
        "Equivalently: "
        r"$P(H_i\mid D)<1/2$" + ".\n"
        "The result was never truly limited to two named hypotheses.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#145a32",
    )
    return fig


# -------------------------------------------------------
# Step 6: Particular rival
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Stronger rule -- lose to a particular rival $H_k$",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        r"Bayesian model selection often asks: is $H_i$ beaten by some specific $H_k$?",
    )

    round_box(ax, 0.4, 3.35, 11.2, 2.7, C_FUN_FACE, C_FUN, lw=2.0)
    ax.text(
        6.0, 4.7,
        r"$P(H_i\mid D)<P(H_k\mid D)"
        r"\;\Longleftrightarrow\;"
        r"P(D\mid H_i)<\dfrac{P(H_k)}{P(H_i)}\,P(D\mid H_k)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
    )

    pair = [
        (0.4, C_NEW, C_LOOP, "Vs the rest",
         r"$P(H_i\mid D)<1/2$" + "\n"
         "threshold uses "
         r"$P(D)/(2P(H_i))$"),
        (6.1, C_OLD, C_OLD_EDGE, "Vs one rival",
         r"$P(H_i\mid D)<P(H_k\mid D)$" + "\n"
         "threshold uses prior odds "
         r"$\times\,P(D\mid H_k)$"),
    ]
    for x, face, edge, title, body in pair:
        round_box(ax, x, 0.4, 5.5, 2.55, face, edge, lw=1.8)
        ax.text(x + 2.75, 2.45, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=edge)
        ax.text(x + 2.75, 1.35, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")
    return fig


# -------------------------------------------------------
# Step 7: Continuous case
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: Continuous case -- use a hypothesis region, not a point",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        r"A single $\theta$ often has prior probability 0. Let $H\subset\Theta$ with $P(H)>0$.",
    )

    round_box(ax, 0.4, 3.4, 11.2, 2.65, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 4.7,
        r"$P(H)=\int_H\pi(\theta)\,d\theta,\quad"
        r"P(D\cap H)=\int_H P(D\mid\theta)\pi(\theta)\,d\theta$"
        + "\n\n"
        r"$P(D\mid H)=\dfrac{\int_H P(D\mid\theta)\pi(\theta)\,d\theta}"
        r"{\int_H\pi(\theta)\,d\theta}$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.45, 11.2, 2.55, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.725,
        "Then the same identity holds for the region H:\n"
        r"$P(D\mid H)<P(D)/(2P(H))"
        r"\;\Longleftrightarrow\;"
        r"P(H\mid D)<P(\bar H\mid D)$" + ".\n"
        "Continuous parameters do not need a new theorem -- only positive-mass sets.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#145a32",
    )
    return fig


# -------------------------------------------------------
# Step 8: World vs belief
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: What is the object of study -- the world, or belief about it?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 4.15, 11.2, 3.2, C_OLD, C_OLD_EDGE, lw=2.0)
    ax.text(6.0, 6.7, "Traditional scientific picture", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_OLD_EDGE)
    ax.text(
        6.0, 5.35,
        "An objective world has fixed but unknown properties.\n"
        "It sends us noisy signals. We try to recover how that world behaves.\n\n"
        r"objective world $\longrightarrow$ noisy data $\longrightarrow$ inference",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.45, 11.2, 3.3, C_FUN_FACE, C_FUN, lw=2.0)
    ax.text(6.0, 3.1, "Bayesian emphasis", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_FUN)
    ax.text(
        6.0, 1.7,
        "Given what we currently know, what should we believe\n"
        "about the possible states of that world?\n\n"
        r"prior belief $+$ data $\longrightarrow$ updated belief "
        r"$P(\theta)\;\longrightarrow\;P(\theta\mid D)$",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 9: Two questions
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Two different questions",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 2.55, 5.5, 4.7, C_NEW, C_LOOP, lw=2.2)
    ax.text(3.15, 6.7, "Frequentist", ha="center",
            fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 4.5,
        "If the world were fixed\n"
        "in this way, how would\n"
        "data behave across\n"
        "hypothetical repetitions?\n\n"
        "Parameter: fixed unknown.\n"
        "Randomness: mainly in data\n"
        "and in the procedure.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.1, 2.55, 5.5, 4.7, C_GOAL, C_OK, lw=2.2)
    ax.text(8.85, 6.7, "Bayesian", ha="center",
            fontsize=FS_AX_TITLE - 3, fontweight="bold", color=C_OK)
    ax.text(
        8.85, 4.5,
        "Given the data I have,\n"
        "how should my beliefs\n"
        "about possible worlds\n"
        "change?\n\n"
        "Put probability on hypotheses.\n"
        "Update when evidence arrives.\n"
        "Act through that belief.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.4, 11.2, 1.8, C_MUTE, "0.55", lw=1.5)
    ax.text(
        6.0, 1.3,
        "Bayesian work need not deny an objective world.\n"
        "It makes the layer of uncertainty about that world explicit.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 10: Epistemic vs physical
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: \"Random parameter\" often means uncertain, not fluctuating",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        "A mass may have one physical value. A prior over that mass describes our ignorance.",
    )

    pair = [
        (0.5, C_WARN, C_ACCENT, "Physical randomness",
         "The quantity itself varies\n"
         "randomly in the world\n"
         "(noise, sampling, chaos)."),
        (6.2, C_FUN_FACE, C_FUN, "Epistemic uncertainty",
         "The quantity may be fixed,\n"
         "but we do not know it.\n"
         "Probability tracks belief."),
    ]
    for x, face, edge, title, body in pair:
        round_box(ax, x, 2.5, 5.3, 3.4, face, edge, lw=2.0)
        ax.text(x + 2.65, 5.3, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=edge)
        ax.text(x + 2.65, 3.7, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.4, 11.2, 1.75, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 1.275,
        "We cannot act on an unknown reality directly.\n"
        "We act through our current model of it -- and update that model with data.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Step 11: Literature caution
# -------------------------------------------------------
def build_step_11():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 11: Related literature -- close, but not the same slogan",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    section_banner(
        ax,
        "Do not claim absolute priority. Do claim a clean, under-emphasized identity.",
    )

    # ~1 cm taller upward; left -2 cm width, right +2 cm width (~0.52 / 1.04 axes units).
    y_mid, h_mid = 2.30, 3.52
    w_left, w_right, gap = 4.46, 6.54, 0.20
    x_left, x_right = 0.4, 0.4 + w_left + gap
    cards = [
        (x_left, w_left, C_NEW, C_LOOP, "Already known broadly",
         "p-values linked to posteriors;\n"
         "Bayesian and Frequentist\n"
         "rejection can coincide under\n"
         "special setups (e.g. Casella\n"
         "and Berger; later p / posterior\n"
         "equivalence papers)."),
        (x_right, w_right, C_GOAL, C_OK, "This formulation",
         r"$P(D\mid H)<\dfrac{P(D)}{2P(H)}"
         r"\;\Longleftrightarrow\;"
         r"P(H\mid D)<P(\bar H\mid D)$"
         "\n"
         "No need of the mysterious "
         r"$p$"
         " in the p-value technique."
         "\n"
         "The reason to reject "
         r"$H$"
         " is clear and intuitive: attend more to "
         r"$\bar H$"
         " than to "
         r"$H$"
         ","
         "\n"
         "and "
         r"$D$"
         " should be neither too large nor too small relative to "
         r"$H$"
         "."),
    ]
    for x, w, face, edge, title, body in cards:
        round_box(ax, x, y_mid, w, h_mid, face, edge, lw=2.0)
        ax.text(x + w / 2, y_mid + h_mid - 0.45, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=edge)
        ax.text(x + w / 2, y_mid + 1.35, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")

    round_box(ax, 0.4, 0.15, 11.2, 1.95, C_WARN, C_ACCENT, lw=2.0)
    ax.text(
        6.0, 1.125,
        "Open question for teaching and research:\n"
        "can the left-hand threshold be used as a Frequentist decision rule\n"
        "without already treating H as an event with prior probability?\n"
        "The algebra is fine either way; the philosophy is the delicate part.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 12: Takeaway
# -------------------------------------------------------
def build_step_12():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 12: Takeaway",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    cards = [
        (0.4, 4.2, 3.7, 3.15, C_NEW, C_LOOP, "Identity",
         r"$P(D\mid H)<\dfrac{P(D)}{2P(H)}$"
         "\n"
         r"$\Longleftrightarrow$"
         "\n"
         r"$P(H\mid D)<P(\bar H\mid D)$"),
        (4.15, 4.2, 3.7, 3.15, C_FUN_FACE, C_FUN, "Extensions",
         "Works for many discrete\nhypotheses, for a chosen\nrival, and for positive-\nmass regions in "
         r"$\Theta$."),
        (7.9, 4.2, 3.7, 3.15, C_GOAL, C_OK, "Perspective",
         "Frequentist: repeated\nbehavior under a fixed\nworld.\n"
         "Bayesian: changing belief\nabout possible worlds."),
    ]
    for x, y, w, h, face, edge, title, body in cards:
        round_box(ax, x, y, w, h, face, edge, lw=2.0)
        ax.text(x + w / 2, y + h - 0.5, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=edge)
        ax.text(x + w / 2, y + 1.25, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")

    round_box(ax, 0.4, 0.4, 11.2, 3.4, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 2.1,
        "Frequentist inference studies the repeated behavior of evidence under a fixed world.\n"
        "Bayesian inference studies the changing state of belief about possible worlds\n"
        "in light of evidence.\n\n"
        "We may never escape the objective world, but we can only reason and act\n"
        "through our beliefs about it. Bayesian statistics makes that layer explicit.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#f9e79f",
    )
    return fig


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
    build_step_12,
])
