"""
1_20_naive_bayesian_classifier.py

Naive Bayes classifier -- Bayes' theorem with a deliberately strong assumption:
given the class, features are conditionally independent.

Sketch:
  1. What it is -- the "naive" idea
  2. Start from Bayes' theorem
  3. Drop the denominator for classification
  4. The independence assumption
  5. The decision rule
  6. Tiny spam example (table + arithmetic)
  7. Why it still works when independence is false
  8. Flavors: Gaussian / Multinomial / Bernoulli
  9. Generative classifier vs discriminative boundary
  10. Tiny code sketch
  11. Takeaway

product-of-randoms. Ties to discriminative vs generative.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from _config import (
    FS_FIGURE_TITLE,
    FS_AX_TITLE,
    FS_AXIS,
    FS_TICK,
    FS_ANNOT,
    FS_LEGEND,
    FIGURE_DPI,
    apply_rcparams,
    screen_figsize,
    run_slide_deck,
)
apply_rcparams(plt)
import numpy as np

np.random.seed(65)


def new_figure(*args, **kwargs):
    kwargs.setdefault("figsize", screen_figsize())
    kwargs.setdefault("dpi", FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


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


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=2.3):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=15),
    )


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_WARN = "#f8d7da"
C_FUN = "#8e44ad"
C_OLD = "#fce4d6"
C_CODE = "#fbfcf8"
C_LOOP = "#2874a6"


# -------------------------------------------------------
# Step 1
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: What is a Naive Bayes classifier?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        "A simple probabilistic classifier based on Bayes' theorem,\n"
        "with one deliberately strong assumption.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.5, 2.6, 11.0, 2.5, C_WARN, C_ACCENT, lw=2.4)
    ax.text(
        6.0, 4.35,
        "The naive assumption",
        ha="center", fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_ACCENT,
    )
    ax.text(
        6.0, 3.35,
        "Given the class, all input features are conditionally independent.",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#641e16",
    )

    round_box(ax, 0.5, 0.45, 11.0, 1.8, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.35,
        "That independence assumption is usually false --\n"
        "and yet the classifier is often surprisingly useful.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: Bayes setup
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: Start with Bayes' theorem (spam email)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.7, 5.3, 1.8, C_NEW, C_LOOP, lw=2.0)
    ax.text(3.15, 7.0, "Classes", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax.text(
        3.15, 6.25,
        r"$C=\mathrm{spam}$"
        "\n"
        r"$C=\mathrm{not\ spam}$",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax, 6.2, 5.7, 5.3, 1.8, C_OLD, "#e67e22", lw=2.0)
    ax.text(8.85, 7.0, "Features (example)", ha="center",
            fontsize=FS_ANNOT, fontweight="bold", color="#a04000")
    ax.text(
        8.85, 6.25,
        r"$x_1$: contains \"free\""
        "\n"
        r"$x_2$: contains \"money\""
        "\n"
        r"$x_3$: contains \"offer\"",
        ha="center", va="center", fontsize=FS_TICK + 1,
    )

    round_box(ax, 0.5, 2.4, 11.0, 2.9, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 4.5,
        "Bayes says",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 3.4,
        r"$P(C\mid x_1,x_2,x_3)=\dfrac{P(x_1,x_2,x_3\mid C)\,P(C)}"
        r"{P(x_1,x_2,x_3)}$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#1c2833",
    )

    round_box(ax, 0.5, 0.4, 11.0, 1.6, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.2,
        "Posterior  ~  how the class would generate these words  x  prior of the class.",
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 3: drop denominator
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: For classification, drop the shared denominator",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.2, 11.0, 2.3, C_NEW, C_LOOP, lw=2.2)
    ax.text(
        6.0, 6.7,
        "The evidence term is the same for every class",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 5.85,
        r"$P(x_1,x_2,x_3)$ does not depend on $C$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#333333",
    )

    round_box(ax, 1.5, 2.5, 9.0, 2.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 3.6,
        r"$P(C\mid x)\;\propto\;P(x\mid C)\,P(C)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.5, 0.45, 11.0, 1.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.3,
        "So we only need a class prior and a model of features given the class.\n"
        "Ranking classes is enough -- absolute probabilities can wait.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 4: independence
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: The naive assumption -- factor the likelihood",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_WARN, C_ACCENT, lw=2.0)
    ax.text(
        6.0, 6.85,
        "Hard part",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT,
    )
    ax.text(
        6.0, 6.15,
        r"Estimating the joint  $P(x_1,x_2,x_3\mid C)$  in high dimensions",
        ha="center", va="center", fontsize=FS_ANNOT, color="#641e16",
    )

    round_box(ax, 0.5, 2.7, 11.0, 2.4, C_NEW, C_LOOP, lw=2.2)
    ax.text(
        6.0, 4.35,
        "Naive Bayes assumes conditional independence",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 3.45,
        r"$P(x_1,x_2,x_3\mid C)=P(x_1\mid C)\,P(x_2\mid C)\,P(x_3\mid C)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1, color="#1c2833",
    )

    round_box(ax, 0.5, 0.4, 11.0, 1.9, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.35,
        r"Therefore  $P(C\mid x)\;\propto\;P(C)\prod_i P(x_i\mid C)$",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 5: decision rule
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: The decision rule",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 1.0, 4.4, 10.0, 3.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        "Pick the class with the largest score",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color="#f9e79f",
    )
    ax.text(
        6.0, 5.4,
        r"$\hat C=\arg\max_C\; P(C)\prod_i P(x_i\mid C)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color="white",
    )

    trio = [
        (0.5, 1.0, "Prior", r"$P(C)$" + "\nhow common\nis the class?"),
        (4.25, 1.0, "Likelihood factors", r"$P(x_i\mid C)$" + "\nhow each feature\nlooks under C"),
        (8.0, 1.0, "Product score", "multiply,\nthen compare\nacross classes"),
    ]
    for x, y, title, body in trio:
        round_box(ax, x, y, 3.5, 2.9, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.75, y + 2.25, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.75, y + 1.1, body, ha="center", va="center",
                fontsize=FS_TICK + 1)
    return fig


# -------------------------------------------------------
# Step 6: tiny example
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Tiny spam example",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    # Table header
    round_box(ax, 0.4, 6.5, 3.6, 1.0, C_DARK, C_DARK, lw=0)
    round_box(ax, 4.1, 6.5, 3.7, 1.0, C_DARK, C_DARK, lw=0)
    round_box(ax, 7.9, 6.5, 3.7, 1.0, C_DARK, C_DARK, lw=0)
    ax.text(2.2, 7.0, "Feature", ha="center", va="center",
            fontsize=FS_TICK, fontweight="bold", color="white")
    ax.text(5.95, 7.0, "Spam", ha="center", va="center",
            fontsize=FS_TICK, fontweight="bold", color="white")
    ax.text(9.75, 7.0, "Not spam", ha="center", va="center",
            fontsize=FS_TICK, fontweight="bold", color="white")

    rows = [
        ('P("free"|C)', "0.8", "0.1"),
        ('P("money"|C)', "0.7", "0.2"),
        ('P("offer"|C)', "0.6", "0.1"),
        ("Prior P(C)", "0.3", "0.7"),
    ]
    for i, (feat, spam, ham) in enumerate(rows):
        y = 5.35 - i * 0.85
        face = C_NEW if i % 2 == 0 else C_MUTE
        round_box(ax, 0.4, y, 3.6, 0.75, face, "0.55", lw=1.2)
        round_box(ax, 4.1, y, 3.7, 0.75, face, "0.55", lw=1.2)
        round_box(ax, 7.9, y, 3.7, 0.75, face, "0.55", lw=1.2)
        ax.text(2.2, y + 0.35, feat, ha="center", va="center", fontsize=FS_TICK)
        ax.text(5.95, y + 0.35, spam, ha="center", va="center",
                fontsize=FS_TICK, fontweight="bold")
        ax.text(9.75, y + 0.35, ham, ha="center", va="center",
                fontsize=FS_TICK, fontweight="bold")

    round_box(ax, 0.4, 0.35, 5.5, 1.55, C_WARN, C_ACCENT, lw=2.0)
    ax.text(
        3.15, 1.1,
        "Spam score\n0.3×0.8×0.7×0.6 = 0.1008",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
        color="#641e16",
    )

    round_box(ax, 6.1, 0.35, 5.5, 1.55, C_GOAL, C_OK, lw=2.0)
    ax.text(
        8.85, 1.1,
        "Not-spam score\n0.7×0.1×0.2×0.1 = 0.0014",
        ha="center", va="center", fontsize=FS_TICK + 1, fontweight="bold",
        color=C_OK,
    )
    return fig


# -------------------------------------------------------
# Step 7: decision from example + why useful
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: Decision -- and why Naive Bayes still helps",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        "0.1008  ≫  0.0014   →   classify as spam",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.5, 2.7, 11.0, 2.4, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 4.35,
        "Independence is usually false",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.0, 3.4,
        '"free" and "offer" are related in spam.\n'
        "Yet relative class scores can still rank the right label first.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    uses = [
        (0.5, 0.4, "spam\ndetection"),
        (2.8, 0.4, "document\nclassification"),
        (5.1, 0.4, "sentiment\n/ text tags"),
        (7.4, 0.4, "simple medical\nscreening"),
        (9.7, 0.4, "sparse high-D\nfeatures"),
    ]
    for x, y, lab in uses:
        round_box(ax, x, y, 2.1, 1.9, C_NEW, C_LOOP, lw=1.6)
        ax.text(x + 1.05, y + 0.95, lab, ha="center", va="center",
                fontsize=FS_TICK, fontweight="bold", color=C_DARK)
    return fig


# -------------------------------------------------------
# Step 8: flavors
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        r"Step 8: Same idea -- different $P(x_i\mid C)$ models",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    cards = [
        (0.4, 3.2, "Gaussian NB",
         r"$x_i\mid C\sim\mathcal{N}(\mu_{i,C},\sigma_{i,C}^2)$"
         "\n\n"
         "Continuous measurements\n(heights, sensors, scores)."),
        (4.2, 3.2, "Multinomial NB",
         r"$P(\mathrm{word}_i\mid C)$"
         "\n\n"
         "Word counts / bag-of-words.\nClassic for documents."),
        (8.0, 3.2, "Bernoulli NB",
         r"$x_i\in\{0,1\}$"
         "\n\n"
         "Presence / absence of\na word or binary flag."),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 4.2, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 3.5, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 1.7, body, ha="center", va="center",
                fontsize=FS_TICK + 1)

    round_box(ax, 0.4, 0.4, 11.2, 2.4, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.6,
        "The skeleton stays the same:\n"
        r"prior × product of feature likelihoods → pick the best class.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 9: generative vs discriminative
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Naive Bayes is a generative classifier",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.5, 4.5, 5.3, 3.0, C_GOAL, C_OK, lw=2.2)
    ax.text(3.15, 6.9, "Generative route", ha="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
    ax.text(
        3.15, 5.5,
        r"Model $P(x\mid C)P(C)$"
        "\n\n"
        "Ask: which class would\n"
        "most likely generate x?\n\n"
        "Naive Bayes lives here.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax, 6.2, 4.5, 5.3, 3.0, C_OLD, "#e67e22", lw=2.2)
    ax.text(8.85, 6.9, "Discriminative route", ha="center",
            fontsize=FS_AX_TITLE - 2, fontweight="bold", color="#a04000")
    ax.text(
        8.85, 5.5,
        r"Model $P(C\mid x)$ directly"
        "\n\n"
        "Learn a boundary / score\n"
        "between classes.\n\n"
        "Logistic regression, etc.",
        ha="center", va="center", fontsize=FS_ANNOT,
    )

    round_box(ax, 0.5, 0.45, 11.0, 3.6, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 2.9,
        "Naive Bayes does not draw the boundary first.\n"
        "It models how each class would produce the features,\n"
        "then asks which story fits the observation better.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    ax.text(
        6.0, 1.2,
        "Clean bridge to discriminative vs generative (later chapter).",
        ha="center", va="center", fontsize=FS_TICK + 1,
        fontweight="bold", color=C_DARK,
    )
    return fig


# -------------------------------------------------------
# Step 10: code sketch
# -------------------------------------------------------
def build_step_10():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 10: Tiny code-shaped sketch (Bernoulli spam)",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Score each class", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.35, 0.6, 9.3, 10.8, C_CODE, "0.55", lw=1.5)
    ax.text(
        5.0, 6.2,
        "def nb_score(prior, likelihoods, x):\n"
        "    # prior: P(C)\n"
        "    # likelihoods[i] = P(x_i=1 | C)\n"
        "    # x[i] in {0,1}\n"
        "    s = prior\n"
        "    for xi, p1 in zip(x, likelihoods):\n"
        "        s *= p1 if xi == 1 else (1 - p1)\n"
        "    return s\n"
        "\n"
        "spam = nb_score(0.3, [0.8, 0.7, 0.6], [1,1,1])\n"
        "ham  = nb_score(0.7, [0.1, 0.2, 0.1], [1,1,1])\n"
        "label = 'spam' if spam > ham else 'ham'",
        ha="center", va="center", fontsize=FS_TICK - 1,
        family="monospace", color="#1c2833",
    )

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Practical notes", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    tips = [
        (0.5, 9.2, "Use logs to avoid underflow:\nsum log p instead of product."),
        (0.5, 6.6, "Add Laplace smoothing so\nunseen words don't zero a class."),
        (0.5, 4.0, "You need relative ranking,\nnot calibrated probabilities."),
        (0.5, 1.4, "Strong baseline for text;\nreplace later if needed."),
    ]
    for x, y, body in tips:
        round_box(ax, x, y, 9.0, 2.1, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 4.5, y + 1.05, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color=C_DARK)

    fig.tight_layout(rect=[0, 0.12, 1, 0.90])
    return fig


# -------------------------------------------------------
# Step 11: takeaway
# -------------------------------------------------------
def build_step_11():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 11: Takeaway",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    points = [
        (0.5, 5.7, "1", "Bayes updates class belief from features."),
        (0.5, 4.3, "2", "Naive = factor the likelihood into a product."),
        (0.5, 2.9, "3", "Classify by argmax of prior × product of P(x_i|C)."),
        (0.5, 1.5, "4", "Generative: model how each class generates x."),
    ]
    for x, y, num, body in points:
        round_box(ax, x, y, 11.0, 1.2, C_NEW, C_LOOP, lw=1.8)
        ax.text(x + 0.7, y + 0.6, num, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_LOOP)
        ax.text(x + 6.0, y + 0.6, body, ha="center", va="center",
                fontsize=FS_ANNOT, color=C_DARK)

    round_box(ax, 0.5, 0.3, 11.0, 1.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 0.8,
        r"$\hat C=\arg\max_C P(C)\prod_i P(x_i\mid C)$"
        "   -- simple, generative, often enough.",
        ha="center", va="center", fontsize=FS_TICK + 1,
        fontweight="bold", color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Deck
# -------------------------------------------------------
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
print("NAIVE BAYES CLASSIFIER")
print("=" * 60)
print("""
Naive Bayes = Bayes + conditional independence of features given class.

  P(C|x) ~ P(C) * product_i P(x_i|C)
  hat C = argmax_C of that score

Independence is usually false; relative ranking can still work.
Flavors: Gaussian / Multinomial / Bernoulli likelihoods.
It is a generative classifier: model P(x|C)P(C), then classify.
""")
