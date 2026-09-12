"""
3_280_YOLO_training.py

Training algorithm of YOLO -- from YOLOv1 intuition to a modern pipeline.

  Detection = localize boxes + classify objects in one forward pass.
  Training = assign targets to predictions, then minimize a multi-part loss.

Sketch:
  1. What YOLO is training to do
  2. YOLOv1 picture -- S x S grid, B boxes, objectness, classes
  3. YOLOv1 targets and loss (sum-squares style)
  4. One training step (classic)
  5. What changed in modern YOLO
  6. Data pipeline for detectors (mosaic / augment)
  7. Modern multi-part loss (box + obj + cls)
  8. Label assignment (who supervises whom)
  9. Modern training algorithm (pseudocode)
  10. Takeaway

same pipeline pattern with a richer sample and a richer loss.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
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

np.random.seed(270)


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


def code_panel(ax, x, y, w, h, code: str, fontsize=None):
    fontsize = FS_TICK if fontsize is None else fontsize
    round_box(ax, x, y, w, h, C_CODE, C_LOOP, lw=2.0)
    ax.text(
        x + 0.35, y + h / 2, code.rstrip("\n"),
        ha="left", va="center",
        fontsize=fontsize,
        family="monospace",
        linespacing=1.35,
        color="#1c2833",
    )


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_WARN = "#f8d7da"
C_OLD = "#fce4d6"
C_LOOP = "#2874a6"
C_CODE = "#fbfcf8"
C_PURPLE = "#8e44ad"


# -------------------------------------------------------
# Step 1: Goal of YOLO training
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: What YOLO training optimizes",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.15, 11.2, 1.55, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.9,
        "Given an image + ground-truth boxes/labels,\n"
        "learn a net that predicts boxes and classes in one shot.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.4, 3.2, "Input", "image\n(+ optional\naugment)"),
        (3.3, 3.2, "Model", "backbone\n+ neck\n+ detect head"),
        (6.2, 3.2, "Targets", "which cell / anchor\nowns each GT box\n+ class id"),
        (9.1, 3.2, "Loss", "box + objectness\n+ classification"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 2.7, 2.7, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.35, y + 2.15, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.35, y + 1.05, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.4, 11.2, 2.5, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.65,
        "Training algorithm =  data pipeline  +  target assignment  +  multi-task loss.\n"
        "Same outer loop as any PyTorch trainer; the detector-specific part is the loss.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: YOLOv1 grid picture
# -------------------------------------------------------
def build_step_2():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        r"Step 2: YOLOv1 intuition -- divide the image into an $S\times S$ grid",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    ax = axes[0]
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.set_title(r"$S\times S$ cells (schematic)", fontsize=FS_AX_TITLE - 2,
                 fontweight="bold")
    S = 7
    for i in range(S + 1):
        ax.plot([0, S], [i, i], color="0.55", lw=1.0)
        ax.plot([i, i], [0, S], color="0.55", lw=1.0)
    # fake objects
    ax.add_patch(Rectangle((1.2, 3.5), 2.2, 1.8, fill=False, edgecolor=C_OK, lw=2.5))
    ax.add_patch(Rectangle((4.0, 1.0), 1.8, 2.4, fill=False, edgecolor=C_ACCENT, lw=2.5))
    ax.plot(2.3, 4.4, "o", color=C_OK, ms=9)
    ax.plot(4.9, 2.2, "o", color=C_ACCENT, ms=9)
    ax.text(2.3, 5.6, "GT box\n(center cell owns it)", ha="center",
            fontsize=FS_TICK, color=C_OK, fontweight="bold")
    ax.text(4.9, 0.35, "another object", ha="center",
            fontsize=FS_TICK, color=C_ACCENT, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Per cell, the head predicts", fontsize=FS_AX_TITLE - 2,
                 fontweight="bold")
    items = [
        (8.2, r"$B$ boxes: $(x,y,w,h)$ relative to the cell"),
        (6.4, r"objectness / confidence for each box"),
        (4.6, r"class probabilities (shared per cell in v1)"),
        (2.8, r"tensor shape idea: $S\times S\times(B\cdot 5 + C)$"),
    ]
    for y, text in items:
        round_box(ax, 0.4, y - 0.7, 9.2, 1.5, C_NEW, C_LOOP, lw=1.8)
        ax.text(5.0, y + 0.05, text, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 3: YOLOv1 loss
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: YOLOv1 training loss -- multi-part sum of squares",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.15, 11.2, 1.55, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.9,
        r"$L = \lambda_{\mathrm{coord}} L_{\mathrm{box}}"
        r"+ L_{\mathrm{obj}} + \lambda_{\mathrm{noobj}} L_{\mathrm{noobj}}"
        r"+ L_{\mathrm{cls}}$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 3,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.4, 3.35, r"$L_{\mathrm{box}}$",
         "only responsible boxes\n"
         r"err on $(x,y,\sqrt{w},\sqrt{h})$"
         "\n"
         r"$\lambda_{\mathrm{coord}}$ upweights"),
        (4.2, 3.35, r"$L_{\mathrm{obj/noobj}}$",
         "confidence → IoU if object\n"
         "push toward 0 if empty\n"
         r"$\lambda_{\mathrm{noobj}}$ downweights"),
        (8.0, 3.35, r"$L_{\mathrm{cls}}$",
         "class scores only on\n"
         "cells that contain\nan object"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.55, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 2.05, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 0.95, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")

    round_box(ax, 0.4, 0.4, 11.2, 2.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.75,
        "Responsible predictor: among the B boxes in the GT's cell,\n"
        "the one with largest IoU to the GT gets the box + obj + cls gradients.\n"
        "Empty cells mainly train \"no object\" confidence.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 4: Classic training step
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: One classic YOLO training step",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    steps = [
        (6.35, "1. Load batch of images + GT boxes / labels"),
        (5.25, "2. Forward: predict S x S grid tensor"),
        (4.15, "3. Assign each GT to a cell (+ responsible box)"),
        (3.05, "4. Build targets for box / obj / noobj / cls"),
        (1.95, "5. Compute weighted multi-part loss"),
        (0.85, "6. backward(); optimizer.step()"),
    ]
    for y, text in steps:
        round_box(ax, 0.5, y - 0.4, 11.0, 0.95, C_NEW, C_LOOP, lw=1.7)
        ax.text(6.0, y + 0.07, text, ha="center", va="center",
                fontsize=FS_ANNOT, color="#333333")
    return fig


# -------------------------------------------------------
# Step 5: Modern changes
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: What changed from v1 to modern YOLO training",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    rows = [
        (6.2, "Multi-scale heads", "detect small/medium/large on different feature maps"),
        (5.0, "Better boxes", "anchors or anchor-free; IoU / CIoU / DFL box losses"),
        (3.8, "Richer assignment", "SimOTA / Task-Aligned / dynamic matching -- not only center cell"),
        (2.6, "Strong augments", "mosaic, mixup, letterbox, color jitter, copy-paste"),
        (1.4, "Training recipe", "AdamW / SGD + warmup + cosine / EMA weights"),
    ]
    for y, left, right in rows:
        round_box(ax, 0.4, y - 0.45, 11.2, 1.05, C_NEW, C_LOOP, lw=1.6)
        ax.text(2.5, y + 0.08, left, ha="center", va="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)
        ax.text(7.7, y + 0.08, right, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.2, 11.2, 0.85, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 0.62,
        "The outer algorithm is the same.  The assignment + loss + augments got smarter.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Step 6: Detector data pipeline
# -------------------------------------------------------
def build_step_6():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 6: Detector data pipeline -- still Dataset + DataLoader",
        fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("One sample", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.5, 5.2, 11.0, 2.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.3,
        "image tensor\n"
        "+ list of boxes (xyxy or xywh)\n"
        "+ class ids",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.5, 2.5, 11.0, 2.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.65,
        "Augment jointly:\n"
        "geometric transforms must move boxes too.\n"
        "Mosaic stitches 4 images → denser objects / epoch.",
        ha="center", va="center", fontsize=FS_TICK + 1, color=C_DARK,
    )
    round_box(ax, 0.5, 0.4, 11.0, 1.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.25,
        "Collate often keeps variable-length labels as lists,\n"
        "then the loss module builds dense targets.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    ax = axes[1]
    # mosaic schematic
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Mosaic (schematic)", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    colors = ["#d6eaf8", "#fce4d6", "#d5f5e3", "#f5b7b1"]
    coords = [(1, 5.2), (5.2, 5.2), (1, 1), (5.2, 1)]
    for (x, y), c in zip(coords, colors):
        ax.add_patch(Rectangle((x, y), 3.8, 3.8, facecolor=c, edgecolor=C_DARK, lw=2))
    ax.add_patch(Rectangle((2.2, 6.2), 1.4, 1.0, fill=False, edgecolor=C_OK, lw=2))
    ax.add_patch(Rectangle((6.0, 6.5), 1.2, 1.5, fill=False, edgecolor=C_ACCENT, lw=2))
    ax.add_patch(Rectangle((2.5, 2.0), 1.6, 1.2, fill=False, edgecolor=C_PURPLE, lw=2))
    ax.add_patch(Rectangle((6.3, 2.3), 1.3, 1.0, fill=False, edgecolor=C_LOOP, lw=2))
    ax.text(5.0, 0.35, "4 images → 1 training image", ha="center",
            fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK)

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 7: Modern loss
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: Modern YOLO loss -- still three jobs",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.2, 11.2, 1.5, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.95,
        r"$L = \lambda_{\mathrm{box}} L_{\mathrm{box}}"
        r"+ \lambda_{\mathrm{obj}} L_{\mathrm{obj}}"
        r"+ \lambda_{\mathrm{cls}} L_{\mathrm{cls}}$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 3,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.4, 3.2, "Box loss",
         "IoU / GIoU / CIoU / DFL\n"
         "how well predicted box\noverlaps the GT"),
        (4.2, 3.2, "Objectness",
         "BCE (or similar):\n"
         "is there an object\nat this location?"),
        (8.0, 3.2, "Classification",
         "BCE / focal on classes\n"
         "for positive matches\n(often multi-label)"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.7, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 2.15, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.8, y + 1.0, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.4, 11.2, 2.5, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.65,
        "v1 used sum-of-squares on coordinates and confidence.\n"
        "Modern YOLO replaces that with IoU-aware box losses and better positives.\n"
        "Same three questions: where?  is anything there?  what class?",
        ha="center", va="center", fontsize=FS_ANNOT, color="#1e4419",
    )
    return fig


# -------------------------------------------------------
# Step 8: Label assignment
# -------------------------------------------------------
def build_step_8():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 8: Label assignment -- who gets the gradient?",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.2, 11.2, 1.5, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.95,
        "Assignment links each ground-truth object to one or more predictions.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    round_box(ax, 0.4, 3.35, 5.4, 2.55, C_OLD, "#d35400", lw=2.0)
    ax.text(
        3.1, 5.4,
        "Classic (v1)",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color="#d35400",
    )
    ax.text(
        3.1, 4.2,
        "GT center falls in a cell.\n"
        "Best of B predictors\nin that cell is responsible.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 6.2, 3.35, 5.4, 2.55, C_GOAL, C_OK, lw=2.0)
    ax.text(
        8.9, 5.4,
        "Modern",
        ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_OK,
    )
    ax.text(
        8.9, 4.2,
        "Dynamic matching:\n"
        "score by IoU + class quality,\n"
        "often several positives / GT.",
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )

    round_box(ax, 0.4, 0.4, 11.2, 2.7, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.75,
        "Unmatched predictions are negatives → train objectness toward 0.\n"
        "Matched predictions get box regression + classification (+ obj) targets.\n"
        "Bad assignment ⇒ noisy gradients even if the architecture is fine.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 9: Modern pseudocode
# -------------------------------------------------------
MODERN_TRAIN = """\
for epoch in range(epochs):
    for images, gt_boxes, gt_labels in loader:   # Dataset + DataLoader
        images = augment(images, gt_boxes)       # mosaic / letterbox / ...

        preds = model(images)                    # multi-scale head outputs

        targets = assign(preds, gt_boxes, gt_labels)
        # -> positive matches, negative masks, encoded box targets

        loss = (lb * box_loss(preds, targets)
              + lo * obj_loss(preds, targets)
              + lc * cls_loss(preds, targets))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)                        # optional
"""


def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Modern YOLO training algorithm (pseudocode)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    code_panel(ax, 0.3, 0.35, 11.4, 7.2, MODERN_TRAIN, fontsize=FS_TICK)
    return fig


# -------------------------------------------------------
# Step 10: Takeaway
# -------------------------------------------------------
def build_step_10():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 10: Takeaway -- YOLO training in one page",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    rows = [
        (6.0, "v1 idea", "grid cell owns the object; predict box + conf + class"),
        (4.7, "Loss", "always multi-task: where / is-object / which-class"),
        (3.4, "Modern", "better augments, assignment, IoU box losses, multi-scale"),
        (2.1, "Loop", "DataLoader batch → assign → weighted loss → backward"),
    ]
    for y, left, right in rows:
        round_box(ax, 0.4, y - 0.5, 11.2, 1.15, C_NEW, C_LOOP, lw=1.8)
        ax.text(2.3, y + 0.08, left, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(7.5, y + 0.08, right, ha="center", va="center",
                fontsize=FS_ANNOT, color="#333333")

    round_box(ax, 0.4, 0.35, 11.2, 1.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 0.95,
        "YOLO trains like any net -- the detector art is targets + loss design.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
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
])


print("\n" + "=" * 60)
print("YOLO TRAINING ALGORITHM")
print("=" * 60)
print("""
YOLOv1:
  SxS grid; each cell predicts B boxes + objectness + classes
  Loss = coord*L_box + L_obj + noobj*L_noobj + L_cls
  Responsible box = highest IoU with GT in that cell

Modern YOLO:
  same outer loop (Dataset/DataLoader -> forward -> loss -> step)
  mosaic/augments, multi-scale heads, dynamic assignment
  L = lb*L_box(IoU-family) + lo*L_obj + lc*L_cls
""")
