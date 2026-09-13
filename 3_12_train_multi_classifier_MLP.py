"""
3_12_train_multi_classifier_MLP.py

Train a multi-class classifier on synthetic Cat / Dog / Pig blobs
that touch each other. Walk the code path, then watch train/val/test
loss move under a chosen learning rate.

Sketch:
  1. Task overview -- three classes that touch
  2. Code -- data
  3. Visual -- data distributions
  4. Code -- task / model definition
  5. Code -- loss function
  6. Code -- data pipeline (split + batches)
  7. Code -- training loop
  8. Interactive -- LR slider + loss animation (train / val / test)
  9. Code -- inference
  10. Code -- testing
  11. Test confusion table
  12. Takeaway
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, Slider
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

torch.manual_seed(45)
np.random.seed(45)


def new_figure(*args, **kwargs):
    kwargs.setdefault("figsize", screen_figsize())
    kwargs.setdefault("dpi", FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0, zorder=2):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, zorder=zorder,
    ))


def code_panel(ax, x, y, w, h, code: str, fontsize=None):
    fontsize = FS_TICK - 1 if fontsize is None else fontsize
    round_box(ax, x, y, w, h, C_CODE, C_LOOP, lw=2.0)
    ax.text(
        x + 0.35, y + h - 0.25, code.rstrip("\n"),
        ha="left", va="top",
        fontsize=fontsize,
        family="monospace",
        linespacing=1.32,
        color="#1c2833",
        zorder=4,
    )


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_OLD = "#fce4d6"
C_LOOP = "#2874a6"
C_CODE = "#fbfcf8"
C_VAL = "#8e44ad"
C_TEST = "#d35400"

CLASS_NAMES = ["cat", "dog", "pig"]
# Saturated / high-contrast class colors (cat, dog, pig).
CLASS_COLORS = ["#d62728", "#ffd400", "#00e5ff"]  # cat red, dog yellow, pig cyan


# -------------------------------------------------------
# Shared synthetic experiment (used by visuals + interactive)
# -------------------------------------------------------
N_PER = 120
# Centers close enough that clouds touch / slightly overlap.
CENTERS = np.array([
    [-1.05, -0.55],  # cat
    [1.05, -0.55],   # dog
    [0.00, 1.05],    # pig
], dtype=np.float32)
BLOB_STD = 0.62

xs, ys = [], []
for c, mean in enumerate(CENTERS):
    pts = mean + BLOB_STD * np.random.randn(N_PER, 2).astype(np.float32)
    xs.append(pts)
    ys.append(np.full(N_PER, c, dtype=np.int64))
X_all = np.vstack(xs)
y_all = np.concatenate(ys)

# Shuffle once, then fixed split: 60% train / 20% val / 20% test
perm = np.random.permutation(len(y_all))
X_all, y_all = X_all[perm], y_all[perm]
n = len(y_all)
n_train, n_val = int(0.60 * n), int(0.20 * n)
X_tr, y_tr = X_all[:n_train], y_all[:n_train]
X_va, y_va = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
X_te, y_te = X_all[n_train + n_val:], y_all[n_train + n_val:]

X_tr_t = torch.from_numpy(X_tr)
y_tr_t = torch.from_numpy(y_tr)
X_va_t = torch.from_numpy(X_va)
y_va_t = torch.from_numpy(y_va)
X_te_t = torch.from_numpy(X_te)
y_te_t = torch.from_numpy(y_te)


class SoftmaxClassifier(nn.Module):
    """Linear map R^2 -> 3 logits (one score per class)."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 3)

    def forward(self, x):
        return self.fc(x)


def dramatic_init_(model: SoftmaxClassifier) -> None:
    """Deliberately wrong, high-confidence init so boundaries move a lot.

    True layout: cat left, dog right, pig top (joint near the origin).
    Init maps: cat<-top, dog<-left, pig<-right  (cycled / flipped),
    with the three-way joint shifted away from the true meeting point.
    """
    with torch.no_grad():
        # Rows = class logits; columns = (x1, x2). Large norms => sharp wrong regions.
        W = torch.tensor(
            [
                [0.0, 5.0],   # "cat" fires on +x2  (pig's home)
                [-5.0, -1.5], # "dog" fires on -x1  (cat's home)
                [5.0, -1.5],  # "pig" fires on +x1  (dog's home)
            ],
            dtype=torch.float32,
        )
        # Offset the initial triple-point from the true ~origin joint.
        joint = torch.tensor([0.65, 0.40], dtype=torch.float32)
        b = -W @ joint  # all three logits equal at `joint`
        model.fc.weight.copy_(W)
        model.fc.bias.copy_(b)


def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train(train)
    total_loss, total_n, correct = 0.0, 0, 0
    for xb, yb in loader:
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += loss.item() * len(yb)
        total_n += len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
    return total_loss / max(total_n, 1), correct / max(total_n, 1)


# Decision grid for animated boundaries (Step 8)
_XX, _YY = np.meshgrid(
    np.linspace(X_all[:, 0].min() - 0.4, X_all[:, 0].max() + 0.4, 160),
    np.linspace(X_all[:, 1].min() - 0.4, X_all[:, 1].max() + 0.4, 160),
)
_GRID = torch.from_numpy(np.c_[_XX.ravel(), _YY.ravel()].astype(np.float32))


def predict_grid(model, grid=_GRID, xx=_XX):
    with torch.no_grad():
        return model(grid).argmax(dim=1).numpy().reshape(xx.shape)


def train_history(lr: float, epochs: int = 40, batch_size: int = 32, seed: int = 0):
    torch.manual_seed(seed)
    model = SoftmaxClassifier()
    dramatic_init_(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_va_t, y_va_t), batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_te_t, y_te_t), batch_size=batch_size, shuffle=False,
    )
    tr_hist, va_hist, te_hist, te_acc_hist, boundaries = [], [], [], [], []
    # Epoch-0 boundary (before any step) so animation can start from init.
    boundaries.append(predict_grid(model).astype(np.uint8))
    for _ in range(epochs):
        tr_loss, _ = run_epoch(model, train_loader, opt)
        with torch.no_grad():
            va_loss, _ = run_epoch(model, val_loader, opt=None)
            te_loss, te_acc = run_epoch(model, test_loader, opt=None)
        tr_hist.append(tr_loss)
        va_hist.append(va_loss)
        te_hist.append(te_loss)
        te_acc_hist.append(te_acc)
        boundaries.append(predict_grid(model).astype(np.uint8))
    return {
        "train": np.asarray(tr_hist, dtype=float),
        "val": np.asarray(va_hist, dtype=float),
        "test": np.asarray(te_hist, dtype=float),
        "test_acc_hist": np.asarray(te_acc_hist, dtype=float),
        "test_loss": float(te_hist[-1]),
        "test_acc": float(te_acc_hist[-1]),
        "model": model,
        # boundaries[0]=init, boundaries[k]=after epoch k  (len = epochs+1)
        "boundaries": boundaries,
    }


# Precompute a few LR trajectories for snappy interactive animation
EPOCHS = 40
LR_CHOICES = [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
DEMO_LR = 0.20  # Step 8 default
SELECTED_LR = DEMO_LR  # updated live by Step 8; later slides read this
LR_HIST = {lr: train_history(lr, epochs=EPOCHS, seed=7) for lr in LR_CHOICES}


def scatter_classes(ax, X, y, alpha=0.95, s=36, edge=True):
    for c, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        m = y == c
        ax.scatter(
            X[m, 0], X[m, 1], c=color, s=s, alpha=alpha,
            edgecolors="#111111" if edge else "none", linewidths=1.15,
            label=name, zorder=3,
        )


# -------------------------------------------------------
# Step 1: Overview
# -------------------------------------------------------
def build_step_1():
    w_in, h_in = screen_figsize()
    extra = 3.0 / 2.54  # each panel +3 cm
    fig, axes = plt.subplots(
        1, 2,
        figsize=(w_in + 2 * extra, h_in),
        dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [w_in / 2 + extra, w_in / 2 + extra]},
    )
    fig.suptitle(
        "Step 1: Multi-class task -- Cat / Dog / Pig",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    left_w_in = max(w_in / 2 + extra, 1.0)
    dx_2cm = (2.0 / 2.54) / left_w_in * 12.0  # blank_slide x: 0..12

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("What we will build", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    steps = [
        (6.4, "1. Data", "synthetic 2D blobs that touch"),
        (5.2, "2. Task", "predict class in {0,1,2}"),
        (4.0, "3. Loss", "cross-entropy on softmax"),
        (2.8, "4. Pipeline", "train / val / test loaders"),
        (1.6, "5. Train → Infer → Test", "watch losses under a chosen LR"),
    ]
    box_w = 11.0 + dx_2cm
    box_x = 6.0 - box_w / 2
    # Left-align title/body columns so every row (including the long bottom
    # title) shares the same horizontal text lines.
    title_x = box_x + 0.45
    body_x = box_x + box_w * 0.42
    for y, title, body in steps:
        round_box(ax, box_x, y - 0.45, box_w, 1.05, C_NEW, C_LOOP, lw=1.7)
        ax.text(title_x, y + 0.05, title, ha="left", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(body_x, y + 0.05, body, ha="left", va="center",
                fontsize=FS_ANNOT, color="#333333")

    ax = axes[1]
    ax.set_title("Why three piles that touch?", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    scatter_classes(ax, X_all, y_all, s=22)
    ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02, 0.02,
        "Touching / overlapping support\n=> non-trivial decision boundaries",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 2: Code -- data
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: Code -- data",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    code = """\
# Three Gaussian blobs that touch (cat=0, dog=1, pig=2)
centers = {
    "cat": (-1.05, -0.55),
    "dog": ( 1.05, -0.55),
    "pig": ( 0.00,  1.05),
}
std = 0.62
X_list, y_list = [], []
for label, (name, mean) in enumerate(centers.items()):
    pts = mean + std * torch.randn(120, 2)
    X_list.append(pts)
    y_list.append(torch.full((120,), label, dtype=torch.long))
X = torch.cat(X_list, dim=0)
y = torch.cat(y_list, dim=0)
# shuffle rows
perm = torch.randperm(len(y))
X, y = X[perm], y[perm]"""
    code_panel(ax, 0.35, 0.55, 11.3, 7.1, code, fontsize=FS_TICK - 1)
    return fig


# -------------------------------------------------------
# Step 3: Visual data
# -------------------------------------------------------
def build_step_3():
    fig, axes = new_figure(1, 3)
    fig.suptitle(
        "Step 3: Data -- train / validation / test piles",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    packs = [
        (axes[0], X_tr, y_tr, "Train (60%)", C_LOOP),
        (axes[1], X_va, y_va, "Validation (20%)", C_VAL),
        (axes[2], X_te, y_te, "Test (20%)", C_TEST),
    ]
    for ax, X, y, title, edge in packs:
        scatter_classes(ax, X, y, s=26)
        ax.set_title(title, fontsize=FS_AX_TITLE - 2, fontweight="bold", color=edge)
        ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
        ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 4: Code -- task / model
# -------------------------------------------------------
def build_step_4():
    w_in, h_in = screen_figsize()
    left_extra = 6.5 / 2.54   # left panel +6.5 cm (was +5, plus 1.5)
    right_extra = 10.0 / 2.54  # right panel / code box extends ~10 cm rightward
    fig, axes = plt.subplots(
        1, 2,
        figsize=(w_in + left_extra + right_extra, h_in),
        dpi=FIGURE_DPI,
        gridspec_kw={
            "width_ratios": [w_in / 2 + left_extra, w_in / 2 + right_extra],
        },
    )
    fig.suptitle(
        "Step 4: Code -- task definition (model)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Idea", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.5, 11.2, 2.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        r"logits $= W x + b \in \mathbb{R}^{3}$"
        + "\n"
        r"class $= \arg\max_k \mathrm{softmax}(\mathrm{logits})_k$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.8, 11.2, 2.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.95,
        "One linear score per class.\n"
        "Softmax turns scores into a distribution over {cat, dog, pig}.\n"
        "Prediction = class with largest probability.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.4, 0.5, 11.2, 1.9, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 1.45,
        "Task = learn W, b so touching clouds become separable enough.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    # Wider data range so the code textbox can grow rightward by ~10 cm.
    right_w_in = max(w_in / 2 + right_extra, 1.0)
    dx_10cm = (10.0 / 2.54) / right_w_in * 12.0
    code_w = 11.3 + dx_10cm
    ax.set_xlim(0, max(12.0, 0.35 + code_w + 0.35))
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Code", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    code = """\
class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 3)  # 2 features -> 3 logits

    def forward(self, x):
        return self.fc(x)          # raw scores

model = SoftmaxClassifier()
# later:
#   logits = model(x)             # (batch, 3)
#   pred   = logits.argmax(dim=1)"""
    code_panel(ax, 0.35, 0.6, code_w, 7.0, code, fontsize=FS_TICK)
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 5: Code -- loss
# -------------------------------------------------------
def build_step_5():
    w_in, h_in = screen_figsize()
    extra = 2.0 / 2.54  # each panel +2 cm
    fig, axes = plt.subplots(
        1, 2,
        figsize=(w_in + 2 * extra, h_in),
        dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [w_in / 2 + extra, w_in / 2 + extra]},
    )
    fig.suptitle(
        "Step 5: Code -- loss function",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Cross-entropy", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.3, 11.2, 2.3, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.45,
        r"$\mathcal{L} = -\log p_{y}$"
        + "\n"
        r"$p = \mathrm{softmax}(Wx+b)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 3,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.5, 11.2, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.7,
        "Punishes confident wrong answers.\n"
        "Natural loss when the model outputs class probabilities.\n"
        "In Torch: F.cross_entropy(logits, y)  (softmax inside).",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.4, 0.4, 11.2, 1.7, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 1.25,
        "Train minimizes mean CE on train batches.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("Code", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    code = """\
import torch.nn.functional as F

logits = model(x)                 # (B, 3)
loss = F.cross_entropy(logits, y) # y: (B,) long

# Equivalent expanded form:
#   log_p = F.log_softmax(logits, dim=1)
#   loss  = F.nll_loss(log_p, y)

loss.backward()                   # dL/dW, dL/db"""
    code_panel(ax, 0.35, 0.6, 11.3, 7.0, code, fontsize=FS_TICK)
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 6: Code -- data pipeline
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Code -- data pipeline",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    code = """\
from torch.utils.data import DataLoader, TensorDataset

n = len(y)
n_train, n_val = int(0.60 * n), int(0.20 * n)
X_tr, y_tr = X[:n_train], y[:n_train]
X_va, y_va = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_te, y_te = X[n_train+n_val:], y[n_train+n_val:]

train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_va, y_va),
                          batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_te, y_te),
                          batch_size=32, shuffle=False)
# Train: fit  |  Val: choose / monitor  |  Test: final report once"""
    code_panel(ax, 0.35, 0.45, 11.3, 7.2, code, fontsize=FS_TICK - 1)
    return fig


# -------------------------------------------------------
# Step 7: Code -- training
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: Code -- training",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    code = """\
model = SoftmaxClassifier()
opt = torch.optim.SGD(model.parameters(), lr=lr)   # lr from slider later

def run_epoch(loader, train=True):
    model.train(train)
    total, n = 0.0, 0
    for xb, yb in loader:
        loss = F.cross_entropy(model(xb), yb)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(yb); n += len(yb)
    return total / n

for epoch in range(epochs):
    tr = run_epoch(train_loader, train=True)
    with torch.no_grad():
        va = run_epoch(val_loader, train=False)
    # log tr, va each epoch  -- test only at the end"""
    code_panel(ax, 0.35, 0.4, 11.3, 7.3, code, fontsize=FS_TICK - 1)
    return fig


# -------------------------------------------------------
# Step 8: Interactive LR slider + animated losses
# -------------------------------------------------------
def build_step_8():
    global SELECTED_LR
    fig, axes = new_figure(1, 2)
    fig.subplots_adjust(bottom=0.20, wspace=0.28)
    fig.suptitle(
        "Step 8: Watch loss fall -- choose learning rate",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    lr0 = DEMO_LR
    SELECTED_LR = lr0
    hist0 = LR_HIST[lr0]
    epochs = np.arange(1, EPOCHS + 1)
    cmap = plt.matplotlib.colors.ListedColormap(CLASS_COLORS)
    n_frames = EPOCHS + 1  # frame 0 = wild init; then one frame per epoch

    ax_data = axes[0]
    ax_data.set_title("Decision boundaries (wild init → trained)", fontsize=FS_AX_TITLE - 3,
                      fontweight="bold")
    scatter_classes(ax_data, X_tr, y_tr, s=22, alpha=0.75)
    # boundaries[0]=dramatic init; boundaries[k]=after epoch k
    zz0 = hist0["boundaries"][0]
    ax_data.contourf(
        _XX, _YY, zz0, levels=[-0.5, 0.5, 1.5, 2.5],
        cmap=cmap, alpha=0.22, zorder=1,
    )
    ax_data.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax_data.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax_data.tick_params(labelsize=FS_TICK)
    ax_data.legend(fontsize=FS_LEGEND - 1, loc="upper right")
    ax_data.set_aspect("equal", adjustable="box")
    ax_data.grid(True, alpha=0.2)

    ax_loss = axes[1]
    ax_loss.set_title("Loss vs epoch", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    (line_tr,) = ax_loss.plot([], [], color=C_LOOP, lw=2.6, label="train")
    (line_va,) = ax_loss.plot([], [], color=C_VAL, lw=2.6, label="val")
    (line_te,) = ax_loss.plot([], [], color=C_TEST, lw=2.4, ls="--", label="test")
    (pt_tr,) = ax_loss.plot([], [], "o", color=C_LOOP, ms=8)
    (pt_va,) = ax_loss.plot([], [], "o", color=C_VAL, ms=8)
    (pt_te,) = ax_loss.plot([], [], "o", color=C_TEST, ms=8)
    ax_loss.set_xlabel("epoch", fontsize=FS_AXIS)
    ax_loss.set_ylabel("cross-entropy", fontsize=FS_AXIS)
    ax_loss.tick_params(labelsize=FS_TICK)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=FS_LEGEND, loc="upper right")
    # Sit just under the upper-right legend.
    hud = ax_loss.text(
        0.98, 0.78, "",
        transform=ax_loss.transAxes, ha="right", va="top",
        fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )

    ax_sl = fig.add_axes([0.18, 0.06, 0.50, 0.04])
    sl = Slider(
        ax_sl, "LR", 0, len(LR_CHOICES) - 1,
        valinit=LR_CHOICES.index(lr0), valstep=1, color=C_LOOP,
    )
    sl.valtext.set_text(f"{lr0:.2f}")

    ax_btn = fig.add_axes([0.80, 0.045, 0.15, 0.055])
    btn = Button(ax_btn, "Stop", color="#f5f5f5", hovercolor="#d6eaf8")
    btn.label.set_fontsize(FS_TICK)

    state = {"lr": lr0, "frame": 0, "playing": True}

    def current_hist():
        return LR_HIST[state["lr"]]

    def draw_boundary(zz, title: str):
        for coll in list(ax_data.collections):
            coll.remove()
        scatter_classes(ax_data, X_tr, y_tr, s=22, alpha=0.75)
        ax_data.contourf(
            _XX, _YY, zz, levels=[-0.5, 0.5, 1.5, 2.5],
            cmap=cmap, alpha=0.22, zorder=1,
        )
        ax_data.set_title(title, fontsize=FS_AX_TITLE - 3, fontweight="bold")
        ax_data.legend(fontsize=FS_LEGEND - 1, loc="upper right")
        ax_data.set_aspect("equal", adjustable="box")
        ax_data.grid(True, alpha=0.2)

    def refresh_legend(h, i):
        if i <= 0:
            line_tr.set_label("train")
            line_va.set_label("val")
            line_te.set_label("test")
        else:
            line_tr.set_label(f"train = {h['train'][i - 1]:.3f}")
            line_va.set_label(f"val = {h['val'][i - 1]:.3f}")
            line_te.set_label(f"test = {h['test'][i - 1]:.3f}")
        ax_loss.legend(fontsize=FS_LEGEND, loc="upper right")

    def show_frame(i):
        h = current_hist()
        i = int(np.clip(i, 0, EPOCHS))
        state["frame"] = i
        # i=0: wild init only; i>=1: after epoch i
        if i == 0:
            line_tr.set_data([], [])
            line_va.set_data([], [])
            line_te.set_data([], [])
            pt_tr.set_data([], [])
            pt_va.set_data([], [])
            pt_te.set_data([], [])
            title = "Decision boundaries (wild init)"
            hud.set_text(
                f"LR={state['lr']:.2f}   epoch 0/{EPOCHS} (wild init)\n"
                f"classes are cycled on purpose -- watch them unswap\n"
                f"final test={h['test_loss']:.3f}   "
                f"acc={100 * h['test_acc']:.1f}%"
            )
        else:
            line_tr.set_data(epochs[:i], h["train"][:i])
            line_va.set_data(epochs[:i], h["val"][:i])
            line_te.set_data(epochs[:i], h["test"][:i])
            pt_tr.set_data([epochs[i - 1]], [h["train"][i - 1]])
            pt_va.set_data([epochs[i - 1]], [h["val"][i - 1]])
            pt_te.set_data([epochs[i - 1]], [h["test"][i - 1]])
            title = f"Decision boundaries (epoch {i}/{EPOCHS})"
            hud.set_text(
                f"LR={state['lr']:.2f}   epoch {i}/{EPOCHS}\n"
                f"train={h['train'][i - 1]:.3f}   val={h['val'][i - 1]:.3f}\n"
                f"test={h['test'][i - 1]:.3f}   "
                f"acc={100 * h['test_acc_hist'][i - 1]:.1f}%"
            )
        refresh_legend(h, i)
        ymax = max(
            float(h["train"].max()), float(h["val"].max()), float(h["test"].max()),
        ) * 1.15
        ax_loss.set_xlim(0, EPOCHS)
        ax_loss.set_ylim(0.0, max(0.5, ymax))
        draw_boundary(h["boundaries"][i], title)
        return line_tr, line_va, line_te, pt_tr, pt_va, pt_te, hud

    def on_lr(_=None):
        global SELECTED_LR
        idx = int(sl.val)
        state["lr"] = LR_CHOICES[idx]
        SELECTED_LR = state["lr"]
        sl.valtext.set_text(f"{state['lr']:.2f}")
        show_frame(0)
        anim.frame_seq = anim.new_frame_seq()
        # Changing LR restarts playback from the wild init.
        if anim.event_source is not None:
            anim.event_source.start()
        btn.label.set_text("Stop")
        state["playing"] = True
        fig.canvas.draw_idle()

    sl.on_changed(on_lr)

    def update(frame):
        artists = show_frame(frame)
        # Stop at the end and flip the toggle to Resume.
        if frame >= n_frames - 1:
            if anim.event_source is not None:
                anim.event_source.stop()
            btn.label.set_text("Resume")
            state["playing"] = False
            fig.canvas.draw_idle()
        return artists

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=120, blit=False, repeat=False,
    )

    def on_toggle(_event):
        if state["playing"]:
            if anim.event_source is not None:
                anim.event_source.stop()
            btn.label.set_text("Resume")
            state["playing"] = False
        else:
            # Finished run: restart from the beginning on Resume.
            if state["frame"] >= n_frames - 1:
                show_frame(0)
                anim.frame_seq = anim.new_frame_seq()
            if anim.event_source is not None:
                anim.event_source.start()
            btn.label.set_text("Stop")
            state["playing"] = True
        fig.canvas.draw_idle()

    btn.on_clicked(on_toggle)

    fig._lecture_anim = anim  # keep alive
    fig._retained_widgets = getattr(fig, "_retained_widgets", [])
    fig._retained_widgets.extend([sl, btn])

    show_frame(0)
    return fig


# -------------------------------------------------------
# Step 9: Code -- inference
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Code -- inference",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    code = """\
model.eval()
names = ["cat", "dog", "pig"]

def predict(x2d):
    # x2d: (2,) or (B, 2) features
    with torch.no_grad():
        logits = model(x2d.reshape(-1, 2))
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
    return pred, prob

x_new = torch.tensor([[0.1, 0.2]])
pred, prob = predict(x_new)
print(names[int(pred)], prob.numpy().round(3))
# -> e.g. ('pig', array([[0.18, 0.21, 0.61]]))"""
    code_panel(ax, 0.35, 0.45, 11.3, 7.2, code, fontsize=FS_TICK - 1)
    return fig


# -------------------------------------------------------
# Step 10: Code -- testing
# -------------------------------------------------------
def build_step_10():
    w_in, h_in = screen_figsize()
    left_extra = 4.0 / 2.54   # left panel +4 cm
    right_extra = 3.0 / 2.54  # right panel +3 cm
    fig, axes = plt.subplots(
        1, 2,
        figsize=(w_in + left_extra + right_extra, h_in),
        dpi=FIGURE_DPI,
        gridspec_kw={
            "width_ratios": [w_in / 2 + left_extra, w_in / 2 + right_extra],
        },
    )
    fig.suptitle(
        "Step 10: Code -- testing (report once)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Code", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    code = """\
model.eval()
total_loss, n, correct = 0.0, 0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        total_loss += loss.item() * len(yb)
        n += len(yb)
        correct += (logits.argmax(1) == yb).sum().item()

test_loss = total_loss / n
test_acc  = correct / n
print(f"test CE={test_loss:.3f}  acc={test_acc:.3f}")
# Do not tune LR / epochs on this number."""
    code_panel(ax, 0.25, 0.45, 11.5, 7.1, code, fontsize=FS_TICK - 2)

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("Roles reminder", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    cards = [
        (5.8, C_NEW, C_LOOP, "Train loss", "fits the model"),
        (3.9, "#f5eef8", C_VAL, "Val loss", "monitor / choose LR"),
        (2.0, C_OLD, C_TEST, "Test loss", "final honest report"),
    ]
    for y, face, edge, title, body in cards:
        round_box(ax, 0.6, y - 0.55, 10.8, 1.5, face, edge, lw=2.0)
        ax.text(3.0, y + 0.15, title, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=edge)
        ax.text(8.0, y + 0.15, body, ha="center", va="center",
                fontsize=FS_ANNOT, color="#333333")
    round_box(ax, 0.6, 0.35, 10.8, 0.95, C_MUTE, "0.45", lw=1.4)
    ax.text(
        6.0, 0.82,
        "Animation shows all three: train & val curves + test line.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 11: Test confusion table
# -------------------------------------------------------
def confusion_matrix_np(model, X_t, y_np, n_classes=3):
    model.eval()
    with torch.no_grad():
        pred = model(X_t).argmax(dim=1).cpu().numpy()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_np, pred):
        cm[int(t), int(p)] += 1
    return cm


def build_step_11():
    """Confusion table on TEST using the LR last chosen on Step 8."""
    fig, axes = new_figure(1, 2)
    lr = SELECTED_LR if SELECTED_LR in LR_HIST else DEMO_LR
    hist = LR_HIST[lr]
    cm = confusion_matrix_np(hist["model"], X_te_t, y_te)
    n_test = int(cm.sum())
    acc = float(np.trace(cm)) / max(n_test, 1)

    fig.suptitle(
        f"Step 11: Test confusion table from Step 8 model  (LR={lr:g}, n={n_test})",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Counts  (rows = true, cols = predicted)", fontsize=FS_AX_TITLE - 3,
                 fontweight="bold")
    # Table geometry in blank_slide coords
    x0, y0 = 2.0, 1.4
    cell = 2.2
    # column headers (predicted)
    ax.text(x0 - 0.3, y0 + 3.5 * cell + 0.35, "true \\ pred",
            ha="right", va="center", fontsize=FS_TICK, color="#555555")
    for j, name in enumerate(CLASS_NAMES):
        ax.text(x0 + (j + 0.5) * cell, y0 + 3.5 * cell + 0.35, name,
                ha="center", va="center", fontsize=FS_ANNOT,
                fontweight="bold", color=CLASS_COLORS[j])
    for i, name in enumerate(CLASS_NAMES):
        yy = y0 + (2 - i) * cell
        ax.text(x0 - 0.3, yy + 0.5 * cell, name,
                ha="right", va="center", fontsize=FS_ANNOT,
                fontweight="bold", color=CLASS_COLORS[i])
        for j in range(3):
            xx = x0 + j * cell
            face = C_GOAL if i == j else C_MUTE
            edge = C_OK if i == j else "0.55"
            round_box(ax, xx + 0.08, yy + 0.08, cell - 0.16, cell - 0.16,
                      face, edge, lw=1.8)
            ax.text(xx + 0.5 * cell, yy + 0.5 * cell, str(cm[i, j]),
                    ha="center", va="center", fontsize=FS_AX_TITLE - 2,
                    fontweight="bold",
                    color=(C_OK if i == j else "#333333"))

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("How to read it", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.5, 5.5, 11.0, 2.0, C_NEW, C_LOOP, lw=1.8)
    ax.text(
        6.0, 6.5,
        f"Test accuracy = trace / n = {np.trace(cm)} / {n_test} = {100 * acc:.1f}%\n"
        "Diagonal = correct · Off-diagonal = confusions",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.5, 3.2, 11.0, 1.9, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 4.15,
        f"Uses the LR last chosen on Step 8 (now LR={lr:g}).\n"
        "Built only on the TEST split -- do not tune on this table.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    # Per-class recall strip
    round_box(ax, 0.5, 0.5, 11.0, 2.3, C_OLD, "#d35400", lw=1.6)
    lines = []
    for i, name in enumerate(CLASS_NAMES):
        row = cm[i].sum()
        hit = cm[i, i]
        rec = hit / max(row, 1)
        lines.append(f"{name}: {hit}/{row} correct ({100 * rec:.0f}% recall)")
    ax.text(
        6.0, 1.65,
        "Per-class (true-row) recall on TEST\n" + "\n".join(lines),
        ha="center", va="center", fontsize=FS_TICK + 1, color="#333333",
    )
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 12: Takeaway
# -------------------------------------------------------
def build_step_12():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 12: Takeaway",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)
    points = [
        (6.0, "Data", "touching Cat/Dog/Pig blobs => real multi-class pressure"),
        (4.7, "Model + loss", "linear logits + cross-entropy"),
        (3.4, "Pipeline", "train fit · val monitor · test once"),
        (2.1, "LR slider", "same data/model; only step size changes the path"),
    ]
    for y, left, right in points:
        round_box(ax, 0.5, y - 0.5, 11.0, 1.15, C_NEW, C_LOOP, lw=1.8)
        ax.text(2.4, y + 0.05, left, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(7.5, y + 0.05, right, ha="center", va="center",
                fontsize=FS_ANNOT, color="#333333")
    round_box(ax, 0.5, 0.3, 11.0, 1.05, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 0.82,
        "A classifier training story = data → task → loss → pipeline → train → infer → test.",
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
    build_step_11,
    build_step_12,
])


print("\n" + "=" * 60)
print("MULTI-CLASS TRAIN EXAMPLE -- Cat / Dog / Pig")
print("=" * 60)
print(
    f"""
Data: {len(y_all)} points, 3 touching Gaussian blobs.
Split: train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}
Model: Linear(2 -> 3) + cross-entropy
Interactive Step 8: drag LR, watch train/val curves animate; test loss shown.

Sample precomputed test CE @ LR={DEMO_LR:g}: {LR_HIST[DEMO_LR]['test_loss']:.3f}
"""
)
print("=" * 60)
