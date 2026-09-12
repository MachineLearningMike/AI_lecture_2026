"""
text.py

RNN training twin of 3_045_train_multi_classifier.py

Classify short noisy 2D trajectories (walk styles):
  circle / zigzag / drift
with a tiny GRU + Softmax/CE, train/val/test split, and an
interactive LR + loss animation (Stop/Resume).

Sketch:
  1. Task overview
  2. Code -- data
  3. Visual -- example sequences
  4. Code -- task / model
  5. Code -- loss
  6. Code -- data pipeline
  7. Code -- training
  8. Interactive -- LR slider + train/val/test loss animation
  9. Code -- inference
  10. Code -- testing
  11. Test confusion table
  12. Takeaway
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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

torch.manual_seed(46)
np.random.seed(46)


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

CLASS_NAMES = ["circle", "zigzag", "drift"]
CLASS_COLORS = ["#d62728", "#ffd400", "#00e5ff"]  # red / yellow / cyan


# -------------------------------------------------------
# Synthetic sequence data
# -------------------------------------------------------
T = 20
N_PER = 80
NOISE = 0.18


def make_circle(rng):
    t = np.linspace(0, 2 * np.pi, T, endpoint=False)
    r = 0.85 + 0.15 * rng.randn()
    phase = rng.uniform(0, 2 * np.pi)
    xy = np.stack([r * np.cos(t + phase), r * np.sin(t + phase)], axis=1)
    return xy + NOISE * rng.randn(T, 2)


def make_zigzag(rng):
    t = np.linspace(-1.0, 1.0, T)
    amp = 0.75 + 0.15 * rng.randn()
    y = amp * ((np.arange(T) % 2) * 2 - 1)
    xy = np.stack([t, y], axis=1)
    return xy + NOISE * rng.randn(T, 2)


def make_drift(rng):
    t = np.linspace(-1.0, 1.0, T)
    a = 0.55 + 0.25 * rng.randn()
    xy = np.stack([t, a * t], axis=1)
    return xy + NOISE * rng.randn(T, 2)


MAKERS = [make_circle, make_zigzag, make_drift]

seqs, labels = [], []
for c, maker in enumerate(MAKERS):
    for i in range(N_PER):
        rng = np.random.RandomState(1000 * c + i)
        seqs.append(maker(rng).astype(np.float32))
        labels.append(c)

X_all = np.stack(seqs, axis=0)          # (N, T, 2)
y_all = np.asarray(labels, dtype=np.int64)

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


class SeqClassifier(nn.Module):
    """GRU over (B,T,2) → logits over 3 walk styles."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.rnn = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 3)

    def forward(self, x):
        _h_seq, h_last = self.rnn(x)          # h_last: (1, B, H)
        return self.fc(h_last.squeeze(0))     # (B, 3)


def dramatic_init_(model: SeqClassifier) -> None:
    """Wrong but recoverable head -- early epochs look bad, then unswap.

    Keep this milder than a hard collapse so every demo LR finishes with
    correct showcase predictions (test accuracy stays high).
    """
    with torch.no_grad():
        # Cycle the classifier rows: prefer the next class.
        W = model.fc.weight.detach().clone()
        model.fc.weight.copy_(torch.roll(W, shifts=1, dims=0))
        # Mild bias toward a wrong class (not so large that Adam can't undo it).
        model.fc.bias.copy_(torch.tensor([0.8, -0.4, -0.4]))


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


def train_history(lr: float, epochs: int = 35, batch_size: int = 32, seed: int = 0):
    torch.manual_seed(seed)
    model = SeqClassifier()
    dramatic_init_(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size, shuffle=False)

    # Fixed showcase sequences (one per class from train) for side panel
    showcase_idx = []
    for c in range(3):
        hits = np.where(y_tr == c)[0]
        showcase_idx.append(int(hits[0]))
    showcase = X_tr_t[showcase_idx]  # (3, T, 2)
    showcase_y = y_tr[showcase_idx]

    tr_hist, va_hist, te_hist, te_acc_hist, preds_hist = [], [], [], [], []
    with torch.no_grad():
        preds_hist.append(model(showcase).argmax(dim=1).numpy().copy())

    for _ in range(epochs):
        tr_loss, _ = run_epoch(model, train_loader, opt)
        with torch.no_grad():
            va_loss, _ = run_epoch(model, val_loader, opt=None)
            te_loss, te_acc = run_epoch(model, test_loader, opt=None)
            pred = model(showcase).argmax(dim=1).numpy().copy()
        tr_hist.append(tr_loss)
        va_hist.append(va_loss)
        te_hist.append(te_loss)
        te_acc_hist.append(te_acc)
        preds_hist.append(pred)

    return {
        "train": np.asarray(tr_hist, dtype=float),
        "val": np.asarray(va_hist, dtype=float),
        "test": np.asarray(te_hist, dtype=float),
        "test_acc_hist": np.asarray(te_acc_hist, dtype=float),
        "test_loss": float(te_hist[-1]),
        "test_acc": float(te_acc_hist[-1]),
        "model": model,
        "showcase": showcase.numpy(),
        "showcase_y": showcase_y,
        "preds": preds_hist,  # list len epochs+1, each (3,)
    }


EPOCHS = 40
LR_CHOICES = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
DEMO_LR = 0.02  # Step 8 default
SELECTED_LR = DEMO_LR  # updated live by Step 8; later slides read this
LR_HIST = {lr: train_history(lr, epochs=EPOCHS, seed=7) for lr in LR_CHOICES}


def plot_seq(ax, seq, color, lw=2.2, label=None):
    ax.plot(seq[:, 0], seq[:, 1], "-", color=color, lw=lw, label=label, zorder=3)
    ax.scatter(seq[:1, 0], seq[:1, 1], c=color, s=40, zorder=4, edgecolors="k", linewidths=0.8)
    ax.scatter(seq[-1:, 0], seq[-1:, 1], c=color, s=55, marker="*", zorder=4,
               edgecolors="k", linewidths=0.8)


# -------------------------------------------------------
# Step 1: Overview
# -------------------------------------------------------
def build_step_1():
    w_in, h_in = screen_figsize()
    extra = 3.0 / 2.54
    fig, axes = plt.subplots(
        1, 2, figsize=(w_in + 2 * extra, h_in), dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [w_in / 2 + extra, w_in / 2 + extra]},
    )
    fig.suptitle(
        "Step 1: RNN task -- classify a walk style",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("What we will build", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    steps = [
        (6.4, "1. Data", "noisy 2D trajectories, length T"),
        (5.2, "2. Task", "predict style in {0,1,2}"),
        (4.0, "3. Loss", "cross-entropy on sequence label"),
        (2.8, "4. Pipeline", "train / val / test loaders"),
        (1.6, "5. Train → Infer → Test", "GRU memory + LR story"),
    ]
    box_w, box_x = 11.2, 0.4
    title_x, body_x = box_x + 0.45, box_x + box_w * 0.42
    for y, title, body in steps:
        round_box(ax, box_x, y - 0.45, box_w, 1.05, C_NEW, C_LOOP, lw=1.7)
        ax.text(title_x, y + 0.05, title, ha="left", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(body_x, y + 0.05, body, ha="left", va="center",
                fontsize=FS_ANNOT, color="#333333")

    ax = axes[1]
    ax.set_title("Three confusable walk styles", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    for c, name, color in zip(range(3), CLASS_NAMES, CLASS_COLORS):
        # pick one train example
        idx = np.where(y_tr == c)[0][0]
        plot_seq(ax, X_tr[idx], color, label=name)
    ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.02, 0.02,
        "Noise makes styles touch -- need memory over time",
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
    fig.suptitle("Step 2: Code -- sequence data", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    blank_slide(ax)
    code = """\
# Each example is a path x[t] in R^2, t = 1..T
T = 20
# circle:  (cos wt, sin wt) + noise
# zigzag:  (t, +/- amp)     + noise
# drift:   (t, a*t)         + noise
X = torch.stack(seqs)      # (N, T, 2)
y = torch.tensor(labels)   # (N,)  in {0,1,2}
perm = torch.randperm(len(y))
X, y = X[perm], y[perm]"""
    code_panel(ax, 0.35, 0.55, 11.3, 7.1, code, fontsize=FS_TICK)
    return fig


# -------------------------------------------------------
# Step 3: Visual examples
# -------------------------------------------------------
def build_step_3():
    fig, axes = new_figure(1, 3)
    fig.suptitle(
        "Step 3: Data -- train / val / test examples",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    packs = [
        (axes[0], X_tr, y_tr, "Train", C_LOOP),
        (axes[1], X_va, y_va, "Validation", C_VAL),
        (axes[2], X_te, y_te, "Test", C_TEST),
    ]
    for ax, X, y, title, edge in packs:
        ax.set_title(title, fontsize=FS_AX_TITLE - 2, fontweight="bold", color=edge)
        for c, name, color in zip(range(3), CLASS_NAMES, CLASS_COLORS):
            hits = np.where(y == c)[0][:3]
            for j, idx in enumerate(hits):
                plot_seq(ax, X[idx], color, lw=1.8 if j == 0 else 1.2,
                         label=name if j == 0 else None)
        ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
        ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=FS_LEGEND - 1, loc="best")
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 4: Model
# -------------------------------------------------------
def build_step_4():
    w_in, h_in = screen_figsize()
    left_extra, right_extra = 4.0 / 2.54, 6.0 / 2.54
    fig, axes = plt.subplots(
        1, 2, figsize=(w_in + left_extra + right_extra, h_in), dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [w_in / 2 + left_extra, w_in / 2 + right_extra]},
    )
    fig.suptitle(
        "Step 4: Code -- task definition (RNN)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Idea", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.4, 11.2, 2.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        r"$h_t = \mathrm{GRU}(x_t, h_{t-1})$"
        + "\n"
        r"logits $= W h_T + b$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.7, 11.2, 2.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.85,
        "Read the path step by step into a memory h.\n"
        "Classify from the final memory h_T.\n"
        "Same Softmax multi-class head as the blob lecture.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.4, 0.45, 11.2, 1.9, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 1.4,
        "New noun vs 3_045: sequence memory. Same training moral.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("Code", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    code = """\
class SeqClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(2, 16, batch_first=True)
        self.fc  = nn.Linear(16, 3)

    def forward(self, x):          # x: (B, T, 2)
        _, h = self.rnn(x)         # h: (1, B, 16)
        return self.fc(h.squeeze(0))

model = SeqClassifier()"""
    code_panel(ax, 0.35, 0.55, 11.3, 7.1, code, fontsize=FS_TICK)
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 5: Loss
# -------------------------------------------------------
def build_step_5():
    w_in, h_in = screen_figsize()
    extra = 2.0 / 2.54
    fig, axes = plt.subplots(
        1, 2, figsize=(w_in + 2 * extra, h_in), dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [w_in / 2 + extra, w_in / 2 + extra]},
    )
    fig.suptitle(
        "Step 5: Code -- loss function",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Cross-entropy on the sequence label", fontsize=FS_AX_TITLE - 3, fontweight="bold")
    round_box(ax, 0.4, 5.3, 11.2, 2.3, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.45,
        r"$\mathcal{L} = -\log p_{y}$"
        + "\n"
        r"$p = \mathrm{softmax}(W h_T + b)$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 3,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.5, 11.2, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.7,
        "One label for the whole trajectory.\n"
        "Same CE as the static multi-class classifier.\n"
        "Torch: F.cross_entropy(logits, y)",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.4, 0.4, 11.2, 1.7, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 1.25,
        "Train minimizes mean CE on train batches of sequences.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("Code", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    code = """\
logits = model(x)                 # (B, 3)
loss = F.cross_entropy(logits, y) # y: (B,) long
loss.backward()
opt.step()"""
    code_panel(ax, 0.35, 1.8, 11.3, 5.0, code, fontsize=FS_TICK + 1)
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 6: Pipeline
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle("Step 6: Code -- data pipeline", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    blank_slide(ax)
    code = """\
from torch.utils.data import DataLoader, TensorDataset

# X: (N, T, 2), y: (N,)
n_train, n_val = int(0.60 * n), int(0.20 * n)
X_tr, y_tr = X[:n_train], y[:n_train]
X_va, y_va = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_te, y_te = X[n_train+n_val:], y[n_train+n_val:]

train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_va, y_va),
                          batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_te, y_te),
                          batch_size=32, shuffle=False)"""
    code_panel(ax, 0.35, 0.4, 11.3, 7.3, code, fontsize=FS_TICK - 1)
    return fig


# -------------------------------------------------------
# Step 7: Training code
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle("Step 7: Code -- training", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    blank_slide(ax)
    code = """\
model = SeqClassifier()
opt = torch.optim.Adam(model.parameters(), lr=lr)

def run_epoch(loader, train=True):
    model.train(train)
    total, n = 0.0, 0
    for xb, yb in loader:              # xb: (B, T, 2)
        loss = F.cross_entropy(model(xb), yb)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(yb); n += len(yb)
    return total / n

for epoch in range(epochs):
    tr = run_epoch(train_loader, train=True)
    with torch.no_grad():
        va = run_epoch(val_loader, train=False)
    # log tr, va -- test only at the end"""
    code_panel(ax, 0.35, 0.35, 11.3, 7.4, code, fontsize=FS_TICK - 1)
    return fig


# -------------------------------------------------------
# Step 8: Interactive
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
    n_frames = EPOCHS + 1

    ax_seq = axes[0]
    ax_seq.set_title(
        "Train showcase (NOT the test set)",
        fontsize=FS_AX_TITLE - 3, fontweight="bold",
    )

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
    sl.valtext.set_text(f"{lr0:.3g}")

    ax_btn = fig.add_axes([0.80, 0.045, 0.15, 0.055])
    btn = Button(ax_btn, "Stop", color="#f5f5f5", hovercolor="#d6eaf8")
    btn.label.set_fontsize(FS_TICK)

    state = {"lr": lr0, "frame": 0, "playing": True}

    def current_hist():
        return LR_HIST[state["lr"]]

    def draw_showcase(preds, epoch_label: str):
        h = current_hist()
        ax_seq.cla()
        n_ok = int(np.sum(np.asarray(preds) == np.asarray(h["showcase_y"])))
        ax_seq.set_title(
            f"Train showcase -- not test  ({epoch_label}; {n_ok}/3 correct)",
            fontsize=FS_AX_TITLE - 3, fontweight="bold",
        )
        for c in range(3):
            true_c = int(h["showcase_y"][c])
            pred_c = int(preds[c])
            color = CLASS_COLORS[true_c]
            plot_seq(ax_seq, h["showcase"][c], color, lw=2.4,
                     label=f"true {CLASS_NAMES[true_c]}")
            mark = "OK" if pred_c == true_c else "WRONG"
            ax_seq.text(
                h["showcase"][c][-1, 0],
                h["showcase"][c][-1, 1],
                f" pred={CLASS_NAMES[pred_c]} ({mark})",
                fontsize=FS_TICK, fontweight="bold",
                color=CLASS_COLORS[pred_c],
                ha="left", va="bottom",
            )
        ax_seq.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
        ax_seq.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
        ax_seq.tick_params(labelsize=FS_TICK)
        ax_seq.grid(True, alpha=0.25)
        ax_seq.set_aspect("equal", adjustable="box")
        handles, labels = ax_seq.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax_seq.legend(uniq.values(), uniq.keys(), fontsize=FS_LEGEND - 1, loc="best")
        ax_seq.text(
            0.02, 0.02,
            "Left = 3 fixed TRAIN examples for intuition.\n"
            "Right test curve is loss on TEST after each epoch.",
            transform=ax_seq.transAxes, ha="left", va="bottom", fontsize=FS_TICK - 1,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
        )

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
        if i == 0:
            line_tr.set_data([], [])
            line_va.set_data([], [])
            line_te.set_data([], [])
            pt_tr.set_data([], [])
            pt_va.set_data([], [])
            pt_te.set_data([], [])
            hud.set_text(
                f"LR={state['lr']:.3g}   epoch 0/{EPOCHS} (wild init)\n"
                f"left panel is TRAIN showcase, not test\n"
                f"final TEST acc={100 * h['test_acc']:.1f}%   "
                f"loss={h['test_loss']:.3f}"
            )
            draw_showcase(h["preds"][0], "wild init")
        else:
            line_tr.set_data(epochs[:i], h["train"][:i])
            line_va.set_data(epochs[:i], h["val"][:i])
            line_te.set_data(epochs[:i], h["test"][:i])
            pt_tr.set_data([epochs[i - 1]], [h["train"][i - 1]])
            pt_va.set_data([epochs[i - 1]], [h["val"][i - 1]])
            pt_te.set_data([epochs[i - 1]], [h["test"][i - 1]])
            n_ok = int(np.sum(h["preds"][i] == h["showcase_y"]))
            hud.set_text(
                f"LR={state['lr']:.3g}   epoch {i}/{EPOCHS}\n"
                f"train={h['train'][i - 1]:.3f}   val={h['val'][i - 1]:.3f}\n"
                f"test={h['test'][i - 1]:.3f}   "
                f"acc={100 * h['test_acc_hist'][i - 1]:.1f}%\n"
                f"showcase (train) correct={n_ok}/3"
            )
            draw_showcase(h["preds"][i], f"epoch {i}/{EPOCHS}")
        refresh_legend(h, i)
        ymax = max(
            float(h["train"].max()), float(h["val"].max()), float(h["test"].max()),
        ) * 1.15
        ax_loss.set_xlim(0, EPOCHS)
        ax_loss.set_ylim(0.0, max(0.5, ymax))
        return line_tr, line_va, line_te, pt_tr, pt_va, pt_te, hud

    def on_lr(_=None):
        global SELECTED_LR
        idx = int(sl.val)
        state["lr"] = LR_CHOICES[idx]
        SELECTED_LR = state["lr"]
        sl.valtext.set_text(f"{state['lr']:.3g}")
        show_frame(0)
        anim.frame_seq = anim.new_frame_seq()
        if anim.event_source is not None:
            anim.event_source.start()
        btn.label.set_text("Stop")
        state["playing"] = True
        fig.canvas.draw_idle()

    sl.on_changed(on_lr)

    def update(frame):
        artists = show_frame(frame)
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
            if state["frame"] >= n_frames - 1:
                show_frame(0)
                anim.frame_seq = anim.new_frame_seq()
            if anim.event_source is not None:
                anim.event_source.start()
            btn.label.set_text("Stop")
            state["playing"] = True
        fig.canvas.draw_idle()

    btn.on_clicked(on_toggle)
    fig._lecture_anim = anim
    fig._retained_widgets = [sl, btn]
    show_frame(0)
    return fig


# -------------------------------------------------------
# Step 9: Inference
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle("Step 9: Code -- inference", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    blank_slide(ax)
    code = """\
model.eval()
names = ["circle", "zigzag", "drift"]

def predict(seq):
    # seq: (T, 2) or (B, T, 2)
    x = seq.reshape(-1, seq.shape[-2], 2)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
    return pred, prob

x_new = torch.from_numpy(X_te[0:1])   # one test path
pred, prob = predict(x_new)
print(names[int(pred)], prob.numpy().round(3))"""
    code_panel(ax, 0.35, 0.4, 11.3, 7.3, code, fontsize=FS_TICK - 1)
    return fig


# -------------------------------------------------------
# Step 10: Testing
# -------------------------------------------------------
def build_step_10():
    w_in, h_in = screen_figsize()
    left_extra, right_extra = 4.0 / 2.54, 3.0 / 2.54
    fig, axes = plt.subplots(
        1, 2, figsize=(w_in + left_extra + right_extra, h_in), dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [w_in / 2 + left_extra, w_in / 2 + right_extra]},
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
        (5.8, C_NEW, C_LOOP, "Train loss", "fits the RNN"),
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
        "Animation shows train & val curves + test line.",
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
    x0, y0 = 2.0, 1.4
    cell = 2.2
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
    fig.suptitle("Step 12: Takeaway", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank_slide(ax)
    points = [
        (6.0, "Data", "short noisy paths with confusable styles"),
        (4.7, "Model + loss", "GRU memory h_T + Softmax/CE"),
        (3.4, "Pipeline", "train fit · val monitor · test once"),
        (2.1, "LR story", "same data/model; step size changes the path"),
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
        "RNN training story = sequences -> memory -> CE -> train/val/test -> LR path.",
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
print("RNN SEQUENCE TRAIN EXAMPLE -- Walk styles")
print("=" * 60)
print(
    f"""
Data: {len(y_all)} sequences, T={T}, classes={CLASS_NAMES}
Split: train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}
Model: GRU(2->16) + Linear(16->3) + cross-entropy
Interactive Step 8: LR slider, Stop/Resume, train/val curves, test line,
                    showcase path predictions per epoch.

Sample test CE @ LR={DEMO_LR:g}: {LR_HIST[DEMO_LR]['test_loss']:.3f}
"""
)
print("=" * 60)
