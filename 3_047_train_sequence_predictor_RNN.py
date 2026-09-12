"""
3_047_train_sequence_predictor_RNN.py

Train an RNN to predict the *next value* in a sequence
(same teaching spirit as 3_045 / 3_046: data -> model -> loss ->
pipeline -> train -> interactive LR story -> infer -> test).

Task:
  Given x_1..x_t in R^2, predict x_{t+1}.
  Synthetic curved trajectories with noise.

Sketch:
  1. Task overview
  2. Code -- data
  3. Visual -- sequences + next-step targets
  4. Code -- task / model
  5. Code -- loss (MSE)
  6. Code -- data pipeline
  7. Code -- training
  8. Interactive -- LR slider + train/val/test loss + prediction overlay
  9. Code -- inference
  10. Code -- testing
  11. Test prediction scatter (true vs pred)
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

torch.manual_seed(47)
np.random.seed(47)


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
C_TRUE = "#1f77b4"
C_PRED = "#d62728"


# -------------------------------------------------------
# Synthetic trajectories: next-step supervision
# -------------------------------------------------------
T = 24          # full length; we predict steps 2..T from 1..T-1
N_SEQ = 240
NOISE = 0.06


def make_orbit(rng):
    """Noisy elliptical orbit -- smooth, predictable next step."""
    t = np.linspace(0, 2 * np.pi, T, endpoint=False)
    a = 0.9 + 0.2 * rng.randn()
    b = 0.55 + 0.15 * rng.randn()
    phase = rng.uniform(0, 2 * np.pi)
    rot = rng.uniform(0, 2 * np.pi)
    xy = np.stack([a * np.cos(t + phase), b * np.sin(t + phase)], axis=1)
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = xy.astype(np.float32) @ R.T
    return xy + NOISE * rng.randn(T, 2).astype(np.float32)


def make_wave(rng):
    """Horizontal progress + vertical oscillation."""
    t = np.linspace(-1.2, 1.2, T)
    amp = 0.45 + 0.15 * abs(rng.randn())
    freq = 1.5 + 0.5 * rng.randn()
    phase = rng.uniform(0, 2 * np.pi)
    xy = np.stack([t, amp * np.sin(freq * np.pi * t + phase)], axis=1)
    return xy.astype(np.float32) + NOISE * rng.randn(T, 2).astype(np.float32)


def make_spiral(rng):
    """Slow outward spiral."""
    t = np.linspace(0, 2.2 * np.pi, T)
    r = np.linspace(0.25, 1.0, T) * (0.9 + 0.2 * rng.randn())
    phase = rng.uniform(0, 2 * np.pi)
    xy = np.stack([r * np.cos(t + phase), r * np.sin(t + phase)], axis=1)
    return xy.astype(np.float32) + NOISE * rng.randn(T, 2).astype(np.float32)


MAKERS = [make_orbit, make_wave, make_spiral]
seqs = []
for i in range(N_SEQ):
    maker = MAKERS[i % len(MAKERS)]
    rng = np.random.RandomState(2000 + i)
    seqs.append(maker(rng))
X_all = np.stack(seqs, axis=0)  # (N, T, 2)

# Next-step pairs: context = x_1..x_{T-1}, target = x_2..x_T
X_ctx_all = X_all[:, :-1, :]
Y_nxt_all = X_all[:, 1:, :]

perm = np.random.permutation(len(X_all))
X_ctx_all, Y_nxt_all, X_all = X_ctx_all[perm], Y_nxt_all[perm], X_all[perm]
n = len(X_all)
n_train, n_val = int(0.60 * n), int(0.20 * n)

def split3(arr):
    return (
        arr[:n_train],
        arr[n_train:n_train + n_val],
        arr[n_train + n_val:],
    )

X_tr, X_va, X_te = split3(X_all)
C_tr, C_va, C_te = split3(X_ctx_all)
Y_tr, Y_va, Y_te = split3(Y_nxt_all)

C_tr_t, Y_tr_t = torch.from_numpy(C_tr), torch.from_numpy(Y_tr)
C_va_t, Y_va_t = torch.from_numpy(C_va), torch.from_numpy(Y_va)
C_te_t, Y_te_t = torch.from_numpy(C_te), torch.from_numpy(Y_te)


class NextStepRNN(nn.Module):
    """Many-to-many: each step predicts the next 2D point."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.rnn = nn.GRU(input_size=2, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        # x: (B, T-1, 2) -> pred next at each time (B, T-1, 2)
        h_seq, _ = self.rnn(x)
        return self.fc(h_seq)


def dramatic_init_(model: NextStepRNN) -> None:
    """Bad head: predictions start far from the true next points."""
    with torch.no_grad():
        model.fc.weight.normal_(0.0, 1.2)
        model.fc.bias.copy_(torch.tensor([1.5, -1.2]))


def batch_mse(model, loader, opt=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    for xb, yb in loader:
        pred = model(xb)
        loss = F.mse_loss(pred, yb)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total += loss.item() * len(yb)
        n += len(yb)
    return total / max(n, 1)


def mean_l2(model, C_t, Y_t):
    model.eval()
    with torch.no_grad():
        pred = model(C_t)
        err = torch.linalg.vector_norm(pred - Y_t, dim=-1)  # (B, T-1)
        return float(err.mean())


def train_history(lr: float, epochs: int = 40, batch_size: int = 32, seed: int = 0):
    torch.manual_seed(seed)
    model = NextStepRNN()
    dramatic_init_(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(C_tr_t, Y_tr_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(C_va_t, Y_va_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(C_te_t, Y_te_t), batch_size=batch_size, shuffle=False)

    # Fixed train showcase sequence for the left interactive panel
    showcase_i = 0
    showcase_full = X_tr[showcase_i]          # (T, 2)
    showcase_ctx = C_tr_t[showcase_i:showcase_i + 1]
    showcase_tgt = Y_tr[showcase_i]           # (T-1, 2)

    tr_hist, va_hist, te_hist, pred_hist = [], [], [], []
    with torch.no_grad():
        pred_hist.append(model(showcase_ctx)[0].numpy().copy())

    for _ in range(epochs):
        tr = batch_mse(model, train_loader, opt)
        with torch.no_grad():
            va = batch_mse(model, val_loader, opt=None)
            te = batch_mse(model, test_loader, opt=None)
            pred_hist.append(model(showcase_ctx)[0].numpy().copy())
        tr_hist.append(tr)
        va_hist.append(va)
        te_hist.append(te)

    te_l2 = mean_l2(model, C_te_t, Y_te_t)

    return {
        "train": np.asarray(tr_hist, dtype=float),
        "val": np.asarray(va_hist, dtype=float),
        "test": np.asarray(te_hist, dtype=float),
        "test_loss": float(te_hist[-1]),
        "test_l2": float(te_l2),
        "model": model,
        "showcase_full": showcase_full,
        "showcase_tgt": showcase_tgt,
        "preds": pred_hist,  # len epochs+1, each (T-1, 2)
    }


EPOCHS = 40
LR_CHOICES = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1]
DEMO_LR = 0.03  # Step 8 default
SELECTED_LR = DEMO_LR  # updated live by Step 8; later slides read this
LR_HIST = {lr: train_history(lr, epochs=EPOCHS, seed=7) for lr in LR_CHOICES}


# -------------------------------------------------------
# Step 1
# -------------------------------------------------------
def build_step_1():
    w_in, h_in = screen_figsize()
    extra = 3.0 / 2.54
    fig, axes = plt.subplots(
        1, 2, figsize=(w_in + 2 * extra, h_in), dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [w_in / 2 + extra, w_in / 2 + extra]},
    )
    fig.suptitle(
        "Step 1: RNN task -- predict the next value",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("What we will build", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    steps = [
        (6.4, "1. Data", "noisy 2D trajectories x_1..x_T"),
        (5.2, "2. Task", "from x_1..x_t predict x_{t+1}"),
        (4.0, "3. Loss", "MSE on next-step vectors"),
        (2.8, "4. Pipeline", "train / val / test loaders"),
        (1.6, "5. Train -> Infer -> Test", "GRU memory + LR story"),
    ]
    box_w, box_x = 11.2, 0.4
    left_w_in = max(w_in / 2 + extra, 1.0)
    dx_body_1_5cm = (1.5 / 2.54) / left_w_in * 12.0  # blank_slide x: 0..12
    title_x = box_x + 0.45
    body_x = box_x + box_w * 0.40 + dx_body_1_5cm
    for y, title, body in steps:
        round_box(ax, box_x, y - 0.45, box_w, 1.05, C_NEW, C_LOOP, lw=1.7)
        ax.text(title_x, y + 0.05, title, ha="left", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(body_x, y + 0.05, body, ha="left", va="center",
                fontsize=FS_ANNOT, color="#333333")

    ax = axes[1]
    ax.set_title("A path is a sequence of next-step targets", fontsize=FS_AX_TITLE - 3,
                 fontweight="bold")
    seq = X_tr[0]
    ax.plot(seq[:, 0], seq[:, 1], "-", color=C_TRUE, lw=2.2, label="trajectory")
    ax.scatter(seq[:-1, 0], seq[:-1, 1], c=C_TRUE, s=28, zorder=3, label="context x_t")
    ax.scatter(seq[1:, 0], seq[1:, 1], c=C_PRED, s=36, marker="x", zorder=4,
               label="target x_{t+1}")
    # a few arrows context -> next
    for i in (3, 8, 14):
        ax.annotate(
            "", xy=seq[i + 1], xytext=seq[i],
            arrowprops=dict(arrowstyle="->", color="0.35", lw=1.4),
        )
    ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
    ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 2: data code
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle("Step 2: Code -- sequence data", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    blank_slide(ax)
    code = """\
# Full path: X[n] has shape (T, 2)
# Next-step supervision (teacher forcing style):
X_ctx = X[:, :-1, :]   # (N, T-1, 2)  inputs  x_1..x_{T-1}
Y_nxt = X[:,  1:, :]   # (N, T-1, 2)  targets x_2..x_T

# Synthetic families: orbit / wave / spiral + small noise
# so the next point is learnable but not trivial."""
    code_panel(ax, 0.35, 0.8, 11.3, 6.6, code, fontsize=FS_TICK)
    return fig


# -------------------------------------------------------
# Step 3: visual
# -------------------------------------------------------
def build_step_3():
    fig, axes = new_figure(1, 3)
    fig.suptitle(
        "Step 3: Data -- train / val / test paths",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    packs = [
        (axes[0], X_tr, "Train", C_LOOP),
        (axes[1], X_va, "Validation", C_VAL),
        (axes[2], X_te, "Test", C_TEST),
    ]
    for ax, X, title, edge in packs:
        ax.set_title(title, fontsize=FS_AX_TITLE - 2, fontweight="bold", color=edge)
        for i in range(min(6, len(X))):
            ax.plot(X[i, :, 0], X[i, :, 1], "-", lw=1.4, alpha=0.85)
        ax.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
        ax.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 4: model
# -------------------------------------------------------
def build_step_4():
    w_in, h_in = screen_figsize()
    left_extra, right_extra = 4.0 / 2.54, 6.0 / 2.54
    fig, axes = plt.subplots(
        1, 2, figsize=(w_in + left_extra + right_extra, h_in), dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [w_in / 2 + left_extra, w_in / 2 + right_extra]},
    )
    fig.suptitle(
        "Step 4: Code -- task definition (next-step RNN)",
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
        r"$\hat{x}_{t+1} = W h_t + b$",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.7, 11.2, 2.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.85,
        "At every time, memory h_t summarizes the past.\n"
        "A linear head emits the next 2D point.\n"
        "Same train/val/test moral as 3_045 -- different output type.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.4, 0.45, 11.2, 1.9, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 1.4,
        "Regression head (R^2) instead of Softmax classes.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("Code", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    code = """\
class NextStepRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(2, 32, batch_first=True)
        self.fc  = nn.Linear(32, 2)

    def forward(self, x):       # x: (B, T-1, 2)
        h, _ = self.rnn(x)      # h: (B, T-1, 32)
        return self.fc(h)       # (B, T-1, 2) = xhat_{2..T}

model = NextStepRNN()"""
    code_panel(ax, 0.35, 0.55, 11.3, 7.1, code, fontsize=FS_TICK)
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 5: loss
# -------------------------------------------------------
def build_step_5():
    w_in, h_in = screen_figsize()
    extra = 2.0 / 2.54
    fig, axes = plt.subplots(
        1, 2, figsize=(w_in + 2 * extra, h_in), dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [w_in / 2 + extra, w_in / 2 + extra]},
    )
    fig.suptitle(
        "Step 5: Code -- loss function (MSE)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Mean squared error on next points", fontsize=FS_AX_TITLE - 3,
                 fontweight="bold")
    round_box(ax, 0.4, 5.3, 11.2, 2.3, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.45,
        r"$\mathcal{L} = \mathrm{mean}\,\| \hat{x}_{t+1} - x_{t+1} \|_2^2$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 3,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.5, 11.2, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.7,
        "Natural loss for continuous next-value prediction.\n"
        "Average over batch and over time steps.\n"
        "Torch: F.mse_loss(pred, target)",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.4, 0.4, 11.2, 1.7, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 1.25,
        "Unlike 3_045/3_046: no Softmax, no class CE.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("Code", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    code = """\
pred = model(x_ctx)                 # (B, T-1, 2)
loss = F.mse_loss(pred, y_next)     # vs true x_2..x_T
loss.backward()
opt.step()"""
    code_panel(ax, 0.35, 1.8, 11.3, 5.0, code, fontsize=FS_TICK + 1)
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 6: pipeline
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle("Step 6: Code -- data pipeline", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    blank_slide(ax)
    code = """\
from torch.utils.data import DataLoader, TensorDataset

# Context / next-target tensors already shaped (N, T-1, 2)
train_loader = DataLoader(TensorDataset(C_tr, Y_tr),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(C_va, Y_va),
                          batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(C_te, Y_te),
                          batch_size=32, shuffle=False)
# Train: fit  |  Val: monitor LR  |  Test: report once"""
    code_panel(ax, 0.35, 0.55, 11.3, 7.1, code, fontsize=FS_TICK - 1)
    return fig


# -------------------------------------------------------
# Step 7: training
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle("Step 7: Code -- training", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    blank_slide(ax)
    code = """\
model = NextStepRNN()
opt = torch.optim.Adam(model.parameters(), lr=lr)

def run_epoch(loader, train=True):
    model.train(train)
    total, n = 0.0, 0
    for xb, yb in loader:                 # xb context, yb next
        loss = F.mse_loss(model(xb), yb)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(yb); n += len(yb)
    return total / n

for epoch in range(epochs):
    tr = run_epoch(train_loader, train=True)
    with torch.no_grad():
        va = run_epoch(val_loader, train=False)
    # log tr, va -- test MSE only at the end"""
    code_panel(ax, 0.35, 0.35, 11.3, 7.4, code, fontsize=FS_TICK - 1)
    return fig


# -------------------------------------------------------
# Step 8: interactive
# -------------------------------------------------------
def build_step_8():
    global SELECTED_LR
    fig, axes = new_figure(1, 2)
    fig.subplots_adjust(bottom=0.20, wspace=0.28)
    fig.suptitle(
        "Step 8: Watch next-step error fall -- choose learning rate",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    lr0 = DEMO_LR
    SELECTED_LR = lr0
    hist0 = LR_HIST[lr0]
    epochs = np.arange(1, EPOCHS + 1)
    n_frames = EPOCHS + 1

    ax_seq = axes[0]
    ax_seq.set_title("Train showcase (NOT test)", fontsize=FS_AX_TITLE - 3, fontweight="bold")

    ax_loss = axes[1]
    ax_loss.set_title("MSE vs epoch", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    (line_tr,) = ax_loss.plot([], [], color=C_LOOP, lw=2.6, label="train")
    (line_va,) = ax_loss.plot([], [], color=C_VAL, lw=2.6, label="val")
    (line_te,) = ax_loss.plot([], [], color=C_TEST, lw=2.4, ls="--", label="test")
    (pt_tr,) = ax_loss.plot([], [], "o", color=C_LOOP, ms=8)
    (pt_va,) = ax_loss.plot([], [], "o", color=C_VAL, ms=8)
    (pt_te,) = ax_loss.plot([], [], "o", color=C_TEST, ms=8)
    ax_loss.set_xlabel("epoch", fontsize=FS_AXIS)
    ax_loss.set_ylabel("MSE", fontsize=FS_AXIS)
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

    def draw_showcase(pred_next, epoch_label: str):
        h = current_hist()
        full = h["showcase_full"]
        tgt = h["showcase_tgt"]
        ax_seq.cla()
        l2 = float(np.linalg.norm(pred_next - tgt, axis=-1).mean())
        ax_seq.set_title(
            f"Train showcase -- not test  ({epoch_label}; mean L2={l2:.3f})",
            fontsize=FS_AX_TITLE - 3, fontweight="bold",
        )
        ctx = full[:-1]  # all context points x_1..x_{T-1}
        # Path through every context point
        ax_seq.plot(ctx[:, 0], ctx[:, 1], "-", color=C_TRUE, lw=2.0,
                    label="context path", zorder=2)
        ax_seq.scatter(ctx[:, 0], ctx[:, 1], c=C_TRUE, s=28, zorder=3,
                       edgecolors="k", linewidths=0.4, label="context x_t")
        ax_seq.scatter(tgt[:, 0], tgt[:, 1], c="#2ca02c", s=30, marker="o",
                       zorder=3, edgecolors="k", linewidths=0.35, label="true next")
        ax_seq.scatter(pred_next[:, 0], pred_next[:, 1], c=C_PRED, s=36, marker="x",
                       label="pred next", zorder=4)
        # Path segment from every context point to its predicted next value
        for i in range(len(ctx)):
            ax_seq.annotate(
                "",
                xy=(pred_next[i, 0], pred_next[i, 1]),
                xytext=(ctx[i, 0], ctx[i, 1]),
                arrowprops=dict(arrowstyle="->", color=C_PRED, lw=1.0, alpha=0.75),
                zorder=2,
            )
        ax_seq.set_xlabel(r"$x_1$", fontsize=FS_AXIS)
        ax_seq.set_ylabel(r"$x_2$", fontsize=FS_AXIS)
        ax_seq.tick_params(labelsize=FS_TICK)
        ax_seq.grid(True, alpha=0.25)
        ax_seq.set_aspect("equal", adjustable="box")
        ax_seq.legend(fontsize=FS_LEGEND - 1, loc="best")
        ax_seq.text(
            0.02, 0.02,
            "Left = one TRAIN path for intuition.\n"
            "Right test curve is MSE on TEST after each epoch.",
            transform=ax_seq.transAxes, ha="left", va="bottom", fontsize=FS_TICK - 1,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
        )

    def refresh_legend(h, i):
        if i <= 0:
            line_tr.set_label("train")
            line_va.set_label("val")
            line_te.set_label("test")
        else:
            line_tr.set_label(f"train = {h['train'][i - 1]:.4f}")
            line_va.set_label(f"val = {h['val'][i - 1]:.4f}")
            line_te.set_label(f"test = {h['test'][i - 1]:.4f}")
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
                f"final TEST MSE={h['test_loss']:.4f}   "
                f"mean L2={h['test_l2']:.3f}"
            )
            draw_showcase(h["preds"][0], "wild init")
        else:
            line_tr.set_data(epochs[:i], h["train"][:i])
            line_va.set_data(epochs[:i], h["val"][:i])
            line_te.set_data(epochs[:i], h["test"][:i])
            pt_tr.set_data([epochs[i - 1]], [h["train"][i - 1]])
            pt_va.set_data([epochs[i - 1]], [h["val"][i - 1]])
            pt_te.set_data([epochs[i - 1]], [h["test"][i - 1]])
            hud.set_text(
                f"LR={state['lr']:.3g}   epoch {i}/{EPOCHS}\n"
                f"train={h['train'][i - 1]:.4f}   val={h['val'][i - 1]:.4f}\n"
                f"test={h['test'][i - 1]:.4f}   "
                f"final={h['test_loss']:.4f}   L2={h['test_l2']:.3f}"
            )
            draw_showcase(h["preds"][i], f"epoch {i}/{EPOCHS}")
        refresh_legend(h, i)
        ymax = max(
            float(h["train"].max()), float(h["val"].max()), float(h["test"].max()),
        ) * 1.15
        ax_loss.set_xlim(0, EPOCHS)
        ax_loss.set_ylim(0.0, max(0.05, ymax))
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
# Step 9: inference
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle("Step 9: Code -- inference", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    blank_slide(ax)
    code = """\
model.eval()

def predict_next(context):
    # context: (Tctx, 2) or (B, Tctx, 2)
    x = context.reshape(-1, context.shape[-2], 2)
    with torch.no_grad():
        pred = model(x)            # (B, Tctx, 2)
    return pred

# One-step ahead along a test path:
ctx = torch.from_numpy(C_te[0:1])  # (1, T-1, 2)
yhat = predict_next(ctx)[0]        # predicted x_2..x_T
ytrue = torch.from_numpy(Y_te[0])
print((yhat - ytrue).pow(2).mean())"""
    code_panel(ax, 0.35, 0.35, 11.3, 7.4, code, fontsize=FS_TICK - 1)
    return fig


# -------------------------------------------------------
# Step 10: testing
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
total, n = 0.0, 0
with torch.no_grad():
    for xb, yb in test_loader:
        loss = F.mse_loss(model(xb), yb)
        total += loss.item() * len(yb)
        n += len(yb)
test_mse = total / n
print(f"test MSE={test_mse:.4f}")
# Do not tune LR / epochs on this number."""
    code_panel(ax, 0.25, 0.55, 11.5, 6.8, code, fontsize=FS_TICK - 1)

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("Roles reminder", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    cards = [
        (5.8, C_NEW, C_LOOP, "Train MSE", "fits the RNN"),
        (3.9, "#f5eef8", C_VAL, "Val MSE", "monitor / choose LR"),
        (2.0, C_OLD, C_TEST, "Test MSE", "final honest report"),
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
# Step 11: test true-vs-pred scatter (regression analogue of confusion)
# -------------------------------------------------------
def build_step_11():
    fig, axes = new_figure(1, 2)
    # Use whichever LR was last selected on Step 8.
    lr = SELECTED_LR if SELECTED_LR in LR_HIST else DEMO_LR
    hist = LR_HIST[lr]
    model = hist["model"]
    model.eval()
    with torch.no_grad():
        pred = model(C_te_t).numpy().reshape(-1, 2)
        true = Y_te.reshape(-1, 2)
    err = np.linalg.norm(pred - true, axis=1)
    test_loader = DataLoader(TensorDataset(C_te_t, Y_te_t), batch_size=64, shuffle=False)
    test_mse = batch_mse(model, test_loader, opt=None)
    test_l2 = mean_l2(model, C_te_t, Y_te_t)

    fig.suptitle(
        f"Step 11: Test results from Step 8 model  (LR={lr:g})",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    ax.set_title("True vs predicted (dim 1)", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.scatter(true[:, 0], pred[:, 0], s=10, alpha=0.35, c=C_LOOP)
    lim = np.array([true[:, 0].min(), true[:, 0].max()])
    ax.plot(lim, lim, "--", color=C_ACCENT, lw=1.6, label="perfect")
    ax.set_xlabel("true x1", fontsize=FS_AXIS)
    ax.set_ylabel("pred x1", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc="best")

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("TEST summary", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.5, 5.3, 11.0, 2.2, C_NEW, C_LOOP, lw=1.8)
    ax.text(
        6.0, 6.4,
        f"test MSE = {test_mse:.4f}\n"
        f"mean L2 error = {test_l2:.3f}\n"
        f"median L2 error = {np.median(err):.3f}",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.5, 2.8, 11.0, 2.0, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 3.8,
        "Regression twin of a confusion table:\n"
        "points on the diagonal = good next-step forecasts.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    round_box(ax, 0.5, 0.5, 11.0, 1.9, C_OLD, "#d35400", lw=1.6)
    ax.text(
        6.0, 1.45,
        f"Uses the LR last chosen on Step 8 (now LR={lr:g}).\n"
        "Built only on TEST -- do not tune LR on this plot.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 12: takeaway
# -------------------------------------------------------
def build_step_12():
    fig, ax = new_figure()
    fig.suptitle("Step 12: Takeaway", fontsize=FS_FIGURE_TITLE, fontweight="bold")
    blank_slide(ax)
    points = [
        (6.0, "Data", "paths become next-step (context, target) pairs"),
        (4.7, "Model + loss", "GRU memory + linear head + MSE"),
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
        "Next-step RNN story = sequence memory -> continuous forecast -> train/val/test.",
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
print("RNN NEXT-STEP PREDICTOR -- sequence regression")
print("=" * 60)
print(
    f"""
Data: {len(X_all)} paths, T={T}, next-step pairs shape (N, {T-1}, 2)
Split: train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}
Model: GRU(2->32) + Linear(32->2) + MSE
Interactive Step 8: LR slider, Stop/Resume, train/val curves, test line,
                    showcase true-vs-pred next points.

Sample test MSE @ LR={DEMO_LR:g}: {LR_HIST[DEMO_LR]['test_loss']:.4f}
"""
)
print("=" * 60)
