"""
3_044_train_regression_MLP.py

Train an MLP for 1D regression (same teaching spirit as 3_045):
data -> model -> loss -> pipeline -> train -> interactive LR story
-> infer -> test.

Task:
  Learn y ≈ f(x) from noisy samples of a curved target function.
  Model: small MLP  (x in R -> y in R)
  Loss:  MSE

Sketch:
  1. Task overview
  2. Code -- data
  3. Visual -- train / val / test clouds
  4. Code -- task / model (MLP)
  5. Code -- loss (MSE)
  6. Code -- data pipeline
  7. Code -- training
  8. Interactive -- LR slider + fitted-curve animation + train/val/test MSE
  9. Code -- inference
  10. Code -- testing
  11. Test true-vs-pred scatter
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

torch.manual_seed(44)
np.random.seed(44)


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
C_FIT = "#d62728"


# -------------------------------------------------------
# Synthetic 1D regression data
# -------------------------------------------------------
def f_true(x):
    """Curved target -- needs a nonlinear MLP, not a single line."""
    return np.sin(1.7 * np.pi * x) + 0.35 * np.cos(3.3 * np.pi * x)


N = 240
NOISE = 0.18
x_all = np.random.uniform(-1.0, 1.0, size=N).astype(np.float32)
y_all = (f_true(x_all) + NOISE * np.random.randn(N)).astype(np.float32)

perm = np.random.permutation(N)
x_all, y_all = x_all[perm], y_all[perm]
n_train, n_val = int(0.60 * N), int(0.20 * N)
x_tr, y_tr = x_all[:n_train], y_all[:n_train]
x_va, y_va = x_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
x_te, y_te = x_all[n_train + n_val:], y_all[n_train + n_val:]

X_tr_t = torch.from_numpy(x_tr).unsqueeze(1)
y_tr_t = torch.from_numpy(y_tr).unsqueeze(1)
X_va_t = torch.from_numpy(x_va).unsqueeze(1)
y_va_t = torch.from_numpy(y_va).unsqueeze(1)
X_te_t = torch.from_numpy(x_te).unsqueeze(1)
y_te_t = torch.from_numpy(y_te).unsqueeze(1)

# Dense grid for plotting the fitted curve
X_GRID = torch.linspace(-1.05, 1.05, 200).unsqueeze(1)
x_grid_np = X_GRID.squeeze(1).numpy()
y_grid_true = f_true(x_grid_np)


class MLPRegressor(nn.Module):
    """Deeper MLP: R -> R (more capacity for the curved target)."""

    def __init__(self, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x)


def dramatic_init_(model: MLPRegressor) -> None:
    """Mildly wrong start -- visible at epoch 0, still easy to unlearn."""
    with torch.no_grad():
        # Keep default-ish hidden weights; only nudge the readout.
        last = model.net[-1]
        last.weight.mul_(0.3)
        last.weight.add_(1.2)
        last.bias.fill_(-0.8)


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


def predict_curve(model):
    model.eval()
    with torch.no_grad():
        return model(X_GRID).squeeze(1).numpy().copy()


def train_history(lr: float, epochs: int = 50, batch_size: int = 32, seed: int = 0):
    torch.manual_seed(seed)
    model = MLPRegressor()
    dramatic_init_(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va_t, y_va_t), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size, shuffle=False)

    tr_hist, va_hist, te_hist, curves = [], [], [], []
    curves.append(predict_curve(model))
    for _ in range(epochs):
        tr = batch_mse(model, train_loader, opt)
        with torch.no_grad():
            va = batch_mse(model, val_loader, opt=None)
            te = batch_mse(model, test_loader, opt=None)
        tr_hist.append(tr)
        va_hist.append(va)
        te_hist.append(te)
        curves.append(predict_curve(model))
    with torch.no_grad():
        te_mae = float(torch.mean(torch.abs(model(X_te_t) - y_te_t)))

    return {
        "train": np.asarray(tr_hist, dtype=float),
        "val": np.asarray(va_hist, dtype=float),
        "test": np.asarray(te_hist, dtype=float),
        "test_loss": float(te_hist[-1]),
        "test_mae": te_mae,
        "model": model,
        "curves": curves,  # len epochs+1
    }


EPOCHS = 100
LR_CHOICES = [0.0005, 0.001, 0.003, 0.01, 0.02, 0.03]
DEMO_LR = 0.01  # Step 8 default
# Updated live by Step 8's LR slider; later slides read this.
SELECTED_LR = DEMO_LR
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
        "Step 1: MLP regression -- learn a curved f(x)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("What we will build", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    steps = [
        (6.4, "1. Data", "noisy samples (x, y) from a curve"),
        (5.2, "2. Task", "predict continuous y from x"),
        (4.0, "3. Loss", "MSE between yhat and y"),
        (2.8, "4. Pipeline", "train / val / test loaders"),
        (1.6, "5. Train -> Infer -> Test", "MLP fit + LR story"),
    ]
    box_w, box_x = 11.2, 0.4
    title_x, body_x = box_x + 0.45, box_x + box_w * 0.40
    for y, title, body in steps:
        round_box(ax, box_x, y - 0.45, box_w, 1.05, C_NEW, C_LOOP, lw=1.7)
        ax.text(title_x, y + 0.05, title, ha="left", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(body_x, y + 0.05, body, ha="left", va="center",
                fontsize=FS_ANNOT, color="#333333")

    ax = axes[1]
    ax.set_title("Target curve + noisy observations", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.plot(x_grid_np, y_grid_true, "-", color=C_TRUE, lw=2.6, label="true f(x)")
    ax.scatter(x_tr, y_tr, s=18, c=C_LOOP, alpha=0.75, label="train samples")
    ax.set_xlabel("x", fontsize=FS_AXIS)
    ax.set_ylabel("y", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc="best")
    ax.text(
        0.02, 0.02,
        "Nonlinear f => a line is not enough; use an MLP.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )
    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 2: data code
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle("Step 2: Code -- data", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    blank_slide(ax)
    code = """\
# Unknown curved target + observation noise
def f_true(x):
    return sin(1.7*pi*x) + 0.35*cos(3.3*pi*x)

x = torch.empty(N).uniform_(-1, 1)
y = f_true(x) + 0.18 * torch.randn(N)

# Later tensors are shaped (N, 1) for nn.Linear
X = x.unsqueeze(1)
Y = y.unsqueeze(1)"""
    code_panel(ax, 0.35, 0.7, 11.3, 6.8, code, fontsize=FS_TICK)
    return fig


# -------------------------------------------------------
# Step 3: visual splits
# -------------------------------------------------------
def build_step_3():
    fig, axes = new_figure(1, 3)
    fig.suptitle(
        "Step 3: Data -- train / val / test",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    packs = [
        (axes[0], x_tr, y_tr, "Train (60%)", C_LOOP),
        (axes[1], x_va, y_va, "Validation (20%)", C_VAL),
        (axes[2], x_te, y_te, "Test (20%)", C_TEST),
    ]
    for ax, x, y, title, color in packs:
        ax.set_title(title, fontsize=FS_AX_TITLE - 2, fontweight="bold", color=color)
        ax.plot(x_grid_np, y_grid_true, "-", color="0.7", lw=1.6, label="true f")
        ax.scatter(x, y, s=20, c=color, alpha=0.8, edgecolors="k", linewidths=0.3)
        ax.set_xlabel("x", fontsize=FS_AXIS)
        ax.set_ylabel("y", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FS_LEGEND - 1, loc="best")
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
        "Step 4: Code -- task definition (MLP)",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    blank_slide(ax)
    ax.set_title("Idea", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.4, 11.2, 2.2, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.5,
        r"$\hat{y} = W_5\,\phi(W_4\,\phi(W_3\,\phi(W_2\,\phi(W_1 x))))$"
        + "\n"
        r"$\phi = \tanh$  (4 hidden layers)",
        ha="center", va="center", fontsize=FS_ANNOT - 1,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.7, 11.2, 2.3, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.85,
        "Deeper tanh stack buys more nonlinear capacity.\n"
        "Output is a single continuous number (no Softmax).\n"
        "Same train/val/test moral as 3_045 -- different output type.",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.4, 0.45, 11.2, 1.9, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 1.4,
        "Regression twin of the multi-class MLP/linear story.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("Code", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    code = """\
class MLPRegressor(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):     # x: (B, 1)
        return self.net(x)    # (B, 1)

model = MLPRegressor()"""
    code_panel(ax, 0.35, 0.45, 11.3, 7.2, code, fontsize=FS_TICK)
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
    ax.set_title("Mean squared error", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    round_box(ax, 0.4, 5.3, 11.2, 2.3, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.45,
        r"$\mathcal{L} = \mathrm{mean}\,(\hat{y} - y)^2$",
        ha="center", va="center", fontsize=FS_AX_TITLE - 2,
        fontweight="bold", color="#f9e79f",
    )
    round_box(ax, 0.4, 2.5, 11.2, 2.4, C_NEW, C_LOOP, lw=2.0)
    ax.text(
        6.0, 3.7,
        "Natural loss for continuous targets.\n"
        "Large residuals are punished quadratically.\n"
        "Torch: F.mse_loss(yhat, y)",
        ha="center", va="center", fontsize=FS_ANNOT, color=C_DARK,
    )
    round_box(ax, 0.4, 0.4, 11.2, 1.7, C_MUTE, "0.45", lw=1.5)
    ax.text(
        6.0, 1.25,
        "Unlike 3_045: no Softmax / cross-entropy.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )

    ax = axes[1]
    blank_slide(ax)
    ax.set_title("Code", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    code = """\
yhat = model(x)                # (B, 1)
loss = F.mse_loss(yhat, y)
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

train_loader = DataLoader(TensorDataset(X_tr, Y_tr),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_va, Y_va),
                          batch_size=32, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_te, Y_te),
                          batch_size=32, shuffle=False)
# Train: fit  |  Val: monitor LR  |  Test: report once"""
    code_panel(ax, 0.35, 0.7, 11.3, 6.8, code, fontsize=FS_TICK - 1)
    return fig


# -------------------------------------------------------
# Step 7: training
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle("Step 7: Code -- training", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")
    blank_slide(ax)
    code = """\
model = MLPRegressor()
opt = torch.optim.Adam(model.parameters(), lr=lr)

def run_epoch(loader, train=True):
    model.train(train)
    total, n = 0.0, 0
    for xb, yb in loader:
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
        "Step 8: Watch the fit improve -- choose learning rate",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    lr0 = DEMO_LR
    SELECTED_LR = lr0
    hist0 = LR_HIST[lr0]
    epochs = np.arange(1, EPOCHS + 1)
    n_frames = EPOCHS + 1

    ax_fit = axes[0]
    ax_fit.set_title("Train fit (NOT the test set)", fontsize=FS_AX_TITLE - 3, fontweight="bold")
    ax_fit.scatter(x_tr, y_tr, s=16, c=C_LOOP, alpha=0.55, label="train data", zorder=2)
    (line_true,) = ax_fit.plot(x_grid_np, y_grid_true, "--", color=C_TRUE, lw=2.0, label="true f")
    (line_fit,) = ax_fit.plot(x_grid_np, hist0["curves"][0], "-", color=C_FIT, lw=2.6, label="MLP fit")
    ax_fit.set_xlabel("x", fontsize=FS_AXIS)
    ax_fit.set_ylabel("y", fontsize=FS_AXIS)
    ax_fit.tick_params(labelsize=FS_TICK)
    ax_fit.grid(True, alpha=0.3)
    ax_fit.legend(fontsize=FS_LEGEND, loc="best")
    ax_fit.text(
        0.02, 0.02,
        "Left = TRAIN cloud + evolving MLP curve.\n"
        "Right test curve is MSE on TEST after each epoch.",
        transform=ax_fit.transAxes, ha="left", va="bottom", fontsize=FS_TICK - 1,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )

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

    def refresh_legend(h, i):
        # Rebuild legend text from the *current* LR / epoch (avoids stale labels).
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
        line_fit.set_ydata(h["curves"][i])
        if i == 0:
            line_tr.set_data([], [])
            line_va.set_data([], [])
            line_te.set_data([], [])
            pt_tr.set_data([], [])
            pt_va.set_data([], [])
            pt_te.set_data([], [])
            ax_fit.set_title("Train fit -- wild init (NOT test)", fontsize=FS_AX_TITLE - 3,
                             fontweight="bold")
            hud.set_text(
                f"LR={state['lr']:.3g}   epoch 0/{EPOCHS} (wild init)\n"
                f"left panel is TRAIN fit, not test\n"
                f"final TEST MSE={h['test_loss']:.4f}   "
                f"MAE={h['test_mae']:.3f}"
            )
        else:
            line_tr.set_data(epochs[:i], h["train"][:i])
            line_va.set_data(epochs[:i], h["val"][:i])
            line_te.set_data(epochs[:i], h["test"][:i])
            pt_tr.set_data([epochs[i - 1]], [h["train"][i - 1]])
            pt_va.set_data([epochs[i - 1]], [h["val"][i - 1]])
            pt_te.set_data([epochs[i - 1]], [h["test"][i - 1]])
            ax_fit.set_title(
                f"Train fit -- epoch {i}/{EPOCHS} (NOT test)",
                fontsize=FS_AX_TITLE - 3, fontweight="bold",
            )
            hud.set_text(
                f"LR={state['lr']:.3g}   epoch {i}/{EPOCHS}\n"
                f"train={h['train'][i - 1]:.4f}   val={h['val'][i - 1]:.4f}\n"
                f"test={h['test'][i - 1]:.4f}   "
                f"final test={h['test_loss']:.4f}   MAE={h['test_mae']:.3f}"
            )
        refresh_legend(h, i)
        ymax = max(
            float(h["train"].max()), float(h["val"].max()), float(h["test"].max()),
        ) * 1.15
        ax_loss.set_xlim(0, EPOCHS)
        ax_loss.set_ylim(0.0, max(0.05, ymax))
        return line_fit, line_tr, line_va, line_te, pt_tr, pt_va, pt_te, hud

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
        fig, update, frames=n_frames, interval=100, blit=False, repeat=False,
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

def predict(x_scalar_or_batch):
    x = torch.as_tensor(x_scalar_or_batch, dtype=torch.float32)
    x = x.reshape(-1, 1)
    with torch.no_grad():
        yhat = model(x)
    return yhat.squeeze(1)

print(predict([-0.5, 0.0, 0.7]))"""
    code_panel(ax, 0.35, 0.7, 11.3, 6.8, code, fontsize=FS_TICK)
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
        (5.8, C_NEW, C_LOOP, "Train MSE", "fits the MLP"),
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
# Step 11: test scatter
# -------------------------------------------------------
def build_step_11():
    fig, axes = new_figure(1, 2)
    # Use whichever LR was last selected on Step 8 (falls back to DEMO_LR).
    lr = SELECTED_LR if SELECTED_LR in LR_HIST else DEMO_LR
    hist = LR_HIST[lr]
    model = hist["model"]
    model.eval()
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=64, shuffle=False)
    test_mse = batch_mse(model, test_loader, opt=None)
    with torch.no_grad():
        pred_t = model(X_te_t)
        test_mae = float(torch.mean(torch.abs(pred_t - y_te_t)))
        pred = pred_t.squeeze(1).numpy()
        fit = model(X_GRID).squeeze(1).numpy()
    true = y_te

    fig.suptitle(
        f"Step 11: Test results from Step 8 model  (LR={lr:g})",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    ax.set_title("Test true vs pred", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.scatter(true, pred, s=28, alpha=0.75, c=C_LOOP, edgecolors="k", linewidths=0.3)
    lim = np.array([min(true.min(), pred.min()), max(true.max(), pred.max())])
    ax.plot(lim, lim, "--", color=C_ACCENT, lw=1.8, label="perfect")
    ax.set_xlabel("true y", fontsize=FS_AXIS)
    ax.set_ylabel("pred y", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc="best")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1]
    ax.set_title("Same model on the curve", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.scatter(x_te, y_te, s=22, c=C_TEST, alpha=0.75, label="test data", zorder=2,
               edgecolors="k", linewidths=0.3)
    ax.plot(x_grid_np, y_grid_true, "--", color=C_TRUE, lw=2.0, label="true f")
    ax.plot(x_grid_np, fit, "-", color=C_FIT, lw=2.4, label=f"MLP (LR={lr:g})")
    ax.set_xlabel("x", fontsize=FS_AXIS)
    ax.set_ylabel("y", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc="best")
    ax.text(
        0.02, 0.02,
        f"test MSE={test_mse:.4f}   MAE={test_mae:.3f}\n"
        "Uses the LR last chosen on Step 8.",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_TICK,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
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
        (6.0, "Data", "noisy samples from a nonlinear curve"),
        (4.7, "Model + loss", "MLP capacity + MSE"),
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
        "MLP regression story = nonlinear fit -> MSE -> train/val/test -> LR path.",
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
print("MLP REGRESSION -- learn a curved f(x)")
print("=" * 60)
print(
    f"""
Data: {N} noisy (x, y) samples from a nonlinear target
Split: train={len(x_tr)}  val={len(x_va)}  test={len(x_te)}
Model: MLP 1->128->128->128->128->1 + MSE
Interactive Step 8: LR slider, Stop/Resume, evolving fit curve,
                    train/val curves + test MSE line.

Sample test MSE @ LR={DEMO_LR:g}: {LR_HIST[DEMO_LR]['test_loss']:.4f}
"""
)
print("=" * 60)
