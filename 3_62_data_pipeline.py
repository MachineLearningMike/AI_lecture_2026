"""
3_62_data_pipeline.py

Standard PyTorch data pipeline:
  Dataset  -- one sample at a time  (__len__, __getitem__)
  DataLoader -- batches, shuffle, workers, iteration

Then a tiny end-to-end training example that uses both.

Sketch:
  1. Pipeline map: raw data -> Dataset -> DataLoader -> training loop
  2. Dataset contract
  3. Dataset code
  4. DataLoader job and knobs
  5. DataLoader code
  6. Training loop pattern (for batch in loader)
  7. Full mini example -- code
  8. Full mini example -- live run (loss curve)
  9. Takeaway
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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

torch.manual_seed(265)
np.random.seed(265)


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
C_OLD = "#fce4d6"
C_LOOP = "#2874a6"
C_CODE = "#fbfcf8"


# -------------------------------------------------------
# Live mini example (also shown on slides)
# -------------------------------------------------------
class ToyRegressionDataset(Dataset):
    """y ≈ 3x + 1 with noise. One sample = (x_i, y_i)."""

    def __init__(self, n: int = 200, noise: float = 0.4):
        x = torch.linspace(-2.0, 2.0, n).unsqueeze(1)
        y = 3.0 * x + 1.0 + noise * torch.randn(n, 1)
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def run_mini_training(epochs: int = 40, batch_size: int = 32, lr: float = 0.05):
    ds = ToyRegressionDataset(n=200, noise=0.4)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    for _ in range(epochs):
        total = 0.0
        n_batches = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n_batches += 1
        losses.append(total / max(n_batches, 1))

    with torch.no_grad():
        w = float(model.weight.item())
        b = float(model.bias.item())
    return losses, w, b, ds


LOSSES, LEARNED_W, LEARNED_B, TOY_DS = run_mini_training()


DATASET_CODE = """\
class ToyRegressionDataset(Dataset):
    def __init__(self, n=200, noise=0.4):
        x = torch.linspace(-2, 2, n).unsqueeze(1)
        y = 3 * x + 1 + noise * torch.randn(n, 1)
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
"""

LOADER_CODE = """\
ds = ToyRegressionDataset(n=200)
loader = DataLoader(
    ds,
    batch_size=32,
    shuffle=True,     # new order each epoch
    num_workers=0,    # >0 for heavier I/O
    drop_last=False,
)

for xb, yb in loader:   # xb, yb are batched tensors
    ...
"""

TRAIN_CODE = """\
ds = ToyRegressionDataset(n=200)
loader = DataLoader(ds, batch_size=32, shuffle=True)

model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

for epoch in range(40):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
"""


# -------------------------------------------------------
# Step 1: Pipeline map
# -------------------------------------------------------
def build_step_1():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 1: Standard PyTorch data pipeline",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    stages = [
        (0.4, "Raw data", "files / arrays\n/ databases"),
        (3.3, "Dataset", "one sample\nvia __getitem__"),
        (6.2, "DataLoader", "batches +\nshuffle"),
        (9.1, "Train loop", "forward / loss\n/ backward"),
    ]
    for x, title, body in stages:
        round_box(ax, x, 3.2, 2.6, 3.2, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.3, 5.7, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(x + 1.3, 4.3, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    for x0 in (2.9, 5.8, 8.7):
        ax.annotate(
            "", xy=(x0 + 0.35, 4.8), xytext=(x0, 4.8),
            arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2.2),
        )

    round_box(ax, 0.4, 0.45, 11.2, 2.3, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.6,
        "Dataset knows samples.\n"
        "DataLoader knows how to feed the optimizer in mini-batches.\n"
        "Training code should almost always iterate the DataLoader -- not raw tensors.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 2: Dataset contract
# -------------------------------------------------------
def build_step_2():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 2: Dataset -- the sample interface",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.2, 11.2, 1.5, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.95,
        "torch.utils.data.Dataset  =  map from index  →  one training example",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    cards = [
        (0.4, 3.2, "__init__", "store paths / tensors\n/ handles\n\ncheap to construct"),
        (4.2, 3.2, "__len__", "how many samples?\n\nlen(dataset)"),
        (8.0, 3.2, "__getitem__", "return sample i\n\n(x_i, y_i) or dict"),
    ]
    for x, y, title, body in cards:
        round_box(ax, x, y, 3.6, 2.7, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.8, y + 2.15, title, ha="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
                family="monospace")
        ax.text(x + 1.8, y + 1.0, body, ha="center", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.4, 0.4, 11.2, 2.5, C_GOAL, C_OK, lw=2.0)
    ax.text(
        6.0, 1.65,
        "Dataset does not batch.\n"
        "It answers: \"give me sample #i\".\n"
        "Transforms (resize, normalize, augment) usually live in __getitem__.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#1e4419",
    )
    return fig


# -------------------------------------------------------
# Step 3: Dataset code
# -------------------------------------------------------
def build_step_3():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 3: Custom Dataset -- minimal working example",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    code_panel(ax, 0.35, 0.45, 11.3, 7.1, DATASET_CODE, fontsize=FS_TICK + 1)
    return fig


# -------------------------------------------------------
# Step 4: DataLoader job
# -------------------------------------------------------
def build_step_4():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 4: DataLoader -- batching and iteration",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    round_box(ax, 0.4, 6.2, 11.2, 1.5, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 6.95,
        "DataLoader wraps a Dataset and yields mini-batches for training.",
        ha="center", va="center", fontsize=FS_ANNOT + 1,
        fontweight="bold", color="#f9e79f",
    )

    knobs = [
        (0.4, 3.15, "batch_size", "how many samples\nper step"),
        (3.3, 3.15, "shuffle", "randomize order\neach epoch"),
        (6.2, 3.15, "num_workers", "parallel load\nprocesses"),
        (9.1, 3.15, "drop_last", "drop incomplete\nfinal batch?"),
    ]
    for x, y, title, body in knobs:
        round_box(ax, x, y, 2.7, 2.7, C_NEW, C_LOOP, lw=2.0)
        ax.text(x + 1.35, y + 2.1, title, ha="center",
                fontsize=FS_TICK + 1, fontweight="bold", color=C_DARK,
                family="monospace")
        ax.text(x + 1.35, y + 1.05, body, ha="center", va="center",
                fontsize=FS_TICK, color="#333333")

    round_box(ax, 0.4, 0.4, 11.2, 2.45, C_MUTE, "0.45", lw=1.6)
    ax.text(
        6.0, 1.6,
        "Also collates samples into tensors (default_collate).\n"
        "For variable-length inputs you may pass a custom collate_fn.\n"
        "pin_memory=True helps host→GPU copies when training on CUDA.",
        ha="center", va="center", fontsize=FS_ANNOT, color="#333333",
    )
    return fig


# -------------------------------------------------------
# Step 5: DataLoader code
# -------------------------------------------------------
def build_step_5():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 5: DataLoader -- construction and iteration",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    code_panel(ax, 0.35, 0.45, 11.3, 7.1, LOADER_CODE, fontsize=FS_TICK + 1)
    return fig


# -------------------------------------------------------
# Step 6: Training pattern
# -------------------------------------------------------
def build_step_6():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 6: Training loop pattern -- iterate the loader",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)

    steps = [
        (6.2, "1. for epoch in range(E):", "one pass over the dataset"),
        (4.7, "2. for xb, yb in loader:", "one mini-batch"),
        (3.2, "3. pred = model(xb);  loss = loss_fn(pred, yb)", "forward"),
        (1.7, "4. zero_grad → backward → step", "update parameters"),
    ]
    for y, title, note in steps:
        round_box(ax, 0.5, y - 0.55, 11.0, 1.3, C_NEW, C_LOOP, lw=1.8)
        ax.text(0.9, y + 0.1, title, ha="left", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
                family="monospace")
        ax.text(11.0, y + 0.1, note, ha="right", va="center",
                fontsize=FS_TICK + 1, color="#333333")

    round_box(ax, 0.5, 0.25, 11.0, 0.95, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 0.72,
        "Dataset defines samples.  DataLoader defines the stream of batches.",
        ha="center", va="center", fontsize=FS_ANNOT,
        fontweight="bold", color="#f9e79f",
    )
    return fig


# -------------------------------------------------------
# Step 7: Full example code
# -------------------------------------------------------
def build_step_7():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 7: End-to-end mini training example",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    blank_slide(ax)
    code_panel(ax, 0.35, 0.35, 11.3, 7.2, TRAIN_CODE, fontsize=FS_TICK)
    return fig


# -------------------------------------------------------
# Step 8: Live run
# -------------------------------------------------------
def build_step_8():
    fig, axes = new_figure(1, 2)
    fig.suptitle(
        "Step 8: Same example -- live run",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(np.arange(1, len(LOSSES) + 1), LOSSES, color=C_LOOP, lw=2.6)
    ax.set_xlabel("epoch", fontsize=FS_AXIS)
    ax.set_ylabel("mean batch MSE", fontsize=FS_AXIS)
    ax.set_title("training loss", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98, 0.95,
        f"final MSE ≈ {LOSSES[-1]:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
    )

    ax = axes[1]
    x = TOY_DS.x.numpy().ravel()
    y = TOY_DS.y.numpy().ravel()
    ax.scatter(x, y, s=18, c="#85929e", alpha=0.55, label="data")
    xs = np.linspace(-2, 2, 100)
    ax.plot(xs, 3 * xs + 1, "k--", lw=2.0, label="true  3x+1")
    ax.plot(xs, LEARNED_W * xs + LEARNED_B, color=C_OK, lw=2.6,
            label=f"learned  {LEARNED_W:.2f}x+{LEARNED_B:.2f}")
    ax.set_xlabel("x", fontsize=FS_AXIS)
    ax.set_ylabel("y", fontsize=FS_AXIS)
    ax.set_title("fit after DataLoader training", fontsize=FS_AX_TITLE - 2,
                 fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND - 1, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# -------------------------------------------------------
# Step 9: Takeaway
# -------------------------------------------------------
def build_step_9():
    fig, ax = new_figure()
    fig.suptitle(
        "Step 9: Takeaway",
        fontsize=FS_FIGURE_TITLE, fontweight="bold",
    )
    blank_slide(ax)

    rows = [
        (5.9, "Dataset", "__len__ + __getitem__  →  one sample"),
        (4.5, "DataLoader", "batches, shuffle, workers, collation"),
        (3.1, "Train loop", "for epoch / for batch in loader / update"),
        (1.7, "Shortcut", "TensorDataset(x, y) when data already tensors"),
    ]
    for y, left, right in rows:
        round_box(ax, 0.5, y - 0.45, 11.0, 1.2, C_NEW, C_LOOP, lw=1.8)
        ax.text(2.4, y + 0.15, left, ha="center", va="center",
                fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
        ax.text(7.6, y + 0.15, right, ha="center", va="center",
                fontsize=FS_ANNOT, color="#333333")

    round_box(ax, 0.5, 0.25, 11.0, 1.0, C_DARK, C_DARK, lw=0)
    ax.text(
        6.0, 0.75,
        "Pipeline = Dataset (what is a sample?) + DataLoader (how are batches streamed?).",
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
])


print("\n" + "=" * 60)
print("PYTORCH DATA PIPELINE")
print("=" * 60)
print(f"""
Dataset:   map index -> sample   (__len__, __getitem__)
DataLoader: Dataset -> mini-batches (batch_size, shuffle, ...)

Mini example result:
  learned y ~= {LEARNED_W:.3f} x + {LEARNED_B:.3f}   (truth 3x+1)
  final mean-batch MSE ~= {LOSSES[-1]:.4f}

Also useful: TensorDataset(x, y) when tensors already live in memory.
""")
