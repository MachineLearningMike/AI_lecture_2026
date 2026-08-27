"""
015_torch_training.py

The simplest PyTorch training loop: one weight, no bias.

    y ≈ w * x

Data: a few (x, y) pairs near the line y = 2x.
Train w by gradient descent on mean squared error.

Companion to 020_model.py (adds bias / richer models).
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
)
apply_rcparams(plt)
torch.manual_seed(42)


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


def show_maximized(fig):
    fig.set_dpi(FIGURE_DPI)
    fig.set_size_inches(screen_figsize(), forward=True)
    fig.canvas.draw_idle()
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None:
        try:
            manager.window.state("zoomed")
        except Exception:
            try:
                manager.window.showMaximized()
            except Exception:
                try:
                    manager.frame.Maximize(True)
                except Exception:
                    pass
    plt.show()

# -------------------------------------------------------
# Data: y ≈ 2 x  (small noise)
# -------------------------------------------------------
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = 2.0 * x + 0.5 * torch.randn(5, 1)

# -------------------------------------------------------
# Model: single neuron, NO bias  ->  y_hat = w * x
# -------------------------------------------------------
model = nn.Linear(1, 1, bias=False)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# -------------------------------------------------------
# Train
# -------------------------------------------------------
losses = []
for step in range(5):
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

w = model.weight.item()
print(f"Learned w = {w:.4f}  (true slope ≈ 2)")
print(f"Final MSE = {losses[-1]:.6f}")

# -------------------------------------------------------
# Plot: data + fit, and loss curve
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)

ax = axes[0]
ax.scatter(x.numpy(), y.numpy(), s=60, c="royalblue", label="data", zorder=3)
xs = torch.linspace(0.5, 5.5, 50).unsqueeze(1)
with torch.no_grad():
    ys = model(xs)
ax.plot(xs.numpy(), ys.numpy(), color="crimson", lw=2.5, label=rf"fit $y={w:.2f}x$")
ax.plot(xs.numpy(), (2.0 * xs).numpy(), "k--", lw=1.5, alpha=0.6, label="true $y=2x$")
ax.set_xlabel("x", fontsize=FS_AXIS)
ax.set_ylabel("y", fontsize=FS_AXIS)
ax.set_title("Single neuron, no bias", fontsize=FS_AX_TITLE)
ax.tick_params(axis="both", labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(losses, color="seagreen", lw=2.0)
ax.set_xlabel("step", fontsize=FS_AXIS)
ax.set_ylabel("MSE", fontsize=FS_AXIS)
ax.set_title("Training loss", fontsize=FS_AX_TITLE)
ax.tick_params(axis="both", labelsize=FS_TICK)
ax.grid(True, alpha=0.3)

fig.suptitle(
    "015: train one weight with PyTorch",
    fontsize=FS_FIGURE_TITLE, fontweight="bold",
)
fig.tight_layout()
show_maximized(fig)
