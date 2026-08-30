"""
070_torch_training.py

The simplest PyTorch training loop: one weight, no bias.

    y ≈ w * x

Data: a few (x, y) pairs near the line y = 2x.
Train w by gradient descent on mean squared error.
An animation shows the prediction line fitting the data one SGD step at a time.

Companion to 080_model.py (adds bias / richer models).
"""

from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import animation

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
    add_step_footer,
)
apply_rcparams(plt)

torch.manual_seed(523)

ANIM_FPS = 4
ANIM_HOLD_SEC = 2.0
OUT_GIF = Path(__file__).resolve().parent / "070_torch_training.gif"


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
    add_step_footer(fig)
    plt.show()


# -------------------------------------------------------
# Data: y ≈ 2 x  (noise)
# -------------------------------------------------------
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = 2.0 * x + 2.5 * torch.randn(5, 1)

# -------------------------------------------------------
# Model: single neuron, NO bias  ->  y_hat = w * x
# -------------------------------------------------------
model = nn.Linear(1, 1, bias=False)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# -------------------------------------------------------
# Train -- record w and loss every SGD step (including init)
# -------------------------------------------------------
SGD_STEPS = 100
xs_line = torch.linspace(0.5, 5.5, 50).unsqueeze(1)

with torch.no_grad():
    w_hist = [model.weight.item()]
    loss_hist = [loss_fn(model(x), y).item()]
    pred_hist = [model(xs_line).squeeze(1).clone()]

for sgd_step in range(SGD_STEPS):
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        w_hist.append(model.weight.item())
        loss_hist.append(loss_fn(model(x), y).item())
        pred_hist.append(model(xs_line).squeeze(1).clone())

w_final = w_hist[-1]
print(f"Learned w = {w_final:.4f}  (true slope ≈ 2)")
print(f"Final MSE = {loss_hist[-1]:.6f}")

x_np = x.numpy()
y_np = y.numpy()
xs_np = xs_line.numpy().ravel()
true_ys = (2.0 * xs_line).numpy().ravel()
pred_np = [p.numpy() for p in pred_hist]


# -------------------------------------------------------
# Animation: prediction line progressively fits the data
# -------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig.suptitle(
    "015: watch one weight learn  (y = w x, no bias)",
    fontsize=FS_FIGURE_TITLE, fontweight="bold",
)

ax_fit = axes[0]
ax_fit.scatter(x_np, y_np, s=80, c="royalblue", label="data", zorder=3)
ax_fit.plot(xs_np, true_ys, "k--", lw=1.8, alpha=0.65, label="true $y=2x$")
fit_line, = ax_fit.plot([], [], color="crimson", lw=2.8, label="predict $y=wx$")
hud = ax_fit.text(
    0.02, 0.98, "", transform=ax_fit.transAxes, va="top", ha="left",
    fontsize=FS_ANNOT,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
)
ax_fit.set_xlabel("x", fontsize=FS_AXIS)
ax_fit.set_ylabel("y", fontsize=FS_AXIS)
ax_fit.set_title("Fit improves with each SGD step", fontsize=FS_AX_TITLE)
ax_fit.tick_params(axis="both", labelsize=FS_TICK)
ax_fit.legend(loc="lower right", fontsize=FS_LEGEND)
ax_fit.grid(True, alpha=0.3)
y_lo = min(float(y_np.min()), float(true_ys.min()), min(p.min() for p in pred_np)) - 1.0
y_hi = max(float(y_np.max()), float(true_ys.max()), max(p.max() for p in pred_np)) + 1.0
ax_fit.set_xlim(0.3, 5.7)
ax_fit.set_ylim(y_lo, y_hi)

ax_loss = axes[1]
loss_line, = ax_loss.plot([], [], color="seagreen", lw=2.5)
loss_dot, = ax_loss.plot([], [], "o", color="seagreen", ms=8)
ax_loss.set_xlim(0, len(loss_hist) - 1)
ax_loss.set_ylim(0.0, max(loss_hist) * 1.1 + 1e-6)
ax_loss.set_xlabel("SGD step", fontsize=FS_AXIS)
ax_loss.set_ylabel("MSE", fontsize=FS_AXIS)
ax_loss.set_title("Mini-batch loss", fontsize=FS_AX_TITLE)
ax_loss.tick_params(axis="both", labelsize=FS_TICK)
ax_loss.grid(True, alpha=0.3)

fig.tight_layout(rect=[0, 0, 1, 0.94])

n_frames = len(w_hist)
hold = max(1, int(round(ANIM_HOLD_SEC * ANIM_FPS)))
frame_ids = list(range(n_frames)) + [n_frames - 1] * hold


def init_anim():
    fit_line.set_data([], [])
    loss_line.set_data([], [])
    loss_dot.set_data([], [])
    hud.set_text("")
    return fit_line, loss_line, loss_dot, hud


def update_anim(frame):
    fit_line.set_data(xs_np, pred_np[frame])
    steps = list(range(frame + 1))
    loss_line.set_data(steps, loss_hist[: frame + 1])
    loss_dot.set_data([frame], [loss_hist[frame]])
    hud.set_text(
        f"SGD step {frame}/{n_frames - 1}\n"
        f"w = {w_hist[frame]:.3f}\n"
        f"MSE = {loss_hist[frame]:.3f}"
    )
    return fit_line, loss_line, loss_dot, hud


anim = animation.FuncAnimation(
    fig, update_anim, init_func=init_anim,
    frames=frame_ids, interval=1000 / ANIM_FPS, blit=False,
)

if OUT_GIF.is_file() and OUT_GIF.stat().st_size > 0:
    print(f"\nSkipping GIF export (already exists): {OUT_GIF}")
else:
    print(f"Saving animation to {OUT_GIF} ...")
    try:
        anim.save(OUT_GIF, writer=animation.PillowWriter(fps=ANIM_FPS))
        print(f"Saved: {OUT_GIF}")
    except Exception as exc:
        print(f"Skipping GIF export ({OUT_GIF.name}): {exc}")
        print("Install pillow to enable animation export: pip install pillow")

show_maximized(fig)
