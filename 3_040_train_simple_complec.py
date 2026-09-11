"""
3_040_train_simple_complec.py

Train a simple linear model and a small non-linear MLP on the same cubic
data -- then watch each converge.

Sketch:
  1. Setup -- cubic signal + noise
  2. Simple model -- before / after
  3. Simple model -- loss curve
  4. Simple model -- live convergence animation
  5. Complex model -- before / after
  6. Complex model -- loss curve
  7. Complex model -- live convergence animation
  8. Takeaway -- capacity, not scaling
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
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
    run_slide_deck,
)
apply_rcparams(plt)


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


# -------------------------------------------------------
# Data (shared by both models)
# -------------------------------------------------------
torch.manual_seed(42)

N = 1000
X_org = torch.randn(N, 1) + 1.0
X = X_org + 0.1 * torch.rand(N, 1)

a, b, power = 2.0, 1.0, 3
Y_true = a * torch.pow(X, power) + b

noise_std = 10.0
Y = Y_true + noise_std * torch.randn(N, 1)

X_mean, X_std = X.mean(), X.std()
Y_mean, Y_std = Y.mean(), Y.std()
X_n = (X - X_mean) / X_std
Y_n = (Y - Y_mean) / Y_std

sorted_indices = torch.argsort(X, dim=0).squeeze()
X_sorted = X[sorted_indices]
Y_true_sorted = Y_true[sorted_indices]

X_grid = torch.linspace(X.min().item(), X.max().item(), 400).unsqueeze(1)
X_grid_n = (X_grid - X_mean) / X_std

X_np = X.numpy()
Y_np = Y.numpy()
X_sorted_np = X_sorted.numpy()
Y_true_sorted_np = Y_true_sorted.numpy()
X_grid_np = X_grid.numpy()


def denormalize_y(y_n: torch.Tensor) -> torch.Tensor:
    return y_n * Y_std + Y_mean


def linear_params_in_original_space(model: nn.Module):
    w = model.neuron.weight.item()
    bias_n = model.neuron.bias.item()
    slope = (Y_std * w / X_std).item()
    intercept = (Y_mean + Y_std * bias_n - Y_std * w * X_mean / X_std).item()
    return slope, intercept


class SimplestAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.neuron = nn.Linear(1, 1)

    def forward(self, x):
        return self.neuron(x)


class ComplexAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.tanh(self.layer3(x))
        x = self.layer4(x)
        return x


def train_case(kind: str):
    """Train one architecture; return histories for plots/animation."""
    torch.manual_seed(42)
    criterion = nn.MSELoss()

    if kind == "simple":
        model = SimplestAI()
        name = "Simple Model (1 linear neuron)"
        # 100x smaller than the original lr=0.1 so the convergence animation is readable.
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        epochs = 1000
        snap_every = 20
        clip = False
        color = "#c0392b"
    else:
        model = ComplexAI()
        name = "Complex Model (4 layers, ReLU + Tanh)"
        # 20x smaller than the original lr=0.001.
        optimizer = optim.Adam(model.parameters(), lr=5e-5)
        epochs = 2000
        snap_every = 40
        clip = True
        color = "#8e44ad"

    model.eval()
    with torch.no_grad():
        preds_before = denormalize_y(model(X_grid_n)).numpy().copy()

    model.train()
    losses = []
    pred_snaps = [preds_before]
    snap_epochs = [0]

    print("=" * 60)
    print(f"Training {name}...")
    print("=" * 60)

    for epoch in range(epochs):
        predictions = model(X_n)
        loss = criterion(predictions, Y_n)
        optimizer.zero_grad()
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % snap_every == 0 or epoch + 1 == epochs:
            model.eval()
            with torch.no_grad():
                snap = denormalize_y(model(X_grid_n)).numpy().copy()
            pred_snaps.append(snap)
            snap_epochs.append(epoch + 1)
            model.train()

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        preds_after = denormalize_y(model(X_grid_n)).numpy().copy()
        preds_data = denormalize_y(model(X_n))
        mse_vs_true = torch.mean((preds_data - Y_true) ** 2).item()
        mse_vs_data = torch.mean((preds_data - Y) ** 2).item()

    return {
        "kind": kind,
        "name": name,
        "color": color,
        "model": model,
        "losses": losses,
        "preds_before": preds_before,
        "preds_after": preds_after,
        "pred_snaps": pred_snaps,
        "snap_epochs": snap_epochs,
        "mse_vs_true": mse_vs_true,
        "mse_vs_data": mse_vs_data,
        "log_loss": kind == "complex",
    }


print("Running BOTH simple and complex training cases...")
SIMPLE = train_case("simple")
COMPLEX = train_case("complex")


def _before_after_slide(result):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle(result["name"], fontsize=FS_FIGURE_TITLE, fontweight="bold")

    for ax, preds, title in (
        (ax1, result["preds_before"], "Before training (random guesses)"),
        (ax2, result["preds_after"], "After training"),
    ):
        ax.scatter(X_np, Y_np, alpha=0.3, label="Data with noise", color="blue", s=5)
        ax.plot(X_sorted_np, Y_true_sorted_np, color="green", linewidth=2,
                label="True function", alpha=0.8)
        ax.plot(X_grid_np, preds, color=result["color"], linewidth=2,
                label="Model")
        ax.set_title(title, fontsize=FS_AX_TITLE)
        ax.set_xlabel("X", fontsize=FS_AXIS)
        ax.set_ylabel("Y", fontsize=FS_AXIS)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.legend(fontsize=FS_LEGEND)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-50, 150)

    ax2.text(
        0.05, 0.95, fr"True: $Y = {a:.1f}X^{power} + {b:.1f}$",
        transform=ax2.transAxes, fontsize=FS_ANNOT, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )
    if result["kind"] == "simple":
        slope, intercept = linear_params_in_original_space(result["model"])
        note = f"Pred: Y = {slope:.2f}X + {intercept:.2f}"
    else:
        note = "Complex MLP\n(non-linear capacity)"
    ax2.text(
        0.05, 0.82, note,
        transform=ax2.transAxes, fontsize=FS_ANNOT, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
    )
    plt.tight_layout()
    return fig


def _loss_slide(result):
    fig, ax = plt.subplots(1, 1, figsize=screen_figsize(), dpi=FIGURE_DPI)
    ax.plot(result["losses"], color=result["color"], linewidth=2)
    ax.set_title(
        f'Training loss -- {result["name"]}',
        fontsize=FS_AX_TITLE,
    )
    ax.set_xlabel("Epoch", fontsize=FS_AXIS)
    ax.set_ylabel("Normalized MSE loss", fontsize=FS_AXIS)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    if result["log_loss"]:
        ax.set_yscale("log")
    ax.text(
        0.98, 0.95,
        f'final loss = {result["losses"][-1]:.4f}\n'
        f'MSE vs true = {result["mse_vs_true"]:.1f}',
        transform=ax.transAxes, ha="right", va="top", fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )
    plt.tight_layout()
    return fig


def _convergence_animation_slide(result):
    """Live animation: prediction curve evolves over epochs."""
    fig, (ax, ax_loss) = plt.subplots(
        1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI,
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )
    fig.suptitle(
        f'Step: Convergence animation -- {result["name"]}',
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )

    snaps = result["pred_snaps"]
    epochs = result["snap_epochs"]
    losses = result["losses"]

    ax.scatter(X_np, Y_np, alpha=0.25, color="blue", s=5, label="Data")
    ax.plot(X_sorted_np, Y_true_sorted_np, color="green", lw=2.2,
            label="True function", alpha=0.85)
    (line_pred,) = ax.plot(
        X_grid_np, snaps[0], color=result["color"], lw=2.6, label="Model",
    )
    title_txt = ax.set_title("epoch 0", fontsize=FS_AX_TITLE)
    ax.set_xlabel("X", fontsize=FS_AXIS)
    ax.set_ylabel("Y", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_ylim(-50, 150)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FS_LEGEND, loc="upper left")

    (loss_line,) = ax_loss.plot([], [], color=result["color"], lw=2.0)
    ax_loss.set_xlim(0, max(len(losses) - 1, 1))
    ymin = min(losses)
    ymax = max(losses)
    if result["log_loss"]:
        ax_loss.set_yscale("log")
        ax_loss.set_ylim(max(ymin * 0.8, 1e-6), ymax * 1.2)
    else:
        pad = 0.05 * (ymax - ymin + 1e-9)
        ax_loss.set_ylim(ymin - pad, ymax + pad)
    ax_loss.set_title("Loss so far", fontsize=FS_AX_TITLE - 1)
    ax_loss.set_xlabel("Epoch", fontsize=FS_AXIS)
    ax_loss.set_ylabel("Normalized MSE", fontsize=FS_AXIS)
    ax_loss.tick_params(labelsize=FS_TICK)
    ax_loss.grid(True, alpha=0.3)

    def _frame(i):
        line_pred.set_ydata(snaps[i].ravel())
        ep = epochs[i]
        title_txt.set_text(f"epoch {ep}")
        xs = list(range(ep)) if ep > 0 else [0]
        ys = losses[:ep] if ep > 0 else [losses[0]]
        loss_line.set_data(xs, ys)
        return (line_pred, loss_line, title_txt)

    anim = animation.FuncAnimation(
        fig, _frame, frames=len(snaps), interval=80, blit=False, repeat=True,
    )
    fig._conv_anim = anim  # keep alive
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    return fig


# -------------------------------------------------------
# Slides
# -------------------------------------------------------
def build_step_1():
    fig, ax = plt.subplots(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle(
        "Step 1: Same data -- cubic signal + large noise",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    ax.scatter(X_np, Y_np, alpha=0.35, color="blue", s=8, label="Noisy observations")
    ax.plot(X_sorted_np, Y_true_sorted_np, color="green", lw=2.5,
            label=fr"True $Y={a:g}X^{power}+{b:g}$")
    ax.set_xlabel("X", fontsize=FS_AXIS)
    ax.set_ylabel("Y", fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_LEGEND)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-50, 150)
    ax.text(
        0.02, 0.98,
        "Both models train on the SAME normalized pairs.\n"
        "Difference = capacity (linear vs non-linear), not the data.",
        transform=ax.transAxes, ha="left", va="top", fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95),
    )
    plt.tight_layout()
    return fig


def build_step_2():
    fig = _before_after_slide(SIMPLE)
    fig.suptitle(
        f'Step 2: {SIMPLE["name"]} -- before / after',
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    return fig


def build_step_3():
    fig = _loss_slide(SIMPLE)
    fig.axes[0].set_title(
        f'Step 3: Loss -- {SIMPLE["name"]}',
        fontsize=FS_AX_TITLE,
    )
    return fig


def build_step_4():
    fig = _convergence_animation_slide(SIMPLE)
    fig.suptitle(
        f'Step 4: Convergence animation -- {SIMPLE["name"]}',
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    return fig


def build_step_5():
    fig = _before_after_slide(COMPLEX)
    fig.suptitle(
        f'Step 5: {COMPLEX["name"]} -- before / after',
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    return fig


def build_step_6():
    fig = _loss_slide(COMPLEX)
    fig.axes[0].set_title(
        f'Step 6: Loss -- {COMPLEX["name"]}',
        fontsize=FS_AX_TITLE,
    )
    return fig


def build_step_7():
    fig = _convergence_animation_slide(COMPLEX)
    fig.suptitle(
        f'Step 7: Convergence animation -- {COMPLEX["name"]}',
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    return fig


def build_step_8():
    fig, axes = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle(
        "Step 8: Takeaway -- capacity decides what can be fit",
        fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
    )
    for ax, result in ((axes[0], SIMPLE), (axes[1], COMPLEX)):
        ax.scatter(X_np, Y_np, alpha=0.2, color="blue", s=4)
        ax.plot(X_sorted_np, Y_true_sorted_np, color="green", lw=2.0, alpha=0.85)
        ax.plot(X_grid_np, result["preds_after"], color=result["color"], lw=2.4)
        ax.set_title(result["name"].split("(")[0].strip(), fontsize=FS_AX_TITLE - 1)
        ax.set_xlabel("X", fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.set_ylim(-50, 150)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02, 0.98,
            f'MSE vs true = {result["mse_vs_true"]:.1f}',
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_ANNOT,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
        )
    axes[0].set_ylabel("Y", fontsize=FS_AXIS)
    fig.text(
        0.5, 0.02,
        "A straight line cannot represent a cubic. Non-linear layers can.",
        ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
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
])

print("\n" + "=" * 60)
print("BOTH CASES COMPLETE")
print("=" * 60)
print(f"True function: Y = {a:.2f} * X^{power} + {b:.2f}")
print(f"Noise floor E[noise^2] ~= {noise_std ** 2:.1f}")
print()
print(f"Simple  final loss={SIMPLE['losses'][-1]:.6f}  MSE vs true={SIMPLE['mse_vs_true']:.2f}")
print(f"Complex final loss={COMPLEX['losses'][-1]:.6f}  MSE vs true={COMPLEX['mse_vs_true']:.2f}")
print()
print("Animations: Steps 4 (simple) and 7 (complex) play live in the deck.")
print("A linear neuron stays a line; the MLP can track the cubic.")
