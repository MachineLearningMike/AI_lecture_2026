import torch
import torch.nn as nn
import torch.optim as optim
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
    add_step_footer,
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


# =======================================================
# CONFIGURATION: Toggle between Simple and Complex Model
# =======================================================
USE_COMPLEX_MODEL = False  # Set to False for simple model, True for complex model

# -------------------------------------------------------
# 1. CREATE THE DATA
# -------------------------------------------------------
torch.manual_seed(42)

N = 1000
X = torch.randn(N, 1) + 1.0  # inputs centered near 1

# True underlying function (noise-free): y = a * x^p + b
a, b, power = 2.0, 1.0, 3
Y_true = a * torch.pow(X, power) + b

# Observe noisy labels (standard Gaussian noise — not a product of Gaussians)
noise_std = 10.0
Y = Y_true + noise_std * torch.randn(N, 1)

# Normalize BOTH x and y for stable training.
# A linear model on normalized data is still a straight line after denormalizing,
# so the simple vs complex comparison stays fair (architecture, not optimizer scale).
X_mean, X_std = X.mean(), X.std()
Y_mean, Y_std = Y.mean(), Y.std()
X_n = (X - X_mean) / X_std
Y_n = (Y - Y_mean) / Y_std

# Sorted true curve for plotting
sorted_indices = torch.argsort(X, dim=0).squeeze()
X_sorted = X[sorted_indices]
Y_true_sorted = Y_true[sorted_indices]


def denormalize_y(y_n: torch.Tensor) -> torch.Tensor:
    return y_n * Y_std + Y_mean


def linear_params_in_original_space(model: nn.Module):
    """Convert normalized-space linear weights to y = slope * x + intercept."""
    w = model.neuron.weight.item()
    bias_n = model.neuron.bias.item()
    slope = (Y_std * w / X_std).item()
    intercept = (Y_mean + Y_std * bias_n - Y_std * w * X_mean / X_std).item()
    return slope, intercept


# -------------------------------------------------------
# 2. BUILD THE NEURAL NETWORK
# -------------------------------------------------------
class SimplestAI(nn.Module):
    """One linear neuron: can only represent straight lines."""

    def __init__(self):
        super().__init__()
        self.neuron = nn.Linear(1, 1)

    def forward(self, x):
        return self.neuron(x)


class ComplexAI(nn.Module):
    """Small MLP with non-linear activations: can fit curved functions."""

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
        # Kaiming for ReLU layers; Xavier for the tanh layer / linear output
        nn.init.kaiming_uniform_(self.layer1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.layer2.weight, nonlinearity='relu')
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


if USE_COMPLEX_MODEL:
    model = ComplexAI()
    model_name = "Complex Model (4 layers with ReLU + Tanh)"
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 2000
else:
    model = SimplestAI()
    model_name = "Simple Model (1 linear neuron)"
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    epochs = 1000

criterion = nn.MSELoss()

# Dense x-grid for a smooth prediction curve (original + normalized space)
X_grid = torch.linspace(X.min().item(), X.max().item(), 400).unsqueeze(1)
X_grid_n = (X_grid - X_mean) / X_std

# -------------------------------------------------------
# 3. PLOT: BEFORE TRAINING (Random Guesses)
# -------------------------------------------------------
model.eval()
with torch.no_grad():
    preds_before = denormalize_y(model(X_grid_n)).numpy()

X_np = X.numpy()
Y_np = Y.numpy()
X_sorted_np = X_sorted.numpy()
Y_true_sorted_np = Y_true_sorted.numpy()
X_grid_np = X_grid.numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=screen_figsize(), dpi=FIGURE_DPI)
fig.suptitle(f'Model: {model_name}', fontsize=FS_FIGURE_TITLE, fontweight='bold')

ax1.scatter(X_np, Y_np, alpha=0.3, label='Data with Noise', color='blue', s=5)
ax1.plot(X_sorted_np, Y_true_sorted_np, color='green', linewidth=2,
         label='True Function', alpha=0.8)
ax1.plot(X_grid_np, preds_before, color='red', linewidth=2, label='AI Guesses')
ax1.set_title('BEFORE Training (Random Guesses)', fontsize=FS_AX_TITLE)
ax1.set_xlabel('X', fontsize=FS_AXIS)
ax1.set_ylabel('Y', fontsize=FS_AXIS)
ax1.tick_params(axis='both', labelsize=FS_TICK)
ax1.legend(fontsize=FS_LEGEND)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-50, 150)

# -------------------------------------------------------
# 4. TRAIN THE MODEL
# -------------------------------------------------------
print("=" * 60)
print(f"Training {model_name}...")
print("=" * 60)

model.train()
losses = []

for epoch in range(epochs):
    predictions = model(X_n)
    loss = criterion(predictions, Y_n)

    optimizer.zero_grad()
    loss.backward()
    if USE_COMPLEX_MODEL:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    losses.append(loss.item())

    if (epoch + 1) % max(1, epochs // 10) == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

# -------------------------------------------------------
# 5. PLOT: AFTER TRAINING (Learned Model)
# -------------------------------------------------------
model.eval()
with torch.no_grad():
    preds_after = denormalize_y(model(X_grid_n)).numpy()
    preds_data = denormalize_y(model(X_n))
    mse_vs_true = torch.mean((preds_data - Y_true) ** 2).item()
    mse_vs_data = torch.mean((preds_data - Y) ** 2).item()

ax2.scatter(X_np, Y_np, alpha=0.3, label='Data with Noise', color='blue', s=5)
ax2.plot(X_sorted_np, Y_true_sorted_np, color='green', linewidth=2,
         label='True Function', alpha=0.8)
ax2.plot(X_grid_np, preds_after, color='red', linewidth=2, label='AI Predictions')
ax2.set_title('AFTER Training', fontsize=FS_AX_TITLE)
ax2.set_xlabel('X', fontsize=FS_AXIS)
ax2.set_ylabel('Y', fontsize=FS_AXIS)
ax2.tick_params(axis='both', labelsize=FS_TICK)
ax2.legend(fontsize=FS_LEGEND)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-50, 150)

ax2.text(
    0.05, 0.95, f'True: Y = {a:.1f}X^{power} + {b:.1f}',
    transform=ax2.transAxes, fontsize=FS_ANNOT, verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
)

if USE_COMPLEX_MODEL:
    ax2.text(
        0.05, 0.82, 'Complex Model\n(4 layers, ReLU + Tanh)',
        transform=ax2.transAxes, fontsize=FS_ANNOT, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
    )
else:
    slope, intercept = linear_params_in_original_space(model)
    ax2.text(
        0.05, 0.82, f'Pred: Y = {slope:.2f}X + {intercept:.2f}',
        transform=ax2.transAxes, fontsize=FS_ANNOT, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
    )

plt.tight_layout()
show_maximized(fig)

# -------------------------------------------------------
# 6. PLOT: TRAINING LOSS CURVE
# -------------------------------------------------------
fig_loss, ax_loss = plt.subplots(1, 1, figsize=screen_figsize(), dpi=FIGURE_DPI)
ax_loss.plot(losses, color='blue', linewidth=2)
ax_loss.set_title(
    f'Training Loss Over Time ({model_name})',
    fontsize=FS_AX_TITLE,
)
ax_loss.set_xlabel('Epoch', fontsize=FS_AXIS)
ax_loss.set_ylabel('Normalized MSE Loss', fontsize=FS_AXIS)
ax_loss.tick_params(axis='both', labelsize=FS_TICK)
ax_loss.grid(True, alpha=0.3)
if USE_COMPLEX_MODEL:
    ax_loss.set_yscale('log')
plt.tight_layout()
show_maximized(fig_loss)

# -------------------------------------------------------
# 7. FINAL RESULTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"True function: Y = {a:.2f} * X^{power} + {b:.2f}")
print(f"Final normalized MSE (vs noisy labels): {losses[-1]:.6f}")
print(f"MSE vs true function (denormalized):    {mse_vs_true:.4f}")
print(f"MSE vs noisy data (denormalized):       {mse_vs_data:.4f}")
print(f"(Noise floor E[noise^2] ~= {noise_std ** 2:.1f})")

if USE_COMPLEX_MODEL:
    print("\nSUCCESS: The complex model can follow the non-linear cubic.")
    print("  - Green curve: true relationship")
    print("  - Red curve: what the MLP learned")
    print("  - MSE vs true << MSE vs noisy data means it fit the signal, not just noise")
else:
    slope, intercept = linear_params_in_original_space(model)
    print(f"\nThe linear model learned: Y = {slope:.2f} * X + {intercept:.2f}")
    print("NOTICE: A straight line cannot capture the cubic pattern.")
    print("  - Green curve: true relationship (curved)")
    print("  - Red curve: best straight-line fit the linear model can do")
    print("  - This is why non-linear models (activations + depth) are needed")

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print("""
Simple Model:  1 linear neuron (can only learn straight lines)
Complex Model: 4 layers with 64->64->32->1 neurons + ReLU/Tanh activations

Both models train on the SAME normalized data. The simple model still fails
because a linear map cannot represent Y = a*X^3 + b -- not because of scaling.

To toggle between models, change the flag at the top of the script:
    USE_COMPLEX_MODEL = True   # Complex model (non-linear)
    USE_COMPLEX_MODEL = False  # Simple model (linear)
""")
