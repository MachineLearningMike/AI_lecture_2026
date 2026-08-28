"""
142_SGD_training.py

Train a model with Gradient Descent: full-batch vs mini-batch.

Same data, same model, same learning rate. Only the batch size changes:

  full-batch GD:
      each SGD step uses all N examples
      -> exact gradient of full-batch loss L(... ; w)

  mini-batch GD:          (what deep learning usually calls "SGD")
      each SGD step uses b << N examples
      -> noisy estimate of that same gradient

What to watch:
  1) Full-batch loss L(... ; w) for mini-batch GD trends to the SAME limit as full-batch GD.
  2) The mini-batch curve is NOT stable -- it jitters around that limit.
  3) Mini-batch GD is economic: each SGD step is cheap, so you buy MORE
     SGD steps per unit of data/compute.

Terminology matches 144_SGD_concept.py:
  full-batch loss L(... ; w), mini-batch loss, SGD step, examples touched.

Companion to 020_torch_training.py and 144_SGD_concept.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn as nn

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

torch.manual_seed(42)
np.random.seed(42)
RNG = np.random.default_rng(42)


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


def new_figure(*args, **kwargs):
    kwargs.setdefault("figsize", screen_figsize())
    kwargs.setdefault("dpi", FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


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


def add_note(ax, text, x=0.98, y=0.98, ha="right", va="top", facecolor="lightyellow"):
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=FS_ANNOT,
        ha=ha, va=va,
        bbox=dict(boxstyle="round", facecolor=facecolor, alpha=0.95),
    )


# -------------------------------------------------------
# Consistent names (aligned with 141_stochastic_gradient_descent.py):
#   full-batch GD         -- batch size = N
#   mini-batch GD         -- batch size = b  (often called SGD in deep learning)
#   w                     -- model parameters (weights + bias); a CONDITION for L (...; w), not sole input of L
#   full-batch loss L(... ; w)  -- mean loss on the WHOLE dataset (the real objective)
#   mini-batch loss       -- mean loss on the current mini-batch only
#   theoretical floor     -- best attainable L(... ; w) for this model/data (demo only:
#                            computed via least squares because model is linear+MSE)
#   SGD step         -- one optimizer.step()
#   examples touched      -- training rows used so far (compute proxy)
# -------------------------------------------------------
N = 512
D = 4
B = 16                   # mini-batch size
LR = 0.08
N_SGD_STEPS = 180            # fair compare #1: same number of SGD steps
DATA_BUDGET = N * 12     # fair compare #2: same number of examples touched

X_np = RNG.normal(size=(N, D)).astype(np.float32)
w_true = np.array([1.4, -0.8, 0.55, 0.2], dtype=np.float32)
b_true = 0.35
y_np = (X_np @ w_true + b_true + 0.35 * RNG.normal(size=N)).astype(np.float32)

X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np).unsqueeze(1)
loss_fn = nn.MSELoss(reduction="mean")

LABEL_FULL = "full-batch GD"
LABEL_MINI = f"mini-batch GD (b={B})"


def make_model():
    model = nn.Linear(D, 1)
    # Shared init so both methods start from the same place.
    with torch.no_grad():
        model.weight.copy_(torch.zeros(1, D))
        model.bias.copy_(torch.zeros(1))
    return model


@torch.no_grad()
def full_batch_loss(model):
    """Full-batch loss L(... ; w): mean loss on the whole dataset."""
    return float(loss_fn(model(X), y).item())


@torch.no_grad()
def params_vec(model):
    return torch.cat([model.weight.data.reshape(-1), model.bias.data.reshape(-1)]).numpy().copy()


def train_gd(n_sgd_steps, batch_size, lr=LR, seed=0):
    """
    Gradient descent with a fixed batch size.
    batch_size == N  -> full-batch GD
    batch_size <  N  -> mini-batch GD
    """
    model = make_model()
    # torch.optim.SGD here means "plain SGD step"; batch size is our choice.
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    L_hist = [full_batch_loss(model)]
    mini_L_hist = [L_hist[0]]
    params = [params_vec(model)]
    examples_touched = [0]

    for t in range(n_sgd_steps):
        if batch_size >= N:
            xb, yb = X, y
            used = N
        else:
            idx = torch.randperm(N, generator=g)[:batch_size]
            xb, yb = X[idx], y[idx]
            used = batch_size

        mini_L = loss_fn(model(xb), yb)
        opt.zero_grad()
        mini_L.backward()
        opt.step()

        L_hist.append(full_batch_loss(model))
        mini_L_hist.append(float(mini_L.item()))
        params.append(params_vec(model))
        examples_touched.append((t + 1) * used)

    return (
        model,
        np.array(L_hist),
        np.array(mini_L_hist),
        np.array(examples_touched),
        np.stack(params),
    )


def train_until_data_budget(batch_size, budget, lr=LR, seed=0):
    """Run until examples_touched >= budget. Record full-batch loss after each SGD step."""
    model = make_model()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    L_hist = [full_batch_loss(model)]
    examples_touched = [0]
    touched = 0
    n_sgd_steps = 0

    while touched < budget:
        if batch_size >= N:
            xb, yb = X, y
            used = N
        else:
            idx = torch.randperm(N, generator=g)[:batch_size]
            xb, yb = X[idx], y[idx]
            used = batch_size

        mini_L = loss_fn(model(xb), yb)
        opt.zero_grad()
        mini_L.backward()
        opt.step()

        touched += used
        n_sgd_steps += 1
        L_hist.append(full_batch_loss(model))
        examples_touched.append(touched)

    return np.array(L_hist), np.array(examples_touched), n_sgd_steps


# Precompute shared trajectories
print(f"Training {LABEL_FULL} and {LABEL_MINI} ...")
_, L_full, _, _, params_full = train_gd(N_SGD_STEPS, batch_size=N, seed=0)
_, L_mini, mini_L_mini, _, params_mini = train_gd(N_SGD_STEPS, batch_size=B, seed=7)
L_full_budget, ex_full_budget, sgd_steps_full_budget = train_until_data_budget(
    batch_size=N, budget=DATA_BUDGET, seed=1,
)
L_mini_budget, ex_mini_budget, sgd_steps_mini_budget = train_until_data_budget(
    batch_size=B, budget=DATA_BUDGET, seed=7,
)

# Theoretical floor of L(... ; w): best attainable full-batch loss for THIS model on THIS data.
# (Here we can compute it in closed form via least squares only because the model is
# linear and the loss is MSE -- that is a demo convenience, not a general DL fact.)
X1 = np.concatenate([X_np, np.ones((N, 1), dtype=np.float32)], axis=1)
w_star, *_ = np.linalg.lstsq(X1, y_np, rcond=None)  # argmin_w L(... ; w) for this demo
L_floor = float(np.mean((X1 @ w_star - y_np) ** 2))


# -------------------------------------------------------
# FIGURE 1: Same algorithm family -- only batch size changes
# -------------------------------------------------------
fig1, axes1 = new_figure(1, 2)
fig1.suptitle(
    "Step 1: Full-batch GD vs mini-batch GD -- same SGD-step rule, different batch size",
    fontsize=FS_FIGURE_TITLE, fontweight="bold",
)

ax = axes1[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("One SGD step: how much data enters the gradient?", fontsize=FS_AX_TITLE)

boxes = [
    (0.4, 6.2, 4.4, 3.0, "#dceaff",
     f"{LABEL_FULL}\n"
     f"batch size = N = {N}\n"
     "exact gradient of\n"
     "full-batch loss L(... ; w)"),
    (5.2, 6.2, 4.4, 3.0, "#f8d7da",
     f"{LABEL_MINI}\n"
     f"batch size = b = {B}\n"
     "noisy estimate of\n"
     "the same gradient"),
    (0.4, 1.2, 9.2, 4.2, "#d4edda",
     "Both minimize the SAME full-batch loss:\n"
     r"$L(\ldots; w)=\frac{1}{N}\sum_i \ell(f_w(x_i), y_i)$" + "\n"
     "where w = model parameters (weights + bias).\n\n"
     "Mini-batch GD does not change the objective --\n"
     "only how we estimate its gradient each SGD step."),
]
for x0, y0, w, h, color, text in boxes:
    ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor=color, edgecolor="k", lw=2))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center",
            fontsize=FS_ANNOT, fontweight="bold")

ax = axes1[1]
modes = [LABEL_FULL, LABEL_MINI]
cost = [N, B]
sgd_steps_per_pass = [1, N // B]
x = np.arange(len(modes))
width = 0.35
# One color per METRIC (not per method) -- only two series in the legend.
C_EXAMPLES = "#c0392b"   # examples / SGD step
C_STEPS = "#2471a3"      # SGD steps / pass over data
ax.bar(x - width / 2, cost, width, color=C_EXAMPLES, edgecolor="k", lw=0.8)
ax2 = ax.twinx()
ax2.bar(x + width / 2, sgd_steps_per_pass, width, color=C_STEPS, edgecolor="k", lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"full-batch\nGD", f"mini-batch\nGD (b={B})"], fontsize=FS_TICK)
ax.set_ylabel("examples touched per SGD step", fontsize=FS_AXIS, color=C_EXAMPLES)
ax2.set_ylabel("SGD steps per pass over the dataset", fontsize=FS_AXIS, color=C_STEPS)
ax.tick_params(axis="y", labelsize=FS_TICK, colors=C_EXAMPLES)
ax2.tick_params(axis="y", labelsize=FS_TICK, colors=C_STEPS)
ax.set_title("Why mini-batch GD is cheaper per SGD step", fontsize=FS_AX_TITLE)
ax.set_ylim(0, max(cost) * 1.35)
ax2.set_ylim(0, max(sgd_steps_per_pass) * 1.45)
ax.legend(
    handles=[
        Patch(facecolor=C_EXAMPLES, edgecolor="k", label="examples / SGD step"),
        Patch(facecolor=C_STEPS, edgecolor="k", label="SGD steps / pass over data"),
    ],
    loc="upper left", fontsize=FS_LEGEND,
)
add_note(
    ax,
    "Economy:\n"
    f"1 full-batch SGD step\n"
    f"touches {N} examples.\n"
    f"Same cost buys\n"
    f"{N // B} mini-batch SGD steps.",
    x=0.98, y=0.98, ha="right", va="top",
)

fig1.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Same number of SGD steps -- same limit, unstable path
# -------------------------------------------------------
fig2, axes2 = new_figure(1, 2)
fig2.suptitle(
    "Step 2: Same number of SGD steps\n"
    "Mini-batch GD reaches the same limit, but jitters",
    fontsize=FS_FIGURE_TITLE, fontweight="bold",
)

ax = axes2[0]
steps = np.arange(len(L_full))
# BOTH curves are full-batch loss L(... ; w) of each method's current weights.
# The crimson line is NOT the mini-batch loss (see right panel).
ax.plot(steps, L_full, color="k", lw=2.6,
        label=rf"{LABEL_FULL}:  $L(\ldots; w)$ of its weights")
ax.plot(steps, L_mini, color="crimson", lw=1.8, alpha=0.85,
        label=rf"{LABEL_MINI}:  $L(\ldots; w)$ of its weights")
ax.axhline(L_floor, color="0.4", ls="--", lw=1.6,
           label=fr"theoretical floor of $L(\ldots; w)$  ({L_floor:.3f})")
ax.set_xlabel(f"SGD step  (same count = {N_SGD_STEPS} for both)", fontsize=FS_AXIS)
ax.set_ylabel(r"full-batch loss $L(\ldots; w)$  (whole dataset)", fontsize=FS_AXIS)
ax.set_title(
    r"Both curves are $L(\ldots; w)$ -- not mini-batch loss",
    fontsize=FS_AX_TITLE - 2,
)
ax.set_yscale("log")
ax.tick_params(axis="both", labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND - 1, loc="upper right")
ax.grid(True, alpha=0.3, which="both")
add_note(
    ax,
    "Same x: both methods take\n"
    f"exactly {N_SGD_STEPS} SGD steps.\n\n"
    "Same y: L(... ; w) = mean loss on\n"
    "ALL N examples, where w =\n"
    "model parameters. The red\n"
    "line is ALSO L(... ; w), just for\n"
    "parameters from mini-batch GD.\n"
    "It is NOT the mini-batch loss\n"
    "used inside the gradient update.\n\n"
    "Why compute L(... ; w) at all?\n"
    "Not to take the SGD step -- only\n"
    "to see if mini-batch GD is\n"
    "really lowering the objective.\n"
    "Similar final L(... ; w) is the\n"
    "result we discover by scoring\n"
    "both on the same L(... ; w).",
    x=0.98, y=0.50, ha="right", va="center",
)

ax = axes2[1]
ax.plot(steps, mini_L_mini, color="darkorange", lw=1.2, alpha=0.75,
        label="mini-batch loss (current batch only)")
ax.plot(steps, L_mini, color="crimson", lw=2.2, label=r"full-batch loss $L(\ldots; w)$")
ax.axhline(L_floor, color="0.4", ls="--", lw=1.6, label=r"theoretical floor of $L(\ldots; w)$")
ax.set_xlabel("SGD step", fontsize=FS_AXIS)
ax.set_ylabel("loss", fontsize=FS_AXIS)
ax.set_title("Mini-batch loss drives the SGD step; L(... ; w) scores the result", fontsize=FS_AX_TITLE - 2)
ax.set_yscale("log")
ax.tick_params(axis="both", labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc="upper right")
ax.grid(True, alpha=0.3, which="both")
add_note(
    ax,
    "Mini-batch loss: noisy look\n"
    "at b rows -- used to BUILD\n"
    "the gradient for this SGD step.\n\n"
    "Full-batch loss L(... ; w): score\n"
    "of the CURRENT w on all data.\n"
    "Needed for fair comparison,\n"
    "not for taking the SGD step.",
    x=0.98, y=0.50, ha="right", va="center",
)

fig2.tight_layout(rect=[0, 0, 1, 0.90])
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Same data budget -- why mini-batch GD is economic
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle(
    "Step 3: Same data budget -- mini-batch GD buys many cheap SGD steps",
    fontsize=FS_FIGURE_TITLE, fontweight="bold",
)

ax = axes3[0]
ax.plot(ex_full_budget, L_full_budget, color="k", lw=2.6,
        label=f"{LABEL_FULL}  ({sgd_steps_full_budget} SGD steps)")
ax.plot(ex_mini_budget, L_mini_budget, color="crimson", lw=1.9, alpha=0.9,
        label=f"{LABEL_MINI}  ({sgd_steps_mini_budget} SGD steps)")
ax.axhline(L_floor, color="0.4", ls="--", lw=1.6, label=r"theoretical floor of $L(\ldots; w)$")
ax.set_xlabel("examples touched (compute proxy)", fontsize=FS_AXIS)
ax.set_ylabel(r"full-batch loss $L(\ldots; w)$", fontsize=FS_AXIS)
ax.set_title(f"Fixed budget = {DATA_BUDGET} examples touched", fontsize=FS_AX_TITLE)
ax.set_yscale("log")
ax.tick_params(axis="both", labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc="upper right", framealpha=0.95)
ax.grid(True, alpha=0.3, which="both")
add_note(
    ax,
    "x-axis = how much data\n"
    "you paid to touch.\n\n"
    "Mini-batch GD spends that\n"
    "budget on MANY more\n"
    "SGD steps, so it\n"
    "usually reaches the\n"
    "floor of L(... ; w) sooner.",
    x=0.98, y=0.68, ha="right", va="top",
)

ax = axes3[1]
labels = ["after equal\n# SGD steps", "after equal\n# examples touched"]
full_vals = [L_full[-1], L_full_budget[-1]]
mini_vals = [L_mini[-1], L_mini_budget[-1]]
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width / 2, full_vals, width, color="0.25", label=LABEL_FULL)
ax.bar(x + width / 2, mini_vals, width, color="crimson", label=LABEL_MINI)
ax.axhline(L_floor, color="0.4", ls="--", lw=1.6, label=r"theoretical floor of $L(\ldots; w)$")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=FS_TICK)
ax.set_ylabel(r"final full-batch loss $L(\ldots; w)$", fontsize=FS_AXIS)
ax.set_title("Fairness depends on what you hold fixed", fontsize=FS_AX_TITLE)
ax.set_yscale("log")
ax.tick_params(axis="both", labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc="upper left", framealpha=0.95)
ax.grid(True, alpha=0.3, axis="y", which="both")
add_note(
    ax,
    "Equal # SGD steps\n"
    "favors full-batch GD\n"
    "(each SGD step sees all data).\n\n"
    "Equal # examples touched\n"
    "favors mini-batch GD\n"
    "(many cheap SGD steps).\n"
    "Practice budgets by data,\n"
    "not by SGD-step count.",
    x=0.02, y=0.68, ha="left", va="top",
)

fig3.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Late training -- same neighborhood, different stability
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 2)
fig4.suptitle(
    "Step 4: Same limit neighborhood -- full-batch settles; mini-batch wanders",
    fontsize=FS_FIGURE_TITLE, fontweight="bold",
)

ax = axes4[0]
ax.plot(params_full[:, 0], params_full[:, 1], "-", color="k", lw=2.4, label=LABEL_FULL)
ax.plot(params_mini[:, 0], params_mini[:, 1], "-", color="crimson", lw=1.6, alpha=0.85,
        label=LABEL_MINI)
ax.scatter([params_full[0, 0]], [params_full[0, 1]], c="orange", s=80, zorder=5, label="start")
ax.scatter([w_star[0]], [w_star[1]], c="k", s=140, marker="*", zorder=5,
           label=r"minimizer $w^\star=\arg\min_w L(\ldots; w)$")
ax.set_xlabel(r"parameter $w_0$", fontsize=FS_AXIS)
ax.set_ylabel(r"parameter $w_1$", fontsize=FS_AXIS)
ax.set_title(r"Path of parameters $w$ (two coordinates)", fontsize=FS_AX_TITLE)
ax.tick_params(axis="both", labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc="best")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(True, alpha=0.3)
add_note(
    ax,
    "w = model parameters\n"
    "(weights + bias).\n\n"
    "Both walk toward the\n"
    r"same $w^\star$." + "\n"
    "Mini-batch path is wiggly.",
    x=0.02, y=0.02, ha="left", va="bottom",
)

ax = axes4[1]
late = slice(-(N_SGD_STEPS // 3), None)
t_late = np.arange(len(L_full))[late]
ax.plot(t_late, L_full[late], color="k", lw=2.6, label=LABEL_FULL)
ax.plot(t_late, L_mini[late], color="crimson", lw=2.0, label=LABEL_MINI)
ax.axhline(L_floor, color="0.4", ls="--", lw=1.6, label=r"theoretical floor of $L(\ldots; w)$")
ax.set_xlabel("SGD step", fontsize=FS_AXIS)
ax.set_ylabel(r"full-batch loss $L(\ldots; w)$", fontsize=FS_AXIS)
ax.set_title("Late training: stability vs noise floor", fontsize=FS_AX_TITLE)
ax.tick_params(axis="both", labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND, loc="upper right")
ax.grid(True, alpha=0.3)
add_note(
    ax,
    "Full-batch GD settles\n"
    "near the floor of L(... ; w).\n\n"
    "Mini-batch GD oscillates\n"
    "in a ball around it\n"
    "(noise floor).\n\n"
    "That ball is usually close\n"
    "enough -- and each\n"
    "SGD step was much\n"
    "cheaper.",
    x=0.02, y=0.98, ha="left", va="top",
)

fig4.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig4)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("FULL-BATCH GD vs MINI-BATCH GD")
print("=" * 60)
print(f"""
Setup: linear model on N={N} examples (D={D}), MSE full-batch loss L(... ; w).
Theoretical floor of L(... ; w) ~ {L_floor:.4f}
  (via least squares only because this demo is linear+MSE)

Glossary (same as 144):
  w                     model parameters; CONDITION in L(... ; w)
  full-batch GD         batch size = N
  mini-batch GD         batch size = b  (deep learning usually calls this "SGD")
  full-batch loss L(... ; w)  mean loss on the WHOLE dataset
  mini-batch loss       mean loss on the current mini-batch only
  theoretical floor     best attainable L(... ; w) for this model/data
  SGD step         one optimizer.step()
  examples touched      total training rows used so far

After {N_SGD_STEPS} SGD steps:
  {LABEL_FULL:20s}  L(... ; w) = {L_full[-1]:.4f}
  {LABEL_MINI:20s}  L(... ; w) = {L_mini[-1]:.4f}

After {DATA_BUDGET} examples touched:
  {LABEL_FULL:20s}  {sgd_steps_full_budget} SGD steps, L(... ; w) = {L_full_budget[-1]:.4f}
  {LABEL_MINI:20s}  {sgd_steps_mini_budget} SGD steps, L(... ; w) = {L_mini_budget[-1]:.4f}

Takeaways:
  1) Mini-batch GD minimizes the SAME full-batch loss L(... ; w) as full-batch GD.
  2) Convergence is in trend -- the mini-batch curve is not monotone-stable.
  3) Economy: cheap noisy SGD steps buy more progress per unit of data.
  4) Late training: expect a noise floor unless you grow b or decay the lr.

Next: 130_momentum.py (smoother SGD steps), 160_normalization.py (better conditioning).
""")
