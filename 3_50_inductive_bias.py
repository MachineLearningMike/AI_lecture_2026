"""
Intuition for inductive bias: the assumptions a learner brings so it can
generalize beyond the finite training set.

Key idea:
    Many hypotheses fit the training points equally well.
    Inductive bias is what chooses AMONG those hypotheses -- and that
    choice shows up as different predictions where we have no data.

Demos:
    1) Same 5 points, many curves through them
    2) Different model families => different extrapolations
    3) Regularization strength as a soft bias toward simplicity
    4) Bias--variance cartoon: too little vs too much bias

(loss choice is also an inductive bias).
"""

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
    run_slide_deck,
)
apply_rcparams(plt)
import numpy as np

np.random.seed(42)
RNG = np.random.default_rng(42)


def screen_figsize(fraction=SCREEN_FRACTION, dpi=FIGURE_DPI):
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width_px = root.winfo_screenwidth()
        height_px = root.winfo_screenheight()
        root.destroy()
        return (width_px / dpi * fraction, height_px / dpi * fraction)
    except Exception:
        return (16 * fraction / 0.95, 9 * fraction / 0.95)


def new_figure(*subplot_args, **subplot_kwargs):
    subplot_kwargs.setdefault('figsize', screen_figsize())
    subplot_kwargs.setdefault('dpi', FIGURE_DPI)
    return plt.subplots(*subplot_args, **subplot_kwargs)


def show_maximized(fig):
    fig.set_dpi(FIGURE_DPI)
    fig.set_size_inches(screen_figsize(), forward=True)
    fig.canvas.draw_idle()
    manager = getattr(fig.canvas, 'manager', None)
    if manager is not None:
        try:
            manager.window.state('zoomed')
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


def add_vcenter_legend_and_note(ax, note, facecolor='white'):
    ax.legend(
        loc='lower right',
        bbox_to_anchor=(0.98, 0.54),
        fontsize=FS_LEGEND,
        framealpha=0.95,
        borderaxespad=0.0,
    )
    ax.text(
        0.98, 0.46, note,
        transform=ax.transAxes,
        fontsize=FS_ANNOT,
        va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.95),
    )


# -------------------------------------------------------
# Shared toy data: a smooth underlying truth + few samples
# -------------------------------------------------------
def true_fn(x):
    return np.sin(1.3 * x) + 0.3 * x


N_TRAIN = 7
x_train = np.linspace(-2.2, 2.2, N_TRAIN)
y_train = true_fn(x_train) + 0.12 * RNG.normal(size=N_TRAIN)

x_grid = np.linspace(-3.5, 3.5, 400)
y_true = true_fn(x_grid)


def vandermonde(x, degree):
    return np.stack([x ** k for k in range(degree + 1)], axis=1)


def fit_polynomial(x, y, degree, ridge=0.0):
    """Least squares (optionally ridge) polynomial fit."""
    Phi = vandermonde(x, degree)
    if ridge <= 0.0:
        coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    else:
        A = Phi.T @ Phi + ridge * np.eye(degree + 1)
        coef = np.linalg.solve(A, Phi.T @ y)
    return coef


def predict_polynomial(x, coef):
    degree = len(coef) - 1
    return vandermonde(x, degree) @ coef


# -------------------------------------------------------
# FIGURE 1: Many hypotheses fit the same points
# -------------------------------------------------------
def build_step_1():
    fig1, ax1 = new_figure()
    fig1.suptitle(
        'Step 1: Training points alone do not pick a unique function',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    ax1.plot(x_grid, y_true, 'k--', linewidth=2.0, alpha=0.7, label='unknown truth')
    ax1.scatter(x_train, y_train, s=80, c='red', zorder=5, edgecolor='k',
                linewidth=0.6, label='training data')

    # Several interpolating / near-interpolating curves with different biases
    demo_specs = [
        (1, 0.0, 'steelblue', 'linear'),
        (3, 0.0, 'darkorange', 'cubic'),
        (6, 1e-6, 'seagreen', 'deg-6 (almost interpolating)'),
        (6, 3.0, 'purple', 'deg-6 + strong ridge'),
    ]
    for deg, ridge, color, label in demo_specs:
        coef = fit_polynomial(x_train, y_train, deg, ridge=ridge)
        ax1.plot(x_grid, predict_polynomial(x_grid, coef), color=color,
                 linewidth=2.4, label=label)

    ax1.set_xlim(-3.5, 3.5)
    ax1.set_ylim(-4.0, 4.0)
    ax1.set_xlabel('x', fontsize=FS_AXIS)
    ax1.set_ylabel('y', fontsize=FS_AXIS)
    ax1.tick_params(axis='both', labelsize=FS_TICK)
    ax1.set_title('Same data, many stories about the gaps', fontsize=FS_AX_TITLE)
    ax1.legend(
        loc='upper center',
        bbox_to_anchor=(1.0 / 3.0, 0.98),
        fontsize=FS_LEGEND,
        framealpha=0.95,
        borderaxespad=0.0,
    )
    ax1.text(
        2.0 / 3.0, 0.98,
        'Finite data leave the\n'
        'function underdetermined\n'
        'between / beyond points.\n\n'
        'Inductive bias = the\n'
        'extra preference that\n'
        'fills those gaps.',
        transform=ax1.transAxes,
        fontsize=FS_ANNOT,
        va='top', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
    )
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig1


# -------------------------------------------------------
# FIGURE 2: Model family = inductive bias
# -------------------------------------------------------
def build_step_2():
    fig2, axes2 = new_figure(1, 3)
    fig2.suptitle(
        'Step 2: Changing the hypothesis class changes generalization',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    families = [
        (1, 0.0, 'Linear bias\n(prefer straight lines)', 'steelblue'),
        (3, 0.0, 'Low-degree poly bias\n(prefer gentle curves)', 'darkorange'),
        (9, 1e-8, 'Very flexible bias\n(prefer fitting EVERY wiggle)', 'seagreen'),
    ]

    for ax, (deg, ridge, title, color) in zip(axes2, families):
        coef = fit_polynomial(x_train, y_train, deg, ridge=ridge)
        y_hat = predict_polynomial(x_grid, coef)
        ax.plot(x_grid, y_true, 'k--', linewidth=1.8, alpha=0.6, label='truth')
        ax.plot(x_grid, y_hat, color=color, linewidth=2.6, label=f'deg={deg}')
        ax.scatter(x_train, y_train, s=55, c='red', zorder=5, edgecolor='k', linewidth=0.5)
        # train / "test" MSE in-range vs extrapolated region
        mask_in = (x_grid >= x_train.min()) & (x_grid <= x_train.max())
        mask_out = ~mask_in
        err_in = float(np.mean((y_hat[mask_in] - y_true[mask_in]) ** 2))
        err_out = float(np.mean((y_hat[mask_out] - y_true[mask_out]) ** 2))
        ax.set_title(title, fontsize=FS_AX_TITLE - 2)
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-5.5, 5.5)
        ax.set_xlabel('x', fontsize=FS_AXIS)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02, 0.98,
            f'in-support MSE={err_in:.2f}\n'
            f'extrapolation MSE={err_out:.2f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=FS_ANNOT,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.92),
        )
        if ax is axes2[0]:
            ax.set_ylabel('y', fontsize=FS_AXIS)
            ax.legend(loc='lower left', fontsize=FS_LEGEND)

    plt.tight_layout()
    return fig2


# -------------------------------------------------------
# FIGURE 3: Regularization as soft inductive bias
# -------------------------------------------------------
def build_step_3():
    fig3, axes3 = new_figure(1, 2)
    fig3.suptitle(
        r'Step 3: Ridge penalty $\lambda\|w\|^2$ softens flexibility into smoothness',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    DEGREE = 8
    lambdas = [1e-8, 1e-2, 1e-1, 1.0, 10.0]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(lambdas)))

    ax = axes3[0]
    ax.plot(x_grid, y_true, 'k--', linewidth=2.0, alpha=0.65, label='truth')
    ax.scatter(x_train, y_train, s=70, c='red', zorder=5, edgecolor='k',
               linewidth=0.5, label='data')
    for lam, color in zip(lambdas, cmap):
        coef = fit_polynomial(x_train, y_train, DEGREE, ridge=lam)
        ax.plot(x_grid, predict_polynomial(x_grid, coef), color=color,
                linewidth=2.2, label=rf'$\lambda={lam:g}$')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-5.0, 5.0)
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.set_ylabel('y', fontsize=FS_AXIS)
    ax.set_title(rf'deg-{DEGREE} polynomials, varying $\lambda$', fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.legend(
        loc='upper right',
        fontsize=FS_LEGEND,
        framealpha=0.95,
    )
    ax.text(
        0.50, 0.98,
        'Same architecture.\n'
        'Larger lambda =\n'
        'stronger bias toward\n'
        'small weights / smoother\n'
        'functions.',
        transform=ax.transAxes,
        fontsize=FS_ANNOT,
        va='top', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
    )
    ax.grid(True, alpha=0.3)

    ax = axes3[1]
    # Coefficient L2 norms and extrapolation error vs lambda
    coef_norms = []
    extrap_errs = []
    train_errs = []
    mask_out = (x_grid < x_train.min()) | (x_grid > x_train.max())
    for lam in lambdas:
        coef = fit_polynomial(x_train, y_train, DEGREE, ridge=lam)
        y_hat = predict_polynomial(x_grid, coef)
        coef_norms.append(float(np.linalg.norm(coef)))
        train_errs.append(float(np.mean((predict_polynomial(x_train, coef) - y_train) ** 2)))
        extrap_errs.append(float(np.mean((y_hat[mask_out] - y_true[mask_out]) ** 2)))

    ax.plot(lambdas, train_errs, 'o-', color='crimson', linewidth=2.5, label='train MSE')
    ax.plot(lambdas, extrap_errs, 's-', color='royalblue', linewidth=2.5, label='extrapolation MSE')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'regularization $\lambda$', fontsize=FS_AXIS)
    ax.set_ylabel('MSE', fontsize=FS_AXIS)
    ax.set_title('Bias knob: fit vs generalization', fontsize=FS_AX_TITLE)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    add_vcenter_legend_and_note(
        ax,
        'Tiny lambda: memorizes,\n'
        'extrapolation explodes.\n\n'
        'Huge lambda: underfits.\n\n'
        'Good bias is a middle\n'
        'ground -- not zero bias.',
    )
    ax.grid(True, alpha=0.3, which='both')

    # small twin text for coef norms
    ax.text(
        0.02, 0.02,
        'coef L2 norms: ' + ', '.join(f'{n:.1f}' for n in coef_norms),
        transform=ax.transAxes, fontsize=FS_ANNOT - 1, va='bottom', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
    )

    plt.tight_layout()
    return fig3


# -------------------------------------------------------
# FIGURE 4: Bias--variance intuition panel
# -------------------------------------------------------
def build_step_4():
    fig4, axes4 = new_figure(1, 3)
    fig4.suptitle(
        'Step 4: Inductive bias is unavoidable -- choose it deliberately',
        fontsize=FS_FIGURE_TITLE, fontweight='bold',
    )

    # Repeated datasets: show variance of flexible model vs bias of rigid model
    N_REPEATS = 25
    x_dense = np.linspace(-2.2, 2.2, 80)

    panel_cfgs = [
        (1, 0.0, 'High bias / low variance\n(linear)', 'steelblue'),
        (4, 0.05, 'Balanced bias\n(deg-4 + mild ridge)', 'darkorange'),
        (9, 1e-10, 'Low bias / high variance\n(deg-9, almost free)', 'seagreen'),
    ]

    for ax, (deg, ridge, title, color) in zip(axes4, panel_cfgs):
        preds = []
        for _ in range(N_REPEATS):
            y_rep = true_fn(x_train) + 0.25 * RNG.normal(size=N_TRAIN)
            coef = fit_polynomial(x_train, y_rep, deg, ridge=ridge)
            preds.append(predict_polynomial(x_dense, coef))
        preds = np.stack(preds, axis=0)
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)

        ax.plot(x_grid, y_true, 'k--', linewidth=1.6, alpha=0.7, label='truth')
        for i in range(min(12, N_REPEATS)):
            ax.plot(x_dense, preds[i], color=color, alpha=0.25, linewidth=1.0)
        ax.plot(x_dense, mean_pred, color=color, linewidth=2.6, label='avg predictor')
        ax.fill_between(x_dense, mean_pred - std_pred, mean_pred + std_pred,
                         color=color, alpha=0.2, label='±1 std')
        ax.scatter(x_train, true_fn(x_train), s=40, c='red', zorder=5, edgecolor='k',
                   linewidth=0.4)
        ax.set_title(title, fontsize=FS_AX_TITLE - 2)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-4.0, 4.0)
        ax.set_xlabel('x', fontsize=FS_AXIS)
        ax.tick_params(axis='both', labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)
        if ax is axes4[0]:
            ax.set_ylabel('y', fontsize=FS_AXIS)
            ax.legend(loc='lower left', fontsize=FS_LEGEND - 1)

    plt.tight_layout()
    return fig4


run_slide_deck([
    build_step_1,
    build_step_2,
    build_step_3,
    build_step_4,
])

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("INDUCTIVE BIAS -- INTUITION")
print("=" * 60)
print(f"""
Definition (practical):
  Inductive bias = the preferences / assumptions that let a learner
  pick ONE hypothesis when many fit the training data.

Why we need it:
  With only {N_TRAIN} points, infinitely many curves pass through them.
  Without a bias, there is no reason to prefer any extrapolation.

Where bias comes from in ML:
  1) Hypothesis class   (linear vs deep net vs tree vs kNN)
  2) Architecture       (convolution => locality/translation structure)
  3) Regularization     (ridge/weight decay => prefer small weights)
  4) Loss function (MSE vs MAE; see )
  5) Optimizer / init (implicit bias of SGD; see )

Lesson from the figures:
  - Too little bias (very flexible fit): great train fit, wild extrapolation
  - Too much bias (too rigid): stable but systematically wrong
  - Useful learning = a bias that matches the structure of the world

No free lunch:
  No universal best bias. The "right" inductive bias depends on the
  problem -- which is why architecture and priors matter so much.
""")
