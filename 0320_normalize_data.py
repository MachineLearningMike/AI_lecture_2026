"""
0320_normalize_data.py

Normalize a distribution -- continuation after 0310_distribution.py.

Sketch:
  9. The same dog/cat clusters, tiny in a corner -- then normalized;
     framing questions on scale, information, and boundaries.
  10. Normalize dataset manually with sliders a, b.
  11. Why N(0, 1)? -- activations live near zero-scale numbers.
  12. Normalization with NumPy or Torch.

See also: 0310_distribution.py (why distribution), 0720_discriminative_generative.py.
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
)
apply_rcparams(plt)
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import numpy as np

np.random.seed(42)


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
    kwargs.setdefault('figsize', screen_figsize())
    kwargs.setdefault('dpi', FIGURE_DPI)
    return plt.subplots(*args, **kwargs)


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


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')


def round_box(ax, x, y, w, h, facecolor, edgecolor='0.2', lw=2.0):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.02,rounding_size=0.15',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(rect)
    return rect


def arrow(ax, x1, y1, x2, y2, color='0.25', lw=2.5):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                        mutation_scale=18),
    )


# Palette
C_DARK = '#1a5276'
C_NEW = '#d6eaf8'
C_OK = '#1e8449'
C_ACCENT = '#c0392b'
C_MUTE = '#f4f6f7'
C_GOAL = '#d4edda'
C_OLD = '#fce4d6'




# -------------------------------------------------------
# Shared dog / cat clusters (same recipe as 040 Step 8)
# -------------------------------------------------------
rng8 = np.random.default_rng(3)
dogs = rng8.normal(loc=[-1.2, -0.3], scale=[0.55, 0.45], size=(120, 2))
cats = rng8.normal(loc=[1.3, 0.4], scale=[0.50, 0.48], size=(120, 2))
dog_c = dogs.mean(axis=0)
cat_c = cats.mean(axis=0)


# FIGURE 9: Corner view + normalized view -- what should we do?
# -------------------------------------------------------
fig9, (ax9, axq) = new_figure(1, 2, gridspec_kw={'width_ratios': [2.35, 1.0]})
fig9.suptitle(
    'Step 9: The same distribution -- from a corner, then normalized',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

# Shrink and translate the Step-8 clusters into the lower-left of a wide frame
scale9 = 0.22
shift9 = np.array([-3.2, -3.2])
dogs9 = dogs * scale9 + shift9
cats9 = cats * scale9 + shift9
dog_c9 = dogs9.mean(axis=0)
cat_c9 = cats9.mean(axis=0)

# Corner (shrunk) copy
ax9.scatter(
    dogs9[:, 0], dogs9[:, 1], s=18, c='#1f77b4', alpha=0.75,
    edgecolors='white', linewidths=0.25, zorder=3,
)
ax9.scatter(
    cats9[:, 0], cats9[:, 1], s=18, c='#2ca02c', alpha=0.75,
    edgecolors='white', linewidths=0.25, zorder=3,
)
# Normalized copy (same points as Step 8)
ax9.scatter(
    dogs[:, 0], dogs[:, 1], s=36, c='#1f77b4', alpha=0.85,
    edgecolors='white', linewidths=0.4, zorder=4,
)
ax9.scatter(
    cats[:, 0], cats[:, 1], s=36, c='#2ca02c', alpha=0.85,
    edgecolors='white', linewidths=0.4, zorder=4,
)

ax9.annotate(
    'Dog photos', xy=dog_c, xytext=(dog_c[0] - 0.15, dog_c[1] + 1.15),
    fontsize=FS_ANNOT + 1, fontweight='bold', color='#1f77b4', ha='center',
    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#1f77b4', alpha=0.95),
)
ax9.annotate(
    'Cat photos', xy=cat_c, xytext=(cat_c[0] + 0.15, cat_c[1] + 1.15),
    fontsize=FS_ANNOT + 1, fontweight='bold', color='#2ca02c', ha='center',
    arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
    bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#2ca02c', alpha=0.95),
)
ax9.annotate(
    'As recorded\n(tiny, in a corner)',
    xy=((dog_c9 + cat_c9) / 2),
    xytext=(dog_c9[0] - 0.15, dog_c9[1] + 1.15),
    fontsize=FS_ANNOT - 1, color='0.25', ha='center',
    arrowprops=dict(arrowstyle='->', color='0.45', lw=1.2),
    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='0.55', alpha=0.95),
)

# Bridge label: corner -> normalized
mid_norm = (dog_c + cat_c) / 2
ax9.annotate(
    'Normalize the distribution\nfor convenience',
    xy=mid_norm, xytext=(-0.4, -2.55),
    fontsize=FS_ANNOT + 1, fontweight='bold', color=C_DARK, ha='center',
    arrowprops=dict(arrowstyle='->', color=C_DARK, lw=2.0,
                    connectionstyle='arc3,rad=0.15'),
    bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
              edgecolor=C_DARK, linewidth=2.0, alpha=0.95),
)

ax9.set_xlim(-4.5, 4.5)
ax9.set_ylim(-4.5, 4.5)
ax9.set_xlabel('feature 1', fontsize=FS_AXIS)
ax9.set_ylabel('feature 2', fontsize=FS_AXIS)
ax9.tick_params(labelsize=FS_TICK)
ax9.grid(True, alpha=0.25)
ax9.set_aspect('equal', adjustable='box')
ax9.axhline(0, color='0.7', lw=0.8)
ax9.axvline(0, color='0.7', lw=0.8)

# Questions live outside the plot axes
axq.set_xlim(0, 1)
axq.set_ylim(0, 1)
axq.axis('off')
axq.text(
    0.98, 0.98,
    'Questions for the room:\n\n'
    '1. Would it help to bring these\n'
    '    clusters to the center and\n'
    '    enlarge them, so we can\n'
    '    study how they relate?\n\n'
    '2. If we do, do we lose the\n'
    '    information that tells cats\n'
    '    from dogs -- or do we merely\n'
    '    change the frame in which\n'
    '    we look?\n\n'
    '3. Beyond discriminating one\n'
    '    cluster from the other,\n'
    '    might we learn the boundary\n'
    '    of each cluster in its own\n'
    '    right?',
    transform=axq.transAxes, ha='right', va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round,pad=0.55', facecolor='lightyellow',
              edgecolor='0.55', alpha=0.98),
    multialignment='right',
)
fig9.tight_layout(rect=[0, 0, 1, 0.94])
show_maximized(fig9)


# -------------------------------------------------------
# FIGURE 10: Normalize dataset manually with sliders
# -------------------------------------------------------
fig10, (ax10_raw, ax10_z) = new_figure(1, 2)
fig10.suptitle(
    'Step 10: Normalize dataset manually',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
fig10.subplots_adjust(bottom=0.22, wspace=0.28)

rng10 = np.random.default_rng(7)
MU_DATA, SIG_DATA = 48.0, 2.2
X10 = rng10.normal(loc=MU_DATA, scale=SIG_DATA, size=(36, 2))
mean_x = float(X10.mean())
std_x = float(X10.std(ddof=0))

ax10_raw.scatter(X10[:, 0], X10[:, 1], s=55, c='#2874a6', alpha=0.85,
                 edgecolors='white', linewidths=0.6, zorder=3)
ax10_raw.axhline(mean_x, color='0.4', ls='--', lw=1.2, alpha=0.7)
ax10_raw.axvline(mean_x, color='0.4', ls='--', lw=1.2, alpha=0.7)
ax10_raw.set_title('Raw points (large mean, small std)', fontsize=FS_AX_TITLE - 2)
ax10_raw.set_xlabel(r'$x_1$', fontsize=FS_AXIS)
ax10_raw.set_ylabel(r'$x_2$', fontsize=FS_AXIS)
ax10_raw.tick_params(labelsize=FS_TICK)
ax10_raw.grid(True, alpha=0.25)
ax10_raw.set_aspect('equal', adjustable='datalim')
ax10_raw.text(
    0.02, 0.98,
    f'mean(x) = {mean_x:.2f}\n'
    f'std(x)  = {std_x:.2f}',
    transform=ax10_raw.transAxes, va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

sc10 = ax10_z.scatter([], [], s=55, c='#1e8449', alpha=0.85,
                      edgecolors='white', linewidths=0.6, zorder=3)
ax10_z.axhline(0, color='0.5', lw=1.0)
ax10_z.axvline(0, color='0.5', lw=1.0)
ax10_z.set_title(r'Transformed $z=(x-a)/b$', fontsize=FS_AX_TITLE - 2)
ax10_z.set_xlabel(r'$z_1$', fontsize=FS_AXIS)
ax10_z.set_ylabel(r'$z_2$', fontsize=FS_AXIS)
ax10_z.tick_params(labelsize=FS_TICK)
ax10_z.grid(True, alpha=0.25)
ax10_z.set_aspect('equal', adjustable='datalim')
hud10 = ax10_z.text(
    0.02, 0.98, '',
    transform=ax10_z.transAxes, va='top', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
)
note10 = ax10_z.text(
    0.98, 0.02, '',
    transform=ax10_z.transAxes, va='bottom', ha='right', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
)

ax_a = fig10.add_axes([0.15, 0.10, 0.70, 0.035])
ax_b = fig10.add_axes([0.15, 0.045, 0.70, 0.035])
slider_a = Slider(ax_a, r'$a$', 0.0, 80.0, valinit=1.0, valstep=0.1)
slider_b = Slider(ax_b, r'$b$', 0.0, 20.0, valinit=0.0, valstep=0.1)


def _update_norm(_=None):
    a = float(slider_a.val)
    b = float(slider_b.val)
    if abs(b) < 1e-8:
        sc10.set_offsets(np.empty((0, 2)))
        hud10.set_text(
            rf'$a={a:.2f},\ b={b:.2f}$' + '\n'
            'b = 0  =>  division undefined.\n'
            'Move b away from 0.'
        )
        note10.set_text('Best target: a = mean(x), b = std(x)')
        ax10_z.relim()
        ax10_z.autoscale_view()
        fig10.canvas.draw_idle()
        return
    Z = (X10 - a) / b
    sc10.set_offsets(Z)
    z_mean = float(Z.mean())
    z_std = float(Z.std(ddof=0))
    hud10.set_text(
        rf'$a={a:.2f},\ b={b:.2f}$' + '\n'
        f'mean(z) = {z_mean:.3f}\n'
        f'std(z)  = {z_std:.3f}'
    )
    near = abs(a - mean_x) < 0.35 and abs(b - std_x) < 0.25
    if near:
        note10.set_text(
            'Best setting reached:\n'
            'a ≈ mean(x), b ≈ std(x)\n'
            '=> z ≈ N(0, 1)'
        )
    else:
        note10.set_text(
            f'Try a → {mean_x:.2f} (= mean)\n'
            f'Try b → {std_x:.2f} (= std)\n'
            'to get z ≈ N(0, 1)'
        )
    pad = max(3.0, 3.0 * abs(b) if abs(b) < 1 else 3.0)
    ax10_z.set_xlim(Z[:, 0].mean() - pad, Z[:, 0].mean() + pad)
    ax10_z.set_ylim(Z[:, 1].mean() - pad, Z[:, 1].mean() + pad)
    fig10.canvas.draw_idle()


slider_a.on_changed(_update_norm)
slider_b.on_changed(_update_norm)
_update_norm()
print("Step 10: drag a and b; best values are a=mean(x), b=std(x).")
show_maximized(fig10)


# -------------------------------------------------------
# FIGURE 11: Why N(0, 1)?
# -------------------------------------------------------
fig11, axes11 = new_figure(1, 3)
fig11.suptitle(
    'Step 11: Why N(0, 1)?',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

xs11 = np.linspace(-4.5, 4.5, 500)
gauss11 = np.exp(-0.5 * xs11 ** 2) / np.sqrt(2 * np.pi)
acts = [
    (
        'sigmoid',
        r'$\sigma(x)=\dfrac{1}{1+e^{-x}}$',
        1.0 / (1.0 + np.exp(-xs11)),
        '#2874a6',
        'Alive near 0;\nsaturates for |x| large.',
    ),
    (
        'tanh',
        r'$\tanh(x)=\dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$',
        np.tanh(xs11),
        '#1e8449',
        'Centered at 0;\nsaturates outside ~[-2, 2].',
    ),
    (
        'ReLU',
        r'$\mathrm{ReLU}(x)=\max(0,x)$',
        np.maximum(0.0, xs11),
        '#c0392b',
        'Half of N(0,1) is active\n(x > 0); half is silent.',
    ),
]

for i, (ax, (name, eq, ys, color, tip)) in enumerate(zip(axes11, acts)):
    ax.axvspan(-2.0, 2.0, color='#d5f5e3', alpha=0.55, label='N(0,1) bulk')
    ax.plot(xs11, ys, color=color, lw=2.6, label=name)
    if name == 'ReLU':
        # Show the correct derivative on a twin axis (step: 0 / 1)
        ax2 = ax.twinx()
        relu_grad = (xs11 > 0).astype(float)
        ax2.plot(xs11, relu_grad, color='#7d3c98', lw=2.2, label=r"ReLU'")
        ax2.set_ylim(-0.05, 1.35)
        ax2.set_ylabel(r"ReLU'", fontsize=FS_AXIS, color='#7d3c98')
        ax2.tick_params(axis='y', labelsize=FS_TICK, colors='#7d3c98')
    else:
        ax.plot(xs11, gauss11 / gauss11.max() * max(float(ys.max()), 1.0),
                color='0.45', lw=1.6, ls='--', alpha=0.85, label='N(0,1) shape')
    ax.axvline(0, color='0.5', lw=0.9, alpha=0.7)
    ax.axhline(0, color='0.5', lw=0.9, alpha=0.7)
    ax.set_title(name, fontsize=FS_AX_TITLE - 1)
    ax.set_xlabel('x', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-4.5, 4.5)
    # Left and middle panels: annotation at bottom-right; right panel keeps top-left
    if i < 2:
        ax.text(
            0.98, 0.02, eq + '\n\n' + tip,
            transform=ax.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
        )
    else:
        ax.text(
            0.02, 0.98, eq + '\n\n' + tip,
            transform=ax.transAxes, va='top', fontsize=FS_ANNOT,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
        )

fig11.text(
    0.5, 0.02,
    'If numbers inside the model live near N(0, 1),\n'
    'they are likely to fall into the living region of popular activations -- not the dead flat tails.',
    ha='center', va='bottom', fontsize=FS_ANNOT + 1, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.35', facecolor=C_GOAL, edgecolor=C_OK, alpha=0.96),
)
fig11.tight_layout(rect=[0, 0.08, 1, 0.94])
show_maximized(fig11)


# -------------------------------------------------------
# FIGURE 12: Normalization with NumPy or Torch
# -------------------------------------------------------
fig12, ax12 = new_figure()
fig12.suptitle(
    'Step 12: Normalization with NumPy or Torch',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax12)

round_box(ax12, 0.5, 5.55, 11.0, 1.85, C_DARK, C_DARK, lw=0)
ax12.text(
    6.0, 6.45,
    r'Same recipe as the sliders:  $z = (x - \mathrm{mean})/\mathrm{std}$' + '\n'
    'Now written in one line -- in NumPy or in Torch.',
    ha='center', va='center', fontsize=FS_AX_TITLE - 2,
    fontweight='bold', color='white',
)

# NumPy panel
round_box(ax12, 0.5, 1.55, 5.3, 3.7, C_NEW, '#2874a6', lw=2.2)
ax12.text(
    3.15, 4.85, 'NumPy',
    ha='center', va='center', fontsize=FS_AX_TITLE - 1,
    fontweight='bold', color=C_DARK,
)
ax12.text(
    3.15, 3.25,
    'import numpy as np\n\n'
    'x = np.asarray(x, dtype=float)\n'
    'z = (x - x.mean()) / x.std()',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    family='monospace',
    bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
              edgecolor='#2874a6', alpha=0.98),
)

# Torch panel
round_box(ax12, 6.2, 1.55, 5.3, 3.7, C_GOAL, C_OK, lw=2.2)
ax12.text(
    8.85, 4.85, 'PyTorch',
    ha='center', va='center', fontsize=FS_AX_TITLE - 1,
    fontweight='bold', color=C_OK,
)
ax12.text(
    8.85, 3.25,
    'import torch\n\n'
    'x = torch.as_tensor(x, dtype=torch.float)\n'
    'z = (x - x.mean()) / x.std()',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    family='monospace',
    bbox=dict(boxstyle='round,pad=0.45', facecolor='white',
              edgecolor=C_OK, alpha=0.98),
)

round_box(ax12, 0.5, 0.35, 11.0, 0.95, 'lightyellow', '0.55', lw=1.6)
ax12.text(
    6.0, 0.8,
    'Per-feature / per-channel: mean and std along the right axis '
    r'(e.g. dim=0).  Watch for std $= 0$.',
    ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
)

show_maximized(fig12)


# -------------------------------------------------------

# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("NORMALIZE A DISTRIBUTION")
print("=" * 60)
print("""
Framing questions (Step 9):
  Would it help to bring these clusters to the center and enlarge them,
  so we can study how they relate?
  If we do, do we lose the information that tells cats from dogs --
  or do we merely change the frame in which we look?
  Beyond discriminating one cluster from the other, might we learn
  the boundary of each cluster in its own right?

Normalize manually (Step 10):
  z = (x - a) / b with sliders. Best: a = mean(x), b = std(x) => z ≈ N(0, 1).

Why N(0, 1)? (Step 11):
  Sigmoid, tanh, and ReLU are most useful near zero-scale inputs.

Normalization in code (Step 12):
  NumPy:  z = (x - x.mean()) / x.std()
  Torch:  z = (x - x.mean()) / x.std()

See also: 0310_distribution.py, 0720_discriminative_generative.py
""")
