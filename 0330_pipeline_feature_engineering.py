"""
0330_pipeline_feature_engineering.py

Data pipeline, features, and feature engineering -- with an end-to-end demo.

Sketch:
  1. What is a data pipeline?
  2. What is a feature?
  3. What is feature engineering? (common moves)
  4. End-to-end setup: raw apartment table -> predict rent
  5. Engineering recipe: raw columns -> model-ready matrix
  6. Before / after: scales and shapes of inputs
  7. Tiny model comparison: raw numerics vs engineered features
  8. Fit transforms on train only -- leakage warning
  9. Pipeline as a reusable chain

Companion to 0320_normalize_data.py, 0260_model.py, 0700_normalize_activations.py,
0240_numpy_torch.py.
"""

from __future__ import annotations

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
from matplotlib.patches import FancyBboxPatch
import numpy as np

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
    rect = FancyBboxPatch(
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
# Shared end-to-end toy: predict monthly rent
# -------------------------------------------------------
N = 220
YEAR_NOW = 2026
NEIGH = np.array(['old_town', 'suburbs', 'dockside'])

# Raw fields (as a landlord's spreadsheet might look)
area_m2 = np.exp(RNG.normal(3.7, 0.35, size=N))          # skewed m^2
year_built = RNG.integers(1955, 2021, size=N)
floor = RNG.integers(0, 12, size=N)
n_floors = np.maximum(floor + RNG.integers(1, 6, size=N), floor + 1)
dist_metro_m = np.clip(RNG.lognormal(mean=6.5, sigma=0.55, size=N), 80, 8000)
neigh = RNG.choice(NEIGH, size=N, p=[0.34, 0.41, 0.25])

# Hidden "truth" rent (model does not see this formula)
neigh_bonus = np.where(
    neigh == 'old_town', 220.0,
    np.where(neigh == 'dockside', 140.0, 40.0),
)
age = YEAR_NOW - year_built
true_rent = (
    8.5 * area_m2
    + 1.8 * np.log(area_m2) * 40.0
    - 3.2 * age
    + 18.0 * (floor / n_floors)
    - 0.045 * dist_metro_m
    + neigh_bonus
    + 900.0
)
true_rent = true_rent + RNG.normal(0.0, 55.0, size=N)
true_rent = np.clip(true_rent, 450.0, None)

# Train / test split (fixed indices)
perm = RNG.permutation(N)
n_train = int(0.7 * N)
idx_tr, idx_te = perm[:n_train], perm[n_train:]


def one_hot(labels, classes):
    out = np.zeros((len(labels), len(classes)), dtype=float)
    for j, c in enumerate(classes):
        out[:, j] = (labels == c).astype(float)
    return out


def standardize_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu, sd


def standardize_apply(X, mu, sd):
    return (X - mu) / sd


def design_raw(area, year, fl, nf, dist):
    """Numeric raw columns only (no categorical, no transforms)."""
    return np.column_stack([area, year, fl, nf, dist])


def design_engineered(area, year, fl, nf, dist, labels, classes, year_now=YEAR_NOW):
    """Feature-engineering recipe used in the demo."""
    age_ = (year_now - year).astype(float)
    log_area = np.log(np.clip(area, 1e-6, None))
    floor_ratio = fl / np.maximum(nf, 1)
    dist_km = dist / 1000.0
    near_metro = (dist < 500).astype(float)
    # mild interaction: large flats near metro are especially valuable
    area_x_near = log_area * near_metro
    oh = one_hot(labels, classes)
    return np.column_stack([
        log_area, age_, floor_ratio, dist_km, near_metro, area_x_near, oh,
    ])


FEAT_NAMES_RAW = ['area_m2', 'year_built', 'floor', 'n_floors', 'dist_m']
FEAT_NAMES_ENG = [
    'log_area', 'age', 'floor_ratio', 'dist_km', 'near_metro',
    'log_area×near', 'oh_old_town', 'oh_suburbs', 'oh_dockside',
]


def fit_linear(X, y):
    """Least squares with intercept column."""
    Phi = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return coef


def predict_linear(X, coef):
    Phi = np.column_stack([np.ones(len(X)), X])
    return Phi @ coef


def rmse(y_hat, y):
    return float(np.sqrt(np.mean((y_hat - y) ** 2)))


# Build matrices
X_raw_all = design_raw(area_m2, year_built, floor, n_floors, dist_metro_m)
X_eng_all = design_engineered(
    area_m2, year_built, floor, n_floors, dist_metro_m, neigh, NEIGH,
)

X_raw_tr, X_raw_te = X_raw_all[idx_tr], X_raw_all[idx_te]
X_eng_tr, X_eng_te = X_eng_all[idx_tr], X_eng_all[idx_te]
y_tr, y_te = true_rent[idx_tr], true_rent[idx_te]

mu_raw, sd_raw = standardize_fit(X_raw_tr)
mu_eng, sd_eng = standardize_fit(X_eng_tr)
Z_raw_tr = standardize_apply(X_raw_tr, mu_raw, sd_raw)
Z_raw_te = standardize_apply(X_raw_te, mu_raw, sd_raw)
Z_eng_tr = standardize_apply(X_eng_tr, mu_eng, sd_eng)
Z_eng_te = standardize_apply(X_eng_te, mu_eng, sd_eng)

coef_raw = fit_linear(Z_raw_tr, y_tr)
coef_eng = fit_linear(Z_eng_tr, y_tr)
pred_raw_te = predict_linear(Z_raw_te, coef_raw)
pred_eng_te = predict_linear(Z_eng_te, coef_eng)
rmse_raw = rmse(pred_raw_te, y_te)
rmse_eng = rmse(pred_eng_te, y_te)


# -------------------------------------------------------
# FIGURE 1: What is a data pipeline?
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    'Step 1: A data pipeline is the path from raw world to model input',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax1)

round_box(ax1, 0.4, 6.35, 11.2, 1.3, C_DARK, C_DARK, lw=0)
ax1.text(
    6.0, 7.0,
    'Pipeline = ordered stages that turn messy reality into tensors the model can use.\n'
    'Same stages should run in training AND in production.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    fontweight='bold', color='white',
)

stages = [
    (0.4, 'Raw\nsources'),
    (2.5, 'Ingest /\nclean'),
    (4.6, 'Feature\nengineering'),
    (6.7, 'Model\ntrain/infer'),
    (8.8, 'Evaluate /\nmonitor'),
    (10.4, 'Serve\npredictions'),
]
for i, (x, title) in enumerate(stages):
    face = C_NEW if i < 3 else (C_GOAL if i == 3 else C_MUTE)
    edge = '#2874a6' if i < 3 else (C_OK if i == 3 else '0.45')
    round_box(ax1, x, 3.4, 1.9, 2.2, face, edge, lw=2.0)
    ax1.text(x + 0.95, 4.5, title, ha='center', va='center',
             fontsize=FS_ANNOT, fontweight='bold')
    if i < len(stages) - 1:
        arrow(ax1, x + 1.95, 4.5, x + 2.4, 4.5, C_DARK, lw=2.0)

round_box(ax1, 0.4, 0.45, 11.2, 2.4, C_OLD, '#e67e22', lw=2.0)
ax1.text(
    6.0, 1.65,
    'Why call it a pipeline?\n'
    'Because later stages assume earlier contracts: types, missingness rules,\n'
    'units, and train-fitted statistics (means, vocabularies, scalers).\n'
    'Break the contract in production and the model silently degrades.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: What is a feature?
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    'Step 2: A feature is one measurable input the model is allowed to use',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax2)

round_box(ax2, 0.5, 5.7, 11.0, 1.9, C_DARK, C_DARK, lw=0)
ax2.text(
    6.0, 6.65,
    'Feature = a column in the design matrix X.\n'
    r'One row = one example;  $X_{i,:}$  is that example\'s feature vector.',
    ha='center', va='center', fontsize=FS_AX_TITLE - 2,
    fontweight='bold', color='white',
)

# Mini design-matrix cartoon
round_box(ax2, 0.5, 1.0, 5.4, 4.2, C_MUTE, '0.45', lw=1.8)
ax2.text(3.2, 4.7, 'Design matrix mindset', ha='center',
         fontsize=FS_ANNOT + 2, fontweight='bold', color=C_DARK)
# table header
headers = ['area', 'age', 'dist_km', 'y']
for j, h in enumerate(headers):
    ax2.text(1.1 + j * 1.15, 3.9, h, ha='center', fontsize=FS_ANNOT,
             fontweight='bold', color=C_DARK)
rows = [
    ['52', '18', '0.4', '1450'],
    ['31', '40', '2.1', '980'],
    ['78', '7', '0.2', '2100'],
]
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        face = C_GOAL if j == 3 else 'white'
        round_box(ax2, 0.7 + j * 1.15, 3.15 - i * 0.7, 1.0, 0.55, face, '0.55', lw=1.2)
        ax2.text(1.2 + j * 1.15, 3.4 - i * 0.7, val, ha='center', va='center',
                 fontsize=FS_ANNOT)

round_box(ax2, 6.3, 1.0, 5.2, 4.2, C_NEW, '#2874a6', lw=2.2)
ax2.text(8.9, 4.7, 'Not every raw field is a good feature',
         ha='center', fontsize=FS_ANNOT + 2, fontweight='bold', color=C_DARK)
ax2.text(
    8.9, 2.7,
    'IDs, timestamps-as-strings,\n'
    'free text, nested JSON, currencies,\n'
    'and mixed units are raw fields.\n\n'
    'A feature is what remains AFTER\n'
    'you decide how the model should\n'
    'see that information.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Feature engineering catalog
# -------------------------------------------------------
fig3, ax3 = new_figure()
fig3.suptitle(
    'Step 3: Feature engineering = invent better X from the raw fields',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax3)

moves = [
    (0.4, 4.4, 'Rescale /\nnormalize', 'log, z-score,\nmin-max, robust'),
    (3.3, 4.4, 'Recode\ncategories', 'one-hot, target\nenc., hashing'),
    (6.2, 4.4, 'Derive /\naggregate', 'age, ratios,\ncounts, windows'),
    (9.1, 4.4, 'Interact /\ncompose', 'products, bins,\ncross features'),
    (0.4, 1.2, 'Parse time\n& text', 'dow, hour,\nTF-IDF, embeds'),
    (3.3, 1.2, 'Handle\nmissing', 'indicator +\nimpute policy'),
    (6.2, 1.2, 'Select /\nreduce', 'drop leaky IDs,\nPCA, filters'),
    (9.1, 1.2, 'Domain\nrules', 'physics units,\nbusiness caps'),
]
for x, y, title, body in moves:
    round_box(ax3, x, y, 2.6, 2.5, C_GOAL, C_OK, lw=2.0)
    ax3.text(x + 1.3, y + 1.75, title, ha='center', va='center',
             fontsize=FS_ANNOT, fontweight='bold', color=C_OK)
    ax3.text(x + 1.3, y + 0.75, body, ha='center', va='center',
             fontsize=FS_ANNOT)
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: End-to-end setup -- raw table
# -------------------------------------------------------
fig4, ax4 = new_figure()
fig4.suptitle(
    'Step 4: Demo setup -- predict rent from a messy apartment spreadsheet',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax4)

round_box(ax4, 0.4, 6.3, 11.2, 1.35, C_DARK, C_DARK, lw=0)
ax4.text(
    6.0, 7.0,
    f'N = {N} apartments  |  train {n_train} / test {N - n_train}\n'
    'Target y = monthly rent.  Raw fields below are what the office stores.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    fontweight='bold', color='white',
)

# Show a few raw rows as a slide table
cols = ['area_m2', 'year_built', 'floor', 'n_floors', 'dist_m', 'neigh', 'rent y']
sample_i = [idx_tr[0], idx_tr[1], idx_tr[2], idx_tr[3], idx_tr[4]]
round_box(ax4, 0.35, 1.55, 11.3, 4.4, C_MUTE, '0.45', lw=1.6)
ax4.text(6.0, 5.55, 'Raw rows (excerpt)', ha='center',
         fontsize=FS_ANNOT + 2, fontweight='bold', color=C_DARK)

x0 = 0.55
widths = [1.45, 1.45, 1.1, 1.25, 1.45, 1.7, 1.55]
# header
x = x0
for w, h in zip(widths, cols):
    round_box(ax4, x, 4.7, w, 0.55, C_NEW, '#2874a6', lw=1.3)
    ax4.text(x + w / 2, 4.97, h, ha='center', va='center',
             fontsize=FS_ANNOT - 1, fontweight='bold')
    x += w + 0.08

for r, i in enumerate(sample_i):
    vals = [
        f'{area_m2[i]:.1f}',
        f'{year_built[i]}',
        f'{floor[i]}',
        f'{n_floors[i]}',
        f'{dist_metro_m[i]:.0f}',
        f'{neigh[i]}',
        f'{true_rent[i]:.0f}',
    ]
    x = x0
    y = 4.0 - r * 0.55
    for w, v in zip(widths, vals):
        face = C_GOAL if w == widths[-1] else 'white'
        round_box(ax4, x, y, w, 0.48, face, '0.55', lw=1.1)
        ax4.text(x + w / 2, y + 0.24, v, ha='center', va='center',
                 fontsize=FS_ANNOT - 1)
        x += w + 0.08

ax4.text(
    6.0, 0.85,
    'Problems already visible: skewed area, huge distance scale (meters),\n'
    'categorical neighborhood, and year_built is not "age".',
    ha='center', va='center', fontsize=FS_ANNOT + 1, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='lightyellow',
              edgecolor='0.55', alpha=0.95),
)
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Engineering recipe
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    'Step 5: Engineering recipe used in this demo',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
blank_slide(ax5)

raw_items = [
    'area_m2', 'year_built', 'floor', 'n_floors', 'dist_metro_m', 'neighborhood',
]
eng_items = [
    'log(area)', 'age = 2026 - year', 'floor / n_floors',
    'dist_km = dist/1000', 'near_metro = 1[dist<500]',
    'log(area) × near_metro', 'one-hot(neighborhood)',
]

round_box(ax5, 0.4, 1.2, 4.4, 5.8, C_OLD, '#e67e22', lw=2.2)
ax5.text(2.6, 6.5, 'RAW fields', ha='center',
         fontsize=FS_AX_TITLE - 2, fontweight='bold', color='#e67e22')
for i, t in enumerate(raw_items):
    ax5.text(2.6, 5.6 - i * 0.7, t, ha='center', va='center',
             fontsize=FS_ANNOT + 1,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       edgecolor='0.55', alpha=0.95))

arrow(ax5, 5.0, 4.2, 6.3, 4.2, C_DARK, lw=3.0)
ax5.text(5.65, 4.7, 'engineer', ha='center', fontsize=FS_ANNOT,
         fontweight='bold', color=C_DARK)

round_box(ax5, 6.5, 0.7, 5.1, 6.5, C_GOAL, C_OK, lw=2.2)
ax5.text(9.05, 6.7, 'ENGINEERED features', ha='center',
         fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_OK)
for i, t in enumerate(eng_items):
    ax5.text(9.05, 5.85 - i * 0.7, t, ha='center', va='center',
             fontsize=FS_ANNOT + 1,
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                       edgecolor=C_OK, alpha=0.95))
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Before / after distributions
# -------------------------------------------------------
fig6, axes6 = new_figure(2, 2)
fig6.suptitle(
    'Step 6: Engineering changes scale and shape -- that is the point',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

ax = axes6[0, 0]
ax.hist(area_m2[idx_tr], bins=22, color='#2874a6', alpha=0.85, edgecolor='white')
ax.set_title('Raw: area_m2 (skewed)', fontsize=FS_AX_TITLE - 2)
ax.set_xlabel('m$^2$', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)

ax = axes6[0, 1]
ax.hist(np.log(area_m2[idx_tr]), bins=22, color=C_OK, alpha=0.85, edgecolor='white')
ax.set_title('Engineered: log(area)', fontsize=FS_AX_TITLE - 2)
ax.set_xlabel('log m$^2$', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)

ax = axes6[1, 0]
ax.hist(dist_metro_m[idx_tr], bins=22, color='#2874a6', alpha=0.85, edgecolor='white')
ax.set_title('Raw: dist_metro_m', fontsize=FS_AX_TITLE - 2)
ax.set_xlabel('meters', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)

ax = axes6[1, 1]
ax.hist(dist_metro_m[idx_tr] / 1000.0, bins=22, color=C_OK, alpha=0.85, edgecolor='white')
ax.axvline(0.5, color=C_ACCENT, lw=2.0, ls='--', label='near_metro threshold')
ax.set_title('Engineered: dist_km (+ threshold)', fontsize=FS_AX_TITLE - 2)
ax.set_xlabel('km', fontsize=FS_AXIS)
ax.tick_params(labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)

for ax in axes6.ravel():
    ax.grid(True, alpha=0.25)

fig6.tight_layout(rect=[0, 0.04, 1, 0.94])
fig6.text(
    0.5, 0.01,
    'Linear models feel units and skew strongly. Transforms are inductive bias about how X should enter f.',
    ha='center', va='bottom', fontsize=FS_ANNOT + 1, fontweight='bold',
)
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Model comparison raw vs engineered
# -------------------------------------------------------
fig7, (ax7a, ax7b) = new_figure(1, 2)
fig7.suptitle(
    'Step 7: Same linear model -- engineered X predicts rent better',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)

lims = [
    min(y_te.min(), pred_raw_te.min(), pred_eng_te.min()) - 50,
    max(y_te.max(), pred_raw_te.max(), pred_eng_te.max()) + 50,
]

ax7a.scatter(y_te, pred_raw_te, s=36, c='#2874a6', alpha=0.8,
             edgecolors='white', linewidths=0.4)
ax7a.plot(lims, lims, color=C_ACCENT, lw=2.0, label='perfect')
ax7a.set_xlim(lims)
ax7a.set_ylim(lims)
ax7a.set_aspect('equal', adjustable='box')
ax7a.set_xlabel('true rent', fontsize=FS_AXIS)
ax7a.set_ylabel('predicted rent', fontsize=FS_AXIS)
ax7a.set_title(f'Raw numerics only\nRMSE = {rmse_raw:.1f}',
               fontsize=FS_AX_TITLE - 2)
ax7a.tick_params(labelsize=FS_TICK)
ax7a.grid(True, alpha=0.25)
ax7a.legend(fontsize=FS_LEGEND, loc='upper left')
ax7a.text(
    0.98, 0.02,
    'Uses area, year, floors,\ndist_m (no neigh, no log/age).',
    transform=ax7a.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax7b.scatter(y_te, pred_eng_te, s=36, c=C_OK, alpha=0.8,
             edgecolors='white', linewidths=0.4)
ax7b.plot(lims, lims, color=C_ACCENT, lw=2.0, label='perfect')
ax7b.set_xlim(lims)
ax7b.set_ylim(lims)
ax7b.set_aspect('equal', adjustable='box')
ax7b.set_xlabel('true rent', fontsize=FS_AXIS)
ax7b.set_ylabel('predicted rent', fontsize=FS_AXIS)
ax7b.set_title(f'Engineered features\nRMSE = {rmse_eng:.1f}',
               fontsize=FS_AX_TITLE - 2)
ax7b.tick_params(labelsize=FS_TICK)
ax7b.grid(True, alpha=0.25)
ax7b.legend(fontsize=FS_LEGEND, loc='upper left')
ax7b.text(
    0.98, 0.02,
    'log/age/ratio/km/near,\ninteraction + one-hot.',
    transform=ax7b.transAxes, ha='right', va='bottom', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor=C_GOAL, edgecolor=C_OK, alpha=0.95),
)

fig7.tight_layout(rect=[0, 0.06, 1, 0.94])
fig7.text(
    0.5, 0.015,
    f'Improvement: RMSE {rmse_raw:.1f}  →  {rmse_eng:.1f}   '
    f'({100.0 * (rmse_raw - rmse_eng) / rmse_raw:.0f}% lower error on the same test set).',
    ha='center', va='bottom', fontsize=FS_ANNOT + 1, fontweight='bold',
)
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Leakage -- fit transforms on train only
# -------------------------------------------------------
fig8, ax8 = new_figure()
fig8.suptitle(
    'Step 8: Fit pipeline statistics on TRAIN only -- then freeze them',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax8)

round_box(ax8, 0.5, 5.5, 5.3, 2.1, C_GOAL, C_OK, lw=2.4)
ax8.text(
    3.15, 6.55, 'Correct',
    ha='center', fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_OK,
)
ax8.text(
    3.15, 5.95,
    '1) split train / test\n'
    '2) fit scaler / vocab on train\n'
    '3) transform train AND test\n'
    '   with those frozen stats',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

round_box(ax8, 6.2, 5.5, 5.3, 2.1, C_OLD, C_ACCENT, lw=2.4)
ax8.text(
    8.85, 6.55, 'Leakage',
    ha='center', fontsize=FS_AX_TITLE - 2, fontweight='bold', color=C_ACCENT,
)
ax8.text(
    8.85, 5.95,
    'Fitting mean/std (or target encoding)\n'
    'on train+test peeks at the future.\n'
    'Test scores become optimistic\n'
    'and production will disappoint.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)

# Show that this demo froze train stats
round_box(ax8, 0.5, 0.5, 11.0, 4.5, C_MUTE, '0.45', lw=1.8)
ax8.text(
    6.0, 4.5, 'In this demo (standardized linear regression)',
    ha='center', fontsize=FS_ANNOT + 2, fontweight='bold', color=C_DARK,
)

# mini numbers
mu_show = ', '.join(f'{v:.1f}' for v in mu_eng[:4])
sd_show = ', '.join(f'{v:.2f}' for v in sd_eng[:4])
ax8.text(
    6.0, 2.7,
    f'Train size = {n_train},  test size = {N - n_train}\n\n'
    f'Engineered feature means (first 4, train-fitted):\n'
    f'  mu ≈ [{mu_show}, ...]\n'
    f'  sd ≈ [{sd_show}, ...]\n\n'
    'Test matrix is transformed with THESE values -- never re-fit on test.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
    family='monospace',
)
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 9: Pipeline as a reusable chain
# -------------------------------------------------------
fig9, ax9 = new_figure()
fig9.suptitle(
    'Step 9: Package the recipe so train and serve share one path',
    fontsize=FS_FIGURE_TITLE - 2, fontweight='bold',
)
blank_slide(ax9)

chain = [
    (0.5, 'raw\nrow'),
    (2.6, 'parse /\nclean'),
    (4.7, 'engineer\nfeatures'),
    (6.8, 'standardize\n(frozen)'),
    (8.9, 'model\nf(x)'),
]
for i, (x, title) in enumerate(chain):
    face = C_NEW if i < 3 else (C_MUTE if i == 3 else C_GOAL)
    edge = '#2874a6' if i < 3 else ('0.45' if i == 3 else C_OK)
    round_box(ax9, x, 4.3, 1.9, 2.2, face, edge, lw=2.2)
    ax9.text(x + 0.95, 5.4, title, ha='center', va='center',
             fontsize=FS_ANNOT, fontweight='bold')
    if i < len(chain) - 1:
        arrow(ax9, x + 2.0, 5.4, x + 2.5, 5.4, C_DARK, lw=2.2)

round_box(ax9, 0.5, 0.6, 11.0, 3.2, 'lightyellow', '0.55', lw=1.8)
ax9.text(
    6.0, 2.2,
    'Feature engineering is not a notebook side quest.\n'
    'It is part of the product: the same transforms must run on every new row.\n\n'
    'Practical rule: if a number is learned from data (mean, vocab, bins),\n'
    'store it with the model artifact and reload it at serving time.',
    ha='center', va='center', fontsize=FS_ANNOT + 1,
)
show_maximized(fig9)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("DATA PIPELINE / FEATURE / FEATURE ENGINEERING")
print("=" * 60)
print(f"""
Step 1 -- data pipeline:
  Ordered stages from raw sources to model I/O.
  Train and production must share the same contracts.

Step 2 -- feature:
  One input coordinate the model may use (a column of X).
  Raw fields become features only after you decide their representation.

Step 3 -- feature engineering:
  Rescale, recode, derive, interact, parse, impute, select, apply domain rules.
  This is inductive bias about how information should enter the model.

Steps 4-7 -- end-to-end rent demo:
  Raw spreadsheet -> engineered matrix -> linear regressor.
  Test RMSE raw numerics:     {rmse_raw:.2f}
  Test RMSE engineered feats: {rmse_eng:.2f}
  Engineered feature names: {FEAT_NAMES_ENG}

Step 8 -- no leakage:
  Fit scaler / vocabularies on TRAIN only; freeze; apply to test / serve.

Step 9 -- ship the pipeline:
  Persist transform parameters with the model.
  Serving a model without its feature recipe is serving a different model.

See also: 0320_normalize_data.py, 0260_model.py, 0700_normalize_activations.py
""")
