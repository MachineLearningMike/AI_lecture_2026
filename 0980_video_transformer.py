"""
0980_video_transformer.py

Video Transformer intuition.

A video is a 3D volume: time x height x width (plus channels).
Video Transformers turn that volume into tokens, then run attention
over space and/or time.

Central design tension:
    Joint space-time attention is expressive but quadratic in (T*N_patches).
    Factorized / divided attention (TimeSformer-style) is much cheaper:
        1) attend over space within each frame
        2) attend over time within each spatial location

Companion to 0850_transformer.py (token attention basics).
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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


def add_note(ax, text, x=0.98, y=0.46, ha='right', va='top', facecolor='white'):
    ax.text(
        x, y, text, transform=ax.transAxes, fontsize=FS_ANNOT,
        va=va, ha=ha,
        bbox=dict(boxstyle='round', facecolor=facecolor, alpha=0.95),
    )


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# -------------------------------------------------------
# Toy video: moving bright blob across frames
# shape (T, H, W)
# -------------------------------------------------------
T, H, W = 6, 8, 8
PATCH = 2          # spatial patch size
TUBE_T = 2         # temporal tubelet extent (ViViT-style)

video = np.zeros((T, H, W), dtype=float)
for t in range(T):
    cy, cx = 2 + t // 2, 1 + t
    yy, xx = np.ogrid[:H, :W]
    video[t] = np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / 1.2)


def extract_spatial_patches(frame, patch=PATCH):
    """frame (H,W) -> (n_h*n_w, patch*patch)."""
    n_h, n_w = H // patch, W // patch
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patches.append(frame[i * patch:(i + 1) * patch,
                                 j * patch:(j + 1) * patch].reshape(-1))
    return np.stack(patches, axis=0), (n_h, n_w)


def extract_tubelets(video, patch=PATCH, tube_t=TUBE_T):
    """
    video (T,H,W) -> tokens (n_t*n_h*n_w, tube_t*patch*patch)
    and index map of (t_idx, i, j) for each token.
    """
    n_t = T // tube_t
    n_h, n_w = H // patch, W // patch
    tokens = []
    index = []
    for ti in range(n_t):
        for i in range(n_h):
            for j in range(n_w):
                tube = video[
                    ti * tube_t:(ti + 1) * tube_t,
                    i * patch:(i + 1) * patch,
                    j * patch:(j + 1) * patch,
                ].reshape(-1)
                tokens.append(tube)
                index.append((ti, i, j))
    return np.stack(tokens, axis=0), index, (n_t, n_h, n_w)


spatial_tokens_f0, (n_h, n_w) = extract_spatial_patches(video[0])
tube_tokens, tube_index, (n_t, n_h, n_w) = extract_tubelets(video)
N_SPATIAL = n_h * n_w
N_TUBE = tube_tokens.shape[0]


# -------------------------------------------------------
# FIGURE 1: Video as a space-time volume
# -------------------------------------------------------
fig1 = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
fig1.suptitle(
    'Step 1: A video is a volume -- time stacks frames of space',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax = fig1.add_subplot(111, projection='3d')

# draw each frame as a translucent plane with the blob
for t in range(T):
    yy = np.arange(H)
    xx = np.arange(W)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.full_like(XX, t, dtype=float)
    ax.plot_surface(
        XX, YY, ZZ, facecolors=plt.cm.magma(video[t] / video.max()),
        rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.55,
        shade=False,
    )

ax.set_xlabel('width x', fontsize=FS_AXIS - 2)
ax.set_ylabel('height y', fontsize=FS_AXIS - 2)
ax.set_zlabel('time t', fontsize=FS_AXIS - 2)
ax.set_title('Toy clip: a blob moves through space-time', fontsize=FS_AX_TITLE)
ax.tick_params(labelsize=FS_ANNOT)
ax.text2D(
    0.02, 0.95,
    'Image Transformer: tokens in 2D space\n'
    'Video Transformer: tokens in 3D space-time',
    transform=ax.transAxes, fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)
plt.tight_layout()
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Frame patches vs space-time tubelets
# -------------------------------------------------------
fig2, axes2 = new_figure(1, 2)
fig2.suptitle(
    'Step 2: Tokenization -- patches (per frame) vs tubelets (space-time)',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes2[0]
ax.imshow(video[0], cmap='magma', vmin=0, vmax=video.max())
# draw patch grid
for i in range(0, H + 1, PATCH):
    ax.axhline(i - 0.5, color='cyan', lw=1.2)
for j in range(0, W + 1, PATCH):
    ax.axvline(j - 0.5, color='cyan', lw=1.2)
ax.set_title(f'Spatial patches on one frame\n({N_SPATIAL} tokens / frame)', fontsize=FS_AX_TITLE)
ax.set_xlabel('x', fontsize=FS_AXIS)
ax.set_ylabel('y', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK)
add_note(
    ax,
    'ViT-style:\n'
    'patchify each frame,\n'
    'then somehow fuse time.\n'
    f'Total if naive: T*N = {T * N_SPATIAL}',
    y=0.02, va='bottom', facecolor='lightyellow',
)

ax = axes2[1]
# show a small tubelet schematically: 2 frames x 2x2 patch
ax.set_xlim(0, 6)
ax.set_ylim(0, 5)
ax.axis('off')
ax.set_title(f'Space-time tubelet\n({tube_tokens.shape[1]} values / token)', fontsize=FS_AX_TITLE)
# two stacked patch squares
for t_off, y0 in enumerate([2.6, 1.0]):
    for ii in range(PATCH):
        for jj in range(PATCH):
            val = video[t_off, 2 + ii, 2 + jj] / video.max()
            rect = patches.Rectangle(
                (2.2 + jj, y0 + (PATCH - 1 - ii) * 0.7), 0.7, 0.7,
                facecolor=plt.cm.magma(val), edgecolor='cyan', lw=1.5,
            )
            ax.add_patch(rect)
    ax.text(1.5, y0 + 0.6, f't={t_off}', fontsize=FS_ANNOT, ha='right')
ax.annotate('', xy=(3.0, 2.5), xytext=(3.0, 2.0),
            arrowprops=dict(arrowstyle='<->', color='crimson', lw=2))
ax.text(3.7, 2.2, 'time', color='crimson', fontsize=FS_ANNOT)
ax.text(
    0.5, 0.08,
    f'ViViT-style tubelets: one token spans {TUBE_T} frames x {PATCH}x{PATCH} pixels.\n'
    f'Number of tubelet tokens here: {N_TUBE}  (vs {T * N_SPATIAL} frame patches).',
    transform=ax.transAxes, ha='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig2)


# -------------------------------------------------------
# Attention helpers on toy tokens
# -------------------------------------------------------
def attention_map(tokens):
    """Simple content attention: softmax of scaled cosine similarities."""
    # tokens (N, d)
    Xn = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8)
    scores = (Xn @ Xn.T) / np.sqrt(tokens.shape[1])
    return softmax(scores, axis=-1)


# Build frame-patch token sequence for all time: (T, N_spatial, d)
frame_tokens = []
for t in range(T):
    toks, _ = extract_spatial_patches(video[t])
    frame_tokens.append(toks)
frame_tokens = np.stack(frame_tokens, axis=0)  # (T, N, d)

# Divided: space-only within a frame, time-only within a patch location
attn_space = attention_map(frame_tokens[2])  # frame t=2 spatial attention

# time attention for a fixed spatial patch (center-ish)
patch_id = (n_h // 2) * n_w + (n_w // 2)
time_series = frame_tokens[:, patch_id, :]  # (T, d)
attn_time = attention_map(time_series)


def cost_joint(t, n):
    return (t * n) ** 2


def cost_divided(t, n):
    # space attention per frame: T * N^2 , time attention per location: N * T^2
    return t * (n ** 2) + n * (t ** 2)


# -------------------------------------------------------
# FIGURE 3: Joint vs divided space-time attention
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 3)
fig3.suptitle(
    'Step 3: Joint attention vs divided (factorized) space-time attention',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes3[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Joint space-time attention', fontsize=FS_AX_TITLE)
# tokens as grid of points over t and space index
for t in range(4):
    for s in range(5):
        ax.scatter(1.5 + s * 1.3, 6.2 - t * 1.3, s=120, c='royalblue', edgecolor='k')
# many connections from one query
qi, qj = 2, 1
qx, qy = 1.5 + qj * 1.3, 6.2 - qi * 1.3
for t in range(4):
    for s in range(5):
        ax.plot([qx, 1.5 + s * 1.3], [qy, 6.2 - t * 1.3],
                color='crimson', alpha=0.25, lw=0.8)
ax.scatter([qx], [qy], s=160, c='crimson', edgecolor='k', zorder=5)
ax.text(
    0.5, 0.08,
    f'Each token attends all T·N others\nCost ~ (T N)^2 = {cost_joint(T, N_SPATIAL)}',
    transform=ax.transAxes, ha='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes3[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Divided: spatial attention', fontsize=FS_AX_TITLE)
# one frame row fully connected
y = 4.5
xs = [1.5 + s * 1.5 for s in range(5)]
for x in xs:
    ax.scatter(x, y, s=130, c='seagreen', edgecolor='k', zorder=3)
for i in range(5):
    for j in range(5):
        ax.plot([xs[i], xs[j]], [y, y], color='seagreen', alpha=0.35, lw=1.0)
ax.text(5, 6.2, 'within a single frame', ha='center', fontsize=FS_ANNOT)
ax.text(
    0.5, 0.12,
    f'Per frame: N^2\nOver T frames: T N^2 = {T * N_SPATIAL ** 2}',
    transform=ax.transAxes, ha='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

ax = axes3[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Divided: temporal attention', fontsize=FS_AX_TITLE)
xs = 5.0
ys = [6.5 - t * 1.2 for t in range(5)]
for y in ys:
    ax.scatter(xs, y, s=130, c='darkorange', edgecolor='k', zorder=3)
for i in range(5):
    for j in range(5):
        ax.plot([xs, xs], [ys[i], ys[j]], color='darkorange', alpha=0.35, lw=1.0)
ax.text(7.2, 4.5, 'same patch\nacross time', ha='left', va='center', fontsize=FS_ANNOT)
ax.text(
    0.5, 0.12,
    f'Per location: T^2\nOver N locations: N T^2 = {N_SPATIAL * T ** 2}',
    transform=ax.transAxes, ha='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)

plt.tight_layout()
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Cost comparison + example attention maps
# -------------------------------------------------------
fig4, axes4 = new_figure(1, 3)
fig4.suptitle(
    'Step 4: Factorization buys scalability; attention still tracks the motion',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes4[0]
Ts = np.array([4, 8, 16, 32])
Ns = 64  # e.g. 8x8 patches
ax.plot(Ts, cost_joint(Ts, Ns) / 1e6, 'o-', color='crimson', lw=2.5, label='joint (TN)^2')
ax.plot(Ts, cost_divided(Ts, Ns) / 1e6, 's-', color='royalblue', lw=2.5, label='divided TN^2+NT^2')
ax.set_xlabel('number of frames T', fontsize=FS_AXIS)
ax.set_ylabel('attention cost  (millions of pairs)', fontsize=FS_AXIS)
ax.set_title(f'Cost vs T  (N={Ns} patches/frame)', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.legend(fontsize=FS_LEGEND)
ax.grid(True, alpha=0.3)
add_note(
    ax,
    'Joint attention grows\n'
    'much faster in T.\n'
    'Divided is the usual\n'
    'practical compromise.',
    y=0.98, va='top', facecolor='lightyellow',
)

ax = axes4[1]
im = ax.imshow(attn_space, cmap='viridis', vmin=0, vmax=1)
ax.set_title('Spatial attention\n(inside one frame)', fontsize=FS_AX_TITLE)
ax.set_xlabel('patch key', fontsize=FS_AXIS)
ax.set_ylabel('patch query', fontsize=FS_AXIS)
ax.tick_params(axis='both', labelsize=FS_TICK - 2)
fig4.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes4[2]
im = ax.imshow(attn_time, cmap='viridis', vmin=0, vmax=1)
ax.set_title('Temporal attention\n(fixed patch over time)', fontsize=FS_AX_TITLE)
ax.set_xlabel('frame key', fontsize=FS_AXIS)
ax.set_ylabel('frame query', fontsize=FS_AXIS)
ax.set_xticks(range(T))
ax.set_yticks(range(T))
ax.tick_params(axis='both', labelsize=FS_TICK - 2)
fig4.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
add_note(
    ax,
    'Moving blob => nearby\n'
    'frames share content.\n'
    'Temporal attention\n'
    'links them.',
    y=0.02, va='bottom',
)

plt.tight_layout()
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Positional structure in video transformers
# -------------------------------------------------------
fig5, axes5 = new_figure(1, 2)
fig5.suptitle(
    'Step 5: Positional embeddings tell tokens where and WHEN they are',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)

ax = axes5[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')
ax.set_title('Additive space + time positions', fontsize=FS_AX_TITLE)
# schematic formula
ax.text(
    5, 5.5,
    r'$z_{t,i,j} = Embed(tubelet_{t,i,j})$'
    '\n'
    r'$\;\;\;\;\;\;\; +\, e^{time}_t + e^{space}_{i,j}$',
    ha='center', va='center', fontsize=FS_AXIS - 1,
    bbox=dict(boxstyle='round', facecolor='#eef5ff', alpha=0.95),
)
ax.text(
    5, 3.2,
    'Without positions, attention is a bag of patches:\n'
    'it cannot know left vs right, or early vs late.\n\n'
    'Video models often factor positions:\n'
    '  • spatial grid embedding\n'
    '  • temporal index embedding\n'
    '  (sometimes relative / RoPE-style variants)',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95),
)
ax.text(
    5, 1.0,
    'Some models use joint 3D positions;\n'
    'factorized ones are easier to transfer from image ViTs.',
    ha='center', va='center', fontsize=FS_ANNOT,
    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95),
)

ax = axes5[1]
# visualize factorized position ids
pos_time = np.repeat(np.arange(n_t), n_h * n_w)
pos_space = np.tile(np.arange(n_h * n_w), n_t)
ax.scatter(pos_space, pos_time, c=pos_time, cmap='plasma', s=80, edgecolor='k')
for k, (ti, i, j) in enumerate(tube_index):
    if k % 5 == 0:
        ax.text(pos_space[k] + 0.15, pos_time[k] + 0.05, f'({i},{j})',
                fontsize=7, color='0.3')
ax.set_xlabel('spatial patch id', fontsize=FS_AXIS)
ax.set_ylabel('temporal id', fontsize=FS_AXIS)
ax.set_title('Each tubelet token has a (time, space) address', fontsize=FS_AX_TITLE)
ax.tick_params(axis='both', labelsize=FS_TICK)
ax.grid(True, alpha=0.3)

plt.tight_layout()
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Architecture zoo (mental map)
# -------------------------------------------------------
fig6, ax6 = new_figure()
fig6.suptitle(
    'Step 6: Common Video Transformer design patterns',
    fontsize=FS_FIGURE_TITLE, fontweight='bold',
)
ax6.set_xlim(0, 12)
ax6.set_ylim(0, 8)
ax6.axis('off')


def card(ax, x, y, w, h, title, body, color):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.1',
        facecolor=color, edgecolor='k', lw=2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.35, title, ha='center', va='top',
            fontsize=FS_AXIS + 2, fontweight='bold')
    ax.text(x + w / 2, y + h / 2 - 0.15, body, ha='center', va='center',
            fontsize=FS_AXIS)


card(
    ax6, 0.4, 4.5, 3.6, 3.0,
    'Joint ST attention',
    'Full attention over\nall space-time tokens\n\n+ most expressive\n- very expensive',
    '#f8d7da',
)
card(
    ax6, 4.2, 4.5, 3.6, 3.0,
    'Divided ST (TimeSformer)',
    'Space-attn then\ntime-attn (or vice versa)\n\n+ scalable\n+ strong baseline',
    '#d1ecf1',
)
card(
    ax6, 8.0, 4.5, 3.6, 3.0,
    'Tubelet embed (ViViT)',
    'Tokens already mix\na bit of local time\n\nThen Transformer\nover tubelets',
    '#d4edda',
)
card(
    ax6, 2.2, 0.6, 3.6, 3.0,
    'Late fusion',
    'Image ViT per frame\nthen pool / attend\nover time\n\n+ init from images\n- weaker motion',
    '#fff3cd',
)
card(
    ax6, 6.2, 0.6, 3.6, 3.0,
    'Masked video models',
    'VideoMAE / similar:\npredict missing\ntubelets\n\nSelf-supervised\npretraining recipe',
    '#e2d5f1',
)

plt.tight_layout()
show_maximized(fig6)


# -------------------------------------------------------
# KEY INSIGHTS
# -------------------------------------------------------
print("\n" + "=" * 60)
print("VIDEO TRANSFORMER INTUITION")
print("=" * 60)
print(f"""
Video tensor:  (T, H, W) = ({T}, {H}, {W}) in this toy clip
Spatial patches / frame:  {N_SPATIAL}   ({n_h}x{n_w} with patch={PATCH})
Tubelet tokens:           {N_TUBE}    (tube_t={TUBE_T})

Attention complexity (order-of-magnitude):
  Joint space-time:   (T N)^2 = {cost_joint(T, N_SPATIAL)}
  Divided space+time: T N^2 + N T^2 = {cost_divided(T, N_SPATIAL)}

Pipeline:
  1) Tokenize frames (patches) or space-time (tubelets)
  2) Add spatial + temporal positional information
  3) Apply Transformer blocks:
        - joint OR divided/factorized attention
        - FFN + residual + LayerNorm  (same as text/image Transformers)
  4) Pool tokens -> video classification / detection / generation head

Intuition to keep:
  Image ViT answers: which places matter together?
  Video Transformer also answers: which moments matter together?
  Motion is correspondence across time -- temporal attention is how
  tokens find those correspondences.

See also: 0850_transformer.py for plain self-attention mechanics.
""")
