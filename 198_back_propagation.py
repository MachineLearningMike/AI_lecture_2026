"""
198_back_propagation.py

Backpropagation on the simplest model, then how PyTorch builds a grad graph,
runs backward(), and applies step() -- with a step-by-step forward/backward seeker.

Simplest model (scalars):
    yhat = w * x + b
    e    = yhat - y
    L    = e ** 2

Torch training beat:
    predict -> loss -> zero_grad -> backward -> step

Sketch:
  1. Simplest model and the loss scalar
  2. Chain rule by hand (local grads on edges)
  3. What PyTorch stores: leaves, values, grad_fn
  4. Forward/backward seeker -- Prev/Next (or slider / arrow keys)
  5. How graph content changes across phases
  6. Why zero_grad matters
  7. One optimizer step closes the loop
  8. Takeaway

Companion to 016_torch_training.py, 140_derivative.py, 197_SGD_concept.py,
141_discriminative.py.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.widgets import Button, Slider
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
import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("198_back_propagation.py needs PyTorch") from exc

np.random.seed(198)
torch.manual_seed(198)


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


def blank_slide(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    return ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    ))


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_FUN = "#8e44ad"
C_BIN = "#d68910"
C_GRAY = "#85929e"
C_ACTIVE = "#f9e79f"
C_DONE = "#a9dfbf"
C_IDLE = "#ebedef"


# Fixed toy numbers (chosen so grads are readable)
X_VAL = 2.0
Y_VAL = 5.0
W0 = 1.5
B0 = 0.5
LR = 0.05


def run_manual(w, b, x=X_VAL, y=Y_VAL):
    u = w * x
    yhat = u + b
    e = yhat - y
    L = e ** 2
    dL_de = 2.0 * e
    dL_dyhat = dL_de
    dL_du = dL_dyhat
    dL_db = dL_dyhat
    dL_dw = dL_du * x
    return {
        "w": w, "b": b, "x": x, "y": y,
        "u": u, "yhat": yhat, "e": e, "L": L,
        "dL_de": dL_de, "dL_dyhat": dL_dyhat,
        "dL_du": dL_du, "dL_db": dL_db, "dL_dw": dL_dw,
    }


# -------------------------------------------------------
# FIGURE 1: Simplest model
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle("Step 1: Simplest model -- one weight, one bias, squared error", fontsize=FS_FIGURE_TITLE, fontweight="bold")
blank_slide(ax1)

boxes = [
    (0.4, 5.2, C_NEW, C_DARK, "x", f"{X_VAL:g}\n(fixed)"),
    (2.7, 5.2, C_GOAL, C_OK, "w", f"{W0:g}\nlearnable"),
    (5.0, 5.2, "#f5eef8", C_FUN, "b", f"{B0:g}\nlearnable"),
    (7.3, 5.2, C_NEW, C_DARK, "y", f"{Y_VAL:g}\ntarget"),
]
for x0, y0, face, edge, title, body in boxes:
    round_box(ax1, x0, y0, 2.0, 2.2, face, edge, lw=2.0)
    ax1.text(x0 + 1.0, y0 + 1.55, title, ha="center", fontsize=FS_AX_TITLE - 2,
             fontweight="bold", color=edge, family="monospace")
    ax1.text(x0 + 1.0, y0 + 0.7, body, ha="center", va="center", fontsize=FS_ANNOT)

round_box(ax1, 0.4, 1.0, 11.2, 3.6, C_MUTE, "0.45", lw=1.8)
ax1.text(
    6.0, 2.8,
    r"$\,\hat{y} = w\,x + b$"
    "\n"
    r"$\,e = \hat{y} - y$"
    "\n"
    r"$\,L = e^{2}$",
    ha="center", va="center", fontsize=FS_FIGURE_TITLE - 4, family="monospace",
    color=C_DARK,
)
ax1.text(
    6.0, 1.35,
    "Backprop = chain rule on this tiny graph.",
    ha="center", va="center", fontsize=FS_ANNOT + 1,
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Chain rule by hand
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle("Step 2: Chain rule by hand -- local grads on every edge", fontsize=FS_FIGURE_TITLE, fontweight="bold")
blank_slide(ax2)

m = run_manual(W0, B0)
nodes = [
    ("w", 1.0, 6.2),
    ("x", 1.0, 4.2),
    ("u=wx", 3.3, 5.2),
    ("b", 3.3, 2.8),
    ("yhat", 5.8, 4.2),
    ("y", 5.8, 2.0),
    ("e", 8.2, 3.3),
    ("L", 10.5, 3.3),
]
pos = {n: (x, y) for n, x, y in nodes}
for name, x, y in nodes:
    face = C_GOAL if name in ("w", "b") else (C_ACTIVE if name == "L" else C_NEW)
    round_box(ax2, x - 0.7, y - 0.45, 1.4, 0.9, face, C_DARK, lw=1.6)
    ax2.text(x, y, name, ha="center", va="center", fontsize=FS_ANNOT,
             fontweight="bold", family="monospace")

edges = [
    ("w", "u=wx", r"$\times x$"),
    ("x", "u=wx", r"$\times w$"),
    ("u=wx", "yhat", r"$+1$"),
    ("b", "yhat", r"$+1$"),
    ("yhat", "e", r"$+1$"),
    ("y", "e", r"$-1$"),
    ("e", "L", r"$2e$"),
]
for a, b, lab in edges:
    x1, y1 = pos[a]; x2, y2 = pos[b]
    ax2.annotate(
        "", xy=(x2 - 0.75, y2), xytext=(x1 + 0.75, y1),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.8, mutation_scale=12),
    )
    ax2.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.35, lab,
             ha="center", fontsize=FS_TICK, color=C_ACCENT)

ax2.text(
    6.0, 0.7,
    f"numbers: L={m['L']:.3f},  dL/dw={m['dL_dw']:.3f},  dL/db={m['dL_db']:.3f}",
    ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK,
)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Torch leaves / grad_fn
# -------------------------------------------------------
fig3, axes3 = new_figure(1, 2)
fig3.suptitle("Step 3: PyTorch stores values + grad_fn while building the graph", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")

w_t = torch.tensor(W0, requires_grad=True)
b_t = torch.tensor(B0, requires_grad=True)
x_t = torch.tensor(X_VAL)
y_t = torch.tensor(Y_VAL)
u_t = w_t * x_t
yhat_t = u_t + b_t
e_t = yhat_t - y_t
L_t = e_t ** 2

rows = [
    ("w", float(w_t), str(w_t.grad_fn), "leaf" if w_t.is_leaf else ""),
    ("b", float(b_t), str(b_t.grad_fn), "leaf" if b_t.is_leaf else ""),
    ("u=w*x", float(u_t), type(u_t.grad_fn).__name__, ""),
    ("yhat", float(yhat_t), type(yhat_t.grad_fn).__name__, ""),
    ("e", float(e_t), type(e_t.grad_fn).__name__, ""),
    ("L", float(L_t), type(L_t.grad_fn).__name__, "root for backward"),
]

axes3[0].axis("off")
axes3[0].set_title("tensor table after forward", fontsize=FS_AX_TITLE - 2, fontweight="bold")
table = axes3[0].table(
    cellText=[[r[0], f"{r[1]:.4g}", r[2], r[3]] for r in rows],
    colLabels=["name", "value", "grad_fn", "note"],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(FS_TICK)
table.scale(1.2, 2.0)

# graph sketch on right
axes3[1].set_xlim(0, 10)
axes3[1].set_ylim(0, 10)
axes3[1].axis("off")
axes3[1].set_title("grad graph (forward-built)", fontsize=FS_AX_TITLE - 2, fontweight="bold")
g_nodes = [
    ("w leaf", 1.5, 8, C_GOAL),
    ("x", 1.5, 5.5, C_MUTE),
    ("MulBackward0", 4.2, 6.8, C_NEW),
    ("b leaf", 4.2, 3.5, C_GOAL),
    ("AddBackward0", 6.5, 5.2, C_NEW),
    ("SubBackward0", 8.2, 3.8, C_NEW),
    ("PowBackward0\n(L)", 8.2, 1.5, C_ACTIVE),
]
for name, x, y, face in g_nodes:
    round_box(axes3[1], x - 1.0, y - 0.55, 2.0, 1.1, face, C_DARK, lw=1.5)
    axes3[1].text(x, y, name, ha="center", va="center", fontsize=FS_TICK, fontweight="bold")
for a, b in [(0, 2), (1, 2), (2, 4), (3, 4), (4, 5), (5, 6)]:
    x1, y1 = g_nodes[a][1], g_nodes[a][2]
    x2, y2 = g_nodes[b][1], g_nodes[b][2]
    axes3[1].annotate(
        "", xy=(x2 - 0.2, y2 + 0.4), xytext=(x1 + 0.2, y1 - 0.4),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.6),
    )
fig3.tight_layout(rect=[0, 0.02, 1, 0.92])
show_maximized(fig3)


# -------------------------------------------------------
# Build seeker snapshots with live torch where useful
# -------------------------------------------------------
# Seeker stages (forward then backward then step/zero)
SEEKER_LABELS = [
    "0 leaves only",
    "1 forward: u = w*x",
    "2 forward: yhat = u+b",
    "3 forward: e = yhat-y",
    "4 forward: L = e**2   [graph complete]",
    "5 backward: seed dL=1",
    "6 backward: dL/de = 2e",
    "7 backward: to yhat, u, b",
    "8 backward: dL/dw filled",
    "9 step: w,b <- w,b - lr*grad",
    "10 graph freed; grads still sit on leaves",
    "11 zero_grad: clear .grad",
]


def build_snapshots():
    snaps = []
    w = W0
    b = B0
    man = run_manual(w, b)

    def base(**kwargs):
        d = {
            "w": w, "b": b,
            "u": None, "yhat": None, "e": None, "L": None,
            "gw": None, "gb": None,
            "active": [],
            "phase": "init",
            "graph_nodes": ["w", "b", "x", "y"],
            "edge_on": set(),
            "note": "",
        }
        d.update(kwargs)
        return d

    # 0
    snaps.append(base(
        phase="leaves",
        active=["w", "b"],
        note="requires_grad leaves exist. No forward graph yet.",
    ))
    # 1
    snaps.append(base(
        u=man["u"], phase="fwd", active=["u=wx"],
        graph_nodes=["w", "b", "x", "y", "u=wx"],
        edge_on={("w", "u=wx"), ("x", "u=wx")},
        note="MulBackward0 attached to u.",
    ))
    # 2
    snaps.append(base(
        u=man["u"], yhat=man["yhat"], phase="fwd", active=["yhat"],
        graph_nodes=["w", "b", "x", "y", "u=wx", "yhat"],
        edge_on={("w", "u=wx"), ("x", "u=wx"), ("u=wx", "yhat"), ("b", "yhat")},
        note="AddBackward0 attached to yhat.",
    ))
    # 3
    snaps.append(base(
        u=man["u"], yhat=man["yhat"], e=man["e"], phase="fwd", active=["e"],
        graph_nodes=["w", "b", "x", "y", "u=wx", "yhat", "e"],
        edge_on={("w", "u=wx"), ("x", "u=wx"), ("u=wx", "yhat"), ("b", "yhat"),
                 ("yhat", "e"), ("y", "e")},
        note="SubBackward0 attached to e.",
    ))
    # 4
    full_edges = {("w", "u=wx"), ("x", "u=wx"), ("u=wx", "yhat"), ("b", "yhat"),
                  ("yhat", "e"), ("y", "e"), ("e", "L")}
    snaps.append(base(
        u=man["u"], yhat=man["yhat"], e=man["e"], L=man["L"], phase="fwd",
        active=["L"],
        graph_nodes=["w", "b", "x", "y", "u=wx", "yhat", "e", "L"],
        edge_on=full_edges,
        note="PowBackward0 is the backward root. Graph is complete.",
    ))
    # 5 seed
    snaps.append(base(
        u=man["u"], yhat=man["yhat"], e=man["e"], L=man["L"], phase="bwd",
        active=["L"], graph_nodes=["w", "b", "x", "y", "u=wx", "yhat", "e", "L"],
        edge_on=full_edges,
        note="backward() starts: upstream grad at L is 1.",
    ))
    # 6 to e
    snaps.append(base(
        u=man["u"], yhat=man["yhat"], e=man["e"], L=man["L"], phase="bwd",
        active=["e"], graph_nodes=["w", "b", "x", "y", "u=wx", "yhat", "e", "L"],
        edge_on=full_edges,
        note=f"PowBackward: dL/de = 2e = {man['dL_de']:.3f}",
    ))
    # 7 to yhat/u/b
    snaps.append(base(
        u=man["u"], yhat=man["yhat"], e=man["e"], L=man["L"],
        gb=man["dL_db"], phase="bwd",
        active=["yhat", "u=wx", "b"],
        graph_nodes=["w", "b", "x", "y", "u=wx", "yhat", "e", "L"],
        edge_on=full_edges,
        note=f"Add/Sub local grads pass through. dL/db = {man['dL_db']:.3f}",
    ))
    # 8 to w
    snaps.append(base(
        u=man["u"], yhat=man["yhat"], e=man["e"], L=man["L"],
        gw=man["dL_dw"], gb=man["dL_db"], phase="bwd",
        active=["w"],
        graph_nodes=["w", "b", "x", "y", "u=wx", "yhat", "e", "L"],
        edge_on=full_edges,
        note=f"MulBackward: dL/dw = dL/du * x = {man['dL_dw']:.3f}. Leaf .grad filled.",
    ))
    # 9 step
    w1 = w - LR * man["dL_dw"]
    b1 = b - LR * man["dL_db"]
    snaps.append({
        "w": w1, "b": b1,
        "u": man["u"], "yhat": man["yhat"], "e": man["e"], "L": man["L"],
        "gw": man["dL_dw"], "gb": man["dL_db"],
        "active": ["w", "b"],
        "phase": "step",
        "graph_nodes": ["w", "b", "x", "y"],  # freed intermediates
        "edge_on": set(),
        "note": f"opt.step(): w {w:.3f}->{w1:.3f}, b {b:.3f}->{b1:.3f}",
    })
    # 10 after backward graph freed (typical without retain_graph)
    snaps.append({
        "w": w1, "b": b1,
        "u": None, "yhat": None, "e": None, "L": None,
        "gw": man["dL_dw"], "gb": man["dL_db"],
        "active": ["w", "b"],
        "phase": "freed",
        "graph_nodes": ["w", "b", "x", "y"],
        "edge_on": set(),
        "note": "Forward graph is gone. Leaf tensors remain; .grad still holds numbers.",
    })
    # 11 zero_grad
    snaps.append({
        "w": w1, "b": b1,
        "u": None, "yhat": None, "e": None, "L": None,
        "gw": 0.0, "gb": 0.0,
        "active": ["w", "b"],
        "phase": "zero",
        "graph_nodes": ["w", "b", "x", "y"],
        "edge_on": set(),
        "note": "opt.zero_grad(): .grad cleared so the next backward does not accumulate.",
    })
    return snaps


SNAPS = build_snapshots()

# verify with real torch once
_w = torch.tensor(W0, requires_grad=True)
_b = torch.tensor(B0, requires_grad=True)
_L = ((_w * X_VAL + _b) - Y_VAL) ** 2
_L.backward()
_man = run_manual(W0, B0)
assert abs(float(_w.grad) - _man["dL_dw"]) < 1e-6
assert abs(float(_b.grad) - _man["dL_db"]) < 1e-6


# -------------------------------------------------------
# FIGURE 4: Forward / backward seeker
# -------------------------------------------------------
fig4, axes4 = plt.subplots(1, 3, figsize=screen_figsize(), dpi=FIGURE_DPI,
                           gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
fig4.suptitle("Step 4: Forward / backward seeker -- drag the step", fontsize=FS_FIGURE_TITLE, fontweight="bold")
fig4.subplots_adjust(bottom=0.20, wspace=0.28)
ax_g, ax_v, ax_h = axes4

NODE_POS = {
    "w": (1.2, 7.0),
    "x": (1.2, 4.6),
    "u=wx": (3.6, 5.8),
    "b": (3.6, 2.8),
    "yhat": (6.0, 4.6),
    "y": (6.0, 2.2),
    "e": (8.3, 3.5),
    "L": (10.4, 3.5),
}
ALL_EDGES = [
    ("w", "u=wx"), ("x", "u=wx"), ("u=wx", "yhat"), ("b", "yhat"),
    ("yhat", "e"), ("y", "e"), ("e", "L"),
]


def _fmt(v):
    if v is None:
        return "—"
    return f"{v:.4g}"


def draw_seeker(step):
    snap = SNAPS[int(step)]
    ax_g.clear(); ax_v.clear(); ax_h.clear()
    ax_g.set_xlim(0, 12)
    ax_g.set_ylim(0, 8.5)
    ax_g.axis("off")
    ax_g.set_title("grad graph", fontsize=FS_AX_TITLE - 2, fontweight="bold")

    # edges
    for a, b in ALL_EDGES:
        if a in snap["graph_nodes"] and b in snap["graph_nodes"] and (a, b) in snap["edge_on"]:
            x1, y1 = NODE_POS[a]; x2, y2 = NODE_POS[b]
            ax_g.annotate(
                "", xy=(x2 - 0.7, y2), xytext=(x1 + 0.7, y1),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2.0, mutation_scale=12),
            )

    for name in ["w", "x", "u=wx", "b", "yhat", "y", "e", "L"]:
        if name not in snap["graph_nodes"]:
            continue
        x, y = NODE_POS[name]
        if name in snap["active"]:
            face = C_ACTIVE
        elif name in ("w", "b"):
            face = C_GOAL
        else:
            face = C_NEW
        round_box(ax_g, x - 0.75, y - 0.42, 1.5, 0.84, face, C_DARK, lw=1.6)
        ax_g.text(x, y, name, ha="center", va="center", fontsize=FS_TICK,
                  fontweight="bold", family="monospace")

    # values panel
    ax_v.axis("off")
    ax_v.set_xlim(0, 1)
    ax_v.set_ylim(0, 1)
    ax_v.set_title("values", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    lines = [
        f"w = {_fmt(snap['w'])}",
        f"b = {_fmt(snap['b'])}",
        f"x = {X_VAL:g}",
        f"y = {Y_VAL:g}",
        f"u = {_fmt(snap['u'])}",
        f"yhat = {_fmt(snap['yhat'])}",
        f"e = {_fmt(snap['e'])}",
        f"L = {_fmt(snap['L'])}",
    ]
    ax_v.text(0.05, 0.92, "\n".join(lines), ha="left", va="top",
              fontsize=FS_ANNOT + 1, family="monospace", color=C_DARK,
              transform=ax_v.transAxes)

    # grads / status panel
    ax_h.axis("off")
    ax_h.set_xlim(0, 1)
    ax_h.set_ylim(0, 1)
    ax_h.set_title("grads / status", fontsize=FS_AX_TITLE - 2, fontweight="bold")
    phase_col = {
        "leaves": C_GRAY, "fwd": "#2874a6", "bwd": C_ACCENT,
        "step": C_OK, "freed": C_BIN, "zero": C_FUN, "init": C_GRAY,
    }.get(snap["phase"], C_DARK)
    grad_lines = [
        f"phase: {snap['phase']}",
        f"w.grad = {_fmt(snap['gw'])}",
        f"b.grad = {_fmt(snap['gb'])}",
        f"lr = {LR:g}",
        "",
        snap["note"],
    ]
    round_box(ax_h, 0.05, 0.42, 0.90, 0.50, C_MUTE, phase_col, lw=2.0)
    ax_h.text(0.5, 0.67, "\n".join(grad_lines[:4]), ha="center", va="center",
              fontsize=FS_ANNOT, family="monospace", color=C_DARK,
              transform=ax_h.transAxes)
    ax_h.text(0.5, 0.18, snap["note"], ha="center", va="center",
              fontsize=FS_TICK + 1, color=C_DARK, wrap=True,
              transform=ax_h.transAxes)

    # stage label under graph
    ax_g.text(
        6.0, 0.35, SEEKER_LABELS[int(step)],
        ha="center", va="center", fontsize=FS_ANNOT, fontweight="bold",
        color=phase_col,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C_MUTE, edgecolor=phase_col),
    )


ax_prev = fig4.add_axes([0.14, 0.05, 0.08, 0.045])
ax_next = fig4.add_axes([0.78, 0.05, 0.08, 0.045])
ax_sl = fig4.add_axes([0.26, 0.055, 0.48, 0.035])
btn_prev = Button(ax_prev, "Prev", color=C_MUTE, hovercolor=C_NEW)
btn_next = Button(ax_next, "Next", color=C_MUTE, hovercolor=C_GOAL)
sl_seek = Slider(ax_sl, "step", 0, len(SNAPS) - 1, valinit=0, valstep=1, color=C_FUN)


def _set_seek_step(step):
    step = int(np.clip(step, 0, len(SNAPS) - 1))
    if int(sl_seek.val) == step:
        draw_seeker(step)
        fig4.canvas.draw_idle()
    else:
        sl_seek.set_val(step)


def _u_seek(_=None):
    draw_seeker(sl_seek.val)
    fig4.canvas.draw_idle()


def _prev(_event=None):
    _set_seek_step(int(sl_seek.val) - 1)


def _next(_event=None):
    _set_seek_step(int(sl_seek.val) + 1)


def _on_key(event):
    if event.key in ("right", " ", "enter", "down", "n"):
        _next()
    elif event.key in ("left", "up", "p", "backspace"):
        _prev()
    elif event.key == "home":
        _set_seek_step(0)
    elif event.key == "end":
        _set_seek_step(len(SNAPS) - 1)


draw_seeker(0)
sl_seek.on_changed(_u_seek)
btn_prev.on_clicked(_prev)
btn_next.on_clicked(_next)
fig4.canvas.mpl_connect("key_press_event", _on_key)
print("Step 4: seeker -- Next/Prev, slider, or arrow keys / space.")
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: How graph content changes (phase strip)
# -------------------------------------------------------
fig5, axes5 = new_figure(2, 3)
fig5.suptitle("Step 5: How the grad-graph content changes", fontsize=FS_FIGURE_TITLE, fontweight="bold")

phase_shots = [
    (0, "before forward\nleaves only"),
    (4, "after forward\ngraph complete"),
    (8, "after backward\n.grad filled"),
    (9, "after step\nparams moved"),
    (10, "graph freed\nintermediates gone"),
    (11, "after zero_grad\n.grad cleared"),
]


def mini_graph(ax, step, title):
    snap = SNAPS[step]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    ax.set_title(title, fontsize=FS_AX_TITLE - 3, fontweight="bold")
    for a, b in ALL_EDGES:
        if a in snap["graph_nodes"] and b in snap["graph_nodes"] and (a, b) in snap["edge_on"]:
            x1, y1 = NODE_POS[a]; x2, y2 = NODE_POS[b]
            ax.annotate(
                "", xy=(x2 - 0.7, y2), xytext=(x1 + 0.7, y1),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.2, mutation_scale=9),
            )
    for name in snap["graph_nodes"]:
        x, y = NODE_POS[name]
        face = C_ACTIVE if name in snap["active"] else (C_GOAL if name in ("w", "b") else C_NEW)
        round_box(ax, x - 0.7, y - 0.38, 1.4, 0.76, face, C_DARK, lw=1.2)
        ax.text(x, y, name, ha="center", va="center", fontsize=7, fontweight="bold")
    ax.text(
        6.0, 0.4,
        f"w.grad={_fmt(snap['gw'])}   b.grad={_fmt(snap['gb'])}",
        ha="center", fontsize=8, color=C_DARK,
    )


for ax, (step, title) in zip(axes5.ravel(), phase_shots):
    mini_graph(ax, step, title)
fig5.tight_layout(rect=[0, 0.02, 1, 0.94])
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Why zero_grad
# -------------------------------------------------------
fig6, (ax6a, ax6b) = new_figure(1, 2)
fig6.suptitle("Step 6: Grads accumulate -- why zero_grad exists", fontsize=FS_FIGURE_TITLE, fontweight="bold")

# Simulate two backwards without zero vs with zero
w_acc = torch.tensor(W0, requires_grad=True)
b_acc = torch.tensor(B0, requires_grad=True)
L1 = ((w_acc * X_VAL + b_acc) - Y_VAL) ** 2
L1.backward()
g1w, g1b = float(w_acc.grad), float(b_acc.grad)
L2 = ((w_acc * X_VAL + b_acc) - Y_VAL) ** 2
L2.backward()
g2w, g2b = float(w_acc.grad), float(b_acc.grad)

w_z = torch.tensor(W0, requires_grad=True)
b_z = torch.tensor(B0, requires_grad=True)
Lz1 = ((w_z * X_VAL + b_z) - Y_VAL) ** 2
Lz1.backward()
w_z.grad.zero_(); b_z.grad.zero_()
Lz2 = ((w_z * X_VAL + b_z) - Y_VAL) ** 2
Lz2.backward()
gzw, gzb = float(w_z.grad), float(b_z.grad)

ax6a.bar([0, 1, 2], [g1w, g2w / 2, g2w], color=[C_GRAY, C_BIN, C_ACCENT], width=0.6)
ax6a.set_xticks([0, 1, 2])
ax6a.set_xticklabels(["after 1st\nbackward", "true grad", "after 2nd\n(no zero)"], fontsize=FS_TICK)
ax6a.set_ylabel("w.grad", fontsize=FS_AXIS)
ax6a.set_title("without zero_grad -- accumulates", fontsize=FS_AX_TITLE - 2, color=C_ACCENT)
ax6a.grid(True, alpha=0.3, axis="y")

ax6b.bar([0, 1], [g1w, gzw], color=[C_GRAY, C_OK], width=0.55)
ax6b.set_xticks([0, 1])
ax6b.set_xticklabels(["1st backward", "2nd after zero_grad"], fontsize=FS_TICK)
ax6b.set_ylabel("w.grad", fontsize=FS_AXIS)
ax6b.set_title("with zero_grad -- fresh each time", fontsize=FS_AX_TITLE - 2, color=C_OK)
ax6b.grid(True, alpha=0.3, axis="y")
fig6.tight_layout(rect=[0, 0.02, 1, 0.92])
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: predict -> loss -> zero_grad -> backward -> step
# -------------------------------------------------------
fig7, ax7 = new_figure()
fig7.suptitle("Step 7: The torch beat that actually trains", fontsize=FS_FIGURE_TITLE, fontweight="bold")
blank_slide(ax7)

cycle = [
    (0.5, 5.0, C_NEW, "#2874a6", "predict", "build graph\nyhat=w*x+b"),
    (2.9, 5.0, C_NEW, "#2874a6", "loss", "scalar L\ngraph root"),
    (5.3, 5.0, "#fdebd0", C_BIN, "zero_grad", "clear old\n.grad"),
    (7.7, 5.0, "#fadbd8", C_ACCENT, "backward", "fill .grad\nvia grad_fn"),
    (10.1, 5.0, C_GOAL, C_OK, "step", "w -= lr*grad\nb -= lr*grad"),
]
for x0, y0, face, edge, title, body in cycle:
    round_box(ax7, x0, y0, 2.2, 2.4, face, edge, lw=2.0)
    ax7.text(x0 + 1.1, y0 + 1.75, title, ha="center", fontsize=FS_ANNOT + 1,
             fontweight="bold", color=edge, family="monospace")
    ax7.text(x0 + 1.1, y0 + 0.85, body, ha="center", va="center", fontsize=FS_TICK + 1)

for i in range(4):
    x1 = 0.5 + 2.2 + i * 2.4
    ax7.annotate(
        "", xy=(x1 + 0.15, 6.2), xytext=(x1, 6.2),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2.0, mutation_scale=12),
    )

# live mini train curve
w = torch.nn.Parameter(torch.tensor(W0))
b = torch.nn.Parameter(torch.tensor(B0))
opt = torch.optim.SGD([w, b], lr=LR)
losses = []
for _ in range(25):
    opt.zero_grad()
    loss = ((w * X_VAL + b) - Y_VAL) ** 2
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))

ax_c = fig7.add_axes([0.18, 0.12, 0.64, 0.32])
ax_c.plot(losses, color=C_OK, lw=2.4)
ax_c.scatter([0], [losses[0]], c=C_ACCENT, s=40, zorder=3)
ax_c.scatter([len(losses) - 1], [losses[-1]], c=C_FUN, s=40, zorder=3)
ax_c.set_xlabel("step", fontsize=FS_TICK)
ax_c.set_ylabel("L", fontsize=FS_TICK)
ax_c.set_title("same toy, 25x (zero_grad -> backward -> step)", fontsize=FS_ANNOT, fontweight="bold")
ax_c.grid(True, alpha=0.3)
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Takeaway
# -------------------------------------------------------
fig8, axes8 = new_figure(1, 3)
fig8.suptitle("Step 8: Graph builds forward -- gradients flow backward -- step learns", fontsize=FS_FIGURE_TITLE - 2, fontweight="bold")

mini_graph(axes8[0], 4, "forward builds\ngrad_fn chain")
mini_graph(axes8[1], 8, "backward fills\nleaf .grad")
mini_graph(axes8[2], 9, "step updates\nparameters")
fig8.tight_layout(rect=[0, 0.02, 1, 0.90])
show_maximized(fig8)


print("\n" + "=" * 60)
print("198 BACK PROPAGATION -- SIMPLEST MODEL + TORCH GRAPH")
print("=" * 60)
print(f"""
Model:  yhat = w*x + b ,  L = (yhat - y)^2
Data:   x={X_VAL:g}, y={Y_VAL:g}, w0={W0:g}, b0={B0:g}

Manual grads match torch.backward():
  dL/dw = {_man['dL_dw']:.6f}
  dL/db = {_man['dL_db']:.6f}

Seeker (Next/Prev, slider, arrow keys/space):
  forward graph growth -> backward grad flow -> step -> free -> zero_grad

See also: 016_torch_training.py, 140_derivative.py, 197_SGD_concept.py
""")
