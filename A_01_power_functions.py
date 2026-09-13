"""
01 — Power functions  y = x^n                                (interactive)
Run:  python 01_power_functions.py

Controls at the bottom of the window:
  n slider -> change the exponent
  x slider -> pick a point; the right panel shows x^1 ... x^8 at that point

What to notice:
  * Every x^n passes through (1, 1), because 1^n = 1.
  * EVEN n -> a symmetric bowl, never negative. That is why the squared
    error (prediction - target)^2 is the classic loss function:
    always >= 0, with one lowest point to aim for.
  * ODD n  -> keeps the sign of x: negative in, negative out.
  * |x| < 1 -> higher powers get SMALLER   (0.5^8 = 0.0039)
    |x| > 1 -> higher powers EXPLODE       (1.5^8 = 25.6)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "legend.fontsize": 10,
})


def header(fig, title, tip):
    """Big title plus a 'Try:' hint at the top of the window."""
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.975)
    fig.text(0.5, 0.925, "Try: " + tip, ha="center", fontsize=10.5,
             color="#444444", style="italic")


def draw_axes(ax):
    """Dark x- and y-axis lines through (0, 0)."""
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)


# ---- window and panels ------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.25, wspace=0.22)
fig.canvas.manager.set_window_title("01 — Power functions")
header(fig, "Power functions   $y = x^n$",
       "drag n from 1 to 8 (even vs odd?). Then drag x into the gray zone −1 < x < 1 "
       "and watch the bars.")

x = np.linspace(-2, 2, 400)
N_MAX = 8
ns = np.arange(1, N_MAX + 1)          # 1, 2, ..., 8

# left panel: every power in light gray, the chosen one in bold blue
for n in ns:
    ax1.plot(x, x**n, color="lightgray", lw=1)
ax1.axvspan(-1, 1, color="gray", alpha=0.1)
ax1.plot(1, 1, "ko", ms=6)
ax1.annotate("(1, 1)", (1, 1), xytext=(-45, 5), textcoords="offset points")
(curve,) = ax1.plot(x, x**2, lw=3, color="tab:blue")
(point,) = ax1.plot([], [], "o", color="tab:red", ms=10)
point_label = ax1.annotate("", (0, 0), xytext=(12, -18), textcoords="offset points",
                           color="tab:red", fontweight="bold")
ax1.set_xlim(-2, 2)
ax1.set_ylim(-4, 4)
ax1.set_xlabel("x")
draw_axes(ax1)

# right panel: bar chart of x^1 ... x^8 at the chosen x
bars = ax2.bar(ns, np.zeros(N_MAX))
bar_labels = [ax2.text(n, 0, "", ha="center", fontsize=9) for n in ns]
ax2.set_xticks(ns, [f"$x^{n}$" for n in ns])
ax2.axhline(0, color="black", lw=0.8)

# ---- controls ---------------------------------------------------------
s_n = Slider(fig.add_axes([0.10, 0.10, 0.33, 0.04]), "n", 1, N_MAX, valinit=2, valstep=1)
s_x = Slider(fig.add_axes([0.60, 0.10, 0.33, 0.04]), "x", -2, 2, valinit=1.5, valstep=0.05)


def update(_=None):
    """Redraw everything from the current slider values."""
    n = int(s_n.val)
    x0 = round(s_x.val, 2)

    # left panel
    curve.set_ydata(x**n)
    y0 = x0**n
    point.set_data([x0], [y0])
    point_label.xy = (x0, np.clip(y0, -3.8, 3.8))
    base = f"({x0:g})" if x0 < 0 else f"{x0:g}"
    point_label.set_text(f"{base}$^{{{n}}}$ = {y0:.4g}")
    kind = "EVEN → symmetric bowl, never negative" if n % 2 == 0 else "ODD → keeps the sign of x"
    ax1.set_title(f"$y = x^{{{n}}}$      {kind}")

    # right panel
    values = x0**ns
    for bar, label, k, v in zip(bars, bar_labels, ns, values):
        bar.set_height(v)
        bar.set_color("tab:blue" if k == n else "lightsteelblue")
        label.set_position((k, v))
        label.set_text(f"{v:.3g}")
        label.set_va("bottom" if v >= 0 else "top")
    lo, hi = min(values.min(), 0), max(values.max(), 0)
    pad = 0.15 * (hi - lo) if hi > lo else 1
    ax2.set_ylim(lo - pad, hi + pad)

    if x0 == 0:
        msg = "every power of 0 is 0"
    elif abs(x0) == 1:
        msg = "every power has size 1"
    elif abs(x0) < 1:
        msg = "|x| < 1 → higher powers get SMALLER"
    else:
        msg = "|x| > 1 → higher powers EXPLODE"
    if x0 < 0:
        msg += "\n(x < 0: even powers are +, odd powers are −)"
    ax2.set_title(f"at x = {x0:g}:  {msg}")
    fig.canvas.draw_idle()


s_n.on_changed(update)
s_x.on_changed(update)
update()
plt.show()