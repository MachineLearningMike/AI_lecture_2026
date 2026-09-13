"""
04 — What is the derivative at a point?                        (interactive)
Run:  python 04_derivative_at_a_point.py

Controls:
  function buttons (bottom left) -> x^2, x^3, sin x, e^x
  point x slider / click-and-drag on the left plot -> where to measure
  step h slider -> the "run" of the rise/run triangle

The idea:
  A SECANT line (red) joins two points on the curve: x and x + h.
      its slope = rise / run = (f(x + h) - f(x)) / h
  Shrink h and the secant rotates until it becomes the TANGENT line
  (green), which just touches the curve at x. The tangent's slope is
  the DERIVATIVE f'(x).
  Right panel: the secant slope for every h. As h -> 0 it settles on
  the true derivative (green dashed line).

Questions to answer by playing:
  * For x^2 at x = 1, what does the secant slope settle on?
  * Where is the slope of x^3 equal to zero?
  * Where is the slope of sin x equal to zero?
  * For e^x, compare the slope with the height of the point.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

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


def widget_axes(fig, rect):
    """An area for a button or checkbox, without plot frame lines."""
    ax = fig.add_axes(rect)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def draw_axes(ax):
    """Dark x- and y-axis lines through (0, 0)."""
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)


# name: (function, its derivative, x-range, y-range, derivative formula)
FUNCS = {
    "$x^2$":      (lambda x: x**2, lambda x: 2 * x,      (-2.5, 2.5), (-1.5, 6.5), "2x"),
    "$x^3$":      (lambda x: x**3, lambda x: 3 * x**2,   (-2.2, 2.2), (-9, 9),     "3x^2"),
    "$\\sin x$":  (np.sin,         np.cos,               (-2 * np.pi, 2 * np.pi), (-1.8, 1.8), "cos(x)"),
    "$e^x$":      (np.exp,         np.exp,               (-3, 2.5),   (-0.8, 9),   "e^x"),
}

fig = plt.figure(figsize=(14, 7.8))
fig.canvas.manager.set_window_title("04 — The derivative at a point")
header(fig, "The derivative = the slope at ONE point",
       "drag the h slider to the right to shrink h: the red secant becomes the green tangent. "
       "Then pick other functions and drag the point.")
ax1 = fig.add_axes([0.06, 0.32, 0.52, 0.53])
ax2 = fig.add_axes([0.66, 0.32, 0.32, 0.53])

# ---- left: the curve, the secant, and the tangent --------------------
(curve,) = ax1.plot([], [], lw=3, color="tab:blue", label="f(x)")
(tangent,) = ax1.plot([], [], "--", lw=2, color="tab:green", label="tangent (slope = derivative)")
(secant,) = ax1.plot([], [], lw=2, color="tab:red", label="secant through x and x + h")
(run_line,) = ax1.plot([], [], color="gray", lw=3)
(rise_line,) = ax1.plot([], [], color="gray", lw=3)
(p0,) = ax1.plot([], [], "o", color="black", ms=9, zorder=5)
(p1,) = ax1.plot([], [], "o", color="tab:red", ms=8, zorder=5)
ax1.set_xlabel("x")
ax1.legend(loc="upper left", fontsize=9)

# ---- right: secant slope as h shrinks ---------------------------------
(sec_curve,) = ax2.semilogx([], [], lw=2.5, color="tab:red", label="secant slope")
true_line = ax2.axhline(0, color="tab:green", ls="--", lw=2, label="true derivative")
(cur_pt,) = ax2.plot([], [], "o", color="tab:red", ms=10)
ax2.set_xlim(1, 0.001)                               # big h on the left, tiny h on the right
ax2.set_xlabel("step size h  (shrinking →)")
ax2.set_ylabel("slope = rise / run")
ax2.set_title("secant slope for every h")
ax2.legend(loc="best", fontsize=9)

readout = fig.text(0.62, 0.225, "", family="monospace", va="top", fontsize=10.5,
                   bbox=dict(boxstyle="round", fc="white", ec="gray"))

# ---- controls ---------------------------------------------------------
fig.text(0.02, 0.235, "function", fontsize=11, fontweight="bold")
radio = RadioButtons(widget_axes(fig, [0.02, 0.03, 0.09, 0.19]), list(FUNCS), active=0)
s_x = Slider(fig.add_axes([0.20, 0.16, 0.36, 0.035]), "point x", -2, 2, valinit=1, valstep=0.01)
s_h = Slider(fig.add_axes([0.20, 0.09, 0.36, 0.035]), "step h", 0, 3, valinit=0, valstep=0.01)
fig.text(0.20, 0.045, "(slide right: h = 1 → 0.1 → 0.01 → 0.001)", fontsize=9, color="#555555")


def update(_=None):
    name = radio.value_selected
    f, df, xlim, ylim, d_formula = FUNCS[name]
    x0 = s_x.val
    h = 10 ** (-s_h.val)                           # slider 0, 1, 2, 3 -> h = 1, 0.1, 0.01, 0.001
    s_h.valtext.set_text(f"{h:.3g}")              # show h itself next to the slider

    y0, y1 = f(x0), f(x0 + h)
    m = (y1 - y0) / h                               # secant slope: rise / run
    d = df(x0)                                      # true derivative

    xs = np.linspace(*xlim, 400)
    curve.set_data(xs, f(xs))
    tangent.set_data(xs, y0 + d * (xs - x0))
    secant.set_data(xs, y0 + m * (xs - x0))
    run_line.set_data([x0, x0 + h], [y0, y0])
    rise_line.set_data([x0 + h, x0 + h], [y0, y1])
    p0.set_data([x0], [y0])
    p1.set_data([x0 + h], [y1])
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_title(f"f(x) = {name}      gray triangle: run = h,  rise = f(x+h) − f(x)")

    H = np.logspace(-3, 0, 200)
    S = (f(x0 + H) - y0) / H
    sec_curve.set_data(H, S)
    true_line.set_ydata([d, d])
    cur_pt.set_data([h], [m])
    lo, hi = min(S.min(), d), max(S.max(), d)
    pad = 0.12 * (hi - lo) if hi > lo else 0.5
    ax2.set_ylim(lo - pad, hi + pad)

    readout.set_text(
        f"secant slope = (f(x+h) − f(x)) / h\n"
        f"             = ({y1:.4f} − {y0:.4f}) / {h:.4g}\n"
        f"             = {m:.5f}\n"
        f"\n"
        f"derivative   f'(x) = {d_formula}\n"
        f"             f'({x0:.2f}) = {d:.5f}\n"
        f"difference   = {abs(m - d):.5f}")
    fig.canvas.draw_idle()


# ---- click and drag on the left plot to move the point ---------------
state = {"dragging": False}


def on_press(event):
    toolbar_busy = getattr(fig.canvas.toolbar, "mode", "") != ""
    if event.inaxes is ax1 and event.button == 1 and not toolbar_busy:
        state["dragging"] = True
        s_x.set_val(np.clip(event.xdata, -2, 2))


def on_move(event):
    if state["dragging"] and event.inaxes is ax1:
        s_x.set_val(np.clip(event.xdata, -2, 2))


def on_release(event):
    state["dragging"] = False


fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("button_release_event", on_release)

radio.on_clicked(update)
s_x.on_changed(update)
s_h.on_changed(update)
update()
plt.show()