"""
05 — The gradient of an exponential: find the magic base      (interactive)
Run:  python 05_exponential_gradient.py

Controls:
  base a slider (and the  − / +  buttons for fine tuning)
  point x slider / click-and-drag on the left plot
  Show answer button

The game:
  For y = a^x, compare the HEIGHT of a point with the SLOPE there.
  Step 1: move the point x. Notice that  slope ÷ height  never changes!
          For any base a, the slope is always the same multiple of the
          height. (That multiple is ln(a).)
  Step 2: tune the base a until slope = height exactly.
          There is only one base where this happens. Which number is it?

Why it matters:
  For that special base, the slope of the function IS the function.
  Its derivatives are the simplest possible, which is why it appears
  everywhere in AI (sigmoid, softmax, the normal distribution, ...).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

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


def slope(f, x, h=1e-5):
    """Numerical derivative = rise / run over a tiny step h."""
    return (f(x + h) - f(x - h)) / (2 * h)


fig = plt.figure(figsize=(14, 7.8))
fig.canvas.manager.set_window_title("05 — The gradient of an exponential")
header(fig, "Find the base a where  slope = height",
       "first move the point x: does 'slope ÷ height' change?  "
       "Then tune a until the two bars are equal.")
ax1 = fig.add_axes([0.06, 0.30, 0.56, 0.53])
ax2 = fig.add_axes([0.72, 0.30, 0.25, 0.53])

# ---- left: a^x with a tangent line ------------------------------------
xs = np.linspace(-3, 2.5, 400)
(curve,) = ax1.plot([], [], lw=3, color="tab:blue")
(tangent,) = ax1.plot([], [], lw=2.5, color="tab:red", label="tangent line")
(height_line,) = ax1.plot([], [], ":", lw=2, color="tab:purple", label="height")
(point,) = ax1.plot([], [], "o", color="black", ms=9, zorder=5)
(x_hit,) = ax1.plot([], [], "x", color="tab:red", ms=11, mew=3,
                    label="where the tangent hits the x-axis")
info = ax1.text(0.02, 0.97, "", transform=ax1.transAxes, va="top", family="monospace",
                fontsize=10.5, bbox=dict(boxstyle="round", fc="white", ec="gray"))
ax1.set_xlim(-3, 2.5)
ax1.set_ylim(-0.5, 10)
ax1.set_xlabel("x")
ax1.legend(loc="center left", fontsize=9)
draw_axes(ax1)

# ---- right: height vs slope as two bars -------------------------------
bars = ax2.bar([0, 1], [0, 0], color=["tab:purple", "tab:red"], width=0.6)
bar_text = [ax2.text(i, 0, "", ha="center", va="bottom", fontweight="bold") for i in (0, 1)]
ax2.set_xticks([0, 1], ["height\n$a^x$", "slope\nat x"])
ax2.set_xlim(-0.6, 1.6)

# ---- controls ---------------------------------------------------------
s_a = Slider(fig.add_axes([0.13, 0.15, 0.42, 0.035]), "base a", 1.5, 4, valinit=2, valstep=0.001)
s_x = Slider(fig.add_axes([0.13, 0.08, 0.42, 0.035]), "point x", -2, 1.5, valinit=1, valstep=0.01)
b_minus = Button(widget_axes(fig, [0.60, 0.14, 0.045, 0.055]), "− 0.01")
b_plus = Button(widget_axes(fig, [0.65, 0.14, 0.045, 0.055]), "+ 0.01")
b_answer = Button(widget_axes(fig, [0.72, 0.14, 0.12, 0.055]), "Show answer")


def update(_=None):
    a = s_a.val
    x0 = s_x.val
    f = lambda t: a**t                  # the function for the current base
    height = f(x0)
    m = slope(f, x0)                    # measured, not assumed
    ratio = m / height
    x_axis_hit = x0 - height / m        # where the tangent line reaches y = 0

    curve.set_data(xs, f(xs))
    curve.set_label(f"$y = {a:.3f}^x$")
    t = np.array([x_axis_hit, x0 + 0.6])
    tangent.set_data(t, height + m * (t - x0))
    height_line.set_data([x0, x0], [0, height])
    point.set_data([x0], [height])
    x_hit.set_data([x_axis_hit], [0])
    ax1.legend(loc="center left", fontsize=9)

    info.set_text(
        f"height   a^x      = {height:.4f}\n"
        f"slope             = {m:.4f}\n"
        f"slope ÷ height    = {ratio:.4f}\n"
        f"tangent hits the x-axis {x0 - x_axis_hit:.3f} steps to the left")

    for bar, txt, v in zip(bars, bar_text, [height, m]):
        bar.set_height(v)
        txt.set_position((txt.get_position()[0], v))
        txt.set_text(f"{v:.3f}")
    ax2.set_ylim(0, 1.3 * max(height, m))
    ax2.set_title(f"slope ÷ height = {ratio:.3f}")

    if abs(ratio - 1) < 0.01:
        ax1.set_title(f"You found it!  a ≈ {a:.3f}   (exactly: e = 2.71828...)",
                      color="tab:green", fontweight="bold")
    elif ratio < 1:
        ax1.set_title("slope is SMALLER than height  →  try a bigger a", color="black",
                      fontweight="normal")
    else:
        ax1.set_title("slope is BIGGER than height  →  try a smaller a", color="black",
                      fontweight="normal")
    fig.canvas.draw_idle()


def nudge(amount):
    return lambda _: s_a.set_val(np.clip(s_a.val + amount, 1.5, 4))


# ---- click and drag on the left plot to move the point ---------------
state = {"dragging": False}


def on_press(event):
    toolbar_busy = getattr(fig.canvas.toolbar, "mode", "") != ""
    if event.inaxes is ax1 and event.button == 1 and not toolbar_busy:
        state["dragging"] = True
        s_x.set_val(np.clip(event.xdata, -2, 1.5))


def on_move(event):
    if state["dragging"] and event.inaxes is ax1:
        s_x.set_val(np.clip(event.xdata, -2, 1.5))


def on_release(event):
    state["dragging"] = False


fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("button_release_event", on_release)

s_a.on_changed(update)
s_x.on_changed(update)
b_minus.on_clicked(nudge(-0.01))
b_plus.on_clicked(nudge(+0.01))
b_answer.on_clicked(lambda _: s_a.set_val(np.e))
update()
plt.show()