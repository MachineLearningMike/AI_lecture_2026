"""
03 — sin and cos: the slope of sin IS cos                     (interactive)
Run:  python 03_sin_cos.py

Controls:
  x slider / click-and-drag on a plot -> move the red point along sin(x)
  Play          -> sweep the point automatically from left to right
  Clear dots    -> erase the black dots
  show cos(x)   -> reveal the answer

The game:
  The red tangent line touches sin(x) at one point. Its steepness is the
  SLOPE of sin at that point. Every slope you measure leaves a black dot
  in the bottom panel (at the same x, with height = slope).
  Sweep across and the dots draw a curve. Which curve is it?

Answer: cos(x).  So  d/dx sin(x) = cos(x).
  sin at its peak (flat)       -> slope 0   -> cos = 0
  sin climbing steepest        -> slope +1  -> cos = +1
  sin falling steepest         -> slope -1  -> cos = -1
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons

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


TWO_PI = 2 * np.pi
X = np.linspace(-TWO_PI, TWO_PI, 600)

fig, (top, bot) = plt.subplots(2, 1, figsize=(13, 8.5), sharex=True)
fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.19, hspace=0.3)
fig.canvas.manager.set_window_title("03 — sin and cos")
header(fig, "The slope of sin(x): what curve does it make?",
       "press Play (or drag the red point). Watch the black dots appear below. "
       "Guess the curve, then tick 'show cos(x)'.")

# ---- top: sin(x) with a tangent line ---------------------------------
top.plot(X, np.sin(X), lw=2.5, color="tab:blue", label="sin(x)")
(tangent,) = top.plot([], [], lw=3, color="tab:red", label="tangent line")
(p_top,) = top.plot([], [], "o", color="tab:red", ms=10)
slope_txt = top.text(0, 0, "", color="tab:red", fontweight="bold", ha="center")
v_top = top.axvline(0, color="tab:red", ls=":", lw=1.2)
top.set_ylim(-1.7, 1.9)
top.set_title("sin(x) and its tangent line: the tangent's steepness is the slope")
top.legend(loc="upper right")

# ---- bottom: the slopes you measured ---------------------------------
(cos_line,) = bot.plot(X, np.cos(X), lw=2.5, color="tab:orange", label="cos(x)", zorder=1)
cos_line.set_visible(False)
(trail,) = bot.plot([], [], ".", color="black", ms=6, label="slopes you measured", zorder=3)
(p_bot,) = bot.plot([], [], "o", color="tab:red", ms=11, zorder=5)
v_bot = bot.axvline(0, color="tab:red", ls=":", lw=1.2)
info = bot.text(0.01, 0.95, "", transform=bot.transAxes, va="top", family="monospace",
                fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="gray"))
bot.set_ylim(-1.7, 1.7)
bot.set_ylabel("slope of sin")

ticks = np.arange(-4, 5) * np.pi / 2
labels = ["$-2\\pi$", "$-3\\pi/2$", "$-\\pi$", "$-\\pi/2$", "0",
          "$\\pi/2$", "$\\pi$", "$3\\pi/2$", "$2\\pi$"]
bot.set_xticks(ticks, labels)
bot.set_xlabel("x")
for ax in (top, bot):
    draw_axes(ax)

# ---- controls ---------------------------------------------------------
s_x = Slider(fig.add_axes([0.10, 0.085, 0.50, 0.035]), "x", -TWO_PI, TWO_PI,
             valinit=-TWO_PI, valstep=0.01)
b_play = Button(widget_axes(fig, [0.655, 0.07, 0.09, 0.06]), "Play")
b_clear = Button(widget_axes(fig, [0.755, 0.07, 0.09, 0.06]), "Clear dots")
c_cos = CheckButtons(widget_axes(fig, [0.855, 0.055, 0.13, 0.09]), ["show cos(x)"], [False])

trail_x, trail_y = [], []            # every slope measured so far
state = {"playing": False, "dragging": False}


def update(_=None):
    x0 = s_x.val
    y0 = np.sin(x0)
    m = slope(np.sin, x0)
    show_cos = c_cos.get_status()[0]

    t = np.array([x0 - 0.9, x0 + 0.9])
    tangent.set_data(t, y0 + m * (t - x0))
    p_top.set_data([x0], [y0])
    # put the label on the side where the tangent line is NOT
    if m > 0.1:
        slope_txt.set_position((x0 - 0.2, y0 + 0.25))
        slope_txt.set_ha("right")
    elif m < -0.1:
        slope_txt.set_position((x0 + 0.2, y0 + 0.25))
        slope_txt.set_ha("left")
    else:
        slope_txt.set_position((x0, y0 + 0.3))
        slope_txt.set_ha("center")
    slope_txt.set_text(f"slope = {round(m, 2) + 0.0:+.2f}")
    v_top.set_xdata([x0, x0])

    trail_x.append(x0)
    trail_y.append(m)
    trail.set_data(trail_x, trail_y)
    p_bot.set_data([x0], [m])
    v_bot.set_xdata([x0, x0])

    text = f"x            = {x0:+.2f}\nslope of sin = {m:+.3f}"
    if show_cos:
        text += f"\ncos(x)       = {np.cos(x0):+.3f}"
    info.set_text(text)
    fig.canvas.draw_idle()


def toggle_cos(_=None):
    show = c_cos.get_status()[0]
    cos_line.set_visible(show)
    handles = [trail, cos_line] if show else [trail]
    bot.legend(handles=handles, loc="upper right")
    if show:
        bot.set_title("The dots sit exactly on cos(x):    d/dx sin(x) = cos(x)")
    else:
        bot.set_title("Each slope you measure leaves a black dot here. What curve is it making?")
    update_info_only()


def update_info_only():
    """Refresh the text box without adding a new dot."""
    x0 = s_x.val
    text = f"x            = {x0:+.2f}\nslope of sin = {slope(np.sin, x0):+.3f}"
    if c_cos.get_status()[0]:
        text += f"\ncos(x)       = {np.cos(x0):+.3f}"
    info.set_text(text)
    fig.canvas.draw_idle()


def clear(_=None):
    trail_x.clear()
    trail_y.clear()
    trail.set_data([], [])
    fig.canvas.draw_idle()


# ---- Play button: a timer moves the slider a little every 40 ms ------
timer = fig.canvas.new_timer(interval=40)


def step():
    new_x = s_x.val + 0.05
    if new_x > TWO_PI:
        stop()
    else:
        s_x.set_val(new_x)


def stop():
    timer.stop()
    state["playing"] = False
    b_play.label.set_text("Play")
    fig.canvas.draw_idle()


def play(_=None):
    if state["playing"]:
        stop()
        return
    if s_x.val >= TWO_PI - 0.05:      # at the end? start again from the left
        s_x.set_val(-TWO_PI)
    state["playing"] = True
    b_play.label.set_text("Pause")
    timer.start()


timer.add_callback(step)


# ---- click and drag on either plot to move the point -----------------
def on_press(event):
    toolbar_busy = getattr(fig.canvas.toolbar, "mode", "") != ""
    if event.inaxes in (top, bot) and event.button == 1 and not toolbar_busy:
        state["dragging"] = True
        s_x.set_val(np.clip(event.xdata, -TWO_PI, TWO_PI))


def on_move(event):
    if state["dragging"] and event.inaxes in (top, bot):
        s_x.set_val(np.clip(event.xdata, -TWO_PI, TWO_PI))


def on_release(event):
    state["dragging"] = False


fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("button_release_event", on_release)

s_x.on_changed(update)
b_play.on_clicked(play)
b_clear.on_clicked(clear)
c_cos.on_clicked(toggle_cos)
toggle_cos()
update()
plt.show()