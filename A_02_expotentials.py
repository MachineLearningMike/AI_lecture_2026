"""
02 — Exponential functions  y = a^x   and  ln(x)              (interactive)
Run:  python 02_exponentials.py

Controls (one under each panel):
  (a) a slider        -> change the base of a^x
  (b) power n slider  -> race x^n against 2^x;  'log scale' checkbox
  (c) p slider        -> a point on e^x and its mirror image on ln(x)

What to notice:
  (a) Every a^x passes through (0, 1), because a^0 = 1.
      a > 1 grows, a < 1 decays, a = 1 stays flat.
  (b) x^n can be ahead for a while (orange zone), but 2^x ALWAYS takes
      the lead for good. An exponential beats any power, eventually.
      On a log scale an exponential becomes a straight line.
  (c) ln(x) "undoes" e^x: e^p = v  means  ln(v) = p.
      The two points are mirror images across the line y = x.
      In AI, the cross-entropy loss is -ln(p).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons

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


fig = plt.figure(figsize=(17, 7.2))
fig.canvas.manager.set_window_title("02 — Exponentials")
header(fig, "Exponential functions   $y = a^x$   (x is in the exponent)",
       "(a) push a below 1, then exactly to 1.   (b) raise the power n: can $x^n$ stay ahead "
       "forever?   (c) move p: ln undoes $e^x$.")
axA = fig.add_axes([0.045, 0.27, 0.28, 0.56])
axB = fig.add_axes([0.375, 0.27, 0.28, 0.56])
axC = fig.add_axes([0.70, 0.27, 0.28, 0.56])

# ======================================================================
# (a) the family a^x
# ======================================================================
xa = np.linspace(-3, 3, 400)
for a in [0.5, 2, 3]:
    axA.plot(xa, a**xa, color="lightgray", lw=1)
(curveA,) = axA.plot([], [], lw=3, color="tab:blue")
axA.plot(0, 1, "ko", ms=7)
axA.annotate("(0, 1):  $a^0 = 1$  for every a", (0, 1), xytext=(-2.9, 6.8),
             arrowprops=dict(arrowstyle="->"))
(pA,) = axA.plot([], [], "o", color="tab:red", ms=8)
labelA = axA.annotate("", (0, 0), xytext=(8, -18), textcoords="offset points",
                      color="tab:red", fontweight="bold")
axA.set_xlim(-3, 3)
axA.set_ylim(-0.5, 8)
axA.set_xlabel("x")
draw_axes(axA)
s_a = Slider(fig.add_axes([0.07, 0.12, 0.24, 0.035]), "a", 0.2, 4, valinit=2, valstep=0.01)


def update_a(_=None):
    a = round(s_a.val, 2)
    curveA.set_data(xa, a**xa)
    pA.set_data([1], [a])
    labelA.xy = (1, min(a, 7.8))
    labelA.set_text(f"$a^1$ = a = {a:.2f}")
    if a == 1:
        what = "a = 1: FLAT (1 to any power is 1)"
    elif a > 1:
        what = f"a = {a:.2f} > 1: GROWS"
    else:
        what = f"a = {a:.2f} < 1: DECAYS"
    axA.set_title(f"(a)  $y = a^x$     {what}")
    fig.canvas.draw_idle()


# ======================================================================
# (b) the race: x^n  vs  2^x
# ======================================================================
(powB,) = axB.plot([], [], lw=2.5, color="tab:orange")
(expB,) = axB.plot([], [], lw=2.5, color="tab:green", label="$2^x$")
(crossB,) = axB.plot([], [], "ko", ms=8)
axB.set_xlabel("x")
shade = {"B": None}                 # remember the orange zone so we can remove it
s_pow = Slider(fig.add_axes([0.405, 0.12, 0.15, 0.035]), "power n", 1, 6, valinit=3, valstep=1)
c_log = CheckButtons(widget_axes(fig, [0.585, 0.085, 0.075, 0.08]), ["log scale"], [False])


def update_b(_=None):
    n = int(s_pow.val)
    # where is x^n ahead of 2^x?  (compare logs:  n*ln(x) > x*ln(2))
    grid = np.linspace(0.01, 60, 60_000)
    ahead = n * np.log(grid) > grid * np.log(2)
    if ahead.any():
        cross = grid[np.where(ahead)[0][-1] + 1]      # last moment x^n is ahead
        xmax = 1.35 * cross
    else:
        cross, xmax = None, 10
    xb = np.linspace(0.01, xmax, 500)
    powB.set_data(xb, xb**n)
    powB.set_label(f"$x^{{{n}}}$")
    expB.set_data(xb, 2**xb)
    if shade["B"] is not None:
        shade["B"].remove()
    shade["B"] = axB.fill_between(xb, 2**xb, xb**n, where=xb**n > 2**xb,
                                  color="tab:orange", alpha=0.2, label=f"$x^{{{n}}}$ is ahead")
    axB.set_xlim(0, xmax)

    top = max(2**xmax, xmax**n)
    if c_log.get_status()[0]:
        axB.set_yscale("log")
        axB.set_ylim(0.1, top * 3)
    else:
        axB.set_yscale("linear")
        axB.set_ylim(0, 2.5 * cross**n if cross else 1.1 * top)

    if cross:
        crossB.set_data([cross], [cross**n])
        axB.set_title(f"(b)  race: $2^x$ passes $x^{{{n}}}$ for good at x ≈ {cross:.2f}")
    else:
        crossB.set_data([], [])
        axB.set_title(f"(b)  $2^x$ is ahead of $x^{{{n}}}$ everywhere")
    axB.legend(loc="upper left")
    fig.canvas.draw_idle()


# ======================================================================
# (c) e^x and its mirror image ln(x)
# ======================================================================
xc = np.linspace(-4, 2.2, 400)
xp = np.linspace(0.02, 7, 400)
axC.plot(xc, np.exp(xc), lw=2.5, color="tab:blue", label="$e^x$")
axC.plot(xp, np.log(xp), lw=2.5, color="tab:orange", label="$\\ln(x)$")
axC.plot([-4, 7], [-4, 7], "k--", lw=1, label="mirror line  y = x")
(link,) = axC.plot([], [], ":", color="black", lw=1.5)
(pP,) = axC.plot([], [], "o", color="tab:blue", ms=10)
(pQ,) = axC.plot([], [], "o", color="tab:orange", ms=10)
axC.set_xlim(-4, 7)
axC.set_ylim(-4, 7)
axC.set_aspect("equal")
axC.set_xlabel("x")
axC.legend(loc="lower right")
draw_axes(axC)
s_p = Slider(fig.add_axes([0.73, 0.12, 0.22, 0.035]), "p", -2.5, 1.9, valinit=1, valstep=0.01)


def update_c(_=None):
    p = s_p.val
    v = np.exp(p)
    pP.set_data([p], [v])
    pQ.set_data([v], [p])
    link.set_data([p, v], [v, p])
    axC.set_title(f"(c)  $e^{{{p:.2f}}}$ = {v:.2f}    so    ln({v:.2f}) = {p:.2f}")
    fig.canvas.draw_idle()


s_a.on_changed(update_a)
s_pow.on_changed(update_b)
c_log.on_clicked(update_b)
s_p.on_changed(update_c)
update_a()
update_b()
update_c()
plt.show()