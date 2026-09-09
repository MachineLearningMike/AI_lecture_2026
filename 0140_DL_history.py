"""
012_historical_achievements_by_DL.py

Historical problems cracked by deep learning -- and the ideas that made them possible.

Pedagogy (adapted and tightened from a historical survey outline):
  - Not only a scoreboard of benchmarks.
  - Separate enabling ideas from headline systems.
  - Credit the people who opened the door AND those who delivered scale.
  - Compress the history into one claim:
      Deep learning made representation itself a learnable object.

Sketch:
  1. List: historical problems DL cracked
  2. Not a better gadget -- a change of question
  3. Timeline of headline breakthroughs
  4. Five kinds of historically hard problems
  5. Interactive: walk the five kinds
  6. Corrections: AlexNet, AlphaGo, AlphaFold
  7. Enabling ideas vs problems cracked
  8. The Godfathers framing -- useful but incomplete
  9. Historical progression ladder
  10. What deep learning has not solved (yet)
  11. Takeaway for this course
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
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

np.random.seed(12)


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


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=2.4):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, mutation_scale=16),
    )


C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_WARN = "#f8d7da"
C_FUN = "#8e44ad"
C_BIN = "#d68910"
C_GRAY = "#85929e"
C_ACTIVE = "#f9e79f"

# Approximate data-units per cm on lecture axes (measured for screen_figsize).
# blank_slide (0..12 x, 0..8 y): ~22.5cm x 11.9cm
CM_X = 0.533
CM_Y = 0.673
# Two-panel Step 5 left axis (0..10 y): ~11.1cm tall
CM_Y_STEP5 = 0.899
# Two-panel Step 7 axis (0..12 y): ~11.9cm tall
CM_Y_STEP7 = 1.009


# -------------------------------------------------------
# FIGURE 0: Full list first
# -------------------------------------------------------
fig0, ax0 = new_figure()
fig0.suptitle(
    "Step 1: Historical problems deep learning cracked -- the list",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax0)

_s1_dy = 2 * CM_Y          # move content down 2cm
_s1_dh = 1 * CM_Y          # shrink box height 1cm (tighten lead/body gap)
_s1_h = 3.3 - _s1_dh       # ~2.63
_s1_y_top = 5.55 - _s1_dy  # ~4.20
_s1_y_bot = 1.55 - _s1_dy  # ~0.20


problems_list = [
    (0.35, _s1_y_top, "2012+",
     "Image classification\nObject detection\nFace recognition"),
    (3.25, _s1_y_top, "2013+",
     "Speech recognition\nSpeech synthesis\nAtari / game control"),
    (6.15, _s1_y_top, "2014+",
     "Machine translation\nImage captioning\nImage generation (GAN)"),
    (9.05, _s1_y_top, "2016+",
     "Go (AlphaGo)\nSelf-play mastery\nSearch guided by nets"),
    (0.35, _s1_y_bot, "2017+",
     "Transformers\nLanguage understanding\nNeural TTS quality"),
    (3.25, _s1_y_bot, "2018+",
     "BERT / GPT era\nPretrained language\nTransfer learning"),
    (6.15, _s1_y_bot, "2020+",
     "Protein folding\nLarge language models\nText-to-image"),
    (9.05, _s1_y_bot, "2022+",
     "AI assistants\nCode generation\nWeather / science nets"),
]
for x0, y0, era, body in problems_list:
    upper = y0 > 3.0
    round_box(ax0, x0, y0, 2.7, _s1_h, C_GOAL if upper else C_NEW,
              C_OK if upper else "#2874a6", lw=1.8)
    # Lead near top; body kept from bottom -- gap narrows as height shrinks
    ax0.text(x0 + 1.35, y0 + _s1_h - 0.60, era, ha="center",
             fontsize=FS_ANNOT, fontweight="bold",
             color=C_OK if upper else "#2874a6")
    ax0.text(x0 + 1.35, y0 + 1.30, body, ha="center", va="center",
             fontsize=FS_TICK + 1)

ax0.text(
    6.0, 0.08 - CM_Y,  # bottom caption down 1cm
    "Next slides: why these count as breakthroughs -- and how to credit them carefully.",
    ha="center", fontsize=FS_ANNOT, fontweight="bold", color=C_DARK,
    clip_on=False,
)
show_maximized(fig0)


# -------------------------------------------------------
# FIGURE 1: Change of question
# -------------------------------------------------------
fig1, ax1 = new_figure()
fig1.suptitle(
    "Step 2: Deep learning's historical contribution is a change of question",
    fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
)
blank_slide(ax1)

round_box(ax1, 0.5, 5.2, 11.0, 2.3, C_DARK, C_DARK, lw=0)
ax1.text(
    6.0, 6.35,
    "Not merely a better machine-learning technique.\n"
    "It made representation itself a learnable object.",
    ha="center", va="center", fontsize=FS_ANNOT + 2,
    fontweight="bold", color="#f9e79f",
)

round_box(ax1, 0.5, 1.0, 5.3, 3.7, C_WARN, C_ACCENT, lw=2.2)
ax1.text(3.15, 4.0, "Old question", ha="center",
         fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_ACCENT)
ax1.text(
    3.15, 2.4,
    '"What features should I\ngive the machine?"',
    ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold",
)

round_box(ax1, 6.2, 1.0, 5.3, 3.7, C_GOAL, C_OK, lw=2.2)
ax1.text(8.85, 4.0, "New question", ha="center",
         fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_OK)
ax1.text(
    8.85, 2.4,
    '"Can the machine discover\nthe features?"',
    ha="center", va="center", fontsize=FS_ANNOT + 2, fontweight="bold",
)
show_maximized(fig1)


# -------------------------------------------------------
# FIGURE 2: Timeline of headline breakthroughs
# -------------------------------------------------------
fig2, ax2 = new_figure()
fig2.suptitle(
    "Step 3: Headline breakthroughs -- from the DL reboot to recent years",
    fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
)
blank_slide(ax2)

# Two bands so recent years are not squeezed off the timeline
era_early = [
    (2012, "AlexNet", "ImageNet / CNN scale"),
    (2014, "GAN", "adversarial generation"),
    (2015, "DQN", "Atari from pixels"),
    (2016, "AlphaGo", "Go + search guidance"),
    (2017, "Transformer", "attention architecture"),
    (2018, "BERT / GPT-1", "pretrained language"),
]
era_recent = [
    (2019, "GPT-2", "fluent long-form text"),
    (2020, "GPT-3 / AlphaFold", "scale + structure"),
    (2021, "CLIP / DALL-E", "vision-language"),
    (2022, "Diffusion / ChatGPT", "gen. + assistants"),
    (2023, "GPT-4 era", "multimodal LLMs"),
    (2024, "o1 / science nets", "reason + GraphCast"),
    (2025, "Open / efficient LLMs", "DeepSeek-class scale"),
    (2026, "Agents / tools", "act, use tools, loop"),
]


def _draw_band(y_spine, items, face, edge, recent=False):
    # Shorten stems from axis to cards by ~2cm (leave a small stub).
    lead_up = max(0.20, 1.15 - 2 * CM_Y)
    lead_dn = max(0.20, 0.70 - 2 * CM_Y)
    box_h = 1.85
    n = len(items)
    xs = np.linspace(1.0, 11.0, n)
    ax2.plot([0.7, 11.3], [y_spine, y_spine], color=C_DARK, lw=2.6,
             zorder=1, clip_on=False)
    for i, ((year, name, note), x) in enumerate(zip(items, xs)):
        up = i % 2 == 0
        y_box = y_spine + lead_up if up else y_spine - lead_dn - box_h
        stem_end = y_box if up else y_box + box_h
        ax2.scatter([x], [y_spine], s=55,
                    c=C_ACCENT if recent else C_FUN, zorder=3, clip_on=False)
        ax2.plot([x, x], [y_spine, stem_end], color="0.55", lw=1.1,
                 clip_on=False)
        patch = round_box(ax2, x - 0.62, y_box, 1.24, box_h, face, edge, lw=1.4)
        patch.set_clip_on(False)
        ax2.text(x, y_box + 1.42, str(year), ha="center",
                 fontsize=FS_TICK - 2, fontweight="bold", color=C_DARK,
                 clip_on=False)
        ax2.text(x, y_box + 0.95, name, ha="center",
                 fontsize=9, fontweight="bold", clip_on=False)
        ax2.text(x, y_box + 0.35, note, ha="center", va="center", fontsize=8,
                 clip_on=False)


# 1st axis: up 0.5in then down 1in; 2nd axis: down 0.5in then down 1cm.
_s3_half_in = 0.5 * 2.54 * CM_Y
_s3_y1 = 6.35 + _s3_half_in - 1.0 * 2.54 * CM_Y
_s3_y2 = 2.55 - _s3_half_in - 1.0 * CM_Y
# Era labels net left 1in from 0.4 (left 2in, then right 1in); down 2cm.
_s3_label_x = 0.4 - 1.0 * 2.54 * CM_X
_s3_label_dy = 2.0 * CM_Y
ax2.text(_s3_label_x, _s3_y1 + 1.0 - _s3_label_dy, "2012 -- 2018", ha="left",
         fontsize=FS_ANNOT, fontweight="bold", color="#2874a6",
         clip_on=False)
_draw_band(_s3_y1, era_early, C_NEW, "#2874a6", recent=False)

ax2.text(_s3_label_x, _s3_y2 + 1.0 - _s3_label_dy, "2019 -- 2026", ha="left",
         fontsize=FS_ANNOT, fontweight="bold", color=C_ACCENT,
         clip_on=False)
_draw_band(_s3_y2, era_recent, C_GOAL, C_OK, recent=True)
show_maximized(fig2)


# -------------------------------------------------------
# FIGURE 3: Five kinds of historically hard problems
# -------------------------------------------------------
fig3, ax3 = new_figure()
fig3.suptitle(
    "Step 4: Five kinds of historically hard problems DL attacked",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax3)

kinds = [
    (0.35, 4.4, "1. Features\nhumans couldn't design", "Vision: edges→objects\nbecame learned.", C_NEW, "#2874a6"),
    (4.25, 4.4, "2. Too many interacting\nvariables", "Speech / translation:\nend-to-end maps.", "#f5eef8", C_FUN),
    (8.15, 4.4, "3. Enormous\nsearch spaces", "Go: nets guide\nwhere to look.", C_GOAL, C_OK),
    (2.2, 0.55, "4. Extremely complicated\nmappings", "Protein folding:\ninfer hidden structure.", C_WARN, C_ACCENT),
    (6.3, 0.55, "5. Hard-to-specify\nobjectives", "Prompt→answer,\ntext→image, code…", C_MUTE, C_DARK),
]
for x0, y0, title, body, face, edge in kinds:
    round_box(ax3, x0, y0, 3.5, 3.2, face, edge, lw=2.0)
    ax3.text(x0 + 1.75, y0 + 2.35, title, ha="center",
             fontsize=FS_TICK + 1, fontweight="bold", color=edge)
    ax3.text(x0 + 1.75, y0 + 1.0, body, ha="center", va="center",
             fontsize=FS_TICK + 1)
show_maximized(fig3)


# -------------------------------------------------------
# FIGURE 4: Interactive walk of the five kinds
# -------------------------------------------------------
fig4, (ax4a, ax4b) = new_figure(1, 2)
fig4.suptitle(
    "Step 5: Walk the five kinds (Prev / Next)",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
fig4.subplots_adjust(bottom=0.16, wspace=0.25)

KIND_DETAIL = [
    ("Features we could not design",
     "Before CNNs: hand-crafted visual pipelines.\n"
     "AlexNet (2012): large CNN + GPU + big data.\n\n"
     "Conceptual shift: stop prescribing features;\nlearn them.",
     "Krizhevsky, Sutskever, Hinton\n(building on LeCun's CNNs)"),
    ("Too many interacting variables",
     "Speech and translation once meant many\nengineered stages.\n\n"
     "Neural MT / deep acoustic models learned\nintermediate representations end-to-end.",
     "Sutskever, Vinyals, Le;\nBahdanau, Cho, Bengio;\nspeech DNN teams"),
    ("Enormous search spaces",
     "Go's tree is astronomical.\n\n"
     "AlphaGo: policy/value nets + RL + MCTS.\n"
     "Deep learning learned where to search.",
     "Silver, Huang, Hassabis\n+ DeepMind team"),
    ("Extremely complicated mappings",
     "Sequence → 3D structure resisted\nclassical search.\n\n"
     "AlphaFold learned structural constraints\nfrom biological data.",
     "Jumper, Evans, Senior, Hassabis\n+ DeepMind team"),
    ("Hard-to-specify objectives",
     "No closed-form algorithm for\n'good answer' or 'good image'.\n\n"
     "LLMs / diffusion learn the mapping\nas a statistical model of experience.",
     "Many labs: OpenAI, Google DeepMind,\nStability, Meta, ..."),
]


def draw_kind(k):
    ax4a.clear(); ax4b.clear()
    k = int(np.clip(k, 0, 4))
    ax4a.set_xlim(0, 10); ax4a.set_ylim(0, 10); ax4a.axis("off")
    # Left panel: down 2cm, then up 1cm.
    _s5_dy = 2 * CM_Y_STEP5 - 1 * CM_Y_STEP5
    _s5_y0 = 8.2 - _s5_dy
    ax4a.text(5.0, _s5_y0 + 1.25 + 0.40, "kind of hardness",
              ha="center", va="center",
              fontsize=FS_AX_TITLE - 2, fontweight="bold", color=C_DARK)
    for i in range(5):
        y = _s5_y0 - i * 1.5
        face = C_ACTIVE if i == k else C_MUTE
        edge = C_ACCENT if i == k else "0.55"
        round_box(ax4a, 0.6, y, 8.8, 1.25, face, edge, lw=2.0 if i == k else 1.3)
        ax4a.text(5.0, y + 0.62, KIND_DETAIL[i][0], ha="center", va="center",
                  fontsize=FS_ANNOT, fontweight="bold",
                  color=C_DARK if i == k else "0.35")

    title, body, people = KIND_DETAIL[k]
    ax4b.set_xlim(0, 10); ax4b.set_ylim(0, 10); ax4b.axis("off")
    round_box(ax4b, 0.5, 1.0, 9.0, 8.0, C_NEW, "#2874a6", lw=2.2)
    ax4b.text(5.0, 8.2, title, ha="center", fontsize=FS_AX_TITLE - 3,
              fontweight="bold", color=C_DARK)
    ax4b.text(5.0, 5.3, body, ha="center", va="center", fontsize=FS_ANNOT)
    ax4b.text(5.0, 1.8, people, ha="center", va="center",
              fontsize=FS_TICK + 1, style="italic", color="#2874a6")


ax_k = fig4.add_axes([0.30, 0.05, 0.40, 0.04])
ax_prev = fig4.add_axes([0.14, 0.045, 0.10, 0.045])
ax_next = fig4.add_axes([0.76, 0.045, 0.10, 0.045])
sl_k = Slider(ax_k, "kind", 0, 4, valinit=0, valstep=1, color=C_FUN)
btn_p = Button(ax_prev, "Prev", color=C_MUTE, hovercolor=C_NEW)
btn_n = Button(ax_next, "Next", color=C_MUTE, hovercolor=C_GOAL)


def _set_k(v):
    v = int(np.clip(v, 0, 4))
    if int(sl_k.val) == v:
        draw_kind(v); fig4.canvas.draw_idle()
    else:
        sl_k.set_val(v)


def _uk(_=None):
    draw_kind(sl_k.val); fig4.canvas.draw_idle()


def _prev(_=None):
    _set_k(int(sl_k.val) - 1)


def _next(_=None):
    _set_k(int(sl_k.val) + 1)


sl_k.on_changed(_uk)
btn_p.on_clicked(_prev)
btn_n.on_clicked(_next)
_uk()
print("Step 5: walk the five kinds.")
show_maximized(fig4)


# -------------------------------------------------------
# FIGURE 5: Important corrections
# -------------------------------------------------------
fig5, ax5 = new_figure()
fig5.suptitle(
    "Step 6: Credit carefully -- three common oversimplifications",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)
blank_slide(ax5)

_s6_w = 5.4 + 3 * CM_X   # extend textbox width by 3cm (~7.0)
_s6_x = (12.0 - _s6_w) / 2.0
# Wider cards no longer fit side-by-side; stack them and keep readable heights.
corr = [
    (_s6_x, 5.35, 2.25, "AlexNet ≠ invention of CNNs",
     "LeCun et al. built practical CNNs decades earlier.\n"
     "2012 showed: large CNN + ImageNet + GPU + ReLU/dropout\n"
     "could crush the old vision pipeline."),
    (_s6_x, 2.85, 2.25, "AlphaGo ≠ \"just a neural net\"",
     "Policy/value nets + RL + Monte Carlo tree search.\n"
     "Deep learning learned how to guide search."),
    (_s6_x, 0.35, 2.25, "AlphaFold ≠ one inventor",
     "Jumper & Hassabis led; Evans, Senior and a large\n"
     "DeepMind team built architecture, data, inference.\n"
     "Credit the team, not a slogan."),
]
for x0, y0, h, title, body in corr:
    round_box(ax5, x0, y0, _s6_w, h, C_MUTE, C_DARK, lw=1.8)
    ax5.text(x0 + _s6_w / 2, y0 + h - 0.40, title, ha="center",
             fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    ax5.text(x0 + _s6_w / 2, y0 + h / 2 - 0.15, body, ha="center", va="center",
             fontsize=FS_TICK + 1)
show_maximized(fig5)


# -------------------------------------------------------
# FIGURE 6: Enabling ideas vs problems cracked
# -------------------------------------------------------
fig6, axes6 = new_figure(1, 2)
fig6.suptitle(
    "Step 7: Separate enabling ideas from problems later cracked",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

_s7_dy = 2 * CM_Y_STEP7  # move both panels down by 2cm
_s7_title_up = 1.0 * 2.54 * CM_Y_STEP7  # panel titles up 1 inch
_s7_title_y = 11.55 - _s7_dy + _s7_title_up - 2.0 * CM_Y_STEP7  # then down 2cm
axes6[0].set_xlim(0, 10); axes6[0].set_ylim(0, 12); axes6[0].axis("off")
axes6[0].text(5.0, _s7_title_y, "Enabling breakthroughs",
              ha="center", va="center", fontsize=FS_AX_TITLE - 2,
              fontweight="bold", color="#2874a6", clip_on=False)
enablers = [
    "1943  McCulloch & Pitts — neuron",
    "1958  Rosenblatt — perceptron",
    "1986  Rumelhart, Hinton, Williams — backprop",
    "1989–98  LeCun et al. — CNNs that work",
    "1997  Hochreiter & Schmidhuber — LSTM",
    "2006  Hinton et al. — deep nets trainable again",
    "2012  Krizhevsky, Sutskever, Hinton — AlexNet",
    "2014  Goodfellow et al. — GANs",
    "2017  Vaswani et al. — Transformer",
]
# Slightly tighter stride so the shifted list still clears y=0.
_s7_step_e = min(1.1, (10.5 - _s7_dy - 0.20) / (len(enablers) - 1))
for i, line in enumerate(enablers):
    y = 10.5 - _s7_dy - i * _s7_step_e
    round_box(axes6[0], 0.4, y, 9.2, 0.95, C_NEW, "#2874a6", lw=1.3)
    axes6[0].text(5.0, y + 0.45, line, ha="center", va="center", fontsize=FS_TICK)

axes6[1].set_xlim(0, 10); axes6[1].set_ylim(0, 12); axes6[1].axis("off")
axes6[1].text(5.0, _s7_title_y, "Problems subsequently cracked",
              ha="center", va="center", fontsize=FS_AX_TITLE - 2,
              fontweight="bold", color=C_OK, clip_on=False)
cracked = [
    "Handwriting / digit recognition",
    "Image classification & detection",
    "Speech recognition & synthesis",
    "Face recognition embeddings",
    "Machine translation",
    "Image / text generation",
    "Game playing (Atari, Go, ...)",
    "Protein structure prediction",
    "Code generation & assistants",
    "Competitive weather forecasting",
]
_s7_step_c = min(1.0, (10.7 - _s7_dy - 0.20) / (len(cracked) - 1))
for i, line in enumerate(cracked):
    y = 10.7 - _s7_dy - i * _s7_step_c
    round_box(axes6[1], 0.4, y, 9.2, 0.88, C_GOAL, C_OK, lw=1.3)
    axes6[1].text(5.0, y + 0.42, line, ha="center", va="center", fontsize=FS_TICK)
show_maximized(fig6)


# -------------------------------------------------------
# FIGURE 7: Three pillars
# -------------------------------------------------------
fig7, ax7 = new_figure()
fig7.suptitle(
    "Step 8: LeCun / Hinton / Bengio -- useful pillars, not the whole story",
    fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
)
blank_slide(ax7)

pillars = [
    (0.5, 1.5, "Yann LeCun", "CNNs &\npractical visual\ngradient learning", C_NEW, "#2874a6"),
    (4.25, 1.5, "Geoffrey Hinton", "Backprop advocacy,\ndeep representations,\nneural revival", C_GOAL, C_OK),
    (8.0, 1.5, "Yoshua Bengio", "Representation &\nlanguage/sequence\nlearning foundations", "#f5eef8", C_FUN),
]
for x0, y0, name, body, face, edge in pillars:
    round_box(ax7, x0, y0, 3.5, 5.5, face, edge, lw=2.2)
    ax7.text(x0 + 1.75, y0 + 4.5, name, ha="center",
             fontsize=FS_ANNOT + 1, fontweight="bold", color=edge)
    ax7.text(x0 + 1.75, y0 + 2.4, body, ha="center", va="center",
             fontsize=FS_ANNOT + 1)

ax7.text(
    6.0, 0.6,
    "A multi-decade accumulation -- not one breakthrough paper.",
    ha="center", fontsize=FS_ANNOT + 1, fontweight="bold", color=C_DARK,
)
show_maximized(fig7)


# -------------------------------------------------------
# FIGURE 8: Progression ladder
# -------------------------------------------------------
fig8, ax8 = new_figure()
fig8.suptitle(
    "Step 9: The deeper historical pattern",
    fontsize=FS_FIGURE_TITLE, fontweight="bold",
)
blank_slide(ax8)

ladder = [
    "Hand-coded rules",
    "Hand-designed features + statistical models",
    "Learned features + statistical models",
    "End-to-end learned representations",
    "Large pretrained representations",
    "Foundation models",
    "One model, many formerly separate tasks",
]
for i, line in enumerate(ladder):
    y = 6.7 - i * 0.9
    face = C_GOAL if i >= 4 else (C_NEW if i >= 2 else C_MUTE)
    edge = C_OK if i >= 4 else ("#2874a6" if i >= 2 else "0.45")
    round_box(ax8, 1.5, y, 9.0, 0.75, face, edge, lw=1.6)
    ax8.text(6.0, y + 0.38, line, ha="center", va="center",
             fontsize=FS_ANNOT, fontweight="bold", color=C_DARK)
    if i < len(ladder) - 1:
        arrow(ax8, 6.0, y - 0.02, 6.0, y - 0.12, edge, lw=1.8)

ax8.text(
    6.0, 0.35,
    "Then: can one learned representation support thousands of tasks?\n"
    "Frontier: discover latent structure humans did not explicitly formulate.",
    ha="center", va="center", fontsize=FS_TICK + 1, color=C_DARK,
)
show_maximized(fig8)


# -------------------------------------------------------
# FIGURE 9: Unsolved / open
# -------------------------------------------------------
fig9, ax9 = new_figure()
fig9.suptitle(
    "Step 10: What deep learning has not solved (yet) -- keep the humility",
    fontsize=FS_FIGURE_TITLE - 4, fontweight="bold",
)
blank_slide(ax9)

open_probs = [
    (0.4, 4.5, "Reliable long-horizon reasoning", "Still brittle outside\ntraining patterns."),
    (4.25, 4.5, "Causal / scientific discovery", "Correlation-rich models\n≠ automatic science."),
    (8.1, 4.5, "Data-efficient learning", "Humans need far less\nthan today's giants."),
    (0.4, 0.7, "Robustness & safety", "Distribution shift,\nadversaries, misuse."),
    (4.25, 0.7, "Grounded world models", "Language fluency ≠\nstable understanding."),
    (8.1, 0.7, "Energy / compute limits", "Capability often still\nbought with scale."),
]
for x0, y0, title, body in open_probs:
    round_box(ax9, x0, y0, 3.5, 3.3, C_WARN, C_ACCENT, lw=1.8)
    ax9.text(x0 + 1.75, y0 + 2.4, title, ha="center",
             fontsize=FS_TICK + 1, fontweight="bold", color=C_ACCENT)
    ax9.text(x0 + 1.75, y0 + 1.15, body, ha="center", va="center",
             fontsize=FS_TICK + 1)
show_maximized(fig9)


# -------------------------------------------------------
# FIGURE 10: Takeaway for the course
# -------------------------------------------------------
fig10, axes10 = new_figure(1, 3)
fig10.suptitle(
    "Step 11: For this course -- history as a compass, not a museum",
    fontsize=FS_FIGURE_TITLE - 2, fontweight="bold",
)

panels = [
    (axes10[0], "Learn the shift",
     "Representation became\nlearnable.",
     C_NEW, "#2874a6"),
    (axes10[1], "Credit carefully",
     "The famous last paper is rarely\n"
     "the whole invention.\n\n"
     "Separate enabling ideas from later\n"
     "scale and systems.\n\n"
     "Credit the lineage and the team,\n"
     "not a slogan or a single name.",
     C_GOAL, C_OK),
    (axes10[2], "Stay unfinished",
     "Many hard problems\nremain open.\nThat is your job.",
     "#f5eef8", C_FUN),
]
for ax, title, body, face, edge in panels:
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    round_box(ax, 0.5, 1.5, 9.0, 7.0, face, edge, lw=2.2)
    ax.text(5.0, 7.0, title, ha="center", fontsize=FS_AX_TITLE - 2,
            fontweight="bold", color=edge)
    ax.text(5.0, 4.2, body, ha="center", va="center", fontsize=FS_ANNOT + 1)
show_maximized(fig10)


print("\n" + "=" * 60)
print("HISTORICAL ACHIEVEMENTS BY DEEP LEARNING")
print("=" * 60)
print("""
Claim:
  Deep learning made representation itself a learnable object.

Pattern:
  hand-coded rules
    -> hand-designed features
    -> learned features
    -> end-to-end representations
    -> pretrained / foundation models
    -> multi-task general systems

Corrections:
  AlexNet scaled CNNs; it did not invent them.
  AlphaGo guided search; it was not \"only a net\".
  AlphaFold was a team achievement.

Open:
  reasoning reliability, causality, data efficiency,
  robustness, grounded world models, compute limits.
""")
