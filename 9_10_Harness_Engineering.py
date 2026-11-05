"""
9_10_Harness_Engineering.py

Learn Harness Engineering: a walk through all 14 lectures, with runnable code.

Course read: https://walkinglabs.github.io/learn-harness-engineering/en/
Source repo: https://github.com/walkinglabs/learn-harness-engineering (MIT, WalkingLab)

Thesis of the course: a capable model is not a reliable agent. Everything outside
the model weights -- instructions, tools, environment, state, feedback -- is the
harness, and the harness decides how much of the model's capability is realized.

Slides (Step N = main idea, Step N.1 = details or code; at most two panels each):
  0. Preface                         11. L9  Agents declare victory too early
  1. Course map                      12. L10 End-to-end testing changes results
  2. L1  Capable agents still fail   13. L11 Observability inside the harness
  3. L2  What a harness is           14. L12 Leave a clean state
  4. L3  Repo = system of record     15. L13 Loop engineering
  5. L4  One giant instruction file  16. CODE: a maker-checker goal loop, live
  6. L5  Continuity across sessions  17. L14 Graph engineering
  7. L6  Initialization phase        18. CODE: the loop as a graph (18.1, 18.2 live)
  8. L7  Overreach / under-finish    19. Projects and starter templates
  9. L8  Feature lists as primitives 20. Takeaways
 10. CODE: harness-owned feature list, live

Part A of this file is a small, real harness (no third-party deps): a feature
list whose pass state only the harness can set, an independent checker that runs
verification commands in fresh processes, a maker-checker loop with stop
conditions, and a minimal graph engine with checkpoints. The code slides display
the actual source of these functions and the results of running them.

Usage:
  python 9_10_Harness_Engineering.py              # slide deck
  python 9_10_Harness_Engineering.py --demo       # run the harness in the terminal
  python 9_10_Harness_Engineering.py --scaffold DIR
      # write AGENTS.md, feature_list.json, claude-progress.md, init.sh into DIR
      # (existing files are never overwritten)
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


# =======================================================
# Part A: a minimal runnable harness
# =======================================================
class HarnessError(Exception):
    """Raised when the agent asks for something the harness does not allow."""


@dataclass
class Feature:
    """Lecture 8's triple: behavior + verification command + state."""
    id: str
    behavior: str
    verify: list[str]
    state: str = "not_started"   # not_started | active | blocked | passing
    evidence: str = ""


def run_check(argv, workdir):
    """Run one verification command in a fresh process; return (ok, last line)."""
    env = {**os.environ, "PYTHONPATH": str(workdir), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(argv, cwd=workdir, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=60)
    lines = (proc.stdout + proc.stderr).strip().splitlines()
    return proc.returncode == 0, (lines[-1] if lines else "ok")


class FeatureList:
    """Harness-owned feature list (Lecture 8). The agent proposes, the harness decides."""

    def __init__(self, features, workdir):
        self.features = {f.id: f for f in features}
        self.workdir = workdir

    def next_feature(self):
        """Scheduler: the first not_started feature, in list order."""
        return next((f.id for f in self.features.values() if f.state == "not_started"), None)

    def activate(self, fid):
        active = [f.id for f in self.features.values() if f.state == "active"]
        if active:
            raise HarnessError(f"WIP=1: {active[0]} is still active, finish it first")
        self.features[fid].state = "active"

    def request_pass(self, fid):
        """Verifier: the ONLY path from active to passing is a passing command."""
        f = self.features[fid]
        if f.state != "active":
            raise HarnessError(f"{fid} is {f.state}, only active features are verified")
        ok, line = run_check(f.verify, self.workdir)
        if ok:
            f.state, f.evidence = "passing", "verify command exited 0"
        return ok, line

    def mark_passing(self, fid):
        raise HarnessError("agents may not set state; call request_pass() instead")

    def back_pressure(self):
        """Progress tracker: features not yet passing (0 means the project is done)."""
        return sum(f.state != "passing" for f in self.features.values())

    def to_json(self):
        return json.dumps([f.__dict__ for f in self.features.values()], indent=2)


# --- A toy project the "agent" works on: slugify(text) -------------------------------
def slug_check(text, expected):
    code = (f"T = {text!r}; E = {expected!r}; from toy import slugify; r = slugify(T); "
            "assert r == E, 'slugify(%r) returned %r, expected %r' % (T, r, E)")
    return [sys.executable, "-c", code]


def toy_features():
    return [
        Feature("F01", "lowercase words joined by dashes", slug_check("Hello World", "hello-world")),
        Feature("F02", "punctuation is dropped", slug_check("Hi, there!", "hi-there")),
        Feature("F03", "runs of spaces/dashes collapse", slug_check("  a  --  b ", "a-b")),
        Feature("F04", "empty input gives empty slug", slug_check("", "")),
        Feature("F05", "accents are transliterated", slug_check("Café Déjà vu", "cafe-deja-vu")),
    ]


# Scripted maker: each attempt is what an agent writes after reading checker feedback.
MAKER_ATTEMPTS = [
    ("lowercase, spaces -> dashes",
     'def slugify(text):\n    return text.lower().replace(" ", "-")\n'),
    ("every non-alphanumeric run -> one dash",
     'import re\n\ndef slugify(text):\n    return re.sub(r"[^a-z0-9]+", "-", text.lower())\n'),
    ("strip leading/trailing dashes",
     'import re\n\ndef slugify(text):\n    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")\n'),
    ("transliterate accents first (NFKD)",
     'import re\nimport unicodedata\n\ndef slugify(text):\n'
     '    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()\n'
     '    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")\n'),
]


def checker(workdir):
    """Independent checker: fresh processes, sees files only, never maker notes."""
    return [(f.id, *run_check(f.verify, workdir)) for f in toy_features()]


def stop_reason(passed, total, stale, patience, rnd, max_rounds):
    """The loop's stop conditions; None means keep going."""
    if passed == total:
        return "goal met, every check passes"
    if stale >= patience:
        return f"no progress for {patience} rounds, call a human"
    if rnd == max_rounds:
        return "max rounds reached"
    return None


def maker_checker_loop(workdir, max_rounds=6, patience=3):
    """Goal loop (Lecture 13): goal + independent verification + stop conditions."""
    rounds, best, stale = [], -1, 0
    for rnd, (note, source) in enumerate(MAKER_ATTEMPTS[:max_rounds], start=1):
        (Path(workdir) / "toy.py").write_text(source, encoding="utf-8")  # maker
        results = checker(workdir)                                       # checker
        passed = sum(ok for _, ok, _ in results)
        stale = 0 if passed > best else stale + 1
        best = max(best, passed)
        reason = stop_reason(passed, len(results), stale, patience, rnd, max_rounds)
        rounds.append({"round": rnd, "note": note, "passed": passed, "results": results,
                       "decision": f"STOP: {reason}" if reason else "continue"})
        if reason:
            break
    return rounds


class MiniGraph:
    """The skeleton every graph engine shares (Lecture 14): nodes, edges, state, routing."""

    def __init__(self, reducers=None):
        self.nodes, self.edges, self.routes = {}, {}, {}
        self.reducers = reducers or {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, src, dst):
        self.edges[src] = dst

    def add_conditional_edges(self, src, router):
        self.routes[src] = router

    def run(self, state, thread, pause_before=(), approve=lambda s: True, max_steps=25):
        node, trace, checkpoints = self.edges["START"], [], []
        while node != "END" and len(trace) < max_steps:
            if node in pause_before and not approve(state):
                break                          # human said no; last checkpoint stays
            for key, value in self.nodes[node](state).items():
                merge = self.reducers.get(key)
                state[key] = merge(state.get(key), value) if merge else value
            trace.append(node)
            checkpoints.append({"thread": thread, "step": len(trace),
                                "node": node, "state": dict(state)})
            if node in self.routes:
                node = self.routes[node](state)
            else:
                node = self.edges[node]
        return state, trace, checkpoints


def build_maker_checker_graph(workdir, max_attempts=6):
    g = MiniGraph(reducers={"attempts": lambda old, new: (old or 0) + new})

    def research(state):
        return {"requirements": [f.behavior for f in toy_features()]}

    def implement(state):
        note, source = MAKER_ATTEMPTS[min(state.get("attempts", 0), len(MAKER_ATTEMPTS) - 1)]
        (Path(workdir) / "toy.py").write_text(source, encoding="utf-8")
        return {"code": note}

    def verify(state):                  # fresh context: reads files, not the maker's reasoning
        passed = sum(ok for _, ok, _ in checker(workdir))
        review = "pass" if passed == len(toy_features()) else "fail"
        return {"review": review, "passed": passed, "attempts": int(review == "fail")}

    def merge(state):
        return {"merged": True}

    def route_after_verify(state):
        if state["review"] == "pass":
            return "merge"
        return "implement" if state["attempts"] < max_attempts else "END"

    for name, fn in [("research", research), ("implement", implement),
                     ("verify", verify), ("merge", merge)]:
        g.add_node(name, fn)
    g.add_edge("START", "research")
    g.add_edge("research", "implement")
    g.add_edge("implement", "verify")
    g.add_conditional_edges("verify", route_after_verify)
    g.add_edge("merge", "END")
    return g


# --- Cached demo runs (shared by the slides and --demo) --------------------------------
@lru_cache(maxsize=None)
def demo_feature_gate():
    """Replay an agent session against the harness-owned feature list."""
    log = []
    with tempfile.TemporaryDirectory() as wd:
        (Path(wd) / "toy.py").write_text(MAKER_ATTEMPTS[0][1], encoding="utf-8")
        fl = FeatureList(toy_features(), wd)

        def attempt(actor, action, call):
            try:
                out = call()
                log.append((actor, action, "OK" if out is None else out))
            except HarnessError as exc:
                log.append((actor, action, f"REFUSED: {exc}"))

        attempt("scheduler", f"next_feature() -> {fl.next_feature()}", lambda: None)
        attempt("agent", "activate('F01')", lambda: fl.activate("F01"))
        attempt("agent", "activate('F02')  # start another", lambda: fl.activate("F02"))
        attempt("agent", "request_pass('F01')",
                lambda: "PASS -> passing" if fl.request_pass("F01")[0] else "FAIL")
        attempt("agent", "activate('F02')", lambda: fl.activate("F02"))
        attempt("agent", "mark_passing('F02')  # 'looks fine'", lambda: fl.mark_passing("F02"))

        def verify_f02():
            ok, line = fl.request_pass("F02")
            return "PASS" if ok else "FAIL, stays active: " + line.replace("AssertionError: ", "")
        attempt("agent", "request_pass('F02')", verify_f02)
        attempt("tracker", "back_pressure()", lambda: f"{fl.back_pressure()} of 5 not passing")
        return tuple(log), fl.to_json()


@lru_cache(maxsize=None)
def demo_loop():
    with tempfile.TemporaryDirectory() as wd:
        return tuple(maker_checker_loop(wd))


@lru_cache(maxsize=None)
def demo_graph():
    with tempfile.TemporaryDirectory() as wd:
        g = build_maker_checker_graph(wd)
        approvals = []

        def human_gate(state):
            approvals.append(dict(state))
            return True                             # auto-approve in the demo
        state, trace, cps = g.run({"attempts": 0}, thread="session-1",
                                  pause_before=("merge",), approve=human_gate)
        return dict(state), tuple(trace), len(cps), len(approvals)


# --- Ports of two course tools (Lectures 7 and 9), same sample data ------------------
SCOPE_LOG = [  # (step, file, feature) from scope-tracker.ts; F-001 is the active feature
    (1, "routes/search.ts", "F-001"), (2, "routes/search.ts", "F-001"),
    (3, "routes/search.ts", "F-001"), (4, "routes/delete.ts", "F-002"),
    (5, "middleware/rate-limit.ts", "F-003"), (6, "routes/search.ts", "F-001"),
    (7, "dashboard/ui.tsx", "F-004"), (8, "routes/search.ts", "F-001"),
    (9, "routes/delete.ts", "F-002"), (10, "routes/search.ts", "F-001"),
]

VICTORY_TASKS = {  # claimed: every check true; actual results from victory-detector.ts
    "Add search endpoint": [True, False, True, False, False, False],
    "Fix pagination bug": [True, True, True, False, False],
    "Refactor auth module": [False, False, True, False],
}


# --- Starter templates (condensed from the course's resources/templates) ---------------
TEMPLATES = {
    "AGENTS.md": """\
        # AGENTS.md

        Goal: leave the repo in a state where the next session can continue without guessing.

        ## Startup Workflow
        1. Read `claude-progress.md` for the latest verified state and next step.
        2. Read `feature_list.json`; pick the highest-priority unfinished feature.
        3. Review recent commits: `git log --oneline -5`.
        4. Run `./init.sh`. If baseline verification fails, fix that first.

        ## Working Rules
        - Work on one feature at a time (WIP=1).
        - Do not mark a feature complete just because code was added.
        - Do not silently change verification rules during implementation.
        - Prefer durable repo artifacts over chat summaries.

        ## Definition of Done
        - Target behavior implemented, required verification actually ran,
          evidence recorded, repo restartable from `./init.sh`.

        ## End of Session
        1. Update `claude-progress.md` and `feature_list.json`.
        2. Record unresolved risks or blockers.
        3. Commit once the work is in a safe state.
        """,
    "feature_list.json": json.dumps([
        {"id": "F01", "priority": 1, "area": "core",
         "behavior": "Describe one user-visible behavior",
         "verification": ["command or step that proves it"],
         "status": "not_started", "evidence": ""},
    ], indent=2) + "\n",
    "claude-progress.md": """\
        # Progress Log

        ## Current Verified State
        - Standard startup path: ./init.sh
        - Standard verification path:
        - Highest-priority unfinished feature:
        - Current blocker:

        ## Session Log
        ### Session 001
        - Goal:
        - Completed:
        - Verification run / evidence:
        - Commits:
        - Known risk:
        - Next best step:
        """,
    "init.sh": """\
        #!/usr/bin/env bash
        # Standard startup + verification path.
        # Edit the three commands for your project.
        set -euo pipefail
        INSTALL_CMD="pip install -r requirements.txt"
        VERIFY_CMD="pytest -x"
        START_CMD=""

        echo "[init] install";  eval "$INSTALL_CMD"
        echo "[init] verify";   eval "$VERIFY_CMD"
        if [ -n "$START_CMD" ]; then
          echo "[init] start";  eval "$START_CMD"
        fi
        """,
}


def scaffold(target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name, body in TEMPLATES.items():
        path = target / name
        if path.exists():
            print(f"skip   {path} (already exists)")
            continue
        path.write_text(textwrap.dedent(body), encoding="utf-8", newline="\n")
        print(f"wrote  {path}")


def print_demo():
    print("== Feature list: the harness owns state transitions (Lecture 8) ==")
    log, _ = demo_feature_gate()
    for actor, action, result in log:
        print(f"  {actor:<9} {action:<36} {result}")
    print("\n== Maker-checker goal loop (Lecture 13) ==")
    for r in demo_loop():
        print(f"  round {r['round']}: {r['note']:<40} {r['passed']}/5  {r['decision']}")
    print("\n== The same loop as a graph (Lecture 14) ==")
    state, trace, n_cp, n_gate = demo_graph()
    print("  path: " + " -> ".join(trace))
    print(f"  checkpoints saved: {n_cp}, human approval gates: {n_gate}, final: {state}")


# =======================================================
# Part B: slides
# =======================================================
# Readability rules for this deck (lecture screen is ~10.8 x 6.1 inches):
# at most two panels per slide, body text >= 12 pt, notes >= 11 pt, code >= 10 pt.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Circle  # noqa: E402

from _config import (  # noqa: E402
    FS_FIGURE_TITLE,
    FS_TICK,
    FS_ANNOT,
    FIGURE_DPI,
    SCREEN_FRACTION,
    apply_rcparams,
    run_slide_deck,
)

apply_rcparams(plt)

FS_TITLE = FS_FIGURE_TITLE - 2   # slide title
FS_HEAD = FS_ANNOT               # panel / card heading
FS_BODY = FS_TICK - 2            # body text
FS_NOTE = FS_TICK - 3            # notes and chart annotations
FS_CODE_MAX, FS_CODE_MIN = FS_TICK - 1, FS_TICK - 4

C_DARK = "#1a5276"
C_NEW = "#d6eaf8"
C_OK = "#1e8449"
C_ACCENT = "#c0392b"
C_MUTE = "#f4f6f7"
C_GOAL = "#d4edda"
C_OLD = "#fce4d6"
C_FUN = "#6c3483"
C_WARN = "#f9e79f"
C_GRAY = "#b0b0b0"
C_CODE = "#1e1e1e"
C_ORANGE = "#e67e22"
C_TEAL = "#117a65"

PART_COLORS = {
    "Why agents fail": C_ACCENT,
    "Repo & instructions": C_DARK,
    "Continuity": C_FUN,
    "Scope & verification": C_OK,
    "Operations": C_ORANGE,
    "Beyond the harness": C_TEAL,
}


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


def slide(title, ncols=1, nrows=1, tag=None, **gs_kw):
    """Figure with a title and a gridspec for <= 2 panels; the title colour marks the part."""
    fig = plt.figure(figsize=screen_figsize(), dpi=FIGURE_DPI)
    fig.suptitle(title, fontsize=FS_TITLE, fontweight="bold", y=0.965,
                 color=PART_COLORS[tag[1]] if tag else "black")
    gs_kw.setdefault("top", 0.85)
    gs_kw.setdefault("bottom", 0.19)
    gs_kw.setdefault("left", 0.08)
    gs_kw.setdefault("right", 0.97)
    gs_kw.setdefault("wspace", 0.25)
    return fig, fig.add_gridspec(nrows, ncols, **gs_kw)


def blank(ax, xlim=(0, 12), ylim=(0, 10)):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.tick_params(labelsize=FS_BODY)
    if title:
        ax.set_title(title, fontsize=FS_BODY + 1, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_BODY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_BODY)


def round_box(ax, x, y, w, h, facecolor, edgecolor="0.2", lw=2.0):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(rect)
    return rect


def arrow(ax, x1, y1, x2, y2, color="0.25", lw=2.5, style="->", rad=0.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=18,
                        connectionstyle=f"arc3,rad={rad}"),
    )


def card(ax, x, y, w, h, title, body, face, edge, title_color=None):
    round_box(ax, x, y, w, h, face, edge, lw=1.8)
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top",
            fontsize=FS_HEAD, fontweight="bold", color=title_color or edge)
    ax.text(x + w / 2, y + (h - 0.8) / 2, body, ha="center", va="center",
            fontsize=FS_BODY, color="0.15", linespacing=1.4, parse_math=False)


def text_block(ax, x, y, text, fs=None, **kw):
    kw.setdefault("ha", "left")
    kw.setdefault("va", "top")
    kw.setdefault("linespacing", 1.45)
    ax.text(x, y, text, fontsize=fs or FS_BODY, parse_math=False, **kw)


def row_list(ax, items, top, step, x=0.2, w=11.6, h=None, face=C_MUTE, edge="0.6",
             title_color="0.1"):
    """Stacked boxes, each with a bold first line and an optional second line."""
    h = h or step * 0.82
    for i, (head, sub) in enumerate(items):
        y = top - i * step
        round_box(ax, x, y - h, w, h, face, edge, lw=1.2)
        if sub:
            ax.text(x + 0.3, y - h * 0.32, head, ha="left", va="center", fontsize=FS_BODY,
                    fontweight="bold", color=title_color, parse_math=False)
            ax.text(x + 0.3, y - h * 0.72, sub, ha="left", va="center", fontsize=FS_NOTE,
                    color="0.25", parse_math=False)
        else:
            ax.text(x + 0.3, y - h / 2, head, ha="left", va="center", fontsize=FS_BODY,
                    fontweight="bold", color=title_color, parse_math=False)


def code_panel(ax, code, title=""):
    """Dark editor-style panel; the font shrinks to fit, never below FS_CODE_MIN."""
    lines = code.rstrip("\n").splitlines() or [""]
    box = ax.get_position()
    fig = ax.figure
    w_pt = 0.93 * box.width * fig.get_figwidth() * 72
    h_pt = 0.9 * box.height * fig.get_figheight() * 72
    fit_w = w_pt / (0.61 * max(len(s) for s in lines))
    fit_h = h_pt / (1.3 * 1.2 * (len(lines) + (2 if title else 0)))
    fs = max(FS_CODE_MIN, min(FS_CODE_MAX, fit_w, fit_h))
    blank(ax, xlim=(0, 1), ylim=(0, 1))
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.02",
                                facecolor=C_CODE, edgecolor="#111111", lw=1.2,
                                transform=ax.transAxes, clip_on=False))
    top = 0.97
    if title:
        ax.text(0.025, top, title, ha="left", va="top", fontsize=FS_BODY, fontweight="bold",
                color="#9cdcfe", transform=ax.transAxes, parse_math=False)
        top -= FS_BODY * 2.0 / (box.height * fig.get_figheight() * 72)
    ax.text(0.025, top, "\n".join(lines), ha="left", va="top", family="monospace",
            fontsize=fs, color="#d4d4d4", linespacing=1.3, transform=ax.transAxes,
            parse_math=False)


def footnote(fig, text, y=0.078):
    fig.text(0.525, y, text, ha="center", va="center", fontsize=FS_NOTE, color="0.35",
             style="italic", parse_math=False)


def source_of(fn):
    return textwrap.dedent(inspect.getsource(fn))


# -------------------------------------------------------
# Preface and map
# -------------------------------------------------------
def build_preface():
    fig, gs = slide("")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    ax.text(6.0, 9.4, "Course reading", ha="center", va="center", fontsize=FS_HEAD + 2,
            fontweight="bold", color="0.45")
    round_box(ax, 0.6, 5.9, 10.8, 2.8, C_GOAL, C_OK, lw=2.2)
    ax.text(6.0, 7.65, "Learn Harness Engineering", ha="center", va="center",
            fontsize=FS_FIGURE_TITLE + 10, fontweight="bold", color="#145a32")
    ax.text(6.0, 6.55, "Making AI coding agents reliable", ha="center", va="center",
            fontsize=FS_HEAD + 4, color="0.25")
    ax.text(6.0, 4.9, "WalkingLabs  ·  14 lectures  ·  8 projects  ·  templates",
            ha="center", va="center", fontsize=FS_HEAD + 4, fontweight="bold")
    ax.text(6.0, 3.7, "References: OpenAI, Harness Engineering;\n"
            "Anthropic, Effective Harnesses for Long-Running Agents",
            ha="center", va="center", fontsize=FS_HEAD, color="0.3", linespacing=1.5)
    ax.text(6.0, 2.1, "walkinglabs.github.io/learn-harness-engineering/en/", ha="center",
            va="center", fontsize=FS_HEAD + 2, fontweight="bold", color=C_DARK)
    ax.text(6.0, 1.2, "Source repo: MIT License  ·  read 2026-11-04", ha="center",
            va="center", fontsize=FS_BODY, color="0.45")
    return fig


LECTURES = [
    ("Why agents fail", "L1", "Capable agents still fail"),
    ("Why agents fail", "L2", "What a harness actually is"),
    ("Repo & instructions", "L3", "Repo = system of record"),
    ("Repo & instructions", "L4", "One giant instruction file fails"),
    ("Continuity", "L5", "Long tasks lose continuity"),
    ("Continuity", "L6", "Initialization is its own phase"),
    ("Scope & verification", "L7", "Overreach and under-finish"),
    ("Scope & verification", "L8", "Feature lists are primitives"),
    ("Scope & verification", "L9", "Victory declared too early"),
    ("Scope & verification", "L10", "End-to-end testing"),
    ("Operations", "L11", "Observability in the harness"),
    ("Operations", "L12", "Leave a clean state"),
    ("Beyond the harness", "L13", "Loop engineering"),
    ("Beyond the harness", "L14", "Graph engineering"),
]


def build_step_1():
    fig, gs = slide("Step 1: The course in one picture", ncols=2, width_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, ylim=(0, 14.5))
    for i, (part, tag, title) in enumerate(LECTURES):
        y = 14.0 - i * 1.0
        col = PART_COLORS[part]
        round_box(ax, 0.1, y - 0.4, 1.7, 0.8, col, col, lw=0)
        ax.text(0.95, y, tag, ha="center", va="center", fontsize=FS_BODY, fontweight="bold",
                color="white")
        ax.text(2.1, y, title, ha="left", va="center", fontsize=FS_BODY)

    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    layers = [
        (0.2, 1.5, "Model weights", "the capability you rent", C_GRAY, "0.35"),
        (2.0, 2.6, "Harness  (L1-L12)", "instructions, tools, environment,\nstate, feedback: ONE reliable run",
         C_GOAL, C_OK),
        (5.0, 2.0, "Loop engineering  (L13)", "goal + verification + stop:\nruns become continuous",
         C_NEW, C_DARK),
        (7.4, 2.0, "Graph engineering  (L14)", "nodes, edges, state, routing:\nmany loops cooperate",
         "#d1f2eb", C_TEAL),
    ]
    for y, h, name, desc, face, edge in layers:
        round_box(ax, 0.2, y, 11.6, h, face, edge, lw=2)
        ax.text(6.0, y + h - 0.3, name, ha="center", va="top", fontsize=FS_HEAD,
                fontweight="bold", color=edge)
        if h > 1.6:
            ax.text(6.0, y + (h - 0.75) / 2, desc, ha="center", va="center", fontsize=FS_BODY,
                    linespacing=1.35)
        else:
            ax.text(6.0, y + 0.35, desc, ha="center", va="center", fontsize=FS_BODY)
    return fig


# -------------------------------------------------------
# Lecture 1
# -------------------------------------------------------
L1 = ("Lecture 1", "Why agents fail")


def build_step_2():
    fig, gs = slide("Step 2: L1 A strong model is not a reliable agent", ncols=2, tag=L1,
                    width_ratios=[1, 1.1], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(2)
    ax.bar(x - 0.2, [20, 360], 0.38, color=C_DARK, label="minutes")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [9, 200], 0.38, color=C_ORANGE, label="USD")
    for xi, m, c in zip(x, [20, 360], [9, 200]):
        ax.text(xi - 0.2, m + 10, f"{m} min", ha="center", fontsize=FS_NOTE, fontweight="bold")
        ax2.text(xi + 0.2, c + 6, f"{c} USD", ha="center", fontsize=FS_NOTE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["bare run", "full harness"])
    ax.set_ylim(0, 430)
    ax2.set_ylim(0, 240)
    ax2.tick_params(labelsize=FS_BODY)
    style(ax, "Same model (Opus 4.5), same prompt", ylabel="minutes")
    ax2.set_ylabel("USD", fontsize=FS_BODY)

    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    card(ax, 0.2, 5.6, 11.6, 4.2, "Bare run: 20 min",
         "the app opened, but game\nentities ignored all input", C_OLD, C_ACCENT)
    card(ax, 0.2, 1.0, 11.6, 4.2, "Planner + generator + evaluator",
         "6 hours: core gameplay works\n(physics still rough). Only\nthe harness changed.",
         C_GOAL, C_OK, title_color="#145a32")
    footnote(fig, "Anthropic's '2D retro game editor' experiment, as cited in Lecture 1")
    return fig


def build_step_2_1():
    fig, gs = slide("Step 2.1: L1 When it fails, find the harness layer", ncols=2, tag=L1,
                    width_ratios=[1.25, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    rows = [("Vague requirements", "1. Task spec"),
            ("Unwritten conventions", "2. Context"),
            ("Broken setup", "3. Environment"),
            ("No tests: 'seems fine'", "4. Verification"),
            ("Lost between sessions", "5. State")]
    ax.text(6.0, 9.7, "Where agents get stuck  ->  layer to fix", ha="center", va="top",
            fontsize=FS_HEAD, fontweight="bold")
    for i, (fail, layer) in enumerate(rows):
        y = 8.2 - i * 1.6
        round_box(ax, 0.1, y - 0.55, 6.3, 1.1, C_OLD, C_ACCENT, lw=1.2)
        ax.text(3.25, y, fail, ha="center", va="center", fontsize=FS_BODY)
        arrow(ax, 6.5, y, 7.3, y)
        round_box(ax, 7.4, y - 0.55, 4.5, 1.1, C_GOAL, C_OK, lw=1.2)
        ax.text(9.65, y, layer, ha="center", va="center", fontsize=FS_BODY, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    code_panel(ax, "Completion criteria:\n"
                   "- GET /api/search?q=xxx\n"
                   "- pagination, default 20\n"
                   "- highlighted snippets\n"
                   "- all new code passes pytest\n"
                   "- mypy --strict passes\n\n"
                   "pytest tests/api/v2/ &&\n"
                   "  python -m mypy src/",
               title="Definition of Done")
    footnote(fig, "Diagnostic loop: execute -> observe failure -> attribute to a layer -> fix it -> re-execute")
    return fig


# -------------------------------------------------------
# Lecture 2
# -------------------------------------------------------
L2 = ("Lecture 2", "Why agents fail")


def build_step_3():
    fig, gs = slide("Step 3: L2 A harness is everything except the weights", ncols=2, tag=L2,
                    width_ratios=[1.15, 1], wspace=0.38)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    ax.add_patch(Circle((6.0, 5.0), 1.2, facecolor=C_GRAY, edgecolor="0.3", lw=2))
    ax.text(6.0, 5.0, "model", ha="center", va="center", fontsize=FS_HEAD, fontweight="bold")
    subs = [("Instructions", "AGENTS.md", C_DARK), ("Tools", "shell, tests", C_FUN),
            ("Environment", "pinned, locked", C_ORANGE), ("State", "PROGRESS.md", C_TEAL),
            ("Feedback", "verify commands", C_OK)]
    for i, (name, desc, col) in enumerate(subs):
        ang = np.pi / 2 + i * 2 * np.pi / 5
        cx, cy = 6.0 + 4.2 * np.cos(ang), 5.0 + 3.6 * np.sin(ang)
        round_box(ax, cx - 1.75, cy - 0.8, 3.5, 1.6, "white", col, lw=2)
        ax.text(cx, cy + 0.3, name, ha="center", va="center", fontsize=FS_BODY,
                fontweight="bold", color=col)
        ax.text(cx, cy - 0.35, desc, ha="center", va="center", fontsize=FS_NOTE)

    ax = fig.add_subplot(gs[0, 1])
    rate = [20, 60, 80, 90]
    bars = ax.bar(range(4), rate, color=[C_GRAY, C_DARK, C_OK, C_TEAL])
    ax.errorbar([3], [90], yerr=[[10], [10]], color="k", capsize=6, lw=1.5)
    for b, lbl in zip(bars, ["20%", "60%", "80%", "80-100%"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 11, lbl, ha="center",
                fontsize=FS_NOTE, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["README", "+AGENTS", "+verify", "+progress"])
    ax.set_ylim(0, 118)
    style(ax, "GPT-4o, 20k-line app: success %")
    footnote(fig, "Same model throughout; one harness piece added per stage (Lecture 2 case study)")
    return fig


def build_step_3_1():
    fig, gs = slide("Step 3.1: L2 Feedback first, then measure each piece", ncols=2, tag=L2,
                    width_ratios=[1, 1], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, "Verification commands:\n"
                   "- Tests: pytest tests/ -x\n"
                   "- Types: mypy src/ --strict\n"
                   "- Lint:  ruff check src/\n"
                   "- All:   make check",
               title="In AGENTS.md (highest ROI)")
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    row_list(ax, [("Map, not manual", "~100-line AGENTS.md, details in docs/"),
                  ("Constrain, don't micromanage", "enforce invariants with checks"),
                  ("Separate worker and checker", "agents overrate their own work"),
                  ("Ablate one piece at a time", "biggest drop = biggest value now"),
                  ("Harnesses rot like code", "audit and pay down harness debt")],
             top=9.8, step=1.95)
    return fig


# -------------------------------------------------------
# Lecture 3
# -------------------------------------------------------
L3 = ("Lecture 3", "Repo & instructions")


def build_step_4():
    fig, gs = slide("Step 4: L3 Not in the repo = does not exist", ncols=2, tag=L3,
                    width_ratios=[1, 1], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, "project/\n"
                   "├── AGENTS.md    # entry point\n"
                   "├── src/\n"
                   "│   ├── api/\n"
                   "│   │   └── ARCHITECTURE.md\n"
                   "│   └── db/\n"
                   "│       └── CONSTRAINTS.md\n"
                   "├── PROGRESS.md  # done/doing/blocked\n"
                   "└── Makefile     # setup test check",
               title="Knowledge lives next to code")
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    ax.text(6.0, 9.8, "Fresh-session test: repo only", ha="center", va="top",
            fontsize=FS_HEAD, fontweight="bold", color=C_DARK)
    qs = ["What is this system?", "How is it organized?", "How do I run it?",
          "How do I verify it?", "What is the progress?"]
    row_list(ax, [(f"{i + 1}. {q}", None) for i, q in enumerate(qs)], top=8.8, step=1.45,
             face=C_NEW, edge=C_DARK)
    footnote(fig, "Every unanswered question becomes a guess, every guess a bug, repeated in every new session")
    return fig


def build_step_4_1():
    fig, gs = slide("Step 4.1: L3 Manage agent state like a database", ncols=2, tag=L3,
                    width_ratios=[1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    row_list(ax, [("Atomicity", "commit a change only when verified"),
                  ("Consistency", "tests pass + lint clean after each op"),
                  ("Isolation", "own branch or worktree per agent"),
                  ("Durability", "cross-session facts in git files")],
             top=9.8, step=2.45, face=C_GOAL, edge=C_OK, title_color="#145a32")
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    row_list(ax, [("Visibility gap", "knowledge NOT in repo; aim < 10%"),
                  ("Discovery cost", "context spent finding one fact"),
                  ("Knowledge decay", "stale docs mislead with confidence"),
                  ("Placement > volume", "50 lines in src/api/ beat a wiki")],
             top=9.8, step=2.45, face=C_NEW, edge=C_DARK)
    return fig


# -------------------------------------------------------
# Lecture 4
# -------------------------------------------------------
L4 = ("Lecture 4", "Repo & instructions")


def build_step_5():
    fig, gs = slide("Step 5: L4 One giant instruction file fails", ncols=2, tag=L4,
                    width_ratios=[1, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    window = 128
    for i, (used, col) in enumerate([(15, C_ACCENT), (2, C_OK)]):
        ax.barh(i, used, color=col)
        ax.barh(i, window - used, left=used, color=C_MUTE, edgecolor="0.7")
        ax.text(used + 3, i, f"~{used}K tokens ({100 * used / window:.0f}%)", va="center",
                fontsize=FS_BODY, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["600 lines", "100 lines"])
    ax.set_xlim(0, window)
    ax.invert_yaxis()
    style(ax, "Spent before any work starts", xlabel="128K-token window")

    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    row_list(ax, [("Lost in the middle", "line 300 of 600 gets ignored"),
                  ("Priority conflicts", "hard rules look like preferences"),
                  ("Maintenance decay", "adding feels free, deleting risky"),
                  ("Contradictions", "rules from different months clash")],
             top=9.8, step=2.45, face=C_OLD, edge=C_ACCENT, title_color=C_ACCENT)
    footnote(fig, "The vicious cycle: mistake -> add a rule -> works for now -> the file keeps growing")
    return fig


def build_step_5_1():
    fig, gs = slide("Step 5.1: L4 The entry file is a router", ncols=2, tag=L4,
                    width_ratios=[1.2, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, "# AGENTS.md\n"
                   "## Project Overview\n"
                   "Python 3.11 FastAPI, PostgreSQL 15.\n"
                   "## Quick Start\n"
                   "- Test: make test\n"
                   "- Full verification: make check\n"
                   "## Hard Constraints  (<= 15)\n"
                   "- OAuth 2.0 on every API\n"
                   "- SQLAlchemy 2.0 syntax only\n"
                   "## Topic Docs  (read on demand)\n"
                   "- docs/api-patterns.md\n"
                   "- docs/database-rules.md",
               title="Entry file: 50-200 lines")
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    row_list(ax, [("Each rule has metadata", "source, when needed, expiry"),
                  ("Critical rules at the edges", "top or bottom, not the middle"),
                  ("Lulla et al. (124 tasks)", "runtime -29%, tokens -17%"),
                  ("ETH Zurich study", "often no gain, +20% cost: test!")],
             top=9.8, step=2.45)
    return fig


# -------------------------------------------------------
# Lecture 5
# -------------------------------------------------------
L5 = ("Lecture 5", "Continuity")


def build_step_6():
    fig, gs = slide("Step 6: L5 Context windows end; state must not", ncols=2, tag=L5,
                    width_ratios=[1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    card(ax, 0.2, 5.3, 11.6, 4.5, "Compaction (same session)",
         "summarize in place: keeps the\nWHAT, often loses the WHY", C_NEW, C_DARK)
    card(ax, 0.2, 0.3, 11.6, 4.5, "Reset (new session)",
         "rebuild from files: clean, but\nonly as good as the files", C_GOAL, C_OK,
         title_color="#145a32")
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    card(ax, 0.2, 5.3, 11.6, 4.5, "Clock in",
         "read PROGRESS.md, DECISIONS.md\nrun make check\ncontinue from 'Next Steps'",
         C_WARN, "#b7950b", title_color="#7d6608")
    card(ax, 0.2, 0.3, 11.6, 4.5, "Clock out",
         "update PROGRESS.md\nrun make check\ncommit completed work",
         C_WARN, "#b7950b", title_color="#7d6608")
    footnote(fig, "Context anxiety: near the limit, agents rush and skip verification")
    return fig


def build_step_6_1():
    fig, gs = slide("Step 6.1: L5 What a handoff file holds", ncols=2, tag=L5,
                    width_ratios=[1.1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, "# PROGRESS.md\n"
                   "## Current State\n"
                   "- commit a1b2c3d, tests 41/42\n"
                   "## In Progress\n"
                   "- Pagination (90%): edge case\n"
                   "## Known Issues\n"
                   "- page_size=0 returns 500\n"
                   "## Next Steps\n"
                   "1. Validate page_size=0\n"
                   "2. make check, commit",
               title="Illustrative handoff")
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    row_list(ax, [("Rebuild cost", "time to an executable state"),
                  ("< ~30 min task", "one session is enough"),
                  ("> ~60% of the window", "start writing the handoff"),
                  ("Fit the model", "Sonnet 4.5 needed resets")],
             top=9.8, step=2.45)
    return fig


# -------------------------------------------------------
# Lecture 6
# -------------------------------------------------------
L6 = ("Lecture 6", "Continuity")


def build_step_7():
    fig, gs = slide("Step 7: L6 Session 1 builds infrastructure", ncols=2, tag=L6,
                    width_ratios=[1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    card(ax, 0.2, 5.3, 11.6, 4.5, "Initialization",
         "goal: make every LATER\nsession reliable", C_NEW, C_DARK)
    card(ax, 0.2, 0.3, 11.6, 4.5, "Implementation",
         "goal: verified features now.\nMixed in, it wins, and\ninfrastructure is skipped",
         C_OLD, C_ACCENT)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    ax.text(6.0, 9.8, "Ready when all four hold", ha="center", va="top", fontsize=FS_HEAD,
            fontweight="bold")
    row_list(ax, [("✓ can start", None), ("✓ can test", None), ("✓ can see progress", None),
                  ("✓ can pick up the next step", None)],
             top=8.6, step=2.0, face=C_GOAL, edge=C_OK, title_color="#145a32")
    footnote(fig, "Metric: time from start to the first passing test")
    return fig


def build_step_7_1():
    fig, gs = slide("Step 7.1: L6 init.sh and the acceptance checklist", ncols=2, tag=L6,
                    width_ratios=[1.1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, textwrap.dedent(TEMPLATES["init.sh"]), title="init.sh (from --scaffold)")
    ax = fig.add_subplot(gs[0, 1])
    code_panel(ax, "- [ ] make setup works from scratch\n"
                   "- [ ] make test: >= 1 passing\n"
                   "- [ ] new session can answer\n"
                   "      'run?' and 'test?' alone\n"
                   "- [ ] task list, >= 3 tasks\n"
                   "- [ ] everything committed",
               title="Initialization done?")
    return fig


# -------------------------------------------------------
# Lecture 7
# -------------------------------------------------------
L7 = ("Lecture 7", "Scope & verification")


def build_step_8():
    fig, gs = slide("Step 8: L7 Start fewer tasks, finish more", ncols=2, tag=L7,
                    width_ratios=[1, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0, 0])
    C, need = 100, 30
    ks = np.arange(1, 7)
    finished = np.where(C / ks >= need, ks, 0)
    ax.bar(ks, finished, color=[C_OK if f else C_ACCENT for f in finished])
    ax.set_ylim(0, 4)
    style(ax, "Each task gets ~C/k attention", xlabel="tasks started at once (k)",
          ylabel="tasks finished")
    ax.text(5.0, 1.4, "C/k below what\none task needs:\nnone finish", ha="center",
            fontsize=FS_NOTE, color=C_ACCENT, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    ax.bar([0, 1], [37.5, 87.5], color=[C_ACCENT, C_OK], width=0.55)
    for x, v in [(0, 37.5), (1, 87.5)]:
        ax.text(x, v + 3, f"{v}%", ha="center", fontsize=FS_BODY, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["5 at once", "WIP = 1"])
    ax.set_ylim(0, 105)
    style(ax, "8-feature API (illustrative)", ylabel="features completed (%)")
    footnote(fig, "WIP = 1: one active feature; the next starts only after end-to-end verification passes")
    return fig


def build_step_8_1():
    fig, gs = slide("Step 8.1: L7 Catch scope drift automatically", ncols=1, nrows=2, tag=L7,
                    height_ratios=[1, 1.15], hspace=0.6)
    ax = fig.add_subplot(gs[0, 0])
    for step, fname, fid in SCOPE_LOG:
        ok = fid == "F-001"
        ax.add_patch(plt.Rectangle((step - 0.45, 0), 0.9, 1, color=C_OK if ok else C_ACCENT))
        ax.text(step, 0.5, fid.replace("F-00", "F"), ha="center", va="center", color="white",
                fontsize=FS_BODY, fontweight="bold")
    drift = sum(fid != "F-001" for _, _, fid in SCOPE_LOG)
    ax.set_xlim(0.4, 10.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(range(1, 11))
    style(ax, f"Port of scope-tracker.ts: {drift} of {len(SCOPE_LOG)} changes drift "
              "away from active F1", xlabel="change step")
    ax = fig.add_subplot(gs[1, 0])
    code_panel(ax, "## Work Rules\n"
                   "- Work on one feature at a time\n"
                   "- Start the next only after this one passes end-to-end\n"
                   "- Don't 'also refactor' feature B while doing feature A\n"
                   "# VCR = verified / activated: block new work while < 1.0",
               title="Paste into AGENTS.md / CLAUDE.md")
    return fig


# -------------------------------------------------------
# Lecture 8
# -------------------------------------------------------
L8 = ("Lecture 8", "Scope & verification")


def build_step_9():
    fig, gs = slide("Step 9: L8 Feature lists are harness primitives", tag=L8)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(0, 24))
    ax.text(12.0, 9.7, "The agent proposes. The harness changes state.", ha="center", va="top",
            fontsize=FS_HEAD + 1, fontweight="bold")
    ax.text(12.0, 8.8, "Each item = (behavior, verification command, state)", ha="center",
            va="top", fontsize=FS_BODY, color="0.3")
    pos = {"not_started": (3.0, 5.8), "active": (12.0, 5.8), "passing": (21.0, 5.8),
           "blocked": (12.0, 2.0)}
    cols = {"not_started": C_GRAY, "active": C_DARK, "passing": C_OK, "blocked": C_ACCENT}
    for s, (x, y) in pos.items():
        round_box(ax, x - 2.4, y - 0.8, 4.8, 1.6, cols[s], cols[s], lw=0)
        ax.text(x, y, s, ha="center", va="center", fontsize=FS_HEAD, fontweight="bold",
                color="white")
    arrow(ax, 5.5, 5.8, 9.5, 5.8, lw=2.5)
    ax.text(7.5, 6.2, "scheduler (WIP=1)", ha="center", va="bottom", fontsize=FS_BODY)
    arrow(ax, 14.5, 5.8, 18.5, 5.8, color=C_OK, lw=3)
    ax.text(16.5, 6.2, "verifier only", ha="center", va="bottom", fontsize=FS_BODY,
            color=C_OK, fontweight="bold")
    ax.text(21.0, 4.5, "irreversible", ha="center", fontsize=FS_BODY, color=C_OK, style="italic")
    arrow(ax, 11.4, 4.9, 11.4, 2.9, color=C_ACCENT)
    arrow(ax, 12.6, 2.9, 12.6, 4.9, color=C_ACCENT)
    ax.text(13.2, 3.9, "dependency resolved", ha="left", va="center", fontsize=FS_BODY)
    footnote(fig, "Consumers: scheduler (next task), verifier (gates), handoff reporter, progress tracker")
    return fig


def build_step_9_1():
    fig, gs = slide("Step 9.1: L8 One entry, with evidence", ncols=2, tag=L8,
                    width_ratios=[1.15, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, json.dumps({
        "id": "F03",
        "behavior": "User can add an item to the cart",
        "verification": ["pytest tests/e2e/test_cart.py"],
        "state": "passing",
        "evidence": "commit 4f2e9a1",
    }, indent=2), title="feature_list.json")
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    row_list(ax, [("Too broad", "'Implement the shopping cart'"),
                  ("Right size: one session", "'User can add items to cart'"),
                  ("Too narrow", "'Create the name field on Cart'"),
                  ("Memo vs list", "~20 min to infer state vs ~3 min")],
             top=9.8, step=2.45)
    return fig


# -------------------------------------------------------
# CODE: harness-owned feature list
# -------------------------------------------------------
def build_step_10():
    fig, gs = slide("Step 10: Code: a feature list the agent cannot cheat")
    ax = fig.add_subplot(gs[0, 0])
    code = source_of(FeatureList.activate) + "\n" + source_of(FeatureList.request_pass) + "\n" + \
        source_of(FeatureList.mark_passing)
    code_panel(ax, code, title="Part A of this file, shown with inspect.getsource")
    return fig


def build_step_10_1():
    fig, gs = slide("Step 10.1: Live run: an agent session vs the harness")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(0, 24))
    log, _ = demo_feature_gate()
    for i, (actor, action, result) in enumerate(log):
        y = 9.6 - i * 1.2
        refused = result.startswith(("REFUSED", "FAIL"))
        face = C_OLD if refused else C_GOAL
        round_box(ax, 0.1, y - 1.0, 23.8, 1.0, face, "0.6", lw=0.8)
        ax.text(0.4, y - 0.5, f"{actor}: {action}", ha="left", va="center", fontsize=FS_BODY,
                family="monospace", fontweight="bold", parse_math=False)
        shown = result.replace("AssertionError: ", "").replace("slugify('Hi, there!') ", "")
        ax.text(11.6, y - 0.5, shown[:70], ha="left",
                va="center", fontsize=FS_NOTE, color=C_ACCENT if refused else C_OK,
                parse_math=False)
    footnote(fig, "Toy project: slugify(); every check runs in a fresh Python process.  "
                  "Try: python 9_10_Harness_Engineering.py --demo", y=0.06)
    return fig


# -------------------------------------------------------
# Lecture 9
# -------------------------------------------------------
L9 = ("Lecture 9", "Scope & verification")


def build_step_11():
    fig, gs = slide("Step 11: L9 Agents declare victory too early", ncols=2, tag=L9,
                    width_ratios=[1.1, 1], wspace=0.15, bottom=0.2)
    ax = fig.add_subplot(gs[0, 0])
    names = ["search\nendpoint", "pagination\nbug", "auth\nrefactor"]
    claimed = [len(v) for v in VICTORY_TASKS.values()]
    actual = [sum(v) for v in VICTORY_TASKS.values()]
    x = np.arange(3)
    ax.bar(x - 0.2, claimed, 0.38, color=C_GRAY, label="claimed")
    ax.bar(x + 0.2, actual, 0.38, color=C_OK, label="actually true")
    for xi, c, a in zip(x, claimed, actual):
        ax.text(xi + 0.2, a + 0.15, f"{a}/{c}", ha="center", fontsize=FS_BODY, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 7.5)
    ax.legend(fontsize=FS_BODY, loc="upper right")
    style(ax, "Port of victory-detector.ts", ylabel="checks")

    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    layers = [("3 System: E2E, scenarios", C_OK), ("2 Runtime: tests, app up", C_DARK),
              ("1 Static: lint, types", C_GRAY)]
    ax.text(6.0, 9.8, "Three-layer 'done' check", ha="center", va="top", fontsize=FS_HEAD,
            fontweight="bold")
    for i, (name, col) in enumerate(layers):
        w = 8.0 + 1.8 * i
        y0 = 6.6 - i * 2.4
        round_box(ax, 6.0 - w / 2, y0, w, 1.9, col, col, lw=0)
        ax.text(6.0, y0 + 0.95, name, ha="center", va="center", fontsize=FS_BODY,
                fontweight="bold", color="white")
    footnote(fig, "All three tasks were reported 'done'; 6 of 15 checks actually hold")
    return fig


def build_step_11_1():
    fig, gs = slide("Step 11.1: L9 Externalize the judgment of 'done'", ncols=2, tag=L9,
                    width_ratios=[1.15, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, "# weak\n"
                   "FAIL test_password_reset\n\n"
                   "# actionable\n"
                   "FAIL POST /api/password-reset\n"
                   "     -> 500\n"
                   "WHY: SMTP_HOST not set\n"
                   "FIX: add it to .env.example\n"
                   "     and config/email.py",
               title="Errors that carry the fix")
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    row_list(ax, [("Overconfidence", "Guo et al., ICML 2017"),
                  ("Self-evaluation bias", "authors grade themselves high"),
                  ("Unit tests mock boundaries", "integration bugs slip through"),
                  ("Fix: separate checker", "runtime signals decide 'done'")],
             top=9.8, step=2.45)
    return fig


# -------------------------------------------------------
# Lecture 10
# -------------------------------------------------------
L10 = ("Lecture 10", "Scope & verification")


def build_step_12():
    fig, gs = slide("Step 12: L10 Unit tests miss the boundaries", ncols=2, tag=L10,
                    width_ratios=[0.8, 1.2], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(-5, 5), ylim=(-5, 5))
    for r, name, col in [(4.8, "E2E", C_OK), (3.3, "integration", C_DARK), (1.8, "unit", "0.35")]:
        ax.add_patch(Circle((0, -4.8 + r), r, facecolor=col, alpha=0.3, edgecolor=col, lw=2))
        ax.text(0, -4.8 + 2 * r - 0.75, name, ha="center", va="center", fontsize=FS_BODY,
                fontweight="bold", color=col)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    ax.text(6.0, 9.8, "Electron export: unit tests all green", ha="center", va="top",
            fontsize=FS_HEAD, fontweight="bold")
    row_list(ax, [("Interface mismatch", "relative vs absolute path"),
                  ("State propagation", "progress never reaches the UI"),
                  ("Resource leak", "file handles not released"),
                  ("Permissions", "different when packaged"),
                  ("Error propagation", "exceptions never reach the UI")],
             top=8.7, step=1.72, face=C_OLD, edge=C_ACCENT)
    footnote(fig, "All five: missed by unit tests, caught end-to-end")
    return fig


def build_step_12_1():
    fig, gs = slide("Step 12.1: L10 Turn rules into checks", ncols=1, nrows=2, tag=L10,
                    height_ratios=[1.5, 0.75], hspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, "# architecture rule as an executable check\n"
                   "grep -r \"require('fs')\" src/renderer/ && exit 1\n\n"
                   "ERROR: direct import of 'fs' in src/renderer/App.tsx:12\n"
                   "WHY:   the renderer process has no Node.js access\n"
                   "FIX:   move it to src/preload/file-ops.ts, call window.api.readFile()",
               title="Agent-oriented error message: what, why, how to fix")
    ax = fig.add_subplot(gs[1, 0])
    blank(ax, xlim=(0, 12), ylim=(0, 3))
    layers = ["Types", "Config", "Repo", "Service", "Runtime", "UI"]
    for i, name in enumerate(layers):
        x0 = 0.1 + i * 2.0
        round_box(ax, x0, 0.9, 1.6, 1.3, C_NEW, C_DARK, lw=1.4)
        ax.text(x0 + 0.8, 1.55, name, ha="center", va="center", fontsize=FS_BODY, fontweight="bold")
        if i < len(layers) - 1:
            arrow(ax, x0 + 1.62, 1.55, x0 + 1.98, 1.55)
    ax.text(6.0, 0.35, "OpenAI's layered architecture: dependencies flow forward only (custom lint)",
            ha="center", va="center", fontsize=FS_NOTE, color="0.3")
    return fig


# -------------------------------------------------------
# Lecture 11
# -------------------------------------------------------
L11 = ("Lecture 11", "Operations")
DAW_PHASES = [  # Anthropic, March 2026, Opus 4.6, browser DAW (from the lecture's table)
    ("Planner", 4.7, 0.46), ("Build 1", 127, 71.08), ("QA 1", 8.8, 3.24),
    ("Build 2", 62, 36.89), ("QA 2", 6.8, 3.09), ("Build 3", 10.9, 5.88), ("QA 3", 9.6, 4.06),
]


def build_step_13():
    fig, gs = slide("Step 13: L11 Make the run observable", ncols=2, tag=L11,
                    width_ratios=[1, 1.1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    card(ax, 0.2, 5.3, 11.6, 4.5, "Runtime: WHAT happened",
         "lifecycle, critical paths,\ndata flow, full error context", C_NEW, C_DARK)
    card(ax, 0.2, 0.3, 11.6, 4.5, "Process: WHY accept it",
         "sprint contract, rubric,\nacceptance criteria, trace", C_GOAL, C_OK,
         title_color="#145a32")
    ax = fig.add_subplot(gs[0, 1])
    names = [p[0] for p in DAW_PHASES]
    costs = [p[2] for p in DAW_PHASES]
    cols = [C_FUN if n == "Planner" else (C_DARK if n.startswith("Build") else C_OK) for n in names]
    y = np.arange(len(names))
    ax.barh(y, costs, color=cols)
    for yi, c in zip(y, costs):
        ax.text(c + 1.5, yi, f"{c:.2f}", va="center", fontsize=FS_NOTE)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 90)
    style(ax, f"3-agent DAW build: {sum(costs):.2f} USD", xlabel="cost (USD)")
    footnote(fig, "Anthropic, March 2026, Opus 4.6: QA rounds are cheap, build rounds are not")
    return fig


def build_step_13_1():
    fig, gs = slide("Step 13.1: L11 Contracts and rubrics", ncols=2, tag=L11,
                    width_ratios=[1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, "Goal: visible citations in Q&A\n"
                   "Done means:\n"
                   "- user asks a question\n"
                   "- app returns an answer\n"
                   "- >= 1 citation is shown\n"
                   "- click opens the source\n"
                   "Out of scope: styling",
               title="Sprint contract (before coding)")
    ax = fig.add_subplot(gs[0, 1])
    code_panel(ax, "Score 1-5 per dimension:\n"
                   "- Grounding\n"
                   "- Citation quality\n"
                   "- Functionality\n"
                   "- Product coherence\n\n"
                   "Tune: read evaluator logs\n"
                   "for bugs it talked itself\n"
                   "out of.",
               title="Evaluator rubric")
    return fig


# -------------------------------------------------------
# Lecture 12
# -------------------------------------------------------
L12 = ("Lecture 12", "Operations")


def build_step_14():
    fig, gs = slide("Step 14: L12 Entropy is the default", ncols=2, tag=L12,
                    width_ratios=[1.2, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    wk = [1, 4, 8, 12]
    ax.plot(wk, [100, 95, 82, 68], "o-", color=C_ACCENT, lw=2.5, label="build, no cleanup")
    ax.plot(wk, [100, 92, 78, 61], "s--", color=C_ACCENT, lw=2, label="tests, no cleanup")
    ax.plot([1, 12], [100, 97], "o-", color=C_OK, lw=2.5, label="build, cleanup")
    ax.plot([1, 12], [100, 95], "s--", color=C_OK, lw=2, label="tests, cleanup")
    ax.set_ylim(55, 103)
    ax.legend(fontsize=FS_NOTE, loc="lower left")
    style(ax, "12-week project (lecture's case)", xlabel="week", ylabel="pass rate (%)")
    ax = fig.add_subplot(gs[0, 1])
    ax.bar([0, 1], [60, 9], color=[C_ACCENT, C_OK], width=0.6)
    for x, v, t in [(0, 60, "60+ min"), (1, 9, "9 min")]:
        ax.text(x, v + 2, t, ha="center", fontsize=FS_BODY, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no cleanup", "cleanup"])
    ax.set_ylim(0, 72)
    style(ax, "Session startup, week 12", ylabel="minutes")
    return fig


def build_step_14_1():
    fig, gs = slide("Step 14.1: L12 Every session leaves a clean state", ncols=2, tag=L12,
                    width_ratios=[1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    row_list(ax, [("1. Build passes", None), ("2. All tests pass", None),
                  ("3. Progress recorded", None), ("4. No stale artifacts", None),
                  ("5. Startup path works", None)],
             top=9.8, step=1.95, face=C_GOAL, edge=C_OK, title_color="#145a32")
    ax = fig.add_subplot(gs[0, 1])
    code_panel(ax, "# idempotent: safe to rerun\n"
                   "rm -f /tmp/debug-*.log\n"
                   "git checkout -- .env.local\n"
                   "npm run test\n\n"
                   "# monthly: switch off one\n"
                   "# harness part, rerun the\n"
                   "# benchmark; drop it if\n"
                   "# results hold",
               title="Cleanup habits")
    footnote(fig, "Session done = task verified AND clean-state check passed. 'Clean up later' means never.")
    return fig


# -------------------------------------------------------
# Lecture 13
# -------------------------------------------------------
L13 = ("Lecture 13", "Beyond the harness")


def build_step_15():
    fig, gs = slide("Step 15: L13 Stop prompting, design the loop", ncols=2, tag=L13,
                    width_ratios=[1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    ax.text(6.0, 9.8, "/goal = goal + verify + stop", ha="center", va="top", fontsize=FS_HEAD,
            fontweight="bold", color=C_DARK)
    row_list(ax, [("Turn-based", "manual prompt; agent thinks done"),
                  ("Goal-based  /goal", "independent judge or max turns"),
                  ("Time-based  /loop", "schedule; stop manually"),
                  ("Event-driven", "PR, CI failure; retry cap")],
             top=8.7, step=2.15, face=C_NEW, edge=C_DARK)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    ax.text(6.0, 9.8, "Six primitives", ha="center", va="top", fontsize=FS_HEAD, fontweight="bold")
    prims = [("Automations", "heartbeat"), ("Worktrees", "isolation"), ("Skills", "SKILL.md"),
             ("Connectors", "MCP"), ("Sub-agents", "maker/checker"), ("External state", "memory")]
    for i, (n, d) in enumerate(prims):
        x0 = 0.2 + (i % 2) * 5.9
        y0 = 6.6 - (i // 2) * 2.9
        round_box(ax, x0, y0, 5.6, 2.4, C_NEW, C_DARK, lw=1.4)
        ax.text(x0 + 2.8, y0 + 1.5, n, ha="center", va="center", fontsize=FS_BODY, fontweight="bold")
        ax.text(x0 + 2.8, y0 + 0.75, d, ha="center", va="center", fontsize=FS_NOTE)
    footnote(fig, "Does the task have an end? yes -> /goal, no -> /loop.  Someone in the crew must not believe you.")
    return fig


def build_step_15_1():
    fig, gs = slide("Step 15.1: L13 The silent costs of loops", ncols=2, tag=L13,
                    width_ratios=[1.1, 1], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    n = np.arange(1, 31)
    base, per_round = 8, 6
    resend = np.cumsum(base + per_round * n)
    compact = np.cumsum(np.minimum(base + per_round * n, base + per_round * 6))
    ax.plot(n, resend, color=C_ACCENT, lw=2.5, label="resend all (~N²)")
    ax.plot(n, compact, color=C_OK, lw=2.5, label="compaction (~N)")
    ax.legend(fontsize=FS_BODY)
    style(ax, "Token blowout (illustrative)", xlabel="rounds N", ylabel="input tokens (K)")
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    row_list(ax, [("Verification debt", "'looks fine' piles up"),
                  ("Comprehension rot", "code outpaces reading"),
                  ("Cognitive surrender", "you stop having opinions"),
                  ("Token blowout", "history reread every round")],
             top=9.8, step=2.45, face=C_OLD, edge=C_ACCENT, title_color=C_ACCENT)
    footnote(fig, "Maturity: goal runner -> scheduled task -> maker/checker -> self-feeding -> fleet")
    return fig


# -------------------------------------------------------
# CODE: maker-checker loop
# -------------------------------------------------------
def build_step_16():
    fig, gs = slide("Step 16: Code: a maker-checker goal loop", top=0.87, bottom=0.13)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, source_of(maker_checker_loop), title="Part A of this file")
    footnote(fig, "stop_reason(): all checks pass, or no progress for 3 rounds, or max rounds",
             y=0.075)
    return fig


def build_step_16_1():
    fig, gs = slide("Step 16.1: Live run of the loop", ncols=2, width_ratios=[1, 1.15], wspace=0.3)
    rounds = demo_loop()
    ax = fig.add_subplot(gs[0, 0])
    r = [d["round"] for d in rounds]
    p = [d["passed"] for d in rounds]
    ax.bar(r, p, color=[C_OK if v == 5 else C_DARK for v in p], width=0.6)
    ax.axhline(5, color=C_OK, ls="--", lw=1.5)
    ax.annotate("self-graded maker\nstops here", xy=(1, p[0]), xytext=(1.6, 0.5),
                fontsize=FS_NOTE, color=C_ACCENT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_ACCENT))
    ax.set_xticks(r)
    ax.set_ylim(0, 6)
    style(ax, "Checks passed per round (goal 5)", xlabel="round", ylabel="checks passed")
    ax = fig.add_subplot(gs[0, 1])
    lines = []
    for d in rounds:
        failed = [fid for fid, ok, _ in d["results"] if not ok]
        lines.append(f"Round {d['round']}: {d['passed']}/5  fail: {', '.join(failed) or '-'}")
        lines.append("  -> " + d["decision"].replace(", every check passes", ""))
    code_panel(ax, "\n".join(lines), title="loop-state.md")
    return fig


# -------------------------------------------------------
# Lecture 14
# -------------------------------------------------------
L14 = ("Lecture 14", "Beyond the harness")


def build_step_17():
    fig, gs = slide("Step 17: L14 From loops to graphs", tag=L14)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(0, 24))
    nodes = {"research": 2.8, "implement": 9.0, "verify": 15.2, "merge": 21.4}
    for n, x in nodes.items():
        round_box(ax, x - 2.4, 4.8, 4.8, 1.8, C_MUTE if n == "merge" else C_NEW, C_DARK, lw=2)
        ax.text(x, 5.7, n, ha="center", va="center", fontsize=FS_HEAD + 1, fontweight="bold")
    arrow(ax, 5.3, 5.7, 6.5, 5.7)
    arrow(ax, 11.5, 5.7, 12.7, 5.7)
    arrow(ax, 17.7, 5.7, 18.9, 5.7, color=C_OK, lw=3)
    ax.text(18.3, 6.9, "pass", ha="center", fontsize=FS_BODY, color=C_OK, fontweight="bold")
    arrow(ax, 14.8, 4.7, 9.4, 4.7, color=C_ACCENT, lw=2.5, rad=-0.45)
    ax.text(12.1, 2.4, "fail: back to implement", ha="center", fontsize=FS_BODY, color=C_ACCENT,
            fontweight="bold")
    ax.text(21.4, 7.6, "human approval", ha="center", fontsize=FS_BODY, color=C_FUN,
            fontweight="bold")
    ax.text(12.0, 9.6, "Shared state: requirements, code, review, attempts", ha="center",
            va="top", fontsize=FS_HEAD, fontweight="bold")
    ax.text(12.0, 8.7, "each node keeps its own private context", ha="center", va="top",
            fontsize=FS_BODY, color="0.35")
    ax.text(12.0, 0.8, "checkpoint after every step: resume, replay, audit", ha="center",
            fontsize=FS_BODY, color=C_DARK, fontweight="bold")
    footnote(fig, "Stack: prompt -> context -> loop -> graph, on a harness foundation")
    return fig


def build_step_17_1():
    fig, gs = slide("Step 17.1: L14 When a graph is worth it", ncols=2, tag=L14,
                    width_ratios=[1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax)
    row_list(ax, [("Goodhart", "metric up, business down"),
                  ("Blindness upward", "never questions its target"),
                  ("Conflict", "loops undercut each other")],
             top=9.8, step=3.2, face=C_OLD, edge=C_ACCENT, title_color=C_ACCENT)
    ax = fig.add_subplot(gs[0, 1])
    blank(ax)
    ax.text(6.0, 9.8, "Use a graph if >= 3 hold", ha="center", va="top", fontsize=FS_HEAD,
            fontweight="bold", color="#145a32")
    row_list(ax, [("1 Parallel independent units", None), ("2 Branch / rollback paths", None),
                  ("3 State worth checkpointing", None), ("4 Checkable done per node", None),
                  ("5 Benefit > coordination cost", None)],
             top=8.7, step=1.7, face=C_GOAL, edge=C_OK, title_color="#145a32")
    footnote(fig, "Orchestration tax: the human reviewer stays the bottleneck, however many nodes run")
    return fig


# -------------------------------------------------------
# CODE: the loop as a graph
# -------------------------------------------------------
LANGGRAPH_EXCERPT = '''\
graph = StateGraph(GraphState)   # attempts: Annotated[int, operator.add]
graph.add_node("research", research)
graph.add_node("implement", implement)
graph.add_node("verify", verify)
graph.add_node("merge", merge)
graph.add_edge(START, "research")
graph.add_edge("research", "implement")
graph.add_edge("implement", "verify")
graph.add_conditional_edges("verify", route_after_verify,
                            {"implement": "implement", "merge": "merge"})
graph.add_edge("merge", END)
app = graph.compile(checkpointer=MemorySaver())
app.invoke({"requirements": "fix the login page bug", "attempts": 0},
           config={"configurable": {"thread_id": "session-1"}})
'''


def build_step_18():
    fig, gs = slide("Step 18: Code: the course's LangGraph graph")
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, LANGGRAPH_EXCERPT, title="maker_checker_graph.py (pip install langgraph)")
    return fig


def build_step_18_1():
    fig, gs = slide("Step 18.1: The same skeleton, no dependencies", top=0.87, bottom=0.13)
    ax = fig.add_subplot(gs[0, 0])
    code_panel(ax, source_of(MiniGraph.run), title="MiniGraph.run (Part A of this file)")
    return fig


def build_step_18_2():
    fig, gs = slide("Step 18.2: Live run of the graph", top=0.8, bottom=0.3)
    state, trace, n_cp, n_gate = demo_graph()
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(0, len(trace)), ylim=(0, 2))
    colors = {"research": C_FUN, "implement": C_DARK, "verify": C_ORANGE, "merge": C_OK}
    short = {"research": "research", "implement": "impl.", "verify": "verify", "merge": "merge"}
    for i, node in enumerate(trace):
        round_box(ax, i + 0.06, 0.9, 0.88, 0.9, colors[node], colors[node], lw=0)
        ax.text(i + 0.5, 1.35, short[node], ha="center", va="center", fontsize=FS_NOTE,
                color="white", fontweight="bold")
    ax.text(len(trace) / 2, 0.35, f"{len(trace)} steps · {n_cp} checkpoints · "
            f"{n_gate} approval gate · passed {state['passed']}/5 after "
            f"{state['attempts']} failed reviews", ha="center", va="center", fontsize=FS_BODY,
            fontweight="bold")
    return fig


# -------------------------------------------------------
# Projects and takeaways
# -------------------------------------------------------
PROJECTS = [
    ("P1", "Prompt-only vs rules-first"), ("P2", "Agent-readable workspace"),
    ("P3", "Multi-session continuity"), ("P4", "Runtime feedback, scope control"),
    ("P5", "Self-verification, role split"), ("P6", "Complete harness (capstone)"),
    ("P7", "Your first automated loop"), ("P8", "Draw your workflow as a graph"),
]


def build_step_19():
    fig, gs = slide("Step 19: Hands-on projects", ncols=2, width_ratios=[1, 1], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, ylim=(0, 10))
    for i, (tag, name) in enumerate(PROJECTS):
        y = 9.4 - i * 1.2
        col = C_DARK if i < 6 else C_TEAL
        round_box(ax, 0.1, y - 0.45, 1.6, 0.9, col, col, lw=0)
        ax.text(0.9, y, tag, ha="center", va="center", color="white", fontsize=FS_BODY,
                fontweight="bold")
        ax.text(2.0, y, name, ha="left", va="center", fontsize=FS_BODY)
    ax = fig.add_subplot(gs[0, 1])
    code_panel(ax, "python 9_10_Harness_Engineering.py \\\n"
                   "    --scaffold my_repo\n\n"
                   "wrote my_repo/AGENTS.md\n"
                   "wrote my_repo/feature_list.json\n"
                   "wrote my_repo/claude-progress.md\n"
                   "wrote my_repo/init.sh",
               title="Start your own harness")
    footnote(fig, "Each project: starter/ (weak harness) vs solution/ (explicit harness)")
    return fig


def build_step_20():
    fig, gs = slide("Step 20: Harness engineering in eight lines")
    ax = fig.add_subplot(gs[0, 0])
    blank(ax, xlim=(0, 24))
    items = [
        ("L1-2", "Debug the harness before buying a bigger model."),
        ("L3-4", "Repo is the record; AGENTS.md is a short map."),
        ("L5-6", "Persist state; session 1 builds infrastructure."),
        ("L7-8", "WIP = 1; only a verifier sets 'passing'."),
        ("L9-10", "'Done' = executable checks, end to end."),
        ("L11-12", "Observe the run; leave a clean state."),
        ("L13", "Loops: goal + independent check + stop."),
        ("L14", "Graphs only when rollbacks earn their cost."),
    ]
    for i, (tag, text) in enumerate(items):
        y = 9.5 - i * 1.12
        part = next(p for p, t, _ in LECTURES if t == tag.split("-")[0])
        col = PART_COLORS[part]
        round_box(ax, 0.2, y - 0.42, 3.0, 0.84, col, col, lw=0)
        ax.text(1.7, y, tag, ha="center", va="center", fontsize=FS_BODY, fontweight="bold",
                color="white")
        ax.text(3.8, y, text, ha="left", va="center", fontsize=FS_HEAD + 1, parse_math=False)
    footnote(fig, "Generation is cheap; judgment is scarce. Build the harness around judgment.", y=0.07)
    return fig


# -------------------------------------------------------
# Main
# -------------------------------------------------------
BUILDERS = [
    build_preface, build_step_1,
    build_step_2, build_step_2_1, build_step_3, build_step_3_1,
    build_step_4, build_step_4_1, build_step_5, build_step_5_1,
    build_step_6, build_step_6_1, build_step_7, build_step_7_1,
    build_step_8, build_step_8_1, build_step_9, build_step_9_1,
    build_step_10, build_step_10_1, build_step_11, build_step_11_1,
    build_step_12, build_step_12_1, build_step_13, build_step_13_1,
    build_step_14, build_step_14_1, build_step_15, build_step_15_1,
    build_step_16, build_step_16_1, build_step_17, build_step_17_1,
    build_step_18, build_step_18_1, build_step_18_2, build_step_19, build_step_20,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn Harness Engineering lecture deck")
    parser.add_argument("--demo", action="store_true", help="run the harness demo in the terminal")
    parser.add_argument("--scaffold", metavar="DIR", help="write starter harness files into DIR")
    args = parser.parse_args()
    if args.scaffold:
        scaffold(args.scaffold)
    elif args.demo:
        print_demo()
    else:
        run_slide_deck(BUILDERS)
