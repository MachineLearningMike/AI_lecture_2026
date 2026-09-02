"""
tools_apply_curriculum.py

Apply _curriculum.py:
  1) write chapter opener .py files
  2) two-phase rename of lecture files
  3) rewrite companion filename strings
  4) write README.md
  5) stamp comment.txt header

Run from repo root:
  python tools_apply_curriculum.py
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from _curriculum import CHAPTERS, CORE_PATH, RENAMES  # noqa: E402


def write_chapter_openers() -> None:
    for meta in CHAPTERS:
        opener = meta["opener"]
        path = ROOT / opener
        code = meta["code"]
        content = f'''"""
{opener}

Chapter opener: {code} -- {meta["title"]}

Roadmap and student attempt for this chapter.
Generated from _curriculum.py -- edit the map, re-run tools_apply_curriculum.py
to refresh openers (or edit this file by hand after freezing the map).
"""

from _chapter import run_chapter
from _curriculum import CHAPTERS

_META = next(c for c in CHAPTERS if c["opener"] == "{opener}")
run_chapter(_META)
'''
        path.write_text(content, encoding="utf-8")
        print("wrote", opener)


def _git_mv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print("SKIP missing", src.name)
        return
    if dst.exists():
        raise FileExistsError(f"destination exists: {dst}")
    try:
        subprocess.run(
            ["git", "mv", "--", str(src), str(dst)],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError:
        # untracked files: plain rename
        src.rename(dst)
    print(f"mv {src.name} -> {dst.relative_to(ROOT)}")


def two_phase_rename() -> None:
    staging = ROOT / ".__staging__"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    # Phase 1: old -> staging/new
    for old, new in RENAMES.items():
        src = ROOT / old
        dst = staging / new
        if not src.exists():
            print("SKIP missing", old)
            continue
        _git_mv(src, dst)

    # Phase 2: staging/new -> root/new
    for p in sorted(staging.iterdir()):
        _git_mv(p, ROOT / p.name)

    if staging.exists():
        try:
            staging.rmdir()
        except OSError:
            # leftover?
            for p in staging.rglob("*"):
                print("leftover", p)
            raise


def rewrite_references() -> None:
    # Longest old basenames first to avoid partial clobber
    pairs = sorted(RENAMES.items(), key=lambda kv: len(kv[0]), reverse=True)
    # Also map stem-only mentions carefully via full filenames only
    text_exts = {".py", ".ipynb", ".md", ".txt", ".json"}
    skip_names = {
        "tools_apply_curriculum.py",
        "_curriculum.py",
        # openers regenerated separately
    }

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in text_exts:
            continue
        if ".__staging__" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name in skip_names:
            continue
        # Don't rewrite the curriculum map itself by old keys inside comments only — actually
        # _curriculum.py uses old names as dict keys; must NOT rewrite those keys.
        if path.name == "_curriculum.py":
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        orig = text
        for old, new in pairs:
            if old in text:
                text = text.replace(old, new)
        if text != orig:
            path.write_text(text, encoding="utf-8")
            print("rewrote refs in", path.relative_to(ROOT))


def patch_docstring_filenames() -> None:
    """If a .py starts with a docstring whose first line is an old-style name, fix to actual name."""
    for path in sorted(ROOT.glob("*.py")):
        if path.name.startswith("_") or path.name.startswith("tools_"):
            continue
        text = path.read_text(encoding="utf-8")
        m = re.match(r'^(\ufeff)?"""([^\n]*)\n', text)
        if not m:
            continue
        first = m.group(2).strip()
        if first.endswith(".py") and first != path.name:
            # replace first docstring line
            text2 = text.replace(f'"""{first}\n', f'"""{path.name}\n', 1)
            if text2 != text:
                path.write_text(text2, encoding="utf-8")
                print("docstring name ->", path.name)


def write_readme() -> None:
    lines = [
        "# AI Lecture 2026",
        "",
        "Deep-learning-focused AI course.",
        "",
        "## Gravity (from the contract)",
        "",
        "> Begin delivery. Advance toward understanding -- fast.",
        "",
        "Habit per module: **see it run → touch it → name it → own it**.",
        "",
        "Teach by **chapter order** below. Do not teach by leftover historical numbers.",
        "",
        "Longer critique of gaps/redundancy: [`comment.txt`](comment.txt).",
        "Machine-readable map: [`_curriculum.py`](_curriculum.py).",
        "",
        "## Chapters",
        "",
    ]
    for meta in CHAPTERS:
        lines.append(f"### {meta['code']}: {meta['title']}")
        lines.append("")
        lines.append(f"- Opener: `{meta['opener']}`")
        lines.append(f"- Attempt: {meta['attempt']}")
        lines.append("")
        lines.append("| Tag | File | Note |")
        lines.append("|-----|------|------|")
        for tag, fname, note in meta["rows"]:
            f = fname if fname else "*(TODO)*"
            lines.append(f"| {tag} | `{f}` | {note} |")
        lines.append("")

    lines.extend([
        "## Minimal core path (time-short)",
        "",
        "Run these in order:",
        "",
    ])
    for f in CORE_PATH:
        lines.append(f"- `{f}`")
    lines.extend([
        "",
        "## Shared helpers",
        "",
        "- `_config.py` -- fonts, footer, screen size",
        "- `_chapter.py` -- chapter opener helpers",
        "- `_curriculum.py` -- rename map + chapter metadata",
        "",
        "## Notes",
        "",
        "- `142_normalize_data.py` scales **features/data**.",
        "- `194_normalize_activations.py` is BatchNorm / LayerNorm.",
        "- Transformer notebooks: `228_transformer_code_scratch.ipynb`, "
        "`228_transformer_code_alt.ipynb` (enrichment; prefer `228_transformer_code.py`).",
        "- Parallel speaking track: `900_ChParallel_Speaking.py`, `901_Asian_interview.py`.",
        "",
    ])
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote README.md")


def stamp_comment() -> None:
    path = ROOT / "comment.txt"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    banner = (
        "NOTE (applied): File renames and chapter openers are live.\n"
        "Use README.md and _curriculum.py as the current map.\n"
        "Names inside this critique may still mention pre-rename IDs.\n"
        "========================================================================\n\n"
    )
    if text.startswith("NOTE (applied):"):
        return
    path.write_text(banner + text, encoding="utf-8")
    print("stamped comment.txt")


def dump_json_map() -> None:
    payload = {
        "renames": RENAMES,
        "chapters": [
            {
                "opener": c["opener"],
                "code": c["code"],
                "title": c["title"],
                "rows": c["rows"],
            }
            for c in CHAPTERS
        ],
        "core_path": CORE_PATH,
    }
    (ROOT / "_curriculum_map.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print("wrote _curriculum_map.json")


def smoke_openers() -> None:
    ok = 0
    for meta in CHAPTERS:
        p = ROOT / meta["opener"]
        ast.parse(p.read_text(encoding="utf-8"))
        ok += 1
    print("ast-ok openers:", ok)


def main() -> None:
    os.chdir(ROOT)
    write_chapter_openers()
    two_phase_rename()
    rewrite_references()
    patch_docstring_filenames()
    write_readme()
    stamp_comment()
    dump_json_map()
    smoke_openers()
    print("DONE")


if __name__ == "__main__":
    main()
