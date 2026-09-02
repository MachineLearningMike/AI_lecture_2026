"""
Pad all lecture file numbers to 4 digits (0100, 0110, ..., 0990).

Needed so future numbers >= 1000 sort correctly next to 0990, and so
the gap-10 scheme can grow without mixed-width prefixes.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def file_num(name: str) -> int | None:
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else None


def to4(name: str) -> str:
    m = re.match(r"^(\d+)_(.*)$", name)
    if not m:
        return name
    return f"{int(m.group(1)):04d}_{m.group(2)}"


def build_map() -> dict[str, str]:
    renames: dict[str, str] = {}
    for p in ROOT.iterdir():
        if not p.is_file():
            continue
        n = file_num(p.name)
        if n is None:
            continue
        new = to4(p.name)
        if new != p.name:
            renames[p.name] = new
    return renames


def git_mv(src: Path, dst: Path) -> None:
    if not src.exists():
        print("SKIP", src.name)
        return
    if dst.exists():
        raise FileExistsError(dst)
    try:
        subprocess.run(
            ["git", "mv", "--", str(src), str(dst)],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError:
        src.rename(dst)
    print(f"mv {src.name} -> {dst.name}")


def two_phase(renames: dict[str, str]) -> None:
    staging = ROOT / ".__staging__"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    for old, new in renames.items():
        git_mv(ROOT / old, staging / new)
    for p in sorted(staging.iterdir()):
        git_mv(p, ROOT / p.name)
    staging.rmdir()


def rewrite_refs(renames: dict[str, str]) -> None:
    # Also rewrite bare number prefixes in filenames mentioned without full map
    # by applying explicit renames longest-first.
    pairs = sorted(renames.items(), key=lambda kv: len(kv[0]), reverse=True)
    mod_pairs = [(o[:-3], n[:-3]) for o, n in pairs if o.endswith(".py")]
    mod_pairs.sort(key=lambda kv: len(kv[0]), reverse=True)

    text_exts = {".py", ".ipynb", ".md", ".txt", ".json"}
    skip = {"tools_pad4_numbers.py"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_exts:
            continue
        if "__pycache__" in path.parts or path.name in skip:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        orig = text
        for o, n in pairs:
            text = text.replace(o, n)
        for o, n in mod_pairs:
            text = text.replace(f'import_module("{o}")', f'import_module("{n}")')
            text = text.replace(f"import_module('{o}')", f"import_module('{n}')")
        # generic: "100_Ch01 -> "0100_Ch01 already handled via pairs
        if text != orig:
            path.write_text(text, encoding="utf-8")
            print("rewrote", path.name)


def patch_formatters() -> None:
    """Ensure tooling uses :04d going forward."""
    for name in (
        "tools_gap10_renumber.py",
        "tools_insert_llm_usage_chapter.py",
        "tools_apply_curriculum.py",
        "tools_pad4_numbers.py",
    ):
        p = ROOT / name
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        t2 = t.replace('f"{n:04d}_', 'f"{n:04d}_').replace("{n:04d}", "{n:04d}")
        if t2 != t:
            p.write_text(t2, encoding="utf-8")
            print("patched formatter", name)


def regenerate_openers_readme() -> None:
    sys.path.insert(0, str(ROOT))
    import importlib
    import _curriculum
    importlib.reload(_curriculum)
    from _curriculum import CHAPTERS, CORE_PATH

    for meta in CHAPTERS:
        opener = meta["opener"]
        content = f'''"""
{opener}

Chapter opener: {meta["code"]} -- {meta["title"]}

Roadmap and student attempt for this chapter.
"""

from _chapter import run_chapter
from _curriculum import CHAPTERS

_META = next(c for c in CHAPTERS if c["opener"] == "{opener}")
run_chapter(_META)
'''
        (ROOT / opener).write_text(content, encoding="utf-8")

    lines = [
        "# AI Lecture 2026",
        "",
        "Deep-learning-focused AI course.",
        "",
        "## Gravity",
        "",
        "> Begin delivery. Advance toward understanding -- fast.",
        "",
        "Consecutive lecture numbers differ by **10**, zero-padded to **4 digits**",
        "(`0100`, `0110`, …) so sorting stays correct past `0999`.",
        "",
        "**Ch03 LLM usage** comes right after the DL story, before training-from-scratch.",
        "",
        "Map: [`_curriculum.py`](_curriculum.py). Critique: [`comment.txt`](comment.txt).",
        "",
        "## Chapters",
        "",
    ]
    for meta in CHAPTERS:
        lines += [
            f"### {meta['code']}: {meta['title']}",
            "",
            f"- Opener: `{meta['opener']}`",
            f"- Attempt: {meta['attempt']}",
            "",
            "| Tag | File | Note |",
            "|-----|------|------|",
        ]
        for tag, fname, note in meta["rows"]:
            f = fname if fname else "*(TODO)*"
            lines.append(f"| {tag} | `{f}` | {note} |")
        lines.append("")
    lines += ["## Minimal core path", ""]
    for f in CORE_PATH:
        lines.append(f"- `{f}`")
    lines += [
        "",
        "## Notes",
        "",
        "- File numbers are 4-digit zero-padded.",
        "- Ch03 = LLM **usage**; Ch15 = LLM **internals**; Ch04 = first **training** delivery.",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("openers+README refreshed")


def patch_docstrings(renames: dict[str, str]) -> None:
    for new in renames.values():
        path = ROOT / new
        if path.suffix != ".py" or not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        m = re.match(r'^(\ufeff)?"""([^\n]*)\n', text)
        if not m:
            continue
        first = m.group(2).strip()
        if first.endswith(".py") and first != path.name:
            text2 = text.replace(f'"""{first}\n', f'"""{path.name}\n', 1)
            if text2 != text:
                path.write_text(text2, encoding="utf-8")


def main() -> None:
    renames = build_map()
    print("renames", len(renames))
    if not renames:
        print("nothing to do")
        return
    (ROOT / "_pad4_map.json").write_text(json.dumps(renames, indent=2), encoding="utf-8")
    two_phase(renames)
    rewrite_refs(renames)
    patch_formatters()
    regenerate_openers_readme()
    patch_docstrings(renames)

    # verify
    sys.path.insert(0, str(ROOT))
    import importlib
    import _curriculum
    importlib.reload(_curriculum)
    from _curriculum import CHAPTERS
    bad = []
    for ch in CHAPTERS:
        for name in [ch["opener"], *[f for _, f, _ in ch["rows"] if f]]:
            if not re.match(r"^\d{4}_", name):
                bad.append(name)
            if not (ROOT / name).exists() and name.endswith(".py"):
                bad.append("MISSING " + name)
    print("bad/missing", bad[:10], "count", len(bad))
    print("DONE pad4")


if __name__ == "__main__":
    main()
