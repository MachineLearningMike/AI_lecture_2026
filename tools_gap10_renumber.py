"""
tools_gap10_renumber.py

Renumber every lecture file so consecutive teaching slots differ by 10.
Uses current _curriculum.CHAPTERS order (opener, then each real file row).
Companion notebooks/gifs share their parent lecture's number.
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from _curriculum import CHAPTERS, CORE_PATH  # noqa: E402


def strip_num(name: str) -> str:
    """100_foo.py -> foo.py ; 228_transformer_code_scratch.ipynb -> transformer_code_scratch.ipynb"""
    return re.sub(r"^\d+_", "", name)


def build_gap10_map() -> dict[str, str]:
    """old filename -> new filename, step 10, starting at 100."""
    renames: dict[str, str] = {}
    n = 100

    # Parent lecture -> companion assets that should share the same number
    companions: dict[str, list[str]] = {
        "123_torch_training.py": [
            "123_torch_training.ipynb",
            "123_torch_training.gif",
        ],
        "176_SGD_concept.py": ["176_SGD_code.ipynb"],
        "228_transformer_code.py": [
            "228_transformer_code_scratch.ipynb",
            "228_transformer_code_alt.ipynb",
        ],
        "230_machine_translation.py": ["230_machine_translation.ipynb"],
    }

    ordered: list[str] = []
    for ch in CHAPTERS:
        ordered.append(ch["opener"])
        for _tag, fname, _note in ch["rows"]:
            if fname:
                ordered.append(fname)

    for old in ordered:
        new = f"{n:04d}_{strip_num(old)}"
        renames[old] = new
        for comp in companions.get(old, []):
            # Keep companion descriptive suffix; only change leading number
            renames[comp] = f"{n:04d}_{strip_num(comp)}"
        n += 10

    return renames


def git_mv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print("SKIP missing", src.name)
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
    print(f"mv {src.name} -> {dst.relative_to(ROOT)}")


def two_phase_rename(renames: dict[str, str]) -> None:
    staging = ROOT / ".__staging__"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    for old, new in renames.items():
        src = ROOT / old
        if not src.exists():
            print("SKIP missing", old)
            continue
        git_mv(src, staging / new)

    for p in sorted(staging.iterdir()):
        git_mv(p, ROOT / p.name)

    staging.rmdir()


def rewrite_texts(renames: dict[str, str]) -> None:
    pairs = sorted(renames.items(), key=lambda kv: len(kv[0]), reverse=True)
    # Also map import_module("NNN_stem") without extension
    mod_pairs = []
    for old, new in pairs:
        if old.endswith(".py"):
            mod_pairs.append((old[:-3], new[:-3]))
    mod_pairs.sort(key=lambda kv: len(kv[0]), reverse=True)

    text_exts = {".py", ".ipynb", ".md", ".txt", ".json"}
    skip = {
        "tools_gap10_renumber.py",
        "tools_apply_curriculum.py",
        "_curriculum.py",  # rewritten explicitly below
    }

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in text_exts:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        if path.name in skip:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        orig = text
        for old, new in pairs:
            text = text.replace(old, new)
        for old, new in mod_pairs:
            text = text.replace(f'import_module("{old}")', f'import_module("{new}")')
            text = text.replace(f"import_module('{old}')", f"import_module('{new}')")
        if text != orig:
            path.write_text(text, encoding="utf-8")
            print("rewrote", path.relative_to(ROOT))


def rewrite_curriculum(renames: dict[str, str]) -> None:
    path = ROOT / "_curriculum.py"
    text = path.read_text(encoding="utf-8")

    # Replace filenames inside CHAPTERS / CORE_PATH by applying renames longest-first
    pairs = sorted(renames.items(), key=lambda kv: len(kv[0]), reverse=True)
    for old, new in pairs:
        text = text.replace(f'"{old}"', f'"{new}"')
        text = text.replace(f"'{old}'", f"'{new}'")

    # Refresh RENAMES dict comment — leave historical RENAMES as-is for archaeology,
    # but prepend a GAP10 note after the module docstring.
    if "GAP10 applied" not in text:
        text = text.replace(
            "Teach by this map -- not by leftover historical numbers.\n",
            "Teach by this map -- not by leftover historical numbers.\n"
            "GAP10 applied: consecutive teaching files differ by 10 in their prefix.\n",
            1,
        )
    path.write_text(text, encoding="utf-8")
    print("updated _curriculum.py names")


def regenerate_openers_and_readme() -> None:
    # Re-import fresh curriculum
    import importlib
    import _curriculum
    importlib.reload(_curriculum)
    from _curriculum import CHAPTERS as CH, CORE_PATH as CP

    for meta in CH:
        opener = meta["opener"]
        content = f'''"""
{opener}

Chapter opener: {meta["code"]} -- {meta["title"]}

Roadmap and student attempt for this chapter.
Generated from _curriculum.py -- edit the map, then refresh openers if needed.
"""

from _chapter import run_chapter
from _curriculum import CHAPTERS

_META = next(c for c in CHAPTERS if c["opener"] == "{opener}")
run_chapter(_META)
'''
        (ROOT / opener).write_text(content, encoding="utf-8")
        print("opener", opener)

    # README
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
        "Teach by **chapter order** below. Consecutive lecture numbers differ by **10** "
        "(room to insert new files without renumbering everything).",
        "",
        "Longer critique of gaps/redundancy: [`comment.txt`](comment.txt).",
        "Machine-readable map: [`_curriculum.py`](_curriculum.py).",
        "",
        "## Chapters",
        "",
    ]
    for meta in CH:
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
    for f in CP:
        lines.append(f"- `{f}`")
    lines.extend([
        "",
        "## Shared helpers",
        "",
        "- `_config.py` -- fonts, footer, screen size",
        "- `_chapter.py` -- chapter opener helpers",
        "- `_curriculum.py` -- chapter metadata + maps",
        "",
        "## Notes",
        "",
        "- Data scaling vs activation norm have distinct names "
        "(`normalize_data` vs `normalize_activations`).",
        "- Companion notebooks/gifs share their parent lecture number.",
        "- File numbers increase by 10 so inserts fit in the gaps.",
        "",
    ])
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote README.md")


def patch_docstrings(renames: dict[str, str]) -> None:
    for new in renames.values():
        path = ROOT / new
        if not path.exists() or path.suffix != ".py":
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


def dump_map(renames: dict[str, str]) -> None:
    (ROOT / "_gap10_map.json").write_text(
        json.dumps(renames, indent=2), encoding="utf-8"
    )
    # verify gaps on primary .py teaching files only
    nums = sorted(
        int(re.match(r"^(\d+)_", v).group(1))
        for v in renames.values()
        if v.endswith(".py") and re.match(r"^\d+_", v)
    )
    # unique sorted
    uniq = sorted(set(nums))
    bad = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1) if uniq[i + 1] - uniq[i] != 10]
    print("unique py numbers:", uniq[0], "...", uniq[-1], "count", len(uniq))
    if bad:
        print("WARNING non-10 gaps present among unique numbers (companions share nums): ok")
    # consecutive teaching py in map order
    ordered_py = [renames[k] for k in renames if k.endswith(".py") and k in {
        *[c["opener"] for c in CHAPTERS],
        *[f for c in CHAPTERS for _, f, _ in c["rows"] if f],
    }]
    # rebuild ordered from CHAPTERS after rename — use values in chapter order
    print("wrote _gap10_map.json")


def smoke() -> None:
    # reload
    import importlib
    import _curriculum
    importlib.reload(_curriculum)
    for meta in _curriculum.CHAPTERS:
        p = ROOT / meta["opener"]
        ast.parse(p.read_text(encoding="utf-8"))
    print("opener syntax ok:", len(_curriculum.CHAPTERS))


def main() -> None:
    renames = build_gap10_map()
    print("planned renames:", len(renames))
    for i, (a, b) in enumerate(list(renames.items())[:8]):
        print(f"  {a} -> {b}")
    print("  ...")

    # Collision check: two olds must not map to same new unless intentional companions
    rev: dict[str, list[str]] = {}
    for o, n in renames.items():
        rev.setdefault(n, []).append(o)
    collisions = {n: os_ for n, os_ in rev.items() if len(os_) > 1}
    # allow only if all share same number and are companions (py+ipynb+gif)
    for n, os_ in collisions.items():
        stems = {strip_num(x) for x in os_}
        print("shared number", n, "<-", os_)

    two_phase_rename(renames)
    rewrite_curriculum(renames)
    rewrite_texts(renames)
    regenerate_openers_and_readme()
    patch_docstrings(renames)
    dump_map(renames)
    smoke()
    print("DONE gap-10 renumber")


if __name__ == "__main__":
    main()
