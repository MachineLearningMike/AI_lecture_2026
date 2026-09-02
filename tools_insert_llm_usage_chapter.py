"""
Insert Ch03 LLM Usage between 160 and former 170.
Shift every file with number >= 170 by +50 (gap-10 preserved).
Renumber chapter codes: old Ch03..Ch15 become Ch04..Ch16.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SHIFT = 50
CUTOFF = 170


def file_num(name: str) -> int | None:
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else None


def with_num(name: str, n: int) -> str:
    stem = re.sub(r"^\d+_", "", name)
    return f"{n:04d}_{stem}"


def ch_bump(name: str) -> str:
    """Ch03 -> Ch04 in filenames like 170_Ch03_First_Delivery.py (after num shift)."""
    def repl(m):
        return f"Ch{int(m.group(1)) + 1:02d}"
    return re.sub(r"Ch(\d+)", repl, name, count=1)


def build_shift_map() -> dict[str, str]:
    renames: dict[str, str] = {}
    for p in ROOT.iterdir():
        if not p.is_file():
            continue
        n = file_num(p.name)
        if n is None or n < CUTOFF:
            continue
        new_n = n + SHIFT
        new_name = with_num(p.name, new_n)
        # bump chapter index in opener / ChXX names
        if re.search(r"_Ch\d+", new_name):
            new_name = ch_bump(new_name)
        # Parallel opener: was ChParallel, keep
        if "ChParallel" in p.name:
            new_name = with_num(p.name, new_n)
        renames[p.name] = new_name
    return renames


def git_mv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
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
    for old, new in sorted(renames.items(), key=lambda kv: file_num(kv[0]) or 0, reverse=True):
        git_mv(ROOT / old, staging / new)
    for p in sorted(staging.iterdir()):
        git_mv(p, ROOT / p.name)
    staging.rmdir()


def rewrite_refs(renames: dict[str, str]) -> None:
    pairs = sorted(renames.items(), key=lambda kv: len(kv[0]), reverse=True)
    mod_pairs = [(o[:-3], n[:-3]) for o, n in pairs if o.endswith(".py")]
    mod_pairs.sort(key=lambda kv: len(kv[0]), reverse=True)
    text_exts = {".py", ".ipynb", ".md", ".txt", ".json"}
    skip = {"tools_insert_llm_usage_chapter.py", "tools_gap10_renumber.py",
            "tools_apply_curriculum.py"}
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
        # chapter code mentions in openers will be regenerated
        if text != orig:
            path.write_text(text, encoding="utf-8")
            print("rewrote", path.name)


def main_shift_only():
    renames = build_shift_map()
    print("shift renames", len(renames), "examples:")
    for i, (a, b) in enumerate(sorted(renames.items(), key=lambda kv: file_num(kv[0]) or 0)[:6]):
        print(f"  {a} -> {b}")
    (ROOT / "_insert_shift_map.json").write_text(json.dumps(renames, indent=2), encoding="utf-8")
    two_phase(renames)
    rewrite_refs(renames)
    print("SHIFT DONE")


if __name__ == "__main__":
    main_shift_only()
