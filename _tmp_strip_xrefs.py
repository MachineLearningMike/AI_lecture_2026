"""Remove cross-slide / cross-file references so each slide file stands alone."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILE_RE = re.compile(r"^(\d+)_(\d+)_(.+)\.py$")

# Filenames that look like lecture materials
RE_PY = re.compile(r"\b\d+[_\d]*_[A-Za-z0-9_]+\.py\b|\b\d{3,4}_[A-Za-z0-9_]+\.py\b")
RE_IPYNB = re.compile(r"\b[A-Za-z0-9_]+\.ipynb\b")
RE_XREF_START = re.compile(
    r"(?i)^\s*(see also|companion(?:\s+to|\s*:)?|pairs with|sits after|sits before|"
    r"sits between|sits near|sits late|continuation after|look at|"
    r"next lecture|previous lecture|next lectures)\b"
)
RE_ONLY_FILENAMES = re.compile(
    r"(?i)^\s*(?:and\s+)?(?:`?[\w.-]+\.(?:py|ipynb)`?\s*,?\s*)+\.?$"
)
RE_INLINE_SEE = re.compile(
    r"(?i)(?:\s*[—(]\s*)?(?:see|look at|cf\.?)\s+(?:also\s+)?"
    r"[\w./-]+\.(?:py|ipynb)(?:\s+for\b[^.——\n]*)?[).]?"
)


def is_slide(path: Path) -> bool:
    return bool(FILE_RE.match(path.name))


def line_has_material_ref(line: str, self_name: str) -> bool:
    for m in RE_PY.finditer(line):
        if m.group(0) != self_name:
            return True
    if RE_IPYNB.search(line):
        return True
    return False


def clean_module_docstring(src: str, self_name: str) -> str:
    if not src.startswith('"""') and not src.startswith("'''"):
        # may start with shebang/comment then docstring
        pass
    # Handle common pattern: optional line comments, then docstring
    m = re.match(r"(?s)^(.*?)([\"']{3})(.*?)(\2)(.*)$", src)
    if not m:
        return src
    prefix, q, body, q2, rest = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    # Only treat as module docstring if it starts within first ~15 lines
    if prefix.count("\n") > 12:
        return src

    lines = body.splitlines(True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Drop xref starter lines and their continuation indent filename lists
        if RE_XREF_START.match(stripped) or (
            line_has_material_ref(stripped, self_name)
            and (
                stripped.lower().startswith("see also")
                or "companion" in stripped.lower()
                or stripped.lower().startswith("sits ")
                or RE_ONLY_FILENAMES.match(stripped)
            )
        ):
            i += 1
            while i < len(lines):
                nxt = lines[i]
                ns = nxt.strip()
                if not ns:
                    # keep a single blank only if next kept content needs separation later
                    break
                if (
                    line_has_material_ref(ns, self_name)
                    or RE_ONLY_FILENAMES.match(ns)
                    or ns.startswith("and ")
                    or (nxt.startswith(" ") or nxt.startswith("\t"))
                    and (".py" in ns or ".ipynb" in ns or ns.endswith(","))
                ):
                    i += 1
                    continue
                break
            # skip following blank lines collapsed later
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue

        # Strip inline file refs inside otherwise useful docstring lines
        if line_has_material_ref(line, self_name):
            cleaned = RE_INLINE_SEE.sub("", line)
            # also remove bare filename tokens that aren't self
            cleaned = re.sub(
                rf"\b(?!{re.escape(self_name[:-3])})\d+[_\w]*_[A-Za-z0-9_]+\.py\b",
                "",
                cleaned,
            )
            cleaned = re.sub(r"\b\d{3,4}_[A-Za-z0-9_]+\.py\b", "", cleaned)
            cleaned = re.sub(r"\b[A-Za-z0-9_]+\.ipynb\b", "", cleaned)
            cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
            cleaned = re.sub(r" ?\([ ,;]*\)", "", cleaned)
            cleaned = re.sub(r" ,", ",", cleaned)
            cleaned = re.sub(r",\s*,", ",", cleaned)
            cleaned = re.sub(r"\s+$", "\n" if line.endswith("\n") else "", cleaned)
            if cleaned.strip() and not RE_ONLY_FILENAMES.match(cleaned.strip()):
                out.append(cleaned if cleaned.endswith("\n") or not line.endswith("\n") else cleaned)
                if line.endswith("\n") and not cleaned.endswith("\n"):
                    out[-1] = cleaned + "\n"
            i += 1
            continue

        out.append(line)
        i += 1

    # Collapse 3+ blanks to 2; strip leading/trailing blank in docstring
    text = "".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
    if text:
        text = "\n" + text + "\n"
    else:
        text = "\n"
    return prefix + q + text + q2 + rest


def clean_source(src: str, self_name: str) -> str:
    src = clean_module_docstring(src, self_name)
    lines = src.splitlines(True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Full-line comments that only cross-ref other materials
        if stripped.startswith("#") and line_has_material_ref(stripped, self_name):
            # keep if it's the historical self banner on line 2-ish: "# oldname.py" matching topic
            # Drop any comment that mentions another lecture file / notebook
            i += 1
            continue

        # print("See also: ...") possibly multi-line string
        if re.match(r"(?i)^\s*print\(\s*$", line) or re.match(
            r"(?i)^\s*print\(\s*[\"'].*see also", line
        ):
            block = line
            j = i
            if '"""' in line or "'''" in line:
                q = '"""' if '"""' in line else "'''"
                if line.count(q) < 2:
                    j += 1
                    while j < len(lines) and q not in lines[j]:
                        block += lines[j]
                        j += 1
                    if j < len(lines):
                        block += lines[j]
            elif "(" in line and ")" not in line:
                j += 1
                while j < len(lines) and ")" not in lines[j]:
                    block += lines[j]
                    j += 1
                if j < len(lines):
                    block += lines[j]
            if re.search(r"(?i)see also", block) and (
                line_has_material_ref(block, self_name) or ".py" in block
            ):
                i = j + 1
                continue

        # Single-line print See also
        if re.match(r"(?i)^\s*print\(.*see also", stripped) and line_has_material_ref(
            stripped, self_name
        ):
            i += 1
            continue

        # Inline comment tails with file refs: code  # see foo.py
        if "#" in line and line_has_material_ref(line, self_name):
            code, _, comment = line.partition("#")
            if code.strip() and line_has_material_ref("#" + comment, self_name):
                # drop only the comment part if code remains meaningful
                if RE_PY.search(comment) or RE_IPYNB.search(comment):
                    out.append(code.rstrip() + ("\n" if line.endswith("\n") else ""))
                    i += 1
                    continue

        # String lines in slide content that are pure See also / Companion with filenames
        if line_has_material_ref(line, self_name) and re.search(
            r"(?i)see also|companion to|sits after|sits before|look at", line
        ):
            # Try to remove just the filename bits from string literals; if line becomes empty string content, drop
            new = RE_PY.sub("", line)
            new = RE_IPYNB.sub("", new)
            new = re.sub(r"(?i)see also:\s*", "", new)
            new = re.sub(r"[ \t]{2,}", " ", new)
            # If it's clearly a dedicated xref string, skip the line
            if re.search(r"(?i)see also|companion to|sits after|sits before", new) and not re.search(
                r"[A-Za-z]{3,}", re.sub(r"(?i)see also|companion to|sits after|sits before|steps?", "", new)
            ):
                i += 1
                continue
            line = new

        out.append(line)
        i += 1

    text = "".join(out)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def main() -> None:
    changed = []
    for path in sorted(ROOT.glob("[0-9]_*.py")):
        if not is_slide(path):
            continue
        original = path.read_text(encoding="utf-8")
        updated = clean_source(original, path.name)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(path.name)
    print(f"updated {len(changed)} files")
    for name in changed:
        print(" ", name)


if __name__ == "__main__":
    main()
