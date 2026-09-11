"""Scan lecture slides for py_compile / import / builder errors."""
from __future__ import annotations

import importlib.util
import py_compile
import re
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import _config

SLIDE_RE = re.compile(r"^\d+_\d+_.*\.py$")
SKIP = {
    "_tmp_scan_slides.py",
}


def is_slide(path: Path) -> bool:
    if path.name in SKIP or path.name.startswith("_") or path.name.startswith("tools_"):
        return False
    return bool(SLIDE_RE.match(path.name))


def scan_one(path: Path) -> list[str]:
    errs: list[str] = []
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as e:
        errs.append(f"COMPILE: {e}")
        return errs

    builders: list = []
    _config.run_slide_deck = lambda bs, **kw: builders.extend(bs)
    # Some older slides call show_maximized at import; stub it.
    if hasattr(_config, "show_maximized"):
        _orig_show = _config.show_maximized
        _config.show_maximized = lambda fig, **kw: None
    else:
        _orig_show = None

    # Also stub local plt.show during import for orphan show_maximized
    _orig_plt_show = plt.show
    plt.show = lambda *a, **k: None

    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            errs.append(f"IMPORT: {type(e).__name__}: {e}")
            return errs

        # Discover builders if not registered via run_slide_deck
        if not builders:
            names = sorted(
                n for n in dir(mod)
                if n.startswith("build_step_") and callable(getattr(mod, n))
            )
            builders = [getattr(mod, n) for n in names]

        if not builders:
            # Single-figure modules that only call show at import are OK if import lived.
            return errs

        for i, b in enumerate(builders, 1):
            try:
                fig = b()
                if fig is None:
                    errs.append(f"STEP {i} ({getattr(b, '__name__', '?')}): returned None")
                else:
                    plt.close("all")
            except Exception as e:
                errs.append(
                    f"STEP {i} ({getattr(b, '__name__', '?')}): "
                    f"{type(e).__name__}: {e}"
                )
                plt.close("all")
    finally:
        plt.show = _orig_plt_show
        if _orig_show is not None:
            _config.show_maximized = _orig_show

    return errs


def main() -> int:
    slides = sorted(p for p in ROOT.glob("*.py") if is_slide(p))
    print(f"scanning {len(slides)} slides...")
    failed = []
    for p in slides:
        errs = scan_one(p)
        if errs:
            failed.append((p.name, errs))
            print(f"FAIL {p.name}")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"OK   {p.name}")

    print("\n" + "=" * 60)
    print(f"failed {len(failed)} / {len(slides)}")
    out = ROOT / "_tmp_scan_results.txt"
    lines = []
    for name, errs in failed:
        lines.append(name)
        lines.extend(f"  {e}" for e in errs)
    out.write_text("\n".join(lines) if lines else "ALL CLEAN\n", encoding="utf-8")
    print(f"wrote {out.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
