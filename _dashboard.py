"""
_dashboard.py

Local browser dashboard for launching lecture slide decks during class.

Usage (on the lecture machine):
  1. Activate your Python env in a terminal (conda/venv/etc.).
  2. cd to this repo root.
  3. python _dashboard.py
  4. Open the printed URL (default http://127.0.0.1:8765).
  5. Click a subject -- it launches with *this same* Python (sys.executable).

The dashboard inherits whatever interpreter is already active in the terminal.
It only serves http://127.0.0.1 and launches local slide files.

Dependency invariant:
  Catalog discovery is filesystem + AST only (no curriculum maps).
  Runtime lecture code depends on slide files + _config.py only.
  This dashboard itself is stdlib-only.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765
FILE_RE = re.compile(r"^(\d+)_(\d+)_(.+)\.py$")
LAST_FILE = ROOT / "_dashboard_last.txt"

_lock = threading.Lock()
_proc: subprocess.Popen | None = None


def pretty_stem(stem: str) -> str:
    return stem.replace("_", " ")


def read_last_file() -> str | None:
    try:
        name = LAST_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not name or not FILE_RE.match(name):
        return None
    if not (ROOT / name).is_file():
        return None
    return name


def write_last_file(filename: str) -> None:
    try:
        LAST_FILE.write_text(filename + "\n", encoding="utf-8")
    except OSError:
        pass


def discover_catalog() -> list[dict]:
    """Return chapters with title + subjects (sorted)."""
    chapters: dict[int, dict] = {}
    for path in sorted(ROOT.glob("[0-9]_*.py")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        ch, num, stem = int(m.group(1)), int(m.group(2)), m.group(3)
        bucket = chapters.setdefault(
            ch,
            {"chapter": ch, "title": f"Chapter {ch}", "subjects": []},
        )
        if num == 0:
            bucket["title"] = pretty_stem(stem)
            bucket["opener"] = path.name
            continue
        bucket["subjects"].append(
            {
                "num": num,
                "subject": pretty_stem(stem),
                "file": path.name,
            }
        )
    out = [chapters[k] for k in sorted(chapters)]
    for ch in out:
        ch["subjects"].sort(key=lambda s: s["num"])
    return out


def _kill_current() -> None:
    global _proc
    if _proc is None:
        return
    if _proc.poll() is None:
        try:
            _proc.kill()
        except Exception:
            pass
        try:
            _proc.wait(timeout=0.4)
        except Exception:
            pass
    _proc = None


def launch_slide(filename: str) -> dict:
    global _proc
    path = ROOT / filename
    if not path.is_file() or not FILE_RE.match(filename):
        return {"ok": False, "error": f"Unknown slide file: {filename}"}

    with _lock:
        _kill_current()

        creationflags = 0
        if sys.platform == "win32":
            # Detach from console Ctrl+C of the dashboard when possible.
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        _proc = subprocess.Popen(
            [sys.executable, "-u", str(path)],
            cwd=str(ROOT),
            creationflags=creationflags,
        )
        write_last_file(filename)
        return {"ok": True, "last": filename}


def render_page(catalog: list[dict], last_file: str | None = None) -> str:
    last_file = last_file or read_last_file()
    chapters_html = []
    for ch in catalog:
        items = []
        for s in ch["subjects"]:
            last_cls = " last" if s["file"] == last_file else ""
            items.append(
                f'<button type="button" class="subject{last_cls}" data-file="{html.escape(s["file"])}">'
                f'<span class="sub-name">{html.escape(s["subject"])}</span>'
                f"</button>"
            )
        opener = ch.get("opener")
        opener_btn = ""
        if opener:
            last_cls = " last" if opener == last_file else ""
            opener_btn = (
                f'<button type="button" class="subject opener{last_cls}" data-file="{html.escape(opener)}">'
                f'<span class="sub-name">Chapter title slide</span>'
                f"</button>"
            )
        n_subj = len(ch["subjects"])
        chapters_html.append(
            f'<section class="chapter">'
            f'<h2 class="chapter-title">'
            f'<span class="ch-name">{html.escape(ch["title"])}</span>'
            f'<span class="ch-count">{n_subj} subject{"s" if n_subj != 1 else ""}</span>'
            f"</h2>"
            f'<div class="grid">{opener_btn}{"".join(items)}</div>'
            f"</section>"
        )

    last_label = html.escape(last_file) if last_file else "none yet"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Lecture Dashboard</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e7ecf3;
      --muted: #9aa8bc;
      --accent: #3d8bfd;
      --border: #2b3a4f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg); color: var(--text);
    }}
    header {{
      position: sticky; top: 0; z-index: 10;
      background: rgba(15,20,25,0.95); backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
      padding: 14px 20px;
    }}
    h1 {{ margin: 0; font-size: 1.25rem; }}
    .last-line {{
      margin-top: 8px; color: var(--muted); font-size: 0.9rem;
    }}
    .last-line strong {{ color: #9ec1ff; font-weight: 700; }}
    main {{ padding: 16px 20px 40px; max-width: 1200px; margin: 0 auto; }}
    .chapter {{ margin-bottom: 28px; }}
    .chapter-title {{
      display: flex; align-items: center; gap: 12px;
      margin: 0 0 12px; padding: 12px 14px;
      border-radius: 12px;
      background: linear-gradient(90deg, #1a3358 0%, #152033 55%, #121a27 100%);
      border: 1px solid #2f4d75;
      border-left: 5px solid var(--accent);
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22);
    }}
    .ch-name {{
      color: #061018;
      background: var(--accent);
      font-size: 1.05rem; font-weight: 800;
      line-height: 1.2;
      padding: 6px 14px;
      border-radius: 999px;
      box-shadow: 0 0 0 3px rgba(61, 139, 253, 0.22);
      max-width: calc(100% - 8rem);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .ch-count {{
      margin-left: auto;
      color: var(--muted); font-size: 0.78rem; font-weight: 600;
      padding: 4px 10px; border-radius: 999px;
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--border);
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 8px;
    }}
    button.subject {{
      appearance: none; -webkit-appearance: none;
      text-align: left; background: var(--panel); color: var(--text);
      border: 2px solid var(--border); border-radius: 10px;
      padding: 10px 12px; cursor: pointer;
    }}
    button.subject:hover {{
      border-color: var(--accent) !important;
      box-shadow: 0 0 0 2px var(--accent);
      transform: translateY(-1px);
    }}
    button.subject.last {{
      border-color: var(--accent) !important;
      box-shadow: 0 0 0 2px var(--accent) !important;
      background: #243b5c !important;
      transform: translateY(-1px);
    }}
    button.subject.opener {{ border-style: dashed; }}
    button.subject.opener.last {{ border-style: solid !important; }}
    .sub-name {{ font-weight: 650; font-size: 0.95rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Lecture Dashboard</h1>
    <div class="last-line">Last launched: <strong id="last-label">{last_label}</strong></div>
  </header>
  <main>
    {"".join(chapters_html)}
  </main>
  <script>
    function markLast(file) {{
      var buttons = document.querySelectorAll("button.subject");
      for (var i = 0; i < buttons.length; i++) {{
        buttons[i].classList.remove("last");
      }}
      var label = document.getElementById("last-label");
      if (!file) {{
        if (label) label.textContent = "none yet";
        return;
      }}
      if (label) label.textContent = file;
      for (var j = 0; j < buttons.length; j++) {{
        if (buttons[j].getAttribute("data-file") === file) {{
          buttons[j].classList.add("last");
          break;
        }}
      }}
    }}

    document.addEventListener("click", function (ev) {{
      var btn = ev.target.closest("button.subject");
      if (!btn) return;
      var file = btn.getAttribute("data-file");
      markLast(file);
      fetch("/api/launch?file=" + encodeURIComponent(file), {{ method: "POST" }})
        .then(function (r) {{ return r.json(); }})
        .then(function (data) {{
          if (!data.ok) {{
            alert(data.error || "Launch failed");
            return;
          }}
          markLast(data.last || file);
        }})
        .catch(function () {{
          alert("Launch failed");
        }});
    }});
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    catalog: list[dict] = []

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            page = render_page(self.catalog, read_last_file())
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/launch":
            qs = parse_qs(parsed.query)
            filename = (qs.get("file") or [""])[0]
            result = launch_slide(filename)
            code = 200 if result.get("ok") else 400
            self._send(code, json.dumps(result).encode("utf-8"), "application/json")
            return
        self._send(404, b"not found", "text/plain")


def main() -> int:
    Handler.catalog = discover_catalog()
    n_subj = sum(len(ch["subjects"]) for ch in Handler.catalog)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("=" * 60)
    print("Lecture Dashboard")
    print("=" * 60)
    print(f"Python : {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print(f"Root   : {ROOT}")
    print(f"Chapters: {len(Handler.catalog)}  subjects: {n_subj}")
    print(f"Open   : http://{HOST}:{PORT}/")
    print("Click a subject to launch it with this Python.")
    print("Closing the browser does NOT stop this server.")
    print("Press Ctrl+C here to stop the dashboard and get the prompt back.")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping dashboard...")
        with _lock:
            _kill_current()
        # shutdown() can deadlock if called from the serve_forever thread;
        # after KeyboardInterrupt that loop has already exited.
        try:
            server.server_close()
        except Exception:
            pass
        print("Dashboard stopped. You can type the next command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
