"""
_dashboard.py

Local browser dashboard for launching lecture slide decks during class.

Usage (on the lecture machine):
  1. Activate your Python env in a terminal (conda/venv/etc.).
  2. cd to this repo root.
  3. python _dashboard.py
  4. Open the printed URL (default http://127.0.0.1:8765).
  5. Click a .py subject -- it launches with *this same* Python (sys.executable).
     Click a .ipynb subject -- opens it in VS Code with the Explorer panel hidden.

The dashboard inherits whatever interpreter is already active in the terminal.
It only serves http://127.0.0.1 and launches local slide / notebook files.

Dependency invariant:
  Catalog discovery is filesystem + AST only (no curriculum maps).
  Runtime lecture code depends on slide files + _config.py only.
  This dashboard itself is stdlib-only.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765
FILE_RE = re.compile(r"^(\d+)_(\d+)_(.+)\.(py|ipynb)$")
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
    """Return chapters with title + subjects (sorted by number, .py before .ipynb)."""
    chapters: dict[int, dict] = {}
    for path in sorted(ROOT.iterdir()):
        m = FILE_RE.match(path.name)
        if not m or not path.is_file():
            continue
        ch, num, stem, ext = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        bucket = chapters.setdefault(
            ch,
            {"chapter": ch, "title": f"Chapter {ch}", "subjects": []},
        )
        if num == 0 and ext == "py":
            bucket["title"] = pretty_stem(stem)
            bucket["opener"] = path.name
            continue
        if num == 0:
            # Non-.py openers (unexpected) still appear as subjects.
            pass
        bucket["subjects"].append(
            {
                "num": num,
                "subject": pretty_stem(stem),
                "file": path.name,
                "ext": ext,
            }
        )
    out = [chapters[k] for k in sorted(chapters)]
    for ch in out:
        # Numbered place: by lecture number, then stem, then .py before .ipynb.
        ch["subjects"].sort(
            key=lambda s: (s["num"], s["subject"].lower(), 0 if s["ext"] == "py" else 1)
        )
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


def _vscode_command() -> list[str] | None:
    """Return argv prefix that opens a file in VS Code, or None if not found."""
    code = shutil.which("code") or shutil.which("code.cmd")
    if code:
        return [code]
    # Common Windows install location when PATH is incomplete.
    if sys.platform == "win32":
        candidate = (
            Path.home()
            / "AppData"
            / "Local"
            / "Programs"
            / "Microsoft VS Code"
            / "bin"
            / "code.cmd"
        )
        if candidate.is_file():
            return [str(candidate)]
    return None


def _spawn_detached(argv: list[str]) -> None:
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    subprocess.Popen(
        argv,
        cwd=str(ROOT),
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


def _close_vscode_sidebar() -> None:
    """Close the primary side bar (Explorer). Safe to call repeatedly."""
    uri = "vscode://command/workbench.action.closeSidebar"
    try:
        if sys.platform == "win32" and hasattr(os, "startfile"):
            os.startfile(uri)  # type: ignore[attr-defined]
            return
        if sys.platform == "darwin":
            _spawn_detached(["open", uri])
            return
        _spawn_detached(["xdg-open", uri])
    except OSError:
        if sys.platform == "win32":
            _spawn_detached(["cmd", "/c", "start", "", uri])


def _hide_explorer_for_notebook(notebook_filename: str) -> None:
    """Focus the notebook's VS Code window, then close the Explorer side bar."""
    if sys.platform == "win32":
        needle = notebook_filename.replace("'", "''")
        # Use the Command Palette close command (not Ctrl+B toggle).
        ps = f"""
$ErrorActionPreference = 'SilentlyContinue'
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class VscUi {{
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr hWnd);
}}
"@
$deadline = (Get-Date).AddSeconds(25)
$hwnd = [IntPtr]::Zero
while ((Get-Date) -lt $deadline) {{
  $matches = Get-Process -Name 'Code','Code - Insiders' |
    Where-Object {{ $_.MainWindowHandle -ne 0 -and $_.MainWindowTitle -like '*{needle}*' }} |
    Sort-Object StartTime -Descending
  if ($matches) {{
    $hwnd = $matches[0].MainWindowHandle
    break
  }}
  Start-Sleep -Milliseconds 200
}}
if ($hwnd -eq [IntPtr]::Zero) {{
  $fallback = Get-Process -Name 'Code','Code - Insiders' |
    Where-Object {{ $_.MainWindowHandle -ne 0 }} |
    Sort-Object StartTime -Descending |
    Select-Object -First 1
  if ($fallback) {{ $hwnd = $fallback.MainWindowHandle }}
}}
if ($hwnd -eq [IntPtr]::Zero) {{ exit 0 }}
if ([VscUi]::IsIconic($hwnd)) {{ [void][VscUi]::ShowWindow($hwnd, 9) }}
[void][VscUi]::SetForegroundWindow($hwnd)
Start-Sleep -Milliseconds 400
# Close Primary Side Bar via Command Palette (idempotent; not a toggle).
[System.Windows.Forms.SendKeys]::SendWait('^+p')
Start-Sleep -Milliseconds 450
[System.Windows.Forms.SendKeys]::SendWait('View: Close Primary Side Bar')
Start-Sleep -Milliseconds 350
[System.Windows.Forms.SendKeys]::SendWait('{{ENTER}}')
Start-Sleep -Milliseconds 300
Start-Process 'vscode://command/workbench.action.closeSidebar'
"""
        _spawn_detached(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps,
            ]
        )
        return

    for _ in range(3):
        _close_vscode_sidebar()


def open_in_vscode(filename: str) -> dict:
    """Open a notebook in VS Code and hide the Explorer panel."""
    path = ROOT / filename
    m = FILE_RE.match(filename)
    if not path.is_file() or not m or m.group(4) != "ipynb":
        return {"ok": False, "error": f"Unknown notebook file: {filename}"}

    cmd = _vscode_command()
    if cmd is None:
        return {
            "ok": False,
            "error": "VS Code CLI not found (expected `code` on PATH).",
        }

    try:
        # File-only open (no folder) so Explorer has no project tree.
        _spawn_detached([*cmd, "-n", str(path.resolve())])
    except OSError as e:
        return {"ok": False, "error": f"Failed to open VS Code: {e}"}

    threading.Timer(1.2, lambda: _hide_explorer_for_notebook(filename)).start()
    threading.Timer(3.5, lambda: _hide_explorer_for_notebook(filename)).start()

    write_last_file(filename)
    return {"ok": True, "last": filename, "action": "vscode"}


def launch_slide(filename: str) -> dict:
    global _proc
    path = ROOT / filename
    m = FILE_RE.match(filename)
    if not path.is_file() or not m or m.group(4) != "py":
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
        return {"ok": True, "last": filename, "action": "python"}


def launch_file(filename: str) -> dict:
    m = FILE_RE.match(filename or "")
    if not m:
        return {"ok": False, "error": f"Unknown file: {filename}"}
    if m.group(4) == "ipynb":
        return open_in_vscode(filename)
    return launch_slide(filename)


def render_page(catalog: list[dict], last_file: str | None = None) -> str:
    last_file = last_file or read_last_file()
    chapters_html = []
    for ch in catalog:
        items = []
        for s in ch["subjects"]:
            last_cls = " last" if s["file"] == last_file else ""
            kind_cls = " notebook" if s["ext"] == "ipynb" else " slide"
            if s["ext"] == "ipynb":
                label = (
                    f'<span class="sub-name">{html.escape(s["subject"])}</span>'
                    f'<span class="sub-badge">.ipynb</span>'
                )
            else:
                label = f'<span class="sub-name">{html.escape(s["subject"])}</span>'
            items.append(
                f'<button type="button" class="subject{kind_cls}{last_cls}" '
                f'data-file="{html.escape(s["file"])}" data-ext="{html.escape(s["ext"])}">'
                f"{label}"
                f"</button>"
            )
        opener = ch.get("opener")
        n_subj = len(ch["subjects"])
        if opener:
            last_cls = " last" if opener == last_file else ""
            title_html = (
                f'<button type="button" class="chapter-title{last_cls}" '
                f'data-file="{html.escape(opener)}" data-ext="py" '
                f'title="Open chapter title slide">'
                f'<span class="ch-name">{html.escape(ch["title"])}</span>'
                f'<span class="ch-count">{n_subj} subject{"s" if n_subj != 1 else ""}</span>'
                f"</button>"
            )
        else:
            title_html = (
                f'<h2 class="chapter-title">'
                f'<span class="ch-name">{html.escape(ch["title"])}</span>'
                f'<span class="ch-count">{n_subj} subject{"s" if n_subj != 1 else ""}</span>'
                f"</h2>"
            )
        chapters_html.append(
            f'<section class="chapter">'
            f"{title_html}"
            f'<div class="grid">{"".join(items)}</div>'
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
      width: 100%;
      margin: 0 0 12px; padding: 12px 14px;
      border-radius: 12px;
      background: linear-gradient(90deg, #1a3358 0%, #152033 55%, #121a27 100%);
      border: 1px solid #2f4d75;
      border-left: 5px solid var(--accent);
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.22);
      color: inherit; font: inherit; text-align: left;
    }}
    button.chapter-title {{
      appearance: none; -webkit-appearance: none;
      cursor: pointer;
    }}
    button.chapter-title:hover {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent), 0 6px 18px rgba(0, 0, 0, 0.22);
      transform: translateY(-1px);
    }}
    button.chapter-title.last {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px var(--accent), 0 6px 18px rgba(0, 0, 0, 0.22);
      background: linear-gradient(90deg, #244a7a 0%, #1a3358 55%, #152033 100%);
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
    button.subject.notebook {{
      border-color: #4a6a3a;
      background: #1a2a1c;
    }}
    button.subject.notebook:hover {{
      border-color: #7dce57 !important;
      box-shadow: 0 0 0 2px #7dce57;
    }}
    button.subject.notebook.last {{
      border-color: #7dce57 !important;
      box-shadow: 0 0 0 2px #7dce57 !important;
      background: #243b28 !important;
    }}
    .sub-name {{ font-weight: 650; font-size: 0.95rem; }}
    .sub-badge {{
      display: inline-block;
      margin-left: 8px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.02em;
      color: #061018;
      background: #7dce57;
      vertical-align: middle;
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Lecture Dashboard</h1>
    <div class="last-line">Last opened: <strong id="last-label">{last_label}</strong> <span style="color:var(--muted)">( .py → Python · .ipynb → VS Code, Explorer hidden )</span></div>
  </header>
  <main>
    {"".join(chapters_html)}
  </main>
  <script>
    function markLast(file) {{
      var buttons = document.querySelectorAll("button.subject, button.chapter-title");
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

    function launchFile(file) {{
      if (!file) return;
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
    }}

    document.addEventListener("click", function (ev) {{
      var btn = ev.target.closest("button.subject, button.chapter-title");
      if (!btn) return;
      launchFile(btn.getAttribute("data-file"));
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
            result = launch_file(filename)
            code = 200 if result.get("ok") else 400
            self._send(code, json.dumps(result).encode("utf-8"), "application/json")
            return
        self._send(404, b"not found", "text/plain")


def main() -> int:
    Handler.catalog = discover_catalog()
    n_subj = sum(len(ch["subjects"]) for ch in Handler.catalog)
    n_nb = sum(
        1
        for ch in Handler.catalog
        for s in ch["subjects"]
        if s.get("ext") == "ipynb"
    )
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("=" * 60)
    print("Lecture Dashboard")
    print("=" * 60)
    print(f"Python : {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    print(f"Root   : {ROOT}")
    print(f"Chapters: {len(Handler.catalog)}  subjects: {n_subj}  notebooks: {n_nb}")
    print(f"Open   : http://{HOST}:{PORT}/")
    print("Click a .py subject to launch it with this Python.")
    print("Click a .ipynb subject to open it in VS Code (Explorer hidden).")
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
