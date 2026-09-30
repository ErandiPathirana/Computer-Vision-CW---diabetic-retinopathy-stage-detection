"""
Local web server for OcuGrade (standard library only, plus reportlab for PDFs).

    python -m webapp.server          ->  http://localhost:8080

Endpoints
  GET  /                      single-page app (webapp/static)
  GET  /api/health            model status
  POST /api/predict           raw PNG/JPEG bytes in the body -> prediction JSON (images are never saved)
  GET  /api/metrics           models/metrics.json
  GET  /api/assets[/name]     figures and tables exported from Colab (webapp/assets)
  GET  /api/samples[/name]    optional demo images (webapp/samples, kept out of Git)
  POST /api/report            prediction JSON -> one-page PDF
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from src import config as C
from .inference import Predictor

STATIC_DIR = Path(__file__).resolve().parent / "static"
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_JSON_BYTES = 25 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/styles.css": "styles.css", "/app.js": "app.js"}
SAMPLE_EXTS = (".png", ".jpg", ".jpeg")


def make_handler(predictor: Predictor, assets_dir: Path, samples_dir: Path, models_dir: Path):
    """Create a request handler class bound to one predictor and folder set."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "OcuGrade/1.0"

        # ---- helpers ----------------------------------------------------- #
        def log_message(self, fmt, *args):                        # never log bodies or image data
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, status: int, body: bytes, ctype: str, extra: Optional[dict] = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

        def _error(self, status: int, message: str) -> None:
            self._json({"error": message}, status)

        def _file(self, path: Path) -> None:
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype in ("application/json", "application/javascript"):
                ctype += "; charset=utf-8"
            self._send(200, path.read_bytes(), ctype)

        def _read_body(self, limit: int) -> Optional[bytes]:
            """Read the request body safely (size-limited). Sends the error itself and returns None on failure."""
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._error(400, "Invalid Content-Length.")
                return None
            if length <= 0:
                self._error(400, "Empty request body.")
                return None
            if length > limit:
                self._drain(length)
                self._error(413, f"File too large (limit {limit // (1024 * 1024)} MB).")
                return None
            return self.rfile.read(length)

        def _drain(self, length: int, cap: int = 64 * 1024 * 1024) -> None:
            """Read and discard an oversized body (up to a cap) so the browser can receive our 413 message."""
            remaining = min(length, cap)
            while remaining > 0:
                chunk = self.rfile.read(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
            self.close_connection = True

        def _named_file(self, folder: Path, name: str, exts: Optional[tuple] = None) -> None:
            """Serve one file from a folder, only if the name is a plain, existing file name (no path tricks)."""
            if not SAFE_NAME.match(name) or (exts and not name.lower().endswith(exts)):
                return self._error(400, "Invalid file name.")
            path = folder / name
            if not path.is_file():
                return self._error(404, "Not found.")
            self._file(path)

        def _list(self, folder: Path, exts: Optional[tuple] = None) -> None:
            names = sorted(p.name for p in folder.iterdir() if p.is_file() and SAFE_NAME.match(p.name)
                           and (not exts or p.name.lower().endswith(exts))) if folder.exists() else []
            self._json(names)

        # ---- routes ------------------------------------------------------ #
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in STATIC_FILES:
                return self._file(STATIC_DIR / STATIC_FILES[path])
            if path == "/api/health":
                return self._json(predictor.info())
            if path == "/api/metrics":
                f = models_dir / "metrics.json"
                return self._file(f) if f.is_file() else self._error(404, "metrics.json not found. Run the Colab export step.")
            if path == "/api/assets":
                return self._list(assets_dir)
            if path.startswith("/api/assets/"):
                return self._named_file(assets_dir, path[len("/api/assets/"):])
            if path == "/api/samples":
                return self._list(samples_dir, SAMPLE_EXTS)
            if path.startswith("/api/samples/"):
                return self._named_file(samples_dir, path[len("/api/samples/"):], SAMPLE_EXTS)
            self._error(404, "Not found.")

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/predict":
                return self._predict()
            if path == "/api/report":
                return self._report()
            self._error(404, "Not found.")

        def _predict(self) -> None:
            data = self._read_body(MAX_IMAGE_BYTES)
            if data is None:
                return
            if not predictor.loaded:
                return self._error(503, predictor.load_error or "The model is not loaded.")
            try:
                self._json(predictor.predict(data))
            except ValueError as exc:                          # bad upload
                self._error(400, str(exc))
            except Exception as exc:                           # unexpected: explain, keep server alive
                sys.stderr.write(f"Prediction failed: {type(exc).__name__}: {exc}\n")
                self._error(500, "Prediction failed. See the server console for details.")

        def _report(self) -> None:
            data = self._read_body(MAX_JSON_BYTES)
            if data is None:
                return
            try:
                from .report import build_pdf
                pdf = build_pdf(json.loads(data))
            except (ValueError, KeyError, TypeError) as exc:
                return self._error(400, f"Could not build the report: {exc}")
            except ImportError:
                return self._error(500, "reportlab is not installed (pip install reportlab).")
            self._send(200, pdf, "application/pdf", {"Content-Disposition": 'attachment; filename="ocugrade_report.pdf"'})

    return Handler


def create_server(host: str = "127.0.0.1", port: int = 8080, predictor: Optional[Predictor] = None,
                  assets_dir: Path = C.ASSETS_DIR, samples_dir: Path = SAMPLES_DIR,
                  models_dir: Path = C.MODELS_DIR) -> ThreadingHTTPServer:
    predictor = predictor or Predictor()
    return ThreadingHTTPServer((host, port), make_handler(predictor, assets_dir, samples_dir, models_dir))


def main() -> None:
    import os
    parser = argparse.ArgumentParser(description="OcuGrade local web app")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--open", action="store_true", help="open the browser automatically")
    args = parser.parse_args()

    try:
        from . import report  # noqa: F401  (import now so the first PDF request is not slow)
    except ImportError:
        print("NOTE: reportlab is not installed, PDF reports are disabled (pip install reportlab).")
    predictor = Predictor()
    print("Loading model (TensorFlow can take a few seconds)...")
    predictor.load()
    print("Model loaded." if predictor.loaded else f"WARNING: {predictor.load_error}")
    server = create_server(args.host, args.port, predictor)
    url = f"http://localhost:{args.port}"
    print(f"OcuGrade is running at {url}  (press Ctrl+C to stop)")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
