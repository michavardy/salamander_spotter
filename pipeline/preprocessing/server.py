"""Stdlib HTTP server for the preprocessing review app — no web framework, no new deps.

The page owns the logic; this layer hands over data and writes the file. Endpoints:

    GET  /                            the single page
    GET  /api/bootstrap               review.json + the individual index + config
    GET  /api/individual?label=ac_3   one family: geometry, quality, machine matches
    GET  /photo?sid=<sid>             the source photo bytes (cached hard — they never change)
    GET  /api/jobs                    regeneration jobs + their streamed logs
    POST /api/edit                    {rev, individual, record, ui?, ...} -> merge + atomic save
    POST /api/export                  write the legacy interesting_spots.json + reprocess CSVs
    POST /api/interest                start the one-off distinctiveness build
    POST /api/generate                {sid, n, mock?} -> regenerate synthetic views (BILLED)

A ``rev`` mismatch on ``/api/edit`` returns **409** with the current store, so a second tab can
resync instead of silently overwriting hand-labelled work.
"""
from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .app import ReviewApp
from .page import INDEX_HTML

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

_CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                  ".webp": "image/webp", ".bmp": "image/bmp", ".tif": "image/tiff",
                  ".tiff": "image/tiff"}


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, app: ReviewApp):
        super().__init__(addr, handler)
        self.app = app


class _Handler(BaseHTTPRequestHandler):
    server_version = "PreprocessingReview/1.0"

    def log_message(self, fmt, *args):  # noqa: A003 — keep the console quiet
        pass

    @property
    def app(self) -> ReviewApp:
        return self.server.app  # type: ignore[attr-defined]

    # ---- send helpers -------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json")

    def _err(self, code: int, msg: str):
        self._json({"error": msg}, code)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    # ---- routing ------------------------------------------------------------

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path, q = parsed.path, parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif path == "/api/bootstrap":
                self._json(self.app.bootstrap_payload())
            elif path == "/api/individual":
                self._individual(q)
            elif path == "/photo":
                self._photo(q)
            elif path == "/api/jobs":
                self._json({"jobs": self.app.jobs.jobs()})
            elif path == "/review.json":
                self._json(self.app.review)
            else:
                self._err(404, "not found")
        except KeyError as exc:
            self._err(404, str(exc))
        except Exception as exc:                       # one bad request never stops serving
            self._err(500, f"{type(exc).__name__}: {exc}")

    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/edit":
                body = self._body()
                result = self.app.apply_edit(int(body.get("rev", -1)), body)
                self._json(result, 409 if result.get("conflict") else 200)
            elif path == "/api/export":
                self._json(self.app.export())
            elif path == "/api/interest":
                self.app.ensure_interest()
                self._json(self.app.interest_status())
            elif path == "/api/generate":
                body = self._body()
                sid = str(body.get("sid") or "")
                if not sid:
                    self._err(400, "sid required")
                    return
                job = self.app.jobs.submit(sid, int(body.get("n", 1)),
                                           mock=bool(body.get("mock")))
                self._json(job)
            else:
                self._err(404, "not found")
        except KeyError as exc:
            self._err(404, str(exc))
        except Exception as exc:
            self._err(400, f"{type(exc).__name__}: {exc}")

    # ---- endpoint bodies ----------------------------------------------------

    def _individual(self, q):
        label = (q.get("label") or [""])[0]
        if not label:
            self._err(400, "label required")
            return

        def _opt(name, cast):
            raw = (q.get(name) or [None])[0]
            return None if raw in (None, "") else cast(raw)

        self._json(self.app.individual_payload(
            label,
            matcher=_opt("matcher", str),
            top_n=_opt("top_n", int),
            cutoff=_opt("cutoff", float),
        ))

    def _photo(self, q):
        sid = (q.get("sid") or [""])[0]
        path = self.app.photo_path(sid)
        if path is None or not path.is_file():
            self._err(404, f"no photo for {sid!r}")
            return
        ctype = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        self._send(200, path.read_bytes(), ctype, cache=True)


def run(app: ReviewApp, host: str = "127.0.0.1", port: int = 8770,
        open_browser: bool = True, build_interest: bool = True) -> None:
    """Serve until Ctrl-C. Opens the browser once the socket is listening."""
    httpd = _Server((host, port), _Handler, app)
    url = f"http://{host}:{port}/"
    logger.info(f"preprocessing-review  ->  {url}")
    logger.info(f"  dataset     : {app.dataset}")
    logger.info(f"  db          : {app.db_path}")
    logger.info(f"  store       : {app.store_path}")
    logger.info(f"  individuals : {app.n_individuals}  ({app.n_images} images)")
    logger.info(f"  resume at   : {app.resume_label()}")
    logger.info("  keys        : ←/→ individual · U next unreviewed · A/R accept/reject · "
          "T train · M miss · Esc clear. Ctrl-C to stop.")
    if build_interest:
        app.ensure_interest()
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopping…")
    finally:
        httpd.shutdown()
        app.close()
