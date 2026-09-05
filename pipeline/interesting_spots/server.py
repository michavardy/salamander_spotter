"""Stdlib HTTP server for the interesting-spot selector — no web framework, no new deps.

Endpoints (all tiny JSON except the two image ones):

    GET  /                       the single page
    GET  /api/index              ordered image list + which are labeled + where to open
    GET  /api/image?i=<n>        metadata for image n (sid, dims, current selection, counts)
    GET  /api/next-unlabeled?i=  index of the next unlabeled image after n (or null)
    GET  /photo?sid=<sid>        the base image bytes (the purple / segmentation image)
    GET  /overlay?sid=&spot=     the green RGBA overlay PNG for one selected spot
    POST /api/click  {sid,x,y}   toggle the spot under the click; persists JSON; returns result
"""
from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .app import SpotSelectorApp
from .page import INDEX_HTML

_REPO_ROOT_FOR_LOGGER = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT_FOR_LOGGER) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_LOGGER))

from pipeline.utils.logger_utils import get_logger
logger = get_logger(Path(__file__).stem)

_CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
                  ".webp": "image/webp", ".bmp": "image/bmp"}


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, app: SpotSelectorApp):
        super().__init__(addr, handler)
        self.app = app


class _Handler(BaseHTTPRequestHandler):
    server_version = "InterestingSpots/1.0"

    # keep the console quiet — one line per real request is enough
    def log_message(self, fmt, *args):  # noqa: A003
        pass

    @property
    def app(self) -> SpotSelectorApp:
        return self.server.app  # type: ignore[attr-defined]

    # ---- send helpers -------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str, cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=86400")
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _err(self, code: int, msg: str):
        self._json({"error": msg}, code)

    # ---- routing ------------------------------------------------------------

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        try:
            if path == "/" or path == "/index.html":
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif path == "/api/index":
                self._json(self.app.index_payload())
            elif path == "/api/image":
                self._api_image(q)
            elif path == "/api/next-unlabeled":
                i = int(q.get("i", ["0"])[0])
                self._json({"index": self.app.next_unlabeled(i)})
            elif path == "/photo":
                self._photo(q)
            elif path == "/overlay":
                self._overlay(q)
            else:
                self._err(404, "not found")
        except Exception as exc:  # never take the server down on one bad request
            self._err(500, str(exc))

    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/click":
            self._err(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            sid = str(body["sid"])
            x, y = int(body["x"]), int(body["y"])
            self._json(self.app.toggle(sid, x, y))
        except Exception as exc:
            self._err(400, str(exc))

    # ---- endpoint bodies ----------------------------------------------------

    def _api_image(self, q):
        i = int(q.get("i", ["0"])[0])
        if not (0 <= i < self.app.total):
            self._err(404, f"index {i} out of range")
            return
        self._json(self.app.index_meta(i))

    def _photo(self, q):
        sid = q.get("sid", [""])[0]
        path = self.app.photo_path(sid)
        if path is None or not path.is_file():
            self._err(404, f"no photo for {sid!r}")
            return
        ctype = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        self._send(200, path.read_bytes(), ctype, cache=True)

    def _overlay(self, q):
        sid = q.get("sid", [""])[0]
        try:
            spot = int(q.get("spot", [""])[0])
        except ValueError:
            self._err(400, "bad spot id")
            return
        png = self.app.overlay_png(sid, spot)
        if png is None:
            self._err(404, "no such spot")
            return
        self._send(200, png, "image/png", cache=True)


def run(app: SpotSelectorApp, host: str = "127.0.0.1", port: int = 8765,
        open_browser: bool = True) -> None:
    """Serve until Ctrl-C. Opens the browser once the socket is listening."""
    httpd = _Server((host, port), _Handler, app)
    url = f"http://{host}:{port}/"
    logger.info(f"interesting-spot-selector  ->  {url}")
    logger.info(f"  folder : {app.input_dir}")
    logger.info(f"  labels : {app.labels_path}")
    logger.info(f"  images : {app.total}  ({app.labeled_count} already labeled)")
    logger.info("  keys   : ← / → prev/next, U next-unlabeled. Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopping…")
    finally:
        httpd.shutdown()
        app.close()
