"""Stdlib HTTP server for the correspondence labeller — mirrors interesting_spots/server.py.

    GET  /                        the page
    GET  /api/index               pair list + where to open
    GET  /api/pair?i=<n>          one pair's metadata + its current links/misses
    GET  /api/next-undone?i=<n>   next pair not marked done
    GET  /photo?sid=              the purple base image
    GET  /overlay?sid=&spot=&kind=&idx=   RGBA overlay for one spot, coloured by role
    POST /api/spot-at {sid,x,y}   resolve a click to a spot_id (no state change)
    POST /api/link  {key,a,b}     record a correspondence
    POST /api/miss  {key,spot}    toggle 'visible in B but never extracted'
    POST /api/undo  {key}         drop the last link (or miss)
    POST /api/done  {key,done}    mark the pair finished
"""
from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .app import CorrespondenceApp
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

    def __init__(self, addr, handler, app: CorrespondenceApp):
        super().__init__(addr, handler)
        self.app = app


class _Handler(BaseHTTPRequestHandler):
    server_version = "SpotCorrespondence/1.0"

    def log_message(self, fmt, *args):  # noqa: A003
        pass

    @property
    def app(self) -> CorrespondenceApp:
        return self.server.app  # type: ignore[attr-defined]

    def _send(self, code, body, ctype, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _err(self, code, msg):
        self._json({"error": msg}, code)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        try:
            if parsed.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
            elif parsed.path == "/api/index":
                self._json(self.app.index_payload())
            elif parsed.path == "/api/pair":
                i = int(q.get("i", ["0"])[0])
                if not (0 <= i < self.app.total):
                    self._err(404, f"index {i} out of range")
                else:
                    self._json(self.app.pair_meta(i))
            elif parsed.path == "/api/next-undone":
                self._json({"index": self.app.next_undone(int(q.get("i", ["0"])[0]))})
            elif parsed.path == "/photo":
                path = self.app.photo_path(q.get("sid", [""])[0])
                if path is None or not path.is_file():
                    self._err(404, "no photo")
                else:
                    self._send(200, path.read_bytes(),
                               _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                               cache=True)
            elif parsed.path == "/overlay":
                png = self.app.overlay_png(q.get("sid", [""])[0], int(q.get("spot", ["0"])[0]),
                                           q.get("kind", ["link"])[0],
                                           int(q.get("idx", ["0"])[0]))
                if png is None:
                    self._err(404, "no such spot")
                else:
                    self._send(200, png, "image/png", cache=True)
            else:
                self._err(404, "not found")
        except Exception as exc:
            self._err(500, str(exc))

    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length", 0))
            b = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/api/spot-at":
                self._json({"spotId": self.app.spot_at(str(b["sid"]), int(b["x"]), int(b["y"]))})
            elif parsed.path == "/api/link":
                self._json(self.app.add_link(str(b["key"]), int(b["a"]), int(b["b"])))
            elif parsed.path == "/api/miss":
                self._json(self.app.toggle_miss(str(b["key"]), int(b["spot"])))
            elif parsed.path == "/api/undo":
                self._json(self.app.undo(str(b["key"])))
            elif parsed.path == "/api/done":
                self._json(self.app.set_done(str(b["key"]), bool(b.get("done", True))))
            else:
                self._err(404, "not found")
        except Exception as exc:
            self._err(400, str(exc))


def run(app: CorrespondenceApp, host="127.0.0.1", port=8772, open_browser=True) -> None:
    httpd = _Server((host, port), _Handler, app)
    url = f"http://{host}:{port}/"
    done = sum(1 for r in app.store.values() if r["done"])
    links = sum(len(r["links"]) for r in app.store.values())
    logger.info(f"spot-correspondence  ->  {url}")
    logger.info(f"  folder : {app.input_dir}")
    logger.info(f"  store  : {app.store_path}")
    logger.info(f"  pairs  : {app.total}  ({done} done, {links} links so far)")
    logger.info("  keys   : click LEFT then RIGHT to link · M miss-mode · U undo · Enter done+next")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopping…")
    finally:
        httpd.shutdown()
        app.close()
