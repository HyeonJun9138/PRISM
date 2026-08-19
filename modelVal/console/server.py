"""Stdlib HTTP server for the validation console (port 8810).

Same pattern as the sibling GUIs: static files at /, JSON API at /api/,
Server-Sent Events for the frame stream. Binds 127.0.0.1 only.
"""
from __future__ import annotations

import json
import mimetypes
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from console import api

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
HOST, DEFAULT_PORT = "127.0.0.1", 8810


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path: str) -> None:
        rel = path.lstrip("/") or "index.html"
        full = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(os.path.normpath(STATIC_DIR)) or not os.path.isfile(full):
            self._json(404, {"error": f"not found: {path}"})
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8"
                         if ctype.startswith("text/") or "javascript" in ctype else ""))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/api/stream":
            self._stream()
            return
        if u.path.startswith("/api/"):
            code, obj = api.handle("GET", u.path,
                                   {k: v[0] for k, v in parse_qs(u.query).items()}, None)
            self._json(code, obj)
            return
        self._static(u.path)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads((self.rfile.read(n) or b"{}").decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
            return
        code, obj = api.handle("POST", u.path, {}, body)
        self._json(code, obj)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        # close-delimited body: without this, an HTTP/1.1 response with
        # neither Content-Length nor chunked encoding violates the spec and
        # strict middleboxes (institutional firewalls, AV web shields) RST
        # the connection -- observed from an external viewer's network.
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            for payload in api.stream_frames():
                self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                                 .encode("utf-8"))
                self.wfile.flush()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            return


class ExclusiveHTTPServer(ThreadingHTTPServer):
    """Exclusive port binding (allow_reuse_address off).

    http.server enables SO_REUSEADDR by default; on Windows that lets a
    second console silently BIND THE SAME PORT and steal a fraction of the
    requests (clicks land on one instance, frames stream from the other).
    With exclusive binding the second bind raises OSError and the port-scan
    loop below does its job instead.
    """

    allow_reuse_address = False


def serve(port: int | None = None, open_browser: bool = True,
          host: str | None = None) -> None:
    bind = host or HOST
    port = port or DEFAULT_PORT
    for p in range(port, port + 20):
        try:
            httpd = ExclusiveHTTPServer((bind, p), Handler)
            port = p
            break
        except OSError:
            continue
    httpd.daemon_threads = True
    url = f"http://{bind}:{port}"
    print(f"PRISM validation console: {url}")
    if open_browser:
        import webbrowser
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()

