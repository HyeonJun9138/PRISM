"""Stdlib-only HTTP server for the training GUI.

* static files from gui/static at /
* JSON API under /api/ routed to gui.api
* Server-Sent Events at /api/train/stream (live metrics/logs/progress)
* binds 127.0.0.1 only; picks the next free port if the default is taken
"""
from __future__ import annotations

import json
import mimetypes
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from common.constants import GUI_HOST, GUI_PORT
from gui import api

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence per-request console spam
        pass

    # ------------------------------------------------------------- plumbing
    def _send_json(self, code: int, obj) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, path: str) -> None:
        rel = path.lstrip("/") or "index.html"
        full = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not full.startswith(os.path.normpath(STATIC_DIR)) or not os.path.isfile(full):
            self._send_json(404, {"error": f"not found: {path}"})
            return
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text/")
                                                  or "javascript" in ctype else ""))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -------------------------------------------------------------- methods
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/api/train/stream":
            self._stream(q)
            return
        if u.path.startswith("/api/"):
            code, obj = api.handle("GET", u.path, q, None)
            self._send_json(code, obj)
            return
        self._send_static(u.path)

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        code, obj = api.handle("POST", u.path, {}, body)
        self._send_json(code, obj)

    # ------------------------------------------------------------------ SSE
    def _stream(self, q: dict) -> None:
        job_id = q.get("job_id", "")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for event, payload in api.stream_events(job_id):
                chunk = f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            return


def _free_port(host: str, start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket() as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError("no free port found")


def serve(open_browser: bool = True, port: int | None = None) -> None:
    port = port or _free_port(GUI_HOST, GUI_PORT)
    httpd = ThreadingHTTPServer((GUI_HOST, port), Handler)
    httpd.daemon_threads = True
    url = f"http://{GUI_HOST}:{port}"
    print(f"PRISM model training GUI: {url}")
    if open_browser:
        import webbrowser
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("shutting down")
        httpd.shutdown()
