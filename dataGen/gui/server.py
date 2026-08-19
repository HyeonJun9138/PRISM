"""Stdlib-only HTTP server for the PRISM dataGen GUI.

Uses :class:`http.server.ThreadingHTTPServer` so that a long-running
Server-Sent-Events stream on one connection cannot block the REST calls made by
the rest of the page. Long batch generation never runs on a request thread --
it is handed to a worker thread through :class:`gui.api.Api`.

Why SSE and not WebSockets: the Python standard library has no server-side
WebSocket implementation, and the project may only import numpy + stdlib.
``text/event-stream`` over plain HTTP/1.1 needs nothing extra and the browser
reconnects automatically.

Security posture: the server binds 127.0.0.1 only, refuses path traversal in
both the static handler and the run-name handler, and echoes no user input into
HTML.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
import queue
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from gui.api import Api, ApiError, STATIC_ROOT, to_jsonable

__all__ = ["PrismHTTPServer", "make_server", "serve", "main", "DEFAULT_PORT"]

#: Preferred TCP port; the next free one is used when it is taken.
DEFAULT_PORT: int = 8770

#: Loopback only. Never expose this server on a routable interface.
BIND_HOST: str = "127.0.0.1"

#: Extra MIME types (Windows registry sometimes lacks these).
_MIME: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
}


class PrismHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server carrying the shared :class:`~gui.api.Api`."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int], handler: type, api: Api) -> None:
        """Create the server.

        Args:
            addr: ``(host, port)`` to bind.
            handler: request handler class.
            api: the shared API implementation.
        """
        super().__init__(addr, handler)
        self.api = api
        self.started_at = time.time()


class Handler(BaseHTTPRequestHandler):
    """Routes ``/api/*`` to :class:`~gui.api.Api` and everything else to files."""

    server_version = "PRISM-dataGen/1.0"
    protocol_version = "HTTP/1.1"

    # -- logging ----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quiet by default; set ``PRISM_GUI_VERBOSE=1`` to see every request."""
        if os.environ.get("PRISM_GUI_VERBOSE") == "1":
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- low-level writers -------------------------------------------------

    def _send_bytes(self, payload: bytes, ctype: str, status: int = 200,
                    extra: Optional[dict[str, str]] = None) -> None:
        """Send a complete response with a fixed body."""
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(self, obj: Any, status: int = 200) -> None:
        """Serialise ``obj`` (NaN-safe) and send it as ``application/json``."""
        body = json.dumps(to_jsonable(obj), allow_nan=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_error_json(self, exc: BaseException) -> None:
        """Send a clean error envelope for any exception."""
        if isinstance(exc, ApiError):
            self._send_json(exc.envelope(), status=exc.status)
            return
        payload = {
            "ok": False,
            "error": {
                "code": "internal_error",
                "message": f"{type(exc).__name__}: {exc}",
                "detail": {"traceback": traceback.format_exc(limit=8).splitlines()[-8:]},
            },
        }
        self._send_json(payload, status=500)

    def _read_body(self) -> dict[str, Any]:
        """Read and JSON-parse the request body (``{}`` when empty)."""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            return {}
        if n > 8 * 1024 * 1024:
            raise ApiError("request body too large", status=413,
                           code="body_too_large")
        raw = self.rfile.read(n)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ApiError(f"request body is not valid JSON: {exc}",
                           code="bad_json") from exc
        if not isinstance(obj, dict):
            raise ApiError("request body must be a JSON object", code="bad_json")
        return obj

    # -- HTTP verbs --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        """Handle GET: API routes first, then static files."""
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path.startswith("/api/"):
                if path == "/api/sim/stream":
                    if self.command == "HEAD":
                        raise ApiError("HEAD is not supported on the SSE stream",
                                       status=405, code="method_not_allowed")
                    self._sse_stream(parse_qs(parsed.query))
                    return
                self._send_json(self._route_get(path, parse_qs(parsed.query)))
                return
            self._serve_static(path)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except BaseException as exc:  # noqa: BLE001 - always answer with JSON
            try:
                self._send_error_json(exc)
            except Exception:
                pass

    def do_HEAD(self) -> None:  # noqa: N802
        """HEAD is handled exactly like GET minus the body."""
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST API routes."""
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if not path.startswith("/api/"):
                raise ApiError(f"no POST route '{path}'", status=404,
                               code="not_found")
            body = self._read_body()
            self._send_json(self._route_post(path, body))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except BaseException as exc:  # noqa: BLE001
            try:
                self._send_error_json(exc)
            except Exception:
                pass

    # -- routing -----------------------------------------------------------

    def _route_get(self, path: str, qs: dict[str, list[str]]) -> Any:
        """Dispatch a GET API route.

        Args:
            path: URL path, already unquoted.
            qs: parsed query string.

        Returns:
            The JSON-serialisable response body.
        """
        api: Api = self.server.api  # type: ignore[attr-defined]
        parts = [p for p in path.strip("/").split("/") if p]  # ['api', ...]

        if path == "/api/meta":
            return api.meta()
        if path == "/api/presets":
            return api.presets()
        if path == "/api/config/defaults":
            return api.config_defaults()
        if path == "/api/health":
            return {"ok": True, "uptime_s": time.time() -
                    self.server.started_at}  # type: ignore[attr-defined]
        if path == "/api/sim/agent":
            sim_id = (qs.get("sim_id") or [""])[0]
            raw_id = (qs.get("id") or qs.get("agent_id") or [""])[0]
            try:
                agent_id = int(raw_id)
            except ValueError:
                raise ApiError("query parameter 'id' must be an integer",
                               code="bad_agent_id") from None
            return api.sim_agent(sim_id, agent_id)
        if path == "/api/runs":
            return api.runs()
        if path == "/api/jobs":
            return api.jobs()
        if len(parts) == 3 and parts[1] == "jobs":
            return api.job(parts[2])
        if len(parts) == 4 and parts[1] == "runs" and parts[3] == "index":
            return api.run_index(parts[2])
        if len(parts) == 5 and parts[1] == "runs" and parts[3] == "episode":
            try:
                idx = int(parts[4])
            except ValueError:
                raise ApiError("episode index must be an integer",
                               code="bad_episode_index") from None
            return api.run_episode(parts[2], idx)
        raise ApiError(f"no GET route '{path}'", status=404, code="not_found")

    def _route_post(self, path: str, body: dict[str, Any]) -> Any:
        """Dispatch a POST API route."""
        api: Api = self.server.api  # type: ignore[attr-defined]
        parts = [p for p in path.strip("/").split("/") if p]

        if path == "/api/sim/new":
            return api.sim_new(body)
        if path == "/api/sim/control":
            return api.sim_control(body)
        if path == "/api/rollout":
            return api.rollout(body)
        if path == "/api/generate":
            return api.generate(body)
        if path == "/api/diagnose":
            return api.diagnose(body)
        if len(parts) == 4 and parts[1] == "jobs" and parts[3] == "cancel":
            return api.job_cancel(parts[2])
        raise ApiError(f"no POST route '{path}'", status=404, code="not_found")

    # -- Server-Sent Events -------------------------------------------------

    def _sse_stream(self, qs: dict[str, list[str]]) -> None:
        """Stream live simulation frames as ``text/event-stream``.

        Frames are pushed by the session's driver thread; this handler only
        drains a bounded queue. A comment heartbeat every few seconds keeps
        proxies and the browser's idle timer happy, and lets the loop notice a
        disconnected client promptly.
        """
        api: Api = self.server.api  # type: ignore[attr-defined]
        sim_id = (qs.get("sim_id") or [""])[0]
        session = api.sim_session(sim_id)  # raises ApiError -> JSON 404
        q = session.subscribe()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            self._chunk(b"retry: 1500\n\n")
            last_beat = time.time()
            while not session.closed:
                try:
                    frame = q.get(timeout=1.0)
                except queue.Empty:
                    if time.time() - last_beat > 3.0:
                        self._chunk(b": ping\n\n")
                        last_beat = time.time()
                    continue
                payload = json.dumps(to_jsonable(frame), allow_nan=False)
                self._chunk(b"event: frame\ndata: " + payload.encode("utf-8") + b"\n\n")
                last_beat = time.time()
            self._chunk(b"event: gone\ndata: {}\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True
        finally:
            session.unsubscribe(q)

    def _chunk(self, payload: bytes) -> None:
        """Write one HTTP/1.1 chunked-transfer body chunk and flush it."""
        self.wfile.write(b"%X\r\n" % len(payload))
        self.wfile.write(payload)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    # -- static files ------------------------------------------------------

    def _serve_static(self, path: str) -> None:
        """Serve a file from ``gui/static`` with traversal protection."""
        if path in ("/", ""):
            path = "/index.html"
        clean = posixpath.normpath(path).lstrip("/")
        if clean in (".", "..") or clean.startswith("../") or ".." in clean.split("/"):
            raise ApiError("path traversal rejected", status=403, code="path_escape")
        root = os.path.abspath(STATIC_ROOT)
        full = os.path.abspath(os.path.join(root, clean.replace("/", os.sep)))
        if os.path.commonpath([root, full]) != root:
            raise ApiError("path traversal rejected", status=403, code="path_escape")
        if not os.path.isfile(full):
            body = (f"404 - {path} not found under {STATIC_ROOT}").encode("utf-8")
            self._send_bytes(body, "text/plain; charset=utf-8", status=404)
            return
        ext = os.path.splitext(full)[1].lower()
        ctype = _MIME.get(ext) or mimetypes.guess_type(full)[0] or "application/octet-stream"
        with open(full, "rb") as fh:
            data = fh.read()
        self._send_bytes(data, ctype)


# --------------------------------------------------------------------------
# Server construction
# --------------------------------------------------------------------------


def _port_free(host: str, port: int) -> bool:
    """Return True if ``(host, port)`` can currently be bound."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def make_server(port: int = DEFAULT_PORT, host: str = BIND_HOST,
                api: Optional[Api] = None) -> PrismHTTPServer:
    """Create the HTTP server, picking a free port if the preferred one is busy.

    Args:
        port: preferred TCP port [-]; 0 asks the OS for any free port.
        host: bind address; keep the 127.0.0.1 default.
        api: an existing API instance (a fresh one is created when omitted).

    Returns:
        A bound, not-yet-serving :class:`PrismHTTPServer`.

    Raises:
        OSError: if no port in the search window could be bound.
    """
    api = api or Api()
    candidates = [port] if port == 0 else [port] + list(range(port + 1, port + 40))
    last: Optional[OSError] = None
    for p in candidates:
        if p != 0 and not _port_free(host, p):
            continue
        try:
            return PrismHTTPServer((host, p), Handler, api)
        except OSError as exc:
            last = exc
    try:
        return PrismHTTPServer((host, 0), Handler, api)
    except OSError as exc:
        raise (last or exc)


def serve(port: int = DEFAULT_PORT, open_browser: bool = False,
          host: str = BIND_HOST) -> PrismHTTPServer:
    """Start the GUI server on a background thread and print the real URL.

    Args:
        port: preferred TCP port [-].
        open_browser: launch the default browser at the served URL.
        host: bind address.

    Returns:
        The running server; call ``.shutdown()`` to stop it.
    """
    httpd = make_server(port=port, host=host)
    real_port = httpd.server_address[1]
    url = f"http://{host}:{real_port}/"
    th = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.2},
                          name="prism-gui-http", daemon=True)
    th.start()
    print(f"PRISM dataGen GUI serving {STATIC_ROOT}")
    print(f"  -> {url}")
    if port not in (0, real_port):
        print(f"  (port {port} was busy; using {real_port})")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return httpd


def main(argv: Optional[list[str]] = None) -> int:
    """Command-line entry point: ``python -m gui.server [--port N] [--open]``.

    Args:
        argv: argument vector (``sys.argv[1:]`` when omitted).

    Returns:
        Process exit code [-].
    """
    ap = argparse.ArgumentParser(description="PRISM dataGen GUI server")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"preferred TCP port (default {DEFAULT_PORT})")
    ap.add_argument("--host", type=str, default=BIND_HOST,
                    help="bind address (loopback only by design)")
    ap.add_argument("--open", action="store_true", help="open a browser window")
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("refusing to bind a non-loopback address", file=sys.stderr)
        return 2
    httpd = serve(port=args.port, open_browser=args.open, host=args.host)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        try:
            httpd.api.shutdown()
        finally:
            httpd.shutdown()
            httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual launch
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    raise SystemExit(main())
