"""The HTTP core: stdlib ThreadingHTTPServer, JSON under /api, the static web/ tree at /, sign-in links and auth.

Routes are `(method, "/api/area/{param}", handler, auth)` from the modules a server is started with. auth is
`public`, `person` (cookie `tw` or `Authorization: Bearer`, a people.token), `operator` (the admin token, or a
founder/staff person) or `admin`. Request logs never carry tokens or query strings. Imports only store and errors,
so the workspace can be served on its own (`tw workspace`).
"""

from __future__ import annotations

import hmac
import json
import mimetypes
import os
import re
import secrets
import sqlite3
import sys
import threading
import time
import traceback
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .errors import Refused, TooLarge, lane
from .store import data_dir, event, one, write_private

def _web() -> Path:
    """The static UI: $TW_WEB, else the source tree's web/, else the copy packaged inside the wheel."""
    if os.environ.get("TW_WEB"):
        return Path(os.environ["TW_WEB"])
    src = Path(__file__).resolve().parents[2] / "web"
    return src if src.is_dir() else Path(__file__).resolve().parent / "web"


WEB = _web()
MAX_UPLOAD = int(os.environ.get("TW_MAX_UPLOAD_MB", "512")) * 1024 * 1024
MAX_BODY = 16 * 1024 * 1024  # req.body (JSON, small files); larger uploads stream through req.stream
COOKIE_DAYS = 30
TRUSTED_PROXIES = set((os.environ.get("TW_TRUSTED_PROXIES") or "127.0.0.1,::1").split(","))
INLINE = {"image/png", "image/jpeg", "image/gif", "image/webp", "image/heic", "video/mp4", "video/quicktime",
          "video/webm", "audio/mpeg", "audio/mp4", "audio/wav", "audio/webm"}  # anything else downloads


class Response:
    def __init__(self, status: int, body: bytes, content_type: str = "application/json", headers: dict | None = None):
        self.status, self.body, self.content_type, self.headers = status, body, content_type, headers or {}


class Request:
    """`body` reads the request body once (at most MAX_BODY); an upload handler streams `stream`/`length` instead
    (agency.save_media(org, name, req.stream, ctype, length=req.length))."""

    def __init__(self, method, path, query, headers, stream, length, params, person, admin, client=""):
        self.method, self.path, self.query, self.headers = method, path, query, headers
        self.stream, self.length, self._body = stream, length, None
        self.params, self.person, self.admin, self.client = params, person, admin, client

    @property
    def body(self) -> bytes:
        if self._body is None:
            if self.length > MAX_BODY:
                raise TooLarge("That request is too large.")
            self._body = self.stream.read(self.length) if self.length else b""
            self.length = 0
        return self._body

    def json(self) -> dict:
        try:
            v = json.loads(self.body or b"{}")
        except ValueError:
            raise Refused("The request body is not valid JSON.")
        if not isinstance(v, dict):
            raise Refused("The request body must be a JSON object.")
        return v

    @property
    def org_id(self) -> str:
        return self.person["org_id"]


def is_operator(person: dict | None) -> bool:
    """Who may see and drive the phones: the admin token, or a founder/staff person (never a customer)."""
    return bool(person) and (person["role"] == "founder" or person["cohort"] in ("founder", "staff"))


def admin_token() -> str:
    t = os.environ.get("TW_ADMIN_TOKEN")
    if t:
        return t
    p = data_dir().parent / "admin.token"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_private(p, secrets.token_urlsafe(32), exclusive=True)  # 0600 from creation; O_EXCL refuses a symlink there
    return p.read_text().strip()


def session_cookie(token: str, headers) -> str:
    """The Set-Cookie value that signs a person in (lanes' self-serve sign-up returns it in Response headers)."""
    secure = "; Secure" if headers.get("X-Forwarded-Proto") == "https" else ""
    return f"tw={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={COOKIE_DAYS * 86400}{secure}"




HEALTH = [("GET", "/api/health", lambda req: {"ok": True}, "public")]


def person_by_token(token: str) -> dict | None:
    return one("SELECT * FROM people WHERE token = ?", (token,)) if token else None


def compile_routes(routes: list[tuple]) -> list[tuple]:
    out = []
    for method, pattern, handler, auth in routes:
        rx = re.compile("^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern) + "$")
        out.append((method, rx, handler, auth))
    return out


def module_routes(names) -> list[tuple]:
    out = []
    for name in names:
        mod = lane(name)
        out += list(getattr(mod, "ROUTES", []) if mod else [])
    return out


# ---- the handler ---------------------------------------------------------------------------------------------

def _redact(path: str) -> str:
    return re.sub(r"^/t/[^/?]+", "/t/…", path.split("?", 1)[0])


class Handler(BaseHTTPRequestHandler):
    server_version = "teachable-worker"
    ROUTES: list = []
    timeout = 60  # seconds per socket read: a stalled client frees its thread

    def log_message(self, fmt, *args):  # the default logs the raw request line (tokens in /t/…, queries)
        pass

    def _send(self, r: Response):
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {_redact(self.path)} {r.status}\n")
        self.send_response(r.status)
        self.send_header("Content-Type", r.content_type)
        self.send_header("Content-Length", str(len(r.body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", r.headers.pop("Referrer-Policy", "same-origin"))
        for k, v in r.headers.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(r.body)

    def _proxied(self) -> bool:
        return self.client_address[0] in TRUSTED_PROXIES

    def _client(self) -> str:
        """The caller's IP: X-Forwarded-For only when the direct peer is a trusted (local) proxy."""
        xff = self.headers.get("X-Forwarded-For") if self._proxied() else None
        return (xff.split(",")[-1].strip() if xff else "") or self.client_address[0]

    def _headers(self) -> dict:
        h = {k: v for k, v in self.headers.items()}
        if not self._proxied():
            for k in ("X-Forwarded-For", "X-Forwarded-Proto", "X-Forwarded-Host"):
                h.pop(k, None)
        return _CI(h)

    def _token(self) -> str:
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        c = SimpleCookie(self.headers.get("Cookie") or "")
        return c["tw"].value if "tw" in c else ""

    def _dispatch(self):
        url = urlsplit(self.path)
        path = url.path
        if path.startswith("/t/"):
            return self._send(self._login(path[3:]))
        if not path.startswith("/api/"):
            return self._send(self._static(path))
        token = self._token()
        is_admin = bool(token) and hmac.compare_digest(token, admin_token())
        person = None if is_admin else person_by_token(token)
        allowed = [r for r in self.ROUTES if r[1].match(path)]
        if not allowed:
            return self._send(_err(404, "Not found."))
        match = [r for r in allowed if r[0] == self.command]
        if not match:
            return self._send(_err(405, "Method not allowed."))
        _, rx, handler, auth = match[0]
        if auth == "admin" and not is_admin:
            return self._send(_err(401 if not token else 403, "Operator sign-in required."))
        if auth == "person" and not person and not is_admin:
            return self._send(_err(401, "Sign in with your link."))
        if auth == "person" and is_admin and not person:
            return self._send(_err(403, "This is a customer route; use /api/ops as the operator."))
        if auth == "operator" and not (is_admin or is_operator(person)):
            return self._send(_err(401 if not token else 403, "Only the phone's operator can see and drive it."))
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0:
            return self._send(_err(400, "Bad Content-Length."))
        if n > MAX_UPLOAD:
            return self._send(_err(413, "That file is too large."))
        req = Request(self.command, path, {k: v[0] for k, v in parse_qs(url.query).items()}, self._headers(),
                      self.rfile, n, rx.match(path).groupdict(), person, is_admin, self._client())
        try:
            out = handler(req)
        except Refused as e:
            return self._send(_err(e.status, str(e)))
        except sqlite3.IntegrityError:
            return self._send(_err(409, "That conflicts with an existing record."))
        except Exception:
            traceback.print_exc()
            return self._send(_err(500, "Something went wrong on our side; it has been logged."))
        if isinstance(out, Response):
            return self._send(out)
        status, obj = out if isinstance(out, tuple) else (200, out)
        return self._send(Response(status, json.dumps(obj, default=str).encode()))

    def _login(self, token: str) -> Response:
        ok_admin = hmac.compare_digest(token, admin_token())
        p = None if ok_admin else person_by_token(token)
        if not ok_admin and not p:
            return _err(404, "This link is not valid.", html=True)
        if p:
            event("signed_in", actor="founder" if is_operator(p) else "customer", org_id=p["org_id"], person_id=p["id"])
        return Response(303, b"", "text/plain", {"Set-Cookie": session_cookie(token, self._headers()),
                                                "Location": "/" if ok_admin or is_operator(p) else "/jobs/",
                                                "Referrer-Policy": "no-referrer"})

    def _static(self, path: str) -> Response:
        root = WEB.resolve()
        f = (root / path.lstrip("/")).resolve()
        if not f.is_relative_to(root):
            return _err(404, "Not found.", html=True)
        if f.is_dir():
            if not path.endswith("/"):
                return Response(301, b"", "text/plain", {"Location": path + "/"})
            f = f / "index.html"
        if not f.is_file():
            return _err(404, "Not found.", html=True)
        import mimetypes
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        return Response(200, f.read_bytes(), ctype, {"Cache-Control": "no-cache"})

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = _dispatch


class _CI(dict):
    """Header lookup that ignores case, like http.client's message object."""

    def __init__(self, d):
        super().__init__({k.lower(): v for k, v in d.items()})

    def get(self, k, default=None):
        return super().get(k.lower(), default)

    def __getitem__(self, k):
        return super().__getitem__(k.lower())

    def __contains__(self, k):
        return super().__contains__(k.lower())


def _err(status: int, msg: str, html: bool = False) -> Response:
    if html:
        return Response(status, f"<!doctype html><title>{status}</title><p>{msg}</p>".encode(), "text/html")
    return Response(status, json.dumps({"error": msg}).encode())


def worker(interval: float, stop: threading.Event, ticks) -> None:
    """Each tick, every housekeeping function the server was started with (a failure is logged, never fatal)."""
    while not stop.wait(interval):
        for tick in ticks():
            try:
                tick and tick()
            except Exception:
                traceback.print_exc()


def serve(routes: list[tuple], host: str = "127.0.0.1", port: int = 8780, work_interval: float = 15.0,
          ticks=lambda: (), banner: str = "teachable-worker") -> None:
    Handler.ROUTES = compile_routes(routes)
    admin_token()  # created on first run, 0600
    from . import workspace
    workspace.recover()
    httpd = ThreadingHTTPServer((host, port), Handler)
    stop = threading.Event()
    if work_interval > 0:
        threading.Thread(target=worker, args=(work_interval, stop, ticks), daemon=True).start()
    sys.stderr.write(f"{banner} on http://{host}:{port} (db {os.environ.get('TW_DB') or 'default'}, "
                     f"worker every {work_interval}s)\n")
    try:
        httpd.serve_forever()
    finally:
        stop.set()
        httpd.server_close()
