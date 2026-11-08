import threading
from http.server import ThreadingHTTPServer

import pytest


@pytest.fixture(autouse=True)
def tw(tmp_path, monkeypatch):
    """Every test on its own temp DB, data tree and admin token; never the live state."""
    monkeypatch.setenv("TW_DB", str(tmp_path / "tw.sqlite"))
    monkeypatch.setenv("TW_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("TW_ADMIN_TOKEN", "admin-test-token")
    monkeypatch.setenv("TW_LAUNCH_AT", "2000-01-01T00:00:00+00:00")  # tests model external traffic; see test_prelaunch
    monkeypatch.setenv("TW_BRIDGE_URL", "http://127.0.0.1:1")
    from teachable_worker import missions, workspace
    monkeypatch.setattr(workspace, "bridge", workspace.Bridge("http://127.0.0.1:1"))
    def no_private_lease(*args, **kwargs):
        raise RuntimeError("Tests must supply a simulated session or the isolated local lease")

    if missions._site is not None:  # the private deployment's session (authhub, Focus) is unreachable from tests
        monkeypatch.setattr(missions._site, "AUTHHUB_PY", tmp_path / "no-private-authhub-in-tests")
        monkeypatch.setattr(missions._site, "FOCUSGATE", str(tmp_path / "no-private-focusgate-in-tests"))
        monkeypatch.setattr(missions._site, "_lease", no_private_lease)
    return tmp_path


@pytest.fixture
def http(tw):
    """The real server on an ephemeral port; returns a call(method, path, body=None, token=None) helper."""
    import json
    import urllib.error
    import urllib.request

    from teachable_worker import server

    server.Handler.ROUTES = server.routes()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(method, path, body=None, token=None, raw=None, ctype=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(base + path, data=data, method=method)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        if ctype or body is not None:
            req.add_header("Content-Type", ctype or "application/json")
        try:
            with urllib.request.urlopen(req) as r:
                b = r.read()
                return r.status, (json.loads(b) if r.headers.get_content_type() == "application/json" else b)
        except urllib.error.HTTPError as e:
            b = e.read()
            return e.code, (json.loads(b) if b and e.headers.get_content_type() == "application/json" else b)

    yield call
    httpd.shutdown()
