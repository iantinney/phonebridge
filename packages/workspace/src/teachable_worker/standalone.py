"""Phone tasks on the same Mac as a locally leased Phonebridge daemon, and `phonebridge-workspace`, which serves the
workspace for them."""

from contextlib import contextmanager
import argparse
import os
import shlex
import shutil
import sys

TEACH_PATHS = ("/api/media", "/api/routines", "/api/accounts")  # what Teach's screens call besides /api/teach (tw-web)


def enabled() -> bool:
    return os.environ.get("TW_PHONE_ADAPTER") == "local"


def lease_command() -> str:
    return os.environ.get("TW_LEASE_CMD") or "phonebridge-local-lease lease"


def available() -> tuple[bool, str]:
    try:
        command = shlex.split(lease_command())
    except ValueError:
        return False, "The local lease command has invalid quoting (TW_LEASE_CMD)."
    if not command or not shutil.which(command[0]):
        return False, "Install Phonebridge on this Mac so phonebridge-local-lease is available."
    return True, ""


@contextmanager
def session(ctx):
    from .missions import Hands
    from .workspace import LeaseSession

    # The daemon and workspace must share PHONEBRIDGE_STATE on this Mac. The
    # daemon checks this token before each input, using its normal lease guard.
    with LeaseSession(ctx, command=lease_command()) as lease:
        hands = Hands(ctx.device["id"], lease.token)
        ctx.hands = hands
        try:
            yield ctx
        finally:
            hands.release()  # release controller ownership before the device lease


def routes() -> list[tuple]:
    """The workspace, its read-only tasks and Teach: those modules' own routes, plus the routine, media and account
    routes Teach's screens call (server.py's, which also serves the agency product this release leaves out)."""
    from . import httpd, server, workspace

    return (httpd.HEALTH + httpd.module_routes((*workspace.WORKSPACE_MODULES, "teaching"))
            + [r for r in server.ROUTES if r[1].startswith(TEACH_PATHS)])


def main(argv: list[str] | None = None) -> int:
    """`phonebridge-workspace [--port 8780]`: the desktop workspace on 127.0.0.1 for the phones of this Mac's
    `phonebridge serve --http`, with the local device lease. Prints the operator sign-in link."""
    ap = argparse.ArgumentParser(prog="phonebridge-workspace", description=main.__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8780)
    a = ap.parse_args(argv)
    os.environ.setdefault("TW_PHONE_ADAPTER", "local")  # before the server imports its lanes
    ok, why = available()
    if not ok:
        print(f"Phone tasks are off: {why}", file=sys.stderr)
    from . import httpd, workspace

    # The operator is a founder person: it drives the phones and also has the person-scoped routes (Teach).
    print(f"Sign in at http://127.0.0.1:{a.port}/t/{workspace.ensure_operator()['token']} (keep this link private)",
          flush=True)
    httpd.serve(routes(), "127.0.0.1", a.port, 60, lambda: (workspace.purge_frames,), banner="phonebridge workspace")
    return 0
