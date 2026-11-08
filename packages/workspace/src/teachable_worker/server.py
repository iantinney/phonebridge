"""The HTTP server: stdlib ThreadingHTTPServer, JSON under /api, the static web/ tree at /.

Routes are `(method, "/api/area/{param}", handler, auth)`; this module's own are below, and every lane module in
MODULES that exists contributes its `ROUTES`. auth: `public`, `person` (cookie `tw` or `Authorization: Bearer`, a
people.token) or `admin` (the admin token). Person routes are always scoped to `req.person["org_id"]`.
Request logs never carry tokens or query strings.
"""

from __future__ import annotations

import hmac
import json
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
from urllib.parse import parse_qs, quote, urlsplit

from . import agency
from .errors import Refused
from .httpd import (COOKIE_DAYS, INLINE, MAX_BODY, MAX_UPLOAD, TRUSTED_PROXIES, WEB, Handler, Request,  # noqa: F401
                    Response, _CI, _err, admin_token, compile_routes, is_operator, module_routes, person_by_token,
                    session_cookie)
from . import httpd
from .store import data_dir, event, one, query

MODULES = ("workspace", "teaching", "phone", "coaching", "growth", "missions", "phone_coach")
def _need(d: dict, *keys: str) -> list:
    missing = [k for k in keys if d.get(k) in (None, "")]
    if missing:
        raise Refused(f"Missing {', '.join(missing)}.")
    return [d[k] for k in keys]


# ---- customer routes ----------------------------------------------------------------------------------------

START_LIMIT = int(os.environ.get("TW_START_PER_HOUR", "20"))
_starts: dict[str, list[float]] = {}
_starts_lock = threading.Lock()


def start(req):
    """Public self-serve sign-up (Game Review, agency pilot request): new external org + owner, signed in."""
    if req.person:
        return {"person": agency.public_person(req.person), "org": agency.get_org(req.org_id)}
    b = req.json()
    if b.get("adult") is not True:
        raise Refused("Game Review is for adults; confirm you are 18 or older.")
    ip = req.client
    t = time.time()
    with _starts_lock:
        recent = [x for x in _starts.get(ip, []) if t - x < 3600]
        if len(recent) >= START_LIMIT:
            raise agency.TooMany("Too many sign-ups from here in the last hour; try again later.")
        _starts[ip] = recent + [t]
    p = agency.start(b.get("track") or "dating", (b.get("name") or "").strip()[:80] or None,
                     (b.get("source") or "")[:64] or None, b.get("ref"), (b.get("org_name") or "").strip()[:120] or None)
    body = {"person": agency.public_person(p), "org": agency.get_org(p["org_id"])}
    return Response(200, json.dumps(body).encode(), headers={"Set-Cookie": session_cookie(p["token"], req.headers)})


def me(req):
    return {"person": agency.public_person(req.person), "org": agency.get_org(req.org_id)}


def put_media(req):
    name = (req.query.get("name") or "upload").replace("/", "_")[:200]
    return agency.save_media(req.org_id, name, req.stream, req.headers.get("Content-Type"), length=req.length)


def get_media(req):
    """Recognized image/video/audio types inline; everything else (HTML, SVG, scripts) only as a download."""
    m = agency.get_media(None if req.admin else req.org_id, req.params["sha"])
    ctype = (m["content_type"] or "").split(";")[0].strip().lower()
    headers = {"Cache-Control": "private, max-age=3600", "Content-Security-Policy": "default-src 'none'; sandbox"}
    if ctype not in INLINE:
        ctype = "application/octet-stream"
        headers["Content-Disposition"] = f'attachment; filename="{quote(m["name"])}"'
    return Response(200, Path(m["path"]).read_bytes(), ctype, headers)


def get_summary(req):
    return agency.summary(req.org_id)


def list_accounts(req):
    return [agency.account_view(a) for a in query("SELECT * FROM accounts WHERE org_id = ? ORDER BY created",
                                                  (req.org_id,))]


def add_account(req):
    b = req.json()
    platform, handle = _need(b, "platform", "handle")
    return agency.account_view(agency.add_account(req.org_id, platform, handle, req.person["id"],
                                                  instructions=b.get("instructions")))


def patch_account(req):
    b = req.json()
    return agency.account_view(agency.update_account(req.params["id"], req.org_id, b.get("instructions") or "",
                                                     req.person["id"]))


def set_permission(req):
    (mode,) = _need(req.json(), "mode")
    if mode not in ("prepare", "approve_each", "delegated"):
        raise Refused("Choose prepare, approve_each or delegated.")
    agency.get_account(req.params["id"], req.org_id)
    return agency.grant(req.params["id"], mode, req.person["id"],
                        f"chosen in the app by {req.person['name']} ({req.person['id']})")


def list_routines(req):
    return [agency.routine_view(r) for r in query("SELECT * FROM routines WHERE org_id = ? ORDER BY created DESC",
                                                  (req.org_id,))]


def get_routine(req):
    return agency.routine_view(agency.get_routine(req.params["id"], req.org_id), detail=True)


def confirm_routine(req):
    return agency.routine_view(agency.confirm_routine(req.params["id"], req.org_id, req.person["id"]))


def correct_routine(req):
    b = req.json()
    return agency.routine_view(agency.correct_routine(req.params["id"], req.org_id, b.get("text") or "",
                                                      req.person["id"], job_id=b.get("job_id")), detail=True)


def patch_routine(req):
    b = req.json()
    return agency.routine_view(agency.edit_routine(req.params["id"], req.org_id, req.person["id"], name=b.get("name"),
                                                   steps=b.get("steps")), detail=True)


def list_jobs(req):
    return agency.jobs_for(req.org_id, open_only=req.query.get("open") in ("1", "true"))


def get_job(req):
    return agency.job_view(agency.get_job(req.params["id"], req.org_id), detail=True)


def post_job(req):
    b = req.json()
    account_id, kind = _need(b, "account_id", "kind")
    j = agency.submit_job(req.org_id, account_id, kind, b.get("inputs") or {}, req.person["id"],
                          origin="customer", due_at=b.get("due_at"), routine_id=b.get("routine_id"))
    return agency.job_view(j, detail=True)


def patch_job(req):
    b = req.json()
    j = agency.update_inputs(req.params["id"], req.org_id, req.person["id"], caption=b.get("caption"),
                             notes=b.get("notes"), media=b.get("media"))
    return agency.job_view(j, detail=True)


def job_action(req):
    jid, act, by, b = req.params["id"], req.params["action"], req.person["id"], req.json()
    if act == "approve":
        j = agency.approve(jid, b.get("sha") or "", by, req.org_id)
    elif act == "accept":
        j = agency.accept(jid, "customer_click", by, req.org_id)
    elif act == "reject":
        j = agency.reject(jid, b.get("reason") or "", by, req.org_id, correction=b.get("correction"))
    elif act == "unblock":
        if agency.get_job(jid, req.org_id)["needs_from"] != "customer":
            raise agency.Conflict("We are handling this one; nothing is needed from you.")
        j = agency.unblock(jid, by, req.org_id, note=b.get("note"))
    elif act == "cancel":
        j = agency.cancel(jid, by, req.org_id)
    else:
        raise agency.NotFound("Unknown action.")
    return agency.job_view(j, detail=True)


# ---- operator (admin) routes ----------------------------------------------------------------------------------

def _link(req, person: dict) -> str:
    host = req.headers.get("X-Forwarded-Host") or req.headers.get("Host") or "127.0.0.1"
    proto = req.headers.get("X-Forwarded-Proto") or "http"
    return f"{proto}://{host}/t/{quote(person['token'])}"


def ops_overview(req):
    return {
        "orgs": query("SELECT * FROM orgs ORDER BY created DESC"),
        "people": [agency.public_person(p) for p in query("SELECT * FROM people ORDER BY created DESC")],
        "accounts": [agency.account_view(a) for a in query("SELECT * FROM accounts ORDER BY created DESC")],
        "attention": [agency.job_view(j) for j in query(
            "SELECT * FROM jobs WHERE (status = 'needs_input' AND needs_from = 'founder') OR status IN "
            "('queued', 'running') ORDER BY updated")],
        "jobs": [agency.job_view(j) for j in query("SELECT * FROM jobs ORDER BY updated DESC LIMIT 200")],
        "events": query("SELECT * FROM events ORDER BY id DESC LIMIT 100"),
    }


ONBOARDED = "onboarded by the operator"


def ops_create_org(req):
    b = req.json()
    name, track, cohort = _need(b, "name", "track", "cohort")
    org = agency.create_org(name, track, cohort, b.get("source"), b.get("referred_by"))
    out = {"org": org}
    if b.get("person"):
        p = agency.add_person(org["id"], **{k: b["person"][k] for k in ("name", "email", "role") if k in b["person"]},
                              verified=b["person"].get("verified_evidence") or ONBOARDED)
        out |= {"person": agency.public_person(p), "login_link": _link(req, p)}
    return out


def ops_add_person(req):
    b = req.json()
    (name,) = _need(b, "name")
    p = agency.add_person(req.params["id"], name, b.get("email"), b.get("role", "owner"), b.get("cohort"),
                          b.get("source"), b.get("referred_by"), verified=b.get("verified_evidence") or ONBOARDED)
    return {"person": agency.public_person(p), "login_link": _link(req, p)}


def ops_verify_person(req):
    (evidence,) = _need(req.json(), "evidence")
    return agency.public_person(agency.verify_person(req.params["id"], evidence))


def ops_cohort(req):
    cohort, reason = _need(req.json(), "cohort", "reason")
    return agency.set_cohort(req.params["id"], cohort, reason)


def ops_link(req):
    p = one("SELECT * FROM people WHERE id = ?", (req.params["id"],))
    if not p:
        raise agency.NotFound("No such person.")
    return {"login_link": _link(req, p)}


def ops_account(req):
    aid, act, b = req.params["id"], req.params["action"], req.json()
    if act == "verify":
        return agency.verify_authority(aid, b.get("evidence") or "")
    if act == "identity":
        identity, evidence = _need(b, "identity", "evidence")
        return agency.map_identity(aid, identity, evidence)
    if act == "access":
        access, evidence = _need(b, "access", "evidence")
        return agency.set_access(aid, access, evidence)
    if act == "check":
        return agency.check_access(aid)
    raise agency.NotFound("Unknown action.")


def ops_job(req):
    jid, act, b = req.params["id"], req.params["action"], req.json()
    if act == "unblock":
        return agency.unblock(jid, None, note=b.get("note"))
    if act == "block":
        needs, needs_from = _need(b, "needs", "needs_from")
        return agency.block(jid, needs, needs_from)
    if act == "publication":  # executed_by: phone (verifying the phone's run) or founder (posted by hand)
        (executed_by,) = _need(b, "executed_by")
        return agency.record_publication(jid, b.get("url") or "", b.get("evidence") or "", None,
                                         executed_by=executed_by)
    if act == "accept":  # acceptance the customer gave by message: say where
        (evidence,) = _need(b, "evidence")
        j = agency.accept(jid, "customer_message", None)
        event("acceptance_evidence", actor="founder", org_id=j["org_id"], job_id=jid, evidence=evidence)
        return j
    if act == "not_posted":
        (evidence,) = _need(b, "evidence")
        return agency.not_posted(jid, evidence)
    if act == "retry":
        return agency.retry(jid, None)
    if act == "cancel":
        return agency.cancel(jid, None)
    raise agency.NotFound("Unknown action.")


def ops_submit_job(req):
    """A job entered by the founder: origin `customer` only for a request the customer made (say where)."""
    b = req.json()
    org_id, account_id, kind, origin = _need(b, "org_id", "account_id", "kind", "origin")
    if origin not in ("customer", "founder", "test"):
        raise Refused("origin is customer (relayed request), founder or test.")
    if origin == "customer":
        _need(b, "requested_by", "evidence")
    j = agency.submit_job(org_id, account_id, kind, b.get("inputs") or {},
                          b.get("requested_by") if origin == "customer" else None, origin=origin,
                          due_at=b.get("due_at"), routine_id=b.get("routine_id"), relayed_by="operator")
    if b.get("evidence"):
        event("request_evidence", actor="founder", org_id=org_id, job_id=j["id"], evidence=b["evidence"])
    return agency.job_view(j, detail=True)


def ops_put_media(req):
    org_id = req.query.get("org_id")
    agency.get_org(org_id or "")
    name = (req.query.get("name") or "upload").replace("/", "_")[:200]
    return agency.save_media(org_id, name, req.stream, req.headers.get("Content-Type"), length=req.length)


def ops_event(req):
    b = req.json()
    (name,) = _need(b, "name")
    if name not in ("founder_rescue", "effort"):
        raise Refused("The console records founder_rescue or effort.")
    data = {k: v for k, v in b.items() if k not in ("name", "org_id", "job_id", "account_id")}
    _need(data, "minutes")
    j = agency.get_job(b["job_id"]) if b.get("job_id") else None
    return {"id": event(name, actor="founder", org_id=(j or {}).get("org_id") or b.get("org_id"),
                        job_id=b.get("job_id"), account_id=(j or {}).get("account_id"), **data)}


def ops_payment(req):
    b = req.json()
    product, amount, status, processor, evidence = _need(b, "product", "amount_cents", "status", "processor",
                                                         "evidence")
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        raise Refused("amount_cents is a whole number.")
    return agency.record_payment(b.get("org_id"), b.get("person_id"), product, amount, status, processor,
                                 b.get("processor_ref"), evidence)


def ops_routine(req):
    b = req.json()
    org_id, name, provenance = _need(b, "org_id", "name", "provenance")
    if provenance in ("customer_demo", "operator_demo"):
        raise Refused("A demonstration routine comes from a recording (teaching), not the console.")
    return agency.routine_view(agency.create_routine(org_id, name, provenance, None, b.get("summary") or "",
                                                     b.get("steps"), b.get("account_id")))


def get_metrics(req):
    if (metrics := agency.lane("metrics")) is None:  # the operator's scorecard lane, when this install has one
        raise agency.NotFound("No scorecard on this install.")
    return metrics.readout()


def health(req):
    return {"ok": True}


ROUTES = [
    ("GET", "/api/health", health, "public"),
    ("POST", "/api/start", start, "public"),
    ("GET", "/api/me", me, "person"),
    ("PUT", "/api/media", put_media, "person"),
    ("GET", "/api/media/{sha}", get_media, "person"),
    ("GET", "/api/summary", get_summary, "person"),
    ("GET", "/api/accounts", list_accounts, "person"),
    ("POST", "/api/accounts", add_account, "person"),
    ("PATCH", "/api/accounts/{id}", patch_account, "person"),
    ("POST", "/api/accounts/{id}/permission", set_permission, "person"),
    ("GET", "/api/routines", list_routines, "person"),
    ("GET", "/api/routines/{id}", get_routine, "person"),
    ("PATCH", "/api/routines/{id}", patch_routine, "person"),
    ("POST", "/api/routines/{id}/confirm", confirm_routine, "person"),
    ("POST", "/api/routines/{id}/corrections", correct_routine, "person"),
    ("GET", "/api/jobs", list_jobs, "person"),
    ("POST", "/api/jobs", post_job, "person"),
    ("GET", "/api/jobs/{id}", get_job, "person"),
    ("PATCH", "/api/jobs/{id}", patch_job, "person"),
    ("POST", "/api/jobs/{id}/{action}", job_action, "person"),
    ("GET", "/api/metrics", get_metrics, "admin"),
    ("GET", "/api/ops/overview", ops_overview, "admin"),
    ("POST", "/api/ops/orgs", ops_create_org, "admin"),
    ("POST", "/api/ops/orgs/{id}/people", ops_add_person, "admin"),
    ("GET", "/api/ops/people/{id}/link", ops_link, "admin"),
    ("POST", "/api/ops/people/{id}/verify", ops_verify_person, "admin"),
    ("POST", "/api/ops/orgs/{id}/cohort", ops_cohort, "admin"),
    ("POST", "/api/ops/accounts/{id}/{action}", ops_account, "admin"),
    ("POST", "/api/ops/jobs", ops_submit_job, "admin"),
    ("POST", "/api/ops/jobs/{id}/{action}", ops_job, "admin"),
    ("PUT", "/api/ops/media", ops_put_media, "admin"),
    ("POST", "/api/ops/events", ops_event, "admin"),
    ("POST", "/api/ops/payments", ops_payment, "admin"),
    ("POST", "/api/ops/routines", ops_routine, "admin"),
]


def routes() -> list[tuple]:
    return compile_routes(ROUTES + module_routes(MODULES))


def _ticks():
    from . import workspace
    return (agency.work_once, getattr(agency.lane("coaching"), "purge_expired", None), workspace.purge_frames)


def serve(host: str = "127.0.0.1", port: int = 8780, work_interval: float = 15.0) -> None:
    httpd.serve(ROUTES + module_routes(MODULES), host, port, work_interval, _ticks)
