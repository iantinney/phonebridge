"""The agency flow: organizations, people, accounts, permissions, media, routines, jobs and results.

This module is the only writer of `jobs.status`:

    requested ─prepare─▶ awaiting_approval ─approve(sha)─▶ queued ─▶ running ─▶ delivered ─accept─▶ accepted
       kind=prepare: requested ─prepare─▶ delivered (a draft result; nothing touches the account)
    delivered | awaiting_approval ─reject─▶ rejected · open ─block─▶ needs_input ─unblock─▶ where it left off
    running ─phone failure─▶ failed ─retry─▶ queued · not running ─cancel─▶ cancelled

Approval binds to `content_sha` (account, kind, media hashes, caption); any edit clears it, and the worker
re-checks it before the phone is touched. `run_ref` is stored before `phone.submit`, so a restart polls the same
run instead of starting another. A publish is delivered only with a verified readback.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import mimetypes
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from . import phone
from .errors import Conflict, NotFound, Refused, TooLarge, TooMany, lane  # noqa: F401 (re-exported)
from .store import data_dir, event, extend, new_id, now, one, private_dir, query, tx

OPEN = ("requested", "awaiting_approval", "queued", "running", "needs_input", "delivered")
FINAL = ("accepted", "rejected", "failed", "cancelled")
PUBLISH_MODES = ("approve_each", "delegated")

MEDIA = """
CREATE TABLE IF NOT EXISTS media (
  sha256 TEXT NOT NULL, org_id TEXT NOT NULL REFERENCES orgs(id), name TEXT NOT NULL, content_type TEXT,
  bytes INTEGER NOT NULL, path TEXT NOT NULL, created TEXT NOT NULL, PRIMARY KEY (org_id, sha256));
"""
extend(MEDIA)


def actor_of(person_id: str | None) -> str:
    if person_id is None:
        return "founder"  # the admin console
    p = one("SELECT role, cohort FROM people WHERE id = ?", (person_id,))
    return "founder" if p and (p["role"] == "founder" or p["cohort"] in ("founder", "staff")) else "customer"


# ---- organizations, people ----------------------------------------------------------------------------------

def create_org(name: str, track: str, cohort: str, source: str | None = None, referred_by: str | None = None) -> dict:
    oid = new_id("org")
    with tx() as con:
        con.execute("INSERT INTO orgs VALUES (?, ?, ?, ?, ?, ?, ?)", (oid, name, track, cohort, source, referred_by, now()))
        event("org_created", actor="founder", org_id=oid, con=con, track=track, cohort=cohort, source=source)
    return get_org(oid)


def get_org(org_id: str) -> dict:
    o = one("SELECT * FROM orgs WHERE id = ?", (org_id,))
    if not o:
        raise NotFound("That organization does not exist.")
    return o


def add_person(org_id: str, name: str, email: str | None = None, role: str = "owner", cohort: str | None = None,
               source: str | None = None, referred_by: str | None = None, verified: str | None = None) -> dict:
    """`verified` is the evidence that this is a distinct human we know (operator onboarding); a self-serve
    browser session stays unverified until a payment or an operator check says otherwise."""
    org = get_org(org_id)
    pid = new_id("per")
    with tx() as con:
        con.execute("INSERT INTO people (id, org_id, name, email, role, cohort, source, referred_by, token, created, "
                    "verified_at, verified_evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (pid, org_id, name, email, role, cohort or org["cohort"], source or org["source"], referred_by,
                     secrets.token_urlsafe(24), now(), now() if verified else None, verified))
        event("person_created", actor="founder" if verified else "customer", org_id=org_id, person_id=pid, con=con,
              role=role, verified=bool(verified))
    return one("SELECT * FROM people WHERE id = ?", (pid,))


def verify_person(person_id: str, evidence: str) -> dict:
    if not evidence:
        raise Refused("Say how you know this is a distinct person.")
    with tx() as con:
        p = one("SELECT * FROM people WHERE id = ?", (person_id,))
        if not p:
            raise NotFound("No such person.")
        con.execute("UPDATE people SET verified_at = COALESCE(verified_at, ?), verified_evidence = "
                    "COALESCE(verified_evidence, ?) WHERE id = ?", (now(), evidence, person_id))
        event("person_verified", actor="founder", org_id=p["org_id"], person_id=person_id, con=con)
    return one("SELECT * FROM people WHERE id = ?", (person_id,))


def start(track: str, name: str | None, source: str | None, ref: str | None, org_name: str | None = None) -> dict:
    """Self-serve sign-up: a new external org of one (or an agency) and its owner. `ref` is the referring person's id;
    an unknown ref is ignored, never an error (links get mangled)."""
    if track not in ("dating", "agency"):
        raise Refused("Choose dating or agency.")
    referrer = one("SELECT * FROM people WHERE id = ?", (ref,)) if ref else None
    with tx():
        org = create_org(org_name or name or ("New agency" if track == "agency" else "Reviewer"), track, "external",
                         source, referrer["org_id"] if referrer else None)
        p = add_person(org["id"], name or "You", role="owner", source=source,
                       referred_by=referrer["id"] if referrer else None)
    return p


def set_cohort(org_id: str, cohort: str, reason: str) -> dict:
    """Operator-only: re-label an org and its people (an internal tester who signed up as external, a friend)."""
    org = get_org(org_id)
    if not reason:
        raise Refused("Say why the cohort changes.")
    with tx() as con:
        con.execute("UPDATE orgs SET cohort = ? WHERE id = ?", (cohort, org_id))
        con.execute("UPDATE people SET cohort = ? WHERE org_id = ?", (cohort, org_id))
        event("cohort_changed", actor="founder", org_id=org_id, con=con, before=org["cohort"], after=cohort,
              reason=reason)
    return get_org(org_id)


def person_by_token(token: str) -> dict | None:
    return one("SELECT * FROM people WHERE token = ?", (token,)) if token else None


def public_person(p: dict) -> dict:
    return {k: v for k, v in p.items() if k != "token"}


# ---- accounts and permissions --------------------------------------------------------------------------------

def add_account(org_id: str, platform: str, handle: str, by: str | None, identity: str | None = None,
                instructions: str | None = None) -> dict:
    get_org(org_id)
    handle = handle.strip().lstrip("@")
    if not platform or not handle:
        raise Refused("Give the platform and the account handle.")
    if one("SELECT 1 FROM accounts WHERE org_id = ? AND platform = ? AND handle = ?", (org_id, platform, handle)):
        raise Conflict(f"@{handle} on {platform} is already connected.")
    aid = new_id("acc")
    with tx() as con:
        con.execute("INSERT INTO accounts (id, org_id, platform, handle, identity, instructions, created) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", (aid, org_id, platform, handle, identity, instructions, now()))
        event("account_added", actor=actor_of(by), org_id=org_id, person_id=by, account_id=aid, con=con,
              platform=platform)
    return get_account(aid)


def get_account(account_id: str, org_id: str | None = None) -> dict:
    a = one("SELECT * FROM accounts WHERE id = ?", (account_id,))
    if not a or (org_id and a["org_id"] != org_id):
        raise NotFound("That account does not exist.")
    return a


def update_account(account_id: str, org_id: str, instructions: str, by: str) -> dict:
    get_account(account_id, org_id)
    with tx() as con:
        con.execute("UPDATE accounts SET instructions = ? WHERE id = ?", (instructions, account_id))
        event("account_instructions", actor=actor_of(by), org_id=org_id, person_id=by, account_id=account_id, con=con)
    return get_account(account_id)


def verify_authority(account_id: str, evidence: str, by: str | None = None) -> dict:
    """Founder-side: the owner's delegated authority is confirmed (how, in `evidence`)."""
    a = get_account(account_id)
    if not evidence:
        raise Refused("Say how the owner's authority was verified.")
    with tx() as con:
        con.execute("UPDATE accounts SET authority = 'verified', authority_evidence = ? WHERE id = ?",
                    (evidence, account_id))
        event("authority_verified", actor=actor_of(by), org_id=a["org_id"], person_id=by, account_id=account_id,
              con=con)
    return get_account(account_id)


def map_identity(account_id: str, identity: str, evidence: str, by: str | None = None) -> dict:
    """Operator-only: which authhub identity / device route executes this account. Customers never set it, and one
    identity maps to one account (a customer typing the founder's handle gets no route to the founder's account)."""
    a = get_account(account_id)
    if not identity or not evidence:
        raise Refused("Give the identity and the evidence that this account is that identity.")
    other = one("SELECT id FROM accounts WHERE identity = ? AND id != ?", (identity, account_id))
    if other:
        raise Conflict(f"{identity} already belongs to account {other['id']}.")
    with tx() as con:
        con.execute("UPDATE accounts SET identity = ? WHERE id = ?", (identity, account_id))
        event("identity_mapped", actor=actor_of(by), org_id=a["org_id"], person_id=by, account_id=account_id, con=con,
              identity=identity, evidence=evidence)
    return get_account(account_id)


def set_access(account_id: str, access: str, evidence: str, by: str | None = None) -> dict:
    a = get_account(account_id)
    with tx() as con:
        con.execute("UPDATE accounts SET access = ?, access_checked = ?, access_evidence = ? WHERE id = ?",
                    (access, now(), evidence, account_id))
        event("access_checked", actor=actor_of(by) if by else "system", org_id=a["org_id"], person_id=by,
              account_id=account_id, con=con, usable=access == "usable")
    return get_account(account_id)


def check_access(account_id: str, by: str | None = None) -> dict:
    r = phone.readiness(get_account(account_id))
    return set_access(account_id, "usable" if r["usable"] else "broken", r.get("reason") or r.get("evidence") or "", by)


def permission(account_id: str) -> dict | None:
    return one("SELECT * FROM permissions WHERE account_id = ? AND revoked IS NULL ORDER BY created DESC LIMIT 1",
               (account_id,))


def grant(account_id: str, mode: str, granted_by: str, evidence: str) -> dict:
    a = get_account(account_id)
    pid = new_id("prm")
    with tx() as con:
        con.execute("UPDATE permissions SET revoked = ? WHERE account_id = ? AND revoked IS NULL", (now(), account_id))
        con.execute("INSERT INTO permissions (id, account_id, mode, granted_by, evidence, created) "
                    "VALUES (?, ?, ?, ?, ?, ?)", (pid, account_id, mode, granted_by, evidence, now()))
        event("permission_granted", actor=actor_of(granted_by), org_id=a["org_id"], person_id=granted_by,
              account_id=account_id, con=con, mode=mode)
    return one("SELECT * FROM permissions WHERE id = ?", (pid,))


def revoke(permission_id: str, by: str) -> None:
    with tx() as con:
        con.execute("UPDATE permissions SET revoked = ? WHERE id = ? AND revoked IS NULL", (now(), permission_id))


def account_view(a: dict) -> dict:
    p = permission(a["id"])
    return {**a, "permission": {"id": p["id"], "mode": p["mode"]} if p else None,
            "readiness": {"usable": a["access"] == "usable", "reason": a["access_evidence"],
                          "checked": a["access_checked"]}}


# ---- media ---------------------------------------------------------------------------------------------------

def save_media(org_id: str, name: str, body, content_type: str | None = None, length: int | None = None) -> dict:
    """Store an upload under its sha256. `body` is bytes or a stream read in chunks (with `length`), so a large
    video never sits in memory."""
    get_org(org_id)
    stream = io.BytesIO(body) if isinstance(body, (bytes, bytearray)) else body
    left = len(body) if isinstance(body, (bytes, bytearray)) else (length or 0)
    if left <= 0:
        raise Refused("The upload was empty.")
    folder = private_dir(data_dir() / "media")  # uploads (recordings, media): 0700; the temp file is 0600 already
    h, size = hashlib.sha256(), 0
    with tempfile.NamedTemporaryFile(dir=folder, delete=False, prefix=".upload-") as tmp:
        try:
            while left > 0:
                chunk = stream.read(min(left, 1 << 20))
                if not chunk:
                    raise Refused("The upload ended early.")
                h.update(chunk)
                tmp.write(chunk)
                size += len(chunk)
                left -= len(chunk)
        except BaseException:
            os.unlink(tmp.name)
            raise
    sha = h.hexdigest()
    ext = "".join(c for c in (name.rsplit(".", 1)[-1].lower() if "." in name else "bin") if c.isalnum())[:8] or "bin"
    path = folder / f"{sha}.{ext}"
    if path.exists():
        os.unlink(tmp.name)
    else:
        os.replace(tmp.name, path)
    ctype = mimetypes.guess_type(f"x.{ext}")[0] or content_type or "application/octet-stream"
    with tx() as con:
        con.execute("INSERT OR IGNORE INTO media (sha256, org_id, name, content_type, bytes, path, created) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", (sha, org_id, name, ctype, size, str(path), now()))
    return {"path": str(path), "sha256": sha, "name": name, "bytes": size, "content_type": ctype}


def delete_media(org_id: str, sha: str, by: str | None = None) -> None:
    """Remove an org's upload (a deleted recording, a privacy request). Refused while an open job still uses it; the
    file goes when no other org holds the same bytes."""
    m = get_media(org_id, sha)
    in_use = [j["id"] for j in query(f"SELECT id, input_json FROM jobs WHERE org_id = ? AND status NOT IN "
                                     f"({','.join('?' * len(FINAL))})", (org_id, *FINAL))
              if any(x["sha256"] == sha for x in json.loads(j["input_json"])["media"])]
    if in_use:
        raise Conflict(f"Open work still uses this file ({', '.join(in_use)}); finish or cancel it first.")
    with tx() as con:
        con.execute("DELETE FROM media WHERE org_id = ? AND sha256 = ?", (org_id, sha))
        others = con.execute("SELECT 1 FROM media WHERE sha256 = ?", (sha,)).fetchone()
        event("media_deleted", actor=actor_of(by) if by else "system", org_id=org_id, person_id=by, con=con)
    if not others:
        Path(m["path"]).unlink(missing_ok=True)


def get_media(org_id: str | None, sha: str) -> dict:
    m = (one("SELECT * FROM media WHERE org_id = ? AND sha256 = ?", (org_id, sha)) if org_id
         else one("SELECT * FROM media WHERE sha256 = ?", (sha,)))
    if not m:
        raise NotFound("That media file does not exist.")
    return m


# ---- routines (the core view; teaching.py derives and corrects them when present) -----------------------------

def create_routine(org_id: str, name: str, provenance: str, by: str | None, summary: str = "",
                   steps: list | None = None, account_id: str | None = None, recording_id: str | None = None) -> dict:
    rid = new_id("rtn")
    with tx() as con:
        con.execute("INSERT INTO routines (id, org_id, account_id, name, provenance, recording_id, summary, steps_json, "
                    "created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (rid, org_id, account_id, name, provenance, recording_id, summary, json.dumps(steps or []), now()))
        event("routine_derived", actor=actor_of(by), org_id=org_id, person_id=by, routine_id=rid, con=con,
              provenance=provenance)
    return get_routine(rid)


def get_routine(routine_id: str, org_id: str | None = None) -> dict:
    r = one("SELECT * FROM routines WHERE id = ?", (routine_id,))
    if not r or (org_id and r["org_id"] != org_id):
        raise NotFound("That routine does not exist.")
    return r


def routine_view(r: dict, detail: bool = False) -> dict:
    out = {**r, "steps": json.loads(r["steps_json"] or "[]")}
    if detail:
        out["corrections"] = query("SELECT id, text, job_id, created, applied_version FROM corrections "
                                   "WHERE routine_id = ? ORDER BY created", (r["id"],))
        teaching = lane("teaching")
        rec = getattr(teaching, "recording", None) if teaching else None
        out["recording"] = rec(r["recording_id"]) if rec and r["recording_id"] else None
        extras = getattr(teaching, "routine_extras", None) if teaching else None
        if extras:  # what the saved workflow asks for (slots) and knows (rules, open questions, pace); never recurses
            out.update(extras(r))
    return out


def confirm_routine(routine_id: str, org_id: str, by: str) -> dict:
    get_routine(routine_id, org_id)
    teaching = lane("teaching")
    if teaching and hasattr(teaching, "confirm"):
        teaching.confirm(routine_id, by)
    else:
        with tx() as con:
            con.execute("UPDATE routines SET confirmed_at = ?, confirmed_by = ? WHERE id = ?", (now(), by, routine_id))
            event("routine_confirmed", actor=actor_of(by), org_id=org_id, person_id=by, routine_id=routine_id, con=con)
    return get_routine(routine_id)


def correct_routine(routine_id: str, org_id: str, text: str, by: str, job_id: str | None = None) -> dict:
    get_routine(routine_id, org_id)
    if job_id:
        get_job(job_id, org_id)
    if not text or not text.strip():
        raise Refused("Write the correction.")
    teaching = lane("teaching")
    if teaching and hasattr(teaching, "correct"):
        teaching.correct(routine_id, text, by, job_id=job_id)
    else:
        _add_correction(routine_id, job_id, text, by)
    return get_routine(routine_id)


def _add_correction(routine_id, job_id, text, by, applied_version=None, con=None):
    row = (new_id("cor"), routine_id, job_id, text, by, now(), applied_version)
    sql = "INSERT INTO corrections VALUES (?, ?, ?, ?, ?, ?, ?)"
    if con is not None:
        con.execute(sql, row)
    else:
        with tx() as c:
            c.execute(sql, row)
    org = one("SELECT org_id FROM routines WHERE id = ?", (routine_id,)) if routine_id else None
    event("routine_corrected", actor=actor_of(by), org_id=org and org["org_id"], person_id=by, routine_id=routine_id,
          job_id=job_id, con=con)


def edit_routine(routine_id: str, org_id: str, by: str, name: str | None = None, steps: list | None = None) -> dict:
    r = get_routine(routine_id, org_id)
    with tx() as con:
        if name:
            con.execute("UPDATE routines SET name = ? WHERE id = ?", (name, routine_id))
        if steps is not None:
            v = r["version"] + 1
            con.execute("UPDATE routines SET steps_json = ?, version = ? WHERE id = ?",
                        (json.dumps(steps), v, routine_id))
            _add_correction(routine_id, None, "Edited the steps directly", by, applied_version=v, con=con)
    return get_routine(routine_id)


# ---- jobs ----------------------------------------------------------------------------------------------------

def content_sha(account_id: str, kind: str, inputs: dict, routine: dict | None = None) -> str:
    """What an approval binds: destination, kind, media hashes, caption, and the routine version that will run."""
    media = [m.get("sha256") for m in inputs.get("media") or []]
    blob = json.dumps({"account_id": account_id, "kind": kind, "media": media, "caption": inputs.get("caption"),
                       "routine": [routine["id"], routine["version"]] if routine else None}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def current_sha(job: dict) -> str:
    r = get_routine(job["routine_id"]) if job["routine_id"] else None
    return content_sha(job["account_id"], job["kind"], json.loads(job["input_json"]), r)


def get_job(job_id: str, org_id: str | None = None) -> dict:
    j = one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if not j or (org_id and j["org_id"] != org_id):
        raise NotFound("That job does not exist.")
    return j


def _set(con, job: dict, **cols) -> None:
    """Change the job `job` was read as (optimistic: its `rev`); a concurrent change raises Conflict instead of
    acting on a stale read (a cancel racing the worker, an edit racing an approval)."""
    cols["updated"] = now()
    n = con.execute(f"UPDATE jobs SET {', '.join(f'{k} = ?' for k in cols)}, rev = rev + 1 WHERE id = ? AND rev = ?",
                    (*cols.values(), job["id"], job["rev"])).rowcount
    if n != 1:
        raise Conflict("This job changed meanwhile; reload it and try again.")
    job.update(cols, rev=job["rev"] + 1)


def _ev(con, name: str, job: dict, by: str | None, actor: str | None = None, **data) -> None:
    event(name, actor=actor or actor_of(by), org_id=job["org_id"], person_id=by, account_id=job["account_id"],
          job_id=job["id"], routine_id=job["routine_id"], con=con, **data)


def parse_due(due_at: str | None) -> str | None:
    """ISO-8601; a time without a zone is New York time (what a datetime-local input sends). Stored as UTC."""
    if not due_at:
        return None
    try:
        t = datetime.fromisoformat(due_at)
    except (TypeError, ValueError):
        raise Refused("The due time must be a date and time, like 2026-11-09T17:00.")
    if t.tzinfo is None:
        t = t.replace(tzinfo=ZoneInfo("America/New_York"))
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def submit_job(org_id: str, account_id: str, kind: str, inputs: dict, requested_by: str | None,
               origin: str = "customer", due_at: str | None = None, routine_id: str | None = None,
               relayed_by: str | None = None) -> dict:
    """`relayed_by`: the operator entered a request the customer made elsewhere (say where in an event); such a
    job is never auto-approved, since the exact content did not come from the customer's own session."""
    get_account(account_id, org_id)
    due_at = parse_due(due_at)
    if requested_by and not one("SELECT 1 FROM people WHERE id = ? AND org_id = ?", (requested_by, org_id)):
        raise NotFound("That person is not part of this organization.")
    if kind not in ("prepare", "publish"):
        raise Refused("A job is either 'prepare' (a ready draft) or 'publish' (posted on the account).")
    routine = get_routine(routine_id, org_id) if routine_id else None
    inputs = _apply_rules(org_id, routine, _clean_inputs(org_id, inputs))
    if kind == "publish" and len(inputs["media"]) > 1:
        raise Refused("A publish job carries one media file; submit the others as their own jobs.")
    jid = new_id("job")
    t = now()
    with tx() as con:
        con.execute("INSERT INTO jobs (id, org_id, account_id, routine_id, kind, input_json, origin, requested_by, "
                    "requested_at, due_at, status, content_sha, updated, relayed_by) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (jid, org_id, account_id, routine_id, kind, json.dumps(inputs), origin, requested_by, t, due_at,
                     "requested", content_sha(account_id, kind, inputs, routine), t, relayed_by))
        _ev(con, "job_requested", get_job_con(con, jid), requested_by, origin=origin, kind=kind)
    return prepare(jid)


INPUT_KEYS = ("caption", "notes", "acceptance")  # + "rules_changed", set only by _apply_rules


def _clean_inputs(org_id: str, inputs: dict) -> dict:
    """Only known keys; media by sha256 of this org's own uploads, the path always from the media table."""
    out = {k: inputs.get(k) for k in INPUT_KEYS}
    if out["acceptance"] not in (None, "objective"):
        raise Refused("Acceptance is either your review (the default) or 'objective' (a verified live post).")
    out["media"] = []
    for m in inputs.get("media") or []:
        sha = m.get("sha256") if isinstance(m, dict) else m
        if not sha:
            raise Refused("Each media item needs the sha256 returned by the upload.")
        row = get_media(org_id, sha)
        out["media"].append({"sha256": sha, "name": row["name"], "path": row["path"]})
    return out


def _apply_rules(org_id: str, routine: dict | None, inputs: dict) -> dict:
    """The routine's rules (e.g. "always end with #acme") shape the content before it is hashed, so the customer
    approves exactly what will run (teaching.apply_rules)."""
    teaching = lane("teaching")
    if not routine or not teaching or not hasattr(teaching, "apply_rules"):
        return inputs
    ruled = _clean_inputs(org_id, teaching.apply_rules(routine, inputs))
    ruled["rules_changed"] = any(ruled.get(k) != inputs.get(k) for k in ("caption", "notes", "media"))
    return ruled


def get_job_con(con, job_id: str) -> dict:
    return dict(con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())


def _draft(job: dict) -> dict:
    a = get_account(job["account_id"])
    inputs = json.loads(job["input_json"])
    out = {"account": {"id": a["id"], "platform": a["platform"], "handle": a["handle"]}, "kind": job["kind"],
           "media": [{"sha256": m["sha256"], "name": m.get("name")} for m in inputs["media"]], "caption": inputs.get("caption") or "", "notes": inputs.get("notes"),
           "account_instructions": a["instructions"], "content_sha": job["content_sha"]}
    if job["routine_id"]:
        r = get_routine(job["routine_id"])
        out["routine"] = {"id": r["id"], "name": r["name"], "version": r["version"], "summary": r["summary"],
                          "steps": json.loads(r["steps_json"] or "[]")}
        out["corrections"] = [c["text"] for c in query("SELECT text FROM corrections WHERE routine_id = ? "
                                                        "ORDER BY created", (r["id"],))]
    return out


def prepare(job_id: str) -> dict:
    """requested → the draft or the approval request; blocks on the one missing thing."""
    job = get_job(job_id)
    if job["status"] != "requested":
        return job
    inputs = json.loads(job["input_json"])
    a = get_account(job["account_id"])
    if not inputs["media"]:
        return block(job_id, "Upload the finished media for this post.", "customer")
    if job["kind"] == "publish":
        p = permission(a["id"])
        if not p or p["mode"] not in PUBLISH_MODES:
            return block(job_id, f"Allow publishing on @{a['handle']}: choose 'approve each' or 'delegated' for it.",
                         "customer")
    draft = _draft(job)
    teaching = lane("teaching")
    if job["kind"] == "prepare" and job["routine_id"] and teaching and hasattr(teaching, "draft"):
        draft = {**draft, **teaching.draft(get_routine(job["routine_id"]), job), "transform": "routine"}
    else:
        draft["transform"] = "echo"  # nothing was done to the customer's inputs; metrics do not count it as work
    with tx() as con:
        if job["kind"] == "prepare":
            con.execute("INSERT INTO results (id, job_id, kind, output_json, created) VALUES (?, ?, 'draft', ?, ?)",
                        (new_id("res"), job_id, json.dumps(draft), now()))
            _set(con, job, status="delivered")
            _ev(con, "job_prepared", job, None, actor="agent")
            _ev(con, "job_delivered", job, None, actor="agent", result="draft")
        else:
            _set(con, job, status="awaiting_approval", content_sha=current_sha(job))
            _ev(con, "job_prepared", job, None, actor="agent")
            p = permission(a["id"])
            if p["mode"] == "delegated" and job["origin"] == "customer" and job["requested_by"] and \
                    not job["relayed_by"] and not inputs.get("rules_changed"):
                # the customer's own exact media + caption under a delegated grant: submitting it is the approval
                _set(con, job, status="queued", approved_sha=job["content_sha"], approved_by=job["requested_by"],
                     approved_at=now())
                _ev(con, "job_approved", job, job["requested_by"], via="delegated", sha=job["content_sha"])
                _ev(con, "job_queued", job, None, actor="system")
    return get_job(job_id)


def approve(job_id: str, sha: str, by: str, org_id: str | None = None) -> dict:
    job = get_job(job_id, org_id)
    if job["status"] != "awaiting_approval":
        raise Conflict("This job is not waiting for approval.")
    if sha != job["content_sha"] or sha != current_sha(job):
        raise Conflict("The content changed since you looked at it; review it again before approving.")
    with tx() as con:
        _set(con, job, status="queued", approved_sha=sha, approved_by=by, approved_at=now())
        _ev(con, "job_approved", job, by, sha=sha)
        _ev(con, "job_queued", job, None, actor="system")
    return get_job(job_id)


def update_inputs(job_id: str, org_id: str, by: str, **changes) -> dict:
    """Edit caption/notes/media; clears approval and prepares again."""
    job = get_job(job_id, org_id)
    if job["status"] not in ("requested", "awaiting_approval", "needs_input", "rejected") or _in_flight(job):
        raise Conflict("This job can no longer be edited.")
    inputs = json.loads(job["input_json"])
    inputs.update({k: v for k, v in changes.items() if v is not None})
    inputs = _apply_rules(org_id, get_routine(job["routine_id"]) if job["routine_id"] else None,
                          _clean_inputs(org_id, inputs))
    with tx() as con:
        _set(con, job, input_json=json.dumps(inputs), status="requested", resume_status=None, needs=None,
             needs_from=None, content_sha=content_sha(job["account_id"], job["kind"], inputs,
                                                      get_routine(job["routine_id"]) if job["routine_id"] else None),
             approved_sha=None,
             approved_by=None, approved_at=None)
        _ev(con, "job_edited", job, by)
    return prepare(job_id)


def _in_flight(job: dict) -> bool:
    """A phone run exists (running, or parked on a block while running)."""
    return job["status"] == "running" or (job["status"] == "needs_input" and job["resume_status"] == "running")


def block(job_id: str, needs: str, needs_from: str, resume: str | None = None) -> dict:
    job = get_job(job_id)
    with tx() as con:
        _set(con, job, status="needs_input", resume_status=resume or (job["status"] if job["status"] !=
             "needs_input" else job["resume_status"]), needs=needs, needs_from=needs_from)
        _ev(con, "job_blocked", job, None, actor="system", needs_from=needs_from, needs=needs)
    return get_job(job_id)


def unblock(job_id: str, by: str | None, org_id: str | None = None, note: str | None = None) -> dict:
    job = get_job(job_id, org_id)
    if job["status"] != "needs_input":
        raise Conflict("This job is not waiting on anything.")
    with tx() as con:
        _set(con, job, status=job["resume_status"] or "requested", resume_status=None, needs=None, needs_from=None)
        _ev(con, "job_unblocked", job, by, note=note)
    return prepare(job_id)


def cancel(job_id: str, by: str | None, org_id: str | None = None) -> dict:
    job = get_job(job_id, org_id)
    if job["status"] in FINAL:
        raise Conflict("This job is already finished.")
    if job["status"] == "running" or _in_flight(job):
        raise Conflict("The phone is working on this job now; it can be stopped from the operator console.")
    with tx() as con:
        _set(con, job, status="cancelled")
        _ev(con, "job_cancelled", job, by)
    return get_job(job_id)


def accept(job_id: str, method: str, by: str | None, org_id: str | None = None) -> dict:
    job = get_job(job_id, org_id)
    if job["status"] != "delivered":
        raise Conflict("Only delivered work can be accepted.")
    with tx() as con:
        _set(con, job, status="accepted", accepted_at=now(), acceptance_method=method, accepted_by=by)
        _ev(con, "job_accepted", job, by, actor="system" if method == "objective_criteria" else None, method=method)
    return get_job(job_id)


def reject(job_id: str, reason: str, by: str, org_id: str | None = None, correction: str | None = None) -> dict:
    job = get_job(job_id, org_id)
    if job["status"] not in ("delivered", "awaiting_approval"):
        raise Conflict("Only a draft or delivered work can be rejected.")
    with tx() as con:
        _set(con, job, status="rejected")
        _ev(con, "job_rejected", job, by, reason=reason)
        if correction and correction.strip():
            _add_correction(job["routine_id"], job_id, correction, by, con=con)
    return get_job(job_id)


def not_posted(job_id: str, evidence: str, by: str | None = None) -> dict:
    """Operator-only: the phone run's outcome was unconfirmed, the account was checked and the post is not there.
    The run is abandoned (the adapter's `cancel` when it has one) and the job fails; a retry is a new attempt."""
    job = get_job(job_id)
    if not evidence:
        raise Refused("Say how you checked that nothing was posted.")
    if not job["run_ref"] or not _in_flight(job):
        raise Conflict("Only a job with an unresolved phone run can be closed as not posted.")
    if hasattr(phone, "cancel"):
        phone.cancel(job["run_ref"])
    with tx() as con:
        _set(con, job, status="failed", resume_status=None, needs="Not posted (checked by the operator).",
             needs_from=None)
        _ev(con, "job_failed", job, by, actor="founder", reason="not_posted", evidence=evidence,
            run_ref=job["run_ref"])
    return get_job(job_id)


def retry(job_id: str, by: str | None) -> dict:
    job = get_job(job_id)
    if job["status"] != "failed":
        raise Conflict("Only a failed job can be retried.")
    with tx() as con:
        _set(con, job, status="queued", run_ref=None)
        _ev(con, "job_queued", job, by, retry=True)
    return get_job(job_id)


def add_result(job_id: str, kind: str, output: dict, url: str | None = None, readback: dict | None = None,
               verified: bool = False, simulated: bool = False, con=None) -> str:
    rid = new_id("res")
    row = (rid, job_id, kind, json.dumps(output), url, json.dumps(readback) if readback is not None else None,
           now() if verified else None, int(simulated), now())
    sql = "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    if con is not None:
        con.execute(sql, row)
    else:
        with tx() as c:
            c.execute(sql, row)
    return rid


EXECUTORS = ("phone", "founder")


def record_publication(job_id: str, url: str, evidence: str, by: str | None, simulated: bool = False,
                       executed_by: str = "phone", readback: dict | None = None) -> dict:
    """A verified readback of the live post: the adapter's, or the founder's own check of the account.
    `executed_by='founder'` (posted by hand) is disclosed assistance: the result says so and it counts as a
    founder rescue for A14; only `phone` publications are mobile-worker proof."""
    job = get_job(job_id)
    if executed_by not in EXECUTORS:
        raise Refused("executed_by is phone or founder.")
    if job["kind"] != "publish" or job["status"] not in ("queued", "running", "needs_input"):
        raise Conflict("Only a publish job in progress can be marked live.")
    if executed_by == "phone" and not job["run_ref"]:
        raise Conflict("The phone never started this job; a hand-posted result is executed_by founder.")
    if not job["approved_sha"] or job["approved_sha"] != job["content_sha"] or job["approved_sha"] != current_sha(job):
        raise Conflict("This content was never approved as it stands; nothing may be posted for it.")
    if not url or not evidence:
        raise Refused("Give the post's link and how it was verified.")
    with tx() as con:
        add_result(job_id, "publication", {**_draft(job), "executed_by": executed_by}, url=url,
                   readback={**(readback or {}), "evidence": evidence, "by": by, "executed_by": executed_by},
                   verified=True,
                   simulated=simulated, con=con)
        _set(con, job, status="delivered", resume_status=None, needs=None, needs_from=None)
        _ev(con, "job_delivered", job, by, result="publication", url=url, executed_by=executed_by)
        if executed_by == "founder":
            _ev(con, "founder_rescue", job, by, actor="founder", minutes=None, note="posted by hand")
    return _objective_accept(job_id)


def _objective_accept(job_id: str) -> dict:
    job = get_job(job_id)
    if job["status"] == "delivered" and json.loads(job["input_json"]).get("acceptance") == "objective":
        return accept(job_id, "objective_criteria", None)
    return job


def record_payment(org_id: str | None, person_id: str | None, product: str, amount_cents: int, status: str,
                   processor: str, processor_ref: str | None, evidence: str) -> dict:
    """One row per processor receipt: recording the same (processor, processor_ref) again changes its status (a
    refund), never adds a second payment. A settled payment from a person verifies them as a distinct buyer."""
    if not evidence or not processor_ref:
        raise Refused("A payment needs the processor's receipt id and where the record is (evidence).")
    if product not in ("pilot", "subscription", "review_pack") or status not in ("settled", "pending", "refunded"):
        raise Refused("product is pilot|subscription|review_pack and status settled|pending|refunded.")
    if not isinstance(amount_cents, int) or amount_cents < 0:
        raise Refused("amount_cents is a whole, non-negative number.")
    if org_id:
        get_org(org_id)
    if person_id and not one("SELECT 1 FROM people WHERE id = ?" + (" AND org_id = ?" if org_id else ""),
                             (person_id, org_id) if org_id else (person_id,)):
        raise NotFound("That person is not part of this organization.")
    with tx() as con:
        old = one("SELECT * FROM payments WHERE processor = ? AND processor_ref = ?", (processor, processor_ref))
        if old:
            if (old["org_id"], old["person_id"], old["product"], old["amount_cents"]) != (org_id, person_id, product,
                                                                                          amount_cents):
                raise Conflict("That receipt is already recorded with different details.")
            con.execute("UPDATE payments SET status = ?, evidence = evidence || ' | ' || ? WHERE id = ?",
                        (status, evidence, old["id"]))
            pid = old["id"]
        else:
            pid = new_id("pay")
            con.execute("INSERT INTO payments (id, org_id, person_id, product, amount_cents, currency, status, "
                        "processor, processor_ref, evidence, at) VALUES (?, ?, ?, ?, ?, 'USD', ?, ?, ?, ?, ?)",
                        (pid, org_id, person_id, product, amount_cents, status, processor, processor_ref, evidence,
                         now()))
        event("payment_recorded", actor="founder", org_id=org_id, person_id=person_id, con=con, product=product,
              amount_cents=amount_cents, status=status, updated=bool(old))
        if person_id and status == "settled":
            verify_person(person_id, f"settled {processor} payment {processor_ref}")
    return one("SELECT * FROM payments WHERE id = ?", (pid,))


def record_refund(processor: str, processor_ref: str, refunded_cents: int, evidence: str) -> dict:
    """Refunds on a recorded receipt (stripe-sync). `refunded_cents` is the total refunded so far, so a repeat call
    changes nothing; a full refund makes the payment `refunded` (out of paid counts), a partial one nets the cash."""
    p = one("SELECT * FROM payments WHERE processor = ? AND processor_ref = ?", (processor, processor_ref))
    if not p:
        raise NotFound("No payment with that receipt is recorded.")
    if not isinstance(refunded_cents, int) or not 0 <= refunded_cents <= p["amount_cents"]:
        raise Refused("refunded_cents is the total refunded, between 0 and the payment amount.")
    if refunded_cents == (p["refunded_cents"] or 0):
        return p
    full = refunded_cents == p["amount_cents"] and refunded_cents > 0
    with tx() as con:
        con.execute("UPDATE payments SET refunded_cents = ?, status = ?, evidence = evidence || ' | ' || ? WHERE id = ?",
                    (refunded_cents, "refunded" if full else p["status"], evidence, p["id"]))
        event("payment_refunded", actor="system", org_id=p["org_id"], person_id=p["person_id"], con=con,
              refunded_cents=refunded_cents, full=full)
    return one("SELECT * FROM payments WHERE id = ?", (p["id"],))


# ---- the worker: queued/running publish jobs through phone.py ------------------------------------------------

def _plan(job: dict) -> dict:
    teaching = lane("teaching")
    if job["routine_id"] and teaching and hasattr(teaching, "plan"):
        return teaching.plan(get_routine(job["routine_id"]), job)
    a, inputs = get_account(job["account_id"]), json.loads(job["input_json"])
    m = inputs["media"][0]
    return {"v": 1, "identity": a["identity"], "media": {"path": m["path"], "sha256": m["sha256"]},
            "caption": inputs.get("caption") or "", "account": {"id": a["id"], "platform": a["platform"],
                                                                "handle": a["handle"]}, "draft": _draft(job)}


def _start(job: dict, run_ref: str) -> dict:
    plan = _plan(job)
    if plan.get("status", "ready") != "ready":
        return block(job["id"], plan.get("needs") or "The routine needs an answer before it can run.", "customer",
                     resume="queued")
    try:
        phone.submit(job, plan, job["approved_sha"], run_ref)
    except Exception as e:  # the adapter refuses before any effect (its contract)
        return block(job["id"], f"The phone run could not start: {e}", "founder", resume="queued")
    return get_job(job["id"])


def work_once() -> int:
    """One pass over queued and running publish jobs, and runs parked on a phone-side wait (Ian's 👍, the phone
    unplugged): those keep being polled and resume by themselves when the wait clears. Returns how many changed."""
    changed = 0
    for job in query("SELECT * FROM jobs WHERE kind = 'publish' AND (status IN ('queued', 'running') OR (status = "
                     "'needs_input' AND resume_status = 'running' AND run_ref IS NOT NULL)) ORDER BY updated"):
        before = (job["status"], job["run_ref"], job["needs"])
        try:
            job = _step(job)
        except Conflict:  # changed under us (a cancel, an edit): the next pass sees the new state
            continue
        changed += (job["status"], job["run_ref"], job["needs"]) != before
    return changed


def _step(job: dict) -> dict:
    a = get_account(job["account_id"])
    if job["status"] == "queued":
        now_sha = current_sha(job)  # the routine may have changed since approval: approve what will run
        if job["approved_sha"] != now_sha or job["approved_sha"] != job["content_sha"]:
            with tx() as con:
                _set(con, job, status="awaiting_approval", content_sha=now_sha, approved_sha=None, approved_by=None,
                     approved_at=None)
                _ev(con, "job_approval_cleared", job, None, actor="system")
            return get_job(job["id"])
        p = permission(a["id"])
        if not p or p["mode"] not in PUBLISH_MODES:
            return block(job["id"], f"Allow publishing on @{a['handle']} again.", "customer", resume="queued")
        if a["authority"] != "verified":
            return block(job["id"], f"Verify the owner's authority over @{a['handle']}.", "founder", resume="queued")
        if not a["identity"]:
            return block(job["id"], f"Map @{a['handle']} to its device identity (operator).", "founder",
                         resume="queued")
        plan = _plan(job)
        if plan.get("status", "ready") != "ready":  # the routine needs an answer first (teaching.plan)
            return block(job["id"], plan.get("needs") or "The routine needs an answer before it can run.", "customer",
                         resume="queued")
        r = phone.readiness(a)
        set_access(a["id"], "usable" if r["usable"] else "broken", r.get("reason") or "")
        if not r["usable"]:
            return block(job["id"], r.get("reason") or "The phone is not ready.", "founder", resume="queued")
        run_ref = f"{job['id']}#{job['attempts'] + 1}"
        with tx() as con:  # stored before the phone is touched: a restart polls this run, never starts another
            # readiness took time: the grant, authority and identity must still hold in the transaction that commits
            now_a, now_p = get_account(a["id"]), permission(a["id"])
            if not now_p or now_p["mode"] not in PUBLISH_MODES or now_p["id"] != p["id"] or \
                    (now_a["authority"], now_a["identity"]) != (a["authority"], a["identity"]):
                raise Conflict("The account's permission changed while the phone was being checked.")
            _set(con, job, status="running", run_ref=run_ref, attempts=job["attempts"] + 1)
            _ev(con, "job_started", job, None, actor="system", run_ref=run_ref)
        return _start(get_job(job["id"]), run_ref)
    s = phone.poll(job["run_ref"])
    parked = job["status"] == "needs_input"
    if s["state"] == "unknown":  # stored but never reached the adapter (or it lost it): same run_ref, idempotent
        return job if parked else _start(job, job["run_ref"])  # a parked run waits for the founder, never restarts
    if s["state"] == "needs":
        needs = s.get("needs") or "The phone needs attention."
        return job if parked and job["needs"] == needs else block(job["id"], needs, "founder", resume="running")
    if parked and s["state"] == "running":  # the wait cleared on the phone side
        with tx() as con:
            _set(con, job, status="running", resume_status=None, needs=None, needs_from=None)
            _ev(con, "job_unblocked", job, None, actor="system", note="the phone resumed the run")
        return get_job(job["id"])
    if s["state"] == "failed":
        with tx() as con:
            _set(con, job, status="failed", needs=s.get("needs"))
            _ev(con, "job_failed", job, None, actor="system", reason=s.get("needs"))
        return get_job(job["id"])
    if s["state"] == "done":
        res = s.get("result") or {}
        if res.get("verified_at") and res.get("url"):
            return record_publication(job["id"], res["url"], "phone adapter readback", None,
                                      simulated=bool(res.get("simulated")), readback=res.get("readback") or {})
        needs = (f"The phone finished but nothing confirmed the post on @{a['handle']}: check the account and record "
                 "the live link.")
        return job if parked and job["needs"] == needs else block(job["id"], needs, "founder", resume="running")
    return job


# ---- views ---------------------------------------------------------------------------------------------------

def summary(org_id: str) -> list[dict]:
    """Per account: what needs the customer, what awaits their review, what was delivered and where it went live."""
    out = []
    for a in query("SELECT * FROM accounts WHERE org_id = ? ORDER BY created", (org_id,)):
        jobs = query("SELECT * FROM jobs WHERE account_id = ? ORDER BY requested_at DESC", (a["id"],))
        by: dict[str, int] = {}
        for j in jobs:
            by[j["status"]] = by.get(j["status"], 0) + 1
        live = query("SELECT r.url, r.verified_at, r.job_id FROM results r JOIN jobs j ON j.id = r.job_id WHERE "
                     "j.account_id = ? AND r.kind = 'publication' AND r.verified_at IS NOT NULL AND r.simulated = 0 "
                     "ORDER BY r.verified_at DESC LIMIT 10", (a["id"],))
        out.append({"account": account_view(a), "jobs_by_status": by,
                    "needs_you": [{"job_id": j["id"], "needs": j["needs"]} for j in jobs
                                  if j["status"] == "needs_input" and j["needs_from"] == "customer"],
                    "awaiting_your_approval": [j["id"] for j in jobs if j["status"] == "awaiting_approval"],
                    "awaiting_your_review": [j["id"] for j in jobs if j["status"] == "delivered"],
                    "in_progress": [j["id"] for j in jobs if j["status"] in ("queued", "running") or
                                    (j["status"] == "needs_input" and j["needs_from"] == "founder")],
                    "live_posts": live})
    return out


def job_view(j: dict, detail: bool = False) -> dict:
    a = get_account(j["account_id"])
    r = get_routine(j["routine_id"]) if j["routine_id"] else None
    inputs = json.loads(j["input_json"])
    inputs["media"] = [{"sha256": m["sha256"], "name": m.get("name")} for m in inputs["media"]]
    out = {**j, "input": inputs,
           "account": {"id": a["id"], "platform": a["platform"], "handle": a["handle"],
                       "instructions": a["instructions"]},
           "routine": {"id": r["id"], "name": r["name"], "version": r["version"]} if r else None,
           "results": [{**{k: x[k] for k in ("id", "kind", "url", "verified_at", "simulated", "created")},
                        "output": json.loads(x["output_json"])}
                       for x in query("SELECT * FROM results WHERE job_id = ? ORDER BY created", (j["id"],))]}
    if detail:
        out["preview"] = {"account": out["account"], "kind": j["kind"], "caption": inputs.get("caption") or "",
                          "content_sha": j["content_sha"],
                          "media": [{**m, "url": f"/api/media/{m['sha256']}"} for m in inputs["media"]]}
    return out


def jobs_for(org_id: str, open_only: bool = False) -> list[dict]:
    where = f" AND status IN ({','.join('?' * len(OPEN))})" if open_only else ""
    rows = query(f"SELECT * FROM jobs WHERE org_id = ?{where} ORDER BY requested_at DESC",
                 (org_id, *(OPEN if open_only else ())))
    return [job_view(j) for j in rows]
