"""Independent review (azure/tw-phone-review) of taught_run's read-only claim. SIMULATED: a fake missions module and
fake looks, as test_teaching's own taught-steps test; a routine imported through teaching.import_workflow (what
POST /api/teach/... lets a person do). Each test asserts what a read-only run must never do; nothing reaches a phone."""

import contextlib
import copy
import re
import time as _time
import types

import pytest

from teachable_worker import teaching
from test_teaching import account_check_fixture, org_with


def _it(text, x, y, h=56):
    return {"text": text, "x": x, "y": y, "w": 22 * len(text), "h": h}


def _look(fid, *items):
    return {"frame_id": fid, "captured_at": fid, "width": 1170, "height": 2532, "items": list(items)}


KEYS = (_it("Q W E R T Y U I O P", 585, 1900), _it("space", 585, 2300))


def _harness(monkeypatch, looks, retry_look=None):
    """missions as taught_run uses it; `act` records what it would tap. With retry_look, act's first try is refused with
    zero input (as the daemon does) and the step is re-resolved on that fresh look, as missions.act does."""
    taps, typed = [], []
    monkeypatch.setattr(_time, "sleep", lambda s: None)

    class Stop(RuntimeError):
        pass

    def act(ctx, o, stops, resolve, note):
        if retry_look is not None and resolve(o):
            o = retry_look  # missions.act: ZeroInput → look again → resolve AGAIN on the fresh look
        spec = resolve(o)
        if spec:
            taps.append(spec[3] | {"what": spec[1]})
        return o if spec else None
    m = types.SimpleNamespace(
        Stop=Stop, NEVER=re.compile(r"^(follow|follow back|following|message|like|share|send|reply|comment|save|subscribe|"
                                    r"not interested|add to favorites|seguir|suivre|requested|mute|block|report|restrict)\b", re.I),
        TASKS={}, _stops=lambda: [], _open_ig=lambda ctx, stops: looks.pop(0), _look=lambda ctx, note, stops: looks.pop(0),
        act=act, tab_screen=lambda items, h: True,
        keyboard_up=lambda items, h: any(teaching.norm(i["text"]) == "space" for i in items))
    monkeypatch.setattr(teaching.TaughtRun, "_missions", staticmethod(lambda: m))
    return m, taps, typed


def _ctx(routine_id, typed, first):
    class Ctx:
        task, device = {"id": "t1", "input": {"routine_id": routine_id, "inputs": {"handle": "nasa"}}}, {"id": "iphone12"}

        @contextlib.contextmanager
        def session(self):
            yield self

        def hands(self, ctx, kind, what, tool, **args):
            typed.append(args.get("text"))

        def observe(self, note=None):
            return first

        def step(self, *a, **k):
            pass

        def check(self):
            pass
    return Ctx()


def _imported(tmp_path, steps):
    """An account-check workflow exported from a taught routine, its steps replaced, imported and confirmed elsewhere."""
    org, owner, acme = org_with()
    rec = teaching.ingest_session(org["id"], account_check_fixture(tmp_path / "s"), "Check @studio_fixture's profile.",
                                  owner["id"], acme["id"], label="agent")
    wf = copy.deepcopy(teaching.export_workflow(teaching.derive_routine(rec, "Account check")))
    wf["steps"] = steps
    org2, owner2, other = org_with("bright_bakery")
    r = teaching.import_workflow(org2["id"], wf, owner2["id"], other["id"])
    return teaching.confirm(r["id"], owner2["id"])


def test_an_imported_routine_never_comments_on_a_read_only_run(tmp_path, monkeypatch):
    """No irreversible flag, no NEVER word: 'Add a comment…' then the handle typed, then 'Post'. taught_run is read-only."""
    r = _imported(tmp_path, [
        {"op": "tap", "target": {"label": "Add a comment…"}, "screen": {}, "expect": {}},
        {"op": "type", "value": {"param": "handle"}, "screen": {}, "expect": {}},
        {"op": "tap", "target": {"label": "Post"}, "screen": {}, "expect": {}},
    ])
    post = _look("p", _it("nasa", 585, 220), _it("Add a comment…", 400, 2200), _it("Post", 1080, 2200))
    typing = _look("k", _it("nasa", 585, 220), _it("Add a comment…", 400, 1500), _it("Post", 1080, 1500), *KEYS)
    m, taps, typed = _harness(monkeypatch, [post, typing, typing, typing, typing, typing])
    with contextlib.suppress(teaching.TeachError):
        teaching.TASKS["taught_run"].run(_ctx(r["id"], typed, post))
    assert not taps and not typed, (taps, typed)  # refused before any input: not a supported read-only step


def test_a_read_only_run_types_only_into_the_search_field(tmp_path, monkeypatch):
    """A raised keyboard isn't the search field: a DM or comment box with the keyboard up must not get the text."""
    r = _imported(tmp_path, [{"op": "type", "value": {"param": "handle"}, "screen": {}, "expect": {}}])
    dm = _look("d", _it("ian_friend", 585, 180), _it("Message...", 400, 1500), *KEYS)
    m, taps, typed = _harness(monkeypatch, [dm, dm, dm])
    with contextlib.suppress(teaching.TeachError):
        teaching.TASKS["taught_run"].run(_ctx(r["id"], typed, dm))
    assert not typed, typed


def test_the_top_bar_icon_gate_is_decided_again_on_the_retry_look(tmp_path, monkeypatch):
    """field_clear is computed on the look before act; after a zero-input refusal act re-resolves on a fresh look, which
    no longer shows a focused field with an old query (here: the keyboard went down on a profile). No blind tap there."""
    r = _imported(tmp_path, [{"op": "tap", "target": {"icon_at": [0.95, 0.08], "what": "clear"}, "screen": {}, "expect": {}}])
    typing = _look("t", _it("natgeo", 400, 230), _it("Cancel", 1060, 230), *KEYS)
    profile = _look("p", _it("nasa", 585, 200), _it("Follow", 400, 900), _it("Message", 800, 900))
    m, taps, typed = _harness(monkeypatch, [typing, profile, profile], retry_look=profile)
    with contextlib.suppress(teaching.TeachError, Exception):
        teaching.TASKS["taught_run"].run(_ctx(r["id"], typed, typing))
    assert not taps, taps
