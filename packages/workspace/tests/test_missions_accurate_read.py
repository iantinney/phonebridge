"""account_check through the REAL workspace (run_task, Ctx, frames table) when the look's fast OCR dropped a one-digit
count (live 11-08, cut a96955d: @apple's '9 following'). The bridge is a fake that answers like phonebridge (look, ocr,
foreground, presence) from a screen state the fake hands move; nothing reaches a phone. tw-verifier's condition for the
final cut: the accurate still is stored with its own text, and every read and the result frame come from it alone."""
import base64
import contextlib
import io
import json

from PIL import Image

from teachable_worker import missions, workspace
from teachable_worker.store import now, query, tx

W, H = 1170, 2532
GRID = "instagram.screen.instagram_explore_search_grid"


def _jpeg(shade: int) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (W, H), (shade, shade, shade)).save(b, "JPEG", quality=60)
    return b.getvalue()


def _i(text, x, y, w=200, h=50):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "conf": 1.0}


HEADER = [_i("apple", 270, 210), _i("1,413", 433, 443), _i("38.1M", 667, 441), _i("posts", 421, 499),
          _i("followers", 679, 494), _i("following", 959, 496), _i("Follow", 280, 970), _i("Message", 770, 975)]
SCREENS = {  # state -> (items, the daemon's screen id)
    "home": ([_i("Instagram", 300, 160), _i("Your story", 160, 620)], "instagram.home"),
    "own": ([_i("samplecreator.co v", 300, 205), _i("Edit profile", 300, 935), _i("Share profile", 780, 935)],
            "instagram.profile"),
    "grid": ([_i("G Search with Meta Al", 359, 213, 526)], GRID),
    "typing": ([_i("Search with Meta Al", 458, 213, 526), _i("Recent", 130, 388), _i("See all", 1053, 388)],
               "instagram.search.typing"),
    "results": ([_i("apple", 300, 213), _i("apple • 38.1M followers", 520, 506, 600)], None),
    "profile": (HEADER, "instagram.profile.other"),  # the fast look: no '9'
}
MOVES = {"Tap the Profile tab": "own", "Tap the Search tab": "grid", "Tap the search field": "typing",
         "Type apple": "results", "Open apple": "profile", "Back to the Home Screen": "home"}


def test_a_dropped_one_digit_count_is_read_from_one_accurate_still_through_the_real_workspace(monkeypatch):
    state, calls, inputs = {"s": "home"}, [], []
    look_jpeg, accurate_jpeg = _jpeg(20), _jpeg(40)

    def tool(name, **args):
        calls.append((name, args))
        if name == "foreground":
            return {"structuredContent": {"result": missions.IG}}
        if name == "look":
            items, screen = SCREENS[state["s"]]
            return {"structuredContent": {"frame_jpeg": base64.b64encode(look_jpeg).decode(), "items": items,
                                          "screen": screen, "width": W, "height": H}}
        if name == "ocr":
            assert state["s"] == "profile" and args == {"device": "iphone12", "fast": False, "with_image": True}
            return {"structuredContent": {"result": {"items": HEADER + [_i("9", 889, 443, 20)], "width": W, "height": H,
                                                     "image_jpeg": base64.b64encode(accurate_jpeg).decode()}}}
        raise AssertionError(f"unexpected bridge call {name}")

    b = workspace.Bridge("http://127.0.0.1:1")
    monkeypatch.setattr(b, "tool", tool)
    monkeypatch.setattr(b, "presence", lambda device: {"foreground": missions.IG})
    monkeypatch.setattr(workspace, "bridge", b)
    # the account the fixture's own profile shows, pinned here: never a deployment's own account map
    monkeypatch.setattr(missions, "ACCOUNTS", {"iphone12": "ig:test"})
    monkeypatch.setattr(missions, "ACCOUNT_HANDLES", {"ig:test": "samplecreator.co"})

    class Hands:
        device = "iphone12"

        def __call__(self, ctx, kind, what, tool, **args):
            inputs.append((kind, what))
            state["s"] = MOVES.get(what, state["s"])

    @contextlib.contextmanager
    def session(ctx):
        ctx.hands = Hands()
        yield

    monkeypatch.setattr(workspace, "session_hook", lambda: session)
    monkeypatch.setattr(missions, "_wait", lambda ctx, why="reading": None)
    monkeypatch.setattr(missions.time, "sleep", lambda s: None)
    with tx() as con:
        con.execute("INSERT INTO tasks (id, device_id, kind, input_json, state, origin, created, updated) VALUES "
                    "('tsk_apple', 'iphone12', 'account_check', ?, 'queued', 'test', ?, ?)",
                    (json.dumps({"handles": "apple"}), now(), now()))
    workspace.run_task(workspace.get_task("tsk_apple"))

    t = workspace.get_task("tsk_apple")
    result = json.loads(t["result_json"]) if t.get("result_json") else t.get("result")
    assert t["state"] == "done", (t["state"], t.get("error"), result)
    prof = result["profiles"][0]
    assert prof["found"] and prof["handle_seen"] == "apple"
    assert prof["counts"] == {"posts": {"value": 1413, "shown": "1,413"}, "followers": {"value": 38100000, "shown": "38.1M"},
                              "following": {"value": 9, "shown": "9"}}
    ocr_calls = [c for c in calls if c[0] == "ocr"]
    assert ocr_calls == [("ocr", {"device": "iphone12", "fast": False, "with_image": True})]  # exactly one accurate read
    f = query("SELECT * FROM frames WHERE id = ?", (prof["frame_id"],))[0]  # the result frame IS the accurate still
    assert f["source"] == "bridge ocr (accurate)" and open(f["path"], "rb").read() == accurate_jpeg
    assert "9" in [i["text"] for i in json.loads(f["items_json"])]
    assert [k for k, _ in inputs] == ["tap", "tap", "tap", "type", "tap", "tap"]  # Profile, Search, field, type, row, Home
