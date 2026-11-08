"""Feed maps as data (director/feedmaps.py): IG/TikTok agree with the director's dicts until it reads this module; X is
read-only; every screen a kind's step expects has a record; X's like state reads pink on the heart and nothing on the
real (unliked) 10-19 rows."""
import io
import json

import pytest
from PIL import Image, ImageDraw

from phonebridge.director import feedmaps as fm
from phonebridge.director import kinds, primitives, session
from phonebridge.screens.store import CORPUS, RECORD_DIRS, Store

W, H = 1170, 2532
XSEQ = CORPUS / "sequences" / "surfaces_x"


def test_ig_and_tiktok_match_the_directors_dicts():
    for app in ("instagram", "tiktok"):
        m = fm.FEEDMAPS[app]
        assert m.feed == kinds.FEED[app] == primitives.FEED_IDS[app]
        assert set(m.feed) <= set(session.FEED_SCREENS[app]) and m.home_tab == session.HOME_TAB[app]
        assert set(m.kinds) <= set(kinds.KINDS)


def test_x_is_read_only():
    x = fm.FEEDMAPS["x"]
    ks = fm.kinds_for("x")
    assert x.read_only and set(ks) == {"thread", "profile"}
    assert all(k.group != "write" and not any(s.action.split(":")[-1] in ("media_double", "comment_icon", "follow") for s in k.steps) for k in ks.values())
    assert x.back is fm.x_back_button and x.like_state is fm.x_like_state  # verification only: no like kind on X


def test_every_expected_screen_has_a_record():
    ids = set(Store(records_dirs=RECORD_DIRS).records)
    for app, m in fm.FEEDMAPS.items():
        assert all(f in ids for f in m.feed if not f.endswith(("reels", "foryou"))), (app, m.feed)
        for k in fm.kinds_for(app).values():
            for st in k.steps:
                want = [e for e in st.expect if e != "*feed"]  # (ids declared before their records land: kinds.kind_ready keeps their kinds off live)
                assert not want or any(i == e or i.startswith(e + ".") for e in want for i in ids) or set(want) <= kinds.PENDING_RECORDS, (app, k.name, st.action, want)


def frames():
    for p in sorted(XSEQ.glob("*/*.ocr.json")):
        yield json.loads(p.read_text()), p.with_suffix("").with_suffix(".png").read_bytes()


@pytest.mark.skipif(not XSEQ.exists(), reason=f"no X captures at {XSEQ}")
def test_x_action_rows_found_and_all_unliked_on_real_frames():
    rows = 0
    for items, png in frames():
        for y in fm.x_action_rows(items, W, H):
            rows += 1
            assert not fm.x_like_state(png, y, W, H), y
    assert rows >= 10


def test_x_like_state_reads_a_pink_heart():
    im = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([0.5 * W - 25, 1300 - 25, 0.5 * W + 25, 1300 + 25], fill=fm.X_PINK)
    b = io.BytesIO()
    im.save(b, "PNG")
    assert fm.x_like_state(b.getvalue(), 1300, W, H) and not fm.x_like_state(b.getvalue(), 1700, W, H)


@pytest.mark.skipif(not XSEQ.exists(), reason=f"no X captures at {XSEQ}")
@pytest.mark.parametrize("frame,want", [("1/01-home", "for_you"), ("3/05-back2_button", "for_you"), ("3/06-following", "following"),
                                        ("1/03-scroll2", None)])
def test_x_selected_tab_from_the_underline(frame, want):
    """T-X-02/03 (PLATFORM_REQUIREMENTS): which home tab is selected, from its underline, never the label alone."""
    items = json.loads((XSEQ / f"{frame}.ocr.json").read_text())
    assert fm.x_selected_tab((XSEQ / f"{frame}.png").read_bytes(), items, W, H) == want
