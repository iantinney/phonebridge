"""SCR-08/13, TST-01: the corpus outside git, phash dedupe, unknown queue, promotion, the manifest."""
import pytest

from phonebridge.screens.cascade import Frame
from phonebridge.screens.schema import Record
from phonebridge.screens.store import Store, phash
from screens_fixtures.synth import frame as synth_frame, W, H


HOME = Record.from_dict({"id": "instagram.home", "platform": "instagram", "bundle": "com.burbn.instagram",
                         "status": "confirmed", "match": {"ocr_any": ["Instagram", "Your story"], "at_least": 2}})


def test_save_frame_dedupes_by_phash_and_caps(tmp_path):
    store = Store(records=[HOME], corpus=tmp_path / "corpus", cap_per_screen=3)
    f = Frame(png=synth_frame(top="pink", seed=1), ocr=[], width=W, height=H)
    p = store.save_frame("instagram", "instagram.home", f, {"app_version": "400.0", "prev": None, "action": None})
    assert p and p.exists() and (p.stat().st_mode & 0o777) == 0o600 and p.with_suffix(".ocr.json").exists()
    assert store.save_frame("instagram", "instagram.home", f, {}) is None  # same phash: dropped
    for i in range(2, 6):
        store.save_frame("instagram", "instagram.home", Frame(png=synth_frame(top="pink", body="noise", seed=i), ocr=[],
                                                              width=W, height=H), {})
    assert len(list((tmp_path / "corpus" / "instagram" / "instagram.home").glob("*.png"))) <= 3


def test_unknown_queue_and_manifest(tmp_path):
    store = Store(records=[HOME], corpus=tmp_path / "corpus")
    f = Frame(png=synth_frame(top="gray", seed=1), ocr=[{"text": "Hello", "x": 1, "y": 1, "w": 1, "h": 1, "conf": 1}],
              width=W, height=H)
    p = store.queue_unknown("instagram", f, {"prev": "instagram.home", "action": "tap create"})
    assert "unknown" in p.parts and p.with_suffix(".ocr.json").exists()
    assert store.stats()["unknown"] == 1
    (tmp_path / "corpus" / "manifest.yaml").write_text("screens:\n  instagram.home: [instagram/instagram.home/x.png]\n")
    assert store.manifest_gaps() == ["instagram.home: instagram/instagram.home/x.png missing"]


def test_record_dirs_load_candidates_beside_repo_records(tmp_path):
    repo = tmp_path / "records" / "instagram"
    repo.mkdir(parents=True)
    HOME.dump(repo / "home.yaml")
    store = Store(records_dirs=[tmp_path / "records"], corpus=tmp_path / "corpus")
    assert "instagram.home" in store.records
    cand = Record.from_dict({"id": "instagram.popup.x", "platform": "instagram", "bundle": "com.burbn.instagram",
                             "kind": "popup", "answer": "no", "targets": {"no": {"ocr": "Not now"}},
                             "match": {"ocr_any": ["Not now", "Enable"], "at_least": 2}, "source": "vlm"})
    store.add_candidate(cand, Frame(png=synth_frame(top="gray", seed=3), ocr=[], width=W, height=H), {})
    store2 = Store(records_dirs=[tmp_path / "records"], corpus=tmp_path / "corpus")
    assert store2.records["instagram.popup.x"].status == "candidate" and store2.records["instagram.popup.x"].evidence
    assert phash(synth_frame(top="gray", seed=3)) == phash(synth_frame(top="gray", seed=3))


def test_manifest_keeps_other_worktrees_records(tmp_path):
    """The corpus is shared by every lane's worktree: regenerating the manifest from one worktree must not drop the
    xfail notes of records only another worktree has (lane tiktok's irreversible screens vanished that way)."""
    import yaml
    from phonebridge.screens.harvest import write_manifest
    from phonebridge.screens.schema import Record
    store = Store(records=[Record.from_dict({"id": "instagram.reel.final", "platform": "instagram", "bundle": "x",
                                             "match": {"ocr_any": ["Share"]}, "targets": {"share": {"ocr": "Share", "irreversible": True}}})],
                  corpus=tmp_path / "corpus")
    (tmp_path / "corpus").mkdir(parents=True, exist_ok=True)
    (tmp_path / "corpus" / "manifest.yaml").write_text(yaml.safe_dump(
        {"irreversible": {"tiktok.post.final": "xfail: 0 < 5", "instagram.gone": "xfail: stale"},
         "alerts": {"tiktok.popup.discard": "xfail: no frame yet"}}))
    m = yaml.safe_load(write_manifest(store).read_text())
    assert m["irreversible"]["instagram.reel.final"].startswith("xfail: 0 < 5")
    assert m["irreversible"]["tiktok.post.final"] == "xfail: 0 < 5" and m["alerts"]["tiktok.popup.discard"] == "xfail: no frame yet"


def test_corpus_candidates_with_newer_keys_still_load(tmp_path):
    """The live corpus is shared by every tree: a record harvested by a newer schema (an unknown key) must not break
    an older tree's suite; repo records stay strict."""
    import yaml
    cand = tmp_path / "corpus" / "candidates" / "instagram"
    cand.mkdir(parents=True)
    d = {**HOME.to_dict(), "id": "instagram.popup.future", "kind": "popup", "answer": "ok", "targets": {"ok": {"ocr": "OK"}},
         "match": {"ocr_any": ["OK", "Later"], "at_least": 2}, "future_key": {"x": 1}}
    (cand / "instagram.popup.future.yaml").write_text(yaml.safe_dump(d))
    store = Store(records_dirs=[], corpus=tmp_path / "corpus")
    assert "instagram.popup.future" in store.records and "future_key" in (store.records["instagram.popup.future"].note or "")
    repo = tmp_path / "records" / "instagram"
    repo.mkdir(parents=True)
    (repo / "x.yaml").write_text(yaml.safe_dump(d))
    with pytest.raises(Exception):
        Store(records_dirs=[tmp_path / "records"], corpus=tmp_path / "corpus2")


def test_refresh_sees_a_record_saved_by_another_process_and_save_syncs(tmp_path, monkeypatch):
    """G3 sync: a store re-reads its dirs when a file appears (the Mac daemon sees a VM-learned record at once),
    and a learned save pushes the file to the other store (rsync, in the background)."""
    import yaml
    from phonebridge.screens.cascade import Cascade, Frame
    corpus = tmp_path / "corpus"
    store = Store(records=[HOME], corpus=corpus, sync_to="mac:~/corpus/candidates")
    assert "instagram.popup.late" not in store.records
    cand = corpus / "candidates" / "instagram"
    cand.mkdir(parents=True)
    d = {**HOME.to_dict(), "id": "instagram.popup.late", "kind": "popup", "answer": "ok", "targets": {"ok": {"ocr": "OK"}},
         "match": {"ocr_any": ["Late popup", "OK"], "at_least": 2}, "source": "vlm"}
    (cand / "instagram.popup.late.yaml").write_text(yaml.safe_dump(d))  # as if written by another process/host
    f = Frame(ocr=[{"text": "Late popup", "x": 585, "y": 1200, "w": 300, "h": 40, "conf": 1.0},
                   {"text": "OK", "x": 585, "y": 1500, "w": 60, "h": 40, "conf": 1.0}], width=W, height=H)
    assert Cascade(store).recognise(f, platform="instagram").screen_id == "instagram.popup.late"  # refreshed on recognise
    assert "instagram.home" in store.records  # in-memory records survive the reload
    ran = []
    monkeypatch.setattr(Store, "_sync_runner", staticmethod(lambda argv, log: ran.append(argv)))
    rec = store.records["instagram.popup.late"]
    store.success(rec.id, "run1:abc")
    assert ran and ran[0][0] == "rsync" and ran[0][-1] == "mac:~/corpus/candidates/instagram/" and ran[0][-2].endswith("instagram.popup.late.yaml")
    assert (corpus / "sync.log").read_text().count("instagram.popup.late.yaml") == 1
    pushed = yaml.safe_load(open(ran[0][-2]).read())  # the pushed copy carries only keys every deployed schema knows
    rec.header = (0.0, 0.1, 1.0, 0.2); rec.avoid = [(0.0, 0.9, 1.0, 1.0)]
    store.success(rec.id, "run2:def")
    pushed = yaml.safe_load(open(ran[-1][-2]).read())
    assert "header" not in pushed and "avoid" not in pushed and pushed["successes"] == 2 and "pushed without" in pushed["note"]
    assert "header" in yaml.safe_load(open(cand / "instagram.popup.late.yaml").read())  # the local file keeps the full record
