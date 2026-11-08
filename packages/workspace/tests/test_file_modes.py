"""Private state is private under any umask: the state and data dirs 0700, the database (+ -wal/-shm), screenshots,
uploads and the admin token 0600; an existing install is tightened on first use; a user's own directory (a TW_DB
parent it chose) and anything behind a symlink are never chmodded."""

import os
import sqlite3
import stat

import pytest

from teachable_worker import agency, httpd, store, workspace


def mode(p):
    return stat.S_IMODE(os.lstat(p).st_mode)


@pytest.fixture(params=[0o022, 0o002, 0o000])
def umask(request):
    old = os.umask(request.param)
    yield request.param
    os.umask(old)


@pytest.fixture
def home(tw, monkeypatch):
    """The defaults (no TW_DB/TW_DATA) under a fresh HOME: ~/.local/state/teachable-worker is the app's own dir."""
    monkeypatch.setenv("HOME", str(tw / "home"))
    for k in ("TW_DB", "TW_DATA", "TW_ADMIN_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(store, "_migrated", set())
    return tw / "home" / ".local" / "state" / "teachable-worker"


def test_a_fresh_install_is_private(home, umask):
    store.query("SELECT 1")
    f = workspace.save_frame("phone", b"\xff\xd8 jpeg", None, "test")
    httpd.admin_token()
    org = agency.create_org("Me", "agency", "founder")
    agency.save_media(org["id"], "clip.mp4", b"video bytes")
    data = home / "data"
    assert {p: mode(p) for p in (home, data, data / "frames", data / "frames" / "phone", data / "media")} == \
        dict.fromkeys((home, data, data / "frames", data / "frames" / "phone", data / "media"), 0o700)
    for p in [home / "tw.sqlite", home / "tw.sqlite-wal", home / "tw.sqlite-shm", home / "admin.token", f["path"],
              *(data / "media").iterdir()]:
        if os.path.exists(p):
            assert mode(p) == 0o600, p
    live = sqlite3.connect(home / "tw.sqlite")  # while a connection writes, SQLite keeps -wal/-shm: they take the db's mode
    live.execute("CREATE TABLE IF NOT EXISTS probe (x)")
    live.execute("INSERT INTO probe VALUES (1)")
    live.commit()
    try:
        assert mode(home / "tw.sqlite-wal") == 0o600 and mode(home / "tw.sqlite-shm") == 0o600
    finally:
        live.close()


def test_an_existing_install_is_tightened_on_first_use(home, umask):
    data = home / "data"
    (data / "frames" / "phone").mkdir(parents=True)
    (data / "media").mkdir()
    for d in (home, data, data / "frames", data / "frames" / "phone", data / "media"):
        d.chmod(0o775)
    con = sqlite3.connect(home / "tw.sqlite")
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("CREATE TABLE t (x)")
    old_wal = home / "tw.sqlite-wal"
    for p in (home / "tw.sqlite", old_wal, home / "tw.sqlite-shm"):
        if p.exists():
            p.chmod(0o644)
    (home / "admin.token").write_text("old-token")
    (home / "admin.token").chmod(0o644)
    shot, upload = data / "frames" / "phone" / "frm_old.jpg", data / "media" / "abc.mp4"
    for p in (shot, upload):
        p.write_bytes(b"x")
        p.chmod(0o644)
    elsewhere = home.parent / "not-ours.txt"
    elsewhere.write_text("a user's file")
    elsewhere.chmod(0o444)
    (data / "frames" / "phone" / "planted.jpg").symlink_to(elsewhere)
    store.query("SELECT 1")
    con.close()
    for d in (home, data, data / "frames", data / "frames" / "phone", data / "media"):
        assert mode(d) == 0o700, d
    for p in (home / "tw.sqlite", home / "admin.token", shot, upload, *[p for p in (old_wal, home / "tw.sqlite-shm") if p.exists()]):
        assert mode(p) == 0o600, p
    assert mode(elsewhere) == 0o444  # behind a symlink: never followed


def test_a_user_chosen_database_folder_is_left_alone(tw, monkeypatch, umask):
    mine = tw / "projects"
    mine.mkdir()
    mine.chmod(0o755)
    monkeypatch.setenv("TW_DB", str(mine / "tw.sqlite"))
    monkeypatch.setattr(store, "_migrated", set())
    store.query("SELECT 1")
    assert mode(mine) == 0o755 and mode(mine / "tw.sqlite") == 0o600


def test_a_user_chosen_data_folder_keeps_its_mode_but_the_apps_folders_inside_are_private(tmp_path, monkeypatch):
    """TW_DATA pointing at a folder the user already had: that folder isn't chmodded; frames/ and media/ inside are."""
    import os
    from teachable_worker import store
    mine = tmp_path / "my-data"
    mine.mkdir(mode=0o755)
    os.chmod(mine, 0o755)
    (mine / "frames").mkdir(mode=0o755)
    monkeypatch.setenv("TW_DATA", str(mine))
    monkeypatch.setenv("TW_DB", str(tmp_path / "db" / "tw.sqlite"))
    store._migrated.clear()
    store.query("SELECT 1")
    assert mine.stat().st_mode & 0o777 == 0o755
    assert (mine / "frames").stat().st_mode & 0o777 == 0o700
