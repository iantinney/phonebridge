import pytest

HOME = [{"text": "9:41", "x": 15, "y": 6, "w": 8, "h": 4, "conf": 1.0},  # small status-bar time
        {"text": "Settings", "x": 98, "y": 190, "w": 20, "h": 6, "conf": 1.0}]


@pytest.fixture(autouse=True)
def fake_ocr(monkeypatch):
    """Vision is macOS-only: tests set what the screen 'reads' (a list of OCR results per screenshot)."""
    reads = []
    monkeypatch.setattr("phonebridge.bridge.vision.recognize", lambda png, fast=False: reads.pop(0) if reads else HOME)
    return reads


@pytest.fixture(autouse=True)
def controller_lock_root(tmp_path, monkeypatch):
    monkeypatch.setattr("phonebridge.bridge.RESERVATIONS", tmp_path / "controllers")
