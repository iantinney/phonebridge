"""SCR-03 gates on the real corpus (outside git). N and confidence intervals go into the report."""
import os
from pathlib import Path

import pytest

from phonebridge.screens import gates
from phonebridge.screens.store import Store, CORPUS, RECORD_DIRS

pytestmark = pytest.mark.skipif(not (CORPUS / "manifest.yaml").exists(),
                                reason=f"no screens corpus at {CORPUS} (outside git; harvest first)")


@pytest.fixture(scope="module")
def report():
    store = Store(records_dirs=RECORD_DIRS, corpus=CORPUS)
    rep = gates.run(store)
    out = Path(os.environ.get("SCREENS_GATES_REPORT", CORPUS / "gates-report.md"))
    out.write_text(gates.render(rep))
    return rep


def fixtures():
    if not (CORPUS / "manifest.yaml").exists():
        return []
    return [p.relative_to(CORPUS).as_posix() for p in sorted(CORPUS.glob("*/*/*.png"))
            if p.parent.parent.name not in ("confusers", "unknown", "sequences", "candidates")]


@pytest.mark.parametrize("fixture", fixtures())
def test_every_fixture_recognised(fixture):
    store = Store(records_dirs=RECORD_DIRS, corpus=CORPUS)
    want, got = gates.recognise_fixture(store, CORPUS / fixture)
    assert got == want, f"{fixture}: expected {want}, got {got}"


def test_no_confuser_matches(report):
    assert report["confusers"]["n"] >= 200
    assert report["confusers"]["matches"] == []


def irreversible_ids():
    if not (CORPUS / "manifest.yaml").exists():
        return []
    return sorted(r.id for r in Store(records_dirs=RECORD_DIRS, corpus=CORPUS).records.values() if r.irreversible)


@pytest.mark.parametrize("screen", irreversible_ids())
def test_irreversible_zero_fp_n_ge_5(report, screen):
    import yaml
    row = report["irreversible"][screen]
    assert row["false_positives"] == 0, f"{screen}: false positives {row['sources']}"
    if row["n"] < 5:
        doc = (yaml.safe_load((CORPUS / "manifest.yaml").read_text()) or {}).get("irreversible", {}).get(screen)
        if isinstance(doc, str) and doc.startswith("xfail"):
            pytest.xfail(f"{screen}: insufficient ({row['n']} < 5): {doc}")
        pytest.fail(f"{screen}: insufficient ({row['n']} < 5) and not documented in manifest.yaml")


def test_overall_accuracy_with_ci(report):
    assert report["overall"]["n"] >= 40  # the first harvest; grows with every run (frames are saved as fixtures)
    assert report["overall"]["accuracy"] >= 0.98
    lo, hi = report["overall"]["ci95"]
    assert 0 <= lo <= report["overall"]["accuracy"] <= hi <= 1
