# Known-screens store (lane screens; PLAN §5, DECISIONS Q3/G3)

One YAML per Screen under `<platform>/<id>.yaml` (schema: `phonebridge/screens/schema.py`, PLAN §5.1):
content-free id, platform, bundle, appearance as observed on the phone, `match` anchors (OCR words in
regions, a chrome dHash, icon templates under `templates/`), `targets` (OCR-anchored, template-located or
region-bounded, never bare coordinates), `next` transitions, `popups`, `way_out`, `checks`, success/failure
counters. Stop screens are records (`kind: stop`, PT's texts verbatim, one matching rule). Records learned
live by the slow model are saved at once under the corpus (`candidates/`) and run from then on; `screens
promote --apply` adopts the ones that work into this directory.

The corpus (frames, OCR, replay sequences, confusers, the manifest, the coverage map) lives outside git at
`~/.local/state/agent-reach/screens/corpus/` (mode 600; it holds account screens).

Commands: `phonebridge screens {list,stats,gates,harvest,add,promote,replay,walk,drift}`.

## Wall-harvest checklist (SCR-08): harvest these on purpose, in rehearse, when the phone is free

Each wall becomes ≥ 1 fixture and a record (popup with its safe answer, alert that parks, or stop).

| Wall | How to reach it | Record |
|---|---|---|
| Photos access None / Limited | Settings › Instagram › Photos (done 2026-10-18: `ios.settings.photos_access`, `ios.alert.photos_full_access`) | alert (parks) |
| Camera + microphone prompt | IG create › camera tile (`instagram.camera_access` seen; the iOS alert behind Continue not yet) | alert |
| Notifications prompt | first launch after reinstall, or Settings › Notifications off then open IG | popup: Don't Allow |
| App Tracking Transparency | first launch after reinstall | popup: Ask App Not to Track |
| Allow Paste | paste the caption on the share screen (`ios.allow_paste`, no frame yet) | popup: Allow Paste |
| Coach marks / "New" tips on a fresh account | profile, reels tab, create flow | popup: dismiss |
| Find contacts / Connect contacts sheet | TikTok profile, IG profile (`("Find contacts", "Don't allow")` in tracer) | popup: Don't allow |
| Start over? discard dialog | back out of the reel editor (done: `instagram.popup.start_over`) | popup: Keep editing |
| Force-update wall | an old build left installed (not now: updates are off) | stop (operator) |
| Login wall | sign out on a test account only (Q4) | stop: login_wall |
| Low battery / storage alerts, incoming call, Clock alarm, Screen Time | iOS, on demand | alert |

Dark frames of the other appearance, half-rendered transitions and header swaps are generated from the
fixtures by `screens harvest` as confusers for the SCR-03 gates.
