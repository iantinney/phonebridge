# Phonebridge

**Give your phone a job.** Phonebridge is an open-source agent workspace for your real iPhone. Watch the phone live
on your Mac, follow each step with the screen it acted on, and stop or take over at any point. Agents such as Claude
Code and Codex can drive the same phone through Phonebridge's MCP server. It works over USB, with no jailbreak and
no app on the phone.

We're building toward describing a result once and getting finished work back. The alpha's demonstrated scope is
below. **[Bring one real phone chore](docs/partners/)** to help choose what comes next: an uncut creator or early-user
attempt, with a reusable job if it works.

**Teach once, run on a new input.** A workflow learned from an agent-run `@natgeo` profile check was exported, imported
unchanged into a fresh install of this release, and run on `@apple` on a real iPhone 12: 88 seconds, 9 inputs (6
taught, 3 to reach its start screen), no one stepping in, and all three counts (1,413 posts, 38.1M followers, 9
following) read from one stored result frame. Evidence, the portable workflow file and the input trace:
[docs/demo/taught-profile-check](docs/demo/taught-profile-check/). In this alpha a taught workflow runs Instagram
profile checks only.

> **Alpha.** Verified on one iPhone 12 (iOS 27.2) with one Mac, in the maintainers' own installs; no outside user has
> run it yet. Teach is new and narrow: Instagram profile checks only. Everything else, with its evidence:
> [What works today](#what-works-today).

**Also from a fresh install:** the workspace's Account check found `@natgeo` and read 32.1K posts, 268M followers and
195 following off the screen, about 3.5 minutes after installing (with the packages already in uv's cache). The
unchanged result frame, its hash and the nine inputs it sent: [docs/demo/native-profile-check](docs/demo/native-profile-check/).

## Quick start

You need a Mac with Apple silicon, Python 3.12 or later and [`uv`](https://docs.astral.sh/uv/), and an iPhone on iOS 27 or later,
connected by USB, trusted, with Developer Mode and UI Automation on ([Set up a phone](#set-up-a-phone)).

```sh
git clone https://github.com/iantinney/phonebridge && cd phonebridge
uv sync
uv run phonebridge setup          # the connected phone -> ~/.config/phonebridge/phonebridge.toml
uv run phonebridge serve --http   # terminal 1: the bridge on 127.0.0.1:8934
uv run phonebridge-workspace      # terminal 2: prints your private sign-in link
```

Open the link, click **Read the screen** (it sends no input), then run **Account check** on a public profile.
`uv tool install .` from the clone puts `phonebridge` and `phonebridge-workspace` on your `PATH` instead.

## Requirements

- A Mac with Apple silicon, macOS 26, Python 3.12 or later and [`uv`](https://docs.astral.sh/uv/). No Xcode.
- An iPhone on **iOS 27 or later** for input. On iOS 26.5 an iPhone 12 refuses the input stream; screenshots, OCR,
  Home, `launch` and `open_url` still work there.
- A USB cable.
- Phonebridge can't enter a passcode. Unlock the phone yourself before a session and keep it awake while the session
  runs (a longer Auto-Lock helps).

Linux installs and runs the test suite, but only macOS can drive a phone (Apple Vision, AVFoundation capture).

**The device tunnel and the screen.** Without any setup, Phonebridge opens its own in-process tunnel to the phone
(no administrator rights, one phone per process). On iOS 27.2 the maintainers saw taps and the phone's screen stream
both arrive over that tunnel (2026-11-08, with pymobiledevice3's tunnel service also running on that Mac; a Mac
without the service hasn't been checked yet).

By default the screen comes through AVFoundation, which works only for a bridge running in your logged-in desktop
session. A bridge started any other way (over SSH, for example) gets no AVFoundation frames and falls back to the
screen stream.

If the live view or input doesn't start, run pymobiledevice3's tunnel service and try again; it needs an
administrator password once per boot:

```sh
sudo uv run pymobiledevice3 remote tunneld
```

The service is also what lets one server drive several phones.

## Set up a phone

On the phone, once:

1. Plug it in by USB, unlock it, tap **Trust** and enter the passcode.
2. Settings › Privacy & Security › **Developer Mode** › On. The phone restarts; unlock it and confirm.
3. Settings › Developer › **UI Automation** › On.

On the Mac:

```sh
uv run phonebridge devices     # the connected phone's UDID and iOS version
uv run phonebridge setup       # writes ~/.config/phonebridge/phonebridge.toml for that phone, alias "phone"
```

`setup` refuses to overwrite an existing config. With several phones connected, pass `--device <UDID>`; `--alias`
names the phone. The config it writes:

- marks the phone `read-only` (reported to agents as advice; Phonebridge can't tell a Post tap from any other tap);
- turns on the local lease: every input holds `device:<alias>` through `phonebridge-local-lease`, so only one
  controller at a time sends input. A second one is refused until the first releases the phone, 60 s after its last
  input. Screenshots and OCR need no lease.

The developer disk image mounts by itself on first use.

## The desktop workspace

With a phone set up, run the bridge and the workspace side by side:

```sh
uv run phonebridge serve --http        # terminal 1: the bridge on 127.0.0.1:8934
uv run phonebridge-workspace           # terminal 2: prints "Sign in at http://127.0.0.1:8780/t/<token>"
```

Open the printed link. The workspace shows each configured phone and its live screen, about one frame a second,
captioned with how old the frame is. A first step that sends no input: click **Read the screen**. It stores one look
(not a task, so it doesn't appear in the task list) and lists the text it read, each line placed on the frame.

Tasks (account check, feed scout) take the phone's device lease through `phonebridge-local-lease`, so the bridge and
the workspace must run as the same user on the same Mac. Each step shows the frame it acted on; **Stop** ends the
task before its next input. The sign-in link is the operator's key: anyone who has it controls the phone.

The workspace assumes the phone is yours. A mission reads which Instagram account is signed in and reports it; to
make missions stop unless a particular account is signed in, set `TW_PHONE_ACCOUNTS='{"phone": "ig:me"}'` and
`TW_ACCOUNT_HANDLES='{"ig:me": "your_handle"}'` before starting it.

### Teach

Teach is where Phonebridge is heading: show the phone a job once, then hand it new inputs. In this alpha it runs one
kind of job, an Instagram profile check (see [What works today](#what-works-today) for how far it has been checked).
The way in starts from a task the workspace already ran:

1. Run **Account check** on a public profile and let it finish.
2. On the finished task, choose **Teach a workflow from this task** (optionally only part of it).
3. Read the steps it found, answer any question it asks once, and **Confirm this workflow**.
4. **Run a saved workflow**: pick it, give a new handle, and **Run on the phone**.

The workflow names each step by the control it used and the screen it was on. An icon with no text (a tab, for
example) is kept as the position you touched, relative to the screen size, and used only on that same screen. Your
corrections are kept for every later run.

What a taught workflow may do when it runs is fixed in this alpha: Instagram's Home, Search and Profile tabs, the
search field and its clear button, typing the handle you give it, the result row for that handle, the
Accounts/Profiles chip, and reading the screen. A workflow with any other step can still be derived, read and
corrected, but running it is refused before the phone is touched. Nothing a taught run does posts, follows, likes or
sends.

Teaching from a task needs nothing extra: each screen the task stored keeps the text the phone read from it.

Teaching from an uploaded screen recording needs more on this computer:

- `ffmpeg` and `ffprobe` on `PATH` (`brew install ffmpeg`).
- Text recognition for the recording's frames, in its own Python:
  `uv venv ~/.local/share/teachable-worker/ocr && uv pip install --python ~/.local/share/teachable-worker/ocr rapidocr_onnxruntime`
  (or `TW_OCR_PYTHON` pointing at a Python that has it, or `tesseract` on `PATH`). A task stored before this release,
  without the text of its screens, needs it too.
- Optionally a model to interpret the recorded steps: `TW_TEACH_MODEL=claude-plan` (your signed-in `claude` CLI) or
  `TW_TEACH_MODEL=local:<url>`. It is off by default, and a recording goes to an external model only if you allow it
  for that recording.

## Use it from an agent

stdio (the client starts the server):

```sh
claude mcp add phonebridge -- uv run --project /path/to/phonebridge phonebridge serve
codex mcp add phonebridge -- uv run --project /path/to/phonebridge phonebridge serve
```

Or keep one server running and connect over HTTP, which keeps the phone's session warm between calls:

```sh
uv run phonebridge serve --http        # http://127.0.0.1:8934/mcp, loopback only
claude mcp add --transport http phonebridge http://127.0.0.1:8934/mcp
codex mcp add phonebridge --url http://127.0.0.1:8934/mcp
```

Other MCP clients take the same command:

```json
{"mcpServers": {"phonebridge": {"command": "uv", "args": ["run", "--project", "/path/to/phonebridge", "phonebridge", "serve"]}}}
```

The server's tools include `devices`, `presence`, `screen`, `frame`, `ocr`, `look`, `find_text`, `foreground`,
`tap`, `double_tap`, `swipe`, `scroll`, `type`, `paste`, `home`, `unlock`, `launch`, `open_url` and `lock_state`.
Coordinates are screenshot pixels, the same space `ocr` returns, so an agent taps what it reads. The server takes
the device lease at an agent's first input and holds it until 60 s after the last; an agent that needs the phone for
longer can hold the lease itself and pass `lease=<token>` on each input.

Three prompts to start with:

1. "Take a screenshot of my phone, read the screen, and tell me which app is open. Don't touch anything."
2. "On my phone, open Settings › General › About and tell me the iOS version and the model name. Don't change any
   setting."
3. "Open https://en.wikipedia.org/wiki/Special:Random in Safari on my phone, scroll to the end of the article, and
   summarise it in three sentences."

Longer read-only tasks written as prompts, each marked with whether it has been run end to end:
[RECIPES.md](RECIPES.md).

## From the command line

Without the workspace, look first. This sends no input:

```sh
uv run phonebridge screen phone -o now.png
uv run phonebridge ocr phone
```

Then the live check. It presses Home, swipes through the home screens (or types in App Library search) to find
**Settings**, opens it, checks that it opened, and goes Home again. It takes no account actions:

```sh
uv run phonebridge check phone
```

Each CLI command opens its own connection to the phone (a few seconds). If an MCP server is driving the same phone,
the CLI's input waits its turn: it is refused while the server holds the lease.

While input flows, iOS shows a purple "remote control" pill. Home-screen labels aren't tappable on iOS 27; the check
taps the icon above the label.

## What works today

Every capability and the evidence behind it, as of this release.

| Capability | Status |
|---|---|
| Screenshots, OCR, find text on screen (`devices`, `screen`, `ocr`, `find`) | Works on the maintainers' iPhone 12 (iOS 27.2). No public run evidence linked yet. |
| Tap, swipe, scroll, type, paste, Home, launch an app, open a URL, with configurable, randomised timing | Works on the same phone, **but live input currently fails intermittently**; being investigated. Needs iOS 27 or later. |
| MCP server for agents (`phonebridge serve`, stdio or HTTP on 127.0.0.1) | Implemented; the maintainers' agents use it. |
| Guards: lock screen, foreground app, stale input, URL allowlist, exclusive device lease | Implemented, with unit tests. Not independently verified on a device. |
| Known-screen recognition: 89 screen records for Instagram, TikTok, X, Reddit and iOS | Records built from real frames of specific app versions; tested on synthetic fixtures. Apps change, so records go stale. |
| `phonebridge setup` and the local device lease (`phonebridge-local-lease`), with no cloud or account | Run end to end once on the maintainers' Mac from a fresh environment, config and state (2026-11-08): setup, the bridge, the workspace, one account check, about 3.5 minutes from install. Not yet tried by anyone else. |
| A model to name unfamiliar screens | Off by default. Opt in with `PHONEBRIDGE_VLM` (see [Limitations](#limitations)). |
| Desktop workspace (`phonebridge-workspace`): live phone view, "Read the screen", tasks with steps, stop | Included. On a fresh install, the imported taught workflow ran on the real phone through the HTTP API; its completed task, displayed counts and linked evidence were then checked in the shipped workspace using a browser. The browser did not initiate that validation run. |
| Read-only Instagram tasks in the workspace: account check (named public profiles) and feed scout | Account check: a recorded run on the maintainers' iPhone 12 (iOS 27.2, 2026-11-08) read 2 of 2 public profiles, its counts checked against the frames it stored; one run from a fresh install is shown in [docs/demo/native-profile-check](docs/demo/native-profile-check/). Feed scout: no checked real run yet. |
| Teach (alpha): a workflow from a finished task or a recording, read and corrected in the workspace, rerun on new handles (`taught_run`: Instagram profile checks only, read-only) | Two verified runs on the maintainers' iPhone 12 (iOS 27.2, 2026-11-08), each from a workflow taught on an `@natgeo` check. On a fresh install of this release, the workflow, imported unchanged, read `@apple`'s 1,413 posts, 38.1M followers and 9 following from one accurate result frame, with 9 inputs and no one stepping in ([evidence](docs/demo/taught-profile-check/)). An earlier run read a different public account in 32 s. Seven attempts failed along the way (four before the first success, three on the standalone release before the second) and led to the fixes in this release. A founder's install, not yet an outside user's; a Mac without the tunnel service is unchecked. |
| Several phones at once | Leases are per device; only one physical phone has been exercised. |
| Posting, messaging, liking, following | No verified recipe and no per-action approval flow. The generic input tools can still tap any button, Post and Send included, so keep agents to read-only tasks and watch them. |

Reports from other Macs and phones are the most useful contribution right now (see [Share a result](#share-a-result)).

## Limitations

- **One Mac, one phone, verified.** The default in-process tunnel carries one phone per server process. Several
  phones in one server need pymobiledevice3's tunnel service running as root
  (`sudo uv run pymobiledevice3 remote tunneld`); that path hasn't been exercised with two phones for this release.
- **The local lease is per Mac.** It keeps two controllers on one Mac from colliding. It doesn't coordinate across
  computers, and it isn't an account policy.
- **The first look over the in-process tunnel** can fail once with "userspace dial plane is closed" (HTTP 503); the
  looks after it work. Seen twice on iOS 27.2 (2026-11-08); try again.
- **The screen stream's size.** On iOS 27.2 the stream's video is coded larger than the screen (1184×2576 for an
  iPhone 12's 1170×2532) with dark padding at the right and bottom. Phonebridge learns the screen's size from one
  screenshot before the stream starts and crops each frame to it; a frame whose margins aren't dark goes out uncropped
  and labelled so, and a tap is refused, with nothing sent, while the screen's size is unknown. Checked on an iPhone 12
  with iOS 27.2 only.
- **Typing is ASCII only.** `paste` handles emoji, accents and long text.
- **While input or the screen stream is open,** the phone shows the capture indicator and in-app camera and microphone
  are blocked. The session closes 45 s after the last input.
- **Unfamiliar screens.** The known-screen records cover specific app versions. A screen they don't recognise stays
  unknown unless you choose a model: `PHONEBRIDGE_VLM=claude-cli` uses your signed-in `claude` CLI (and AWS Bedrock on
  its usage limit, with `uv sync --extra vlm` and AWS credentials); `PHONEBRIDGE_VLM=bedrock` uses Bedrock only. The
  chosen provider receives a downscaled screenshot and the screen's OCR text.
- **Optional pieces that need extra setup:** `phonebridge understand serve` (a local vision model, extra
  `understand`, about 800 MB of packages plus 3 GB of weights). The `bthid` input backend needs a Bluetooth daemon
  that isn't part of this release.
- **The `run_*` task tools** (a journaled task queue with a read-only feed visit) come from the maintainers' own
  setup. They are experimental and undocumented here.
- **The workspace has one operator** (the sign-in link) and listens on 127.0.0.1 only. Its data, including the frames
  tasks keep as evidence (7 days), lives in `~/.local/state/teachable-worker/`.
- **Instagram version.** The Instagram tasks and screen cues were built and checked on Instagram 454 (iOS 27.2).
  Other versions are unverified; an unfamiliar screen stops the task.
- **Apps and accounts.** Phonebridge drives apps through their normal screens, with pauses you can configure
  (`[humanize]` in `phonebridge.toml`), and stops on screens it doesn't know. The pauses in the runs shown here came
  from those settings and the tasks' own fixed reading times, not from a person's recorded pace. It doesn't promise
  that an app can't tell it is automated, and it doesn't know any app's rules. Follow the terms of the apps and
  accounts you use.

## Privacy and security

- No telemetry: nothing is sent to the maintainers. Phonebridge sends a screenshot off the Mac only to a model you
  turned on with `PHONEBRIDGE_VLM`.
- The HTTP server listens on 127.0.0.1 only. If `~/.config/phonebridge/phonebridge.token` exists, every request needs
  `Authorization: Bearer <token>`.
- Every call is logged to `~/.local/state/phonebridge/calls.jsonl` (mode 600). Tap coordinates are kept; typed text
  isn't. Screenshots and evidence folders show whatever was on your screen, including other people's messages and
  posts. Don't attach them to issues.
- Report a vulnerability privately: see [SECURITY.md](SECURITY.md).

## Share a result

If you installed Phonebridge and ran a first task, whether it worked or not, a
[first-task report](https://github.com/iantinney/phonebridge/issues/new?template=first-task.yml) helps the most. It
asks for text only: device, iOS version, what you ran, what broke. Nothing is collected automatically.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Tests run on Linux or macOS: `uv run pytest`, then `uv run pytest packages/workspace/tests`.

## Credits and license

Phonebridge stands on [pymobiledevice3](https://github.com/doronz88/pymobiledevice3) by doronz88 and contributors,
which does the USB, tunnel, screenshot and CoreDevice input work. It also uses Apple's Vision and AVFoundation
frameworks (through [PyObjC](https://github.com/ronaldoussoren/pyobjc)), the
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk), [Pillow](https://github.com/python-pillow/Pillow)
and [PyYAML](https://github.com/yaml/pyyaml); optionally [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) with
[Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and the
[Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python). Versions and licenses:
[DEPENDENCIES.md](DEPENDENCIES.md).

Other projects named "phonebridge" exist and are unrelated to this one.

GPL-3.0-or-later (see [LICENSE](LICENSE)), because Phonebridge imports pymobiledevice3, which is GPL-3.0-or-later.
The small MCP client in `packages/phonebridge-client` is part of the same release under the same license. The small
icon images in `src/phonebridge/screens/records/templates/` are crops of the apps' own interface, used only to
match those screens locally, and the screenshot in `docs/demo/` documents one run; neither is covered by this
project's license.
