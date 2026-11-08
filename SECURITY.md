# Security

Phonebridge sends real input to a real phone, so a bug in it can act on your accounts. Please report a vulnerability
privately through GitHub's **Report a vulnerability** button on this repository's Security tab, not in a public
issue. Include the version (commit), your macOS and iOS versions, and the smallest steps that show the problem.
Expect an acknowledgement within a few days; there is no bounty.

In scope, for example:

- input reaching a phone without the device lease, past a guard (lock screen, foreground app, stale input), or
  outside the `open_url` allowlist;
- the MCP HTTP server answering anything but loopback, or accepting a request without the bearer token when
  `phonebridge.token` exists;
- a token, typed text or screenshot written somewhere other than the documented, mode-600 files.

## What Phonebridge keeps and sends

- **No telemetry.** Nothing is sent to the maintainers.
- **Logs** stay on the Mac under `~/.local/state/phonebridge/` (mode 600). `calls.jsonl` records each call with its
  timing and tap coordinates, never typed text. Lease tokens are stored only as hashes.
- **Screenshots** are written only where a command or tool is told to write them (`screen -o`, `check`, recordings,
  evidence folders). They contain whatever your phone showed, including other people's names and messages.
- **The model for unfamiliar screens** is used only if you have set one up (your `claude` CLI, or AWS Bedrock with
  the `vlm` extra). It receives a downscaled screenshot and the screen's OCR text.
- **Sharing a run** is voluntary: the first-task issue form asks for text only. Remove names, handles, phone numbers
  and message text before posting anything.
