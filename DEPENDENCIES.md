# Dependencies

| Dependency | Version | License | Source | Why |
|---|---|---|---|---|
| `pymobiledevice3` | 11.34.1 (pinned) | GPL-3.0-or-later | <https://github.com/doronz88/pymobiledevice3> | USB lockdown, the RemoteXPC tunnel, DVT screenshots, CoreDevice HID input, pasteboard and app services. It tracks Apple's protocol, so bump it deliberately and rerun `phonebridge check`. **Its license makes Phonebridge GPL-3.0-or-later.** |
| `mcp` | 2.3.0 | MIT | <https://github.com/modelcontextprotocol/python-sdk> | The MCP server (stdio and streamable HTTP) and the client in `phonebridge_client`. |
| `pillow` | ≥ 11 | MIT-CMU (HPND) | <https://github.com/python-pillow/Pillow> | JPEG for MCP screenshots, thumbnails for the screen checks. Already required by pymobiledevice3. |
| `pyyaml` | ≥ 6 | MIT | <https://github.com/yaml/pyyaml> | The known-screen records, one YAML file per screen. |
| `pyobjc-framework-Vision` | 12.2.2 (macOS) | MIT | <https://github.com/ronaldoussoren/pyobjc> | Apple Vision text recognition (OCR) and scene labels, fed PNG bytes directly. |
| `pyobjc-framework-AVFoundation`, `-libdispatch` | 12.2.2 (macOS) | MIT | <https://github.com/ronaldoussoren/pyobjc> | The iPhone's screen as a capture device (`avf.py`, ~60 fps), on a GCD queue. Pulls `-CoreMedia` and `-Quartz` (same license). |
| `pyobjc-framework-NaturalLanguage` | 12.2.2 (macOS) | MIT | <https://github.com/ronaldoussoren/pyobjc> | On-device sentence embeddings for the feed visit's interest cue; a bag-of-words fallback elsewhere. |
| VideoToolbox, CoreMedia, Vision, AVFoundation | system (macOS) | Apple | macOS SDK | Decoding the CoreDevice screen stream (through pymobiledevice3's ctypes binding), OCR and capture. Off macOS the stream decoder uses PyAV, already a pymobiledevice3 dependency there. |

Optional extras:

| Extra | Dependency | License | Why |
|---|---|---|---|
| `vlm` | `anthropic[bedrock]` ≥ 0.80 (pulls boto3, Apache-2.0) | MIT | <https://github.com/anthropics/anthropic-sdk-python>: the AWS Bedrock fallback for naming unfamiliar screens. The first route, your own `claude` CLI, needs no package. |
| `understand` | `mlx-vlm` 0.7.8 with `mlx` (macOS) | MIT | <https://github.com/Blaizzy/mlx-vlm>: a local vision model on the Mac GPU (`phonebridge understand serve`). About 800 MB of packages. |
| (weights, downloaded on first use) | `mlx-community/Qwen3.5-4B-4bit` | Apache-2.0 | <https://huggingface.co/Qwen/Qwen3.5-4B>: the model `understand` serves; about 3 GB in the Hugging Face cache, never in git. |
| dev group | `pytest` ≥ 8, `numpy` ≥ 2.1 | MIT, BSD-3-Clause | The test suite. |

Notable transitive dependencies through pymobiledevice3: `pmd-pytcp`, `pmd-net-addr`, `opack2`, `ipsw-parser`
(GPL-3.0-or-later); `pylzss` (LGPL-3.0); `pyimg4` (MIT), `srptools` (BSD-3), `bytestruct` (MIT), `pytun-pmd3`
(MIT-style). All are compatible with GPL-3.0.

**Third-party images.** The icon templates in `src/phonebridge/screens/records/templates/` (close, comment, create,
tabs, bookmarks, menu dots, compose) are small crops of Instagram's, TikTok's and X's own interface, kept
only to recognise those screens on your phone. They are not project work and not covered by its license.

**License boundary.** Phonebridge imports GPL-3.0 code, so the whole release, `phonebridge_client` included, is
GPL-3.0-or-later. A program that only talks to Phonebridge through its MCP server or CLI is a separate work.

Not used, by design: `leeguooooo/iphone-use` (PolyForm Shield, not an open-source license); Appium WebDriverAgent
(needs Xcode and code signing; a later option for an element tree); `apple/ml-mobileclip` (its weights' license
excludes use in products).
