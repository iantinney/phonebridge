# Development history

This public repository began as a standalone export on **November 8, 2026**. It was not the
first day of development. Earlier Phonebridge source lived in a larger private repository,
whose unrelated projects, account data and deployment settings are not part of this release.

The following dates come from the existing private source history and the linked public
artifacts. The private history is not publicly inspectable; this is a maintainer-reported
timeline, not a reconstruction of public commits.

| Date | Recorded work | Evidence available here |
| --- | --- | --- |
| October 18, 2026 | Initial Phonebridge interface, Apple Vision OCR, leased input, MCP server, CoreDevice backend and CLI; configurable touch and typing generation; host and foreground guards. | Implementations retained under `src/phonebridge/`; original dates come from private source commit metadata. |
| October 19–20, 2026 | Screen-stream, observation, controller and recovery work continued in the private repository. | The public source includes those components; historical development notes remain private. |
| November 8, 2026 | Standalone packaging and the first public alpha; fresh Mac install and an unchanged imported workflow operated a real iPhone. | [Native result, input receipt and workflow](demo/taught-profile-check/); [initial public export](https://github.com/iantinney/phonebridge/commit/159da456). |
| November 8, 2026 | Seven-device configuration, isolation and concurrent software validation. | [Multi-iPhone design and validation](multi-iphone.md), tests and accurately dated public commits. |

The earliest Phonebridge source commit located for this timeline is October 18. This timeline
does not establish a year of development or backdate newly written code. The public export
and subsequent changes retain their actual commit dates.
