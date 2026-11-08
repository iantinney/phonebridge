# Recipes

Two recipes included in this release, in the desktop workspace (`phonebridge-workspace`; the Run and Teach tabs), and one planned that isn't packaged yet (Coach). All are read-only by default. Every step carries its exact status. "Verified" means a recorded run on a real phone whose result was checked against its stored, timestamped frames; anything else says what is and isn't shown.

**What each recipe touches**

| Recipe | Whose data | What happens on the phone |
|---|---|---|
| Profile check | **Public profiles you name** (observed, never controlled, not a customer account) | Uses the phone's own logged-in Instagram to search and open them. The searches stay in that account's recent searches. Never Follow or Message |
| Feed scout | **Your own account's home feed** | Scrolls and reads only; no likes, follows, saves or messages |
| Conversation coach | **Your own conversation, between adults** | `review_only` (default) never touches the phone; `draft` types a reply and reads it back, never sends; `approve_each` sends only the exact text you approved |

## 1. Profile check → teach it → run it on a new handle

**a. Run: Account check** (`account_check`). Inputs: `handles`, comma separated, at most 5 (e.g. `natgeo, nasa`); `record: yes` keeps the run as a demonstration.
It opens Instagram's Search tab and types each handle. It opens the account, through the Profiles chip if a keyword page opens, and reads the handle, posts, followers and following as the profile shows them.

Per profile you get: `found`, the counts as displayed (rounded the way the app rounds them), the stored frame id and its capture time, the search row, and `recent_posts_text`: the profile grid's visible text only. Thumbnails carry no text, so there's no per-post extraction.
*Status: verified 2026-11-08 on one iPhone 12 (iOS 27.2).* A recorded run read 2 of 2 profiles: @natgeo 32.1K posts / 268M followers / 195 following, and @nasa 5,000 / 104M / 90. Both sets of counts were independently checked against the stored frames. An earlier run stopped on @nasa at its story viewer. This is the maintainers' own use, not an outside install.

**b. Teach: "Teach a workflow from this task."** A finished task that kept its frames is a demonstration. In the Teach tab, describe what you did, answer the questions it asks about steps it couldn't see, and confirm the routine.
*Status: done once on the maintainers' phone:* the routine verified in (c) was derived from a recorded Account check and confirmed in the Teach tab.

**c. Run the taught workflow on a fresh handle** (`taught_run`, inputs: the routine id and `{"handle": "<new handle>"}`).
*Status: verified twice.* On 2026-11-08 a workflow taught from the `@natgeo` check ran on the maintainers' iPhone 12: once on a different public account (32 s, 6 inputs), and once, imported unchanged into a fresh install of this release, on `@apple` (1,413 posts / 38.1M followers / 9 following, all from one accurate result frame; 9 inputs). Neither needed anyone to step in; four earlier attempts failed and led to fixes. In this alpha a taught workflow can only run Instagram profile checks.

## 2. Feed scout: a short list with why and where

**Run: Feed scout** (`feed_scout`). Inputs: `request` in a few words (e.g. "AI tools", "new phone launches") and `posts`, how many feed posts to look at (1–15, default 8).
It scrolls your Instagram home feed and keeps the posts whose visible text matches. The Mac's local vision model judges relevance, and each find says what judged it. Each find comes with its frame as the source; no share links are read back.
*Status: in the workspace; no checked real-phone run recorded yet.*

## 3. Conversation coach: talk it through on the computer

**Run: Coach** (`coach_observe`). Inputs: `consent: true` (your own conversation, between adults, after reading the processing notice); `mode` `review_only` | `draft` | `approve_each`; an optional `goal` and `style`.
It reads the conversation open on the phone from stored frames: their bubbles on the left, yours on the right. Then it answers your questions on the computer, citing the observed messages ([m4]); it looks at the phone again before each answer. "Review" gives a message-by-message review.
- In `draft` it types a reply into the message box and reads it back. It never sends.
- In `approve_each` it sends only the exact text you approved, to the same conversation, and checks the new bubble.

Privacy: the other person's name is masked; nothing goes into events or metrics; the transcript is deleted 24 hours after the last read unless you keep it. Some apps' terms restrict third-party automation or AI-written messages, so check before using it there. Don't post real conversations in issues.
*Status: not in this release. The coaching lane is optional and isn't packaged yet; the workspace lists Coach as unavailable.*
