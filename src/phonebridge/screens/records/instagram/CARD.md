# Instagram (com.burbn.instagram)

Use when a task names an Instagram account (ig:public). Screen records in this directory are the structured truth;
this card is prose for the task planner (design task-agent §3.5, mobile-harness CARD shape).

## Useful labels
- Tab bar (bottom, icons only): Home, Reels, the DM tray, Search, Profile. The Search tab opens the Explore grid with
  the search bar on top ("Search with Meta AI"); tap the bar to type.
- Search: while typing, matching accounts are listed under the field (handle, the name under it) and tags (#tag with a
  post count). Tap the row whose handle is exactly the one asked for; "Cancel" closes the keyboard.
- Profile of another account: counts "posts / followers / following" with the number above each word; "Follow" (or
  "Following"), "Message"; the grid below.
- A post: title "Posts" with a back chevron, the author's handle, the media, the action row (heart, comment, share,
  and the bookmark at the right end = Save), the caption starting with the author's handle, "View all N comments".

## Flow notes
- Account by handle: Search tab → the search bar → type the handle → tap the matching account row → the profile.
- Latest post: on the profile, the first grid tile WITHOUT the pin glyph (top-right corner) → the post.
- Hashtag: type the tag in the search bar → tap the "#tag" row → its top posts grid → the first tile.
- Follower count: read the number above "followers" exactly as shown (e.g. "283M").

## Traps
- Pinned posts come first in a profile grid (up to three); the newest post is the first tile without the pin glyph.
- The typed text also shows in the search field itself: the account row is the line under it, not the field.
- A similar handle (natgeotravel for natgeo, a fan page) is not the account: the handle must match exactly.
- The bookmark and the heart toggle: tapping a filled bookmark unsaves, a red heart unlikes. Check the state first.
- The Save toast / "Save to collection" sheet may show for a moment after a save; it is not an error.
- Never tap Follow, Message, Share, or a comment field unless the task asks for that.
