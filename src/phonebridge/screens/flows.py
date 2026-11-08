"""The flows the engine runs today (data: records + `next` transitions). The IG reel flow mirrors tracer's
`feedagent.flows.INSTAGRAM` way_in up to the picker; the walks are the read-only canary (SCR-12)."""

from .engine import Flow, Step

IG_REEL = Flow(name="ig_reel", platform="instagram", bundle="com.burbn.instagram", final="instagram.reel.final", steps=(
    Step(at="instagram.home", do="tap", target="create"),
    Step(at="instagram.new_post.picker", do="tap", target="next"),
    Step(at="instagram.reel.editor", do="tap", target="next"),
    Step(at="instagram.reel.final", do="tap", target="share"),  # irreversible: rehearse stops here
))
IG_REEL_TO_PICKER = Flow(name="ig_reel", platform="instagram", bundle="com.burbn.instagram", final="instagram.new_post.picker",
                         steps=(Step(at="instagram.home", do="tap", target="create"),))
IG_WALK = Flow(name="ig_walk", platform="instagram", bundle="com.burbn.instagram", final="instagram.about_account", steps=(
    Step(at="instagram.home", do="tap", target="profile_tab"),
    Step(at="instagram.profile", do="tap", target="menu"),
    Step(at="instagram.settings", do="tap", target="search"),
    Step(at="instagram.settings.search", do="tap", target="about_account"),
))
TT_WALK = Flow(name="tt_walk", platform="tiktok", bundle="com.zhiliaoapp.musically", final="tiktok.settings", steps=(
    Step(at="tiktok.feed", do="tap", target="profile_tab"),
    Step(at="tiktok.profile", do="tap", target="menu"),
    Step(at="tiktok.menu", do="tap", target="settings"),
))
# TikTok video (lane tiktok) mirrors feedagent.flows.TIKTOK: + → camera → Upload → picker (our clip is picked by its
# duration in feedagent) → Next → editor → Next → the post screen, whose Post is irreversible: rehearse stops there
TT_POST = Flow(name="tt_post", platform="tiktok", bundle="com.zhiliaoapp.musically", final="tiktok.post.final", steps=(
    Step(at="tiktok.feed", do="tap", target="create"),
    Step(at="tiktok.create.camera", do="tap", target="upload"),
    Step(at="tiktok.upload.picker", do="tap", target="next"),
    Step(at="tiktok.upload.editor", do="tap", target="next"),
    Step(at="tiktok.post.final", do="tap", target="post"),  # irreversible: rehearse stops here
))
FLOWS = {"ig_reel": IG_REEL_TO_PICKER, "ig_reel_full": IG_REEL, "ig_walk": IG_WALK, "tt_walk": TT_WALK, "tt_post": TT_POST}
