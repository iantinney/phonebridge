"""Every decision has a reason (director v1): ads and suggestions skipped, his interests kept him, questions open comments."""

from phonebridge.director import cues
from phonebridge.director.cues import Cue, decide, interest

PROFILE = {"topics": ["gym", "fitness", "calisthenics"], "keywords": ["protein", "pullups"]}


def items(*texts):
    return [{"text": t, "x": 1, "y": 1, "w": 1, "h": 1, "conf": 0.9} for t in texts]


def test_interest_scores_caption_hashtags_and_labels():
    assert interest(items("gympolar Strong nutrition... more", "#fitness #protein"), [], PROFILE)[0] == 3
    assert interest(items("a sunset over the bay"), ["people"], PROFILE)[0] == 1
    assert interest(items("new pullups PR today"), [], PROFILE) == (2, ["pullups"])
    assert interest(items("anything"), [], {}) == (1, [])  # no profile yet: neutral


def test_decisions_carry_reasons():
    assert decide(Cue("ad", "skip", "ocr", 3.0, markers=["ad"]), items("Sponsored"), PROFILE).action == "skip"
    d = decide(Cue("video", "stay", "motion", 3.0), items("gympolar Strong nutrition... more #fitness"), PROFILE)
    assert d.action == "like_candidate" and "fitness" in d.reason and d.interest == 3
    d = decide(Cue("video", "stay", "motion", 3.0), items("Best gym split for calisthenics beginners?"), PROFILE)
    assert d.action == "open_comments" and "question" in d.reason
    d = decide(Cue("photo", "stay", "ocr", 3.0), items("samplehomestudio Upgrade your bedroom with... more"), PROFILE)
    assert d.action == "skip" and "glance" in d.reason
    d = decide(Cue(cues.BLIND, "stay", "late", 250.0), items("x"), PROFILE)
    assert d.action == "stay" and "blind" in d.reason
    assert set(d.as_dict()) == {"action", "reason", "interest", "matches"}


def test_suggested_posts_are_judged_on_interest_not_the_label():
    sug = Cue("suggested", "skip", "ocr", 2.0, markers=["suggested"])
    assert decide(sug, items("Suggested for you", "a sunset"), PROFILE).action == "skip"
    d = decide(sug, items("Suggested for you", "calisthenics pullups progression"), PROFILE)
    assert d.action == "stay" and "his thing" in d.reason and d.interest >= 2


def test_handle_comes_from_the_ocr_not_the_model():
    from phonebridge.director.cues import who, with_handle
    ig = [{"text": "9:41", "y": 30}, {"text": "the.sampleia", "y": 300}, {"text": "sam.postern", "y": 760}, {"text": "Follow", "y": 760}, {"text": "kim_chef_", "y": 1700}, {"text": "Follow", "y": 1700}, {"text": "62.1K", "y": 2000}]
    assert who(ig, 2532) == "sam.postern"  # not the story-tray name above the band, not the caption's 18pro, not the counter
    tt = [{"text": "Following Friends For You", "y": 140}, {"text": "@samplechess", "y": 2000}, {"text": "When Magnus Carlsen chooses you", "y": 2100}, {"text": "6,567", "y": 1500}]
    assert who(tt, 2532) == "samplechess"
    assert who([{"text": "Follow", "y": 400}, {"text": "1,239", "y": 500}], 2532) is None
    assert who([{"text": "ALIEN ARE", "y": 1200}, {"text": "DO YOU PLAY MORE", "y": 1300}, {"text": "26.6K", "y": 1400}], 2532) is None  # caption words aren't handles
    assert who([{"text": "sam.postern", "y": 700}, {"text": "Follow", "y": 760}], 2532) == "sam.postern"
    assert who([{"text": "samplytechreview", "y": 700}, {"text": "Follow", "y": 760}], 2532) is None  # letters only: ambiguous without layout, left to the model
    assert with_handle("A tech review video from rise.with.ab showing an iPhone", "samplytechreview") == "A tech review video from rise.with.ab showing an iPhone"  # the model named an account: kept
    assert with_handle("User sam.posternn posted a motivational quote", "sam.postern") == "User sam.postern posted a motivational quote"  # near-miss spelling corrected
    assert with_handle("tech_tianya posted an unboxing", "tech_tianya") == "tech_tianya posted an unboxing"
    assert with_handle("A user shares a video of a sunset", "travelbookofagirl") == "travelbookofagirl: A user shares a video of a sunset"
    assert with_handle("alien: alien: x", "alien") == "alien: alien: x"  # never prefixed twice


def test_model_handle_is_confirmed_against_the_post_card_ocr():
    from phonebridge.director.cues import confirm_handle, model_handle
    assert model_handle("User sam.posternn posted a motivational quote") == "sam.posternn"
    assert model_handle("A user shares a video of a sunset") is None
    items = [{"text": "the.sampleia", "y": 300}, {"text": "sam.postern", "y": 760}, {"text": "Follow", "y": 760}, {"text": "62.1K", "y": 2000}, {"text": "September 27", "y": 1900}]
    who, ok, s = confirm_handle("Verified user sam.posternn posted a motivational quote", items, 2532, "instagram")
    assert (who, ok) == ("sam.postern", True) and "sam.postern posted" in s and "sam.posternn" not in s
    who, ok, s = confirm_handle("samplytechreview posted a tech review", [{"text": "rise.with.abarat", "y": 700}, {"text": "62.1K", "y": 900}], 2532, "instagram")
    assert (who, ok) == ("samplytechreview", False) and s.startswith("samplytechreview")  # unconfirmed, kept, never prefixed
    who, ok, s = confirm_handle("User samplecook posted a TikTok clip", [{"text": "@samplecook", "y": 1900}, {"text": "Following Friends For You", "y": 140}], 2532, "tiktok")
    assert (who, ok) == ("samplecook", True)


def test_trailing_underscores_keep_the_longer_spelling():
    from phonebridge.director.cues import confirm_handle
    who, ok, s = confirm_handle("kim_chef__ posted a video of a laptop", [{"text": "kim_chef_", "y": 700}, {"text": "Follow", "y": 700}], 2532, "instagram")
    assert (who, ok) == ("kim_chef__", True) and s.startswith("kim_chef__")


def test_header_zone_names_a_post_the_model_left_nameless():
    from phonebridge.director.cues import confirm_handle
    items = [{"text": "the.sampleia", "x": 100, "y": 600, "w": 200, "h": 30}, {"text": "fullyclass", "x": 100, "y": 700, "w": 200, "h": 30},
             {"text": "Follow", "x": 900, "y": 700, "w": 100, "h": 30}, {"text": "SOCIAL MEDIA IS FAKE", "x": 100, "y": 1500, "w": 600, "h": 40}]
    who, ok, s = confirm_handle("A user shares a photo of a t-shirt", items, 2532, "instagram", header=[0.08, 0.26, 0.8, 0.33], w=1170)
    assert (who, ok) == ("fullyclass", True) and s.startswith("fullyclass: ")
    who, ok, s = confirm_handle("A user shares a photo", [{"text": "a", "x": 1, "y": 700, "w": 10, "h": 10}], 2532, "instagram", header=[0.08, 0.26, 0.8, 0.33], w=1170)
    assert who is None  # nothing handle-shaped in the zone: stays nameless


def test_the_sentence_may_name_the_account_as_the_source():
    from phonebridge.director.cues import model_handle, confirm_handle
    assert model_handle("A tech review video from samplytechreview showing an iPhone") == "samplytechreview"
    assert model_handle("A tech influencer shares a scenic road trip video") is None
    assert model_handle("A user shares a video of a sunset", ["travelbookofagirl"]) is None  # nothing in the sentence matches
    assert model_handle("Travelbookofagirl shares a sunrise video", ["travelbookofagirl"]) == "Travelbookofagirl"
    who, ok, s = confirm_handle("A tech influencer shares a road trip video", [{"text": "ringke_tech", "x": 120, "y": 700, "w": 200, "h": 30, "conf": 0.9}, {"text": "Follow", "x": 900, "y": 700, "w": 90, "h": 30}], 2532, "instagram")
    assert (who, ok) == ("ringke_tech", True)  # the store's default header zone named it


def test_fuzzy_fallback_only_takes_handle_shaped_tokens():
    from phonebridge.director.cues import model_handle
    assert model_handle("A user shares a photo of a t-shirt with 'Social media is fake'", ["social", "fullyclass"]) is None
    assert model_handle("A tech store posted a video showing how to clean a speaker", ["clean", "ny_tech_store", "speaker"]) is None
    assert model_handle("A post from 'your_he4ler.' with flowers", ["your_he4ler"]) == "your_he4ler"
