"""The self-learning known-screens store (PLAN §5, DECISIONS Q3).

Records are data (schema.py), kept in the repo and learned live; the corpus of frames lives outside git
(store.py). The cascade (cascade.py) recognises a frame from its UI chrome, never from the content area, and
returns `unknown` rather than a weak guess. The flow engine (engine.py) walks records by their `next`
transitions; an Unfamiliar screen (GLOSSARY: not matched, or progress stalls) goes to the learner
(learn.py), which asks the slow model once (vlm.py), acts, and saves what worked as a Learned response
(a record with its answer target, success/failure counters) that runs fast from then on, no review.
"""
