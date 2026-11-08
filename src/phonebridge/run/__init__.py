"""Tasks the VM hands the device operator: the fsync'd journal (journal.py) and the task API (taskapi.py)."""

from . import do, taskapi

try:  # the publishing adapter; the public release leaves it out until its phone review clears it
    import phonebridge.run.post as post
except ModuleNotFoundError as e:
    if e.name != "phonebridge.run.post":
        raise
else:
    taskapi.FLOWS["post"] = post.Post  # the one loop's post kind (task protocol 2), defined on taskapi's PostFlow
taskapi.FLOWS["do"] = do.Do  # "do X on this account" (task protocol 2): a visit whose detour is the task
