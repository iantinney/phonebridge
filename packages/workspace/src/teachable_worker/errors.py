"""Refusals every module raises (the server turns them into an HTTP status and a plain sentence), and lane lookup."""

from __future__ import annotations

import importlib
import importlib.util


class Refused(ValueError):
    """A request the flow will not perform; the message is a plain sentence for the person."""

    status = 400


class Conflict(Refused):
    status = 409


class NotFound(Refused):
    status = 404


class TooMany(Refused):
    status = 429


class TooLarge(Refused):
    status = 413


def lane(name: str):
    """A lane module (teaching, coaching, growth, missions) when it exists; its own import errors are not hidden."""
    if importlib.util.find_spec(f"{__package__}.{name}") is None:
        return None
    return importlib.import_module(f"{__package__}.{name}")
