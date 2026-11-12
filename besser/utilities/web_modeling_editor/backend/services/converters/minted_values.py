"""The ids and timestamps a conversion to editor JSON makes up: fresh by default, fixed on request.

A model written in Python has no editor ids and no save time, so ``buml_to_json``
mints them: a new uuid per element, diagram and project, and the current time.
That is what the editor wants when it imports a file. A caller that needs the
same JSON for the same model (``besser convert``, ``besser new``) wraps the
conversion in :func:`fixed_minted_values`: every id is then derived from a seed,
the scope it is minted in (:func:`next_id_scope`) and its count in that scope, and
every timestamp is the one given.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator, Optional

# Namespace of the ids derived from a seed (uuid5), so they cannot collide with another tool's uuid5 ids.
_SEEDED_ID_NAMESPACE = uuid.UUID("5d3f6c2e-8a41-4f0b-9c7d-2b1e0a9f4c63")


@dataclass
class _Fixed:
    seed: str
    timestamp: str
    scope: int = field(default=0)
    count: int = field(default=0)


_fixed: ContextVar[Optional[_Fixed]] = ContextVar("besser_minted_values", default=None)


def new_id() -> str:
    """A new element, diagram or project id: a fresh uuid4, or the next id derived from the seed."""
    fixed = _fixed.get()
    if fixed is None:
        return str(uuid.uuid4())
    fixed.count += 1
    return str(uuid.uuid5(_SEEDED_ID_NAMESPACE, f"{fixed.seed}:{fixed.scope}:{fixed.count}"))


def next_id_scope() -> None:
    """Start the next scope of derived ids (one per diagram): an id an element carries from its
    model, which takes no derived id, then shifts only the ids after it in its own diagram."""
    fixed = _fixed.get()
    if fixed is not None:
        fixed.scope += 1
        fixed.count = 0


def now_iso() -> str:
    """The time to record as a save or creation time: now in UTC, or the fixed timestamp."""
    fixed = _fixed.get()
    if fixed is None:
        return datetime.now(timezone.utc).isoformat()
    return fixed.timestamp


@contextmanager
def fixed_minted_values(seed: str, timestamp: str) -> Iterator[None]:
    """Within the block, ids come from ``seed`` (the n-th id minted is the same on every run) and
    every timestamp is ``timestamp``. The same input then converts to the same JSON."""
    token = _fixed.set(_Fixed(seed=seed, timestamp=timestamp))
    try:
        yield
    finally:
        _fixed.reset(token)
