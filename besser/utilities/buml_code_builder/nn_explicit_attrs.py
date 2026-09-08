"""
Sidecar bookkeeping for "attribute was explicitly set by the user" metadata
on NN metamodel objects.

The web editor needs to round-trip whether a boolean/numeric attribute (e.g.
``input_reused``, ``bidirectional``, ``dropout``) was explicitly ticked or
left at its default, because metamodel defaults and user-set defaults are
indistinguishable in the object itself. Previously this was tracked by
writing a ``layer._set_attrs`` attribute directly onto metamodel objects,
which mutates shared domain types from outside their package.

This module keeps the bookkeeping in a ``WeakKeyDictionary`` keyed on the
metamodel object, so the metamodel classes stay untouched and entries are
garbage-collected automatically when the model is dropped.

Importers (json_to_buml) call :func:`mark_explicit` when a user-facing
attribute is present in the JSON; exporters (buml_to_json, code builders)
call :func:`is_explicit` to decide whether to emit it.
"""

from __future__ import annotations

from typing import Set
from weakref import WeakKeyDictionary

_EXPLICIT: "WeakKeyDictionary[object, Set[str]]" = WeakKeyDictionary()


def mark_explicit(obj: object, attr_name: str) -> None:
    """Record that ``attr_name`` was explicitly set on ``obj``."""
    bucket = _EXPLICIT.get(obj)
    if bucket is None:
        bucket = set()
        _EXPLICIT[obj] = bucket
    bucket.add(attr_name)


def is_explicit(obj: object, attr_name: str) -> bool:
    """True if ``attr_name`` was explicitly marked on ``obj``."""
    bucket = _EXPLICIT.get(obj)
    return bucket is not None and attr_name in bucket


def clear(obj: object) -> None:
    """Remove all tracked attributes for ``obj`` (test helper)."""
    _EXPLICIT.pop(obj, None)
