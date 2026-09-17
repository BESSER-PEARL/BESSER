"""Shared primary-key python-type map for code generators.

A ForeignKey field, path parameter, or relationship slot must use the
SAME python type as the primary key it references — a ``guest: int``
Pydantic field pointing at a String PK 422-rejects every real id even
though everything imports. Generators that emit cross-entity references
build this map once and thread it into their templates.
"""

from __future__ import annotations

_PK_PY_TYPES = {
    "str": "str",
    "string": "str",
    "int": "int",
    "integer": "int",
    "float": "float",
}


def pk_python_types(model) -> dict:
    """Class name -> python type of its primary key (default ``int``).

    Delegates to ``structural_utils.get_pk_py_types`` so there is ONE
    implementation: this file previously carried its own copy that looked only
    at a class's own attributes, so a subclass in joined-table inheritance --
    whose PK *is* the parent's, emitted as a ForeignKey to it -- was absent
    from the map and every reference to it fell back to ``int`` against a
    String PK.
    """
    try:
        from besser.generators.structural_utils import get_pk_py_types
        return get_pk_py_types(model)
    except Exception:
        return {}
