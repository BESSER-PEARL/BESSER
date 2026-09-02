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

    Classes with no declared id attribute are absent — they get the
    integer surrogate and every layer already agrees on ``int``.
    """
    pk_types: dict = {}
    try:
        classes = list(model.get_classes())
    except Exception:
        return pk_types
    for cls in classes:
        id_attr = next((a for a in cls.attributes if getattr(a, "is_id", False)), None)
        if id_attr is None:
            id_attr = next((a for a in cls.attributes if a.name == "id"), None)
        if id_attr is not None:
            type_name = (getattr(id_attr.type, "name", "") or "").lower()
            pk_types[cls.name] = _PK_PY_TYPES.get(type_name, "int")
    return pk_types
