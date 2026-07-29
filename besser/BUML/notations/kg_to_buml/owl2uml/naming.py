"""Identifier sanitisation and auxiliary-class naming conventions.

The aux-class names follow the tokens used in ``conversion_kg2uml.tex`` exactly:
``_some_p_D``, ``_min_n_p_D``, ``_all_p_D``, ``_not_C``,
``_C1_..._Cn_Union``/``_Intersection``, ``_hasValue_p_a`` (O08), etc.
"""
from __future__ import annotations

import re

_INVALID = re.compile(r"[^0-9A-Za-z_]")


def local_name(iri: str) -> str:
    """Return the local part of an IRI (after the last ``#``, ``/`` or ``:``).

    The ``:`` fallback only applies to IRIs with no ``#`` or ``/`` at all — in
    practice URN-style identifiers such as ``urn:besser:Thing``, which the KG
    editor mints for synthetic classes. An ``http(s)`` IRI always contains ``/``
    and so never reaches it.
    """
    s = str(iri).rstrip("#/")
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
            return s or str(iri)
    if ":" in s:
        s = s.rsplit(":", 1)[-1]
    return s or str(iri)


def sanitize(name: str) -> str:
    """Turn an arbitrary label into a valid UML/PlantUML identifier."""
    s = _INVALID.sub("_", name)
    if not s:
        s = "_"
    if s[0].isdigit():
        s = "_" + s
    return s


def namespace_hint(iri: str) -> str:
    """A short, human-ish token identifying an IRI's namespace.

    Used to disambiguate two IRIs that share a local name
    """
    s = str(iri)
    # strip the local part
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[0]
    # take the last meaningful path/host token
    token = local_name(s)
    token = token.replace(".owl", "").replace(".rdf", "")
    return sanitize(token) or "ns"


# --- auxiliary-class names ------------------------------------------------

def union_name(operands: list[str]) -> str:
    return "_" + "_".join(operands) + "_Union"


def intersection_name(operands: list[str]) -> str:
    return "_" + "_".join(operands) + "_Intersection"


def one_of_name(individuals: list[str]) -> str:
    return "_" + "_".join(individuals) + "_OneOf"


def not_name(cls: str) -> str:
    return f"_not_{cls}"


def restriction_name(kind: str, prop: str, filler: str, n: int | None = None) -> str:
    """Aux name for a property restriction.

    ``kind`` is one of: some, all, min, max, exact, hasValue, hasSelf.
    """
    if kind == "hasSelf":
        return f"_hasSelf_{prop}"
    if n is not None:
        return f"_{kind}_{n}_{prop}_{filler}"
    return f"_{kind}_{prop}_{filler}"
