"""IRI helpers shared by the KG importer, exporter and converters."""

from __future__ import annotations

from typing import Optional

__all__ = ["local_name"]


def local_name(iri: Optional[str]) -> str:
    """Return the local part of an IRI (after the last ``#``, ``/`` or ``:``).

    The ``:`` fallback only applies to IRIs with no ``#`` or ``/`` at all — in
    practice URN-style identifiers such as ``urn:besser:Thing``, which the KG
    editor mints for synthetic classes. An ``http(s)`` IRI always contains ``/``
    and so never reaches it.

    Trailing ``#`` / ``/`` are ignored, so an ontology IRI such as
    ``http://purl.org/ontology/bibo/`` yields ``bibo`` rather than an empty
    string. If stripping leaves nothing, the IRI itself is returned.

    ``None`` and ``""`` yield ``""``.
    """
    if not iri:
        return ""
    s = str(iri).rstrip("#/")
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
            return s or str(iri)
    if ":" in s:
        s = s.rsplit(":", 1)[-1]
    return s or str(iri)
