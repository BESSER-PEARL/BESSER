"""Vocabulary checks and the warning record shared by the KG transformations.

Used by this package and by :mod:`besser.BUML.notations.kg_to_buml`, which
re-exports them from its own ``_common``. They live here, below both, so the
RDF projection does not have to import the KG → B-UML package that consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from besser.BUML.metamodel.kg.constants import (
    OWL as OWL_NS,
    RDF as RDF_NS,
    RDFS as RDFS_NS,
    SH as SH_NS,
    XSD as XSD_NS,
)

__all__ = [
    "KGConversionWarning",
    "add_warning",
    "is_meta_vocab",
    "looks_like_datatype_iri",
]


#: The RDF/RDFS literal types that behave like datatypes in an ontology.
_LITERAL_TYPE_IRIS = frozenset({
    f"{RDF_NS}langString",
    f"{RDFS_NS}Literal",
    f"{RDF_NS}PlainLiteral",
})


def looks_like_datatype_iri(iri: Optional[str]) -> bool:
    """True when ``iri`` names a literal datatype rather than a class.

    Every ``xsd:*`` term, plus the three RDF/RDFS literal types that behave
    like datatypes in an ontology (``rdf:langString``, ``rdfs:Literal``,
    ``rdf:PlainLiteral``).
    """
    if not iri:
        return False
    return iri.startswith(XSD_NS) or iri in _LITERAL_TYPE_IRIS


# OWL/RDF/RDFS/XSD/SHACL framework namespaces. IRIs in these namespaces are
# vocabulary terms (``owl:Class``, ``owl:DatatypeProperty``, ``rdfs:Class``,
# ``sh:NodeShape``, …), not user concepts. They appear as objects of
# ``rdf:type`` triples in every well-formed ontology, but they should NEVER
# produce user-facing BUML classes when the converter synthesises classes from
# references.
_META_VOCAB_NAMESPACES = (
    OWL_NS,
    RDF_NS,
    RDFS_NS,
    XSD_NS,
    SH_NS,
)


def is_meta_vocab(iri: Optional[str]) -> bool:
    """True when ``iri`` is in OWL / RDF / RDFS / XSD / SHACL framework vocabulary."""
    if not iri:
        return False
    return any(iri.startswith(ns) for ns in _META_VOCAB_NAMESPACES)


@dataclass
class KGConversionWarning:
    """A non-fatal anomaly detected while projecting a KG to RDF or converting it to a BUML model."""

    code: str
    message: str
    node_id: Optional[str] = None
    edge_id: Optional[str] = None


def add_warning(
    warnings: List[KGConversionWarning],
    code: str,
    message: str,
    *,
    node_id: Optional[str] = None,
    edge_id: Optional[str] = None,
) -> None:
    """Append a :class:`KGConversionWarning` to ``warnings``.

    ``code`` is the stable machine-readable identifier the editor keys on,
    ``message`` the human text; ``node_id`` / ``edge_id`` point at the KG element
    concerned, when there is one.
    """
    warnings.append(KGConversionWarning(code=code, message=message, node_id=node_id, edge_id=edge_id))
