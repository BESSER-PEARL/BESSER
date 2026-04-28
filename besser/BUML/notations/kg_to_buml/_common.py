"""Shared helpers for the KG → BUML notation."""

from __future__ import annotations

import keyword
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urldefrag

from besser.BUML.metamodel.kg import KGEdge, KGNode, KnowledgeGraph

# --- Well-known predicate IRIs (full form, as emitted by rdflib) ---
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS_OF = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
RDFS_RANGE = "http://www.w3.org/2000/01/rdf-schema#range"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

# Compact forms occasionally appear when KGs are hand-edited in the web
# editor; treat both forms as equivalent.
_PREDICATE_ALIASES: Dict[str, str] = {
    "rdf:type": RDF_TYPE,
    "a": RDF_TYPE,
    "rdfs:subClassOf": RDFS_SUBCLASS_OF,
    "rdfs:domain": RDFS_DOMAIN,
    "rdfs:range": RDFS_RANGE,
    "rdfs:label": RDFS_LABEL,
}


def normalize_predicate(iri_or_label: Optional[str]) -> Optional[str]:
    """Normalise a predicate identifier to its canonical full IRI when possible."""
    if not iri_or_label:
        return iri_or_label
    return _PREDICATE_ALIASES.get(iri_or_label, iri_or_label)


def local_name(iri: Optional[str]) -> str:
    """Return the local-name part of an IRI (after ``#`` or the last ``/``)."""
    if not iri:
        return ""
    # Strip fragment first; if there is one, that's the local name.
    base, frag = urldefrag(iri)
    if frag:
        return frag
    if "/" in base:
        return base.rsplit("/", 1)[-1]
    if ":" in base:
        return base.rsplit(":", 1)[-1]
    return base


_IDENT_FIRST_CHAR = re.compile(r"[^A-Za-z_]")
_IDENT_OTHER = re.compile(r"[^A-Za-z0-9_]")


def sanitize_python_identifier(
    label: Optional[str],
    fallback_prefix: str,
    used_names: Set[str],
) -> str:
    """Produce a unique Python-identifier-safe name based on ``label``.

    - Replaces any non-identifier characters with ``_``.
    - Prefixes a leading digit with the fallback prefix.
    - Avoids collisions by suffixing ``_2``, ``_3``, … against ``used_names``.
    - Avoids Python keywords.

    The chosen name is added to ``used_names`` before returning.
    """
    base = (label or "").strip()
    if not base:
        base = fallback_prefix
    base = _IDENT_OTHER.sub("_", base)
    if not base or _IDENT_FIRST_CHAR.match(base[0]) or base[0].isdigit():
        base = f"{fallback_prefix}_{base.lstrip('_')}" if base.lstrip("_") else fallback_prefix
    base = base.strip("_") or fallback_prefix
    if keyword.iskeyword(base):
        base = f"{base}_"

    candidate = base
    suffix = 2
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


@dataclass
class KGConversionWarning:
    """A non-fatal anomaly detected while converting a KG to a BUML model."""

    code: str
    message: str
    node_id: Optional[str] = None
    edge_id: Optional[str] = None


@dataclass
class GraphIndexes:
    """Pre-computed lookup tables over a :class:`KnowledgeGraph`."""

    nodes_by_id: Dict[str, KGNode] = field(default_factory=dict)
    out_edges: Dict[str, List[KGEdge]] = field(default_factory=lambda: defaultdict(list))
    in_edges: Dict[str, List[KGEdge]] = field(default_factory=lambda: defaultdict(list))

    def out_with_predicate(self, node_id: str, predicate_iri: str) -> List[KGEdge]:
        """All outgoing edges of ``node_id`` whose predicate matches ``predicate_iri``."""
        return [e for e in self.out_edges.get(node_id, []) if normalize_predicate(e.iri) == predicate_iri]

    def in_with_predicate(self, node_id: str, predicate_iri: str) -> List[KGEdge]:
        """All incoming edges of ``node_id`` whose predicate matches ``predicate_iri``."""
        return [e for e in self.in_edges.get(node_id, []) if normalize_predicate(e.iri) == predicate_iri]


def build_indexes(kg: KnowledgeGraph) -> GraphIndexes:
    """Build :class:`GraphIndexes` from a :class:`KnowledgeGraph`."""
    idx = GraphIndexes()
    for node in kg.nodes:
        idx.nodes_by_id[node.id] = node
    for edge in kg.edges:
        idx.out_edges[edge.source.id].append(edge)
        idx.in_edges[edge.target.id].append(edge)
    return idx


def sorted_by_id(items):
    """Stable sort by ``.id`` for deterministic output."""
    return sorted(items, key=lambda x: x.id)


def add_warning(
    warnings: List[KGConversionWarning],
    code: str,
    message: str,
    *,
    node_id: Optional[str] = None,
    edge_id: Optional[str] = None,
) -> None:
    warnings.append(KGConversionWarning(code=code, message=message, node_id=node_id, edge_id=edge_id))


__all__ = [
    "RDF_TYPE",
    "RDFS_SUBCLASS_OF",
    "RDFS_DOMAIN",
    "RDFS_RANGE",
    "RDFS_LABEL",
    "normalize_predicate",
    "local_name",
    "sanitize_python_identifier",
    "KGConversionWarning",
    "GraphIndexes",
    "build_indexes",
    "sorted_by_id",
    "add_warning",
]
