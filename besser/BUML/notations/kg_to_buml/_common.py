"""Shared helpers for the KG → BUML notation."""

from __future__ import annotations

import hashlib
import keyword
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from besser.BUML.metamodel.kg import KGEdge, KGNode, KnowledgeGraph
from besser.BUML.metamodel.kg.iri import local_name as _local_name
from besser.BUML.metamodel.kg.constants import (
    OWL as OWL_NS,
    RDF as RDF_NS,
    RDFS as RDFS_NS,
    SH as SH_NS,
    XSD as XSD_NS,
)

# --- Well-known predicate IRIs (full form, as emitted by rdflib) ---
RDF_TYPE = f"{RDF_NS}type"
RDFS_SUBCLASS_OF = f"{RDFS_NS}subClassOf"
RDFS_DOMAIN = f"{RDFS_NS}domain"
RDFS_RANGE = f"{RDFS_NS}range"
RDFS_LABEL = f"{RDFS_NS}label"

# --- Framework namespaces, used by the datatype-IRI predicates below ---
XSD_NAMESPACE = XSD_NS
RDF_NAMESPACE = RDF_NS
RDFS_NAMESPACE = RDFS_NS


def looks_like_datatype_iri(iri: Optional[str]) -> bool:
    """True when ``iri`` names a literal datatype rather than a class.

    Every ``xsd:*`` term, plus the three RDF/RDFS literal types that behave
    like datatypes in an ontology (``rdf:langString``, ``rdfs:Literal``,
    ``rdf:PlainLiteral``).
    """
    if not iri:
        return False
    return iri.startswith(XSD_NAMESPACE) or iri in {
        f"{RDF_NAMESPACE}langString",
        f"{RDFS_NAMESPACE}Literal",
        f"{RDF_NAMESPACE}PlainLiteral",
    }


def all_datatype_refs(members: List[str]) -> bool:
    """True when ``members`` is non-empty and every member is a datatype IRI."""
    return bool(members) and all(looks_like_datatype_iri(m) for m in members)


# Private aliases kept for call sites that imported the underscore names.
_looks_like_datatype_iri = looks_like_datatype_iri
_all_datatype_refs = all_datatype_refs

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
    """True when ``iri`` is in OWL / RDF / RDFS / XSD framework vocabulary."""
    if not iri:
        return False
    return any(iri.startswith(ns) for ns in _META_VOCAB_NAMESPACES)

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


# Re-exported: the one implementation lives in the metamodel package.
local_name = _local_name


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


def short_digest(raw: str, length: int, *, algo: str = "sha1") -> str:
    """Hex digest of ``raw``, truncated to ``length`` characters.

    The shared tail of every id/signature helper in the KG code. What goes into
    ``raw`` (separators, sort order) and the ``algo``/``length`` are deliberately
    left to each caller: those ids are echoed to the editor and back (issue ids,
    ``kgSignature``), so changing one invalidates a round-trip.

    Encoding uses ``errors="replace"`` so a lone surrogate in a client-supplied
    id degrades to ``?`` instead of raising ``UnicodeEncodeError`` (a 500).
    """
    return hashlib.new(algo, raw.encode("utf-8", errors="replace")).hexdigest()[:length]


def sorted_by_id(items):
    """Stable sort by ``.id`` for deterministic output."""
    return sorted(items, key=lambda x: x.id)


def find_cycles(parents: Dict[str, Set[str]]) -> List[Tuple[str, ...]]:
    """Find the ``rdfs:subClassOf`` cycles in a ``child -> {parents}`` map.

    Returns one canonical tuple per cycle, each closed (first id repeated last)
    and rotated to start at its lexicographically smallest member, so the same
    cycle is never reported twice under a different rotation. Sorted, so the
    result is stable across runs and hash seeds.

    Uses the three-colour DFS that :func:`to_buml._break_cycles` already relies
    on: a node is grey while it is on the current path and black once fully
    explored, and only a grey hit is a cycle. The earlier implementation here
    tracked just the current path, so a node reachable by several routes was
    re-explored once per route -- it enumerated every simple path rather than
    every cycle, and a cycle-free stack of diamonds was enough to make it
    quadratic then exponential (55 classes took 1.7 s, and preflight runs this
    on every call and again after each decision).
    """
    GREY, BLACK = 0, 1
    state: Dict[str, int] = {}
    path: List[str] = []
    found: Set[Tuple[str, ...]] = set()

    def visit(node: str) -> None:
        state[node] = GREY
        path.append(node)
        for parent in sorted(parents.get(node, ())):
            if state.get(parent) == GREY:
                cycle = path[path.index(parent):]
                # Rotate so the smallest id leads: A->B->A and B->A->B are one
                # cycle, and reporting both makes applying both decisions fail.
                pivot = cycle.index(min(cycle))
                rotated = cycle[pivot:] + cycle[:pivot]
                found.add(tuple(rotated) + (rotated[0],))
            elif parent not in state:
                visit(parent)
        path.pop()
        state[node] = BLACK

    for node in sorted(parents):
        if node not in state:
            visit(node)

    return sorted(found)


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
    "find_cycles",
    "short_digest",
    "RDF_TYPE",
    "RDFS_SUBCLASS_OF",
    "RDFS_DOMAIN",
    "RDFS_RANGE",
    "RDFS_LABEL",
    "XSD_NAMESPACE",
    "RDF_NAMESPACE",
    "RDFS_NAMESPACE",
    "looks_like_datatype_iri",
    "all_datatype_refs",
    "is_meta_vocab",
    "normalize_predicate",
    "local_name",
    "sanitize_python_identifier",
    "KGConversionWarning",
    "GraphIndexes",
    "build_indexes",
    "sorted_by_id",
    "add_warning",
]
