"""Knowledge Graph diagram processing — JSON → BUML.

Turns the KG diagram JSON (produced by the web editor / ``/import-owl``) into
a ``KnowledgeGraph`` metamodel instance.
"""

from __future__ import annotations

from typing import Any, Dict, List

from besser.BUML.metamodel.kg import (
    ImportAxiom,
    KGAxiom,
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNodeConstraint,
    KGProperty,
    KGPropertyConstraint,
    KnowledgeGraph,
)
from besser.utilities.web_modeling_editor.backend.services.converters.json_to_buml.utils import (
    sanitize_name,
)
from besser.utilities.web_modeling_editor.backend.services.exceptions import ConversionError


# Derived from the class hierarchy, not hand-listed. The hand-written table
# was missing EquivalentPropertiesAxiom, and because an unknown kind was
# silently dropped, nothing failed: bibo.ttl simply came back with 51 of its
# 61 axioms. Deriving it means a new KGAxiom subclass round-trips the day it
# is added.
_AXIOM_KIND_TO_CLASS = {cls.__name__: cls for cls in KGAxiom.__subclasses__()}


_NODE_TYPE_TO_CLASS = {
    "class": KGClass,
    "individual": KGIndividual,
    "property": KGProperty,
    "blank": KGBlank,
    "nodeconstraint": KGNodeConstraint,
    "propertyconstraint": KGPropertyConstraint,
}


def process_kg_diagram(json_data: Dict[str, Any]) -> KnowledgeGraph:
    """Convert a KG diagram JSON payload into a ``KnowledgeGraph``.

    Accepts either the diagram envelope ``{"title": ..., "model": {...}}`` or
    the raw model dict ``{"nodes": [...], "edges": [...]}``.

    Editor view state is ignored: a node's ``position`` and the model's
    ``settings`` have no place in the metamodel and do not survive B-UML (see
    the ``kg_diagram_converter`` module docstring).

    Raises:
        ConversionError: For any malformed node, edge or axiom.
    """
    model = json_data.get("model") if isinstance(json_data, dict) else None
    if not model:
        model = json_data or {}

    if not isinstance(model, dict):
        raise ConversionError(
            f"KG diagram model must be an object, got {type(model).__name__}."
        )

    raw_title = (json_data.get("title") if isinstance(json_data, dict) else None) or "knowledge_graph"
    title = sanitize_name(raw_title)

    kg = KnowledgeGraph(name=title)

    for raw_node in _entries(model, "nodes"):
        node = _build_node(raw_node)
        kg.add_node(node)

    nodes_by_id = {n.id: n for n in kg.nodes}
    for raw_edge in _entries(model, "edges"):
        edge = _build_edge(raw_edge, nodes_by_id)
        kg.add_edge(edge)

    for raw_axiom in _entries(model, "axioms"):
        kg.add_axiom(_build_axiom(raw_axiom, nodes_by_id))

    return kg


def _entries(model: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    """The ``key`` list of ``model``, with every entry checked to be an object.

    A non-dict entry used to reach ``raw.get`` and raise ``AttributeError``,
    which the project endpoint turned into a 500.
    """
    raw = model.get(key) or []
    if not isinstance(raw, list):
        raise ConversionError(f"KG diagram '{key}' must be a list, got {type(raw).__name__}.")
    singular = key[:-1]
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConversionError(
                f"KG {singular} #{index} must be an object, got {type(entry).__name__}."
            )
    return raw


def _build_axiom(raw: Dict[str, Any], nodes_by_id: Dict[str, Any]):
    """Rebuild one axiom from its JSON form.

    Raises ``ConversionError`` rather than returning ``None`` for anything
    malformed, matching :func:`_build_node` / :func:`_build_edge`. Returning
    ``None`` meant a bad axiom vanished from the model without a word.

    Every field the axiom class declares in ``FIELDS`` must be present, every
    id field non-empty, and every id must name a node of the graph: the
    constructors default each field to ``""`` / ``[]``, so a bare
    ``{"kind": "SubPropertyOfAxiom"}`` would otherwise build an axiom about
    nothing. ``ImportAxiom`` is the exception: it carries ontology IRIs, not
    node ids, and its ``source_ontology_iri`` is optional.
    """
    kind = raw.get("kind")
    if not kind:
        raise ConversionError("KG axiom is missing 'kind'.")
    cls = _AXIOM_KIND_TO_CLASS.get(kind)
    if cls is None:
        known = ", ".join(sorted(_AXIOM_KIND_TO_CLASS))
        raise ConversionError(f"Unknown KG axiom kind '{kind}'. Expected one of: {known}.")
    payload = {k: v for k, v in raw.items() if k != "kind"}
    try:
        axiom = cls(**payload)
    except (TypeError, ValueError) as exc:
        # TypeError: an unknown or duplicated field; ValueError: a field of the wrong type.
        raise ConversionError(f"Malformed '{kind}' axiom: {exc}") from exc
    missing = [name for name in cls.FIELDS if name not in payload]
    if missing:
        raise ConversionError(f"Malformed '{kind}' axiom: missing field(s) {', '.join(missing)}.")
    if cls is ImportAxiom:
        if not axiom.target_iri:
            raise ConversionError("Malformed 'ImportAxiom' axiom: 'target_iri' is empty.")
        return axiom
    for name in cls.FIELDS:
        value = getattr(axiom, name)
        ids = value if isinstance(value, list) else [value]
        if not isinstance(value, list) and not value:
            raise ConversionError(f"Malformed '{kind}' axiom: '{name}' is empty.")
        unknown = [node_id for node_id in ids if node_id not in nodes_by_id]
        if unknown:
            raise ConversionError(
                f"'{kind}' axiom field '{name}' references unknown node(s) "
                f"{', '.join(repr(u) for u in unknown)}."
            )
    return axiom


def _build_node(raw: Dict[str, Any]):
    node_type = (raw.get("nodeType") or "").lower()
    node_id = raw.get("id")
    if not node_id:
        raise ConversionError("KG node is missing 'id'.")
    label = raw.get("label", "") or ""
    iri = raw.get("iri")
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else None

    if node_type == "literal":
        value = raw.get("value")
        if value is None:
            value = label or ""
        datatype = raw.get("datatype")
        return KGLiteral(id=node_id, value=value, datatype=datatype, label=label, metadata=metadata)

    cls = _NODE_TYPE_TO_CLASS.get(node_type)
    if cls is None:
        raise ConversionError(
            f"Unknown KG nodeType '{raw.get('nodeType')!r}' (expected one of "
            "'class', 'individual', 'property', 'literal', 'blank', "
            "'nodeConstraint', 'propertyConstraint')."
        )
    return cls(id=node_id, label=label, iri=iri, metadata=metadata)


def _build_edge(raw: Dict[str, Any], nodes_by_id: Dict[str, Any]) -> KGEdge:
    edge_id = raw.get("id")
    if not edge_id:
        raise ConversionError("KG edge is missing 'id'.")
    source_id = raw.get("source")
    target_id = raw.get("target")
    if not isinstance(source_id, str) or not isinstance(target_id, str):
        raise ConversionError(f"KG edge {edge_id!r} must name its source and target by id.")
    if source_id not in nodes_by_id:
        raise ConversionError(f"KG edge {edge_id!r} references unknown source node {source_id!r}.")
    if target_id not in nodes_by_id:
        raise ConversionError(f"KG edge {edge_id!r} references unknown target node {target_id!r}.")
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else None
    return KGEdge(
        id=edge_id,
        source=nodes_by_id[source_id],
        target=nodes_by_id[target_id],
        label=raw.get("label", "") or "",
        iri=raw.get("iri"),
        metadata=metadata,
    )

