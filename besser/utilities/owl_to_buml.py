"""OWL / RDF → KnowledgeGraph conversion.

Deterministic, offline parser that loads an RDF graph with rdflib and maps
each term (URI, blank node, literal) onto the five KG node types used by the
web editor's Knowledge Graph diagram. On top of the lossless triple → edge
emission, the importer runs a small dispatch pipeline of decoders that lift
OWL/OWL2 constructs into structured annotations:

* property characteristics (``owl:FunctionalProperty`` etc.) → ``KGProperty.metadata``
* ``owl:Restriction`` blank nodes → ``KGBlank.metadata`` (kind, on_property, …)
* ``owl:unionOf`` / ``intersectionOf`` / ``complementOf`` / ``oneOf`` → ``KGBlank.metadata``
* ``rdfs:comment`` / ``rdfs:isDefinedBy`` → ``KGNode.metadata['description' / 'defined_by']``
* ``owl:imports`` → ``ImportAxiom`` (logged, not followed)
* OWL axioms that don't fit on one node/edge (``equivalentClass``, ``disjointWith``,
  ``disjointUnionOf``, ``subPropertyOf``, ``inverseOf``, ``propertyChainAxiom``,
  ``hasKey``) → ``KnowledgeGraph.axioms``
* OWL2 punning (same IRI as class + individual) → both nodes kept, linked via
  ``metadata['punned_with']``
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import rdflib
from rdflib import RDF, RDFS, OWL, BNode, Literal, Namespace, URIRef

from besser.BUML.metamodel.kg import (
    DisjointClassesAxiom,
    DisjointUnionAxiom,
    EquivalentClassesAxiom,
    HasKeyAxiom,
    ImportAxiom,
    InversePropertiesAxiom,
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNode,
    KGNodeConstraint,
    KGProperty,
    KGPropertyConstraint,
    KnowledgeGraph,
    PropertyChainAxiom,
    SubPropertyOfAxiom,
)
from besser.BUML.metamodel.kg.constants import (
    CONSTRAINT_TARGET_CLASS,
    CONSTRAINT_TARGET_PROPERTY,
    SH_PATH,
    SH_PROPERTY,
)


SH = Namespace("http://www.w3.org/ns/shacl#")


__all__ = ["owl_file_to_knowledge_graph"]


_PROPERTY_TYPES = {
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    RDF.Property,
}
_CLASS_TYPES = {OWL.Class, RDFS.Class}

_XSD_NAMESPACE = "http://www.w3.org/2001/XMLSchema#"
_RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDFS_NAMESPACE = "http://www.w3.org/2000/01/rdf-schema#"
_DATATYPE_IRIS = {
    f"{_RDF_NAMESPACE}langString",
    f"{_RDF_NAMESPACE}PlainLiteral",
    f"{_RDFS_NAMESPACE}Literal",
}

# OWL property characteristics: each maps a term type to a metadata flag
# stored on KGProperty.metadata.characteristics (a set of strings).
_PROPERTY_CHARACTERISTIC_TYPES: Dict[Any, str] = {
    OWL.FunctionalProperty: "Functional",
    OWL.InverseFunctionalProperty: "InverseFunctional",
    OWL.TransitiveProperty: "Transitive",
    OWL.SymmetricProperty: "Symmetric",
    OWL.AsymmetricProperty: "Asymmetric",
    OWL.ReflexiveProperty: "Reflexive",
    OWL.IrreflexiveProperty: "Irreflexive",
}

# Restriction predicates that carry a literal cardinality value.
_CARDINALITY_RESTRICTIONS = {
    OWL.cardinality: "cardinality",
    OWL.minCardinality: "minCardinality",
    OWL.maxCardinality: "maxCardinality",
    OWL.qualifiedCardinality: "qualifiedCardinality",
    OWL.minQualifiedCardinality: "minQualifiedCardinality",
    OWL.maxQualifiedCardinality: "maxQualifiedCardinality",
}

# Restriction predicates that carry a class/value reference.
_VALUE_RESTRICTIONS = {
    OWL.someValuesFrom: "someValuesFrom",
    OWL.allValuesFrom: "allValuesFrom",
    OWL.hasValue: "hasValue",
    OWL.hasSelf: "hasSelf",
}


def _is_datatype_iri(iri: str) -> bool:
    """Return True for XSD datatype IRIs and the few RDF/RDFS literal-shaped IRIs.

    Datatype IRIs sometimes appear as objects of ``rdfs:range`` even though they
    are never declared via ``rdf:type owl:Class``. Treat them as KGClass nodes
    so the BUML class-diagram converter can filter them out via its existing
    ``_looks_like_datatype_iri`` check (rather than letting them leak into the
    graph as KGIndividual nodes).
    """
    if not iri:
        return False
    return iri.startswith(_XSD_NAMESPACE) or iri in _DATATYPE_IRIS


def _local_name(iri: str) -> str:
    """Return the local name of an IRI (after ``#`` or the last ``/``)."""
    if not iri:
        return ""
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    if "/" in iri:
        return iri.rsplit("/", 1)[-1]
    return iri


def _literal_id(lit: Literal) -> str:
    """Stable id for a literal based on lexical form + datatype."""
    raw = f"{lit}\0{lit.datatype or ''}\0{lit.language or ''}"
    digest = hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"lit:{digest}"


def _term_str(term) -> Optional[str]:
    """Return the IRI string of a URIRef, or None for non-URIRefs."""
    if isinstance(term, URIRef):
        return str(term)
    return None


def _classify(term, classes: set, properties: set) -> str:
    """Decide the KG node kind for an rdflib term (``class`` / ``individual`` /
    ``property`` / ``literal`` / ``blank``)."""
    if isinstance(term, Literal):
        return "literal"
    if isinstance(term, BNode):
        return "blank"
    if isinstance(term, URIRef):
        if term in classes:
            return "class"
        if term in properties:
            return "property"
        if _is_datatype_iri(str(term)):
            return "class"
        return "individual"
    return "individual"


# ----------------------------------------------------------------------
# Decoders — each takes the rdflib graph and produces structured payloads
# keyed by rdflib terms (URIRef / BNode). They are pure functions and run
# before nodes are emitted, so the node-emission loop can attach metadata.
# ----------------------------------------------------------------------


def _collect_property_kinds(g: rdflib.Graph) -> Dict[URIRef, Dict[str, Any]]:
    """Detect OWL property characteristics for each property IRI.

    Returns a mapping ``{property_term: {"characteristics": {"Functional", ...},
    "kind": "Object"|"Datatype"|"Annotation"|"Property"}}``. Properties without
    any explicit characteristic still appear in the map if they have a
    declared ``rdf:type`` matching ``_PROPERTY_TYPES``.
    """
    out: Dict[URIRef, Dict[str, Any]] = {}
    for char_type, label in _PROPERTY_CHARACTERISTIC_TYPES.items():
        for s, _, _ in g.triples((None, RDF.type, char_type)):
            if not isinstance(s, URIRef):
                continue
            entry = out.setdefault(s, {"characteristics": set()})
            entry["characteristics"].add(label)
    # Also record the most specific declared property kind, if any.
    kind_priority = [
        (OWL.ObjectProperty, "Object"),
        (OWL.DatatypeProperty, "Datatype"),
        (OWL.AnnotationProperty, "Annotation"),
        (RDF.Property, "Property"),
    ]
    for type_term, label in kind_priority:
        for s, _, _ in g.triples((None, RDF.type, type_term)):
            if not isinstance(s, URIRef):
                continue
            entry = out.setdefault(s, {"characteristics": set()})
            entry.setdefault("kind", label)
    # Convert characteristic sets to sorted lists for deterministic output.
    for entry in out.values():
        entry["characteristics"] = sorted(entry["characteristics"])
    return out


def _decode_lists(g: rdflib.Graph) -> Dict[BNode, List[Any]]:
    """Resolve every ``rdf:List`` head into a Python list of items.

    Walks ``rdf:first`` / ``rdf:rest`` chains. Loops or malformed lists
    short-circuit with whatever we have so far (RDF lists are well-formed
    in practice, but we don't trust the input).
    """
    list_heads: Set[BNode] = set()
    for s, _, _ in g.triples((None, RDF.first, None)):
        if isinstance(s, BNode):
            list_heads.add(s)
    # Only true *heads* — those that aren't the rest of another list.
    rests: Set[BNode] = set()
    for _, _, o in g.triples((None, RDF.rest, None)):
        if isinstance(o, BNode):
            rests.add(o)
    heads = list_heads - rests
    out: Dict[BNode, List[Any]] = {}
    for head in heads:
        items: List[Any] = []
        seen: Set[BNode] = set()
        cursor: Any = head
        while isinstance(cursor, BNode) and cursor not in seen:
            seen.add(cursor)
            first = next(g.objects(cursor, RDF.first), None)
            if first is None:
                break
            items.append(first)
            cursor = next(g.objects(cursor, RDF.rest), None)
            if cursor == RDF.nil or cursor is None:
                break
        out[head] = items
    return out


def _decode_restrictions(
    g: rdflib.Graph,
    lists: Dict[BNode, List[Any]],
) -> Dict[BNode, Dict[str, Any]]:
    """Decode every ``owl:Restriction`` blank node into a structured payload.

    Each restriction must carry an ``owl:onProperty`` and one restriction-
    type predicate (``owl:cardinality``, ``owl:someValuesFrom``, …). The
    result is a dict keyed by the blank-node term:

        {
            "kind": "restriction",
            "on_property": iri,
            "restriction_type": "minCardinality" | ... ,
            "value": int | iri | bool,
            "on_class": iri | None,   # for qualified cardinality
        }
    """
    out: Dict[BNode, Dict[str, Any]] = {}
    for restriction, _, _ in g.triples((None, RDF.type, OWL.Restriction)):
        if not isinstance(restriction, BNode):
            continue
        on_prop = next(g.objects(restriction, OWL.onProperty), None)
        on_class = next(g.objects(restriction, OWL.onClass), None)
        payload: Dict[str, Any] = {
            "kind": "restriction",
            "on_property": _term_str(on_prop),
            "on_class": _term_str(on_class),
        }
        # Cardinality-flavoured.
        for pred, label in _CARDINALITY_RESTRICTIONS.items():
            v = next(g.objects(restriction, pred), None)
            if v is not None:
                try:
                    payload["restriction_type"] = label
                    payload["value"] = int(str(v))
                except (TypeError, ValueError):
                    payload["restriction_type"] = label
                    payload["value"] = str(v)
                break
        else:
            # Value-flavoured.
            for pred, label in _VALUE_RESTRICTIONS.items():
                v = next(g.objects(restriction, pred), None)
                if v is not None:
                    payload["restriction_type"] = label
                    if label == "hasSelf":
                        payload["value"] = bool(v) if isinstance(v, Literal) else True
                    elif isinstance(v, Literal):
                        payload["value"] = str(v)
                    else:
                        payload["value"] = _term_str(v) or str(v)
                    break
            else:
                # Restriction with no recognised predicate — keep it but flag it.
                payload["restriction_type"] = "unknown"
        out[restriction] = payload
    return out


def _decode_class_combinators(
    g: rdflib.Graph,
    lists: Dict[BNode, List[Any]],
) -> Dict[BNode, Dict[str, Any]]:
    """Annotate anonymous classes built with unionOf/intersectionOf/oneOf/complementOf.

    Each combinator references either an ``rdf:List`` (union/intersection/oneOf)
    or a single class (complementOf).
    """
    out: Dict[BNode, Dict[str, Any]] = {}
    list_combinators = {
        OWL.unionOf: "unionOf",
        OWL.intersectionOf: "intersectionOf",
        OWL.oneOf: "oneOf",
    }
    for pred, label in list_combinators.items():
        for s, _, o in g.triples((None, pred, None)):
            if not isinstance(s, BNode):
                continue
            members = lists.get(o) if isinstance(o, BNode) else None
            member_iris = [_term_str(m) or str(m) for m in (members or [])]
            payload = out.setdefault(s, {"kind": "class_expression"})
            payload["combinator"] = label
            payload["members"] = member_iris
    for s, _, o in g.triples((None, OWL.complementOf, None)):
        if not isinstance(s, BNode):
            continue
        payload = out.setdefault(s, {"kind": "class_expression"})
        payload["combinator"] = "complementOf"
        payload["members"] = [_term_str(o) or str(o)]
    return out


def _collect_descriptions(g: rdflib.Graph) -> Dict[Any, Dict[str, str]]:
    """Map terms to a description payload using ``rdfs:comment`` / ``rdfs:isDefinedBy``."""
    out: Dict[Any, Dict[str, str]] = {}
    for s, _, o in g.triples((None, RDFS.comment, None)):
        if isinstance(o, Literal):
            out.setdefault(s, {})["description"] = str(o)
    for s, _, o in g.triples((None, RDFS.isDefinedBy, None)):
        out.setdefault(s, {})["defined_by"] = str(o)
    return out


def _emit_axioms(g: rdflib.Graph, lists: Dict[BNode, List[Any]]) -> List[Any]:
    """Decode OWL axioms that don't fit on a single node/edge."""
    axioms: List[Any] = []
    # equivalentClass: pairwise.
    for s, _, o in g.triples((None, OWL.equivalentClass, None)):
        a = _term_str(s)
        b = _term_str(o)
        if a and b:
            axioms.append(EquivalentClassesAxiom(class_ids=[a, b]))
    # disjointWith: pairwise.
    for s, _, o in g.triples((None, OWL.disjointWith, None)):
        a = _term_str(s)
        b = _term_str(o)
        if a and b:
            axioms.append(DisjointClassesAxiom(class_ids=[a, b]))
    # AllDisjointClasses (n-ary).
    for axiom_node, _, _ in g.triples((None, RDF.type, OWL.AllDisjointClasses)):
        members_head = next(g.objects(axiom_node, OWL.members), None)
        members = lists.get(members_head, []) if isinstance(members_head, BNode) else []
        ids = [_term_str(m) for m in members if _term_str(m)]
        if ids:
            axioms.append(DisjointClassesAxiom(class_ids=ids))
    # disjointUnionOf.
    for s, _, o in g.triples((None, OWL.disjointUnionOf, None)):
        union_iri = _term_str(s)
        parts = lists.get(o, []) if isinstance(o, BNode) else []
        part_ids = [_term_str(p) for p in parts if _term_str(p)]
        if union_iri and part_ids:
            axioms.append(DisjointUnionAxiom(union_class_id=union_iri, part_class_ids=part_ids))
    # subPropertyOf.
    for s, _, o in g.triples((None, RDFS.subPropertyOf, None)):
        a = _term_str(s)
        b = _term_str(o)
        if a and b:
            axioms.append(SubPropertyOfAxiom(sub_property_id=a, super_property_id=b))
    # inverseOf.
    for s, _, o in g.triples((None, OWL.inverseOf, None)):
        a = _term_str(s)
        b = _term_str(o)
        if a and b:
            axioms.append(InversePropertiesAxiom(property_a_id=a, property_b_id=b))
    # propertyChainAxiom.
    for s, _, o in g.triples((None, OWL.propertyChainAxiom, None)):
        chain = lists.get(o, []) if isinstance(o, BNode) else []
        chain_ids = [_term_str(p) for p in chain if _term_str(p)]
        prop_iri = _term_str(s)
        if prop_iri and chain_ids:
            axioms.append(PropertyChainAxiom(property_id=prop_iri, chain_property_ids=chain_ids))
    # hasKey.
    for s, _, o in g.triples((None, OWL.hasKey, None)):
        keys = lists.get(o, []) if isinstance(o, BNode) else []
        key_ids = [_term_str(k) for k in keys if _term_str(k)]
        cls_iri = _term_str(s)
        if cls_iri and key_ids:
            axioms.append(HasKeyAxiom(class_id=cls_iri, property_ids=key_ids))
    return axioms


def _emit_imports(g: rdflib.Graph) -> List[ImportAxiom]:
    """Emit one :class:`ImportAxiom` per ``owl:imports`` triple. Imports are
    logged but not followed."""
    out: List[ImportAxiom] = []
    for s, _, o in g.triples((None, OWL.imports, None)):
        target = _term_str(o)
        if not target:
            continue
        out.append(ImportAxiom(target_iri=target, source_ontology_iri=_term_str(s)))
    return out


# ----------------------------------------------------------------------
# SHACL Core constraint components.
# ----------------------------------------------------------------------

# Maps SHACL predicate → (constraint kind, value extractor).
# Each extractor receives (graph, object_term, lists) and returns the spec value.
def _shacl_literal_int(_g, o, _lists):
    try:
        return int(str(o))
    except (TypeError, ValueError):
        return str(o)


def _shacl_literal_value(_g, o, _lists):
    if isinstance(o, Literal):
        try:
            return float(str(o))
        except (TypeError, ValueError):
            return str(o)
    return _term_str(o) or str(o)


def _shacl_iri(_g, o, _lists):
    return _term_str(o) or str(o)


def _shacl_literal_str(_g, o, _lists):
    return str(o)


def _shacl_literal_bool(_g, o, _lists):
    if isinstance(o, Literal):
        return bool(o)
    return str(o).lower() in ("true", "1")


def _shacl_iri_list(_g, o, lists):
    if isinstance(o, BNode):
        members = lists.get(o, []) or []
        return [_term_str(m) or str(m) for m in members]
    return [_term_str(o) or str(o)]


def _shacl_literal_list(_g, o, lists):
    if isinstance(o, BNode):
        members = lists.get(o, []) or []
        out = []
        for m in members:
            if isinstance(m, Literal):
                out.append({"value": str(m), "datatype": str(m.datatype) if m.datatype else None,
                            "language": m.language})
            else:
                out.append(_term_str(m) or str(m))
        return out
    return [str(o)]


# Single-value SHACL property-shape constraints.
_SHACL_PROPERTY_COMPONENTS: Dict[Any, Tuple[str, Any]] = {
    SH.minCount: ("minCardinality", _shacl_literal_int),
    SH.maxCount: ("maxCardinality", _shacl_literal_int),
    SH.minInclusive: ("minInclusive", _shacl_literal_value),
    SH.maxInclusive: ("maxInclusive", _shacl_literal_value),
    SH.minExclusive: ("minExclusive", _shacl_literal_value),
    SH.maxExclusive: ("maxExclusive", _shacl_literal_value),
    SH.minLength: ("minLength", _shacl_literal_int),
    SH.maxLength: ("maxLength", _shacl_literal_int),
    SH.pattern: ("pattern", _shacl_literal_str),
    SH.flags: ("flags", _shacl_literal_str),
    SH.datatype: ("datatype", _shacl_iri),
    SH.nodeKind: ("nodeKind", _shacl_iri),
    SH.hasValue: ("hasValue", _shacl_literal_value),
    SH.uniqueLang: ("uniqueLang", _shacl_literal_bool),
    SH.languageIn: ("languageIn", _shacl_literal_list),
    SH["class"]: ("someValuesFrom", _shacl_iri),  # sh:class maps onto someValuesFrom-of-class
    SH["in"]: ("in", _shacl_literal_list),
    SH.severity: ("shaclSeverity", _shacl_iri),
    SH.message: ("shaclMessage", _shacl_literal_str),
    SH.name: ("shaclName", _shacl_literal_str),
    SH.description: ("shaclDescription", _shacl_literal_str),
    SH.deactivated: ("shaclDeactivated", _shacl_literal_bool),
    SH.order: ("shaclOrder", _shacl_literal_value),
    SH.group: ("shaclGroup", _shacl_iri),
}

# Node-shape components (closure plus shape-level shaclXXX meta).
_SHACL_NODE_COMPONENTS: Dict[Any, Tuple[str, Any]] = {
    SH.closed: ("shaclClosed", _shacl_literal_bool),
    SH.ignoredProperties: ("shaclIgnoredProperties", _shacl_iri_list),
    SH.disjoint: ("shaclDisjoint", _shacl_iri),
    SH.severity: ("shaclSeverity", _shacl_iri),
    SH.message: ("shaclMessage", _shacl_literal_str),
    SH.name: ("shaclName", _shacl_literal_str),
    SH.description: ("shaclDescription", _shacl_literal_str),
    SH.deactivated: ("shaclDeactivated", _shacl_literal_bool),
}


def _restriction_payload_to_specs(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translate a decoded OWL ``owl:Restriction`` payload into ``constraintSpecs``.

    The decoder produces one ``{restriction_type, value, on_class}`` triple per
    restriction. This helper maps that into the normalised spec shape used by
    :class:`KGPropertyConstraint`.
    """
    rt = payload.get("restriction_type")
    val = payload.get("value")
    on_class = payload.get("on_class")
    kind_map = {
        "minCardinality": "minCardinality",
        "maxCardinality": "maxCardinality",
        "cardinality": "exactCardinality",
        "minQualifiedCardinality": "minQualifiedCardinality",
        "maxQualifiedCardinality": "maxQualifiedCardinality",
        "qualifiedCardinality": "exactQualifiedCardinality",
        "someValuesFrom": "someValuesFrom",
        "allValuesFrom": "allValuesFrom",
        "hasValue": "hasValue",
        "hasSelf": "hasSelf",
    }
    kind = kind_map.get(rt)
    if not kind:
        return []
    spec: Dict[str, Any] = {"kind": kind, "value": val}
    if on_class:
        spec["on_class"] = on_class
    return [spec]


def _detect_punning(g: rdflib.Graph, classes: Set[URIRef]) -> Set[URIRef]:
    """Detect IRIs that are explicitly *both* a class and an individual.

    Returns the set of URIRefs that are declared via both
    ``rdf:type owl:Class`` (already in ``classes``) and ``rdf:type
    owl:NamedIndividual``. These IRIs get **two** nodes — a KGClass and a
    KGIndividual — linked via ``metadata['punned_with']``.
    """
    named_individuals: Set[URIRef] = set()
    for s, _, _ in g.triples((None, RDF.type, OWL.NamedIndividual)):
        if isinstance(s, URIRef):
            named_individuals.add(s)
    return classes & named_individuals


# ----------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------


def _decode_shacl_shapes(
    g: rdflib.Graph,
    lists: Dict[BNode, List[Any]],
) -> Dict[Any, Dict[str, Any]]:
    """Decode SHACL shapes (NodeShape / PropertyShape) into structured payloads.

    Returns a dict keyed by the shape's RDF term (URIRef or BNode) with shape
    fields:

        {
            "shape_kind": "nodeShape" | "propertyShape",
            "target_class": iri | None,
            "target_node": [iri, ...],
            "target_subjects_of": [iri, ...],
            "target_objects_of": [iri, ...],
            "path": iri | None,           # property shapes only
            "property_shape_terms": [term, ...],  # node shapes only
            "constraint_specs": [{...}, ...],
            "logical_terms": {"shaclAnd": [term, ...], ...},
            "qualified": [{count_kind, count, value_shape_term}, ...],
        }

    Targets / paths that aren't simple IRIs (e.g. sequence paths) are recorded
    verbatim as strings so the editor can preserve them.
    """
    out: Dict[Any, Dict[str, Any]] = {}

    # Identify shape terms.
    node_shapes: Set[Any] = set()
    prop_shapes: Set[Any] = set()
    for s, _, _ in g.triples((None, RDF.type, SH.NodeShape)):
        node_shapes.add(s)
    for s, _, _ in g.triples((None, RDF.type, SH.PropertyShape)):
        prop_shapes.add(s)
    # SHACL infers PropertyShape via sh:path / sh:property targets.
    for s, _, _ in g.triples((None, SH.path, None)):
        prop_shapes.add(s)
    for _, _, o in g.triples((None, SH.property, None)):
        prop_shapes.add(o)
    # SHACL infers NodeShape via sh:targetClass / sh:property subjects.
    for s, _, _ in g.triples((None, SH.targetClass, None)):
        node_shapes.add(s)
    for s, _, _ in g.triples((None, SH.property, None)):
        if s not in prop_shapes:
            node_shapes.add(s)
    # Shapes referenced only via the four logical operators (sh:not, sh:and,
    # sh:or, sh:xone) — these may be anonymous inline shapes that carry no
    # other shape-defining predicate. We still need to decode their
    # constraint components so the lifter can inline them as nested shapes.
    logical_obj_preds = [SH["not"], SH["and"], SH["or"], SH.xone]
    for pred in logical_obj_preds:
        for _, _, obj in g.triples((None, pred, None)):
            if pred == SH["not"]:
                terms = [obj]
            else:
                terms = lists.get(obj, []) if isinstance(obj, BNode) else [obj]
            for term in terms:
                if term in node_shapes or term in prop_shapes:
                    continue
                # Default to property-shape semantics for nested inline shapes
                # (property-style constraints — datatype, pattern, cardinality
                # — are by far the most common nested-shape contents). The
                # constraint-component decoder treats both shape kinds
                # uniformly anyway via _SHACL_PROPERTY_COMPONENTS.
                prop_shapes.add(term)

    def _init(term):
        return out.setdefault(
            term,
            {
                "target_class": None,
                "target_node": [],
                "target_subjects_of": [],
                "target_objects_of": [],
                "path": None,
                "property_shape_terms": [],
                "constraint_specs": [],
                "logical_terms": {},
                "qualified": [],
            },
        )

    for term in node_shapes:
        payload = _init(term)
        payload["shape_kind"] = "nodeShape"

    for term in prop_shapes:
        payload = _init(term)
        payload["shape_kind"] = "propertyShape"

    # Targets.
    for s, _, o in g.triples((None, SH.targetClass, None)):
        if s in out:
            out[s]["target_class"] = _term_str(o) or str(o)
    for s, _, o in g.triples((None, SH.targetNode, None)):
        if s in out:
            out[s]["target_node"].append(_term_str(o) or str(o))
    for s, _, o in g.triples((None, SH.targetSubjectsOf, None)):
        if s in out:
            out[s]["target_subjects_of"].append(_term_str(o) or str(o))
    for s, _, o in g.triples((None, SH.targetObjectsOf, None)):
        if s in out:
            out[s]["target_objects_of"].append(_term_str(o) or str(o))

    # sh:path on property shapes (only IRI paths handled deterministically;
    # other path shapes get stringified).
    for s, _, o in g.triples((None, SH.path, None)):
        if s in out:
            if isinstance(o, URIRef):
                out[s]["path"] = str(o)
            else:
                out[s]["path"] = str(o)

    # sh:property links NodeShape → PropertyShape.
    for s, _, o in g.triples((None, SH.property, None)):
        if s in out:
            out[s]["property_shape_terms"].append(o)

    # Simple constraint components on property shapes.
    for pred, (kind, extractor) in _SHACL_PROPERTY_COMPONENTS.items():
        for s, _, o in g.triples((None, pred, None)):
            if s not in out:
                continue
            spec_value = extractor(g, o, lists)
            spec: Dict[str, Any] = {"kind": kind, "value": spec_value}
            out[s]["constraint_specs"].append(spec)

    # Node-shape components (these may overlap with property-shape preds for
    # the SHACL meta vocabulary, but the predicate set above keeps both shapes
    # accumulating in their own payloads).
    for pred, (kind, extractor) in _SHACL_NODE_COMPONENTS.items():
        for s, _, o in g.triples((None, pred, None)):
            if s not in out or out[s].get("shape_kind") != "nodeShape":
                continue
            spec_value = extractor(g, o, lists)
            # Avoid duplicate emission if the same predicate was also added via
            # the property-shape table (only happens for meta predicates).
            already = any(
                existing.get("kind") == kind and existing.get("value") == spec_value
                for existing in out[s]["constraint_specs"]
            )
            if not already:
                out[s]["constraint_specs"].append({"kind": kind, "value": spec_value})

    # sh:not / sh:and / sh:or / sh:xone: record the linked shape terms for the
    # post-processor to wire as nested-spec references.
    logical_preds = {
        SH["not"]: "shaclNot",
        SH["and"]: "shaclAnd",
        SH["or"]: "shaclOr",
        SH.xone: "shaclXone",
    }
    for pred, kind in logical_preds.items():
        for s, _, o in g.triples((None, pred, None)):
            if s not in out:
                continue
            if pred == SH["not"]:
                terms = [o]
            else:
                terms = lists.get(o, []) if isinstance(o, BNode) else [o]
            out[s]["logical_terms"].setdefault(kind, []).extend(terms)

    # Qualified value shapes: sh:qualifiedValueShape + sh:qualifiedMinCount /
    # sh:qualifiedMaxCount.
    for s, _, o in g.triples((None, SH.qualifiedValueShape, None)):
        if s not in out:
            continue
        q_min = next(g.objects(s, SH.qualifiedMinCount), None)
        q_max = next(g.objects(s, SH.qualifiedMaxCount), None)
        entry = {"value_shape_term": o}
        if q_min is not None:
            try:
                entry["min_count"] = int(str(q_min))
            except (TypeError, ValueError):
                pass
        if q_max is not None:
            try:
                entry["max_count"] = int(str(q_max))
            except (TypeError, ValueError):
                pass
        out[s]["qualified"].append(entry)

    return out


def _lift_to_constraint_nodes(
    kg: KnowledgeGraph,
    g: rdflib.Graph,
    restriction_terms: Dict[BNode, str],
    shacl_payloads: Dict[Any, Dict[str, Any]],
    shacl_term_to_id: Dict[Any, str],
) -> None:
    """Mutate the KG in place so OWL restrictions and SHACL shapes appear as
    first-class :class:`KGNodeConstraint` / :class:`KGPropertyConstraint` nodes.

    Steps:

    1. Each OWL ``owl:Restriction`` (currently emitted as ``KGBlank``) is
       replaced by a ``KGPropertyConstraint`` carrying the equivalent
       ``constraintSpecs`` payload. ``constraintTargetProperty`` edges are
       added to the property the restriction is ``owl:onProperty``.
    2. ``rdfs:subClassOf`` edges from a class to such a restriction are removed
       and replaced by a wrapping ``KGNodeConstraint`` linked via
       ``constraintTargetClass`` to the class and via ``sh:property`` to each
       associated PropertyConstraint.
    3. Class-level axioms (``equivalentClass``, ``disjointWith``, ``oneOf``,
       ``hasKey``, ``disjointUnionOf``) materialise as anonymous
       ``KGNodeConstraint`` nodes (in addition to remaining on
       ``kg.axioms`` for the existing class-diagram conversion).
    4. SHACL shapes from ``shacl_payloads`` materialise their full
       ``constraintSpecs`` and target / ``sh:property`` edges.
    """
    nodes_by_id: Dict[str, KGNode] = {n.id: n for n in kg.nodes}
    edges_by_id: Dict[str, KGEdge] = {e.id: e for e in kg.edges}

    def _next_edge_id() -> str:
        i = len(edges_by_id) + 1
        while f"cedge:{i}" in edges_by_id:
            i += 1
        return f"cedge:{i}"

    def _next_node_id(prefix: str) -> str:
        i = 1
        while f"{prefix}{i}" in nodes_by_id:
            i += 1
        return f"{prefix}{i}"

    def _add_edge(src: KGNode, tgt: KGNode, label: str, iri: str) -> KGEdge:
        eid = _next_edge_id()
        edge = KGEdge(id=eid, source=src, target=tgt, label=label, iri=iri)
        kg.add_edge(edge)
        edges_by_id[eid] = edge
        return edge

    def _replace_node(old: KGNode, new: KGNode) -> None:
        """Swap a node in the KG, preserving all edges pointing at the old node."""
        # Remove old, add new.
        kg.nodes.remove(old)
        kg.nodes.add(new)
        nodes_by_id.pop(old.id, None)
        nodes_by_id[new.id] = new
        # Re-wire edges.
        for e in list(kg.edges):
            if e.source is old:
                e.source = new
            if e.target is old:
                e.target = new

    # 1. Replace OWL restriction KGBlank nodes with KGPropertyConstraint.
    blank_to_pc: Dict[str, KGPropertyConstraint] = {}
    for bnode_term, blank_id in restriction_terms.items():
        old = nodes_by_id.get(blank_id)
        if old is None or not isinstance(old, KGBlank):
            continue
        payload = dict(old.metadata)
        specs = _restriction_payload_to_specs(payload)
        if not specs:
            continue  # leave unknown restrictions as blank nodes
        on_property_iri = payload.get("on_property")
        pc = KGPropertyConstraint(
            id=old.id,
            label=old.label or "PropertyConstraint",
            iri=None,
            metadata={
                "constraintSpecs": specs,
                "isAnonymous": True,
                "source": "owl",
                "onPropertyIri": on_property_iri,
            },
        )
        _replace_node(old, pc)
        blank_to_pc[old.id] = pc
        # Add constraintTargetProperty edge to the property node (if known).
        if on_property_iri and on_property_iri in nodes_by_id:
            prop_node = nodes_by_id[on_property_iri]
            if isinstance(prop_node, KGProperty):
                _add_edge(pc, prop_node, "constraintTargetProperty", CONSTRAINT_TARGET_PROPERTY)

    # 2. For each class that's rdfs:subClassOf a restriction-now-PC, replace
    #    that edge with a wrapping NodeConstraint.
    if blank_to_pc:
        class_to_pcs: Dict[str, List[KGPropertyConstraint]] = {}
        edges_to_remove: List[KGEdge] = []
        for e in list(kg.edges):
            if e.iri == str(RDFS.subClassOf):
                tgt_id = e.target.id
                if tgt_id in blank_to_pc and isinstance(e.source, KGClass):
                    class_to_pcs.setdefault(e.source.id, []).append(blank_to_pc[tgt_id])
                    edges_to_remove.append(e)
        for e in edges_to_remove:
            kg.edges.discard(e)
            edges_by_id.pop(e.id, None)
        for class_id, pcs in class_to_pcs.items():
            class_node = nodes_by_id[class_id]
            wrapper_id = _next_node_id(f"nc:{class_id}#")
            nc = KGNodeConstraint(
                id=wrapper_id,
                label=f"{class_node.label or 'Class'} restrictions",
                metadata={"constraintSpecs": [], "isAnonymous": True, "source": "owl"},
            )
            kg.add_node(nc)
            nodes_by_id[nc.id] = nc
            _add_edge(nc, class_node, "constraintTargetClass", CONSTRAINT_TARGET_CLASS)
            for pc in pcs:
                _add_edge(nc, pc, "property", SH_PROPERTY)

    # 3. Materialise class-level axioms as NodeConstraints (in addition to the
    #    typed kg.axioms records).
    def _axiom_nc(label: str, target_class_id: str, specs: List[Dict[str, Any]]) -> None:
        target_node = nodes_by_id.get(target_class_id)
        if not isinstance(target_node, KGClass):
            return
        nid = _next_node_id(f"nc:axiom:{target_class_id}#")
        nc = KGNodeConstraint(
            id=nid,
            label=label,
            metadata={"constraintSpecs": specs, "isAnonymous": True, "source": "owl-axiom"},
        )
        kg.add_node(nc)
        nodes_by_id[nc.id] = nc
        _add_edge(nc, target_node, "constraintTargetClass", CONSTRAINT_TARGET_CLASS)

    for axiom in list(kg.axioms):
        if isinstance(axiom, EquivalentClassesAxiom) and axiom.class_ids:
            anchor = axiom.class_ids[0]
            others = axiom.class_ids[1:]
            if others:
                _axiom_nc("Equivalent", anchor, [{"kind": "equivalentClasses", "value": others}])
        elif isinstance(axiom, DisjointClassesAxiom) and axiom.class_ids:
            anchor = axiom.class_ids[0]
            others = axiom.class_ids[1:]
            if others:
                _axiom_nc("Disjoint", anchor, [{"kind": "disjointWith", "value": others}])
        elif isinstance(axiom, DisjointUnionAxiom) and axiom.union_class_id:
            _axiom_nc(
                "Disjoint union",
                axiom.union_class_id,
                [{"kind": "disjointUnionOf", "value": list(axiom.part_class_ids)}],
            )
        elif isinstance(axiom, HasKeyAxiom) and axiom.class_id:
            _axiom_nc(
                "Has key",
                axiom.class_id,
                [{"kind": "hasKey", "value": list(axiom.property_ids)}],
            )

    # 4. SHACL shapes.
    if shacl_payloads:
        # Decide which shapes are "top-level" (worth materialising as a
        # KGNodeConstraint / KGPropertyConstraint node) vs "nested-only"
        # (only referenced from a logical operator on another shape — we
        # inline these into the parent's spec value as ``{"specs": [...]}``).
        def _is_top_level(term: Any, payload: Dict[str, Any]) -> bool:
            if payload["shape_kind"] == "nodeShape":
                if any(payload.get(k) for k in (
                    "target_class", "target_node", "target_subjects_of", "target_objects_of"
                )):
                    return True
            else:  # propertyShape
                if payload.get("path"):
                    return True
            # Named (URIRef) shapes are always materialised so the user can
            # reuse them by name across the diagram.
            if isinstance(term, URIRef):
                return True
            # Referenced from a top-level node shape via sh:property → still
            # materialised so the editor can show the sh:property edge.
            for other_term, other in shacl_payloads.items():
                if other.get("shape_kind") != "nodeShape":
                    continue
                if term in other.get("property_shape_terms", []):
                    return True
            return False

        top_level_terms: Set[Any] = {
            t for t, p in shacl_payloads.items() if _is_top_level(t, p)
        }

        shape_term_to_node: Dict[Any, KGNode] = {}
        for term, payload in shacl_payloads.items():
            if term not in top_level_terms:
                # Nested-only: kept in shacl_payloads for the logical-spec
                # builder below, but no node is created.
                continue
            existing_id = shacl_term_to_id.get(term)
            existing = nodes_by_id.get(existing_id) if existing_id else None
            label = ""
            for spec in payload["constraint_specs"]:
                if spec["kind"] == "shaclName":
                    label = str(spec.get("value") or "")
                    break
            if not label:
                if isinstance(term, URIRef):
                    label = _local_name(str(term))
                elif payload.get("path"):
                    label = _local_name(payload["path"]) + " shape"
                elif payload.get("target_class"):
                    label = _local_name(payload["target_class"]) + " shape"
                else:
                    label = "Shape"
            iri = str(term) if isinstance(term, URIRef) else None
            specs = list(payload["constraint_specs"])
            if payload["shape_kind"] == "nodeShape":
                nc = KGNodeConstraint(
                    id=existing.id if existing else (iri or _next_node_id("nc:shacl:")),
                    label=label,
                    iri=iri,
                    metadata={"constraintSpecs": specs, "source": "shacl", "isAnonymous": iri is None},
                )
                if existing:
                    _replace_node(existing, nc)
                else:
                    kg.add_node(nc)
                    nodes_by_id[nc.id] = nc
                shape_term_to_node[term] = nc
                # Wire targetClass edge.
                target_class_iri = payload.get("target_class")
                if target_class_iri and target_class_iri in nodes_by_id:
                    tgt = nodes_by_id[target_class_iri]
                    if isinstance(tgt, KGClass):
                        _add_edge(nc, tgt, "constraintTargetClass", CONSTRAINT_TARGET_CLASS)
            else:
                pc = KGPropertyConstraint(
                    id=existing.id if existing else (iri or _next_node_id("pc:shacl:")),
                    label=label,
                    iri=iri,
                    metadata={"constraintSpecs": specs, "source": "shacl", "isAnonymous": iri is None},
                )
                if existing:
                    _replace_node(existing, pc)
                else:
                    kg.add_node(pc)
                    nodes_by_id[pc.id] = pc
                shape_term_to_node[term] = pc
                # Wire constraintTargetProperty edge.
                path_iri = payload.get("path")
                if path_iri and path_iri in nodes_by_id:
                    prop = nodes_by_id[path_iri]
                    if isinstance(prop, KGProperty):
                        _add_edge(pc, prop, "constraintTargetProperty", CONSTRAINT_TARGET_PROPERTY)

        # Wire sh:property edges from NodeShape → PropertyShape.
        for term, payload in shacl_payloads.items():
            src = shape_term_to_node.get(term)
            if not isinstance(src, KGNodeConstraint):
                continue
            for ps_term in payload.get("property_shape_terms", []):
                tgt = shape_term_to_node.get(ps_term)
                if isinstance(tgt, KGPropertyConstraint):
                    _add_edge(src, tgt, "property", SH_PROPERTY)

        # Append shaclNot / shaclAnd / shaclOr / shaclXone specs to each
        # materialised shape. Each entry resolves to either a {"ref": node-id}
        # (for top-level referenced shapes) or {"specs": [...]} (for inline /
        # nested-only shapes), recursing through nested logical operators so
        # arbitrarily-deep SHACL combinator trees round-trip.
        def _payload_to_full_specs(payload: Dict[str, Any], visited: Set[Any]) -> List[Dict[str, Any]]:
            out_specs: List[Dict[str, Any]] = list(payload.get("constraint_specs", []))
            for kind, terms in (payload.get("logical_terms") or {}).items():
                nested = [_resolve_nested(t, visited) for t in terms]
                # shaclNot must have exactly one entry; the importer never
                # produces more, but guard against malformed input.
                if kind == "shaclNot":
                    nested = nested[:1] if nested else [{"specs": []}]
                out_specs.append({"kind": kind, "value": nested})
            return out_specs

        def _resolve_nested(term: Any, visited: Set[Any]) -> Dict[str, Any]:
            if term in visited:
                # Cyclic reference: bail out with an empty inline shape; the
                # rdflib graph can express cycles via blank-node loops, but
                # the editor's spec tree can't.
                return {"specs": []}
            tnode = shape_term_to_node.get(term)
            if tnode is not None:
                return {"ref": tnode.id}
            payload = shacl_payloads.get(term)
            if payload is None:
                return {"specs": []}
            return {"specs": _payload_to_full_specs(payload, visited | {term})}

        for term in top_level_terms:
            payload = shacl_payloads[term]
            tnode = shape_term_to_node.get(term)
            if tnode is None:
                continue
            logical_specs: List[Dict[str, Any]] = []
            for kind, terms in (payload.get("logical_terms") or {}).items():
                nested = [_resolve_nested(t, {term}) for t in terms]
                if kind == "shaclNot":
                    nested = nested[:1] if nested else [{"specs": []}]
                logical_specs.append({"kind": kind, "value": nested})
            if not logical_specs:
                continue
            meta = dict(tnode.metadata)
            specs = list(meta.get("constraintSpecs", []))
            specs.extend(logical_specs)
            meta["constraintSpecs"] = specs
            tnode.metadata = meta

        # Cleanup: nested-only SHACL shapes were materialised as KGBlank
        # nodes during the first emission pass (every term becomes a node),
        # and their raw shape predicates (sh:datatype, sh:minLength, …) plus
        # the rdf:List spine that joined them (rdf:first / rdf:rest blanks)
        # live on as KGEdges. Now that the contents are folded into a parent
        # constraint's nested-spec value, the raw blanks + their incident
        # edges are redundant. Drop them so the exporter doesn't leak the
        # same constraints twice (once via the nested spec, once as raw
        # blank-node triples).
        nested_blank_ids: Set[str] = set()
        for term in shacl_payloads:
            if term in top_level_terms:
                continue
            if isinstance(term, BNode):
                nid = f"_:{str(term)}"
                if nid in nodes_by_id and isinstance(nodes_by_id[nid], KGBlank):
                    nested_blank_ids.add(nid)
        # Walk every list head that fed a logical operator, collecting the
        # list spine blank nodes (the chained rdf:rest cursors). They have
        # served their decoding purpose and are about to be re-emitted via
        # _emit_rdf_list on the export side.
        rdf_first = str(RDF.first)
        rdf_rest = str(RDF.rest)
        list_spine_ids: Set[str] = set()
        for payload in shacl_payloads.values():
            for kind, terms in (payload.get("logical_terms") or {}).items():
                if kind == "shaclNot":
                    continue  # sh:not takes a single shape, no list
                # `terms` is the flat list of members; we need the list-head
                # itself, which we recover by scanning the original Turtle
                # rdflib graph for the head whose first item is terms[0].
                for t in terms:
                    nid_t = f"_:{str(t)}" if isinstance(t, BNode) else None
                    if nid_t:
                        list_spine_ids.add(nid_t)  # member blanks already in nested_blank_ids
        # Sweep any blank node whose only incident edges are rdf:first /
        # rdf:rest to also-doomed nodes. That's the spine.
        for blank_node in [n for n in kg.nodes if isinstance(n, KGBlank)]:
            edges_in = [e for e in kg.edges if e.source.id == blank_node.id or e.target.id == blank_node.id]
            if not edges_in:
                continue
            if all(e.iri in (rdf_first, rdf_rest) for e in edges_in):
                # If at least one neighbour is a doomed nested blank, this
                # spine node is doomed too.
                touching = {e.source.id for e in edges_in} | {e.target.id for e in edges_in}
                touching.discard(blank_node.id)
                if touching & nested_blank_ids:
                    list_spine_ids.add(blank_node.id)
        # Repeat the spine sweep until no further nodes are added, so chains
        # of rdf:rest blanks get pruned end-to-end.
        prev = -1
        doomed = nested_blank_ids | list_spine_ids
        while len(doomed) != prev:
            prev = len(doomed)
            for blank_node in [n for n in kg.nodes if isinstance(n, KGBlank) and n.id not in doomed]:
                edges_in = [e for e in kg.edges if e.source.id == blank_node.id or e.target.id == blank_node.id]
                if not edges_in:
                    continue
                if all(e.iri in (rdf_first, rdf_rest) for e in edges_in):
                    touching = {e.source.id for e in edges_in} | {e.target.id for e in edges_in}
                    touching.discard(blank_node.id)
                    if touching & doomed:
                        doomed.add(blank_node.id)
        if doomed:
            surviving_edges = {
                e for e in kg.edges
                if e.source.id not in doomed and e.target.id not in doomed
            }
            kg.edges = surviving_edges
            for nid in doomed:
                node = nodes_by_id.pop(nid, None)
                if node is not None:
                    kg.nodes.discard(node)
            edges_by_id.clear()
            edges_by_id.update({e.id: e for e in kg.edges})


def owl_file_to_knowledge_graph(path: str) -> KnowledgeGraph:
    """Parse an OWL/RDF file and return a populated ``KnowledgeGraph``.

    Every triple becomes a ``KGEdge`` (including ``rdf:type`` triples) so the
    import is transparent — the UI can later choose to hide type edges.

    Args:
        path: Filesystem path to the ontology file. Supported serializations:
            ``.owl`` (RDF/XML), ``.rdf``, ``.xml``, ``.ttl``, ``.nt``, ``.n3``.
            The format is auto-detected via ``rdflib.util.guess_format``;
            ``.owl`` defaults to ``xml`` when guess fails.
    """
    g = rdflib.Graph()
    fmt = rdflib.util.guess_format(path)
    if fmt is None:
        ext = os.path.splitext(path)[1].lower()
        fmt = "xml" if ext in (".owl", ".rdf") else None
    g.parse(path, format=fmt) if fmt else g.parse(path)

    # First pass: collect explicit class and property IRIs.
    classes: set = set()
    properties: set = set()
    type_objects: set = set()
    for s, p, o in g.triples((None, RDF.type, None)):
        type_objects.add(o)
        if o in _CLASS_TYPES:
            classes.add(s)
        elif o in _PROPERTY_TYPES:
            properties.add(s)
    # Property characteristic types (FunctionalProperty etc.) imply the
    # subject is also a property.
    for char_type in _PROPERTY_CHARACTERISTIC_TYPES:
        for s, _, _ in g.triples((None, RDF.type, char_type)):
            if isinstance(s, URIRef):
                properties.add(s)
    # Terms appearing as the object of rdf:type (and not already a property) are
    # treated as classes — they're being used as types.
    for o in type_objects:
        if isinstance(o, URIRef) and o not in properties and o not in _CLASS_TYPES and o not in _PROPERTY_TYPES:
            classes.add(o)
    # Datatype IRIs used as rdfs:range targets must classify as "class" too.
    for _, _, o in g.triples((None, RDFS.range, None)):
        if isinstance(o, URIRef) and _is_datatype_iri(str(o)):
            classes.add(o)

    # Decoder pass: pre-compute structured annotations and axioms.
    property_kinds = _collect_property_kinds(g)
    list_resolutions = _decode_lists(g)
    restriction_payloads = _decode_restrictions(g, list_resolutions)
    combinator_payloads = _decode_class_combinators(g, list_resolutions)
    descriptions = _collect_descriptions(g)
    punned = _detect_punning(g, classes)
    shacl_payloads = _decode_shacl_shapes(g, list_resolutions)

    # Second pass: build nodes.
    nodes_by_key: Dict[str, KGNode] = {}

    def _attach_metadata(term, node: KGNode) -> None:
        # Common: rdfs:comment / isDefinedBy → metadata.
        desc = descriptions.get(term)
        if desc:
            node.metadata.update(desc)
        # KGProperty: characteristics + kind.
        if isinstance(node, KGProperty):
            entry = property_kinds.get(term) if isinstance(term, URIRef) else None
            if entry:
                if entry.get("characteristics"):
                    node.metadata["characteristics"] = list(entry["characteristics"])
                if entry.get("kind"):
                    node.metadata["kind"] = entry["kind"]
        # KGBlank: restriction or class-expression payload.
        if isinstance(node, KGBlank) and isinstance(term, BNode):
            if term in restriction_payloads:
                node.metadata.update(restriction_payloads[term])
            elif term in combinator_payloads:
                node.metadata.update(combinator_payloads[term])

    def _get_or_create_node(term) -> KGNode:
        kind = _classify(term, classes, properties)
        if kind == "literal":
            nid = _literal_id(term)
            if nid in nodes_by_key:
                return nodes_by_key[nid]
            datatype = str(term.datatype) if term.datatype is not None else None
            node: KGNode = KGLiteral(id=nid, value=str(term), datatype=datatype)
        elif kind == "blank":
            nid = f"_:{str(term)}"
            if nid in nodes_by_key:
                return nodes_by_key[nid]
            node = KGBlank(id=nid, label=str(term))
        else:
            iri = str(term)
            nid = iri
            if nid in nodes_by_key:
                return nodes_by_key[nid]
            label = _label_for(g, term) or _local_name(iri)
            if kind == "class":
                node = KGClass(id=nid, label=label, iri=iri)
            elif kind == "property":
                node = KGProperty(id=nid, label=label, iri=iri)
            else:
                node = KGIndividual(id=nid, label=label, iri=iri)
        _attach_metadata(term, node)
        nodes_by_key[nid] = node
        return node

    # Build the punning twins eagerly so they're available regardless of
    # whether the IRI also appears as the object of a triple later.
    punned_twins: Dict[str, KGIndividual] = {}
    for iri_term in punned:
        # Make sure the class node exists first.
        cls_node = _get_or_create_node(iri_term)
        twin_id = f"{str(iri_term)}#individual"
        twin = KGIndividual(
            id=twin_id,
            label=cls_node.label,
            iri=str(iri_term),
            metadata={"punned_with": cls_node.id},
        )
        nodes_by_key[twin_id] = twin
        punned_twins[str(iri_term)] = twin
        # Cross-link from the class side too.
        cls_node.metadata["punned_with"] = twin_id

    edges = []
    edge_counter = 0
    for s, p, o in g:
        # For punned subjects, route NamedIndividual-typed triples to the
        # individual twin; everything else stays on the class node so the
        # TBox round-trip remains intact.
        src_term = s
        tgt_term = o
        if isinstance(s, URIRef) and str(s) in punned_twins and p == RDF.type and o == OWL.NamedIndividual:
            src = punned_twins[str(s)]
        else:
            src = _get_or_create_node(src_term)
        tgt = _get_or_create_node(tgt_term)
        edge_counter += 1
        predicate_iri = str(p)
        edges.append(
            KGEdge(
                id=f"edge:{edge_counter}",
                source=src,
                target=tgt,
                label=_local_name(predicate_iri),
                iri=predicate_iri,
            )
        )

    # Axioms.
    axioms = _emit_axioms(g, list_resolutions)
    axioms.extend(_emit_imports(g))

    kg = KnowledgeGraph(name="knowledge_graph", nodes=set(nodes_by_key.values()), axioms=axioms)
    for e in edges:
        kg.add_edge(e)

    # Track which terms correspond to which node ids so the lifter can swap
    # them in place: OWL restriction blank nodes already in nodes_by_key as
    # ``_:<bnode>`` ids, SHACL shapes via their term -> node-id mapping.
    restriction_terms: Dict[BNode, str] = {}
    for bnode_term in restriction_payloads:
        nid = f"_:{str(bnode_term)}"
        if nid in nodes_by_key:
            restriction_terms[bnode_term] = nid

    shacl_term_to_id: Dict[Any, str] = {}
    for shape_term in shacl_payloads:
        if isinstance(shape_term, URIRef):
            shacl_term_to_id[shape_term] = str(shape_term)
        elif isinstance(shape_term, BNode):
            nid = f"_:{str(shape_term)}"
            if nid in nodes_by_key:
                shacl_term_to_id[shape_term] = nid

    _lift_to_constraint_nodes(kg, g, restriction_terms, shacl_payloads, shacl_term_to_id)
    return kg


def _label_for(g: rdflib.Graph, term) -> Optional[str]:
    """Return the first ``rdfs:label`` for a term, if any."""
    for _, _, lbl in g.triples((term, RDFS.label, None)):
        if isinstance(lbl, Literal):
            return str(lbl)
    return None
