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
from rdflib import RDF, RDFS, OWL, BNode, Literal, URIRef

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
    KGProperty,
    KnowledgeGraph,
    PropertyChainAxiom,
    SubPropertyOfAxiom,
)


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
    return kg


def _label_for(g: rdflib.Graph, term) -> Optional[str]:
    """Return the first ``rdfs:label`` for a term, if any."""
    for _, _, lbl in g.triples((term, RDFS.label, None)):
        if isinstance(lbl, Literal):
            return str(lbl)
    return None
