"""OWL2 + SHACL consistency checking for a :class:`KnowledgeGraph`.

Delegates the actual validation to :mod:`pyshacl` (which itself wraps
:mod:`owlrl` for OWL2-RL inference). The KG is first serialised to an
``rdflib.Graph`` via the existing :func:`knowledge_graph_to_rdf`
exporter (``vocab="both"`` so both OWL and SHACL triples are present),
then augmented with a small set of synthetic SHACL shapes for OWL DL
constructs that ``pyshacl + owlrl`` doesn't translate into validation
violations on its own (notably ``owl:disjointWith``,
``owl:complementOf``, ``owl:FunctionalProperty``).

The validation report is parsed back into our
:class:`ConsistencyIssue` shape, with focus-node IRIs mapped to KG
node ids. The shim shapes never escape this module — they live only
in the in-memory graph passed to pyshacl.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

import pyshacl
import rdflib
from rdflib import BNode, Literal, RDF, URIRef, XSD
from rdflib.namespace import OWL, RDFS

from besser.BUML.metamodel.kg import (
    KnowledgeGraph,
    KGBlank,
    KGLiteral,
    KGNodeConstraint,
    KGPropertyConstraint,
)
from besser.BUML.notations.kg_to_buml.preflight import kg_signature
from besser.utilities.kg_to_owl import DEFAULT_NAMESPACE, _slugify, knowledge_graph_to_rdf


__all__ = [
    "ConsistencyIssue",
    "ConsistencyReport",
    "check_kg_consistency",
]


SH = rdflib.Namespace("http://www.w3.org/ns/shacl#")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConsistencyIssue:
    """A single inconsistency surfaced by the OWL/SHACL check."""

    id: str
    code: str
    severity: str           # "info" | "warning" | "violation"
    message: str
    affected_node_ids: List[str]
    affected_edge_ids: List[str] = field(default_factory=list)
    constraint_node_id: Optional[str] = None
    spec_kind: Optional[str] = None
    path_iri: Optional[str] = None
    #: One-line human-readable summary of the violated constraint
    #: (e.g. "hasPrincipalInvestigator: exactly 1 of SeniorResearcher"). Built
    #: from the source-shape's SHACL triples — gives the UI a clear "this is
    #: the constraint" label without the user having to read the verbose
    #: pyshacl message.
    constraint_label: Optional[str] = None


@dataclass
class ConsistencyReport:
    issues: List[ConsistencyIssue]
    issue_count: int
    severity_counts: Dict[str, int]
    kg_signature: str
    inference_used: str


# ---------------------------------------------------------------------------
# Constraint-component → ConstraintKind label mapping
# ---------------------------------------------------------------------------


_COMPONENT_TO_KIND: Dict[str, str] = {
    f"{SH}MinCountConstraintComponent": "minCardinality",
    f"{SH}MaxCountConstraintComponent": "maxCardinality",
    f"{SH}QualifiedMinCountConstraintComponent": "minQualifiedCardinality",
    f"{SH}QualifiedMaxCountConstraintComponent": "maxQualifiedCardinality",
    f"{SH}DatatypeConstraintComponent": "datatype",
    f"{SH}NodeKindConstraintComponent": "nodeKind",
    f"{SH}PatternConstraintComponent": "pattern",
    f"{SH}MinLengthConstraintComponent": "minLength",
    f"{SH}MaxLengthConstraintComponent": "maxLength",
    f"{SH}MinInclusiveConstraintComponent": "minInclusive",
    f"{SH}MaxInclusiveConstraintComponent": "maxInclusive",
    f"{SH}MinExclusiveConstraintComponent": "minExclusive",
    f"{SH}MaxExclusiveConstraintComponent": "maxExclusive",
    f"{SH}LanguageInConstraintComponent": "languageIn",
    f"{SH}UniqueLangConstraintComponent": "uniqueLang",
    f"{SH}InConstraintComponent": "in",
    f"{SH}HasValueConstraintComponent": "hasValue",
    f"{SH}ClassConstraintComponent": "someValuesFrom",
    f"{SH}NotConstraintComponent": "shaclNot",
    f"{SH}AndConstraintComponent": "shaclAnd",
    f"{SH}OrConstraintComponent": "shaclOr",
    f"{SH}XoneConstraintComponent": "shaclXone",
    f"{SH}ClosedConstraintComponent": "shaclClosed",
    f"{SH}DisjointConstraintComponent": "shaclDisjoint",
    f"{SH}EqualsConstraintComponent": "shaclEquals",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_kg_consistency(
    kg: KnowledgeGraph,
    *,
    inference: str = "owlrl",
) -> ConsistencyReport:
    """Validate ``kg``'s ABox against its OWL/SHACL constraints.

    Args:
        kg: The KG to check.
        inference: ``"rdfs"`` (faster, SHACL-only checks) or ``"owlrl"``
            (default; runs OWL2-RL pre-inference so OWL DL constructs
            are also checked).

    Returns:
        :class:`ConsistencyReport` whose ``issues`` list is empty when
        the KG is fully conformant.
    """
    data_graph = knowledge_graph_to_rdf(kg, vocab="both")
    _normalize_count_literals(data_graph)
    _augment_with_owl_shim(data_graph)

    _conforms, results_graph, _results_text = pyshacl.validate(
        data_graph,
        shacl_graph=data_graph,
        ont_graph=data_graph,
        inference=inference,
        allow_warnings=True,
        advanced=True,
        meta_shacl=False,
        debug=False,
    )

    issues = _parse_validation_report(results_graph, kg, data_graph)
    issues = _dedupe(issues)

    return ConsistencyReport(
        issues=issues,
        issue_count=len(issues),
        severity_counts=_count_severities(issues),
        kg_signature=kg_signature(kg),
        inference_used=inference,
    )


# ---------------------------------------------------------------------------
# Graph pre-processing
# ---------------------------------------------------------------------------


def _normalize_count_literals(g: rdflib.Graph) -> None:
    """pyshacl strictly requires ``xsd:integer`` for ``sh:minCount`` /
    ``sh:maxCount`` and other count predicates, but the OWL exporter
    emits ``xsd:nonNegativeInteger`` for the matching OWL cardinality
    predicates. Rewrite in place — semantics are preserved (the value
    range is the same, only the datatype IRI differs)."""
    to_replace = []
    for s, p, o in g:
        if isinstance(o, Literal) and o.datatype == XSD.nonNegativeInteger:
            try:
                lit = Literal(int(o), datatype=XSD.integer)
            except (TypeError, ValueError):
                continue
            to_replace.append((s, p, o, lit))
    for s, p, old, new in to_replace:
        g.remove((s, p, old))
        g.add((s, p, new))


def _augment_with_owl_shim(g: rdflib.Graph) -> None:
    """Add synthetic SHACL shapes for OWL DL constructs that pyshacl +
    owlrl don't surface as validation violations on their own.

    The shapes are anonymous blank nodes whose subjects never leak
    outside the consistency check — the caller-visible RDF graph is
    untouched. Translations:

    - ``:A owl:disjointWith :B`` → ``[a sh:NodeShape ; sh:targetClass :A ;
      sh:not [sh:class :B]]`` (and symmetrically).
    - ``:A owl:complementOf :B`` → ``[a sh:NodeShape ; sh:targetClass :A ;
      sh:not [sh:class :B]]``.
    - ``:p a owl:FunctionalProperty`` → ``[a sh:NodeShape ; sh:targetSubjectsOf
      :p ; sh:property [sh:path :p ; sh:maxCount 1]]``.
    - ``owl:AllDisjointClasses`` axioms expand to all pairwise disjointWith.
    """
    seen_pairs: Set[Tuple[URIRef, URIRef]] = set()

    def _add_disjoint_shape(a: URIRef, b: URIRef) -> None:
        if (a, b) in seen_pairs:
            return
        seen_pairs.add((a, b))
        shape = BNode()
        inner = BNode()
        g.add((shape, RDF.type, SH.NodeShape))
        g.add((shape, SH.targetClass, a))
        g.add((shape, SH["not"], inner))
        g.add((inner, SH["class"], b))

    # owl:disjointWith — symmetric.
    for a, b in list(g.subject_objects(OWL.disjointWith)):
        if isinstance(a, URIRef) and isinstance(b, URIRef):
            _add_disjoint_shape(a, b)
            _add_disjoint_shape(b, a)

    # owl:complementOf.
    for a, b in list(g.subject_objects(OWL.complementOf)):
        if isinstance(a, URIRef) and isinstance(b, URIRef):
            _add_disjoint_shape(a, b)

    # owl:AllDisjointClasses — expand each membership list into pairwise.
    for axiom in list(g.subjects(RDF.type, OWL.AllDisjointClasses)):
        members_head = next(g.objects(axiom, OWL.members), None)
        if members_head is None:
            continue
        members = list(rdflib.collection.Collection(g, members_head))
        members = [m for m in members if isinstance(m, URIRef)]
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                _add_disjoint_shape(a, b)
                _add_disjoint_shape(b, a)

    # owl:FunctionalProperty → per-subject max-1 constraint.
    for prop in list(g.subjects(RDF.type, OWL.FunctionalProperty)):
        if not isinstance(prop, URIRef):
            continue
        shape = BNode()
        ps = BNode()
        g.add((shape, RDF.type, SH.NodeShape))
        g.add((shape, SH.targetSubjectsOf, prop))
        g.add((shape, SH.property, ps))
        g.add((ps, SH.path, prop))
        g.add((ps, SH.maxCount, Literal(1, datatype=XSD.integer)))

    # Our SHACL exporter marks inline anonymous shapes under sh:not / sh:and /
    # sh:or / sh:xone as `rdf:type sh:PropertyShape`, but pyshacl strictly
    # requires every PropertyShape to carry an `sh:path`. Inline shapes
    # nested under a logical operator inherit their path from the outer
    # shape and don't repeat it — strip the explicit type so pyshacl's
    # structural inference treats them as the implicit NodeShape they are.
    bnodes_without_path = [
        bn for bn in g.subjects(RDF.type, SH.PropertyShape)
        if isinstance(bn, BNode) and not any(g.objects(bn, SH.path))
    ]
    for bn in bnodes_without_path:
        g.remove((bn, RDF.type, SH.PropertyShape))


# ---------------------------------------------------------------------------
# Validation-report parsing
# ---------------------------------------------------------------------------


_SEVERITY_MAP = {
    f"{SH}Violation": "violation",
    f"{SH}Warning": "warning",
    f"{SH}Info": "info",
}


def _parse_validation_report(
    results_graph: rdflib.Graph,
    kg: KnowledgeGraph,
    data_graph: rdflib.Graph,
) -> List[ConsistencyIssue]:
    iri_to_node_id = _build_iri_to_node_id(kg)

    issues: List[ConsistencyIssue] = []
    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        focus = next(results_graph.objects(result, SH.focusNode), None)
        path = next(results_graph.objects(result, SH.resultPath), None)
        message = next(results_graph.objects(result, SH.resultMessage), None)
        severity_iri = next(results_graph.objects(result, SH.resultSeverity), None)
        source_shape = next(results_graph.objects(result, SH.sourceShape), None)
        component = next(results_graph.objects(result, SH.sourceConstraintComponent), None)
        value = next(results_graph.objects(result, SH.value), None)

        focus_id = _term_to_node_id(focus, iri_to_node_id)
        affected = [focus_id] if focus_id else []
        value_id = _term_to_node_id(value, iri_to_node_id)
        if value_id and value_id not in affected:
            affected.append(value_id)

        component_iri = str(component) if component else ""
        component_local = component_iri.split("#")[-1] if component_iri else "Unknown"

        spec_kind = _COMPONENT_TO_KIND.get(component_iri)
        severity = _SEVERITY_MAP.get(str(severity_iri), "violation") if severity_iri else "violation"

        constraint_node_id = _term_to_node_id(source_shape, iri_to_node_id)

        path_iri: Optional[str] = None
        if isinstance(path, URIRef):
            path_iri = str(path)

        msg_text = str(message) if message else _synthesize_message(focus, component_local, path_iri)
        constraint_label = _constraint_label_for(source_shape, path_iri, spec_kind, component_local, data_graph)

        issue_id = _hash_id(focus, source_shape, component, value)

        issues.append(
            ConsistencyIssue(
                id=issue_id,
                code=component_local,
                severity=severity,
                message=msg_text,
                affected_node_ids=affected,
                affected_edge_ids=[],
                constraint_node_id=constraint_node_id,
                spec_kind=spec_kind,
                path_iri=path_iri,
                constraint_label=constraint_label,
            )
        )
    issues.sort(key=lambda i: (i.severity, i.code, i.id))
    return issues


def _local_name(iri: str) -> str:
    """Return the local part of an IRI for display."""
    if not iri:
        return ""
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    if "/" in iri:
        return iri.rsplit("/", 1)[-1]
    return iri


def _shape_value(g: rdflib.Graph, shape: Any, predicate: URIRef) -> Optional[Any]:
    """First object of (shape, predicate, *) in the graph, or None."""
    if shape is None:
        return None
    return next(g.objects(shape, predicate), None)


def _format_literal_or_iri(term: Any) -> str:
    if isinstance(term, URIRef):
        return _local_name(str(term))
    if isinstance(term, Literal):
        return f'"{term}"'
    return str(term)


def _constraint_label_for(
    source_shape: Any,
    path_iri: Optional[str],
    spec_kind: Optional[str],
    component_local: str,
    data_graph: rdflib.Graph,
) -> Optional[str]:
    """Compose a short human-readable label for the violated constraint.

    Reads the source shape's SHACL triples (counts, datatypes, value classes,
    …) and combines them with the property path (when present) so users can
    grok the rule at a glance without parsing pyshacl's verbose message.

    Returns ``None`` only when there is nothing useful to say beyond the
    constraint-component name (which the row already shows as a chip).
    """
    if source_shape is None:
        return None
    kind = spec_kind or component_local

    # Pull the most relevant value(s) for each kind.
    parts: List[str] = []
    if kind in ("minCardinality", "MinCountConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.minCount)
        parts.append(f"at least {n}" if n is not None else "at least N")
    elif kind in ("maxCardinality", "MaxCountConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.maxCount)
        parts.append(f"at most {n}" if n is not None else "at most N")
    elif kind in ("minQualifiedCardinality", "QualifiedMinCountConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.qualifiedMinCount)
        qv = _shape_value(data_graph, source_shape, SH.qualifiedValueShape)
        cls = _shape_value(data_graph, qv, SH["class"]) if qv is not None else None
        max_qv = _shape_value(data_graph, source_shape, SH.qualifiedMaxCount)
        if n is not None and max_qv is not None and str(n) == str(max_qv):
            parts.append(f"exactly {n} of {_format_literal_or_iri(cls) if cls is not None else 'class'}")
        elif n is not None:
            parts.append(f"at least {n} of {_format_literal_or_iri(cls) if cls is not None else 'class'}")
    elif kind in ("maxQualifiedCardinality", "QualifiedMaxCountConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.qualifiedMaxCount)
        qv = _shape_value(data_graph, source_shape, SH.qualifiedValueShape)
        cls = _shape_value(data_graph, qv, SH["class"]) if qv is not None else None
        parts.append(f"at most {n} of {_format_literal_or_iri(cls) if cls is not None else 'class'}")
    elif kind in ("datatype", "DatatypeConstraintComponent"):
        d = _shape_value(data_graph, source_shape, SH.datatype)
        parts.append(f"datatype {_format_literal_or_iri(d)}" if d is not None else "datatype")
    elif kind in ("pattern", "PatternConstraintComponent"):
        p = _shape_value(data_graph, source_shape, SH.pattern)
        parts.append(f"matches /{p}/" if p is not None else "matches regex")
    elif kind in ("minLength", "MinLengthConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.minLength)
        parts.append(f"length ≥ {n}" if n is not None else "minimum length")
    elif kind in ("maxLength", "MaxLengthConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.maxLength)
        parts.append(f"length ≤ {n}" if n is not None else "maximum length")
    elif kind in ("minInclusive", "MinInclusiveConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.minInclusive)
        parts.append(f"≥ {n}" if n is not None else "minimum inclusive")
    elif kind in ("maxInclusive", "MaxInclusiveConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.maxInclusive)
        parts.append(f"≤ {n}" if n is not None else "maximum inclusive")
    elif kind in ("minExclusive", "MinExclusiveConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.minExclusive)
        parts.append(f"> {n}" if n is not None else "exclusive minimum")
    elif kind in ("maxExclusive", "MaxExclusiveConstraintComponent"):
        n = _shape_value(data_graph, source_shape, SH.maxExclusive)
        parts.append(f"< {n}" if n is not None else "exclusive maximum")
    elif kind in ("in", "InConstraintComponent"):
        head = _shape_value(data_graph, source_shape, SH["in"])
        if head is not None:
            items = list(rdflib.collection.Collection(data_graph, head))
            parts.append(f"in {{{', '.join(_format_literal_or_iri(i) for i in items)}}}")
        else:
            parts.append("in set")
    elif kind in ("hasValue", "HasValueConstraintComponent"):
        v = _shape_value(data_graph, source_shape, SH.hasValue)
        parts.append(f"must equal {_format_literal_or_iri(v)}" if v is not None else "must equal a value")
    elif kind in ("nodeKind", "NodeKindConstraintComponent"):
        v = _shape_value(data_graph, source_shape, SH.nodeKind)
        parts.append(f"node kind {_format_literal_or_iri(v)}" if v is not None else "node kind")
    elif kind in ("someValuesFrom", "ClassConstraintComponent"):
        v = _shape_value(data_graph, source_shape, SH["class"])
        parts.append(f"of class {_format_literal_or_iri(v)}" if v is not None else "of class")
    elif kind in ("languageIn", "LanguageInConstraintComponent"):
        head = _shape_value(data_graph, source_shape, SH.languageIn)
        if head is not None:
            tags = list(rdflib.collection.Collection(data_graph, head))
            parts.append(f"language in {{{', '.join(str(t) for t in tags)}}}")
    elif kind in ("uniqueLang", "UniqueLangConstraintComponent"):
        parts.append("each language at most once")
    elif kind in ("shaclNot", "NotConstraintComponent"):
        # The negated inner shape commonly carries a sh:class — surface it.
        inner = _shape_value(data_graph, source_shape, SH["not"])
        inner_cls = _shape_value(data_graph, inner, SH["class"]) if inner is not None else None
        if inner_cls is not None:
            parts.append(f"must NOT be of class {_format_literal_or_iri(inner_cls)}")
        else:
            parts.append("must NOT satisfy nested shape")
    elif kind in ("shaclAnd", "AndConstraintComponent"):
        parts.append("must satisfy ALL nested shapes")
    elif kind in ("shaclOr", "OrConstraintComponent"):
        parts.append("must satisfy ANY nested shape")
    elif kind in ("shaclXone", "XoneConstraintComponent"):
        parts.append("must satisfy EXACTLY ONE nested shape")
    elif kind in ("shaclDisjoint", "DisjointConstraintComponent"):
        other = _shape_value(data_graph, source_shape, SH.disjoint)
        parts.append(f"disjoint from {_format_literal_or_iri(other)}" if other is not None else "disjoint")
    elif kind in ("shaclClosed", "ClosedConstraintComponent"):
        parts.append("closed shape (no extra properties)")
    else:
        # Fallback — use the component local name as-is.
        parts.append(component_local.replace("ConstraintComponent", ""))

    body = ", ".join(parts)
    if path_iri:
        return f"{_local_name(path_iri)}: {body}"
    return body


def _build_iri_to_node_id(kg: KnowledgeGraph) -> Dict[str, str]:
    """Map every rdflib-term-as-string to the KG node id that produced it.

    The exporter at :func:`knowledge_graph_to_rdf` uses three strategies in
    ``_term_for_node`` for non-literal nodes:

    1. ``node.iri`` directly when set;
    2. an explicit :class:`BNode` for KGBlank, KGNodeConstraint, and
       KGPropertyConstraint without an iri (KGBlank uses ``node.id``
       without the ``_:`` prefix as the BNode handle);
    3. otherwise (KGIndividual, KGClass, KGProperty without an iri) a
       synthesised IRI ``<DEFAULT_NAMESPACE><slug(label or id)>``.

    We replicate that logic here so the consistency report can map every
    focus-node term pyshacl emits back to a KG node id — without this,
    UI-created individuals (which have no explicit iri) come back with
    empty ``affected_node_ids`` and the "Fix in KG" button never appears.
    """
    mapping: Dict[str, str] = {}
    for node in kg.nodes:
        nid = getattr(node, "id", None)
        if not nid:
            continue
        # Identity mapping always.
        mapping[nid] = nid
        iri = getattr(node, "iri", None)
        if iri:
            mapping[iri] = nid
            continue
        # Mirror the exporter's BNode-vs-synthetic-URI choice.
        if isinstance(node, KGBlank):
            bn_id = nid[2:] if nid.startswith("_:") else nid
            if bn_id:
                mapping[f"_:{bn_id}"] = nid
        elif isinstance(node, (KGNodeConstraint, KGPropertyConstraint)):
            # The exporter mints a deterministic BNode from
            # `_slugify(node.id)` for constraint nodes without an iri so
            # pyshacl's `sh:sourceShape` can be traced back here.
            slug = _slugify(nid)
            if slug:
                mapping[f"_:{slug}"] = nid
        elif isinstance(node, KGLiteral):
            # Literals can't be a focus node anyway.
            continue
        else:
            # KGIndividual / KGClass / KGProperty without an iri.
            slug = _slugify(getattr(node, "label", None) or nid)
            mapping[f"{DEFAULT_NAMESPACE}{slug}"] = nid
    return mapping


def _term_to_node_id(
    term: Optional[Any], iri_to_node_id: Dict[str, str]
) -> Optional[str]:
    if term is None:
        return None
    if isinstance(term, URIRef):
        return iri_to_node_id.get(str(term))
    if isinstance(term, BNode):
        # The exporter materialises stable BNode handles for KGBlank /
        # KGNodeConstraint / KGPropertyConstraint nodes (see
        # _build_iri_to_node_id). Consult the mapping first; if we can't
        # resolve, surface the raw bnode handle so the frontend at least
        # has *something* to display.
        key = f"_:{str(term)}"
        mapped = iri_to_node_id.get(key)
        return mapped if mapped is not None else key
    if isinstance(term, Literal):
        return None
    return iri_to_node_id.get(str(term))


def _synthesize_message(focus: Any, component_local: str, path_iri: Optional[str]) -> str:
    focus_label = str(focus) if focus is not None else "<unknown>"
    if path_iri:
        return f"{focus_label} violates {component_local} on <{path_iri}>"
    return f"{focus_label} violates {component_local}"


def _hash_id(*terms: Any) -> str:
    h = hashlib.md5()
    for t in terms:
        h.update(str(t).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _dedupe(issues: List[ConsistencyIssue]) -> List[ConsistencyIssue]:
    seen: Set[Tuple[str, str, Tuple[str, ...]]] = set()
    out: List[ConsistencyIssue] = []
    for i in issues:
        key = (i.code, i.constraint_node_id or "", tuple(i.affected_node_ids))
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


def _count_severities(issues: List[ConsistencyIssue]) -> Dict[str, int]:
    counts = {"info": 0, "warning": 0, "violation": 0}
    for i in issues:
        if i.severity in counts:
            counts[i.severity] += 1
    return counts
