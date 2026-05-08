"""KG → BUML preflight: surface every non-1-to-1 mapping for user review.

The deterministic ``kg_to_class_diagram`` / ``kg_to_object_diagram``
converters happily produce a BUML model from any KG, but several KG
elements don't have a clean 1-to-1 mapping into BUML (blank nodes used
as instances, properties without a domain, multi-valued literals,
unsupported restrictions, …). For each such element we emit a
:class:`KGIssue`. The frontend renders one row per issue, lets the user
choose ``"accept"`` (apply the recommended action) or ``"skip"`` (drop
the element from the BUML output), and POSTs the decisions back to the
conversion endpoint along with a ``kgSignature`` to detect a stale
graph.

The "blocking vs advisory" distinction has been dropped: every issue is
presented uniformly. The user can always proceed (or fix manually in
the editor and re-run preflight).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from besser.BUML.metamodel.kg import (
    EquivalentClassesAxiom,
    InversePropertiesAxiom,
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNode,
    KGProperty,
    KnowledgeGraph,
)

from besser.BUML.notations.kg_to_buml._common import (
    RDFS_DOMAIN,
    RDFS_RANGE,
    RDFS_SUBCLASS_OF,
    RDF_TYPE,
    build_indexes,
    is_meta_vocab,
    local_name,
    normalize_predicate,
    sanitize_python_identifier,
    sorted_by_id,
)
from besser.BUML.notations.kg_to_buml.datatype_mapping import (
    parse_literal,
    xsd_to_primitive,
)
from besser.BUML.notations.kg_to_buml.kg_to_class_diagram import (
    _looks_like_datatype_iri,
)


__all__ = [
    "KGAction",
    "KGIssue",
    "KGPreflightReport",
    "analyze_kg_for_class_diagram",
    "analyze_kg_for_object_diagram",
    "kg_signature",
]


_XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"

ORPHAN_DESCRIPTION_TRUNCATE = 25
_CLASS_ANCHOR_PREDICATES = frozenset({RDF_TYPE, RDFS_DOMAIN, RDFS_RANGE, RDFS_SUBCLASS_OF})


@dataclass
class KGAction:
    """A pre-filled action ready to be applied to a KG.

    Attributes:
        key: Handler key dispatched in :mod:`resolutions`.
        parameters: Fully-resolved parameters (no user input required).
            Deterministic defaults are computed at preflight time so the
            frontend doesn't need to render any form fields.
        label: Human-readable label shown next to the recommendation
            checkbox in the modal.
    """

    key: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    label: str = ""


@dataclass
class KGIssue:
    """A KG element that does not have a 1-to-1 mapping into BUML.

    Each issue carries one *recommended* and one *skip* action — the
    user picks between them via a checkbox in the frontend modal.
    """

    id: str
    code: str
    description: str
    affected_node_ids: List[str] = field(default_factory=list)
    affected_edge_ids: List[str] = field(default_factory=list)
    recommended_action: Optional[KGAction] = None
    skip_action: Optional[KGAction] = None


@dataclass
class KGPreflightReport:
    """Result of :func:`analyze_kg_for_class_diagram` /
    :func:`analyze_kg_for_object_diagram`."""

    issues: List[KGIssue] = field(default_factory=list)
    issue_count: int = 0
    kg_signature: str = ""
    diagram_type: str = "ClassDiagram"  # or "ObjectDiagram"


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def kg_signature(kg: KnowledgeGraph) -> str:
    """Short stable hash of the KG's node-id and edge-id sets."""
    node_ids = sorted(n.id for n in kg.nodes)
    edge_ids = sorted(e.id for e in kg.edges)
    raw = "\0".join(node_ids) + "\0\0" + "\0".join(edge_ids)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:16]


def analyze_kg_for_class_diagram(kg: KnowledgeGraph) -> KGPreflightReport:
    """Walk a KG and return every issue the user should review before
    converting to a Class Diagram."""
    issues: List[KGIssue] = []
    indexes = build_indexes(kg)

    orphan_issues, orphan_node_ids = _detect_orphan_nodes(kg, indexes)

    issues.extend(_detect_blank_node_instance(kg, indexes))
    issues.extend(_detect_undeclared_class(kg, indexes))
    issues.extend(_detect_no_domain(kg, indexes))
    issues.extend(_detect_multiple_domains(kg, indexes))
    issues.extend(_detect_unattached_restrictions(kg, indexes))
    issues.extend(_detect_unsupported_restrictions(kg, indexes))
    issues.extend(_detect_property_name_collisions(kg, indexes))
    issues.extend(_detect_range_both_datatype_and_class(kg, indexes))
    issues.extend(_detect_range_not_type(kg, indexes))
    issues.extend(_detect_no_range(kg, indexes))
    issues.extend(_detect_unmapped_datatypes(kg, indexes))
    issues.extend(_detect_cyclic_subclass(kg, indexes))
    issues.extend(_detect_multivalued_literal(kg, indexes))
    issues.extend(_detect_punning(kg))
    issues.extend(_detect_equivalent_classes(kg))
    issues.extend(_detect_inverse_properties(kg))
    issues.extend(orphan_issues)

    return KGPreflightReport(
        issues=issues,
        issue_count=len(issues),
        kg_signature=kg_signature(kg),
        diagram_type="ClassDiagram",
    )


def analyze_kg_for_object_diagram(kg: KnowledgeGraph) -> KGPreflightReport:
    """Walk a KG and return every issue the user should review before
    converting to an Object Diagram. Class-level issues are NOT included
    here (they only appear when the user converts to Class Diagram)."""
    issues: List[KGIssue] = []
    indexes = build_indexes(kg)

    orphan_issues, orphan_node_ids = _detect_orphan_nodes(kg, indexes)

    issues.extend(_detect_blank_node_as_object(kg, indexes))
    issues.extend(_detect_individual_no_type(kg, indexes, suppress_node_ids=orphan_node_ids))
    issues.extend(_detect_multiple_types(kg, indexes))
    issues.extend(_detect_literal_type_coerced(kg, indexes))
    issues.extend(_detect_link_type_mismatch(kg, indexes))
    issues.extend(_detect_multivalued_literal(kg, indexes))
    issues.extend(orphan_issues)

    return KGPreflightReport(
        issues=issues,
        issue_count=len(issues),
        kg_signature=kg_signature(kg),
        diagram_type="ObjectDiagram",
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _issue_id(code: str, *parts: str) -> str:
    raw = code + "\0" + "\0".join(sorted(parts))
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:12]


def _classes_in(kg: KnowledgeGraph) -> List[KGClass]:
    return sorted_by_id([n for n in kg.nodes if isinstance(n, KGClass)])


def _properties_in(kg: KnowledgeGraph) -> List[KGProperty]:
    return sorted_by_id([n for n in kg.nodes if isinstance(n, KGProperty)])


def _individuals_in(kg: KnowledgeGraph) -> List[KGIndividual]:
    return sorted_by_id([n for n in kg.nodes if isinstance(n, KGIndividual)])


def _domain_targets(prop: KGProperty, indexes) -> List[Any]:
    return [
        e.target for e in indexes.out_with_predicate(prop.id, RDFS_DOMAIN)
        if not isinstance(e.target, (KGBlank, KGLiteral))
    ]


def _range_targets(prop: KGProperty, indexes) -> List[Any]:
    return [e.target for e in indexes.out_with_predicate(prop.id, RDFS_RANGE)]


def _has_abox_usage(prop_iri: Optional[str], kg: KnowledgeGraph) -> bool:
    if not prop_iri:
        return False
    for edge in kg.edges:
        if edge.iri == prop_iri and isinstance(edge.source, KGIndividual):
            return True
    return False


# ----------------------------------------------------------------------
# Class-diagram detectors
# ----------------------------------------------------------------------


def _detect_no_domain(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    out: List[KGIssue] = []
    for prop in _properties_in(kg):
        if _domain_targets(prop, indexes):
            continue
        if not _has_abox_usage(prop.iri, kg):
            continue
        prop_label = prop.label or local_name(prop.iri) or prop.id
        out.append(KGIssue(
            id=_issue_id("PROPERTY_NO_DOMAIN", prop.id),
            code="PROPERTY_NO_DOMAIN",
            description=(
                f"Property '{prop_label}' has no rdfs:domain but is used in the ABox. "
                f"Without a domain it cannot be attached to a specific class."
            ),
            affected_node_ids=[prop.id],
            recommended_action=KGAction(
                key="attach_to_thing",
                parameters={"property_iri": prop.iri or prop.id},
                label=f"Attach '{prop_label}' to a synthetic 'Thing' class",
            ),
            skip_action=KGAction(
                key="drop_property",
                parameters={"property_iri": prop.iri or prop.id},
                label=f"Drop the '{prop_label}' property",
            ),
        ))
    return out


def _detect_multiple_domains(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    out: List[KGIssue] = []
    parents_of: Dict[str, Set[str]] = defaultdict(set)
    for edge in kg.edges:
        if normalize_predicate(edge.iri) == RDFS_SUBCLASS_OF:
            parents_of[edge.source.id].add(edge.target.id)

    def _ancestors(node_id: str) -> Set[str]:
        seen: Set[str] = set()
        stack = list(parents_of.get(node_id, ()))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(parents_of.get(x, ()))
        return seen

    for prop in _properties_in(kg):
        domains = _domain_targets(prop, indexes)
        if len(domains) <= 1:
            continue
        domain_ids = sorted({d.id for d in domains})
        # If they're all in a chain, it's not an issue.
        chain_ok = False
        for d in domains:
            if set(domain_ids) - {d.id} <= _ancestors(d.id):
                chain_ok = True
                break
            descendants = _descendants_of(d.id, parents_of)
            if set(domain_ids) - {d.id} <= descendants:
                chain_ok = True
                break
        if chain_ok:
            continue
        prop_label = prop.label or local_name(prop.iri) or prop.id
        first_domain_id = domain_ids[0]  # deterministic: lex-first
        first_domain_label = next(
            (d.label or local_name(d.iri) or d.id for d in domains if d.id == first_domain_id),
            first_domain_id,
        )
        out.append(KGIssue(
            id=_issue_id("MULTIPLE_DOMAINS", prop.id),
            code="MULTIPLE_DOMAINS",
            description=(
                f"Property '{prop_label}' has {len(domains)} unrelated rdfs:domain values. "
                f"Only one can be the owner class in BUML."
            ),
            affected_node_ids=[prop.id, *domain_ids],
            recommended_action=KGAction(
                key="pick_domain",
                parameters={"property_iri": prop.iri or prop.id, "class_id": first_domain_id},
                label=f"Keep only '{first_domain_label}' as the domain",
            ),
            skip_action=KGAction(
                key="split_per_domain",
                parameters={"property_iri": prop.iri or prop.id},
                label=f"Split into one property per domain",
            ),
        ))
    return out


def _descendants_of(node_id: str, parents_of: Dict[str, Set[str]]) -> Set[str]:
    children_of: Dict[str, Set[str]] = defaultdict(set)
    for child, parents in parents_of.items():
        for p in parents:
            children_of[p].add(child)
    seen: Set[str] = set()
    stack = list(children_of.get(node_id, ()))
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(children_of.get(x, ()))
    return seen


def _detect_unattached_restrictions(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    out: List[KGIssue] = []
    LINK_PREDS = {RDFS_SUBCLASS_OF, "http://www.w3.org/2002/07/owl#equivalentClass"}
    attached: Set[str] = set()
    for edge in kg.edges:
        if normalize_predicate(edge.iri) not in LINK_PREDS:
            continue
        if isinstance(edge.target, KGBlank) and edge.target.metadata.get("kind") == "restriction":
            if isinstance(edge.source, KGClass):
                attached.add(edge.target.id)
    for blank in sorted_by_id([
        n for n in kg.nodes
        if isinstance(n, KGBlank) and n.metadata.get("kind") == "restriction"
    ]):
        if blank.id in attached:
            continue
        on_prop = blank.metadata.get("on_property") or "<unknown>"
        out.append(KGIssue(
            id=_issue_id("RESTRICTION_UNATTACHED", blank.id),
            code="RESTRICTION_UNATTACHED",
            description=(
                f"An owl:Restriction on property '{on_prop}' is not attached to any class. "
                f"It cannot be lifted into BUML without an owner."
            ),
            affected_node_ids=[blank.id],
            recommended_action=KGAction(
                key="drop_restriction",
                parameters={"blank_id": blank.id},
                label="Drop the unattached restriction",
            ),
            skip_action=KGAction(
                key="drop_restriction",
                parameters={"blank_id": blank.id},
                label="Drop the unattached restriction",
            ),
        ))
    return out


def _detect_unsupported_restrictions(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    """Restrictions that *are* attached but use semantics with no clean BUML mapping
    (allValuesFrom, hasValue, hasSelf)."""
    UNSUPPORTED = {"allValuesFrom", "hasValue", "hasSelf"}
    out: List[KGIssue] = []
    for blank in sorted_by_id([
        n for n in kg.nodes
        if isinstance(n, KGBlank)
        and n.metadata.get("kind") == "restriction"
        and n.metadata.get("restriction_type") in UNSUPPORTED
    ]):
        rtype = blank.metadata.get("restriction_type")
        on_prop = blank.metadata.get("on_property") or "<unknown>"
        out.append(KGIssue(
            id=_issue_id("RESTRICTION_UNSUPPORTED", blank.id),
            code="RESTRICTION_UNSUPPORTED",
            description=(
                f"Restriction '{rtype}' on property '{on_prop}' has no clean mapping into BUML "
                f"multiplicity. The semantic constraint cannot be enforced by the class diagram alone."
            ),
            affected_node_ids=[blank.id],
            recommended_action=KGAction(
                key="record_as_description",
                parameters={"blank_id": blank.id},
                label="Record as a textual constraint on the class description",
            ),
            skip_action=KGAction(
                key="drop_restriction",
                parameters={"blank_id": blank.id},
                label="Drop the restriction entirely",
            ),
        ))
    return out


def _detect_property_name_collisions(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    out: List[KGIssue] = []
    bucket: Dict[Tuple[str, str], List[KGProperty]] = defaultdict(list)
    for prop in _properties_in(kg):
        if not prop.iri:
            continue
        name_base = sanitize_python_identifier(
            prop.label or local_name(prop.iri), "prop", set()
        )
        for dom in _domain_targets(prop, indexes):
            bucket[(name_base, dom.id)].append(prop)
    for (name, owner_id), props in bucket.items():
        if len(props) <= 1:
            continue
        # Deterministic: keep lex-first property unchanged; rename the rest with _2, _3.
        sorted_props = sorted(props, key=lambda p: p.id)
        rename_pairs = [
            {"target_property_id": p.id, "new_name": f"{name}_{idx}"}
            for idx, p in enumerate(sorted_props[1:], start=2)
        ]
        out.append(KGIssue(
            id=_issue_id("PROPERTY_NAME_COLLISION", *(p.id for p in sorted_props), owner_id),
            code="PROPERTY_NAME_COLLISION",
            description=(
                f"Properties { [p.iri for p in sorted_props] } sanitise to the same name "
                f"'{name}' on the same owner class."
            ),
            affected_node_ids=[p.id for p in sorted_props],
            recommended_action=KGAction(
                key="rename_with_suffix",
                parameters={"renames": rename_pairs},
                label=f"Rename later properties to '{name}_2', '{name}_3', …",
            ),
            skip_action=KGAction(
                key="drop_property",
                parameters={"property_iri": sorted_props[-1].iri or sorted_props[-1].id},
                label=f"Drop the last colliding property",
            ),
        ))
    return out


def _detect_range_both_datatype_and_class(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    out: List[KGIssue] = []
    for prop in _properties_in(kg):
        ranges = _range_targets(prop, indexes)
        has_dt = any(_looks_like_datatype_iri(getattr(r, "iri", None)) for r in ranges)
        has_class = any(
            isinstance(r, KGClass) and not _looks_like_datatype_iri(getattr(r, "iri", None))
            for r in ranges
        )
        if has_dt and has_class:
            prop_label = prop.label or prop.id
            out.append(KGIssue(
                id=_issue_id("RANGE_BOTH_DATATYPE_AND_CLASS", prop.id),
                code="RANGE_BOTH_DATATYPE_AND_CLASS",
                description=(
                    f"Property '{prop_label}' has both a datatype and a class as rdfs:range. "
                    f"In BUML it must be either an attribute or an association — pick one role."
                ),
                affected_node_ids=[prop.id],
                recommended_action=KGAction(
                    key="keep_object_property",
                    parameters={"property_iri": prop.iri or prop.id},
                    label=f"Keep '{prop_label}' as an object property (drop datatype range)",
                ),
                skip_action=KGAction(
                    key="drop_property",
                    parameters={"property_iri": prop.iri or prop.id},
                    label=f"Drop the property",
                ),
            ))
    return out


def _detect_range_not_type(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    """Property whose rdfs:range targets an individual or blank node — neither
    is a valid class for an association nor a datatype for an attribute."""
    out: List[KGIssue] = []
    for prop in _properties_in(kg):
        ranges = _range_targets(prop, indexes)
        bad_ranges = [
            r for r in ranges
            if isinstance(r, (KGIndividual, KGBlank))
            and not _looks_like_datatype_iri(getattr(r, "iri", None))
        ]
        if not bad_ranges:
            continue
        prop_label = prop.label or prop.id
        out.append(KGIssue(
            id=_issue_id("RANGE_NOT_TYPE", prop.id),
            code="RANGE_NOT_TYPE",
            description=(
                f"Property '{prop_label}' has rdfs:range pointing to an individual or blank node "
                f"({[r.id for r in bad_ranges]}). Neither maps to a BUML class or datatype."
            ),
            affected_node_ids=[prop.id, *(r.id for r in bad_ranges)],
            recommended_action=KGAction(
                key="treat_as_string",
                parameters={"property_iri": prop.iri or prop.id},
                label=f"Treat '{prop_label}' as a string attribute",
            ),
            skip_action=KGAction(
                key="drop_property",
                parameters={"property_iri": prop.iri or prop.id},
                label=f"Drop the property",
            ),
        ))
    return out


def _detect_cyclic_subclass(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    out: List[KGIssue] = []
    parents_of: Dict[str, Set[str]] = defaultdict(set)
    parent_edge_id: Dict[Tuple[str, str], str] = {}
    for edge in kg.edges:
        if normalize_predicate(edge.iri) != RDFS_SUBCLASS_OF:
            continue
        if isinstance(edge.source, KGBlank) or isinstance(edge.target, KGBlank):
            continue
        if edge.source.id == edge.target.id:
            continue
        parents_of[edge.source.id].add(edge.target.id)
        parent_edge_id[(edge.source.id, edge.target.id)] = edge.id

    seen_cycle: Set[Tuple[str, ...]] = set()

    def _dfs(node_id: str, stack: List[str]):
        if node_id in stack:
            cycle = tuple(stack[stack.index(node_id):] + [node_id])
            seen_cycle.add(cycle)
            return
        stack.append(node_id)
        for p in parents_of.get(node_id, ()):
            _dfs(p, stack)
        stack.pop()

    for n in parents_of:
        _dfs(n, [])

    for cycle in sorted(seen_cycle):
        # Deterministic: pick the lex-last edge in the cycle to break.
        edge_ids = []
        for i in range(len(cycle) - 1):
            eid = parent_edge_id.get((cycle[i], cycle[i + 1]))
            if eid is not None:
                edge_ids.append(eid)
        if not edge_ids:
            continue
        break_edge = sorted(edge_ids)[-1]
        out.append(KGIssue(
            id=_issue_id("CYCLIC_SUBCLASS", *cycle),
            code="CYCLIC_SUBCLASS",
            description=(
                f"Multi-class rdfs:subClassOf cycle: " + " → ".join(cycle)
            ),
            affected_node_ids=list(cycle),
            affected_edge_ids=edge_ids,
            recommended_action=KGAction(
                key="break_at_edge",
                parameters={"edge_id": break_edge},
                label=f"Break the cycle at edge '{break_edge}'",
            ),
            skip_action=KGAction(
                key="break_at_edge",
                parameters={"edge_id": break_edge},
                label=f"Break the cycle at edge '{break_edge}'",
            ),
        ))
    return out


def _detect_no_range(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    out: List[KGIssue] = []
    for prop in _properties_in(kg):
        if _range_targets(prop, indexes):
            continue
        prop_label = prop.label or local_name(prop.iri) or prop.id
        out.append(KGIssue(
            id=_issue_id("PROPERTY_NO_RANGE", prop.id),
            code="PROPERTY_NO_RANGE",
            description=(
                f"Property '{prop_label}' has no rdfs:range. Without a range we have to "
                f"guess whether it's a datatype attribute or an association."
            ),
            affected_node_ids=[prop.id],
            recommended_action=KGAction(
                key="set_range",
                parameters={"property_iri": prop.iri or prop.id, "range_iri": _XSD_STRING},
                label=f"Default the range to xsd:string (datatype attribute)",
            ),
            skip_action=KGAction(
                key="drop_property",
                parameters={"property_iri": prop.iri or prop.id},
                label=f"Drop the property",
            ),
        ))
    return out


def _detect_unmapped_datatypes(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    out: List[KGIssue] = []
    for prop in _properties_in(kg):
        for rng in _range_targets(prop, indexes):
            iri = getattr(rng, "iri", None)
            if not iri or not _looks_like_datatype_iri(iri):
                continue
            _, known = xsd_to_primitive(iri)
            if known:
                continue
            prop_label = prop.label or prop.id
            out.append(KGIssue(
                id=_issue_id("UNMAPPED_DATATYPE", prop.id, iri),
                code="UNMAPPED_DATATYPE",
                description=(
                    f"Datatype IRI '{iri}' on property '{prop_label}' has no XSD mapping. "
                    f"It can only be represented as a string in BUML."
                ),
                affected_node_ids=[prop.id],
                recommended_action=KGAction(
                    key="keep_as_string",
                    parameters={},
                    label=f"Map '{iri}' to xsd:string",
                ),
                skip_action=KGAction(
                    key="drop_property",
                    parameters={"property_iri": prop.iri or prop.id},
                    label=f"Drop the property",
                ),
            ))
    return out


def _detect_punning(kg: KnowledgeGraph) -> List[KGIssue]:
    out: List[KGIssue] = []
    for node in sorted_by_id([n for n in kg.nodes if isinstance(n, KGClass)]):
        twin = node.metadata.get("punned_with") if node.metadata else None
        if not twin:
            continue
        out.append(KGIssue(
            id=_issue_id("PUNNING", node.id),
            code="PUNNING",
            description=(
                f"IRI '{node.iri or node.id}' is used as both a class and an individual "
                f"(OWL2 punning). BUML can model only one of those roles per name."
            ),
            affected_node_ids=[node.id, twin],
            recommended_action=KGAction(
                key="keep_both",
                parameters={},
                label="Keep both the class and the individual (default)",
            ),
            skip_action=KGAction(
                key="prefer_class",
                parameters={"individual_id": twin},
                label="Drop the individual side, keep only the class",
            ),
        ))
    return out


def _detect_equivalent_classes(kg: KnowledgeGraph) -> List[KGIssue]:
    out: List[KGIssue] = []
    for axiom in kg.axioms:
        if not isinstance(axiom, EquivalentClassesAxiom):
            continue
        ids = sorted({c for c in axiom.class_ids if c})
        if len(ids) < 2:
            continue
        out.append(KGIssue(
            id=_issue_id("EQUIVALENT_CLASSES", *ids),
            code="EQUIVALENT_CLASSES",
            description=f"Classes {ids} are declared equivalent (owl:equivalentClass).",
            affected_node_ids=ids,
            recommended_action=KGAction(
                key="keep_separate",
                parameters={},
                label="Keep them as separate classes (default)",
            ),
            skip_action=KGAction(
                key="merge_classes",
                parameters={"keep_class_id": ids[0], "drop_class_id": ids[1]},
                label=f"Merge into '{ids[0]}'",
            ),
        ))
    return out


def _detect_inverse_properties(kg: KnowledgeGraph) -> List[KGIssue]:
    out: List[KGIssue] = []
    for axiom in kg.axioms:
        if not isinstance(axiom, InversePropertiesAxiom):
            continue
        a, b = axiom.property_a_id, axiom.property_b_id
        if not (a and b):
            continue
        out.append(KGIssue(
            id=_issue_id("INVERSE_PROPERTY", a, b),
            code="INVERSE_PROPERTY",
            description=f"Properties '{a}' and '{b}' are inverses (owl:inverseOf).",
            affected_node_ids=[a, b],
            recommended_action=KGAction(
                key="keep_separate",
                parameters={},
                label="Keep two separate associations (default)",
            ),
            skip_action=KGAction(
                key="merge_associations",
                parameters={"property_a_iri": a, "property_b_iri": b},
                label="Merge into one bidirectional association",
            ),
        ))
    return out


def _detect_blank_node_instance(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    """Blank nodes that act as instances (subject of rdf:type or instance-edge endpoints)
    rather than restriction or class-expression markers."""
    out: List[KGIssue] = []
    structural_kinds = {"restriction", "class_expression"}
    for blank in sorted_by_id([n for n in kg.nodes if isinstance(n, KGBlank)]):
        if blank.metadata.get("kind") in structural_kinds:
            continue
        # Is this blank used as an instance (typed via rdf:type)?
        type_edges = indexes.out_with_predicate(blank.id, RDF_TYPE)
        # Or as the subject/object of any instance-level edge?
        any_outgoing = bool(indexes.out_edges.get(blank.id))
        any_incoming = bool(indexes.in_edges.get(blank.id))
        if not (type_edges or any_outgoing or any_incoming):
            continue
        out.append(KGIssue(
            id=_issue_id("BLANK_NODE_INSTANCE", blank.id),
            code="BLANK_NODE_INSTANCE",
            description=(
                f"Blank node '{blank.id}' is used as an instance but has no IRI. "
                f"BUML classes need named instances."
            ),
            affected_node_ids=[blank.id],
            recommended_action=KGAction(
                key="materialize_as_individual",
                parameters={"blank_id": blank.id},
                label="Promote it to a named individual",
            ),
            skip_action=KGAction(
                key="drop_node",
                parameters={"node_id": blank.id},
                label="Drop the blank node and its edges",
            ),
        ))
    return out


def _detect_undeclared_class(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    """Nodes referenced as classes (via rdf:type, rdfs:domain, rdfs:range, rdfs:subClassOf
    targets) but not explicitly declared as KGClass.

    The detector skips OWL/RDF/RDFS/XSD meta-vocabulary IRIs (e.g. ``owl:Class``,
    ``owl:DatatypeProperty``) — those are framework terms, not user classes,
    and the importer keeps them only because every triple becomes an edge.
    """
    declared_class_ids = {n.id for n in kg.nodes if isinstance(n, KGClass)}
    referenced_as_class: Set[str] = set()
    for edge in kg.edges:
        pred = normalize_predicate(edge.iri)
        if pred == RDF_TYPE:
            tgt = edge.target
            if not isinstance(tgt, (KGBlank, KGLiteral, KGProperty)):
                referenced_as_class.add(tgt.id)
        elif pred == RDFS_DOMAIN:
            referenced_as_class.add(edge.target.id)
        elif pred == RDFS_RANGE:
            tgt = edge.target
            if not isinstance(tgt, (KGBlank, KGLiteral, KGProperty)) and not _looks_like_datatype_iri(getattr(tgt, "iri", None)):
                referenced_as_class.add(tgt.id)
        elif pred == RDFS_SUBCLASS_OF:
            if not isinstance(edge.source, KGBlank):
                referenced_as_class.add(edge.source.id)
            if not isinstance(edge.target, KGBlank):
                referenced_as_class.add(edge.target.id)
    undeclared = referenced_as_class - declared_class_ids
    out: List[KGIssue] = []
    for node_id in sorted(undeclared):
        node = kg.get_node(node_id)
        if node is None or isinstance(node, (KGProperty, KGBlank, KGLiteral)):
            continue
        if isinstance(node, KGClass):
            continue
        if is_meta_vocab(getattr(node, "iri", None) or node_id):
            continue
        out.append(KGIssue(
            id=_issue_id("UNDECLARED_CLASS", node_id),
            code="UNDECLARED_CLASS",
            description=(
                f"Node '{node_id}' is referenced as a class but never declared via "
                f"`rdf:type owl:Class`. The converter can synthesise a class for it."
            ),
            affected_node_ids=[node_id],
            recommended_action=KGAction(
                key="synthesize_class",
                parameters={"node_id": node_id},
                label=f"Synthesise a class for '{node.label or node_id}'",
            ),
            skip_action=KGAction(
                key="drop_references",
                parameters={"node_id": node_id},
                label=f"Drop edges that reference '{node_id}' as a class",
            ),
        ))
    return out


def _detect_multivalued_literal(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    """Same KGIndividual + predicate has more than one literal value."""
    counts: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for edge in kg.edges:
        if not edge.iri:
            continue
        if isinstance(edge.source, KGIndividual) and isinstance(edge.target, KGLiteral):
            counts[(edge.source.id, edge.iri)].append(edge.id)
    out: List[KGIssue] = []
    seen_props: Set[str] = set()
    for (indiv_id, prop_iri), edge_ids in counts.items():
        if len(edge_ids) <= 1:
            continue
        if prop_iri in seen_props:
            continue
        seen_props.add(prop_iri)
        out.append(KGIssue(
            id=_issue_id("MULTIVALUED_LITERAL", prop_iri, indiv_id),
            code="MULTIVALUED_LITERAL",
            description=(
                f"Individual '{indiv_id}' has {len(edge_ids)} literal values for property "
                f"'{prop_iri}'. BUML attributes default to single-valued."
            ),
            affected_node_ids=[indiv_id],
            affected_edge_ids=edge_ids,
            recommended_action=KGAction(
                key="bump_to_unbounded",
                parameters={"property_iri": prop_iri},
                label="Bump multiplicity to 0..* (allow many values)",
            ),
            skip_action=KGAction(
                key="keep_first_only",
                parameters={"property_iri": prop_iri, "individual_id": indiv_id},
                label="Keep only the first literal value",
            ),
        ))
    return out


# ----------------------------------------------------------------------
# Object-diagram detectors
# ----------------------------------------------------------------------


def _detect_blank_node_as_object(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    out: List[KGIssue] = []
    structural_kinds = {"restriction", "class_expression"}
    for blank in sorted_by_id([n for n in kg.nodes if isinstance(n, KGBlank)]):
        if blank.metadata.get("kind") in structural_kinds:
            continue
        out.append(KGIssue(
            id=_issue_id("BLANK_NODE_AS_OBJECT", blank.id),
            code="BLANK_NODE_AS_OBJECT",
            description=(
                f"Blank node '{blank.id}' would be silently skipped in the object diagram. "
                f"It has no IRI to use as an object name."
            ),
            affected_node_ids=[blank.id],
            recommended_action=KGAction(
                key="materialize_as_individual",
                parameters={"blank_id": blank.id},
                label="Promote it to a named individual",
            ),
            skip_action=KGAction(
                key="drop_node",
                parameters={"node_id": blank.id},
                label="Drop the blank node and its edges",
            ),
        ))
    return out


def _detect_individual_no_type(
    kg: KnowledgeGraph,
    indexes,
    *,
    suppress_node_ids: Optional[Set[str]] = None,
) -> List[KGIssue]:
    out: List[KGIssue] = []
    suppressed = suppress_node_ids or set()
    for ind in _individuals_in(kg):
        if ind.id in suppressed:
            continue
        if is_meta_vocab(getattr(ind, "iri", None) or ind.id):
            continue
        type_edges = indexes.out_with_predicate(ind.id, RDF_TYPE)
        if type_edges:
            continue
        out.append(KGIssue(
            id=_issue_id("INDIVIDUAL_NO_TYPE", ind.id),
            code="INDIVIDUAL_NO_TYPE",
            description=(
                f"Individual '{ind.id}' has no rdf:type. Without a type the converter "
                f"can't pick a classifier."
            ),
            affected_node_ids=[ind.id],
            recommended_action=KGAction(
                key="assign_thing_class",
                parameters={"individual_id": ind.id},
                label="Type it as a synthetic 'Thing'",
            ),
            skip_action=KGAction(
                key="drop_node",
                parameters={"node_id": ind.id},
                label="Drop the individual",
            ),
        ))
    return out


def _detect_multiple_types(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    """Individual has multiple rdf:type edges that aren't in a subclass chain."""
    out: List[KGIssue] = []
    parents_of: Dict[str, Set[str]] = defaultdict(set)
    for edge in kg.edges:
        if normalize_predicate(edge.iri) == RDFS_SUBCLASS_OF:
            parents_of[edge.source.id].add(edge.target.id)

    def _ancestors(nid: str) -> Set[str]:
        seen: Set[str] = set()
        stack = list(parents_of.get(nid, ()))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(parents_of.get(x, ()))
        return seen

    for ind in _individuals_in(kg):
        type_targets = [
            e.target for e in indexes.out_with_predicate(ind.id, RDF_TYPE)
            if isinstance(e.target, KGClass)
        ]
        if len(type_targets) <= 1:
            continue
        type_ids = sorted({t.id for t in type_targets})
        # If they're all in a chain, no issue.
        chain_ok = False
        for t in type_targets:
            if set(type_ids) - {t.id} <= _ancestors(t.id):
                chain_ok = True
                break
        if chain_ok:
            continue
        most_specific = type_ids[0]  # deterministic: lex-first
        out.append(KGIssue(
            id=_issue_id("MULTIPLE_TYPES", ind.id),
            code="MULTIPLE_TYPES",
            description=(
                f"Individual '{ind.id}' has {len(type_ids)} incomparable rdf:type values "
                f"({type_ids}). BUML objects have a single classifier."
            ),
            affected_node_ids=[ind.id, *type_ids],
            recommended_action=KGAction(
                key="pick_most_specific",
                parameters={"individual_id": ind.id, "class_id": most_specific},
                label=f"Keep '{most_specific}' as the classifier",
            ),
            skip_action=KGAction(
                key="drop_node",
                parameters={"node_id": ind.id},
                label="Drop the individual",
            ),
        ))
    return out


def _detect_literal_type_coerced(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    """Literal value can't parse cleanly into its declared datatype."""
    out: List[KGIssue] = []
    for lit in [n for n in kg.nodes if isinstance(n, KGLiteral)]:
        if not lit.datatype or lit.datatype == _XSD_STRING:
            continue
        try:
            parse_literal(lit.value, lit.datatype)
        except Exception:
            out.append(KGIssue(
                id=_issue_id("LITERAL_TYPE_COERCED", lit.id),
                code="LITERAL_TYPE_COERCED",
                description=(
                    f"Literal '{lit.value}' (datatype '{lit.datatype}') can't be parsed; "
                    f"it would be coerced to a raw string."
                ),
                affected_node_ids=[lit.id],
                recommended_action=KGAction(
                    key="coerce_to_string",
                    parameters={"literal_id": lit.id},
                    label="Coerce the value to a plain string",
                ),
                skip_action=KGAction(
                    key="drop_slot",
                    parameters={"literal_id": lit.id},
                    label="Drop the literal slot entirely",
                ),
            ))
            continue
        # parse_literal returns the raw string on failure too — check for that.
        # The function never raises in practice; to be safe, we only flag on exceptions above.
    return out


def _detect_link_type_mismatch(kg: KnowledgeGraph, indexes) -> List[KGIssue]:
    """Best-effort detector: an instance edge between two individuals whose
    types aren't connected by any property with matching domain/range."""
    out: List[KGIssue] = []
    # Collect property-IRI → (domain class id, range class id) pairs.
    prop_dom_rng: Dict[str, Tuple[Set[str], Set[str]]] = {}
    for prop in _properties_in(kg):
        if not prop.iri:
            continue
        doms = {d.id for d in _domain_targets(prop, indexes) if isinstance(d, KGClass)}
        rngs = {r.id for r in _range_targets(prop, indexes) if isinstance(r, KGClass)}
        prop_dom_rng[prop.iri] = (doms, rngs)
    # For each instance edge, check.
    indiv_types: Dict[str, Set[str]] = defaultdict(set)
    for edge in kg.edges:
        if normalize_predicate(edge.iri) != RDF_TYPE:
            continue
        if isinstance(edge.source, KGIndividual) and isinstance(edge.target, KGClass):
            indiv_types[edge.source.id].add(edge.target.id)
    seen: Set[Tuple[str, str, str]] = set()
    for edge in kg.edges:
        if not isinstance(edge.source, KGIndividual) or not isinstance(edge.target, KGIndividual):
            continue
        if not edge.iri or edge.iri in (RDF_TYPE,):
            continue
        dr = prop_dom_rng.get(edge.iri)
        if dr is None:
            continue
        doms, rngs = dr
        if not (doms and rngs):
            continue
        src_ts = indiv_types.get(edge.source.id, set())
        tgt_ts = indiv_types.get(edge.target.id, set())
        if not src_ts or not tgt_ts:
            continue
        if (src_ts & doms) and (tgt_ts & rngs):
            continue
        key = (edge.iri, edge.source.id, edge.target.id)
        if key in seen:
            continue
        seen.add(key)
        out.append(KGIssue(
            id=_issue_id("LINK_TYPE_MISMATCH", edge.id),
            code="LINK_TYPE_MISMATCH",
            description=(
                f"Edge '{edge.id}' ({edge.iri}) connects individuals whose types "
                f"don't match the property's declared domain/range."
            ),
            affected_node_ids=[edge.source.id, edge.target.id],
            affected_edge_ids=[edge.id],
            recommended_action=KGAction(
                key="coerce_to_string_link",
                parameters={"edge_id": edge.id},
                label="Keep the edge as a textual annotation",
            ),
            skip_action=KGAction(
                key="drop_link",
                parameters={"edge_id": edge.id},
                label="Drop the link",
            ),
        ))
    return out


# ----------------------------------------------------------------------
# Shared detector — orphan nodes (used by both class and object analyzers)
# ----------------------------------------------------------------------


def _detect_orphan_nodes(
    kg: KnowledgeGraph, indexes
) -> Tuple[List[KGIssue], Set[str]]:
    """Find Individuals/Literals/Blanks with no transitive class-anchored link.

    A node is *class-anchored* if it can reach (undirected) a ``KGClass`` along
    edges with predicate in ``{rdf:type, rdfs:domain, rdfs:range,
    rdfs:subClassOf}``. Anything else of kind Individual / Literal / Blank is
    orphan: it cannot contribute to a class diagram and forces the LLM
    pipeline (or the user) to decide whether to drop it or classify it.

    Returns ``(issues, orphan_node_ids)`` so the caller can suppress
    overlapping detectors (e.g. ``INDIVIDUAL_NO_TYPE``).

    Orphans are grouped into one issue per disconnected component (in the
    orphan-induced subgraph, edges of any predicate). Each issue exposes:

    - ``recommended_action``: ``drop_orphan_nodes`` — batch-drop the component.
    - ``skip_action``: ``defer_to_llm_classification`` — defer to a per-node
      LLM classification step (raises a typed exception caught by the
      apply endpoint).

    Structural blank nodes (``kind in {restriction, class_expression}``) are
    excluded — they're handled by their own detectors and live as auxiliary
    schema graph fragments rather than data.
    """
    structural_kinds = {"restriction", "class_expression"}
    eligible_kinds = (KGIndividual, KGLiteral, KGBlank)

    # Step 1 — class-anchored closure via DFS over the schema-edge subgraph.
    schema_neighbors: Dict[str, Set[str]] = defaultdict(set)
    for edge in kg.edges:
        if normalize_predicate(edge.iri) not in _CLASS_ANCHOR_PREDICATES:
            continue
        schema_neighbors[edge.source.id].add(edge.target.id)
        schema_neighbors[edge.target.id].add(edge.source.id)

    seeds = {n.id for n in kg.nodes if isinstance(n, KGClass)}
    anchored: Set[str] = set()
    stack = list(seeds)
    while stack:
        nid = stack.pop()
        if nid in anchored:
            continue
        anchored.add(nid)
        stack.extend(schema_neighbors.get(nid, ()))

    # Step 2 — collect orphan candidates.
    orphan_ids: Set[str] = set()
    for node in kg.nodes:
        if not isinstance(node, eligible_kinds):
            continue
        if isinstance(node, KGBlank) and node.metadata.get("kind") in structural_kinds:
            continue
        if node.id in anchored:
            continue
        orphan_ids.add(node.id)

    if not orphan_ids:
        return [], set()

    # Step 3 — orphan-induced subgraph + connected components (any predicate).
    orphan_neighbors: Dict[str, Set[str]] = {nid: set() for nid in orphan_ids}
    for edge in kg.edges:
        s, t = edge.source.id, edge.target.id
        if s in orphan_ids and t in orphan_ids:
            orphan_neighbors[s].add(t)
            orphan_neighbors[t].add(s)

    components: List[List[str]] = []
    unvisited = set(orphan_ids)
    while unvisited:
        seed = next(iter(sorted(unvisited)))
        comp: List[str] = []
        stack = [seed]
        while stack:
            nid = stack.pop()
            if nid not in unvisited:
                continue
            unvisited.remove(nid)
            comp.append(nid)
            stack.extend(orphan_neighbors.get(nid, ()))
        components.append(sorted(comp))

    components.sort(key=lambda c: (len(c), c[0] if c else ""))

    out: List[KGIssue] = []
    for comp in components:
        node_ids = list(comp)
        n = len(node_ids)
        preview = node_ids[:ORPHAN_DESCRIPTION_TRUNCATE]
        if n > ORPHAN_DESCRIPTION_TRUNCATE:
            preview_str = ", ".join(preview) + f", … ({n - ORPHAN_DESCRIPTION_TRUNCATE} more)"
        else:
            preview_str = ", ".join(preview)
        description = (
            f"{n} node(s) have no link (direct or transitive) to any class: "
            f"{preview_str}. They will not appear in a class diagram. "
            f"Drop them, or send them to the LLM for individual classification."
        )
        out.append(KGIssue(
            id=_issue_id("ORPHAN_NODE_NO_CLASS_LINK", *node_ids),
            code="ORPHAN_NODE_NO_CLASS_LINK",
            description=description,
            affected_node_ids=node_ids,
            recommended_action=KGAction(
                key="drop_orphan_nodes",
                parameters={"node_ids": node_ids},
                label=f"Drop {n} orphan node{'s' if n != 1 else ''}",
            ),
            skip_action=KGAction(
                key="defer_to_llm_classification",
                parameters={"node_ids": node_ids},
                label=f"Send {n} node{'s' if n != 1 else ''} to the LLM for classification",
            ),
        ))
    return out, orphan_ids
