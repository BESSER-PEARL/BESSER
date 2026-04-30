"""KG → BUML Class Diagram (TBox extraction).

``kg_to_class_diagram(kg)`` walks a :class:`KnowledgeGraph` and produces a
:class:`DomainModel` whose:

* classes correspond to :class:`KGClass` nodes (or are synthesised from
  references when a class isn't explicitly declared),
* attributes correspond to :class:`KGProperty` nodes whose ``rdfs:range``
  points to an XSD datatype,
* binary associations correspond to :class:`KGProperty` nodes whose
  ``rdfs:range`` points to another class,
* generalizations correspond to ``rdfs:subClassOf`` edges.

After the schema-based pass, the ABox is scanned: any datatype property
that any individual uses with more than one literal value gets its
multiplicity bumped to ``0..*`` so the class diagram stays consistent
with the object diagram derived from the same KG.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KGNode,
    KGProperty,
    KnowledgeGraph,
)
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    DomainModel,
    Generalization,
    Metadata,
    Multiplicity,
    Property,
    StringType,
    UNLIMITED_MAX_MULTIPLICITY,
)

from besser.BUML.notations.kg_to_buml._common import (
    KGConversionWarning,
    RDF_TYPE,
    RDFS_DOMAIN,
    RDFS_RANGE,
    RDFS_SUBCLASS_OF,
    add_warning,
    build_indexes,
    is_meta_vocab,
    local_name,
    normalize_predicate,
    sanitize_python_identifier,
    sorted_by_id,
)
from besser.BUML.notations.kg_to_buml.datatype_mapping import xsd_to_primitive


_XSD_NAMESPACE = "http://www.w3.org/2001/XMLSchema#"
_RDF_NAMESPACE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDFS_NAMESPACE = "http://www.w3.org/2000/01/rdf-schema#"


@dataclass
class ClassConversionResult:
    """Output of :func:`kg_to_class_diagram`."""

    domain_model: DomainModel
    iri_to_class: Dict[str, Class] = field(default_factory=dict)
    property_iri_to_attribute: Dict[str, Property] = field(default_factory=dict)
    property_iri_to_association: Dict[str, BinaryAssociation] = field(default_factory=dict)
    # For each derived BinaryAssociation, the Property end that represents the
    # *source* (KG triple subject) side. Important for self-referential
    # associations where both ends share the same type and would otherwise be
    # indistinguishable.
    assoc_source_end: Dict[int, Property] = field(default_factory=dict)
    warnings: List[KGConversionWarning] = field(default_factory=list)


def _looks_like_datatype_iri(iri: Optional[str]) -> bool:
    if not iri:
        return False
    return iri.startswith(_XSD_NAMESPACE) or iri in {
        f"{_RDF_NAMESPACE}langString",
        f"{_RDFS_NAMESPACE}Literal",
        f"{_RDF_NAMESPACE}PlainLiteral",
    }


def _model_name(kg: KnowledgeGraph, override: Optional[str]) -> str:
    raw = override or kg.name or "KGClassDiagram"
    safe = sanitize_python_identifier(raw, "KGClassDiagram", set())
    return safe


def kg_to_class_diagram(
    kg: KnowledgeGraph,
    *,
    model_name: Optional[str] = None,
    resolutions: Optional[List["KGResolution"]] = None,
) -> ClassConversionResult:
    """Convert a :class:`KnowledgeGraph` into a :class:`DomainModel` (TBox).

    Args:
        kg: Source KG.
        model_name: Override for the resulting :class:`DomainModel.name`.
        resolutions: Optional list of user choices from a preflight report
            (see :mod:`besser.BUML.notations.kg_to_buml.preflight`). When
            provided, :func:`apply_resolutions` rewrites a deep-copied
            KG before conversion (so the input is untouched).
    """
    if resolutions:
        # Lazy import to avoid a top-of-module cycle through the package's
        # ``__init__.py`` re-exports.
        from besser.BUML.notations.kg_to_buml.resolutions import apply_resolutions
        kg = apply_resolutions(kg, resolutions)
    indexes = build_indexes(kg)
    warnings: List[KGConversionWarning] = []

    used_class_names: Set[str] = set()
    iri_to_class: Dict[str, Class] = {}
    id_to_class: Dict[str, Class] = {}

    # ------------------------------------------------------------------
    # Step 1: declared classes (KGClass) → Class. Skip nodes whose IRI is
    # a known datatype IRI — the OWL parser falls back to KGClass for any
    # untyped URIRef, which means XSD datatype IRIs would otherwise leak
    # into the class diagram as empty classes.
    # ------------------------------------------------------------------
    for node in sorted_by_id([n for n in kg.nodes if isinstance(n, KGClass)]):
        if _looks_like_datatype_iri(getattr(node, "iri", None)):
            continue
        if is_meta_vocab(getattr(node, "iri", None)):
            # OWL/RDFS framework terms (owl:Class, rdfs:Class, …) are not user
            # concepts; the importer occasionally classifies them as KGClass
            # when an OWL2 punning ontology declares them. Skip them so they
            # don't pollute the BUML output.
            continue
        cls = _build_class(node, used_class_names)
        id_to_class[node.id] = cls
        if node.iri:
            iri_to_class.setdefault(node.iri, cls)

    # ------------------------------------------------------------------
    # Step 2: synthesise classes from references (rdf:type targets,
    # rdfs:domain/range targets pointing to non-class nodes,
    # rdfs:subClassOf endpoints) — but never from KGBlank/KGLiteral nodes
    # and never from OWL/RDFS framework vocabulary IRIs (owl:Class,
    # owl:DatatypeProperty, …) which would otherwise leak into the
    # diagram as spurious "Class", "DatatypeProperty" boxes.
    # ------------------------------------------------------------------
    def _ensure_class_for_node(node: KGNode) -> Optional[Class]:
        if isinstance(node, (KGBlank, KGLiteral, KGProperty)):
            return None
        if is_meta_vocab(getattr(node, "iri", None)):
            return None
        existing = id_to_class.get(node.id)
        if existing is not None:
            return existing
        cls = _build_class(node, used_class_names, synthesised=True)
        id_to_class[node.id] = cls
        if node.iri:
            iri_to_class.setdefault(node.iri, cls)
        add_warning(
            warnings,
            "UNDECLARED_CLASS",
            f"Node {node.id!r} ({type(node).__name__}) was referenced as a class but never declared as KGClass; "
            f"a class '{cls.name}' has been synthesised.",
            node_id=node.id,
        )
        return cls

    for edge in kg.edges:
        pred = normalize_predicate(edge.iri)
        if pred == RDF_TYPE and not isinstance(edge.target, (KGBlank, KGLiteral, KGProperty)):
            _ensure_class_for_node(edge.target)
        elif pred in (RDFS_SUBCLASS_OF,):
            _ensure_class_for_node(edge.source)
            _ensure_class_for_node(edge.target)
        elif pred == RDFS_DOMAIN:
            _ensure_class_for_node(edge.target)
        elif pred == RDFS_RANGE:
            tgt = edge.target
            if not isinstance(tgt, (KGBlank, KGLiteral, KGProperty)) and not _looks_like_datatype_iri(getattr(tgt, "iri", None)):
                _ensure_class_for_node(tgt)

    # ------------------------------------------------------------------
    # Step 3: generalizations from rdfs:subClassOf edges (cycle-safe).
    # ------------------------------------------------------------------
    generalizations: Set[Generalization] = set()
    parents_of: Dict[str, Set[str]] = defaultdict(set)
    for edge in sorted_by_id(kg.edges):
        if normalize_predicate(edge.iri) != RDFS_SUBCLASS_OF:
            continue
        if isinstance(edge.source, KGBlank) or isinstance(edge.target, KGBlank):
            continue
        child_cls = id_to_class.get(edge.source.id)
        parent_cls = id_to_class.get(edge.target.id)
        if child_cls is None or parent_cls is None or child_cls is parent_cls:
            continue
        if _would_create_cycle(child_cls, parent_cls, parents_of, id_to_class):
            add_warning(
                warnings,
                "CYCLIC_SUBCLASS",
                f"rdfs:subClassOf edge {edge.id!r} would introduce a cycle "
                f"({child_cls.name} → {parent_cls.name}); dropped.",
                edge_id=edge.id,
            )
            continue
        try:
            gen = Generalization(general=parent_cls, specific=child_cls)
        except ValueError as exc:  # pragma: no cover - defensive
            add_warning(
                warnings,
                "INVALID_GENERALIZATION",
                f"Could not create generalization {child_cls.name} → {parent_cls.name}: {exc}",
                edge_id=edge.id,
            )
            continue
        generalizations.add(gen)
        parents_of[edge.source.id].add(edge.target.id)

    # ------------------------------------------------------------------
    # Step 4: KGProperty nodes → attributes / associations.
    # ------------------------------------------------------------------
    used_assoc_names: Set[str] = set()
    associations: Set[BinaryAssociation] = set()
    property_iri_to_attribute: Dict[str, Property] = {}
    property_iri_to_association: Dict[str, BinaryAssociation] = {}
    assoc_source_end: Dict[int, Property] = {}
    synthetic_thing: Optional[Class] = None

    def _get_or_create_thing() -> Class:
        nonlocal synthetic_thing
        if synthetic_thing is None:
            synthetic_thing = _build_synthetic_thing(used_class_names)
        return synthetic_thing

    for prop_node in sorted_by_id([n for n in kg.nodes if isinstance(n, KGProperty)]):
        domain_targets = [
            e.target for e in indexes.out_with_predicate(prop_node.id, RDFS_DOMAIN)
            if not isinstance(e.target, (KGBlank, KGLiteral, KGProperty))
        ]
        range_targets = [e.target for e in indexes.out_with_predicate(prop_node.id, RDFS_RANGE)]

        prop_label = prop_node.label or local_name(prop_node.iri)
        prop_name_base = sanitize_python_identifier(prop_label, "prop", set())

        # Determine owner class(es).
        if not domain_targets:
            owner = _get_or_create_thing()
            add_warning(
                warnings,
                "PROPERTY_NO_DOMAIN",
                f"Property '{prop_name_base}' has no rdfs:domain; assigned to synthetic class 'Thing'.",
                node_id=prop_node.id,
            )
            owners: List[Class] = [owner]
        else:
            owners = []
            for dom_node in domain_targets:
                owner = id_to_class.get(dom_node.id) or _ensure_class_for_node(dom_node)
                if owner is not None:
                    owners.append(owner)
            if len(domain_targets) > 1:
                add_warning(
                    warnings,
                    "MULTIPLE_DOMAINS",
                    f"Property '{prop_name_base}' has multiple rdfs:domain values; "
                    f"property/association will be added to the first ({owners[0].name if owners else 'n/a'}) only.",
                    node_id=prop_node.id,
                )
            owners = owners[:1]

        if not range_targets:
            # Treat as datatype string attribute.
            for owner in owners:
                _add_datatype_attribute(
                    owner, prop_node, prop_name_base, StringType,
                    multiplicity=Multiplicity(0, 1),
                    property_iri_to_attribute=property_iri_to_attribute,
                    used_attr_names_per_owner={},
                    warnings=warnings,
                )
            add_warning(
                warnings,
                "PROPERTY_NO_RANGE",
                f"Property '{prop_name_base}' has no rdfs:range; treated as a string attribute.",
                node_id=prop_node.id,
            )
            continue

        used_attr_names_per_owner: Dict[int, Set[str]] = defaultdict(
            lambda: set()
        )
        # Pre-seed with already-existing attribute names per owner so we
        # detect collisions deterministically.
        for owner in owners:
            used_attr_names_per_owner[id(owner)] = {a.name for a in owner.attributes}

        for rng in range_targets:
            rng_iri = getattr(rng, "iri", None)
            # Datatype IRIs (xsd:*, rdf:langString, rdfs:Literal) always
            # become datatype attributes — even when the parser classified
            # the node as a KGClass (the OWL parser falls back to KGClass
            # for any URIRef that isn't otherwise typed).
            if _looks_like_datatype_iri(rng_iri) or isinstance(rng, KGLiteral):
                primitive, known = xsd_to_primitive(rng_iri)
                if not known and rng_iri:
                    add_warning(
                        warnings,
                        "UNMAPPED_DATATYPE",
                        f"Datatype IRI {rng_iri!r} on property '{prop_name_base}' is not in the XSD mapping; "
                        f"falling back to string.",
                        node_id=prop_node.id,
                    )
                for owner in owners:
                    _add_datatype_attribute(
                        owner, prop_node, prop_name_base, primitive,
                        multiplicity=Multiplicity(0, 1),
                        property_iri_to_attribute=property_iri_to_attribute,
                        used_attr_names_per_owner=used_attr_names_per_owner,
                        warnings=warnings,
                    )
                continue
            if isinstance(rng, KGClass) or (rng_iri and not isinstance(rng, (KGLiteral, KGBlank))):
                # Object property → BinaryAssociation
                target_cls = id_to_class.get(rng.id) or _ensure_class_for_node(rng)
                if target_cls is None:
                    continue
                for owner in owners:
                    result = _build_binary_association(
                        owner=owner,
                        target=target_cls,
                        prop_node=prop_node,
                        prop_name_base=prop_name_base,
                        used_assoc_names=used_assoc_names,
                    )
                    if result is not None:
                        assoc, source_end = result
                        associations.add(assoc)
                        assoc_source_end[id(assoc)] = source_end
                        if prop_node.iri:
                            property_iri_to_association.setdefault(prop_node.iri, assoc)
                continue
            # Anomalous: range points to an individual or blank; treat as string attribute.
            add_warning(
                warnings,
                "RANGE_NOT_TYPE",
                f"Range of property '{prop_name_base}' is a {type(rng).__name__}; "
                f"treated as a string attribute.",
                node_id=prop_node.id,
            )
            for owner in owners:
                _add_datatype_attribute(
                    owner, prop_node, prop_name_base, StringType,
                    multiplicity=Multiplicity(0, 1),
                    property_iri_to_attribute=property_iri_to_attribute,
                    used_attr_names_per_owner=used_attr_names_per_owner,
                    warnings=warnings,
                )

    # ------------------------------------------------------------------
    # Step 4.5: lift OWL restrictions and property characteristics into
    # BUML Multiplicity. owl:FunctionalProperty caps max=1; owl:Restriction
    # blank nodes attached via rdfs:subClassOf or owl:equivalentClass refine
    # the multiplicity of the matching attribute or association end.
    # ------------------------------------------------------------------
    explicit_multiplicity_props: Set[str] = _apply_restrictions_and_characteristics(
        kg=kg,
        id_to_class=id_to_class,
        property_iri_to_attribute=property_iri_to_attribute,
        property_iri_to_association=property_iri_to_association,
        assoc_source_end=assoc_source_end,
        warnings=warnings,
    )

    # ------------------------------------------------------------------
    # Step 5: ABox-driven multiplicity bump for datatype properties.
    # ------------------------------------------------------------------
    if property_iri_to_attribute:
        # Count, for each (individual, predicate IRI, target=KGLiteral),
        # how many literal targets exist. Skip blanks.
        usage: Dict[Tuple[str, str], int] = defaultdict(int)
        for edge in kg.edges:
            if not edge.iri or edge.iri not in property_iri_to_attribute:
                continue
            if isinstance(edge.source, KGBlank) or isinstance(edge.target, KGBlank):
                continue
            if isinstance(edge.target, KGLiteral) and isinstance(edge.source, KGIndividual):
                usage[(edge.source.id, edge.iri)] += 1

        max_per_property: Dict[str, int] = defaultdict(int)
        for (_indiv_id, prop_iri), count in usage.items():
            if count > max_per_property[prop_iri]:
                max_per_property[prop_iri] = count

        for prop_iri, max_count in max_per_property.items():
            if max_count <= 1:
                continue
            attr = property_iri_to_attribute.get(prop_iri)
            if attr is None:
                continue
            if prop_iri in explicit_multiplicity_props:
                # An explicit OWL restriction (or FunctionalProperty) already
                # set the multiplicity; don't override it from ABox usage.
                continue
            attr.multiplicity = Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY)
            add_warning(
                warnings,
                "MULTIVALUED_LITERAL",
                f"Property '{attr.name}' on '{attr.owner.name if attr.owner else '?'}' has multi-valued instance "
                f"data ({max_count} literals on a single individual); multiplicity bumped to 0..*.",
            )

    # ------------------------------------------------------------------
    # Assemble the DomainModel.
    # ------------------------------------------------------------------
    types: Set[Class] = set(id_to_class.values())
    if synthetic_thing is not None:
        types.add(synthetic_thing)

    domain_model = DomainModel(
        name=_model_name(kg, model_name),
        types=types,
        associations=associations,
        generalizations=generalizations,
    )

    return ClassConversionResult(
        domain_model=domain_model,
        iri_to_class=iri_to_class,
        property_iri_to_attribute=property_iri_to_attribute,
        property_iri_to_association=property_iri_to_association,
        assoc_source_end=assoc_source_end,
        warnings=warnings,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_class(node: KGNode, used_names: Set[str], *, synthesised: bool = False) -> Class:
    raw_label = node.label or local_name(getattr(node, "iri", None)) or "Class"
    name = sanitize_python_identifier(raw_label, "Class", used_names)
    metadata: Optional[Metadata] = None
    iri = getattr(node, "iri", None)
    if iri:
        metadata = Metadata(uri=iri)
    cls = Class(name=name, metadata=metadata)
    if synthesised:
        # Mark the synthetic class with a hint in its metadata description so
        # downstream consumers can distinguish it from declared classes.
        cls.metadata = Metadata(uri=iri, description="Synthesised from KG reference")
    return cls


def _build_synthetic_thing(used_names: Set[str]) -> Class:
    name = sanitize_python_identifier("Thing", "Thing", used_names)
    return Class(name=name, metadata=Metadata(description="Synthetic root class for KG nodes without an explicit type"))


def _add_datatype_attribute(
    owner: Class,
    prop_node: KGProperty,
    prop_name_base: str,
    primitive,
    *,
    multiplicity: Multiplicity,
    property_iri_to_attribute: Dict[str, Property],
    used_attr_names_per_owner: Dict[int, Set[str]],
    warnings: List[KGConversionWarning],
) -> None:
    used_for_owner = used_attr_names_per_owner.setdefault(id(owner), {a.name for a in owner.attributes})
    name = sanitize_python_identifier(prop_name_base, "prop", used_for_owner)
    if name != prop_name_base:
        add_warning(
            warnings,
            "PROPERTY_NAME_COLLISION",
            f"Property name '{prop_name_base}' collides on class '{owner.name}'; renamed to '{name}'.",
            node_id=prop_node.id,
        )
    attr = Property(
        name=name,
        type=primitive,
        multiplicity=multiplicity,
        is_optional=multiplicity.min == 0,
        metadata=Metadata(uri=prop_node.iri) if prop_node.iri else None,
    )
    owner.add_attribute(attr)
    if prop_node.iri:
        property_iri_to_attribute.setdefault(prop_node.iri, attr)


def _build_binary_association(
    *,
    owner: Class,
    target: Class,
    prop_node: KGProperty,
    prop_name_base: str,
    used_assoc_names: Set[str],
) -> Optional[Tuple[BinaryAssociation, Property]]:
    assoc_name = sanitize_python_identifier(prop_name_base, "assoc", used_assoc_names)

    target_end_name = _unique_end_name(target, prop_name_base, prefer="prop")
    source_base = (owner.name[:1].lower() + owner.name[1:]) if owner.name else "source"
    if owner is target:
        # Self-referential association: avoid colliding with the target end name.
        source_base = f"source_{source_base}"
    source_end_name = _unique_end_name(owner, source_base, prefer="src")

    target_end = Property(
        name=target_end_name,
        type=target,
        multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY),
    )
    source_end = Property(
        name=source_end_name,
        type=owner,
        multiplicity=Multiplicity(0, UNLIMITED_MAX_MULTIPLICITY),
    )
    try:
        assoc = BinaryAssociation(
            name=assoc_name,
            ends={source_end, target_end},
            metadata=Metadata(uri=prop_node.iri) if prop_node.iri else None,
        )
    except ValueError:
        return None
    return assoc, source_end


def _unique_end_name(owner: Class, base: str, *, prefer: str) -> str:
    used = {e.name for e in owner.association_ends()} | {a.name for a in owner.attributes}
    return sanitize_python_identifier(base, prefer, used)


def _would_create_cycle(
    child: Class,
    parent: Class,
    parents_of: Dict[str, Set[str]],
    id_to_class: Dict[str, Class],
) -> bool:
    """Return True if adding (child → parent) would create a cycle in the
    directed subclass graph."""
    parent_id = next((nid for nid, c in id_to_class.items() if c is parent), None)
    child_id = next((nid for nid, c in id_to_class.items() if c is child), None)
    if parent_id is None or child_id is None:
        return False
    if parent_id == child_id:
        return True
    stack = [parent_id]
    seen: Set[str] = set()
    while stack:
        node = stack.pop()
        if node == child_id:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(parents_of.get(node, ()))
    return False


_OWL_RESTRICTION_LINK_PREDICATES = {
    RDFS_SUBCLASS_OF,
    "http://www.w3.org/2002/07/owl#equivalentClass",
}


def _apply_restrictions_and_characteristics(
    *,
    kg: KnowledgeGraph,
    id_to_class: Dict[str, Class],
    property_iri_to_attribute: Dict[str, Property],
    property_iri_to_association: Dict[str, BinaryAssociation],
    assoc_source_end: Dict[int, Property],
    warnings: List[KGConversionWarning],
) -> Set[str]:
    """Lift OWL property characteristics + ``owl:Restriction`` payloads into
    BUML ``Multiplicity`` on attributes and association target ends.

    Returns the set of property IRIs whose multiplicity was set explicitly
    (so ABox-driven heuristics in Step 5 can avoid overriding them).
    """
    explicit: Set[str] = set()

    # 1. Property characteristics on KGProperty.metadata
    #    Functional → max=1.
    for node in kg.nodes:
        if not isinstance(node, KGProperty):
            continue
        chars = node.metadata.get("characteristics") if node.metadata else None
        if not chars or "Functional" not in chars:
            continue
        prop_iri = node.iri or node.id
        attr = property_iri_to_attribute.get(prop_iri)
        if attr is not None:
            new_min = attr.multiplicity.min if attr.multiplicity is not None else 0
            attr.multiplicity = Multiplicity(min(new_min, 1), 1)
            attr.is_optional = attr.multiplicity.min == 0
            explicit.add(prop_iri)
            continue
        assoc = property_iri_to_association.get(prop_iri)
        if assoc is not None:
            target_end = _target_end_of(assoc, assoc_source_end)
            if target_end is not None:
                tmin = target_end.multiplicity.min if target_end.multiplicity is not None else 0
                target_end.multiplicity = Multiplicity(min(tmin, 1), 1)
                explicit.add(prop_iri)

    # 2. owl:Restriction payloads on KGBlank nodes attached via rdfs:subClassOf
    #    or owl:equivalentClass to an owner KGClass.
    #    Group restrictions by (owner_class_id, on_property_iri).
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for edge in kg.edges:
        pred = normalize_predicate(edge.iri)
        if pred not in _OWL_RESTRICTION_LINK_PREDICATES:
            continue
        if not isinstance(edge.target, KGBlank):
            continue
        if edge.target.metadata.get("kind") != "restriction":
            continue
        owner_cls = id_to_class.get(edge.source.id)
        if owner_cls is None:
            continue
        on_prop = edge.target.metadata.get("on_property")
        if not on_prop:
            continue
        grouped[(edge.source.id, on_prop)].append(edge.target.metadata)

    for (owner_id, prop_iri), payloads in grouped.items():
        attr = property_iri_to_attribute.get(prop_iri)
        assoc = property_iri_to_association.get(prop_iri)
        target_end: Optional[Property] = (
            _target_end_of(assoc, assoc_source_end) if assoc is not None else None
        )
        if attr is None and target_end is None:
            # Restriction on a property we didn't lift — skip silently; the
            # blank metadata is preserved on the KG and surfaced by preflight.
            continue

        # Combine OWL restriction values across all payloads for this
        # (owner_class, property) pair. Multiple ``minCardinality`` axioms
        # combine via max() (most restrictive lower bound); multiple
        # ``maxCardinality`` axioms combine via min(). Default (BUML)
        # multiplicity is *not* used as a constraint — an explicit OWL
        # restriction overrides the default.
        explicit_min: Optional[int] = None
        explicit_max: Optional[int] = None
        owner_name = id_to_class[owner_id].name if owner_id in id_to_class else "?"
        for payload in payloads:
            kind = payload.get("restriction_type")
            value = payload.get("value")
            if kind in ("cardinality", "qualifiedCardinality") and isinstance(value, int):
                explicit_min = value if explicit_min is None else max(explicit_min, value)
                explicit_max = value if explicit_max is None else min(explicit_max, value)
            elif kind in ("minCardinality", "minQualifiedCardinality") and isinstance(value, int):
                explicit_min = value if explicit_min is None else max(explicit_min, value)
            elif kind in ("maxCardinality", "maxQualifiedCardinality") and isinstance(value, int):
                explicit_max = value if explicit_max is None else min(explicit_max, value)
            elif kind == "someValuesFrom":
                explicit_min = 1 if explicit_min is None else max(explicit_min, 1)
            else:
                add_warning(
                    warnings,
                    "ADV_RESTRICTION_UNSUPPORTED",
                    f"Restriction '{kind}' on property '{prop_iri}' (class '{owner_name}') has no clean BUML mapping; "
                    f"left unmodelled.",
                )

        if explicit_min is None and explicit_max is None:
            continue  # only unsupported payloads → already warned

        current = attr.multiplicity if attr is not None else target_end.multiplicity
        cur_min = current.min if current is not None else 0
        cur_max = current.max if current is not None else UNLIMITED_MAX_MULTIPLICITY
        new_min = explicit_min if explicit_min is not None else cur_min
        new_max = explicit_max if explicit_max is not None else cur_max
        if new_max != UNLIMITED_MAX_MULTIPLICITY and new_min > new_max:
            add_warning(
                warnings,
                "ADV_RESTRICTION_UNSUPPORTED",
                f"Restrictions on property '{prop_iri}' (class '{owner_name}') yield contradictory multiplicity "
                f"({new_min}..{new_max}); ignored.",
            )
            continue
        new_mult = Multiplicity(new_min, new_max)
        if attr is not None:
            attr.multiplicity = new_mult
            attr.is_optional = new_min == 0
        elif target_end is not None:
            target_end.multiplicity = new_mult
        explicit.add(prop_iri)

    return explicit


def _target_end_of(
    assoc: Optional[BinaryAssociation],
    assoc_source_end: Dict[int, Property],
) -> Optional[Property]:
    """Return the *target* end of a binary association (the end opposite the
    recorded source end). Returns None if either input is missing."""
    if assoc is None:
        return None
    src_end = assoc_source_end.get(id(assoc))
    if src_end is None:
        return None
    for end in assoc.ends:
        if end is not src_end:
            return end
    return None


__all__ = ["ClassConversionResult", "kg_to_class_diagram"]
