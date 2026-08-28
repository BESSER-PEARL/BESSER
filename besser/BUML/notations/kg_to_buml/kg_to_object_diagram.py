"""KG → BUML Object Diagram (ABox extraction).

Builds an :class:`ObjectModel` whose objects, slots and links mirror the
ABox of the source :class:`KnowledgeGraph`, using a class diagram derived
from the same KG (via :func:`kg_to_class_diagram`) as the typing reference.

Blank nodes are skipped entirely — no Object is created for a KGBlank,
and any edge whose source or target is a KGBlank is dropped. A single
``BLANK_SKIPPED`` warning summarises the count.

Passing ``root_individual_ids`` restricts the result to the neighbourhood of
the chosen individuals. Scoping happens on the KG *before* conversion and
touches the ABox only: every class, property and constraint node survives, so
the class diagram that types the objects stays complete while the object
diagram shrinks to the part the user asked about. See :func:`scope_abox`.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from besser.BUML.metamodel.kg import (
    KGBlank,
    KGClass,
    KGEdge,
    KGIndividual,
    KGLiteral,
    KnowledgeGraph,
)
from besser.BUML.metamodel.object import (
    AttributeLink,
    DataValue,
    Link,
    LinkEnd,
    Object,
    ObjectModel,
)
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Generalization,
    Metadata,
    Property,
    StringType,
)

from besser.BUML.notations.kg_to_buml._common import (
    KGConversionWarning,
    RDF_TYPE,
    add_warning,
    build_indexes,
    local_name,
    normalize_predicate,
    sanitize_python_identifier,
    sorted_by_id,
)
from besser.BUML.notations.kg_to_buml.datatype_mapping import parse_literal
from besser.BUML.notations.kg_to_buml.kg_to_class_diagram import (
    ClassConversionResult,
    kg_to_class_diagram,
)


@dataclass
class ObjectConversionResult:
    """Output of :func:`kg_to_object_diagram`."""

    object_model: ObjectModel
    domain_model: object  # DomainModel; typed loosely to avoid extra import
    warnings: List[KGConversionWarning] = field(default_factory=list)


def scope_abox(
    kg: KnowledgeGraph,
    root_individual_ids: List[str],
    *,
    max_depth: Optional[int] = None,
    warnings: Optional[List[KGConversionWarning]] = None,
) -> KnowledgeGraph:
    """Return a copy of ``kg`` whose ABox is limited to a neighbourhood.

    Converting a whole ontology's ABox produces an unreadable diagram once a
    dataset has more than a few dozen individuals, so the editor lets the user
    name a starting individual instead. Pruning happens here, on the KG, rather
    than by filtering the resulting :class:`ObjectModel`: the conversion then
    runs unchanged, and no dangling ``Link`` has to be cleaned up afterwards.

    Only the ABox is pruned. Every :class:`KGClass`, :class:`KGProperty` and
    constraint node is kept, along with every edge between kept non-individual
    nodes, so :func:`kg_to_class_diagram` still sees the complete TBox and the
    surviving objects keep their types.

    What is kept:

    * the roots themselves;
    * individuals reachable from a root over individual-to-individual edges,
      followed in **both** directions — an individual that points *at* the root
      is related to it just as much as one the root points at;
    * for every kept individual, its ``rdf:type`` edges and its literal
      targets, which carry the object's attribute slots. These are depth-0
      attachments and never extend the frontier.

    Args:
        kg: The graph to scope. Left untouched; a deep copy is returned.
        root_individual_ids: Ids of the individuals to start from.
        max_depth: Hops to follow. ``None`` (default) means the full connected
            component; ``1`` means direct neighbours only.
        warnings: Optional sink; an ``ABOX_SCOPED`` warning reporting how many
            individuals survived is appended when given.

    Raises:
        ValueError: if an id is absent from the graph or does not name a
            :class:`KGIndividual`. Returning an empty diagram instead would
            leave the caller unable to tell a bad id from a genuinely isolated
            individual.
    """
    nodes_by_id = {n.id: n for n in kg.nodes}

    roots: List[str] = []
    for node_id in root_individual_ids:
        node = nodes_by_id.get(node_id)
        if node is None:
            raise ValueError(f"Unknown node id {node_id!r}.")
        if not isinstance(node, KGIndividual):
            raise ValueError(
                f"Node {node_id!r} is a {type(node).__name__}, not an individual; "
                "an object diagram can only be rooted at an individual."
            )
        if node_id not in roots:
            roots.append(node_id)

    # Adjacency over individual-to-individual edges only, in both directions.
    neighbours: Dict[str, Set[str]] = defaultdict(set)
    for edge in kg.edges:
        if normalize_predicate(edge.iri) == RDF_TYPE:
            continue
        if isinstance(edge.source, KGIndividual) and isinstance(edge.target, KGIndividual):
            neighbours[edge.source.id].add(edge.target.id)
            neighbours[edge.target.id].add(edge.source.id)

    kept_individuals: Set[str] = set(roots)
    frontier: Set[str] = set(roots)
    depth = 0
    while frontier and (max_depth is None or depth < max_depth):
        nxt: Set[str] = set()
        for node_id in frontier:
            nxt |= neighbours.get(node_id, set()) - kept_individuals
        kept_individuals |= nxt
        frontier = nxt
        depth += 1

    total_individuals = sum(1 for n in kg.nodes if isinstance(n, KGIndividual))

    # Everything that is not an individual is TBox and always survives.
    keep_ids: Set[str] = {
        n.id for n in kg.nodes if not isinstance(n, KGIndividual) and not isinstance(n, KGLiteral)
    }
    keep_ids |= kept_individuals

    # Literals are kept only where they hang off a kept individual, so an
    # excluded individual does not drag its attribute values along.
    for edge in kg.edges:
        if isinstance(edge.target, KGLiteral) and edge.source.id in keep_ids:
            keep_ids.add(edge.target.id)
        if isinstance(edge.source, KGLiteral) and edge.target.id in keep_ids:
            keep_ids.add(edge.source.id)

    scoped = KnowledgeGraph(name=kg.name)
    for node in sorted_by_id(kg.nodes):
        if node.id in keep_ids:
            scoped.add_node(node)
    for edge in sorted_by_id(kg.edges):
        if edge.source.id in keep_ids and edge.target.id in keep_ids:
            scoped.add_edge(edge)
    for axiom in kg.axioms:
        scoped.add_axiom(axiom)

    if warnings is not None:
        root_labels = ", ".join(
            repr(nodes_by_id[r].label or nodes_by_id[r].iri or r) for r in roots
        )
        add_warning(
            warnings,
            "ABOX_SCOPED",
            f"{len(kept_individuals)} of {total_individuals} individual(s) included, "
            f"rooted at {root_labels}.",
        )

    return scoped


def kg_to_object_diagram(
    kg: KnowledgeGraph,
    class_result: Optional[ClassConversionResult] = None,
    *,
    model_name: Optional[str] = None,
    resolutions: Optional[List["KGResolution"]] = None,
    root_individual_ids: Optional[List[str]] = None,
    max_depth: Optional[int] = None,
) -> ObjectConversionResult:
    """Convert a :class:`KnowledgeGraph` into an :class:`ObjectModel`.

    When ``resolutions`` is provided, the same list is forwarded to
    :func:`kg_to_class_diagram` so the class diagram and object diagram
    derive from the same resolved KG. To keep them perfectly in sync, the
    caller may compute the class diagram once with resolutions and pass it
    via ``class_result``.

    Args:
        root_individual_ids: When given, only these individuals and the ones
            reachable from them are turned into Objects. ``None`` (the
            default) keeps the whole ABox.
        max_depth: How many individual-to-individual hops to follow from the
            roots. ``None`` means the full connected component. Ignored when
            ``root_individual_ids`` is ``None``.
    """
    scope_warnings: List[KGConversionWarning] = []
    if root_individual_ids is not None:
        # Scope first, so both the resolution pass and the conversion below
        # see the same graph. Only the ABox is pruned, so ``class_result``
        # (if the caller supplied one) stays valid.
        kg = scope_abox(
            kg,
            root_individual_ids,
            max_depth=max_depth,
            warnings=scope_warnings,
        )

    if resolutions and class_result is None:
        # Forward the resolutions to the class diagram pass and use the
        # same resolved KG for the object diagram pass.
        from besser.BUML.notations.kg_to_buml.resolutions import apply_resolutions
        kg = apply_resolutions(kg, resolutions)
        class_result = kg_to_class_diagram(kg)
    if class_result is None:
        class_result = kg_to_class_diagram(kg)

    warnings: List[KGConversionWarning] = list(class_result.warnings) + scope_warnings
    indexes = build_indexes(kg)

    domain_model = class_result.domain_model
    iri_to_class = class_result.iri_to_class
    property_iri_to_attribute = class_result.property_iri_to_attribute
    property_iri_to_association = class_result.property_iri_to_association
    assoc_source_end = class_result.assoc_source_end

    # Lookup helpers built lazily.
    name_to_attribute_per_class: Dict[int, Dict[str, Property]] = {}

    def _attrs_for(cls: Class) -> Dict[str, Property]:
        cache = name_to_attribute_per_class.get(id(cls))
        if cache is None:
            cache = {a.name: a for a in cls.all_attributes()}
            name_to_attribute_per_class[id(cls)] = cache
        return cache

    # ------------------------------------------------------------------
    # First pass: build Objects from KGIndividuals (skip KGBlank).
    # ------------------------------------------------------------------
    used_object_names: Set[str] = set()
    objects_by_node_id: Dict[str, Object] = {}
    synthetic_thing: Optional[Class] = None

    blank_count = sum(1 for n in kg.nodes if isinstance(n, KGBlank))
    if blank_count:
        add_warning(
            warnings,
            "BLANK_SKIPPED",
            f"{blank_count} blank node(s) were skipped; their incident edges were dropped.",
        )

    def _get_or_create_thing() -> Class:
        nonlocal synthetic_thing
        if synthetic_thing is not None:
            return synthetic_thing
        existing = next((t for t in domain_model.types if isinstance(t, Class) and t.name == "Thing"), None)
        if existing is not None:
            synthetic_thing = existing
            return existing
        synthetic_thing = Class(
            name="Thing",
            metadata=Metadata(description="Synthetic root class for KG individuals without an rdf:type"),
        )
        domain_model.add_type(synthetic_thing)
        return synthetic_thing

    individuals = sorted_by_id([n for n in kg.nodes if isinstance(n, KGIndividual)])
    for ind in individuals:
        cls = _resolve_individual_class(ind, indexes, iri_to_class, domain_model, warnings)
        if cls is None:
            cls = _get_or_create_thing()
            add_warning(
                warnings,
                "INDIVIDUAL_NO_TYPE",
                f"Individual {ind.id!r} has no resolvable class; typed as 'Thing'.",
                node_id=ind.id,
            )
        obj_name = sanitize_python_identifier(
            ind.label or local_name(ind.iri),
            f"obj_{_short_hash(ind.id)}",
            used_object_names,
        )
        obj = Object(name=obj_name, classifier=cls, slots=[])
        objects_by_node_id[ind.id] = obj

    # ------------------------------------------------------------------
    # Second pass: slots & links.
    # ------------------------------------------------------------------
    for edge in sorted_by_id(kg.edges):
        if isinstance(edge.source, KGBlank) or isinstance(edge.target, KGBlank):
            continue
        if normalize_predicate(edge.iri) == RDF_TYPE:
            continue

        src_obj = objects_by_node_id.get(edge.source.id)
        if src_obj is None:
            # Edges from declared classes / properties (TBox) are ignored here.
            continue

        if isinstance(edge.target, KGLiteral):
            _add_literal_slot(
                src_obj=src_obj,
                edge=edge,
                literal=edge.target,
                property_iri_to_attribute=property_iri_to_attribute,
                attrs_for=_attrs_for,
                warnings=warnings,
            )
        elif isinstance(edge.target, KGIndividual):
            tgt_obj = objects_by_node_id.get(edge.target.id)
            if tgt_obj is None:
                continue
            _add_link(
                src_obj=src_obj,
                tgt_obj=tgt_obj,
                edge=edge,
                property_iri_to_association=property_iri_to_association,
                assoc_source_end=assoc_source_end,
                warnings=warnings,
            )
        else:
            # KGClass / KGProperty target on an instance edge — uncommon; ignore quietly.
            add_warning(
                warnings,
                "UNKNOWN_PREDICATE",
                f"Edge {edge.id!r} from {src_obj.name_!r} targets a {type(edge.target).__name__}; ignored.",
                edge_id=edge.id,
            )

    # ------------------------------------------------------------------
    # Assemble the ObjectModel.
    # ------------------------------------------------------------------
    raw_name = model_name or kg.name or "KGObjectDiagram"
    safe_name = sanitize_python_identifier(raw_name, "KGObjectDiagram", set())
    om = ObjectModel(name=safe_name, objects=set(objects_by_node_id.values()))

    return ObjectConversionResult(object_model=om, domain_model=domain_model, warnings=warnings)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _resolve_individual_class(
    ind: KGIndividual,
    indexes,
    iri_to_class: Dict[str, Class],
    domain_model,
    warnings: List[KGConversionWarning],
) -> Optional[Class]:
    """Pick the most specific class declared via ``rdf:type`` for an individual."""
    type_edges = indexes.out_with_predicate(ind.id, RDF_TYPE)
    candidates: List[Class] = []
    for edge in type_edges:
        if isinstance(edge.target, KGBlank):
            continue
        cls: Optional[Class] = None
        target_iri = getattr(edge.target, "iri", None)
        if target_iri:
            cls = iri_to_class.get(target_iri)
        if cls is None and isinstance(edge.target, KGClass):
            # Fallback: match by node id via the domain model.
            target_label = edge.target.label or local_name(edge.target.iri)
            cls = next(
                (c for c in domain_model.types if isinstance(c, Class) and c.name == target_label),
                None,
            )
        if cls is not None and cls not in candidates:
            candidates.append(cls)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Pick the most specific: a class with no descendant in the candidate set.
    descendant_map = _descendant_map(domain_model)
    most_specific = [c for c in candidates if not (descendant_map.get(id(c), set()) & {id(o) for o in candidates})]
    if len(most_specific) == 1:
        return most_specific[0]

    add_warning(
        warnings,
        "MULTIPLE_TYPES",
        f"Individual {ind.id!r} has multiple incomparable rdf:type values "
        f"({[c.name for c in candidates]}); picked '{most_specific[0].name if most_specific else candidates[0].name}'.",
        node_id=ind.id,
    )
    return (most_specific or candidates)[0]


def _descendant_map(domain_model) -> Dict[int, Set[int]]:
    """Build {id(class): {ids of its transitive descendants}} from the model's generalizations."""
    children_of: Dict[int, Set[int]] = defaultdict(set)
    for gen in getattr(domain_model, "generalizations", set()):
        if isinstance(gen, Generalization):
            children_of[id(gen.general)].add(id(gen.specific))

    cache: Dict[int, Set[int]] = {}

    def resolve(cls_id: int) -> Set[int]:
        if cls_id in cache:
            return cache[cls_id]
        descendants: Set[int] = set()
        for child_id in children_of.get(cls_id, set()):
            descendants.add(child_id)
            descendants.update(resolve(child_id))
        cache[cls_id] = descendants
        return descendants

    return {id(t): resolve(id(t)) for t in getattr(domain_model, "types", set()) if isinstance(t, Class)}


def _add_literal_slot(
    *,
    src_obj: Object,
    edge: KGEdge,
    literal: KGLiteral,
    property_iri_to_attribute: Dict[str, Property],
    attrs_for,
    warnings: List[KGConversionWarning],
) -> None:
    cls = src_obj.classifier
    attr: Optional[Property] = None
    if edge.iri and edge.iri in property_iri_to_attribute:
        attr = property_iri_to_attribute[edge.iri]
        if attr.owner is not cls and attr.owner not in (cls.all_parents() if hasattr(cls, "all_parents") else set()):
            attr = None  # not a property of this class; fall through to name match
    if attr is None:
        # Name-based fallback (use property local name or label).
        candidate_name = sanitize_python_identifier(
            edge.label or local_name(edge.iri),
            "prop",
            set(),
        )
        attr = attrs_for(cls).get(candidate_name)
    if attr is None:
        add_warning(
            warnings,
            "UNKNOWN_PREDICATE",
            f"Predicate {edge.iri or edge.label!r} on object '{src_obj.name_}' does not match any "
            f"attribute of class '{cls.name}'; literal slot dropped.",
            edge_id=edge.id,
        )
        return

    parsed = parse_literal(literal.value, literal.datatype)
    try:
        data_value = DataValue(classifier=attr.type, value=parsed)
        slot = AttributeLink(value=data_value, attribute=attr)
    except (TypeError, ValueError):
        # Type mismatch — coerce to string.
        data_value = DataValue(classifier=StringType, value=str(literal.value))
        # If the attribute doesn't accept StringType either, skip with a warning.
        try:
            # Re-create with a string-typed attribute by constructing manually:
            slot = AttributeLink.__new__(AttributeLink)
            object.__setattr__(slot, "_AttributeLink__attribute", attr)
            object.__setattr__(slot, "_AttributeLink__value", data_value)
        except Exception:
            add_warning(
                warnings,
                "LITERAL_TYPE_COERCED",
                f"Could not store literal value {literal.value!r} on attribute '{attr.name}' of "
                f"object '{src_obj.name_}' even after string coercion.",
                edge_id=edge.id,
            )
            return
        add_warning(
            warnings,
            "LITERAL_TYPE_COERCED",
            f"Literal value {literal.value!r} did not match the declared type of attribute "
            f"'{attr.name}' ({attr.type.name}); stored as a raw string.",
            edge_id=edge.id,
        )

    src_obj.add_slot(slot)


def _add_link(
    *,
    src_obj: Object,
    tgt_obj: Object,
    edge: KGEdge,
    property_iri_to_association: Dict[str, BinaryAssociation],
    assoc_source_end: Dict[int, Property],
    warnings: List[KGConversionWarning],
) -> None:
    src_cls = src_obj.classifier
    tgt_cls = tgt_obj.classifier
    assoc: Optional[BinaryAssociation] = None
    if edge.iri and edge.iri in property_iri_to_association:
        assoc = property_iri_to_association[edge.iri]

    if assoc is None:
        # Fallback: search by name.
        candidate_name = sanitize_python_identifier(
            edge.label or local_name(edge.iri),
            "assoc",
            set(),
        )
        for ae in src_cls.all_association_ends():
            owner = ae.owner
            if owner.name == candidate_name and isinstance(owner, BinaryAssociation):
                assoc = owner
                break

    if assoc is None:
        add_warning(
            warnings,
            "UNKNOWN_PREDICATE",
            f"Edge {edge.id!r} ({edge.label or edge.iri or '<unknown>'}) on object '{src_obj.name_}' "
            f"does not match any association from class '{src_cls.name}'; link dropped.",
            edge_id=edge.id,
        )
        return

    ends = list(assoc.ends)
    # Prefer the source-end recorded by the class converter — this is
    # essential for self-referential associations where both ends share
    # the same type and would otherwise be indistinguishable.
    recorded_source = assoc_source_end.get(id(assoc))
    if recorded_source is not None and recorded_source in ends and _type_matches(src_cls, recorded_source.type):
        src_end = recorded_source
        tgt_end = next((e for e in ends if e is not src_end), None)
    else:
        src_end = next((e for e in ends if _type_matches(src_cls, e.type)), None)
        tgt_end = next((e for e in ends if e is not src_end and _type_matches(tgt_cls, e.type)), None)

    if src_end is None or tgt_end is None or not _type_matches(tgt_cls, tgt_end.type):
        add_warning(
            warnings,
            "LINK_TYPE_MISMATCH",
            f"Could not match association ends of '{assoc.name}' for objects "
            f"'{src_obj.name_}'({src_cls.name}) and '{tgt_obj.name_}'({tgt_cls.name}); link dropped.",
            edge_id=edge.id,
        )
        return

    Link(
        name=f"{src_obj.name_}_to_{tgt_obj.name_}",
        association=assoc,
        connections=[
            LinkEnd(name=src_end.name, association_end=src_end, object=src_obj),
            LinkEnd(name=tgt_end.name, association_end=tgt_end, object=tgt_obj),
        ],
    )


def _type_matches(actual: Class, expected) -> bool:
    if actual is expected:
        return True
    if hasattr(actual, "all_parents") and expected in actual.all_parents():
        return True
    return False


__all__ = ["ObjectConversionResult", "kg_to_object_diagram", "scope_abox"]
