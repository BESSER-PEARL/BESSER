"""Lower the intermediate :class:`UMLModel` onto a BUML ``DomainModel``.

The paper's rules are expressed against plain UML, which is more permissive
than BUML's metamodel. This module is where every BUML-specific concession
lives, so the ported rules in
:mod:`besser.BUML.notations.kg_to_buml.owl2uml` stay faithful to the paper.

The concessions, and why each is forced:

* **Association end names must be unique across a class, all its ancestors and
  all its descendants** (``Class._validate_unique_end_names``). Meanwhile OCL
  navigation ``self.<p>`` resolves through the *opposite* end's name, so
  renaming an end silently breaks every invariant that mentions it. Colliding
  associations are therefore **merged onto an auxiliary union class** rather
  than renamed — see :func:`_merge_fanout`.
* **Only nine primitives exist** and they cannot be extended, so materialised
  data ranges (O01-O05) become ordinary classes carrying a ``value``
  attribute. ``Property.owner`` outright rejects a ``DataType`` owner, so this
  is a hard constraint, not a preference.
* **``Multiplicity.max`` rejects ``<= 0``**, so an ``owl:maxCardinality 0`` is
  clamped.
* **``all_parents()`` recurses without a visited set**, so a generalization
  cycle would hang the first downstream consumer. Cycles are broken here.
* **The editor drops any OCL constraint** whose expression does not start with
  ``context <TypeName>`` and have at least four whitespace-separated tokens
  (``ocl_parser.py``), so that shape is mandatory rather than stylistic.

Construction order is deliberate: generalizations are built *before*
associations because ``Generalization``'s setters re-run the end-name
validator, which is a no-op while no associations exist. That makes
association construction the single place a validation error can surface.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from besser.BUML.metamodel.structural import (
    UNLIMITED_MAX_MULTIPLICITY,
    BinaryAssociation,
    Class,
    Constraint,
    DomainModel,
    Enumeration,
    EnumerationLiteral,
    Generalization,
    Metadata,
    Multiplicity,
    Property,
    StringType,
)

from besser.BUML.notations.kg_to_buml._common import KGConversionWarning, add_warning
from besser.BUML.notations.kg_to_buml.datatype_mapping import PRIMITIVE_BY_NAME
from besser.BUML.notations.kg_to_buml.owl2uml import naming
from besser.BUML.notations.kg_to_buml.owl2uml.model import (
    Association as UMLAssociation,
    Class as UMLClass,
    UMLModel,
)

__all__ = ["lower_to_buml", "LoweredModel"]


#: OCL standard operations that may follow ``self.`` without naming a feature.
_OCL_OPERATIONS = frozenset({
    "oclIsKindOf", "oclIsTypeOf", "oclAsType", "oclIsNew", "oclIsUndefined",
    "allInstances", "size", "language", "asSet", "asBag", "asSequence",
    "isEmpty", "notEmpty", "sum", "count", "first", "last", "flatten",
})

_TYPE_CHECK_RE = re.compile(r"ocl(?:IsKindOf|IsTypeOf|AsType)\(\s*([A-Za-z_]\w*)\s*\)")
_SELF_FEATURE_RE = re.compile(r"self\.([A-Za-z_]\w*)")


class LoweredModel:
    """Result of :func:`lower_to_buml`."""

    def __init__(self) -> None:
        self.domain_model: Optional[DomainModel] = None
        self.iri_to_class: Dict[str, Class] = {}
        self.property_iri_to_attribute: Dict[str, Property] = {}
        self.property_iri_to_association: Dict[str, BinaryAssociation] = {}
        self.assoc_source_end: Dict[int, Property] = {}
        self.class_stereotypes: Dict[str, List[str]] = {}
        self.warnings: List[KGConversionWarning] = []


def lower_to_buml(
    model: UMLModel,
    *,
    model_name: str,
    emit_ocl: bool = True,
    warnings: Optional[List[KGConversionWarning]] = None,
) -> LoweredModel:
    """Build a ``DomainModel`` (plus OCL constraints) from ``model``."""
    out = LoweredModel()
    out.warnings = warnings if warnings is not None else []

    renames = _plan_renames(model, out.warnings)
    generalizations = _break_cycles(model, renames, out.warnings)

    classes = _build_classes(model, renames, out)
    enumerations = _build_enumerations(model, renames, out.warnings)
    _build_attributes(model, renames, classes, enumerations, out)

    buml_generalizations = _build_generalizations(generalizations, classes)
    associations = _build_associations(model, renames, classes, buml_generalizations, out)

    domain_model = DomainModel(
        name=_safe_model_name(model_name),
        types=set(classes.values()) | set(enumerations.values()),
        associations=set(associations),
        generalizations=set(buml_generalizations),
    )
    out.domain_model = domain_model

    if emit_ocl:
        _build_constraints(model, renames, classes, domain_model, out)
    return out


# ----------------------------------------------------------------------
# Naming
# ----------------------------------------------------------------------


def _safe_model_name(raw: str) -> str:
    name = naming.sanitize((raw or "KGClassDiagram").strip())
    return name or "KGClassDiagram"


def _plan_renames(model: UMLModel, warnings) -> Dict[str, str]:
    """Map UML type name → BUML-safe type name.

    ``naming.sanitize`` already satisfies ``NamedElement.name`` (no spaces, no
    hyphens, non-empty). The one residual hazard is a class named after a BUML
    primitive: ``DomainModel.types`` treats the presence of a primitive-named
    type as "primitives already supplied", suppresses its own injection, and
    then raises on the duplicate name.
    """
    renames: Dict[str, str] = {}
    taken: Set[str] = set()
    names = sorted(set(model.classes) | set(model.enumerations))
    for name in names:
        safe = naming.sanitize(name) or "_"
        if safe in PRIMITIVE_BY_NAME:
            safe = f"{safe}_"
            add_warning(
                warnings,
                "NAME_SHADOWS_PRIMITIVE",
                f"Class {name!r} collides with the BUML primitive of the same name; "
                f"renamed to {safe!r}.",
            )
        candidate, index = safe, 2
        while candidate in taken:
            candidate = f"{safe}_{index}"
            index += 1
        taken.add(candidate)
        renames[name] = candidate
    return renames


def _resolve(name: str, renames: Dict[str, str]) -> str:
    return renames.get(name, naming.sanitize(name) or "_")


# ----------------------------------------------------------------------
# Generalizations
# ----------------------------------------------------------------------


def _break_cycles(model: UMLModel, renames, warnings) -> List[Tuple[str, str]]:
    """Return ``(subclass, superclass)`` pairs with cycle-forming edges removed.

    ``Class.all_parents()`` and ``all_specializations()`` recurse without a
    visited set, so a single cycle makes them hang. An ontology with mutually
    ``rdfs:subClassOf``-related classes (or an ``owl:equivalentClass`` pair the
    canonicaliser could not merge) is enough to produce one.
    """
    edges = sorted(
        {(_resolve(g.subclass, renames), _resolve(g.superclass, renames))
         for g in model.generalizations}
    )
    parents: Dict[str, List[str]] = {}
    for sub, sup in edges:
        parents.setdefault(sub, []).append(sup)

    kept: List[Tuple[str, str]] = []
    state: Dict[str, int] = {}  # 0 = visiting, 1 = done
    dropped: Set[Tuple[str, str]] = set()

    def visit(node: str) -> None:
        state[node] = 0
        for parent in parents.get(node, []):
            if state.get(parent) == 0:
                dropped.add((node, parent))
                add_warning(
                    warnings,
                    "CYCLIC_SUBCLASS",
                    f"Generalization {node} -> {parent} closes an inheritance cycle; dropped.",
                )
                continue
            if parent not in state:
                visit(parent)
        state[node] = 1

    for sub, _ in edges:
        if sub not in state:
            visit(sub)

    for sub, sup in edges:
        if sub != sup and (sub, sup) not in dropped:
            kept.append((sub, sup))
    return kept


def _build_generalizations(pairs, classes) -> List[Generalization]:
    out: List[Generalization] = []
    for sub, sup in pairs:
        specific, general = classes.get(sub), classes.get(sup)
        if specific is None or general is None:
            continue
        out.append(Generalization(general=general, specific=specific))
    return out


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------


def _build_classes(model: UMLModel, renames, out: LoweredModel) -> Dict[str, Class]:
    """Create one ``Class`` per UML class, plus one per materialised data range.

    Materialised data ranges (``DataUnionOf`` and friends, O01-O05) are the
    paper's ``<<dataType>>`` classes. They must be BUML ``Class`` objects, not
    ``DataType`` objects: ``Property.owner`` raises for a ``DataType`` owner, so
    a ``DataType`` cannot carry the ``value`` attribute the rule requires.
    """
    classes: Dict[str, Class] = {}

    for name in sorted(model.classes):
        uml_class = model.classes[name]
        safe = _resolve(name, renames)
        buml_class = Class(
            name=safe,
            is_abstract=uml_class.is_abstract,
            metadata=_metadata_for(uml_class),
        )
        classes[safe] = buml_class
        if uml_class.uri:
            out.iri_to_class[uml_class.uri] = buml_class
        stereotypes = uml_class.stereotypes()
        if stereotypes:
            out.class_stereotypes[safe] = stereotypes

    for name in sorted(model.datatypes):
        datatype = model.datatypes[name]
        if datatype.is_primitive:
            continue  # resolves to a BUML primitive singleton, not a type of its own
        safe = _resolve(name, renames) if name in renames else naming.sanitize(name)
        if safe in classes:
            continue
        description = "; ".join(datatype.facets) if datatype.facets else None
        classes[safe] = Class(
            name=safe,
            metadata=Metadata(uri=datatype.uri, description=description)
            if (datatype.uri or description)
            else None,
        )
        out.class_stereotypes.setdefault(safe, []).append("dataType")

    _report_stereotypes(out)
    return classes


def _metadata_for(uml_class: UMLClass) -> Optional[Metadata]:
    if not (uml_class.uri or uml_class.comment):
        return None
    return Metadata(uri=uml_class.uri, description=uml_class.comment)


def _report_stereotypes(out: LoweredModel) -> None:
    """Summarise the classifications BUML has no place to store.

    BUML has no stereotype concept, and ``Metadata.description`` round-trips
    into a visible comment box in the editor — misrepresenting a classification
    as authored prose. The classification is exposed on the result object and
    summarised as warnings instead.
    """
    buckets: Dict[str, List[str]] = {}
    for name, stereotypes in out.class_stereotypes.items():
        for stereotype in stereotypes:
            buckets.setdefault(stereotype, []).append(name)
    labels = {
        "aux": ("AUX_CLASSES_MATERIALISED", "auxiliary class(es) materialised from OWL expressions"),
        "external": ("EXTERNAL_CLASSES_STUBBED", "class(es) referenced but never declared"),
        "deprecated": ("DEPRECATED_CLASSES", "class(es) marked owl:deprecated"),
        "dataType": ("DATA_RANGES_MATERIALISED", "materialised data range(s)"),
    }
    for stereotype, names in sorted(buckets.items()):
        code, description = labels.get(stereotype, ("CLASS_STEREOTYPE", stereotype))
        add_warning(
            out.warnings,
            code,
            f"{len(names)} {description}: {', '.join(sorted(names)[:8])}"
            + (" …" if len(names) > 8 else ""),
        )


def _build_enumerations(model: UMLModel, renames, warnings) -> Dict[str, Enumeration]:
    enumerations: Dict[str, Enumeration] = {}
    for name in sorted(model.enumerations):
        uml_enum = model.enumerations[name]
        safe = _resolve(name, renames)
        literals, taken = [], set()
        for raw in uml_enum.literals:
            # ``Mapper.literal_value`` quotes strings; a literal like "foo bar"
            # would otherwise be rejected by ``NamedElement.name``.
            text = str(raw).strip().strip('"')
            candidate = naming.sanitize(text) or "_"
            base, index = candidate, 2
            while candidate in taken:
                candidate = f"{base}_{index}"
                index += 1
            taken.add(candidate)
            literals.append(EnumerationLiteral(name=candidate))
        enumerations[safe] = Enumeration(name=safe, literals=set(literals))
    return enumerations


def _lower_type(type_name: str, renames, classes, enumerations):
    """Resolve a UML type name to a BUML type object."""
    primitive = PRIMITIVE_BY_NAME.get(type_name)
    if primitive is not None:
        return primitive
    safe = _resolve(type_name, renames)
    return classes.get(safe) or enumerations.get(safe) or StringType


def _multiplicity(lower: int, upper, warnings, *, label: str) -> Multiplicity:
    """Build a fresh ``Multiplicity``.

    Never reuse the ``Property`` default — it is a shared module-level
    singleton, so mutating it would affect every other property.
    """
    maximum = UNLIMITED_MAX_MULTIPLICITY if upper == "*" else int(upper)
    if maximum <= 0:
        add_warning(
            warnings,
            "MULTIPLICITY_CLAMPED",
            f"{label} has an upper bound of {upper}, which BUML rejects; clamped to 1.",
        )
        maximum = 1
    minimum = max(0, int(lower))
    if minimum > maximum:
        minimum = maximum
    return Multiplicity(minimum, maximum)


# ----------------------------------------------------------------------
# Attributes
# ----------------------------------------------------------------------


def _build_attributes(model: UMLModel, renames, classes, enumerations, out: LoweredModel) -> None:
    for name in sorted(model.classes):
        uml_class = model.classes[name]
        safe = _resolve(name, renames)
        buml_class = classes.get(safe)
        if buml_class is None or not uml_class.attributes:
            continue
        _attach_attributes(buml_class, uml_class.attributes, renames, classes, enumerations, out)

    # Materialised data ranges carry a single `value` attribute (O01-O05).
    for name in sorted(model.datatypes):
        datatype = model.datatypes[name]
        if datatype.is_primitive or not datatype.attributes:
            continue
        safe = _resolve(name, renames) if name in renames else naming.sanitize(name)
        buml_class = classes.get(safe)
        if buml_class is None:
            continue
        _attach_attributes(buml_class, datatype.attributes, renames, classes, enumerations, out)


def _attach_attributes(buml_class, attributes, renames, classes, enumerations, out) -> None:
    built: List[Property] = []
    taken: Set[str] = set()
    seen_id = False
    for attribute in attributes:
        attr_name = naming.sanitize(attribute.name) or "_"
        if attr_name in taken:
            continue
        taken.add(attr_name)
        is_id = bool(attribute.is_id)
        if is_id and seen_id:
            # BUML permits at most one identifier per class.
            add_warning(
                out.warnings,
                "MULTIPLE_ID_ATTRIBUTES",
                f"{buml_class.name}.{attr_name} is a second identifier attribute "
                f"(from owl:hasKey); kept as an ordinary attribute.",
            )
            is_id = False
        seen_id = seen_id or is_id
        prop = Property(
            name=attr_name,
            type=_lower_type(attribute.type, renames, classes, enumerations),
            multiplicity=_multiplicity(
                attribute.lower, attribute.upper, out.warnings,
                label=f"Attribute {buml_class.name}.{attr_name}",
            ),
            is_id=is_id,
            is_derived=bool(attribute.is_derived),
            metadata=Metadata(uri=attribute.uri) if attribute.uri else None,
        )
        built.append(prop)
        if attribute.uri:
            out.property_iri_to_attribute[attribute.uri] = prop
    if built:
        buml_class.attributes = set(built)


# ----------------------------------------------------------------------
# Associations
# ----------------------------------------------------------------------


def _build_associations(
    model: UMLModel, renames, classes, generalizations, out: LoweredModel
) -> List[BinaryAssociation]:
    resolved = _resolve_association_types(model, renames, classes)
    resolved = _merge_fanout(resolved, classes, generalizations, out)
    return _instantiate_associations(resolved, classes, generalizations, out)


class _PlannedAssociation:
    """A binary association after type resolution but before instantiation."""

    __slots__ = ("source", "target", "role", "source_role", "src_mult", "tgt_mult",
                 "navigable_source", "uri", "name")

    def __init__(self, assoc: UMLAssociation, source: str, target: str):
        self.source = source
        self.target = target
        self.role = naming.sanitize(assoc.target.role or assoc.name) or "target"
        self.source_role = naming.sanitize(assoc.source.role) if assoc.source.role else None
        self.src_mult = (assoc.source.lower, assoc.source.upper)
        self.tgt_mult = (assoc.target.lower, assoc.target.upper)
        self.navigable_source = assoc.source.navigable
        self.uri = assoc.uri
        self.name = naming.sanitize(assoc.name) or self.role


def _resolve_association_types(model, renames, classes) -> List[_PlannedAssociation]:
    planned: List[_PlannedAssociation] = []
    seen: Set[Tuple[str, str, str]] = set()
    ordered = sorted(
        model.associations,
        key=lambda a: (a.source.type, a.target.role or a.name, a.target.type, a.uri or ""),
    )
    for assoc in ordered:
        source = _resolve(assoc.source.type, renames)
        target = _resolve(assoc.target.type, renames)
        if source not in classes or target not in classes:
            continue
        key = (source, target, naming.sanitize(assoc.target.role or assoc.name) or "target")
        if key in seen:
            continue
        seen.add(key)
        planned.append(_PlannedAssociation(assoc, source, target))
    return planned


def _merge_fanout(planned, classes, generalizations, out: LoweredModel):
    """Collapse associations that would give one class two ends of the same name.

    Rules D30/D31 link a union-typed ``rdfs:domain``/``rdfs:range`` directly to
    every member class rather than through the union class, which yields several
    associations sharing a source and a role name. BUML rejects that, and
    renaming the ends would invalidate every ``self.<role>`` in the generated
    OCL — so the group is reinstated as the D19 union construction: one
    association to an abstract auxiliary class that every former target
    specialises. ``oclIsKindOf(Ti)`` still holds, so the invariants stay true.
    """
    groups: Dict[Tuple[str, str], List[_PlannedAssociation]] = {}
    for assoc in planned:
        groups.setdefault((assoc.source, assoc.role), []).append(assoc)

    merged: List[_PlannedAssociation] = []
    for (source, role), group in sorted(groups.items()):
        targets = sorted({a.target for a in group})
        if len(targets) <= 1:
            merged.extend(group)
            continue
        aux_name = _ensure_union_class(targets, classes, generalizations, out)
        head = group[0]
        head.target = aux_name
        head.tgt_mult = _widest([a.tgt_mult for a in group])
        head.src_mult = _widest([a.src_mult for a in group])
        merged.append(head)
        add_warning(
            out.warnings,
            "ASSOC_FANOUT_MERGED",
            f"{source}.{role} pointed at {len(targets)} classes "
            f"({', '.join(targets[:5])}{' …' if len(targets) > 5 else ''}); "
            f"merged onto the auxiliary union class {aux_name}.",
        )
    return merged


def _ensure_union_class(targets, classes, generalizations, out) -> str:
    aux_name = naming.sanitize(naming.union_name(list(targets)))
    if aux_name in classes:
        return aux_name
    aux = Class(name=aux_name, is_abstract=True)
    classes[aux_name] = aux
    out.class_stereotypes.setdefault(aux_name, []).append("aux")
    for target in targets:
        specific = classes.get(target)
        if specific is not None:
            generalizations.append(Generalization(general=aux, specific=specific))
    return aux_name


def _widest(bounds: Sequence[Tuple[int, Any]]) -> Tuple[int, Any]:
    lower = min(b[0] for b in bounds)
    if any(b[1] == "*" for b in bounds):
        return lower, "*"
    return lower, max(int(b[1]) for b in bounds)


class _EndNameSpace:
    """Mirrors BUML's association-end-name rule so we never trip it.

    ``Association.ends`` validates each end against the *other* end's owning
    class: for an association ``(S:A, T:B)`` it requires ``T`` to be unused
    across ``A``, ``A``'s ancestors and ``A``'s descendants, and ``S`` to be
    unused across ``B``'s. "Used" means appearing in ``Class.association_ends()``,
    which returns the *opposite* ends — i.e. the names that class can navigate.
    Attribute names do not participate.

    Tracking this here lets us rename the free end before construction instead
    of catching a ``ValueError`` afterwards and losing the association.
    """

    def __init__(self, classes, generalizations):
        parents: Dict[str, List[str]] = {}
        children: Dict[str, List[str]] = {}
        for generalization in generalizations:
            sub, sup = generalization.specific.name, generalization.general.name
            parents.setdefault(sub, []).append(sup)
            children.setdefault(sup, []).append(sub)
        self._closures = {
            name: _closure(name, parents) | _closure(name, children) | {name}
            for name in classes
        }
        self._visible: Dict[str, Set[str]] = {name: set() for name in classes}

    def blocked(self, class_name: str) -> Set[str]:
        """Names already navigable anywhere in ``class_name``'s hierarchy."""
        names: Set[str] = set()
        for relative in self._closures.get(class_name, {class_name}):
            names |= self._visible.get(relative, set())
        return names

    def register(self, source: str, target: str, source_role: str, target_role: str) -> None:
        self._visible.setdefault(source, set()).add(target_role)
        self._visible.setdefault(target, set()).add(source_role)
        if source == target:
            # A self-association keeps both ends navigable on the same class.
            self._visible[source].add(source_role)


def _closure(start: str, adjacency: Dict[str, List[str]]) -> Set[str]:
    seen: Set[str] = set()
    stack = list(adjacency.get(start, []))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency.get(current, []))
    return seen


def _ancestors_first(planned, generalizations):
    """Order associations so a superclass's copy of a role name wins.

    When a class and one of its ancestors both declare the same role, BUML keeps
    only one and the other is dropped. The ancestor's is the one to keep: the
    auxiliary classes materialised for restrictions (D22-D25) sit *above* the
    classes they constrain and carry the tighter multiplicity, so
    ``_some_owns_Pet.owns [1..*]`` must survive rather than the unconstrained
    ``Person.owns [0..*]`` it subsumes. Processing shallow sources first makes
    that happen without depending on the input order.
    """
    parents: Dict[str, List[str]] = {}
    for generalization in generalizations:
        parents.setdefault(generalization.specific.name, []).append(generalization.general.name)

    depths: Dict[str, int] = {}

    def depth(name: str, seen: Optional[Set[str]] = None) -> int:
        if name in depths:
            return depths[name]
        seen = seen or set()
        if name in seen:
            return 0
        seen.add(name)
        value = max((depth(p, seen) + 1 for p in parents.get(name, [])), default=0)
        depths[name] = value
        return value

    return sorted(planned, key=lambda a: (depth(a.source), a.source, a.role, a.target))


def _instantiate_associations(planned, classes, generalizations, out: LoweredModel):
    associations: List[BinaryAssociation] = []
    used_names: Set[str] = set()
    namespace = _EndNameSpace(classes, generalizations)
    attribute_names = {
        name: {attribute.name for attribute in buml_class.attributes}
        for name, buml_class in classes.items()
    }

    for assoc in _ancestors_first(planned, generalizations):
        source_class, target_class = classes.get(assoc.source), classes.get(assoc.target)
        if source_class is None or target_class is None:
            continue

        target_role = assoc.role
        if target_role in namespace.blocked(assoc.source):
            # The role name is already navigable from this class (usually via an
            # ancestor that declares the same property). The inherited feature
            # satisfies every ``self.<role>`` reference, so the duplicate is
            # redundant rather than lost.
            add_warning(
                out.warnings,
                "ASSOC_INHERITED_SHADOWED",
                f"{assoc.source}.{target_role} is already navigable from this class's "
                f"hierarchy; the redundant association was dropped.",
            )
            continue
        if target_role in attribute_names.get(assoc.source, set()):
            add_warning(
                out.warnings,
                "NAME_SHADOWS_ATTRIBUTE",
                f"{assoc.source}.{target_role} names both an attribute and an "
                f"association end; kept as-is (the ambiguity is in the source ontology).",
            )

        # The reverse end can always be renamed; the forward one cannot without
        # invalidating the generated OCL.
        source_role = _unique_name(
            assoc.source_role or _default_source_role(assoc),
            namespace.blocked(assoc.target) | {target_role},
        )

        source_end = Property(
            name=source_role,
            type=source_class,
            multiplicity=_multiplicity(
                *assoc.src_mult, out.warnings, label=f"Association end {assoc.source}.{source_role}"
            ),
            is_navigable=assoc.navigable_source,
        )
        target_end = Property(
            name=target_role,
            type=target_class,
            multiplicity=_multiplicity(
                *assoc.tgt_mult, out.warnings, label=f"Association end {assoc.target}.{target_role}"
            ),
        )
        name = _unique_association_name(assoc, used_names)
        try:
            association = BinaryAssociation(name=name, ends={source_end, target_end})
        except (ValueError, TypeError) as exc:
            add_warning(
                out.warnings,
                "ASSOC_DROPPED",
                f"Could not build association {name!r} "
                f"({assoc.source} -> {assoc.target}): {exc}",
            )
            continue
        namespace.register(assoc.source, assoc.target, source_role, target_role)
        associations.append(association)
        out.assoc_source_end[id(association)] = source_end
        if assoc.uri:
            out.property_iri_to_association[assoc.uri] = association
    return associations


def _default_source_role(assoc: _PlannedAssociation) -> str:
    """Name for the reverse end, which the paper's model leaves anonymous.

    Always safe to choose freely: generated OCL only navigates a source role
    when the reference marked it navigable, which happens exactly when the
    source ontology named it via ``owl:inverseOf``.
    """
    base = assoc.source[:1].lower() + assoc.source[1:] if assoc.source else "source"
    if assoc.source == assoc.target:
        # Self-association: BUML keeps both ends only when their names differ.
        base = f"source_{base}"
    return naming.sanitize(base) or "source"


def _unique_name(candidate: str, taken: Set[str]) -> str:
    name, index = candidate, 2
    while name in taken:
        name = f"{candidate}_{index}"
        index += 1
    return name


def _unique_association_name(assoc: _PlannedAssociation, used: Set[str]) -> str:
    for candidate in (assoc.name, f"{assoc.name}_{assoc.source}"):
        if candidate not in used:
            used.add(candidate)
            return candidate
    base, index = f"{assoc.name}_{assoc.source}", 2
    name = f"{base}_{index}"
    while name in used:
        index += 1
        name = f"{base}_{index}"
    used.add(name)
    return name


# ----------------------------------------------------------------------
# OCL constraints
# ----------------------------------------------------------------------


def _build_constraints(model: UMLModel, renames, classes, domain_model, out: LoweredModel) -> None:
    """Attach the invariants, re-validated against the finished model.

    The mapper already guards emission with ``_has_property``, but that runs
    before the fan-out merges and end renames above. Re-checking here is what
    guarantees no constraint survives referencing a feature or type the
    ``DomainModel`` does not actually have — which would otherwise surface as a
    silent drop (or a crash) in the editor and the OCL generators.
    """
    type_names = {t.name for t in domain_model.types}
    features = _reachable_features(classes, domain_model)
    used_names: Set[str] = set()

    for invariant in model.ocl_constraints():
        context_name = _resolve(invariant.context, renames)
        context = classes.get(context_name)
        if context is None:
            continue
        body = " ".join(invariant.body.split())
        if not body:
            continue
        if re.search(r"\bcontext\b", body):
            add_warning(
                out.warnings,
                "OCL_DROPPED_MALFORMED_BODY",
                f"Invariant on {context_name} contains a bare 'context' token, which "
                f"the editor's OCL parser uses as a block separator; dropped.",
            )
            continue
        if not _features_resolve(body, context_name, features, out, context_name):
            continue
        if not _types_resolve(body, type_names, out, context_name):
            continue

        name = _unique_constraint_name(context_name, invariant, used_names)
        domain_model.add_constraint(Constraint(
            name=name,
            context=context,
            expression=f"context {context_name} inv {name}: {body}",
            language="OCL",
        ))


def _reachable_features(classes, domain_model) -> Dict[str, Set[str]]:
    """Feature names an invariant on each class may legitimately navigate.

    Its own features plus every ancestor's, and also every *descendant's*.
    Descendants matter because of the † rule in the paper's Table 3: the
    auxiliary class materialised for a data restriction carries the invariant,
    while the feature it constrains is declared on the named classes that become
    its subclasses. Excluding descendants would drop exactly those invariants
    (O10-O15) even though the model does define the feature.

    ``Thing``'s features are folded into every class because the mapper attaches
    domain-less properties there, and every OWL class is implicitly a subclass
    of ``owl:Thing`` even without an explicit generalization edge.
    """
    direct: Dict[str, Set[str]] = {}
    for name, buml_class in classes.items():
        # ``association_ends()`` already returns the *opposite* ends — i.e. the
        # names this class can navigate — so their names are the features.
        direct[name] = (
            {attribute.name for attribute in buml_class.attributes}
            | {end.name for end in buml_class.association_ends()}
        )

    parents: Dict[str, List[str]] = {}
    children: Dict[str, List[str]] = {}
    for generalization in domain_model.generalizations:
        sub, sup = generalization.specific.name, generalization.general.name
        parents.setdefault(sub, []).append(sup)
        children.setdefault(sup, []).append(sub)

    thing_features = direct.get("Thing", set())
    resolved: Dict[str, Set[str]] = {}
    for name in classes:
        names = set(thing_features) | direct.get(name, set())
        for relative in _closure(name, parents) | _closure(name, children):
            names |= direct.get(relative, set())
        resolved[name] = names
    return resolved


def _features_resolve(body, context_name, features, out, label) -> bool:
    available = features.get(context_name, set())
    for feature in _SELF_FEATURE_RE.findall(body):
        if feature in _OCL_OPERATIONS or feature in available:
            continue
        add_warning(
            out.warnings,
            "OCL_DROPPED_UNRESOLVED_FEATURE",
            f"Invariant on {label} navigates self.{feature}, which the class does not have; dropped.",
        )
        return False
    return True


def _types_resolve(body, type_names, out, label) -> bool:
    for type_name in _TYPE_CHECK_RE.findall(body):
        if type_name in type_names or type_name in PRIMITIVE_BY_NAME:
            continue
        add_warning(
            out.warnings,
            "OCL_DROPPED_UNKNOWN_TYPE",
            f"Invariant on {label} references the unknown type {type_name}; dropped.",
        )
        return False
    return True


def _unique_constraint_name(context_name, invariant, used: Set[str]) -> str:
    hint = invariant.name or invariant.origin_rule or "inv"
    base = naming.sanitize(f"{context_name}_{hint}") or f"{context_name}_inv"
    name, index = base, 2
    while name in used:
        name = f"{base}_{index}"
        index += 1
    used.add(name)
    return name
