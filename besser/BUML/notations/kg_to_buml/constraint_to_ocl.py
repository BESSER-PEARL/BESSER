"""Translate KG constraint nodes + axioms into B-UML OCL ``Constraint`` objects.

Called from :func:`kg_to_class_diagram` after structural lifting completes.
Walks every reachable :class:`KGNodeConstraint` and :class:`KGPropertyConstraint`,
plus :attr:`KnowledgeGraph.axioms`, and emits one :class:`Constraint` per
translatable spec / axiom (attached to the appropriate context class).

Structural cardinality (``min/max/exactCardinality``) is already lifted into
``Property.multiplicity`` upstream, so we deliberately skip those kinds here
to avoid duplicate / contradictory invariants. Everything else that maps
cleanly to the B-OCL grammar (see :mod:`besser.BUML.notations.ocl`) is emitted
as an ``inv``; specs/axioms we can't lower produce a ``KGConversionWarning``.

The emitted expression follows the grammar's ``contextDeclaration`` shape::

    context <ContextClassName> inv <ConstraintName>: <body>

so the result parses cleanly through ``BOCLLexer`` / ``BOCLParser`` (with the
exception of ``matches()`` for regex patterns — see the plan).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from besser.BUML.metamodel.kg import (
    KGClass,
    KGIndividual,
    KGLiteral,
    KGNode,
    KGNodeConstraint,
    KGProperty,
    KGPropertyConstraint,
    KnowledgeGraph,
)
from besser.BUML.metamodel.kg.axioms import (
    DisjointClassesAxiom,
    DisjointUnionAxiom,
    EquivalentClassesAxiom,
    HasKeyAxiom,
    InversePropertiesAxiom,
    PropertyChainAxiom,
    SubPropertyOfAxiom,
)
from besser.BUML.metamodel.kg.constraint_specs import LOGICAL_KINDS
from besser.BUML.metamodel.structural import (
    BinaryAssociation,
    Class,
    Constraint,
    Property,
)

from besser.BUML.notations.kg_to_buml._common import (
    KGConversionWarning,
    add_warning,
    sanitize_python_identifier,
)


__all__ = ["OCLEmissionResult", "emit_ocl_constraints"]


_MAX_REF_DEPTH = 8


@dataclass
class OCLEmissionResult:
    constraints: List[Constraint] = field(default_factory=list)
    warnings: List[KGConversionWarning] = field(default_factory=list)


@dataclass
class _LookupCtx:
    """Shared resolution maps + name machinery passed through the emitter."""

    iri_to_class: Dict[str, Class]
    id_to_buml_class: Dict[str, Class]
    property_iri_to_attribute: Dict[str, Property]
    property_iri_to_association: Dict[str, BinaryAssociation]
    assoc_source_end: Dict[int, Property]
    pc_to_property: Dict[str, str]
    id_to_constraint_node: Dict[str, KGNode]
    id_to_property_node: Dict[str, KGProperty]
    individual_ids: Set[str]
    used_constraint_names: Set[str]
    name_counters: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    emitted_keys: Set[Tuple] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def emit_ocl_constraints(
    kg: KnowledgeGraph,
    *,
    iri_to_class: Dict[str, Class],
    property_iri_to_attribute: Dict[str, Property],
    property_iri_to_association: Dict[str, BinaryAssociation],
    assoc_source_end: Dict[int, Property],
    class_to_ncs: Dict[str, List[KGNodeConstraint]],
    nc_to_pcs: Dict[str, List[KGPropertyConstraint]],
    pc_to_property: Dict[str, str],
) -> OCLEmissionResult:
    """Build OCL :class:`Constraint` objects from a KG's constraint nodes + axioms.

    Args:
        kg: The source knowledge graph.
        iri_to_class: Map produced by ``kg_to_class_diagram`` (KGClass.iri → B-UML Class).
        property_iri_to_attribute: KGProperty.iri → B-UML ``Property`` attribute.
        property_iri_to_association: KGProperty.iri → B-UML ``BinaryAssociation``.
        assoc_source_end: id(BinaryAssociation) → the *source* end property
            (used to identify the *target* end's name).
        class_to_ncs / nc_to_pcs / pc_to_property: index dicts already built
            inside ``_apply_restrictions_and_characteristics``.
    """
    result = OCLEmissionResult()

    id_to_buml_class: Dict[str, Class] = {}
    individual_ids: Set[str] = set()
    id_to_property_node: Dict[str, KGProperty] = {}
    id_to_constraint_node: Dict[str, KGNode] = {}

    for node in kg.nodes:
        if isinstance(node, KGClass):
            buml_cls = iri_to_class.get(getattr(node, "iri", None) or "")
            if buml_cls is not None:
                id_to_buml_class[node.id] = buml_cls
        elif isinstance(node, KGProperty):
            id_to_property_node[node.id] = node
            if node.iri:
                id_to_property_node[node.iri] = node
        elif isinstance(node, KGIndividual):
            individual_ids.add(node.id)
            if node.iri:
                individual_ids.add(node.iri)
        if isinstance(node, (KGNodeConstraint, KGPropertyConstraint)):
            id_to_constraint_node[node.id] = node

    used_names: Set[str] = set()

    ctx = _LookupCtx(
        iri_to_class=iri_to_class,
        id_to_buml_class=id_to_buml_class,
        property_iri_to_attribute=property_iri_to_attribute,
        property_iri_to_association=property_iri_to_association,
        assoc_source_end=assoc_source_end,
        pc_to_property=pc_to_property,
        id_to_constraint_node=id_to_constraint_node,
        id_to_property_node=id_to_property_node,
        individual_ids=individual_ids,
        used_constraint_names=used_names,
    )

    _emit_property_constraints(kg, class_to_ncs, nc_to_pcs, ctx, result)
    _emit_node_constraints(kg, class_to_ncs, ctx, result)
    _emit_axioms(kg, ctx, result)

    return result


# ---------------------------------------------------------------------------
# Per-source emission
# ---------------------------------------------------------------------------


def _emit_property_constraints(
    kg: KnowledgeGraph,
    class_to_ncs: Dict[str, List[KGNodeConstraint]],
    nc_to_pcs: Dict[str, List[KGPropertyConstraint]],
    ctx: _LookupCtx,
    out: OCLEmissionResult,
) -> None:
    """Emit OCL invariants from every PropertyConstraint we can attach to an owner class.

    Source of (owner_class, property_constraint) pairs:
      1. NodeConstraint → Class via constraintTargetClass, then NC → PC via sh:property.
      2. Standalone PC linked to a class via rdfs:subClassOf / owl:equivalentClass.
    """
    seen_pairs: Set[Tuple[str, str]] = set()
    rdfs_sub = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
    owl_equiv = "http://www.w3.org/2002/07/owl#equivalentClass"

    # Path 1: NC → Class wraps PCs.
    for class_id, ncs in class_to_ncs.items():
        owner = ctx.id_to_buml_class.get(class_id)
        if owner is None:
            continue
        for nc in ncs:
            if _specs_are_deactivated(nc.get_specs()):
                add_warning(
                    out.warnings,
                    "OCL_SHAPE_DEACTIVATED",
                    f"Skipped emission of NodeConstraint {nc.id!r} (sh:deactivated=true).",
                    node_id=nc.id,
                )
                continue
            for pc in nc_to_pcs.get(nc.id, []):
                _emit_one_property_constraint(pc, owner, ctx, out, seen_pairs)

    # Path 2: standalone PC → Class via subClassOf / equivalentClass.
    for edge in kg.edges:
        if edge.iri not in (rdfs_sub, owl_equiv):
            continue
        if not isinstance(edge.target, KGPropertyConstraint):
            continue
        owner = ctx.id_to_buml_class.get(edge.source.id)
        if owner is None:
            continue
        _emit_one_property_constraint(edge.target, owner, ctx, out, seen_pairs)


def _emit_one_property_constraint(
    pc: KGPropertyConstraint,
    owner: Class,
    ctx: _LookupCtx,
    out: OCLEmissionResult,
    seen_pairs: Set[Tuple[str, str]],
) -> None:
    prop_iri = ctx.pc_to_property.get(pc.id)
    if not prop_iri:
        return
    key = (id(owner), pc.id)
    if key in seen_pairs:
        return
    seen_pairs.add(key)
    specs = pc.get_specs()
    if _specs_are_deactivated(specs):
        add_warning(
            out.warnings,
            "OCL_SHAPE_DEACTIVATED",
            f"Skipped emission of PropertyConstraint {pc.id!r} (sh:deactivated=true).",
            node_id=pc.id,
        )
        return
    prop_name = _resolve_property_name(prop_iri, owner, ctx)
    if prop_name is None:
        add_warning(
            out.warnings,
            "OCL_PROP_UNRESOLVED",
            f"PropertyConstraint {pc.id!r} targets a property ({prop_iri}) not present on class "
            f"{owner.name!r}; constraints skipped.",
            node_id=pc.id,
        )
        return
    is_multi = _is_multi_valued(prop_iri, ctx)
    for spec in specs:
        body = _translate_property_spec(spec, owner, prop_name, is_multi, ctx, out, set(), 0)
        if not body:
            continue
        _attach_constraint(owner, body, spec, pc.get_specs(), ctx, out, owner_label=prop_name)


def _emit_node_constraints(
    kg: KnowledgeGraph,
    class_to_ncs: Dict[str, List[KGNodeConstraint]],
    ctx: _LookupCtx,
    out: OCLEmissionResult,
) -> None:
    for class_id, ncs in class_to_ncs.items():
        owner = ctx.id_to_buml_class.get(class_id)
        if owner is None:
            continue
        for nc in ncs:
            specs = nc.get_specs()
            if _specs_are_deactivated(specs):
                continue  # already warned in _emit_property_constraints
            for spec in specs:
                body = _translate_node_spec(spec, owner, ctx, out, set(), 0)
                if not body:
                    continue
                _register_dedup_key(spec, owner, ctx)
                _attach_constraint(owner, body, spec, specs, ctx, out, owner_label=None)


def _register_dedup_key(spec: Dict[str, Any], owner: Class, ctx: _LookupCtx) -> None:
    """When an NC spec mirrors a class-level axiom, register the canonical
    dedup key so the axiom pass doesn't emit the same constraint a second
    time."""
    kind = spec.get("kind")
    value = spec.get("value")
    if kind == "hasKey":
        prop_iris = value if isinstance(value, list) else []
        names = [_resolve_property_name(str(p), owner, ctx) for p in prop_iris]
        names = [n for n in names if n]
        if names:
            ctx.emitted_keys.add(("has_key", id(owner), tuple(sorted(names))))
    elif kind == "disjointWith":
        items = value if isinstance(value, list) else [value]
        classes = [_resolve_class(item, ctx) for item in items]
        classes = [c for c in classes if c is not None]
        if classes:
            ctx.emitted_keys.add(("disjoint", id(owner), tuple(sorted(c.name for c in classes))))
    elif kind == "disjointUnionOf":
        items = value if isinstance(value, list) else []
        classes = [_resolve_class(item, ctx) for item in items]
        classes = [c for c in classes if c is not None]
        if len(classes) >= 2:
            ctx.emitted_keys.add(("disjoint_union", id(owner), tuple(sorted(c.name for c in classes))))
    elif kind == "equivalentClasses":
        items = value if isinstance(value, list) else [value]
        classes = [_resolve_class(item, ctx) for item in items]
        for c in classes:
            if c is not None:
                ctx.emitted_keys.add(("equiv", id(owner), c.name))


def _emit_axioms(kg: KnowledgeGraph, ctx: _LookupCtx, out: OCLEmissionResult) -> None:
    """Translate ``kg.axioms`` into OCL. De-duplicates against constraints
    already emitted from materialised NodeConstraints (see
    ``ctx.emitted_keys``)."""
    for axiom in kg.axioms:
        if isinstance(axiom, EquivalentClassesAxiom):
            classes = [ctx.id_to_buml_class.get(cid) for cid in axiom.class_ids]
            classes = [c for c in classes if c is not None]
            if len(classes) < 2:
                continue
            for owner in classes:
                others = [c for c in classes if c is not owner]
                for other in others:
                    key = ("equiv", id(owner), other.name)
                    if key in ctx.emitted_keys:
                        continue
                    ctx.emitted_keys.add(key)
                    body = f"self.oclIsKindOf({other.name}) = self.oclIsKindOf({owner.name})"
                    _attach_axiom_constraint(owner, body, "equiv", ctx, out)
        elif isinstance(axiom, DisjointClassesAxiom):
            classes = [ctx.id_to_buml_class.get(cid) for cid in axiom.class_ids]
            classes = [c for c in classes if c is not None]
            if len(classes) < 2:
                continue
            for owner in classes:
                others = [c for c in classes if c is not owner]
                others_sig = tuple(sorted(c.name for c in others))
                key = ("disjoint", id(owner), others_sig)
                if key in ctx.emitted_keys:
                    continue
                ctx.emitted_keys.add(key)
                body = " and ".join(f"not self.oclIsKindOf({c.name})" for c in others)
                _attach_axiom_constraint(owner, body, "disjoint", ctx, out)
        elif isinstance(axiom, DisjointUnionAxiom):
            owner = ctx.id_to_buml_class.get(axiom.union_class_id)
            parts = [ctx.id_to_buml_class.get(cid) for cid in axiom.part_class_ids]
            parts = [c for c in parts if c is not None]
            if owner is None or len(parts) < 2:
                continue
            parts_sig = tuple(sorted(c.name for c in parts))
            key = ("disjoint_union", id(owner), parts_sig)
            if key in ctx.emitted_keys:
                continue
            ctx.emitted_keys.add(key)
            body = " xor ".join(f"self.oclIsKindOf({c.name})" for c in parts)
            _attach_axiom_constraint(owner, body, "disjoint_union", ctx, out)
        elif isinstance(axiom, HasKeyAxiom):
            owner = ctx.id_to_buml_class.get(axiom.class_id)
            if owner is None:
                continue
            prop_names = [
                _resolve_property_name_by_id(pid, owner, ctx)
                for pid in axiom.property_ids
            ]
            prop_names = [p for p in prop_names if p]
            if not prop_names:
                continue
            sig = tuple(sorted(prop_names))
            key = ("has_key", id(owner), sig)
            if key in ctx.emitted_keys:
                continue
            ctx.emitted_keys.add(key)
            body = _has_key_body(owner.name, prop_names)
            _attach_axiom_constraint(owner, body, "has_key", ctx, out)
        elif isinstance(axiom, SubPropertyOfAxiom):
            owner, sub_name = _owner_and_name_for_property(axiom.sub_property_id, ctx)
            _, sup_name = _owner_and_name_for_property(axiom.super_property_id, ctx)
            if owner is None or not sub_name or not sup_name:
                add_warning(
                    out.warnings,
                    "OCL_AXIOM_UNRESOLVED",
                    f"SubPropertyOfAxiom: could not resolve {axiom.sub_property_id} ⊑ {axiom.super_property_id}; skipped.",
                )
                continue
            body = f"self.{sup_name}->includesAll(self.{sub_name})"
            _attach_axiom_constraint(owner, body, f"sub_{sub_name}", ctx, out)
        elif isinstance(axiom, InversePropertiesAxiom):
            owner, a_name = _owner_and_name_for_property(axiom.property_a_id, ctx)
            _, b_name = _owner_and_name_for_property(axiom.property_b_id, ctx)
            if owner is None or not a_name or not b_name:
                add_warning(
                    out.warnings,
                    "OCL_AXIOM_UNRESOLVED",
                    f"InversePropertiesAxiom: could not resolve {axiom.property_a_id} inverseOf {axiom.property_b_id}; skipped.",
                )
                continue
            body = f"self.{a_name}->forAll(o | o.{b_name}->includes(self))"
            _attach_axiom_constraint(owner, body, f"inverse_{a_name}", ctx, out)
        elif isinstance(axiom, PropertyChainAxiom):
            add_warning(
                out.warnings,
                "OCL_PROPERTY_CHAIN_UNSUPPORTED",
                f"PropertyChainAxiom on {axiom.property_id} is not translated to OCL.",
            )
        # ImportAxiom is metadata-only; skip silently.


# ---------------------------------------------------------------------------
# Property-spec translator
# ---------------------------------------------------------------------------


def _translate_property_spec(
    spec: Dict[str, Any],
    owner: Class,
    prop_name: str,
    is_multi: bool,
    ctx: _LookupCtx,
    out: OCLEmissionResult,
    visited: Set[str],
    depth: int,
) -> Optional[str]:
    kind = spec.get("kind")
    value = spec.get("value")
    on_class = spec.get("on_class")

    if kind in ("minCardinality", "maxCardinality", "exactCardinality"):
        return None  # structural Multiplicity handles these
    if kind == "datatype":
        return None  # enforced via attribute type
    if kind in ("nodeKind", "languageIn", "uniqueLang",
                "shaclClosed", "shaclIgnoredProperties",
                "shaclSeverity", "shaclMessage", "shaclName",
                "shaclDescription", "shaclOrder", "shaclGroup",
                "shaclDeactivated"):
        return None  # silent skip (handled elsewhere or non-constraint metadata)

    if kind in ("minQualifiedCardinality", "maxQualifiedCardinality", "exactQualifiedCardinality"):
        if on_class is None:
            return None  # falls back to plain cardinality (already structural)
        cls = _resolve_class(on_class, ctx)
        if cls is None:
            add_warning(out.warnings, "OCL_CLASS_UNRESOLVED",
                        f"Qualified cardinality on '{prop_name}' references unknown class {on_class!r}.")
            return None
        op = {"minQualifiedCardinality": ">=", "maxQualifiedCardinality": "<=", "exactQualifiedCardinality": "="}[kind]
        return f"self.{prop_name}->select(x | x.oclIsKindOf({cls.name}))->size() {op} {int(value)}"
    if kind == "someValuesFrom":
        cls = _resolve_class(value, ctx)
        if cls is None:
            add_warning(out.warnings, "OCL_CLASS_UNRESOLVED",
                        f"someValuesFrom on '{prop_name}' references unknown class {value!r}.")
            return None
        return f"self.{prop_name}->exists(x | x.oclIsKindOf({cls.name}))"
    if kind == "allValuesFrom":
        cls = _resolve_class(value, ctx)
        if cls is None:
            add_warning(out.warnings, "OCL_CLASS_UNRESOLVED",
                        f"allValuesFrom on '{prop_name}' references unknown class {value!r}.")
            return None
        return f"self.{prop_name}->forAll(x | x.oclIsKindOf({cls.name}))"
    if kind == "hasValue":
        formatted = _format_literal(value, out)
        if formatted is None:
            add_warning(out.warnings, "OCL_HASVALUE_IRI_SKIPPED",
                        f"hasValue with IRI on '{prop_name}' (value={value!r}) is not translated.")
            return None
        return f"self.{prop_name}->includes({formatted})" if is_multi else f"self.{prop_name} = {formatted}"
    if kind == "hasSelf":
        if not value:
            return None
        return f"self.{prop_name}->includes(self)"
    if kind == "in":
        items, has_iri = _format_literal_list(value, out)
        if has_iri:
            add_warning(out.warnings, "OCL_IN_IRI_SKIPPED",
                        f"`in` on '{prop_name}' contains IRI/individual entries; constraint skipped.")
            return None
        if not items:
            return None
        set_literal = "Set{" + ", ".join(items) + "}"
        if is_multi:
            return f"self.{prop_name}->forAll(x | {set_literal}->includes(x))"
        return f"{set_literal}->includes(self.{prop_name})"
    if kind == "pattern":
        regex = value if isinstance(value, str) else _format_literal(value, out)
        if regex is None:
            return None
        # Look for a sibling `flags` spec in the same constraint node's spec list.
        flag_string = _find_sibling_flags(spec, ctx)
        if flag_string:
            embedded = _embed_regex_flags(flag_string, out)
            regex_str = embedded + str(value) if not str(value).startswith("(?") else str(value)
        else:
            regex_str = str(value)
        quoted = _quote_string_literal(regex_str)
        return f"self.{prop_name}.matches({quoted})"
    if kind == "flags":
        return None  # absorbed by sibling `pattern`
    if kind == "minLength":
        return f"self.{prop_name}.size() >= {int(value)}"
    if kind == "maxLength":
        return f"self.{prop_name}.size() <= {int(value)}"
    if kind == "minInclusive":
        return f"self.{prop_name} >= {_format_number(value)}"
    if kind == "maxInclusive":
        return f"self.{prop_name} <= {_format_number(value)}"
    if kind == "minExclusive":
        return f"self.{prop_name} > {_format_number(value)}"
    if kind == "maxExclusive":
        return f"self.{prop_name} < {_format_number(value)}"
    if kind == "shaclDisjoint":
        # value is the disjoint property's IRI / id
        other = _resolve_property_name(str(value), owner, ctx) if value else None
        if other is None:
            add_warning(out.warnings, "OCL_DISJOINT_PROPERTY_UNRESOLVED",
                        f"shaclDisjoint on '{prop_name}' references unknown property {value!r}.")
            return None
        return f"self.{prop_name}->excludesAll(self.{other})"
    if kind in LOGICAL_KINDS:
        return _translate_logical(
            kind, value, owner, prop_name, is_multi, ctx, out, visited, depth, on_property=True,
        )
    add_warning(out.warnings, "OCL_KIND_UNSUPPORTED",
                f"Spec kind {kind!r} on '{prop_name}' has no OCL translation; skipped.")
    return None


# ---------------------------------------------------------------------------
# Node-spec translator
# ---------------------------------------------------------------------------


def _translate_node_spec(
    spec: Dict[str, Any],
    owner: Class,
    ctx: _LookupCtx,
    out: OCLEmissionResult,
    visited: Set[str],
    depth: int,
) -> Optional[str]:
    kind = spec.get("kind")
    value = spec.get("value")

    if kind in ("subClassOf",):
        return None  # structural Generalization
    if kind in ("nodeKind", "shaclClosed", "shaclIgnoredProperties",
                "shaclSeverity", "shaclMessage", "shaclName",
                "shaclDescription", "shaclOrder", "shaclGroup",
                "shaclDeactivated"):
        return None
    if kind == "oneOf":
        add_warning(out.warnings, "OCL_ONEOF_SKIPPED",
                    f"oneOf on class '{owner.name}' enumerates individuals; not translated.")
        return None
    if kind == "equivalentClasses":
        targets = _format_class_list(value, ctx, out, owner)
        if not targets:
            return None
        # one constraint per target; emit them via caller — here we
        # join into a single conjunction so the caller's single Constraint
        # carries the same semantics: ∀ D in targets, A ≡ D.
        parts = [f"self.oclIsKindOf({c.name}) = self.oclIsKindOf({owner.name})" for c in targets]
        return " and ".join(parts) if parts else None
    if kind == "disjointWith":
        targets = _format_class_list(value, ctx, out, owner)
        if not targets:
            return None
        return " and ".join(f"not self.oclIsKindOf({c.name})" for c in targets)
    if kind == "disjointUnionOf":
        targets = _format_class_list(value, ctx, out, owner)
        if len(targets) < 2:
            return None
        return " xor ".join(f"self.oclIsKindOf({c.name})" for c in targets)
    if kind == "unionOf":
        targets = _format_class_list(value, ctx, out, owner)
        if not targets:
            return None
        return " or ".join(f"self.oclIsKindOf({c.name})" for c in targets)
    if kind == "intersectionOf":
        targets = _format_class_list(value, ctx, out, owner)
        if not targets:
            return None
        return " and ".join(f"self.oclIsKindOf({c.name})" for c in targets)
    if kind == "complementOf":
        cls = _resolve_class(value, ctx)
        if cls is None:
            add_warning(out.warnings, "OCL_CLASS_UNRESOLVED",
                        f"complementOf on class '{owner.name}' references unknown class {value!r}.")
            return None
        return f"not self.oclIsKindOf({cls.name})"
    if kind == "hasKey":
        prop_iris = value if isinstance(value, list) else []
        prop_names: List[str] = []
        for iri in prop_iris:
            name = _resolve_property_name(str(iri), owner, ctx)
            if name:
                prop_names.append(name)
        if not prop_names:
            add_warning(out.warnings, "OCL_HASKEY_UNRESOLVED",
                        f"hasKey on class '{owner.name}' has no resolvable properties.")
            return None
        return _has_key_body(owner.name, prop_names)
    if kind in LOGICAL_KINDS:
        return _translate_logical(
            kind, value, owner, None, False, ctx, out, visited, depth, on_property=False,
        )
    add_warning(out.warnings, "OCL_KIND_UNSUPPORTED",
                f"Spec kind {kind!r} on class '{owner.name}' has no OCL translation; skipped.")
    return None


# ---------------------------------------------------------------------------
# Logical-operator + nested-shape translator
# ---------------------------------------------------------------------------


def _translate_logical(
    kind: str,
    value: Any,
    owner: Class,
    prop_name: Optional[str],
    is_multi: bool,
    ctx: _LookupCtx,
    out: OCLEmissionResult,
    visited: Set[str],
    depth: int,
    *,
    on_property: bool,
) -> Optional[str]:
    if not isinstance(value, list) or not value:
        return None
    operand_bodies: List[str] = []
    for slot in value:
        body = _translate_nested_slot(
            slot, owner, prop_name, is_multi, ctx, out, visited, depth + 1, on_property,
        )
        if body:
            operand_bodies.append(body)
    if not operand_bodies:
        add_warning(out.warnings, "OCL_NESTED_EMPTY",
                    f"Logical operator {kind!r} on class '{owner.name}' has no translatable operands; skipped.")
        return None
    if kind == "shaclNot":
        if len(operand_bodies) != 1:
            return None
        return f"not ({operand_bodies[0]})"
    if kind == "shaclAnd":
        return " and ".join(f"({b})" for b in operand_bodies)
    if kind == "shaclOr":
        return " or ".join(f"({b})" for b in operand_bodies)
    if kind == "shaclXone":
        # Exactly-one over n terms expressed as disjunction of n conjunctions.
        if len(operand_bodies) < 2:
            return None
        conjuncts = []
        for i, t_i in enumerate(operand_bodies):
            terms = []
            for j, t_j in enumerate(operand_bodies):
                terms.append(f"({t_j})" if j == i else f"not ({t_j})")
            conjuncts.append("(" + " and ".join(terms) + ")")
        return " or ".join(conjuncts)
    return None


def _translate_nested_slot(
    slot: Dict[str, Any],
    owner: Class,
    prop_name: Optional[str],
    is_multi: bool,
    ctx: _LookupCtx,
    out: OCLEmissionResult,
    visited: Set[str],
    depth: int,
    on_property: bool,
) -> Optional[str]:
    if depth > _MAX_REF_DEPTH:
        add_warning(out.warnings, "OCL_NESTED_DEPTH",
                    f"Nested shape exceeds max depth {_MAX_REF_DEPTH} on class '{owner.name}'; truncated.")
        return None
    if "ref" in slot:
        ref = slot["ref"]
        if ref in visited:
            add_warning(out.warnings, "OCL_NESTED_CYCLE",
                        f"Cyclic nested-shape reference {ref!r} on class '{owner.name}'; slot dropped.")
            return None
        ref_node = ctx.id_to_constraint_node.get(ref)
        if ref_node is None:
            add_warning(out.warnings, "OCL_NESTED_REF_UNKNOWN",
                        f"Nested-shape ref {ref!r} on class '{owner.name}' is unknown; slot dropped.")
            return None
        specs = ref_node.get_specs() if hasattr(ref_node, "get_specs") else []
        if _specs_are_deactivated(specs):
            return None
        return _conjoin_specs(specs, owner, prop_name, is_multi, ctx, out, visited | {ref}, depth, on_property)
    specs = slot.get("specs", [])
    return _conjoin_specs(specs, owner, prop_name, is_multi, ctx, out, visited, depth, on_property)


def _conjoin_specs(
    specs: List[Dict[str, Any]],
    owner: Class,
    prop_name: Optional[str],
    is_multi: bool,
    ctx: _LookupCtx,
    out: OCLEmissionResult,
    visited: Set[str],
    depth: int,
    on_property: bool,
) -> Optional[str]:
    if not specs:
        return None
    bodies: List[str] = []
    for s in specs:
        if on_property and prop_name is not None:
            b = _translate_property_spec(s, owner, prop_name, is_multi, ctx, out, visited, depth)
        else:
            b = _translate_node_spec(s, owner, ctx, out, visited, depth)
        if b:
            bodies.append(b)
    if not bodies:
        return None
    if len(bodies) == 1:
        return bodies[0]
    return " and ".join(f"({b})" for b in bodies)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_class(iri_or_id: Any, ctx: _LookupCtx) -> Optional[Class]:
    if not isinstance(iri_or_id, str) or not iri_or_id:
        return None
    cls = ctx.iri_to_class.get(iri_or_id)
    if cls is not None:
        return cls
    return ctx.id_to_buml_class.get(iri_or_id)


def _resolve_property_name(prop_iri_or_id: str, owner: Class, ctx: _LookupCtx) -> Optional[str]:
    """Locate the property attribute / association-target name within owner's class."""
    attr = ctx.property_iri_to_attribute.get(prop_iri_or_id)
    if attr is not None and attr in owner.attributes:
        return attr.name
    assoc = ctx.property_iri_to_association.get(prop_iri_or_id)
    if assoc is not None:
        # The end opposite the source end of the association points at the target
        # navigated via owner.{end.name}.
        src_end = ctx.assoc_source_end.get(id(assoc))
        for end in assoc.ends:
            if end is src_end:
                continue
            if src_end is not None and src_end.type is owner:
                return end.name
            # When owner is the target side, navigation uses the source end's name.
            if end.type is owner and src_end is not None:
                return src_end.name
        # Fallback: pick the end whose type isn't `owner`.
        for end in assoc.ends:
            if end.type is not owner:
                return end.name
    # Fallback by id (KGProperty.id-keyed scenarios). Guard against the
    # cycle that would arise when ``prop_iri_or_id`` was already the
    # property's IRI (lookup would return the same node forever).
    prop_node = ctx.id_to_property_node.get(prop_iri_or_id)
    if prop_node is not None and prop_node.iri and prop_node.iri != prop_iri_or_id:
        return _resolve_property_name(prop_node.iri, owner, ctx)
    return None


def _resolve_property_name_by_id(prop_id_or_iri: str, owner: Class, ctx: _LookupCtx) -> Optional[str]:
    """Like `_resolve_property_name` but accepts a KG node id and falls back via the node's IRI."""
    prop_node = ctx.id_to_property_node.get(prop_id_or_iri)
    if prop_node is not None and prop_node.iri:
        name = _resolve_property_name(prop_node.iri, owner, ctx)
        if name:
            return name
    return _resolve_property_name(prop_id_or_iri, owner, ctx)


def _owner_and_name_for_property(prop_id_or_iri: str, ctx: _LookupCtx) -> Tuple[Optional[Class], Optional[str]]:
    """Locate the owner class + local name of a property, for axiom emission."""
    prop_node = ctx.id_to_property_node.get(prop_id_or_iri)
    iri = prop_node.iri if prop_node is not None else prop_id_or_iri
    attr = ctx.property_iri_to_attribute.get(iri) if iri else None
    if attr is not None and attr.owner is not None:
        return attr.owner, attr.name
    assoc = ctx.property_iri_to_association.get(iri) if iri else None
    if assoc is not None:
        src_end = ctx.assoc_source_end.get(id(assoc))
        if src_end is not None:
            owner = src_end.type if isinstance(src_end.type, Class) else None
            for end in assoc.ends:
                if end is not src_end:
                    return owner, end.name
    return None, None


def _is_multi_valued(prop_iri: str, ctx: _LookupCtx) -> bool:
    attr = ctx.property_iri_to_attribute.get(prop_iri)
    if attr is not None and attr.multiplicity is not None:
        return attr.multiplicity.max != 1
    assoc = ctx.property_iri_to_association.get(prop_iri)
    if assoc is not None:
        src_end = ctx.assoc_source_end.get(id(assoc))
        for end in assoc.ends:
            if end is src_end:
                continue
            if end.multiplicity is not None:
                return end.multiplicity.max != 1
    return True  # default to multi-valued (safer for ->includes)


def _format_class_list(value: Any, ctx: _LookupCtx, out: OCLEmissionResult, owner: Class) -> List[Class]:
    if not isinstance(value, list):
        items = [value]
    else:
        items = list(value)
    classes: List[Class] = []
    for item in items:
        cls = _resolve_class(item, ctx)
        if cls is None:
            add_warning(out.warnings, "OCL_CLASS_UNRESOLVED",
                        f"Class reference {item!r} on '{owner.name}' could not be resolved.")
            continue
        classes.append(cls)
    return classes


# ---------------------------------------------------------------------------
# Literal / value formatting
# ---------------------------------------------------------------------------


_INDIVIDUAL_LIKE_DATATYPES = {
    "http://www.w3.org/2001/XMLSchema#anyURI",
}


def _format_literal(value: Any, out: OCLEmissionResult) -> Optional[str]:
    """Format a single literal value as OCL; return None for IRI / individual values."""
    if isinstance(value, dict):
        v = value.get("value")
        dt = value.get("datatype")
        if dt in _INDIVIDUAL_LIKE_DATATYPES:
            return None
        return _format_literal(v, out)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, str):
        # A bare http(s) string is presumed to be an IRI (individual ref) → skip.
        if value.startswith("http://") or value.startswith("https://"):
            return None
        return _quote_string_literal(value)
    return None


def _format_literal_list(value: Any, out: OCLEmissionResult) -> Tuple[List[str], bool]:
    """Format a list of literal values; second tuple entry signals whether
    any IRI-shaped entries were detected (so the caller can skip the whole
    constraint)."""
    if not isinstance(value, list):
        return [], False
    items: List[str] = []
    has_iri = False
    for v in value:
        formatted = _format_literal(v, out)
        if formatted is None:
            has_iri = True
            continue
        items.append(formatted)
    return items, has_iri


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        s = repr(value)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s
    if isinstance(value, str):
        try:
            n = int(value)
            return str(n)
        except ValueError:
            try:
                return _format_number(float(value))
            except ValueError:
                return _quote_string_literal(value)
    return _quote_string_literal(str(value))


def _quote_string_literal(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _embed_regex_flags(flags: str, out: OCLEmissionResult) -> str:
    safe = {"i", "m", "s", "x"}
    used = ""
    for ch in flags:
        if ch in safe:
            if ch not in used:
                used += ch
        else:
            add_warning(out.warnings, "OCL_REGEX_FLAG_UNKNOWN",
                        f"Regex flag {ch!r} is not recognised by the embedded-flag form; ignored.")
    return f"(?{used})" if used else ""


def _find_sibling_flags(pattern_spec: Dict[str, Any], ctx: _LookupCtx) -> Optional[str]:
    """If the constraint node carrying ``pattern_spec`` also carries a ``flags``
    spec, return its value. We don't have direct access to the parent here,
    so we rely on the caller passing the same spec list and rely on the
    plan's invariant that flags is a sibling of pattern in the same PC's
    spec list. The simplest robust thing: search every constraint node for
    a matching pattern-and-flags pair. Cheap because the list is small."""
    # Look across constraint nodes — find one whose specs include this exact
    # pattern dict and also a flags spec. This avoids threading the parent
    # spec-list down.
    for node in ctx.id_to_constraint_node.values():
        specs = node.get_specs() if hasattr(node, "get_specs") else []
        if pattern_spec in specs:
            for s in specs:
                if s.get("kind") == "flags":
                    v = s.get("value")
                    if isinstance(v, str) and v:
                        return v
    return None


def _has_key_body(class_name: str, prop_names: List[str]) -> str:
    if len(prop_names) == 1:
        return f"{class_name}.allInstances()->isUnique(c | c.{prop_names[0]})"
    diffs = " or ".join(f"a.{p} <> b.{p}" for p in prop_names)
    return (
        f"{class_name}.allInstances()->forAll(a, b | a <> b implies ({diffs}))"
    )


def _specs_are_deactivated(specs: List[Dict[str, Any]]) -> bool:
    return any(s.get("kind") == "shaclDeactivated" and bool(s.get("value")) for s in specs)


# ---------------------------------------------------------------------------
# Constraint construction
# ---------------------------------------------------------------------------


def _attach_constraint(
    owner: Class,
    body: str,
    spec: Dict[str, Any],
    sibling_specs: List[Dict[str, Any]],
    ctx: _LookupCtx,
    out: OCLEmissionResult,
    *,
    owner_label: Optional[str],
) -> None:
    label_hint = _shacl_name_in(sibling_specs) or _kind_to_label(spec.get("kind"), owner_label)
    name = _next_name(owner, label_hint, ctx)
    expression = f"context {owner.name} inv {name}: {body}"
    out.constraints.append(Constraint(name=name, context=owner, expression=expression, language="ocl"))


def _attach_axiom_constraint(
    owner: Class,
    body: str,
    label_hint: str,
    ctx: _LookupCtx,
    out: OCLEmissionResult,
) -> None:
    name = _next_name(owner, label_hint, ctx)
    expression = f"context {owner.name} inv {name}: {body}"
    out.constraints.append(Constraint(name=name, context=owner, expression=expression, language="ocl"))


def _shacl_name_in(specs: List[Dict[str, Any]]) -> Optional[str]:
    for s in specs:
        if s.get("kind") == "shaclName":
            v = s.get("value")
            if isinstance(v, dict):
                v = v.get("value")
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _kind_to_label(kind: Optional[str], owner_label: Optional[str]) -> str:
    parts = []
    if owner_label:
        parts.append(owner_label)
    if kind:
        parts.append(kind)
    return "_".join(parts) or "ocl"


def _next_name(owner: Class, hint: str, ctx: _LookupCtx) -> str:
    base = sanitize_python_identifier(f"{owner.name}_{hint}", "ocl", set())
    name = base
    seq = 2
    while name in ctx.used_constraint_names:
        name = f"{base}_{seq}"
        seq += 1
    ctx.used_constraint_names.add(name)
    return name
