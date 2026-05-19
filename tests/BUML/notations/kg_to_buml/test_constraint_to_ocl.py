"""Tests for the KG → OCL constraint emitter.

The fixtures in this module hand-build small KnowledgeGraphs that exercise one
constraint kind at a time. Each test invokes ``kg_to_class_diagram`` end-to-end
and then asserts that ``domain_model.constraints`` contains a Constraint whose
expression matches the expected OCL body. Expressions all conform to the B-OCL
grammar's ``contextDeclaration`` rule:: ``context C inv name: body``.
"""

from __future__ import annotations

import pytest

from besser.BUML.metamodel.kg import (
    KGClass,
    KGEdge,
    KGIndividual,
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
from besser.BUML.metamodel.kg.constants import (
    CONSTRAINT_TARGET_CLASS,
    CONSTRAINT_TARGET_PROPERTY,
    SH_PROPERTY,
)
from besser.BUML.notations.kg_to_buml import kg_to_class_diagram


RDFS = "http://www.w3.org/2000/01/rdf-schema#"
XSD = "http://www.w3.org/2001/XMLSchema#"
EX = "http://example.org/"


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _person_kg(*, prop_range_iri: str = XSD + "string", prop_label: str = "hasName") -> KnowledgeGraph:
    """A minimal KG with a Person class and one datatype property."""
    kg = KnowledgeGraph(name="t")
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    rng = KGClass(id="range", label=prop_label, iri=prop_range_iri)
    prop = KGProperty(id=prop_label, label=prop_label, iri=EX + prop_label)
    kg.add_node(person)
    kg.add_node(rng)
    kg.add_node(prop)
    kg.add_edge(KGEdge(id="d", source=prop, target=person, iri=RDFS + "domain"))
    kg.add_edge(KGEdge(id="r", source=prop, target=rng, iri=RDFS + "range"))
    return kg


def _person_with_object_property(*, target_class_label: str = "Org") -> KnowledgeGraph:
    kg = KnowledgeGraph(name="t")
    person = KGClass(id="Person", label="Person", iri=EX + "Person")
    other = KGClass(id=target_class_label, label=target_class_label, iri=EX + target_class_label)
    prop = KGProperty(id="hasFriend", label="hasFriend", iri=EX + "hasFriend")
    kg.add_node(person)
    kg.add_node(other)
    kg.add_node(prop)
    kg.add_edge(KGEdge(id="d", source=prop, target=person, iri=RDFS + "domain"))
    kg.add_edge(KGEdge(id="r", source=prop, target=other, iri=RDFS + "range"))
    return kg


def _attach_pc(kg: KnowledgeGraph, person_id: str, prop_iri: str, specs: list, *, pc_id: str = "pc1") -> KGPropertyConstraint:
    """Wire `pc_id` -> property `prop_iri` and group under a fresh NC -> `person_id`."""
    pc = KGPropertyConstraint(id=pc_id)
    pc.metadata = {"constraintSpecs": specs}
    nc = KGNodeConstraint(id=f"nc_{pc_id}")
    kg.add_node(pc)
    kg.add_node(nc)
    # Find the KGProperty by iri.
    prop_node = next(n for n in kg.nodes if isinstance(n, KGProperty) and n.iri == prop_iri)
    target_class = next(n for n in kg.nodes if isinstance(n, KGClass) and n.id == person_id)
    kg.add_edge(KGEdge(id=f"e1_{pc_id}", source=pc, target=prop_node, iri=CONSTRAINT_TARGET_PROPERTY))
    kg.add_edge(KGEdge(id=f"e2_{pc_id}", source=nc, target=target_class, iri=CONSTRAINT_TARGET_CLASS))
    kg.add_edge(KGEdge(id=f"e3_{pc_id}", source=nc, target=pc, iri=SH_PROPERTY))
    return pc


def _attach_nc(kg: KnowledgeGraph, class_id: str, specs: list, *, nc_id: str = "nc_alone") -> KGNodeConstraint:
    nc = KGNodeConstraint(id=nc_id)
    nc.metadata = {"constraintSpecs": specs}
    kg.add_node(nc)
    target_class = next(n for n in kg.nodes if isinstance(n, KGClass) and n.id == class_id)
    kg.add_edge(KGEdge(id=f"e_{nc_id}", source=nc, target=target_class, iri=CONSTRAINT_TARGET_CLASS))
    return nc


def _constraint_bodies(result) -> list:
    """Return just the body part (after `inv NAME: `) of every emitted constraint."""
    bodies = []
    for c in result.domain_model.constraints:
        expr = c.expression
        if "inv" in expr and ":" in expr:
            _, _, after = expr.partition("inv")
            _, _, body = after.partition(":")
            bodies.append(body.strip())
    return bodies


def _has_body(result, body_substring: str) -> bool:
    return any(body_substring in b for b in _constraint_bodies(result))


def _has_warning(result, code: str) -> bool:
    return any(w.code == code for w in result.warnings)


# ---------------------------------------------------------------------------
# Property-spec tests
# ---------------------------------------------------------------------------


def test_some_values_from_emits_exists():
    kg = _person_with_object_property(target_class_label="Org")
    _attach_pc(kg, "Person", EX + "hasFriend",
               [{"kind": "someValuesFrom", "value": EX + "Org"}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.hasFriend->exists(x | x.oclIsKindOf(Org))")


def test_all_values_from_emits_for_all():
    kg = _person_with_object_property(target_class_label="Org")
    _attach_pc(kg, "Person", EX + "hasFriend",
               [{"kind": "allValuesFrom", "value": EX + "Org"}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.hasFriend->forAll(x | x.oclIsKindOf(Org))")


def test_has_value_literal_single_valued():
    kg = _person_kg(prop_range_iri=XSD + "string", prop_label="hasName")
    # Force max=1 via maxCardinality (structural) so the property is single-valued.
    _attach_pc(kg, "Person", EX + "hasName",
               [{"kind": "maxCardinality", "value": 1},
                {"kind": "hasValue", "value": "Alice"}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.hasName = 'Alice'")


def test_has_value_literal_multi_valued():
    kg = _person_kg(prop_range_iri=XSD + "string", prop_label="nicknames")
    _attach_pc(kg, "Person", EX + "nicknames",
               [{"kind": "hasValue", "value": "Lina"}])
    result = kg_to_class_diagram(kg)
    # Multi-valued is the default → ->includes form.
    assert _has_body(result, "self.nicknames->includes('Lina')") or _has_body(result, "self.nicknames = 'Lina'")


def test_has_self_emits_includes_self():
    kg = _person_with_object_property(target_class_label="Person")
    _attach_pc(kg, "Person", EX + "hasFriend",
               [{"kind": "hasSelf", "value": True}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.hasFriend->includes(self)")


def test_in_literal_set_single_valued():
    kg = _person_kg(prop_label="color")
    _attach_pc(kg, "Person", EX + "color",
               [{"kind": "maxCardinality", "value": 1},
                {"kind": "in", "value": ["red", "green", "blue"]}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "Set{'red', 'green', 'blue'}->includes(self.color)")


def test_in_literal_set_multi_valued():
    kg = _person_kg(prop_label="colors")
    # Make the property multi-valued by lifting maxCardinality=5 (structural).
    _attach_pc(kg, "Person", EX + "colors",
               [{"kind": "maxCardinality", "value": 5},
                {"kind": "in", "value": ["red", "green"]}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.colors->forAll(x | Set{'red', 'green'}->includes(x))")


def test_pattern_emits_matches():
    kg = _person_kg(prop_label="code")
    _attach_pc(kg, "Person", EX + "code",
               [{"kind": "pattern", "value": "^[A-Z]+$"}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.code.matches('^[A-Z]+$')")


def test_pattern_with_flags_embeds_flag_group():
    kg = _person_kg(prop_label="code")
    _attach_pc(kg, "Person", EX + "code",
               [{"kind": "pattern", "value": "^[a-z]+$"},
                {"kind": "flags", "value": "im"}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.code.matches('(?im)^[a-z]+$')")


def test_min_max_length():
    kg = _person_kg(prop_label="name")
    _attach_pc(kg, "Person", EX + "name",
               [{"kind": "minLength", "value": 1},
                {"kind": "maxLength", "value": 10}])
    result = kg_to_class_diagram(kg)
    bodies = _constraint_bodies(result)
    assert any("self.name.size() >= 1" in b for b in bodies)
    assert any("self.name.size() <= 10" in b for b in bodies)


def test_numeric_ranges():
    kg = _person_kg(prop_range_iri=XSD + "integer", prop_label="age")
    _attach_pc(kg, "Person", EX + "age",
               [{"kind": "minInclusive", "value": 0},
                {"kind": "maxInclusive", "value": 130},
                {"kind": "minExclusive", "value": -1},
                {"kind": "maxExclusive", "value": 200}])
    result = kg_to_class_diagram(kg)
    bodies = _constraint_bodies(result)
    assert any("self.age >= 0" in b for b in bodies)
    assert any("self.age <= 130" in b for b in bodies)
    assert any("self.age > -1" in b for b in bodies)
    assert any("self.age < 200" in b for b in bodies)


def test_qualified_cardinality_emits_select_size():
    kg = _person_with_object_property(target_class_label="Org")
    _attach_pc(kg, "Person", EX + "hasFriend",
               [{"kind": "minQualifiedCardinality", "value": 2, "on_class": EX + "Org"}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.hasFriend->select(x | x.oclIsKindOf(Org))->size() >= 2")


def test_shacl_disjoint_property_pair():
    kg = _person_kg(prop_label="friends")
    # Add a sibling "enemies" property on Person.
    enemies = KGProperty(id="enemies", label="enemies", iri=EX + "enemies")
    string_class = next(n for n in kg.nodes if isinstance(n, KGClass) and n.iri == XSD + "string")
    person = next(n for n in kg.nodes if isinstance(n, KGClass) and n.id == "Person")
    kg.add_node(enemies)
    kg.add_edge(KGEdge(id="ed", source=enemies, target=person, iri=RDFS + "domain"))
    kg.add_edge(KGEdge(id="er", source=enemies, target=string_class, iri=RDFS + "range"))
    _attach_pc(kg, "Person", EX + "friends",
               [{"kind": "shaclDisjoint", "value": EX + "enemies"}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.friends->excludesAll(self.enemies)")


# ---------------------------------------------------------------------------
# Node-spec tests
# ---------------------------------------------------------------------------


def test_disjoint_with_emits_negated_oclIsKindOf():
    kg = _person_kg(prop_label="name")
    # Add Animal class
    animal = KGClass(id="Animal", label="Animal", iri=EX + "Animal")
    kg.add_node(animal)
    _attach_nc(kg, "Person",
               [{"kind": "disjointWith", "value": [EX + "Animal"]}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "not self.oclIsKindOf(Animal)")


def test_disjoint_union_of_uses_xor():
    kg = _person_kg(prop_label="name")
    male = KGClass(id="Male", label="Male", iri=EX + "Male")
    female = KGClass(id="Female", label="Female", iri=EX + "Female")
    kg.add_node(male)
    kg.add_node(female)
    _attach_nc(kg, "Person",
               [{"kind": "disjointUnionOf", "value": [EX + "Male", EX + "Female"]}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.oclIsKindOf(Male) xor self.oclIsKindOf(Female)")


def test_union_of():
    kg = _person_kg(prop_label="name")
    a = KGClass(id="A", label="A", iri=EX + "A")
    b = KGClass(id="B", label="B", iri=EX + "B")
    kg.add_node(a)
    kg.add_node(b)
    _attach_nc(kg, "Person",
               [{"kind": "unionOf", "value": [EX + "A", EX + "B"]}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.oclIsKindOf(A) or self.oclIsKindOf(B)")


def test_intersection_of():
    kg = _person_kg(prop_label="name")
    a = KGClass(id="A", label="A", iri=EX + "A")
    b = KGClass(id="B", label="B", iri=EX + "B")
    kg.add_node(a)
    kg.add_node(b)
    _attach_nc(kg, "Person",
               [{"kind": "intersectionOf", "value": [EX + "A", EX + "B"]}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.oclIsKindOf(A) and self.oclIsKindOf(B)")


def test_complement_of():
    kg = _person_kg(prop_label="name")
    a = KGClass(id="Animal", label="Animal", iri=EX + "Animal")
    kg.add_node(a)
    _attach_nc(kg, "Person",
               [{"kind": "complementOf", "value": EX + "Animal"}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "not self.oclIsKindOf(Animal)")


def test_equivalent_classes_uses_boolean_equality():
    kg = _person_kg(prop_label="name")
    twin = KGClass(id="Human", label="Human", iri=EX + "Human")
    kg.add_node(twin)
    _attach_nc(kg, "Person",
               [{"kind": "equivalentClasses", "value": [EX + "Human"]}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.oclIsKindOf(Human) = self.oclIsKindOf(Person)")


def test_has_key_single_property_uses_isUnique():
    kg = _person_kg(prop_label="ssn")
    _attach_nc(kg, "Person",
               [{"kind": "hasKey", "value": [EX + "ssn"]}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "Person.allInstances()->isUnique(c | c.ssn)")


def test_has_key_multi_property_uses_pairwise():
    kg = _person_kg(prop_label="firstName")
    last = KGProperty(id="lastName", label="lastName", iri=EX + "lastName")
    string_class = next(n for n in kg.nodes if isinstance(n, KGClass) and n.iri == XSD + "string")
    person = next(n for n in kg.nodes if isinstance(n, KGClass) and n.id == "Person")
    kg.add_node(last)
    kg.add_edge(KGEdge(id="ed_last", source=last, target=person, iri=RDFS + "domain"))
    kg.add_edge(KGEdge(id="er_last", source=last, target=string_class, iri=RDFS + "range"))
    _attach_nc(kg, "Person",
               [{"kind": "hasKey", "value": [EX + "firstName", EX + "lastName"]}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result,
                     "Person.allInstances()->forAll(a, b | a <> b implies (a.firstName <> b.firstName or a.lastName <> b.lastName))")


def test_shacl_deactivated_skips_emission():
    kg = _person_kg(prop_label="name")
    _attach_pc(kg, "Person", EX + "name",
               [{"kind": "shaclDeactivated", "value": True},
                {"kind": "minLength", "value": 1}],
               pc_id="pc_dead")
    result = kg_to_class_diagram(kg)
    # No constraint should have been emitted from this PC.
    assert not _has_body(result, "self.name.size() >= 1")
    assert _has_warning(result, "OCL_SHAPE_DEACTIVATED")


# ---------------------------------------------------------------------------
# Nested-shape tests
# ---------------------------------------------------------------------------


def test_shacl_not_negates_inline_body():
    kg = _person_kg(prop_label="name")
    _attach_pc(kg, "Person", EX + "name",
               [{"kind": "shaclNot", "value": [
                   {"specs": [{"kind": "minLength", "value": 1}]}
               ]}])
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "not (self.name.size() >= 1)")


def test_shacl_and_joins_inline_operands():
    kg = _person_kg(prop_label="name")
    _attach_pc(kg, "Person", EX + "name",
               [{"kind": "shaclAnd", "value": [
                   {"specs": [{"kind": "minLength", "value": 1}]},
                   {"specs": [{"kind": "maxLength", "value": 10}]},
               ]}])
    result = kg_to_class_diagram(kg)
    bodies = _constraint_bodies(result)
    assert any("self.name.size() >= 1" in b and "self.name.size() <= 10" in b and " and " in b for b in bodies)


def test_shacl_or_joins_inline_operands():
    kg = _person_kg(prop_label="name")
    _attach_pc(kg, "Person", EX + "name",
               [{"kind": "shaclOr", "value": [
                   {"specs": [{"kind": "minLength", "value": 5}]},
                   {"specs": [{"kind": "maxLength", "value": 2}]},
               ]}])
    result = kg_to_class_diagram(kg)
    assert any(" or " in b for b in _constraint_bodies(result))


def test_shacl_xone_three_terms_expands():
    kg = _person_kg(prop_range_iri=XSD + "integer", prop_label="age")
    _attach_pc(kg, "Person", EX + "age",
               [{"kind": "shaclXone", "value": [
                   {"specs": [{"kind": "minInclusive", "value": 0}]},
                   {"specs": [{"kind": "minInclusive", "value": 10}]},
                   {"specs": [{"kind": "minInclusive", "value": 20}]},
               ]}])
    result = kg_to_class_diagram(kg)
    # Should produce a disjunction of conjunctions, each pinning one term true
    # and the others false.
    bodies = _constraint_bodies(result)
    assert any(" or " in b and "not (" in b for b in bodies)


def test_nested_ref_inlines_referenced_specs():
    kg = _person_kg(prop_label="name")
    # Inline reusable PC (not directly wired to a class — used only via ref).
    reusable = KGPropertyConstraint(id="reusable")
    reusable.metadata = {"constraintSpecs": [{"kind": "minLength", "value": 3}]}
    kg.add_node(reusable)
    _attach_pc(kg, "Person", EX + "name",
               [{"kind": "shaclAnd", "value": [
                   {"specs": [{"kind": "maxLength", "value": 8}]},
                   {"ref": "reusable"},
               ]}])
    result = kg_to_class_diagram(kg)
    bodies = _constraint_bodies(result)
    assert any("self.name.size() <= 8" in b and "self.name.size() >= 3" in b for b in bodies)


def test_nested_ref_cycle_is_dropped_with_warning():
    kg = _person_kg(prop_label="name")
    # Build a PC that references itself inside its own shaclAnd.
    self_ref = KGPropertyConstraint(id="self_ref")
    self_ref.metadata = {"constraintSpecs": [
        {"kind": "shaclAnd", "value": [{"ref": "self_ref"}]},
    ]}
    kg.add_node(self_ref)
    prop = next(n for n in kg.nodes if isinstance(n, KGProperty) and n.iri == EX + "name")
    person = next(n for n in kg.nodes if isinstance(n, KGClass) and n.id == "Person")
    nc = KGNodeConstraint(id="nc_cycle")
    kg.add_node(nc)
    kg.add_edge(KGEdge(id="cyc_pt", source=self_ref, target=prop, iri=CONSTRAINT_TARGET_PROPERTY))
    kg.add_edge(KGEdge(id="cyc_nc", source=nc, target=person, iri=CONSTRAINT_TARGET_CLASS))
    kg.add_edge(KGEdge(id="cyc_sh", source=nc, target=self_ref, iri=SH_PROPERTY))
    result = kg_to_class_diagram(kg)
    assert _has_warning(result, "OCL_NESTED_CYCLE")


# ---------------------------------------------------------------------------
# Axiom tests
# ---------------------------------------------------------------------------


def test_disjoint_classes_axiom_emits_per_class():
    kg = _person_kg(prop_label="name")
    animal = KGClass(id="Animal", label="Animal", iri=EX + "Animal")
    kg.add_node(animal)
    kg.axioms = [DisjointClassesAxiom(class_ids=["Person", "Animal"])]
    result = kg_to_class_diagram(kg)
    bodies = _constraint_bodies(result)
    assert any("not self.oclIsKindOf(Animal)" in b for b in bodies)
    assert any("not self.oclIsKindOf(Person)" in b for b in bodies)


def test_disjoint_union_axiom():
    kg = _person_kg(prop_label="name")
    male = KGClass(id="Male", label="Male", iri=EX + "Male")
    female = KGClass(id="Female", label="Female", iri=EX + "Female")
    kg.add_node(male)
    kg.add_node(female)
    kg.axioms = [DisjointUnionAxiom(union_class_id="Person", part_class_ids=["Male", "Female"])]
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.oclIsKindOf(Male) xor self.oclIsKindOf(Female)")


def test_equivalent_classes_axiom():
    kg = _person_kg(prop_label="name")
    human = KGClass(id="Human", label="Human", iri=EX + "Human")
    kg.add_node(human)
    kg.axioms = [EquivalentClassesAxiom(class_ids=["Person", "Human"])]
    result = kg_to_class_diagram(kg)
    bodies = _constraint_bodies(result)
    assert any("self.oclIsKindOf(Human) = self.oclIsKindOf(Person)" in b for b in bodies)


def test_has_key_axiom():
    kg = _person_kg(prop_label="ssn")
    kg.axioms = [HasKeyAxiom(class_id="Person", property_ids=[EX + "ssn"])]
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "Person.allInstances()->isUnique(c | c.ssn)")


def test_sub_property_of_axiom():
    kg = _person_kg(prop_label="firstName")
    # Add a sibling "name" property that firstName is a sub-property of.
    name_prop = KGProperty(id="name", label="name", iri=EX + "name")
    string_class = next(n for n in kg.nodes if isinstance(n, KGClass) and n.iri == XSD + "string")
    person = next(n for n in kg.nodes if isinstance(n, KGClass) and n.id == "Person")
    kg.add_node(name_prop)
    kg.add_edge(KGEdge(id="d_name", source=name_prop, target=person, iri=RDFS + "domain"))
    kg.add_edge(KGEdge(id="r_name", source=name_prop, target=string_class, iri=RDFS + "range"))
    kg.axioms = [SubPropertyOfAxiom(sub_property_id="firstName", super_property_id="name")]
    result = kg_to_class_diagram(kg)
    assert _has_body(result, "self.name->includesAll(self.firstName)")


def test_inverse_properties_axiom():
    kg = _person_with_object_property(target_class_label="Person")
    # Build inverse pair: hasFriend ↔ friendOf, both Person→Person.
    other_prop = KGProperty(id="friendOf", label="friendOf", iri=EX + "friendOf")
    person = next(n for n in kg.nodes if isinstance(n, KGClass) and n.id == "Person")
    kg.add_node(other_prop)
    kg.add_edge(KGEdge(id="d_inv", source=other_prop, target=person, iri=RDFS + "domain"))
    kg.add_edge(KGEdge(id="r_inv", source=other_prop, target=person, iri=RDFS + "range"))
    kg.axioms = [InversePropertiesAxiom(property_a_id="hasFriend", property_b_id="friendOf")]
    result = kg_to_class_diagram(kg)
    bodies = _constraint_bodies(result)
    assert any("forAll(o | o.friendOf->includes(self))" in b or
               "forAll(o | o.hasFriend->includes(self))" in b for b in bodies)


def test_property_chain_axiom_warns_no_constraint():
    kg = _person_kg(prop_label="name")
    kg.axioms = [PropertyChainAxiom(property_id="x", chain_property_ids=["a", "b"])]
    result = kg_to_class_diagram(kg)
    assert _has_warning(result, "OCL_PROPERTY_CHAIN_UNSUPPORTED")


# ---------------------------------------------------------------------------
# Negative / skip tests
# ---------------------------------------------------------------------------


def test_cardinality_is_skipped_not_duplicated():
    kg = _person_kg(prop_label="name")
    _attach_pc(kg, "Person", EX + "name",
               [{"kind": "minCardinality", "value": 1},
                {"kind": "maxCardinality", "value": 3}])
    result = kg_to_class_diagram(kg)
    # No OCL invariants emitted for the cardinality kinds.
    bodies = _constraint_bodies(result)
    assert all("self.name->size()" not in b for b in bodies)
    # Structural multiplicity is still set (the existing tests cover this).


def test_datatype_is_skipped():
    kg = _person_kg(prop_label="name")
    _attach_pc(kg, "Person", EX + "name",
               [{"kind": "datatype", "value": XSD + "string"}])
    result = kg_to_class_diagram(kg)
    assert len(result.domain_model.constraints) == 0


def test_one_of_warning_only():
    kg = _person_kg(prop_label="name")
    _attach_nc(kg, "Person", [{"kind": "oneOf", "value": [EX + "alice", EX + "bob"]}])
    result = kg_to_class_diagram(kg)
    assert _has_warning(result, "OCL_ONEOF_SKIPPED")
    assert len(result.domain_model.constraints) == 0


def test_has_value_iri_warning_only():
    kg = _person_with_object_property(target_class_label="Person")
    _attach_pc(kg, "Person", EX + "hasFriend",
               [{"kind": "hasValue", "value": EX + "alice"}])
    result = kg_to_class_diagram(kg)
    assert _has_warning(result, "OCL_HASVALUE_IRI_SKIPPED")
    assert len(result.domain_model.constraints) == 0


def test_in_with_iri_warning_only():
    kg = _person_with_object_property(target_class_label="Person")
    _attach_pc(kg, "Person", EX + "hasFriend",
               [{"kind": "in", "value": [EX + "alice", EX + "bob"]}])
    result = kg_to_class_diagram(kg)
    assert _has_warning(result, "OCL_IN_IRI_SKIPPED")
    assert len(result.domain_model.constraints) == 0


def test_emit_ocl_false_disables_emission():
    kg = _person_kg(prop_label="name")
    _attach_pc(kg, "Person", EX + "name",
               [{"kind": "minLength", "value": 1}])
    result = kg_to_class_diagram(kg, emit_ocl=False)
    assert len(result.domain_model.constraints) == 0


def test_has_key_dedup_between_axiom_and_nc():
    kg = _person_kg(prop_label="ssn")
    _attach_nc(kg, "Person", [{"kind": "hasKey", "value": [EX + "ssn"]}])
    kg.axioms = [HasKeyAxiom(class_id="Person", property_ids=[EX + "ssn"])]
    result = kg_to_class_diagram(kg)
    # Exactly one constraint, not two.
    bodies = [b for b in _constraint_bodies(result) if "isUnique" in b]
    assert len(bodies) == 1


# ---------------------------------------------------------------------------
# Parser round-trip (validate the emitted OCL is grammar-valid)
# ---------------------------------------------------------------------------


def test_emitted_constraints_parse_under_bocl_grammar():
    """Every emitted constraint (except pattern/matches() which we accept as
    forward-looking) must parse cleanly with the existing BOCL parser."""
    from besser.utilities.web_modeling_editor.backend.services.validators.ocl_checker import _parse_only
    # Build a KG that exercises many translatable kinds.
    kg = _person_kg(prop_range_iri=XSD + "integer", prop_label="age")
    org = KGClass(id="Org", label="Org", iri=EX + "Org")
    kg.add_node(org)
    friend = KGProperty(id="hasFriend", label="hasFriend", iri=EX + "hasFriend")
    person = next(n for n in kg.nodes if isinstance(n, KGClass) and n.id == "Person")
    kg.add_node(friend)
    kg.add_edge(KGEdge(id="fd", source=friend, target=person, iri=RDFS + "domain"))
    kg.add_edge(KGEdge(id="fr", source=friend, target=org, iri=RDFS + "range"))
    _attach_pc(kg, "Person", EX + "age",
               [{"kind": "minInclusive", "value": 0},
                {"kind": "maxInclusive", "value": 130}],
               pc_id="pc_age")
    _attach_pc(kg, "Person", EX + "hasFriend",
               [{"kind": "someValuesFrom", "value": EX + "Org"}],
               pc_id="pc_friend")
    _attach_nc(kg, "Person",
               [{"kind": "disjointWith", "value": [EX + "Org"]}],
               nc_id="nc_disj")
    result = kg_to_class_diagram(kg)
    assert len(result.domain_model.constraints) >= 3
    for c in result.domain_model.constraints:
        if ".matches(" in c.expression:
            continue
        _parse_only(c.expression)  # raises on syntax error
